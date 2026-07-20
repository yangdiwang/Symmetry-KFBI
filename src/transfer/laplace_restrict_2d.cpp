#include "laplace_restrict_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "../geometry/p2_curve_2d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../local_cauchy/laplace_corner_patch_solver_2d.hpp"
#include "laplace_projection_correction_2d.hpp"

namespace kfbim {

namespace {

constexpr int kQuadraticExpansionCentersPerPanel = 4;

Eigen::Vector2d interface_point(const Interface2D& iface, int q) {
    return iface.points().row(q).transpose();
}

Eigen::Vector2d grid_point(const CartesianGrid2D& grid, int idx) {
    return structured_grid::point(grid, idx);
}

CurveProjection2D project_point_to_fixed_panel(
    const Interface2D& iface,
    int                grid_node,
    int                panel,
    int                component,
    Eigen::Vector2d    target,
    double             initial_s)
{
    constexpr int kMaxIterations = 16;
    constexpr double kResidualTol = 1.0e-12;
    constexpr double kStepTol = 1.0e-13;

    double s = geometry2d::clamp_to_panel(initial_s);
    bool clamped = false;
    bool converged = false;
    int iterations = 0;

    for (; iterations < kMaxIterations; ++iterations) {
        const Eigen::Vector2d point =
            geometry2d::panel_point(iface, panel, s);
        const Eigen::Vector2d tangent =
            geometry2d::panel_tangent(iface, panel, s);
        const Eigen::Vector2d second =
            geometry2d::panel_second_derivative(iface, panel, s);
        const Eigen::Vector2d r = point - target;
        const double f = r.dot(tangent);
        const double tangent_len = tangent.norm();
        const double residual =
            tangent_len > 1.0e-14 ? std::abs(f) / tangent_len : std::abs(f);
        if (residual <= kResidualTol) {
            converged = !clamped;
            break;
        }

        const double jac = tangent.squaredNorm() + r.dot(second);
        if (std::abs(jac) <= 1.0e-28 || !std::isfinite(jac))
            break;

        const double raw_next = s - f / jac;
        const double next = geometry2d::clamp_to_panel(raw_next);
        if (next != raw_next)
            clamped = true;
        if (std::abs(next - s) <= kStepTol) {
            s = next;
            break;
        }
        s = next;
    }

    const Eigen::Vector2d point = geometry2d::panel_point(iface, panel, s);
    const Eigen::Vector2d tangent = geometry2d::panel_tangent(iface, panel, s);
    const Eigen::Vector2d normal = geometry2d::panel_normal(iface, panel, s);
    const Eigen::Vector2d r = point - target;
    const double tangent_len = tangent.norm();

    CurveProjection2D projection;
    projection.grid_node = grid_node;
    projection.panel = panel;
    projection.component = component;
    projection.local_s = s;
    projection.point = point;
    projection.normal = normal;
    projection.signed_distance = (target - point).dot(normal);
    projection.distance = r.norm();
    projection.tangential_residual =
        tangent_len > 1.0e-14 ? std::abs(r.dot(tangent)) / tangent_len
                              : std::abs(r.dot(tangent));
    projection.iterations = iterations;
    projection.converged =
        converged || projection.tangential_residual <= 10.0 * kResidualTol;
    return projection;
}

bool corner_patch_side_matches_label(const Interface2D& iface,
                                     int                patch_id,
                                     int                label)
{
    if (patch_id < 0
        || patch_id >= static_cast<int>(iface.corner_patches().size())) {
        return false;
    }
    const CornerPatch2D& patch =
        iface.corner_patches()[static_cast<std::size_t>(patch_id)];
    return patch.side == CornerPatchSide2D::Interior ? label > 0
                                                     : label == 0;
}

bool corner_patch_data_is_transmission_pair(
    const CornerPatchCorrectionData2D& data)
{
    bool transmission_pair = !data.modes.empty();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        transmission_pair =
            transmission_pair
            && (mode.model
                    == CornerPatchSingularModel2D::
                           InteriorDirichletTransmissionPair
                || mode.model
                       == CornerPatchSingularModel2D::
                              InteriorNeumannTransmissionPair);
    }
    return transmission_pair;
}

bool point_is_in_corner_patch_sector(
    const CornerPatchCorrectionData2D& data,
    const Eigen::Vector2d&             pt)
{
    if (corner_patch_data_is_transmission_pair(data)) {
        return (pt - data.center).norm() <= data.radius + 1.0e-12
            && (pt - data.center).norm() > 1.0e-14;
    }

    CornerPatch2D patch_geometry;
    patch_geometry.corner = -1;
    patch_geometry.point = -1;
    patch_geometry.opening_angle = data.opening_angle;
    patch_geometry.radius = data.radius;
    patch_geometry.center = data.center;
    patch_geometry.ray_minus = data.ray_minus;
    patch_geometry.ray_plus = data.ray_plus;
    return corner_patch_sector_contains_point_2d(patch_geometry, pt);
}

double corner_patch_bulk_singular_contribution(
    const Interface2D&                 iface,
    const CornerPatchCorrectionData2D& data,
    const Eigen::Vector2d&             pt)
{
    (void)iface;
    return evaluate_corner_patch_singular_2d(data, pt);
}

struct CornerPatchAverageTraceRestore2D {
    double value = 0.0;
    Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
    const char* source = "singular";
};

CornerPatchAverageTraceRestore2D corner_patch_average_trace_restore_at_point(
    const Interface2D&                 iface,
    const CornerPatchCorrectionData2D& data,
    int                                boundary_iter_point,
    const Eigen::Vector2d&             pt)
{
    (void)iface;
    (void)boundary_iter_point;
    CornerPatchAverageTraceRestore2D restore;
    restore.value = evaluate_corner_patch_singular_average_trace_2d(data, pt);
    restore.gradient =
        evaluate_corner_patch_singular_average_gradient_2d(data, pt);
    restore.source = "singular";
    return restore;
}

CornerPatchAverageRestoreBasis2D build_corner_patch_average_restore_basis_at_point(
    const Interface2D& iface,
    const std::vector<CornerPatchCorrectionData2D>& corrections,
    int q)
{
    CornerPatchAverageRestoreBasis2D basis;
    const int patch = patch_id_of_boundary_iter_point_2d(iface, q);
    basis.patch = patch;
    if (patch < 0)
        return basis;
    const CornerPatchCorrectionData2D& data =
        laplace_projection_correction_detail::
            corner_patch_correction_by_patch(corrections, patch);
    const Eigen::Vector2d pt = iface.points().row(q).transpose();

    if (data.modes.empty()) {
        CornerPatchCorrectionData2D unit = data;
        unit.amplitude = 1.0;
        const CornerPatchAverageTraceRestore2D restore =
            corner_patch_average_trace_restore_at_point(iface, unit, q, pt);
        basis.value_basis.push_back(restore.value);
        basis.gradient_basis.push_back(restore.gradient);
        basis.valid = true;
        return basis;
    }

    basis.value_basis.reserve(data.modes.size());
    basis.gradient_basis.reserve(data.modes.size());
    for (std::size_t k = 0; k < data.modes.size(); ++k) {
        CornerPatchCorrectionData2D unit = data;
        unit.amplitude = 0.0;
        for (CornerPatchSingularMode2D& mode : unit.modes)
            mode.amplitude = 0.0;
        unit.modes[k].amplitude = 1.0;
        const CornerPatchAverageTraceRestore2D restore =
            corner_patch_average_trace_restore_at_point(iface, unit, q, pt);
        basis.value_basis.push_back(restore.value);
        basis.gradient_basis.push_back(restore.gradient);
    }
    basis.valid = true;
    return basis;
}

std::vector<CornerPatchAverageRestoreBasis2D>
build_corner_patch_average_restore_basis_by_point(
    const Interface2D& iface,
    const std::vector<CornerPatchCorrectionData2D>& corrections)
{
    std::vector<CornerPatchAverageRestoreBasis2D> basis_by_point(
        static_cast<std::size_t>(iface.num_points()));
    for (int q = 0; q < iface.num_points(); ++q) {
        basis_by_point[static_cast<std::size_t>(q)] =
            build_corner_patch_average_restore_basis_at_point(
                iface, corrections, q);
    }
    return basis_by_point;
}

CornerPatchAverageTraceRestore2D evaluate_cached_average_trace_restore(
    const Interface2D& iface,
    const std::vector<CornerPatchCorrectionData2D>& corrections,
    const std::vector<CornerPatchAverageRestoreBasis2D>& basis_by_point,
    int q,
    const Eigen::Vector2d& pt)
{
    CornerPatchAverageTraceRestore2D restore;
    const int patch = patch_id_of_boundary_iter_point_2d(iface, q);
    if (patch < 0)
        return restore;
    const CornerPatchCorrectionData2D& data =
        laplace_projection_correction_detail::
            corner_patch_correction_by_patch(corrections, patch);
    const CornerPatchAverageRestoreBasis2D* basis =
        q >= 0 && q < static_cast<int>(basis_by_point.size())
            ? &basis_by_point[static_cast<std::size_t>(q)]
            : nullptr;
    if (basis != nullptr && basis->valid && basis->patch == patch) {
        if (data.modes.empty() && basis->value_basis.size() == 1
            && basis->gradient_basis.size() == 1) {
            restore.value = data.amplitude * basis->value_basis[0];
            restore.gradient = data.amplitude * basis->gradient_basis[0];
            restore.source = "cached_singular_basis";
            return restore;
        }
        if (!data.modes.empty()
            && basis->value_basis.size() == data.modes.size()
            && basis->gradient_basis.size() == data.modes.size()) {
            restore.value = 0.0;
            restore.gradient = Eigen::Vector2d::Zero();
            for (std::size_t k = 0; k < data.modes.size(); ++k) {
                restore.value +=
                    data.modes[k].amplitude * basis->value_basis[k];
                restore.gradient +=
                    data.modes[k].amplitude * basis->gradient_basis[k];
            }
            restore.source = "cached_singular_basis";
            return restore;
        }
    }
    return corner_patch_average_trace_restore_at_point(iface, data, q, pt);
}

int panel_containing_interface_point(const Interface2D& iface, int point)
{
    if (point < 0 || point >= iface.num_points())
        return -1;
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (int local = 0; local < iface.points_per_panel(); ++local) {
            if (iface.point_index(panel, local) == point)
                return panel;
        }
    }
    return -1;
}

int nearest_available_center_on_panel(const Eigen::Vector2d& anchor_pt,
                                      const std::vector<LocalPoly2D>& polys,
                                      int panel,
                                      int anchor_node)
{
    if (panel < 0 || anchor_node < 0)
        return -1;
    const int first = panel * kQuadraticExpansionCentersPerPanel;
    const int last = first + kQuadraticExpansionCentersPerPanel;
    if (first < 0 || first >= static_cast<int>(polys.size()))
        return -1;

    int best = -1;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int center = first;
         center < last && center < static_cast<int>(polys.size());
         ++center) {
        if (polys[static_cast<std::size_t>(center)].coeffs.size() == 0)
            continue;
        const double dist2 =
            (polys[static_cast<std::size_t>(center)].center - anchor_pt)
                .squaredNorm();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = center;
        }
    }
    return best;
}

} // namespace

class LaplaceRestrictCorrectionEvaluator2D {
public:
    virtual ~LaplaceRestrictCorrectionEvaluator2D() = default;
    virtual double half_correction(int boundary_iter_point,
                                   int restrict_patch,
                                   int anchor_node,
                                   int grid_node,
                                   int stencil_row,
                                   Eigen::Vector2d pt) const = 0;
    virtual double bulk_singular_restore(int restrict_patch,
                                         int grid_node,
                                         Eigen::Vector2d pt) const = 0;
    virtual void restore_physical_average_trace(int boundary_iter_point,
                                                LocalPoly2D& poly) const = 0;
};

namespace {

class NearestExpansionCenterRestrictCorrectionEvaluator2D final
    : public LaplaceRestrictCorrectionEvaluator2D {
public:
    NearestExpansionCenterRestrictCorrectionEvaluator2D(
        const GridPair2D& grid_pair,
        const LaplaceSpreadResult2D& spread_result,
        const std::vector<int>& interface_panel_by_point,
        const std::vector<int>& patch_boundary_center_by_point,
        const std::vector<int>& full_boundary_center_by_point,
        const std::vector<RestrictHalfCorrectionBasis2D>&
            half_correction_basis_by_point,
        const std::vector<CornerPatchAverageRestoreBasis2D>&
            average_restore_basis_by_point)
        : grid_pair_(grid_pair)
        , correction_polys_(spread_result.correction_polys)
        , patch_regularized_correction_polys_(
              spread_result.patch_regularized_correction_polys)
        , corner_patch_corrections_(spread_result.corner_patch_corrections)
        , sd_corner_lifting_(spread_result.sd_corner_lifting)
        , jump_mode_(spread_result.restrict_jump_mode)
        , interface_panel_by_point_(interface_panel_by_point)
        , patch_boundary_center_by_point_(patch_boundary_center_by_point)
        , full_boundary_center_by_point_(full_boundary_center_by_point)
        , half_correction_basis_by_point_(half_correction_basis_by_point)
        , average_restore_basis_by_point_(average_restore_basis_by_point)
    {}

    double half_correction(int boundary_iter_point,
                           int restrict_patch,
                           int anchor_node,
                           int grid_node,
                           int stencil_row,
                           Eigen::Vector2d pt) const override
    {
        if (!sd_corner_lifting_.enabled
            && boundary_iter_point >= 0
            && boundary_iter_point
                   < static_cast<int>(half_correction_basis_by_point_.size())
            && stencil_row >= 0 && stencil_row < 6) {
            const RestrictHalfCorrectionBasis2D& cached =
                half_correction_basis_by_point_[
                    static_cast<std::size_t>(boundary_iter_point)];
            const RestrictHalfCorrectionRowBasis2D& row =
                cached.rows[static_cast<std::size_t>(stencil_row)];
            if (row.grid_node == grid_node) {
                int center_idx = cached.anchor_center;
                const Eigen::Matrix<double, 1, 6>* basis = &row.full_basis;
                const LocalPoly2D* poly =
                    center_idx >= 0
                    && center_idx < static_cast<int>(correction_polys_.size())
                        ? &correction_polys_[static_cast<std::size_t>(
                              center_idx)]
                        : nullptr;
                if (jump_mode_ == PatchInterfaceJumpMode2D::SingularRemovedJump
                    && !patch_regularized_correction_polys_.empty()) {
                    const int patch = restrict_patch >= 0
                        ? restrict_patch
                        : cached.patch;
                    if (patch >= 0
                        && patch
                               < static_cast<int>(
                                   patch_regularized_correction_polys_.size())) {
                        const int patch_center_idx = cached.patch_center >= 0
                            ? cached.patch_center
                            : cached.anchor_center;
                        const LocalPoly2D* patch_poly =
                            find_indexed_local_poly_2d(
                                patch_regularized_correction_polys_[
                                    static_cast<std::size_t>(patch)],
                                patch_center_idx);
                        if (patch_poly != nullptr) {
                            center_idx = patch_center_idx;
                            poly = patch_poly;
                            basis = &row.patch_basis;
                        }
                    }
                }
                if (poly != nullptr && poly->coeffs.size() == 6) {
                    const Eigen::VectorXd& coeffs = poly->coeffs;
                    double value = 0.0;
                    for (int k = 0; k < 6; ++k)
                        value += (*basis)(0, k) * coeffs[k];
                    return 0.5 * value;
                }
            }
        }
        const int anchor_center_idx =
            grid_pair_.nearest_p2_expansion_center(anchor_node);
        int center_idx = anchor_center_idx;
        const LocalPoly2D* poly =
            center_idx >= 0
            && center_idx < static_cast<int>(correction_polys_.size())
                ? &correction_polys_[static_cast<std::size_t>(center_idx)]
                : nullptr;
        if (jump_mode_ == PatchInterfaceJumpMode2D::SingularRemovedJump
            && !patch_regularized_correction_polys_.empty()) {
            const int patch = restrict_patch >= 0
                ? restrict_patch
                : patch_id_of_boundary_iter_point_2d(
                    grid_pair_.interface(), boundary_iter_point);
            if (patch >= 0
                && patch
                       < static_cast<int>(
                           patch_regularized_correction_polys_.size())
                && find_indexed_local_poly_2d(
                       patch_regularized_correction_polys_[
                           static_cast<std::size_t>(patch)],
                       anchor_center_idx)
                       != nullptr) {
                const std::vector<int>& cached_boundary_centers =
                    sd_corner_lifting_.enabled
                        ? full_boundary_center_by_point_
                        : patch_boundary_center_by_point_;
                int boundary_center_idx =
                    boundary_iter_point >= 0
                    && boundary_iter_point
                           < static_cast<int>(cached_boundary_centers.size())
                        ? cached_boundary_centers[
                              static_cast<std::size_t>(boundary_iter_point)]
                        : -1;
                if (boundary_center_idx >= 0
                    && find_indexed_local_poly_2d(
                           patch_regularized_correction_polys_[
                               static_cast<std::size_t>(patch)],
                           boundary_center_idx)
                           == nullptr) {
                    boundary_center_idx = -1;
                }
                center_idx = boundary_center_idx >= 0
                    ? boundary_center_idx
                    : anchor_center_idx;
                poly = find_indexed_local_poly_2d(
                    patch_regularized_correction_polys_[
                        static_cast<std::size_t>(patch)],
                    center_idx);
            }
        }
        if (poly == nullptr) {
            throw std::invalid_argument(
                "LaplaceQuadraticPanelCenterRestrict2D correction_polys missing expansion center");
        }
        const double value = evaluate_taylor_poly_2d(*poly, pt);
        return 0.5 * value;
    }

    double bulk_singular_restore(int restrict_patch,
                                 int grid_node,
                                 Eigen::Vector2d pt) const override
    {
        double restore = 0.0;
        if (sd_corner_lifting_.enabled) {
            const int node_patch = grid_pair_.corner_patch_label(grid_node);
            if (restrict_patch >= 0
                && jump_mode_ == PatchInterfaceJumpMode2D::SingularRemovedJump) {
                if (node_patch != restrict_patch) {
                    restore -= sd_corner_lifting_.lifting.value_for_patch(
                        grid_pair_.interface(), restrict_patch, pt);
                }
            } else if (restrict_patch < 0 && node_patch >= 0) {
                restore += sd_corner_lifting_.lifting.value_for_patch(
                    grid_pair_.interface(), node_patch, pt);
            }
        }

        if (corner_patch_corrections_.empty())
            return restore;

        if (restrict_patch >= 0
            && jump_mode_ == PatchInterfaceJumpMode2D::SingularRemovedJump) {
            if (grid_pair_.corner_patch_label(grid_node) == restrict_patch)
                return restore;
            const CornerPatchCorrectionData2D& data =
                laplace_projection_correction_detail::
                    corner_patch_correction_by_patch(
                        corner_patch_corrections_, restrict_patch);
            if (!corner_patch_data_is_transmission_pair(data)
                && !corner_patch_side_matches_label(
                    grid_pair_.interface(),
                    restrict_patch,
                    grid_pair_.domain_label(grid_node))) {
                return restore;
            }
            if (!point_is_in_corner_patch_sector(data, pt))
                return restore;
            return restore
                 - corner_patch_bulk_singular_contribution(
                       grid_pair_.interface(), data, pt);
        }

        if (restrict_patch >= 0)
            return restore;
        const int patch = grid_pair_.corner_patch_label(grid_node);
        if (patch < 0)
            return restore;

        const CornerPatchCorrectionData2D& data =
            laplace_projection_correction_detail::
                corner_patch_correction_by_patch(
                    corner_patch_corrections_, patch);
        return restore
             + corner_patch_bulk_singular_contribution(
                   grid_pair_.interface(), data, pt);
    }

    void restore_physical_average_trace(int boundary_iter_point,
                                        LocalPoly2D& poly) const override
    {
        if (jump_mode_ != PatchInterfaceJumpMode2D::SingularRemovedJump)
            return;
        const Interface2D& iface = grid_pair_.interface();
        const int patch =
            patch_id_of_boundary_iter_point_2d(iface, boundary_iter_point);
        if (patch < 0)
            return;
        const Eigen::Vector2d pt =
            iface.points().row(boundary_iter_point).transpose();

        double value_restore = 0.0;
        Eigen::Vector2d gradient_restore = Eigen::Vector2d::Zero();
        if (sd_corner_lifting_.enabled) {
            const Eigen::Vector2d normal =
                sd_corner_lifting_detail::normalized_or_throw(
                    iface.normals().row(boundary_iter_point).transpose(),
                    "S_D average trace normal must be nonzero");
            const double eps = sd_corner_lifting_.normal_probe;
            const Eigen::Vector2d inside = pt - eps * normal;
            const Eigen::Vector2d outside = pt + eps * normal;
            value_restore +=
                0.5
                * (sd_corner_lifting_.lifting.value_for_patch(
                       iface, patch, inside)
                   + sd_corner_lifting_.lifting.value_for_patch(
                       iface, patch, outside));
            gradient_restore +=
                0.5
                * (sd_corner_lifting_.lifting.gradient_for_patch(
                       iface, patch, inside)
                   + sd_corner_lifting_.lifting.gradient_for_patch(
                       iface, patch, outside));
        } else if (!corner_patch_corrections_.empty()) {
            const CornerPatchAverageTraceRestore2D patch_restore =
                evaluate_cached_average_trace_restore(
                    iface,
                    corner_patch_corrections_,
                    average_restore_basis_by_point_,
                    boundary_iter_point,
                    pt);
            value_restore += patch_restore.value;
            gradient_restore += patch_restore.gradient;
        }

        poly.coeffs[0] += value_restore;
        poly.coeffs[1] += gradient_restore[0];
        poly.coeffs[2] += gradient_restore[1];
    }

private:
    const GridPair2D& grid_pair_;
    const std::vector<LocalPoly2D>& correction_polys_;
    const std::vector<IndexedLocalPolys2D>&
        patch_regularized_correction_polys_;
    const std::vector<CornerPatchCorrectionData2D>& corner_patch_corrections_;
    const SdCornerLiftingOptions2D& sd_corner_lifting_;
    PatchInterfaceJumpMode2D jump_mode_;
    const std::vector<int>& interface_panel_by_point_;
    const std::vector<int>& patch_boundary_center_by_point_;
    const std::vector<int>& full_boundary_center_by_point_;
    const std::vector<RestrictHalfCorrectionBasis2D>&
        half_correction_basis_by_point_;
    const std::vector<CornerPatchAverageRestoreBasis2D>&
        average_restore_basis_by_point_;
};

class ProjectionPointRestrictCorrectionEvaluator2D final
    : public LaplaceRestrictCorrectionEvaluator2D {
public:
    ProjectionPointRestrictCorrectionEvaluator2D(
        const GridPair2D& grid_pair,
        const LaplaceSpreadResult2D& spread_result)
        : grid_pair_(grid_pair)
        , spread_result_(spread_result)
        , projection_band_(spread_result.projection_cache)
        , jump_mode_(spread_result.restrict_jump_mode)
    {}

    double half_correction(int boundary_iter_point,
                           int restrict_patch,
                           int anchor_node,
                           int grid_node,
                           int stencil_row,
                           Eigen::Vector2d pt) const override
    {
        (void)stencil_row;
        if (!projection_band_.has_projection(anchor_node)) {
            throw std::runtime_error(
                "LaplaceQuadraticPanelCenterRestrict2D missing anchor projection");
        }
        const CurveProjection2D& anchor_projection =
            projection_band_.projection(anchor_node);
        const CurveProjection2D projection =
            project_point_to_fixed_panel(grid_pair_.interface(),
                                         grid_node,
                                         anchor_projection.panel,
                                         anchor_projection.component,
                                         pt,
                                         anchor_projection.local_s);
        const PanelJumpValues2D values =
            build_restrict_panel_jump_values(grid_pair_.interface(),
                                             spread_result_,
                                             projection.panel,
                                             boundary_iter_point,
                                             jump_mode_,
                                             nullptr,
                                             restrict_patch);
        const double correction =
            laplace_projection_correction_detail::
                evaluate_projection_point_correction_from_panel_values_2d(
                    grid_pair_.interface(),
                    projection,
                    spread_result_,
                    values);

        return 0.5 * correction;
    }

    double bulk_singular_restore(int restrict_patch,
                                 int grid_node,
                                 Eigen::Vector2d pt) const override
    {
        if (spread_result_.corner_patch_corrections.empty()) {
            return 0.0;
        }

        if (restrict_patch >= 0
            && jump_mode_ == PatchInterfaceJumpMode2D::SingularRemovedJump) {
            if (grid_pair_.corner_patch_label(grid_node) == restrict_patch)
                return 0.0;
            const CornerPatchCorrectionData2D& data =
                laplace_projection_correction_detail::
                    corner_patch_correction_by_patch(
                        spread_result_.corner_patch_corrections,
                        restrict_patch);
            if (!corner_patch_data_is_transmission_pair(data)
                && !corner_patch_side_matches_label(
                    grid_pair_.interface(),
                    restrict_patch,
                    grid_pair_.domain_label(grid_node))) {
                return 0.0;
            }
            if (!point_is_in_corner_patch_sector(data, pt))
                return 0.0;
            return -corner_patch_bulk_singular_contribution(
                grid_pair_.interface(), data, pt);
        }

        if (restrict_patch >= 0)
            return 0.0;
        const int patch = grid_pair_.corner_patch_label(grid_node);
        if (patch < 0)
            return 0.0;

        const CornerPatchCorrectionData2D& data =
            laplace_projection_correction_detail::
                corner_patch_correction_by_patch(
                    spread_result_.corner_patch_corrections, patch);
        return corner_patch_bulk_singular_contribution(
            grid_pair_.interface(), data, pt);
    }

    void restore_physical_average_trace(int boundary_iter_point,
                                        LocalPoly2D& poly) const override
    {
        if (jump_mode_ != PatchInterfaceJumpMode2D::SingularRemovedJump)
            return;
        const Interface2D& iface = grid_pair_.interface();
        const int patch =
            patch_id_of_boundary_iter_point_2d(iface, boundary_iter_point);
        if (patch < 0)
            return;
        if (spread_result_.corner_patch_corrections.empty())
            return;

        const CornerPatchCorrectionData2D& data =
            laplace_projection_correction_detail::
                corner_patch_correction_by_patch(
                    spread_result_.corner_patch_corrections, patch);
        const Eigen::Vector2d pt =
            iface.points().row(boundary_iter_point).transpose();
        const CornerPatchAverageTraceRestore2D patch_restore =
            corner_patch_average_trace_restore_at_point(
                iface, data, boundary_iter_point, pt);
        poly.coeffs[0] += patch_restore.value;
        poly.coeffs[1] += patch_restore.gradient[0];
        poly.coeffs[2] += patch_restore.gradient[1];
    }

private:
    const GridPair2D& grid_pair_;
    const LaplaceSpreadResult2D& spread_result_;
    const NarrowBandProjection2D& projection_band_;
    PatchInterfaceJumpMode2D jump_mode_;
};

std::unique_ptr<LaplaceRestrictCorrectionEvaluator2D> make_restrict_correction_evaluator(
    const GridPair2D& grid_pair,
    const LaplaceSpreadResult2D& spread_result,
    int expected_centers,
    const std::vector<int>& interface_panel_by_point,
    const std::vector<int>& patch_boundary_center_by_point,
    const std::vector<int>& full_boundary_center_by_point,
    const std::vector<RestrictHalfCorrectionBasis2D>&
        half_correction_basis_by_point,
    const std::vector<CornerPatchAverageRestoreBasis2D>&
        average_restore_basis_by_point)
{
    const auto& iface = grid_pair.interface();
    if (spread_result.correction_method
        == LaplaceCorrectionMethod2D::NearestExpansionCenter) {
        if (static_cast<int>(spread_result.correction_polys.size())
            != expected_centers) {
            throw std::invalid_argument(
                "LaplaceQuadraticPanelCenterRestrict2D correction_polys size must equal 4*num_panels");
        }
        return std::make_unique<NearestExpansionCenterRestrictCorrectionEvaluator2D>(
            grid_pair,
            spread_result,
            interface_panel_by_point,
            patch_boundary_center_by_point,
            full_boundary_center_by_point,
            half_correction_basis_by_point,
            average_restore_basis_by_point);
    }

    if (spread_result.correction_method == LaplaceCorrectionMethod2D::ProjectionPoint) {
        require_projection_curve_data(iface, spread_result);
        return std::make_unique<ProjectionPointRestrictCorrectionEvaluator2D>(
            grid_pair, spread_result);
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPanelCenterRestrict2D unsupported correction method");
}

std::vector<Eigen::Matrix<double, 6, 6>> build_restrict_fit_inverse_by_point(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support)
{
    const auto& grid = grid_pair.grid();
    const Interface2D& iface = grid_pair.interface();
    std::vector<Eigen::Matrix<double, 6, 6>> inverses(
        static_cast<std::size_t>(iface.num_points()));

    for (int q = 0; q < iface.num_points(); ++q) {
        const Eigen::Vector2d center = interface_point(iface, q);
        const std::array<int, 6>& stencil_nodes =
            support.restrict_stencils[static_cast<std::size_t>(q)];
        Eigen::Matrix<double, 6, 6> A;
        for (int r = 0; r < static_cast<int>(stencil_nodes.size()); ++r) {
            const int idx = stencil_nodes[static_cast<std::size_t>(r)];
            const Eigen::Vector2d pt = grid_point(grid, idx);
            const double dx = pt[0] - center[0];
            const double dy = pt[1] - center[1];
            A(r, 0) = 1.0;
            A(r, 1) = dx;
            A(r, 2) = dy;
            A(r, 3) = 0.5 * dx * dx;
            A(r, 4) = dx * dy;
            A(r, 5) = 0.5 * dy * dy;
        }
        Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(A);
        if (!lu.isInvertible()) {
            throw std::runtime_error(
                "LaplaceQuadraticPanelCenterRestrict2D singular fixed quadratic interpolation stencil");
        }
        inverses[static_cast<std::size_t>(q)] =
            lu.solve(Eigen::Matrix<double, 6, 6>::Identity());
    }

    return inverses;
}

std::vector<int> build_interface_panel_by_point(const Interface2D& iface)
{
    std::vector<int> panel_by_point(
        static_cast<std::size_t>(iface.num_points()), -1);
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (int local = 0; local < iface.points_per_panel(); ++local) {
            const int q = iface.point_index(panel, local);
            if (q >= 0 && q < iface.num_points())
                panel_by_point[static_cast<std::size_t>(q)] = panel;
        }
    }
    return panel_by_point;
}

std::vector<std::vector<char>> build_patch_selected_center_mask(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support)
{
    const Interface2D& iface = grid_pair.interface();
    const int n_patches =
        static_cast<int>(iface.corner_patches().size());
    const int n_centers =
        kQuadraticExpansionCentersPerPanel * iface.num_panels();
    std::vector<std::vector<char>> selected(
        static_cast<std::size_t>(n_patches),
        std::vector<char>(static_cast<std::size_t>(n_centers), 0));
    for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops) {
        if (op.kind != LaplaceCrossingKind::InterfaceJump
            || op.patch < 0
            || op.patch >= n_patches) {
            continue;
        }
        const int center =
            grid_pair.nearest_p2_expansion_center_between(op.rhs_node,
                                                          op.correction_node);
        if (center >= 0 && center < n_centers) {
            selected[static_cast<std::size_t>(op.patch)]
                    [static_cast<std::size_t>(center)] = 1;
        }
    }
    return selected;
}

std::vector<int> build_boundary_center_by_point(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support,
    const std::vector<int>& panel_by_point,
    bool selected_only)
{
    const Interface2D& iface = grid_pair.interface();
    const CartesianGrid2D& grid = grid_pair.grid();
    const int n_iface = iface.num_points();
    const int n_centers =
        kQuadraticExpansionCentersPerPanel * iface.num_panels();
    const std::vector<std::vector<char>> selected =
        selected_only ? build_patch_selected_center_mask(grid_pair, support)
                      : std::vector<std::vector<char>>();
    std::vector<int> center_by_point(
        static_cast<std::size_t>(n_iface), -1);

    for (int q = 0; q < n_iface; ++q) {
        const int panel =
            q < static_cast<int>(panel_by_point.size())
                ? panel_by_point[static_cast<std::size_t>(q)]
                : -1;
        if (panel < 0)
            continue;
        const int patch = patch_id_of_boundary_iter_point_2d(iface, q);
        if (selected_only && patch < 0)
            continue;
        if (selected_only
            && patch >= static_cast<int>(selected.size())) {
            continue;
        }

        const int anchor_node = grid_pair.closest_bulk_node(q);
        const Eigen::Vector2d anchor_pt = grid_point(grid, anchor_node);
        const int first = panel * kQuadraticExpansionCentersPerPanel;
        const int last = first + kQuadraticExpansionCentersPerPanel;
        int best = -1;
        double best_dist2 = std::numeric_limits<double>::infinity();
        for (int center = first; center < last && center < n_centers;
             ++center) {
            if (selected_only
                && !selected[static_cast<std::size_t>(patch)]
                            [static_cast<std::size_t>(center)]) {
                continue;
            }
            const int local =
                center - panel * kQuadraticExpansionCentersPerPanel;
            const double s = geometry2d::kP2CenterS[
                static_cast<std::size_t>(local)];
            const Eigen::Vector2d center_pt =
                geometry2d::panel_point(iface, panel, s);
            const double dist2 = (center_pt - anchor_pt).squaredNorm();
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best = center;
            }
        }
        center_by_point[static_cast<std::size_t>(q)] = best;
    }
    return center_by_point;
}

Eigen::Vector2d expansion_center_point_2d(const Interface2D& iface,
                                          int center_idx)
{
    if (center_idx < 0)
        return Eigen::Vector2d::Zero();
    const int panel = center_idx / kQuadraticExpansionCentersPerPanel;
    const int local = center_idx % kQuadraticExpansionCentersPerPanel;
    if (panel < 0 || panel >= iface.num_panels()
        || local < 0
        || local >= static_cast<int>(geometry2d::kP2CenterS.size())) {
        return Eigen::Vector2d::Zero();
    }
    return geometry2d::panel_point(
        iface,
        panel,
        geometry2d::kP2CenterS[static_cast<std::size_t>(local)]);
}

Eigen::Matrix<double, 1, 6> taylor_basis_at_point_2d(
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& pt)
{
    const double dx = pt[0] - center[0];
    const double dy = pt[1] - center[1];
    Eigen::Matrix<double, 1, 6> basis;
    basis << 1.0, dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy;
    return basis;
}

std::vector<RestrictHalfCorrectionBasis2D>
build_restrict_half_correction_basis_by_point(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support,
    const std::vector<int>& patch_boundary_center_by_point)
{
    const Interface2D& iface = grid_pair.interface();
    const CartesianGrid2D& grid = grid_pair.grid();
    std::vector<RestrictHalfCorrectionBasis2D> basis_by_point(
        static_cast<std::size_t>(iface.num_points()));

    for (int q = 0; q < iface.num_points(); ++q) {
        RestrictHalfCorrectionBasis2D& basis =
            basis_by_point[static_cast<std::size_t>(q)];
        basis.patch = patch_id_of_boundary_iter_point_2d(iface, q);
        const int anchor_node = grid_pair.closest_bulk_node(q);
        basis.anchor_center = grid_pair.nearest_p2_expansion_center(anchor_node);
        basis.patch_center =
            q < static_cast<int>(patch_boundary_center_by_point.size())
            && patch_boundary_center_by_point[static_cast<std::size_t>(q)] >= 0
                ? patch_boundary_center_by_point[static_cast<std::size_t>(q)]
                : basis.anchor_center;

        const Eigen::Vector2d full_center =
            expansion_center_point_2d(iface, basis.anchor_center);
        const Eigen::Vector2d patch_center =
            expansion_center_point_2d(iface, basis.patch_center);
        const std::array<int, 6>& stencil =
            support.restrict_stencils[static_cast<std::size_t>(q)];
        for (int r = 0; r < static_cast<int>(stencil.size()); ++r) {
            RestrictHalfCorrectionRowBasis2D& row =
                basis.rows[static_cast<std::size_t>(r)];
            row.grid_node = stencil[static_cast<std::size_t>(r)];
            const Eigen::Vector2d pt = grid_point(grid, row.grid_node);
            row.full_basis = taylor_basis_at_point_2d(full_center, pt);
            row.patch_basis = taylor_basis_at_point_2d(patch_center, pt);
        }
    }
    return basis_by_point;
}

} // namespace

LaplaceQuadraticPanelCenterRestrict2D::LaplaceQuadraticPanelCenterRestrict2D(
    const GridPair2D& grid_pair,
    int               stencil_radius)
    : grid_pair_(grid_pair)
    , stencil_radius_(stencil_radius)
    , support_(build_laplace_correction_support_2d(
          grid_pair,
          "LaplaceQuadraticPanelCenterRestrict2D"))
    , fit_inverse_by_point_(
          build_restrict_fit_inverse_by_point(grid_pair_, support_))
    , interface_panel_by_point_(
          build_interface_panel_by_point(grid_pair_.interface()))
    , patch_boundary_center_by_point_(
          build_boundary_center_by_point(grid_pair_,
                                         support_,
                                         interface_panel_by_point_,
                                         true))
    , full_boundary_center_by_point_(
          build_boundary_center_by_point(grid_pair_,
                                         support_,
                                         interface_panel_by_point_,
                                         false))
    , half_correction_basis_by_point_(
          build_restrict_half_correction_basis_by_point(
              grid_pair_,
              support_,
              patch_boundary_center_by_point_))
{
    if (stencil_radius_ < 1)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D stencil_radius must be positive");
}

std::vector<LocalPoly2D> LaplaceQuadraticPanelCenterRestrict2D::apply(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    return apply_impl(bulk_solution, spread_result, TraceRoute::Average);
}

std::vector<LocalPoly2D>
LaplaceQuadraticPanelCenterRestrict2D::apply_interior_virtual(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    if (spread_result.sd_corner_lifting.enabled
        || !spread_result.corner_patch_corrections.empty()
        || !spread_result.patch_regularized_correction_polys.empty()) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterRestrict2D interior-virtual route "
            "does not support corner or singular corrections");
    }
    return apply_impl(bulk_solution,
                      spread_result,
                      TraceRoute::InteriorVirtual);
}

std::vector<LocalPoly2D>
LaplaceQuadraticPanelCenterRestrict2D::apply_exterior_virtual(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    if (spread_result.sd_corner_lifting.enabled
        || !spread_result.corner_patch_corrections.empty()
        || !spread_result.patch_regularized_correction_polys.empty()) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterRestrict2D exterior-virtual route "
            "does not support corner or singular corrections");
    }
    return apply_impl(bulk_solution,
                      spread_result,
                      TraceRoute::ExteriorVirtual);
}

std::vector<LocalPoly2D> LaplaceQuadraticPanelCenterRestrict2D::apply_impl(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result,
    TraceRoute                   trace_route) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int expected_centers = kQuadraticExpansionCentersPerPanel * iface.num_panels();

    if (bulk_solution.size() != grid.num_dofs())
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D bulk_solution size must equal grid DOF count");
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D requires P2 quadratic 3-point panels");
    }

    if (!spread_result.sd_corner_lifting.enabled
        && !spread_result.corner_patch_corrections.empty()
        && !average_restore_basis_cache_ready_) {
        average_restore_basis_by_point_ =
            build_corner_patch_average_restore_basis_by_point(
                iface,
                spread_result.corner_patch_corrections);
        average_restore_basis_cache_ready_ = true;
    }

    const auto correction_evaluator =
        make_restrict_correction_evaluator(grid_pair_,
                                           spread_result,
                                           expected_centers,
                                           interface_panel_by_point_,
                                           patch_boundary_center_by_point_,
                                           full_boundary_center_by_point_,
                                           half_correction_basis_by_point_,
                                           average_restore_basis_by_point_);

    std::vector<LocalPoly2D> result(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        result[q] = fit_at_interface_point(bulk_solution,
                                           q,
                                           *correction_evaluator,
                                           trace_route);
    }

    return result;
}

std::vector<LocalPoly2D> LaplaceQuadraticPanelCenterRestrict2D::apply(
    const Eigen::VectorXd&          bulk_solution,
    const std::vector<LocalPoly2D>& correction_polys) const
{
    LaplaceSpreadResult2D spread_result;
    spread_result.correction_method =
        LaplaceCorrectionMethod2D::NearestExpansionCenter;
    spread_result.correction_polys = correction_polys;
    return apply(bulk_solution, spread_result);
}

int restrict_patch_for_stencil(const GridPair2D& grid_pair,
                               const std::array<int, 6>& stencil_nodes,
                               int q)
{
    (void)stencil_nodes;
    const Interface2D& iface = grid_pair.interface();
    return patch_id_of_boundary_iter_point_2d(iface, q);
}

LocalPoly2D LaplaceQuadraticPanelCenterRestrict2D::fit_at_interface_point(
    const Eigen::VectorXd&                   bulk_solution,
    int                                      q,
    const LaplaceRestrictCorrectionEvaluator2D& correction_evaluator,
    TraceRoute                               trace_route) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const Eigen::Vector2d center = interface_point(iface, q);
    const std::array<int, 6>& stencil_nodes = support_.restrict_stencils[q];
    const int anchor_node = grid_pair_.closest_bulk_node(q);
    const int restrict_patch =
        restrict_patch_for_stencil(grid_pair_, stencil_nodes, q);

    Eigen::Matrix<double, 6, 1> rhs;
    for (int r = 0; r < static_cast<int>(stencil_nodes.size()); ++r) {
        const int idx = stencil_nodes[r];
        const Eigen::Vector2d pt = grid_point(grid, idx);
        const double raw_val = bulk_solution[idx];
        const double singular_restore =
            correction_evaluator.bulk_singular_restore(
                restrict_patch, idx, pt);
        double val = raw_val + singular_restore;
        const double correction =
            correction_evaluator.half_correction(
                q, restrict_patch, anchor_node, idx, r, pt);
        const int label = grid_pair_.domain_label(idx);
        double correction_scale = 0.0;
        switch (trace_route) {
        case TraceRoute::Average:
            correction_scale = label == 0 ? 1.0 : -1.0;
            break;
        case TraceRoute::InteriorVirtual:
            correction_scale = label == 0 ? 2.0 : 0.0;
            break;
        case TraceRoute::ExteriorVirtual:
            correction_scale = label == 0 ? 0.0 : -2.0;
            break;
        }
        val += correction_scale * correction;
        rhs[r] = val;
    }

    LocalPoly2D poly;
    poly.center = center;
    poly.coeffs = fit_inverse_by_point_[static_cast<std::size_t>(q)] * rhs;
    if (trace_route == TraceRoute::Average)
        correction_evaluator.restore_physical_average_trace(q, poly);
    return poly;
}

} // namespace kfbim
