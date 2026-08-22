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
#include "laplace_arc_length_bspline_crossing_jet_2d.hpp"
#include "laplace_crossing_local_polynomial_2d.hpp"
#include "laplace_nurbs_density_trace_state_2d.hpp"
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

bool make_interface_trace_crossing_owner(
    const Interface2D& iface,
    int point,
    int panel,
    P2CrossingOwner2D& owner)
{
    if (point < 0 || point >= iface.num_points()
        || panel < 0 || panel >= iface.num_panels()) {
        return false;
    }
    int local = -1;
    for (int candidate = 0; candidate < iface.points_per_panel(); ++candidate) {
        if (iface.point_index(panel, candidate) == point) {
            local = candidate;
            break;
        }
    }
    if (local < 0 || local >= 3)
        return false;

    owner = {};
    owner.panel_index = panel;
    owner.local_s =
        geometry2d::kP2NodeS[static_cast<std::size_t>(local)];
    owner.edge_parameter = 0.0;
    owner.crossing_point =
        geometry2d::panel_point(iface, panel, owner.local_s);
    owner.crossing_normal =
        geometry2d::panel_normal(iface, panel, owner.local_s);
    owner.crossing_residual =
        (owner.crossing_point
         - iface.points().row(point).transpose()).norm();
    owner.exact_intersection_count = 1;
    owner.transverse_intersection_count = 1;
    owner.status = P2CrossingOwnerStatus2D::ExactIntersection;
    return owner.crossing_point.allFinite()
        && owner.crossing_normal.allFinite();
}

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
        , spread_result_(spread_result)
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
    {
        build_crossing_trace_point_polynomials();
    }

    double half_correction(int boundary_iter_point,
                           int restrict_patch,
                           int anchor_node,
                           int grid_node,
                           int stencil_row,
                           Eigen::Vector2d pt) const override
    {
        if (!sd_corner_lifting_.enabled && restrict_patch < 0
            && boundary_iter_point >= 0
            && boundary_iter_point
                   < static_cast<int>(
                       crossing_trace_point_poly_valid_.size())
            && crossing_trace_point_poly_valid_[
                   static_cast<std::size_t>(boundary_iter_point)]) {
            return 0.5
                 * crossing_trace_point_polys_[
                       static_cast<std::size_t>(boundary_iter_point)]
                       .evaluate(pt);
        }
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
    void build_crossing_trace_point_polynomials()
    {
        const Interface2D& iface = grid_pair_.interface();
        crossing_trace_point_polys_.resize(
            static_cast<std::size_t>(iface.num_points()));
        crossing_trace_point_poly_valid_.assign(
            static_cast<std::size_t>(iface.num_points()), 0);
        if (spread_result_.correction_method
                != LaplaceCorrectionMethod2D::CrossingOwner
            || spread_result_.u_jump.size() != iface.num_points()
            || spread_result_.un_jump.size() != iface.num_points()
            || spread_result_.rhs_jump.size() != iface.num_points()) {
            return;
        }

        for (int q = 0; q < iface.num_points(); ++q) {
            const int panel = q >= 0
                    && q < static_cast<int>(
                        interface_panel_by_point_.size())
                ? interface_panel_by_point_[static_cast<std::size_t>(q)]
                : -1;
            P2CrossingOwner2D owner;
            if (!make_interface_trace_crossing_owner(
                    iface, q, panel, owner)) {
                continue;
            }

            if (spread_result_.direct_nurbs_density_trace_state
                && spread_result_.direct_nurbs_density_trace_state
                       ->can_evaluate(owner)) {
                crossing_trace_point_polys_[static_cast<std::size_t>(q)] =
                    spread_result_.direct_nurbs_density_trace_state
                        ->build_local_polynomial(
                            owner,
                            spread_result_.rhs_jump,
                            spread_result_.alpha);
                crossing_trace_point_poly_valid_[
                    static_cast<std::size_t>(q)] = 1;
                continue;
            }

            if ((spread_result_.crossing_jet_scheme
                     == LaplaceCrossingJetScheme2D::
                            ArcLengthBSplineCrossingJet
                 || spread_result_.crossing_jet_scheme
                     == LaplaceCrossingJetScheme2D::
                            NurbsSameParameterCrossingJet)
                && spread_result_.arc_length_bspline_crossing_jet_plan
                && spread_result_.arc_length_bspline_trace_state
                && spread_result_.arc_length_bspline_crossing_jet_plan
                       ->can_evaluate(
                           owner,
                           spread_result_.crossing_trace_stencil)) {
                crossing_trace_point_polys_[static_cast<std::size_t>(q)] =
                    spread_result_.arc_length_bspline_crossing_jet_plan
                        ->build_local_polynomial(
                        owner,
                        *spread_result_.arc_length_bspline_trace_state,
                        spread_result_.rhs_jump,
                        spread_result_.alpha);
                crossing_trace_point_poly_valid_[
                    static_cast<std::size_t>(q)] = 1;
                continue;
            }

            if (can_build_laplace_p2_crossing_local_polynomial_2d(
                    iface,
                    owner,
                    spread_result_.crossing_trace_stencil)) {
                crossing_trace_point_polys_[static_cast<std::size_t>(q)] =
                    build_laplace_p2_crossing_local_polynomial_2d(
                        iface,
                        owner,
                        spread_result_.u_jump,
                        spread_result_.un_jump,
                        spread_result_.rhs_jump,
                        spread_result_.alpha,
                        spread_result_.crossing_trace_stencil);
                crossing_trace_point_poly_valid_[
                    static_cast<std::size_t>(q)] = 1;
            }
        }
    }

    const GridPair2D& grid_pair_;
    const LaplaceSpreadResult2D& spread_result_;
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
    std::vector<LaplaceP2CrossingLocalPolynomial2D>
        crossing_trace_point_polys_;
    std::vector<char> crossing_trace_point_poly_valid_;
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
            == LaplaceCorrectionMethod2D::NearestExpansionCenter
        || spread_result.correction_method
               == LaplaceCorrectionMethod2D::CrossingOwner) {
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

struct LaplaceP2CrossingOwnerJointPolynomialRestrict2D::Impl {
    struct InterpolationNode {
        int grid_node = -1;
        double weight = 0.0;
        int correction_center = -1;
        P2CrossingOwner2D correction_crossing;
        double correction_scale = 0.0;
        int secondary_correction_center = -1;
        P2CrossingOwner2D secondary_correction_crossing;
        double secondary_correction_scale = 0.0;
        Eigen::Vector2d point = Eigen::Vector2d::Zero();
    };

    struct TraceSample {
        Eigen::Vector2d query = Eigen::Vector2d::Zero();
        std::vector<InterpolationNode> nodes;
        int query_correction_center = -1;
        double query_correction_scale = 0.0;
        bool virtual_side_flipped = false;
    };

    struct TraceReferencePlan {
        LaplaceNurbsTraceReferencePoint2D reference;
        // Stored as [inside layers..., outside layers...].
        std::vector<TraceSample> samples;
    };

    explicit Impl(const GridPair2D& grid_pair,
                  LaplaceP2JointPolynomialRestrictOptions2D options)
        : grid_pair_(grid_pair)
        , options_(std::move(options))
    {
        const CartesianGrid2D& grid = grid_pair.grid();
        const Interface2D& iface = grid_pair.interface();
        if (!geometry2d::is_quadratic_lagrange_panel_layout(iface)) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D requires P2 quadratic panels");
        }
        degree_ = static_cast<int>(options_.degree);
        if (degree_ != 2 && degree_ != 3) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D degree must be quadratic or cubic");
        }
        if (options_.jump_evaluation
                == LaplaceP2JointJumpEvaluation2D::
                       NearestIncidentPanelCenterCauchy
            && degree_ != 2) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D panel-center Cauchy jump evaluation currently requires quadratic restrict");
        }
        if (options_.quadratic_grid_stencil
                != LaplaceP2QuadraticGridStencil2D::
                       TensorProduct3x3
            && degree_ != 2) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D same-side cross stencil requires quadratic restrict");
        }
        if (options_.quadratic_grid_stencil
                != LaplaceP2QuadraticGridStencil2D::
                       TensorProduct3x3
            && options_.allow_virtual_side_flip) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D same-side cross stencil cannot be combined with virtual-side flip");
        }
        grid_side_ = degree_ + 1;
        layer_count_ = options_.normal_layer_count;
        if (layer_count_ != degree_ + 1) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D requires three layers for quadratic or four layers for cubic");
        }
        if (options_.normal_fit
                == LaplaceP2JointNormalFit2D::UnifiedQuadratic
            && (degree_ != 2 || layer_count_ != 3)) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D unified normal P2 requires quadratic degree and three layers");
        }
        if (options_.virtual_trace_route
                == LaplaceP2JointVirtualTraceRoute2D::
                       InterfaceDofCauchyExterior
            && (options_.jump_evaluation
                    != LaplaceP2JointJumpEvaluation2D::
                           InterfaceTracePointCauchy
                || options_.normal_fit
                    != LaplaceP2JointNormalFit2D::UnifiedQuadratic
                || options_.quadratic_grid_stencil
                    != LaplaceP2QuadraticGridStencil2D::
                           UnifiedCompleteP2SharedSidePolynomial)) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D DOF-Cauchy exterior continuation requires the unified spatial P2, trace-point Cauchy jump, and unified normal P2 options");
        }
        normal_sample_count_ = 2 * layer_count_;
        grid_nodes_per_sample_ =
            uses_six_point_cross_stencil()
                ? 6
                : grid_side_ * grid_side_;
        const auto spacing = grid.spacing();
        h_ = spacing[0];
        if (!(h_ > 0.0)
            || std::abs(spacing[1] - h_)
                   > 64.0 * std::numeric_limits<double>::epsilon()
                       * std::max(spacing[1], h_)) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D requires a uniform square grid");
        }
        for (int layer = 0; layer < layer_count_; ++layer) {
            const double value =
                options_.normal_layers[static_cast<std::size_t>(layer)];
            if (!(value > 0.0) || !std::isfinite(value)
                || (layer > 0
                    && !(value
                         > options_.normal_layers[
                             static_cast<std::size_t>(layer - 1)]))) {
                throw std::invalid_argument(
                    "LaplaceP2CrossingOwnerJointPolynomialRestrict2D normal layers must be finite, positive, and increasing");
            }
        }

        build_joint_fit();
        build_trace_samples(grid_pair);
    }

    int trace_sample_index(int q, int side, int layer) const
    {
        return (q * 2 + side) * layer_count_ + layer;
    }

    double physical_sample(
        const Eigen::VectorXd& bulk_solution,
        const LaplaceSpreadResult2D& spread_result,
        int q,
        int side,
        int layer,
        const LaplaceP2CrossingLocalPolynomial2D*
            interface_dof_cauchy = nullptr,
        int interface_dof_branch = -1) const
    {
        const TraceSample& sample =
            trace_samples_[static_cast<std::size_t>(
                trace_sample_index(q, side, layer))];
        return physical_sample(
            bulk_solution,
            spread_result,
            sample,
            interface_dof_cauchy,
            interface_dof_branch);
    }

    double physical_sample(
        const Eigen::VectorXd& bulk_solution,
        const LaplaceSpreadResult2D& spread_result,
        const TraceSample& sample,
        const LaplaceP2CrossingLocalPolynomial2D*
            interface_dof_cauchy = nullptr,
        int interface_dof_branch = -1) const
    {
        const auto center_correction_value =
            [&](int center, const Eigen::Vector2d& point) {
                if (center < 0
                    || center
                           >= static_cast<int>(
                               spread_result.correction_polys.size())) {
                    throw std::invalid_argument(
                        "LaplaceP2CrossingOwnerJointPolynomialRestrict2D crossing owner is outside correction polynomial storage");
                }
                const LocalPoly2D& correction =
                    spread_result.correction_polys[
                        static_cast<std::size_t>(center)];
                if (correction.coeffs.size() != 6
                    && correction.coeffs.size() != 10) {
                    throw std::invalid_argument(
                        "LaplaceP2CrossingOwnerJointPolynomialRestrict2D requires complete quadratic or cubic Cauchy polynomials");
                }
                return evaluate_taylor_poly_2d(correction, point);
            };
        const auto correction_value =
            [&](int center,
                const P2CrossingOwner2D& crossing,
                const Eigen::Vector2d& point) {
                if (interface_dof_cauchy != nullptr) {
                    if (interface_dof_branch < 0)
                        return interface_dof_cauchy->evaluate(point);
                    const auto& state =
                        spread_result.direct_nurbs_density_trace_state;
                    if (!state) {
                        return interface_dof_cauchy->evaluate(point);
                    }
                    const std::shared_ptr<const NurbsDensitySpace2D>& space =
                        state->phi_space()
                        ? state->phi_space() : state->psi_space();
                    if (space && state->can_evaluate(crossing)
                        && space->resolve_crossing(crossing).branch
                               == interface_dof_branch) {
                        return interface_dof_cauchy->evaluate(point);
                    }
                }
                const Interface2D& iface = grid_pair_.interface();
                if (spread_result.direct_nurbs_density_trace_state
                    && spread_result.direct_nurbs_density_trace_state
                           ->can_evaluate(crossing)) {
                    return spread_result.direct_nurbs_density_trace_state
                        ->build_local_polynomial(
                            crossing,
                            spread_result.rhs_jump,
                            spread_result.alpha)
                        .evaluate(point);
                }
                if ((spread_result.crossing_jet_scheme
                         == LaplaceCrossingJetScheme2D::
                                ArcLengthBSplineCrossingJet
                     || spread_result.crossing_jet_scheme
                         == LaplaceCrossingJetScheme2D::
                                NurbsSameParameterCrossingJet)
                    && spread_result.arc_length_bspline_crossing_jet_plan
                    && spread_result.arc_length_bspline_trace_state
                    && spread_result.arc_length_bspline_crossing_jet_plan
                           ->can_evaluate(
                               crossing,
                               spread_result.crossing_trace_stencil)) {
                    return spread_result
                        .arc_length_bspline_crossing_jet_plan
                        ->build_local_polynomial(
                            crossing,
                            *spread_result.arc_length_bspline_trace_state,
                            spread_result.rhs_jump,
                            spread_result.alpha)
                        .evaluate(point);
                }
                if (can_build_laplace_p2_crossing_local_polynomial_2d(
                        iface,
                        crossing,
                        spread_result.crossing_trace_stencil)) {
                    return build_laplace_p2_crossing_local_polynomial_2d(
                               iface,
                               crossing,
                               spread_result.u_jump,
                               spread_result.un_jump,
                               spread_result.rhs_jump,
                               spread_result.alpha,
                               spread_result.crossing_trace_stencil)
                        .evaluate(point);
                }
                if (uses_unified_spatial_p2()) {
                    throw std::runtime_error(
                        "LaplaceP2CrossingOwnerJointPolynomialRestrict2D unified spatial P2 forbids expansion-center correction fallback");
                }
                return center_correction_value(center, point);
            };
        double value = 0.0;
        for (const InterpolationNode& node : sample.nodes) {
            double node_value = bulk_solution[node.grid_node];
            if (node.correction_center >= 0) {
                node_value += node.correction_scale
                    * correction_value(
                        node.correction_center,
                        node.correction_crossing,
                        node.point);
            }
            if (node.secondary_correction_center >= 0) {
                node_value += node.secondary_correction_scale
                    * correction_value(
                        node.secondary_correction_center,
                        node.secondary_correction_crossing,
                        node.point);
            }
            value += node.weight * node_value;
        }
        if (sample.query_correction_center >= 0) {
            value += sample.query_correction_scale
                * center_correction_value(
                    sample.query_correction_center, sample.query);
        }
        return value;
    }

    bool uses_center_cauchy_jump() const
    {
        return options_.jump_evaluation
            == LaplaceP2JointJumpEvaluation2D::
                   NearestIncidentPanelCenterCauchy;
    }

    bool uses_interface_trace_point_cauchy_jump() const
    {
        return options_.jump_evaluation
            == LaplaceP2JointJumpEvaluation2D::
                   InterfaceTracePointCauchy;
    }

    bool uses_unified_normal_fit() const
    {
        return options_.normal_fit
            == LaplaceP2JointNormalFit2D::UnifiedQuadratic;
    }

    bool uses_interface_dof_cauchy_exterior_route() const
    {
        return options_.virtual_trace_route
            == LaplaceP2JointVirtualTraceRoute2D::
                   InterfaceDofCauchyExterior;
    }

    bool uses_unified_spatial_p2() const
    {
        return options_.quadratic_grid_stencil
            == LaplaceP2QuadraticGridStencil2D::
                   UnifiedCompleteP2SharedSidePolynomial;
    }

    bool uses_six_point_cross_stencil() const
    {
        return options_.quadratic_grid_stencil
            != LaplaceP2QuadraticGridStencil2D::
                   TensorProduct3x3;
    }

    bool uses_grid_edge_cross_stencil() const
    {
        return options_.quadratic_grid_stencil
                   == LaplaceP2QuadraticGridStencil2D::
                          SameSideCrossGridEdgeDiagonal
            || options_.quadratic_grid_stencil
                   == LaplaceP2QuadraticGridStencil2D::
                          SameSideCrossGridEdgeDiagonalSharedSidePolynomial;
    }

    bool uses_shared_side_spatial_polynomial() const
    {
        return options_.quadratic_grid_stencil
                   == LaplaceP2QuadraticGridStencil2D::
                          SameSideCrossGridEdgeDiagonalSharedSidePolynomial
            || uses_unified_spatial_p2();
    }

    bool has_incident_center(int q) const
    {
        return q >= 0
            && q < static_cast<int>(trace_center_by_point_.size())
            && trace_center_by_point_[static_cast<std::size_t>(q)] >= 0;
    }

    double center_cauchy_jump(
        const LaplaceSpreadResult2D& spread_result,
        int q,
        int side,
        int layer) const
    {
        if (!has_incident_center(q)) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D trace point has no incident P2 panel center");
        }
        const int center =
            trace_center_by_point_[static_cast<std::size_t>(q)];
        if (center < 0
            || center
                   >= static_cast<int>(
                       spread_result.correction_polys.size())) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D incident panel center is outside correction polynomial storage");
        }
        const LocalPoly2D& correction =
            spread_result.correction_polys[
                static_cast<std::size_t>(center)];
        if (correction.coeffs.size() != 6
            && correction.coeffs.size() != 10) {
            throw std::invalid_argument(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D panel-center jump evaluation requires complete quadratic or cubic Cauchy polynomials");
        }
        const TraceSample& sample =
            trace_samples_[static_cast<std::size_t>(
                trace_sample_index(q, side, layer))];
        return evaluate_taylor_poly_2d(correction, sample.query);
    }

    LaplaceP2CrossingLocalPolynomial2D
    interface_trace_point_cauchy_polynomial(
        const LaplaceSpreadResult2D& spread_result,
        int q) const
    {
        const Interface2D& iface = grid_pair_.interface();
        if (q < 0
            || q >= static_cast<int>(trace_panel_by_point_.size())
            || trace_panel_by_point_[static_cast<std::size_t>(q)] < 0) {
            throw std::runtime_error(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D unified spatial P2 trace point has no incident panel");
        }
        P2CrossingOwner2D owner;
        if (!make_interface_trace_crossing_owner(
                iface,
                q,
                trace_panel_by_point_[static_cast<std::size_t>(q)],
                owner)) {
            throw std::runtime_error(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D failed to construct its interface trace-point owner");
        }

        if (spread_result.direct_nurbs_density_trace_state
            && spread_result.direct_nurbs_density_trace_state
                   ->can_evaluate(owner)) {
            return spread_result.direct_nurbs_density_trace_state
                ->build_local_polynomial(
                    owner,
                    spread_result.rhs_jump,
                    spread_result.alpha);
        }

        if ((spread_result.crossing_jet_scheme
                 == LaplaceCrossingJetScheme2D::
                        ArcLengthBSplineCrossingJet
             || spread_result.crossing_jet_scheme
                 == LaplaceCrossingJetScheme2D::
                        NurbsSameParameterCrossingJet)
            && spread_result.arc_length_bspline_crossing_jet_plan
            && spread_result.arc_length_bspline_trace_state
            && spread_result.arc_length_bspline_crossing_jet_plan
                   ->can_evaluate(
                       owner,
                       spread_result.crossing_trace_stencil)) {
            return spread_result.arc_length_bspline_crossing_jet_plan
                ->build_local_polynomial(
                owner,
                *spread_result.arc_length_bspline_trace_state,
                spread_result.rhs_jump,
                spread_result.alpha);
        }
        if (can_build_laplace_p2_crossing_local_polynomial_2d(
                iface,
                owner,
                spread_result.crossing_trace_stencil)) {
            return build_laplace_p2_crossing_local_polynomial_2d(
                iface,
                owner,
                spread_result.u_jump,
                spread_result.un_jump,
                spread_result.rhs_jump,
                spread_result.alpha,
                spread_result.crossing_trace_stencil);
        }
        throw std::runtime_error(
            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D unified spatial P2 forbids interface trace-point jump fallback");
    }

    Eigen::Vector2d trace_sample_query(int q, int side, int layer) const
    {
        return trace_samples_[static_cast<std::size_t>(
            trace_sample_index(q, side, layer))].query;
    }

    std::pair<double, double> joint_fit(
        const Eigen::VectorXd& samples) const
    {
        return {c0_weights_.dot(samples),
                c1_weights_.dot(samples) / h_};
    }

    void configure_trace_references(
        std::vector<LaplaceNurbsTraceReferencePoint2D> references)
    {
        if (trace_references_configured_) {
            throw std::logic_error(
                "Laplace P2 reference traces may only be configured once");
        }
        if (references.empty()) {
            throw std::invalid_argument(
                "Laplace P2 reference trace set must not be empty");
        }
        if (!uses_interface_dof_cauchy_exterior_route()
            || !uses_unified_spatial_p2()
            || !uses_unified_normal_fit()) {
            throw std::invalid_argument(
                "exact-NURBS reference traces require the DOF-Cauchy exterior, unified spatial P2, and unified normal P2 restrict route");
        }
        const int wrong_side_before = diagnostics_.wrong_side_nodes;
        const int corrected_before = diagnostics_.corrected_nodes;
        const int exact_before = diagnostics_.exact_crossing_owners;
        const int gap_before = diagnostics_.gap_fallback_owners;
        const int unresolved_before =
            diagnostics_.unresolved_gap_fallback_owners;
        const int endpoint_before = diagnostics_.endpoint_fallback_owners;
        const int unified_before = diagnostics_.unified_p2_stencils;
        const int relocated_before =
            diagnostics_.unified_p2_relocated_stencils;
        const int rejected_before =
            diagnostics_.unified_p2_candidate_rejections;
        const Interface2D& iface = grid_pair_.interface();
        if (!iface.has_panel_geometry()) {
            throw std::invalid_argument(
                "exact-NURBS reference traces require panel geometry");
        }
        const auto* geometry = dynamic_cast<const
            geometry2d::NurbsBoundaryPanelGeometry2D*>(
                &iface.panel_geometry());
        if (geometry == nullptr) {
            throw std::invalid_argument(
                "exact-NURBS reference traces require NURBS geometry");
        }

        trace_references_ = std::move(references);
        trace_reference_plans_.reserve(trace_references_.size());
        const double point_tolerance = 2.0e-11 * std::max(1.0, h_);
        const double selection_layer =
            0.5 * (options_.normal_layers.front()
                   + options_.normal_layers[
                       static_cast<std::size_t>(layer_count_ - 1)]);
        for (LaplaceNurbsTraceReferencePoint2D& reference
             : trace_references_) {
            if (reference.branch < 0
                || reference.branch >= geometry->num_spans()
                || !std::isfinite(reference.parameter)
                || !(reference.physical_weight > 0.0)
                || !std::isfinite(reference.physical_weight)
                || !reference.point.allFinite()
                || !reference.normal.allFinite()
                || !(reference.normal.norm() > 1.0e-14)) {
                throw std::invalid_argument(
                    "exact-NURBS reference trace descriptor is invalid");
            }
            const geometry2d::NurbsBoundaryGeometry2D exact =
                geometry->evaluate_on_span(
                    reference.branch, reference.parameter);
            if ((reference.point - exact.point).norm() > point_tolerance
                || reference.normal.normalized().dot(exact.normal)
                       < 1.0 - 2.0e-11) {
                throw std::invalid_argument(
                    "exact-NURBS reference trace geometry is inconsistent");
            }
            reference.point = exact.point;
            reference.normal = exact.normal;

            TraceReferencePlan plan;
            plan.reference = reference;
            plan.samples.resize(
                static_cast<std::size_t>(normal_sample_count_));
            for (int side = 0; side < 2; ++side) {
                const bool desired_inside = side == 0;
                const double side_sign = desired_inside ? -1.0 : 1.0;
                const Eigen::Vector2d selection_query =
                    reference.point
                    + side_sign * selection_layer * h_
                          * reference.normal;
                for (int layer = 0; layer < layer_count_; ++layer) {
                    const Eigen::Vector2d query =
                        reference.point
                        + side_sign
                              * options_.normal_layers[
                                  static_cast<std::size_t>(layer)]
                              * h_ * reference.normal;
                    const TraceSample* layer_zero = layer == 0
                        ? nullptr
                        : &plan.samples[static_cast<std::size_t>(
                              side * layer_count_)];
                    plan.samples[static_cast<std::size_t>(
                        side * layer_count_ + layer)] =
                        build_unified_p2_shared_side_sample(
                            grid_pair_,
                            selection_query,
                            query,
                            desired_inside,
                            -1,
                            side,
                            layer,
                            &reference.point,
                            &reference.normal,
                            layer_zero);
                }
            }
            trace_reference_plans_.push_back(std::move(plan));
        }
        diagnostics_.reference_trace_points =
            static_cast<int>(trace_reference_plans_.size());
        diagnostics_.reference_trace_samples =
            diagnostics_.reference_trace_points * normal_sample_count_;
        diagnostics_.reference_wrong_side_nodes =
            diagnostics_.wrong_side_nodes - wrong_side_before;
        diagnostics_.reference_corrected_nodes =
            diagnostics_.corrected_nodes - corrected_before;
        diagnostics_.reference_exact_crossing_owners =
            diagnostics_.exact_crossing_owners - exact_before;
        diagnostics_.reference_gap_fallback_owners =
            diagnostics_.gap_fallback_owners - gap_before;
        diagnostics_.reference_unresolved_gap_fallback_owners =
            diagnostics_.unresolved_gap_fallback_owners
            - unresolved_before;
        diagnostics_.reference_endpoint_fallback_owners =
            diagnostics_.endpoint_fallback_owners - endpoint_before;
        diagnostics_.reference_unified_p2_stencils =
            diagnostics_.unified_p2_stencils - unified_before;
        diagnostics_.reference_unified_p2_relocated_stencils =
            diagnostics_.unified_p2_relocated_stencils - relocated_before;
        diagnostics_.reference_unified_p2_candidate_rejections =
            diagnostics_.unified_p2_candidate_rejections - rejected_before;
        for (const TraceReferencePlan& plan : trace_reference_plans_) {
            for (const TraceSample& sample : plan.samples) {
                diagnostics_.reference_interpolation_nodes +=
                    static_cast<int>(sample.nodes.size());
                double weight_l1 = 0.0;
                for (const InterpolationNode& node : sample.nodes)
                    weight_l1 += std::abs(node.weight);
                diagnostics_.reference_max_six_point_weight_l1 = std::max(
                    diagnostics_.reference_max_six_point_weight_l1,
                    weight_l1);
            }
        }
        trace_references_configured_ = true;
    }

    LaplaceRestrictedReferenceTrace2D
    apply_exterior_virtual_at_references(
        const Eigen::VectorXd& bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const
    {
        if (!trace_references_configured_
            || trace_reference_plans_.size() != trace_references_.size()) {
            throw std::logic_error(
                "Laplace P2 reference traces have not been configured");
        }
        if (!spread_result.direct_nurbs_density_trace_state) {
            throw std::invalid_argument(
                "exact-NURBS reference traces require a direct density state");
        }
        LaplaceRestrictedReferenceTrace2D result;
        const int count = static_cast<int>(trace_reference_plans_.size());
        result.value.resize(count);
        result.normal_derivative.resize(count);
        for (int index = 0; index < count; ++index) {
            const TraceReferencePlan& plan =
                trace_reference_plans_[static_cast<std::size_t>(index)];
            const NurbsDensityLocation2D location{
                plan.reference.branch, plan.reference.parameter};
            const LaplaceP2CrossingLocalPolynomial2D jump =
                spread_result.direct_nurbs_density_trace_state
                    ->build_local_polynomial_at_location(
                        location,
                        spread_result.rhs_jump,
                        spread_result.alpha);
            Eigen::VectorXd exterior_samples(normal_sample_count_);
            for (int layer = 0; layer < layer_count_; ++layer) {
                const TraceSample& inside_sample =
                    plan.samples[static_cast<std::size_t>(layer)];
                const TraceSample& outside_sample =
                    plan.samples[static_cast<std::size_t>(
                        layer_count_ + layer)];
                const double inside = physical_sample(
                    bulk_solution,
                    spread_result,
                    inside_sample,
                    &jump,
                    plan.reference.branch);
                const double outside = physical_sample(
                    bulk_solution,
                    spread_result,
                    outside_sample,
                    &jump,
                    plan.reference.branch);
                exterior_samples[layer] =
                    inside - jump.evaluate(inside_sample.query);
                exterior_samples[layer_count_ + layer] = outside;
            }
            const std::pair<double, double> fit =
                joint_fit(exterior_samples);
            result.value[index] = fit.first;
            result.normal_derivative[index] = fit.second;
        }
        return result;
    }

    const std::vector<LaplaceNurbsTraceReferencePoint2D>&
    trace_references() const
    {
        return trace_references_;
    }

    const LaplaceP2JointPolynomialRestrictDiagnostics2D& diagnostics() const
    {
        return diagnostics_;
    }

    double h() const { return h_; }
    const LaplaceP2JointPolynomialRestrictOptions2D& options() const
    {
        return options_;
    }
    int layer_count() const { return layer_count_; }
    int normal_sample_count() const { return normal_sample_count_; }

private:
    static bool same_crossing(const P2CrossingOwner2D& lhs,
                              const P2CrossingOwner2D& rhs)
    {
        if (lhs.panel_index != rhs.panel_index
            || lhs.status != rhs.status
            || lhs.exact_intersection_count
                   != rhs.exact_intersection_count
            || lhs.transverse_intersection_count
                   != rhs.transverse_intersection_count
            || lhs.explicit_gap_intersection
                   != rhs.explicit_gap_intersection) {
            return false;
        }
        if (lhs.panel_index < 0)
            return true;
        return std::abs(lhs.local_s - rhs.local_s) <= 1.0e-12
            && (lhs.crossing_point - rhs.crossing_point).norm()
                   <= 1.0e-12;
    }

    static std::vector<double> lagrange_weights(double fraction,
                                                int grid_side)
    {
        std::vector<double> nodes(static_cast<std::size_t>(grid_side));
        std::vector<double> weights(
            static_cast<std::size_t>(grid_side), 1.0);
        for (int i = 0; i < grid_side; ++i)
            nodes[static_cast<std::size_t>(i)] =
                static_cast<double>(i - 1);
        for (int i = 0; i < grid_side; ++i) {
            for (int j = 0; j < grid_side; ++j) {
                if (i == j)
                    continue;
                weights[static_cast<std::size_t>(i)] *=
                    (fraction - nodes[static_cast<std::size_t>(j)])
                    / (nodes[static_cast<std::size_t>(i)]
                       - nodes[static_cast<std::size_t>(j)]);
            }
        }
        return weights;
    }

    std::vector<int> nearby_same_side_nodes(
        const GridPair2D& grid_pair,
        const Eigen::Vector2d& query,
        bool desired_inside) const
    {
        const CartesianGrid2D& grid = grid_pair.grid();
        const auto dims = grid.dof_dims();
        const auto spacing = grid.spacing();
        const Eigen::Vector2d first =
            structured_grid::point(grid, grid.index(0, 0));
        const double grid_x = (query[0] - first[0]) / spacing[0];
        const double grid_y = (query[1] - first[1]) / spacing[1];
        const int base_i = std::max(
            0,
            std::min(dims[0] - 1,
                     static_cast<int>(std::llround(grid_x))));
        const int base_j = std::max(
            0,
            std::min(dims[1] - 1,
                     static_cast<int>(std::llround(grid_y))));

        // The first distance shell is the requested nearest same-side node.
        // A strict unique-crossing diagonal need not exist for that shell
        // (for example after a sub-grid translation), so retain a small,
        // ordered set of subsequent shells.  A center farther than four grid
        // steps cannot provide a genuinely local six-point stencil for the
        // normal layers used here.
        constexpr int kMaximumAnchorRadius = 4;
        const int i_min =
            std::max(0, base_i - kMaximumAnchorRadius);
        const int i_max =
            std::min(dims[0] - 1, base_i + kMaximumAnchorRadius);
        const int j_min =
            std::max(0, base_j - kMaximumAnchorRadius);
        const int j_max =
            std::min(dims[1] - 1, base_j + kMaximumAnchorRadius);
        std::vector<int> candidates;
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                const int node = grid.index(i, j);
                const bool node_inside =
                    grid_pair.domain_label(node) > 0;
                if (node_inside == desired_inside)
                    candidates.push_back(node);
            }
        }
        std::sort(
            candidates.begin(),
            candidates.end(),
            [&](int lhs, int rhs) {
                const structured_grid::Index2D lhs_index =
                    structured_grid::index_2d(grid, lhs);
                const structured_grid::Index2D rhs_index =
                    structured_grid::index_2d(grid, rhs);
                const double lhs_distance2 =
                    std::pow(
                        (static_cast<double>(lhs_index.i) - grid_x)
                            * spacing[0],
                        2)
                    + std::pow(
                        (static_cast<double>(lhs_index.j) - grid_y)
                            * spacing[1],
                        2);
                const double rhs_distance2 =
                    std::pow(
                        (static_cast<double>(rhs_index.i) - grid_x)
                            * spacing[0],
                        2)
                    + std::pow(
                        (static_cast<double>(rhs_index.j) - grid_y)
                            * spacing[1],
                        2);
                if (lhs_distance2 < rhs_distance2) {
                    return true;
                }
                if (rhs_distance2 < lhs_distance2) {
                    return false;
                }
                return lhs < rhs;
            });
        if (!candidates.empty())
            return candidates;
        throw std::runtime_error(
            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D could not find a nearby same-side grid node");
    }

    TraceSample build_six_point_query_ray_sample(
        const GridPair2D& grid_pair,
        const Eigen::Vector2d& query,
        bool desired_inside,
        int q,
        int side,
        int layer)
    {
        const CartesianGrid2D& grid = grid_pair.grid();
        const auto dims = grid.dof_dims();
        const std::vector<int> center_nodes =
            nearby_same_side_nodes(
                grid_pair, query, desired_inside);

        struct DiagonalCandidate {
            int node = -1;
            double distance2 = std::numeric_limits<double>::infinity();
            P2CrossingOwner2D owner;
        };
        DiagonalCandidate diagonal;
        int center_node = -1;
        structured_grid::Index2D center_index;
        const double nearest_center_distance2 =
            (structured_grid::point(grid, center_nodes.front())
             - query)
                .squaredNorm();
        double selected_center_distance2 =
            std::numeric_limits<double>::infinity();
        int local_zero_crossings = 0;
        int local_multiple_crossings = 0;
        for (const int candidate_center_node : center_nodes) {
            const double candidate_center_distance2 =
                (structured_grid::point(
                     grid, candidate_center_node)
                 - query)
                    .squaredNorm();
            const double center_distance_tolerance =
                256.0 * std::numeric_limits<double>::epsilon()
                * std::max(
                    {h_ * h_,
                     candidate_center_distance2,
                     std::isfinite(selected_center_distance2)
                         ? selected_center_distance2
                         : 0.0});
            if (center_node >= 0
                && candidate_center_distance2
                       > selected_center_distance2
                             + center_distance_tolerance) {
                break;
            }
            const structured_grid::Index2D candidate_center_index =
                structured_grid::index_2d(
                    grid, candidate_center_node);
            const std::array<structured_grid::Index2D, 5>
                candidate_cross_indices{{
                    candidate_center_index,
                    {candidate_center_index.i - 1,
                     candidate_center_index.j},
                    {candidate_center_index.i + 1,
                     candidate_center_index.j},
                    {candidate_center_index.i,
                     candidate_center_index.j - 1},
                    {candidate_center_index.i,
                     candidate_center_index.j + 1}}};
            bool cross_is_inside_grid = true;
            for (const structured_grid::Index2D index :
                 candidate_cross_indices) {
                cross_is_inside_grid =
                    cross_is_inside_grid
                    && structured_grid::is_valid_index(grid, index);
            }
            if (!cross_is_inside_grid)
                continue;

            DiagonalCandidate candidate_diagonal;
            const std::array<structured_grid::Index2D, 4>
                diagonal_indices{{
                    {candidate_center_index.i - 1,
                     candidate_center_index.j - 1},
                    {candidate_center_index.i + 1,
                     candidate_center_index.j - 1},
                    {candidate_center_index.i - 1,
                     candidate_center_index.j + 1},
                    {candidate_center_index.i + 1,
                     candidate_center_index.j + 1}}};
            for (const structured_grid::Index2D index :
                 diagonal_indices) {
                if (!structured_grid::is_valid_index(grid, index))
                    continue;
                const int node = grid.index(index.i, index.j);
                const Eigen::Vector2d point =
                    structured_grid::point(grid, node);
                const P2CrossingOwner2D owner =
                    grid_pair.p2_crossing_owner_between(
                        query, point);
                if (owner.status
                        != P2CrossingOwnerStatus2D::
                               ExactIntersection
                    || owner.exact_intersection_count == 0) {
                    ++local_zero_crossings;
                    ++diagnostics_
                          .diagonal_zero_crossing_rejections;
                    continue;
                }
                if (owner.exact_intersection_count != 1) {
                    ++local_multiple_crossings;
                    ++diagnostics_
                          .diagonal_multiple_crossing_rejections;
                    continue;
                }
                const double distance2 =
                    (point - query).squaredNorm();
                const double distance_tolerance =
                    256.0 * std::numeric_limits<double>::epsilon()
                    * std::max(
                        {h_ * h_,
                         distance2,
                         std::isfinite(
                             candidate_diagonal.distance2)
                             ? candidate_diagonal.distance2
                             : 0.0});
                if (candidate_diagonal.node < 0
                    || distance2
                        < candidate_diagonal.distance2
                              - distance_tolerance
                    || (std::abs(
                            distance2
                            - candidate_diagonal.distance2)
                            <= distance_tolerance
                        && (candidate_diagonal.node < 0
                            || node
                                   < candidate_diagonal.node))) {
                    candidate_diagonal.node = node;
                    candidate_diagonal.distance2 = distance2;
                    candidate_diagonal.owner = owner;
                }
            }
            if (candidate_diagonal.node < 0)
                continue;

            if (center_node < 0)
                selected_center_distance2 =
                    candidate_center_distance2;
            const double diagonal_distance_tolerance =
                256.0 * std::numeric_limits<double>::epsilon()
                * std::max(
                    {h_ * h_,
                     candidate_diagonal.distance2,
                     std::isfinite(diagonal.distance2)
                         ? diagonal.distance2
                         : 0.0});
            if (center_node < 0
                || candidate_diagonal.distance2
                    < diagonal.distance2
                          - diagonal_distance_tolerance
                || (std::abs(candidate_diagonal.distance2
                            - diagonal.distance2)
                        <= diagonal_distance_tolerance
                    && (center_node < 0
                        || candidate_center_node < center_node
                        || (candidate_center_node == center_node
                            && candidate_diagonal.node
                                   < diagonal.node)))) {
                center_node = candidate_center_node;
                center_index = candidate_center_index;
                diagonal = candidate_diagonal;
            }
        }
        if (diagonal.node < 0) {
            throw std::runtime_error(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D "
                "same-side cross stencil has no diagonal with exactly one "
                "crossing at q="
                + std::to_string(q)
                + " side=" + std::to_string(side)
                + " layer=" + std::to_string(layer)
                + " zero=" + std::to_string(local_zero_crossings)
                + " multiple="
                + std::to_string(local_multiple_crossings));
        }
        const double center_distance_tolerance =
            256.0 * std::numeric_limits<double>::epsilon()
            * std::max(
                {h_ * h_,
                 nearest_center_distance2,
                 selected_center_distance2});
        if (selected_center_distance2
            > nearest_center_distance2
                  + center_distance_tolerance) {
            ++diagnostics_.expanded_same_side_center_stencils;
        }
        diagnostics_.max_same_side_center_distance_over_h =
            std::max(
                diagnostics_.max_same_side_center_distance_over_h,
                std::sqrt(selected_center_distance2) / h_);

        const std::array<structured_grid::Index2D, 5> cross_indices{{
            center_index,
            {center_index.i - 1, center_index.j},
            {center_index.i + 1, center_index.j},
            {center_index.i, center_index.j - 1},
            {center_index.i, center_index.j + 1}}};
        TraceSample sample;
        sample.query = query;
        sample.nodes.resize(6);
        for (int position = 0; position < 5; ++position) {
            const structured_grid::Index2D index =
                cross_indices[static_cast<std::size_t>(position)];
            sample.nodes[static_cast<std::size_t>(position)].grid_node =
                grid.index(index.i, index.j);
        }
        sample.nodes[5].grid_node = diagonal.node;

        Eigen::Matrix<double, 6, 6> interpolation;
        for (int row = 0; row < 6; ++row) {
            InterpolationNode& node =
                sample.nodes[static_cast<std::size_t>(row)];
            node.point =
                structured_grid::point(grid, node.grid_node);
            const double dx = (node.point[0] - query[0]) / h_;
            const double dy = (node.point[1] - query[1]) / h_;
            interpolation(row, 0) = 1.0;
            interpolation(row, 1) = dx;
            interpolation(row, 2) = dy;
            interpolation(row, 3) = 0.5 * dx * dx;
            interpolation(row, 4) = dx * dy;
            interpolation(row, 5) = 0.5 * dy * dy;
        }
        Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(
            interpolation.transpose());
        if (!lu.isInvertible()) {
            throw std::runtime_error(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D "
                "same-side cross interpolation stencil is singular");
        }
        Eigen::Matrix<double, 6, 1> query_basis =
            Eigen::Matrix<double, 6, 1>::Zero();
        query_basis[0] = 1.0;
        const Eigen::Matrix<double, 6, 1> weights =
            lu.solve(query_basis);
        diagnostics_.max_six_point_weight_l1 =
            std::max(
                diagnostics_.max_six_point_weight_l1,
                weights.cwiseAbs().sum());

        for (int row = 0; row < 6; ++row) {
            InterpolationNode& node =
                sample.nodes[static_cast<std::size_t>(row)];
            node.weight = weights[row];
            const bool node_inside =
                grid_pair.domain_label(node.grid_node) > 0;
            if (node_inside == desired_inside)
                continue;

            ++diagnostics_.wrong_side_nodes;
            P2CrossingOwner2D owner;
            if (row == 5) {
                owner = diagonal.owner;
            } else {
                owner = grid_pair.p2_crossing_owner_between(
                    query, node.point);
            }
            node.correction_center = owner.center_index;
            node.correction_crossing = owner;
            node.correction_scale =
                desired_inside ? 1.0 : -1.0;
            ++diagnostics_.corrected_nodes;
            switch (owner.status) {
            case P2CrossingOwnerStatus2D::ExactIntersection:
                ++diagnostics_.exact_crossing_owners;
                break;
            case P2CrossingOwnerStatus2D::GapFallback:
                ++diagnostics_.gap_fallback_owners;
                if (owner.explicit_gap_intersection)
                    ++diagnostics_.identified_gap_crossing_owners;
                else
                    ++diagnostics_.unresolved_gap_fallback_owners;
                break;
            case P2CrossingOwnerStatus2D::EndpointNearestCenter:
                ++diagnostics_.endpoint_fallback_owners;
                break;
            }
        }
        ++diagnostics_.six_point_cross_stencils;
        return sample;
    }

    TraceSample build_six_point_grid_edge_sample(
        const GridPair2D& grid_pair,
        const Eigen::Vector2d& selection_query,
        const Eigen::Vector2d& query,
        bool desired_inside,
        int q,
        int side,
        int layer)
    {
        const CartesianGrid2D& grid = grid_pair.grid();
        const std::vector<int> center_nodes =
            nearby_same_side_nodes(
                grid_pair, selection_query, desired_inside);

        struct DiagonalCandidate {
            int node = -1;
            double distance2 = std::numeric_limits<double>::infinity();
            std::array<P2CrossingOwner2D, 2> owners;
            int owner_count = 0;
            bool needs_correction = true;
        };

        const auto node_inside = [&](int node) {
            return grid_pair.domain_label(node) > 0;
        };
        const auto identified_crossing =
            [](const P2CrossingOwner2D& owner) {
                return (owner.status
                            == P2CrossingOwnerStatus2D::
                                   ExactIntersection
                        && owner.exact_intersection_count == 1)
                    || (owner.status
                            == P2CrossingOwnerStatus2D::GapFallback
                        && owner.explicit_gap_intersection);
            };
        const auto identified_no_crossing =
            [](const P2CrossingOwner2D& owner) {
                return owner.exact_intersection_count == 0
                    && !owner.explicit_gap_intersection;
            };

        int center_node = -1;
        structured_grid::Index2D center_index;
        DiagonalCandidate diagonal;
        std::array<bool, 4> cardinal_needs_correction{{false, false,
                                                       false, false}};
        std::array<P2CrossingOwner2D, 4> cardinal_owners;
        const double nearest_center_distance2 =
            (structured_grid::point(grid, center_nodes.front())
             - selection_query)
                .squaredNorm();
        double selected_center_distance2 =
            std::numeric_limits<double>::infinity();

        for (const int candidate_center_node : center_nodes) {
            const Eigen::Vector2d candidate_center_point =
                structured_grid::point(grid, candidate_center_node);
            const double candidate_center_distance2 =
                (candidate_center_point - selection_query).squaredNorm();
            const double center_distance_tolerance =
                256.0 * std::numeric_limits<double>::epsilon()
                * std::max(
                    {h_ * h_,
                     candidate_center_distance2,
                     std::isfinite(selected_center_distance2)
                         ? selected_center_distance2
                         : 0.0});
            if (center_node >= 0
                && candidate_center_distance2
                       > selected_center_distance2
                             + center_distance_tolerance) {
                break;
            }

            const structured_grid::Index2D candidate_center_index =
                structured_grid::index_2d(grid, candidate_center_node);
            const std::array<structured_grid::Index2D, 4>
                candidate_cardinal_indices{{
                    {candidate_center_index.i - 1,
                     candidate_center_index.j},
                    {candidate_center_index.i + 1,
                     candidate_center_index.j},
                    {candidate_center_index.i,
                     candidate_center_index.j - 1},
                    {candidate_center_index.i,
                     candidate_center_index.j + 1}}};
            bool cardinal_indices_valid = true;
            for (const structured_grid::Index2D index :
                 candidate_cardinal_indices) {
                cardinal_indices_valid =
                    cardinal_indices_valid
                    && structured_grid::is_valid_index(grid, index);
            }
            if (!cardinal_indices_valid)
                continue;

            std::array<bool, 4> candidate_cardinal_needs{{
                false, false, false, false}};
            std::array<P2CrossingOwner2D, 4>
                candidate_cardinal_owners;
            bool cardinal_edges_valid = true;
            for (int cardinal = 0; cardinal < 4; ++cardinal) {
                const structured_grid::Index2D index =
                    candidate_cardinal_indices[
                        static_cast<std::size_t>(cardinal)];
                const int node = grid.index(index.i, index.j);
                const bool crosses =
                    node_inside(node) != desired_inside;
                const P2CrossingOwner2D owner =
                    grid_pair.p2_crossing_owner_between(
                        candidate_center_point,
                        structured_grid::point(grid, node));
                if (crosses) {
                    if (!identified_crossing(owner)) {
                        ++diagnostics_
                              .grid_edge_cardinal_ambiguous_rejections;
                        cardinal_edges_valid = false;
                        break;
                    }
                    candidate_cardinal_needs[
                        static_cast<std::size_t>(cardinal)] = true;
                    candidate_cardinal_owners[
                        static_cast<std::size_t>(cardinal)] = owner;
                } else if (!identified_no_crossing(owner)) {
                    ++diagnostics_
                          .grid_edge_cardinal_ambiguous_rejections;
                    cardinal_edges_valid = false;
                    break;
                }
            }
            if (!cardinal_edges_valid)
                continue;

            DiagonalCandidate candidate_diagonal;
            const std::array<structured_grid::Index2D, 4>
                diagonal_indices{{
                    {candidate_center_index.i - 1,
                     candidate_center_index.j - 1},
                    {candidate_center_index.i + 1,
                     candidate_center_index.j - 1},
                    {candidate_center_index.i - 1,
                     candidate_center_index.j + 1},
                    {candidate_center_index.i + 1,
                     candidate_center_index.j + 1}}};
            for (const structured_grid::Index2D index :
                 diagonal_indices) {
                if (!structured_grid::is_valid_index(grid, index))
                    continue;
                const int node = grid.index(index.i, index.j);
                const bool diagonal_needs_correction =
                    node_inside(node) != desired_inside;
                if (!diagonal_needs_correction
                    && !uses_shared_side_spatial_polynomial()) {
                    ++diagnostics_
                          .grid_edge_diagonal_zero_owner_rejections;
                    continue;
                }

                const int offset_i =
                    index.i - candidate_center_index.i;
                const int offset_j =
                    index.j - candidate_center_index.j;
                const std::array<structured_grid::Index2D, 2>
                    adjacent_indices{{
                        {candidate_center_index.i + offset_i,
                         candidate_center_index.j},
                        {candidate_center_index.i,
                         candidate_center_index.j + offset_j}}};
                DiagonalCandidate tested;
                tested.node = node;
                tested.needs_correction =
                    diagonal_needs_correction;
                tested.distance2 =
                    (structured_grid::point(grid, node)
                     - selection_query)
                        .squaredNorm();
                bool edges_valid = true;
                for (const structured_grid::Index2D adjacent_index :
                     adjacent_indices) {
                    const int adjacent_node =
                        grid.index(adjacent_index.i, adjacent_index.j);
                    const bool crosses =
                        node_inside(adjacent_node) != node_inside(node);
                    const P2CrossingOwner2D owner =
                        grid_pair.p2_crossing_owner_between(
                            structured_grid::point(grid, adjacent_node),
                            structured_grid::point(grid, node));
                    if (crosses) {
                        if (!identified_crossing(owner)) {
                            ++diagnostics_
                                  .grid_edge_diagonal_ambiguous_edge_rejections;
                            edges_valid = false;
                            break;
                        }
                        if (diagonal_needs_correction) {
                            tested.owners[static_cast<std::size_t>(
                                tested.owner_count++)] = owner;
                        }
                    } else if (!identified_no_crossing(owner)) {
                        ++diagnostics_
                              .grid_edge_diagonal_ambiguous_edge_rejections;
                        edges_valid = false;
                        break;
                    }
                }
                if (!edges_valid)
                    continue;
                if (diagonal_needs_correction
                    && tested.owner_count == 0) {
                    ++diagnostics_
                          .grid_edge_diagonal_zero_owner_rejections;
                    continue;
                }

                const double distance_tolerance =
                    256.0 * std::numeric_limits<double>::epsilon()
                    * std::max(
                        {h_ * h_,
                         tested.distance2,
                         std::isfinite(candidate_diagonal.distance2)
                             ? candidate_diagonal.distance2
                             : 0.0});
                if (candidate_diagonal.node < 0
                    || tested.distance2
                           < candidate_diagonal.distance2
                                 - distance_tolerance
                    || (std::abs(tested.distance2
                                - candidate_diagonal.distance2)
                            <= distance_tolerance
                        && tested.node < candidate_diagonal.node)) {
                    candidate_diagonal = tested;
                }
            }
            if (candidate_diagonal.node < 0)
                continue;

            if (center_node < 0)
                selected_center_distance2 =
                    candidate_center_distance2;
            const double diagonal_distance_tolerance =
                256.0 * std::numeric_limits<double>::epsilon()
                * std::max(
                    {h_ * h_,
                     candidate_diagonal.distance2,
                     std::isfinite(diagonal.distance2)
                         ? diagonal.distance2
                         : 0.0});
            if (center_node < 0
                || candidate_diagonal.distance2
                       < diagonal.distance2
                             - diagonal_distance_tolerance
                || (std::abs(candidate_diagonal.distance2
                            - diagonal.distance2)
                        <= diagonal_distance_tolerance
                    && (candidate_center_node < center_node
                        || (candidate_center_node == center_node
                            && candidate_diagonal.node
                                   < diagonal.node)))) {
                center_node = candidate_center_node;
                center_index = candidate_center_index;
                diagonal = candidate_diagonal;
                cardinal_needs_correction =
                    candidate_cardinal_needs;
                cardinal_owners = candidate_cardinal_owners;
            }
        }

        if (diagonal.node < 0) {
            throw std::runtime_error(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D "
                "grid-edge cross stencil has no opposite-side diagonal "
                "with one or two identified incident-edge crossings at q="
                + std::to_string(q)
                + " side=" + std::to_string(side)
                + " layer=" + std::to_string(layer));
        }
        const double center_distance_tolerance =
            256.0 * std::numeric_limits<double>::epsilon()
            * std::max(
                {h_ * h_,
                 nearest_center_distance2,
                 selected_center_distance2});
        const bool expanded_center =
            selected_center_distance2
            > nearest_center_distance2 + center_distance_tolerance;
        if (expanded_center) {
            ++diagnostics_.expanded_same_side_center_stencils;
        }
        diagnostics_.max_same_side_center_distance_over_h =
            std::max(
                diagnostics_.max_same_side_center_distance_over_h,
                std::sqrt(selected_center_distance2) / h_);

        const std::array<structured_grid::Index2D, 5> cross_indices{{
            center_index,
            {center_index.i - 1, center_index.j},
            {center_index.i + 1, center_index.j},
            {center_index.i, center_index.j - 1},
            {center_index.i, center_index.j + 1}}};
        TraceSample sample;
        sample.query = query;
        sample.nodes.resize(6);
        for (int position = 0; position < 5; ++position) {
            const structured_grid::Index2D index =
                cross_indices[static_cast<std::size_t>(position)];
            sample.nodes[static_cast<std::size_t>(position)].grid_node =
                grid.index(index.i, index.j);
        }
        sample.nodes[5].grid_node = diagonal.node;

        Eigen::Matrix<double, 6, 6> interpolation;
        for (int row = 0; row < 6; ++row) {
            InterpolationNode& node =
                sample.nodes[static_cast<std::size_t>(row)];
            node.point =
                structured_grid::point(grid, node.grid_node);
            const double dx = (node.point[0] - query[0]) / h_;
            const double dy = (node.point[1] - query[1]) / h_;
            interpolation(row, 0) = 1.0;
            interpolation(row, 1) = dx;
            interpolation(row, 2) = dy;
            interpolation(row, 3) = 0.5 * dx * dx;
            interpolation(row, 4) = dx * dy;
            interpolation(row, 5) = 0.5 * dy * dy;
        }
        Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(
            interpolation.transpose());
        if (!lu.isInvertible()) {
            throw std::runtime_error(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D "
                "grid-edge cross interpolation stencil is singular");
        }
        Eigen::Matrix<double, 6, 1> query_basis =
            Eigen::Matrix<double, 6, 1>::Zero();
        query_basis[0] = 1.0;
        const Eigen::Matrix<double, 6, 1> weights =
            lu.solve(query_basis);
        diagnostics_.max_six_point_weight_l1 =
            std::max(
                diagnostics_.max_six_point_weight_l1,
                weights.cwiseAbs().sum());

        const double phase_scale = desired_inside ? 1.0 : -1.0;
        const auto record_owner =
            [&](const P2CrossingOwner2D& owner) {
                ++diagnostics_.grid_edge_owner_terms;
                switch (owner.status) {
                case P2CrossingOwnerStatus2D::ExactIntersection:
                    ++diagnostics_.exact_crossing_owners;
                    break;
                case P2CrossingOwnerStatus2D::GapFallback:
                    ++diagnostics_.gap_fallback_owners;
                    if (owner.explicit_gap_intersection)
                        ++diagnostics_.identified_gap_crossing_owners;
                    else
                        ++diagnostics_.unresolved_gap_fallback_owners;
                    break;
                case P2CrossingOwnerStatus2D::EndpointNearestCenter:
                    ++diagnostics_.endpoint_fallback_owners;
                    break;
                }
            };
        for (int row = 0; row < 6; ++row)
            sample.nodes[static_cast<std::size_t>(row)].weight =
                weights[row];
        for (int cardinal = 0; cardinal < 4; ++cardinal) {
            if (!cardinal_needs_correction[
                    static_cast<std::size_t>(cardinal)]) {
                continue;
            }
            InterpolationNode& node =
                sample.nodes[static_cast<std::size_t>(cardinal + 1)];
            const P2CrossingOwner2D& owner =
                cardinal_owners[static_cast<std::size_t>(cardinal)];
            node.correction_center = owner.center_index;
            node.correction_crossing = owner;
            node.correction_scale = phase_scale;
            ++diagnostics_.wrong_side_nodes;
            ++diagnostics_.corrected_nodes;
            ++diagnostics_.grid_edge_cardinal_corrections;
            record_owner(owner);
        }

        InterpolationNode& diagonal_node = sample.nodes[5];
        if (!diagonal.needs_correction) {
            ++diagnostics_.grid_edge_diagonal_same_side_stencils;
        } else if (diagonal.owner_count == 1) {
            ++diagnostics_.wrong_side_nodes;
            ++diagnostics_.corrected_nodes;
            diagonal_node.correction_center =
                diagonal.owners[0].center_index;
            diagonal_node.correction_crossing = diagonal.owners[0];
            diagonal_node.correction_scale = phase_scale;
            ++diagnostics_
                  .grid_edge_diagonal_single_owner_stencils;
            record_owner(diagonal.owners[0]);
        } else {
            ++diagnostics_.wrong_side_nodes;
            ++diagnostics_.corrected_nodes;
            diagonal_node.correction_center =
                diagonal.owners[0].center_index;
            diagonal_node.correction_crossing = diagonal.owners[0];
            diagonal_node.correction_scale = 0.5 * phase_scale;
            diagonal_node.secondary_correction_center =
                diagonal.owners[1].center_index;
            diagonal_node.secondary_correction_crossing =
                diagonal.owners[1];
            diagonal_node.secondary_correction_scale =
                0.5 * phase_scale;
            ++diagnostics_
                  .grid_edge_diagonal_blended_owner_stencils;
            ++diagnostics_.grid_edge_blended_correction_nodes;
            record_owner(diagonal.owners[0]);
            record_owner(diagonal.owners[1]);
        }
        LaplaceP2JointPolynomialRestrictDiagnostics2D::GridEdgeSample
            sample_diagnostics;
        sample_diagnostics.q = q;
        sample_diagnostics.side = side;
        sample_diagnostics.layer = layer;
        sample_diagnostics.query = query;
        sample_diagnostics.selection_query = selection_query;
        sample_diagnostics.center_node = center_node;
        sample_diagnostics.diagonal_node = diagonal.node;
        sample_diagnostics.nearest_center_distance_over_h =
            std::sqrt(nearest_center_distance2) / h_;
        sample_diagnostics.selected_center_distance_over_h =
            std::sqrt(selected_center_distance2) / h_;
        sample_diagnostics.evaluation_center_distance_over_h =
            (structured_grid::point(grid, center_node) - query).norm()
            / h_;
        sample_diagnostics.expanded_center = expanded_center;
        for (int row = 0; row < 6; ++row) {
            sample_diagnostics.grid_nodes[static_cast<std::size_t>(row)] =
                sample.nodes[static_cast<std::size_t>(row)].grid_node;
            sample_diagnostics.grid_points[static_cast<std::size_t>(row)] =
                sample.nodes[static_cast<std::size_t>(row)].point;
            sample_diagnostics.interpolation_weights[
                static_cast<std::size_t>(row)] = weights[row];
        }
        sample_diagnostics.cardinal_corrected =
            cardinal_needs_correction;
        sample_diagnostics.cardinal_owners = cardinal_owners;
        sample_diagnostics.diagonal_corrected =
            diagonal.needs_correction;
        sample_diagnostics.diagonal_owner_count =
            diagonal.owner_count;
        sample_diagnostics.diagonal_owners = diagonal.owners;
        diagnostics_.grid_edge_samples.push_back(
            std::move(sample_diagnostics));
        ++diagnostics_.grid_edge_cross_stencils;
        ++diagnostics_.six_point_cross_stencils;
        return sample;
    }

    TraceSample build_unified_p2_shared_side_sample(
        const GridPair2D& grid_pair,
        const Eigen::Vector2d& selection_query,
        const Eigen::Vector2d& query,
        bool desired_inside,
        int q,
        int side,
        int layer,
        const Eigen::Vector2d* explicit_trace_point = nullptr,
        const Eigen::Vector2d* explicit_trace_normal = nullptr,
        const TraceSample* explicit_layer_zero = nullptr)
    {
        const CartesianGrid2D& grid = grid_pair.grid();
        const Interface2D& iface = grid_pair.interface();
        const std::vector<int> center_nodes =
            nearby_same_side_nodes(
                grid_pair, selection_query, desired_inside);

        struct Candidate {
            std::array<int, 6> nodes{{-1, -1, -1, -1, -1, -1}};
            std::array<bool, 6> corrected{{false, false, false,
                                           false, false, false}};
            std::array<P2CrossingOwner2D, 6> owners;
            double quality = std::numeric_limits<double>::infinity();
            double center_distance2 =
                std::numeric_limits<double>::infinity();
        };

        const auto interpolation_weights =
            [&](const std::array<int, 6>& nodes,
                const Eigen::Vector2d& evaluation_query) {
                Eigen::Matrix<double, 6, 6> interpolation;
                for (int row = 0; row < 6; ++row) {
                    const Eigen::Vector2d point =
                        structured_grid::point(
                            grid,
                            nodes[static_cast<std::size_t>(row)]);
                    const double dx =
                        (point[0] - evaluation_query[0]) / h_;
                    const double dy =
                        (point[1] - evaluation_query[1]) / h_;
                    interpolation(row, 0) = 1.0;
                    interpolation(row, 1) = dx;
                    interpolation(row, 2) = dy;
                    interpolation(row, 3) = 0.5 * dx * dx;
                    interpolation(row, 4) = dx * dy;
                    interpolation(row, 5) = 0.5 * dy * dy;
                }
                Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(
                    interpolation.transpose());
                if (!lu.isInvertible()) {
                    throw std::runtime_error(
                        "LaplaceP2CrossingOwnerJointPolynomialRestrict2D unified spatial P2 stencil is singular");
                }
                Eigen::Matrix<double, 6, 1> query_basis =
                    Eigen::Matrix<double, 6, 1>::Zero();
                query_basis[0] = 1.0;
                const Eigen::Matrix<double, 6, 1> weights =
                    lu.solve(query_basis);
                return weights;
            };

        // The first layer selects the spatial P2 stencil and its crossing
        // owners.  Every remaining normal layer reuses that exact stencil;
        // only the evaluation weights change with the query location.
        if (layer > 0) {
            const TraceSample& reference = explicit_layer_zero != nullptr
                ? *explicit_layer_zero
                : trace_samples_[static_cast<std::size_t>(
                      trace_sample_index(q, side, 0))];
            if (reference.nodes.size() != 6) {
                throw std::runtime_error(
                    "LaplaceP2CrossingOwnerJointPolynomialRestrict2D unified spatial P2 reference stencil is missing");
            }
            std::array<int, 6> nodes{{-1, -1, -1, -1, -1, -1}};
            for (int position = 0; position < 6; ++position) {
                nodes[static_cast<std::size_t>(position)] =
                    reference.nodes[static_cast<std::size_t>(position)]
                        .grid_node;
            }
            const Eigen::Matrix<double, 6, 1> weights =
                interpolation_weights(nodes, query);
            diagnostics_.max_six_point_weight_l1 = std::max(
                diagnostics_.max_six_point_weight_l1,
                weights.cwiseAbs().sum());

            TraceSample sample = reference;
            sample.query = query;
            for (int position = 0; position < 6; ++position) {
                InterpolationNode& node =
                    sample.nodes[static_cast<std::size_t>(position)];
                node.weight = weights[position];
                if (node.correction_center < 0)
                    continue;
                ++diagnostics_.wrong_side_nodes;
                ++diagnostics_.corrected_nodes;
                ++diagnostics_.exact_crossing_owners;
            }
            ++diagnostics_.six_point_cross_stencils;
            return sample;
        }

        const Eigen::Vector2d interface_point_q =
            explicit_trace_point != nullptr
                ? *explicit_trace_point
                : iface.points().row(q).transpose();
        Eigen::Vector2d normal = explicit_trace_normal != nullptr
            ? *explicit_trace_normal
            : iface.normals().row(q).transpose();
        normal.normalize();
        const double side_sign = desired_inside ? -1.0 : 1.0;
        const auto node_inside = [&](int node) {
            return grid_pair.domain_label(node) > 0;
        };
        const auto unique_transverse_crossing =
            [](const P2CrossingOwner2D& owner) {
                return owner.status
                           == P2CrossingOwnerStatus2D::ExactIntersection
                    && owner.exact_intersection_count == 1
                    && owner.transverse_intersection_count == 1;
            };

        Candidate best;
        int selected_center_rank = -1;
        constexpr double kMaximumCenterDistanceOverH = 2.5;
        for (std::size_t center_rank = 0;
             center_rank < center_nodes.size();
             ++center_rank) {
            const int center_node = center_nodes[center_rank];
            const Eigen::Vector2d center_point =
                structured_grid::point(grid, center_node);
            const double center_distance2 =
                (center_point - selection_query).squaredNorm();
            if (center_distance2
                > kMaximumCenterDistanceOverH
                      * kMaximumCenterDistanceOverH * h_ * h_) {
                break;
            }
            Candidate best_at_center;
            const structured_grid::Index2D center_index =
                structured_grid::index_2d(grid, center_node);
            const std::array<structured_grid::Index2D, 5>
                cross_indices{{
                    center_index,
                    {center_index.i - 1, center_index.j},
                    {center_index.i + 1, center_index.j},
                    {center_index.i, center_index.j - 1},
                    {center_index.i, center_index.j + 1}}};
            bool cross_is_inside_grid = true;
            for (const structured_grid::Index2D index : cross_indices) {
                cross_is_inside_grid =
                    cross_is_inside_grid
                    && structured_grid::is_valid_index(grid, index);
            }
            if (!cross_is_inside_grid)
                continue;

            const std::array<structured_grid::Index2D, 4>
                diagonal_indices{{
                    {center_index.i - 1, center_index.j - 1},
                    {center_index.i + 1, center_index.j - 1},
                    {center_index.i - 1, center_index.j + 1},
                    {center_index.i + 1, center_index.j + 1}}};
            for (const structured_grid::Index2D diagonal_index :
                 diagonal_indices) {
                if (!structured_grid::is_valid_index(
                        grid, diagonal_index)) {
                    continue;
                }

                Candidate candidate;
                candidate.center_distance2 = center_distance2;
                for (int position = 0; position < 5; ++position) {
                    const structured_grid::Index2D index =
                        cross_indices[static_cast<std::size_t>(position)];
                    candidate.nodes[static_cast<std::size_t>(position)] =
                        grid.index(index.i, index.j);
                }
                candidate.nodes[5] =
                    grid.index(diagonal_index.i, diagonal_index.j);

                bool valid = true;
                for (int position = 0; position < 6; ++position) {
                    const int node =
                        candidate.nodes[static_cast<std::size_t>(position)];
                    if (node_inside(node) == desired_inside)
                        continue;
                    const P2CrossingOwner2D owner =
                        grid_pair.p2_crossing_owner_between(
                            selection_query,
                            structured_grid::point(grid, node));
                    if (!unique_transverse_crossing(owner)) {
                        ++diagnostics_.unified_p2_candidate_rejections;
                        valid = false;
                        break;
                    }
                    candidate.corrected[
                        static_cast<std::size_t>(position)] = true;
                    candidate.owners[
                        static_cast<std::size_t>(position)] = owner;
                }
                if (!valid)
                    continue;

                double maximum_weight_l1 = 0.0;
                double maximum_correction_weight = 0.0;
                for (int sample_layer = 0;
                     sample_layer < layer_count_;
                     ++sample_layer) {
                    const Eigen::Vector2d layer_query =
                        interface_point_q
                        + side_sign
                            * options_.normal_layers[
                                static_cast<std::size_t>(sample_layer)]
                            * h_ * normal;
                    const Eigen::Matrix<double, 6, 1> weights =
                        interpolation_weights(
                            candidate.nodes, layer_query);
                    maximum_weight_l1 = std::max(
                        maximum_weight_l1,
                        weights.cwiseAbs().sum());
                    double correction_weight = 0.0;
                    for (int position = 0; position < 6; ++position) {
                        if (candidate.corrected[
                                static_cast<std::size_t>(position)]) {
                            correction_weight += std::abs(
                                weights[position]);
                        }
                    }
                    maximum_correction_weight = std::max(
                        maximum_correction_weight,
                        correction_weight);
                }
                candidate.quality =
                    maximum_weight_l1 + maximum_correction_weight;

                constexpr double tolerance = 1.0e-14;
                const bool better =
                    candidate.quality
                            < best_at_center.quality - tolerance
                    || (std::abs(candidate.quality
                                 - best_at_center.quality)
                            <= tolerance
                        && candidate.nodes < best_at_center.nodes);
                if (better)
                    best_at_center = candidate;
            }
            if (best_at_center.nodes[0] >= 0) {
                best = std::move(best_at_center);
                selected_center_rank = static_cast<int>(center_rank);
                break;
            }
        }

        if (best.nodes[0] < 0) {
            throw std::runtime_error(
                "LaplaceP2CrossingOwnerJointPolynomialRestrict2D unified spatial P2 found no local six-node stencil with only unique transverse crossing owners at q="
                + std::to_string(q)
                + " side=" + std::to_string(side));
        }
        if (selected_center_rank > 0)
            ++diagnostics_.unified_p2_relocated_stencils;
        diagnostics_.max_same_side_center_distance_over_h = std::max(
            diagnostics_.max_same_side_center_distance_over_h,
            std::sqrt(best.center_distance2) / h_);

        TraceSample sample;
        sample.query = query;
        sample.nodes.resize(6);
        const Eigen::Matrix<double, 6, 1> weights =
            interpolation_weights(best.nodes, query);
        diagnostics_.max_six_point_weight_l1 = std::max(
            diagnostics_.max_six_point_weight_l1,
            weights.cwiseAbs().sum());
        for (int position = 0; position < 6; ++position) {
            InterpolationNode& node =
                sample.nodes[static_cast<std::size_t>(position)];
            node.grid_node =
                best.nodes[static_cast<std::size_t>(position)];
            node.point = structured_grid::point(grid, node.grid_node);
            node.weight = weights[position];
            if (!best.corrected[static_cast<std::size_t>(position)])
                continue;
            node.correction_center =
                best.owners[static_cast<std::size_t>(position)]
                    .center_index;
            node.correction_crossing =
                best.owners[static_cast<std::size_t>(position)];
            node.correction_scale = desired_inside ? 1.0 : -1.0;
            ++diagnostics_.wrong_side_nodes;
            ++diagnostics_.corrected_nodes;
            ++diagnostics_.exact_crossing_owners;
        }
        ++diagnostics_.unified_p2_stencils;
        ++diagnostics_.six_point_cross_stencils;
        return sample;
    }

    TraceSample build_six_point_cross_sample(
        const GridPair2D& grid_pair,
        const Eigen::Vector2d& selection_query,
        const Eigen::Vector2d& query,
        bool desired_inside,
        int q,
        int side,
        int layer)
    {
        if (uses_unified_spatial_p2()) {
            return build_unified_p2_shared_side_sample(
                grid_pair,
                selection_query,
                query,
                desired_inside,
                q,
                side,
                layer);
        }
        if (uses_grid_edge_cross_stencil()) {
            return build_six_point_grid_edge_sample(
                grid_pair,
                selection_query,
                query,
                desired_inside,
                q,
                side,
                layer);
        }
        return build_six_point_query_ray_sample(
            grid_pair, query, desired_inside, q, side, layer);
    }

    void build_joint_fit()
    {
        if (uses_unified_normal_fit()) {
            Eigen::MatrixXd design = Eigen::MatrixXd::Zero(
                normal_sample_count_, 3);
            for (int layer = 0; layer < layer_count_; ++layer) {
                const double xm =
                    -options_.normal_layers[
                        static_cast<std::size_t>(layer)];
                const double xp =
                    options_.normal_layers[
                        static_cast<std::size_t>(layer)];
                design(layer, 0) = 1.0;
                design(layer, 1) = xm;
                design(layer, 2) = xm * xm;
                design(layer_count_ + layer, 0) = 1.0;
                design(layer_count_ + layer, 1) = xp;
                design(layer_count_ + layer, 2) = xp * xp;
            }
            const Eigen::MatrixXd pseudo_inverse =
                design.colPivHouseholderQr().solve(
                    Eigen::MatrixXd::Identity(
                        normal_sample_count_, normal_sample_count_));
            c0_weights_ = pseudo_inverse.row(0).transpose();
            c1_weights_ = pseudo_inverse.row(1).transpose();
            return;
        }

        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(
            normal_sample_count_, 2 * degree_);
        for (int layer = 0; layer < layer_count_; ++layer) {
            const double xm =
                -options_.normal_layers[static_cast<std::size_t>(layer)];
            const double xp =
                 options_.normal_layers[static_cast<std::size_t>(layer)];
            design(layer, 0) = 1.0;
            design(layer, 1) = xm;
            design(layer_count_ + layer, 0) = 1.0;
            design(layer_count_ + layer, 1) = xp;
            double xm_power = xm;
            double xp_power = xp;
            for (int power = 2; power <= degree_; ++power) {
                xm_power *= xm;
                xp_power *= xp;
                design(layer, power) = xm_power;
                design(layer_count_ + layer,
                       degree_ + power - 1) = xp_power;
            }
        }
        const Eigen::MatrixXd pseudo_inverse =
            design.colPivHouseholderQr().solve(
                Eigen::MatrixXd::Identity(
                    normal_sample_count_, normal_sample_count_));
        c0_weights_ = pseudo_inverse.row(0).transpose();
        c1_weights_ = pseudo_inverse.row(1).transpose();
    }

    void build_trace_samples(const GridPair2D& grid_pair)
    {
        const CartesianGrid2D& grid = grid_pair.grid();
        const Interface2D& iface = grid_pair.interface();
        const auto dims = grid.dof_dims();
        const Eigen::Vector2d grid_first =
            structured_grid::point(grid, grid.index(0, 0));
        const auto spacing = grid.spacing();

        trace_samples_.resize(
            static_cast<std::size_t>(
                iface.num_points() * 2 * layer_count_));
        diagnostics_.trace_samples =
            iface.num_points() * 2 * layer_count_;
        diagnostics_.interpolation_nodes = 0;

        // A flipped virtual-side interpolation has no query-to-node crossing
        // for the nodes that must be converted.  Tie that continuation to the
        // smooth P2 panel that owns the trace point, and use the same center at
        // every converted node and at the query.
        trace_center_by_point_.assign(
            static_cast<std::size_t>(iface.num_points()), -1);
        trace_panel_by_point_.assign(
            static_cast<std::size_t>(iface.num_points()), -1);
        std::vector<double> trace_center_distance2(
            static_cast<std::size_t>(iface.num_points()),
            std::numeric_limits<double>::infinity());
        for (int panel = 0; panel < iface.num_panels(); ++panel) {
            for (int local_point = 0;
                 local_point < iface.points_per_panel();
                 ++local_point) {
                const int q = iface.point_index(panel, local_point);
                const Eigen::Vector2d point_q =
                    iface.points().row(q).transpose();
                for (int local_center = 0;
                     local_center
                         < static_cast<int>(
                             geometry2d::kP2CenterS.size());
                     ++local_center) {
                    const int center =
                        static_cast<int>(
                            geometry2d::kP2CenterS.size())
                            * panel
                        + local_center;
                    const Eigen::Vector2d center_point =
                        geometry2d::panel_point(
                            iface,
                            panel,
                            geometry2d::kP2CenterS[
                                static_cast<std::size_t>(
                                    local_center)]);
                    const double distance2 =
                        (center_point - point_q).squaredNorm();
                    double& best_distance2 =
                        trace_center_distance2[
                            static_cast<std::size_t>(q)];
                    int& best_center =
                        trace_center_by_point_[
                            static_cast<std::size_t>(q)];
                    if (distance2 < best_distance2 - 1.0e-14
                        || (std::abs(distance2 - best_distance2)
                                <= 1.0e-14
                            && (best_center < 0
                                || center < best_center))) {
                        best_distance2 = distance2;
                        best_center = center;
                        trace_panel_by_point_[
                            static_cast<std::size_t>(q)] = panel;
                    }
                }
            }
        }
        if (uses_center_cauchy_jump()) {
            for (int center : trace_center_by_point_) {
                if (center >= 0) {
                    diagnostics_.center_cauchy_jump_samples +=
                        2 * layer_count_;
                } else {
                    ++diagnostics_
                          .trace_points_without_incident_center;
                }
            }
        }

        struct Candidate {
            TraceSample sample;
            int start_x = 0;
            int start_y = 0;
            int geometric_wrong_side_nodes = 0;
            int corrected_nodes = 0;
            int exact_owners = 0;
            int gap_owners = 0;
            int identified_gap_owners = 0;
            int unresolved_gap_owners = 0;
            int endpoint_owners = 0;
            double correction_weight = 0.0;
            double weight_norm = 0.0;
            bool flipped = false;
        };

        const auto candidate_less =
            [](const Candidate& lhs, const Candidate& rhs) {
                constexpr double tolerance = 1.0e-14;
                if (lhs.correction_weight
                    < rhs.correction_weight - tolerance) {
                    return true;
                }
                if (rhs.correction_weight
                    < lhs.correction_weight - tolerance) {
                    return false;
                }
                if (lhs.weight_norm < rhs.weight_norm - tolerance)
                    return true;
                if (rhs.weight_norm < lhs.weight_norm - tolerance)
                    return false;
                if (lhs.corrected_nodes != rhs.corrected_nodes)
                    return lhs.corrected_nodes < rhs.corrected_nodes;
                if (lhs.flipped != rhs.flipped)
                    return !lhs.flipped;
                if (lhs.start_y != rhs.start_y)
                    return lhs.start_y < rhs.start_y;
                return lhs.start_x < rhs.start_x;
        };

        for (int q = 0; q < iface.num_points(); ++q) {
            if (uses_unified_spatial_p2()
                && trace_center_by_point_[static_cast<std::size_t>(q)] < 0) {
                // Geometry-only corner markers are deliberately not trace
                // DOFs.  The L-shape operator gives them zero quadrature
                // weight and never reads their restricted values, so do not
                // manufacture tensor/fallback templates for them.
                diagnostics_.trace_samples -= 2 * layer_count_;
                continue;
            }
            const Eigen::Vector2d interface_point_q =
                iface.points().row(q).transpose();
            Eigen::Vector2d normal = iface.normals().row(q).transpose();
            const double normal_length = normal.norm();
            if (!(normal_length > 1.0e-14)
                || !normal.allFinite()) {
                throw std::invalid_argument(
                    "LaplaceP2CrossingOwnerJointPolynomialRestrict2D interface normal must be finite and nonzero");
            }
            normal /= normal_length;

            for (int side = 0; side < 2; ++side) {
                const bool desired_inside = side == 0;
                const double side_sign = desired_inside ? -1.0 : 1.0;
                const int fixed_center =
                    trace_center_by_point_[static_cast<std::size_t>(q)];
                const double selection_layer =
                    0.5
                    * (options_.normal_layers.front()
                       + options_.normal_layers[
                           static_cast<std::size_t>(
                               layer_count_ - 1)]);
                const Eigen::Vector2d shared_selection_query =
                    interface_point_q
                    + side_sign * selection_layer * h_ * normal;
                for (int layer = 0; layer < layer_count_; ++layer) {
                    const Eigen::Vector2d query = interface_point_q
                        + side_sign
                            * options_.normal_layers[
                                static_cast<std::size_t>(layer)]
                            * h_ * normal;
                    if (uses_six_point_cross_stencil()
                        && fixed_center >= 0) {
                        TraceSample& sample =
                            trace_samples_[static_cast<std::size_t>(
                                trace_sample_index(
                                    q, side, layer))];
                        sample = build_six_point_cross_sample(
                            grid_pair,
                            uses_shared_side_spatial_polynomial()
                                ? shared_selection_query
                                : query,
                            query,
                            desired_inside,
                            q,
                            side,
                            layer);
                        continue;
                    }

                    const double grid_x =
                        (query[0] - grid_first[0]) / spacing[0];
                    const double grid_y =
                        (query[1] - grid_first[1]) / spacing[1];
                    if (dims[0] < grid_side_ || dims[1] < grid_side_) {
                        throw std::runtime_error(
                            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D grid is too small for tensor interpolation");
                    }
                    const int anchor_x =
                        degree_ == 3
                            ? static_cast<int>(std::floor(grid_x))
                            : static_cast<int>(std::floor(grid_x + 0.5));
                    const int anchor_y =
                        degree_ == 3
                            ? static_cast<int>(std::floor(grid_y))
                            : static_cast<int>(std::floor(grid_y + 0.5));
                    const int start_x = std::max(
                        0,
                        std::min(
                            dims[0] - grid_side_, anchor_x - 1));
                    const int start_y = std::max(
                        0,
                        std::min(
                            dims[1] - grid_side_, anchor_y - 1));

                    std::vector<int> starts_x;
                    std::vector<int> starts_y;
                    if (degree_ == 2) {
                        const int floor_x =
                            static_cast<int>(std::floor(grid_x));
                        const int floor_y =
                            static_cast<int>(std::floor(grid_y));
                        for (int shift = -2; shift <= 0; ++shift) {
                            starts_x.push_back(std::max(
                                0,
                                std::min(
                                    dims[0] - grid_side_,
                                    floor_x + shift)));
                            starts_y.push_back(std::max(
                                0,
                                std::min(
                                    dims[1] - grid_side_,
                                    floor_y + shift)));
                        }
                        starts_x.push_back(start_x);
                        starts_y.push_back(start_y);
                        std::sort(starts_x.begin(), starts_x.end());
                        starts_x.erase(
                            std::unique(
                                starts_x.begin(), starts_x.end()),
                            starts_x.end());
                        std::sort(starts_y.begin(), starts_y.end());
                        starts_y.erase(
                            std::unique(
                                starts_y.begin(), starts_y.end()),
                            starts_y.end());
                    } else {
                        starts_x.push_back(start_x);
                        starts_y.push_back(start_y);
                    }

                    const bool allow_flip =
                        degree_ == 2
                        && options_.allow_virtual_side_flip
                        && fixed_center >= 0
                        && desired_inside
                        && layer == 0;

                    const auto build_candidate =
                        [&](int candidate_start_x,
                            int candidate_start_y,
                            bool flipped) {
                            Candidate candidate;
                            candidate.sample.query = query;
                            candidate.sample.nodes.resize(
                                static_cast<std::size_t>(
                                    grid_side_ * grid_side_));
                            candidate.start_x = candidate_start_x;
                            candidate.start_y = candidate_start_y;
                            candidate.flipped = flipped;
                            candidate.sample.virtual_side_flipped =
                                flipped;
                            const bool virtual_inside =
                                flipped
                                    ? !desired_inside
                                    : desired_inside;
                            const double node_correction_scale =
                                virtual_inside ? 1.0 : -1.0;
                            if (flipped) {
                                candidate.sample
                                    .query_correction_center =
                                    fixed_center;
                                candidate.sample
                                    .query_correction_scale =
                                    desired_inside ? 1.0 : -1.0;
                            }

                            const std::vector<double> wx =
                                lagrange_weights(
                                    grid_x
                                        - static_cast<double>(
                                            candidate_start_x + 1),
                                    grid_side_);
                            const std::vector<double> wy =
                                lagrange_weights(
                                    grid_y
                                        - static_cast<double>(
                                            candidate_start_y + 1),
                                    grid_side_);
                            int position = 0;
                            for (int local_y = 0;
                                 local_y < grid_side_;
                                 ++local_y) {
                                const int j =
                                    candidate_start_y + local_y;
                                for (int local_x = 0;
                                     local_x < grid_side_;
                                     ++local_x) {
                                    const int i =
                                        candidate_start_x + local_x;
                                    InterpolationNode& node =
                                        candidate.sample.nodes[
                                            static_cast<std::size_t>(
                                                position++)];
                                    node.grid_node = grid.index(i, j);
                                    node.weight =
                                        wx[static_cast<std::size_t>(
                                            local_x)]
                                        * wy[static_cast<std::size_t>(
                                            local_y)];
                                    node.point =
                                        structured_grid::point(
                                            grid, node.grid_node);
                                    candidate.weight_norm +=
                                        std::abs(node.weight);
                                    const bool node_inside =
                                        grid_pair.domain_label(
                                            node.grid_node)
                                        > 0;
                                    if (node_inside
                                        != desired_inside) {
                                        ++candidate
                                              .geometric_wrong_side_nodes;
                                    }
                                    if (node_inside == virtual_inside)
                                        continue;

                                    ++candidate.corrected_nodes;
                                    candidate.correction_weight +=
                                        std::abs(node.weight);
                                    if (flipped) {
                                        node.correction_center =
                                            fixed_center;
                                        node.correction_scale =
                                            node_correction_scale;
                                    }
                                }
                            }
                            return candidate;
                        };

                    std::vector<Candidate> direct_candidates;
                    std::vector<Candidate> flip_candidates;
                    direct_candidates.reserve(
                        starts_x.size() * starts_y.size());
                    if (allow_flip) {
                        flip_candidates.reserve(
                            starts_x.size() * starts_y.size());
                    }
                    for (int candidate_start_y : starts_y) {
                        for (int candidate_start_x : starts_x) {
                            direct_candidates.push_back(
                                build_candidate(
                                    candidate_start_x,
                                    candidate_start_y,
                                    false));
                            if (allow_flip) {
                                flip_candidates.push_back(
                                    build_candidate(
                                        candidate_start_x,
                                        candidate_start_y,
                                        true));
                            }
                        }
                    }

                    Candidate* best_flip = nullptr;
                    for (Candidate& candidate : flip_candidates) {
                        if (best_flip == nullptr
                            || candidate_less(
                                candidate, *best_flip)) {
                            best_flip = &candidate;
                        }
                    }

                    Candidate* selected = nullptr;
                    Candidate* default_direct = nullptr;
                    for (Candidate& candidate : direct_candidates) {
                        if (candidate.start_x == start_x
                            && candidate.start_y == start_y) {
                            default_direct = &candidate;
                            break;
                        }
                    }
                    if (default_direct == nullptr) {
                        throw std::runtime_error(
                            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D default interpolation stencil is missing");
                    }

                    for (InterpolationNode& node
                         : default_direct->sample.nodes) {
                        const bool node_inside =
                            grid_pair.domain_label(node.grid_node) > 0;
                        if (node_inside == desired_inside)
                            continue;

                        P2CrossingOwner2D owner;
                        if ((node.point - query).norm() <= 1.0e-14) {
                            owner.center_index =
                                grid_pair.nearest_p2_expansion_center(
                                    node.grid_node);
                            owner.status =
                                P2CrossingOwnerStatus2D::
                                    EndpointNearestCenter;
                        } else {
                            owner =
                                grid_pair.p2_crossing_owner_between(
                                    query, node.point);
                        }
                        node.correction_center = owner.center_index;
                        node.correction_crossing = owner;
                        node.correction_scale =
                            desired_inside ? 1.0 : -1.0;
                        switch (owner.status) {
                        case P2CrossingOwnerStatus2D::
                                 ExactIntersection:
                            ++default_direct->exact_owners;
                            break;
                        case P2CrossingOwnerStatus2D::GapFallback:
                            ++default_direct->gap_owners;
                            if (owner.explicit_gap_intersection)
                                ++default_direct->identified_gap_owners;
                            else
                                ++default_direct->unresolved_gap_owners;
                            break;
                        case P2CrossingOwnerStatus2D::
                                 EndpointNearestCenter:
                            ++default_direct->endpoint_owners;
                            break;
                        }
                    }

                    const bool default_is_exact =
                        default_direct->gap_owners == 0
                        && default_direct->endpoint_owners == 0;
                    if (default_is_exact || best_flip == nullptr) {
                        selected = default_direct;
                    } else {
                        // Only an ambiguous default stencil may switch
                        // virtual phase.  Smooth-edge samples retain the
                        // original centered 3x3 route exactly.
                        selected = best_flip;
                    }

                    if (selected == nullptr) {
                        throw std::runtime_error(
                            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D failed to select an interpolation route");
                    }
                    TraceSample& sample =
                        trace_samples_[static_cast<std::size_t>(
                            trace_sample_index(q, side, layer))];
                    sample = std::move(selected->sample);
                    diagnostics_.wrong_side_nodes +=
                        selected->geometric_wrong_side_nodes;
                    diagnostics_.corrected_nodes +=
                        selected->corrected_nodes;
                    diagnostics_.exact_crossing_owners +=
                        selected->exact_owners;
                    diagnostics_.gap_fallback_owners +=
                        selected->gap_owners;
                    diagnostics_.identified_gap_crossing_owners +=
                        selected->identified_gap_owners;
                    diagnostics_.unresolved_gap_fallback_owners +=
                        selected->unresolved_gap_owners;
                    diagnostics_.endpoint_fallback_owners +=
                        selected->endpoint_owners;
                    if (selected->flipped) {
                        ++diagnostics_.virtual_side_flips;
                        if (desired_inside)
                            ++diagnostics_.interior_sample_flips;
                        else
                            ++diagnostics_.exterior_sample_flips;
                    }
                    if (selected->start_x != start_x
                        || selected->start_y != start_y) {
                        ++diagnostics_.shifted_stencils;
                    }
                }
                if (uses_shared_side_spatial_polynomial()
                    && fixed_center >= 0) {
                    const TraceSample& reference =
                        trace_samples_[static_cast<std::size_t>(
                            trace_sample_index(q, side, 0))];
                    for (int layer = 1;
                         layer < layer_count_;
                         ++layer) {
                        const TraceSample& sample =
                            trace_samples_[static_cast<std::size_t>(
                                trace_sample_index(
                                    q, side, layer))];
                        if (sample.nodes.size()
                            != reference.nodes.size()) {
                            throw std::runtime_error(
                                "shared-side quadratic restrict changed "
                                "stencil size across normal layers");
                        }
                        for (std::size_t node = 0;
                             node < sample.nodes.size();
                             ++node) {
                            const InterpolationNode& lhs =
                                reference.nodes[node];
                            const InterpolationNode& rhs =
                                sample.nodes[node];
                            if (lhs.grid_node != rhs.grid_node
                                || lhs.correction_center
                                       != rhs.correction_center
                                || !same_crossing(
                                       lhs.correction_crossing,
                                       rhs.correction_crossing)
                                || lhs.secondary_correction_center
                                       != rhs
                                              .secondary_correction_center
                                || !same_crossing(
                                       lhs.secondary_correction_crossing,
                                       rhs.secondary_correction_crossing)
                                || lhs.correction_scale
                                       != rhs.correction_scale
                                || lhs.secondary_correction_scale
                                       != rhs
                                              .secondary_correction_scale) {
                                throw std::runtime_error(
                                    "shared-side quadratic restrict changed "
                                    "a grid node or crossing owner across "
                                    "normal layers");
                            }
                        }
                    }
                    ++diagnostics_.shared_side_spatial_polynomials;
                    diagnostics_
                        .shared_side_spatial_polynomial_samples +=
                        layer_count_;
                }
            }
        }
        for (const TraceSample& sample : trace_samples_) {
            diagnostics_.interpolation_nodes +=
                static_cast<int>(sample.nodes.size());
        }
    }

    const GridPair2D& grid_pair_;
    LaplaceP2JointPolynomialRestrictOptions2D options_;
    int degree_ = 3;
    int grid_side_ = 4;
    int layer_count_ = 4;
    int normal_sample_count_ = 8;
    int grid_nodes_per_sample_ = 16;
    double h_ = 0.0;
    std::vector<TraceSample> trace_samples_;
    std::vector<LaplaceNurbsTraceReferencePoint2D> trace_references_;
    std::vector<TraceReferencePlan> trace_reference_plans_;
    bool trace_references_configured_ = false;
    std::vector<int> trace_center_by_point_;
    std::vector<int> trace_panel_by_point_;
    Eigen::VectorXd c0_weights_;
    Eigen::VectorXd c1_weights_;
    LaplaceP2JointPolynomialRestrictDiagnostics2D diagnostics_;
};

LaplaceP2CrossingOwnerJointPolynomialRestrict2D::
LaplaceP2CrossingOwnerJointPolynomialRestrict2D(
    const GridPair2D& grid_pair,
    LaplaceP2JointPolynomialRestrictOptions2D options)
    : grid_pair_(grid_pair)
    , impl_(std::make_unique<Impl>(grid_pair, std::move(options)))
{}

LaplaceP2CrossingOwnerJointPolynomialRestrict2D::
~LaplaceP2CrossingOwnerJointPolynomialRestrict2D() = default;

const LaplaceP2JointPolynomialRestrictDiagnostics2D&
LaplaceP2CrossingOwnerJointPolynomialRestrict2D::diagnostics() const
{
    return impl_->diagnostics();
}

std::vector<LocalPoly2D>
LaplaceP2CrossingOwnerJointPolynomialRestrict2D::apply(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    return apply_impl(bulk_solution, spread_result, TraceRoute::Average);
}

std::vector<LocalPoly2D>
LaplaceP2CrossingOwnerJointPolynomialRestrict2D::apply_interior_virtual(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    return apply_impl(
        bulk_solution, spread_result, TraceRoute::InteriorVirtual);
}

std::vector<LocalPoly2D>
LaplaceP2CrossingOwnerJointPolynomialRestrict2D::apply_exterior_virtual(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    return apply_impl(
        bulk_solution, spread_result, TraceRoute::ExteriorVirtual);
}

void LaplaceP2CrossingOwnerJointPolynomialRestrict2D::
configure_trace_references(
    std::vector<LaplaceNurbsTraceReferencePoint2D> references)
{
    impl_->configure_trace_references(std::move(references));
}

const std::vector<LaplaceNurbsTraceReferencePoint2D>&
LaplaceP2CrossingOwnerJointPolynomialRestrict2D::trace_references() const
{
    return impl_->trace_references();
}

LaplaceRestrictedReferenceTrace2D
LaplaceP2CrossingOwnerJointPolynomialRestrict2D::
apply_exterior_virtual_at_references(
    const Eigen::VectorXd& bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    const CartesianGrid2D& grid = grid_pair_.grid();
    const Interface2D& iface = grid_pair_.interface();
    if (bulk_solution.size() != grid.num_dofs()) {
        throw std::invalid_argument(
            "Laplace P2 reference-trace bulk solution has invalid size");
    }
    if (spread_result.correction_method
            != LaplaceCorrectionMethod2D::CrossingOwner
        || spread_result.u_jump.size() != iface.num_points()
        || spread_result.un_jump.size() != iface.num_points()
        || spread_result.rhs_jump.size() != iface.num_points()) {
        throw std::invalid_argument(
            "Laplace P2 reference traces require a complete crossing-owner spread state");
    }
    if (spread_result.sd_corner_lifting.enabled
        || !spread_result.corner_patch_corrections.empty()
        || !spread_result.patch_regularized_correction_polys.empty()) {
        throw std::invalid_argument(
            "Laplace P2 reference traces do not support corner-patch singular corrections");
    }
    return impl_->apply_exterior_virtual_at_references(
        bulk_solution, spread_result);
}

std::vector<LocalPoly2D>
LaplaceP2CrossingOwnerJointPolynomialRestrict2D::apply_impl(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result,
    TraceRoute                   route) const
{
    const CartesianGrid2D& grid = grid_pair_.grid();
    const Interface2D& iface = grid_pair_.interface();
    if (bulk_solution.size() != grid.num_dofs()) {
        throw std::invalid_argument(
            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D bulk solution size must equal grid DOF count");
    }
    if (spread_result.correction_method
        != LaplaceCorrectionMethod2D::CrossingOwner) {
        throw std::invalid_argument(
            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D requires crossing-owner spread correction");
    }
    const int expected_centers =
        static_cast<int>(geometry2d::kP2CenterS.size())
        * iface.num_panels();
    if (static_cast<int>(spread_result.correction_polys.size())
        != expected_centers) {
        throw std::invalid_argument(
            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D requires one Cauchy polynomial per P2 expansion center");
    }
    if (spread_result.u_jump.size() != iface.num_points()
        || spread_result.un_jump.size() != iface.num_points()
        || spread_result.rhs_jump.size() != iface.num_points()) {
        throw std::invalid_argument(
            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D requires interface value, normal, and RHS jumps");
    }
    if (spread_result.sd_corner_lifting.enabled
        || !spread_result.corner_patch_corrections.empty()
        || !spread_result.patch_regularized_correction_polys.empty()) {
        throw std::invalid_argument(
            "LaplaceP2CrossingOwnerJointPolynomialRestrict2D does not support corner-patch singular corrections");
    }

    std::vector<LocalPoly2D> result(
        static_cast<std::size_t>(iface.num_points()));
    Eigen::VectorXd trace_values =
        Eigen::VectorXd::Zero(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const bool has_incident_trace_dof = impl_->has_incident_center(q);
        if (impl_->uses_interface_trace_point_cauchy_jump()
            && !has_incident_trace_dof) {
            const bool has_point_weights =
                iface.weights().size() == iface.num_points();
            const double weight_scale = has_point_weights
                ? iface.weights().cwiseAbs().maxCoeff()
                : 1.0;
            const double zero_weight_tolerance =
                64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, weight_scale);
            const double point_weight = has_point_weights
                ? iface.weights()[q]
                : 1.0;
            if (std::abs(point_weight) > zero_weight_tolerance) {
                throw std::runtime_error(
                    "LaplaceP2CrossingOwnerJointPolynomialRestrict2D DOF-Cauchy exterior route found an active trace point without an incident P2 panel");
            }
            LocalPoly2D inactive;
            inactive.center = iface.points().row(q).transpose();
            inactive.coeffs = Eigen::VectorXd::Zero(6);
            result[static_cast<std::size_t>(q)] = std::move(inactive);
            continue;
        }
        const bool dof_cauchy_exterior =
            impl_->uses_interface_dof_cauchy_exterior_route()
            && has_incident_trace_dof;
        const bool use_trace_point_cauchy =
            impl_->uses_interface_trace_point_cauchy_jump()
            && has_incident_trace_dof;
        LaplaceP2CrossingLocalPolynomial2D trace_point_jump;
        if (use_trace_point_cauchy) {
            trace_point_jump =
                impl_->interface_trace_point_cauchy_polynomial(
                    spread_result, q);
        }
        Eigen::VectorXd interior_virtual_samples;
        if (!dof_cauchy_exterior) {
            interior_virtual_samples.resize(
                impl_->normal_sample_count());
        }
        Eigen::VectorXd exterior_virtual_samples(
            impl_->normal_sample_count());
        for (int layer = 0; layer < impl_->layer_count(); ++layer) {
            const double inside = impl_->physical_sample(
                bulk_solution,
                spread_result,
                q,
                0,
                layer,
                dof_cauchy_exterior ? &trace_point_jump : nullptr);
            const double outside = impl_->physical_sample(
                bulk_solution,
                spread_result,
                q,
                1,
                layer,
                dof_cauchy_exterior ? &trace_point_jump : nullptr);
            if (dof_cauchy_exterior) {
                // The interior-side spatial interpolation first recovers the
                // physical interior value using the same DOF-local Cauchy
                // polynomial for every wrong-side node. Subtracting that
                // polynomial once at the normal query maps the sample
                // directly onto the exterior virtual solution. The exterior
                // samples already lie on that branch, so no second normal
                // continuation or branch average is needed.
                exterior_virtual_samples[layer] =
                    inside - trace_point_jump.evaluate(
                        impl_->trace_sample_query(q, 0, layer));
                exterior_virtual_samples[
                    impl_->layer_count() + layer] = outside;
                continue;
            }
            const double tau =
                impl_->options().normal_layers[
                    static_cast<std::size_t>(layer)];
            double jump_inside = 0.0;
            double jump_outside = 0.0;
            if (use_trace_point_cauchy) {
                jump_inside = trace_point_jump.evaluate(
                    impl_->trace_sample_query(q, 0, layer));
                jump_outside = trace_point_jump.evaluate(
                    impl_->trace_sample_query(q, 1, layer));
            } else if (impl_->uses_center_cauchy_jump()
                && impl_->has_incident_center(q)) {
                jump_inside = impl_->center_cauchy_jump(
                    spread_result, q, 0, layer);
                jump_outside = impl_->center_cauchy_jump(
                    spread_result, q, 1, layer);
            } else {
                // Geometry-only metadata points need not belong to a P2
                // panel.  They are inactive in the exterior-trace operator;
                // retain the original linear continuation so the full
                // Interface2D result remains defined.
                jump_inside =
                    spread_result.u_jump[q]
                    - tau * impl_->h() * spread_result.un_jump[q];
                jump_outside =
                    spread_result.u_jump[q]
                    + tau * impl_->h() * spread_result.un_jump[q];
            }

            interior_virtual_samples[layer] = inside;
            interior_virtual_samples[impl_->layer_count() + layer] =
                outside + jump_outside;
            exterior_virtual_samples[layer] =
                inside - jump_inside;
            exterior_virtual_samples[impl_->layer_count() + layer] =
                outside;
        }

        const std::pair<double, double> exterior_fit =
            impl_->joint_fit(exterior_virtual_samples);
        double value = 0.0;
        double normal_derivative = 0.0;
        if (dof_cauchy_exterior) {
            double trace_jump_scale = 0.0;
            switch (route) {
            case TraceRoute::Average:
                trace_jump_scale = 0.5;
                break;
            case TraceRoute::InteriorVirtual:
                trace_jump_scale = 1.0;
                break;
            case TraceRoute::ExteriorVirtual:
                trace_jump_scale = 0.0;
                break;
            }
            value = exterior_fit.first
                  + trace_jump_scale * trace_point_jump.value;
            normal_derivative = exterior_fit.second
                  + trace_jump_scale
                        * trace_point_jump.normal_derivative;
        } else {
            const std::pair<double, double> interior_fit =
                impl_->joint_fit(interior_virtual_samples);
            switch (route) {
            case TraceRoute::Average:
                value =
                    0.5 * (interior_fit.first + exterior_fit.first);
                normal_derivative =
                    0.5 * (interior_fit.second + exterior_fit.second);
                break;
            case TraceRoute::InteriorVirtual:
                value = interior_fit.first;
                normal_derivative = interior_fit.second;
                break;
            case TraceRoute::ExteriorVirtual:
                value = exterior_fit.first;
                normal_derivative = exterior_fit.second;
                break;
            }
        }

        Eigen::Vector2d normal = iface.normals().row(q).transpose();
        normal.normalize();
        LocalPoly2D poly;
        poly.center = iface.points().row(q).transpose();
        poly.coeffs = Eigen::VectorXd::Zero(6);
        poly.coeffs[0] = value;
        poly.coeffs[1] = normal_derivative * normal[0];
        poly.coeffs[2] = normal_derivative * normal[1];
        result[static_cast<std::size_t>(q)] = std::move(poly);
        trace_values[q] = value;
    }

    // Complete the Cartesian gradient with the P2 derivative of the recovered
    // trace. This does not change the fitted normal derivative, but preserves
    // the LocalPoly2D contract for callers that also inspect tangential data.
    std::vector<Eigen::Vector2d> tangential_gradients(
        static_cast<std::size_t>(iface.num_points()),
        Eigen::Vector2d::Zero());
    std::vector<int> tangential_counts(
        static_cast<std::size_t>(iface.num_points()), 0);
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (int local = 0; local < 3; ++local) {
            const int q = iface.point_index(panel, local);
            const double local_s =
                geometry2d::kP2NodeS[static_cast<std::size_t>(local)];
            const Eigen::Vector2d tangent =
                geometry2d::panel_tangent(iface, panel, local_s);
            const double speed = tangent.norm();
            if (!(speed > 1.0e-14))
                continue;
            const double trace_derivative =
                geometry2d::panel_scalar_deriv(
                    iface, panel, trace_values, local_s) / speed;
            tangential_gradients[static_cast<std::size_t>(q)] +=
                trace_derivative * tangent / speed;
            ++tangential_counts[static_cast<std::size_t>(q)];
        }
    }
    for (int q = 0; q < iface.num_points(); ++q) {
        const int count =
            tangential_counts[static_cast<std::size_t>(q)];
        if (count <= 0)
            continue;
        Eigen::Vector2d tangent_gradient =
            tangential_gradients[static_cast<std::size_t>(q)]
            / static_cast<double>(count);
        Eigen::Vector2d normal = iface.normals().row(q).transpose();
        normal.normalize();
        tangent_gradient -= normal * tangent_gradient.dot(normal);
        result[static_cast<std::size_t>(q)].coeffs[1] +=
            tangent_gradient[0];
        result[static_cast<std::size_t>(q)].coeffs[2] +=
            tangent_gradient[1];
    }
    return result;
}

} // namespace kfbim
