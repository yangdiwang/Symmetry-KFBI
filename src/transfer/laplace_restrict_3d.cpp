#include "laplace_restrict_3d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../geometry/p2_surface_3d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "laplace_projection_correction_3d.hpp"

namespace kfbim {

namespace {

constexpr int kExpansionCentersPerPanel3D = 16;

Eigen::Vector3d interface_point(const Interface3D& iface, int q)
{
    return iface.points().row(q).transpose();
}

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int idx)
{
    return structured_grid::point(grid, idx);
}

bool finite_vector(Eigen::Vector2d v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]);
}

bool reference_triangle_contains(Eigen::Vector3d bary, double tol)
{
    return bary[0] >= -tol
        && bary[1] >= -tol
        && bary[2] >= -tol
        && std::abs(bary.sum() - 1.0) <= tol;
}

Eigen::Vector2d clamp_uv_to_reference_triangle(Eigen::Vector2d uv)
{
    if (!finite_vector(uv))
        return Eigen::Vector2d::Zero();

    const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    if (reference_triangle_contains(bary, 0.0))
        return uv;

    Eigen::Vector2d best(0.0, std::max(0.0, std::min(1.0, uv[1])));
    double best_dist2 = (best - uv).squaredNorm();

    auto try_candidate = [&](Eigen::Vector2d candidate) {
        const double dist2 = (candidate - uv).squaredNorm();
        if (dist2 < best_dist2) {
            best = candidate;
            best_dist2 = dist2;
        }
    };

    try_candidate({std::max(0.0, std::min(1.0, uv[0])), 0.0});

    double edge_u = 0.5 * (uv[0] - uv[1] + 1.0);
    edge_u = std::max(0.0, std::min(1.0, edge_u));
    try_candidate({edge_u, 1.0 - edge_u});

    return best;
}

SurfaceProjection3D project_point_to_fixed_panel(
    const Interface3D& iface,
    int                grid_node,
    int                panel,
    int                component,
    Eigen::Vector3d    target,
    Eigen::Vector3d    initial_bary)
{
    constexpr int kMaxIterations = 30;
    constexpr double kResidualTol = 1.0e-11;
    constexpr double kStepTol = 1.0e-13;

    Eigen::Vector2d uv =
        geometry3d::uv_from_barycentric(initial_bary);
    uv = clamp_uv_to_reference_triangle(uv);

    const Eigen::Vector3d Xuu = geometry3d::panel_second_uu(iface, panel);
    const Eigen::Vector3d Xuv = geometry3d::panel_second_uv(iface, panel);
    const Eigen::Vector3d Xvv = geometry3d::panel_second_vv(iface, panel);

    int iterations = 0;
    bool converged = false;
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
        const Eigen::Vector3d point =
            geometry3d::panel_point(iface, panel, bary);
        const Eigen::Vector3d r = point - target;
        const Eigen::Vector3d Xu =
            geometry3d::panel_tangent_u(iface, panel, bary);
        const Eigen::Vector3d Xv =
            geometry3d::panel_tangent_v(iface, panel, bary);

        const Eigen::Vector2d equations{r.dot(Xu), r.dot(Xv)};
        Eigen::Matrix2d metric;
        metric << Xu.dot(Xu), Xu.dot(Xv),
                  Xv.dot(Xu), Xv.dot(Xv);
        double tangential_residual = equations.norm();
        if (std::abs(metric.determinant()) > 1.0e-28) {
            const Eigen::Vector2d coeff =
                metric.fullPivLu().solve(equations);
            tangential_residual =
                (coeff[0] * Xu + coeff[1] * Xv).norm();
        }
        if (tangential_residual <= kResidualTol
            && reference_triangle_contains(bary, 1.0e-8)) {
            converged = true;
            iterations = iter;
            break;
        }

        Eigen::Matrix2d J;
        J(0, 0) = Xu.dot(Xu) + r.dot(Xuu);
        J(0, 1) = Xv.dot(Xu) + r.dot(Xuv);
        J(1, 0) = Xu.dot(Xv) + r.dot(Xuv);
        J(1, 1) = Xv.dot(Xv) + r.dot(Xvv);
        if (std::abs(J.determinant()) <= 1.0e-28)
            break;

        const Eigen::Vector2d delta = J.fullPivLu().solve(equations);
        if (!finite_vector(delta))
            break;

        bool accepted = false;
        Eigen::Vector2d accepted_uv = uv;
        Eigen::Vector2d accepted_step = Eigen::Vector2d::Zero();
        const double merit = equations.squaredNorm();
        for (double damping = 1.0; damping >= 1.0 / 64.0; damping *= 0.5) {
            Eigen::Vector2d candidate = uv - damping * delta;
            if (candidate[0] < -0.5 || candidate[1] < -0.5
                || candidate[0] + candidate[1] > 1.5) {
                candidate = clamp_uv_to_reference_triangle(candidate);
            }
            if (!finite_vector(candidate))
                continue;
            const Eigen::Vector3d candidate_bary =
                geometry3d::barycentric_from_uv(candidate);
            const Eigen::Vector3d candidate_point =
                geometry3d::panel_point(iface, panel, candidate_bary);
            const Eigen::Vector3d candidate_r = candidate_point - target;
            const Eigen::Vector3d candidate_Xu =
                geometry3d::panel_tangent_u(iface, panel, candidate_bary);
            const Eigen::Vector3d candidate_Xv =
                geometry3d::panel_tangent_v(iface, panel, candidate_bary);
            const Eigen::Vector2d candidate_equations{
                candidate_r.dot(candidate_Xu),
                candidate_r.dot(candidate_Xv)};
            if (candidate_equations.squaredNorm() <= merit
                || damping <= 1.0 / 64.0) {
                accepted = true;
                accepted_uv = candidate;
                accepted_step = damping * delta;
                break;
            }
        }
        if (!accepted)
            break;

        uv = accepted_uv;
        iterations = iter + 1;
        if (accepted_step.norm() <= kStepTol)
            break;
    }

    Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    if (!reference_triangle_contains(bary, 1.0e-8)) {
        uv = clamp_uv_to_reference_triangle(uv);
        bary = geometry3d::barycentric_from_uv(uv);
        converged = false;
    }

    const Eigen::Vector3d point = geometry3d::panel_point(iface, panel, bary);
    const Eigen::Vector3d normal =
        geometry3d::panel_oriented_normal(iface, panel, bary);
    const Eigen::Vector3d r = point - target;
    const Eigen::Vector3d Xu = geometry3d::panel_tangent_u(iface, panel, bary);
    const Eigen::Vector3d Xv = geometry3d::panel_tangent_v(iface, panel, bary);
    const Eigen::Vector2d equations{r.dot(Xu), r.dot(Xv)};
    Eigen::Matrix2d metric;
    metric << Xu.dot(Xu), Xu.dot(Xv),
              Xv.dot(Xu), Xv.dot(Xv);
    double tangential_residual = equations.norm();
    if (std::abs(metric.determinant()) > 1.0e-28) {
        const Eigen::Vector2d coeff = metric.fullPivLu().solve(equations);
        tangential_residual = (coeff[0] * Xu + coeff[1] * Xv).norm();
    }

    SurfaceProjection3D projection;
    projection.grid_node = grid_node;
    projection.panel = panel;
    projection.component = component;
    projection.barycentric = bary;
    projection.point = point;
    projection.normal = normal;
    projection.signed_distance = (target - point).dot(normal);
    projection.distance = r.norm();
    projection.tangential_residual = tangential_residual;
    projection.iterations = iterations;
    projection.converged =
        converged || tangential_residual <= kResidualTol;
    return projection;
}

bool profile_projection_transfer_3d()
{
    return std::getenv("KFBIM_PROFILE_INTERFACE_3D") != nullptr
        || std::getenv("KFBIM_PROFILE_TRANSFER_3D") != nullptr;
}

using ProfileClock = std::chrono::steady_clock;

double seconds_since(ProfileClock::time_point start)
{
    return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

} // namespace

class LaplaceRestrictCorrectionEvaluator3D {
public:
    virtual ~LaplaceRestrictCorrectionEvaluator3D() = default;
    virtual double half_correction(int interface_point,
                                   int anchor_node,
                                   int grid_node,
                                   Eigen::Vector3d pt) const = 0;
    virtual double bulk_branch_restore(int interface_point,
                                       int grid_node,
                                       Eigen::Vector3d pt) const = 0;
    virtual void restore_physical_average_trace(
        int interface_point,
        LocalPoly3D& poly) const = 0;
};

namespace {

class NearestExpansionCenterRestrictCorrectionEvaluator3D final
    : public LaplaceRestrictCorrectionEvaluator3D {
public:
    NearestExpansionCenterRestrictCorrectionEvaluator3D(
        const GridPair3D& grid_pair,
        const std::vector<int>& owner_center_by_point,
        const LaplaceSpreadResult3D& spread_result)
        : grid_pair_(grid_pair)
        , iface_(grid_pair.interface())
        , owner_center_by_point_(owner_center_by_point)
        , correction_polys_(spread_result.correction_polys)
        , edge_regularized_polys_(
              spread_result.edge_regularized_correction_polys)
        , edge_patches_(spread_result.edge_regularization_patches)
        , edge_restrict_data_(spread_result.edge_restrict_data)
    {}

    double half_correction(int interface_point,
                           int,
                           int,
                           Eigen::Vector3d pt) const override
    {
        if (interface_point < 0
            || interface_point >= static_cast<int>(owner_center_by_point_.size())) {
            throw std::out_of_range(
                "LaplaceQuadraticPatchCenterRestrict3D interface point is outside owner cache");
        }
        const int center_idx = owner_center_by_point_[
            static_cast<std::size_t>(interface_point)];
        if (center_idx < 0
            || center_idx >= static_cast<int>(correction_polys_.size())) {
            throw std::runtime_error(
                "LaplaceQuadraticPatchCenterRestrict3D invalid interface-point owner center");
        }
        const LocalPoly3D* poly = &correction_polys_[center_idx];
        if (!edge_patches_.empty()) {
            const Eigen::Vector3d interface_location =
                iface_.points().row(interface_point).transpose();
            bool inside_edge_patch = false;
            for (const geometry3d::EdgePatch3D& patch : edge_patches_) {
                if (geometry3d::edge_patch_contains_point_3d(
                        patch, interface_location)) {
                    inside_edge_patch = true;
                    break;
                }
            }
            if (inside_edge_patch) {
                if (const LocalPoly3D* regularized =
                        find_indexed_local_poly_3d(edge_regularized_polys_,
                                                   center_idx)) {
                    poly = regularized;
                }
            }
        }
        return 0.5 * evaluate_taylor_poly_3d(*poly, pt);
    }

    double bulk_branch_restore(int interface_point,
                               int grid_node,
                               Eigen::Vector3d pt) const override
    {
        if (interface_point < 0
            || interface_point >= iface_.num_points()) {
            throw std::out_of_range(
                "LaplaceQuadraticPatchCenterRestrict3D interface point is outside edge restore data");
        }
        double restore = 0.0;
        const Eigen::Vector3d interface_location =
            iface_.points().row(interface_point).transpose();
        for (const EdgePatchRestrictData3D& data : edge_restrict_data_) {
            if (grid_node < 0
                || grid_node >= data.continuation_grid_values.size()) {
                throw std::out_of_range(
                    "LaplaceQuadraticPatchCenterRestrict3D grid node is outside edge restore data");
            }
            if (grid_pair_.domain_label(grid_node)
                != data.active_domain_label) {
                continue;
            }
            const bool target_inside =
                geometry3d::edge_patch_contains_point_3d(
                    data.patch, interface_location);
            const bool node_inside =
                geometry3d::edge_patch_contains_point_3d(data.patch, pt);
            restore += (static_cast<double>(node_inside)
                        - static_cast<double>(target_inside))
                     * data.continuation_grid_values[grid_node];
        }
        return restore;
    }

    void restore_physical_average_trace(
        int interface_point,
        LocalPoly3D& poly) const override
    {
        if (interface_point < 0
            || interface_point >= iface_.num_points()) {
            throw std::out_of_range(
                "LaplaceQuadraticPatchCenterRestrict3D interface point is outside edge trace data");
        }
        const Eigen::Vector3d interface_location =
            iface_.points().row(interface_point).transpose();
        for (const EdgePatchRestrictData3D& data : edge_restrict_data_) {
            if (!geometry3d::edge_patch_contains_point_3d(
                    data.patch, interface_location)) {
                continue;
            }
            if (interface_point >= data.interface_values.size()
                || interface_point >= data.interface_gradients.rows()) {
                throw std::runtime_error(
                    "LaplaceQuadraticPatchCenterRestrict3D edge trace data size mismatch");
            }
            poly.coeffs[0] += 0.5 * data.interface_values[interface_point];
            poly.coeffs[1] +=
                0.5 * data.interface_gradients(interface_point, 0);
            poly.coeffs[2] +=
                0.5 * data.interface_gradients(interface_point, 1);
            poly.coeffs[3] +=
                0.5 * data.interface_gradients(interface_point, 2);
        }
    }

private:
    const GridPair3D& grid_pair_;
    const Interface3D& iface_;
    const std::vector<int>& owner_center_by_point_;
    const std::vector<LocalPoly3D>& correction_polys_;
    const IndexedLocalPolys3D& edge_regularized_polys_;
    const std::vector<geometry3d::EdgePatch3D>& edge_patches_;
    const std::vector<EdgePatchRestrictData3D>& edge_restrict_data_;
};

class ProjectionPointRestrictCorrectionEvaluator3D final
    : public LaplaceRestrictCorrectionEvaluator3D {
public:
    ProjectionPointRestrictCorrectionEvaluator3D(
        const GridPair3D& grid_pair,
        const LaplaceSpreadResult3D& spread_result,
        const std::vector<int>& owner_center_by_point)
        : grid_pair_(grid_pair)
        , spread_result_(spread_result)
        , owner_center_by_point_(owner_center_by_point)
    {}

    double half_correction(int interface_point,
                           int,
                           int grid_node,
                           Eigen::Vector3d) const override
    {
        if (interface_point < 0
            || interface_point >= static_cast<int>(owner_center_by_point_.size())) {
            throw std::out_of_range(
                "LaplaceQuadraticPatchCenterRestrict3D interface point is outside owner cache");
        }
        const int owner_center = owner_center_by_point_[
            static_cast<std::size_t>(interface_point)];
        const int panel = owner_center / kExpansionCentersPerPanel3D;
        const int local_center = owner_center % kExpansionCentersPerPanel3D;
        const auto center_barycentrics =
            geometry3d::expansion_center_barycentrics();
        if (owner_center < 0
            || panel < 0
            || panel >= grid_pair_.interface().num_panels()
            || local_center < 0
            || local_center >= static_cast<int>(center_barycentrics.size())) {
            throw std::runtime_error(
                "LaplaceQuadraticPatchCenterRestrict3D invalid projection owner center");
        }
        const SurfaceProjection3D projection =
            project_point_to_fixed_panel(grid_pair_.interface(),
                                         grid_node,
                                         panel,
                                         grid_pair_.interface().panel_components()[panel],
                                         grid_point(grid_pair_.grid(), grid_node),
                                         center_barycentrics[
                                             static_cast<std::size_t>(local_center)]);
        return 0.5 * evaluate_projection_point_correction_3d(
                         grid_pair_.interface(),
                         projection,
                         spread_result_);
    }

    double bulk_branch_restore(int,
                               int,
                               Eigen::Vector3d) const override
    {
        return 0.0;
    }

    void restore_physical_average_trace(int,
                                        LocalPoly3D&) const override
    {}

private:
    const GridPair3D& grid_pair_;
    const LaplaceSpreadResult3D& spread_result_;
    const std::vector<int>& owner_center_by_point_;
};

std::unique_ptr<LaplaceRestrictCorrectionEvaluator3D> make_restrict_correction_evaluator(
    const GridPair3D& grid_pair,
    const LaplaceSpreadResult3D& spread_result,
    int expected_centers,
    const std::vector<int>& owner_center_by_point)
{
    const auto& iface = grid_pair.interface();
    if (spread_result.correction_method
        == LaplaceCorrectionMethod3D::NearestExpansionCenter) {
        if (static_cast<int>(spread_result.correction_polys.size())
            != expected_centers) {
            throw std::invalid_argument(
                "LaplaceQuadraticPatchCenterRestrict3D correction_polys size must equal 16*num_panels");
        }
        return std::make_unique<NearestExpansionCenterRestrictCorrectionEvaluator3D>(
            grid_pair, owner_center_by_point, spread_result);
    }

    if (spread_result.correction_method
        == LaplaceCorrectionMethod3D::ProjectionPoint) {
        require_projection_surface_data(iface, spread_result);
        if (profile_projection_transfer_3d()) {
            std::printf("      restrict projection_cache nodes=%zu projections=%zu\n",
                        spread_result.projection_cache.nodes().size(),
                        spread_result.projection_cache.projections().size());
        }
        return std::make_unique<ProjectionPointRestrictCorrectionEvaluator3D>(
            grid_pair, spread_result, owner_center_by_point);
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPatchCenterRestrict3D unsupported correction method");
}

} // namespace

void restore_edge_patch_physical_solution_3d(
    Eigen::VectorXd& u_bulk,
    const LaplaceSpreadResult3D& spread_result)
{
    for (const EdgePatchRestrictData3D& data
         : spread_result.edge_restrict_data) {
        if (data.physical_grid_values.size() != u_bulk.size()) {
            throw std::invalid_argument(
                "edge-patch physical restore vector size mismatch");
        }
        u_bulk += data.physical_grid_values;
    }
}

LaplaceQuadraticPatchCenterRestrict3D::LaplaceQuadraticPatchCenterRestrict3D(
    const GridPair3D& grid_pair,
    int               stencil_radius)
    : grid_pair_(grid_pair)
    , stencil_radius_(stencil_radius)
    , support_(build_laplace_correction_support_3d(
          grid_pair,
          "LaplaceQuadraticPatchCenterRestrict3D"))
{
    if (stencil_radius_ < 1)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterRestrict3D stencil_radius must be positive");

    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const auto spacing = grid.spacing();
    owner_center_by_point_.resize(static_cast<std::size_t>(iface.num_points()));
    fit_inverse_dimensionless_by_point_.resize(
        static_cast<std::size_t>(iface.num_points()));
    fit_rcond_by_point_.resize(static_cast<std::size_t>(iface.num_points()));

    for (int q = 0; q < iface.num_points(); ++q) {
        owner_center_by_point_[static_cast<std::size_t>(q)] =
            grid_pair_.nearest_p2_expansion_center_for_interface_point(q);

        const Eigen::Vector3d center = interface_point(iface, q);
        Eigen::Matrix<double, 10, 10> dimensionless_fit;
        const std::array<int, 10>& stencil_nodes =
            support_.restrict_stencils[static_cast<std::size_t>(q)];
        for (int r = 0; r < static_cast<int>(stencil_nodes.size()); ++r) {
            const Eigen::Vector3d pt = grid_point(grid, stencil_nodes[r]);
            const double x = (pt[0] - center[0]) / spacing[0];
            const double y = (pt[1] - center[1]) / spacing[1];
            const double z = (pt[2] - center[2]) / spacing[2];
            dimensionless_fit(r, 0) = 1.0;
            dimensionless_fit(r, 1) = x;
            dimensionless_fit(r, 2) = y;
            dimensionless_fit(r, 3) = z;
            dimensionless_fit(r, 4) = 0.5 * x * x;
            dimensionless_fit(r, 5) = x * y;
            dimensionless_fit(r, 6) = x * z;
            dimensionless_fit(r, 7) = 0.5 * y * y;
            dimensionless_fit(r, 8) = y * z;
            dimensionless_fit(r, 9) = 0.5 * z * z;
        }

        Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> lu(dimensionless_fit);
        const double rcond = lu.rcond();
        if (!lu.isInvertible() || !std::isfinite(rcond) || rcond <= 1.0e-10) {
            throw std::runtime_error(
                "LaplaceQuadraticPatchCenterRestrict3D ill-conditioned dimensionless fixed stencil");
        }
        fit_inverse_dimensionless_by_point_[static_cast<std::size_t>(q)] =
            lu.solve(Eigen::Matrix<double, 10, 10>::Identity());
        fit_rcond_by_point_[static_cast<std::size_t>(q)] = rcond;
    }
}

std::vector<LocalPoly3D> LaplaceQuadraticPatchCenterRestrict3D::apply(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult3D& spread_result) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int expected_centers =
        kExpansionCentersPerPanel3D * iface.num_panels();

    if (bulk_solution.size() != grid.num_dofs())
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterRestrict3D bulk_solution size must equal grid DOF count");
    if (iface.points_per_panel() != 6
        || iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterRestrict3D requires P2 QuadraticLagrange panels");
    }

    const auto correction_evaluator =
        make_restrict_correction_evaluator(grid_pair_,
                                           spread_result,
                                           expected_centers,
                                           owner_center_by_point_);

    const ProfileClock::time_point fit_start = ProfileClock::now();
    std::vector<LocalPoly3D> result(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        result[q] = fit_at_interface_point(bulk_solution,
                                           q,
                                           *correction_evaluator);
    }
    const double t_fit = seconds_since(fit_start);
    if (profile_projection_transfer_3d()) {
        std::printf("      restrict fit iface=%d cache_radius=%d fit %.3fs\n",
                    n_iface,
                    stencil_radius_,
                    t_fit);
    }

    return result;
}

LocalPoly3D LaplaceQuadraticPatchCenterRestrict3D::fit_at_interface_point(
    const Eigen::VectorXd&                      bulk_solution,
    int                                         q,
    const LaplaceRestrictCorrectionEvaluator3D& correction_evaluator) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const Eigen::Vector3d center = interface_point(iface, q);
    const std::array<int, 10>& stencil_nodes = support_.restrict_stencils[q];
    const int anchor_node = grid_pair_.closest_bulk_node(q);

    Eigen::Matrix<double, 10, 1> rhs;
    for (int r = 0; r < static_cast<int>(stencil_nodes.size()); ++r) {
        const int idx = stencil_nodes[r];
        const Eigen::Vector3d pt = grid_point(grid, idx);

        double val = bulk_solution[idx];
        val += correction_evaluator.bulk_branch_restore(q, idx, pt);
        const double correction =
            correction_evaluator.half_correction(q, anchor_node, idx, pt);
        if (grid_pair_.domain_label(idx) == 0)
            val += correction;
        else
            val -= correction;
        rhs[r] = val;
    }

    const Eigen::Matrix<double, 10, 1> dimensionless_coeffs =
        fit_inverse_dimensionless_by_point_[static_cast<std::size_t>(q)] * rhs;
    const auto spacing = grid.spacing();

    LocalPoly3D poly;
    poly.center = center;
    poly.coeffs = Eigen::VectorXd::Zero(10);
    poly.coeffs[0] = dimensionless_coeffs[0];
    poly.coeffs[1] = dimensionless_coeffs[1] / spacing[0];
    poly.coeffs[2] = dimensionless_coeffs[2] / spacing[1];
    poly.coeffs[3] = dimensionless_coeffs[3] / spacing[2];
    poly.coeffs[4] = dimensionless_coeffs[4] / (spacing[0] * spacing[0]);
    poly.coeffs[5] = dimensionless_coeffs[5] / (spacing[0] * spacing[1]);
    poly.coeffs[6] = dimensionless_coeffs[6] / (spacing[0] * spacing[2]);
    poly.coeffs[7] = dimensionless_coeffs[7] / (spacing[1] * spacing[1]);
    poly.coeffs[8] = dimensionless_coeffs[8] / (spacing[1] * spacing[2]);
    poly.coeffs[9] = dimensionless_coeffs[9] / (spacing[2] * spacing[2]);
    correction_evaluator.restore_physical_average_trace(q, poly);
    return poly;
}

int LaplaceQuadraticPatchCenterRestrict3D::owner_center(
    int interface_point) const
{
    if (interface_point < 0
        || interface_point >= static_cast<int>(owner_center_by_point_.size())) {
        throw std::out_of_range(
            "LaplaceQuadraticPatchCenterRestrict3D owner query is outside the interface");
    }
    return owner_center_by_point_[static_cast<std::size_t>(interface_point)];
}

int LaplaceQuadraticPatchCenterRestrict3D::owner_panel(
    int interface_point) const
{
    return owner_center(interface_point) / kExpansionCentersPerPanel3D;
}

double LaplaceQuadraticPatchCenterRestrict3D::dimensionless_fit_rcond(
    int interface_point) const
{
    if (interface_point < 0
        || interface_point >= static_cast<int>(fit_rcond_by_point_.size())) {
        throw std::out_of_range(
            "LaplaceQuadraticPatchCenterRestrict3D fit-condition query is outside the interface");
    }
    return fit_rcond_by_point_[static_cast<std::size_t>(interface_point)];
}

} // namespace kfbim
