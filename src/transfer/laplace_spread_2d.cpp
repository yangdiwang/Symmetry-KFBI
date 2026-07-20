#include "laplace_spread_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include "../geometry/corner_patch_2d.hpp"
#include "../geometry/p2_projection_2d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../local_cauchy/laplace_corner_patch_solver_2d.hpp"
#include "../local_cauchy/laplace_panel_solver_2d.hpp"
#include "laplace_projection_correction_2d.hpp"

namespace kfbim {

namespace {

bool is_outer_boundary_node(const CartesianGrid2D& grid, int idx) {
    return structured_grid::is_boundary_node(grid, idx);
}

int side_from_label(int label) {
    return label == 0 ? 0 : 1;
}

double stencil_weight_for_neighbor(const CartesianGrid2D& grid, int neighbor_slot) {
    return structured_grid::stencil_weight_for_neighbor(grid, neighbor_slot);
}

Eigen::Vector2d node_coord(const CartesianGrid2D& grid, int idx) {
    return structured_grid::point(grid, idx);
}

LocalPoly2D center_poly(const PanelCenterCauchyResult2D& cauchy, int idx) {
    LocalPoly2D poly;
    poly.center = cauchy.centers.row(idx).transpose();
    poly.coeffs.resize(6);
    poly.coeffs << cauchy.C[idx],
                   cauchy.Cx[idx],
                   cauchy.Cy[idx],
                   cauchy.Cxx[idx],
                   cauchy.Cxy[idx],
                   cauchy.Cyy[idx];
    return poly;
}

PanelCenterJumpEvaluator2D make_patch_regularized_panel_jump_evaluator(
    const Interface2D& iface,
    const CornerPatchCorrectionData2D& data,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump_values,
    double alpha,
    const std::vector<int>* selected_centers = nullptr)
{
    struct PanelLocalJumpCache2D {
        bool direct_regular = false;
        bool initialized = false;
        std::array<double, 3> value{};
        std::array<double, 3> normal{};
        std::array<double, 3> rhs{};
    };

    auto panel_cache =
        std::make_shared<std::vector<PanelLocalJumpCache2D>>(iface.num_panels());
    auto fill_panel_cache =
        [&iface,
         &data,
         &u_jump,
         &un_jump,
         &rhs_jump_values,
         panel_cache](int panel) {
        if (panel < 0 || panel >= iface.num_panels())
            return;
        PanelLocalJumpCache2D& cached =
            (*panel_cache)[static_cast<std::size_t>(panel)];
        if (cached.initialized)
            return;
        cached.initialized = true;
        cached.direct_regular = false;

        const bool subtract_singular =
            laplace_projection_correction_detail::
                panel_is_in_corner_patch_support(data, panel);
        for (int local = 0; local < 3; ++local) {
            const int q = iface.point_index(panel, local);
            const Eigen::Vector2d node_pt =
                iface.points().row(q).transpose();
            const Eigen::Vector2d node_normal =
                iface.panel_normal(panel, local);
            cached.value[static_cast<std::size_t>(local)] = u_jump[q];
            cached.normal[static_cast<std::size_t>(local)] = un_jump[q];
            if (subtract_singular) {
                cached.value[static_cast<std::size_t>(local)] -=
                    evaluate_corner_patch_singular_jump_trace_2d(data, node_pt);
                cached.normal[static_cast<std::size_t>(local)] -=
                    evaluate_corner_patch_singular_normal_jump_trace_2d(
                        data, node_pt, node_normal);
            }
            cached.rhs[static_cast<std::size_t>(local)] = rhs_jump_values[q];
        }
    };

    if (selected_centers != nullptr) {
        std::vector<int> selected_panels;
        selected_panels.reserve(selected_centers->size());
        for (int center_idx : *selected_centers) {
            if (center_idx < 0)
                continue;
            selected_panels.push_back(
                center_idx / detail::kQuadraticPanelExpansionCount);
        }
        std::sort(selected_panels.begin(), selected_panels.end());
        selected_panels.erase(
            std::unique(selected_panels.begin(), selected_panels.end()),
            selected_panels.end());
        for (int panel : selected_panels)
            fill_panel_cache(panel);
    } else {
        for (int panel = 0; panel < iface.num_panels(); ++panel)
            fill_panel_cache(panel);
    }

    return [&data, panel_cache, fill_panel_cache, alpha](int panel,
                                   double local_s,
                                   Eigen::Vector2d pt,
                                   Eigen::Vector2d normal,
                                   double& value_jump,
                                   double& normal_jump,
                                   double& rhs_jump) {
        fill_panel_cache(panel);
        const PanelLocalJumpCache2D& cached =
            (*panel_cache)[static_cast<std::size_t>(panel)];
        if (cached.direct_regular) {
            const LocalPoly2D& regular = data.interface_regular_part;
            value_jump = evaluate_taylor_poly_2d(regular, pt);
            normal_jump =
                evaluate_taylor_poly_gradient_2d(regular, pt).dot(normal);
            rhs_jump =
                alpha * value_jump
                - laplace_projection_correction_detail::
                      evaluate_taylor_poly_laplacian_2d(regular, pt);
            return true;
        }

        value_jump =
            laplace_projection_correction_detail::
                panel_scalar_from_local_values(cached.value.data(), local_s);
        normal_jump =
            laplace_projection_correction_detail::
                panel_scalar_from_local_values(cached.normal.data(), local_s);
        rhs_jump =
            laplace_projection_correction_detail::
                panel_scalar_from_local_values(cached.rhs.data(), local_s);
    return true;
};
}

PanelCenterJumpEvaluator2D make_sd_patch_regularized_panel_jump_evaluator(
    const Interface2D& iface,
    const SdCornerLiftingOptions2D& sd_lifting,
    int patch_id,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump_values)
{
    return [&iface, &sd_lifting, patch_id, &u_jump, &un_jump, &rhs_jump_values](
               int panel,
               double local_s,
               Eigen::Vector2d pt,
               Eigen::Vector2d normal,
               double& value_jump,
               double& normal_jump,
               double& rhs_jump) {
        value_jump = geometry2d::panel_scalar(iface, panel, u_jump, local_s);
        normal_jump =
            geometry2d::panel_scalar(iface, panel, un_jump, local_s);
        rhs_jump =
            geometry2d::panel_scalar(iface, panel, rhs_jump_values, local_s);

        if (patch_id >= 0
            && patch_id < static_cast<int>(iface.corner_patches().size())) {
            if (sd_lifting.regularized_jump_mode
                == SdCornerRegularizedJumpMode2D::InterpolateResidualAtP2Nodes) {
                double shape[3];
                geometry2d::p2_shape(local_s, shape);
                value_jump = 0.0;
                for (int local_q = 0; local_q < 3; ++local_q) {
                    const int q = iface.point_index(panel, local_q);
                    const Eigen::Vector2d q_pt =
                        iface.points().row(q).transpose();
                    const Eigen::Vector2d q_normal =
                        iface.normals().row(q).transpose();
                    value_jump +=
                        shape[local_q]
                        * (u_jump[q]
                           - sd_lifting.lifting.jump_value_for_patch(
                                 iface,
                                 patch_id,
                                 q_pt,
                                 q_normal,
                                 sd_lifting.normal_probe));
                }
            } else {
                value_jump -=
                    sd_lifting.lifting.jump_value_for_patch(
                        iface,
                        patch_id,
                        pt,
                        normal,
                        sd_lifting.normal_probe);
            }
        }
        return true;
    };
}

SdCornerLiftingOptions2D effective_sd_corner_lifting_options(
    const SdCornerLiftingOptions2D& options,
    const Interface2D& iface,
    const Eigen::VectorXd& u_jump)
{
    SdCornerLiftingOptions2D effective = options;
    if (!effective.enabled)
        return effective;

    if (effective.strength_source
        == SdCornerStrengthSource2D::P2EndpointFromUJump) {
        effective.lifting =
            build_p2_endpoint_sd_lifting_from_u_jump(
                iface, u_jump, effective.normal_probe);
    }
    return effective;
}

double evaluate_interface_jump_correction_for_node(
    const GridPair2D& grid_pair,
    const std::vector<LocalPoly2D>& full_polys,
    const std::vector<IndexedLocalPolys2D>& artificial_polys_by_patch,
    int forced_patch,
    int rhs_node,
    int correction_node,
    const Eigen::Vector2d& pt,
    int cached_center_idx = -1)
{
    const int center_idx =
        cached_center_idx >= 0
            ? cached_center_idx
            : grid_pair.nearest_p2_expansion_center_between(rhs_node,
                                                            correction_node);
    const LocalPoly2D* patch_poly = nullptr;
    if (forced_patch >= 0
        && forced_patch < static_cast<int>(artificial_polys_by_patch.size())
        && center_idx >= 0) {
        patch_poly = find_indexed_local_poly_2d(
            artificial_polys_by_patch[static_cast<std::size_t>(forced_patch)],
            center_idx);
    }
    if (patch_poly != nullptr) {
        return evaluate_taylor_poly_2d(*patch_poly, pt);
    }
    return evaluate_taylor_poly_2d(full_polys[center_idx], pt);
}

double evaluate_corner_patch_boundary_difference_for_node(
    const Interface2D& iface,
    const std::vector<CornerPatchCorrectionData2D>& corrections,
    const LaplaceCrossingCorrectionOp& op,
    Eigen::Vector2d pt)
{
    auto singular_value = [&](int patch) {
        if (patch < 0)
            return 0.0;
        const CornerPatchCorrectionData2D& data =
            laplace_projection_correction_detail::
                corner_patch_correction_by_patch(corrections, patch);
        return evaluate_corner_patch_singular_2d(data, pt);
    };
    return singular_value(op.rhs_patch)
         - singular_value(op.correction_patch);
}

PatchBoundarySingularValueBasis2D build_patch_boundary_singular_value_basis(
    const std::vector<CornerPatchCorrectionData2D>& corrections,
    int patch,
    const Eigen::Vector2d& pt)
{
    PatchBoundarySingularValueBasis2D basis;
    basis.patch = patch;
    if (patch < 0)
        return basis;
    const CornerPatchCorrectionData2D& data =
        laplace_projection_correction_detail::
            corner_patch_correction_by_patch(corrections, patch);
    if (!data.modes.empty()) {
        const CornerPatch2D geometry = corner_patch_data_geometry_2d(data);
        basis.mode_basis.reserve(data.modes.size());
        for (const CornerPatchSingularMode2D& mode : data.modes) {
            basis.mode_basis.push_back(
                laplace_corner_patch_detail::
                    evaluate_singular_mode_basis(geometry, mode, pt));
        }
    }
    basis.legacy_basis = evaluate_corner_patch_singular_basis_2d(data, pt);
    return basis;
}

std::vector<PatchBoundarySingularDifferenceBasis2D>
build_patch_boundary_singular_basis_by_crossing(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support,
    const std::vector<CornerPatchCorrectionData2D>& corrections)
{
    std::vector<PatchBoundarySingularDifferenceBasis2D> bases(
        support.crossing_ops.size());
    const CartesianGrid2D& grid = grid_pair.grid();
    for (std::size_t op_idx = 0; op_idx < support.crossing_ops.size();
         ++op_idx) {
        const LaplaceCrossingCorrectionOp& op =
            support.crossing_ops[op_idx];
        if (op.kind != LaplaceCrossingKind::CornerPatchBoundary)
            continue;
        const Eigen::Vector2d pt = node_coord(grid, op.correction_node);
        PatchBoundarySingularDifferenceBasis2D basis;
        basis.rhs = build_patch_boundary_singular_value_basis(
            corrections, op.rhs_patch, pt);
        basis.correction = build_patch_boundary_singular_value_basis(
            corrections, op.correction_patch, pt);
        basis.valid = true;
        bases[op_idx] = std::move(basis);
    }
    return bases;
}

double evaluate_cached_patch_boundary_singular_value(
    const std::vector<CornerPatchCorrectionData2D>& corrections,
    const PatchBoundarySingularValueBasis2D& basis,
    const Eigen::Vector2d& pt)
{
    if (basis.patch < 0)
        return 0.0;
    const CornerPatchCorrectionData2D& data =
        laplace_projection_correction_detail::
            corner_patch_correction_by_patch(corrections, basis.patch);
    if (data.modes.empty())
        return data.amplitude * basis.legacy_basis;
    if (basis.mode_basis.size() == data.modes.size()) {
        double value = 0.0;
        for (std::size_t k = 0; k < basis.mode_basis.size(); ++k)
            value += data.modes[k].amplitude * basis.mode_basis[k];
        return value;
    }
    return evaluate_corner_patch_singular_2d(data, pt);
}

double evaluate_corner_patch_boundary_difference_for_node_cached(
    const std::vector<CornerPatchCorrectionData2D>& corrections,
    const LaplaceCrossingCorrectionOp& op,
    const Eigen::Vector2d& pt,
    const PatchBoundarySingularDifferenceBasis2D* basis)
{
    if (basis == nullptr || !basis->valid) {
        const Interface2D* unused_iface = nullptr;
        (void)unused_iface;
        auto singular_value = [&](int patch) {
            if (patch < 0)
                return 0.0;
            const CornerPatchCorrectionData2D& data =
                laplace_projection_correction_detail::
                    corner_patch_correction_by_patch(corrections, patch);
            return evaluate_corner_patch_singular_2d(data, pt);
        };
        return singular_value(op.rhs_patch)
             - singular_value(op.correction_patch);
    }
    return evaluate_cached_patch_boundary_singular_value(
               corrections, basis->rhs, pt)
         - evaluate_cached_patch_boundary_singular_value(
               corrections, basis->correction, pt);
}

std::vector<int> patch_regularized_center_indices_for_spread(
    const LaplaceCorrectionSupport2D& support,
    const std::vector<int>& crossing_center_indices,
    int patch)
{
    std::vector<int> centers;
    for (std::size_t op_idx = 0; op_idx < support.crossing_ops.size();
         ++op_idx) {
        const LaplaceCrossingCorrectionOp& op =
            support.crossing_ops[op_idx];
        if (op.kind != LaplaceCrossingKind::InterfaceJump
            || op.patch != patch) {
            continue;
        }
        const int center_idx =
            op_idx < crossing_center_indices.size()
                ? crossing_center_indices[op_idx]
                : -1;
        if (center_idx >= 0)
            centers.push_back(center_idx);
    }
    std::sort(centers.begin(), centers.end());
    centers.erase(std::unique(centers.begin(), centers.end()), centers.end());
    return centers;
}

std::vector<int> build_crossing_center_indices_for_spread(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support)
{
    std::vector<int> center_indices(support.crossing_ops.size(), -1);
    for (std::size_t op_idx = 0; op_idx < support.crossing_ops.size();
         ++op_idx) {
        const LaplaceCrossingCorrectionOp& op =
            support.crossing_ops[op_idx];
        if (op.kind != LaplaceCrossingKind::InterfaceJump)
            continue;
        center_indices[op_idx] =
            grid_pair.nearest_p2_expansion_center_between(op.rhs_node,
                                                          op.correction_node);
    }
    return center_indices;
}

std::vector<std::vector<int>> build_patch_regularized_center_indices_by_patch(
    const Interface2D& iface,
    const LaplaceCorrectionSupport2D& support,
    const std::vector<int>& crossing_center_indices)
{
    const int n_patches =
        static_cast<int>(iface.corner_patches().size());
    std::vector<std::vector<int>> centers_by_patch(
        static_cast<std::size_t>(n_patches));
    for (int patch = 0; patch < n_patches; ++patch) {
        centers_by_patch[static_cast<std::size_t>(patch)] =
            patch_regularized_center_indices_for_spread(
                support, crossing_center_indices, patch);
    }
    return centers_by_patch;
}

double evaluate_sd_circle_crossing_correction_for_node(
    const GridPair2D& grid_pair,
    const NarrowBandProjection2D& projection_cache,
    const SdCornerLiftingOptions2D& sd_lifting,
    const std::vector<LocalPoly2D>& full_polys,
    const std::vector<IndexedLocalPolys2D>& artificial_polys_by_patch,
    const LaplaceCrossingCorrectionOp& op,
    const Eigen::Vector2d& pt)
{
    const int rhs_side = side_from_label(grid_pair.domain_label(op.rhs_node));
    const int correction_side =
        side_from_label(grid_pair.domain_label(op.correction_node));
    const int physical_side_delta = rhs_side - correction_side;
    const int rhs_patch = grid_pair.corner_patch_label(op.rhs_node);
    const int correction_patch =
        grid_pair.corner_patch_label(op.correction_node);

    double correction = 0.0;
    if (physical_side_delta != 0) {
        if (rhs_patch >= 0 && rhs_patch == correction_patch) {
            correction +=
                static_cast<double>(physical_side_delta)
                * evaluate_interface_jump_correction_for_node(
                      grid_pair,
                      full_polys,
                      artificial_polys_by_patch,
                      rhs_patch,
                      op.rhs_node,
                      op.correction_node,
                      pt);
            return correction;
        }

        correction +=
            static_cast<double>(physical_side_delta)
            * evaluate_interface_jump_correction_for_node(
                  grid_pair,
                  full_polys,
                  {},
                  -1,
                  op.rhs_node,
                  op.correction_node,
                  pt);
    }

    auto branch_point = [&](int side_label) -> Eigen::Vector2d {
        if (physical_side_delta == 0)
            return pt;
        const CurveProjection2D projection =
            crossing_reference_projection_2d(grid_pair, projection_cache, op);
        const double d = std::abs(projection.signed_distance);
        const double sign = side_label == 0 ? 1.0 : -1.0;
        Eigen::Vector2d x = projection.point;
        x += sign * d * projection.normal;
        return x;
    };

    if (rhs_patch >= 0) {
        correction -=
            sd_lifting.lifting.value_for_patch(
                grid_pair.interface(), rhs_patch, branch_point(rhs_side));
    }
    if (correction_patch >= 0) {
        correction +=
            sd_lifting.lifting.value_for_patch(
                grid_pair.interface(),
                correction_patch,
                pt);
    }
    return correction;
}

double evaluate_crossing_correction_2d(
    const GridPair2D&                  grid_pair,
    const NarrowBandProjection2D&      projection_cache,
    const LaplaceCrossingCorrectionOp& op,
    const LaplaceSpreadResult2D&       spread_result)
{
    const Eigen::Vector2d pt =
        node_coord(grid_pair.grid(), op.correction_node);
    const LaplaceJumpCorrectionSource2D source =
        laplace_crossing_source_2d(op);
    if (source == LaplaceJumpCorrectionSource2D::PatchBoundarySingular) {
        if (spread_result.sd_corner_lifting.enabled) {
            const double rhs_value =
                op.rhs_patch >= 0
                    ? spread_result.sd_corner_lifting.lifting.value_for_patch(
                          grid_pair.interface(), op.rhs_patch, pt)
                    : 0.0;
            const double correction_value =
                op.correction_patch >= 0
                    ? spread_result.sd_corner_lifting.lifting.value_for_patch(
                          grid_pair.interface(), op.correction_patch, pt)
                    : 0.0;
            return correction_value - rhs_value;
        }
        return evaluate_corner_patch_boundary_difference_for_node(
            grid_pair.interface(),
            spread_result.corner_patch_corrections,
            op,
            pt);
    }

    const CurveProjection2D projection =
        crossing_reference_projection_2d(grid_pair, projection_cache, op);
    const int forced_patch =
        source == LaplaceJumpCorrectionSource2D::PatchRegularizedInterface
            ? op.patch
            : -1;
    return evaluate_projection_point_correction_2d(grid_pair.interface(),
                                                   projection,
                                                   spread_result,
                                                   forced_patch);
}

} // namespace

LaplacePanelSpread2D::LaplacePanelSpread2D(const GridPair2D& grid_pair,
                                           double            kappa)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
{}

LaplaceSpreadResult2D LaplacePanelSpread2D::apply(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_grid = grid.num_dofs();
    const int n_iface = iface.num_points();

    if (iface.points_per_panel() != 3)
        throw std::invalid_argument("LaplacePanelSpread2D requires 3 interface points per panel");
    if (iface.panel_node_layout() != PanelNodeLayout2D::LegacyGaussLegendre)
        throw std::invalid_argument("LaplacePanelSpread2D is legacy and requires Gauss-Legendre panel nodes");
    if (static_cast<int>(jumps.size()) != n_iface)
        throw std::invalid_argument("LaplacePanelSpread2D jumps size must equal interface point count");
    if (rhs_correction.size() != n_grid)
        throw std::invalid_argument("LaplacePanelSpread2D rhs_correction size must equal grid DOF count");

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    Eigen::VectorXd rhs_jump(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (jumps[q].rhs_derivs.size() < 1)
            throw std::invalid_argument("LaplacePanelSpread2D requires rhs_derivs[0] at every interface point");
        u_jump[q] = jumps[q].u_jump;
        un_jump[q] = jumps[q].un_jump;
        rhs_jump[q] = jumps[q].rhs_derivs[0];
    }

    const PanelCauchyResult2D cauchy =
        laplace_panel_cauchy_2d(iface, u_jump, un_jump, rhs_jump, kappa_);

    std::vector<LocalPoly2D> polys(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        polys[q].center = iface.points().row(q).transpose();
        polys[q].coeffs.resize(6);
        polys[q].coeffs << cauchy.C[q],
                           cauchy.Cx[q],
                           cauchy.Cy[q],
                           cauchy.Cxx[q],
                           cauchy.Cxy[q],
                           cauchy.Cyy[q];
    }

    for (int n = 0; n < n_grid; ++n) {
        if (is_outer_boundary_node(grid, n))
            continue;

        const int side_n = side_from_label(grid_pair_.domain_label(n));
        const auto neighbors = grid.neighbors(n);

        for (int slot = 0; slot < 4; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || is_outer_boundary_node(grid, nb))
                continue;

            const int side_nb = side_from_label(grid_pair_.domain_label(nb));
            if (side_nb == side_n)
                continue;

            const int q = grid_pair_.closest_interface_point(nb);
            const double correction = evaluate_taylor_poly_2d(polys[q], node_coord(grid, nb));
            rhs_correction[n] += static_cast<double>(side_n - side_nb)
                                 * correction
                                 * stencil_weight_for_neighbor(grid, slot);
        }
    }

    LaplaceSpreadResult2D result;
    result.correction_method = LaplaceCorrectionMethod2D::NearestExpansionCenter;
    result.correction_polys = std::move(polys);
    return result;
}

LaplaceQuadraticPanelCenterSpread2D::LaplaceQuadraticPanelCenterSpread2D(
    const GridPair2D& grid_pair,
    double            kappa,
    LaplaceCorrectionMethod2D correction_method,
    int               projection_restrict_stencil_radius,
    CornerPatchSolverOptions2D corner_patch_solver_options,
    SdCornerLiftingOptions2D sd_corner_lifting_options)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
    , correction_method_(correction_method)
    , projection_restrict_stencil_radius_(projection_restrict_stencil_radius)
    , support_(build_laplace_correction_support_2d(
          grid_pair,
          "LaplaceQuadraticPanelCenterSpread2D"))
    , crossing_center_indices_(
          build_crossing_center_indices_for_spread(grid_pair_, support_))
    , patch_regularized_center_indices_by_patch_(
          build_patch_regularized_center_indices_by_patch(
              grid_pair_.interface(),
              support_,
              crossing_center_indices_))
    , cauchy_geometry_cache_(
          build_panel_center_cauchy_geometry_cache_2d(grid_pair_.interface(),
                                                      kappa_))
    , corner_patch_solver_options_(std::move(corner_patch_solver_options))
    , sd_corner_lifting_options_(std::move(sd_corner_lifting_options))
{
    if (projection_restrict_stencil_radius_ < 1) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterSpread2D projection restrict stencil radius must be positive");
    }
    if (sd_corner_lifting_options_.enabled
        && correction_method_ == LaplaceCorrectionMethod2D::ProjectionPoint) {
        throw std::invalid_argument(
            "S_D corner patch lifting currently supports nearest-center correction only");
    }
    const CornerPatchSingularFitOptions2D& singular_fit =
        corner_patch_solver_options_.singular_fit;
    if (singular_fit.enabled
        && singular_fit.coefficient_fit_method
               == CornerPatchSingularCoefficientFitMethod2D::EdgeP2Trace
        && singular_fit.edge_aligned_half_step_sampling) {
        const Interface2D& iface = grid_pair_.interface();
        for (const CornerPatch2D& patch : iface.corner_patches()) {
            laplace_corner_patch_detail::
                prepare_aligned_half_step_trace_geometry_plan_2d(
                    iface, patch, singular_fit);
        }
    }
    projection_cache_ =
        project_p2_grid_nodes_to_interface_2d(grid_pair_,
                                             support_.projection_nodes);
}

LaplaceSpreadResult2D LaplaceQuadraticPanelCenterSpread2D::apply(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_grid = grid.num_dofs();
    const int n_iface = iface.num_points();

    if (iface.points_per_panel() != 3)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D requires 3 interface points per panel");
    if (iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D requires P2 quadratic panel nodes");
    if (static_cast<int>(jumps.size()) != n_iface)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D jumps size must equal interface point count");
    if (rhs_correction.size() != n_grid)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D rhs_correction size must equal grid DOF count");

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    Eigen::VectorXd rhs_jump(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (jumps[q].rhs_derivs.size() < 1)
            throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D requires rhs_derivs[0] at every interface point");
        u_jump[q] = jumps[q].u_jump;
        un_jump[q] = jumps[q].un_jump;
        rhs_jump[q] = jumps[q].rhs_derivs[0];
    }

    LaplaceSpreadResult2D result;
    result.correction_method = correction_method_;
    result.u_jump = u_jump;
    result.un_jump = un_jump;
    result.rhs_jump = rhs_jump;
    result.alpha = kappa_;
    result.projection_cache = projection_cache_;
    const SdCornerLiftingOptions2D sd_corner_lifting =
        effective_sd_corner_lifting_options(
            sd_corner_lifting_options_, iface, u_jump);
    result.sd_corner_lifting = sd_corner_lifting;
    if (!sd_corner_lifting.enabled) {
        result.corner_patch_corrections =
            laplace_corner_patch_solver_2d(iface,
                                           u_jump,
                                           un_jump,
                                           rhs_jump,
                                           kappa_,
                                           corner_patch_solver_options_);
    }

    if (correction_method_ == LaplaceCorrectionMethod2D::NearestExpansionCenter) {
        const PanelCenterCauchyResult2D cauchy =
            laplace_panel_quadratic_center_cauchy_2d(
                iface,
                u_jump,
                un_jump,
                rhs_jump,
                kappa_,
                {},
                nullptr,
                &cauchy_geometry_cache_);

        std::vector<LocalPoly2D> center_polys(cauchy.centers.rows());
        for (int i = 0; i < cauchy.centers.rows(); ++i)
            center_polys[i] = center_poly(cauchy, i);

        std::vector<IndexedLocalPolys2D> artificial_center_polys_by_patch(
            sd_corner_lifting.enabled
                ? iface.corner_patches().size()
                : result.corner_patch_corrections.size());
        if (sd_corner_lifting.enabled) {
            for (int patch_id = 0;
                 patch_id < static_cast<int>(iface.corner_patches().size());
                 ++patch_id) {
                const PanelCenterCauchyResult2D artificial_cauchy =
                    laplace_panel_quadratic_center_cauchy_2d(
                        iface,
                        u_jump,
                        un_jump,
                        rhs_jump,
                        kappa_,
                        make_sd_patch_regularized_panel_jump_evaluator(
                            iface,
                            sd_corner_lifting,
                            patch_id,
                            u_jump,
                            un_jump,
                            rhs_jump),
                        nullptr,
                        &cauchy_geometry_cache_);
                IndexedLocalPolys2D& patch_polys =
                    artificial_center_polys_by_patch[
                        static_cast<std::size_t>(patch_id)];
                patch_polys.center_indices.resize(
                    static_cast<std::size_t>(artificial_cauchy.centers.rows()));
                patch_polys.polys.resize(
                    static_cast<std::size_t>(artificial_cauchy.centers.rows()));
                for (int i = 0; i < artificial_cauchy.centers.rows(); ++i)
                    patch_polys.center_indices[static_cast<std::size_t>(i)] =
                        i;
                for (int i = 0; i < artificial_cauchy.centers.rows(); ++i)
                    patch_polys.polys[static_cast<std::size_t>(i)] =
                        center_poly(artificial_cauchy, i);
            }
        }
        for (const CornerPatchCorrectionData2D& data
             : result.corner_patch_corrections) {
            if (data.patch < 0)
                continue;
            if (data.patch
                >= static_cast<int>(
                    patch_regularized_center_indices_by_patch_.size())) {
                continue;
            }
            const std::vector<int>& selected_centers =
                patch_regularized_center_indices_by_patch_[
                    static_cast<std::size_t>(data.patch)];
            if (selected_centers.empty())
                continue;
            const PanelCenterCauchyResult2D artificial_cauchy =
                laplace_panel_quadratic_center_cauchy_2d(iface,
                                                         u_jump,
                                                         un_jump,
                                                         rhs_jump,
                                                         kappa_,
                                                         make_patch_regularized_panel_jump_evaluator(
                                                             iface,
                                                             data,
                                                             u_jump,
                                                             un_jump,
                                                             rhs_jump,
                                                             kappa_,
                                                             &selected_centers),
                                                         &selected_centers,
                                                         &cauchy_geometry_cache_);
            IndexedLocalPolys2D& patch_polys =
                artificial_center_polys_by_patch[
                    static_cast<std::size_t>(data.patch)];
            patch_polys.center_indices = selected_centers;
            patch_polys.polys.resize(selected_centers.size());
            for (std::size_t pos = 0; pos < selected_centers.size(); ++pos) {
                const int i = selected_centers[pos];
                patch_polys.polys[pos] = center_poly(artificial_cauchy, i);
            }
        }

        if (!sd_corner_lifting.enabled
            && !patch_boundary_singular_basis_cache_ready_
            && !result.corner_patch_corrections.empty()) {
            patch_boundary_singular_basis_by_crossing_ =
                build_patch_boundary_singular_basis_by_crossing(
                    grid_pair_,
                    support_,
                    result.corner_patch_corrections);
            patch_boundary_singular_basis_cache_ready_ = true;
        }

        for (std::size_t op_idx = 0; op_idx < support_.crossing_ops.size();
             ++op_idx) {
            const LaplaceCrossingCorrectionOp& op =
                support_.crossing_ops[op_idx];
            const Eigen::Vector2d pt = node_coord(grid, op.correction_node);
            double correction = 0.0;
            if (sd_corner_lifting.enabled) {
                correction =
                    evaluate_sd_circle_crossing_correction_for_node(
                        grid_pair_,
                        projection_cache_,
                        sd_corner_lifting,
                        center_polys,
                        artificial_center_polys_by_patch,
                        op,
                        pt);
                rhs_correction[op.rhs_node] += correction * op.stencil_weight;
                continue;
            }

            if (op.kind == LaplaceCrossingKind::InterfaceJump) {
                // Physical-interface crossings use the polynomial tied to the
                // crossing panel/center.  If the crossing touches a corner
                // patch, use the panel polynomial built from singular-removed
                // jump data for that patch.
                correction = evaluate_interface_jump_correction_for_node(
                    grid_pair_,
                    center_polys,
                    artificial_center_polys_by_patch,
                    op.patch,
                    op.rhs_node,
                    op.correction_node,
                    pt,
                    op_idx < crossing_center_indices_.size()
                        ? crossing_center_indices_[op_idx]
                        : -1);
            } else {
                const PatchBoundarySingularDifferenceBasis2D* basis =
                    op_idx < patch_boundary_singular_basis_by_crossing_.size()
                        ? &patch_boundary_singular_basis_by_crossing_[op_idx]
                        : nullptr;
                correction =
                    evaluate_corner_patch_boundary_difference_for_node_cached(
                        result.corner_patch_corrections,
                        op,
                        pt,
                        basis);
            }
            // Across an artificial patch boundary the regular problem
            // compensates the removed singular field, so the singular
            // difference is oriented opposite to the crossing direction.
            const double side_scale =
                op.kind == LaplaceCrossingKind::CornerPatchBoundary
                    ? -1.0
                    :  static_cast<double>(op.side_delta);
            const double contribution =
                side_scale * correction * op.stencil_weight;
            rhs_correction[op.rhs_node] += contribution;
        }
        result.patch_regularized_correction_polys =
            std::move(artificial_center_polys_by_patch);
        result.correction_polys = std::move(center_polys);
        return result;
    }

    if (correction_method_ == LaplaceCorrectionMethod2D::ProjectionPoint) {
        for (const LaplaceCrossingCorrectionOp& op : support_.crossing_ops) {
            const double correction =
                evaluate_crossing_correction_2d(grid_pair_,
                                                projection_cache_,
                                                op,
                                                result);
            // Across an artificial patch boundary the regular problem
            // compensates the removed singular field, so the singular
            // difference is oriented opposite to the crossing direction.
            const double side_scale =
                op.kind == LaplaceCrossingKind::CornerPatchBoundary
                    ? -1.0
                    :  static_cast<double>(op.side_delta);
            rhs_correction[op.rhs_node] += side_scale
                                         * correction
                                         * op.stencil_weight;
        }
        return result;
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPanelCenterSpread2D unsupported correction method");
}

} // namespace kfbim
