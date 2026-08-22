#include "laplace_spread_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include "../geometry/corner_patch_2d.hpp"
#include "../geometry/p2_curve_2d.hpp"
#include "../geometry/p2_projection_2d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../local_cauchy/laplace_corner_patch_solver_2d.hpp"
#include "../local_cauchy/laplace_panel_solver_2d.hpp"
#include "laplace_crossing_local_polynomial_2d.hpp"
#include "laplace_arc_length_bspline_crossing_jet_2d.hpp"
#include "laplace_nurbs_density_trace_state_2d.hpp"
#include "laplace_projection_correction_2d.hpp"

namespace kfbim {

struct LaplaceP2CubicHarmonicCache2D {
    struct CenterFit {
        Eigen::Vector2d center = Eigen::Vector2d::Zero();
        Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
        Eigen::Vector2d normal = Eigen::Vector2d::Zero();
        std::vector<int> value_neighbor_ids;
        std::vector<int> normal_neighbor_ids;
        Eigen::MatrixXd value_map;
        Eigen::MatrixXd normal_map;
        double condition = 0.0;
    };

    double h = 0.0;
    double max_condition = 0.0;
    std::vector<CenterFit> centers;
};

namespace {

constexpr int kCubicHarmonicDimension2D = 7;

Eigen::Matrix<double, kCubicHarmonicDimension2D, 1>
cubic_harmonic_basis_2d(double s, double r)
{
    Eigen::Matrix<double, kCubicHarmonicDimension2D, 1> basis;
    basis << 1.0,
             s,
             r,
             s * s - r * r,
             2.0 * s * r,
             s * s * s - 3.0 * s * r * r,
             3.0 * s * s * r - r * r * r;
    return basis;
}

Eigen::Matrix<double, 2, kCubicHarmonicDimension2D>
cubic_harmonic_gradient_2d(double s, double r)
{
    Eigen::Matrix<double, 2, kCubicHarmonicDimension2D> gradient;
    gradient <<
        0.0, 1.0, 0.0, 2.0 * s, 2.0 * r,
        3.0 * (s * s - r * r), 6.0 * s * r,
        0.0, 0.0, 1.0, -2.0 * r, 2.0 * s,
        -6.0 * s * r, 3.0 * (s * s - r * r);
    return gradient;
}

Eigen::MatrixXd spread_pseudo_inverse_2d(const Eigen::MatrixXd& matrix,
                                         double relative_cutoff,
                                         double* condition)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() == 0 || !(singular[0] > 0.0))
        throw std::runtime_error(
            "cubic harmonic spread received a rank-zero fit matrix");

    const double cutoff = relative_cutoff * singular[0];
    Eigen::VectorXd inverse = singular;
    for (int i = 0; i < inverse.size(); ++i)
        inverse[i] = singular[i] > cutoff ? 1.0 / singular[i] : 0.0;

    if (condition != nullptr) {
        *condition = singular[singular.size() - 1] > 0.0
            ? singular[0] / singular[singular.size() - 1]
            : std::numeric_limits<double>::infinity();
    }
    return svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose();
}

std::shared_ptr<const LaplaceP2CubicHarmonicCache2D>
build_p2_cubic_harmonic_cache_2d(
    const GridPair2D& grid_pair,
    const LaplaceP2CubicHarmonicSpreadOptions2D& options)
{
    if (options.value_neighbors < 1
        || options.normal_neighbors < 1
        || options.normal_neighbors > options.value_neighbors
        || options.value_neighbors + options.normal_neighbors
               < kCubicHarmonicDimension2D) {
        throw std::invalid_argument(
            "P2 cubic harmonic spread requires normal_neighbors <= "
            "value_neighbors and at least seven total Cauchy samples");
    }
    if (!(options.relative_svd_cutoff > 0.0)
        || !(options.distance_weight_shift > 0.0)
        || !(options.normal_row_weight > 0.0)) {
        throw std::invalid_argument(
            "P2 cubic harmonic spread weights and SVD cutoff must be positive");
    }

    const CartesianGrid2D& grid = grid_pair.grid();
    const Interface2D& iface = grid_pair.interface();
    const std::array<double, 2> spacing = grid.spacing();
    if (!(spacing[0] > 0.0) || !(spacing[1] > 0.0)) {
        throw std::invalid_argument(
            "P2 cubic harmonic spread requires positive grid spacing");
    }
    const double spacing_scale = std::max(spacing[0], spacing[1]);
    if (std::abs(spacing[0] - spacing[1])
        > 64.0 * std::numeric_limits<double>::epsilon() * spacing_scale) {
        throw std::invalid_argument(
            "P2 cubic harmonic spread currently requires an isotropic grid");
    }
    const int center_count =
        static_cast<int>(geometry2d::kP2CenterS.size())
        * iface.num_panels();
    if (center_count < options.value_neighbors) {
        throw std::invalid_argument(
            "P2 cubic harmonic spread has too few expansion centers");
    }

    auto cache = std::make_shared<LaplaceP2CubicHarmonicCache2D>();
    cache->h = spacing[0];
    cache->centers.resize(static_cast<std::size_t>(center_count));

    // Keep the exact 4*panel center ordering consumed by GridPair2D's
    // crossing-owner map and by the joint restrict operator.
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (int local = 0;
             local < static_cast<int>(geometry2d::kP2CenterS.size());
             ++local) {
            const int center_index =
                static_cast<int>(geometry2d::kP2CenterS.size()) * panel
                + local;
            LaplaceP2CubicHarmonicCache2D::CenterFit& fit =
                cache->centers[static_cast<std::size_t>(center_index)];
            const double panel_s =
                geometry2d::kP2CenterS[static_cast<std::size_t>(local)];
            fit.center = geometry2d::panel_point(iface, panel, panel_s);
            fit.normal = geometry2d::panel_normal(iface, panel, panel_s);
            fit.normal.normalize();
            fit.tangent = Eigen::Vector2d(-fit.normal[1], fit.normal[0]);
        }
    }

    // Mirror the standalone smooth implementation on the P2 expansion-center
    // cloud: each center uses its four nearest center values and the first
    // three corresponding normal derivatives. The center jump samples
    // themselves are obtained by P2 interpolation in apply().
    for (int center_index = 0; center_index < center_count; ++center_index) {
            LaplaceP2CubicHarmonicCache2D::CenterFit& fit =
                cache->centers[static_cast<std::size_t>(center_index)];
            std::vector<int> order(
                static_cast<std::size_t>(center_count));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(
                order.begin(), order.end(), [&](int lhs, int rhs) {
                    const double dl =
                        (cache->centers[static_cast<std::size_t>(lhs)].center
                         - fit.center).squaredNorm();
                    const double dr =
                        (cache->centers[static_cast<std::size_t>(rhs)].center
                         - fit.center).squaredNorm();
                    if (dl != dr)
                        return dl < dr;
                    return lhs < rhs;
                });

            fit.value_neighbor_ids.assign(
                order.begin(),
                order.begin() + options.value_neighbors);
            fit.normal_neighbor_ids.assign(
                order.begin(),
                order.begin() + options.normal_neighbors);

            const int rows =
                options.value_neighbors + options.normal_neighbors;
            Eigen::MatrixXd design(
                rows, kCubicHarmonicDimension2D);
            Eigen::VectorXd sqrt_weights(rows);
            for (int pos = 0; pos < options.value_neighbors; ++pos) {
                const int neighbor =
                    fit.value_neighbor_ids[static_cast<std::size_t>(pos)];
                const Eigen::Vector2d d =
                    (cache->centers[
                         static_cast<std::size_t>(neighbor)].center
                     - fit.center) / cache->h;
                design.row(pos) = cubic_harmonic_basis_2d(
                    d.dot(fit.tangent), d.dot(fit.normal)).transpose();
                sqrt_weights[pos] = std::sqrt(
                    1.0
                    / std::pow(options.distance_weight_shift + d.norm(), 2));
            }
            for (int pos = 0; pos < options.normal_neighbors; ++pos) {
                const int neighbor =
                    fit.normal_neighbor_ids[static_cast<std::size_t>(pos)];
                const Eigen::Vector2d d =
                    (cache->centers[
                         static_cast<std::size_t>(neighbor)].center
                     - fit.center) / cache->h;
                const Eigen::Vector2d& sample_normal =
                    cache->centers[
                        static_cast<std::size_t>(neighbor)].normal;
                const Eigen::Vector2d components(
                    sample_normal.dot(fit.tangent),
                    sample_normal.dot(fit.normal));
                design.row(options.value_neighbors + pos) =
                    components.transpose()
                    * cubic_harmonic_gradient_2d(
                        d.dot(fit.tangent), d.dot(fit.normal));
                sqrt_weights[options.value_neighbors + pos] = std::sqrt(
                    options.normal_row_weight
                    / std::pow(options.distance_weight_shift + d.norm(), 2));
            }

            Eigen::MatrixXd weighted_design = design;
            for (int row = 0; row < rows; ++row)
                weighted_design.row(row) *= sqrt_weights[row];
            const Eigen::MatrixXd pinv = spread_pseudo_inverse_2d(
                weighted_design,
                options.relative_svd_cutoff,
                &fit.condition);
            cache->max_condition =
                std::max(cache->max_condition, fit.condition);

            fit.value_map = Eigen::MatrixXd::Zero(
                kCubicHarmonicDimension2D, options.value_neighbors);
            fit.normal_map = Eigen::MatrixXd::Zero(
                kCubicHarmonicDimension2D, options.normal_neighbors);
            for (int pos = 0; pos < options.value_neighbors; ++pos) {
                fit.value_map.col(pos) =
                    pinv.col(pos) * sqrt_weights[pos];
            }
            for (int pos = 0; pos < options.normal_neighbors; ++pos) {
                fit.normal_map.col(pos) =
                    pinv.col(options.value_neighbors + pos)
                    * sqrt_weights[options.value_neighbors + pos]
                    * cache->h;
            }
    }

    return cache;
}

LocalPoly2D cubic_harmonic_poly_to_cartesian_taylor_2d(
    const LaplaceP2CubicHarmonicCache2D::CenterFit& fit,
    double h,
    const Eigen::Matrix<double, kCubicHarmonicDimension2D, 1>& a)
{
    const double sx = fit.tangent[0] / h;
    const double sy = fit.tangent[1] / h;
    const double rx = fit.normal[0] / h;
    const double ry = fit.normal[1] / h;

    const double x = a[1] * sx + a[2] * rx;
    const double y = a[1] * sy + a[2] * ry;

    const double x2 =
        a[3] * (sx * sx - rx * rx)
        + a[4] * (2.0 * sx * rx);
    const double xy =
        a[3] * (2.0 * sx * sy - 2.0 * rx * ry)
        + a[4] * (2.0 * (sx * ry + sy * rx));
    const double y2 =
        a[3] * (sy * sy - ry * ry)
        + a[4] * (2.0 * sy * ry);

    const double s3_x3 = sx * sx * sx;
    const double s3_x2y = 3.0 * sx * sx * sy;
    const double s3_xy2 = 3.0 * sx * sy * sy;
    const double s3_y3 = sy * sy * sy;
    const double sr2_x3 = sx * rx * rx;
    const double sr2_x2y = sy * rx * rx + 2.0 * sx * rx * ry;
    const double sr2_xy2 = sx * ry * ry + 2.0 * sy * rx * ry;
    const double sr2_y3 = sy * ry * ry;
    const double s2r_x3 = sx * sx * rx;
    const double s2r_x2y = sx * sx * ry + 2.0 * sx * sy * rx;
    const double s2r_xy2 = sy * sy * rx + 2.0 * sx * sy * ry;
    const double s2r_y3 = sy * sy * ry;
    const double r3_x3 = rx * rx * rx;
    const double r3_x2y = 3.0 * rx * rx * ry;
    const double r3_xy2 = 3.0 * rx * ry * ry;
    const double r3_y3 = ry * ry * ry;

    const double x3 =
        a[5] * (s3_x3 - 3.0 * sr2_x3)
        + a[6] * (3.0 * s2r_x3 - r3_x3);
    const double x2y =
        a[5] * (s3_x2y - 3.0 * sr2_x2y)
        + a[6] * (3.0 * s2r_x2y - r3_x2y);
    const double xy2 =
        a[5] * (s3_xy2 - 3.0 * sr2_xy2)
        + a[6] * (3.0 * s2r_xy2 - r3_xy2);
    const double y3 =
        a[5] * (s3_y3 - 3.0 * sr2_y3)
        + a[6] * (3.0 * s2r_y3 - r3_y3);

    LocalPoly2D poly;
    poly.center = fit.center;
    poly.coeffs.resize(10);
    poly.coeffs << a[0],
                   x,
                   y,
                   2.0 * x2,
                   xy,
                   2.0 * y2,
                   6.0 * x3,
                   2.0 * x2y,
                   2.0 * xy2,
                   6.0 * y3;
    return poly;
}

std::vector<LocalPoly2D> fit_p2_cubic_harmonic_polys_2d(
    const Interface2D& iface,
    const LaplaceP2CubicHarmonicCache2D& cache,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump)
{
    Eigen::VectorXd center_value_jump(
        static_cast<int>(cache.centers.size()));
    Eigen::VectorXd center_normal_jump(
        static_cast<int>(cache.centers.size()));
    for (int center = 0;
         center < static_cast<int>(cache.centers.size());
         ++center) {
        const int panel =
            center / static_cast<int>(geometry2d::kP2CenterS.size());
        const int local =
            center % static_cast<int>(geometry2d::kP2CenterS.size());
        const double panel_s =
            geometry2d::kP2CenterS[static_cast<std::size_t>(local)];
        center_value_jump[center] =
            geometry2d::panel_scalar(iface, panel, value_jump, panel_s);
        center_normal_jump[center] =
            geometry2d::panel_scalar(iface, panel, normal_jump, panel_s);
    }

    std::vector<LocalPoly2D> polys(cache.centers.size());
    for (std::size_t center = 0; center < cache.centers.size(); ++center) {
        const LaplaceP2CubicHarmonicCache2D::CenterFit& fit =
            cache.centers[center];
        Eigen::VectorXd local_value(
            static_cast<int>(fit.value_neighbor_ids.size()));
        Eigen::VectorXd local_normal(
            static_cast<int>(fit.normal_neighbor_ids.size()));
        for (int i = 0; i < local_value.size(); ++i) {
            local_value[i] = center_value_jump[
                fit.value_neighbor_ids[static_cast<std::size_t>(i)]];
        }
        for (int i = 0; i < local_normal.size(); ++i) {
            local_normal[i] = center_normal_jump[
                fit.normal_neighbor_ids[static_cast<std::size_t>(i)]];
        }
        const Eigen::Matrix<double, kCubicHarmonicDimension2D, 1>
            coefficients =
                fit.value_map * local_value
                + fit.normal_map * local_normal;
        polys[center] = cubic_harmonic_poly_to_cartesian_taylor_2d(
            fit, cache.h, coefficients);
    }
    return polys;
}

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

std::vector<P2CrossingOwner2D> build_crossing_owners_for_spread(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support,
    LaplaceCorrectionMethod2D correction_method)
{
    std::vector<P2CrossingOwner2D> owners(support.crossing_ops.size());
    if (correction_method != LaplaceCorrectionMethod2D::CrossingOwner)
        return owners;

    for (std::size_t op_idx = 0; op_idx < support.crossing_ops.size();
         ++op_idx) {
        const LaplaceCrossingCorrectionOp& op =
            support.crossing_ops[op_idx];
        if (op.kind == LaplaceCrossingKind::InterfaceJump) {
            owners[op_idx] = grid_pair.p2_crossing_owner_between(
                op.rhs_node, op.correction_node);
        }
    }
    return owners;
}

std::vector<int> build_crossing_center_indices_for_spread(
    const GridPair2D& grid_pair,
    const LaplaceCorrectionSupport2D& support,
    LaplaceCorrectionMethod2D correction_method,
    const std::vector<P2CrossingOwner2D>& crossing_owners)
{
    std::vector<int> center_indices(support.crossing_ops.size(), -1);
    for (std::size_t op_idx = 0; op_idx < support.crossing_ops.size();
         ++op_idx) {
        const LaplaceCrossingCorrectionOp& op =
            support.crossing_ops[op_idx];
        if (op.kind != LaplaceCrossingKind::InterfaceJump)
            continue;
        if (correction_method == LaplaceCorrectionMethod2D::CrossingOwner
            && op_idx < crossing_owners.size()
            && crossing_owners[op_idx].center_index >= 0) {
            center_indices[op_idx] = crossing_owners[op_idx].center_index;
        } else {
            center_indices[op_idx] =
                grid_pair.nearest_p2_expansion_center_between(
                    op.rhs_node, op.correction_node);
        }
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
    SdCornerLiftingOptions2D sd_corner_lifting_options,
    LaplaceP2PanelCenterSpreadMode2D spread_mode,
    LaplaceP2CubicHarmonicSpreadOptions2D cubic_harmonic_options,
    LaplaceCrossingTraceStencil2D crossing_trace_stencil,
    LaplaceCrossingJetScheme2D crossing_jet_scheme)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
    , correction_method_(correction_method)
    , projection_restrict_stencil_radius_(projection_restrict_stencil_radius)
    , support_(build_laplace_correction_support_2d(
          grid_pair,
          "LaplaceQuadraticPanelCenterSpread2D"))
    , crossing_owners_(build_crossing_owners_for_spread(
          grid_pair_, support_, correction_method_))
    , crossing_center_indices_(
          build_crossing_center_indices_for_spread(
              grid_pair_,
              support_,
              correction_method_,
              crossing_owners_))
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
    , spread_mode_(spread_mode)
    , cubic_harmonic_options_(std::move(cubic_harmonic_options))
    , crossing_trace_stencil_(crossing_trace_stencil)
    , crossing_jet_scheme_(crossing_jet_scheme)
{
    if (!is_valid_crossing_trace_stencil(crossing_trace_stencil_)) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterSpread2D has an invalid crossing trace stencil");
    }
    if (crossing_jet_scheme_
            != LaplaceCrossingJetScheme2D::LocalArclengthLagrange
        && crossing_jet_scheme_
            != LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet
        && crossing_jet_scheme_
            != LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterSpread2D has an invalid crossing jet scheme");
    }
    if (projection_restrict_stencil_radius_ < 1) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterSpread2D projection restrict stencil radius must be positive");
    }
    if (sd_corner_lifting_options_.enabled
        && correction_method_ == LaplaceCorrectionMethod2D::ProjectionPoint) {
        throw std::invalid_argument(
            "S_D corner patch lifting currently supports nearest-center correction only");
    }
    if (spread_mode_
        == LaplaceP2PanelCenterSpreadMode2D::CubicHarmonic) {
        if (kappa_ != 0.0) {
            throw std::invalid_argument(
                "P2 cubic harmonic spread currently requires kappa=0");
        }
        if (correction_method_ == LaplaceCorrectionMethod2D::ProjectionPoint) {
            throw std::invalid_argument(
                "P2 cubic harmonic spread does not support projection-point correction");
        }
        if (!grid_pair_.interface().corner_patches().empty()
            || sd_corner_lifting_options_.enabled) {
            throw std::invalid_argument(
                "P2 cubic harmonic spread currently requires a smooth interface without corner corrections");
        }
        cubic_harmonic_cache_ = build_p2_cubic_harmonic_cache_2d(
            grid_pair_, cubic_harmonic_options_);
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
    if (crossing_jet_scheme_
            == LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet
        || crossing_jet_scheme_
            == LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet) {
        arc_length_bspline_crossing_jet_plan_ =
            std::make_shared<LaplaceArcLengthBSplineCrossingJetPlan2D>(
                grid_pair_.interface(), crossing_jet_scheme_);
        if (crossing_jet_scheme_
                == LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet
            && !arc_length_bspline_crossing_jet_plan_
                    ->uses_nurbs_same_parameter()) {
            throw std::invalid_argument(
                "NSP-CJ requires a complete NurbsBoundaryPanelGeometry2D provider");
        }
    }
}

double LaplaceQuadraticPanelCenterSpread2D::
cubic_harmonic_max_condition() const
{
    return cubic_harmonic_cache_
        ? cubic_harmonic_cache_->max_condition : 0.0;
}

LaplaceSpreadResult2D LaplaceQuadraticPanelCenterSpread2D::apply(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    return apply_with_crossing_trace_stencil(
        jumps, rhs_correction, crossing_trace_stencil_);
}

LaplaceSpreadResult2D
LaplaceQuadraticPanelCenterSpread2D::apply_with_crossing_trace_stencil(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd&                      rhs_correction,
    LaplaceCrossingTraceStencil2D         trace_stencil) const
{
    return apply_impl(jumps, rhs_correction, trace_stencil, nullptr);
}

LaplaceSpreadResult2D
LaplaceQuadraticPanelCenterSpread2D::
apply_with_direct_nurbs_density_state(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd& rhs_correction,
    std::shared_ptr<const LaplaceNurbsDensityTraceState2D> direct_state) const
{
    if (!direct_state) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterSpread2D requires a direct NURBS density state");
    }
    if (correction_method_ != LaplaceCorrectionMethod2D::CrossingOwner) {
        throw std::invalid_argument(
            "direct NURBS density spread requires crossing-owner correction");
    }
    // Function-argument evaluation order is not guaranteed here.  Capture
    // the stencil before moving the shared pointer; otherwise a compiler may
    // evaluate std::move(direct_state) first and dereference an empty pointer.
    const LaplaceCrossingTraceStencil2D trace_stencil =
        direct_state->trace_stencil();
    return apply_impl(jumps,
                      rhs_correction,
                      trace_stencil,
                      std::move(direct_state));
}

LaplaceSpreadResult2D LaplaceQuadraticPanelCenterSpread2D::apply_impl(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd& rhs_correction,
    LaplaceCrossingTraceStencil2D trace_stencil,
    std::shared_ptr<const LaplaceNurbsDensityTraceState2D> direct_state) const
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
    if (direct_state) {
        const auto* geometry = iface.has_panel_geometry()
            ? dynamic_cast<const geometry2d::NurbsBoundaryPanelGeometry2D*>(
                  &iface.panel_geometry())
            : nullptr;
        if (geometry == nullptr || geometry != &direct_state->geometry()
            || geometry->num_parameterized_points() != n_iface) {
            throw std::invalid_argument(
                "direct NURBS density state does not match the spread interface geometry");
        }
    }

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
    if (!is_valid_crossing_trace_stencil(trace_stencil)) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterSpread2D received an invalid crossing trace stencil");
    }
    result.crossing_trace_stencil = trace_stencil;
    result.crossing_jet_scheme = crossing_jet_scheme_;
    result.u_jump = u_jump;
    result.un_jump = un_jump;
    result.rhs_jump = rhs_jump;
    result.alpha = kappa_;
    result.projection_cache = projection_cache_;
    result.direct_nurbs_density_trace_state = direct_state;
    if (!direct_state
        && correction_method_ == LaplaceCorrectionMethod2D::CrossingOwner
        && arc_length_bspline_crossing_jet_plan_) {
        result.arc_length_bspline_crossing_jet_plan =
            arc_length_bspline_crossing_jet_plan_;
        result.arc_length_bspline_trace_state =
            std::make_shared<LaplaceArcLengthBSplineTraceState2D>(
                arc_length_bspline_crossing_jet_plan_->fit(
                    u_jump, un_jump, trace_stencil));
    }
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

    if (correction_method_ == LaplaceCorrectionMethod2D::NearestExpansionCenter
        || correction_method_ == LaplaceCorrectionMethod2D::CrossingOwner) {
        std::vector<LocalPoly2D> center_polys;
        std::vector<IndexedLocalPolys2D> artificial_center_polys_by_patch(
            sd_corner_lifting.enabled
                ? iface.corner_patches().size()
                : result.corner_patch_corrections.size());

        if (spread_mode_
            == LaplaceP2PanelCenterSpreadMode2D::CubicHarmonic) {
            if (!rhs_jump.isZero(0.0)) {
                throw std::invalid_argument(
                    "P2 cubic harmonic spread requires zero RHS jump");
            }
            if (!cubic_harmonic_cache_) {
                throw std::runtime_error(
                    "P2 cubic harmonic spread cache is unavailable");
            }
            center_polys = fit_p2_cubic_harmonic_polys_2d(
                iface, *cubic_harmonic_cache_, u_jump, un_jump);
        } else {
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
            center_polys.resize(cauchy.centers.rows());
            for (int i = 0; i < cauchy.centers.rows(); ++i)
                center_polys[static_cast<std::size_t>(i)] =
                    center_poly(cauchy, i);

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
                        static_cast<std::size_t>(
                            artificial_cauchy.centers.rows()));
                    patch_polys.polys.resize(
                        static_cast<std::size_t>(
                            artificial_cauchy.centers.rows()));
                    for (int i = 0;
                         i < artificial_cauchy.centers.rows();
                         ++i) {
                        patch_polys.center_indices[
                            static_cast<std::size_t>(i)] = i;
                        patch_polys.polys[static_cast<std::size_t>(i)] =
                            center_poly(artificial_cauchy, i);
                    }
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
                for (std::size_t pos = 0;
                     pos < selected_centers.size();
                     ++pos) {
                    const int i = selected_centers[pos];
                    patch_polys.polys[pos] =
                        center_poly(artificial_cauchy, i);
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
                const P2CrossingOwner2D* crossing =
                    op_idx < crossing_owners_.size()
                        ? &crossing_owners_[op_idx]
                        : nullptr;
                const bool crossing_owner_is_active =
                    correction_method_
                            == LaplaceCorrectionMethod2D::CrossingOwner
                    && op.patch < 0
                    && crossing != nullptr;
                const bool use_direct_density =
                    crossing_owner_is_active
                    && result.direct_nurbs_density_trace_state
                    && result.direct_nurbs_density_trace_state
                           ->can_evaluate(*crossing);
                const bool use_als_cj =
                    !use_direct_density
                    && crossing_owner_is_active
                    && result.arc_length_bspline_crossing_jet_plan
                    && result.arc_length_bspline_trace_state
                    && result.arc_length_bspline_crossing_jet_plan
                           ->can_evaluate(*crossing, trace_stencil);
                if (use_direct_density) {
                    correction =
                        result.direct_nurbs_density_trace_state
                            ->build_local_polynomial(
                                *crossing, rhs_jump, kappa_)
                            .evaluate(pt);
                } else if (use_als_cj) {
                    correction =
                        result.arc_length_bspline_crossing_jet_plan
                            ->build_local_polynomial(
                                *crossing,
                                *result.arc_length_bspline_trace_state,
                                rhs_jump,
                                kappa_)
                            .evaluate(pt);
                } else if (crossing_owner_is_active
                           && can_build_laplace_p2_crossing_local_polynomial_2d(
                               iface, *crossing, trace_stencil)) {
                    // Compatibility route for open, cornered, positional, or
                    // otherwise non-periodic components that have no ALS-CJ
                    // component plan.
                    correction =
                        build_laplace_p2_crossing_local_polynomial_2d(
                            iface,
                            *crossing,
                            u_jump,
                            un_jump,
                            rhs_jump,
                            kappa_,
                            trace_stencil)
                            .evaluate(pt);
                } else {
                    // Corner-patch, gap, multiple-hit, and endpoint fallback
                    // cases retain the established expansion-center route.
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
                }
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
