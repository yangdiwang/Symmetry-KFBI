#pragma once

#include <cstddef>
#include <stdexcept>

#include "i_spread.hpp"
#include "laplace_correction_support.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../geometry/p2_curve_2d.hpp"

namespace kfbim {

struct PanelJumpValues2D {
    double u[3] = {0.0, 0.0, 0.0};
    double un[3] = {0.0, 0.0, 0.0};
    double rhs[3] = {0.0, 0.0, 0.0};
};

namespace laplace_projection_correction_detail {

inline bool panel_is_in_corner_patch_support(
    const CornerPatchCorrectionData2D& data,
    int panel)
{
    if (data.patch < 0)
        return false;
    if (data.interface_regular_panels.empty())
        return true;
    for (int candidate : data.interface_regular_panels) {
        if (candidate == panel)
            return true;
    }
    return false;
}

inline bool panel_uses_fitted_corner_regular_part(
    const CornerPatchCorrectionData2D& data,
    int panel)
{
    if (!data.has_interface_regular_part
        || data.interface_regular_part.coeffs.size() == 0
        || data.interface_regular_panels.empty()) {
        return false;
    }
    for (int candidate : data.interface_regular_panels) {
        if (candidate == panel)
            return true;
    }
    return false;
}

inline const CornerPatchCorrectionData2D& corner_patch_correction_by_patch(
    const std::vector<CornerPatchCorrectionData2D>& corrections,
    int patch)
{
    if (patch < 0)
        throw std::invalid_argument("negative corner patch id");
    for (const CornerPatchCorrectionData2D& data : corrections) {
        if (data.patch == patch)
            return data;
    }
    if (patch < static_cast<int>(corrections.size())
        && corrections[static_cast<std::size_t>(patch)].patch == patch) {
        return corrections[static_cast<std::size_t>(patch)];
    }
    throw std::invalid_argument(
        "missing corner patch correction data for forced patch");
}

inline double evaluate_taylor_poly_laplacian_2d(const LocalPoly2D& poly,
                                                Eigen::Vector2d    pt)
{
    const int max_degree = taylor_degree_from_num_coeffs_2d(
        static_cast<int>(poly.coeffs.size()));
    const double dx = pt[0] - poly.center[0];
    const double dy = pt[1] - poly.center[1];

    double value = 0.0;
    int idx = 0;
    for (int degree = 0; degree <= max_degree; ++degree) {
        for (int ay = 0; ay <= degree; ++ay) {
            const int ax = degree - ay;
            const double coeff = poly.coeffs[idx]
                / (local_poly_factorial(ax) * local_poly_factorial(ay));
            if (ax >= 2) {
                value += coeff * static_cast<double>(ax * (ax - 1))
                       * local_poly_pow(dx, ax - 2)
                       * local_poly_pow(dy, ay);
            }
            if (ay >= 2) {
                value += coeff * local_poly_pow(dx, ax)
                       * static_cast<double>(ay * (ay - 1))
                       * local_poly_pow(dy, ay - 2);
            }
            ++idx;
        }
    }
    return value;
}

inline Eigen::Matrix2d evaluate_taylor_poly_hessian_2d(const LocalPoly2D& poly,
                                                       Eigen::Vector2d    pt)
{
    const int max_degree = taylor_degree_from_num_coeffs_2d(
        static_cast<int>(poly.coeffs.size()));
    const double dx = pt[0] - poly.center[0];
    const double dy = pt[1] - poly.center[1];

    Eigen::Matrix2d hessian = Eigen::Matrix2d::Zero();
    int idx = 0;
    for (int degree = 0; degree <= max_degree; ++degree) {
        for (int ay = 0; ay <= degree; ++ay) {
            const int ax = degree - ay;
            const double coeff = poly.coeffs[idx]
                / (local_poly_factorial(ax) * local_poly_factorial(ay));
            if (ax >= 2) {
                hessian(0, 0) += coeff * static_cast<double>(ax * (ax - 1))
                               * local_poly_pow(dx, ax - 2)
                               * local_poly_pow(dy, ay);
            }
            if (ax >= 1 && ay >= 1) {
                const double mixed =
                    coeff * static_cast<double>(ax * ay)
                    * local_poly_pow(dx, ax - 1)
                    * local_poly_pow(dy, ay - 1);
                hessian(0, 1) += mixed;
                hessian(1, 0) += mixed;
            }
            if (ay >= 2) {
                hessian(1, 1) += coeff * static_cast<double>(ay * (ay - 1))
                               * local_poly_pow(dx, ax)
                               * local_poly_pow(dy, ay - 2);
            }
            ++idx;
        }
    }
    return hessian;
}

inline bool panel_contains_point(const Interface2D& iface,
                                 int panel,
                                 int point)
{
    for (int local = 0; local < iface.points_per_panel(); ++local) {
        if (iface.point_index(panel, local) == point)
            return true;
    }
    return false;
}

inline int patch_id_of_interface_point_by_location_2d(const Interface2D& iface,
                                                      int q)
{
    if (q < 0 || q >= iface.num_points())
        return -1;
    const Eigen::Vector2d point = iface.points().row(q).transpose();
    for (int patch = 0;
         patch < static_cast<int>(iface.corner_patches().size());
         ++patch) {
        const CornerPatch2D& corner_patch =
            iface.corner_patches()[static_cast<std::size_t>(patch)];
        if (corner_patch.point == q
            || (point - corner_patch.center).norm() <= 1.0e-12
            || corner_patch_contains_point_2d(corner_patch, point)) {
            return patch;
        }
    }
    return -1;
}

inline bool panel_uses_corner_regular_direct_values(
    const Interface2D& iface,
    const CornerPatchCorrectionData2D& data,
    int panel)
{
    return data.point >= 0
        && iface.is_corner_point(data.point)
        && panel_contains_point(iface, panel, data.point)
        && panel_uses_fitted_corner_regular_part(data, panel);
}

inline double panel_scalar_from_local_values(const double values[3], double s)
{
    double N[3];
    geometry2d::p2_shape(s, N);
    return N[0] * values[0] + N[1] * values[1] + N[2] * values[2];
}

inline double panel_scalar_deriv_from_local_values(const double values[3],
                                                   double s)
{
    double dN[3];
    geometry2d::p2_shape_deriv(s, dN);
    return dN[0] * values[0] + dN[1] * values[1] + dN[2] * values[2];
}

inline double panel_scalar_second_deriv_from_local_values(
    const double values[3])
{
    double ddN[3];
    geometry2d::p2_shape_second_deriv(0.0, ddN);
    return ddN[0] * values[0] + ddN[1] * values[1]
         + ddN[2] * values[2];
}

inline double panel_laplace_beltrami_from_local_values(
    const Interface2D& iface,
    int panel,
    const double values[3],
    double s)
{
    const Eigen::Vector2d tangent = geometry2d::panel_tangent(iface, panel, s);
    const Eigen::Vector2d second =
        geometry2d::panel_second_derivative(iface, panel, s);
    const double speed2 = tangent.squaredNorm();
    if (speed2 <= 1.0e-28)
        throw std::invalid_argument("degenerate P2 panel tangent");

    const double c_s = panel_scalar_deriv_from_local_values(values, s);
    const double c_ss = panel_scalar_second_deriv_from_local_values(values);
    return c_ss / speed2
         - c_s * tangent.dot(second) / (speed2 * speed2);
}

inline double panel_laplace_beltrami_from_regular_2d(
    const Interface2D& iface,
    int panel,
    const LocalPoly2D& regular,
    double s)
{
    const Eigen::Vector2d pt = geometry2d::panel_point(iface, panel, s);
    const Eigen::Vector2d tangent = geometry2d::panel_tangent(iface, panel, s);
    const Eigen::Vector2d second =
        geometry2d::panel_second_derivative(iface, panel, s);
    const double speed2 = tangent.squaredNorm();
    if (speed2 <= 1.0e-28)
        throw std::invalid_argument("degenerate P2 panel tangent");

    const Eigen::Vector2d grad =
        evaluate_taylor_poly_gradient_2d(regular, pt);
    const Eigen::Matrix2d hessian =
        evaluate_taylor_poly_hessian_2d(regular, pt);
    const double c_s = grad.dot(tangent);
    const double c_ss = tangent.dot(hessian * tangent) + grad.dot(second);
    return c_ss / speed2
         - c_s * tangent.dot(second) / (speed2 * speed2);
}

inline PanelJumpValues2D build_projection_panel_jump_values_2d(
    const Interface2D& iface,
    const LaplaceSpreadResult2D& spread_result,
    int panel,
    const CornerPatchCorrectionData2D* data)
{
    PanelJumpValues2D values;
    const bool subtract_singular =
        data != nullptr && panel_is_in_corner_patch_support(*data, panel);
    for (int local = 0; local < 3; ++local) {
        const int q = iface.point_index(panel, local);
        const Eigen::Vector2d pt = iface.points().row(q).transpose();
        const Eigen::Vector2d normal = iface.panel_normal(panel, local);
        values.u[local] = spread_result.u_jump[q];
        values.un[local] = spread_result.un_jump[q];
        values.rhs[local] = spread_result.rhs_jump[q];
        if (subtract_singular) {
            values.u[local] -=
                evaluate_corner_patch_singular_jump_trace_2d(*data, pt);
            values.un[local] +=
                -evaluate_corner_patch_singular_normal_jump_trace_2d(
                    *data, pt, normal);
        }
    }
    return values;
}

inline PanelJumpValues2D build_sd_residual_panel_jump_values_2d(
    const Interface2D& iface,
    const LaplaceSpreadResult2D& spread_result,
    int panel,
    int patch_id)
{
    PanelJumpValues2D values;
    for (int local = 0; local < 3; ++local) {
        const int q = iface.point_index(panel, local);
        const Eigen::Vector2d pt = iface.points().row(q).transpose();
        const Eigen::Vector2d normal = iface.panel_normal(panel, local);
        values.u[local] =
            spread_result.u_jump[q]
            - spread_result.sd_corner_lifting.lifting.jump_value_for_patch(
                  iface,
                  patch_id,
                  pt,
                  normal,
                  spread_result.sd_corner_lifting.normal_probe);
        values.un[local] = spread_result.un_jump[q];
        values.rhs[local] = spread_result.rhs_jump[q];
    }
    return values;
}

inline double evaluate_projection_point_correction_from_panel_values_2d(
    const Interface2D& iface,
    const CurveProjection2D& projection,
    const LaplaceSpreadResult2D& spread_result,
    const PanelJumpValues2D& values)
{
    const double s = projection.local_s;
    const double c = panel_scalar_from_local_values(values.u, s);
    const double cn = panel_scalar_from_local_values(values.un, s);
    const double rhs_jump = panel_scalar_from_local_values(values.rhs, s);
    const double normal_divergence =
        geometry2d::panel_normal_divergence(iface, projection.panel, s);
    const double surface_laplace =
        panel_laplace_beltrami_from_local_values(
            iface, projection.panel, values.u, s);
    const double cnn = spread_result.alpha * c
                     - rhs_jump
                     - normal_divergence * cn
                     - surface_laplace;

    const double d = projection.signed_distance;
    return c + d * cn + 0.5 * d * d * cnn;
}

inline double evaluate_projection_point_correction_from_regular_2d(
    const Interface2D& iface,
    const CurveProjection2D& projection,
    const LaplaceSpreadResult2D& spread_result,
    const LocalPoly2D& regular)
{
    const double s = projection.local_s;
    const Eigen::Vector2d pt = projection.point;
    const Eigen::Vector2d normal =
        geometry2d::panel_normal(iface, projection.panel, s);
    const double c = evaluate_taylor_poly_2d(regular, pt);
    const double cn =
        evaluate_taylor_poly_gradient_2d(regular, pt).dot(normal);
    const double rhs_jump =
        spread_result.alpha * c
        - evaluate_taylor_poly_laplacian_2d(regular, pt);
    const double normal_divergence =
        geometry2d::panel_normal_divergence(iface, projection.panel, s);
    const double surface_laplace =
        panel_laplace_beltrami_from_regular_2d(
            iface, projection.panel, regular, s);
    const double cnn = spread_result.alpha * c
                     - rhs_jump
                     - normal_divergence * cn
                     - surface_laplace;

    const double d = projection.signed_distance;
    return c + d * cn + 0.5 * d * d * cnn;
}

inline double evaluate_regularized_projection_point_correction_2d(
    const Interface2D& iface,
    const CurveProjection2D& projection,
    const LaplaceSpreadResult2D& spread_result,
    const CornerPatchCorrectionData2D& data)
{
    if (panel_uses_fitted_corner_regular_part(data, projection.panel)) {
        return evaluate_projection_point_correction_from_regular_2d(
            iface, projection, spread_result, data.interface_regular_part);
    }
    const PanelJumpValues2D values =
        build_projection_panel_jump_values_2d(
            iface, spread_result, projection.panel, &data);
    return evaluate_projection_point_correction_from_panel_values_2d(
        iface, projection, spread_result, values);
}

} // namespace laplace_projection_correction_detail

inline int patch_id_of_boundary_iter_point_2d(const Interface2D& iface, int q)
{
    return laplace_projection_correction_detail::
        patch_id_of_interface_point_by_location_2d(iface, q);
}

inline PanelJumpValues2D build_restrict_panel_jump_values(
    const Interface2D& iface,
    const LaplaceSpreadResult2D& spread_result,
    int panel,
    int boundary_iter_point,
    PatchInterfaceJumpMode2D mode,
    const CornerPatchCorrectionData2D* patch_data,
    int forced_patch = -1)
{
    const CornerPatchCorrectionData2D* data = nullptr;
    const int patch_id = forced_patch >= 0
        ? forced_patch
        : patch_id_of_boundary_iter_point_2d(iface, boundary_iter_point);
    if (patch_id >= 0 && mode == PatchInterfaceJumpMode2D::SingularRemovedJump
        && spread_result.sd_corner_lifting.enabled
        && spread_result.sd_corner_lifting.regularized_jump_mode
               == SdCornerRegularizedJumpMode2D::InterpolateResidualAtP2Nodes) {
        return laplace_projection_correction_detail::
            build_sd_residual_panel_jump_values_2d(
                iface, spread_result, panel, patch_id);
    }
    if (patch_id >= 0 && mode == PatchInterfaceJumpMode2D::SingularRemovedJump) {
        if (patch_data != nullptr && patch_data->patch == patch_id) {
            data = patch_data;
        } else {
            data = &laplace_projection_correction_detail::
                corner_patch_correction_by_patch(
                    spread_result.corner_patch_corrections, patch_id);
        }
    }
    return laplace_projection_correction_detail::
        build_projection_panel_jump_values_2d(
            iface, spread_result, panel, data);
}

inline CurveProjection2D projection_on_reference_panel_2d(
    const GridPair2D&           grid_pair,
    const CurveProjection2D&    reference,
    int                         grid_node)
{
    const Eigen::Vector2d target =
        structured_grid::point(grid_pair.grid(), grid_node);
    CurveProjection2D projection = reference;
    projection.grid_node = grid_node;
    projection.signed_distance =
        (target - reference.point).dot(reference.normal);
    projection.distance = (target - reference.point).norm();
    return projection;
}

inline CurveProjection2D crossing_reference_projection_2d(
    const GridPair2D&                  grid_pair,
    const NarrowBandProjection2D&      band,
    const LaplaceCrossingCorrectionOp& op)
{
    if (!band.has_projection(op.rhs_node)
        || !band.has_projection(op.correction_node)) {
        throw std::runtime_error(
            "crossing-driven correction missing endpoint projection");
    }

    const CurveProjection2D& rhs_projection = band.projection(op.rhs_node);
    const CurveProjection2D& correction_projection =
        band.projection(op.correction_node);
    const Eigen::Vector2d rhs_point =
        structured_grid::point(grid_pair.grid(), op.rhs_node);
    const Eigen::Vector2d correction_point =
        structured_grid::point(grid_pair.grid(), op.correction_node);

    const double rhs_score =
        (rhs_projection.point - rhs_point).norm()
        + (rhs_projection.point - correction_point).norm();
    const double correction_score =
        (correction_projection.point - rhs_point).norm()
        + (correction_projection.point - correction_point).norm();

    const CurveProjection2D& reference =
        rhs_score <= correction_score ? rhs_projection : correction_projection;
    return projection_on_reference_panel_2d(
        grid_pair, reference, op.correction_node);
}

inline void require_projection_curve_data(const Interface2D&           iface,
                                          const LaplaceSpreadResult2D& spread_result)
{
    const int n_iface = iface.num_points();
    if (spread_result.u_jump.size() != n_iface
        || spread_result.un_jump.size() != n_iface
        || spread_result.rhs_jump.size() != n_iface) {
        throw std::invalid_argument(
            "projection-point correction requires curve jump data");
    }
}

inline double evaluate_projection_point_correction_2d(
    const Interface2D&         iface,
    const CurveProjection2D&   projection,
    const LaplaceSpreadResult2D& spread_result)
{
    require_projection_curve_data(iface, spread_result);
    if (projection.panel < 0 || projection.panel >= iface.num_panels())
        throw std::invalid_argument("projection-point correction has invalid panel");

    const double s = projection.local_s;
    const double c = geometry2d::panel_scalar(iface,
                                              projection.panel,
                                              spread_result.u_jump,
                                              s);
    const double cn = geometry2d::panel_scalar(iface,
                                               projection.panel,
                                               spread_result.un_jump,
                                               s);
    const double rhs_jump = geometry2d::panel_scalar(iface,
                                                     projection.panel,
                                                     spread_result.rhs_jump,
                                                     s);
    const double normal_divergence =
        geometry2d::panel_normal_divergence(iface, projection.panel, s);
    const double surface_laplace =
        geometry2d::panel_laplace_beltrami_scalar(iface,
                                                  projection.panel,
                                                  spread_result.u_jump,
                                                  s);
    const double cnn = spread_result.alpha * c
                     - rhs_jump
                     - normal_divergence * cn
                     - surface_laplace;

    const double d = projection.signed_distance;
    return c + d * cn + 0.5 * d * d * cnn;
}

inline double evaluate_projection_point_correction_2d(
    const Interface2D&           iface,
    const CurveProjection2D&     projection,
    const LaplaceSpreadResult2D& spread_result,
    int                          forced_patch)
{
    if (forced_patch < 0) {
        return evaluate_projection_point_correction_2d(
            iface, projection, spread_result);
    }
    require_projection_curve_data(iface, spread_result);
    if (projection.panel < 0 || projection.panel >= iface.num_panels()) {
        throw std::invalid_argument(
            "projection-point correction has invalid panel");
    }
    const CornerPatchCorrectionData2D& data =
        laplace_projection_correction_detail::
            corner_patch_correction_by_patch(
                spread_result.corner_patch_corrections, forced_patch);
    return laplace_projection_correction_detail::
        evaluate_regularized_projection_point_correction_2d(
            iface, projection, spread_result, data);
}

inline double evaluate_projection_point_correction_2d(
    const GridPair2D&             grid_pair,
    const NarrowBandProjection2D& band,
    int                           grid_node,
    const LaplaceSpreadResult2D&  spread_result,
    int                           forced_patch)
{
    if (!band.has_projection(grid_node)) {
        throw std::runtime_error(
            "projection-point correction missing narrow-band projection");
    }
    const CurveProjection2D& projection = band.projection(grid_node);
    return evaluate_projection_point_correction_2d(grid_pair.interface(),
                                                   projection,
                                                   spread_result,
                                                   forced_patch);
}

inline double evaluate_projection_point_correction_2d(
    const GridPair2D&             grid_pair,
    const NarrowBandProjection2D& band,
    int                           grid_node,
    const LaplaceSpreadResult2D&  spread_result)
{
    return evaluate_projection_point_correction_2d(
        grid_pair, band, grid_node, spread_result, -1);
}

inline double restore_physical_solution_2d(
    double                             regular_value,
    int                                patch,
    const Eigen::Vector2d&             x,
    const CornerPatchCorrectionData2D& data)
{
    if (patch < 0 || data.patch != patch)
        return regular_value;

    bool add_singular = false;
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        if (mode.model
                == CornerPatchSingularModel2D::
                       InteriorDirichletTransmissionPair
            || mode.model
                   == CornerPatchSingularModel2D::
                          InteriorNeumannTransmissionPair) {
            add_singular = true;
            break;
        }
    }
    if (add_singular) {
        const double r = (x - data.center).norm();
        if (r > data.radius + 1.0e-12 || r <= 1.0e-14)
            return regular_value;
        return regular_value + evaluate_corner_patch_singular_2d(data, x);
    }

    CornerPatch2D patch_geometry;
    patch_geometry.corner = -1;
    patch_geometry.point = -1;
    patch_geometry.opening_angle = data.opening_angle;
    patch_geometry.radius = data.radius;
    patch_geometry.center = data.center;
    patch_geometry.ray_minus = data.ray_minus;
    patch_geometry.ray_plus = data.ray_plus;
    if (!corner_patch_sector_contains_point_2d(patch_geometry, x)
        || (x - data.center).norm() > data.radius + 1.0e-12) {
        return regular_value;
    }
    const double singular = evaluate_corner_patch_singular_2d(data, x);
    return regular_value - singular;
}

inline Eigen::VectorXd restore_physical_solution_2d(
    const GridPair2D&                                  grid_pair,
    const Eigen::VectorXd&                             regular_values,
    const std::vector<CornerPatchCorrectionData2D>&    corner_patch_data)
{
    const auto& grid = grid_pair.grid();
    if (regular_values.size() != grid.num_dofs()) {
        throw std::invalid_argument(
            "restore_physical_solution_2d regular_values size must equal grid DOF count");
    }

    Eigen::VectorXd physical = regular_values;
    if (corner_patch_data.empty())
        return physical;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const int patch = grid_pair.corner_patch_label(n);
        if (patch < 0)
            continue;
        const CornerPatchCorrectionData2D& data =
            laplace_projection_correction_detail::
                corner_patch_correction_by_patch(corner_patch_data, patch);
        physical[n] = restore_physical_solution_2d(
            regular_values[n], patch, structured_grid::point(grid, n), data);
    }
    return physical;
}

} // namespace kfbim
