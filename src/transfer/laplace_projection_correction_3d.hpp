#pragma once

#include <stdexcept>

#include "i_spread.hpp"
#include "../geometry/p2_surface_3d.hpp"

namespace kfbim {

inline void require_projection_surface_data(const Interface3D&           iface,
                                            const LaplaceSpreadResult3D& spread_result)
{
    const int n_iface = iface.num_points();
    if (spread_result.u_jump.size() != n_iface
        || spread_result.un_jump.size() != n_iface
        || spread_result.rhs_jump.size() != n_iface) {
        throw std::invalid_argument(
            "projection-point correction requires surface jump data");
    }
}

inline double evaluate_projection_point_correction_3d(
    const Interface3D&           iface,
    const SurfaceProjection3D&   projection,
    const LaplaceSpreadResult3D& spread_result)
{
    require_projection_surface_data(iface, spread_result);
    if (projection.panel < 0 || projection.panel >= iface.num_panels())
        throw std::invalid_argument("projection-point correction has invalid panel");

    const Eigen::Vector3d bary = projection.barycentric;
    const double c = geometry3d::panel_scalar(iface,
                                              projection.panel,
                                              spread_result.u_jump,
                                              bary);
    const double cn = geometry3d::panel_scalar(iface,
                                               projection.panel,
                                               spread_result.un_jump,
                                               bary);
    const double rhs_jump = geometry3d::panel_scalar(iface,
                                                     projection.panel,
                                                     spread_result.rhs_jump,
                                                     bary);
    const double mean_curvature =
        geometry3d::panel_mean_curvature(iface, projection.panel, bary);
    const double surface_laplace =
        geometry3d::panel_laplace_beltrami_scalar(iface,
                                                  projection.panel,
                                                  spread_result.u_jump,
                                                  bary);
    const double cnn = spread_result.alpha * c
                     - rhs_jump
                     - mean_curvature * cn
                     - surface_laplace;

    const double s = projection.signed_distance;
    return c + s * cn + 0.5 * s * s * cnn;
}

inline double evaluate_projection_point_correction_3d(
    const GridPair3D&             grid_pair,
    const NarrowBandProjection3D& band,
    int                           grid_node,
    const LaplaceSpreadResult3D&  spread_result)
{
    if (!band.has_projection(grid_node)) {
        throw std::runtime_error(
            "projection-point correction missing narrow-band projection");
    }
    return evaluate_projection_point_correction_3d(grid_pair.interface(),
                                                   band.projection(grid_node),
                                                   spread_result);
}

} // namespace kfbim
