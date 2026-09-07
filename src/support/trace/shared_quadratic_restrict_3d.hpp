#pragma once

#include "src/grid/cartesian_grid_3d.hpp"

#include <Eigen/Dense>

#include <array>

namespace kfbim::app3d {

constexpr int kSharedQuadraticRestrictNodeCount3D = 10;
constexpr int kSharedQuadraticRestrictQueryCount3D = 3;

// One complete-quadratic Cartesian stencil shared by three query points.
// The polynomial basis is
//   1, x, y, z, x^2, y^2, z^2, xy, xz, yz,
// in grid-spacing-scaled coordinates about the stencil center.
struct SharedQuadraticRestrictStencil3D {
    std::array<int, kSharedQuadraticRestrictNodeCount3D> grid_ids{};
    Eigen::Matrix<double,
                  kSharedQuadraticRestrictQueryCount3D,
                  kSharedQuadraticRestrictNodeCount3D> weights =
        Eigen::Matrix<double,
                      kSharedQuadraticRestrictQueryCount3D,
                      kSharedQuadraticRestrictNodeCount3D>::Zero();
    double condition = 0.0;
};

[[nodiscard]] SharedQuadraticRestrictStencil3D
build_shared_quadratic_restrict_stencil_3d(
    const CartesianGrid3D& grid,
    const std::array<Eigen::Vector3d,
                     kSharedQuadraticRestrictQueryCount3D>& query_points);

enum class QuadraticRestrictNormalSide3D {
    Interior = -1,
    Exterior = 1,
};

// The legacy route keeps the experimentally tuned near-interface layers and
// the split-quadratic joint recovery.  The topology-affine route follows the
// implementation specification and uses the wider 3+3 layer set together
// with a single cubic least-squares recovery.  Keeping this choice explicit
// prevents the new route from changing the established shared-quadratic
// results.
enum class SharedQuadraticNormalProfile3D {
    LegacySplitQuadratic,
    TopologyAffineCubic,
};

[[nodiscard]] std::array<double,
                         kSharedQuadraticRestrictQueryCount3D>
shared_quadratic_signed_rho_3d(
    QuadraticRestrictNormalSide3D side,
    SharedQuadraticNormalProfile3D profile);

// The signed layer coordinate is rho = signed_distance / h, where positive
// rho points along the outward normal.  normal_weights therefore recover the
// derivative with respect to the outward unit normal on either side.
struct OneSidedQuadraticTraceWeights3D {
    std::array<double, kSharedQuadraticRestrictQueryCount3D> signed_rho{};
    Eigen::RowVector3d value_weights = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d normal_weights = Eigen::RowVector3d::Zero();
};

[[nodiscard]] OneSidedQuadraticTraceWeights3D
build_one_sided_quadratic_trace_weights_3d(
    QuadraticRestrictNormalSide3D side,
    double h);

[[nodiscard]] std::array<Eigen::Vector3d,
                         kSharedQuadraticRestrictQueryCount3D>
quadratic_restrict_normal_query_points_3d(
    const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal,
    double h,
    QuadraticRestrictNormalSide3D side);

[[nodiscard]] std::array<Eigen::Vector3d,
                         kSharedQuadraticRestrictQueryCount3D>
quadratic_restrict_normal_query_points_3d(
    const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal,
    double h,
    QuadraticRestrictNormalSide3D side,
    SharedQuadraticNormalProfile3D profile);

} // namespace kfbim::app3d
