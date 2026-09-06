#pragma once

#include "src/grid/cartesian_grid_3d.hpp"

#include <Eigen/Dense>

#include <vector>

namespace kfbim::app3d {

enum class TensorProductCoverKind3D {
    Q27Cover3,
    Q64Cover4,
};

[[nodiscard]] int tensor_product_cover_width_3d(
    TensorProductCoverKind3D kind) noexcept;

[[nodiscard]] int tensor_product_cover_node_count_3d(
    TensorProductCoverKind3D kind) noexcept;

// Tensor-product Lagrange cover at one interface point.  Q27Cover3 uses
// three consecutive Cartesian nodes per axis (Q2, 27 nodes); Q64Cover4 uses
// four (Q3, 64 nodes).  value_weights evaluate the polynomial at the trace
// point and normal_weights evaluate its outward physical normal derivative.
struct TensorProductCoverRestrictStencil3D {
    TensorProductCoverKind3D kind =
        TensorProductCoverKind3D::Q27Cover3;
    int width = 0;
    std::vector<int> grid_ids;
    Eigen::VectorXd value_weights;
    Eigen::VectorXd normal_weights;
    double value_weight_l1 = 0.0;
    double scaled_normal_weight_l1 = 0.0;
};

[[nodiscard]] TensorProductCoverRestrictStencil3D
build_tensor_product_cover_restrict_stencil_3d(
    const CartesianGrid3D& grid,
    const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal,
    TensorProductCoverKind3D kind);

} // namespace kfbim::app3d
