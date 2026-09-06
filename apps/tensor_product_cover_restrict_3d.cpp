#include "tensor_product_cover_restrict_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

struct AxisLagrangeData3D {
    int start = 0;
    Eigen::VectorXd value;
    Eigen::VectorXd derivative;
};

AxisLagrangeData3D axis_lagrange_data(
    double coordinate,
    double origin,
    double spacing,
    int dimension,
    int width)
{
    if (!std::isfinite(coordinate) || !std::isfinite(origin)
        || !std::isfinite(spacing) || !(spacing > 0.0)
        || dimension < width || width < 2) {
        throw std::invalid_argument(
            "tensor-product cover received invalid axis data");
    }
    double grid_coordinate = (coordinate - origin) / spacing;
    const double upper = static_cast<double>(dimension - 1);
    const double coordinate_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(grid_coordinate), upper});
    if (!std::isfinite(grid_coordinate)
        || grid_coordinate < -coordinate_tolerance
        || grid_coordinate > upper + coordinate_tolerance) {
        throw std::out_of_range(
            "tensor-product cover trace point lies outside the Cartesian node box");
    }
    grid_coordinate = std::clamp(grid_coordinate, 0.0, upper);
    const int anchor = width == 3
        ? static_cast<int>(std::floor(grid_coordinate + 0.5))
        : static_cast<int>(std::floor(grid_coordinate));

    AxisLagrangeData3D result;
    // An odd three-node cover is centred on the nearest Cartesian node;
    // an even four-node cover straddles the containing grid cell.  This is
    // the symmetric Q2/Q3 convention already used by the 2-D tensor route.
    result.start = std::clamp(anchor - 1, 0, dimension - width);
    result.value = Eigen::VectorXd::Ones(width);
    result.derivative = Eigen::VectorXd::Zero(width);
    const double local = grid_coordinate - static_cast<double>(result.start);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if (i == j)
                continue;
            result.value[i] *=
                (local - static_cast<double>(j))
                / static_cast<double>(i - j);
        }
        for (int differentiated = 0;
             differentiated < width; ++differentiated) {
            if (differentiated == i)
                continue;
            double term = 1.0 / static_cast<double>(i - differentiated);
            for (int j = 0; j < width; ++j) {
                if (j == i || j == differentiated)
                    continue;
                term *=
                    (local - static_cast<double>(j))
                    / static_cast<double>(i - j);
            }
            result.derivative[i] += term / spacing;
        }
    }
    if (!result.value.allFinite() || !result.derivative.allFinite()) {
        throw std::runtime_error(
            "tensor-product cover generated non-finite axis weights");
    }
    return result;
}

} // namespace

int tensor_product_cover_width_3d(TensorProductCoverKind3D kind) noexcept
{
    switch (kind) {
    case TensorProductCoverKind3D::Q27Cover3:
        return 3;
    case TensorProductCoverKind3D::Q64Cover4:
        return 4;
    }
    return 0;
}

int tensor_product_cover_node_count_3d(
    TensorProductCoverKind3D kind) noexcept
{
    const int width = tensor_product_cover_width_3d(kind);
    return width * width * width;
}

TensorProductCoverRestrictStencil3D
build_tensor_product_cover_restrict_stencil_3d(
    const CartesianGrid3D& grid,
    const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal,
    TensorProductCoverKind3D kind)
{
    if (grid.layout() != DofLayout3D::Node) {
        throw std::invalid_argument(
            "tensor-product cover restrict requires a node-centred Cartesian grid");
    }
    if (!trace_point.allFinite() || !outward_normal.allFinite()
        || std::abs(outward_normal.norm() - 1.0) > 1.0e-10) {
        throw std::invalid_argument(
            "tensor-product cover requires a finite trace point and unit normal");
    }
    const int width = tensor_product_cover_width_3d(kind);
    if (width != 3 && width != 4)
        throw std::invalid_argument("unknown tensor-product cover kind");

    const auto origin = grid.origin();
    const auto spacing = grid.spacing();
    const auto dimensions = grid.dof_dims();
    std::array<AxisLagrangeData3D, 3> axis;
    for (int d = 0; d < 3; ++d) {
        axis[static_cast<std::size_t>(d)] = axis_lagrange_data(
            trace_point[d], origin[static_cast<std::size_t>(d)],
            spacing[static_cast<std::size_t>(d)],
            dimensions[static_cast<std::size_t>(d)], width);
    }

    TensorProductCoverRestrictStencil3D result;
    result.kind = kind;
    result.width = width;
    const int count = width * width * width;
    result.grid_ids.reserve(static_cast<std::size_t>(count));
    result.value_weights.resize(count);
    result.normal_weights.resize(count);

    int slot = 0;
    for (int k = 0; k < width; ++k) {
        for (int j = 0; j < width; ++j) {
            for (int i = 0; i < width; ++i, ++slot) {
                result.grid_ids.push_back(grid.index(
                    axis[0].start + i,
                    axis[1].start + j,
                    axis[2].start + k));
                const double lx = axis[0].value[i];
                const double ly = axis[1].value[j];
                const double lz = axis[2].value[k];
                result.value_weights[slot] = lx * ly * lz;
                result.normal_weights[slot] =
                    outward_normal.x() * axis[0].derivative[i] * ly * lz
                    + outward_normal.y() * lx * axis[1].derivative[j] * lz
                    + outward_normal.z() * lx * ly * axis[2].derivative[k];
            }
        }
    }
    const double h_min = std::min({spacing[0], spacing[1], spacing[2]});
    result.value_weight_l1 = result.value_weights.lpNorm<1>();
    result.scaled_normal_weight_l1 =
        h_min * result.normal_weights.lpNorm<1>();
    if (static_cast<int>(result.grid_ids.size()) != count
        || !result.value_weights.allFinite()
        || !result.normal_weights.allFinite()
        || !std::isfinite(result.value_weight_l1)
        || !std::isfinite(result.scaled_normal_weight_l1)) {
        throw std::runtime_error(
            "tensor-product cover produced malformed interpolation data");
    }
    return result;
}

} // namespace kfbim::app3d
