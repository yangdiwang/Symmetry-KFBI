#include "shared_quadratic_restrict_3d.hpp"

#include "src/grid/structured_grid_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

using QuadraticRow3D =
    Eigen::Matrix<double, 1, kSharedQuadraticRestrictNodeCount3D>;
using QuadraticMatrix3D =
    Eigen::Matrix<double,
                  kSharedQuadraticRestrictNodeCount3D,
                  kSharedQuadraticRestrictNodeCount3D>;

QuadraticMatrix3D invert_quadratic_matrix(const QuadraticMatrix3D& matrix)
{
    constexpr int size = kSharedQuadraticRestrictNodeCount3D;
    double augmented[size][2 * size]{};
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
            augmented[row][column] = matrix(row, column);
            augmented[row][size + column] = row == column ? 1.0 : 0.0;
        }
    }
    for (int column = 0; column < size; ++column) {
        int pivot = column;
        for (int row = column + 1; row < size; ++row) {
            if (std::abs(augmented[row][column])
                > std::abs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (!(std::abs(augmented[pivot][column]) > 1.0e-13)) {
            throw std::runtime_error(
                "shared quadratic restrict stencil is rank deficient");
        }
        if (pivot != column) {
            for (int entry = 0; entry < 2 * size; ++entry)
                std::swap(augmented[pivot][entry],
                          augmented[column][entry]);
        }
        const double scale = augmented[column][column];
        for (int entry = 0; entry < 2 * size; ++entry)
            augmented[column][entry] /= scale;
        for (int row = 0; row < size; ++row) {
            if (row == column)
                continue;
            const double multiplier = augmented[row][column];
            for (int entry = 0; entry < 2 * size; ++entry) {
                augmented[row][entry] -=
                    multiplier * augmented[column][entry];
            }
        }
    }
    QuadraticMatrix3D inverse;
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column)
            inverse(row, column) = augmented[row][size + column];
    }
    return inverse;
}

double infinity_condition(const QuadraticMatrix3D& matrix,
                          const QuadraticMatrix3D& inverse)
{
    const auto matrix_norm = [](const QuadraticMatrix3D& value) {
        double result = 0.0;
        for (int row = 0; row < value.rows(); ++row)
            result = std::max(result, value.row(row).cwiseAbs().sum());
        return result;
    };
    return matrix_norm(matrix) * matrix_norm(inverse);
}

QuadraticRow3D complete_quadratic_row(const Eigen::Vector3d& coordinate)
{
    QuadraticRow3D row;
    const double x = coordinate.x();
    const double y = coordinate.y();
    const double z = coordinate.z();
    row << 1.0, x, y, z, x * x, y * y, z * z, x * y, x * z,
        y * z;
    return row;
}

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int node)
{
    const std::array<double, 3> coordinate = grid.coord(node);
    return {coordinate[0], coordinate[1], coordinate[2]};
}

void validate_grid(const CartesianGrid3D& grid)
{
    const std::array<double, 3> spacing = grid.spacing();
    const std::array<int, 3> dims = grid.dof_dims();
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(spacing[static_cast<std::size_t>(axis)])
            || !(spacing[static_cast<std::size_t>(axis)] > 0.0)) {
            throw std::invalid_argument(
                "shared quadratic restrict requires positive grid spacing");
        }
        if (dims[static_cast<std::size_t>(axis)] < 3) {
            throw std::invalid_argument(
                "shared quadratic restrict requires at least three grid "
                "DOFs in every direction");
        }
    }
}

} // namespace

SharedQuadraticRestrictStencil3D
build_shared_quadratic_restrict_stencil_3d(
    const CartesianGrid3D& grid,
    const std::array<Eigen::Vector3d,
                     kSharedQuadraticRestrictQueryCount3D>& query_points)
{
    validate_grid(grid);
    for (const Eigen::Vector3d& query : query_points) {
        if (!query.allFinite()) {
            throw std::invalid_argument(
                "shared quadratic restrict query must be finite");
        }
    }

    const Eigen::Vector3d centroid =
        (query_points[0] + query_points[1] + query_points[2]) / 3.0;
    const std::array<double, 3> spacing_array = grid.spacing();
    const Eigen::Vector3d spacing(
        spacing_array[0], spacing_array[1], spacing_array[2]);
    const std::array<double, 3> first_array = grid.coord(0, 0, 0);
    const Eigen::Vector3d first(first_array[0], first_array[1], first_array[2]);
    const std::array<int, 3> dims = grid.dof_dims();

    std::array<int, 3> center_index{};
    for (int axis = 0; axis < 3; ++axis) {
        const double continuous = (centroid[axis] - first[axis]) / spacing[axis];
        const int nearest = static_cast<int>(std::llround(continuous));
        center_index[static_cast<std::size_t>(axis)] =
            std::clamp(nearest, 1, dims[static_cast<std::size_t>(axis)] - 2);
    }

    SharedQuadraticRestrictStencil3D result;
    QuadraticMatrix3D design;
    const int center_node =
        grid.index(center_index[0], center_index[1], center_index[2]);
    result.grid_ids = structured_grid::quadratic_restrict_stencil_nodes_3d(
        "shared quadratic restrict", grid, center_node, centroid);
    const Eigen::Vector3d stencil_center = grid_point(grid, center_node);
    for (int row = 0; row < kSharedQuadraticRestrictNodeCount3D; ++row) {
        const int node = result.grid_ids[static_cast<std::size_t>(row)];
        const Eigen::Vector3d local =
            (grid_point(grid, node) - stencil_center).cwiseQuotient(spacing);
        design.row(row) = complete_quadratic_row(local);
    }

    const QuadraticMatrix3D inverse = invert_quadratic_matrix(design);
    result.condition = infinity_condition(design, inverse);
    for (int query = 0; query < kSharedQuadraticRestrictQueryCount3D;
         ++query) {
        const Eigen::Vector3d local =
            (query_points[static_cast<std::size_t>(query)] - stencil_center)
                .cwiseQuotient(spacing);
        result.weights.row(query) = complete_quadratic_row(local) * inverse;
    }
    if (!result.weights.allFinite() || !std::isfinite(result.condition)) {
        throw std::runtime_error(
            "shared quadratic restrict produced invalid weights");
    }
    return result;
}

OneSidedQuadraticTraceWeights3D
build_one_sided_quadratic_trace_weights_3d(
    QuadraticRestrictNormalSide3D side,
    double h)
{
    if (!std::isfinite(h) || !(h > 0.0)) {
        throw std::invalid_argument(
            "one-sided quadratic trace recovery requires positive h");
    }
    const double sign =
        side == QuadraticRestrictNormalSide3D::Exterior ? 1.0 : -1.0;
    OneSidedQuadraticTraceWeights3D result;
    result.signed_rho = {{sign * 0.2, sign * 0.6, sign * 1.0}};

    result.value_weights << 1.875, -1.25, 0.375;
    result.normal_weights =
        sign * Eigen::RowVector3d(-5.0, 7.5, -2.5) / h;
    if (!result.value_weights.allFinite()
        || !result.normal_weights.allFinite()) {
        throw std::runtime_error(
            "one-sided quadratic trace recovery produced invalid weights");
    }
    return result;
}

std::array<double, kSharedQuadraticRestrictQueryCount3D>
shared_quadratic_signed_rho_3d(
    QuadraticRestrictNormalSide3D side,
    SharedQuadraticNormalProfile3D profile)
{
    const double sign =
        side == QuadraticRestrictNormalSide3D::Exterior ? 1.0 : -1.0;
    if (profile == SharedQuadraticNormalProfile3D::TopologyAffineCubic)
        return {{sign * 0.5, sign * 0.75, sign * 1.5}};
    return {{sign * 0.2, sign * 0.6, sign * 1.0}};
}

std::array<Eigen::Vector3d, kSharedQuadraticRestrictQueryCount3D>
quadratic_restrict_normal_query_points_3d(
    const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal,
    double h,
    QuadraticRestrictNormalSide3D side)
{
    return quadratic_restrict_normal_query_points_3d(
        trace_point, outward_normal, h, side,
        SharedQuadraticNormalProfile3D::LegacySplitQuadratic);
}

std::array<Eigen::Vector3d, kSharedQuadraticRestrictQueryCount3D>
quadratic_restrict_normal_query_points_3d(
    const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal,
    double h,
    QuadraticRestrictNormalSide3D side,
    SharedQuadraticNormalProfile3D profile)
{
    if (!trace_point.allFinite() || !outward_normal.allFinite()
        || std::abs(outward_normal.norm() - 1.0) > 1.0e-10) {
        throw std::invalid_argument(
            "quadratic restrict normal queries require a finite point and "
            "unit normal");
    }
    const auto signed_rho = shared_quadratic_signed_rho_3d(side, profile);
    std::array<Eigen::Vector3d, kSharedQuadraticRestrictQueryCount3D> result;
    for (int layer = 0; layer < kSharedQuadraticRestrictQueryCount3D;
         ++layer) {
        result[static_cast<std::size_t>(layer)] =
            trace_point
            + h * signed_rho[static_cast<std::size_t>(layer)]
                * outward_normal;
    }
    return result;
}

} // namespace kfbim::app3d
