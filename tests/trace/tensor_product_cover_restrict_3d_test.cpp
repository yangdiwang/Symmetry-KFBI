#include "src/support/trace/tensor_product_cover_restrict_3d.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

using kfbim::CartesianGrid3D;
using kfbim::DofLayout3D;
using kfbim::app3d::TensorProductCoverKind3D;

double integer_power(double x, int degree)
{
    double result = 1.0;
    for (int q = 0; q < degree; ++q)
        result *= x;
    return result;
}

void require_close(double actual, double expected, double tolerance,
                   const char* message)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message);
}

void check_kind_at_point(
    TensorProductCoverKind3D kind,
    int degree,
    const Eigen::Vector3d& point)
{
    const CartesianGrid3D grid(
        {-1.2, -0.9, -1.4}, {0.13, 0.17, 0.11},
        {24, 22, 26}, DofLayout3D::Node);
    const Eigen::Vector3d normal =
        Eigen::Vector3d(1.0, -2.0, 3.0).normalized();
    const auto stencil =
        kfbim::app3d::build_tensor_product_cover_restrict_stencil_3d(
            grid, point, normal, kind);
    const int width = degree + 1;
    if (stencil.width != width
        || static_cast<int>(stencil.grid_ids.size())
               != width * width * width) {
        throw std::runtime_error("tensor cover has the wrong size");
    }
    require_close(stencil.value_weights.sum(), 1.0, 2.0e-13,
                  "tensor cover does not preserve constants");
    require_close(stencil.normal_weights.sum(), 0.0, 2.0e-12,
                  "tensor-cover derivative does not annihilate constants");

    for (int ax = 0; ax <= degree; ++ax) {
        for (int ay = 0; ay <= degree; ++ay) {
            for (int az = 0; az <= degree; ++az) {
                Eigen::VectorXd values(stencil.value_weights.size());
                for (int q = 0; q < values.size(); ++q) {
                    const auto coordinate = grid.coord(
                        stencil.grid_ids[static_cast<std::size_t>(q)]);
                    values[q] = integer_power(coordinate[0], ax)
                        * integer_power(coordinate[1], ay)
                        * integer_power(coordinate[2], az);
                }
                const double exact_value = integer_power(point.x(), ax)
                    * integer_power(point.y(), ay)
                    * integer_power(point.z(), az);
                double exact_normal = 0.0;
                if (ax > 0) {
                    exact_normal += normal.x() * static_cast<double>(ax)
                        * integer_power(point.x(), ax - 1)
                        * integer_power(point.y(), ay)
                        * integer_power(point.z(), az);
                }
                if (ay > 0) {
                    exact_normal += normal.y() * static_cast<double>(ay)
                        * integer_power(point.x(), ax)
                        * integer_power(point.y(), ay - 1)
                        * integer_power(point.z(), az);
                }
                if (az > 0) {
                    exact_normal += normal.z() * static_cast<double>(az)
                        * integer_power(point.x(), ax)
                        * integer_power(point.y(), ay)
                        * integer_power(point.z(), az - 1);
                }
                require_close(stencil.value_weights.dot(values), exact_value,
                              2.0e-11,
                              "tensor cover failed value reproduction");
                require_close(stencil.normal_weights.dot(values), exact_normal,
                              3.0e-10,
                              "tensor cover failed normal reproduction");
            }
        }
    }
}

void check_kind(TensorProductCoverKind3D kind, int degree)
{
    // Exercise both halves of a grid cell and both clamped domain ends.  In
    // particular, Q27 must move its central node when the fractional phase
    // crosses one half, while Q64 continues to straddle the containing cell.
    for (const Eigen::Vector3d& point : {
             Eigen::Vector3d(0.137, -0.083, 0.219),
             Eigen::Vector3d(0.211, 0.046, -0.172),
             Eigen::Vector3d(-1.191, -0.887, -1.393),
             Eigen::Vector3d(1.911, 2.821, 1.459)}) {
        check_kind_at_point(kind, degree, point);
    }
}

void check_outside_box_rejected()
{
    const CartesianGrid3D grid(
        {-1.0, -1.0, -1.0}, {0.1, 0.1, 0.1},
        {21, 21, 21}, DofLayout3D::Node);
    bool rejected = false;
    try {
        (void)kfbim::app3d::build_tensor_product_cover_restrict_stencil_3d(
            grid, Eigen::Vector3d(1.11, 0.0, 0.0),
            Eigen::Vector3d::UnitX(), TensorProductCoverKind3D::Q64Cover4);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("tensor cover accepted a point outside the grid");
}

} // namespace

int main()
{
    try {
        check_kind(TensorProductCoverKind3D::Q27Cover3, 2);
        check_kind(TensorProductCoverKind3D::Q64Cover4, 3);
        check_outside_box_rejected();
        std::cout << "tensor-product cover restrict 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor-product cover restrict 3D test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
