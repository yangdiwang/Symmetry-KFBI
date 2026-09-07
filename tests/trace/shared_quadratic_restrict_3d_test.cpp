#include "src/support/trace/shared_quadratic_restrict_3d.hpp"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using kfbim::CartesianGrid3D;
using kfbim::DofLayout3D;
using kfbim::app3d::OneSidedQuadraticTraceWeights3D;
using kfbim::app3d::QuadraticRestrictNormalSide3D;
using kfbim::app3d::SharedQuadraticRestrictStencil3D;
using kfbim::app3d::SharedQuadraticNormalProfile3D;
using kfbim::app3d::build_one_sided_quadratic_trace_weights_3d;
using kfbim::app3d::build_shared_quadratic_restrict_stencil_3d;
using kfbim::app3d::quadratic_restrict_normal_query_points_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double complete_quadratic(const Eigen::Vector3d& x)
{
    return 1.3 - 0.7 * x.x() + 0.4 * x.y() + 1.1 * x.z()
         + 0.8 * x.x() * x.x() - 0.2 * x.y() * x.y()
         + 0.35 * x.z() * x.z() + 0.6 * x.x() * x.y()
         - 0.45 * x.x() * x.z() + 0.25 * x.y() * x.z();
}

void test_shared_stencil_reproduces_complete_quadratics()
{
    const CartesianGrid3D grid(
        {{-1.1, -0.9, -1.3}},
        {{0.2, 0.25, 0.3}},
        {{12, 10, 9}},
        DofLayout3D::Node);
    const std::array<Eigen::Vector3d, 3> queries{{
        {0.13, -0.17, 0.22},
        {0.18, -0.11, 0.28},
        {0.23, -0.05, 0.34},
    }};
    const SharedQuadraticRestrictStencil3D stencil =
        build_shared_quadratic_restrict_stencil_3d(grid, queries);
    require(std::set<int>(stencil.grid_ids.begin(), stencil.grid_ids.end())
                    .size()
                == 10,
            "shared quadratic stencil has ten distinct nodes");
    require(std::isfinite(stencil.condition) && stencil.condition < 100.0,
            "shared quadratic stencil reports a bounded condition number");

    Eigen::VectorXd nodal_values(10);
    for (int node = 0; node < 10; ++node) {
        const auto coordinate =
            grid.coord(stencil.grid_ids[static_cast<std::size_t>(node)]);
        nodal_values[node] = complete_quadratic(
            {coordinate[0], coordinate[1], coordinate[2]});
    }
    const Eigen::Vector3d interpolated = stencil.weights * nodal_values;
    for (int query = 0; query < 3; ++query) {
        require(
            std::abs(interpolated[query]
                     - complete_quadratic(
                         queries[static_cast<std::size_t>(query)]))
                < 2.0e-12,
            "shared stencil exactly reproduces a complete quadratic");
    }
}

void test_boundary_stencil_is_valid()
{
    const CartesianGrid3D grid(
        {{0.0, 0.0, 0.0}},
        {{0.1, 0.1, 0.1}},
        {{4, 4, 4}},
        DofLayout3D::CellCenter);
    const std::array<Eigen::Vector3d, 3> queries{{
        {-0.02, 0.02, 0.03},
        {0.01, 0.04, 0.05},
        {0.04, 0.06, 0.07},
    }};
    const SharedQuadraticRestrictStencil3D stencil =
        build_shared_quadratic_restrict_stencil_3d(grid, queries);
    require(std::set<int>(stencil.grid_ids.begin(), stencil.grid_ids.end())
                    .size()
                == 10,
            "boundary-clamped stencil retains ten distinct nodes");
    require((stencil.weights.rowwise().sum().array() - 1.0).abs().maxCoeff()
                < 2.0e-13,
            "boundary-clamped stencil reproduces constants");
}

void test_one_sided_quadratic_trace_recovery()
{
    constexpr double h = 0.125;
    for (const QuadraticRestrictNormalSide3D side : {
             QuadraticRestrictNormalSide3D::Interior,
             QuadraticRestrictNormalSide3D::Exterior}) {
        const OneSidedQuadraticTraceWeights3D weights =
            build_one_sided_quadratic_trace_weights_3d(side, h);
        Eigen::Vector3d samples;
        for (int layer = 0; layer < 3; ++layer) {
            const double rho =
                weights.signed_rho[static_cast<std::size_t>(layer)];
            samples[layer] = 2.1 - 0.7 * rho + 0.45 * rho * rho;
        }
        require(std::abs(weights.value_weights.dot(samples) - 2.1)
                    < 2.0e-14,
                "one-sided weights recover the interface value");
        require(std::abs(weights.normal_weights.dot(samples) + 0.7 / h)
                    < 2.0e-13,
                "one-sided weights recover the outward-normal derivative");
    }
}

void test_signed_normal_query_points()
{
    constexpr double h = 0.2;
    const Eigen::Vector3d point(0.3, -0.2, 0.1);
    const Eigen::Vector3d normal(0.0, 0.0, 1.0);
    const auto interior = quadratic_restrict_normal_query_points_3d(
        point, normal, h, QuadraticRestrictNormalSide3D::Interior);
    const auto exterior = quadratic_restrict_normal_query_points_3d(
        point, normal, h, QuadraticRestrictNormalSide3D::Exterior);
    const std::array<double, 3> layers{{0.2, 0.6, 1.0}};
    for (int layer = 0; layer < 3; ++layer) {
        require(
            (interior[static_cast<std::size_t>(layer)]
                 - (point - h * layers[static_cast<std::size_t>(layer)]
                                * normal))
                    .norm()
                < 1.0e-15,
            "interior query has the requested signed layer");
        require(
            (exterior[static_cast<std::size_t>(layer)]
                 - (point + h * layers[static_cast<std::size_t>(layer)]
                                * normal))
                    .norm()
                < 1.0e-15,
            "exterior query has the requested signed layer");
    }
}

void test_topology_affine_cubic_query_points()
{
    constexpr double h = 0.2;
    const Eigen::Vector3d point(0.3, -0.2, 0.1);
    const Eigen::Vector3d normal(0.0, 1.0, 0.0);
    const auto interior = quadratic_restrict_normal_query_points_3d(
        point, normal, h, QuadraticRestrictNormalSide3D::Interior,
        SharedQuadraticNormalProfile3D::TopologyAffineCubic);
    const auto exterior = quadratic_restrict_normal_query_points_3d(
        point, normal, h, QuadraticRestrictNormalSide3D::Exterior,
        SharedQuadraticNormalProfile3D::TopologyAffineCubic);
    const std::array<double, 3> layers{{0.5, 0.75, 1.5}};
    for (int layer = 0; layer < 3; ++layer) {
        require(
            (interior[static_cast<std::size_t>(layer)]
                 - (point - h * layers[static_cast<std::size_t>(layer)]
                                * normal))
                    .norm()
                < 1.0e-15,
            "topology-affine interior query uses the specified cubic layer");
        require(
            (exterior[static_cast<std::size_t>(layer)]
                 - (point + h * layers[static_cast<std::size_t>(layer)]
                                * normal))
                    .norm()
                < 1.0e-15,
            "topology-affine exterior query uses the specified cubic layer");
    }
}

} // namespace

int main()
{
    try {
        test_shared_stencil_reproduces_complete_quadratics();
        test_boundary_stencil_is_valid();
        test_one_sided_quadratic_trace_recovery();
        test_signed_normal_query_points();
        test_topology_affine_cubic_query_points();
        std::cout << "3D shared quadratic restrict tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D shared quadratic restrict test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
