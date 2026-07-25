#include "neumann_edge_continuity_3d.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {
using namespace kfbim::app3d;

void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

Eigen::Vector3d edge_point(const NativeNurbsSurface3D& surface,
                           const kfbim::geometry3d::NurbsPatchEdgeInterval3D& interval,
                           double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    switch (interval.edge) {
    case PatchEdge3D::UMin: return patch.evaluate(patch.domain_start_u(), parameter);
    case PatchEdge3D::UMax: return patch.evaluate(patch.domain_end_u(), parameter);
    case PatchEdge3D::VMin: return patch.evaluate(parameter, patch.domain_start_v());
    case PatchEdge3D::VMax: return patch.evaluate(parameter, patch.domain_end_v());
    }
    throw std::runtime_error("unknown edge");
}

double edge_length(const NativeNurbsSurface3D& surface,
                   const kfbim::geometry3d::NurbsPatchEdgeInterval3D& interval)
{
    constexpr std::array<double, 8> x = {{-0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346424956498, 0.1834346424956498,
         0.5255324099163290, 0.7966664774136267, 0.9602898564975363}};
    constexpr std::array<double, 8> w = {{0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620, 0.3626837833783620,
        0.3137066458778873, 0.2223810344533745, 0.1012285362903763}};
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const double half = 0.5 * (interval.end - interval.begin);
    const double middle = 0.5 * (interval.begin + interval.end);
    double result = 0.0;
    for (std::size_t q = 0; q < x.size(); ++q) {
        const double parameter = middle + half * x[q];
        const auto d = interval.edge == PatchEdge3D::UMin
            ? patch.evaluate_with_derivatives(patch.domain_start_u(), parameter)
            : interval.edge == PatchEdge3D::UMax
            ? patch.evaluate_with_derivatives(patch.domain_end_u(), parameter)
            : interval.edge == PatchEdge3D::VMin
            ? patch.evaluate_with_derivatives(parameter, patch.domain_start_v())
            : patch.evaluate_with_derivatives(parameter, patch.domain_end_v());
        result += w[q] * ((interval.edge == PatchEdge3D::UMin || interval.edge == PatchEdge3D::UMax) ? d.dv.norm() : d.du.norm());
    }
    return half * result;
}

double exact_value(const Eigen::Vector3d& x)
{
    return std::exp(0.35 * x.x()) * std::cos(0.21 * x.y())
         * std::cos(std::sqrt(0.35 * 0.35 - 0.21 * 0.21) * x.z());
}

Eigen::VectorXd exact_density(const SurfaceDofCloud3D& cloud)
{
    Eigen::VectorXd values(static_cast<Eigen::Index>(cloud.dofs.size()));
    for (std::size_t i = 0; i < cloud.dofs.size(); ++i)
        values[static_cast<Eigen::Index>(i)] = exact_value(cloud.dofs[i].point);
    return values;
}

void test_non_g1_topology_and_rows()
{
    const auto surface = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const double h = 3.0 / 32.0;
    const auto cloud = make_native_surface_dofs_3d(surface, h);
    const auto constraints = build_neumann_edge_constraints_3d(surface, cloud, h);
    require(constraints.matrix.rows() == static_cast<Eigen::Index>(constraints.samples.size())
                && constraints.matrix.rows() == constraints.quadrature_weights.size()
                && constraints.matrix.cols() == static_cast<Eigen::Index>(cloud.dofs.size()),
            "constraint dimensions");
    std::map<int, int> row_counts;
    std::map<int, double> previous_second;
    const double gap_limit = 1.0e-11 * surface.geometry_model().control_bounds().diameter();
    for (const auto& sample : constraints.samples) {
        const auto& connection = surface.geometric_connections.at(sample.connection_index);
        require(!connection.g1, "G1 edge was constrained");
        require(sample.first_parameter >= connection.first.begin && sample.first_parameter <= connection.first.end
                    && sample.second_parameter >= connection.second.begin && sample.second_parameter <= connection.second.end,
                "partial interval not preserved");
        require(std::isfinite(sample.quadrature_weight) && sample.quadrature_weight > 0.0
                    && sample.mapped_point_gap <= gap_limit
                    && (edge_point(surface, connection.first, sample.first_parameter)
                        - edge_point(surface, connection.second, sample.second_parameter)).norm() <= gap_limit,
                "invalid edge sample geometry or quadrature");
        const int before = row_counts[sample.connection_index]++;
        require(sample.sample_index == before && sample.sample_count >= 2,
                "cell-center sample indexing");
        if (before > 0)
            require(connection.reversed ? sample.second_parameter < previous_second[sample.connection_index]
                                        : sample.second_parameter > previous_second[sample.connection_index],
                    "second-edge orientation");
        previous_second[sample.connection_index] = sample.second_parameter;
    }
    int non_g1 = 0;
    for (int c = 0; c < static_cast<int>(surface.geometric_connections.size()); ++c) {
        if (surface.geometric_connections[c].g1) require(row_counts[c] == 0, "G1 row count");
        else { ++non_g1; require(row_counts[c] == std::max(2, static_cast<int>(std::ceil(edge_length(surface, surface.geometric_connections[c].first) / h))), "non-G1 connection row count"); }
    }
    require(constraints.non_g1_connection_count == non_g1, "non-G1 count");
    require(apply_neumann_edge_constraints_3d(constraints,
                Eigen::VectorXd::Ones(constraints.density_size)).lpNorm<Eigen::Infinity>() <= 5.0e-13,
            "constant partition of unity");
}

void test_fallback_and_order()
{
    const auto surface = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const auto coarse_cloud = make_native_surface_dofs_3d(surface, 10.0);
    const auto fallback = build_neumann_edge_constraints_3d(surface, coarse_cloud, 10.0);
    require(fallback.reduced_order_row_count > 0, "missing 2x2 fallback");
    for (int row = 0; row < fallback.matrix.outerSize(); ++row)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(fallback.matrix, row); it; ++it)
            require(it.col() >= 0 && it.col() < fallback.matrix.cols() && std::isfinite(it.value()), "fallback entry");
    const double h32 = 3.0 / 32.0, h64 = 3.0 / 64.0;
    const auto cloud32 = make_native_surface_dofs_3d(surface, h32);
    const auto cloud64 = make_native_surface_dofs_3d(surface, h64);
    const double m32 = neumann_edge_mismatch_weighted_rms_3d(
        build_neumann_edge_constraints_3d(surface, cloud32, h32), exact_density(cloud32));
    const double m64 = neumann_edge_mismatch_weighted_rms_3d(
        build_neumann_edge_constraints_3d(surface, cloud64, h64), exact_density(cloud64));
    require(std::isfinite(m32) && std::isfinite(m64) && m32 > 0.0 && m64 > 0.0 && m32 / m64 >= 6.0,
            "exact trace mismatch is not at least third-order");
}
}

int main()
{
    try {
        test_non_g1_topology_and_rows();
        test_fallback_and_order();
        std::cout << "3D Neumann edge-continuity tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D Neumann edge-continuity test failure: " << error.what() << '\n';
        return 1;
    }
}