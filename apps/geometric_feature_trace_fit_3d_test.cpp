#include "geometric_feature_trace_fit_3d.hpp"
#include "harmonic_polynomial_space_3d.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::GeometricFeatureTraceFit3D;
using kfbim::app3d::GeometricFeatureTraceKind3D;
using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::HarmonicPolynomialSpace3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::SurfaceDof3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double ambient_cubic(const Eigen::Vector3d& point)
{
    const double x = point.x();
    const double y = point.y();
    const double z = point.z();
    return 0.83 + 0.21 * x - 0.17 * y + 0.13 * z
         + 0.09 * x * y - 0.07 * x * z + 0.05 * y * z
         + 0.04 * x * x - 0.03 * y * y + 0.02 * z * z
         + 0.018 * x * x * y - 0.014 * x * z * z
         + 0.011 * y * y * z + 0.007 * x * y * z;
}

void check_local_trace(
    const GeometricFeatureTraceFit3D& fit,
    const SurfaceDofCloud3D& cloud,
    int center,
    const Eigen::VectorXd& values,
    double h,
    double tolerance)
{
    const SurfaceDof3D& target =
        cloud.dofs[static_cast<std::size_t>(center)];
    const HarmonicPolynomialSpace3D space(fit.degree());
    const Eigen::VectorXd coefficients = fit.coefficients(center, values);
    const std::vector<Eigen::Vector2d> samples{
        {0.0, 0.0}, {0.23, -0.17}, {-0.19, 0.21}, {0.31, 0.08}};
    for (const Eigen::Vector2d& sample : samples) {
        const Eigen::Vector3d physical =
            target.point
            + h * (sample.x() * target.tangent1
                   + sample.y() * target.tangent2);
        const double actual =
            space.basis(sample.x(), sample.y(), 0.0).dot(coefficients);
        require(
            std::abs(actual - ambient_cubic(physical)) <= tolerance,
            "feature trace failed target-sheet cubic reproduction");
        const double normal_derivative =
            space.gradient(sample.x(), sample.y(), 0.0)
                .row(2)
                .dot(coefficients);
        require(
            std::abs(normal_derivative) <= 5.0 * tolerance,
            "feature trace lift does not have zero normal derivative");
    }
}

void test_l_prism_feature_hierarchy(int degree)
{
    constexpr double h = 3.0 / 32.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, h);
    const GeometricFeatureTraceFit3D fit(surface, cloud, h, degree);
    const auto& summary = fit.summary();
    require(summary.feature_segments == 22,
            "L-prism feature connection count");
    require(summary.feature_vertices == 12,
            "L-prism true corner count excludes smooth split points");
    require(summary.active_maps > 0 && summary.edge_maps > 0,
            "L-prism edge trace maps were not built");
    require(summary.vertex_maps > 0,
            "L-prism vertex trace maps were not built");
    require(summary.maximum_sheets_per_map >= 3,
            "vertex trace did not couple all incident smooth sheets");
    require(summary.maximum_constraint_residual < 2.0e-9,
            "feature constraints are not satisfied by the data map");

    Eigen::VectorXd constant =
        Eigen::VectorXd::Ones(static_cast<int>(cloud.dofs.size()));
    Eigen::VectorXd cubic(static_cast<int>(cloud.dofs.size()));
    for (int q = 0; q < cubic.size(); ++q) {
        cubic[q] =
            ambient_cubic(cloud.dofs[static_cast<std::size_t>(q)].point);
    }

    int edge_center = -1;
    int vertex_center = -1;
    double maximum_constant_error = 0.0;
    const HarmonicPolynomialSpace3D space(degree);
    for (int center = 0;
         center < static_cast<int>(cloud.dofs.size());
         ++center) {
        const auto& map = fit.map(center);
        if (!map.active())
            continue;
        if (map.kind == GeometricFeatureTraceKind3D::Edge
            && edge_center < 0) {
            edge_center = center;
        }
        if (map.kind == GeometricFeatureTraceKind3D::Vertex
            && vertex_center < 0) {
            vertex_center = center;
        }
        const Eigen::VectorXd coefficients =
            fit.coefficients(center, constant);
        maximum_constant_error = std::max(
            maximum_constant_error,
            std::abs(
                space.basis(0.13, -0.22, 0.0).dot(coefficients) - 1.0));
        maximum_constant_error = std::max(
            maximum_constant_error,
            std::abs(
                space.gradient(0.13, -0.22, 0.0)
                    .row(2)
                    .dot(coefficients)));
    }
    require(edge_center >= 0 && vertex_center >= 0,
            "missing representative edge/vertex maps");
    require(maximum_constant_error < 3.0e-9,
            "feature hierarchy failed constant reproduction");
    const double tolerance = degree == 3 ? 2.0e-8 : 8.0e-8;
    check_local_trace(fit, cloud, edge_center, cubic, h, tolerance);
    check_local_trace(fit, cloud, vertex_center, cubic, h, tolerance);

    const std::vector<double> conditions = fit.condition_values();
    require(static_cast<int>(conditions.size()) == summary.active_maps,
            "feature condition count");
    require(std::all_of(
                conditions.begin(), conditions.end(),
                [](double value) {
                    return std::isfinite(value) && value >= 1.0;
                }),
            "feature condition values are invalid");
}

void test_smooth_geometry_has_no_feature_maps()
{
    constexpr double h = 3.0 / 24.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, h);
    const GeometricFeatureTraceFit3D fit(surface, cloud, h, 3);
    require(fit.summary().feature_segments == 0,
            "smooth torus unexpectedly contains non-G1 features");
    require(fit.summary().active_maps == 0,
            "smooth torus unexpectedly activated feature trace maps");
}

} // namespace

int main()
{
    try {
        test_l_prism_feature_hierarchy(3);
        test_l_prism_feature_hierarchy(4);
        test_smooth_geometry_has_no_feature_maps();
        std::cout << "3D geometric feature trace fit tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D geometric feature trace fit test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
