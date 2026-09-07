#include "src/support/geometry/native_nurbs_surface_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_transform_3d.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::RigidTransform3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::smooth_patch_component;
using kfbim::app3d::transform_native_nurbs_surface_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double planar_patch_area(
    const kfbim::geometry3d::NurbsSurfacePatch3D& patch)
{
    const double u0 = patch.domain_start_u();
    const double u1 = patch.domain_end_u();
    const double v0 = patch.domain_start_v();
    const double v1 = patch.domain_end_v();
    const auto derivatives = patch.evaluate_with_derivatives(
        0.5 * (u0 + u1), 0.5 * (v0 + v1));
    return derivatives.du.cross(derivatives.dv).norm()
         * (u1 - u0) * (v1 - v0);
}

double actual_area(const NativeNurbsSurface3D& surface)
{
    double result = 0.0;
    for (const auto& patch : surface.patches)
        result += planar_patch_area(patch);
    return result;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> control_bounds(
    const NativeNurbsSurface3D& surface)
{
    Eigen::Vector3d lower = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::infinity());
    Eigen::Vector3d upper = Eigen::Vector3d::Constant(
        -std::numeric_limits<double>::infinity());
    for (const auto& patch : surface.patches) {
        for (const auto& row : patch.control_net()) {
            for (const Eigen::Vector3d& point : row) {
                lower = lower.cwiseMin(point);
                upper = upper.cwiseMax(point);
            }
        }
    }
    return {lower, upper};
}

void require_closed(const NativeNurbsSurface3D& surface)
{
    const auto model = surface.geometry_model();
    const auto diagnostic = model.validate_closed();
    require(diagnostic.uncovered_interval_count == 0,
            "uncovered U-prism edge interval");
    require(diagnostic.multiply_covered_interval_count == 0,
            "multiply covered U-prism edge interval");
    require(diagnostic.position_mismatch_count == 0,
            "U-prism connected endpoints disagree");
    require(diagnostic.orientation_mismatch_count == 0,
            "U-prism orientation mismatch");
    require(diagnostic.g1_normal_mismatch_count == 0,
            "U-prism G1 normal mismatch");
}

void test_baseline()
{
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::UPrism);
    require(surface.name == "u_prism", "canonical U-prism name");
    require(surface.patches.size() == 18, "U-prism patch count");
    require(surface.geometric_connections.size() == 40,
            "U-prism connection count");
    require_closed(surface);

    constexpr double expected_area = 6.912;
    const double area = actual_area(surface);
    require(std::abs(surface.expected_area - expected_area) < 2.0e-14,
            "U-prism metadata area");
    require(std::abs(area - expected_area) < 2.0e-13,
            "U-prism integrated patch area");

    const auto bounds = control_bounds(surface);
    const Eigen::Vector3d expected_lower(-0.47, -0.43, -0.63);
    const Eigen::Vector3d expected_upper(0.61, 0.29, 0.67);
    require((bounds.first - expected_lower).lpNorm<Eigen::Infinity>()
                < 2.0e-14,
            "U-prism lower AABB");
    require((bounds.second - expected_upper).lpNorm<Eigen::Infinity>()
                < 2.0e-14,
            "U-prism upper AABB");

    require(surface.exact_inside({0.07, -0.25, 0.0}),
            "bottom bar inside predicate");
    require(surface.exact_inside({-0.29, 0.11, 0.0}),
            "left arm inside predicate");
    require(surface.exact_inside({0.43, 0.11, 0.0}),
            "right arm inside predicate");
    require(surface.exact_inside({-0.29, -0.07, 0.0}),
            "cap-cell seam inside predicate");
    require(!surface.exact_inside({0.07, 0.11, 0.0}),
            "central slot outside predicate");

    require(smooth_patch_component(surface, 0).size() == 5,
            "five-patch bottom G1 sheet");
    require(smooth_patch_component(surface, 5).size() == 5,
            "five-patch top G1 sheet");
    for (int side = 10; side < 18; ++side) {
        require(smooth_patch_component(surface, side).size() == 1,
                "C0 wall sheet is not merged across a crease");
    }

    std::cout << "baseline patches=" << surface.patches.size()
              << " connections=" << surface.geometric_connections.size()
              << " area=" << area
              << " aabb=[" << bounds.first.transpose() << "]-["
              << bounds.second.transpose() << "]\n";
}

void test_rigid_transform()
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const NativeNurbsSurface3D source =
        make_native_nurbs_surface_3d(GeometryKind3D::UPrism);
    const RigidTransform3D transform = RigidTransform3D::from_axis_angle(
        {1.0, 2.0, 3.0}, 17.0 * pi / 180.0,
        {0.07, -0.07, 0.02}, {0.137, -0.083, 0.061});
    const NativeNurbsSurface3D moved =
        transform_native_nurbs_surface_3d(source, transform);

    require_closed(moved);
    require(moved.patch_names == source.patch_names,
            "rigid transform preserves patch ordering");
    require(moved.topological_patch_neighbors
                == source.topological_patch_neighbors,
            "rigid transform preserves topology");
    require(moved.geometric_connections.size()
                == source.geometric_connections.size(),
            "rigid transform preserves interval connections");
    const double area_error = std::abs(actual_area(moved) - actual_area(source));
    require(area_error < 2.0e-12, "rigid transform preserves U-prism area");

    const Eigen::Vector3d inside(0.43, 0.11, 0.0);
    const Eigen::Vector3d slot(0.07, 0.11, 0.0);
    require(moved.exact_inside(transform.forward_point(inside)),
            "rigid U-prism inside predicate");
    require(!moved.exact_inside(transform.forward_point(slot)),
            "rigid U-prism slot predicate");

    double point_error = 0.0;
    double normal_error = 0.0;
    for (std::size_t patch_id = 0; patch_id < source.patches.size(); ++patch_id) {
        const auto& a = source.patches[patch_id];
        const auto& b = moved.patches[patch_id];
        const double u = 0.5 * (a.domain_start_u() + a.domain_end_u());
        const double v = 0.5 * (a.domain_start_v() + a.domain_end_v());
        point_error = std::max(
            point_error,
            (b.evaluate(u, v) - transform.forward_point(a.evaluate(u, v)))
                .norm());
        normal_error = std::max(
            normal_error,
            (b.normal(u, v) - transform.forward_vector(a.normal(u, v)))
                .norm());
    }
    require(point_error < 5.0e-13, "rigid patch point covariance");
    require(normal_error < 5.0e-13, "rigid patch normal covariance");
    const auto bounds = control_bounds(moved);
    require(bounds.first.minCoeff() > -1.5
                && bounds.second.maxCoeff() < 1.5,
            "rigid U-prism remains inside the solver box");

    std::cout << "rigid area_error=" << area_error
              << " point_error=" << point_error
              << " normal_error=" << normal_error
              << " aabb=[" << bounds.first.transpose() << "]-["
              << bounds.second.transpose() << "]\n";
}

} // namespace

int main()
{
    try {
        test_baseline();
        test_rigid_transform();
        std::cout << "U-prism geometry tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "U-prism geometry test failure: " << error.what() << '\n';
        return 1;
    }
}
