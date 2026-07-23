#include "native_nurbs_surface_transform_3d.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::RigidTransform3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::transform_native_nurbs_surface_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(const Eigen::Vector3d& actual,
                  const Eigen::Vector3d& expected,
                  double tolerance,
                  const std::string& message)
{
    require((actual - expected).norm() <= tolerance, message);
}
bool same_smooth_neighbors(
    const std::vector<std::array<std::optional<kfbim::app3d::SmoothPatchNeighbor3D>, 4>>& left,
    const std::vector<std::array<std::optional<kfbim::app3d::SmoothPatchNeighbor3D>, 4>>& right)
{
    if (left.size() != right.size())
        return false;
    for (std::size_t patch = 0; patch < left.size(); ++patch) {
        for (std::size_t edge = 0; edge < left[patch].size(); ++edge) {
            const auto& a = left[patch][edge];
            const auto& b = right[patch][edge];
            if (static_cast<bool>(a) != static_cast<bool>(b))
                return false;
            if (a && (a->patch != b->patch || a->edge != b->edge ||
                      a->reversed != b->reversed))
                return false;
        }
    }
    return true;
}

template <class Function>
void require_throws(Function&& function, const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

RigidTransform3D rigid_transform()
{
    constexpr double degrees_to_radians =
        3.141592653589793238462643383279502884 / 180.0;
    return RigidTransform3D::from_axis_angle(
        {1.0, 2.0, 3.0}, 17.0 * degrees_to_radians,
        {0.07, -0.07, 0.02}, {0.137, -0.083, 0.061});
}

void test_transform_points_and_vectors()
{
    const RigidTransform3D identity;
    const Eigen::Vector3d point(0.2, -0.4, 0.6);
    require_near(identity.forward_point(point), point, 0.0,
                 "default transform is identity");

    const RigidTransform3D transform = rigid_transform();
    const Eigen::Vector3d moved = transform.forward_point(point);
    require_near(transform.inverse_point(moved), point, 2.0e-14,
                 "point forward/inverse round trip");

    const Eigen::Vector3d vector(-0.35, 0.21, 0.13);
    require_near(transform.forward_vector(vector), transform.rotation() * vector,
                 2.0e-14, "vectors rotate without center or translation");
    require_near(transform.inverse_point(transform.forward_point(point) +
                                         transform.forward_vector(vector)),
                 point + vector, 2.0e-14, "inverse geometry relationship");
}

void test_transformed_l_prism_geometry_and_metadata()
{
    const NativeNurbsSurface3D original =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const RigidTransform3D transform = rigid_transform();
    const NativeNurbsSurface3D moved =
        transform_native_nurbs_surface_3d(original, transform);

    require(moved.name == original.name && moved.description == original.description,
            "surface metadata is unchanged");
    require(moved.patch_names == original.patch_names,
            "patch names and order are unchanged");
    require(same_smooth_neighbors(moved.smooth_neighbors, original.smooth_neighbors),
            "smooth topology is unchanged");
    require(moved.topological_patch_neighbors == original.topological_patch_neighbors,
            "topological neighbors are unchanged");
    require(moved.patch_components == original.patch_components,
            "component data is unchanged");
    require(moved.expected_area == original.expected_area,
            "expected area is unchanged");
    require(moved.geometric_connections.size() == original.geometric_connections.size(),
            "geometric connection count is unchanged");
    for (std::size_t connection = 0; connection < original.geometric_connections.size(); ++connection) {
        const auto& source = original.geometric_connections[connection];
        const auto& target = moved.geometric_connections[connection];
        require(source.first.patch == target.first.patch && source.first.edge == target.first.edge &&
                    source.first.begin == target.first.begin && source.first.end == target.first.end &&
                    source.second.patch == target.second.patch && source.second.edge == target.second.edge &&
                    source.second.begin == target.second.begin && source.second.end == target.second.end &&
                    source.reversed == target.reversed && source.g1 == target.g1,
                "geometric connections are unchanged");
    }

    require(moved.patches.size() == original.patches.size(), "patch count is unchanged");
    for (std::size_t patch = 0; patch < original.patches.size(); ++patch) {
        const auto& source = original.patches[patch];
        const auto& target = moved.patches[patch];
        require(source.basis_u().degree() == target.basis_u().degree() &&
                    source.basis_v().degree() == target.basis_v().degree() &&
                    source.basis_u().knots() == target.basis_u().knots() &&
                    source.basis_v().knots() == target.basis_v().knots(),
                "NURBS bases and knots are unchanged");
        require(source.weights() == target.weights(), "weights are unchanged");

        const double u = 0.37 * source.domain_start_u() + 0.63 * source.domain_end_u();
        const double v = 0.41 * source.domain_start_v() + 0.59 * source.domain_end_v();
        const auto source_derivatives = source.evaluate_with_derivatives(u, v);
        const auto target_derivatives = target.evaluate_with_derivatives(u, v);
        require_near(target_derivatives.point, transform.forward_point(source_derivatives.point),
                     5.0e-13, "transformed patch evaluation");
        require_near(target_derivatives.du, transform.forward_vector(source_derivatives.du),
                     5.0e-13, "du rotates consistently");
        require_near(target_derivatives.dv, transform.forward_vector(source_derivatives.dv),
                     5.0e-13, "dv rotates consistently");
        require_near(target.normal(u, v), transform.forward_vector(source.normal(u, v)),
                     5.0e-13, "normal rotates consistently");
    }

    const Eigen::Vector3d inside(0.10, -0.10, 0.15);
    const Eigen::Vector3d outside(0.30, 0.20, 0.15);
    require(moved.exact_inside(transform.forward_point(inside)) == original.exact_inside(inside),
            "transformed exact-inside predicate accepts inside point");
    require(moved.exact_inside(transform.forward_point(outside)) == original.exact_inside(outside),
            "transformed exact-inside predicate rejects outside point");
}

void test_invalid_transforms_are_rejected()
{
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    require_throws([&] { (void)RigidTransform3D(Eigen::DiagonalMatrix<double, 3>(2.0, 1.0, 1.0), zero, zero); }, "scale matrix is rejected");
    require_throws([&] { (void)RigidTransform3D(Eigen::DiagonalMatrix<double, 3>(-1.0, 1.0, 1.0), zero, zero); }, "reflection matrix is rejected");
    Eigen::Matrix3d nan_rotation = Eigen::Matrix3d::Identity();
    nan_rotation(0, 0) = std::numeric_limits<double>::quiet_NaN();
    require_throws([&] { (void)RigidTransform3D(nan_rotation, zero, zero); }, "NaN rotation is rejected");
    Eigen::Vector3d nan_center = zero;
    nan_center.x() = std::numeric_limits<double>::quiet_NaN();
    require_throws([&] { (void)RigidTransform3D(Eigen::Matrix3d::Identity(), nan_center, zero); }, "NaN center is rejected");
    Eigen::Vector3d nan_translation = zero;
    nan_translation.z() = std::numeric_limits<double>::quiet_NaN();
    require_throws([&] { (void)RigidTransform3D(Eigen::Matrix3d::Identity(), zero, nan_translation); }, "NaN translation is rejected");
    require_throws([&] { (void)RigidTransform3D::from_axis_angle(zero, 0.1, zero, zero); }, "zero axis is rejected");
    require_throws([&] { (void)RigidTransform3D::from_axis_angle(Eigen::Vector3d::UnitX(), std::numeric_limits<double>::quiet_NaN(), zero, zero); }, "NaN angle is rejected");
}

void test_empty_predicate_is_rejected()
{
    NativeNurbsSurface3D source = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    source.exact_inside = {};
    require_throws([&] { (void)transform_native_nurbs_surface_3d(source, rigid_transform()); },
                   "empty exact-inside predicate is rejected");
}

} // namespace

int main()
{
    try {
        test_transform_points_and_vectors();
        test_transformed_l_prism_geometry_and_metadata();
        test_invalid_transforms_are_rejected();
        test_empty_predicate_is_rejected();
        std::cout << "native NURBS rigid transform tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native NURBS rigid transform test failure: " << error.what() << '\n';
        return 1;
    }
}