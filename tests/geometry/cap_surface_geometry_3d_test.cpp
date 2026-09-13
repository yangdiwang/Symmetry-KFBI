#include "src/geometry/models3d/analytic/cap_surface_geometry_3d.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using kfbim::geometry3d::AnalyticCapGeometry3D;
using kfbim::geometry3d::AnalyticCapGeometryOptions3D;
using kfbim::geometry3d::AnalyticCapShape3D;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

void require_near(const Eigen::Vector3d& actual,
                  const Eigen::Vector3d& expected,
                  double tolerance,
                  const std::string& message)
{
    require((actual - expected).norm() <= tolerance, message);
}

void test_rigid_ellipsoid_queries_use_world_coordinates()
{
    AnalyticCapGeometryOptions3D options;
    options.shape = AnalyticCapShape3D::Ellipsoid;
    options.ellipsoid_axes = {2.0, 3.0, 4.0};
    options.rigid_rotation << 0.0, -1.0, 0.0,
                              1.0,  0.0, 0.0,
                              0.0,  0.0, 1.0;
    options.rigid_center = {1.0, -2.0, 0.5};
    options.rigid_translation = {5.0, -1.0, 0.25};

    const AnalyticCapGeometry3D geometry(options);
    const Eigen::Vector3d surface =
        geometry.surface_point_from_direction(Eigen::Vector3d::UnitX());
    require_near(surface, {4.0, -2.0, 0.25}, 1.0e-14,
                 "rigid ellipsoid surface direction map");
    require_near(geometry.level_set(surface), 0.0, 1.0e-14,
                 "mapped ellipsoid point lies on its level set");
    require_near(geometry.outward_normal(surface), Eigen::Vector3d::UnitY(),
                 1.0e-14, "ellipsoid normal follows the rigid rotation");
    require(geometry.inside(Eigen::Vector3d(4.0, -4.0, 0.25)),
            "rigidly mapped ellipsoid center is inside");

    const auto north = geometry.evaluate(0, 0.5, 0.5);
    require_near(north.point, {4.0, -4.0, 4.25}, 1.0e-14,
                 "central north-cap parameterization");
    require_near(north.normal, Eigen::Vector3d::UnitZ(), 1.0e-14,
                 "central north-cap normal");
    require(north.area_element > 0.0,
            "central north-cap parameterization is regular");
}

void check_all_chart_interiors(const AnalyticCapGeometry3D& geometry,
                               const std::string& shape)
{
    for (int patch = 0; patch < geometry.patch_count(); ++patch) {
        const double u = 0.29 + 0.017 * static_cast<double>(patch % 4);
        const double v = 0.41 + 0.013 * static_cast<double>(patch % 5);
        const auto sample = geometry.evaluate(patch, u, v);
        const std::string where = shape + " patch " + std::to_string(patch);

        require_near(geometry.level_set(sample.point), 0.0, 2.0e-13,
                     where + " lies on level set");
        require_near(sample.normal.norm(), 1.0, 2.0e-14,
                     where + " has a unit normal");
        for (int axis = 0; axis < 2; ++axis) {
            const double scale = std::max(1.0, sample.tangents.col(axis).norm());
            require(std::abs(sample.normal.dot(sample.tangents.col(axis)))
                        <= 2.0e-13 * scale,
                    where + " normal is tangent-orthogonal");
        }

        const auto location = geometry.locate(sample.point);
        require(location.valid, where + " can be located");
        const auto recovered =
            geometry.evaluate(location.patch, location.u, location.v);
        require_near(recovered.point, sample.point, 3.0e-12,
                     where + " locate/evaluate round trip");

        if (patch == 0 || patch == 5 || patch == 9 || patch == 13) {
            constexpr double step = 1.0e-6;
            const Eigen::Vector3d finite_u =
                (geometry.evaluate(patch, u + step, v).point
                 - geometry.evaluate(patch, u - step, v).point)
                / (2.0 * step);
            const Eigen::Vector3d finite_v =
                (geometry.evaluate(patch, u, v + step).point
                 - geometry.evaluate(patch, u, v - step).point)
                / (2.0 * step);
            require_near(finite_u, sample.tangents.col(0), 2.0e-9,
                         where + " u tangent matches finite difference");
            require_near(finite_v, sample.tangents.col(1), 2.0e-9,
                         where + " v tangent matches finite difference");
        }
    }
}

void test_all_cap_atlas_charts_round_trip()
{
    AnalyticCapGeometryOptions3D ellipsoid_options;
    ellipsoid_options.shape = AnalyticCapShape3D::Ellipsoid;
    ellipsoid_options.ellipsoid_axes = {1.2, 0.9, 0.72};
    ellipsoid_options.rigid_rotation << 0.36, -0.8, 0.48,
                                          0.8,  0.0, -0.6,
                                          0.48, 0.6, 0.64;
    ellipsoid_options.rigid_center = {0.17, -0.23, 0.11};
    ellipsoid_options.rigid_translation = {0.09, 0.14, -0.07};
    check_all_chart_interiors(
        AnalyticCapGeometry3D(ellipsoid_options), "ellipsoid");

    AnalyticCapGeometryOptions3D flower_options;
    flower_options.shape = AnalyticCapShape3D::Flower;
    flower_options.flower_epsilon = 0.16;
    flower_options.flower_eta = 0.035;
    check_all_chart_interiors(
        AnalyticCapGeometry3D(flower_options), "flower");
}

void test_flower_radius_and_normal_preserve_existing_formula()
{
    AnalyticCapGeometryOptions3D options;
    options.shape = AnalyticCapShape3D::Flower;
    options.flower_epsilon = 0.16;
    options.flower_eta = 0.035;

    const AnalyticCapGeometry3D geometry(options);
    const Eigen::Vector3d equator =
        geometry.surface_point_from_direction(Eigen::Vector3d::UnitX());
    require_near(equator, {1.125, 0.0, 0.0}, 1.0e-14,
                 "flower equatorial radius");
    require_near(geometry.level_set(equator), 0.0, 1.0e-14,
                 "flower direction map lies on the level set");
    require_near(geometry.outward_normal(equator), Eigen::Vector3d::UnitX(),
                 1.0e-14, "flower axial normal");

    const Eigen::Vector3d pole =
        geometry.surface_point_from_direction(Eigen::Vector3d::UnitZ());
    require_near(pole, {0.0, 0.0, 1.07}, 1.0e-14,
                 "flower polar radius");
    require(geometry.inside(Eigen::Vector3d::Zero()),
            "flower origin is inside");
}

void test_invalid_physical_shape_is_rejected()
{
    AnalyticCapGeometryOptions3D options;
    options.ellipsoid_axes.x() = 0.0;
    try {
        (void)AnalyticCapGeometry3D(options);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("nonpositive ellipsoid axis was accepted");
}

} // namespace

int main()
{
    try {
        test_rigid_ellipsoid_queries_use_world_coordinates();
        test_flower_radius_and_normal_preserve_existing_formula();
        test_all_cap_atlas_charts_round_trip();
        test_invalid_physical_shape_is_rejected();
        std::cout << "analytic cap surface geometry tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "analytic cap surface geometry test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
