#include "construction.hpp"
#include "models_3d.hpp"

#include <array>
#include <utility>

namespace kfbim::app3d::native_detail {

namespace {

geometry3d::NurbsSurfacePatch3D make_cylinder_wall_quarter(
    double radius,
    int quarter,
    double z_first,
    double z_second)
{
    constexpr double cx = 0.06;
    constexpr double cy = -0.05;
    const RationalQuarterArc2D arc = quarter_arc(0.5 * kPi * quarter);
    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(2));
    std::vector<std::vector<double>> weights(3, std::vector<double>(2));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            controls[i][j] = {
                cx + radius * arc.controls[i].x(),
                cy + radius * arc.controls[i].y(),
                j == 0 ? z_first : z_second};
            weights[i][j] = arc.weights[i];
        }
    }
    return geometry3d::NurbsSurfacePatch3D(
        quadratic_unit_basis(), linear_unit_basis(),
        std::move(controls), std::move(weights));
}

geometry3d::NurbsSurfacePatch3D make_annulus_quarter(
    double z,
    int quarter,
    double radius_first,
    double radius_second)
{
    constexpr double cx = 0.06;
    constexpr double cy = -0.05;
    const RationalQuarterArc2D arc = quarter_arc(0.5 * kPi * quarter);
    const std::array<double, 2> radii{{radius_first, radius_second}};
    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(2));
    std::vector<std::vector<double>> weights(3, std::vector<double>(2));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            controls[i][j] = {
                cx + radii[j] * arc.controls[i].x(),
                cy + radii[j] * arc.controls[i].y(),
                z};
            weights[i][j] = arc.weights[i];
        }
    }
    return geometry3d::NurbsSurfacePatch3D(
        quadratic_unit_basis(), linear_unit_basis(),
        std::move(controls), std::move(weights));
}

} // namespace

NativeNurbsSurface3D make_hollow_cylinder_3d()
{
    constexpr double outer_radius = 0.55;
    constexpr double inner_radius = 0.25;
    constexpr double z0 = -0.63;
    constexpr double z1 = 0.67;
    NativeNurbsSurface3D surface;
    surface.name = "cylinder";
    surface.description = "exact rational-quadratic NURBS hollow cylinder";
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "outer_wall_q" + std::to_string(quarter),
                     make_cylinder_wall_quarter(
                         outer_radius, quarter, z0, z1));
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "inner_wall_q" + std::to_string(quarter),
                     make_cylinder_wall_quarter(
                         inner_radius, quarter, z1, z0));
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "top_annulus_q" + std::to_string(quarter),
                     make_annulus_quarter(
                         z1, quarter, outer_radius, inner_radius));
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "bottom_annulus_q" + std::to_string(quarter),
                     make_annulus_quarter(
                         z0, quarter, inner_radius, outer_radius));
    }
    for (int sheet = 0; sheet < 4; ++sheet) {
        const int first = 4 * sheet;
        for (int quarter = 0; quarter < 4; ++quarter) {
            connect_smooth(surface,
                           first + quarter, PatchEdge3D::UMax,
                           first + (quarter + 1) % 4, PatchEdge3D::UMin,
                           false);
        }
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        connect_feature(surface,
            quarter, PatchEdge3D::VMax, 0.0, 1.0,
            8 + quarter, PatchEdge3D::VMin, 0.0, 1.0, false);
        connect_feature(surface,
            quarter, PatchEdge3D::VMin, 0.0, 1.0,
            12 + quarter, PatchEdge3D::VMax, 0.0, 1.0, false);
        connect_feature(surface,
            4 + quarter, PatchEdge3D::VMin, 0.0, 1.0,
            8 + quarter, PatchEdge3D::VMax, 0.0, 1.0, false);
        connect_feature(surface,
            4 + quarter, PatchEdge3D::VMax, 0.0, 1.0,
            12 + quarter, PatchEdge3D::VMin, 0.0, 1.0, false);
    }
    const double height = z1 - z0;
    surface.expected_area = 2.0 * kPi * (outer_radius + inner_radius) * height
        + 2.0 * kPi * (outer_radius * outer_radius
                       - inner_radius * inner_radius);
    surface.exact_inside = [](const Eigen::Vector3d& x) {
        const double radius_sq = (x.x() - 0.06) * (x.x() - 0.06)
                               + (x.y() + 0.05) * (x.y() + 0.05);
        return radius_sq > 0.25 * 0.25
            && radius_sq < 0.55 * 0.55
            && x.z() > -0.63 && x.z() < 0.67;
    };
    return surface;
}

} // namespace kfbim::app3d::native_detail
