#include "native_nurbs_surface_3d.hpp"

#include "src/geometry/nurbs_basis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

using geometry::NurbsBasis1D;
using geometry3d::NurbsSurfacePatch3D;

struct RationalQuarterArc2D {
    std::array<Eigen::Vector2d, 3> controls;
    std::array<double, 3> weights;
};

int edge_index(PatchEdge3D edge)
{
    return static_cast<int>(edge);
}

RationalQuarterArc2D quarter_arc(double angle0)
{
    const double angle1 = angle0 + 0.5 * kPi;
    const double angle_mid = 0.5 * (angle0 + angle1);
    const double middle_weight = std::sqrt(0.5);
    return {{{{std::cos(angle0), std::sin(angle0)},
              {std::cos(angle_mid) / middle_weight,
               std::sin(angle_mid) / middle_weight},
              {std::cos(angle1), std::sin(angle1)}}},
            {{1.0, middle_weight, 1.0}}};
}

NurbsBasis1D quadratic_unit_basis()
{
    return NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
}

NurbsBasis1D linear_unit_basis()
{
    return NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0});
}

void append_patch(NativeNurbsSurface3D& surface,
                  std::string name,
                  NurbsSurfacePatch3D patch)
{
    surface.patch_names.push_back(std::move(name));
    surface.patches.push_back(std::move(patch));
    surface.smooth_neighbors.emplace_back();
}

void connect_smooth(NativeNurbsSurface3D& surface,
                    int patch_a,
                    PatchEdge3D edge_a,
                    int patch_b,
                    PatchEdge3D edge_b,
                    bool reversed)
{
    if (patch_a < 0 || patch_a >= static_cast<int>(surface.patches.size())
        || patch_b < 0 || patch_b >= static_cast<int>(surface.patches.size())) {
        throw std::out_of_range("smooth NURBS connection has invalid patch");
    }
    auto& slot_a = surface.smooth_neighbors[static_cast<std::size_t>(patch_a)]
                                           [static_cast<std::size_t>(edge_index(edge_a))];
    auto& slot_b = surface.smooth_neighbors[static_cast<std::size_t>(patch_b)]
                                           [static_cast<std::size_t>(edge_index(edge_b))];
    if (slot_a || slot_b)
        throw std::runtime_error("NURBS patch edge has multiple smooth neighbors");
    slot_a = SmoothPatchNeighbor3D{patch_b, edge_b, reversed};
    slot_b = SmoothPatchNeighbor3D{patch_a, edge_a, reversed};
}

NurbsSurfacePatch3D make_torus_quarter_patch(int u_quarter,
                                             int v_quarter)
{
    constexpr double cx = 0.07;
    constexpr double cy = -0.04;
    constexpr double cz = 0.03;
    constexpr double major_radius = 0.55;
    constexpr double minor_radius = 0.20;
    const RationalQuarterArc2D u_arc = quarter_arc(0.5 * kPi * u_quarter);
    const RationalQuarterArc2D v_arc = quarter_arc(0.5 * kPi * v_quarter);

    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(3));
    std::vector<std::vector<double>> weights(3, std::vector<double>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double rho = major_radius
                             + minor_radius * v_arc.controls[j].x();
            controls[i][j] = {
                cx + rho * u_arc.controls[i].x(),
                cy + rho * u_arc.controls[i].y(),
                cz + minor_radius * v_arc.controls[j].y()};
            weights[i][j] = u_arc.weights[i] * v_arc.weights[j];
        }
    }
    return NurbsSurfacePatch3D(
        quadratic_unit_basis(), quadratic_unit_basis(),
        std::move(controls), std::move(weights));
}

NativeNurbsSurface3D make_torus()
{
    NativeNurbsSurface3D surface;
    surface.name = "torus";
    surface.description = "exact rational-quadratic NURBS torus";
    for (int iu = 0; iu < 4; ++iu) {
        for (int iv = 0; iv < 4; ++iv) {
            append_patch(surface,
                         "torus_u" + std::to_string(iu)
                             + "_v" + std::to_string(iv),
                         make_torus_quarter_patch(iu, iv));
        }
    }
    auto patch_id = [](int iu, int iv) {
        return 4 * (iu % 4) + (iv % 4);
    };
    for (int iu = 0; iu < 4; ++iu) {
        for (int iv = 0; iv < 4; ++iv) {
            connect_smooth(surface,
                           patch_id(iu, iv), PatchEdge3D::UMax,
                           patch_id(iu + 1, iv), PatchEdge3D::UMin,
                           false);
            connect_smooth(surface,
                           patch_id(iu, iv), PatchEdge3D::VMax,
                           patch_id(iu, iv + 1), PatchEdge3D::VMin,
                           false);
        }
    }
    surface.expected_area = 4.0 * kPi * kPi * 0.55 * 0.20;
    surface.exact_inside = [](const Eigen::Vector3d& x) {
        const double rho = std::hypot(x.x() - 0.07, x.y() + 0.04);
        return (rho - 0.55) * (rho - 0.55)
             + (x.z() - 0.03) * (x.z() - 0.03) < 0.20 * 0.20;
    };
    return surface;
}

NurbsSurfacePatch3D make_cylinder_wall_quarter(double radius,
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
    return NurbsSurfacePatch3D(
        quadratic_unit_basis(), linear_unit_basis(),
        std::move(controls), std::move(weights));
}

NurbsSurfacePatch3D make_annulus_quarter(double z,
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
    return NurbsSurfacePatch3D(
        quadratic_unit_basis(), linear_unit_basis(),
        std::move(controls), std::move(weights));
}

NativeNurbsSurface3D make_hollow_cylinder()
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

void append_l_prism_top(NativeNurbsSurface3D& surface,
                        const std::string& name,
                        double xmin,
                        double xmax,
                        double ymin,
                        double ymax,
                        double z)
{
    append_patch(surface, name, NurbsSurfacePatch3D::make_bilinear_plane(
        {xmin, ymin, z}, {xmax, ymin, z},
        {xmin, ymax, z}, {xmax, ymax, z}));
}

void append_l_prism_bottom(NativeNurbsSurface3D& surface,
                           const std::string& name,
                           double xmin,
                           double xmax,
                           double ymin,
                           double ymax,
                           double z)
{
    append_patch(surface, name, NurbsSurfacePatch3D::make_bilinear_plane(
        {xmin, ymin, z}, {xmin, ymax, z},
        {xmax, ymin, z}, {xmax, ymax, z}));
}

NativeNurbsSurface3D make_l_prism()
{
    constexpr double sx = 0.07;
    constexpr double sy = -0.07;
    constexpr double z0 = -0.63;
    constexpr double z1 = 0.67;
    constexpr double arm = 0.60;
    const std::array<std::array<double, 4>, 3> cells{{
        {{sx - arm, sx, sy - arm, sy}},
        {{sx, sx + arm, sy - arm, sy}},
        {{sx - arm, sx, sy, sy + arm}}}};
    NativeNurbsSurface3D surface;
    surface.name = "l_prism";
    surface.description = "twelve native bilinear NURBS L-prism patches";
    for (int cell = 0; cell < 3; ++cell) {
        const auto& bounds = cells[static_cast<std::size_t>(cell)];
        append_l_prism_bottom(surface,
                              "bottom_cell_" + std::to_string(cell),
                              bounds[0], bounds[1], bounds[2], bounds[3], z0);
    }
    for (int cell = 0; cell < 3; ++cell) {
        const auto& bounds = cells[static_cast<std::size_t>(cell)];
        append_l_prism_top(surface,
                           "top_cell_" + std::to_string(cell),
                           bounds[0], bounds[1], bounds[2], bounds[3], z1);
    }
    const std::array<Eigen::Vector2d, 6> boundary{{
        {sx - arm, sy - arm}, {sx + arm, sy - arm},
        {sx + arm, sy},       {sx, sy},
        {sx, sy + arm},       {sx - arm, sy + arm}}};
    for (int side = 0; side < 6; ++side) {
        const Eigen::Vector2d a = boundary[static_cast<std::size_t>(side)];
        const Eigen::Vector2d b = boundary[static_cast<std::size_t>((side + 1) % 6)];
        append_patch(surface,
                     "side_" + std::to_string(side),
                     NurbsSurfacePatch3D::make_bilinear_plane(
                         {a.x(), a.y(), z0}, {b.x(), b.y(), z0},
                         {a.x(), a.y(), z1}, {b.x(), b.y(), z1}));
    }

    connect_smooth(surface, 0, PatchEdge3D::VMax,
                   1, PatchEdge3D::VMin, false);
    connect_smooth(surface, 0, PatchEdge3D::UMax,
                   2, PatchEdge3D::UMin, false);
    connect_smooth(surface, 3, PatchEdge3D::UMax,
                   4, PatchEdge3D::UMin, false);
    connect_smooth(surface, 3, PatchEdge3D::VMax,
                   5, PatchEdge3D::VMin, false);

    surface.expected_area = 2.0 * 3.0 * arm * arm
                          + 8.0 * arm * (z1 - z0);
    surface.exact_inside = [](const Eigen::Vector3d& x) {
        const double px = x.x() - 0.07;
        const double py = x.y() + 0.07;
        const bool in_plan =
            (px > -0.60 && px < 0.60 && py > -0.60 && py < 0.0)
         || (px > -0.60 && px < 0.0 && py > 0.0 && py < 0.60);
        return in_plan && x.z() > -0.63 && x.z() < 0.67;
    };
    return surface;
}

} // namespace

NativeNurbsSurface3D make_native_nurbs_surface_3d(GeometryKind3D kind)
{
    switch (kind) {
    case GeometryKind3D::Torus:
        return make_torus();
    case GeometryKind3D::HollowCylinder:
        return make_hollow_cylinder();
    case GeometryKind3D::LPrism:
        return make_l_prism();
    }
    throw std::invalid_argument("unknown native 3D geometry kind");
}

} // namespace kfbim::app3d
