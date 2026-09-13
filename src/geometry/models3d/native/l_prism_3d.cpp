#include "construction.hpp"
#include "models_3d.hpp"

#include <array>

namespace kfbim::app3d::native_detail {

NativeNurbsSurface3D make_l_prism_3d()
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
        append_prism_bottom(surface,
                            "bottom_cell_" + std::to_string(cell),
                            bounds[0], bounds[1], bounds[2], bounds[3], z0);
    }
    for (int cell = 0; cell < 3; ++cell) {
        const auto& bounds = cells[static_cast<std::size_t>(cell)];
        append_prism_top(surface,
                         "top_cell_" + std::to_string(cell),
                         bounds[0], bounds[1], bounds[2], bounds[3], z1);
    }
    const std::array<Eigen::Vector2d, 6> boundary{{
        {sx - arm, sy - arm}, {sx + arm, sy - arm},
        {sx + arm, sy},       {sx, sy},
        {sx, sy + arm},       {sx - arm, sy + arm}}};
    for (int side = 0; side < 6; ++side) {
        const Eigen::Vector2d a = boundary[static_cast<std::size_t>(side)];
        const Eigen::Vector2d b =
            boundary[static_cast<std::size_t>((side + 1) % 6)];
        append_patch(surface,
                     "side_" + std::to_string(side),
                     geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
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

    for (int side = 0; side < 6; ++side) {
        const int side_patch = 6 + side;
        connect_feature(surface,
            side_patch, PatchEdge3D::UMax, 0.0, 1.0,
            6 + (side + 1) % 6, PatchEdge3D::UMin, 0.0, 1.0, false);
    }

    connect_feature(surface, 6, PatchEdge3D::VMin, 0.0, 0.5,
                    0, PatchEdge3D::UMin, 0.0, 1.0, false);
    connect_feature(surface, 6, PatchEdge3D::VMin, 0.5, 1.0,
                    1, PatchEdge3D::UMin, 0.0, 1.0, false);
    connect_feature(surface, 7, PatchEdge3D::VMin, 0.0, 1.0,
                    1, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface, 8, PatchEdge3D::VMin, 0.0, 1.0,
                    1, PatchEdge3D::UMax, 0.0, 1.0, true);
    connect_feature(surface, 9, PatchEdge3D::VMin, 0.0, 1.0,
                    2, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface, 10, PatchEdge3D::VMin, 0.0, 1.0,
                    2, PatchEdge3D::UMax, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMin, 0.0, 0.5,
                    2, PatchEdge3D::VMin, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMin, 0.5, 1.0,
                    0, PatchEdge3D::VMin, 0.0, 1.0, true);

    connect_feature(surface, 6, PatchEdge3D::VMax, 0.0, 0.5,
                    3, PatchEdge3D::VMin, 0.0, 1.0, false);
    connect_feature(surface, 6, PatchEdge3D::VMax, 0.5, 1.0,
                    4, PatchEdge3D::VMin, 0.0, 1.0, false);
    connect_feature(surface, 7, PatchEdge3D::VMax, 0.0, 1.0,
                    4, PatchEdge3D::UMax, 0.0, 1.0, false);
    connect_feature(surface, 8, PatchEdge3D::VMax, 0.0, 1.0,
                    4, PatchEdge3D::VMax, 0.0, 1.0, true);
    connect_feature(surface, 9, PatchEdge3D::VMax, 0.0, 1.0,
                    5, PatchEdge3D::UMax, 0.0, 1.0, false);
    connect_feature(surface, 10, PatchEdge3D::VMax, 0.0, 1.0,
                    5, PatchEdge3D::VMax, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMax, 0.0, 0.5,
                    5, PatchEdge3D::UMin, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMax, 0.5, 1.0,
                    3, PatchEdge3D::UMin, 0.0, 1.0, true);

    surface.expected_area = 2.0 * 3.0 * arm * arm
                          + 8.0 * arm * (z1 - z0);
    surface.exact_inside = [](const Eigen::Vector3d& x) {
        const double px = x.x() - 0.07;
        const double py = x.y() + 0.07;
        const bool in_plan =
            (px > -0.60 && px < 0.60 && py > -0.60 && py < 0.0)
         || (px > -0.60 && px < 0.0 && py >= 0.0 && py < 0.60);
        return in_plan && x.z() > -0.63 && x.z() < 0.67;
    };
    return surface;
}

} // namespace kfbim::app3d::native_detail
