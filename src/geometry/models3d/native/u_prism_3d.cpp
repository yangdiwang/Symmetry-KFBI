#include "construction.hpp"
#include "models_3d.hpp"

#include <array>

namespace kfbim::app3d::native_detail {

NativeNurbsSurface3D make_u_prism_3d()
{
    // Five congruent cells form a planar U: three cells on the bottom row
    // and one additional cell above each outer column.  Keeping the caps
    // split by cell exposes their true G1 topology while the eight vertical
    // walls remain separate sheets across the physical C0 creases.
    constexpr double sx = 0.07;
    constexpr double sy = -0.07;
    constexpr double z0 = -0.63;
    constexpr double z1 = 0.67;
    constexpr double cell_size = 0.36;
    constexpr double x0 = sx - 1.5 * cell_size;
    constexpr double x1 = sx - 0.5 * cell_size;
    constexpr double x2 = sx + 0.5 * cell_size;
    constexpr double x3 = sx + 1.5 * cell_size;
    constexpr double y0 = sy - cell_size;
    constexpr double y1 = sy;
    constexpr double y2 = sy + cell_size;
    const std::array<std::array<double, 4>, 5> cells{{
        {{x0, x1, y0, y1}},
        {{x1, x2, y0, y1}},
        {{x2, x3, y0, y1}},
        {{x0, x1, y1, y2}},
        {{x2, x3, y1, y2}}}};

    NativeNurbsSurface3D surface;
    surface.name = "u_prism";
    surface.description =
        "eighteen native bilinear NURBS U-prism patches";
    for (int cell = 0; cell < 5; ++cell) {
        const auto& bounds = cells[static_cast<std::size_t>(cell)];
        append_prism_bottom(surface,
                            "bottom_cell_" + std::to_string(cell),
                            bounds[0], bounds[1], bounds[2], bounds[3], z0);
    }
    for (int cell = 0; cell < 5; ++cell) {
        const auto& bounds = cells[static_cast<std::size_t>(cell)];
        append_prism_top(surface,
                         "top_cell_" + std::to_string(cell),
                         bounds[0], bounds[1], bounds[2], bounds[3], z1);
    }

    // Counter-clockwise boundary as viewed from +z.  This ordering and the
    // (boundary,z) side parameterization give every wall its outward normal.
    const std::array<Eigen::Vector2d, 8> boundary{{
        {x0, y0}, {x3, y0}, {x3, y2}, {x2, y2},
        {x2, y1}, {x1, y1}, {x1, y2}, {x0, y2}}};
    for (int side = 0; side < 8; ++side) {
        const Eigen::Vector2d a = boundary[static_cast<std::size_t>(side)];
        const Eigen::Vector2d b =
            boundary[static_cast<std::size_t>((side + 1) % 8)];
        append_patch(surface,
                     "side_" + std::to_string(side),
                     geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
                         {a.x(), a.y(), z0}, {b.x(), b.y(), z0},
                         {a.x(), a.y(), z1}, {b.x(), b.y(), z1}));
    }

    // Smooth cell-to-cell cap joins.  Bottom caps use (u,v)=(y,x), whereas
    // top caps use (u,v)=(x,y), following the orientation convention shared
    // with the established L-prism geometry.
    connect_smooth(surface, 0, PatchEdge3D::VMax,
                   1, PatchEdge3D::VMin, false);
    connect_smooth(surface, 1, PatchEdge3D::VMax,
                   2, PatchEdge3D::VMin, false);
    connect_smooth(surface, 0, PatchEdge3D::UMax,
                   3, PatchEdge3D::UMin, false);
    connect_smooth(surface, 2, PatchEdge3D::UMax,
                   4, PatchEdge3D::UMin, false);
    connect_smooth(surface, 5, PatchEdge3D::UMax,
                   6, PatchEdge3D::UMin, false);
    connect_smooth(surface, 6, PatchEdge3D::UMax,
                   7, PatchEdge3D::UMin, false);
    connect_smooth(surface, 5, PatchEdge3D::VMax,
                   8, PatchEdge3D::VMin, false);
    connect_smooth(surface, 7, PatchEdge3D::VMax,
                   9, PatchEdge3D::VMin, false);

    for (int side = 0; side < 8; ++side) {
        const int side_patch = 10 + side;
        connect_feature(surface,
            side_patch, PatchEdge3D::UMax, 0.0, 1.0,
            10 + (side + 1) % 8, PatchEdge3D::UMin, 0.0, 1.0, false);
    }

    // Bottom cap to wall connections.  Long outer sides are partitioned so
    // every interval is covered exactly once without inventing cap patches.
    for (int column = 0; column < 3; ++column) {
        connect_feature(surface,
            10, PatchEdge3D::VMin,
            static_cast<double>(column) / 3.0,
            static_cast<double>(column + 1) / 3.0,
            column, PatchEdge3D::UMin, 0.0, 1.0, false);
    }
    connect_feature(surface, 11, PatchEdge3D::VMin, 0.0, 0.5,
                    2, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface, 11, PatchEdge3D::VMin, 0.5, 1.0,
                    4, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface, 12, PatchEdge3D::VMin, 0.0, 1.0,
                    4, PatchEdge3D::UMax, 0.0, 1.0, true);
    connect_feature(surface, 13, PatchEdge3D::VMin, 0.0, 1.0,
                    4, PatchEdge3D::VMin, 0.0, 1.0, true);
    connect_feature(surface, 14, PatchEdge3D::VMin, 0.0, 1.0,
                    1, PatchEdge3D::UMax, 0.0, 1.0, true);
    connect_feature(surface, 15, PatchEdge3D::VMin, 0.0, 1.0,
                    3, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface, 16, PatchEdge3D::VMin, 0.0, 1.0,
                    3, PatchEdge3D::UMax, 0.0, 1.0, true);
    connect_feature(surface, 17, PatchEdge3D::VMin, 0.0, 0.5,
                    3, PatchEdge3D::VMin, 0.0, 1.0, true);
    connect_feature(surface, 17, PatchEdge3D::VMin, 0.5, 1.0,
                    0, PatchEdge3D::VMin, 0.0, 1.0, true);

    // Top cap to wall connections use the top-cap parameter orientation.
    for (int column = 0; column < 3; ++column) {
        connect_feature(surface,
            10, PatchEdge3D::VMax,
            static_cast<double>(column) / 3.0,
            static_cast<double>(column + 1) / 3.0,
            5 + column, PatchEdge3D::VMin, 0.0, 1.0, false);
    }
    connect_feature(surface, 11, PatchEdge3D::VMax, 0.0, 0.5,
                    7, PatchEdge3D::UMax, 0.0, 1.0, false);
    connect_feature(surface, 11, PatchEdge3D::VMax, 0.5, 1.0,
                    9, PatchEdge3D::UMax, 0.0, 1.0, false);
    connect_feature(surface, 12, PatchEdge3D::VMax, 0.0, 1.0,
                    9, PatchEdge3D::VMax, 0.0, 1.0, true);
    connect_feature(surface, 13, PatchEdge3D::VMax, 0.0, 1.0,
                    9, PatchEdge3D::UMin, 0.0, 1.0, true);
    connect_feature(surface, 14, PatchEdge3D::VMax, 0.0, 1.0,
                    6, PatchEdge3D::VMax, 0.0, 1.0, true);
    connect_feature(surface, 15, PatchEdge3D::VMax, 0.0, 1.0,
                    8, PatchEdge3D::UMax, 0.0, 1.0, false);
    connect_feature(surface, 16, PatchEdge3D::VMax, 0.0, 1.0,
                    8, PatchEdge3D::VMax, 0.0, 1.0, true);
    connect_feature(surface, 17, PatchEdge3D::VMax, 0.0, 0.5,
                    8, PatchEdge3D::UMin, 0.0, 1.0, true);
    connect_feature(surface, 17, PatchEdge3D::VMax, 0.5, 1.0,
                    5, PatchEdge3D::UMin, 0.0, 1.0, true);

    const double height = z1 - z0;
    surface.expected_area = 2.0 * 5.0 * cell_size * cell_size
                          + 12.0 * cell_size * height;
    surface.exact_inside = [=](const Eigen::Vector3d& x) {
        const bool in_bottom =
            x.x() > x0 && x.x() < x3 && x.y() > y0 && x.y() < y1;
        const bool in_left_arm =
            x.x() > x0 && x.x() < x1 && x.y() >= y1 && x.y() < y2;
        const bool in_right_arm =
            x.x() > x2 && x.x() < x3 && x.y() >= y1 && x.y() < y2;
        return (in_bottom || in_left_arm || in_right_arm)
            && x.z() > z0 && x.z() < z1;
    };
    return surface;
}

} // namespace kfbim::app3d::native_detail
