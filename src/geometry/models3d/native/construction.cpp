#include "construction.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d::native_detail {

namespace {

int edge_index(PatchEdge3D edge)
{
    return static_cast<int>(edge);
}

void connect_topological_patches(NativeNurbsSurface3D& surface,
                                 int patch_a,
                                 int patch_b)
{
    if (patch_a < 0 || patch_a >= static_cast<int>(surface.patches.size())
        || patch_b < 0 || patch_b >= static_cast<int>(surface.patches.size())
        || patch_a == patch_b) {
        throw std::out_of_range(
            "topological NURBS connection has invalid patch");
    }
    auto insert_unique = [](std::vector<int>& neighbors, int patch) {
        if (std::find(neighbors.begin(), neighbors.end(), patch)
            == neighbors.end()) {
            neighbors.push_back(patch);
            std::sort(neighbors.begin(), neighbors.end());
        }
    };
    insert_unique(
        surface.topological_patch_neighbors[static_cast<std::size_t>(patch_a)],
        patch_b);
    insert_unique(
        surface.topological_patch_neighbors[static_cast<std::size_t>(patch_b)],
        patch_a);
}

} // namespace

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

geometry::NurbsBasis1D quadratic_unit_basis()
{
    return geometry::NurbsBasis1D(
        2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
}

geometry::NurbsBasis1D linear_unit_basis()
{
    return geometry::NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0});
}

void append_patch(NativeNurbsSurface3D& surface,
                  std::string name,
                  geometry3d::NurbsSurfacePatch3D patch)
{
    surface.patch_names.push_back(std::move(name));
    surface.patches.push_back(std::move(patch));
    surface.smooth_neighbors.emplace_back();
    surface.topological_patch_neighbors.emplace_back();
    surface.patch_components.push_back(0);
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
    connect_topological_patches(surface, patch_a, patch_b);
    auto& slot_a = surface.smooth_neighbors[static_cast<std::size_t>(patch_a)]
                                           [static_cast<std::size_t>(edge_index(edge_a))];
    auto& slot_b = surface.smooth_neighbors[static_cast<std::size_t>(patch_b)]
                                           [static_cast<std::size_t>(edge_index(edge_b))];
    if (slot_a || slot_b)
        throw std::runtime_error("NURBS patch edge has multiple smooth neighbors");
    slot_a = SmoothPatchNeighbor3D{patch_b, edge_b, reversed};
    slot_b = SmoothPatchNeighbor3D{patch_a, edge_a, reversed};
    surface.geometric_connections.push_back({
        geometry3d::full_patch_edge_interval(
            surface.patches[static_cast<std::size_t>(patch_a)], patch_a, edge_a),
        geometry3d::full_patch_edge_interval(
            surface.patches[static_cast<std::size_t>(patch_b)], patch_b, edge_b),
        reversed,
        true});
}

void connect_feature(NativeNurbsSurface3D& surface,
                     int patch_a,
                     PatchEdge3D edge_a,
                     double begin_a,
                     double end_a,
                     int patch_b,
                     PatchEdge3D edge_b,
                     double begin_b,
                     double end_b,
                     bool reversed)
{
    connect_topological_patches(surface, patch_a, patch_b);
    surface.geometric_connections.push_back({
        {patch_a, edge_a, begin_a, end_a},
        {patch_b, edge_b, begin_b, end_b},
        reversed,
        false});
}

void append_prism_top(NativeNurbsSurface3D& surface,
                      const std::string& name,
                      double xmin,
                      double xmax,
                      double ymin,
                      double ymax,
                      double z)
{
    append_patch(surface, name,
                 geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
                     {xmin, ymin, z}, {xmax, ymin, z},
                     {xmin, ymax, z}, {xmax, ymax, z}));
}

void append_prism_bottom(NativeNurbsSurface3D& surface,
                         const std::string& name,
                         double xmin,
                         double xmax,
                         double ymin,
                         double ymax,
                         double z)
{
    append_patch(surface, name,
                 geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
                     {xmin, ymin, z}, {xmin, ymax, z},
                     {xmax, ymin, z}, {xmax, ymax, z}));
}

} // namespace kfbim::app3d::native_detail
