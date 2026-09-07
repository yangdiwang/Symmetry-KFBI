#include "src/support/geometry/native_nurbs_surface_3d.hpp"
#include "src/support/trace/restrict_crossing_selector_3d.hpp"

#include "src/geometry/nurbs_surface_intersector_3d.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::CartesianGridlineCrossingRecord3D;
using kfbim::app3d::G1PatchTopology3D;
using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::GridlineCrossingSelectionKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::PatchEdge3D;
using kfbim::app3d::SmoothPatchNeighbor3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;
using kfbim::app3d::nearest_g1_cauchy_dofs;
using kfbim::app3d::parameter_dof_candidates_2x2;
using kfbim::app3d::select_nearest_gridline_crossing_on_g1_sheet_3d;
using kfbim::app3d::smooth_patch_component;
using kfbim::app3d::smooth_patch_neighbors_3d;
using kfbim::geometry3d::NurbsPatchEdgeConnection3D;
using kfbim::geometry3d::NurbsSurfaceCrossing3D;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class IdRange>
bool contains_patch(const IdRange& ids,
                    const SurfaceDofCloud3D& cloud,
                    int patch)
{
    return std::any_of(ids.begin(), ids.end(), [&](int id) {
        return id >= 0 && id < static_cast<int>(cloud.dofs.size())
            && cloud.dofs[static_cast<std::size_t>(id)].patch_id == patch;
    });
}

NativeNurbsSurface3D make_smooth_t_dirichlet_sheet()
{
    NativeNurbsSurface3D surface;
    surface.name = "smooth_t_dirichlet_sheet";
    surface.description =
        "one long G1 edge joined by two short edge intervals";
    surface.patches = {
        kfbim::geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
            {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
            {-1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}),
        kfbim::geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}),
        kfbim::geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0})};
    surface.patch_names = {"long", "short_lower", "short_upper"};

    // Deliberately leave smooth_neighbors empty.  Its fixed one-slot edge
    // representation cannot retain both children of patch 0's UMax edge.
    surface.topological_patch_neighbors = {{1, 2}, {0, 2}, {0, 1}};
    surface.geometric_connections = {
        NurbsPatchEdgeConnection3D{
            {0, PatchEdge3D::UMax, 0.0, 0.5},
            {1, PatchEdge3D::UMin, 0.0, 1.0}, false, true},
        NurbsPatchEdgeConnection3D{
            {0, PatchEdge3D::UMax, 0.5, 1.0},
            {2, PatchEdge3D::UMin, 0.0, 1.0}, false, true},
        NurbsPatchEdgeConnection3D{
            {1, PatchEdge3D::VMax, 0.0, 1.0},
            {2, PatchEdge3D::VMin, 0.0, 1.0}, false, true}};
    surface.patch_components = {0, 0, 0};
    surface.expected_area = 2.0;
    surface.exact_inside = [](const Eigen::Vector3d&) { return false; };
    return surface;
}

G1PatchTopology3D shared_q10_topology(const NativeNurbsSurface3D& surface)
{
    G1PatchTopology3D result;
    result.surface_component_by_patch = surface.patch_components;
    result.g1_neighbors.resize(surface.patches.size());
    for (int patch = 0;
         patch < static_cast<int>(surface.patches.size()); ++patch) {
        result.g1_neighbors[static_cast<std::size_t>(patch)] =
            smooth_patch_neighbors_3d(surface, patch);
    }
    return result;
}

void test_smooth_t_dirichlet_sheet()
{
    const NativeNurbsSurface3D surface = make_smooth_t_dirichlet_sheet();
    require(surface.smooth_neighbors.empty(),
            "smooth-T fixture must not rely on the legacy edge table");
    require(smooth_patch_neighbors_3d(surface, 0)
                == std::vector<int>({1, 2}),
            "long edge retains both G1 interval neighbors");
    require(smooth_patch_neighbors_3d(surface, 1)
                == std::vector<int>({0, 2}),
            "short child retains its long-edge and sibling neighbors");
    require(smooth_patch_component(surface, 0)
                == std::vector<int>({0, 1, 2}),
            "smooth-T patches form one Dirichlet sheet");

    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, 0.2);
    for (const auto& patch : cloud.patches) {
        require(patch.smooth_patch_ids == std::vector<int>({0, 1, 2}),
                "each tensor patch records the complete G1 sheet");
    }

    const std::array<int, 4> lower = parameter_dof_candidates_2x2(
        surface, cloud, 0, 0.99, 0.25);
    require(contains_patch(lower, cloud, 1)
                && !contains_patch(lower, cloud, 2),
            "lower long-edge interval crosses only to the lower child");
    const std::array<int, 4> upper = parameter_dof_candidates_2x2(
        surface, cloud, 0, 0.99, 0.75);
    require(contains_patch(upper, cloud, 2)
                && !contains_patch(upper, cloud, 1),
            "upper long-edge interval crosses only to the upper child");

    const auto& long_patch = cloud.patches[0];
    const int center = long_patch.dof_index(
        long_patch.nu - 1, long_patch.nv / 2);
    const std::vector<int> cauchy = nearest_g1_cauchy_dofs(
        surface, cloud, center, static_cast<int>(cloud.dofs.size()));
    require(contains_patch(cauchy, cloud, 1)
                && contains_patch(cauchy, cloud, 2),
            "nearest-G1 Cauchy BFS retains both smooth-T children");

    NurbsSurfaceCrossing3D root;
    root.patch_index = 0;
    root.component = 0;
    root.point = {0.0, 0.75, 0.0};
    root.transversality = 1.0;
    root.reliable_transversality_tolerance = 1.0e-8;
    const std::vector<CartesianGridlineCrossingRecord3D> records = {
        {10, 1, 0, {0.0, 0.1, 0.0}},
        {20, 2, 0, {0.0, 0.75, 0.0}}};
    const auto selected = select_nearest_gridline_crossing_on_g1_sheet_3d(
        root, shared_q10_topology(surface), records, 1.0e-12);
    require(selected.kind == GridlineCrossingSelectionKind3D::Selected
                && selected.payload_index == 20
                && selected.same_g1_sheet_count == 2,
            "shared-Q10 G1 search sees both children of the long edge");
}

void test_legacy_full_edge_fallback()
{
    // Existing production geometries still populate the fixed table.  Their
    // full-edge topology must remain bit-for-bit representable.
    NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    require(!torus.smooth_neighbors.empty(),
            "legacy production topology remains populated");
    torus.geometric_connections.clear();
    const std::vector<int> legacy_neighbors =
        smooth_patch_neighbors_3d(torus, 0);
    require(legacy_neighbors.size() == 4,
            "legacy-only full-edge torus retains four smooth neighbors");
    require(smooth_patch_component(torus, 0).size() == 16,
            "legacy-only full-edge topology retains its smooth sheet");
}

} // namespace

int main()
{
    try {
        test_smooth_t_dirichlet_sheet();
        test_legacy_full_edge_fallback();
        std::cout << "smooth-T topology tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "smooth-T topology test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
