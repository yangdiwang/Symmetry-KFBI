# 3D Nonconvex Domain Labeling Implementation Plan

> Superseded by the approved NURBS-native geometry design. Do not execute the
> closed-triangle-mesh tasks in this plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace nearest-face-normal labels with closed-surface Cartesian-edge barriers and component flood fill so the L-prism is labeled correctly through `N=128`.

**Architecture:** A new `ClosedSurfaceGeometry3D` owns a welded, validated CGAL mesh built from the complete `crossing_geometry`. It rasterizes triangle/Cartesian-edge intersections into barrier arrays, labels grid connected components globally, and also supplies the segment intersection/closest-face query already needed by `GridPair3D`.

**Tech Stack:** C++17, Eigen, CGAL exact-predicate/inexact-construction kernel, CMake/MSBuild x64, existing `CartesianGrid3D`, `Interface3D`, `GridPair3D`, and native-NURBS 3D tests.

## Global Constraints

- Work directly on `main`; preserve unrelated user changes.
- Use `crossing_geometry`, never the edge-buffered correction interface, as the domain-label source.
- Keep `GridPair3D::domain_label()` and its `0 = exterior`, `component + 1 = interior` contract.
- Keep correction P2 centers for spread, restrict, projection, and correction ownership only.
- Do not change Cauchy stencils, native NURBS DOFs, GMRES tolerances, spread, or restrict.
- Weld only coordinate-coincident vertices within the same interface component using a geometry-scale tolerance independent of `h`.
- Reject a surface passing through a Cartesian node or overlapping a Cartesian edge.
- Do not run `N=256` until the L-prism `N=128` label, crossing, convergence, and order gates pass.

---

### Task 1: Build and validate the closed labeling mesh

**Files:**
- Create: `src/geometry/closed_surface_geometry_3d.hpp`
- Create: `src/geometry/closed_surface_geometry_3d.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: complete `Interface3D crossing_geometry`.
- Produces: `ClosedSurfaceGeometry3D`, `ClosedSurfaceMeshDiagnostics3D`, and a face-to-original-panel-preserving mesh query object.

- [ ] **Step 1: Write the failing welded-mesh test**

Add the new header include and this test shape to
`apps/native_nurbs_surface_3d_test.cpp`:

```cpp
void test_closed_surface_mesh_welding()
{
    constexpr double h = 3.0 / 32.0;
    for (const GeometryKind3D kind : {GeometryKind3D::Torus,
                                      GeometryKind3D::HollowCylinder,
                                      GeometryKind3D::LPrism}) {
        const NativeNurbsSurface3D surface =
            make_native_nurbs_surface_3d(kind);
        kfbim::geometry3d::NurbsPatchTriangulatorOptions3D options;
        options.H_factor = 2.4;
        options.max_edge_over_H = 1.10;
        options.max_depth = 6;
        const auto triangulation =
            kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
                surface.patches, h, options);
        const kfbim::ClosedSurfaceGeometry3D closed(
            triangulation.geometry_interface);
        const auto& diagnostic = closed.mesh_diagnostics();
        require(diagnostic.face_count
                    == triangulation.geometry_interface.num_panels(),
                "closed label mesh retains every geometry panel");
        require(diagnostic.boundary_edge_count == 0,
                "welded label mesh has no boundary edge");
        require(diagnostic.nonmanifold_edge_count == 0,
                "welded label mesh is manifold");
        require(diagnostic.orientation_error_count == 0,
                "welded label mesh has consistent orientation");
    }
}
```

Call it from `main()` before the existing crossing-owner test.

- [ ] **Step 2: Run the test to verify RED**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
```

Expected: compile failure because `closed_surface_geometry_3d.hpp` and
`ClosedSurfaceGeometry3D` do not exist.

- [ ] **Step 3: Add the public geometry-only interface**

Create `closed_surface_geometry_3d.hpp` with these exact public types:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "../grid/cartesian_grid_3d.hpp"
#include "../interface/interface_3d.hpp"

namespace kfbim {

struct ClosedSurfaceMeshDiagnostics3D {
    int input_vertex_count = 0;
    int welded_vertex_count = 0;
    int face_count = 0;
    int boundary_edge_count = 0;
    int nonmanifold_edge_count = 0;
    int orientation_error_count = 0;
};

struct ClosedSurfaceSegmentQuery3D {
    bool exact_intersection = false;
    bool overlap = false;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    int geometry_panel_index = -1;
};

struct ClosedSurfaceLabelDiagnostics3D {
    ClosedSurfaceMeshDiagnostics3D mesh;
    std::array<std::size_t, 3> barrier_edge_counts{{0, 0, 0}};
    int grid_component_count = 0;
    int box_exterior_component_count = 0;
    int representative_query_count = 0;
    std::vector<std::size_t> component_sizes;
};

struct ClosedSurfaceDomainLabels3D {
    std::vector<int> labels;
    ClosedSurfaceLabelDiagnostics3D diagnostics;
};

class ClosedSurfaceGeometry3D {
public:
    explicit ClosedSurfaceGeometry3D(const Interface3D& geometry);
    ~ClosedSurfaceGeometry3D();
    ClosedSurfaceGeometry3D(const ClosedSurfaceGeometry3D&) = delete;
    ClosedSurfaceGeometry3D& operator=(const ClosedSurfaceGeometry3D&) = delete;

    const ClosedSurfaceMeshDiagnostics3D& mesh_diagnostics() const;
    ClosedSurfaceDomainLabels3D label_grid(
        const CartesianGrid3D& grid) const;
    ClosedSurfaceSegmentQuery3D intersect_or_closest(
        const Eigen::Vector3d& a,
        const Eigen::Vector3d& b) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim
```

- [ ] **Step 4: Implement component-aware welding and validation**

In `closed_surface_geometry_3d.cpp`:

- use `CGAL::Exact_predicates_inexact_constructions_kernel` and
  `CGAL::Surface_mesh`;
- compute `weld_tolerance = max(1e-13, 1e-12 * geometry_diameter)`;
- spatially hash vertices by `(component, floor(x/tol), floor(y/tol),
  floor(z/tol))` and search the 27 neighboring buckets before creating a new
  welded vertex;
- reject coordinate-coincident triangles assigned to different components
  with a message containing `coincident faces in distinct components`;
- map every original triangle to three welded IDs and preserve its original
  panel index;
- count undirected edge incidences and directed orientations;
- throw messages containing `boundary edge`, `nonmanifold edge`, or
  `orientation mismatch` when the corresponding count is nonzero;
- add each face to the welded `Surface_mesh` and build one AABB tree for all
  faces plus one closed mesh per `panel_components()` value for side queries.

Use this exact edge invariant before building the trees:

```cpp
struct EdgeUse {
    int count = 0;
    int signed_sum = 0;
};
const auto key = std::minmax(a, b);
EdgeUse& use = edge_uses[{key.first, key.second}];
++use.count;
use.signed_sum += (a < b) ? 1 : -1;
```

Every successful closed edge must have `count == 2` and `signed_sum == 0`.
Reject a degenerate welded triangle before inserting its edges.

Add `geometry/closed_surface_geometry_3d.cpp` to the `KFBIM_BUILD_3D`
`target_sources` list in `src/CMakeLists.txt`.

At this task boundary, define the two operations implemented by later tasks
as explicit stubs so Task 1 links and Task 2 has a deterministic RED state:

```cpp
ClosedSurfaceDomainLabels3D ClosedSurfaceGeometry3D::label_grid(
    const CartesianGrid3D&) const
{
    throw std::logic_error(
        "closed-surface grid labeling is not implemented");
}

ClosedSurfaceSegmentQuery3D ClosedSurfaceGeometry3D::intersect_or_closest(
    const Eigen::Vector3d&, const Eigen::Vector3d&) const
{
    throw std::logic_error(
        "closed-surface segment query is not implemented");
}
```

- [ ] **Step 5: Run GREEN and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git add src/geometry/closed_surface_geometry_3d.hpp src/geometry/closed_surface_geometry_3d.cpp src/CMakeLists.txt apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: build watertight 3d labeling meshes"
```

Expected: `native NURBS model tests passed`.

### Task 2: Construct Cartesian barriers and flood-fill labels

**Files:**
- Modify: `src/geometry/closed_surface_geometry_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: `ClosedSurfaceGeometry3D` from Task 1 and a node-layout `CartesianGrid3D`.
- Produces: `ClosedSurfaceGeometry3D::label_grid()` with labels and barrier/component diagnostics.

- [ ] **Step 1: Add the low-cost N128-aligned L-corner regression**

Add this test. It uses the `N=128` value of `h` but a cropped shifted box, so
it reproduces the failing corner without allocating the full 129-cubed grid.

```cpp
void test_nonconvex_closed_surface_labels()
{
    constexpr double h = 3.0 / 128.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    kfbim::geometry3d::NurbsPatchTriangulatorOptions3D options;
    options.H_factor = 2.4;
    options.max_edge_over_H = 1.10;
    options.max_depth = 6;
    const auto triangulation =
        kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
            surface.patches, h, options);
    kfbim::CartesianGrid3D grid(
        {-0.5625, -0.703125, -0.6328125},
        {h, h, h}, {54, 54, 56}, kfbim::DofLayout3D::Node);
    const kfbim::ClosedSurfaceGeometry3D closed(
        triangulation.geometry_interface);
    const auto result = closed.label_grid(grid);

    require(result.labels[grid.index(27, 27, 4)] > 0,
            "L-prism reentrant lower-arm node must be inside");
    require(result.labels[grid.index(28, 28, 4)] == 0,
            "L-prism missing-quadrant node must be outside");
    require(result.diagnostics.barrier_edge_counts[0]
                + result.diagnostics.barrier_edge_counts[1]
                + result.diagnostics.barrier_edge_counts[2] > 0,
            "closed surface must cut Cartesian edges");
    require(result.diagnostics.grid_component_count >= 2,
            "closed surface must separate grid components");
}
```

Add targeted, away-from-interface checks for both curved geometries. This
grid encloses each complete surface and its shift avoids their analytic seam
coordinates:

```cpp
void test_curved_closed_surface_target_labels()
{
    constexpr double h = 0.05;
    const kfbim::CartesianGrid3D grid(
        {-0.781, -0.841, -0.721}, {h, h, h}, {34, 32, 31},
        kfbim::DofLayout3D::Node);
    for (const GeometryKind3D kind : {GeometryKind3D::Torus,
                                      GeometryKind3D::HollowCylinder}) {
        const NativeNurbsSurface3D surface =
            make_native_nurbs_surface_3d(kind);
        kfbim::geometry3d::NurbsPatchTriangulatorOptions3D options;
        options.H_factor = 2.4;
        options.max_edge_over_H = 1.10;
        options.max_depth = 6;
        const auto triangulation =
            kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
                surface.patches, h, options);
        const kfbim::ClosedSurfaceGeometry3D closed(
            triangulation.geometry_interface);
        const auto result = closed.label_grid(grid);
        const int hole = grid.index(17, 16, 15);
        const int material = kind == GeometryKind3D::Torus
            ? grid.index(28, 16, 15)
            : grid.index(25, 16, 15);
        const auto hole_coord = grid.coord(hole);
        const auto material_coord = grid.coord(material);
        const Eigen::Vector3d hole_point(
            hole_coord[0], hole_coord[1], hole_coord[2]);
        const Eigen::Vector3d material_point(
            material_coord[0], material_coord[1], material_coord[2]);
        require(result.labels[hole] == 0
                    && !surface.exact_inside(hole_point),
                "curved geometry target hole is exterior");
        require(result.labels[material] > 0
                    && surface.exact_inside(material_point),
                "curved geometry target material is interior");
    }
}
```

Add this reusable exception assertion near the existing `require()` helper:

```cpp
template <class Function>
void require_throws_contains(Function&& function,
                             const std::string& needle,
                             const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                message + ": " + error.what());
        return;
    }
    throw std::runtime_error(message + ": no exception");
}
```

Add a deliberately aligned L-prism grid. The reentrant feature point
`(0.07, -0.07, 0)` is node `(13,13,13)`, so the call must fail instead of
silently assigning a side:

```cpp
void test_closed_surface_grid_degeneracy()
{
    constexpr double h = 0.05;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    kfbim::geometry3d::NurbsPatchTriangulatorOptions3D options;
    options.H_factor = 2.4;
    options.max_edge_over_H = 1.10;
    options.max_depth = 6;
    const auto triangulation =
        kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
            surface.patches, h, options);
    const kfbim::ClosedSurfaceGeometry3D closed(
        triangulation.geometry_interface);
    const kfbim::CartesianGrid3D grid(
        {-0.58, -0.72, -0.65}, {h, h, h}, {26, 26, 27},
        kfbim::DofLayout3D::Node);

    require_throws_contains(
        [&] { (void)closed.label_grid(grid); },
        "surface intersects Cartesian node",
        "surface-aligned grid node is rejected");
}
```

Add this exact two-tetrahedra fixture. Each tetrahedron is outward-oriented
and has a different `panel_components()` value:

```cpp
kfbim::Interface3D make_two_tetrahedra_interface()
{
    Eigen::MatrixX3d vertices(8, 3);
    vertices <<
        -0.55, -0.10, -0.10,
        -0.25, -0.10, -0.10,
        -0.55,  0.20, -0.10,
        -0.55, -0.10,  0.20,
         0.25, -0.10, -0.10,
         0.55, -0.10, -0.10,
         0.25,  0.20, -0.10,
         0.25, -0.10,  0.20;
    Eigen::MatrixX3i panels(8, 3);
    panels <<
        0, 2, 1,  0, 1, 3,  0, 3, 2,  1, 2, 3,
        4, 6, 5,  4, 5, 7,  4, 7, 6,  5, 6, 7;
    Eigen::MatrixX3d points(8, 3);
    Eigen::MatrixX3d normals(8, 3);
    Eigen::VectorXd weights(8);
    for (int panel = 0; panel < panels.rows(); ++panel) {
        const Eigen::Vector3d a =
            vertices.row(panels(panel, 0)).transpose();
        const Eigen::Vector3d b =
            vertices.row(panels(panel, 1)).transpose();
        const Eigen::Vector3d c =
            vertices.row(panels(panel, 2)).transpose();
        const Eigen::Vector3d area_normal = (b - a).cross(c - a);
        points.row(panel) = ((a + b + c) / 3.0).transpose();
        normals.row(panel) = area_normal.normalized().transpose();
        weights[panel] = 0.5 * area_normal.norm();
    }
    Eigen::VectorXi components(8);
    components << 0, 0, 0, 0, 1, 1, 1, 1;
    return kfbim::Interface3D(
        vertices, panels, points, normals, weights, 1, components,
        kfbim::PanelNodeLayout3D::Raw);
}

void test_disconnected_closed_surface_labels()
{
    const kfbim::ClosedSurfaceGeometry3D closed(
        make_two_tetrahedra_interface());
    const kfbim::CartesianGrid3D grid(
        {-0.825, -0.375, -0.375}, {0.05, 0.05, 0.05},
        {32, 14, 14}, kfbim::DofLayout3D::Node);
    const auto result = closed.label_grid(grid);

    require(result.labels[grid.index(6, 6, 6)] == 1,
            "first closed component retains label one");
    require(result.labels[grid.index(22, 6, 6)] == 2,
            "second closed component retains label two");
    require(result.labels[grid.index(16, 6, 6)] == 0,
            "space between disconnected components remains exterior");
}
```

Call the nonconvex, curved-target, degeneracy, and disconnected-component
tests from `main()` after the welded-mesh test.

- [ ] **Step 2: Run RED**

Run the focused executable. Expected: failure because Task 1's
`label_grid()` still throws `closed-surface grid labeling is not implemented`.

- [ ] **Step 3: Implement compact barrier arrays**

Inside the `.cpp`, add three `std::vector<std::uint8_t>` arrays with exact
edge indexing:

```cpp
x[(k * ny + j) * (nx - 1) + i];
y[(k * (ny - 1) + j) * nx + i];
z[(k * ny + j) * nx + i];
```

For each original triangle, convert its expanded physical AABB to clamped
index ranges for x-, y-, and z-directed segments. Test only those candidate
segments with `CGAL::intersection(segment, triangle)`.

- a point strictly inside the Cartesian segment marks the edge barrier;
- a point equal to either segment endpoint throws
  `surface intersects Cartesian node (i,j,k) panel=<p>`;
- a segment-valued intersection throws
  `surface overlaps Cartesian edge axis=<x|y|z> (i,j,k) panel=<p>`.

Use exact CGAL predicates for the intersection and a reporting tolerance of
`max(1e-13 * box_diagonal, 1e-14)` only to describe endpoint coincidence; do
not move the surface or grid.

- [ ] **Step 4: Implement grid connected components**

Use an integer component array initialized to `-1` and a vector-backed BFS
queue. Traverse the six Cartesian neighbors only when their connecting edge
barrier is zero. Record component size, representative node, and whether it
touches any of the six box faces.

Classify components exactly as follows:

```cpp
if (touches_box_boundary) {
    component_label = 0;
} else {
    ++diagnostics.representative_query_count;
    component_label = side_of_closed_component(representative_point);
}
```

`side_of_closed_component` queries each validated component mesh. Return
`component + 1` for `CGAL::ON_BOUNDED_SIDE`, continue for
`CGAL::ON_UNBOUNDED_SIDE`, return `0` if no component contains the point, and
throw for `CGAL::ON_BOUNDARY`. If more than one component contains the
representative, throw `overlapping domain components`. Map the per-node graph
component IDs to the resulting domain labels in one final linear pass.

- [ ] **Step 5: Run GREEN, regression tests, and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git add src/geometry/closed_surface_geometry_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: label 3d grids from closed surface topology"
```

Expected: the shifted L-corner assertions pass.

### Task 3: Integrate topology labels and mesh queries into GridPair3D

**Files:**
- Modify: `src/geometry/grid_pair_3d.hpp`
- Modify: `src/geometry/grid_pair_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: `ClosedSurfaceGeometry3D::label_grid()` and `intersect_or_closest()`.
- Produces: unchanged `GridPair3D::domain_label()` behavior with a new read-only `domain_label_diagnostics()` accessor.

- [ ] **Step 1: Add a failing GridPair integration assertion**

Extend `test_nonconvex_closed_surface_labels()`:

```cpp
kfbim::GridPair3D pair(
    grid, triangulation.interface, triangulation.geometry_interface);
require(pair.domain_label(grid.index(27, 27, 4)) > 0,
        "GridPair uses topology-aware L-prism label");
require(pair.domain_label(grid.index(28, 28, 4)) == 0,
        "GridPair preserves missing-quadrant exterior label");
```

Run the test before integration. Expected: the first assertion fails because
the current `GridPair3D` still uses the nearest-center normal.

- [ ] **Step 2: Store the closed geometry and diagnostics in GridPair**

Include the new header from `grid_pair_3d.hpp` and add:

```cpp
const ClosedSurfaceLabelDiagnostics3D& domain_label_diagnostics() const;
```

In `GridPair3D::Impl`, replace the active-P2 `crossing_mesh`, `crossing_tree`,
and `geometry_panel_for_face` ownership with:

```cpp
std::unique_ptr<ClosedSurfaceGeometry3D> closed_geometry;
ClosedSurfaceLabelDiagnostics3D label_diagnostics;
```

Construct it from `crossing_geometry`, call `label_grid(grid)`, move the labels
into `domain_label_vec`, and save the diagnostics. Delete
`build_p2_nearest_center_domain_labels` from the active label path.

Keep the raw non-P2 `Side_of_triangle_mesh` branch unchanged.

- [ ] **Step 3: Route crossing ownership through the shared welded mesh**

Implement `ClosedSurfaceGeometry3D::intersect_or_closest()` so it reproduces
the current behavior:

- return an exact point and original geometry panel for point intersection;
- return the midpoint of an overlap and set `overlap=true`;
- when no segment intersection exists, return the closest mesh point to the
  segment midpoint with `exact_intersection=false`;
- always preserve `geometry_panel_index` through the welded-face property map.

Update `p2_crossing_owner_between()` to map these fields to
`ExactIntersection` or `GapFallback`, then compute the existing geometry
barycentric coordinate and edge parameter. No correction-panel ownership or
NURBS parameter-selection code changes in this step.

- [ ] **Step 4: Strengthen the crossing-owner test**

For every unique crossing edge in `test_grid_edge_triangle_owners()`, require:

```cpp
require(owner.status ==
            kfbim::P2CrossingOwnerStatus3D::ExactIntersection,
        "label-changing Cartesian edge has an exact surface intersection");
```

Run the native test. Expected: every existing torus crossing and the new
L-prism regression pass without a fallback owner.

- [ ] **Step 5: Commit the integration**

```powershell
git diff --check
git add src/geometry/grid_pair_3d.hpp src/geometry/grid_pair_3d.cpp src/geometry/closed_surface_geometry_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "fix: use closed topology for 3d domain labels"
```

### Task 4: Expose readiness diagnostics and CI coverage

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `.github/workflows/build-and-smoke.yml`

**Interfaces:**
- Consumes: `GridPair3D::domain_label_diagnostics()`.
- Produces: console/CSV diagnostics and smoke assertions for topology labeling.

- [ ] **Step 1: Add failing smoke assertions**

After the captured 3D output in the workflow, require:

```bash
printf '%s\n' "$three_d_output" | grep -Fq \
  'domain_label_mode=closed_surface_components'
printf '%s\n' "$three_d_output" | grep -Fq \
  'gap_crossings=0'
```

Run the current executable output check locally and verify RED because neither
diagnostic is printed yet.

- [ ] **Step 2: Add readiness fields and output**

Extend `ReadinessResult` and `geometry_readiness.csv` with:

```text
label_mesh_input_vertices,label_mesh_welded_vertices,label_mesh_faces,
barrier_x,barrier_y,barrier_z,grid_components,box_exterior_components,
representative_queries
```

Print one stable line per level:

```text
domain_label_mode=closed_surface_components barriers=x/y/z components=c/e queries=q gap_crossings=g
```

Populate values only from `grid_pair.domain_label_diagnostics()`; do not
recompute geometry in the app.

- [ ] **Step 3: Verify all three geometries at smoke resolution**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
```

Expected: both formulations converge, every reported gap crossing is zero,
and the new label mode is printed. Curved-surface exact-predicate mismatch
counts remain diagnostic because the flat crossing triangulation approximates
the exact NURBS surface.

- [ ] **Step 4: Commit diagnostics and CI**

```powershell
git diff --check
git add apps/neumann_exterior_zero_trace_3d.cpp .github/workflows/build-and-smoke.yml
git commit -m "test: cover topology-aware 3d labels"
```

### Task 5: Run the L-prism N32/64/128 acceptance study

**Files:**
- Create: `docs/superpowers/results/2026-07-21-3d-nonconvex-domain-labeling.md`
- Outputs: `output/neumann_exterior_zero_trace_3d/same_patch/v32_n20/*.csv`

**Interfaces:**
- Consumes: completed topology labeling and cubic `same_patch 32/20` configuration.
- Produces: acceptance evidence and the decision whether `N=256` is allowed.

- [ ] **Step 1: Run the exact three-level command**

```powershell
$env:KFBIM_3D_CAUCHY_POLICY='same_patch'
$env:KFBIM_3D_CAUCHY_VALUE_COUNT='32'
$env:KFBIM_3D_CAUCHY_NORMAL_COUNT='20'
try {
    & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
        l_prism 32 64 128
} finally {
    Remove-Item Env:KFBIM_3D_CAUCHY_POLICY,
                Env:KFBIM_3D_CAUCHY_VALUE_COUNT,
                Env:KFBIM_3D_CAUCHY_NORMAL_COUNT `
        -ErrorAction SilentlyContinue
}
```

- [ ] **Step 2: Enforce the acceptance gate from CSV**

Use `Import-Csv` to require exactly three rows and assert:

```powershell
$dir='output\neumann_exterior_zero_trace_3d\same_patch\v32_n20'
$g=Import-Csv "$dir\geometry_readiness.csv"
$d=Import-Csv "$dir\dirichlet_normal_results.csv"
if($g.Count -ne 3 -or $d.Count -ne 3) {
    throw 'acceptance CSVs must each contain N=32,64,128'
}
if(($g | Where-Object {[int]$_.label_mismatches -ne 0}).Count) {
    throw 'L-prism label mismatch remains'
}
if(($g | Where-Object {[int]$_.gap_crossings -ne 0}).Count) {
    throw 'L-prism gap crossing remains'
}
if(($d | Where-Object {[int]$_.converged -ne 1}).Count) {
    throw 'Dirichlet solve did not converge'
}
if([double]$d[2].density_order_linf -lt 2.0 -or
   [double]$d[2].interior_order_linf -lt 2.0) {
    throw 'N64-to-N128 order gate failed'
}
if([double]$d[1].density_linf -ge [double]$d[0].density_linf -or
   [double]$d[2].density_linf -ge [double]$d[1].density_linf -or
   [double]$d[1].interior_linf -ge [double]$d[0].interior_linf -or
   [double]$d[2].interior_linf -ge [double]$d[1].interior_linf) {
    throw 'Dirichlet errors did not decrease on both refinements'
}
```

- [ ] **Step 3: Record exact numerical and performance results**

The result document must include:

- mesh/barrier/component diagnostics at every level;
- label mismatch and exact/gap/endpoint crossing counts;
- Dirichlet and Neumann GMRES counts/residuals/runtime;
- density/interior errors and orders;
- comparison with the failed nearest-normal `N=128` result
  (`15` mismatches, `74` gaps, Dirichlet density `6.2212368e-6`, 42 steps);
- an explicit `N=256 allowed: yes/no` decision based only on the acceptance
  gates.

- [ ] **Step 4: Run full verification and request review**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
ctest --test-dir build -C Release --output-on-failure
git diff --check
git status --short --branch
```

Request an independent code review of the complete range. Fix every Critical
or Important finding and rerun the relevant verification.

- [ ] **Step 5: Commit the result document**

```powershell
git add docs/superpowers/results/2026-07-21-3d-nonconvex-domain-labeling.md
git commit -m "test: verify nonconvex 3d domain labels"
```

Do not run `N=256` in this task unless every Task 5 acceptance assertion has
passed.
