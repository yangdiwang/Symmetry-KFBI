# Native NURBS Surface DOFs 3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate every 3D iteration DOF from the same native NURBS patches used by geometry triangulation, and associate Cartesian crossings through native patch parameters and a four-point tensor-grid candidate set.

**Architecture:** Add one app-local native-NURBS geometry module shared by the 3D executable and a focused test executable. The module owns exact torus/cylinder/L-prism patch collections, G1 edge connectivity, uniform parameter-cell-center DOFs, and four-point candidate lookup; `GridPair3D` supplies approximate flat-triangle crossing ownership, while the existing app retains the harmonic Cauchy, spread, restrict, and GMRES algorithms.

**Tech Stack:** C++17, Eigen 3.4, CGAL 5.6, CMake/MSVC x64, existing `NurbsSurfacePatch3D`, `NurbsPatchTriangulation3D`, `GridPair3D`, and `neumann_exterior_zero_trace_3d`.

## Global Constraints

- Work directly on `main`; do not create a worktree.
- Keep the three geometries in the existing 3D app and share only an app-local helper module.
- Represent the torus and hollow cylinder exactly with rational NURBS patches; retain the L-prism's twelve bilinear native patches.
- Use a regular tensor grid in each patch's native parameter domain; divide each parameter interval uniformly and place DOFs only at cell centers.
- Use exactly four logical `2 x 2` candidates for crossing ownership whenever each patch direction has at least two cells.
- Cross a patch edge only through an explicit G1 connection; never cross a non-G1 edge.
- Use triangle intersection and barycentric parameter interpolation only to select candidates, not to evaluate differential traces.
- Preserve the degree-three local harmonic Cauchy polynomial, 48/28 requested samples, tricubic restriction, normal cubic fit, and both existing Krylov equations.
- Verify all three geometries at `N=16,32,64` and compare with the committed baseline in `docs/superpowers/results/2026-07-21-3d-harmonic-jet-results.md`.

---

### Task 1: Native NURBS models for all three geometries

**Files:**
- Create: `apps/native_nurbs_surface_3d.hpp`
- Create: `apps/native_nurbs_surface_3d.cpp`
- Create: `apps/native_nurbs_surface_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Produces: `kfbim::app3d::GeometryKind3D`.
- Produces: `kfbim::app3d::PatchEdge3D` with `UMin`, `UMax`, `VMin`, and `VMax`.
- Produces: `SmoothPatchNeighbor3D { int patch; PatchEdge3D edge; bool reversed; }`.
- Produces: `NativeNurbsSurface3D { name, description, patches, patch_names, smooth_neighbors, expected_area, exact_inside }`.
- Produces: `NativeNurbsSurface3D make_native_nurbs_surface_3d(GeometryKind3D kind)`.

- [ ] **Step 1: Add the test target and write failing native-model tests**

Add this target inside the existing `if(KFBIM_BUILD_3D)` blocks in `apps/CMakeLists.txt`:

```cmake
add_library(kfbim_3d_app_geometry STATIC
    native_nurbs_surface_3d.cpp
)
target_link_libraries(kfbim_3d_app_geometry PUBLIC kfbim_core)
target_compile_features(kfbim_3d_app_geometry PUBLIC cxx_std_17)

add_executable(native_nurbs_surface_3d_test
    native_nurbs_surface_3d_test.cpp
)
target_link_libraries(native_nurbs_surface_3d_test PRIVATE
    kfbim_3d_app_geometry
)
target_compile_features(native_nurbs_surface_3d_test PRIVATE cxx_std_17)
```

Start `apps/native_nurbs_surface_3d_test.cpp` with a small `require(bool,const char*)` harness and these assertions:

```cpp
const auto torus = make_native_nurbs_surface_3d(GeometryKind3D::Torus);
const auto cylinder = make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
const auto lprism = make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
require(torus.patches.size() == 16, "torus must have 4x4 native patches");
require(cylinder.patches.size() == 16,
        "cylinder must have four quarters on each of four sheets");
require(lprism.patches.size() == 12,
        "L-prism must retain its twelve native patches");
require(torus.patch_names.size() == torus.patches.size(), "torus names");
require(cylinder.patch_names.size() == cylinder.patches.size(), "cylinder names");
require(lprism.patch_names.size() == lprism.patches.size(), "L-prism names");
```

For every patch, sample the native parameter midpoint, require finite point and derivatives and require `|du x dv| > 1e-12`. For the torus require the sampled point satisfies its analytic torus equation to `1e-11`; for the cylinder require each sample lies on the named wall/cap; for the L-prism require its exact area is `2*3*0.6*0.6 + 8*0.6*1.3`.

- [ ] **Step 2: Run the new target and verify RED**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test -- /m
```

Expected: compilation fails because `native_nurbs_surface_3d.hpp` and `make_native_nurbs_surface_3d` do not exist yet.

- [ ] **Step 3: Implement exact native patch collections**

Define the public model in `apps/native_nurbs_surface_3d.hpp`:

```cpp
namespace kfbim::app3d {
enum class GeometryKind3D { Torus, HollowCylinder, LPrism };
enum class PatchEdge3D { UMin, UMax, VMin, VMax };
struct SmoothPatchNeighbor3D {
    int patch = -1;
    PatchEdge3D edge = PatchEdge3D::UMin;
    bool reversed = false;
};
struct NativeNurbsSurface3D {
    std::string name;
    std::string description;
    std::vector<geometry3d::NurbsSurfacePatch3D> patches;
    std::vector<std::string> patch_names;
    std::vector<std::array<std::optional<SmoothPatchNeighbor3D>, 4>>
        smooth_neighbors;
    double expected_area = 0.0;
    std::function<bool(const Eigen::Vector3d&)> exact_inside;
};
NativeNurbsSurface3D make_native_nurbs_surface_3d(GeometryKind3D kind);
}
```

In `apps/native_nurbs_surface_3d.cpp`, implement a rational quadratic quarter-circle helper with control points at the two endpoints and tangent intersection and weights `{1,sqrt(0.5),1}`. Build:

```text
torus patch index     = 4*u_quarter + v_quarter, 16 tensor-product patches
cylinder patch 0..3  = outer wall quarters
cylinder patch 4..7  = inner wall quarters with reversed z parameter
cylinder patch 8..11 = top annulus quarters with radial parameter outer->inner
cylinder patch 12..15= bottom annulus quarters with radial parameter inner->outer
L-prism patch 0..2   = bottom rectangles
L-prism patch 3..5   = top rectangles
L-prism patch 6..11  = side rectangles
```

Construct each control net and weight net directly so `du x dv` is the outward normal. Add explicit smooth connections for both wrapped torus directions, each cylinder sheet's quarter seams, and the two coplanar internal seams on each L-prism cap. Do not connect cylinder wall/cap edges or any L-prism geometric edge.

- [ ] **Step 4: Build and run the native-model tests**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test -- /m
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected: `native NURBS model tests passed` and exit code zero.

- [ ] **Step 5: Commit Task 1**

```powershell
git add apps/native_nurbs_surface_3d.hpp apps/native_nurbs_surface_3d.cpp apps/native_nurbs_surface_3d_test.cpp apps/CMakeLists.txt
git commit -m "feat: add native NURBS models for 3d apps"
```

---

### Task 2: Uniform native-parameter surface DOFs

**Files:**
- Modify: `apps/native_nurbs_surface_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: `NativeNurbsSurface3D` from Task 1.
- Produces: `SurfaceDof3D`, including native `patch_id`, `i`, `j`, `u`, and `v`.
- Produces: `SurfaceDofPatch3D`, including `nu`, `nv`, `first_dof`, and directly smooth neighboring patch IDs.
- Produces: `SurfaceDofCloud3D make_native_surface_dofs_3d(const NativeNurbsSurface3D&, double h)`.
- Produces: `int SurfaceDofPatch3D::dof_index(int i, int j) const`.

- [ ] **Step 1: Write failing DOF-generation tests**

Add tests that call `make_native_surface_dofs_3d(surface, 3.0/32.0)` and require:

```cpp
require(cloud.patches.size() == surface.patches.size(), "one DOF patch per NURBS patch");
for (const auto& patch : cloud.patches) {
    require(patch.nu >= 2 && patch.nv >= 2, "each tensor direction needs two cells");
    require(patch.dof_count() == patch.nu * patch.nv, "dense tensor indexing");
}
for (const auto& q : cloud.dofs) {
    const auto& patch = surface.patches.at(q.patch_id);
    require(q.u > patch.domain_start_u() && q.u < patch.domain_end_u(), "u center");
    require(q.v > patch.domain_start_v() && q.v < patch.domain_end_v(), "v center");
    require((q.point - patch.evaluate(q.u, q.v)).norm() < 1e-13, "native point");
    require(std::abs(q.normal.norm() - 1.0) < 1e-13, "unit normal");
    require(q.weight > 0.0 && std::isfinite(q.weight), "positive weight");
}
```

Require relative midpoint-area errors below `3e-2` at `N=32` and require the error to decrease from `N=16` to `N=32` for all three geometries. Check outward orientation with `exact_inside(q.point - 0.1*h*q.normal)` true and `exact_inside(q.point + 0.1*h*q.normal)` false.

- [ ] **Step 2: Run and verify RED**

Run the test target. Expected: compilation fails because `SurfaceDofCloud3D` and `make_native_surface_dofs_3d` are missing.

- [ ] **Step 3: Implement uniform tensor DOFs**

Add these data members:

```cpp
struct SurfaceDof3D {
    Eigen::Vector3d point, normal, tangent1, tangent2;
    double weight = 0.0;
    double u = 0.0, v = 0.0;
    int patch_id = -1, i = -1, j = -1;
};
struct SurfaceDofPatch3D {
    std::string name;
    int nu = 0, nv = 0, first_dof = 0;
    std::vector<int> smooth_patch_ids;
    int dof_index(int i, int j) const { return first_dof + i * nv + j; }
    int dof_count() const { return nu * nv; }
};
struct SurfaceDofCloud3D {
    std::vector<SurfaceDofPatch3D> patches;
    std::vector<SurfaceDof3D> dofs;
    double expected_area = 0.0;
};
```

Estimate each direction's physical length by sampling five transverse
isolines and using sixteen straight subsegments per isoline. Set
`nu=max(2,ceil(max_u_length/h))` and `nv=max(2,ceil(max_v_length/h))`, then
divide the native intervals uniformly. At each midpoint evaluate
`S`, `S_u`, `S_v`, use the outward `S_u x S_v`, build an orthonormal tangent
frame, and set `weight=|S_u x S_v|*delta_u*delta_v`.

- [ ] **Step 4: Run the test target and verify GREEN**

Expected: all native-model and uniform-DOF tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add apps/native_nurbs_surface_3d.hpp apps/native_nurbs_surface_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: generate 3d DOFs from native NURBS parameters"
```

---

### Task 3: Four-candidate parameter lookup across G1 seams

**Files:**
- Modify: `apps/native_nurbs_surface_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: native patch edge connections and dense tensor DOF indexing.
- Produces: `std::array<int,4> parameter_dof_candidates_2x2(const NativeNurbsSurface3D&, const SurfaceDofCloud3D&, int patch, double u, double v)`.
- Produces: `Eigen::Vector2d interpolate_triangle_parameter(const geometry3d::NurbsParamTriangle3D&, const Eigen::Vector3d& barycentric)`.
- Produces: `std::vector<int> smooth_patch_component(const NativeNurbsSurface3D&, int patch)` for Cauchy-topology expansion and tests.

- [ ] **Step 1: Write failing candidate tests**

Cover these behaviors:

```cpp
const auto interior = parameter_dof_candidates_2x2(torus, cloud, 0, 0.5, 0.5);
require(four_unique_valid_ids(interior, cloud), "interior query has four DOFs");

const double near_umin = torus.patches[0].domain_start_u() + 0.1*delta_u;
const auto periodic = parameter_dof_candidates_2x2(torus, cloud, 0, near_umin, 0.5);
require(contains_patch(periodic, cloud, 12), "torus closed seam reaches previous u quarter");

const auto lsharp = parameter_dof_candidates_2x2(lprism, lcloud, 3,
                                                  lpatch.domain_start_u()+0.1*du,
                                                  0.5);
require(all_on_patch(lsharp, lcloud, 3), "non-G1 edge remains on source patch");
```

Also test a smooth L-cap seam, a reversed edge-parameter mapping, and barycentric interpolation with known triangle UV values.

- [ ] **Step 2: Run and verify RED**

Expected: compilation fails because both lookup functions are missing.

- [ ] **Step 3: Implement glued four-point lookup**

For each parameter direction compute the two bracketing cell-center lattice
indices using `floor((parameter-start)/delta - 0.5)` and its successor. For
each of the four tensor combinations:

```text
inside source patch     -> direct dense-grid index
outside through G1 edge -> map to the neighbor boundary row/column
reversed G1 edge        -> reverse the normalized along-edge coordinate
outside through sharp edge -> replace by the next inward source index
```

Apply at most two edge mappings so a smooth four-patch corner reaches its
diagonal patch. Deduplicate only as a safety check and throw unless four unique
DOFs remain; every patch is guaranteed to have at least two cells per
direction. Implement parameter interpolation as the weighted sum of the three
triangle UV vertices. Implement `smooth_patch_component` as a deterministic
breadth-first walk over the explicit smooth edge connections.

- [ ] **Step 4: Run the test target and verify GREEN**

Expected: interior, periodic, smooth-seam, reversed-seam, sharp-edge, and UV interpolation tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add apps/native_nurbs_surface_3d.hpp apps/native_nurbs_surface_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: locate 3d surface DOFs through patch parameters"
```

---

### Task 4: Approximate Cartesian-edge/triangle crossings in GridPair3D

**Files:**
- Modify: `src/geometry/grid_pair_3d.cpp`
- Modify: `src/geometry/grid_pair_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Preserves: `P2CrossingOwner3D GridPair3D::p2_crossing_owner_between(int,int) const`.
- Changes: populate `geometry_panel_index`, `geometry_barycentric`, and `edge_parameter` from the flat corner triangle of `crossing_geometry()`.
- Uses: `ExactIntersection` for a segment/triangle hit and `GapFallback` for nearest-triangle projection; reserve `EndpointNearestCenter` only for an unavailable geometry tree.

- [ ] **Step 1: Write a failing crossing-owner test**

Triangulate the native torus at `h=3.0/16.0`, construct a `16^3` node grid and
`GridPair3D`, build `LaplaceCorrectionSupport3D`, select every unique crossing
edge, and require:

```cpp
const auto owner = pair.p2_crossing_owner_between(a, b);
require(owner.status != P2CrossingOwnerStatus3D::EndpointNearestCenter,
        "closed NURBS geometry must provide a triangle owner");
require(owner.geometry_panel_index >= 0
        && owner.geometry_panel_index < triangulation.geometry_interface.num_panels(),
        "geometry panel index");
require(std::abs(owner.geometry_barycentric.sum() - 1.0) < 1e-12,
        "triangle barycentric partition");
require(owner.edge_parameter >= 0.0 && owner.edge_parameter <= 1.0,
        "grid-edge phase");
```

- [ ] **Step 2: Run and verify RED**

Expected: the test runs but fails because the current implementation always reports `EndpointNearestCenter` and leaves geometry fields unset.

- [ ] **Step 3: Add the flat-triangle AABB tree**

In `GridPair3D::Impl`, retain a CGAL surface mesh built from
`crossing_geometry.vertices()` and `crossing_geometry.panels()` and an
`AABB_tree<AABB_traits_3<K3,AABB_face_graph_triangle_primitive<CMesh3>>>`.
Build it once for active P2 geometry. Keep face insertion order aligned with
the `Interface3D` panel order.

For a query segment, use the first point intersection, compute its phase on
the Cartesian edge and its barycentric coordinates on that flat triangle, and
return `ExactIntersection`. If the finite segment misses because the P2 label
and flat tessellation differ slightly, use the tree's closest point to the
segment midpoint, project its phase to `[0,1]`, compute barycentric coordinates,
and return `GapFallback`. Preserve the existing nearest correction expansion
center in `center_index`, `panel_index`, and `barycentric`.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the native test executable. Expected: all torus crossing edges have a
geometry panel owner, valid barycentric coordinates, and no endpoint fallback.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/geometry/grid_pair_3d.cpp src/geometry/grid_pair_3d.hpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: expose approximate 3d triangle crossings"
```

---

### Task 5: Integrate native patches, DOFs, candidates, and smooth Cauchy topology

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: all native geometry/DOF APIs from Tasks 1-3.
- Consumes: geometry triangle ownership from Task 4.
- Changes: `GeometryBundle` retains the native model plus correction and full-geometry `NurbsParamTriangle3D` arrays.
- Changes: `PanelCenterHarmonicJetKFBI3D` receives these arrays and selects each crossing center from exactly four parameter candidates.

- [ ] **Step 1: Add failing topology assertions**

In the focused test, require that each L-prism side patch's smooth reachable
component contains only itself, each top patch's component contains exactly
the three top patches, and each bottom patch's component contains exactly the
three bottom patches. Require the torus smooth component to contain all sixteen
patches and each cylinder component to contain exactly one four-patch sheet.

Run the existing app before integration and record that it reports `1/4/8`
surface patches and nonzero endpoint owners; this is the expected RED behavior
relative to the new `16/16/12` requirement.

- [ ] **Step 2: Replace geometry-specific analytic builders**

Remove `make_torus_surface_dofs`, `make_cylinder_surface_dofs`,
`make_l_prism_surface_dofs`, the hand-built torus/cylinder P2 interfaces, and
the duplicate L-prism patch builder from
`apps/neumann_exterior_zero_trace_3d.cpp`. Construct one
`NativeNurbsSurface3D`, pass its `patches` to
`triangulate_nurbs_surface_patches_3d` for every geometry, retain both triangle
metadata arrays, and call `make_native_surface_dofs_3d(model,h)`.

- [ ] **Step 3: Restrict Cauchy topology to G1 components**

Replace the old manually broad `adjacent_patch_ids` with a breadth-first walk
over `smooth_neighbors`. Candidate patch expansion stops as soon as it contains
enough samples for the requested 48/28 stencil; it can never leave the smooth
connected component. Keep final ranking by three-dimensional Euclidean
distance and keep the center DOF mandatory.

- [ ] **Step 4: Replace global crossing-to-DOF scans**

Pass `NativeNurbsSurface3D`, correction triangles, and geometry triangles to
`PanelCenterHarmonicJetKFBI3D`. In `build_crossing_rows()`:

```cpp
const auto& tri = owner.geometry_panel_index >= 0
    ? geometry_triangles.at(owner.geometry_panel_index)
    : correction_triangles.at(owner.panel_index);
const Eigen::Vector3d bary = owner.geometry_panel_index >= 0
    ? owner.geometry_barycentric : owner.barycentric;
const Eigen::Vector2d uv = interpolate_triangle_parameter(tri, bary);
const auto candidates = parameter_dof_candidates_2x2(
    native_surface, cloud, tri.patch_index, uv.x(), uv.y());
```

Choose the final center from those four IDs by physical distance to the
approximate crossing, rejecting candidates whose normal dot the native normal
at `(u,v)` is below `0.5`. Delete the full-cloud `nearest_surface_dof` scan.

- [ ] **Step 5: Build and run the `N=16` integration test**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d native_nurbs_surface_3d_test -- /m
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
```

Expected:

```text
torus surface patches = 16
cylinder surface patches = 16
l_prism surface patches = 12
owners endpoint = 0
all six GMRES solves converged
```

Also require finite positive areas, no label failure, finite Cauchy condition
data, and no local Cauchy stencil containing patches from different G1
components.

- [ ] **Step 6: Commit Task 5**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/CMakeLists.txt apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: use native NURBS DOFs in 3d KFBI app"
```

---

### Task 6: Full convergence verification and result comparison

**Files:**
- Modify: `docs/superpowers/results/2026-07-21-3d-harmonic-jet-results.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: runtime CSV files in `output/neumann_exterior_zero_trace_3d/`.
- Produces: updated Neumann/Dirichlet errors, observed orders, iteration counts, residuals, native patch counts, and comparison with commit `984f5b1` baseline values.

- [ ] **Step 1: Run focused verification from a clean build target**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d -- /m
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
```

Expected: all tests and all `N=16` solves pass.

- [ ] **Step 2: Run the complete `16/32/64` study**

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16 32 64
```

Expected: all eighteen solves converge and both result CSV files contain nine
rows with finite errors and observed orders for `N=32,64`.

- [ ] **Step 3: Check numerical invariants**

Use PowerShell `Import-Csv` to require:

```text
geometry_readiness.csv: surface_patches are torus=16, cylinder=16, l_prism=12
geometry_readiness.csv: endpoint_crossings=0 for every row
neumann_results.csv: converged=1 and gmres_relative_residual <= 2.1e-10
dirichlet_normal_results.csv: converged=1 and gmres_relative_residual <= 2.1e-10
all reported errors and residuals are finite
```

Inspect the `N=16 -> 32` and `N=32 -> 64` orders instead of imposing an
unjustified minimum: this geometry/ownership change is being measured, and any
order regression must be diagnosed before completion.

- [ ] **Step 4: Update result documentation**

Replace the two tables in the result document with the new CSV values. Add a
short comparison table containing, for each geometry/formulation at `N=64`,
baseline error, native-NURBS error, baseline iterations, and native-NURBS
iterations. Update README's 3D description to say that geometry triangulation
and iteration DOFs share the same native NURBS patches and that crossing
association uses barycentric UV plus four tensor candidates.

- [ ] **Step 5: Run final verification and inspect the diff**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d -- /m
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16 32 64
git diff --check
git status --short
```

Expected: builds and tests pass, all eighteen solves converge, `git diff --check`
is silent, and only intended source/document/result changes are present.

- [ ] **Step 6: Commit Task 6**

```powershell
git add README.md docs/superpowers/results/2026-07-21-3d-harmonic-jet-results.md
git commit -m "test: record native NURBS 3d convergence"
```
