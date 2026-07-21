# Separate 3D Crossing and Cauchy Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep crossing-to-DOF ownership confined to G1 seams while allowing each local Cauchy fit to collect its 48/28 samples across topologically adjacent non-G1 patch edges.

**Architecture:** `NativeNurbsSurface3D` stores a complete patch-edge adjacency graph in addition to its existing G1 graph. A reusable Cauchy selector expands the complete graph breadth-first by whole topological rings, stops once at least 48 DOFs are available, and then ranks only that pool by physical distance. The existing `2x2` parameter candidate function continues to use only `smooth_neighbors`.

**Tech Stack:** C++17, Eigen, native NURBS patches, CMake/MSBuild, existing standalone 3D regression executable.

## Global Constraints

- Crossing ownership must never cross a non-G1 edge.
- Cauchy candidates may cross G1 and non-G1 edges only through explicit patch-edge topology.
- Do not use global Euclidean nearest-neighbor search across unrelated patches.
- Preserve 48 value samples, 28 normal samples, the cubic harmonic polynomial, tricubic/cubic restriction, and both Krylov equations.
- Run all torus, hollow-cylinder, and L-prism cases at `N=16,32,64`.

---

### Task 1: Complete patch topology and topological Cauchy selector

**Files:**
- Modify: `apps/native_nurbs_surface_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d.cpp`
- Test: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Produces: `NativeNurbsSurface3D::topological_neighbors` with the same edge-pair mapping type as `smooth_neighbors`.
- Produces: `std::vector<int> nearest_topological_cauchy_dofs(const NativeNurbsSurface3D&, const SurfaceDofCloud3D&, int center_dof, int count)`.
- Preserves: `parameter_dof_candidates_2x2(...)` reads only `smooth_neighbors`.

- [ ] **Step 1: Write failing topology and Cauchy-selection tests**

Extend `check_patch_regular` and `test_parameter_candidates` with checks equivalent to:

```cpp
require(surface.topological_neighbors.size() == surface.patches.size(),
        surface.name + " topological-neighbor count");

const SurfaceDofCloud3D lcloud = make_native_surface_dofs_3d(lprism, 3.0 / 16.0);
const int lcenter = lcloud.patches[7].dof_index(
    lcloud.patches[7].nu / 2, lcloud.patches[7].nv / 2);
const auto lids = nearest_topological_cauchy_dofs(
    lprism, lcloud, lcenter, 48);
require(contains_non_source_patch(lids, lcloud, 7),
        "L side Cauchy pool crosses a non-G1 topological edge");

const SurfaceDofCloud3D ccloud = make_native_surface_dofs_3d(
    cylinder, 3.0 / 16.0);
const int ccenter = ccloud.patches[0].dof_index(
    ccloud.patches[0].nu / 2, ccloud.patches[0].nv / 2);
const auto cids = nearest_topological_cauchy_dofs(
    cylinder, ccloud, ccenter, 48);
require(no_patch_in_range(cids, ccloud, 4, 8),
        "outer-wall Cauchy pool does not jump to the inner wall");
```

Keep the existing L-prism `parameter_dof_candidates_2x2` assertion that all
four candidates remain on the source patch at a non-G1 edge.

- [ ] **Step 2: Build and run the focused test to verify RED**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
```

Expected: compilation fails because `topological_neighbors` and
`nearest_topological_cauchy_dofs` do not exist.

- [ ] **Step 3: Add complete edge adjacency without changing G1 lookup**

Add `topological_neighbors` to `NativeNurbsSurface3D`. Make `append_patch`
append empty entries to both graphs. Add an edge-matching pass that compares
the two endpoints and midpoint of every unpaired native patch edge, records
the paired edge and its orientation in `topological_neighbors`, and throws if
one edge receives two topological neighbors. Existing `connect_smooth` must
also register the same pair topologically, while the cylinder wall/cap and
L-prism cap/side and side/side pairs remain absent from `smooth_neighbors`.

The resulting relationship is:

```cpp
smooth_neighbors[patch][edge]       // G1 subset: crossing 2x2 lookup
topological_neighbors[patch][edge]  // all shared edges: Cauchy BFS
```

- [ ] **Step 4: Remove the obsolete per-G1-component 48-DOF refinement**

Change the builder back to direct mesh-based uniform parameter counts:

```cpp
SurfaceDofCloud3D make_native_surface_dofs_3d(
    const NativeNurbsSurface3D& surface, double h);
```

Retain the G1 along-edge count harmonization needed for four distinct `2x2`
candidates, but delete `minimum_smooth_component_dofs` and its refinement
loop. The Cauchy BFS, rather than artificial refinement of each sharp patch,
will provide 48 samples.

- [ ] **Step 5: Implement breadth-first topological selection**

Implement `nearest_topological_cauchy_dofs` as follows:

```cpp
visited = {center.patch_id};
frontier = {center.patch_id};
candidate_count = dofs_on(center.patch_id);
while (candidate_count < count && !frontier.empty()) {
    next = every unvisited topological neighbor of the whole frontier;
    sort_and_unique(next);
    add the entire next ring to visited and candidate_count;
    frontier = next;
}
if (candidate_count < count) throw runtime_error(...);
sort all DOFs on visited patches by (distance_squared, dof_id);
return the first count IDs;
```

Validate `center_dof`, positive `count`, metadata sizes, and retention of the
center DOF. Whole-ring expansion prevents patch-number order from changing the
topological search radius.

- [ ] **Step 6: Build and run the focused test to verify GREEN**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected: exit code 0 and `native NURBS model tests passed`.

- [ ] **Step 7: Commit Task 1**

```powershell
git add apps/native_nurbs_surface_3d.hpp apps/native_nurbs_surface_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "fix: separate 3d Cauchy and crossing topology"
```

### Task 2: Use topological samples in the Cauchy fit and recompute results

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/results/2026-07-21-3d-harmonic-jet-results.md`

**Interfaces:**
- Consumes: `nearest_topological_cauchy_dofs(...)` from Task 1.
- Preserves: `PanelCenterHarmonicJetKFBI3D::surface_dof_for_crossing(...)` and `parameter_dof_candidates_2x2(...)` behavior.

- [ ] **Step 1: Replace smooth-component Cauchy candidates**

Set the requested counts exactly and use the Task 1 selector for each center:

```cpp
result.value_count = requested_value_count;
result.derivative_count = requested_derivative_count;
for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
    const std::vector<int> selected =
        app3d::nearest_topological_cauchy_dofs(
            surface, cloud, center, result.value_count);
    stencil.value_ids = selected;
    stencil.derivative_ids.assign(
        selected.begin(), selected.begin() + result.derivative_count);
}
```

Compute radius and incident-patch diagnostics from `selected`. Remove the
old `smooth_patch_component` candidate construction. Do not change crossing
ownership, harmonic fitting, spread, restriction, or GMRES code.

- [ ] **Step 2: Build both affected targets**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d --parallel 2
```

Expected: both targets build with exit code 0.

- [ ] **Step 3: Run focused and full numerical verification**

Run:

```powershell
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16 32 64
```

Expected:

- focused test exits 0;
- all 18 GMRES solves converge;
- every level reports Cauchy counts `48/28`;
- endpoint crossing fallback remains zero;
- L-prism Cauchy diagnostics select more than one patch near non-G1 edges.

- [ ] **Step 4: Update documentation with fresh data**

In the README and results document, replace statements that Cauchy sampling
stops at non-G1 edges. Record the new 16/32/64 errors, orders, iterations,
DOF counts, stencil radii, and the maximum GMRES residual from the fresh run.
Explain that crossing ownership remains G1-only while Cauchy sampling follows
complete topological adjacency.

- [ ] **Step 5: Verify diffs and commit Task 2**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and only intended source/test/document changes.

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp README.md docs/superpowers/results/2026-07-21-3d-harmonic-jet-results.md
git commit -m "test: record topological 3d Cauchy convergence"
```
