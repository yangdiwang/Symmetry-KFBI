# 3D Cauchy Stencil Policy Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement same-patch and patch-balanced 3D Cauchy sample selectors and compare them with the current selector on the L-prism at `N=16,32,64`.

**Architecture:** Keep selection in the app-local native-NURBS geometry module and pass a policy into the existing stencil builder. Allow per-row sample counts for coarse same-patch stencils, while retaining the active cubic harmonic fit and every transfer/GMRES setting. Add selection, conditioning, and policy diagnostics to policy-separated output directories.

**Tech Stack:** C++17, Eigen, CMake/MSBuild x64, existing native-NURBS 3D app and GMRES solver.

## Global Constraints

- Work directly on `main`; do not create a worktree or branch.
- Keep the harmonic Cauchy degree fixed at three (16 coefficients).
- Keep requested sample counts at 48 values and 28 normals.
- Keep GMRES relative tolerance fixed at `2e-10`.
- Preserve the current command-line syntax and the default output path.
- Do not change crossing ownership, spread, restriction, or manufactured data.

---

### Task 1: Add tested native-NURBS sample selectors

**Files:**
- Modify: `apps/native_nurbs_surface_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d.cpp`
- Test: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Produces: `nearest_same_patch_cauchy_dofs(const SurfaceDofCloud3D&, int, int)`.
- Produces: `balanced_topological_cauchy_dofs(const NativeNurbsSurface3D&, const SurfaceDofCloud3D&, int, int)`.
- Preserves: `nearest_topological_cauchy_dofs(...)` behavior.

- [ ] **Step 1: Write failing selector tests**

Add `using` declarations for both new functions. On the coarse L-prism, request
48 same-patch samples from patch 7 and assert that the returned size equals
that patch's available DOF count, contains the center, and every returned DOF
has `patch_id == 7`. On the fine L-prism, select a center next to a non-G1
edge, form the 48-point control and balanced sets, assert both contain the
center and have size 48, assert every balanced patch occurs in the control's
patch set, and assert active-patch balanced counts differ by at most one.

- [ ] **Step 2: Run the focused build and verify RED**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test
```

Expected: compile failure because the two selector functions are undeclared.

- [ ] **Step 3: Implement same-patch selection**

Validate the center and count, enumerate the contiguous DOF range belonging
to the center patch, sort by squared Euclidean distance and DOF id, and return
the first `min(count, patch.dof_count())` ids. Throw on invalid ownership or
an empty patch.

- [ ] **Step 4: Implement balanced topological selection**

Refactor the existing BFS candidate construction into an internal helper that
returns distance-sorted candidates after including at least the first
topological ring. Use the first `count` candidates to determine active patch
ids. Group all eligible candidates for those active patches, preserve distance
order inside each group, and select round-robin until `count` ids are obtained.
Skip exhausted groups and throw if the requested count cannot be filled. Sort
the final ids by distance and id and verify that the center was retained.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected: build succeeds and prints `native NURBS model tests passed`.

- [ ] **Step 6: Commit selectors**

```powershell
git add apps/native_nurbs_surface_3d.hpp apps/native_nurbs_surface_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: add 3d Cauchy stencil selectors"
```

### Task 2: Integrate policy selection and variable stencil sizes

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `.github/workflows/build-and-smoke.yml`

**Interfaces:**
- Consumes: both selectors from Task 1.
- Produces: `CauchyStencilPolicy3D` app-local enum with
  `TopologicalNearest`, `SamePatch`, and `BalancedPatches`.
- Produces: `KFBIM_3D_CAUCHY_POLICY` parser; default is
  `topological_nearest`.

- [ ] **Step 1: Add a failing smoke assertion**

Extend the 3D smoke command to require the default app output to contain
`cauchy_policy=topological_nearest`.

- [ ] **Step 2: Run the existing app and verify RED**

Run:

```powershell
$text = & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16 2>&1 | Out-String
if ($text -notmatch 'cauchy_policy=topological_nearest') { throw 'missing Cauchy policy diagnostic' }
```

Expected: failure with `missing Cauchy policy diagnostic`.

- [ ] **Step 3: Add policy parsing and dispatch**

Parse `KFBIM_3D_CAUCHY_POLICY`, reject unknown values, pass the policy through
`run_readiness_case` to `build_cauchy_stencils`, and dispatch value and normal
selection independently. The default calls the existing selector; same-patch
calls the capacity-limited selector; balanced calls the balanced selector.

- [ ] **Step 4: Make fit maps consume per-row counts**

In `PanelCenterCauchyFit3D::coefficients` and `build_map`, derive counts from
`stencil.value_ids.size()` and `stencil.derivative_ids.size()` instead of the
requested global counts. Reject fewer than 16 total rows. After SVD, reject a
non-finite condition or a smallest singular value not exceeding
`3e-12 * largest`; keep the existing pseudoinverse threshold and weights.

- [ ] **Step 5: Add diagnostics**

Track requested and minimum/maximum actual value/normal counts, maximum
per-stencil selected-patch count imbalance, radius over `h`, and incident
patch counts. Expose the fit-map condition values through the pipeline and
compute median, nearest-rank 95th percentile, and maximum. Print them and add
them, together with the policy, to readiness CSV output. Pad missing wide CSV
sample columns with empty fields so coarse same-patch rows remain parseable.

- [ ] **Step 6: Separate non-default outputs**

Keep `output/neumann_exterior_zero_trace_3d` for the default. Append
`same_patch` or `balanced_patches` for non-default policies. Print the selected
policy before the levels and include it in each result row.

- [ ] **Step 7: Build and verify all policy smoke runs**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
$env:KFBIM_3D_CAUCHY_POLICY='topological_nearest'; & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
$env:KFBIM_3D_CAUCHY_POLICY='same_patch'; & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
$env:KFBIM_3D_CAUCHY_POLICY='balanced_patches'; & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
Remove-Item Env:KFBIM_3D_CAUCHY_POLICY
```

Expected: focused tests pass; all six formulation solves converge; each run
prints its policy, finite condition statistics, and the expected sample-count
range.

- [ ] **Step 8: Commit integration**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp .github/workflows/build-and-smoke.yml
git commit -m "feat: compare 3d Cauchy stencil policies"
```

### Task 3: Run the controlled study and record the conclusion

**Files:**
- Create: `docs/superpowers/results/2026-07-21-3d-cauchy-stencil-policy-comparison.md`

**Interfaces:**
- Consumes: policy-separated readiness, Neumann, and Dirichlet CSV files.
- Produces: one comparison table and a policy recommendation.

- [ ] **Step 1: Run all policies at all levels**

Run the executable for `l_prism 16 32 64` with each of the three policy
environment values, clearing the environment variable afterward.

- [ ] **Step 2: Verify numerical acceptance**

Check all 18 formulation rows report convergence and finite residuals. For
each policy, verify Dirichlet density and interior `Linf` errors strictly
decrease from 16 to 32 and from 32 to 64. Extract GMRES iterations, errors,
orders, route mismatch, condition statistics, stencil radii, imbalances, and
runtimes.

- [ ] **Step 3: Write the result document**

Record exact commands, configuration, a concise table for the three
Dirichlet sequences, Neumann regression behavior, stencil diagnostics, and a
conclusion that distinguishes iteration stability from accuracy/order.

- [ ] **Step 4: Run final verification**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git status --short --branch
```

Expected: build and tests pass, `git diff --check` is silent, and only the
result document remains uncommitted.

- [ ] **Step 5: Commit results**

```powershell
git add docs/superpowers/results/2026-07-21-3d-cauchy-stencil-policy-comparison.md
git commit -m "test: compare 3d Cauchy stencil policies"
```
