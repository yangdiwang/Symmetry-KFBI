# 3D Crossing-Owner Normal Restrict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain a selectable joint-tricubic restrict route that reroutes reliable foreign non-G1 support-node corrections to the actual NURBS crossing owner, then compare it with the unchanged legacy route.

**Architecture:** Extract the crossing-to-DOF decision into a small testable helper, precompute both legacy and owner-grouped correction terms in each trace sample, and select the representation only at exterior-normal trace recovery. Extend the existing causal probe with a third route and detailed owner diagnostics while keeping geometry work outside GMRES.

**Tech Stack:** C++17, Eigen, native NURBS surface intersector, CMake/Visual Studio x64, existing KFBI3D harmonic-jet pipeline and GMRES.

## Global Constraints

- Work directly on the current `main` branch, as previously approved.
- Do not modify the normal-layer locations or `4x4x4` grid interpolation.
- Do not change Cauchy degree, Cauchy neighborhoods, spread, bulk solve, or GMRES parameters.
- The default route remains `JointTricubicCauchy`.
- Only a reliable single transverse crossing on a foreign non-G1 patch may change the Cauchy owner.
- All ambiguous intersection results fall back to the target owner and are diagnosed.
- Perform all NURBS intersections during template construction, never in GMRES.
- Do not stage or modify the two unrelated untracked same-patch documents.

---

### Task 1: Testable Crossing-Owner Selection

**Files:**
- Create: `apps/crossing_owner_restrict_3d.hpp`
- Create: `apps/crossing_owner_restrict_3d.cpp`
- Create: `apps/crossing_owner_restrict_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: target DOF, query/support segment, native NURBS intersector, surface topology, and surface DOF cloud.
- Produces:

```cpp
namespace kfbim::app3d {

enum class RestrictOwnerDecisionKind3D {
    TargetSideNode,
    TargetOrG1SingleCrossing,
    ForeignNonG1SingleCrossing,
    NoCrossingFallback,
    MultipleCrossingFallback,
    DegenerateCrossingFallback,
    AmbiguousEdgeFallback
};

struct RestrictOwnerDecision3D {
    int owner_dof = -1;
    RestrictOwnerDecisionKind3D kind =
        RestrictOwnerDecisionKind3D::TargetSideNode;
    int crossing_patch = -1;
    double segment_parameter = 0.0;
    double residual = 0.0;
    double transversality = 0.0;
};

RestrictOwnerDecision3D select_restrict_correction_owner_3d(
    int target_dof,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support,
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection);

} // namespace kfbim::app3d
```

- [ ] **Step 1: Write failing owner-selection tests**

Construct synthetic intersection results with exact `patch,u,v` roots against
the L-prism surface and DOF cloud. Assert:

```cpp
require(same_patch.owner_dof == target,
        "same-patch crossing retains target owner");
require(g1_patch.owner_dof == target,
        "G1 crossing retains target owner");
require(foreign.kind ==
            RestrictOwnerDecisionKind3D::ForeignNonG1SingleCrossing,
        "foreign non-G1 crossing is rerouted");
require(cloud.dofs[foreign.owner_dof].patch_id == foreign_patch,
        "foreign crossing uses its parameter-owned patch");
require(multiple.owner_dof == target,
        "multiple roots fall back to target");
```

Add empty, tangent/overlap, and ambiguous non-G1-edge cases. Register the test
target in `apps/CMakeLists.txt`.

- [ ] **Step 2: Build the focused test and verify RED**

Run:

```powershell
cmake --build build --config Release --target crossing_owner_restrict_3d_test
```

Expected: compilation fails because the helper does not exist.

- [ ] **Step 3: Implement the minimal owner decision**

Require one finite transverse root, determine whether its patch is in:

```cpp
cloud.patches[target_patch].smooth_patch_ids
```

For a foreign patch call:

```cpp
const auto candidates = parameter_dof_candidates_2x2(
    surface, cloud, root.patch_index, root.u, root.v);
```

Select the Euclidean-nearest candidate with a compatible NURBS normal, using
the same two-pass normal rule as the existing crossing-row association.
Return the target DOF for every non-single or degenerate result.

- [ ] **Step 4: Build and verify GREEN**

Run:

```powershell
cmake --build build --config Release --target crossing_owner_restrict_3d_test
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
```

Expected: all owner, G1, and fallback assertions pass.

---

### Task 2: Precomputed Legacy and Crossing-Owner Corrections

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:73-76`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:757-762`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:764-802`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:939-989`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1119-1198`

**Interfaces:**
- Consumes: `select_restrict_correction_owner_3d`.
- Produces:

```cpp
struct HarmonicTraceCorrectionTerm3D {
    int owner_dof = -1;
    Eigen::VectorXd evaluation;
};

struct HarmonicTraceSample3D {
    std::array<int, 64> grid_ids{};
    std::array<double, 64> weights{};
    Eigen::VectorXd legacy_correction_evaluation;
    std::vector<HarmonicTraceCorrectionTerm3D> owner_corrections;
    RestrictOwnerSampleDiagnostics3D owner_diagnostics;
};
```

- [ ] **Step 1: Add a failing mode/application test**

Add:

```cpp
JointTricubicCrossingOwner
```

to `ExteriorNormalRestrictMode3D` and route it through
`exterior_normal_trace`. Reference a not-yet-implemented owner-correction
application so the app fails to compile.

- [ ] **Step 2: Build and verify RED**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
```

Expected: failure for the missing owner-correction representation.

- [ ] **Step 3: Precompute owner terms in `build_trace_sample`**

Construct one `NurbsSurfaceIntersector3D` with the native geometry model in
the pipeline. For every wrong-side support node:

```cpp
const auto intersection = intersector_.intersect_segment(query, node_point);
const auto decision = select_restrict_correction_owner_3d(
    center, query, node_point, native_surface_, cloud_, intersection);
const int owner = decision.owner_dof;
const Eigen::Vector3d xi = local_coordinate(owner, node_point);
const Eigen::VectorXd evaluation =
    correction_sign * weight
    * fit_.space().basis(xi.x(), xi.y(), xi.z());
```

Accumulate evaluations by owner DOF. Independently preserve the exact legacy
target-center evaluation. Convert the map to a deterministic owner-sorted
vector after all 64 nodes are processed.

- [ ] **Step 4: Select correction representation without geometry work**

Pass the mode into `continued_samples`. For the legacy route apply only:

```cpp
sample.legacy_correction_evaluation.dot(
    field.coefficients.row(center).transpose());
```

For the crossing-owner route sum:

```cpp
for (const auto& term : sample.owner_corrections)
    value += term.evaluation.dot(
        field.coefficients.row(term.owner_dof).transpose());
```

Keep value traces and the default exterior-normal overload on the legacy
route.

- [ ] **Step 5: Build and run legacy regressions**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --rigid-study 16
```

Expected: both exit zero and retain the legacy route and numerical output.

---

### Task 3: Extend the Causal Probe and Diagnostics

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:3085-3635`

**Interfaces:**
- Consumes: all three restrict modes and precomputed owner diagnostics.
- Produces:
  - a third `joint_tricubic_crossing_owner` route;
  - `restrict_owner_summary.csv`;
  - `restrict_owner_terms.csv`;
  - extended GMRES and numerical summaries.

- [ ] **Step 1: Add the third route and verify a failing smoke**

Resize:

```cpp
std::array<NormalRestrictRouteProbe3D, 3> routes;
```

and enumerate:

```cpp
{JointTricubicCauchy,
 JointTricubicCrossingOwner,
 ExteriorOnlyHarmonicCubic}
```

Reference missing owner diagnostic writers, then build to verify RED.

- [ ] **Step 2: Implement aggregate and per-term output**

Write `restrict_owner_summary.csv` with counts and `sum_abs_weight` for every
decision kind by case, `N`, target DOF, side, and layer. Write
`restrict_owner_terms.csv` for every foreign reroute with target/owner patch
and DOF, node, weight, crossing parameters, residual, and transversality.

Preserve incremental output after every completed route.

- [ ] **Step 3: Add contraction metrics**

For each residual sequence compute:

```cpp
rho(begin,end) =
    std::pow(residual[end] / residual[begin],
             1.0 / static_cast<double>(end - begin));
```

Report iterations 10--30 when present and the maximum five-step rho over the
history. Do not treat a capped non-converged solve as converged.

- [ ] **Step 4: Run the N=16 smoke**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe 16
```

Expected: three cases times three routes complete or report the existing
160-step cap; all CSV values are finite; owner diagnostics contain both
retained and rerouted entries.

---

### Task 4: Verification and Numerical A/B

**Files:**
- Verify: all modified and created source files.
- Generate: `output/dirichlet_normal_restrict_causal_probe_3d/*.csv`

**Interfaces:**
- Consumes: completed optional route and probe.
- Produces: a numerical comparison of unchanged legacy versus crossing-owner
  correction.

- [ ] **Step 1: Build all affected Release targets**

Run:

```powershell
cmake --build build --config Release --target crossing_owner_restrict_3d_test exterior_only_cubic_normal_restrict_3d_test native_nurbs_surface_3d_test native_nurbs_surface_transform_3d_test dirichlet_rigid_transform_study_3d_test neumann_exterior_zero_trace_3d
```

Expected: all targets build.

- [ ] **Step 2: Run focused and regression tests**

Run:

```powershell
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\exterior_only_cubic_normal_restrict_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
```

Expected: all exit zero.

- [ ] **Step 3: Run the approved evidence cases**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe 32 64
```

Expected: baseline, rotation, and rotation-plus-translation complete for all
three routes with incremental output retained.

- [ ] **Step 4: Compare the strict A/B**

Report legacy versus crossing-owner for each case and level:

```text
physical_iterations
physical_final_residual
rho_10_30
worst_rho_5
physical_interior_linf
exact_grid_linf
exact_equation_linf
seconds
```

Compute the `32 -> 64` interior-error order. State whether the rotated
residual plateaus are shortened and whether baseline accuracy or convergence
is degraded.

- [ ] **Step 5: Final repository verification**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and the two pre-existing unrelated same-patch
documents remain untouched.
