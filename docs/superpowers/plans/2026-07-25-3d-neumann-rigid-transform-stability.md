# 3D Neumann L-Prism Rigid-Transform Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated Neumann L-prism rigid-transform study that runs the accepted crossing-owner route for the Dirichlet study's eight poses at `N=32,64,128` and reports accuracy, order, GMRES, timing, geometry, owner, and acceptance results.

**Architecture:** Preserve `--neumann-owner-study` as the focused two-route A/B study. Add a pure data/evaluation module for level validation and rigid-study acceptance, extract one reusable production worker for a `(pose,N)` Neumann value-trace pipeline, and drive the new crossing-owner-only study through that worker. Reuse one generic eight-pose catalog so Dirichlet and Neumann transforms cannot diverge.

**Tech Stack:** C++17, Eigen, existing native-NURBS/KFBI3D pipeline, ZFFT bulk solver, CMake/Visual Studio Release build, CSV checkpoints.

## Global Constraints

- Geometry is the native twelve-patch NURBS L-prism only.
- Default and final levels are exactly `32,64,128`.
- Poses are the existing eight Dirichlet rigid cases without numerical changes.
- Neumann uses `JointTricubicCrossingOwner` and `RegionClosestHybrid` only.
- Cauchy fitting remains degree 3, `G1Nearest`, 48 value and 28 normal conditions.
- GMRES uses relative tolerance `2e-10`, restart 80, and cap 80.
- Existing `--rigid-study` and `--neumann-owner-study` behavior remains available.
- Generated CSVs stay ignored under `output/neumann_rigid_transform_stability_3d`.
- No geometry, FFT, spread, Cauchy, or integral-equation redesign is in scope.

---

### Task 1: Shared rigid catalog and pure Neumann study evaluation

**Files:**
- Modify: `apps/dirichlet_rigid_transform_study_3d.hpp`
- Modify: `apps/dirichlet_rigid_transform_study_3d.cpp`
- Modify: `apps/dirichlet_rigid_transform_study_3d_test.cpp`
- Create: `apps/neumann_rigid_transform_study_3d.hpp`
- Create: `apps/neumann_rigid_transform_study_3d.cpp`
- Create: `apps/neumann_rigid_transform_study_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: `RigidTransform3D`, `RigidStudyCriterionStatus3D`, and `combine_rigid_study_criteria_3d`.
- Produces:

```cpp
using LPrismRigidStudyCase3D = DirichletRigidStudyCase3D;

std::vector<LPrismRigidStudyCase3D>
make_l_prism_rigid_study_cases_3d();

struct NeumannRigidStudyMeasurement3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    bool finite_metrics = false;
    bool gmres_converged = false;
    int gmres_iterations = 0;
    double gmres_relative_residual = 0.0;
    double interior_linf = 0.0;
    bool geometry_diagnostics_pass = false;
    bool owner_invariants_pass = false;
};

struct NeumannRigidStudyDerivedRow3D {
    NeumannRigidStudyMeasurement3D measurement;
    double interior_order;
    double baseline_error_ratio;
    double baseline_iteration_ratio;
    RigidStudyCriterionStatus3D gmres_pass;
    RigidStudyCriterionStatus3D baseline_ratio_pass;
    RigidStudyCriterionStatus3D geometry_diagnostics_pass;
    RigidStudyCriterionStatus3D owner_invariants_pass;
    RigidStudyCriterionStatus3D row_pass;
};

struct NeumannRigidStudyAcceptance3D {
    std::string case_id;
    RigidStudyCriterionStatus3D completeness_pass;
    RigidStudyCriterionStatus3D row_pass;
    RigidStudyCriterionStatus3D gmres_pass;
    RigidStudyCriterionStatus3D monotone_error_pass;
    RigidStudyCriterionStatus3D order_64_128_pass;
    RigidStudyCriterionStatus3D baseline_ratio_pass;
    RigidStudyCriterionStatus3D geometry_diagnostics_pass;
    RigidStudyCriterionStatus3D owner_invariants_pass;
    RigidStudyCriterionStatus3D overall_pass;
};

struct NeumannRigidStudyEvaluation3D {
    std::vector<NeumannRigidStudyDerivedRow3D> rows;
    std::vector<NeumannRigidStudyAcceptance3D> cases;
    bool all_pass = false;
};

std::vector<int> normalize_neumann_rigid_levels_3d(
    std::vector<int> levels);

NeumannRigidStudyEvaluation3D evaluate_neumann_rigid_study_3d(
    const std::vector<NeumannRigidStudyMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_acceptance);
```

- [ ] **Step 1: Add failing catalog and evaluation tests**

Extend the Dirichlet test to compare the generic and compatibility catalogs
item-by-item.  Create the Neumann test with literal fixtures that catch:

```cpp
void test_level_prefix_validation()
{
    require(normalize_neumann_rigid_levels_3d({128, 32, 64, 64})
                == std::vector<int>({32, 64, 128}),
            "levels are sorted and deduplicated");
    require_throws([] {
        normalize_neumann_rigid_levels_3d({64});
    }, "N=64 requires N=32");
    require_throws([] {
        normalize_neumann_rigid_levels_3d({32, 128});
    }, "N=128 requires N=32 and N=64");
    require_throws([] {
        normalize_neumann_rigid_levels_3d({16, 32});
    }, "only N=32,64,128 are accepted");
}
```

Use two synthetic cases with deliberately different errors so a missing
`case_id` grouping produces the wrong order:

```cpp
const std::vector<NeumannRigidStudyMeasurement3D> rows = {
    passing("baseline", 32, 0.09375, 8.0e-6, 40),
    passing("shifted",  32, 0.09375, 3.2e-5, 44),
    passing("baseline", 64, 0.046875, 2.0e-6, 41),
    passing("shifted",  64, 0.046875, 8.0e-6, 45),
    passing("baseline",128, 0.0234375, 5.0e-7, 42),
    passing("shifted", 128, 0.0234375, 2.0e-6, 46),
};
```

Assert both cases have order `2.0`, the shifted/baseline error ratio is
`4.0`, the baseline ratio criterion fails only for `shifted`, and complete
passing rows with ratio at most `3.0` pass.  Separate fixtures must make
missing `N=128`, GMRES failure, non-finite metrics, geometry failure, and
owner-invariant failure observable in the corresponding statuses.

Production mutations caught: separate pose catalogs, order lookup without
`case_id`, baseline lookup at the wrong level, incomplete results marked
complete, and ignored row failures.

- [ ] **Step 2: Build the new test and verify RED**

Run:

```powershell
cmake --build build --config Release --target neumann_rigid_transform_study_3d_test -- /m:1
```

Expected: build fails because
`neumann_rigid_transform_study_3d.hpp` and the generic catalog API do not
exist.

- [ ] **Step 3: Implement the generic catalog and pure evaluator**

Move the literal case construction into
`make_l_prism_rigid_study_cases_3d()`.  Keep:

```cpp
std::vector<DirichletRigidStudyCase3D>
make_l_prism_dirichlet_rigid_study_cases_3d()
{
    return make_l_prism_rigid_study_cases_3d();
}
```

Implement level sorting/deduplication and prefix validation.  In the
evaluator:

- group previous rows by `case_id`;
- match baseline rows by identical `N`;
- use `log(E_coarse/E_fine)/log(N_fine/N_coarse)`;
- pass GMRES only for convergence, iterations `<=80`, and residual
  `<=2e-10`;
- pass baseline ratio only when finite and `<=3`;
- mark completeness pass only with `32`, `64`, and `128`;
- require strict error decrease and `64->128` order `>=1.8`;
- aggregate with `combine_rigid_study_criteria_3d`, treating unavailable
  full-grid criteria as neutral only when
  `require_complete_acceptance == false`.

- [ ] **Step 4: Register and run tests to verify GREEN**

Add the new `.cpp` to `kfbim_3d_app_geometry`; add, link, and set C++17 for
`neumann_rigid_transform_study_3d_test`.

Run:

```powershell
cmake --build build --config Release --target dirichlet_rigid_transform_study_3d_test neumann_rigid_transform_study_3d_test -- /m:1
.\\build\\apps\\Release\\dirichlet_rigid_transform_study_3d_test.exe
.\\build\\apps\\Release\\neumann_rigid_transform_study_3d_test.exe
```

Expected: both exit 0.

- [ ] **Step 5: Commit**

```powershell
git add apps/CMakeLists.txt apps/dirichlet_rigid_transform_study_3d.hpp apps/dirichlet_rigid_transform_study_3d.cpp apps/dirichlet_rigid_transform_study_3d_test.cpp apps/neumann_rigid_transform_study_3d.hpp apps/neumann_rigid_transform_study_3d.cpp apps/neumann_rigid_transform_study_3d_test.cpp
git commit -m "feat: add Neumann rigid study evaluation"
```

---

### Task 2: Reusable Neumann pose worker and rigid-study CLI

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`

**Interfaces:**
- Consumes: generic rigid catalog and pure evaluation API from Task 1.
- Produces: `--neumann-rigid-study [N ...]`, one crossing-owner solve for
  each selected `(case,N)`, and checkpoint CSV writers.

- [ ] **Step 1: Verify the CLI behavior is RED**

Run the real application boundary:

```powershell
$help = .\\build\\apps\\Release\\neumann_exterior_zero_trace_3d.exe --help |
    Out-String
if ($help -notmatch '--neumann-rigid-study') { throw 'route missing' }
```

Expected: failure `route missing`.

- [ ] **Step 2: Extend the internal row without changing owner A/B semantics**

Add the following to `NeumannOwnerStudyRow3D`:

```cpp
app3d::LPrismRigidStudyCase3D study_case;
double setup_seconds = 0.0;
double total_seconds = 0.0;
int label_mismatches = 0;
int unsafe_label_changing_edges = 0;
int gap_crossings = 0;
int endpoint_crossings = 0;
int triangle_fallback_crossings = 0;
```

Change `assign_neumann_owner_order_3d` to require both matching
`study_case.id` and `mode`.  Preserve the existing owner-study CSV columns
and calculations, adding `case_id` only where keys otherwise become
ambiguous.

- [ ] **Step 3: Extract one production `(pose,N)` worker**

Extract the body that constructs grid, transformed geometry, domain,
surface DOFs, Cauchy stencils, `GridPair3D`, and the
`RegionClosestHybrid` pipeline into:

```cpp
std::vector<NeumannOwnerStudyRow3D>
run_neumann_value_trace_pose_3d(
    int N,
    const app3d::LPrismRigidStudyCase3D& study_case,
    const std::vector<ExteriorValueRestrictMode3D>& modes);
```

The worker must:

- compare every numerical grid label with `geometry.exact_inside`;
- retain the existing default-versus-explicit-legacy bitwise probe;
- require a finite crossing-owner probe;
- capture native unsafe-edge diagnostics;
- inspect unique correction-support crossing edges for gap, endpoint, and
  triangle ownership fallback counts;
- record owner fingerprints/query counts before and after each solve;
- run `run_neumann_case` with the supplied mode;
- record setup, solve, and total wall times;
- preserve the mode iteration order passed by the caller.

Refactor `run_neumann_owner_study_3d` to call this worker for its existing
single rotated case and its existing two modes.  Do not change its default
levels, output directory, pass gate, or numerical route.

- [ ] **Step 4: Add rigid-study writers and driver**

Implement:

```cpp
int run_neumann_rigid_study_3d(std::vector<int> levels);
```

It normalizes levels through Task 1, obtains all eight generic cases, and
calls the worker with only:

```cpp
{ExteriorValueRestrictMode3D::JointTricubicCrossingOwner}
```

After every solve, map all completed raw rows into
`NeumannRigidStudyMeasurement3D`, evaluate them, and atomically rewrite:

```text
output/neumann_rigid_transform_stability_3d/rigid_transform_results.csv
output/neumann_rigid_transform_stability_3d/gmres_residuals.csv
output/neumann_rigid_transform_stability_3d/owner_diagnostics.csv
output/neumann_rigid_transform_stability_3d/rigid_transform_acceptance.csv
```

The results writer includes the full 3x3 rotation, axis, angle, center,
translation, N/h, DOFs, setup/solve/total time, GMRES metrics, density and
interior errors, derived order and baseline ratios, geometry counts, owner
invariants, and row pass.

Add command detection, default levels `{32,64,128}`, dispatch, and help text
for `--neumann-rigid-study`.

- [ ] **Step 5: Build and verify CLI GREEN**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d -- /m:1
$help = .\\build\\apps\\Release\\neumann_exterior_zero_trace_3d.exe --help |
    Out-String
if ($help -notmatch '--neumann-rigid-study') { throw 'route missing' }
.\\build\\apps\\Release\\neumann_exterior_zero_trace_3d.exe --neumann-rigid-study 16
```

Expected: build and help check pass; invalid N exits nonzero before geometry
setup with a message allowing only 32, 64, and 128.

- [ ] **Step 6: Run the N=32 integration study**

Run:

```powershell
$env:KFBIM_3D_NEUMANN_RIGID_STUDY_OUTPUT_DIR =
    'output/neumann_rigid_transform_stability_3d_n32'
.\\build\\apps\\Release\\neumann_exterior_zero_trace_3d.exe --neumann-rigid-study 32
```

Verify:

- exit code 0;
- exactly eight result rows and eight acceptance rows;
- every case ID occurs once;
- all rows use `joint_tricubic_crossing_owner` and
  `region_closest_hybrid`;
- all GMRES and row passes are true;
- all query counts are unchanged;
- full-grid order criteria are `not_evaluated`, not falsely passed.

- [ ] **Step 7: Run focused regressions and commit**

```powershell
.\\build\\apps\\Release\\neumann_rigid_transform_study_3d_test.exe
.\\build\\apps\\Release\\dirichlet_rigid_transform_study_3d_test.exe
.\\build\\apps\\Release\\harmonic_trace_correction_3d_test.exe
.\\build\\apps\\Release\\crossing_owner_restrict_3d_test.exe
.\\build\\apps\\Release\\restrict_owner_geometry_preprocessor_3d_test.exe
.\\build\\apps\\Release\\neumann_exterior_zero_trace_3d.exe --neumann-owner-study 32
git diff --check
```

Expected: all commands exit 0 and the existing owner study still writes two
rows for its single rotated case.

Commit:

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add Neumann rigid transform study"
```

---

### Task 3: Full numerical study, acceptance audit, and result report

**Files:**
- Create: `docs/superpowers/results/2026-07-25-3d-neumann-rigid-transform-stability.md`
- Generate, untracked: `output/neumann_rigid_transform_stability_3d/*.csv`

**Interfaces:**
- Consumes: completed CLI from Task 2.
- Produces: audited N=32/64/128 data and a concise checked-in result table.

- [ ] **Step 1: Run all eight poses at all three levels**

```powershell
$env:KFBIM_3D_NEUMANN_RIGID_STUDY_OUTPUT_DIR =
    'output/neumann_rigid_transform_stability_3d'
.\\build\\apps\\Release\\neumann_exterior_zero_trace_3d.exe `
    --neumann-rigid-study 32 64 128
```

Let the executable checkpoint after each solve.  If any row exceeds the
80-iteration cap, has a non-finite metric, changes owner diagnostics during
GMRES, or fails geometry integrity, stop refinement and diagnose before
continuing.

- [ ] **Step 2: Audit generated CSVs**

Use `Import-Csv` to require:

- 24 result rows, 24 owner rows, and 8 acceptance rows;
- exactly three levels for each of eight case IDs;
- no duplicate `(case_id,N)` keys;
- every result is crossing-owner/hybrid and finite;
- query counts and owner fingerprints match before/after GMRES;
- each acceptance status agrees with independently recomputed error
  monotonicity, `64->128` order, and baseline ratio;
- no geometry mismatch/fallback count is nonzero.

Print a compact table with:

```text
case_id,N,interior_linf,interior_order,gmres_iterations,
gmres_relative_residual,setup_seconds,solve_seconds,total_seconds
```

- [ ] **Step 3: Write the result report**

Document:

- the exact route and eight transformations;
- one table per pose or one compact 24-row table;
- `32->64` and `64->128` interior maximum-error orders;
- GMRES iteration counts and residuals;
- setup, solve, total time and overall wall time;
- worst pose/baseline error and iteration ratios;
- every failed acceptance criterion without suppressing it;
- an algorithmic conclusion about translation/rotation sensitivity.

- [ ] **Step 4: Run final verification**

```powershell
cmake --build build --config Release -- /m:1
.\\build\\apps\\Release\\neumann_rigid_transform_study_3d_test.exe
.\\build\\apps\\Release\\dirichlet_rigid_transform_study_3d_test.exe
.\\build\\apps\\Release\\harmonic_trace_correction_3d_test.exe
.\\build\\apps\\Release\\crossing_owner_restrict_3d_test.exe
.\\build\\apps\\Release\\restrict_owner_geometry_preprocessor_3d_test.exe
.\\build\\apps\\Release\\native_nurbs_surface_3d_test.exe
.\\build\\apps\\Release\\native_nurbs_surface_transform_3d_test.exe
git diff --check
```

Expected: Release build succeeds, all seven executables exit 0, and the
diff check is clean.

- [ ] **Step 5: Commit**

```powershell
git add docs/superpowers/results/2026-07-25-3d-neumann-rigid-transform-stability.md
git commit -m "docs: report Neumann rigid transform stability"
```
