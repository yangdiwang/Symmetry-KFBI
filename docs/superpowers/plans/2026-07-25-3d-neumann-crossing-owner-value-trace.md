# 3D Neumann Crossing-Owner Value-Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the existing center-owned 3D Neumann exterior value trace with the accepted crossing-owner/RegionClosestHybrid correction on the rotated L-prism through `N=128`.

**Architecture:** Factor the scalar correction accumulation into a small tested app-layer helper, then let the value-trace operator select legacy-center or crossing-owner correction while sharing one owner-preprocessed pipeline. Add a dedicated study route that writes complete comparison CSVs and never changes the first-kind Neumann equation, mean constraint, spread, FFT, Cauchy fit, or GMRES.

**Tech Stack:** C++17, Eigen, existing NURBS geometry and KFBI harmonic-jet pipeline, Visual Studio x64 Release, CMake, zFFT, GMRES, PowerShell.

## Global Constraints

- Work only on `codex/rotated-n128-phase-profile`.
- Preserve the current default Neumann route.
- Use degree-3 Cauchy fitting, G1-nearest 48/28 surface stencils, tricubic grid interpolation, and cubic normal-line recovery.
- Use `RegionClosestHybrid` for the crossing-owner pipeline.
- Preserve correction accumulation order `q=0..63`.
- Perform no geometry query during GMRES.
- Keep GMRES tolerance `2e-10` and cap iterations at `80`.
- Run the rotated L-prism at `N=32`, `64`, and `128`; do not start `128` unless both smaller levels pass.
- Keep generated CSV and logs untracked.
- Use `/m:1`, never `/m`, for Visual Studio builds.

---

### Task 1: Add a tested correction-route helper

**Files:**
- Create: `apps/harmonic_trace_correction_3d.hpp`
- Create: `apps/harmonic_trace_correction_3d.cpp`
- Create: `apps/harmonic_trace_correction_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `enum class TraceCorrectionOwnerMode3D { CenterDof, CrossingOwner };`
  - `struct HarmonicTraceCorrectionTermInput3D { int owner_dof; Eigen::VectorXd evaluation; };`
  - `double apply_harmonic_trace_correction_3d(...)`
- Consumes coefficient rows and correction terms in their existing stored order.

- [ ] **Step 1: Write the failing unit tests**

Test center-owned evaluation, foreign-owner evaluation, ordered accumulation
with cancellation-sensitive values, empty crossing terms, and invalid owner
indices. The ordered case computes its expected result with:

```cpp
double expected = 0.0;
for (const auto& term : terms)
    expected += term.evaluation.dot(coefficients.row(term.owner_dof));
```

- [ ] **Step 2: Build to verify the test fails**

```powershell
cmake --build build --config Debug --target harmonic_trace_correction_3d_test -- /m:1
```

Expected: failure because the helper target or declarations do not exist.

- [ ] **Step 3: Implement the minimal helper**

The center route returns:

```cpp
legacy_evaluation.dot(coefficients.row(center_dof).transpose())
```

The crossing route loops over `terms` without sorting, grouping, or summing
through a different reduction.

- [ ] **Step 4: Build and run the test**

```powershell
cmake --build build --config Debug --target harmonic_trace_correction_3d_test -- /m:1
.\build\apps\Debug\harmonic_trace_correction_3d_test.exe
```

Expected: exit `0`.

- [ ] **Step 5: Commit**

```powershell
git add apps/harmonic_trace_correction_3d.hpp apps/harmonic_trace_correction_3d.cpp apps/harmonic_trace_correction_3d_test.cpp apps/CMakeLists.txt
git commit -m "test: add ordered harmonic trace correction routes"
```

---

### Task 2: Make Neumann exterior value restriction mode-selectable

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Test: `apps/harmonic_trace_correction_3d_test.cpp`

**Interfaces:**
- Produces:
  - `enum class ExteriorValueRestrictMode3D { JointTricubicCauchy, JointTricubicCrossingOwner };`
  - an `exterior_trace(..., ExteriorValueRestrictMode3D)` overload;
  - mode parameters on `ExteriorZeroTraceOperator3D`,
    `solve_exterior_zero_trace_neumann_3d`, and `run_neumann_case`.
- Preserves no-mode calls as `JointTricubicCauchy`.

- [ ] **Step 1: Add compile-time route calls before the overload exists**

Add the two route names and invoke both from a focused constant-jump probe in
the application. Build once and verify the missing overload fails.

- [ ] **Step 2: Replace local correction accumulation with the helper**

Convert each existing owner term to
`HarmonicTraceCorrectionTermInput3D`. In `continued_samples`, select
`CenterDof` or `CrossingOwner` and call the helper. Do not change the 64 grid
weight accumulation preceding the correction.

- [ ] **Step 3: Thread the value mode through the Neumann operator**

`ExteriorZeroTraceOperator3D::apply` and `right_hand_side` must call the new
value-trace overload. The default constructor path remains center-owned.

- [ ] **Step 4: Add runtime invariants**

For an owner-built pipeline require:

- default `exterior_trace` equals explicit `JointTricubicCauchy` bit-for-bit;
- both routes leave owner diagnostics and query counts unchanged;
- a crossing-owner call rejects a pipeline without owner templates.

- [ ] **Step 5: Build and run focused tests**

```powershell
cmake --build build --config Debug --target neumann_exterior_zero_trace_3d crossing_owner_restrict_3d_test restrict_owner_geometry_preprocessor_3d_test harmonic_trace_correction_3d_test -- /m:1
.\build\apps\Debug\harmonic_trace_correction_3d_test.exe
.\build\apps\Debug\crossing_owner_restrict_3d_test.exe
.\build\apps\Debug\restrict_owner_geometry_preprocessor_3d_test.exe
```

Expected: all exit `0`.

- [ ] **Step 6: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/harmonic_trace_correction_3d_test.cpp
git commit -m "feat: add crossing-owner Neumann value restriction"
```

---

### Task 3: Add the rotated L-prism comparison study

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`

**Interfaces:**
- Produces CLI `--neumann-owner-study [N ...]`.
- Writes `summary.csv`, `gmres_residuals.csv`, and
  `owner_diagnostics.csv` below
  `output/neumann_value_trace_crossing_owner_3d`.

- [ ] **Step 1: Add strict level parsing and usage text**

Accept powers of two from `32` through `128`. If `128` is selected, require
both `32` and `64`.

- [ ] **Step 2: Build one shared case context per level**

Create the rotated L-prism, NURBS domain, surface DOFs, G1-nearest 48/28
Cauchy stencils, and one pipeline using `RegionClosestHybrid`. Verify exact
grid labels and retain the pipeline setup time.

- [ ] **Step 3: Run the two Neumann routes**

Run in this order:

```text
joint_tricubic_cauchy
joint_tricubic_crossing_owner
```

Use identical prescribed flux, constant-shift evaluation, GMRES tolerance,
restart `80`, and maximum `80`. Snapshot owner diagnostics and query counts
before each GMRES solve and compare after it.

- [ ] **Step 4: Write checkpoint CSVs after each level**

`summary.csv` includes:

```text
N,h,mode,dofs,pipeline_setup_seconds,solve_seconds,converged,iterations,
final_residual,operator_residual_linf,exterior_condition_linf,
route_mismatch_linf,density_linf,density_l2,interior_linf,interior_l2,
interior_order,geometry_queries_before_gmres,
geometry_queries_after_gmres,pass
```

`gmres_residuals.csv` contains `N,mode,iteration,residual`.
`owner_diagnostics.csv` contains the preprocessing mode, workload
fingerprint, output digest, wrong-side query count, and pre/post GMRES query
counts.

- [ ] **Step 5: Enforce per-level gates**

Require both routes to converge, final residual `<=2e-10`, finite errors,
unchanged owner query counts, and unchanged owner diagnostics. Stop before
the next level on failure.

- [ ] **Step 6: Build Release**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d harmonic_trace_correction_3d_test crossing_owner_restrict_3d_test restrict_owner_geometry_preprocessor_3d_test kfbim_phase_profile_3d_test -- /m:1
```

- [ ] **Step 7: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add Neumann crossing-owner comparison study"
```

---

### Task 4: Run the numerical comparison and report

**Files:**
- Generated: `output/neumann_value_trace_crossing_owner_3d/*.csv`

**Interfaces:**
- Consumes the Task 3 study route.
- Produces accepted N32/N64/N128 evidence and the recommendation.

- [ ] **Step 1: Run fresh focused Release tests**

```powershell
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\kfbim_phase_profile_3d_test.exe
```

- [ ] **Step 2: Run N32 and N64**

```powershell
$env:KFBIM_3D_NEUMANN_OWNER_STUDY_OUTPUT_DIR='output/neumann_value_trace_crossing_owner_3d'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-owner-study 32 64
```

Inspect convergence, iteration counts, errors, route mismatch, and the first
observed order before allowing `N=128`.

- [ ] **Step 3: Run the accepted final chain**

Use a clean output directory and execute:

```powershell
$env:KFBIM_3D_NEUMANN_OWNER_STUDY_OUTPUT_DIR='output/neumann_value_trace_crossing_owner_3d'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-owner-study 32 64 128
```

- [ ] **Step 4: Verify complete results**

Require exactly six summary rows, zero failed rows, two rows per level,
complete GMRES histories, and identical pre/post geometry-query counts.
Recompute both orders directly from the maximum errors.

- [ ] **Step 5: Report the result**

Report, per level and route, interior maximum error, observed order, density
error, exterior trace residual, route mismatch, GMRES iterations/final
residual, setup time, and solve time. State whether crossing-owner improves
accuracy or pose stability and whether it changes first-kind GMRES behavior.

- [ ] **Step 6: Final verification**

```powershell
git diff --check
git status --short
```

Generated output must remain ignored and the source worktree must be clean.
