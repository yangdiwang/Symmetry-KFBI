# 3D Neumann Crossing-Owner Value-Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a crossing-owner exterior-value restriction route to the 3D Neumann solver and compare it with the unchanged legacy route on three L-prism poses at `N=32,64,128`.

**Architecture:** Extract the scalar Cauchy-correction ownership choice into a small tested helper, then pass one trace-route enum through the Neumann operator, RHS, solve and diagnostics. A dedicated study route builds one owner-enabled pipeline per pose/level and runs legacy and crossing-owner solves against it, ensuring geometry work remains outside GMRES.

**Tech Stack:** C++17, Eigen, existing KFBI3D/NURBS infrastructure, MSVC/CMake, CSV diagnostics.

## Global Constraints

- Work directly on `main`, preserving unrelated untracked files.
- Keep Neumann unknown `[u]`, prescribed `[u_n]`, exterior value trace zero, compatibility correction and scalar mean constraint unchanged.
- Keep Cauchy degree `3`, G1-nearest `48/28`, tricubic interpolation, GMRES tolerance `2e-10`, cap `80`.
- `JointTricubicCauchy` must remain bitwise/order compatible with the current center-owned correction loop.
- `JointTricubicCrossingOwner` must use only precomputed owner templates; no geometry query may occur during GMRES.
- Value traces use `c0_weights_` with scale `1`; normal traces retain `c1_weights_/h`.
- Operator apply, RHS, final exterior trace and route mismatch must all use the same selected route.
- Generated experiment outputs remain untracked under `output/neumann_value_trace_crossing_owner_3d`.

---

### Task 1: Tested trace-correction ownership kernel

**Files:**
- Create: `apps/harmonic_trace_correction_3d.hpp`
- Create: `apps/harmonic_trace_correction_3d.cpp`
- Create: `apps/harmonic_trace_correction_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`

**Interfaces:**
- Produces `HarmonicTraceCorrectionMode3D { CenterOwned, CrossingOwned }`.
- Produces `HarmonicTraceOwnerTerm3D { int owner_dof; Eigen::VectorXd evaluation; }`.
- Produces `apply_harmonic_trace_correction_3d(center_dof, coefficients, center_evaluation, owner_terms, mode)`.
- Consumes only precomputed evaluations; performs no geometry operation.

- [ ] **Step 1: Write the failing unit test**

Use literal coefficients and evaluations so center-owned returns `11.0` and crossing-owned returns `23.0`. Add invalid-index and dimension tests. The production mutation caught is selecting the center row when crossing ownership was requested.

- [ ] **Step 2: Run the test target and verify RED**

```powershell
cmake --build build --config Release --target harmonic_trace_correction_3d_test -- /m:1 /nr:false
```

Expected: compilation or link failure because the new API is absent.

- [ ] **Step 3: Implement the minimal kernel**

```cpp
if (mode == HarmonicTraceCorrectionMode3D::CenterOwned)
    return center_evaluation.dot(coefficients.row(center_dof).transpose());
double result = 0.0;
for (const HarmonicTraceOwnerTerm3D& term : owner_terms)
    result += term.evaluation.dot(coefficients.row(term.owner_dof).transpose());
return result;
```

Validate all indices and dimensions before evaluating.

- [ ] **Step 4: Integrate the kernel into `continued_samples`**

Replace only the legacy/owner correction branch. Preserve the surrounding `q=0..63` grid-potential accumulation and jump-continuation loops.

- [ ] **Step 5: Verify GREEN and existing owner tests**

Run the new test and `crossing_owner_restrict_3d_test`; both must exit zero.

- [ ] **Step 6: Commit**

```powershell
git add apps/harmonic_trace_correction_3d.* apps/CMakeLists.txt apps/neumann_exterior_zero_trace_3d.cpp
git commit -m 'refactor: isolate 3d harmonic trace ownership'
```

---

### Task 2: Neumann crossing-owner operator route

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Test: `apps/harmonic_trace_correction_3d_test.cpp`

**Interfaces:**
- Produces `ExteriorValueRestrictMode3D { JointTricubicCauchy, JointTricubicCrossingOwner }`.
- Produces `PanelCenterHarmonicJetKFBI3D::exterior_trace(..., ExteriorValueRestrictMode3D mode)` while retaining the legacy overload.
- Adds mode parameters to `ExteriorZeroTraceOperator3D`, `solve_exterior_zero_trace_neumann_3d` and `run_neumann_case`.

- [ ] **Step 1: Write the failing route test**

Add a literal mapping test proving `JointTricubicCauchy -> CenterOwned` and `JointTricubicCrossingOwner -> CrossingOwned`. The mutation caught is changing the legacy default or ignoring the crossing mode.

- [ ] **Step 2: Verify RED**

Build the focused target and confirm failure because the value-route mapping/API is missing.

- [ ] **Step 3: Add the exterior-value overload**

The legacy overload delegates explicitly to `JointTricubicCauchy`. The new overload calls:

```cpp
recover_trace(
    continued_samples(field, value_jump, normal_jump, false, mapped_mode),
    c0_weights_, 1.0);
```

- [ ] **Step 4: Thread the mode through the full Neumann equation**

Store the mode in `ExteriorZeroTraceOperator3D`; use it in both `apply` and `right_hand_side`. Pass it through solve/run functions and use it again for direct exterior trace and route-mismatch diagnostics. Keep existing call sites compiling with legacy mode.

- [ ] **Step 5: Verify GREEN and legacy behavior**

Run the focused test, Release build of `neumann_exterior_zero_trace_3d`, and one legacy `l_prism 32` smoke. Compare its CSV row with the saved legacy baseline within existing tolerances.

- [ ] **Step 6: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/harmonic_trace_correction_3d_test.cpp
git commit -m 'feat: add Neumann crossing-owner value trace'
```

---

### Task 3: Dedicated Neumann owner study and diagnostics

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Create: `docs/superpowers/results/2026-07-25-3d-neumann-crossing-owner-value-trace.md`

**Interfaces:**
- Produces CLI `--neumann-owner-study [N ...]`.
- Produces `summary.csv`, `gmres_residuals.csv`, and `owner_diagnostics.csv`.

- [ ] **Step 1: Verify the CLI is RED**

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-owner-study 16
```

Expected: nonzero exit because the mode is not recognized.

- [ ] **Step 2: Implement the study driver**

Select rigid cases `baseline`, `rot_axis123_17deg`, and `rot_axis123_17deg_t_xyz_1`. Build one owner-enabled pipeline per case/level, snapshot owner query count and deterministic correction fingerprint, then run both value routes with identical data.

- [ ] **Step 3: Write complete diagnostics incrementally**

After each route, flush a summary row and residual history. Record case, N, route, convergence, iterations, final residual, density/interior errors and orders, exterior condition, operator residual, route mismatch, setup/solve time, owner queries and fingerprint before/after GMRES.

- [ ] **Step 4: Verify GREEN at N=16 and N=32**

Require finite complete rows, convergence within 80 iterations, final residual at most `2e-10`, unchanged query/fingerprint values, and matching route-independent geometry metadata.

- [ ] **Step 5: Run the formal matrix**

Run `N=32,64,128` for all three poses. If a route fails, preserve its rows and stop only dependent finer levels; do not hide the negative result.

- [ ] **Step 6: Analyze and document**

Compute observed orders per case/route and compare GMRES/setup/solve time. State whether crossing ownership changes numerical behavior and whether a Hybrid owner preprocessor was integrated or remains future performance work.

- [ ] **Step 7: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp docs/superpowers/results/2026-07-25-3d-neumann-crossing-owner-value-trace.md
git commit -m 'test: compare Neumann crossing-owner value trace'
```

---

### Task 4: Full verification and review

**Files:**
- Verify all files changed by Tasks 1-3.

**Interfaces:**
- Consumes all preceding commits; produces reviewed, reproducible evidence.

- [ ] **Step 1: Build serial Release**

```powershell
cmake --build build --config Release -- /m:1 /nr:false
```

- [ ] **Step 2: Run focused and geometry regressions**

Run `harmonic_trace_correction_3d_test`, `crossing_owner_restrict_3d_test`, `native_nurbs_surface_3d_test`, and the existing L-prism legacy smoke.

- [ ] **Step 3: Audit generated CSVs**

Check row counts, finite values, route pairs, residual limits, query/fingerprint invariance and order calculations independently with PowerShell `Import-Csv`.

- [ ] **Step 4: Review the complete diff**

Require zero Critical/Important findings and resolve any regression in the Neumann compatibility/mean equation before completion.

- [ ] **Step 5: Final whitespace/status check**

Run `git diff --check` and confirm only the two pre-existing unrelated untracked documents remain outside this task.
