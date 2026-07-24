# 3D Owner-Only Restrict Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a retained owner-only high-resolution probe and use it to compute the L-prism `N=256` crossing-owner result.

**Architecture:** Parameterize the existing normal-restrict probe with a route-set flag.  The full A/B entry point retains all three routes and its current output directory; the new CLI selects only `JointTricubicCrossingOwner`, omits the exterior-only object, writes a separate directory, and skips the A/B verdict.

**Tech Stack:** C++17, CMake/MSBuild, Eigen, existing KFBI3D app pipeline and CSV diagnostics.

## Global Constraints

- Keep `--restrict-probe` behavior unchanged.
- Add `--restrict-probe-owner [N ...]`.
- Preserve the GMRES tolerance `2e-10`, restart `0`, and cap `160`.
- Preserve all crossing-owner diagnostics and geometry-query invariants.
- Do not modify the KFBI operator, geometry, Cauchy fit, or restrict formulas.

---

### Task 1: Add and validate the owner-only entry point

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:3923-4226`
- Test: executable-level CLI checks using `build/apps/Release/neumann_exterior_zero_trace_3d.exe`

**Interfaces:**
- Consumes: `run_normal_restrict_causal_probe(std::vector<int>)` and `ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner`.
- Produces: `run_normal_restrict_causal_probe(std::vector<int>, bool owner_only)` and CLI `--restrict-probe-owner`.

- [ ] **Step 1: Run the missing CLI to verify RED**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe-owner 16
```

Expected: nonzero exit because `--restrict-probe-owner` is treated as an unknown geometry selection.

- [ ] **Step 2: Implement the minimal route filter**

Change the probe signature to:

```cpp
int run_normal_restrict_causal_probe(std::vector<int> levels,
                                     bool owner_only)
```

Select the output leaf with:

```cpp
const char* output_leaf = owner_only
    ? "dirichlet_normal_restrict_crossing_owner_3d"
    : "dirichlet_normal_restrict_causal_probe_3d";
```

Construct the pipeline with `build_exterior_only_restrict = !owner_only` and
`build_crossing_owner_restrict = true`.  Use a route vector containing only
`JointTricubicCrossingOwner` for owner-only mode and all existing modes
otherwise.  Print the A/B hypothesis only in full mode.

Recognize `--restrict-probe-owner` in `main`, exclude it from geometry
selection, add it to usage, and call:

```cpp
return run_normal_restrict_causal_probe(levels, restrict_probe_owner);
```

- [ ] **Step 3: Build and verify GREEN at N=16**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d -- /m:2
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe-owner 16
```

Expected: exit `0`; `summary.csv` contains three data rows and every route is
`joint_tricubic_crossing_owner`.

- [ ] **Step 4: Verify the existing A/B entry point**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe 16
```

Expected: exit `0`; `summary.csv` contains nine data rows covering all three
routes.

- [ ] **Step 5: Commit the implementation**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add owner-only 3d restrict probe"
```

### Task 2: Run and verify N=256

**Files:**
- Generate: `output/dirichlet_normal_restrict_crossing_owner_3d/*.csv`
- Read: `D:/KFBI-Symmetric/output/dirichlet_normal_restrict_causal_probe_3d_N128/summary.csv`

**Interfaces:**
- Consumes: `--restrict-probe-owner 256`.
- Produces: three `N=256` owner-route rows and `128 -> 256` convergence orders.

- [ ] **Step 1: Run the high-resolution probe**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe-owner 256
```

Expected: three completed, converged crossing-owner rows.

- [ ] **Step 2: Compute convergence**

For each pose, compute:

```text
p = log2(physical_interior_linf(N=128) /
         physical_interior_linf(N=256))
```

Report `physical_iterations`, `common_iterations`, the final residual, setup
time, route time, and whether `p >= 2` within numerical tolerance.

- [ ] **Step 3: Verify diagnostics**

Check that the summary has exactly three owner rows, all physical/common solves
converged, and owner classification/audit counts are conserved.  Run
`git diff --check` and report any remaining untracked files without adding
them.

