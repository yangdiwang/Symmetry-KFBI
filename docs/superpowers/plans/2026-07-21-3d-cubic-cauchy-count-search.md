# 3D Cubic Cauchy Sample-count Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add controlled 3D Cauchy sample-count overrides and find the most stable same-patch count pair for the L-prism Dirichlet normal-jump formulation.

**Architecture:** Parse two optional environment variables in the existing 3D app, pass the counts into the existing stencil builder, and isolate each non-default run in a count-named output directory. Keep the cubic Cauchy and cubic restrict formulas unchanged and make their degrees explicit in diagnostics. Use a staged numerical sweep rather than embedding an optimizer in production code.

**Tech Stack:** C++17, Eigen, CMake/MSBuild x64, PowerShell, existing native-NURBS 3D app and GMRES solver.

## Global Constraints

- Work directly on `main`.
- Optimize the Dirichlet normal-jump second-kind solve; Neumann is diagnostic only.
- Use `same_patch` for every count-search run.
- Keep Cauchy, Cartesian interpolation, and normal restrict degree equal to three.
- Keep GMRES relative tolerance at `2e-10`.
- Keep the default sample counts and output paths backward compatible.

---

### Task 1: Add tested count overrides and degree diagnostics

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `.github/workflows/build-and-smoke.yml`

**Interfaces:**
- Produces: `KFBIM_3D_CAUCHY_VALUE_COUNT`, default `48`.
- Produces: `KFBIM_3D_CAUCHY_NORMAL_COUNT`, default `28`.
- Produces startup diagnostics `cauchy_degree=3`,
  `restrict_grid_degree=3`, `restrict_normal_degree=3`, and
  `cauchy_counts=<value>/<normal>`.

- [ ] **Step 1: Add a failing default-diagnostic assertion**

Extend the existing 3D CI smoke checks to require
`cauchy_degree=3 restrict_grid_degree=3 restrict_normal_degree=3` and
`cauchy_counts=48/28`.

- [ ] **Step 2: Run the old executable and verify RED**

```powershell
$text = & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16 2>&1 | Out-String
if ($text -notmatch 'cauchy_counts=48/28') { throw 'missing count diagnostic' }
```

Expected: failure with `missing count diagnostic`.

- [ ] **Step 3: Implement strict integer parsing**

Add a helper that reads the complete environment string with `std::stoi`,
rejects trailing characters, and rejects non-positive values. Parse both
counts in `main` and reject `normal_count > value_count`. Pass both through
`run_readiness_case` to `build_cauchy_stencils` instead of the compile-time
defaults.

- [ ] **Step 4: Add degree and count diagnostics**

Define named constants for cubic Cauchy, grid interpolation, and normal fit
degrees, use the Cauchy constant at the `PanelCenterCauchyFit3D` construction,
and print all three degrees plus the requested count pair. Keep the existing
tricubic interpolation and joint cubic trace-fit code unchanged.

- [ ] **Step 5: Isolate count outputs**

After applying the policy output directory, append `v<value>_n<normal>` when
the pair differs from `48/28`. This makes the same-patch sweep paths such as
`output/neumann_exterior_zero_trace_3d/same_patch/v16_n10`.

- [ ] **Step 6: Verify valid and invalid inputs**

Build the app. Run same-patch `16/10` at `N=16` and require the startup output
to contain `cauchy_counts=16/10`. Run values `0`, `-1`, `abc`, and a pair
`10/12`; require each process to exit nonzero with a message naming the
invalid count constraint. Clear all environment variables afterward.

- [ ] **Step 7: Run regressions and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
```

Expected: build and focused tests pass and the default numerical route still
reports `48/28`.

Commit:

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp .github/workflows/build-and-smoke.yml
git commit -m "feat: configure 3d Cauchy sample counts"
```

### Task 2: Execute the staged count search

**Files:**
- No production files.
- Outputs: `output/neumann_exterior_zero_trace_3d/same_patch/v*_n*/*.csv`

**Interfaces:**
- Consumes: count environment variables from Task 1.
- Produces: a ranked list of full-rank, converged count pairs.

- [ ] **Step 1: Run Stage 1 paired candidates**

For each of `10/6`, `12/8`, `16/10`, `20/12`, `24/16`, `32/20`, and
`48/28`, run `l_prism 16 32` under `same_patch`. A rank-deficient candidate
is an experimental rejection, not a code failure.

- [ ] **Step 2: Parse and screen Stage 1**

For each completed pair, extract the two Dirichlet rows and readiness rows.
Reject nonconvergence, nondecreasing density/interior errors, non-finite
metrics, or a failed SVD run. Rank remaining pairs by maximum GMRES, then
iteration growth, errors, p95 condition, radius, and sample count.

- [ ] **Step 3: Run the local cross sweep**

Take the best two adjacent paired candidates and run the cross products of
their value and normal counts at `N=16,32`, adding the immediately neighboring
listed count if the best pair lies at one end. Apply the same screening.

- [ ] **Step 4: Select three fine-grid finalists**

Choose the three distinct full-rank pairs with the lowest stable Dirichlet
iteration scores. Include `48/28` as a control even if it is not top three.

### Task 3: Verify finalists and record results

**Files:**
- Create: `docs/superpowers/results/2026-07-21-3d-cubic-cauchy-count-search.md`

**Interfaces:**
- Consumes: finalists from Task 2.
- Produces: the recommended default count pair for the Dirichlet same-patch
  route; this task does not change the global defaults.

- [ ] **Step 1: Run finalists at all three levels**

Run each finalist and `48/28` at `N=16,32,64`. Confirm all Dirichlet rows
converge and both errors decrease twice.

- [ ] **Step 2: Enforce fine-grid order and rank**

Reject a pair when either `N=32` to `N=64` density or interior order is below
two. Rank accepted pairs by maximum GMRES, positive iteration growth, `N=64`
errors, p95 condition, radius, and sample count.

- [ ] **Step 3: Write exact numerical results**

Record the Stage 1 rejections, local sweep, finalist tables, Neumann
diagnostics, exact commands, and comparison against same-patch `48/28` and
topological-nearest `48/28`. State explicitly that restrict remained cubic.

- [ ] **Step 4: Final verification and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git status --short --branch
```

Commit the result document with:

```powershell
git add docs/superpowers/results/2026-07-21-3d-cubic-cauchy-count-search.md
git commit -m "test: optimize cubic 3d Cauchy sample counts"
```
