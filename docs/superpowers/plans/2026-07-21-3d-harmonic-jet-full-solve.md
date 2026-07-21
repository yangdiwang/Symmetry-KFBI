# 3D Harmonic-Jet Full-Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the existing 3D app with shared-cubic-Cauchy Neumann and Dirichlet-normal-second-kind solves on torus, hollow-cylinder, and L-prism grids `N=16,32,64`.

**Architecture:** Keep every implementation change in `apps/neumann_exterior_zero_trace_3d.cpp`. Extend the existing panel-center harmonic-jet class so one degree-three coefficient field drives spread, tricubic restriction, exterior value trace, and exterior normal trace; then place both Krylov formulations and the convergence-study orchestration around that pipeline.

**Tech Stack:** C++17, Eigen, CGAL 5.6.2, project ZFFT Laplace solver, project GMRES, CMake/MSVC x64.

## Global Constraints

- Preserve the executable target `neumann_exterior_zero_trace_3d` and the old `geometry N` command syntax.
- Keep all algorithm changes in `apps/neumann_exterior_zero_trace_3d.cpp`; README and design/result documents may also change.
- Use a degree-three 3D harmonic Cauchy polynomial with 16 coefficients and retain the topology-filtered 48/28 sample counts.
- Spread and every restrict route must consume the same per-DOF Cauchy coefficient field.
- Use `4x4x4` tricubic interpolation at normal distances `0.5h,1.5h,2.5h,3.5h` on each side and a joint cubic normal fit.
- Use relative GMRES tolerance `2e-10` independent of `h` and report physical residuals separately.
- Preserve homogeneous Dirichlet data on the outer box.
- Final acceptance is `all 16 32 64`, producing 18 finite converged solve rows.

---

## File map

- Modify: `apps/neumann_exterior_zero_trace_3d.cpp` — all 3D pipeline, operators, manufactured data, CLI, diagnostics, and CSV output.
- Modify: `README.md` — final commands and the distinction between readiness and full solves.
- Create: `docs/superpowers/results/2026-07-21-3d-harmonic-jet-full-solve.md` — measured `16/32/64` results and conclusions.

### Task 1: Multi-level CLI and manufactured harmonic data

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:2117`

**Interfaces:**
- Produces: `double manufactured_u_3d(const Eigen::Vector3d&)`.
- Produces: `Eigen::Vector3d manufactured_gradient_3d(const Eigen::Vector3d&)`.
- Produces: `std::vector<int> parse_levels(int argc, char** argv)` behavior inside `main`.

- [ ] **Step 1: Verify the existing CLI rejects multiple levels**

Run:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe torus 16 32
```

Expected: nonzero exit with `too many command-line arguments`.

- [ ] **Step 2: Add the analytic manufactured solution**

Add beside the existing geometry helpers:

```cpp
double manufactured_u_3d(const Eigen::Vector3d& p)
{
    return std::exp(0.35 * p.x())
         * std::cos(0.21 * p.y())
         * std::cos(0.28 * p.z());
}

Eigen::Vector3d manufactured_gradient_3d(const Eigen::Vector3d& p)
{
    const double ex = std::exp(0.35 * p.x());
    return {
        0.35 * ex * std::cos(0.21 * p.y()) * std::cos(0.28 * p.z()),
       -0.21 * ex * std::sin(0.21 * p.y()) * std::cos(0.28 * p.z()),
       -0.28 * ex * std::cos(0.21 * p.y()) * std::sin(0.28 * p.z())};
}
```

- [ ] **Step 3: Accept one or more grid levels without breaking the old form**

Replace the single `N` parse with:

```cpp
std::vector<int> levels{16};
if (argc >= 3) {
    levels.clear();
    for (int arg = 2; arg < argc; ++arg) {
        const int level = std::stoi(argv[arg]);
        if (level < 16 || !is_power_of_two(level))
            throw std::invalid_argument(
                "every N must be a power of two and at least 16");
        levels.push_back(level);
    }
}
```

Loop over `levels` outside the geometry loop so `geometry N` and
`geometry N1 N2 ...` both work.

- [ ] **Step 4: Build and verify the new parse**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d -- /m
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe torus 16 32
```

Expected: build succeeds and both `N=16` and `N=32` readiness cases finish.

- [ ] **Step 5: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add multilevel 3d manufactured study input"
```

### Task 2: Shared cubic Cauchy coefficients and normal restriction

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1318`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1848`

**Interfaces:**
- Consumes: `PanelCenterCauchyFit3D`, `HarmonicJetField3D`, existing trace samples.
- Produces: `Eigen::VectorXd interior_trace(...) const`.
- Produces: `Eigen::VectorXd exterior_normal_trace(...) const`.
- Produces: `Eigen::VectorXd interior_normal_trace(...) const`.

- [ ] **Step 1: Add a failing readiness check for the exterior normal trace**

Extend `ReadinessResult` with `constant_exterior_normal_linf`.  In
`run_readiness_case`, construct `PanelCenterHarmonicJetKFBI3D`, evaluate
`[u]=1,[u_n]=0`, call `exterior_normal_trace`, and accumulate its infinity
norm.  Add the new value to console and CSV output.

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d -- /m
```

Expected: compilation fails because `exterior_normal_trace` is not defined.

- [ ] **Step 2: Make the canonical fit cubic**

Change the pipeline initializer from:

```cpp
fit_(cloud, stencils, h_, 4)
```

to:

```cpp
fit_(cloud, stencils, h_, 3)
```

Keep the existing topology-filtered stencil construction and 48/28 sample
counts unchanged.

- [ ] **Step 3: Centralize corrected one-sided samples**

Add a private helper with this exact interface:

```cpp
Eigen::MatrixXd continued_samples(
    const HarmonicJetField3D& field,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    bool continue_to_inside) const;
```

It must first evaluate all eight existing tricubic samples with
`field.coefficients`.  For exterior continuation, transform each inside
sample at `rho=-tau*h` with

```cpp
sample -= value_jump[center];
sample += tau * h_ * normal_jump[center];
```

For interior continuation, transform each outside sample at `rho=+tau*h`
with

```cpp
sample += value_jump[center];
sample += tau * h_ * normal_jump[center];
```

Return an `surface_size() x 8` matrix ordered as four inside layers followed
by four outside layers.

- [ ] **Step 4: Expose all four traces from the same samples**

Store both pseudoinverse rows in `build_joint_trace_fit()`:

```cpp
c0_weights_[q] = pinv(0, q);
c1_weights_[q] = pinv(1, q);
```

Implement value traces with `c0_weights_` and normal traces with
`c1_weights_/h_`:

```cpp
Eigen::VectorXd exterior_normal_trace(
    const HarmonicJetField3D& field,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump) const;

Eigen::VectorXd interior_normal_trace(
    const HarmonicJetField3D& field,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump) const;
```

`exterior_trace()` and the new `interior_trace()` use the same helper, ensuring
spread and restrict consume `field.coefficients` rather than refitting.

- [ ] **Step 5: Verify the constant trace and derivative identities**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d -- /m
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
```

Expected: all three readiness cases exit zero; constant exterior value and
normal traces are finite and below `1e-11`.

- [ ] **Step 6: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add shared cubic 3d normal restrict"
```

### Task 3: Execute and diagnose the 3D Neumann solve

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1629`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1848`

**Interfaces:**
- Consumes: `solve_exterior_zero_trace_neumann_3d(...)`.
- Produces: `SolveMetrics3D run_neumann_case(...)`.
- Produces: one `formulation="neumann_exterior_zero_trace"` result row.

- [ ] **Step 1: Record the failing black-box behavior**

Run:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe torus 16 |
  Select-String "\[neumann\]"
```

Expected: no matching output because the compiled solver is not invoked.

- [ ] **Step 2: Add shared numerical metrics and weighted helpers**

Add `SolveMetrics3D` with formulation, geometry, `N`, `h`, DOFs, GMRES status,
iterations, relative residual, target trace residual, route mismatch, value
and normal density errors, interior bulk `Linf/L2`, exterior bulk `Linf`,
compatibility shift, gauge, runtime, and refinement-order fields.

Add:

```cpp
double surface_weighted_mean(const SurfaceDofCloud& cloud,
                             const Eigen::VectorXd& values);
double vector_linf(const Eigen::VectorXd& values);
```

- [ ] **Step 3: Build compatible manufactured Neumann data**

At each surface DOF, sample `manufactured_u_3d(point)` and
`manufactured_gradient_3d(point).dot(normal)`.  Subtract the weighted mean of
the normal data before solving and store the removed mean.  Subtract the
weighted mean of the manufactured value jump for density comparison.

- [ ] **Step 4: Invoke the existing bordered GMRES solver**

Implement:

```cpp
SolveMetrics3D run_neumann_case(
    const std::string& geometry_name,
    int N,
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const SurfaceDofCloud& cloud,
    const CauchyStencilSet& stencils,
    double tolerance,
    int restart,
    int max_iterations);
```

Construct `PanelCenterHarmonicJetKFBI3D`, call
`solve_exterior_zero_trace_neumann_3d`, calculate the direct exterior trace,
the independent `interior_trace-value_jump` route, aligned density and bulk
errors, and print a single `[neumann]` line.

- [ ] **Step 5: Verify all three N=16 Neumann solves**

Run:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
```

Expected: three `[neumann]` rows, all `converged=1`, with finite errors and
relative residuals.

- [ ] **Step 6: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: execute 3d exterior-zero-trace Neumann solves"
```

### Task 4: Add the 3D Dirichlet normal-jump second-kind solve

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1692`

**Interfaces:**
- Produces: `ExteriorNormalTraceOperator3D : IKFBIOperator`.
- Produces: `DirichletNormalSolution3D solve_exterior_zero_normal_dirichlet_3d(...)`.
- Produces: `SolveMetrics3D run_dirichlet_normal_case(...)`.

- [ ] **Step 1: Record the failing black-box behavior**

Run:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe torus 16 |
  Select-String "\[dirichlet-normal\]"
```

Expected: no matching output because the Dirichlet operator does not exist.

- [ ] **Step 2: Implement the second-kind operator**

Add:

```cpp
class ExteriorNormalTraceOperator3D final : public IKFBIOperator {
public:
    explicit ExteriorNormalTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline);
    int problem_size() const override;
    void apply(const Eigen::VectorXd& normal_jump,
               Eigen::VectorXd& result) const override;
    Eigen::VectorXd right_hand_side(
        const Eigen::VectorXd& prescribed_value_jump) const;
private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
};
```

`apply(q)` evaluates `P_h(0,q)` and returns its direct exterior normal trace.
`right_hand_side(g_D)` returns minus the exterior normal trace of `P_h(g_D,0)`.

- [ ] **Step 3: Implement the solve wrapper**

Add `DirichletNormalSolution3D` containing normal jump, potential, residual,
GMRES residual history, iterations, and convergence.  Implement:

```cpp
DirichletNormalSolution3D solve_exterior_zero_normal_dirichlet_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_value_jump,
    double tolerance,
    int restart,
    int max_iterations);
```

Use an unbordered `GMRES` solve and evaluate the final field with
`P_h(g_D,q)`.

- [ ] **Step 4: Add physical and route diagnostics**

Implement `run_dirichlet_normal_case(...)` with the same construction inputs
as the Neumann runner.  Report:

```cpp
direct_exterior_normal = pipeline.exterior_normal_trace(...);
from_jump_exterior_normal = pipeline.interior_normal_trace(...) - q;
route_mismatch = direct_exterior_normal - from_jump_exterior_normal;
interior_value = pipeline.exterior_trace(...) + g_D;
```

Compare `q` with the analytic normal jump and compute interior/exterior bulk
errors without a constant shift.  Print one `[dirichlet-normal]` row.

- [ ] **Step 5: Verify all three N=16 Dirichlet solves**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d -- /m
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
```

Expected: three `[dirichlet-normal]` rows, all `converged=1`, with finite
exterior-normal residuals and route mismatches.

- [ ] **Step 6: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add 3d Dirichlet normal-jump second-kind solves"
```

### Task 5: CSV output, refinement orders, and full acceptance study

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:2040`
- Create: `docs/superpowers/results/2026-07-21-3d-harmonic-jet-full-solve.md`

**Interfaces:**
- Consumes: all `SolveMetrics3D` rows.
- Produces: `neumann_convergence.csv` and `dirichlet_normal_second_kind_convergence.csv`.

- [ ] **Step 1: Add the failing result-count assertion**

After the study loop, calculate the number of numerical rows and require:

```cpp
const std::size_t expected = geometries.size() * levels.size() * 2;
if (solve_results.size() != expected)
    throw std::runtime_error("3D solve result count mismatch");
```

Run the current app before wiring both pushes into `solve_results`.

Expected: nonzero exit with `3D solve result count mismatch`.

- [ ] **Step 2: Accumulate rows and calculate refinement orders**

For each fixed `(formulation,geometry)`, sort by `h` descending and calculate:

```cpp
order = std::log(previous_error / current_error)
      / std::log(previous_h / current_h);
```

Apply it independently to interior bulk `Linf` and `L2` errors.

- [ ] **Step 3: Write the two convergence CSV files**

Write all fields named in `SolveMetrics3D` with `std::setprecision(17)`.  Split
rows by formulation, preserve the existing readiness CSV, and include `N` in
per-level filenames where the existing readiness output expects it.

- [ ] **Step 4: Run the full study**

Run:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16 32 64
```

Expected: exit zero, nine readiness cases, nine Neumann rows, nine Dirichlet
rows, all 18 solve rows finite and converged, and every `N=64` interior bulk
`L2` error below its matching `N=16` error.

- [ ] **Step 5: Record measured results**

Create `docs/superpowers/results/2026-07-21-3d-harmonic-jet-full-solve.md`
with the exact command, dependency/build configuration, one table per
formulation containing geometry, level, iterations, residual, `Linf`, `L2`,
and observed order, plus a concise discussion of edge/corner behavior.

- [ ] **Step 6: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp docs/superpowers/results/2026-07-21-3d-harmonic-jet-full-solve.md
git commit -m "test: record 3d harmonic jet convergence study"
```

### Task 6: Documentation and final regression verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: final executable behavior and measured result files.
- Produces: reproducible build/run documentation.

- [ ] **Step 1: Update README behavior and commands**

Document that the 3D app now runs readiness, Neumann, and Dirichlet normal
second-kind solves.  Add the exact command:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16 32 64
```

State that the active route is cubic 3D harmonic Cauchy spread plus
`4x4x4` tricubic interpolation and joint cubic normal restriction.

- [ ] **Step 2: Run a clean incremental Release build**

Run:

```powershell
cmake --build build --config Release -- /m
```

Expected: exit zero and all app targets link.

- [ ] **Step 3: Re-run smoke and full studies**

Run:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16 32 64
```

Expected: both commands exit zero with all requested rows converged.

- [ ] **Step 4: Validate files and repository state**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended source, README, plan, and result
changes are present before the final commit.

- [ ] **Step 5: Commit**

```powershell
git add README.md apps/neumann_exterior_zero_trace_3d.cpp docs/superpowers/results/2026-07-21-3d-harmonic-jet-full-solve.md
git commit -m "docs: describe completed 3d harmonic jet solves"
```
