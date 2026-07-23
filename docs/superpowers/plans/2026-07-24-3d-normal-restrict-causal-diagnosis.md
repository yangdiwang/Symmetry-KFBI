# 3D Normal-Restrict Causal Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a falsifiable diagnostic that isolates the 3D exterior-normal restrict and then changes only that restrict in a controlled GMRES A/B experiment.

**Architecture:** Extract the existing harmonic polynomial basis into a reusable app module, then build an independent exterior-only cubic harmonic least-squares restrict. The current harmonic-jet pipeline selects either the unchanged joint-tricubic/Cauchy route or the reference route, while a dedicated CLI mode runs exact-grid isolation, common-RHS GMRES, physical GMRES, and feature-edge localization.

**Tech Stack:** C++17, Eigen SVD, CMake/Visual Studio x64, existing KFBI3D native NURBS, ZFFT bulk solve, and GMRES.

## Global Constraints

- Work directly on the current `main` branch, as previously approved.
- Study only the L-prism cases `baseline`, `rot_axis123_17deg`, and `rot_axis123_17deg_t_xyz_1`.
- Run `N=32,64` first; use `N=128` only if the decision rule remains inconclusive.
- Keep `g1_nearest`, degree-three Cauchy fitting, 48 value samples, 28 normal samples, spread, FFT bulk solve, and manufactured data fixed.
- Use GMRES tolerance `2e-10`, zero initial guess, no restart, and diagnostic maximum 160.
- The legacy app and `--rigid-study` retain the current joint-tricubic route and output unchanged.
- Diagnostic output is isolated under `output/dirichlet_normal_restrict_causal_probe_3d`.
- Never fall back from an invalid reference stencil to the current restrict.
- Do not stage or modify the two unrelated untracked same-patch design/plan files.

---

### Task 1: Reusable Harmonic Polynomial Space

**Files:**
- Create: `apps/harmonic_polynomial_space_3d.hpp`
- Create: `apps/harmonic_polynomial_space_3d.cpp`
- Create: `apps/harmonic_polynomial_space_3d_test.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:520-650`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: polynomial degree and local coordinates.
- Produces:

```cpp
namespace kfbim::app3d {

class HarmonicPolynomialSpace3D {
public:
    explicit HarmonicPolynomialSpace3D(int degree);
    int degree() const noexcept;
    int dimension() const noexcept;
    Eigen::VectorXd basis(double x, double y, double z) const;
    Eigen::MatrixXd gradient(double x, double y, double z) const;
};

Eigen::MatrixXd svd_pseudoinverse_3d(
    const Eigen::MatrixXd& matrix,
    double relative_cutoff);

} // namespace kfbim::app3d
```

- [ ] **Step 1: Write the failing harmonic-space test and register its target**

Add tests for dimensions `4,9,16` at degrees `1,2,3`, harmonicity of every
basis vector by centered second differences, and
`pinv * matrix == identity` for a full-column-rank matrix:

```cpp
const app3d::HarmonicPolynomialSpace3D cubic(3);
require(cubic.dimension() == 16, "cubic harmonic dimension");
const double eps = 1.0e-4;
for (int k = 0; k < cubic.dimension(); ++k) {
    const auto value = [&](double x, double y, double z) {
        return cubic.basis(x, y, z)[k];
    };
    const double laplacian =
        (value(eps,0,0) - 2.0*value(0,0,0) + value(-eps,0,0))/(eps*eps)
      + (value(0,eps,0) - 2.0*value(0,0,0) + value(0,-eps,0))/(eps*eps)
      + (value(0,0,eps) - 2.0*value(0,0,0) + value(0,0,-eps))/(eps*eps);
    require(std::abs(laplacian) < 2.0e-6, "basis is harmonic");
}
```

Add `harmonic_polynomial_space_3d.cpp` to `kfbim_3d_app_geometry` and add
`harmonic_polynomial_space_3d_test`.

- [ ] **Step 2: Build the new target and verify RED**

Run:

```powershell
cmake --build build --config Release --target harmonic_polynomial_space_3d_test
```

Expected: compilation fails because `harmonic_polynomial_space_3d.hpp` and its
symbols do not exist.

- [ ] **Step 3: Move the existing implementation into the new module**

Move `PolynomialPower3D`, integer power evaluation, null-space construction,
`basis`, `gradient`, and SVD pseudoinverse without changing their formulas.
Use:

```cpp
const int expected_dimension = (degree_ + 1) * (degree_ + 1);
transform_ = svd.matrixV().rightCols(expected_dimension);
```

Replace the anonymous definitions in the 3D app with:

```cpp
#include "harmonic_polynomial_space_3d.hpp"
using app3d::HarmonicPolynomialSpace3D;
using app3d::svd_pseudoinverse_3d;
```

and rename both existing pseudoinverse calls to
`svd_pseudoinverse_3d(...)`.

- [ ] **Step 4: Verify GREEN and the existing app**

Run:

```powershell
cmake --build build --config Release --target harmonic_polynomial_space_3d_test neumann_exterior_zero_trace_3d
.\build\apps\Release\harmonic_polynomial_space_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --help
```

Expected: the test exits 0 and help still lists the legacy and rigid-study
routes.

- [ ] **Step 5: Commit the extraction**

```powershell
git add -- apps/harmonic_polynomial_space_3d.hpp apps/harmonic_polynomial_space_3d.cpp apps/harmonic_polynomial_space_3d_test.cpp apps/neumann_exterior_zero_trace_3d.cpp apps/CMakeLists.txt
git commit -m "refactor: share 3d harmonic polynomial space"
```

---

### Task 2: Independent Exterior-Only Cubic Restrict

**Files:**
- Create: `apps/exterior_only_cubic_normal_restrict_3d.hpp`
- Create: `apps/exterior_only_cubic_normal_restrict_3d.cpp`
- Create: `apps/exterior_only_cubic_normal_restrict_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: Cartesian grid, native surface DOF cloud, domain labels, and a
  bulk potential.
- Produces:

```cpp
namespace kfbim::app3d {

struct ExteriorOnlyCubicNormalStencil3D {
    std::vector<int> grid_ids;
    Eigen::VectorXd weights;
    double condition = 0.0;
};

ExteriorOnlyCubicNormalStencil3D
build_exterior_only_cubic_normal_stencil_3d(
    const Eigen::Vector3d& center,
    const Eigen::Vector3d& tangent1,
    const Eigen::Vector3d& tangent2,
    const Eigen::Vector3d& normal,
    double h,
    const std::vector<std::pair<int, Eigen::Vector3d>>& exterior_candidates,
    int sample_count = 48);

class ExteriorOnlyCubicNormalRestrict3D {
public:
    ExteriorOnlyCubicNormalRestrict3D(
        const CartesianGrid3D& grid,
        const GridPair3D& grid_pair,
        const SurfaceDofCloud3D& cloud,
        int sample_count = 48,
        int maximum_index_radius = 4);

    Eigen::VectorXd apply(const Eigen::VectorXd& potential) const;
    const std::vector<ExteriorOnlyCubicNormalStencil3D>& stencils() const;
};

} // namespace kfbim::app3d
```

- [ ] **Step 1: Write polynomial and failure tests**

Generate a rotated orthonormal frame and at least 64 candidate points. For
each harmonic polynomial below, sample physical values and verify the
recovered physical normal derivative:

```text
1
x
z
x*y
x*x-y*y
x*x*x-3*x*y*y
z*(x*x-y*y)
```

Use:

```cpp
const auto stencil = build_exterior_only_cubic_normal_stencil_3d(
    center, t1, t2, normal, h, candidates, 48);
double recovered = 0.0;
for (int k = 0; k < stencil.weights.size(); ++k)
    recovered += stencil.weights[k] * values[k];
require(std::abs(recovered - exact_normal) < 2.0e-11,
        "cubic harmonic normal derivative");
```

Also require throws for fewer than 16 candidates, rank-deficient coplanar
candidates, non-positive `h`, and non-orthonormal frames.

- [ ] **Step 2: Build and verify RED**

Run:

```powershell
cmake --build build --config Release --target exterior_only_cubic_normal_restrict_3d_test
```

Expected: compilation fails because the reference restrict is missing.

- [ ] **Step 3: Implement deterministic candidate selection and weights**

For each DOF, scan index offsets within `maximum_index_radius`, keep only
nodes satisfying:

```cpp
grid_pair.domain_label(node) == 0
```

Sort by `(squared_distance, node_id)` and retain exactly 48. In the stencil
builder, form:

```cpp
xi = {
    (point-center).dot(tangent1)/h,
    (point-center).dot(tangent2)/h,
    (point-center).dot(normal)/h
};
design.row(k) = space.basis(xi.x(), xi.y(), xi.z()).transpose();
sqrt_weight[k] = std::sqrt(1.0 / std::pow(0.35 + xi.norm(), 2.0));
weighted = sqrt_weight.asDiagonal() * design;
```

Reject unless the 16 singular values are finite and
`sigma_min > 3e-12 * sigma_max`. Precompute:

```cpp
const Eigen::RowVectorXd derivative =
    space.gradient(0.0, 0.0, 0.0).row(2) / h;
stencil.weights =
    (derivative * svd_pseudoinverse_3d(weighted, 3.0e-12)
     * sqrt_weight.asDiagonal()).transpose();
```

`apply` checks `potential.size() == grid.num_dofs()` and performs only the
precomputed dot products.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
cmake --build build --config Release --target exterior_only_cubic_normal_restrict_3d_test
.\build\apps\Release\exterior_only_cubic_normal_restrict_3d_test.exe
```

Expected: all polynomial, rotation, translation, validation, and
exterior-label tests pass.

- [ ] **Step 5: Commit the independent restrict**

```powershell
git add -- apps/exterior_only_cubic_normal_restrict_3d.hpp apps/exterior_only_cubic_normal_restrict_3d.cpp apps/exterior_only_cubic_normal_restrict_3d_test.cpp apps/CMakeLists.txt
git commit -m "feat: add exterior-only cubic normal restrict"
```

---

### Task 3: Switch Only the Restrict in the Existing Operator

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:880-1470`

**Interfaces:**
- Consumes: `ExteriorOnlyCubicNormalRestrict3D`.
- Produces:

```cpp
enum class ExteriorNormalRestrictMode3D {
    JointTricubicCauchy,
    ExteriorOnlyHarmonicCubic
};

Eigen::VectorXd PanelCenterHarmonicJetKFBI3D::exterior_normal_trace(
    const HarmonicJetField3D& field,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    ExteriorNormalRestrictMode3D mode) const;
```

- [ ] **Step 1: Add a compile-failing mode selection test path**

Add the enum and pass it through the constructors and solve helper:

```cpp
ExteriorNormalTraceOperator3D(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    ExteriorNormalRestrictMode3D mode);

ExteriorNormalTraceSolution3D solve_exterior_zero_normal_dirichlet_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_value_jump,
    ExteriorNormalRestrictMode3D mode,
    double tolerance,
    int restart,
    int max_iterations);
```

Keep the legacy caller explicit:

```cpp
ExteriorNormalRestrictMode3D::JointTricubicCauchy
```

Call a not-yet-implemented overload of `exterior_normal_trace` so compilation
fails at this interface.

- [ ] **Step 2: Build and verify RED**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
```

Expected: compilation fails because the mode-aware trace is not implemented.

- [ ] **Step 3: Implement the mode-aware trace**

Construct one `ExteriorOnlyCubicNormalRestrict3D` in the pipeline constructor.
Switch only at the final recovery boundary:

```cpp
if (mode == ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic)
    return exterior_only_restrict_.apply(field.potential);
return recover_trace(
    continued_samples(field, value_jump, normal_jump, false),
    c1_weights_, 1.0 / h_);
```

Use the same `mode` for `ExteriorNormalTraceOperator3D::apply` and
`right_hand_side`. Do not change `evaluate`, crossing rows, correction RHS,
bulk solve, Cauchy coefficients, GMRES, or legacy value-trace code.

- [ ] **Step 4: Build and run regression smoke checks**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --rigid-study 16
```

Expected: both existing invocations exit 0 and continue to report the current
joint-tricubic route.

- [ ] **Step 5: Commit the operator switch**

```powershell
git add -- apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: switch 3d exterior normal restrict diagnostically"
```

---

### Task 4: Exact-Grid Probe, Localization, and Diagnostic CLI

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:120-300`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:800-900`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1450-1750`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1750-end`

**Interfaces:**
- Consumes: both restrict modes and the three approved rigid cases.
- Produces:
  - CLI `--restrict-probe [N ...]`;
  - `summary.csv`;
  - `dof_diagnostics.csv`;
  - `gmres_residuals.csv`;
  - `edge_distance_bins.csv`.

- [ ] **Step 1: Add the failing CLI branch**

Parse before legacy positional arguments:

```cpp
const bool restrict_probe =
    argc >= 2 && std::string(argv[1]) == "--restrict-probe";
```

Default levels to `{32,64}`, require powers of two at least 16, and call the
not-yet-defined:

```cpp
return run_normal_restrict_causal_probe(levels);
```

Add the invocation to `--help`.

- [ ] **Step 2: Build and verify RED**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
```

Expected: link or compile failure for
`run_normal_restrict_causal_probe`.

- [ ] **Step 3: Add exact-grid and smooth-grid isolation**

Expose a pipeline helper that combines supplied grid values with fitted jump
coefficients:

```cpp
HarmonicJetField3D field_from_grid_and_jumps(
    const Eigen::VectorXd& potential,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump) const
{
    return {potential, fit_.coefficients(value_jump, normal_jump)};
}
```

For the piecewise exact field use `u_T` on inside nodes and zero outside. For
the smooth control use `u_T` on every node and zero jumps; subtract the exact
`grad(u_T).dot(normal)` from both recovered traces.

For each mode compute:

```cpp
exact_grid_error = restrict(piecewise_exact, value_data, exact_normal);
smooth_grid_error =
    restrict(smooth_exact, zero_jump, zero_jump) - exact_normal;
exact_equation_residual =
    operator.apply(exact_normal) - operator.right_hand_side(value_data);
```

Store `Linf`, surface-weighted RMS, and per-DOF vectors.

- [ ] **Step 4: Add common-RHS and physical GMRES probes**

Create the deterministic common right-hand side:

```cpp
rhs[q] = std::sin(0.73 * static_cast<double>(q + 1))
       + 0.25 * std::cos(0.19 * static_cast<double>(q + 1));
rhs /= rhs.norm();
```

For each mode run zero-initial-guess GMRES on this same vector. Separately run
the physical Dirichlet solve with the same mode used in both `apply` and
`right_hand_side`. Save every relative residual with:

```text
case_id,N,route,solve_kind,iteration,relative_residual
```

- [ ] **Step 5: Add feature-edge localization**

Retain `triangulation.feature_edges` in `GeometryBundle`. Compute point-to-
segment distance for every surface DOF and divide by `h`. Add a pipeline
accessor returning, per DOF, the count of the current trace template's 512
support entries whose domain label differs from the desired side.

Write per-DOF values and aggregate the fixed bins:

```cpp
const std::array<double, 5> upper{{1.0, 2.0, 4.0, 8.0,
    std::numeric_limits<double>::infinity()}};
```

Each bin reports count, current/reference exact-grid `Linf` and weighted RMS,
plus current/reference exact-equation-residual `Linf` and weighted RMS.

- [ ] **Step 6: Implement incremental output and the decision fields**

Write accumulated files after every completed route. `summary.csv` contains:

```text
case_id,N,route,restrict_condition_median,restrict_condition_p95,
restrict_condition_max,exact_grid_linf,exact_grid_wrms,smooth_grid_linf,
smooth_grid_wrms,exact_equation_linf,exact_equation_wrms,
physical_converged,physical_iterations,physical_final_residual,
physical_interior_linf,common_converged,common_iterations,
common_final_residual,seconds
```

After all `N=32,64` rows exist, report the registered excess-iteration test:

```cpp
reference_excess <= 0.5 * current_excess
```

for both physical and common RHS. If the current route does not reproduce its
known inflation, or if the reference route is rank deficient, inaccurate on
the baseline by more than `2x`, or fails the smooth-control rotation check,
mark the restrict hypothesis `not_proven`.

- [ ] **Step 7: Run the N=16 CLI smoke**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe 16
```

Expected: six route rows complete, all three CSV detail files are non-empty,
every reported value is finite, and the command exits without altering the
rigid-study CSVs.

- [ ] **Step 8: Commit the diagnostic mode**

```powershell
git add -- apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add normal restrict causal probe"
```

---

### Task 5: Verification and Numerical Evidence

**Files:**
- Verify: all modified and created source files.
- Generate: `output/dirichlet_normal_restrict_causal_probe_3d/*.csv`

**Interfaces:**
- Consumes: completed diagnostic executable.
- Produces: the causal conclusion, including an explicit `supported` or
  `not_proven` result.

- [ ] **Step 1: Build all affected Release targets**

Run:

```powershell
cmake --build build --config Release --target harmonic_polynomial_space_3d_test exterior_only_cubic_normal_restrict_3d_test native_nurbs_surface_3d_test native_nurbs_surface_transform_3d_test dirichlet_rigid_transform_study_3d_test neumann_exterior_zero_trace_3d
```

Expected: all targets build.

- [ ] **Step 2: Run focused and regression tests**

Run:

```powershell
.\build\apps\Release\harmonic_polynomial_space_3d_test.exe
.\build\apps\Release\exterior_only_cubic_normal_restrict_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
```

Expected: every executable exits 0.

- [ ] **Step 3: Run the approved evidence grid**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe 32 64
```

Expected: three cases times two levels times two restrict routes complete
within the 160-step cap, with incremental CSV output preserved if a route
does not converge.

- [ ] **Step 4: Apply the pre-registered decision rule**

Check:

```powershell
Import-Csv output\dirichlet_normal_restrict_causal_probe_3d\summary.csv |
    Format-Table case_id,N,route,exact_grid_linf,smooth_grid_linf,exact_equation_linf,physical_iterations,common_iterations
```

Report separately:

- whether exact-grid isolation identifies the current recovery path;
- whether smooth-grid control identifies interpolation or cross-interface
  continuation;
- whether errors concentrate within four `h` of non-G1 edges;
- whether changing only restrict removes at least half the excess iterations
  for both GMRES right-hand sides;
- whether the causal claim is `supported` or `not_proven`.

- [ ] **Step 5: Run N=128 only if the decision is inconclusive**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe 128
```

Expected: this step is skipped when the N=32/64 evidence already determines
the registered rule.

- [ ] **Step 6: Final repository verification**

Run:

```powershell
git diff --check
git status --short
git log -6 --oneline
```

Expected: no whitespace errors; only the two pre-existing unrelated
same-patch documents remain untracked; implementation commits are present.
