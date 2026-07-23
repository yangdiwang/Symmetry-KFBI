# 3D Dirichlet Rigid-Transform Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native-NURBS rigid-transform study mode that runs the current main 3D Dirichlet normal-jump formulation for eight L-prism placements at `N=32,64,128` and reports stability, convergence, cost, and acceptance.

**Architecture:** A focused geometry module applies validated proper rigid transforms directly to NURBS control points while preserving all parameter and topology metadata. A separate study-case module owns the eight transforms and the covariantly transformed harmonic exact solution. The existing 3D app gains a backward-compatible `--rigid-study` path that reuses its full geometry/intersection/Cauchy pipeline but executes only the Dirichlet solve and writes isolated study CSVs.

**Tech Stack:** C++17, Eigen, CMake/Visual Studio x64, existing KFBI3D native NURBS and harmonic-jet modules.

## Global Constraints

- Study only the L-prism and the Dirichlet exterior-zero-normal-trace formulation.
- Use `g1_nearest`, degree-three Cauchy fitting, 48 value samples, 28 normal samples, tricubic grid restriction, and cubic normal fitting.
- Use the fixed box `[-1.5,1.5]^3` and levels `N=32,64,128` for numerical acceptance.
- Use GMRES relative tolerance `2e-10` and a default maximum of 80 iterations.
- Transform the NURBS geometry and manufactured harmonic solution together.
- Preserve knots, weights, UV coordinates, patch order, topology, G1/non-G1 relations, components, and expected area.
- Preserve the legacy positional CLI and its output schema.
- Write study output under `output/dirichlet_rigid_transform_stability_3d`.
- Do not stage or modify the unrelated untracked same-patch design/plan files.

---

### Task 1: Validated Native-NURBS Rigid Transform

**Files:**
- Create: `apps/native_nurbs_surface_transform_3d.hpp`
- Create: `apps/native_nurbs_surface_transform_3d.cpp`
- Create: `apps/native_nurbs_surface_transform_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: `app3d::NativeNurbsSurface3D` and `geometry3d::NurbsSurfacePatch3D`.
- Produces:

```cpp
namespace kfbim::app3d {

class RigidTransform3D {
public:
    RigidTransform3D();
    RigidTransform3D(Eigen::Matrix3d rotation,
                     Eigen::Vector3d center,
                     Eigen::Vector3d translation);

    static RigidTransform3D from_axis_angle(
        const Eigen::Vector3d& axis,
        double angle_radians,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& translation);

    Eigen::Vector3d forward_point(const Eigen::Vector3d& point) const;
    Eigen::Vector3d inverse_point(const Eigen::Vector3d& point) const;
    Eigen::Vector3d forward_vector(const Eigen::Vector3d& vector) const;

    const Eigen::Matrix3d& rotation() const noexcept;
    const Eigen::Vector3d& center() const noexcept;
    const Eigen::Vector3d& translation() const noexcept;

private:
    Eigen::Matrix3d rotation_;
    Eigen::Vector3d center_;
    Eigen::Vector3d translation_;
};

NativeNurbsSurface3D transform_native_nurbs_surface_3d(
    const NativeNurbsSurface3D& source,
    const RigidTransform3D& transform);

} // namespace kfbim::app3d
```

- [ ] **Step 1: Add the failing transform tests and CMake target**

Add tests that:

```cpp
const auto transform = RigidTransform3D::from_axis_angle(
    Eigen::Vector3d(1.0, 2.0, 3.0),
    17.0 * std::acos(-1.0) / 180.0,
    Eigen::Vector3d(0.07, -0.07, 0.02),
    Eigen::Vector3d(0.137, -0.083, 0.061));

const Eigen::Vector3d point(0.21, -0.32, 0.17);
require((transform.inverse_point(transform.forward_point(point)) - point).norm()
            < 2.0e-14,
        "rigid point round trip failed");

const NativeNurbsSurface3D original =
    make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
const NativeNurbsSurface3D moved =
    transform_native_nurbs_surface_3d(original, transform);
for (std::size_t p = 0; p < original.patches.size(); ++p) {
    const double u = 0.37 * original.patches[p].domain_start_u()
                   + 0.63 * original.patches[p].domain_end_u();
    const double v = 0.41 * original.patches[p].domain_start_v()
                   + 0.59 * original.patches[p].domain_end_v();
    require((moved.patches[p].evaluate(u, v)
             - transform.forward_point(original.patches[p].evaluate(u, v))).norm()
                < 5.0e-13,
            "transformed NURBS evaluation mismatch");
}
```

Also compare bases, knots, weights, names, adjacency, connections, components,
area, and transformed `exact_inside`; verify derivatives rotate by `Q`; and
verify `diag(2,1,1)` is rejected as an invalid rotation.

Add the new `.cpp` to `kfbim_3d_app_geometry` and create the
`native_nurbs_surface_transform_3d_test` executable linked to that library.

- [ ] **Step 2: Build the new test to verify it fails**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_transform_3d_test
```

Expected: compilation fails because `native_nurbs_surface_transform_3d.hpp`
and its interfaces do not exist yet.

- [ ] **Step 3: Implement transform validation and native-surface rebuilding**

Validate:

```cpp
const Eigen::Matrix3d orthogonality =
    rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
if (!rotation.allFinite() || !center.allFinite() || !translation.allFinite()
    || orthogonality.cwiseAbs().maxCoeff() > 1.0e-12
    || std::abs(rotation.determinant() - 1.0) > 1.0e-12) {
    throw std::invalid_argument(
        "RigidTransform3D requires a finite proper orthogonal rotation");
}
```

Implement:

```cpp
Eigen::Vector3d RigidTransform3D::forward_point(
    const Eigen::Vector3d& point) const
{
    return center_ + rotation_ * (point - center_) + translation_;
}

Eigen::Vector3d RigidTransform3D::inverse_point(
    const Eigen::Vector3d& point) const
{
    return center_ + rotation_.transpose()
        * (point - center_ - translation_);
}
```

Rebuild each patch with copied bases/weights and transformed control points.
Copy all metadata, then wrap membership as:

```cpp
const auto source_inside = source.exact_inside;
result.exact_inside = [source_inside, transform](const Eigen::Vector3d& point) {
    return source_inside(transform.inverse_point(point));
};
```

- [ ] **Step 4: Build and run the transform test**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_transform_3d_test
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
```

Expected: build succeeds and the executable exits with code 0.

- [ ] **Step 5: Run the existing native-surface regression**

Run:

```powershell
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected: exit code 0.

- [ ] **Step 6: Commit the geometry unit**

```powershell
git add -- apps/native_nurbs_surface_transform_3d.hpp apps/native_nurbs_surface_transform_3d.cpp apps/native_nurbs_surface_transform_3d_test.cpp apps/CMakeLists.txt
git commit -m "feat: add native nurbs rigid transforms"
```

---

### Task 2: Study Cases and Covariant Harmonic Data

**Files:**
- Create: `apps/dirichlet_rigid_transform_study_3d.hpp`
- Create: `apps/dirichlet_rigid_transform_study_3d.cpp`
- Create: `apps/dirichlet_rigid_transform_study_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: `app3d::RigidTransform3D`.
- Produces:

```cpp
struct DirichletRigidStudyCase3D {
    std::string id;
    RigidTransform3D transform;
    Eigen::Vector3d rotation_axis;
    double rotation_angle_degrees = 0.0;
};

std::vector<DirichletRigidStudyCase3D>
make_l_prism_dirichlet_rigid_study_cases_3d();

double manufactured_harmonic_value_3d(const Eigen::Vector3d& point);
Eigen::Vector3d manufactured_harmonic_gradient_3d(
    const Eigen::Vector3d& point);
double transformed_manufactured_harmonic_value_3d(
    const RigidTransform3D& transform,
    const Eigen::Vector3d& point);
Eigen::Vector3d transformed_manufactured_harmonic_gradient_3d(
    const RigidTransform3D& transform,
    const Eigen::Vector3d& point);
```

- [ ] **Step 1: Add failing catalog and exact-solution tests**

Verify the IDs and metadata of exactly these cases:

```text
baseline
tx_p0137
ty_m0083
tz_p0061
t_xyz_1
t_xyz_2
rot_axis123_17deg
rot_axis123_17deg_t_xyz_1
```

Verify the five translations, the normalized `(1,2,3)` axis, 17-degree angle,
and center `(0.07,-0.07,0.02)`. For each case and sample point `x`, verify:

```cpp
const Eigen::Vector3d moved = item.transform.forward_point(x);
require(std::abs(transformed_manufactured_harmonic_value_3d(
                     item.transform, moved)
                 - manufactured_harmonic_value_3d(x))
            < 2.0e-14,
        "transformed harmonic value is not covariant");
require((transformed_manufactured_harmonic_gradient_3d(
             item.transform, moved)
         - item.transform.forward_vector(
             manufactured_harmonic_gradient_3d(x))).norm()
            < 2.0e-14,
        "transformed harmonic gradient is not covariant");
```

Register `dirichlet_rigid_transform_study_3d_test` in CMake.

- [ ] **Step 2: Build to verify the study test fails**

Run:

```powershell
cmake --build build --config Release --target dirichlet_rigid_transform_study_3d_test
```

Expected: compilation fails because the study catalog and harmonic interfaces
are missing.

- [ ] **Step 3: Implement the exact eight-case catalog and transformed field**

Use the approved translations:

```cpp
{0.137, 0.0, 0.0}
{0.0, -0.083, 0.0}
{0.0, 0.0, 0.061}
{0.137, -0.083, 0.061}
{-0.109, 0.151, -0.047}
```

Keep the existing harmonic field:

```cpp
u = exp(0.35*x) * cos(0.21*y) * cos(0.28*z);
```

Evaluate transformed data with the inverse point and rotated gradient:

```cpp
const Eigen::Vector3d source = transform.inverse_point(point);
return transform.forward_vector(manufactured_harmonic_gradient_3d(source));
```

- [ ] **Step 4: Build and run both focused tests**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_transform_3d_test dirichlet_rigid_transform_study_3d_test
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
```

Expected: both exit with code 0.

- [ ] **Step 5: Commit the study-definition unit**

```powershell
git add -- apps/dirichlet_rigid_transform_study_3d.hpp apps/dirichlet_rigid_transform_study_3d.cpp apps/dirichlet_rigid_transform_study_3d_test.cpp apps/CMakeLists.txt
git commit -m "feat: define 3d dirichlet rigid study cases"
```

---

### Task 3: Backward-Compatible Dirichlet-Only Study Mode

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`

**Interfaces:**
- Consumes:
  - `DirichletRigidStudyCase3D`
  - `transform_native_nurbs_surface_3d`
  - transformed manufactured value and gradient functions
- Produces:
  - CLI: `neumann_exterior_zero_trace_3d --rigid-study [N ...]`
  - CSV: `output/dirichlet_rigid_transform_stability_3d/rigid_transform_results.csv`
  - CSV: `output/dirichlet_rigid_transform_stability_3d/rigid_transform_acceptance.csv`

- [ ] **Step 1: Add the new CLI branch and compile it against missing run helpers**

Parse before the legacy geometry selection:

```cpp
const bool rigid_study = argc >= 2
                      && std::string(argv[1]) == "--rigid-study";
```

For rigid mode, parse levels from `argv[2...]`, default to `{32,64,128}`, and
require powers of two at least 16. Extend `--help` with the exact new
invocation. Call a not-yet-defined `run_dirichlet_rigid_study(...)` so the
first build fails at the intended integration boundary.

- [ ] **Step 2: Build to verify the app integration fails**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
```

Expected: compilation fails because `run_dirichlet_rigid_study` is not yet
defined.

- [ ] **Step 3: Generalize geometry and exact-data inputs without changing legacy behavior**

Change geometry construction to:

```cpp
GeometryBundle make_geometry(
    GeometryKind kind,
    double h,
    const app3d::RigidTransform3D& transform);
```

Build the original native surface, apply the transform before triangulation,
and verify every transformed control point satisfies:

```cpp
point.allFinite()
&& point.array().minCoeff() > kBoxMin
&& point.array().maxCoeff() < kBoxMin + kBoxSide
```

Pass a `RigidTransform3D` into both legacy solve functions. Replace direct
calls to the anonymous manufactured functions with the shared transformed
field functions. Identity transforms must reproduce the legacy problem.

Add:

```cpp
enum class SolveSelection3D {
    Both,
    DirichletNormalOnly
};
```

Teach `run_readiness_case` to skip Neumann construction, solve reporting, and
convergence checks when `DirichletNormalOnly` is selected. Keep all geometry,
intersection, classification, Cauchy, constant-probe, and Dirichlet
diagnostics active.

- [ ] **Step 4: Implement study results, orders, ratios, and incremental CSV output**

Add a study row containing transform metadata, `ReadinessResult`,
total elapsed seconds, observed order, baseline error ratio, baseline
iteration ratio, and criterion flags.

Compute:

```cpp
order = std::log(previous_error / current_error)
      / std::log(static_cast<double>(current_N) / previous_N);
error_ratio = current_error / baseline_error_at_same_N;
iteration_ratio = static_cast<double>(current_iterations)
                / baseline_iterations_at_same_N;
```

Write `rigid_transform_results.csv` after every completed solve. Write one
acceptance row per case with:

```text
gmres_pass
monotone_error_pass
order_64_128_pass
baseline_ratio_pass
geometry_diagnostics_pass
overall_pass
```

For a smoke run without both `N=64` and `128`, mark the order criterion
`not_evaluated` and do not fail solely because the acceptance grid pair was
not requested. For the full acceptance grid, require:

```text
iterations <= 80
error(N32) > error(N64) > error(N128)
order(N64,N128) >= 1.8
error(case,N) / error(baseline,N) <= 3.0
all specified geometry fallback/mismatch counters == 0
```

Stop immediately on GMRES nonconvergence, while preserving completed CSV
rows. Complete all cases before returning failure for an accuracy/order
criterion, so the comparison remains diagnosable.

- [ ] **Step 5: Enforce the main scheme and isolated output**

Rigid mode must reject active settings other than:

```text
KFBIM_3D_CAUCHY_POLICY=g1_nearest
KFBIM_3D_CAUCHY_VALUE_COUNT=48
KFBIM_3D_CAUCHY_NORMAL_COUNT=28
```

Allow `KFBIM_3D_GMRES_MAX_ITERATIONS`, default 80, but mark a result with more
than 80 iterations as an acceptance failure. Write only beneath:

```text
output/dirichlet_rigid_transform_stability_3d
```

Do not call the legacy readiness or solve-summary writers in rigid mode, so
their CSV schema and files remain unchanged.

- [ ] **Step 6: Build and run CLI regression smoke checks**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --help
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --rigid-study 16
```

Expected:

- help displays both legacy and rigid-study invocations;
- legacy `l_prism 16` still executes both formulations;
- rigid `N=16` executes all eight cases, prints no Neumann solve row, writes
  only the new study CSVs, and every GMRES solve converges.

- [ ] **Step 7: Commit the app integration**

```powershell
git add -- apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add dirichlet rigid transform study mode"
```

---

### Task 4: Full Verification and Numerical Acceptance

**Files:**
- Verify: `apps/native_nurbs_surface_transform_3d_test.cpp`
- Verify: `apps/dirichlet_rigid_transform_study_3d_test.cpp`
- Verify: `apps/native_nurbs_surface_3d_test.cpp`
- Generate: `output/dirichlet_rigid_transform_stability_3d/rigid_transform_results.csv`
- Generate: `output/dirichlet_rigid_transform_stability_3d/rigid_transform_acceptance.csv`

**Interfaces:**
- Consumes: completed study-mode executable.
- Produces: the requested `N=32,64,128` numerical comparison.

- [ ] **Step 1: Build all affected Release targets**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test native_nurbs_surface_transform_3d_test dirichlet_rigid_transform_study_3d_test neumann_exterior_zero_trace_3d
```

Expected: build succeeds with no compile or link errors.

- [ ] **Step 2: Run all focused and regression tests**

Run:

```powershell
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
```

Expected: all three exit with code 0.

- [ ] **Step 3: Run the complete numerical study**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --rigid-study 32 64 128
```

Expected: 24 Dirichlet solves complete; no Neumann solve is run; accumulated
CSV files are updated after every solve.

- [ ] **Step 4: Inspect numerical acceptance**

Run:

```powershell
Import-Csv output\dirichlet_rigid_transform_stability_3d\rigid_transform_acceptance.csv |
    Format-Table case_id,gmres_pass,monotone_error_pass,order_64_128_pass,baseline_ratio_pass,geometry_diagnostics_pass,overall_pass
```

Expected: eight rows. Report every criterion exactly as measured; do not hide
or relabel a failed criterion.

- [ ] **Step 5: Verify repository state and final diff**

Run:

```powershell
git diff --check
git status --short
git log -4 --oneline
```

Expected: no whitespace errors; only the two pre-existing unrelated untracked
same-patch documents may remain untracked; implementation commits are visible.

- [ ] **Step 6: Report results**

Report, for every transform and level:

```text
interior Linf error
observed order
GMRES iterations
solve seconds
total seconds
baseline error ratio
acceptance status
```

Also identify the worst transform by `N=128` error ratio and GMRES iteration
growth, and state whether rigid translation/rotation materially affects the
current main scheme.
