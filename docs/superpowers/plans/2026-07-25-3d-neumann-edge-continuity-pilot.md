# 3D Neumann Non-G1 Edge-Continuity Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, causally isolated Neumann L-prism formulation that projects the patchwise value-jump density onto a space with one reconstructed trace across every non-G1 NURBS patch connection, then compare it with the current patch-independent formulation at `N=32,64` for the baseline, difficult translation, and representative rotation.

**Architecture:** Keep the existing native patch-centered surface DOFs, degree-3 `G1Nearest` Cauchy stencils, `JointTricubicCrossingOwner` restriction, `RegionClosestHybrid` preprocessing, FFT solve, and constant-mode augmentation. Build sparse mortar-like edge-trace rows from `NativeNurbsSurface3D::geometric_connections`, form a surface-mass orthogonal projector once, and wrap the existing augmented exterior-zero-trace operator with `P A P + (I-P)`. A pure evaluator applies structural and numerical A/B acceptance gates, while a dedicated CLI reuses one preprocessed pipeline for both density spaces.

**Tech Stack:** C++17, Eigen dense/sparse linear algebra, existing native-NURBS/KFBI3D application geometry, ZFFT bulk solver, restarted GMRES, CMake/Visual Studio x64 Release build, CSV checkpoints.

## Global Constraints

- Geometry is the native twelve-patch NURBS L-prism only.
- Pilot poses are exactly `baseline`, `ty_m0083`, and `rot_axis123_17deg`.
- Default pilot levels are exactly `32,64`; `N=128` is accepted only as the prefix `32,64,128` and is not run unless the completed `32,64` pilot passes.
- The existing patch-independent Neumann route remains the production default.
- The A/B pair uses `JointTricubicCrossingOwner`, `RegionClosestHybrid`, degree-3 `G1Nearest` Cauchy fitting, 48 value conditions, 28 normal conditions, GMRES tolerance `2e-10`, restart 80, and cap 80.
- No Cauchy stencil is allowed to cross a non-G1 edge in this pilot; only the admissible value-density space changes.
- Surface DOFs remain independent cell-centered tensor grids on native NURBS patches.
- Constraints use only `geometric_connections` entries with `g1 == false`, preserve partial parameter intervals, and honor `reversed`.
- Each edge-side trace uses cubic interpolation along the edge and quadratic extrapolation through the first three inward cell-center rows, with recorded lower-order fallback only when a patch is too coarse.
- The projector uses surface quadrature weights and is applied matrix-free after one setup factorization.
- The existing single constant-mode Lagrange multiplier remains unchanged.
- Geometry labels, crossing ownership, owner workload fingerprints, and geometry-query counts must be identical between A/B solves.
- Generated CSVs remain untracked under `output/neumann_edge_continuity_3d`.
- No penalty parameter, extra edge multiplier, shared global edge basis, FFT change, spread change, or restrict change is in scope.

## File and Responsibility Map

| File | Responsibility |
|---|---|
| `apps/neumann_edge_continuity_3d.hpp` | Public edge-row, projector, projected-operator, measurement, and evaluation interfaces |
| `apps/neumann_edge_continuity_3d.cpp` | Edge parameter mapping, trace reconstruction, sparse constraints, rank filtering, projector algebra, projected wrapper, and pure acceptance evaluation |
| `apps/neumann_edge_continuity_3d_test.cpp` | Geometry/topology, reconstruction order, projector invariants, wrapper algebra, and acceptance-gate tests |
| `apps/neumann_exterior_zero_trace_3d.cpp` | Optional solver hook, shared-pipeline A/B worker, CLI, checkpoints, and production metrics |
| `apps/CMakeLists.txt` | Library source and focused test target registration |
| `docs/superpowers/results/2026-07-25-3d-neumann-edge-continuity-pilot.md` | Checked-in numerical evidence and algorithmic conclusion |

---

### Task 1: Sparse non-G1 edge-trace constraints

**Files:**
- Create: `apps/neumann_edge_continuity_3d.hpp`
- Create: `apps/neumann_edge_continuity_3d.cpp`
- Create: `apps/neumann_edge_continuity_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Public interfaces:**

```cpp
namespace kfbim::app3d {

enum class NeumannDensitySpace3D {
    PatchIndependent,
    NonG1EdgeProjected
};

const char* neumann_density_space_name_3d(
    NeumannDensitySpace3D mode);

struct NeumannEdgeConstraintSample3D {
    int connection_index = -1;
    int sample_index = -1;
    int sample_count = 0;
    int first_patch = -1;
    int second_patch = -1;
    PatchEdge3D first_edge = PatchEdge3D::UMin;
    PatchEdge3D second_edge = PatchEdge3D::UMin;
    double normalized_parameter = 0.0;
    double first_parameter = 0.0;
    double second_parameter = 0.0;
    double quadrature_weight = 0.0;
    double mapped_point_gap = 0.0;
    bool first_reduced_order = false;
    bool second_reduced_order = false;
};

struct NeumannEdgeConstraintSet3D {
    int density_size = 0;
    int non_g1_connection_count = 0;
    int reduced_order_row_count = 0;
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix;
    Eigen::VectorXd quadrature_weights;
    std::vector<NeumannEdgeConstraintSample3D> samples;
};

NeumannEdgeConstraintSet3D build_neumann_edge_constraints_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h);

Eigen::VectorXd apply_neumann_edge_constraints_3d(
    const NeumannEdgeConstraintSet3D& constraints,
    const Eigen::VectorXd& density);

double neumann_edge_mismatch_linf_3d(
    const NeumannEdgeConstraintSet3D& constraints,
    const Eigen::VectorXd& density);

double neumann_edge_mismatch_weighted_rms_3d(
    const NeumannEdgeConstraintSet3D& constraints,
    const Eigen::VectorXd& density);

} // namespace kfbim::app3d
```

- [ ] **Step 1: Register a test target and write failing topology/reconstruction tests**

Add `neumann_edge_continuity_3d_test` to `apps/CMakeLists.txt`, link it to
`kfbim_3d_app_geometry`, and set C++17. Do not add the production `.cpp`
to the library yet, so the first failure is the missing public API rather
than an accidental stub implementation.

The test must construct:

```cpp
const auto surface =
    make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
const double h32 = 3.0 / 32.0;
const auto cloud32 = make_native_surface_dofs_3d(surface, h32);
const auto constraints32 =
    build_neumann_edge_constraints_3d(surface, cloud32, h32);
```

Add literal assertions that catch:

- every `g1 == false` connection index occurring in the metadata;
- every `g1 == true` connection index occurring zero times;
- `samples.size() == matrix.rows() == quadrature_weights.size()`;
- `matrix.cols() == cloud.dofs.size()`;
- each connection contributes exactly
  `max(2, ceil(connection_length / h))` rows;
- every first/second parameter lies inside its declared partial interval;
- the second-side parameter reverses monotonically when
  `connection.reversed` is true;
- both mapped NURBS points agree to `1e-11` times the geometry diameter;
- all quadrature weights are finite and positive;
- a vector of ones satisfies
  `||C 1||inf <= 5e-13`;
- a connection with fewer than the required four along-edge or three
  inward rows is marked reduced-order rather than indexed out of range.

Exercise the fallback explicitly with
`make_native_surface_dofs_3d(surface, 10.0)`, for which the native
generator supplies only its minimum `2x2` cells on the small L-prism
patches. Require a nonzero reduced-order row count, finite coefficients,
and valid column indices for every resulting row.

Use a non-polynomial harmonic trace in the order test:

```cpp
double exact_value(const Eigen::Vector3d& x)
{
    return std::exp(0.35 * x.x())
         * std::cos(0.21 * x.y())
         * std::cos(std::sqrt(0.35 * 0.35 - 0.21 * 0.21) * x.z());
}
```

Sample it at all surface DOFs for `h=3/32` and `h=3/64`. Require finite
edge mismatches and:

```cpp
require(mismatch32 / mismatch64 >= 6.0,
        "exact trace mismatch is not at least third-order");
```

Production mutations caught: processing G1 edges, ignoring partial
intervals, missing reversal, using endpoint rather than cell-center
samples, wrong tensor index direction, non-partition-of-unity
interpolation, and only second-order edge reconstruction.

- [ ] **Step 2: Build the test and verify RED**

Run:

```powershell
cmake --build build --config Release `
    --target neumann_edge_continuity_3d_test -- /m:1
```

Expected: compilation fails because
`apps/neumann_edge_continuity_3d.hpp` does not exist.

- [ ] **Step 3: Implement deterministic edge sampling and trace rows**

Implement helpers privately in `neumann_edge_continuity_3d.cpp`:

```cpp
double edge_parameter(
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double s);

Eigen::Vector3d edge_point(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double parameter);

std::vector<std::pair<int, double>> edge_trace_coefficients(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double parameter,
    bool& reduced_order);
```

For each non-G1 connection:

1. Integrate the first-side physical speed over its active interval with
   fixed 8-point Gauss-Legendre quadrature to obtain
   `connection_length`.
2. Set `n_edge = max(2, ceil(connection_length / h))`.
3. Use `s_j=(j+0.5)/n_edge`.
4. Map the first side with `s_j`; map the second with
   `connection.reversed ? 1-s_j : s_j`.
5. Set midpoint quadrature weight to the first-side physical speed times
   the parameter-cell width.
6. Build each side as a tensor product of one-dimensional Lagrange
   weights:
   - the nearest contiguous four cell centers along the edge;
   - the first three cell-center rows inward from the edge;
   - all available rows when the required count is unavailable.
7. Add first-side coefficients with `+1` and second-side coefficients
   with `-1`, combine duplicate column indices, and reject non-finite
   coefficients or a row whose coefficient sum exceeds `5e-13`.

The edge/index mapping must be explicit:

| Patch edge | Fixed coordinate | Along-edge tensor index | Inward tensor rows |
|---|---|---|---|
| `UMin` | `u=u0` | `j` / `v` | `i=0,1,2` |
| `UMax` | `u=u1` | `j` / `v` | `i=nu-1,nu-2,nu-3` |
| `VMin` | `v=v0` | `i` / `u` | `j=0,1,2` |
| `VMax` | `v=v1` | `i` / `u` | `j=nv-1,nv-2,nv-3` |

The sparse matrix stores the unweighted physical trace differences.
`quadrature_weights` remains separate so diagnostics report physical
mismatch and the projector can apply `sqrt(weight)` exactly once.

- [ ] **Step 4: Add the implementation to the geometry library and verify GREEN**

Add `neumann_edge_continuity_3d.cpp` to
`kfbim_3d_app_geometry`.

Run:

```powershell
cmake --build build --config Release `
    --target neumann_edge_continuity_3d_test -- /m:1
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected: both executables exit 0, the exact non-polynomial trace ratio is
at least 6, and no row reports a mapped point gap above tolerance.

- [ ] **Step 5: Commit**

```powershell
git add apps/CMakeLists.txt `
    apps/neumann_edge_continuity_3d.hpp `
    apps/neumann_edge_continuity_3d.cpp `
    apps/neumann_edge_continuity_3d_test.cpp
git commit -m "feat: build Neumann non-G1 edge constraints"
```

---

### Task 2: Surface-mass projector and projected augmented operator

**Files:**
- Modify: `apps/neumann_edge_continuity_3d.hpp`
- Modify: `apps/neumann_edge_continuity_3d.cpp`
- Modify: `apps/neumann_edge_continuity_3d_test.cpp`

**Additional public interfaces:**

```cpp
struct NeumannEdgeProjectionDiagnostics3D {
    int input_constraint_count = 0;
    int retained_constraint_rank = 0;
    double rank_tolerance = 0.0;
    double factorization_seconds = 0.0;
    double constant_constraint_defect = 0.0;
};

class NeumannEdgeContinuityProjector3D {
public:
    NeumannEdgeContinuityProjector3D(
        const NeumannEdgeConstraintSet3D& constraints,
        const SurfaceDofCloud3D& cloud,
        double rank_tolerance = 1e-12);

    int density_size() const;
    int constraint_count() const;
    int retained_rank() const;

    Eigen::VectorXd project(const Eigen::VectorXd& density) const;
    Eigen::VectorXd complement(const Eigen::VectorXd& density) const;
    Eigen::VectorXd constraint_mismatch(
        const Eigen::VectorXd& density) const;
    double mismatch_linf(const Eigen::VectorXd& density) const;
    double mismatch_weighted_rms(
        const Eigen::VectorXd& density) const;

    const NeumannEdgeProjectionDiagnostics3D& diagnostics() const;
};

class NeumannEdgeProjectedAugmentedOperator3D final
    : public IKFBIOperator {
public:
    NeumannEdgeProjectedAugmentedOperator3D(
        const IKFBIOperator& base,
        const NeumannEdgeContinuityProjector3D& projector);

    int problem_size() const override;
    void apply(
        const Eigen::VectorXd& unknown,
        Eigen::VectorXd& result) const override;

    Eigen::VectorXd project_right_hand_side(
        const Eigen::VectorXd& base_rhs) const;
};
```

- [ ] **Step 1: Add failing projector-invariant tests**

Extend the focused test with the L-prism `N=32` constraint set and a
deterministic input:

```cpp
Eigen::VectorXd x(cloud.dofs.size());
for (int q = 0; q < x.size(); ++q)
    x[q] = std::sin(0.37 * (q + 1)) + 0.2 * std::cos(0.11 * (q + 1));

NeumannEdgeContinuityProjector3D projector(
    constraints, cloud, 1e-12);
const Eigen::VectorXd px = projector.project(x);
```

Require normalized infinity defects:

```text
||C P x||inf / max(1,||x||inf)       <= 1e-11
||P(Px)-Px||inf / max(1,||x||inf)    <= 1e-11
||P 1-1||inf                          <= 1e-11
```

Also require:

- `0 < retained_rank <= input_constraint_count`;
- all surface masses are used and positive;
- `complement(x) == x-project(x)` to roundoff;
- projector mismatch helpers equal direct `C*x` calculations;
- wrong vector sizes throw `std::invalid_argument`;
- a zero-row synthetic constraint set produces the identity projector.

- [ ] **Step 2: Add a failing algebra test for the augmented wrapper**

Create a synthetic base operator of size `density_size+1`:

```cpp
top = M * phi + lambda * ones;
bottom = mass.dot(phi) / mass.sum();
```

For arbitrary `(phi,lambda)`, compare the wrapper against the independently
formed expression:

```cpp
expected.head(n) =
    P(base_top(P(phi))) + (phi - P(phi));
expected[n] = weighted_mean(P(phi));
```

Check separately that:

- a pure excluded component `q=(I-P)x` is returned unchanged in the top
  block;
- `project_right_hand_side` projects only the density block and preserves
  the last scalar;
- constructor/input size mismatches throw;
- the wrapper does not mutate the base operator.

Production mutations caught: Euclidean instead of surface-mass
projection, applying `P` on only one side of `A`, dropping `(I-P)`,
projecting the Lagrange scalar, or using the unprojected density in the
mean equation.

- [ ] **Step 3: Build and verify RED**

Run:

```powershell
cmake --build build --config Release `
    --target neumann_edge_continuity_3d_test -- /m:1
```

Expected: compilation fails on the missing projector and wrapper APIs.

- [ ] **Step 4: Implement rank filtering and the matrix-free projector**

Validate positive finite surface masses and edge weights. Form the sparse
scaled constraint:

```text
B = diag(sqrt(edge_weight)) * C * diag(1/sqrt(surface_mass)).
```

Run rank-revealing sparse QR on `B.transpose()`:

```cpp
Eigen::SparseQR<
    Eigen::SparseMatrix<double>,
    Eigen::COLAMDOrdering<int>> qr;
qr.setPivotThreshold(rank_tolerance);
qr.compute(B.transpose());
```

Use the first `qr.rank()` entries of `qr.colsPermutation()` to select
independent original constraint rows. Preserve the selected original row
indices in deterministic sorted order for diagnostics. Construct:

```text
Cr = selected rows of diag(sqrt(edge_weight)) * C
G  = Cr * W^{-1} * Cr^T
```

Factor `G` once with `Eigen::LDLT<Eigen::MatrixXd>`. Reject a failed,
non-finite, or non-positive factorization. Apply:

```text
r     = Cr * x
alpha = G^{-1} r
P x   = x - W^{-1} Cr^T alpha.
```

Retain the full unweighted `C` for physical mismatch diagnostics and for
checking that dropped rows are still satisfied. Store no dense
`density_size x density_size` projector.

For a zero-row constraint set, return identity without invoking QR or
LDLT.

- [ ] **Step 5: Implement the projected augmented wrapper**

In `apply`:

```cpp
const int n = projector_.density_size();
Eigen::VectorXd projected_unknown = unknown;
projected_unknown.head(n) =
    projector_.project(unknown.head(n));

Eigen::VectorXd base_result;
base_.apply(projected_unknown, base_result);

result = base_result;
result.head(n) =
    projector_.project(base_result.head(n))
    + unknown.head(n) - projected_unknown.head(n);
```

The base operator therefore supplies both
`A(P phi)+lambda*1` and `weighted_mean(P phi)`. Constant preservation
ensures the left projection leaves `lambda*1` unchanged. The RHS helper
projects only `head(n)`.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```powershell
cmake --build build --config Release `
    --target neumann_edge_continuity_3d_test -- /m:1
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
git diff --check
```

Expected: the test exits 0 and all three normalized projection defects are
at most `1e-11`.

Commit:

```powershell
git add apps/neumann_edge_continuity_3d.hpp `
    apps/neumann_edge_continuity_3d.cpp `
    apps/neumann_edge_continuity_3d_test.cpp
git commit -m "feat: add mass-projected Neumann density space"
```

---

### Task 3: Pure A/B evaluation and acceptance gates

**Files:**
- Modify: `apps/neumann_edge_continuity_3d.hpp`
- Modify: `apps/neumann_edge_continuity_3d.cpp`
- Modify: `apps/neumann_edge_continuity_3d_test.cpp`

**Additional public interfaces:**

```cpp
struct NeumannEdgeContinuityMeasurement3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    NeumannDensitySpace3D density_space =
        NeumannDensitySpace3D::PatchIndependent;
    bool finite_metrics = false;
    bool gmres_converged = false;
    int gmres_iterations = 0;
    double gmres_relative_residual = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double interior_linf = 0.0;
    double interior_l2 = 0.0;
    double edge_mismatch_linf = 0.0;
    double edge_mismatch_weighted_rms = 0.0;
    double exact_edge_mismatch_linf = 0.0;
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int duplicate_connection_intervals = 0;
    int g1_constraint_rows = 0;
    int unrelated_constraint_rows = 0;
    int constraint_rows = 0;
    int constraint_rank = 0;
    int reduced_order_rows = 0;
    double constant_constraint_defect = 0.0;
    double projected_constraint_defect = 0.0;
    double projection_idempotence_defect = 0.0;
    double constant_projection_defect = 0.0;
    bool geometry_diagnostics_pass = false;
    bool owner_invariants_pass = false;
    bool shared_preprocess_pass = false;
};

struct NeumannEdgeContinuityDerivedRow3D {
    NeumannEdgeContinuityMeasurement3D measurement;
    double density_linf_order;
    double density_l2_order;
    double interior_linf_order;
    double interior_l2_order;
    double exact_edge_mismatch_order;
    double density_linf_ratio_to_unconstrained;
    double density_l2_ratio_to_unconstrained;
    double interior_linf_ratio_to_unconstrained;
    double interior_l2_ratio_to_unconstrained;
    double edge_reduction_ratio;
    RigidStudyCriterionStatus3D row_pass;
};

struct NeumannEdgeContinuityAcceptance3D {
    RigidStudyCriterionStatus3D completeness_pass;
    RigidStudyCriterionStatus3D topology_pass;
    RigidStudyCriterionStatus3D projector_pass;
    RigidStudyCriterionStatus3D exact_trace_order_pass;
    RigidStudyCriterionStatus3D gmres_pass;
    RigidStudyCriterionStatus3D error_guard_pass;
    RigidStudyCriterionStatus3D edge_reduction_pass;
    RigidStudyCriterionStatus3D trend_pass;
    RigidStudyCriterionStatus3D geometry_owner_pass;
    RigidStudyCriterionStatus3D overall_pass;
};

struct NeumannEdgeContinuityEvaluation3D {
    std::vector<NeumannEdgeContinuityDerivedRow3D> rows;
    NeumannEdgeContinuityAcceptance3D acceptance;
    bool all_pass = false;
};

std::vector<int> normalize_neumann_edge_continuity_levels_3d(
    std::vector<int> levels);

NeumannEdgeContinuityEvaluation3D
evaluate_neumann_edge_continuity_study_3d(
    const std::vector<NeumannEdgeContinuityMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_pilot);
```

- [ ] **Step 1: Write failing level and paired-evaluation tests**

Require:

- an empty level list normalizes to `{32,64}`;
- `{64}` and `{32,128}` throw;
- `{32}`, `{32,64}`, and `{32,64,128}` are accepted prefixes;
- any other level throws.

Build a complete literal fixture with three cases, two levels, and two
density spaces. Make the projected rows:

- retain all topology/projector invariants;
- reduce each baseline edge mismatch by `1e5`;
- keep all four errors at or below baseline;
- reduce the worst iteration count and the `ty_m0083` maximum;
- improve each `32->64` error ratio;
- reduce exact mismatch by a factor of 8.

Require every acceptance field and `all_pass` to pass. Add one mutation
fixture for each independent failure:

- missing mode/level/case row;
- duplicate `(case_id,N,density_space)` key;
- missing or duplicated non-G1 interval;
- any G1/unrelated constraint row;
- any normalized projector defect above `1e-11`;
- exact mismatch ratio below 6;
- projected worst GMRES above baseline or no `ty_m0083` decrease;
- any density/interior `Linf` or `L2` error above `1.10` times baseline;
- edge mismatch reduction below `1e4`;
- any projected refinement ratio worse than its baseline pair;
- changed geometry/owner/shared-preprocess invariant.

Ensure a valid `N=32` smoke prefix checks execution integrity but leaves
two-level numerical criteria `not_evaluated`.

- [ ] **Step 2: Build and verify RED**

Run:

```powershell
cmake --build build --config Release `
    --target neumann_edge_continuity_3d_test -- /m:1
```

Expected: compilation fails on the missing measurement/evaluation APIs.

- [ ] **Step 3: Implement keying, derived values, and exact gates**

Key rows by `(case_id,N,density_space)` and reject duplicates. Compute
orders only within one case and one density space:

```text
p = log(error_coarse / error_fine)
    / log(h_coarse / h_fine).
```

Pair projected and unconstrained rows at identical `(case_id,N)`.
Implement the design gates literally:

- structural defects are normalized infinity defects and must be
  `<=1e-11`;
- every expected non-G1 interval is covered once, with zero G1 and
  unrelated rows;
- `exact_mismatch_32 / exact_mismatch_64 >= 6` for all three poses;
- `max(projected_iterations) <= max(unconstrained_iterations)`;
- `max(projected_iterations for ty_m0083)
  < max(unconstrained_iterations for ty_m0083)`;
- every projected density/interior `Linf` and `L2` error is
  `<=1.10 * unconstrained_error`;
- every numerical edge mismatch reduction is `>=1e4`;
- for all four error norms, the projected `E64/E32` ratio is no larger
  than the unconstrained ratio, allowing only `64*epsilon` relative
  roundoff;
- geometry, owner, and shared-preprocess flags pass for every row.

For `require_complete_pilot == false`, unavailable two-level gates are
`NotEvaluated` and neutral to the smoke exit decision. For a complete
`N=32,64` pilot, every gate participates in `overall_pass`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```powershell
cmake --build build --config Release `
    --target neumann_edge_continuity_3d_test -- /m:1
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
git diff --check
```

Expected: all synthetic positive and negative acceptance cases pass.

Commit:

```powershell
git add apps/neumann_edge_continuity_3d.hpp `
    apps/neumann_edge_continuity_3d.cpp `
    apps/neumann_edge_continuity_3d_test.cpp
git commit -m "feat: evaluate Neumann edge-continuity pilot"
```

---

### Task 4: Optional Neumann solve path and dedicated A/B CLI

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`

**Application interfaces:**

```cpp
ExteriorZeroTraceSolution3D solve_exterior_zero_trace_neumann_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_normal_jump,
    double tolerance,
    int restart,
    int max_iterations,
    ExteriorValueRestrictMode3D mode,
    const app3d::NeumannEdgeContinuityProjector3D*
        edge_projector = nullptr);

int run_neumann_edge_continuity_study_3d(
    std::vector<int> levels);
```

- [ ] **Step 1: Verify the CLI boundary is RED**

Run:

```powershell
$help = .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --help | Out-String
if ($help -notmatch '--neumann-edge-continuity-study') {
    throw 'Neumann edge-continuity route missing'
}
```

Expected: PowerShell throws
`Neumann edge-continuity route missing`.

- [ ] **Step 2: Add the optional projector to the existing solver**

Keep `ExteriorZeroTraceOperator3D` unchanged as the base operator. When
`edge_projector == nullptr`, preserve the existing solve path byte for
byte. Otherwise:

1. wrap the base operator with
   `NeumannEdgeProjectedAugmentedOperator3D`;
2. project the base RHS through `project_right_hand_side`;
3. run the same GMRES configuration from a zero augmented vector;
4. project the returned density once more;
5. replace `augmented_unknown.head(size)` with that projected density
   before evaluating the physical field and augmented residual.

Extend `run_neumann_case` only with trailing optional outputs:

```cpp
const app3d::NeumannEdgeContinuityProjector3D*
    edge_projector = nullptr,
Eigen::VectorXd* solved_value_jump = nullptr
```

All existing calls therefore retain their current source and numerical
behavior.

- [ ] **Step 3: Implement one shared-pipeline A/B worker**

Add an internal row type containing:

- pose, `N`, `h`, density-space mode, DOF count;
- geometry, pipeline, projector, solve, and total times;
- `SolveMetrics3D` and GMRES residual history;
- solved and exact edge mismatch `Linf`/weighted-RMS;
- constraint counts/rank/reduced-order count and all projection defects;
- current label, unsafe-edge, gap, endpoint, triangle-fallback, owner
  fingerprint, output digest, wrong-side query, and before/after GMRES
  diagnostics.

Implement:

```cpp
std::array<NeumannEdgeStudyRow3D, 2>
run_neumann_edge_continuity_pair_3d(
    int N,
    const app3d::LPrismRigidStudyCase3D& study_case);
```

The worker constructs exactly once:

```text
CartesianGrid3D
transformed native L-prism GeometryBundle
NurbsCartesianDomain3D
SurfaceDofCloud3D
degree-3 G1Nearest CauchyStencilSet
GridPair3D
RegionClosestHybrid PanelCenterHarmonicJetKFBI3D
NeumannEdgeConstraintSet3D
NeumannEdgeContinuityProjector3D
```

Then it runs, in order:

```text
patch_independent
non_g1_edge_projected
```

Both use `JointTricubicCrossingOwner`. Compute the exact density as the
transformed manufactured harmonic value sampled at the surface DOFs,
minus its surface-weighted mean. Apply the same constraint set to the
exact, unconstrained, and projected densities.

For a deterministic projector probe
`x_q=sin(0.37(q+1))+0.2*cos(0.11(q+1))`, record:

```text
||C P x||inf / max(1,||x||inf)
||P(Px)-Px||inf / max(1,||x||inf)
||P 1-1||inf
||C 1||inf
```

Assert that pipeline geometry-query counts and preprocessing diagnostics
do not change after either solve. Copy identical geometry/preprocess
diagnostics into both rows and set `shared_preprocess_pass` only after
comparing their fingerprints, digests, and counts.

- [ ] **Step 4: Add checkpoint writers and the study driver**

Select the three pilot cases by exact ID from
`make_l_prism_rigid_study_cases_3d()`. The driver writes after every
completed A/B pair under:

```text
output/neumann_edge_continuity_3d/summary.csv
output/neumann_edge_continuity_3d/gmres_residuals.csv
output/neumann_edge_continuity_3d/edge_diagnostics.csv
output/neumann_edge_continuity_3d/owner_diagnostics.csv
output/neumann_edge_continuity_3d/acceptance.csv
```

`summary.csv` must include, at minimum:

```text
case_id,N,h,density_space,dofs,
geometry_setup_seconds,pipeline_setup_seconds,projector_setup_seconds,
solve_seconds,total_seconds,converged,iterations,final_residual,
operator_residual_linf,density_linf,density_l2,
interior_linf,interior_l2,density_linf_order,density_l2_order,
interior_linf_order,interior_l2_order,
edge_mismatch_linf,edge_mismatch_weighted_rms,
exact_edge_mismatch_linf,exact_edge_mismatch_order,
edge_reduction_ratio,constraint_rows,constraint_rank,
expected_non_g1_connections,covered_non_g1_connections,
duplicate_connection_intervals,g1_constraint_rows,
unrelated_constraint_rows,reduced_order_rows,
constant_constraint_defect,projected_constraint_defect,
projection_idempotence_defect,constant_projection_defect,
geometry_diagnostics_pass,owner_invariants_pass,
shared_preprocess_pass,row_pass
```

`edge_diagnostics.csv` writes one row per edge sample with its connection
index, patch/edge pair, both mapped parameters, physical weight,
mapped-point gap, reduced-order flags, and exact/unconstrained/projected
signed mismatches. This preserves evidence needed to locate a failed
edge without rerunning GMRES.

Support:

```text
--neumann-edge-continuity-study [N ...]
KFBIM_3D_NEUMANN_EDGE_CONTINUITY_OUTPUT_DIR
```

Defaults are `32 64`. If `128` is requested, finish and evaluate all
`32,64` pairs first; return nonzero without starting `N=128` unless the
coarse pilot passes every structural and numerical gate.

- [ ] **Step 5: Build and verify the CLI GREEN**

Run:

```powershell
cmake --build build --config Release `
    --target neumann_exterior_zero_trace_3d `
             neumann_edge_continuity_3d_test -- /m:1
$help = .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --help | Out-String
if ($help -notmatch '--neumann-edge-continuity-study') {
    throw 'Neumann edge-continuity route missing'
}
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --neumann-edge-continuity-study 16
```

Expected: build and help check pass; invalid `N=16` exits nonzero before
geometry setup and reports the accepted prefixes.

- [ ] **Step 6: Run an `N=32` integration smoke**

```powershell
$env:KFBIM_3D_NEUMANN_EDGE_CONTINUITY_OUTPUT_DIR = `
    'output/neumann_edge_continuity_3d_n32_smoke'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --neumann-edge-continuity-study 32
```

Require:

- exit code 0;
- six summary rows: three poses times two density spaces;
- three unique shared preprocess fingerprints, one per pose;
- all solves converge within 80 iterations;
- all geometry and owner invariants pass;
- projected mismatch and all three projection defects satisfy the
  structural thresholds;
- two-level gates are `not_evaluated`, not falsely passed;
- existing `--neumann-rigid-study` and `--neumann-owner-study` output
  schemas remain unchanged.

- [ ] **Step 7: Run focused regressions and commit**

```powershell
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
.\build\apps\Release\neumann_rigid_transform_study_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
```

Expected: every executable exits 0 and the diff check is clean.

Commit:

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add Neumann edge-continuity A/B study"
```

---

### Task 5: Numerical pilot, conditional refinement, and checked-in report

**Files:**
- Create: `docs/superpowers/results/2026-07-25-3d-neumann-edge-continuity-pilot.md`
- Generate, untracked: `output/neumann_edge_continuity_3d/*.csv`

- [ ] **Step 1: Run the complete `N=32,64` pilot**

```powershell
$env:KFBIM_3D_NEUMANN_EDGE_CONTINUITY_OUTPUT_DIR = `
    'output/neumann_edge_continuity_3d'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --neumann-edge-continuity-study 32 64
```

The executable checkpoints after each A/B pair. If a row reaches the
80-iteration cap, produces a non-finite value, violates a structural
invariant, or changes geometry/owner diagnostics, stop the remaining
solve sequence and diagnose before claiming a numerical comparison.

- [ ] **Step 2: Independently audit all CSVs**

Use `Import-Csv` to require:

- 12 unique summary rows;
- exactly two modes for every `(case_id,N)`;
- exactly three cases and two levels;
- no duplicate `(case_id,N,density_space)` key;
- all requested topology, projection, geometry, and owner gates agree
  with values independently recomputed from the CSVs;
- exact-density mismatch ratios are at least 6;
- projected numerical mismatch reductions are at least `1e4`;
- acceptance statuses agree with independently recomputed GMRES, error,
  and refinement comparisons.

Print the review table:

```text
case_id,N,density_space,
density_linf,density_linf_order,
interior_linf,interior_linf_order,
edge_mismatch_linf,
iterations,final_residual,
pipeline_setup_seconds,projector_setup_seconds,solve_seconds,total_seconds
```

- [ ] **Step 3: Run `N=128` only on a passing pilot**

If and only if `acceptance.csv` marks the complete `32,64` pilot
`overall_pass`, run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --neumann-edge-continuity-study 32 64 128
```

The driver itself must recheck the coarse acceptance before beginning
`N=128`. Report `N=128` as extended evidence, not as a prerequisite for
the pilot conclusion.

If the pilot fails, do not loosen tolerances or add a penalty. Record
which gate failed and recommend the next causal experiment:

- phase-continuous blending of crossing-owner interpolation if edge
  continuity reduces mismatch but not GMRES/error;
- mass-scaled constant-mode stabilization if the remaining residual is
  dominated by the mean equation;
- a globally shared NURBS edge basis only if the projection succeeds
  numerically but its setup/application cost is unacceptable.

- [ ] **Step 4: Write the numerical result report**

The report must state:

- exact branch/commit and Release command;
- fixed route, owner, Cauchy, GMRES, poses, and levels;
- all structural diagnostics and any reduced-order rows;
- per-row density/interior maximum errors and observed orders;
- GMRES iterations, final residual, and residual-shape comparison;
- exact and solved edge mismatches before/after projection;
- geometry, pipeline, projector, solve, and total times;
- whether every acceptance gate passed;
- a causal conclusion: supported, rejected, or inconclusive;
- the next experiment selected by the result, without presenting a
  failed pilot as a production improvement.

- [ ] **Step 5: Run final verification**

```powershell
cmake --build build --config Release -- /m:1
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
.\build\apps\Release\neumann_rigid_transform_study_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
git diff --check
```

Expected: Release build succeeds, all eight executables exit 0, and the
diff check is clean.

- [ ] **Step 6: Commit the audited result**

```powershell
git add docs/superpowers/results/2026-07-25-3d-neumann-edge-continuity-pilot.md
git commit -m "docs: report Neumann edge-continuity pilot"
```
