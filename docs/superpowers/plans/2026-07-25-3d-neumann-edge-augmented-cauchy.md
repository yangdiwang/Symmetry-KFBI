# 3D Neumann Edge-Augmented Cauchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Neumann reconstruction that fits one shared value jump at each non-G1 edge sample from both incident faces, inserts those values as soft samples in nearby local Cauchy fits, and compares the result with the unchanged legacy route.

**Architecture:** Precompute a sparse two-sided edge-value map from the fixed NURBS geometry and a set of augmented patch-centered Cauchy maps for only the centers whose legacy stencil reaches a feature edge. During a matrix-free application, compute shared edge values by sparse matrix-vector products, obtain the legacy coefficient matrix, and overwrite only affected rows with the augmented maps. Keep the legacy density space, spread, FFT solve, crossing-owner restrict, and GMRES system unchanged so the A/B study isolates edge samples in the Cauchy reconstruction.

**Tech Stack:** C++17, Eigen dense/sparse linear algebra and SVD, native NURBS topology and surface DOFs, degree-three harmonic polynomial space, ZFFT, restarted GMRES, CMake/Visual Studio x64 Release, CSV checkpoints.

## Global Constraints

- Geometry is the native twelve-patch NURBS L-prism.
- The ordinary face stencil remains degree-three `G1Nearest` with exactly 48 value and 28 normal conditions.
- Each non-G1 edge auxiliary fit uses exactly 24 value and 14 normal conditions from each incident side.
- Auxiliary face-side selection may traverse G1 neighbors only and may not cross the target or any other non-G1 connection.
- Each non-G1 interval uses `max(4, ceil(connection_length / h))` midpoint samples and honors partial intervals and `reversed`.
- Each eligible patch-centered fit receives the nearest four edge samples from every incident non-G1 connection entering its legacy value-stencil radius.
- Edge rows are soft value conditions with the existing distance weight and an exact weight scale of `1.0`.
- The first version has no edge or vertex GMRES unknown, no vertex fit, no endpoint sample, no penalty, and no global density projection.
- Corner-near centers receive samples from all incident non-G1 edges; duplicate endpoint samples cannot occur because edge samples are interval midpoints.
- The old Cauchy route remains the default and must be bitwise unchanged.
- The A/B study compares `none` with `non_g1_auxiliary_values` on the same patch-independent Neumann density space.
- The A/B pair uses `JointTricubicCrossingOwner`, `RegionClosestHybrid`, GMRES tolerance `2e-10`, restart 80, and cap 80.
- Pilot poses are exactly `baseline`, `ty_m0083`, and `rot_axis123_17deg`.
- Default pilot levels are exactly `32,64`; `N=128` is accepted only as the prefix `32,64,128` and runs only after the complete `N=32,64` pilot passes.
- Geometry, surface DOFs, crossing rows, owner preprocessing, FFT grid,
  prescribed Neumann data, exact fields, and solver tolerances are shared
  between the two modes.  Each mode builds its own consistent discrete
  right-hand side through that mode's Cauchy reconstruction.
- Rank-deficient, asymmetric, topologically unrelated, or non-finite fits fail setup with IDs; there is no reduced-degree or topological-nearest fallback.
- Generated data remains untracked under `output/neumann_edge_cauchy_3d`.

## File and Responsibility Map

| File | Responsibility |
|---|---|
| `apps/neumann_edge_augmented_cauchy_3d.hpp` | Public edge-sample, sparse edge-value map, attachment, augmented local-map, diagnostics, and runtime interfaces |
| `apps/neumann_edge_augmented_cauchy_3d.cpp` | Edge geometry, symmetric G1-side selection, weighted edge fits, local edge attachments, affected-center maps, and runtime matvecs |
| `apps/neumann_edge_augmented_cauchy_3d_test.cpp` | Edge reproduction, topology, map linearity, attachment, corner, rigid-covariance, failure, and overwrite tests |
| `apps/neumann_edge_cauchy_study_3d.hpp` | Pure measurement, derived-row, acceptance, level-normalization, and exit-decision interfaces |
| `apps/neumann_edge_cauchy_study_3d.cpp` | Pure A/B comparison and coarse/extended acceptance evaluation |
| `apps/neumann_edge_cauchy_study_3d_test.cpp` | Missing/duplicate keys, all acceptance gates, N=128 isolation, and process-exit tests |
| `apps/neumann_exterior_zero_trace_3d.cpp` | Optional augmented coefficient route, shared-pipeline pair worker, edge discrepancy, CLI, checkpoints, and numerical rows |
| `apps/kfbi_phase_profile_3d.hpp/.cpp/.test.cpp` | Separate `edge_auxiliary_values` runtime timing phase |
| `apps/CMakeLists.txt` | New library sources and focused test executables |
| `docs/superpowers/results/2026-07-25-3d-neumann-edge-augmented-cauchy.md` | Checked-in numerical evidence and conclusion |

---

### Task 1: Two-sided auxiliary edge-value map

**Files:**
- Create: `apps/neumann_edge_augmented_cauchy_3d.hpp`
- Create: `apps/neumann_edge_augmented_cauchy_3d.cpp`
- Create: `apps/neumann_edge_augmented_cauchy_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: `NativeNurbsSurface3D`, `SurfaceDofCloud3D`, `PatchEdge3D`, `HarmonicPolynomialSpace3D`, and `svd_pseudoinverse_3d`.
- Produces:

```cpp
namespace kfbim::app3d {

enum class NeumannEdgeCauchyMode3D {
    None,
    NonG1AuxiliaryValues
};

const char* neumann_edge_cauchy_mode_name_3d(
    NeumannEdgeCauchyMode3D mode);

struct NeumannEdgeAuxiliaryOptions3D {
    int degree = 3;
    int value_samples_per_side = 24;
    int normal_samples_per_side = 14;
    int minimum_edge_samples = 4;
    double rank_relative_cutoff = 3.0e-12;
};

struct NeumannEdgeAuxiliarySample3D {
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
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d oriented_tangent = Eigen::Vector3d::Zero();
    Eigen::Vector3d first_normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d second_normal = Eigen::Vector3d::Zero();
    Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
    double mapped_point_gap = 0.0;
    int first_owner_dof = -1;
    int second_owner_dof = -1;
    std::vector<int> first_value_dofs;
    std::vector<int> first_normal_dofs;
    std::vector<int> second_value_dofs;
    std::vector<int> second_normal_dofs;
    double condition = 0.0;
};

struct NeumannEdgeAuxiliaryDiagnostics3D {
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int edge_sample_count = 0;
    int unrelated_sample_count = 0;
    int asymmetric_sample_count = 0;
    int rank_deficient_fit_count = 0;
    double mapped_point_gap_max = 0.0;
    double frame_orthogonality_defect_max = 0.0;
    double harmonic_cubic_reproduction_defect_max = 0.0;
    double condition_max = 0.0;
    bool pass = false;
};

struct NeumannEdgeAuxiliaryValueMap3D {
    int surface_size = 0;
    Eigen::SparseMatrix<double, Eigen::RowMajor> value_map;
    Eigen::SparseMatrix<double, Eigen::RowMajor> normal_map;
    std::vector<NeumannEdgeAuxiliarySample3D> samples;
    NeumannEdgeAuxiliaryDiagnostics3D diagnostics;
};

NeumannEdgeAuxiliaryValueMap3D
build_neumann_edge_auxiliary_value_map_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    const NeumannEdgeAuxiliaryOptions3D& options = {});

Eigen::VectorXd evaluate_neumann_edge_values_3d(
    const NeumannEdgeAuxiliaryValueMap3D& map,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump);

} // namespace kfbim::app3d
```

The pilot options are checked configuration, not tuning knobs.  Reject
any value other than degree 3, 24/14 samples per side, minimum four edge
samples, and relative cutoff `3e-12`.

- [ ] **Step 1: Register the focused target and write failing geometry, topology, and reproduction tests**

Add `neumann_edge_augmented_cauchy_3d_test` to `apps/CMakeLists.txt`,
link it to `kfbim_3d_app_geometry`, and set C++17. Include the not-yet
created header in the test so the initial build proves the API is absent.

The first test fixture is:

```cpp
const auto surface =
    make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
const double h = 3.0 / 32.0;
const auto cloud = make_native_surface_dofs_3d(surface, h);
const auto edge_map =
    build_neumann_edge_auxiliary_value_map_3d(surface, cloud, h);
```

Require:

```cpp
require(edge_map.surface_size == static_cast<int>(cloud.dofs.size()),
        "edge map has wrong surface size");
require(edge_map.value_map.rows()
            == static_cast<int>(edge_map.samples.size())
        && edge_map.normal_map.rows() == edge_map.value_map.rows()
        && edge_map.value_map.cols() == edge_map.surface_size
        && edge_map.normal_map.cols() == edge_map.surface_size,
        "edge map dimensions are inconsistent");
require(edge_map.diagnostics.expected_non_g1_connections == 22
        && edge_map.diagnostics.covered_non_g1_connections == 22,
        "L-prism non-G1 coverage is wrong");
```

For every sample, assert:

- `sample_count == max(4, ceil(connection_length / h))`;
- literal indices `0..sample_count-1` occur once;
- first and second parameters honor the two partial intervals;
- the second parameter follows `reversed`;
- mapped NURBS points agree within `1e-11` times geometry diameter;
- `frame.transpose()*frame` differs from identity by at most `1e-12`
  and its determinant is positive;
- each of the two value lists has size 24;
- each of the two normal lists has size 14;
- every selected patch is reachable from the correct incident patch by
  G1 edges only;
- neither side list contains a DOF owned by the opposite non-G1 side;
- `first_owner_dof` and `second_owner_dof` are the first
  distance-sorted value DOFs on their sides;
- every condition number is finite and positive.

For exact reproduction, loop over all 16 columns of
`HarmonicPolynomialSpace3D(3)`.  For one edge sample at a time, generate
value and physical normal derivative data from the basis expressed in
that sample's stored frame and scaled coordinate.  Apply the sparse
maps and require:

```cpp
require(std::abs(edge_value[sample_index] - exact_at_origin) <= 1.0e-11,
        "edge map does not reproduce a harmonic cubic");
```

Also form two finite vectors `v1,n1,v2,n2` and verify linearity:

```cpp
const Eigen::VectorXd lhs =
    evaluate_neumann_edge_values_3d(map, v1 + 0.37*v2, n1 + 0.37*n2);
const Eigen::VectorXd rhs =
    evaluate_neumann_edge_values_3d(map, v1, n1)
    + 0.37*evaluate_neumann_edge_values_3d(map, v2, n2);
require((lhs-rhs).lpNorm<Eigen::Infinity>() <= 2.0e-13,
        "edge value map is not linear");
```

- [ ] **Step 2: Build and verify RED**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_augmented_cauchy_3d_test -- /m:1
```

Expected: compilation fails because
`neumann_edge_augmented_cauchy_3d.hpp` is missing.

- [ ] **Step 3: Implement deterministic edge geometry and symmetric side selection**

In `neumann_edge_augmented_cauchy_3d.cpp`, implement private helpers for:

```cpp
double connection_length_8_point_gauss(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsPatchEdgeInterval3D& interval);
double edge_parameter(
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double s);
Eigen::Vector3d edge_point(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double parameter);
Eigen::Vector3d oriented_edge_tangent(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double parameter);
std::vector<int> nearest_g1_side_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int incident_patch,
    int forbidden_patch,
    const Eigen::Vector3d& point,
    int count);
```

`nearest_g1_side_dofs` must:

1. start from `incident_patch`;
2. expand breadth-first through `smooth_neighbors` only;
3. exclude `forbidden_patch`;
4. stop after at least one G1 ring and enough candidate DOFs;
5. sort candidates by `(squared_distance, dof_id)`;
6. throw with the connection/sample/side IDs when fewer than `count`
   candidates exist.

Construct the edge frame as:

```cpp
e0 = normalized oriented edge tangent;
bisector = first_normal + second_normal;
e1 = normalized(bisector - bisector.dot(e0)*e0);
e2 = e0.cross(e1);
frame.col(0) = e0;
frame.col(1) = e1;
frame.col(2) = e2;
```

Evaluate both mapped points first.  Reject their gap before constructing
`point = 0.5*(first_point+second_point)`.  Form tangents with respect to
the common normalized connection parameter; reverse the second tangent
when `connection.reversed` is true.  Require the normalized tangent dot
product to be at least `1-1e-11`, then normalize their sum as the stored
common oriented tangent.

Reject a tangent, bisector, or final determinant below `1e-12`.
Compute the physical length independently on both incident intervals.
Reject
`abs(first_length-second_length) >
1e-11*max(first_length,second_length)` and use their arithmetic mean for
the sample count.  This prevents one incident parametrization from
silently controlling the shared sampling.

- [ ] **Step 4: Implement weighted edge fits and sparse maps**

For each edge sample, build a `76 x 16` design matrix in the fixed order:

```text
24 first-side values
24 second-side values
14 first-side normal derivatives
14 second-side normal derivatives
```

Use the current Cauchy coordinate and weight rules:

```cpp
xi = frame.transpose() * (sample_point - edge_point) / h;
w_value = 1.0 / pow(0.35 + xi.norm(), 2.0);
w_normal = 0.85 / pow(0.35 + xi.norm(), 2.0);
```

Value rows use `space.basis(xi.x(), xi.y(), xi.z())`.  Normal rows use
the sample normal resolved in the edge frame and
`space.gradient(xi.x(), xi.y(), xi.z())`.  Multiply the normal columns
of the pseudoinverse by `h`, matching the existing
`PanelCenterCauchyFit3D` convention.

Require the smallest singular value to exceed
`3e-12 * largest_singular_value`.  The edge value is the basis at the
origin applied to the coefficient map.  Accumulate its value and normal
coefficients into row-major sparse matrices; reject non-finite entries.

Compute `harmonic_cubic_reproduction_defect_max` by applying each sparse
row to the same 16 local harmonic basis data used by the unit test.

- [ ] **Step 5: Add direct-map, shared-value, rigid-covariance, and negative tests**

For deterministic finite value/normal vectors, independently rebuild the
weighted `76 x 16` system for every sample, solve it directly, evaluate
the direct polynomial at the origin, and compare with the sparse-map
value to `1e-12`.  This test must not call the production sparse-row
assembly helper.

Verify that each physical sample produces exactly one sparse-map row and
that both incident owner IDs refer to this same sample index; no
per-incident-side duplicate row or value is allowed.  Rebuild the
baseline, translated, and rotated L-prism with the existing
rigid-transform helpers; require identical connection/sample keys and
value predictions for transformed harmonic data within `1e-11`.  Record
the maximum as the rigid-covariance defect in the test output.

Add tests that:

- mutate one connection to an invalid patch and require
  `std::invalid_argument`;
- erase one smooth-neighbor relation needed by a deliberately restricted
  fixture and require an insufficient-sample error;
- use `make_native_surface_dofs_3d(surface, 10.0)` and require rejection
  rather than lower order;
- change any fixed pilot option (`degree`, 24/14 side counts, minimum
  edge samples, or relative cutoff), or use a nonpositive `h`, and
  require explicit input rejection;
- replace one side's normals with a degenerate frame fixture and require
  an error naming the connection/sample.

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_augmented_cauchy_3d_test `
           native_nurbs_surface_3d_test `
           native_nurbs_surface_transform_3d_test -- /m:1
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
```

Expected: all three exit 0 and the new test prints
`3D Neumann auxiliary edge-value tests passed`.

- [ ] **Step 6: Commit**

```powershell
git add apps/CMakeLists.txt `
        apps/neumann_edge_augmented_cauchy_3d.hpp `
        apps/neumann_edge_augmented_cauchy_3d.cpp `
        apps/neumann_edge_augmented_cauchy_3d_test.cpp
git commit -m "feat: build Neumann auxiliary edge values"
```

---

### Task 2: Edge-attached local Cauchy maps

**Files:**
- Modify: `apps/neumann_edge_augmented_cauchy_3d.hpp`
- Modify: `apps/neumann_edge_augmented_cauchy_3d.cpp`
- Modify: `apps/neumann_edge_augmented_cauchy_3d_test.cpp`

**Interfaces:**
- Consumes: the Task 1 edge map and the exact legacy value/normal stencil
  ID lists built by `build_cauchy_stencils`.
- Produces:

```cpp
namespace kfbim::app3d {

struct NeumannEdgeFaceStencil3D {
    std::vector<int> value_dofs;
    std::vector<int> normal_dofs;
};

struct NeumannEdgeLocalMap3D {
    int center_dof = -1;
    std::vector<int> value_dofs;
    std::vector<int> normal_dofs;
    std::vector<int> edge_sample_indices;
    Eigen::MatrixXd value_map;
    Eigen::MatrixXd normal_map;
    Eigen::MatrixXd edge_map;
    double condition = 0.0;
};

struct NeumannEdgeAugmentedCauchyOptions3D {
    int degree = 3;
    int edge_samples_per_connection = 4;
    double edge_weight_scale = 1.0;
    double rank_relative_cutoff = 3.0e-12;
};

struct NeumannEdgeAugmentedCauchyDiagnostics3D {
    NeumannEdgeAuxiliaryDiagnostics3D edge;
    int affected_center_count = 0;
    int corner_center_count = 0;
    int unrelated_attachment_count = 0;
    int rank_deficient_local_fit_count = 0;
    int factorization_count = 0;
    double harmonic_cubic_reproduction_defect_max = 0.0;
    double local_condition_max = 0.0;
    bool pass = false;
};

class NeumannEdgeAugmentedCauchy3D {
public:
    int surface_size() const;
    int edge_sample_count() const;
    const NeumannEdgeAuxiliaryValueMap3D& edge_value_map() const;
    const std::vector<NeumannEdgeLocalMap3D>& local_maps() const;
    const NeumannEdgeAugmentedCauchyDiagnostics3D& diagnostics() const;

    Eigen::VectorXd edge_values(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const;

    void overwrite_affected_coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        const Eigen::VectorXd& edge_values,
        Eigen::MatrixXd& coefficients) const;

private:
    friend NeumannEdgeAugmentedCauchy3D
    build_neumann_edge_augmented_cauchy_3d(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        double h,
        const std::vector<NeumannEdgeFaceStencil3D>& face_stencils,
        const NeumannEdgeAuxiliaryOptions3D& edge_options,
        const NeumannEdgeAugmentedCauchyOptions3D& local_options);
    int surface_size_ = 0;
    NeumannEdgeAuxiliaryValueMap3D edge_value_map_;
    std::vector<NeumannEdgeLocalMap3D> local_maps_;
    NeumannEdgeAugmentedCauchyDiagnostics3D diagnostics_;
};

NeumannEdgeAugmentedCauchy3D
build_neumann_edge_augmented_cauchy_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    const std::vector<NeumannEdgeFaceStencil3D>& face_stencils,
    const NeumannEdgeAuxiliaryOptions3D& edge_options = {},
    const NeumannEdgeAugmentedCauchyOptions3D& local_options = {});

} // namespace kfbim::app3d
```

The local options are also fixed pilot configuration: degree 3, four
samples per incident connection, weight scale `1.0`, and relative cutoff
`3e-12`.  Reject any other values.

- [ ] **Step 1: Write failing attachment, corner, and overwrite tests**

Build the exact legacy stencil input:

```cpp
std::vector<NeumannEdgeFaceStencil3D> stencils;
for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
    stencils.push_back({
        nearest_g1_cauchy_dofs(surface, cloud, center, 48),
        nearest_g1_cauchy_dofs(surface, cloud, center, 28)});
}
const auto augmented =
    build_neumann_edge_augmented_cauchy_3d(
        surface, cloud, h, stencils);
```

For each local map, independently recompute the legacy value-stencil
radius and require:

- a group exists exactly when its nearest sample is within that radius;
- samples are grouped by connection;
- every group contains exactly the nearest four samples by
  `(distance, sample_index)`, while samples two through four may lie just
  outside the radius;
- the center patch lies in exactly one G1 side component of the
  connection;
- no unrelated connection is attached;
- `value_map.cols()==48`, `normal_map.cols()==28`, and
  `edge_map.cols()==edge_sample_indices.size()`;
- all three maps have 16 rows and finite entries;
- the center ID occurs only once in `local_maps`.

Require at least one center with edge samples from two or more non-G1
connections and record it as a corner center.  Require at least one far
center absent from `local_maps`.

Initialize a coefficient matrix with deterministic nonzero sentinel rows,
call `overwrite_affected_coefficients`, and require every absent center
to remain bitwise identical.

- [ ] **Step 2: Build and verify RED**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_augmented_cauchy_3d_test -- /m:1
```

Expected: compilation fails because
`NeumannEdgeAugmentedCauchy3D` and its builder do not exist.

- [ ] **Step 3: Implement topology-aware attachments**

For each center:

1. compute the maximum Euclidean distance from the center to its 48
   legacy value DOFs;
2. determine its G1 component with `smooth_patch_component`;
3. consider a non-G1 connection only when exactly one incident patch is
   in that component;
4. consider the connection eligible when its nearest edge sample enters
   that radius;
5. sort all samples of that connection by
   `(squared_distance, sample_index)` and keep the nearest four, even
   when the fourth sample lies just outside the radius near an interval
   endpoint;
6. append groups in `(connection_index, sample_index)` order.

The Task 1 minimum of four samples guarantees a complete group.  Never
attach a partial group of one to three samples.

Count a center with at least two connection groups as a corner center.
Reject a connection whose two incident patches are both in or both out
of the center's G1 component when it would otherwise be selected.

- [ ] **Step 4: Implement affected-center weighted maps**

For each affected center, rebuild the complete degree-three harmonic
least-squares design:

```text
48 ordinary value rows
28 ordinary normal rows
4 * incident_connection_count edge value rows
```

Use the center's existing
`(tangent1,tangent2,normal)` frame and the existing `h` scaling.
Ordinary rows and weights must be byte-for-byte the formulas at
`PanelCenterCauchyFit3D::build_map`.  Each edge row uses:

```cpp
xi = center_frame.transpose() * (edge_point-center_point) / h;
design.row = HarmonicPolynomialSpace3D(3).basis(
    xi.x(), xi.y(), xi.z());
sqrt_weight = std::sqrt(
    1.0 / std::pow(0.35 + xi.norm(), 2.0));
```

`edge_weight_scale` must equal exactly `1.0`; reject other values in this
pilot.  Split the weighted pseudoinverse into `value_map`, `normal_map`,
and `edge_map`; multiply normal columns by `h`.

`overwrite_affected_coefficients` gathers only the stored DOFs and edge
sample indices, computes each affected row, and performs no SVD,
topology query, nearest-neighbor search, or allocation proportional to
all surface DOFs.

- [ ] **Step 5: Add exact-field and mutation tests**

For all 16 harmonic-cubic basis members:

1. generate value and physical normal derivative data on all surface
   DOFs;
2. evaluate edge values;
3. start from any finite legacy coefficient matrix;
4. overwrite affected rows;
5. evaluate every affected polynomial at its center and at all attached
   edge points.

Require coefficient and evaluation defects at most `1e-11`.  Store the
maximum of the edge-map defect and all affected-local-map defects in
`diagnostics.harmonic_cubic_reproduction_defect_max`.

For every auxiliary edge sample, require that both `first_owner_dof` and
`second_owner_dof` have an affected local map and that the corresponding
`edge_sample_indices` contain this sample.  This makes the later
incident-polynomial diagnostic unambiguous.

Rebuild all maps after the same translation and rotation used in Task 1.
Require identical center/connection/sample attachment keys and require
all transformed harmonic-cubic coefficient/evaluation defects to remain
at most `1e-11`.  This extends rigid covariance from the edge map to the
combined local map.

Add mutations for:

- a face-stencil vector with the wrong number of centers;
- a value list of 47 or normal list of 27;
- duplicate edge sample IDs;
- an unrelated edge attachment;
- any local pilot option different from degree 3, four edge samples,
  weight scale `1.0`, or relative cutoff `3e-12`;
- a deliberately repeated ordinary sample producing rank deficiency;
- wrong-sized value, normal, edge-value, or coefficient inputs.

Verify evaluation does not change
`diagnostics.factorization_count`, proving all factorizations occur at
setup.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_augmented_cauchy_3d_test `
           harmonic_polynomial_space_3d_test `
           native_nurbs_surface_3d_test -- /m:1
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\harmonic_polynomial_space_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
```

Expected: all exit 0; far rows remain bitwise unchanged; at least one
corner center has multiple edge groups.

Commit:

```powershell
git add apps/neumann_edge_augmented_cauchy_3d.hpp `
        apps/neumann_edge_augmented_cauchy_3d.cpp `
        apps/neumann_edge_augmented_cauchy_3d_test.cpp
git commit -m "feat: add edge samples to local Cauchy fits"
```

---

### Task 3: Matrix-free Neumann route and phase timing

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:681-818`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1586-2046`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:2806-2932`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:3074-3184`
- Modify: `apps/kfbi_phase_profile_3d.hpp`
- Modify: `apps/kfbi_phase_profile_3d.cpp`
- Modify: `apps/kfbi_phase_profile_3d_test.cpp`
- Modify: `apps/neumann_edge_augmented_cauchy_3d_test.cpp`

**Interfaces:**
- Consumes: Task 2 `NeumannEdgeAugmentedCauchy3D`.
- Produces:
  - optional augmented setup inside `PanelCenterHarmonicJetKFBI3D`;
  - a Cauchy mode propagated through LHS, RHS, final reconstruction, and
    diagnostics;
  - `PhaseProfileKind3D::EdgeAuxiliaryValues`;
  - `run_neumann_case` with an explicit `NeumannEdgeCauchyMode3D` and no density
    projector on this route.

- [ ] **Step 1: Extend the phase-profiler test and verify RED**

Insert `EdgeAuxiliaryValues` immediately before `CauchyCoefficients`.
Change the literal expected-name array from 17 to 18 and include:

```cpp
"edge_auxiliary_values",
"cauchy_coefficients",
```

The algorithm-phase boundary increases from 15 to 16.  Add:

```cpp
profile.add(PhaseProfileKind3D::EdgeAuxiliaryValues, 0.125, 3);
require(profile.record(
    PhaseProfileKind3D::EdgeAuxiliaryValues).calls == 3,
    "edge-value phase call count is wrong");
```

Run:

```powershell
cmake --build build --config Release `
  --target kfbim_phase_profile_3d_test -- /m:1
```

Expected: compilation fails because `EdgeAuxiliaryValues` is absent.

- [ ] **Step 2: Add the timing kind and verify the profiler**

Update the enum, name switch, algorithm-phase classification, CSV row
count, and tests.  Run:

```powershell
cmake --build build --config Release `
  --target kfbim_phase_profile_3d_test -- /m:1
.\build\apps\Release\kfbim_phase_profile_3d_test.exe
```

Expected: exit 0.

- [ ] **Step 3: Add an optional augmented fit to the pipeline**

Extend `PanelCenterHarmonicJetKFBI3D` construction with:

```cpp
bool build_neumann_edge_augmented_cauchy = false
```

When true, convert every `CauchyStencil` into
`NeumannEdgeFaceStencil3D` and build one
`NeumannEdgeAugmentedCauchy3D`.  Store it in:

```cpp
std::unique_ptr<app3d::NeumannEdgeAugmentedCauchy3D>
    neumann_edge_augmented_cauchy_;
```

Expose a checked const accessor for the A/B worker:

```cpp
const app3d::NeumannEdgeAugmentedCauchy3D&
neumann_edge_augmented_cauchy() const;
```

Add a private coefficient router:

```cpp
Eigen::MatrixXd cauchy_coefficients(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    app3d::NeumannEdgeCauchyMode3D mode) const;
```

Its exact behavior is:

```cpp
if (mode == NeumannEdgeCauchyMode3D::None) {
    return profile_phase_3d(
        phase_profile_, PhaseProfileKind3D::CauchyCoefficients, 1,
        [&] { return fit_.coefficients(value_jump, normal_jump); });
}
if (!neumann_edge_augmented_cauchy_)
    throw std::runtime_error("edge-augmented Cauchy was not initialized");
const Eigen::VectorXd edge_values = profile_phase_3d(
    phase_profile_, PhaseProfileKind3D::EdgeAuxiliaryValues, 1,
    [&] {
        return neumann_edge_augmented_cauchy_->edge_values(
            value_jump, normal_jump);
    });
return profile_phase_3d(
    phase_profile_, PhaseProfileKind3D::CauchyCoefficients, 1,
    [&] {
        Eigen::MatrixXd coefficients =
            fit_.coefficients(value_jump, normal_jump);
        neumann_edge_augmented_cauchy_->overwrite_affected_coefficients(
            value_jump, normal_jump, edge_values, coefficients);
        return coefficients;
    });
```

The overwrite itself belongs to `CauchyCoefficients`; do not time it a
second time or include geometry/setup work in the runtime phase.

- [ ] **Step 4: Propagate the mode through every Neumann field build**

Add a defaulted Cauchy mode to:

```cpp
HarmonicJetField3D evaluate(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    NeumannEdgeCauchyMode3D cauchy_mode =
        NeumannEdgeCauchyMode3D::None) const;

HarmonicJetField3D field_from_grid_and_jumps(
    const Eigen::VectorXd& potential,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    NeumannEdgeCauchyMode3D cauchy_mode =
        NeumannEdgeCauchyMode3D::None) const;
```

Rename the existing restriction argument to `restrict_mode`; never reuse
`mode` for both enums.

`ExteriorZeroTraceOperator3D` stores the Cauchy mode and applies it in:

- `apply(value_jump, zero_normal)`;
- `right_hand_side(zero_value, prescribed_normal_jump)`.

`solve_exterior_zero_trace_neumann_3d` uses the same mode for:

- the operator;
- the projected or unprojected RHS construction;
- post-GMRES `pipeline.evaluate(value_jump, prescribed_normal_jump)`.

Change the full `run_neumann_case` tail to:

```cpp
ExteriorValueRestrictMode3D restrict_mode =
    ExteriorValueRestrictMode3D::JointTricubicCauchy,
app3d::NeumannEdgeCauchyMode3D cauchy_mode =
    app3d::NeumannEdgeCauchyMode3D::None,
std::vector<double>* residual_history = nullptr,
const app3d::NeumannEdgeContinuityProjector3D* edge_projector = nullptr,
Eigen::VectorXd* solved_value_jump = nullptr,
Eigen::MatrixXd* solved_coefficients = nullptr,
Eigen::VectorXd* prescribed_normal_jump_output = nullptr
```

Pass the Cauchy mode to the solver and populate all three optional outputs
from the already-computed solution and prescribed data.  Update every
existing positional caller to insert explicit
`NeumannEdgeCauchyMode3D::None`, preserving legacy semantics.  A density
projector remains available only for the separate legacy edge-continuity
command; the new edge-Cauchy study always passes `nullptr`.

- [ ] **Step 5: Verify legacy compatibility and augmented runtime**

Add a focused unit assertion that mode names are exactly:

```text
none
non_g1_auxiliary_values
```

Build the complete app and run the unchanged default N=16 L-prism route:

```powershell
cmake --build build --config Release `
  --target neumann_exterior_zero_trace_3d `
           neumann_edge_augmented_cauchy_3d_test `
           kfbim_phase_profile_3d_test -- /m:1
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\kfbim_phase_profile_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
```

Expected:

- all commands exit 0;
- the default output still reports `g1_nearest/degree3/48/28`;
- no edge-auxiliary phase call occurs in the default route;
- existing default numerical rows and route names are unchanged.

Run the relevant regression executables:

```powershell
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
```

Expected: all exit 0.

- [ ] **Step 6: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp `
        apps/kfbi_phase_profile_3d.hpp `
        apps/kfbi_phase_profile_3d.cpp `
        apps/kfbi_phase_profile_3d_test.cpp `
        apps/neumann_edge_augmented_cauchy_3d_test.cpp
git commit -m "feat: integrate edge-augmented Neumann Cauchy route"
```

---

### Task 4: Pure evaluator and dedicated A/B command

**Files:**
- Create: `apps/neumann_edge_cauchy_study_3d.hpp`
- Create: `apps/neumann_edge_cauchy_study_3d.cpp`
- Create: `apps/neumann_edge_cauchy_study_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:8070-8700`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:8700-8810`

**Interfaces:**
- Consumes: the Task 3 route and
  `RigidStudyCriterionStatus3D`/`combine_rigid_study_criteria_3d`.
- Produces:

```cpp
namespace kfbim::app3d {

struct NeumannEdgeCauchyMeasurement3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    NeumannEdgeCauchyMode3D mode =
        NeumannEdgeCauchyMode3D::None;
    bool finite_metrics = false;
    bool gmres_converged = false;
    int gmres_iterations = 0;
    double gmres_relative_residual = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double interior_linf = 0.0;
    double interior_l2 = 0.0;
    double incident_edge_discrepancy_linf = 0.0;
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int edge_sample_count = 0;
    int affected_center_count = 0;
    int corner_center_count = 0;
    int unrelated_sample_or_attachment_count = 0;
    int rank_deficient_fit_count = 0;
    double harmonic_cubic_reproduction_defect = 0.0;
    double edge_condition_max = 0.0;
    double local_condition_max = 0.0;
    double shared_setup_seconds = 0.0;
    double mode_runtime_seconds = 0.0;
    double total_seconds = 0.0;
    bool far_centers_bitwise_legacy = false;
    bool geometry_diagnostics_pass = false;
    bool owner_invariants_pass = false;
    bool shared_preprocess_pass = false;
};

struct NeumannEdgeCauchyDerivedRow3D {
    NeumannEdgeCauchyMeasurement3D measurement;
    double density_linf_order =
        std::numeric_limits<double>::quiet_NaN();
    double density_l2_order =
        std::numeric_limits<double>::quiet_NaN();
    double interior_linf_order =
        std::numeric_limits<double>::quiet_NaN();
    double interior_l2_order =
        std::numeric_limits<double>::quiet_NaN();
    double density_linf_ratio_to_legacy =
        std::numeric_limits<double>::quiet_NaN();
    double density_l2_ratio_to_legacy =
        std::numeric_limits<double>::quiet_NaN();
    double interior_linf_ratio_to_legacy =
        std::numeric_limits<double>::quiet_NaN();
    double interior_l2_ratio_to_legacy =
        std::numeric_limits<double>::quiet_NaN();
    double edge_discrepancy_ratio_to_legacy =
        std::numeric_limits<double>::quiet_NaN();
    RigidStudyCriterionStatus3D row_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannEdgeCauchyAcceptance3D {
    RigidStudyCriterionStatus3D completeness_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D structure_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D reproduction_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D gmres_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D error_guard_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D order_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D rigid_spread_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D edge_discrepancy_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D geometry_owner_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D extended_evidence_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D overall_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannEdgeCauchyEvaluation3D {
    std::vector<NeumannEdgeCauchyDerivedRow3D> rows;
    NeumannEdgeCauchyAcceptance3D acceptance;
    bool all_pass = false;
};

std::vector<int> normalize_neumann_edge_cauchy_levels_3d(
    std::vector<int> levels);

NeumannEdgeCauchyEvaluation3D
evaluate_neumann_edge_cauchy_study_3d(
    const std::vector<NeumannEdgeCauchyMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_pilot);

bool neumann_edge_cauchy_study_exit_pass_3d(
    const NeumannEdgeCauchyEvaluation3D& evaluation,
    bool require_complete_pilot);

} // namespace kfbim::app3d
```

- [ ] **Step 1: Register the evaluator test and write failing gate fixtures**

Add the study source to `kfbim_3d_app_geometry` and a new
`neumann_edge_cauchy_study_3d_test` executable.

Construct a literal passing `N=32,64` fixture with three case IDs and two
modes per `(case,N)`.  Use errors with order 2, augmented/legacy ratios
below 1.10, lower augmented edge discrepancy, augmented worst GMRES no
higher than legacy, and complete structural flags.

Add one mutation per gate:

- missing, duplicate, unknown-case, invalid-level, and invalid-mode key;
- uncovered connection or unrelated attachment;
- reproduction defect above `1e-11`;
- nonconvergence, residual above `2e-10`, or iteration above 80;
- one same-level error ratio above 1.10;
- one augmented order below 1.8;
- increased error or iteration spread across the three poses;
- one augmented incident-edge discrepancy not below legacy;
- failed geometry, owner, or shared-preprocess flag.

Require `require_complete_pilot=true` to mark a missing coarse key
`Fail`, while an intentional `N=32` prefix with
`require_complete_pilot=false` remains `NotEvaluated`.

- [ ] **Step 2: Add N=128 isolation tests and verify RED**

Append a complete clean N=128 pair and require:

```cpp
clean.acceptance.extended_evidence_pass == Status::Pass;
clean.acceptance.overall_pass == Status::Pass;
```

Then make every augmented N=128 row fail numerical comparisons and
require:

```cpp
bad.acceptance.extended_evidence_pass == Status::Fail;
bad.acceptance.overall_pass == Status::Pass;
neumann_edge_cauchy_study_exit_pass_3d(bad, true);
```

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_cauchy_study_3d_test -- /m:1
```

Expected: compilation fails because the evaluator API is absent.

- [ ] **Step 3: Implement the pure evaluator**

Key measurements by `(case_id,N,mode)` and reject duplicates.  Partition
rows explicitly into coarse `{32,64}` and extended `{128}` sets.

Coarse gates:

```text
structure:
  expected == covered > 0
  edge_sample_count > 0
  affected_center_count > 0
  corner_center_count > 0
  unrelated_sample_or_attachment_count == 0
  rank_deficient_fit_count == 0
  far_centers_bitwise_legacy

reproduction:
  harmonic_cubic_reproduction_defect <= 1e-11

gmres:
  every row converged, iterations <= 80, residual <= 2e-10
  max augmented iterations <= max legacy iterations

error_guard:
  every augmented/legacy density/interior Linf/L2 ratio <= 1.10

order:
  every augmented density/interior Linf/L2 N32->N64 order >= 1.8

rigid_spread:
  at each coarse N, augmented max(error)/min(error) <= legacy ratio
  augmented max(iter)-min(iter) <= legacy spread

edge_discrepancy:
  augmented discrepancy < legacy discrepancy for every case and N

geometry_owner:
  every geometry, owner, and shared-preprocess flag passes
```

Use a relative allowance of `64*epsilon` only in ratio/spread
comparisons.  Do not relax the 1.10, 1.8, 80, `2e-10`, or `1e-11`
thresholds.

`overall_pass` combines only coarse fields.  `extended_evidence_pass`
requires complete N=128 keys, the N=128 structure/reproduction/GMRES,
same-level error-ratio, rigid-spread, edge-discrepancy, and
geometry/owner rules, plus every augmented density/interior
N64-to-N128 order at least `1.8`.  It never changes coarse `overall_pass`
or process exit.

For `require_complete_pilot=false`, `neumann_edge_cauchy_study_exit_pass_3d`
requires all completed rows, structure, reproduction, GMRES execution,
and geometry/owner invariants to pass, but does not gate on incomplete
coarse comparison or order hypotheses.  For `true`, it returns the coarse
`all_pass` value.

- [ ] **Step 4: Verify evaluator GREEN**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_cauchy_study_3d_test `
           neumann_edge_augmented_cauchy_3d_test `
           neumann_rigid_transform_study_3d_test -- /m:1
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\neumann_rigid_transform_study_3d_test.exe
```

Expected: all exit 0.

- [ ] **Step 5: Implement the shared-pipeline A/B worker**

Add these worker-only records in
`neumann_exterior_zero_trace_3d.cpp`:

```cpp
using PhaseRecordArray3D = std::array<
    app3d::PhaseProfileRecord3D,
    app3d::phase_profile_kind_count_3d()>;

struct NeumannEdgeCauchyEdgeValueRow3D {
    int connection_index = -1;
    int sample_index = -1;
    int sample_count = 0;
    int first_owner_dof = -1;
    int second_owner_dof = -1;
    double first_value = 0.0;
    double second_value = 0.0;
    double shared_auxiliary_value = 0.0;
    double first_second_difference = 0.0;
    double first_shared_difference = 0.0;
    double second_shared_difference = 0.0;
};

struct NeumannEdgeCauchyPairRun3D {
    std::array<app3d::NeumannEdgeCauchyMeasurement3D, 2> measurements;
    std::array<std::vector<double>, 2> residual_histories;
    std::array<std::vector<NeumannEdgeCauchyEdgeValueRow3D>, 2>
        edge_value_rows;
    PhaseRecordArray3D shared_setup_phases{};
    std::array<PhaseRecordArray3D, 2> runtime_phase_deltas{};
    std::vector<app3d::NeumannEdgePreprocessInvariantSnapshot3D>
        owner_snapshots;
};

NeumannEdgeCauchyPairRun3D run_neumann_edge_cauchy_pair_3d(
    int N,
    const app3d::LPrismRigidStudyCase3D& study_case);
```

Build geometry, surface DOFs, G1-nearest 48/28 stencils, grid pair,
crossing rows, `RegionClosestHybrid` owner preprocessing, and the
edge-augmented maps once with one `PhaseProfile3D`.  Use the existing
nonoverlapping setup timers and remainder checks; require the setup phase
sum to match the common setup wall interval within
`max(1e-9,1e-8*setup_wall_time)`.  Capture that phase array immediately
after construction and a stable owner/preprocess snapshot before either
solve.

Before the solves, form one deterministic finite `value_jump` probe and
the prescribed `normal_jump`.  Call `field_from_grid_and_jumps` in both
modes on these identical inputs.  Build a boolean affected-center mask
from `local_maps`; compare every unmasked coefficient row bitwise and
store the result as `far_centers_bitwise_legacy` in both measurements.
Take the per-solve phase baseline only after this probe so it cannot
contaminate runtime deltas.  Copy the combined edge/local reproduction
defect from `NeumannEdgeAugmentedCauchy3D::diagnostics()` into both rows.

Run modes in the fixed order:

```cpp
{NeumannEdgeCauchyMode3D::None,
 NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues}
```

For each mode, snapshot all phase records immediately before the route.
Measure one wall interval around `run_neumann_case`; after it returns,
sum the deltas of `EdgeAuxiliaryValues`, `CauchyCoefficients`,
`SpreadRhsAssembly`, `FftBulkSolve`, `RestrictContinuedSamples`, and
`RestrictRecovery`, then add the checked nonnegative remainder to
`GmresAndOtherRoute`.  Take the final snapshot only after that addition,
subtract seconds and calls with checked nonnegative arithmetic, and
store only that delta in `runtime_phase_deltas`.  Require the sum of the
runtime delta seconds to equal the measured mode wall time within
`max(1e-9,1e-8*mode_wall_time)`.  Set `shared_setup_seconds` from the
common setup wall interval, `mode_runtime_seconds` from this route
interval, and `total_seconds` to their sum.

Pass `nullptr` for the density projector and pass output pointers for the
solved value jump, solved coefficient matrix, and prescribed normal jump.
Capture fingerprints, output digests, wrong-side queries, and
geometry-query counts before and after each GMRES solve.  Require the six
snapshots (stable, before/after both solves, final) to agree.

For each edge sample, evaluate the solved coefficient matrix at the edge
point using `first_owner_dof` and `second_owner_dof`.  For an owner DOF,
compute

```cpp
const Eigen::Vector3d d = (sample.point-owner.point) / h;
const Eigen::Vector3d xi(
    d.dot(owner.tangent1),
    d.dot(owner.tangent2),
    d.dot(owner.normal));
const double owner_value =
    HarmonicPolynomialSpace3D(3)
        .basis(xi.x(), xi.y(), xi.z())
        .dot(solved_coefficients.row(owner_dof).transpose());
```

Compute the shared auxiliary value from the same solved value jump and
prescribed normal jump even for legacy mode, then record:

```cpp
first_value;
second_value;
shared_auxiliary_value;
std::abs(first_value-second_value);
std::abs(first_value-shared_auxiliary_value);
std::abs(second_value-shared_auxiliary_value);
```

The mode-level discrepancy is the maximum first/second difference.

- [ ] **Step 6: Implement CLI, checkpoints, and output schemas**

Add:

```text
--neumann-edge-cauchy-study [N1 [N2 [N3]]]
```

Default levels are `32 64`.  Normalize only the prefixes `32`,
`32 64`, and `32 64 128`.  Use output directory:

```text
output/neumann_edge_cauchy_3d
```

and optional override:

```text
KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR
```

Write after every completed A/B pair:

- `summary.csv`;
- `edge_values.csv`;
- `gmres_residuals.csv`;
- `owner_diagnostics.csv`;
- `phase_profile.csv`;
- `acceptance.csv`.

`summary.csv` contains all measurement and derived fields plus setup,
edge-value, coefficient, spread, FFT, restrict, solve, and total times.
`phase_profile.csv` has a `scope` column: write one `shared_setup` block
per `(case,N)` from `shared_setup_phases`, and one `mode_runtime` block
per mode from the checked phase deltas.  Thus legacy must report zero
edge-value calls while augmented must report positive calls without
charging the common structural probe.  `edge_values.csv` contains the
literal connection/sample metadata and the six values listed in Step 5.
`acceptance.csv` keeps `extended_evidence_pass` immediately before
`overall_pass`.

If a coarse row fails structural or GMRES execution, write the
checkpoint and stop with exit 1.  Before N=128, evaluate the complete
coarse pilot; if it fails, write acceptance and stop without N=128.
An N=128 failure changes only `extended_evidence_pass`, emits a warning,
and cannot change process success.

- [ ] **Step 7: Build and run the N=32 integration smoke**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_exterior_zero_trace_3d `
           neumann_edge_cauchy_study_3d_test -- /m:1
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
$env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR = `
  'output/neumann_edge_cauchy_3d_task4_n32'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
  --neumann-edge-cauchy-study 32
Remove-Item Env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR
```

Expected:

- command exits 0 as a structurally valid prefix;
- exactly six summary rows and six owner rows;
- two modes for each of the three poses;
- every edge group has literal indices `0..sample_count-1`;
- all legacy/augmented rows share geometry and owner snapshots;
- legacy has zero `edge_auxiliary_values` calls;
- augmented has positive edge-value calls and no setup factorization
  count change during GMRES;
- owner geometry-query counts are unchanged before/after both GMRES
  solves, making this the operator-level no-SVD/no-geometry-query check;
- completeness, order, and every other gate requiring both coarse levels
  report `NotEvaluated`; same-level structure, reproduction, GMRES,
  error, edge-discrepancy, and geometry/owner gates are evaluated.

- [ ] **Step 8: Run regressions and commit**

Run:

```powershell
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
.\build\apps\Release\neumann_rigid_transform_study_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
.\build\apps\Release\kfbim_phase_profile_3d_test.exe
git diff --check
```

Expected: all eleven executables exit 0 and the diff check is clean.

Commit:

```powershell
git add apps/CMakeLists.txt `
        apps/neumann_edge_cauchy_study_3d.hpp `
        apps/neumann_edge_cauchy_study_3d.cpp `
        apps/neumann_edge_cauchy_study_3d_test.cpp `
        apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add Neumann edge-Cauchy A/B study"
```

---

### Task 5: Numerical pilot and checked-in report

**Files:**
- Create: `docs/superpowers/results/2026-07-25-3d-neumann-edge-augmented-cauchy.md`
- Generated, untracked: `output/neumann_edge_cauchy_3d/*.csv`

**Interfaces:**
- Consumes: the complete Task 4 command and CSV schemas.
- Produces: independently audited numerical evidence and a decision on
  whether edge-augmented Cauchy fitting improves the Neumann L-prism.

- [ ] **Step 1: Run a fresh full verification before the expensive pilot**

Run:

```powershell
cmake --build build --config Release -- /m:1
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
.\build\apps\Release\neumann_rigid_transform_study_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
.\build\apps\Release\kfbim_phase_profile_3d_test.exe
```

Expected: the full Release build and all eleven executables exit 0.
Stop before numerical work on any failure.

- [ ] **Step 2: Run the gated `N=32,64,128` request**

Run one command so the application itself enforces the coarse gate:

```powershell
$env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR = `
  'output/neumann_edge_cauchy_3d'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
  --neumann-edge-cauchy-study 32 64 128
$pilotExit = $LASTEXITCODE
Remove-Item Env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR
Write-Output "PILOT_EXIT=$pilotExit"
```

Interpretation:

- exit 0 with N=128 rows means the coarse pilot passed and extended
  evidence completed;
- exit 1 with 12 complete N=32/64 rows and no N=128 rows is a valid
  negative pilot conclusion;
- fewer than 12 coarse rows, a structural row failure, nonconvergence,
  non-finite output, or a missing checkpoint is an implementation/run
  failure that must be diagnosed before reporting.

- [ ] **Step 3: Independently audit every CSV**

Use `Import-Csv`; do not trust `acceptance.csv` as the source of truth.
Verify:

```text
summary:
  12 unique coarse keys, or 18 keys only when N128 ran
  exactly three case IDs and two modes per (case,N)
  all finite structural and numerical fields

edge_values:
  complete connection/sample groups for every case,N,mode
  one shared auxiliary value per physical sample
  recomputed Linf discrepancies match summary

gmres_residuals:
  contiguous iteration indices from zero to terminal iteration
  finite histories and terminal value matching summary

owner_diagnostics:
  same keys as summary
  before/after fingerprints, digests, and query counts unchanged

phase_profile:
  legacy edge_auxiliary_values calls == 0
  augmented edge_auxiliary_values calls > 0
  phase totals and wall time are internally consistent

acceptance:
  every status equals an independent recomputation
```

Recompute all four error orders, all same-level augmented/legacy ratios,
all pose spreads, the worst GMRES comparison, and every incident-edge
discrepancy comparison.

- [ ] **Step 4: Write the numerical report**

The checked-in report must include:

- exact branch, commit, Release command, exit code, and wall time;
- configuration and fixed thresholds;
- row counts and independent CSV-audit result;
- one compact table with case, N, mode, density/interior Linf, orders,
  edge discrepancy, GMRES iterations, residual, and total time;
- L2 errors and orders;
- setup and runtime phase timing, including edge-value overhead;
- edge/local condition maxima and affected/corner center counts;
- all acceptance gates with recomputed and recorded values;
- whether N=128 ran and why;
- a direct conclusion:
  - adopt only if all coarse gates pass;
  - otherwise keep the mode experimental and identify the failed
    mechanism without changing weights or thresholds.

- [ ] **Step 5: Re-run final verification and commit the report**

Run:

```powershell
cmake --build build --config Release -- /m:1
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
.\build\apps\Release\neumann_rigid_transform_study_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
.\build\apps\Release\kfbim_phase_profile_3d_test.exe
git diff --check
```

Expected: full build, eleven executables, and diff check pass.

Commit:

```powershell
git add docs/superpowers/results/2026-07-25-3d-neumann-edge-augmented-cauchy.md
git commit -m "docs: report Neumann edge-augmented Cauchy pilot"
```
