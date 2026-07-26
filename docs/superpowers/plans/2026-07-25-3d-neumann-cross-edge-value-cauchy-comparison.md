# 3D Neumann Two-Level Shared-Edge Cauchy Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-level Neumann Cauchy reconstruction that recovers globally shared value-jump samples on native non-G1 NURBS edges and inserts them into nearby surface-centered cubic harmonic fits, then compare it with G1-only and direct-cross-face controls.

**Architecture:** Extract the current application-local Cauchy fit into a reusable `HarmonicCauchyFit3D` component. Its build phase constructs native shared-edge points, first-level edge maps, and second-level surface maps for one of three routes; its apply phase performs only gathers and small matrix products. The existing Neumann pipeline continues to own KFBI correction, bulk solve, and crossing-owner restrict unchanged.

**Tech Stack:** C++17, Eigen, existing native-NURBS/KFBI3D/GMRES code, CMake/MSVC x64 Release, CSV diagnostics, PowerShell audits.

## Global Constraints

- Work directly on `main`; preserve unrelated changes and both pre-existing untracked `2026-07-23` documents.
- Use `NativeNurbsSurface3D::geometric_connections`; never define shared points from triangulator feature-edge chords.
- Keep `JointTricubicCrossingOwner`, cubic harmonic degree `3`, surface counts `48/28`, current distance weights, SVD cutoff `3e-12`, tricubic/cubic restrict, compatibility correction, bordered equation, GMRES tolerance `2e-10`, restart `80`, and cap `80`.
- Each first-level edge fit uses exactly `24/14` samples from each incident G1 sector.
- The shared-edge second level keeps G1 `48/28` and adds at most four value rows per relevant edge; it adds no edge-normal row.
- Every relevant edge contributes its nearest shared point; only its additional points must lie within Euclidean distance `2h`.
- Outside the closed `2h` band, execute the old G1 map path without an added zero product.
- No geometry query, topology search, SVD, or least-squares solve may occur during GMRES.
- Short/rank-deficient stencils fail the route with diagnostics; no silent topology expansion.
- Preserve legacy `g1_nearest`, `same_patch`, `topological_nearest`, and `balanced_patches` application modes and keep the production default unchanged.
- Generated outputs remain untracked under `output/neumann_two_level_edge_cauchy_3d`.

## File Structure

- Modify `src/geometry/nurbs_patch_triangulator_3d.hpp/.cpp`: expose the existing derivative-based edge-length rule for arbitrary native edge intervals.
- Create `apps/harmonic_cauchy_fit_3d.hpp/.cpp`: route configuration, shared-edge geometry, direct selector, first/second-level maps, diagnostics, sparse apply, counters, and fingerprints.
- Create `apps/harmonic_cauchy_fit_3d_test.cpp`: wedge, L-prism, reproduction, linearity, topology, rigid-invariance, and runtime-purity tests.
- Modify `apps/CMakeLists.txt`: compile and test the new component.
- Modify `apps/neumann_exterior_zero_trace_3d.cpp`: replace the application-local fit, add the three-route study and diagnostics.
- Modify `apps/neumann_exterior_zero_trace_3d_route_test.cpp`: add pipeline/operator splitting and writer tests.
- Create `apps/audit_neumann_two_level_edge_cauchy_3d.ps1`: executable
  completeness, convergence, order, ratio, and route-decision audit.
- Create `docs/superpowers/results/2026-07-25-3d-neumann-two-level-edge-cauchy-comparison.md`: formal evidence and decision.

---

### Task 1: Native shared-edge geometry and deterministic selectors

**Files:**
- Modify: `src/geometry/nurbs_patch_triangulator_3d.hpp`
- Modify: `src/geometry/nurbs_patch_triangulator_3d.cpp`
- Create: `apps/harmonic_cauchy_fit_3d.hpp`
- Create: `apps/harmonic_cauchy_fit_3d.cpp`
- Create: `apps/harmonic_cauchy_fit_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: native patch derivatives, geometric connections, transitive G1 components, and surface DOFs.
- Produces:

```cpp
namespace kfbim::geometry3d {
struct NurbsPatchEdgeClosestPoint3D {
    bool converged = false;
    double parameter = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    double distance = std::numeric_limits<double>::infinity();
    double distance_error_bound = std::numeric_limits<double>::infinity();
    int knot_span_count = 0;
    int refinement_level = 0;
};

[[nodiscard]] double estimate_nurbs_patch_edge_interval_length_3d(
    const NurbsSurfacePatch3D& patch,
    NurbsPatchEdge3D edge,
    double begin,
    double end,
    int parameter_sample_count = 64);

[[nodiscard]] NurbsPatchEdgeClosestPoint3D
closest_point_to_nurbs_patch_edge_interval_3d(
    const NurbsSurfacePatch3D& patch,
    NurbsPatchEdge3D edge,
    double begin,
    double end,
    const Eigen::Vector3d& query,
    double model_diameter);
}

namespace kfbim::app3d {
enum class HarmonicCauchyRoute3D {
    G1ValueG1Normal,
    DirectCrossFaceValue,
    EdgeReconstructedValue
};

struct SharedEdgePoint3D {
    int id = -1;
    int connection_id = -1;
    int cell_id = -1;
    int cell_count = 0;
    double fraction = 0.0;
    double quadrature_weight = 0.0;
    std::array<double,2> native_parameters{{0.0,0.0}};
    std::array<Eigen::Vector2d,2> native_uv;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent = Eigen::Vector3d::Zero();
    Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
    std::array<std::vector<int>,2> sector_patch_ids;
};

struct SurfaceNonG1EdgeDistance3D {
    int connection_id = -1;
    double distance = std::numeric_limits<double>::infinity();
    std::array<double,2> native_parameters{{0.0,0.0}};
    Eigen::Vector3d closest_point = Eigen::Vector3d::Zero();
    double distance_error_bound = std::numeric_limits<double>::infinity();
};

struct SurfaceNonG1EdgeNeighborhood3D {
    int center_dof = -1;
    double nearest_distance_over_h =
        std::numeric_limits<double>::infinity();
    std::vector<SurfaceNonG1EdgeDistance3D> incident_distances;
    std::vector<int> relevant_connection_ids;
};

struct SurfaceNonG1EdgeNeighborhoodSet3D {
    std::vector<SurfaceNonG1EdgeNeighborhood3D> centers;
    std::size_t geometry_query_count = 0;
    std::uint64_t fingerprint = 0;
};

struct HarmonicCauchyFailure3D {
    std::string stage;
    std::string entity_kind;
    int entity_id = -1;
    int connection_id = -1;
    std::vector<std::vector<int>> incident_sectors;
    std::vector<int> actual_value_counts;
    std::vector<int> actual_normal_counts;
    int required_value_count = 0;
    int required_normal_count = 0;
    int actual_edge_count = 0;
    double value_radius_over_h = 0.0;
    double normal_radius_over_h = 0.0;
    double edge_radius_over_h = 0.0;
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
    std::string message;
};

class HarmonicCauchyError3D : public std::runtime_error {
public:
    explicit HarmonicCauchyError3D(HarmonicCauchyFailure3D diagnostic);
    [[nodiscard]] const HarmonicCauchyFailure3D& diagnostic() const noexcept;
private:
    HarmonicCauchyFailure3D diagnostic_;
};

struct SharedEdgePointSet3D {
    std::vector<SharedEdgePoint3D> points;
    std::vector<std::vector<int>> point_ids_by_connection;
    double max_position_mismatch = 0.0;
    double min_mapped_tangent_dot = 1.0;
    std::uint64_t fingerprint = 0;
};

struct DirectCrossFaceSelection3D {
    std::vector<int> dof_ids;
    std::vector<int> relevant_connection_ids;
    std::vector<std::vector<int>> sector_patch_ids;
    std::vector<int> sector_sample_counts;
    double nearest_edge_distance_over_h =
        std::numeric_limits<double>::infinity();
};

struct SurfaceEdgePointSelection3D {
    std::vector<int> edge_point_ids;
    std::vector<int> relevant_connection_ids;
    double nearest_edge_distance_over_h =
        std::numeric_limits<double>::infinity();
};

[[nodiscard]] SharedEdgePointSet3D make_shared_edge_points_3d(
    const NativeNurbsSurface3D& surface,
    double h,
    int edge_length_parameter_samples = 64);

[[nodiscard]] SurfaceNonG1EdgeNeighborhoodSet3D
build_surface_non_g1_edge_neighborhoods_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h);

[[nodiscard]] DirectCrossFaceSelection3D
select_direct_cross_face_value_dofs_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    int center_dof,
    int count,
    double h);

[[nodiscard]] SurfaceEdgePointSelection3D
select_surface_edge_points_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SharedEdgePointSet3D& edge_points,
    const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    int center_dof,
    double h,
    int max_points_per_edge = 4);
}
```

- [ ] **Step 1: Add the target and write RED interval tests**

Add `harmonic_cauchy_fit_3d.cpp` to `kfbim_3d_app_geometry`. Add and link
`harmonic_cauchy_fit_3d_test` with C++17.

Create a reversed two-patch unit wedge: patch 0 is
`make_bilinear_plane({0,0,0},{1,0,0},{0,1,0},{1,1,0})`; patch 1 is
`make_bilinear_plane({1,0,0},{0,0,0},{1,0,-1},{0,0,-1})`. Connect their
`VMin` intervals `[0,1]` with `reversed=true`; with no smooth neighbors, each
transitive G1 sector contains exactly its own incident patch.

Test the public length helper on `[0,1]` and `[0.25,0.75]`, expecting `1` and
`0.5` within `1e-14`. With `h=0.2`, require five edge points at fractions
`0.1,0.3,0.5,0.7,0.9`, first parameters equal to the fractions, second
parameters equal to `1-fraction`, no endpoints, physical points
`(fraction,0,0)`, mismatch `<1e-12`, mapped tangent dot `>1-1e-10`, and
orthonormal right-handed frames.

Also test the closest-point query on the L-prism patch-6 `VMin` split. A query
at native parameter `.25` offset by `.1` normal distance must project to `.25`
on `[0,.5]` and to the clamped endpoint `.5` on `[.5,1]`. A query above `.5`
must report the two partial connections tied within the physical tolerance.
For a circular cylinder edge compare parameter and distance with the analytic
circle result before and after the fixed rigid transform.

- [ ] **Step 2: Build and verify RED**

```powershell
cmake --build build --config Release --target harmonic_cauchy_fit_3d_test --parallel 2
```

Expected: compile/link failure because the helper and new component are absent.

- [ ] **Step 3: Implement native interval length/distance and shared points**

Move the current anonymous edge-length implementation behind the public API.
Use a 64-point composite trapezoid including both endpoints:

```cpp
const double delta = (end-begin)/(parameter_sample_count-1);
double sum = 0.5*speed(begin) + 0.5*speed(end);
for (int q=1; q<parameter_sample_count-1; ++q)
    sum += speed(begin + q*delta);
return delta*sum;
```

For closest distance, define the edge curve from the patch itself, split
`[begin,end]` at every internal knot of the varying native basis, and include
both interval endpoints as candidates. On each nonzero knot span evaluate 17
fixed equally spaced samples of

```text
f(t) = |C(t)-query|^2
g(t) = (C(t)-query) dot C'(t)
```

Use safeguarded Brent root solves on every bracket with
`g_left<=0 && g_right>=0`, plus bounded Brent minimization around every sampled
local minimum as a tangential-root fallback. Cap each solve at 64 iterations;
stop when the parameter bracket is at most
`64*epsilon*max(1,|begin|,|end|)` or its endpoint images are within
`max(1e-12*model_diameter,1e-14)`. Failure to converge is a structured route
failure, never a chord-distance fallback. Tie connection distances within the
same physical tolerance. Snap distances within that tolerance of `h` or `2h`
to the boundary before applying the disjoint `<h`, `[h,2h]`, `>2h` rules.

For each `g1==false` connection set `m=max(1,ceil(length/h))`,
`s=(cell+0.5)/m`, `t1=a1+s*(b1-a1)`, and
`t2=a2+(reversed?1-s:s)*(b2-a2)`. Include the reversed sign in `dX2/ds`,
validate physical points with `max(1e-12*model_diameter,1e-14)`, and require
mapped unit-tangent dot `>=1-1e-10`.

Store the two transitive G1 components as distinct sectors. Construct frame
columns `[t,b.cross(t),b]`, where `b` is the projected normal bisector; fall
back to the incident normal from the sector with the lower minimum patch ID.

- [ ] **Step 4: Write RED L-prism selector tests**

At `h=3/32`, require L-prism split connections for side 6 `[0,.5]` and
`[.5,1]` to produce seven points each, disjoint IDs, no point at `.5`, and
correct decreasing second parameters on reversed connections.

On an `N=32` cloud require:

- a regular direct-cross-face center returns 48 unique IDs split `24/24`;
- a three-sector feature-vertex selection has counts differing by at most one,
  gives the remainder to the center sector first, and contains the center;
- no physically close, topology-unrelated patch is admitted;
- outside `2h`, direct IDs equal `nearest_g1_cauchy_dofs` element-for-element;
- every relevant connection contributes one nearest shared point and at most
  three more points whose point distance is `<=2h`;
- a hollow-cylinder center next to a periodic seam admits outer-wall patches
  `{0,1,2,3}` and top-annulus patches `{8,9,10,11}`, including patches across
  both periodic G1 seams, while excluding inner wall `{4,5,6,7}` and bottom
  annulus `{12,13,14,15}`;
- repeated builds/selections have identical IDs and fingerprints.

- [ ] **Step 5: Implement the two selectors**

Build one route-independent neighborhood set per `(pose,N)` and reuse it for
all three routes and diagnostics. For each center inspect only non-G1
connections incident to its transitive G1 component, query each native partial
interval independently, store the nearest distance even outside the edge
band, and retain every connection with distance `<=2h` after boundary
snapping. Require identical neighborhood fingerprints for all route rows.

For each selector consume that cached neighborhood rather than repeating
closest-point queries.
Deduplicate admitted G1 sectors. For `S` direct sectors allocate `count/S`,
give remainders to the center sector then ascending minimum patch ID, rank by
`(distance_squared,dof_id)`, and fill a short sector only from the globally
nearest remaining admitted candidate. Explicitly insert and retain the center
DOF; fail if the final 48-ID set does not contain it.

Assign shared-point IDs strictly by `(connection vector index,cell index)` and
give every point edge quadrature weight `interval_length/cell_count`. For
edge-point selection, choose the nearest point from every relevant
connection unconditionally, then up to three additional points with point
distance `<=2h`; sort by `(distance_squared,point_id)`.

- [ ] **Step 6: Verify GREEN and regressions**

```powershell
cmake --build build --config Release --target harmonic_cauchy_fit_3d_test native_nurbs_surface_3d_test --parallel 2
.\build\apps\Release\harmonic_cauchy_fit_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

- [ ] **Step 7: Commit**

```powershell
git add src/geometry/nurbs_patch_triangulator_3d.hpp src/geometry/nurbs_patch_triangulator_3d.cpp apps/harmonic_cauchy_fit_3d.hpp apps/harmonic_cauchy_fit_3d.cpp apps/harmonic_cauchy_fit_3d_test.cpp apps/CMakeLists.txt
git commit -m 'feat: add native shared-edge Cauchy geometry'
```

---

### Task 2: Precomputed first- and second-level harmonic maps

**Files:**
- Modify: `apps/harmonic_cauchy_fit_3d.hpp`
- Modify: `apps/harmonic_cauchy_fit_3d.cpp`
- Modify: `apps/harmonic_cauchy_fit_3d_test.cpp`

**Interfaces:**
- Consumes: Task 1 geometry/selectors and `HarmonicPolynomialSpace3D`.
- Produces:

```cpp
enum class LegacySurfaceCauchyPolicy3D {
    G1Nearest,
    TopologicalNearest,
    SamePatch,
    BalancedPatches
};

struct EdgeValueMap3D {
    SharedEdgePoint3D point;
    std::array<std::vector<int>,2> sector_patch_ids;
    std::array<int,2> value_sector_counts{{0,0}};
    std::array<int,2> normal_sector_counts{{0,0}};
    std::vector<int> value_ids;
    std::vector<int> normal_ids;
    Eigen::RowVectorXd E_value;
    Eigen::RowVectorXd E_normal;
    double value_radius_over_h = 0.0;
    double normal_radius_over_h = 0.0;
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
};

struct SurfaceCauchyMap3D {
    std::vector<int> value_ids;
    std::vector<int> normal_ids;
    std::vector<int> edge_point_ids;
    std::vector<std::vector<int>> value_sector_patch_ids;
    std::vector<int> value_sector_counts;
    std::vector<std::vector<int>> normal_sector_patch_ids;
    std::vector<int> normal_sector_counts;
    std::vector<int> relevant_connection_ids;
    int incident_patch_count = 0;
    int value_patch_imbalance = 0;
    int normal_patch_imbalance = 0;
    Eigen::MatrixXd M_value;
    Eigen::MatrixXd M_normal;
    Eigen::MatrixXd M_edge;
    double value_radius_over_h = 0.0;
    double normal_radius_over_h = 0.0;
    double edge_radius_over_h = 0.0;
    double nearest_edge_distance_over_h =
        std::numeric_limits<double>::infinity();
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
};

struct HarmonicCauchyApplyResult3D {
    Eigen::VectorXd edge_values;
    Eigen::MatrixXd coefficients;
    std::size_t runtime_geometry_query_count = 0;
    std::size_t runtime_svd_factorization_count = 0;
    std::uint64_t fingerprint_before = 0;
    std::uint64_t fingerprint_after = 0;
};

struct HarmonicCauchyPreprocessAudit3D {
    std::size_t geometry_query_count = 0;
    std::size_t svd_factorization_count = 0;
    std::uint64_t fingerprint = 0;
};

struct LegacyCauchySummary3D {
    LegacySurfaceCauchyPolicy3D policy =
        LegacySurfaceCauchyPolicy3D::G1Nearest;
    int degree = 0;
    int value_count = 0;
    int normal_count = 0;
    int value_count_min = 0;
    int value_count_max = 0;
    int normal_count_min = 0;
    int normal_count_max = 0;
    double radius_max_over_h = 0.0;
    double radius_mean_over_h = 0.0;
    int incident_patch_count_min = 0;
    int incident_patch_count_max = 0;
    int value_patch_imbalance_max = 0;
    int normal_patch_imbalance_max = 0;
};

class HarmonicCauchyFit3D {
public:
    static HarmonicCauchyFit3D build(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        const SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
        double h,
        HarmonicCauchyRoute3D route,
        int degree = 3,
        int value_count = 48,
        int normal_count = 28,
        double relative_svd_cutoff = 3e-12);

    static HarmonicCauchyFit3D build_legacy(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        double h,
        LegacySurfaceCauchyPolicy3D policy,
        int degree,
        int value_count,
        int normal_count,
        double relative_svd_cutoff = 3e-12);

    [[nodiscard]] HarmonicCauchyApplyResult3D apply(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const;
    [[nodiscard]] const HarmonicPolynomialSpace3D& space() const noexcept;
    [[nodiscard]] int degree() const noexcept;
    [[nodiscard]] int value_count() const noexcept;
    [[nodiscard]] int normal_count() const noexcept;
    [[nodiscard]] std::optional<LegacySurfaceCauchyPolicy3D>
    legacy_policy() const noexcept;
    [[nodiscard]] std::optional<LegacyCauchySummary3D>
    legacy_summary() const;
    [[nodiscard]] std::vector<double> condition_values() const;
    [[nodiscard]] const std::vector<EdgeValueMap3D>& edge_maps() const noexcept;
    [[nodiscard]] const std::vector<SurfaceCauchyMap3D>& surface_maps() const noexcept;
    [[nodiscard]] const HarmonicCauchyPreprocessAudit3D& audit() const noexcept;
};
```

The fit object owns every ID and matrix and retains no `NativeNurbsSurface3D`
or temporary-stencil reference. `apply` therefore cannot query geometry.

- [ ] **Step 1: Write RED cubic-reproduction and map-linearity tests**

Use a two-plane wedge with shared edge `(0,y,0)`, `h=1/8`, and

```cpp
p(x,y,z) = 1 + y - 0.5*z + 2*x*z + x*x*x - 3*x*y*y;
grad_p = {2*z + 3*x*x - 3*y*y, 1 - 6*x*y, -0.5 + 2*x};
```

Set `mu[q]=p(dof.point)` and
`eta[q]=grad_p(dof.point).dot(dof.normal)`. Build
`EdgeReconstructedValue`, require eight edge maps, exact `24+24` value and
`14+14` normal IDs per map, and reconstructed values within `2e-11` of
`p(edge_point.point)`.

For deterministic vectors and scalars `a=.37,b=-1.2`, require both returned
edge values and coefficients to satisfy
`F(a*x+b*y)=a*F(x)+b*F(y)` within relative `2e-13`.

Choose one surface center on each incident side of the same wedge edge.
Require their second-level maps to reference the same nearest global edge
point ID, and instrument `apply` to prove that point's scalar value is formed
once in the global edge vector and gathered by both surface maps.

- [ ] **Step 2: Build and verify RED**

```powershell
cmake --build build --config Release --target harmonic_cauchy_fit_3d_test --parallel 2
```

Expected: failure because the maps and fit class are not implemented.

- [ ] **Step 3: Build the first-level edge maps**

For each edge point, rank candidates separately inside its two stored sectors
and take exactly 24 value and 14 normal DOFs from each. Never borrow across a
short sector. Use local coordinate
`xi=point.frame.transpose()*(sample.point-point.point)/h` and the existing
weights:

```cpp
value_sqrt_weight = 1.0/(0.35 + xi.norm());
normal_sqrt_weight = std::sqrt(0.85)/(0.35 + xi.norm());
```

Form the `76 x 16` weighted design. Normal rows use each sample normal in the
edge frame. Require `sigma_min > 3e-12*sigma_max`. With
`origin=space.basis(0,0,0)`, store

```cpp
E_value[k] = origin.dot(pinv.col(k))*sqrt_weights[k];
E_normal[k] = origin.dot(pinv.col(48+k))*sqrt_weights[48+k]*h;
```

Do not assume coefficient zero equals the origin value because the harmonic
basis is an SVD-generated rotation.

- [ ] **Step 4: Build the three second-level route maps**

For every surface center:

- `G1ValueG1Normal`: current G1 value and normal IDs, no edge IDs;
- `DirectCrossFaceValue`: Task 1 direct value IDs, current G1 normal IDs, no
  edge IDs;
- `EdgeReconstructedValue`: current G1 value/normal IDs plus Task 1 edge IDs.

Assemble the weighted surface design in row order ordinary value, ordinary
normal, shared-edge value. Store `M_value`, `M_normal` multiplied by `h` only
for normal columns, and `M_edge`. Every shared-edge value row must use exactly
the ordinary-value formula `1/(0.35+xi.norm())`, with no new parameter.
Preserve singular extrema, all radii, per-sector patch IDs/counts, and relevant
connection IDs.

All three routes receive the same route-independent neighborhood set. Copy
its nearest native-edge distance into every surface map, including G1 maps
with no edge rows, so bin membership and its fingerprint are identical across
routes.
For zero edge rows, execute the original two-term coefficient expression so
the G1 route does not add a third zero product.

`build_legacy` must reproduce the four current environment policies and use no
edge maps. Delete the application-local duplicate fit types only in Task 3,
after this class is tested.

- [ ] **Step 5: Implement sparse apply and immutable audit**

At the beginning of one `apply`, snapshot the immutable map fingerprint and
zero local runtime query/SVD counters, then compute the global edge vector
once:

```cpp
mu_E[e] = E_value.dot(gather(value_jump,value_ids))
        + E_normal.dot(gather(normal_jump,normal_ids));
```

Then gather `mu_E` in each surface map and evaluate its coefficient row. Hash
point geometry, all IDs, matrices, singular diagnostics, and route into
`audit.fingerprint`. Count every build-time NURBS/curve query and SVD. Return
the local runtime counters and before/after fingerprints in the apply result;
the gather/matrix-only path must leave both counters zero and the fingerprints
equal. Never report preprocessing counts as runtime counts.

- [ ] **Step 6: Add rigid-invariance and failure tests**

Rotate the wedge by 17 degrees about `(1,2,3)` and translate it. Require
unchanged IDs, fractions, and counts; transformed points/frames within
`2e-12`; singular values and conditions within relative `5e-12`; reconstructed
manufactured edge values within `2e-11`.

Use an undersampled sector to require an exception containing point ID,
sector patch IDs, required/actual counts, radii, and failure stage. Declare a
non-G1 connection whose sides belong to one G1 component and require an
explicit distinct-sector failure.

Add explicit fixtures for a zero/non-finite edge tangent, degenerate normal
bisector plus degenerate fallback, first-level rank failure, and second-level
rank failure. Every failure must carry `HarmonicCauchyFailure3D`, including
entity/connection ID, incident sectors, actual/required counts, radii,
singular extrema, condition number when available, and the exact failure
stage. Add one test in which an unrelated patch would be closer and require a
structured topology-selection failure if it is ever admitted.

Snapshot the audit, call `apply` twice, and require identical query count, SVD
count, and fingerprint. Require each apply result's runtime geometry/SVD
counters to be exactly zero and its before/after fingerprints to match the
snapshot.

- [ ] **Step 7: Verify GREEN and all route maps**

```powershell
cmake --build build --config Release --target harmonic_cauchy_fit_3d_test --parallel 2
.\build\apps\Release\harmonic_cauchy_fit_3d_test.exe
```

- [ ] **Step 8: Commit**

```powershell
git add apps/harmonic_cauchy_fit_3d.hpp apps/harmonic_cauchy_fit_3d.cpp apps/harmonic_cauchy_fit_3d_test.cpp
git commit -m 'feat: add two-level harmonic Cauchy maps'
```

---

### Task 3: Integrate the two-level maps into the KFBI pipeline

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d_route_test.cpp`
- Create: `apps/audit_neumann_two_level_edge_cauchy_3d.ps1`

**Interfaces:**
- Consumes: `HarmonicCauchyFit3D` from Task 2.
- Produces:

```cpp
const char* harmonic_cauchy_route_name_3d(
    app3d::HarmonicCauchyRoute3D route);

class PanelCenterHarmonicJetKFBI3D {
public:
    PanelCenterHarmonicJetKFBI3D(
        const CartesianGrid3D& grid,
        const GridPair3D& grid_pair,
        const NativeNurbsSurface3D& native_surface,
        const std::vector<geometry3d::NurbsParamTriangle3D>& correction,
        const std::vector<geometry3d::NurbsParamTriangle3D>& geometry,
        const app3d::SurfaceDofCloud3D& cloud,
        app3d::HarmonicCauchyFit3D fit,
        bool build_exterior_only,
        bool build_crossing_owner);

    [[nodiscard]] const app3d::HarmonicCauchyFit3D& cauchy_fit() const noexcept;
};
```

The route enum controls only Cauchy reconstruction. It must remain independent
of `ExteriorValueRestrictMode3D`, which controls crossing ownership.

- [ ] **Step 1: Write RED route and second-level structure tests**

In the route test, require exact names:

```text
g1_value_g1_normal
direct_cross_face_value
edge_reconstructed_value
```

Build the three L-prism `N=32` fits and require:

- G1 has no edge maps and no surface edge IDs;
- direct has balanced cross-face value IDs, G1 normal IDs, and no edge IDs;
- shared-edge has G1 value/normal IDs, nonempty global edge maps, and at least
  one surface center reusing the same edge-point ID from each incident face;
- a shared-edge center outside `2h` has the exact G1 IDs, matrices, and
  two-term apply path.

- [ ] **Step 2: Build and verify RED**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test --parallel 2
```

Expected: compile failure because the application still owns the old fit.

- [ ] **Step 3: Replace the application-local fit without duplicating types**

Remove `CauchyStencilPolicy3D`, `CauchyStencilSet`, `CauchyFitMap3D`,
`PanelCenterCauchyFit3D`, `select_cauchy_dofs`, and `build_cauchy_stencils`
from the application after all call sites use the public component. Replace
every signature and stored field, including study-only paths around the old
direct `CauchyStencilSet` constructions. Map the four existing environment
strings to `app3d::LegacySurfaceCauchyPolicy3D` and `build_legacy`; preserve
requested nondefault counts, policy names, and output paths. Build one owning
fit before each pipeline and move it into the pipeline.

Change both `evaluate` and `field_from_grid_and_jumps` to call
`fit_.apply(value_jump,normal_jump)` and use its coefficient matrix. Do not
leave any probe on the deleted legacy coefficient path.

Expose degree/value/normal counts, legacy policy, per-center selected IDs,
radii, singular extrema, and condition values through explicit const
accessors on `HarmonicCauchyFit3D`; forward all existing CSV diagnostics from
those accessors. Add a route test comparing the old captured G1 fixture IDs
and aggregate diagnostics with `build_legacy(G1Nearest)` before deleting the
old types.

- [ ] **Step 4: Write the RED end-to-end splitting test**

Build an owner-enabled L-prism `N=32` shared-edge pipeline. For deterministic
`mu,eta`, define `T` as pipeline evaluate followed by
`JointTricubicCrossingOwner` exterior trace. Require

```cpp
linf(T(mu,eta) - T(mu,zero) - T(zero,eta)) < 5e-11;
```

Then construct the bordered operator with unknown `[mu;lambda]`. Require
`op.apply(x)-op.right_hand_side(eta)` to equal
`[T(mu,eta)+lambda; weighted_mean(mu)]` within `5e-11`.

Snapshot Cauchy query/SVD/fingerprint and owner query/fingerprint before the
two calls and require every snapshot unchanged afterward.

- [ ] **Step 5: Implement one global edge evaluation per fit apply**

Ensure the shared edge vector is computed once at the start of
`HarmonicCauchyFit3D::apply` and reused by every surface center. The pipeline
must not independently recompute edge values for exterior trace or owner
continuation; those consume the already computed polynomial coefficients.

- [ ] **Step 6: Verify GREEN and legacy compatibility**

```powershell
cmake --build build --config Release --target harmonic_cauchy_fit_3d_test neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\harmonic_cauchy_fit_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

Run the normal application with each existing policy string and require zero
exit plus the expected policy text/output subdirectory:

```powershell
$base = Join-Path (Get-Location) 'output\neumann_exterior_zero_trace_3d'
$cases = @(
    @{ policy='g1_nearest'; path=(Join-Path $base 'g1_nearest') },
    @{ policy='same_patch'; path=(Join-Path $base 'same_patch') },
    @{ policy='topological_nearest'; path=$base },
    @{ policy='balanced_patches'; path=(Join-Path $base 'balanced_patches') }
)
foreach ($case in $cases) {
    $env:KFBIM_3D_CAUCHY_POLICY = $case.policy
    $text = (.\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw ('legacy policy failed: '+$case.policy) }
    if (-not $text.Contains('cauchy_policy='+$case.policy)) { throw ('policy text missing: '+$case.policy) }
    if (-not $text.Contains($case.path)) { throw ('output path mismatch: '+$case.policy) }
}
Remove-Item Env:KFBIM_3D_CAUCHY_POLICY
$defaultText = (.\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or -not $defaultText.Contains('cauchy_policy=g1_nearest')) {
    throw 'default Cauchy policy changed'
}
```

The final run without the environment variable proves the production default
remains `g1_nearest`.

- [ ] **Step 7: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/neumann_exterior_zero_trace_3d_route_test.cpp
git commit -m 'refactor: integrate two-level Cauchy maps'
```

---

### Task 4: Reproducible physical/common probes and edge diagnostics

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d_route_test.cpp`

**Interfaces:**
- Consumes: the integrated pipeline and bordered operator.
- Produces:

```cpp
struct NeumannManufacturedData3D {
    Eigen::VectorXd prescribed_normal_jump;
    Eigen::VectorXd exact_density;
    double density_mean_shift = 0.0;
};

struct EdgeBinMetrics3D {
    std::string bin;
    int count = 0;
    double weight_sum = 0.0;
    double density_linf = 0.0;
    double density_weighted_rms = 0.0;
    double defect_linf = 0.0;
    double defect_weighted_rms = 0.0;
};

struct NeumannRouteProbe3D {
    SolveMetrics3D physical;
    std::vector<double> physical_residuals;
    CommonRhsGmresProbe3D common;
    Eigen::VectorXd density_error;
    Eigen::VectorXd exact_equation_defect;
    Eigen::VectorXd exact_input_edge_values;
    Eigen::VectorXd exact_edge_value_error;
    Eigen::VectorXd exact_edge_quadrature_weights;
    std::optional<double> edge_value_linf;
    std::optional<double> edge_value_weighted_rms;
    double exact_mean_row_defect = 0.0;
    std::array<EdgeBinMetrics3D,3> bins;
};
```

- [ ] **Step 1: Write RED common-RHS and exact-defect tests**

For each DOF normalize native parameters to `uhat,vhat` in `[0,1]`, set

```cpp
r[q] = sin(2*pi*uhat + 0.37*(patch_id+1))
     + 0.5*cos(2*pi*vhat - 0.23*(patch_id+1))
     + 0.25*sin(2*pi*(uhat+vhat));
```

Test that weighted demeaning and weighted-RMS normalization produce mean
`<=5e-13`, RMS within `5e-13` of one, augmented tail exactly zero, and an
identical vector for all three routes at fixed `(pose,N)`.

On a detailed crossing-owner solve require physical/common histories of size
`iterations+1`, density and defect vectors of surface size, finite mean-row
defect, and an exact defect equal to the literal operator application below.

- [ ] **Step 2: Build and verify RED**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test --parallel 2
```

- [ ] **Step 3: Extract manufactured data and preserve old values**

Move current value/normal loops into a helper. Apply the current discrete
compatibility correction to normal data and subtract the surface-weighted
mean from exact density. Keep `run_neumann_case` as a wrapper so old studies
remain source-compatible.

- [ ] **Step 4: Implement the common solve with fixed GMRES settings**

Use zero initial guess, tolerance `2e-10`, restart `80`, cap `80`, and the same
augmented operator. Store every relative residual and audit mean/RMS before
calling GMRES.

- [ ] **Step 5: Compute literal exact-density defect and edge error**

```cpp
Eigen::VectorXd exact_augmented = Eigen::VectorXd::Zero(op.problem_size());
exact_augmented.head(size) = data.exact_density;
Eigen::VectorXd applied;
op.apply(exact_augmented,applied);
const Eigen::VectorXd residual =
    applied - op.right_hand_side(data.prescribed_normal_jump);
probe.exact_equation_defect = residual.head(size);
probe.exact_mean_row_defect = residual[size];
```

For the shared-edge route evaluate the Cauchy fit on the same exact density
and compatibility-corrected normal data. Compare its edge vector with
`manufactured_value(edge_point)-density_mean_shift`. Controls store empty edge
vectors and explicit N/A optionals rather than fabricated zeros. Compute edge
`Linf` normally and edge weighted RMS with each point's native
`interval_length/cell_count` quadrature weight.

- [ ] **Step 6: Implement exact disjoint bins**

Use the one shared route-independent neighborhood distance for every route;
assert its fingerprint agrees with the copy stored by each surface map:

- `lt_h`: `d/h<1`;
- `h_to_2h`: `1<=d/h<=2`;
- `gt_2h`: `d/h>2`.

For density error and equation defect record count, weight sum, Linf, and
`sqrt(sum(weight*error^2)/sum(weight))`. Empty bins store zero metrics and an
explicit empty flag in CSV output.

- [ ] **Step 7: Verify GREEN and commit**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test --parallel 2
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
git add apps/neumann_exterior_zero_trace_3d.cpp apps/neumann_exterior_zero_trace_3d_route_test.cpp
git commit -m 'test: add two-level Neumann probes'
```

---

### Task 5: Three-route study driver and incremental output

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d_route_test.cpp`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces command:

```text
neumann_exterior_zero_trace_3d.exe --neumann-edge-cauchy-study [N ...]
```

- Produces `output/neumann_two_level_edge_cauchy_3d/`:

```text
summary.csv
gmres_residuals.csv
edge_point_diagnostics.csv
edge_fit_diagnostics.csv
surface_fit_diagnostics.csv
dof_diagnostics.csv
edge_distance_bins.csv
owner_diagnostics.csv
decision.json
```

- [ ] **Step 1: Write RED route/writer tests**

Require exact route strings and write one successful plus one failed synthetic
row to a scoped output directory. Read all files back and require:

- CSV quoting preserves commas and quotes in failure text;
- every row begins with `case_id,N,route`;
- failed summary/owner/bin rows preserve `status`, `failure_stage`, and
  `failure_message` even when later diagnostics are unavailable;
- every structured failure column exists and a synthetic first-level,
  second-level, frame, rank, and unrelated-selection failure round-trips its
  entity/connection ID, sectors, actual/required counts, radii, singular
  extrema, and condition when available;
- residual rows distinguish `physical` and `common`;
- controls produce no invented edge-point/edge-fit rows;
- a successful shared-edge row contains all point and map diagnostics.

Make the audit script accept `-OutputDirectory`, `-ExpectedLevels`,
`-AllowNumericalFailure`, `-SchemaOnly`, and `-DecisionJson`. Invoke
`-SchemaOnly` on the synthetic directory and require it to reject a duplicate
key, a missing failure field, a missing bin, and a mismatched residual length.

- [ ] **Step 2: Verify CLI RED**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-edge-cauchy-study 32
```

Expected: nonzero exit because the command is absent.

- [ ] **Step 3: Implement the fixed study matrix**

Default levels are `32,64,128`; explicit levels must be powers of two at least
16. Select `baseline`, `rot_axis123_17deg`, and
`rot_axis123_17deg_t_xyz_1` from the existing rigid-case factory.

For each `(pose,N)` build grid, native geometry, domain, labels, surface DOFs,
and the route-independent native-edge neighborhood set once. Then build one
fit and owner-enabled pipeline for each route:

```text
g1_value_g1_normal
direct_cross_face_value
edge_reconstructed_value
```

Always use `JointTricubicCrossingOwner`. Snapshot Cauchy audit and owner audit
before both GMRES solves and after them. Also snapshot label counts/fingerprint
and the common neighborhood fingerprint. Owner query/fingerprint and label
fingerprint must match the G1 reference across routes; each route's own Cauchy
fingerprint must remain unchanged; all neighborhood fingerprints must match.
Aggregate the apply-result runtime geometry/SVD counters and require zero.
Hash and compare the common augmented RHS across routes.

Catch setup, stencil, fit, and solve exceptions per route, append a failed row,
and continue every remaining route and finer level. Attempt all requested
configurations regardless of an earlier numerical failure.

- [ ] **Step 4: Implement append-and-flush writers**

Truncate files and write headers once at command start. After each completed
or failed route, append every available route block and flush all streams.
Always emit one summary row, three bin rows, and one owner row.

`summary.csv` includes route/status, geometry counts, physical/common
convergence/iterations/final residual/contractions, common RHS mean/RMS/hash,
density/interior errors and the independently checked `32->64` and `64->128`
adjacent orders, global defect norms, virtual-edge error
norms, setup/fit/pipeline/solve times, map counts/radii/condition statistics,
preprocess and runtime query/SVD counters, before/after fingerprints, label and
neighborhood invariants, and all structured failure fields. Controls retain
the virtual-edge columns with explicit `NA` values; they never write fake zero
errors.

`gmres_residuals.csv` schema is
`case_id,N,route,rhs_kind,iteration,relative_residual`.

`edge_point_diagnostics.csv` stores connection/cell IDs, fraction, both native
parameters/UVs, point, tangent, sectors, exact/reconstructed value, and error.
`edge_fit_diagnostics.csv` stores the per-sector `24/14` counts, radii,
singular extrema, and condition. `surface_fit_diagnostics.csv` stores
ordinary/edge counts, per-sector patch IDs/counts, radii, edge distance,
singular extrema, and condition per center.
`dof_diagnostics.csv` stores point/patch/weight, edge distance, density error,
and equation defect. `edge_distance_bins.csv` stores the three Task 4 bins.
`owner_diagnostics.csv` stores owner aggregates, before/after Cauchy and owner
audits, reference equality, and common-RHS hash equality.

The PowerShell audit must recompute keys, history lengths, adjacent orders,
combined near-edge weighted norms, `W`/`S`, `N=128` ratios, edge-error trends,
and the direct-vs-shared tie-break from raw CSV rows. It must not trust a
precomputed pass flag from the executable. Write every computed scalar and
failed predicate to `decision.json` in deterministic key order.

- [ ] **Step 5: Verify writer tests GREEN**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

- [ ] **Step 6: Run and audit the N=32 smoke**

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-edge-cauchy-study 32
```

Exit code `2` is an admissible recorded numerical failure, but missing rows or
an unexplained failure are not. Run the executable audit rather than a second
hand-written gate:

```powershell
$out = 'output/neumann_two_level_edge_cauchy_3d'
& .\apps\audit_neumann_two_level_edge_cauchy_3d.ps1 `
    -OutputDirectory $out -ExpectedLevels @(32) `
    -AllowNumericalFailure -DecisionJson (Join-Path $out 'decision.json')
```

- [ ] **Step 7: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/neumann_exterior_zero_trace_3d_route_test.cpp apps/audit_neumann_two_level_edge_cauchy_3d.ps1
git commit -m 'feat: add two-level Neumann Cauchy study'
```

---

### Task 6: Formal 32/64/128 comparison and decision

**Files:**
- Create:
  `docs/superpowers/results/2026-07-25-3d-neumann-two-level-edge-cauchy-comparison.md`
- Generate but do not commit:
  `output/neumann_two_level_edge_cauchy_3d/*`

**Interfaces:**
- Consumes: the formal study command and CSV schema from Task 5.
- Produces: the complete-pair numerical decision for all three routes.

- [ ] **Step 1: Build and execute the formal matrix**

```powershell
cmake --build build --config Release --target harmonic_cauchy_fit_3d_test native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\harmonic_cauchy_fit_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-edge-cauchy-study 32 64 128
```

The formal command must execute exactly three fixed L-prism poses, three
routes, and the three resolutions: 27 configurations. Every configuration
must run both the physical manufactured RHS and the common RHS.

- [ ] **Step 2: Enforce structural and solver gates**

Run the committed audit, which must exit nonzero on any failed gate and write
all intermediate metrics and the final route decision to JSON:

```powershell
$out = 'output/neumann_two_level_edge_cauchy_3d'
& .\apps\audit_neumann_two_level_edge_cauchy_3d.ps1 `
    -OutputDirectory $out -ExpectedLevels @(32,64,128) `
    -DecisionJson (Join-Path $out 'decision.json')
```

The script must fail unless:

- `summary.csv` has 27 unique `(case_id,N,route)` rows;
- `edge_distance_bins.csv` has 81 unique
  `(case_id,N,route,distance_bin)` rows;
- `owner_diagnostics.csv` has 27 unique `(case_id,N,route)` rows;
- every formal row has `status=ok`;
- physical and common GMRES both converge within 80 iterations;
- every final relative residual is strictly `<2e-10`;
- each residual history contains exactly `iterations+1` entries;
- runtime query and SVD counters are zero while preprocessing counters retain
  their actual nonzero work;
- preprocessing and post-apply fingerprints match;
- label counts/fingerprints and neighborhood fingerprints agree across routes;
- each shared point satisfies its mismatch/tangent/frame checks;
- all first- and second-level maps have the prescribed row counts and full
  cubic-harmonic rank.

Any missing/failed configuration invalidates the whole decision; do not
compare a surviving subset.

- [ ] **Step 3: Compute the iteration decision**

For route `r`, RHS kind `k`, pose `p`, and grid `N`, compute:

```text
I(r,k,p,N) = the recorded GMRES iteration count
W(r,k)     = max over p and N I(r,k,p,N)
S(r,k,N)   = max over p I(r,k,p,N) - min over p I(r,k,p,N)
```

The primary `edge_reconstructed_value` route passes the iteration comparison
only if, for both RHS kinds, its `W` is no worse than both controls and, for
both RHS kinds and every `N`, its `S` is no worse than both controls. It must
also be strictly better than a control in at least one complete worst-count
comparison and strictly better in at least one complete pose-spread
comparison.

- [ ] **Step 4: Compute local-defect, accuracy, and edge-recovery gates**

At `N=128`, combine the `<h` and `[h,2h]` rows. For every pose, require the
primary route's equation-defect `Linf` and RMS to be strictly below both
controls.

For each pose and each of density `Linf`, density `L2`, interior `Linf`, and
interior `L2`:

```text
order = log2(error_N64/error_N128)
ratio = primary_error_N128/control_error_N128
```

Require the primary route to have order `>=0` for all four errors. Require
every primary/control `N=128` ratio to be `<=1.10`. For every pose require the
shared-edge reconstruction `Linf` and edge-quadrature-weighted RMS errors to decrease
from `N=64` to `N=128`, with RMS weighted by the native edge-cell quadrature
weights.

If the direct route passes the corresponding completeness, iteration,
defect, order, and error-ratio gates relative to G1, keep that simpler route
unless the shared-edge route strictly improves both an iteration metric and
an edge-defect metric relative to the direct route. If neither candidate
passes, report the specified next design: sector-wise polynomials with
explicit shared-edge value/tangential constraints.

- [ ] **Step 5: Write the evidence report**

The result document must include:

- exact commit and build configuration;
- the 27-row convergence/error/iteration table;
- physical and common `W`/`S` tables;
- per-pose `N=32 -> 64` and `N=64 -> 128` adjacent convergence orders;
- per-pose `N=128` primary/control error ratios;
- combined near-edge equation-defect table;
- shared-edge value recovery and conditioning table;
- preprocessing time, solve time, runtime counters, and owner diagnostics;
- every failed gate, with no selective omission;
- one explicit conclusion: adopt as an experimental route, reject, or rerun
  after a named blocker.

- [ ] **Step 6: Commit the report only**

```powershell
git add docs/superpowers/results/2026-07-25-3d-neumann-two-level-edge-cauchy-comparison.md
git commit -m 'docs: report two-level Neumann edge reconstruction'
```

Do not add generated CSV or diagnostic directories.

---

### Task 7: Final verification, review, and clean handoff

**Files:**
- Verify all files modified by Tasks 1-6.
- Modify only files required by accepted review fixes.

**Interfaces:**
- Consumes: the preserved full formal output from Task 6.
- Produces: a reviewed implementation with no production-default change.

- [ ] **Step 1: Run the complete build and direct-test gate**

```powershell
cmake --build build --config Release --target harmonic_cauchy_fit_3d_test native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\harmonic_cauchy_fit_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

Require zero exit from every command. Do not replace the Task 6 formal
directory by running an `N=32`-only study.

- [ ] **Step 2: Re-audit the preserved formal output**

Run the same executable audit against the existing `32/64/128` files:

```powershell
$out = 'output/neumann_two_level_edge_cauchy_3d'
& .\apps\audit_neumann_two_level_edge_cauchy_3d.ps1 `
    -OutputDirectory $out -ExpectedLevels @(32,64,128) `
    -DecisionJson (Join-Path $out 'decision.json')
```

Require 27 complete summary rows, 81 distance-bin rows, and 27 owner rows.
Require both residual-history lengths to equal the corresponding iteration
count plus one.

- [ ] **Step 3: Request a code review**

Use `superpowers:requesting-code-review` on the full implementation diff.
The review must explicitly check:

- interval orientation and partial native NURBS connections;
- absence of cross-non-G1 DOFs in the direct selector outside its route;
- first-level `24/14` per-sector counts and actual incident normals;
- second-level nearest-point inclusion and the `2h` rule for extras;
- `basis(0,0,0)` evaluation instead of assuming the constant coefficient;
- exactly one `h` scaling of normal data;
- precomputation/runtime separation and fingerprint stability;
- affine operator splitting and existing Neumann sign conventions;
- failed-route diagnostics and strict complete-pair comparisons;
- preservation of legacy routes and the production default.

- [ ] **Step 4: Resolve review findings and reverify**

Fix every Critical or Important finding with a failing test first. Rerun the
full Task 6 `32/64/128` study whenever a fix changes geometry, selection,
linear maps, operator algebra, solver configuration, error probes, or CSV
decision logic. Documentation-only and diagnostic-label fixes may reuse the
preserved formal run after the direct tests pass.

- [ ] **Step 5: Check the final tree**

```powershell
git diff --check
git status --short
```

The only unrelated untracked files permitted are:

```text
docs/superpowers/plans/2026-07-23-3d-same-patch-cauchy-default.md
docs/superpowers/specs/2026-07-23-3d-same-patch-cauchy-default-design.md
```

Do not commit generated files below
`output/neumann_two_level_edge_cauchy_3d`. Do not change the production route
default based on this experiment.
