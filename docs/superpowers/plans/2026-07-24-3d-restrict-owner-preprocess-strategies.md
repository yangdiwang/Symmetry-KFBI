# 3D Restrict-Owner Preprocessing Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two selectable, fail-closed 3D KFBI restrict-owner preprocessing strategies, prove their owner decisions against the current full NURBS intersection oracle, and measure preprocessing speed, memory, and end-to-end Dirichlet accuracy on the agreed geometry/pose matrix through `N=128`.

**Architecture:** Core geometry gains an opaque rational-Bezier candidate facade, a typed `CertifiedMiss / CertifiedUniqueTransverseRoot / Unresolved` certificate, and a filtered two-independent-crossing query that reuse the current tolerances, root canonicalization, topology, and closest-point assistance. A focused app-layer preprocessor implements full-reference, optimized-intersection, and region/closest hybrid policies over one complete 64-node tricubic sample; the existing KFBI pipeline supplies the exact sample and preserves its original `q=0..63` floating-point accumulation order. The study driver records the full oracle once, replays the identical compact workload for candidate timing and accuracy, and runs GMRES only after zero owner mismatch is established.

**Tech Stack:** C++17, Eigen, existing rational NURBS/Bezier geometry and CGAL-enabled `kfbim_core`, Visual Studio x64 Release, CMake, PowerShell, zFFT, and the existing 3D harmonic-jet KFBI/GMRES application.

## Global Constraints

- Modify only crossing-owner selection in `PanelCenterHarmonicJetKFBI3D`; do not replace `NurbsCartesianDomain3D`, Cartesian-edge barriers, grid labels, component parity, Cauchy fitting, spread, FFT, GMRES, or restrict formulas.
- Keep `FullIntersectionReference` as the production default and as the final fail-closed fallback until both candidate policies pass every gate.
- Define the owner-compatible set as the target patch plus exactly `cloud.patches[target_patch].smooth_patch_ids`; never use a transitive G1 closure.
- Reuse the existing geometry tolerance, local subdivision depth `4`, terminal certificate depth `6`, producer transversality threshold, seam canonicalization, feature-contact rules, and existing 2-by-2 parameter-space DOF association.
- Process at most `2` independent canonical crossings in the filtered query; two raw roots that may collapse at a seam do not satisfy this cap.
- Accept a closest-point root only when the segment, Bezier element, and NURBS patch parameters are all strictly interior and the existing single-root and reliable-transversality certificates pass.
- Treat overlap, endpoints, near tangency, parameter boundaries, G1/periodic seams, non-G1 edges/vertices, unrelated coincidence, multiple roots, and unresolved subdivision as target-owned fail-closed outcomes or as reasons to invoke the next conservative fallback.
- Normal production mode records aggregate counters without per-query clocks; detailed stage clocks are enabled only in a separate benchmark instrumentation replay and are excluded from median/p95 speed measurements.
- Preserve the current correction accumulation order exactly: support slots are still applied in `q=0..63` order and each owner polynomial receives terms in that order.
- Run the validation oracle once per case; no separate oracle comparison pass may run inside a candidate timing interval. Any `FullIntersectionFallback` selected by the candidate policy is part of that policy and its cost remains inside the timer.
- Do not start a case's GMRES solves until both candidates have `owner_mismatch == 0` on its complete wrong-side workload.
- Do not run `N=128` unless all `N=16`, `N=32`, and `N=64` preprocessing accuracy gates pass.
- Generated CSV files, mismatch files, and run logs stay uncommitted.

---

## File Map

### Core geometry

- Modify `src/geometry/nurbs_bezier_intersection_3d.hpp`: declare the typed element/segment certificate.
- Modify `src/geometry/nurbs_bezier_intersection_3d.cpp`: extract the existing control-hull, closest-point, terminal separation, unique-root, and transversality logic into the typed certificate implementation.
- Modify `src/geometry/nurbs_surface_intersector_3d.hpp`: declare the opaque candidate handle, candidate certificate, and filtered query result.
- Modify `src/geometry/nurbs_surface_intersector_3d.cpp`: expose conservative BVH candidate collection without exposing BVH nodes or mutable Bezier elements; refactor the existing candidate loop for filtered queries while preserving full-query behavior.
- Modify `apps/native_nurbs_surface_3d_test.cpp`: test the typed certificate, candidate ownership validation, deterministic metadata, root cap, seam deduplication, and full-query regression.

### Application policy

- Create `apps/restrict_owner_geometry_preprocessor_3d.hpp`: define modes, normalized classes, query paths, fallback causes, 64-node input/result types, diagnostics, and the policy class.
- Create `apps/restrict_owner_geometry_preprocessor_3d.cpp`: implement full reference, optimized intersection, sample/segment region gates, closest-point certification, fallback, diagnostics, and parsing.
- Create `apps/restrict_owner_geometry_preprocessor_3d_test.cpp`: cover topology, geometry pathologies, 64-slot ordering, parsing, diagnostics, and exact oracle agreement.
- Modify `apps/CMakeLists.txt`: compile the new policy into `kfbim_3d_app_geometry`, add its test target, and link `Psapi` to the study executable on Windows.

### KFBI integration and study

- Modify `apps/neumann_exterior_zero_trace_3d.cpp`: construct one 64-node sample before owner selection, delegate to the policy, preserve accumulation order, capture/replay the exact workload, add mode selection, execute the case matrix, write all six CSV files, and enforce the numerical gates.
- Modify `apps/kfbi_phase_profile_3d.hpp`: add one aggregate restrict-owner preprocessing phase.
- Modify `apps/kfbi_phase_profile_3d.cpp`: name and serialize that phase without changing existing phase semantics.
- Modify `apps/kfbi_phase_profile_3d_test.cpp`: update exhaustive phase-ledger expectations and prove that the new phase remains outside GMRES.

No new source entry is required in `src/CMakeLists.txt`; all modified core `.cpp` files are already part of `kfbim_core`.

## Stable Interfaces

Add the following core certificate to
`src/geometry/nurbs_bezier_intersection_3d.hpp`:

```cpp
enum class NurbsElementSegmentCertificateKind3D {
    CertifiedMiss,
    CertifiedUniqueTransverseRoot,
    Unresolved
};

struct NurbsElementSegmentCertificate3D {
    NurbsElementSegmentCertificateKind3D kind =
        NurbsElementSegmentCertificateKind3D::Unresolved;
    std::optional<NurbsElementRoot3D> root;
    NurbsElementIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

NurbsElementSegmentCertificate3D certify_nurbs_bezier_element_segment_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementIntersectionOptions3D& options,
    std::optional<Eigen::Vector2d> preferred_seed = std::nullopt);
```

Add the following facade to
`src/geometry/nurbs_surface_intersector_3d.hpp`:

```cpp
class NurbsSurfaceIntersector3D;

class NurbsSurfaceCandidate3D {
public:
    int patch_index() const noexcept;
    int component() const noexcept;
    const NurbsAabb3D& bounds() const noexcept;

private:
    friend class NurbsSurfaceIntersector3D;
    NurbsSurfaceCandidate3D(
        const NurbsSurfaceIntersector3D* source,
        std::size_t query_element,
        int patch_index,
        int component,
        NurbsAabb3D bounds);

    const NurbsSurfaceIntersector3D* source_ = nullptr;
    std::size_t query_element_ = 0;
    int patch_index_ = -1;
    int component_ = -1;
    NurbsAabb3D bounds_;
};

struct NurbsSurfaceCandidateCertificate3D {
    NurbsElementSegmentCertificateKind3D kind =
        NurbsElementSegmentCertificateKind3D::Unresolved;
    std::optional<NurbsSurfaceCrossing3D> crossing;
    NurbsElementIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
    bool segment_parameter_strictly_interior = false;
    bool element_parameter_strictly_interior = false;
    bool patch_parameter_strictly_interior = false;
};

struct NurbsSurfaceFilteredIntersectionOptions3D {
    int maximum_independent_crossings = 2;
};

struct NurbsSurfaceFilteredIntersectionResult3D {
    NurbsSurfaceIntersectionResult3D intersection;
    bool all_candidates_processed = true;
    bool independent_crossing_limit_reached = false;
};
```

Add these public methods to `NurbsSurfaceIntersector3D`:

```cpp
std::vector<NurbsSurfaceCandidate3D> conservative_candidates(
    const NurbsAabb3D& query_bounds) const;

std::vector<NurbsSurfaceCandidate3D> conservative_segment_candidates(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end) const;

NurbsSurfaceCandidateCertificate3D certify_candidate_segment(
    const NurbsSurfaceCandidate3D& candidate,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end) const;

NurbsSurfaceFilteredIntersectionResult3D intersect_segment_candidates(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    const std::vector<NurbsSurfaceCandidate3D>& ordered_candidates,
    const NurbsSurfaceFilteredIntersectionOptions3D& options = {}) const;
```

Add this application interface to
`apps/restrict_owner_geometry_preprocessor_3d.hpp`:

```cpp
namespace kfbim::app3d {

constexpr std::size_t kRestrictOwnerTricubicSupportCount3D = 64;

enum class RestrictOwnerPreprocessMode3D {
    FullIntersectionReference,
    OptimizedIntersection,
    RegionClosestHybrid
};

enum class RestrictOwnerNormalizedClass3D {
    Target,
    UniqueForeign,
    FailClosedTarget
};

enum class RestrictOwnerQueryPath3D {
    FullIntersection,
    SweepTargetOnly,
    SegmentTargetOnly,
    ClosestCertifiedMiss,
    ClosestCertifiedRoot,
    OptimizedIntersection,
    FullIntersectionFallback,
    Count
};

enum class RestrictOwnerFallbackCause3D {
    None,
    MultipleCrossings,
    Unresolved,
    Overlap,
    Endpoint,
    NearTangent,
    FeatureContact,
    ParameterBoundary,
    Seam,
    Coincidence,
    Count
};

const char* restrict_owner_preprocess_mode_name_3d(
    RestrictOwnerPreprocessMode3D mode);

RestrictOwnerPreprocessMode3D
parse_restrict_owner_preprocess_mode_3d(const std::string& text);

struct RestrictOwnerSampleInput3D {
    int target_dof = -1;
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
    std::array<Eigen::Vector3d,
               kRestrictOwnerTricubicSupportCount3D> support_points{};
    std::array<bool,
               kRestrictOwnerTricubicSupportCount3D> wrong_side{};
};

struct RestrictOwnerPreprocessResult3D {
    int owner_dof = -1;
    RestrictOwnerNormalizedClass3D owner_class =
        RestrictOwnerNormalizedClass3D::Target;
    RestrictOwnerQueryPath3D query_path =
        RestrictOwnerQueryPath3D::FullIntersection;
    RestrictOwnerFallbackCause3D fallback_cause =
        RestrictOwnerFallbackCause3D::None;
    std::optional<geometry3d::NurbsSurfaceCrossing3D> foreign_crossing;
    std::optional<RestrictOwnerDecisionKind3D>
        legacy_decision_kind;
};

struct RestrictOwnerSampleResult3D {
    std::array<std::optional<RestrictOwnerPreprocessResult3D>,
               kRestrictOwnerTricubicSupportCount3D> nodes;
};

struct RestrictOwnerPreprocessOptions3D {
    bool collect_stage_timings = false;
    int maximum_independent_crossings = 2;
};

struct RestrictOwnerPreprocessDiagnostics3D {
    std::uint64_t wrong_side_queries = 0;
    std::array<std::uint64_t,
        static_cast<std::size_t>(
            RestrictOwnerQueryPath3D::Count)> path_counts{};
    std::array<std::uint64_t,
        static_cast<std::size_t>(
            RestrictOwnerFallbackCause3D::Count)> fallback_counts{};
    std::uint64_t compatible_aabb_candidates = 0;
    std::uint64_t foreign_aabb_candidates = 0;
    std::uint64_t control_hull_rejections = 0;
    std::uint64_t closest_point_attempts = 0;
    std::uint64_t closest_point_converged = 0;
    std::uint64_t closest_point_iterations = 0;
    std::uint64_t closest_certified_misses = 0;
    std::uint64_t closest_certified_roots = 0;
    std::uint64_t closest_unresolved = 0;
    std::uint64_t optimized_intersection_calls = 0;
    std::uint64_t full_fallback_calls = 0;
    std::uint64_t target_decisions = 0;
    std::uint64_t foreign_decisions = 0;
    std::uint64_t fail_closed_target_decisions = 0;
    double region_seconds = 0.0;
    double closest_point_seconds = 0.0;
    double optimized_intersection_seconds = 0.0;
    double full_fallback_seconds = 0.0;
};

class RestrictOwnerGeometryPreprocessor3D {
public:
    RestrictOwnerGeometryPreprocessor3D(
        const NativeNurbsSurface3D& surface,
        const SurfaceDofCloud3D& cloud,
        double grid_spacing,
        RestrictOwnerPreprocessMode3D mode,
        RestrictOwnerPreprocessOptions3D options = {});

    RestrictOwnerSampleResult3D preprocess_sample(
        const RestrictOwnerSampleInput3D& input);

    RestrictOwnerPreprocessMode3D mode() const noexcept;
    const RestrictOwnerPreprocessDiagnostics3D&
    diagnostics() const noexcept;
    void reset_diagnostics();

private:
    using Candidate = geometry3d::NurbsSurfaceCandidate3D;

    struct CandidatePartition {
        std::vector<Candidate> compatible;
        std::vector<Candidate> foreign;
    };

    bool is_compatible_patch(int target_patch, int patch) const;
    CandidatePartition partition_candidates(
        int target_patch,
        std::vector<Candidate> candidates) const;
    RestrictOwnerPreprocessResult3D make_target_result(
        int target_dof,
        RestrictOwnerQueryPath3D path) const;
    RestrictOwnerPreprocessResult3D make_fail_closed_target_result(
        int target_dof,
        RestrictOwnerQueryPath3D path,
        RestrictOwnerFallbackCause3D cause) const;
    RestrictOwnerPreprocessResult3D make_foreign_result(
        int owner_dof,
        const geometry3d::NurbsSurfaceCrossing3D& crossing,
        RestrictOwnerQueryPath3D path) const;
    RestrictOwnerFallbackCause3D classify_fallback_cause(
        const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
        const RestrictOwnerDecision3D& decision) const;
    RestrictOwnerPreprocessResult3D normalize_intersection_result(
        int target_dof,
        const Eigen::Vector3d& query,
        const Eigen::Vector3d& support,
        const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
        RestrictOwnerQueryPath3D path) const;
    RestrictOwnerPreprocessResult3D full_reference_result(
        int target_dof,
        const Eigen::Vector3d& query,
        const Eigen::Vector3d& support,
        RestrictOwnerQueryPath3D path);
    RestrictOwnerPreprocessResult3D optimized_intersection_result(
        int target_dof,
        const Eigen::Vector3d& query,
        const Eigen::Vector3d& support);

    const NativeNurbsSurface3D& surface_;
    const SurfaceDofCloud3D& cloud_;
    geometry3d::NurbsSurfaceIntersector3D intersector_;
    RestrictOwnerPreprocessMode3D mode_;
    RestrictOwnerPreprocessOptions3D options_;
    RestrictOwnerPreprocessDiagnostics3D diagnostics_;
};

} // namespace kfbim::app3d
```

The preprocessor owns its `NurbsSurfaceIntersector3D`, configured with
`maximum_element_extent = 2.0 * grid_spacing` and
`local_max_subdivision_depth = 4`. Application code never receives a BVH
node index or a `RationalBezierElement3D`.

---

### Task 1: Add typed core certificates and opaque candidate queries

**Files:**
- Modify: `src/geometry/nurbs_bezier_intersection_3d.hpp`
- Modify: `src/geometry/nurbs_bezier_intersection_3d.cpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.hpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.cpp`
- Test: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: the stable core declarations above; existing `intersect_nurbs_bezier_element_3d()`, `closest_point_nurbs_bezier_element_to_segment_3d()`, `select_sample_seeds()`, root canonicalization, close-root analysis, and topology feature detection.
- Produces: `NurbsElementSegmentCertificate3D`, `NurbsSurfaceCandidate3D`, `NurbsSurfaceCandidateCertificate3D`, and `NurbsSurfaceFilteredIntersectionResult3D` for Tasks 2–4.

- [ ] **Step 1: Write failing element-certificate tests**

Add `test_nurbs_element_segment_certificates()` next to
`test_closest_point_classifies_terminal_intersection_boxes()`. Reuse
`NurbsSurfacePatch3D::make_unit_square_xy()`, the existing separated-plane
fixture, the existing quadratic two-root graph, and the quarter cylinder.
The assertions must enforce these exact invariants:

```cpp
using CertificateKind =
    kfbim::geometry3d::NurbsElementSegmentCertificateKind3D;

require(miss.kind == CertificateKind::CertifiedMiss
            && !miss.root.has_value()
            && !miss.overlap_detected,
        "typed certificate proves a terminal miss");
require(root.kind == CertificateKind::CertifiedUniqueTransverseRoot
            && root.root.has_value()
            && root.root->transversality
                   > root.root->reliable_transversality_tolerance,
        "typed certificate preserves the producer reliability threshold");
require(tangent.kind == CertificateKind::Unresolved,
        "typed certificate fails closed at tangency");
require(two_root.kind == CertificateKind::Unresolved,
        "typed certificate does not collapse two roots");
require(overlap.kind == CertificateKind::Unresolved
            && overlap.overlap_detected,
        "typed certificate reports overlap as unresolved");
```

Register the function in the test executable's `main()`.

- [ ] **Step 2: Write failing candidate-facade and filtered-query tests**

Add `test_nurbs_surface_candidate_facade()` and
`test_filtered_nurbs_surface_intersection()` beside
`test_native_nurbs_surface_intersector()`. Check:

```cpp
const auto candidates =
    intersector.conservative_segment_candidates(start, end);
require(std::is_sorted(
            candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
                return std::make_tuple(
                           a.patch_index(), a.component())
                     < std::make_tuple(
                           b.patch_index(), b.component());
            }),
        "candidate metadata is deterministic");

NurbsSurfaceFilteredIntersectionOptions3D capped;
capped.maximum_independent_crossings = 2;
const auto filtered =
    intersector.intersect_segment_candidates(
        long_torus_start, long_torus_end, ordered, capped);
require(filtered.independent_crossing_limit_reached
            && filtered.intersection.crossings.size() == 2,
        "filtered query stops at two independent canonical roots");
```

Also assert that a periodic/G1 seam duplicate remains one canonical crossing,
that a unique torus crossing agrees with `intersect_segment()` in patch,
`u`, `v`, `edge_parameter`, and residual tolerance, and that passing a
candidate created by a different intersector throws `std::invalid_argument`.

- [ ] **Step 3: Build to verify the new tests fail**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test -- /m
```

Expected: compilation fails because the certificate and candidate facade
declarations are not present.

- [ ] **Step 4: Implement the typed element certificate**

Add `<optional>` to the header and implement
`certify_nurbs_bezier_element_segment_3d()` by extracting the current
projected/control-hull rejection, terminal separation, closest-point
assistance, root recovery, and terminal single-root proof. Return only:

```cpp
if (terminal_separation_is_proved)
    return {CertificateKind::CertifiedMiss, std::nullopt,
            diagnostics, false};
if (one_root_is_proved
    && root.transversality > root.reliable_transversality_tolerance)
    return {CertificateKind::CertifiedUniqueTransverseRoot, root,
            diagnostics, false};
return {CertificateKind::Unresolved, diagnostic_root,
        diagnostics, overlap_detected};
```

`diagnostic_root` may be populated for an unresolved contact, but callers may
consume `root` as a numerical owner only when `kind` is
`CertifiedUniqueTransverseRoot`. Keep the current terminal certificate depth
of `6`; do not add a second tolerance.

- [ ] **Step 5: Implement the opaque facade and filtered query**

Make candidate construction private to the intersector and validate
`candidate.source_ == this` on every consuming call. Validate finite ordered
AABBs, distinct finite segment endpoints, candidate uniqueness, and
`maximum_independent_crossings >= 2`.

Refactor the current candidate loop into one internal routine used by both
`intersect_segment_impl()` and `intersect_segment_candidates()`. The filtered
routine must:

```cpp
for (const NurbsSurfaceCandidate3D& candidate : ordered_candidates) {
    validate_candidate_owner(candidate);
    process_existing_element_intersection(candidate.query_element_);
    canonicalize_accumulated_roots();
    if (has_two_disjoint_canonical_root_intervals()) {
        result.all_candidates_processed = false;
        result.independent_crossing_limit_reached = true;
        break;
    }
}
```

Two canonical roots are independent only when their segment-parameter
uncertainty intervals are disjoint using the existing segment parameter
tolerance. Preserve `intersect_segment()` output and diagnostics for the
unfiltered path.

- [ ] **Step 6: Run core tests**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test -- /m
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected: exit code `0` and final line
`native NURBS model tests passed`.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/geometry/nurbs_bezier_intersection_3d.hpp src/geometry/nurbs_bezier_intersection_3d.cpp src/geometry/nurbs_surface_intersector_3d.hpp src/geometry/nurbs_surface_intersector_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: expose certified 3d nurbs candidate queries"
```

---

### Task 2: Implement full-reference ownership and exact local topology semantics

**Files:**
- Create: `apps/restrict_owner_geometry_preprocessor_3d.hpp`
- Create: `apps/restrict_owner_geometry_preprocessor_3d.cpp`
- Create: `apps/restrict_owner_geometry_preprocessor_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1 core APIs, `select_restrict_correction_owner_3d()`,
  `NativeNurbsSurface3D`, and `SurfaceDofCloud3D`.
- Produces: the complete stable app interface above, with a working
  `FullIntersectionReference` mode and exact non-transitive compatibility.

- [ ] **Step 1: Add the new test target and failing mode tests**

Add the implementation source to `kfbim_3d_app_geometry`, create
`restrict_owner_geometry_preprocessor_3d_test`, link it to
`kfbim_3d_app_geometry`, and require C++17.

In the test file, add a local `require(bool, const std::string&)`, a
`require_throws_contains()` helper, and tests for exact strings:

```cpp
using Mode = kfbim::app3d::RestrictOwnerPreprocessMode3D;
using NormalizedClass =
    kfbim::app3d::RestrictOwnerNormalizedClass3D;
using QueryPath = kfbim::app3d::RestrictOwnerQueryPath3D;
using FallbackCause =
    kfbim::app3d::RestrictOwnerFallbackCause3D;
using CertificateKind =
    kfbim::geometry3d::NurbsElementSegmentCertificateKind3D;

require(parse_restrict_owner_preprocess_mode_3d(
            "full_intersection_reference")
            == Mode::FullIntersectionReference,
        "parse full reference mode");
require(parse_restrict_owner_preprocess_mode_3d(
            "optimized_intersection")
            == Mode::OptimizedIntersection,
        "parse optimized mode");
require(parse_restrict_owner_preprocess_mode_3d(
            "region_closest_hybrid")
            == Mode::RegionClosestHybrid,
        "parse hybrid mode");
```

Reject the empty string, `full`, and mixed trailing text.

- [ ] **Step 2: Add failing full-reference and topology tests**

Build real sample inputs with one wrong-side slot and assert:

- no crossing returns the target with class `Target`;
- a same-patch or explicitly listed G1 crossing returns the target;
- one reliable foreign non-G1 crossing returns the existing 2-by-2 nearest
  DOF and class `UniqueForeign`;
- overlap, tangent, feature contact, multiple roots, unresolved geometry, and
  unrelated coincidence return the target with class `FailClosedTarget`;
- in a three-patch chain where patch `0` explicitly lists only patch `1`,
  a crossing on patch `2` is foreign even if patch `1` lists patch `2`.

Every full-reference result must use `RestrictOwnerQueryPath3D::FullIntersection`.

- [ ] **Step 3: Build to verify the app tests fail**

Run:

```powershell
cmake --build build --config Release --target restrict_owner_geometry_preprocessor_3d_test -- /m
```

Expected: compilation or linking fails because the new preprocessor is not
implemented.

- [ ] **Step 4: Implement validation, parsing, normalization, and reference mode**

The constructor must reject non-finite/non-positive spacing, empty or
incompatible cloud topology, any target DOF outside the cloud, non-finite
sample points, a wrong-side query with zero-length segment, and any
`maximum_independent_crossings` other than `2`.

Map the existing selector as follows:

```cpp
switch (decision.kind) {
case RestrictOwnerDecisionKind3D::ForeignNonG1SingleCrossing:
    return make_foreign_result(
        decision.owner_dof, intersection.crossings.front(),
        QueryPath::FullIntersection);
case RestrictOwnerDecisionKind3D::MultipleCrossingFallback:
    return make_fail_closed_target_result(
        target_dof, QueryPath::FullIntersection,
        FallbackCause::MultipleCrossings);
case RestrictOwnerDecisionKind3D::DegenerateCrossingFallback:
case RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback:
    return make_fail_closed_target_result(
        target_dof, QueryPath::FullIntersection,
        classify_fallback_cause(intersection, decision));
case RestrictOwnerDecisionKind3D::TargetSideNode:
case RestrictOwnerDecisionKind3D::TargetOrG1SingleCrossing:
case RestrictOwnerDecisionKind3D::NoCrossingFallback:
    return make_target_result(target_dof, QueryPath::FullIntersection);
}
throw std::logic_error("invalid restrict-owner decision kind");
```

Catch only `UnresolvedNurbsIntersectionCandidate3D` and the existing exact
`"coincident roots on unrelated NURBS patches"` runtime error; both become
`FailClosedTarget`. Propagate every other exception.

- [ ] **Step 5: Implement 64-slot result and diagnostics invariants**

For `wrong_side[q] == false`, return `std::nullopt` and do not increment any
query counter. For `wrong_side[q] == true`, return one result and increment
exactly one path, one normalized-class count, and
`wrong_side_queries`. At function exit enforce:

```cpp
if (classified != diagnostics_.wrong_side_queries - before_queries)
    throw std::logic_error(
        "restrict-owner preprocessing did not classify every wrong-side node");
```

`reset_diagnostics()` must zero the entire diagnostics struct without
rebuilding the intersector.

- [ ] **Step 6: Run focused and existing ownership tests**

Run:

```powershell
cmake --build build --config Release --target crossing_owner_restrict_3d_test restrict_owner_geometry_preprocessor_3d_test -- /m
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
```

Expected: both executables exit `0`; the existing selector behavior is
unchanged and the new executable prints
`restrict-owner geometry preprocessor tests passed`.

- [ ] **Step 7: Commit Task 2**

```powershell
git add apps/CMakeLists.txt apps/restrict_owner_geometry_preprocessor_3d.hpp apps/restrict_owner_geometry_preprocessor_3d.cpp apps/restrict_owner_geometry_preprocessor_3d_test.cpp
git commit -m "feat: add reference restrict owner preprocessor"
```

---

### Task 3: Add decision-oriented optimized intersection

**Files:**
- Modify: `apps/restrict_owner_geometry_preprocessor_3d.cpp`
- Modify: `apps/restrict_owner_geometry_preprocessor_3d_test.cpp`

**Interfaces:**
- Consumes: Task 1 conservative candidates and filtered intersection; Task 2
  normalized result and diagnostics.
- Produces: exact `OptimizedIntersection` mode with conservative target
  certification and full-query fallback.

- [ ] **Step 1: Write failing optimized-path tests**

Add cases that assert:

```cpp
require(no_foreign.owner_dof == target
            && no_foreign.query_path == QueryPath::SegmentTargetOnly,
        "no foreign AABB candidate certifies the target");
require(unique_foreign.owner_dof == oracle.owner_dof
            && unique_foreign.query_path == QueryPath::OptimizedIntersection,
        "filtered unique foreign root matches the oracle");
require(two_roots.owner_dof == target
            && two_roots.owner_class == NormalizedClass::FailClosedTarget,
        "two independent roots stop with the target");
require(unresolved.owner_dof == oracle.owner_dof
            && unresolved.query_path == QueryPath::FullIntersectionFallback,
        "unresolved filtered work falls back to the full oracle");
```

Include one root on the target patch, one root on an explicit local G1 patch,
one unique foreign non-G1 root, two roots on one Bezier leaf, two roots on
different leaves, a periodic seam root, unrelated coincidence, and multiple
components.

- [ ] **Step 2: Run the test to verify optimized mode fails**

Run:

```powershell
cmake --build build --config Release --target restrict_owner_geometry_preprocessor_3d_test -- /m
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
```

Expected: the new optimized assertions fail because the mode still lacks its
candidate path.

- [ ] **Step 3: Implement the foreign AABB gate**

For each wrong-side segment, call
`conservative_segment_candidates(query, support)`, partition candidates by:

```cpp
const bool compatible =
    candidate.patch_index() == target_patch
    || std::find(local_smooth.begin(), local_smooth.end(),
                 candidate.patch_index()) != local_smooth.end();
```

Do not expand `local_smooth` through another patch. Accumulate compatible and
foreign candidate counts. If the foreign vector is empty, return target with
`SegmentTargetOnly` without running a root solver.

- [ ] **Step 4: Implement foreign-first filtered decision and fallback**

Construct `ordered_candidates` with all foreign candidates first and all
compatible candidates second, preserving deterministic element order inside
each partition. Invoke the filtered query with cap `2`.

Accept directly only these outcomes:

```cpp
if (filtered.independent_crossing_limit_reached)
    return make_fail_closed_target_result(
        target, QueryPath::OptimizedIntersection,
        FallbackCause::MultipleCrossings);
if (filtered.all_candidates_processed
    && !filtered.intersection.overlap_detected
    && filtered.intersection.diagnostics.unresolved_candidates == 0
    && filtered.intersection.diagnostics.ambiguous_root_clusters == 0)
    return normalize_intersection_result(
        target, query, support, filtered.intersection,
        QueryPath::OptimizedIntersection);
```

Every incomplete, overlap, ambiguous, unresolved, endpoint, feature, seam,
near-tangent, or coincidence outcome invokes `FullIntersectionReference`
internally and changes the returned path to `FullIntersectionFallback`.
Increment `full_fallback_calls` and exactly one fallback cause.

- [ ] **Step 5: Run the optimized suite**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test crossing_owner_restrict_3d_test restrict_owner_geometry_preprocessor_3d_test -- /m
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
```

Expected: all three executables exit `0`; every optimized test owner equals
the reference owner and the path/fallback counters sum to the number of
wrong-side queries.

- [ ] **Step 6: Commit Task 3**

```powershell
git add apps/restrict_owner_geometry_preprocessor_3d.cpp apps/restrict_owner_geometry_preprocessor_3d_test.cpp
git commit -m "feat: add optimized restrict owner intersections"
```

---

### Task 4: Add sample-region and certified-closest hybrid

**Files:**
- Modify: `apps/restrict_owner_geometry_preprocessor_3d.cpp`
- Modify: `apps/restrict_owner_geometry_preprocessor_3d_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–3 candidate certificates, local compatibility, optimized
  fallback, and normalized outputs.
- Produces: exact `RegionClosestHybrid` mode.

- [ ] **Step 1: Write failing sample and closest-path tests**

Add these independent path assertions:

- a sweep AABB containing only target/local-G1 leaves classifies every
  wrong-side slot as `SweepTargetOnly`;
- a failed sweep followed by a segment with no foreign leaf uses
  `SegmentTargetOnly`;
- a segment/leaf AABB false positive with a terminal separation certificate
  uses `ClosestCertifiedMiss`;
- one strictly interior, reliable foreign root with every other leaf
  certified absent uses `ClosestCertifiedRoot` and the oracle owner;
- two unresolved foreign leaves, any compatible unresolved leaf, an element
  or patch boundary, a periodic/G1 seam, a non-G1 feature, endpoint, tangent,
  overlap, or multi-root event uses optimized fallback.

Also add a partial one-to-many L-prism connection case and prove that a sample
near that connection never receives a false sweep certificate.

- [ ] **Step 2: Run the test to verify hybrid mode fails**

Run:

```powershell
cmake --build build --config Release --target restrict_owner_geometry_preprocessor_3d_test -- /m
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
```

Expected: hybrid path assertions fail because the mode still delegates
directly to optimized intersection.

- [ ] **Step 3: Implement the sample sweep certificate**

Construct the conservative AABB from `input.query` and all 64 support points:

```cpp
NurbsAabb3D sweep{input.query, input.query};
for (const Eigen::Vector3d& point : input.support_points) {
    sweep.lower = sweep.lower.cwiseMin(point);
    sweep.upper = sweep.upper.cwiseMax(point);
}
```

Call `conservative_candidates(sweep)`. If every candidate is locally
compatible, fill every wrong-side result with target,
`RestrictOwnerNormalizedClass3D::Target`, and `SweepTargetOnly`. Multiple or
tangential compatible roots do not invalidate this owner certificate.

- [ ] **Step 4: Implement segment separation and closest certification**

For a sample that fails the sweep gate:

1. collect and partition segment candidates;
2. return `SegmentTargetOnly` if there is no foreign candidate;
3. call `certify_candidate_segment()` for deterministic candidate order;
4. count a control-hull rejection when
   `diagnostics.conservative_rejections > 0` and
   `diagnostics.closest_point_attempts == 0`;
5. if every foreign candidate is a certified miss, return the target; use
   `ClosestCertifiedMiss` when at least one foreign certificate attempted
   closest point and `SegmentTargetOnly` when control-hull separation alone
   proved all foreign misses;
6. permit a closest positive-root path only when exactly one foreign
   candidate is a certified unique root and every other candidate, including
   compatible candidates, is a certified miss.

Accept the remaining foreign candidate only with:

```cpp
const bool safe_root =
    certificate.kind == CertificateKind::CertifiedUniqueTransverseRoot
    && certificate.crossing.has_value()
    && certificate.segment_parameter_strictly_interior
    && certificate.element_parameter_strictly_interior
    && certificate.patch_parameter_strictly_interior
    && !certificate.crossing->feature_edge_contact
    && certificate.crossing->transversality
           > certificate.crossing->reliable_transversality_tolerance;
```

The `safe_root` is passed through the existing 2-by-2 DOF selector and
returns `ClosestCertifiedRoot`. Every unresolved, non-interior, or
multi-candidate state calls the Task 3 optimized path.

- [ ] **Step 5: Prove oracle agreement on all unit/pathology cases**

For every fixture, evaluate `FullIntersectionReference`,
`OptimizedIntersection`, and `RegionClosestHybrid` from the same
`RestrictOwnerSampleInput3D`. Require equal `owner_dof` and equal
target-versus-foreign class; require exact path counts but do not require the
fast paths to reproduce the old seven detailed reason codes.

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test crossing_owner_restrict_3d_test restrict_owner_geometry_preprocessor_3d_test -- /m
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
```

Expected: all three executables exit `0`; the preprocessor test prints its
passed line and reports zero unit-workload owner mismatch for both candidates.

- [ ] **Step 6: Commit Task 4**

```powershell
git add apps/restrict_owner_geometry_preprocessor_3d.cpp apps/restrict_owner_geometry_preprocessor_3d_test.cpp
git commit -m "feat: add hybrid restrict owner preprocessing"
```

---

### Task 5: Integrate policy selection into the KFBI trace pipeline

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/kfbi_phase_profile_3d.hpp`
- Modify: `apps/kfbi_phase_profile_3d.cpp`
- Modify: `apps/kfbi_phase_profile_3d_test.cpp`

**Interfaces:**
- Consumes: complete preprocessor from Task 4.
- Produces: mode-selectable fixed trace templates, exact workload capture,
  correction comparison, aggregate phase profiling, and the no-GMRES-query
  invariant used by Tasks 6–8.

- [ ] **Step 1: Add the shared tricubic sample builder and workload record**

In the application anonymous namespace, define:

```cpp
struct RestrictOwnerTraceStencil3D {
    app3d::RestrictOwnerSampleInput3D owner_input;
    std::array<int, 64> grid_nodes{};
    std::array<double, 64> weights{};
};

struct RestrictOwnerOracleNode3D {
    std::uint8_t slot = 0;
    int owner_dof = -1;
    app3d::RestrictOwnerNormalizedClass3D owner_class =
        app3d::RestrictOwnerNormalizedClass3D::Target;
    int crossing_patch = -1;
};

struct RestrictOwnerWorkloadSample3D {
    int target_dof = -1;
    int side = -1;
    int layer = -1;
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
    std::array<int, 3> lower_stencil_index{};
    std::vector<RestrictOwnerOracleNode3D> oracle_nodes;
};

using RestrictOwnerWorkload3D =
    std::vector<RestrictOwnerWorkloadSample3D>;
```

Add one helper that creates `RestrictOwnerTraceStencil3D` from the grid,
grid labels, target, desired side, query, and lower stencil index. Both the
pipeline and benchmark replay must call this same helper; there is no second
copy of cubic weights, grid indexing, or wrong-side classification.

- [ ] **Step 2: Write an integration failure check before changing the pipeline**

Add an internal deterministic fingerprint over target DOF, side, layer,
lower index, all 64 grid IDs, exact weight bit patterns, and wrong-side bits.
Expose:

```cpp
std::uint64_t restrict_owner_workload_fingerprint() const noexcept;
const app3d::RestrictOwnerPreprocessDiagnostics3D&
restrict_owner_preprocess_diagnostics() const;
double restrict_owner_correction_linf_difference(
    const PanelCenterHarmonicJetKFBI3D& other) const;
```

Before implementation, build the executable. Expected compilation failure is
the absence of these methods and the mode-aware constructor.

- [ ] **Step 3: Replace the boolean owner flag with an optional mode**

Change the constructor tail to:

```cpp
bool build_exterior_only_restrict = false,
std::optional<app3d::RestrictOwnerPreprocessMode3D>
    restrict_owner_mode = std::nullopt,
app3d::PhaseProfile3D* phase_profile = nullptr,
RestrictOwnerWorkload3D* workload_capture = nullptr
```

`std::nullopt` preserves the ordinary readiness route with no crossing-owner
templates. Existing owner probes pass
`FullIntersectionReference` unless an explicit mode was selected. Construct
one `RestrictOwnerGeometryPreprocessor3D` member when the optional mode has a
value.

- [ ] **Step 4: Refactor `build_trace_sample()` without changing arithmetic order**

First fill all 64 grid IDs, weights, support points, wrong-side bits, legacy
corrections, and the optional compact workload record. Then call
`preprocess_sample()` once. Finally loop over `q=0..63` and accumulate owner
corrections:

```cpp
for (int q = 0; q < 64; ++q) {
    if (!trace.owner_input.wrong_side[static_cast<std::size_t>(q)])
        continue;
    const auto& decision =
        owner_result.nodes[static_cast<std::size_t>(q)].value();
    const int owner = decision.owner_dof;
    const Eigen::Vector3d owner_xi =
        local_coordinate(owner,
            trace.owner_input.support_points[static_cast<std::size_t>(q)]);
    owner_evaluations[owner] +=
        correction_sign
        * trace.weights[static_cast<std::size_t>(q)]
        * fit_.space().basis(
            owner_xi.x(), owner_xi.y(), owner_xi.z());
}
```

Initialize a newly inserted `Eigen::VectorXd` to
`Eigen::VectorXd::Zero(fit_.dimension())` before `+=`, exactly as in the
current map path. Keep old seven-kind counts only for full-reference
diagnostics; fast paths use normalized-class and query-path counters.

- [ ] **Step 5: Add aggregate phase profiling and GMRES immutability checks**

Add `RestrictOwnerGeometryPreprocessing` to `PhaseProfileKind3D`. Time the
whole fixed-template preprocessing region once, not each query. Update the
phase test so the fixed-initialization leaves remain exhaustive.

Snapshot the complete diagnostics struct immediately before GMRES and require
byte-for-byte counter equality immediately after GMRES. The existing
`restrict_owner_geometry_query_count()` must remain unchanged as a compatible
summary.

- [ ] **Step 6: Add mode parsing for existing owner probe/profile routes**

Read:

```text
KFBIM_3D_RESTRICT_OWNER_MODE
```

with accepted values:

```text
full_intersection_reference
optimized_intersection
region_closest_hybrid
```

An unset variable selects `full_intersection_reference`. Add the values to
`print_usage()`. Use a consumed-position checked integer parser for new CLI
levels so `16junk` and `16.5` are rejected.

- [ ] **Step 7: Build and run focused integration checks**

Run:

```powershell
cmake --build build --config Release --target kfbim_phase_profile_3d_test restrict_owner_geometry_preprocessor_3d_test neumann_exterior_zero_trace_3d -- /m
.\build\apps\Release\kfbi_phase_profile_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
$env:KFBIM_3D_RESTRICT_OWNER_MODE='full_intersection_reference'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe-owner 16
```

Expected: both unit executables pass; the owner probe converges, reports a
nonzero preprocessing query count, reports identical pre/post-GMRES geometry
counters, and retains the reference numerical output.

- [ ] **Step 8: Commit Task 5**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/kfbi_phase_profile_3d.hpp apps/kfbi_phase_profile_3d.cpp apps/kfbi_phase_profile_3d_test.cpp
git commit -m "feat: integrate restrict owner preprocessing modes"
```

---

### Task 6: Add the reproducible preprocessing and KFBI study driver

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 5 workload capture, candidate replay, pipeline comparison,
  existing `make_geometry()`, rigid transforms, Cauchy construction, and
  `run_dirichlet_normal_case()`.
- Produces: `--owner-preprocess-study`, six checkpointed CSV files, timing
  summaries, memory measurements, mismatch rows, and automated gates.

- [ ] **Step 1: Define the exact study matrix**

Add case IDs and transforms:

```cpp
torus_identity
hollow_cylinder_identity
l_prism_identity
torus_rot_axis123_17deg
hollow_cylinder_rot_axis123_17deg
l_prism_rot_axis123_17deg
l_prism_rot_axis123_17deg_t_xyz_1
```

Reuse `baseline`, `rot_axis123_17deg`, and
`rot_axis123_17deg_t_xyz_1` from
`make_l_prism_dirichlet_rigid_study_cases_3d()`; the transforms are geometry
independent. Select:

- `N=16`: the three identity cases, preprocessing only;
- `N=32` and `N=64`: all seven cases, preprocessing plus all three
  end-to-end Dirichlet modes;
- `N=128`: only `l_prism_rot_axis123_17deg`, preprocessing plus all three
  end-to-end Dirichlet modes.

- [ ] **Step 2: Add exact accuracy comparison**

For every wrong-side node compare the candidate to the compact oracle and
compute:

```cpp
owner_mismatch
foreign_true_positive
foreign_false_positive
foreign_false_negative
foreign_wrong_owner
foreign_precision
foreign_recall
max_mismatched_abs_weight
```

Define precision or recall as `1.0` when its denominator is zero and the
corresponding oracle/candidate foreign count is also zero. A foreign true
positive requires both `UniqueForeign` and identical `owner_dof`.

Compute `legacy_reason_mismatch` only where both paths produced an actual
`legacy_decision_kind`. Leave the comparison not-applicable if either
optional is empty. Report the mismatch count in the CSV, but never include
it in `pass`; a
certified target fast path is not required to fabricate an oracle crossing.

Write only mismatches to `preprocess_mismatches.csv`.

- [ ] **Step 3: Add timing protocol and memory sampler**

The full pipeline/reference run records the oracle once. Construct each
candidate preprocessor outside its query timer, run one untimed warm-up, then
alternate timed order:

```cpp
if (repetition % 2 == 0) {
    time(OptimizedIntersection);
    time(RegionClosestHybrid);
} else {
    time(RegionClosestHybrid);
    time(OptimizedIntersection);
}
```

Use `5` candidate repetitions at `N=16` and `N=32`, `3` at `N=64`, and `1`
at `N=128`. Median is the ordinary sorted median. p95 is nearest-rank
`sorted[ceil(0.95 * count) - 1]`.

Run one separate instrumentation replay with
`collect_stage_timings=true`; do not include it in timing medians or
speedups. On Windows sample `PROCESS_MEMORY_COUNTERS_EX::WorkingSetSize`
during the timed region and link `Psapi`; on non-Windows sample
`getrusage(RUSAGE_SELF)`. Report absolute peak working-set bytes and the
working-set increase from the pre-run baseline.

- [ ] **Step 4: Add all six CSV schemas and checkpoint writers**

Create the output directory from
`KFBIM_3D_RESTRICT_OWNER_STUDY_OUTPUT_DIR`, defaulting to
`output/restrict_owner_preprocess_study`. At startup create headers, and
after each completed case rewrite all accumulated rows so an interrupted
long run retains completed cases.

Use these exact files and key columns:

```text
preprocess_timing_repeats.csv
  case_id,geometry,pose,N,h,mode,repetition,run_order,
  wrong_side_queries,construction_seconds,query_seconds,
  total_preprocess_seconds,queries_per_second,
  peak_working_set_bytes,working_set_increase_bytes

preprocess_summary.csv
  case_id,geometry,pose,N,h,mode,repetitions,wrong_side_queries,
  construction_seconds_median,query_seconds_median,query_seconds_p95,
  total_seconds_median,total_seconds_p95,queries_per_second_median,
  speedup_vs_full,peak_working_set_bytes_max,accepted

preprocess_accuracy.csv
  case_id,geometry,pose,N,h,mode,query_count,oracle_target,
  oracle_foreign,owner_mismatch,foreign_true_positive,
  foreign_false_positive,foreign_false_negative,foreign_wrong_owner,
  foreign_precision,foreign_recall,max_mismatched_abs_weight,
  max_owner_correction_abs_difference,legacy_reason_mismatch,pass

preprocess_path_counts.csv
  case_id,geometry,pose,N,h,mode,path,count,fraction,
  compatible_aabb_candidates,foreign_aabb_candidates,
  control_hull_rejections,closest_attempts,closest_converged,
  closest_iterations,closest_certified_misses,closest_certified_roots,
  closest_unresolved,optimized_calls,full_fallback_calls,
  region_seconds,closest_seconds,optimized_seconds,full_fallback_seconds

preprocess_mismatches.csv
  case_id,geometry,pose,N,h,mode,target_dof,target_patch,side,layer,
  grid_node,weight,query_x,query_y,query_z,support_x,support_y,support_z,
  oracle_owner_dof,oracle_owner_patch,candidate_owner_dof,
  candidate_owner_patch,oracle_class,candidate_class,candidate_path,
  oracle_crossing_patch,candidate_crossing_patch

kfbi_numerical_results.csv
  case_id,geometry,pose,N,h,mode,dofs,preprocess_seconds,
  pipeline_setup_seconds,pipeline_speedup_vs_full,solve_seconds,
  converged,iterations,final_residual,operator_residual_linf,
  exterior_condition_linf,boundary_residual_linf,route_mismatch_linf,
  interior_linf,interior_order,exact_grid_linf,exact_equation_linf,
  interior_relative_difference_vs_full,
  exact_grid_relative_difference_vs_full,
  exact_equation_relative_difference_vs_full,
  geometry_queries_before_gmres,geometry_queries_after_gmres,pass
```

Fallback causes are emitted as additional long-form path rows named
`full_intersection_fallback:<cause>`.

- [ ] **Step 5: Add end-to-end numerical gates**

After zero owner mismatch, construct one pipeline per mode and require equal
workload fingerprints. Compare owner correction arrays using:

```cpp
max_abs_difference
    <= 64.0 * std::numeric_limits<double>::epsilon()
       * std::max(1.0, reference_correction_linf)
```

Run the existing manufactured harmonic Dirichlet crossing-owner solve and
require:

- identical convergence flags and iteration counts;
- unchanged geometry counters across GMRES;
- final relative residual `<= 2.0e-10`;
- candidate/reference relative differences for `interior_linf`,
  `exact_grid_linf`, and `exact_equation_linf` each `<= 1.0e-12`;
- candidate observed interior order plus `1.0e-12` is not below the reference
  order.

Use:

`std::abs(a-b) / std::max({1.0e-300, std::abs(a), std::abs(b)})` for reported relative
differences.

- [ ] **Step 6: Add CLI and fail-fast progression**

Add:

```text
neumann_exterior_zero_trace_3d.exe --owner-preprocess-study [N ...]
```

Default levels are `16 32 64`. Reject non-powers of two, values below `16`,
numeric suffixes, and decimals. If `128` is requested, execute it only after
the selected run has completed the smaller accuracy gates; otherwise print
the failing case/mode and return nonzero without starting `N=128`.

- [ ] **Step 7: Build and run the `N=16` smoke study**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test crossing_owner_restrict_3d_test restrict_owner_geometry_preprocessor_3d_test kfbim_phase_profile_3d_test neumann_exterior_zero_trace_3d -- /m
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\kfbi_phase_profile_3d_test.exe
$env:KFBIM_3D_RESTRICT_OWNER_STUDY_OUTPUT_DIR='output/restrict_owner_preprocess_n16'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --owner-preprocess-study 16
```

Expected: all tests exit `0`; all six CSV files exist; the three identity
cases are present; both candidates have zero owner mismatch, false positive,
false negative, and wrong foreign owner; no GMRES row is emitted for `N=16`.

- [ ] **Step 8: Commit Task 6**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/CMakeLists.txt
git commit -m "feat: add restrict owner preprocessing study"
```

---

### Task 7: Validate the complete small and medium geometry matrix

**Files:**
- Test: generated files under `output/restrict_owner_preprocess_n16_n32`
- Test: generated files under `output/restrict_owner_preprocess_n32_n64`

**Interfaces:**
- Consumes: Task 6 study executable and CSV gates.
- Produces: accepted `N=16`, `N=32`, and `N=64` evidence permitting the final
  `N=128` run.

- [ ] **Step 1: Run `N=16` and `N=32`**

```powershell
$env:KFBIM_3D_RESTRICT_OWNER_STUDY_OUTPUT_DIR='output/restrict_owner_preprocess_n16_n32'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --owner-preprocess-study 16 32
```

Expected: exit `0`; N16 contains three preprocessing cases; N32 contains all
seven preprocessing and end-to-end cases.

- [ ] **Step 2: Enforce the `N=32` CSV gates**

```powershell
$accuracy = Import-Csv output\restrict_owner_preprocess_n16_n32\preprocess_accuracy.csv
$candidate = $accuracy | Where-Object { $_.mode -ne 'full_intersection_reference' }
if (($candidate | Where-Object {
    [int64]$_.owner_mismatch -ne 0 -or
    [int64]$_.foreign_false_positive -ne 0 -or
    [int64]$_.foreign_false_negative -ne 0 -or
    [int64]$_.foreign_wrong_owner -ne 0 -or
    $_.pass -ne '1'
}).Count -ne 0) { throw 'N16/N32 owner accuracy gate failed' }

$numerical = Import-Csv output\restrict_owner_preprocess_n16_n32\kfbi_numerical_results.csv
if (($numerical | Where-Object {
    $_.mode -ne 'full_intersection_reference' -and $_.pass -ne '1'
}).Count -ne 0) { throw 'N32 KFBI numerical gate failed' }
```

Expected: no exception.

- [ ] **Step 3: Run `N=32` and `N=64`**

```powershell
$env:KFBIM_3D_RESTRICT_OWNER_STUDY_OUTPUT_DIR='output/restrict_owner_preprocess_n32_n64'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --owner-preprocess-study 32 64
```

Expected: exit `0`; all seven cases exist at both levels; candidate timing
rows contain five repeats at N32 and three at N64.

- [ ] **Step 4: Enforce `N=64`, GMRES, and order gates**

```powershell
$accuracy = Import-Csv output\restrict_owner_preprocess_n32_n64\preprocess_accuracy.csv
if (($accuracy | Where-Object {
    $_.mode -ne 'full_intersection_reference' -and $_.pass -ne '1'
}).Count -ne 0) { throw 'N32/N64 owner accuracy gate failed' }

$numerical = Import-Csv output\restrict_owner_preprocess_n32_n64\kfbi_numerical_results.csv
if (($numerical | Where-Object {
    $_.mode -ne 'full_intersection_reference' -and (
        $_.pass -ne '1' -or
        [int64]$_.geometry_queries_before_gmres -ne
            [int64]$_.geometry_queries_after_gmres)
}).Count -ne 0) { throw 'N32/N64 KFBI or GMRES geometry gate failed' }
```

Expected: no exception. If the executable or a gate fails, stop before
`N=128`, preserve the mismatch CSV, and diagnose the first mismatching
sample against the full oracle; do not weaken a tolerance or reroute an
uncertain case to a foreign owner.

- [ ] **Step 5: Verify the worktree remains free of generated artifacts**

```powershell
git status --short
```

Expected: no output paths or CSV files are tracked or staged.

---

### Task 8: Run the accepted `N=128` rotated L-prism case and report

**Files:**
- Test: generated files under `output/restrict_owner_preprocess_final`
- Report: final user-facing result assembled from the six generated CSVs

**Interfaces:**
- Consumes: Task 7 zero-mismatch evidence and Task 6 final study driver.
- Produces: the requested speed/accuracy comparison through `N=128` and the
  evidence-based policy recommendation.

- [ ] **Step 1: Re-run focused tests immediately before the expensive case**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test crossing_owner_restrict_3d_test restrict_owner_geometry_preprocessor_3d_test kfbim_phase_profile_3d_test neumann_exterior_zero_trace_3d -- /m
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\kfbi_phase_profile_3d_test.exe
```

Expected: all four executables exit `0`.

- [ ] **Step 2: Run the final matrix including `N=128`**

```powershell
$env:KFBIM_3D_RESTRICT_OWNER_STUDY_OUTPUT_DIR='output/restrict_owner_preprocess_final'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --owner-preprocess-study 32 64 128
```

Expected: N32/N64 gates pass first; one rotated L-prism N128 reference run,
one warm-up per candidate, one timed repetition per candidate, one
instrumentation replay, and all three end-to-end modes complete.

- [ ] **Step 3: Enforce final accuracy and numerical gates**

```powershell
$accuracy = Import-Csv output\restrict_owner_preprocess_final\preprocess_accuracy.csv
if (($accuracy | Where-Object {
    $_.mode -ne 'full_intersection_reference' -and $_.pass -ne '1'
}).Count -ne 0) { throw 'final owner accuracy gate failed' }

$numerical = Import-Csv output\restrict_owner_preprocess_final\kfbi_numerical_results.csv
if (($numerical | Where-Object {
    $_.mode -ne 'full_intersection_reference' -and $_.pass -ne '1'
}).Count -ne 0) { throw 'final KFBI numerical gate failed' }

$mismatches = Import-Csv output\restrict_owner_preprocess_final\preprocess_mismatches.csv
if ($mismatches.Count -ne 0) { throw 'final mismatch CSV is not empty' }
```

Expected: no exception and zero mismatch rows.

- [ ] **Step 4: Extract the requested comparison**

From `preprocess_summary.csv`, report for each geometry/pose/N/mode:
median preprocessing seconds, p95, queries/s, speedup versus full reference,
peak working set, and acceptance. From `preprocess_path_counts.csv`, report
sweep/segment/closest/optimized/full-fallback coverage and the dominant
fallback causes. From `kfbi_numerical_results.csv`, report interior maximum
error, observed order for N32-to-N64 and N64-to-N128 where available, GMRES
iterations, final residual, setup time, solve time, and pre/post-GMRES
geometry query counts.

- [ ] **Step 5: Select or reject a candidate production policy**

Keep both modes selectable. Recommend `RegionClosestHybrid` only if it passes
every gate and has a material additional preprocessing speedup over
`OptimizedIntersection` on the rotated/identity torus, hollow cylinder, and
L-prism cases. Otherwise recommend `OptimizedIntersection` if it passes all
gates; if either policy has any mismatch, label it experimental and retain
`FullIntersectionReference` as default.

- [ ] **Step 6: Final verification and implementation commit**

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` is silent; generated CSV/log files are untracked
only under ignored output directories; source changes are already committed
at the preceding task boundaries. Do not commit benchmark outputs.
