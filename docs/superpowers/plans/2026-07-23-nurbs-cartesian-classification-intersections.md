# NURBS Cartesian Classification Intersections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-crossing Cartesian-edge contract with a bounded, grid-scaled multi-candidate path that preserves confirmed NURBS roots and computes point-classification parity without requiring a complete general-purpose intersection solver.

**Architecture:** Keep exact rational Bezier leaves as the BVH geometry and attach reusable 4-by-4 parameter samples to each leaf. The element layer consumes several separated seeds and records stationary line-surface pairs; the surface layer canonicalizes roots with topology-aware provenance and returns an edge summary. The Cartesian domain stores every confirmed crossing while using per-component parity, or cached endpoint classification for ambiguous clusters, to decide flood barriers.

**Tech Stack:** C++17, Eigen, CGAL predicates, CMake/Visual Studio Release, native_nurbs_surface_3d_test.

## Global Constraints

- Work directly on main and preserve unrelated changes.
- Query-leaf maximum extent is exactly 2 * max(h).
- Precompute 16 samples per leaf and retain at most four separated sample seeds per edge/leaf.
- Main local depth remains 4; difficult fallback work is bounded by depth 6.
- Geometry and deduplication tolerances do not scale with h.
- Multiple crossings and tangential contacts are records, not fatal errors.
- Surface/edge overlap and a Cartesian node on the surface remain fatal.
- Do not infer tangency order or silently select one root from a multi-root edge.
- Every task follows RED, GREEN, regression, commit, and a review checkpoint.

---

## File Map

- src/geometry/nurbs_bezier_intersection_3d.hpp/.cpp: supplied seeds, root provenance, bounded difficult boxes.
- src/geometry/nurbs_bezier_segment_closest_point_3d.hpp/.cpp: retain the existing best-local-minimum API.
- src/geometry/nurbs_bezier_segment_stationary_points_3d.hpp/.cpp: new multi-stationary-point solver.
- src/geometry/nurbs_surface_intersector_3d.hpp/.cpp: leaf samples, seed selection, root clustering, edge summary.
- src/geometry/nurbs_cartesian_domain_3d.hpp/.cpp: crossing ranges, interface edges, parity barriers, endpoint cache.
- src/CMakeLists.txt: compile the new source.
- apps/native_nurbs_surface_3d_test.cpp: analytic and geometry regressions.
- apps/neumann_exterior_zero_trace_3d.cpp: new diagnostics only.

---

### Task 1: Multi-crossing Cartesian edge result

**Files:**
- Modify: src/geometry/nurbs_surface_intersector_3d.hpp
- Modify: src/geometry/nurbs_surface_intersector_3d.cpp
- Test: apps/native_nurbs_surface_3d_test.cpp

**Produces:**

~~~cpp
struct NurbsCartesianEdgeIntersections3D {
    std::vector<NurbsSurfaceCrossing3D> crossings;
    std::vector<int> toggled_components;
    NurbsSurfaceIntersectionDiagnostics3D diagnostics;
    bool root_count_known = true;
    bool parity_known_from_roots = true;
    bool has_near_tangent_candidate = false;
};

NurbsCartesianEdgeIntersections3D intersect_cartesian_edge(
    const NurbsCartesianEdgeQuery3D& edge) const;
~~~

- [ ] Add analytic ruled-graph tests for two roots, three roots, quadratic contact, and zero-derivative odd crossing.
- [ ] Build and run native_nurbs_surface_3d_test; expect the old optional return type or multiple-crossing exception to fail.
- [ ] Add the result type and remove the fatal multiple-crossing/tangential branches from the Cartesian wrapper.
- [ ] Keep endpoint and overlap degeneracies fatal.
- [ ] Toggle obvious transverse roots per component; mark near-tangent/unresolved parity unknown.
- [ ] Update existing edge tests to inspect result.crossings.
- [ ] Run GREEN.
- [ ] Commit: feat: return all Cartesian NURBS crossings.

Analytic tests use S(u,v)=(u,v,f(u)), L(t)=(t,0.5,0), and:
- f=(u-0.3)(u-0.7): two roots, even same-component parity.
- f=(u-0.2)(u-0.5)(u-0.8): three roots, odd parity.
- f=(u-0.5)^2: contact record, parity fallback required.
- f=(u-0.5)^3: zero derivative but endpoint labels differ.

---

### Task 2: Grid-scaled 4-by-4 sample seeds

**Files:**
- Modify: src/geometry/nurbs_bezier_intersection_3d.hpp/.cpp
- Modify: src/geometry/nurbs_surface_intersector_3d.hpp/.cpp
- Test: apps/native_nurbs_surface_3d_test.cpp

**Produces:**

~~~cpp
struct NurbsElementParameterSeed3D {
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
};

struct NurbsQueryElementSample3D {
    double u = 0.0;
    double v = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
};
~~~

NurbsElementIntersectionOptions3D gains a vector of parameter seeds.

- [ ] Add failing tests for 16 finite samples per query leaf, at most four accepted seeds, and triangle-seed independence.
- [ ] Run RED.
- [ ] Precompute normalized coordinates {0,1/3,2/3,1} squared for every BVH leaf.
- [ ] Project the 16 points to each finite edge.
- [ ] Select discrete local extrema by distance, sort them, and reject retained seeds closer than 1/3 in normalized parameter Chebyshev distance.
- [ ] Always retain the best sample if no extremum survives; triangle seeds keep first priority.
- [ ] Run supplied seeds before midpoint recursion and deduplicate with fixed geometry/parameter tolerances.
- [ ] Assert main depth <= 4 and accepted seeds <= 4.
- [ ] Run GREEN.
- [ ] Commit: feat: seed NURBS intersections from leaf samples.

---

### Task 3: Stationary witnesses and close-root clusters

**Files:**
- Create: src/geometry/nurbs_bezier_segment_stationary_points_3d.hpp
- Create: src/geometry/nurbs_bezier_segment_stationary_points_3d.cpp
- Modify: src/CMakeLists.txt
- Modify: src/geometry/nurbs_surface_intersector_3d.hpp/.cpp
- Test: apps/native_nurbs_surface_3d_test.cpp

**Produces:**

~~~cpp
struct NurbsParameterBox2D {
    double u0 = 0.0, u1 = 0.0;
    double v0 = 0.0, v1 = 0.0;
};

struct NurbsLineSurfaceStationaryPoint3D {
    bool converged = false;
    double u = 0.0, v = 0.0, t = 0.0;
    Eigen::Vector3d surface_point = Eigen::Vector3d::Zero();
    double distance = std::numeric_limits<double>::infinity();
    double tangent_measure = std::numeric_limits<double>::infinity();
    int iterations = 0;
};
~~~

- [ ] Add a failing close-root test with roots 0.5 +/- 1e-3 and a positive-distance stationary witness at 0.5.
- [ ] Repeat at leaf extents corresponding to N=32,64,128; always require two roots.
- [ ] Add a split-boundary duplicate-root control test; require one physical root and no positive witness.
- [ ] Run RED.
- [ ] Eliminate t by segment projection and solve [(S-L).Su,(S-L).Sv]=0.
- [ ] Use projected damped Newton with a finite-difference 2-by-2 Jacobian clipped to the parameter box; do not impose descent of squared distance.
- [ ] Deduplicate stationary results by fixed parameter/physical tolerances.
- [ ] Estimate root t uncertainty from residual divided by segment length and guarded transversality.
- [ ] For overlapping same-smooth-branch root intervals, search from the parameter midpoint. A positive witness between roots prevents merging.
- [ ] If no proof exists, retain an ambiguous cluster and set root_count_known=false; absence of a witness never forces a merge.
- [ ] Run GREEN.
- [ ] Commit: feat: protect close NURBS roots with stationary witnesses.

---

### Task 4: Topology-aware G1 and non-G1 canonicalization

**Files:**
- Modify: src/geometry/nurbs_surface_intersector_3d.hpp/.cpp
- Test: apps/native_nurbs_surface_3d_test.cpp

- [ ] Add failing tests for a coincident shared-edge root on G1 and non-G1 connections.
- [ ] Add a wedge test containing two close one-sided roots with no smooth stationary witness.
- [ ] Run RED; expect FeatureEdgeAlignment or incorrect distance-only behavior.
- [ ] Merge coincident reports on any declared shared topological edge/vertex within the same component, regardless of G1.
- [ ] Retain the lowest-residual representative and mark feature_edge_contact for non-G1; never average one-sided normals.
- [ ] Never use the smooth witness rule across non-G1.
- [ ] Keep unrelated coincident patches fatal and never merge different components.
- [ ] Run GREEN.
- [ ] Commit: fix: canonicalize NURBS roots with patch topology.

---

### Task 5: Domain crossing ranges and parity fallback

**Files:**
- Modify: src/geometry/nurbs_cartesian_domain_3d.hpp/.cpp
- Test: apps/native_nurbs_surface_3d_test.cpp

**Produces:**

~~~cpp
class NurbsSurfaceCrossingRange3D {
public:
    const NurbsSurfaceCrossing3D* begin() const noexcept;
    const NurbsSurfaceCrossing3D* end() const noexcept;
    std::size_t size() const noexcept;
    const NurbsSurfaceCrossing3D& operator[](std::size_t) const;
};

bool has_interface_between(int node_a, int node_b) const;
NurbsSurfaceCrossingRange3D crossings_between(
    int node_a, int node_b) const;
~~~

The legacy crossing_between API succeeds only when exactly one crossing exists.

- [ ] Replace the under-resolved torus fail-fast test: construction succeeds, at least one edge has multiple crossings, and analytic torus node labels match.
- [ ] Add same-component even-parity and two-component membership-change tests.
- [ ] Run RED.
- [ ] Replace edge-to-single-index storage with contiguous {begin,count} ranges.
- [ ] Maintain separate interface-edge and parity-barrier bit arrays.
- [ ] Toggle sparse component bits for resolved roots; barrier iff any component bit is odd.
- [ ] When parity_known_from_roots=false, cache containing_components for both endpoint node indices and use their symmetric difference.
- [ ] Do not infer physical root count from endpoint fallback.
- [ ] Verify full component label changes, not only binary inside/outside.
- [ ] Update triangle-seed-independent domain checks to compare crossing ranges.
- [ ] Run GREEN.
- [ ] Commit: feat: label NURBS domains from crossing parity.

---

### Task 6: Diagnostics and full regression

**Files:**
- Modify: src/geometry/nurbs_bezier_intersection_3d.hpp
- Modify: src/geometry/nurbs_surface_intersector_3d.hpp
- Modify: src/geometry/nurbs_cartesian_domain_3d.hpp/.cpp
- Modify: apps/neumann_exterior_zero_trace_3d.cpp
- Modify: apps/native_nurbs_surface_3d_test.cpp

- [ ] Add checked counters for sample seeds, stationary solves/witnesses, root clusters, multi-crossing edges, component parity, non-G1 merges, high-degree fallbacks, and endpoint fallbacks.
- [ ] Assert at most four sample seeds, main depth <= 4, fallback depth <= 6, and expected analytic counters.
- [ ] Update console and CSV output without changing the PDE algorithm.
- [ ] Build all Release targets with cmake --build build --config Release --parallel 1.
- [ ] Run .\build\apps\Release\native_nurbs_surface_3d_test.exe and require its pass banner.
- [ ] Run ctest --test-dir build -C Release --output-on-failure and record if no tests are registered.
- [ ] Run git diff --check.
- [ ] Run L-prism and hollow-cylinder at N=32,64,128.
- [ ] Run the torus case that formerly failed on a two-crossing edge and require successful labels.
- [ ] Compare triangle seeds enabled/disabled, root residuals, nested common-node labels, and PDE convergence trends.
- [ ] Request final code review.
- [ ] Commit: test: verify NURBS classification intersections.
