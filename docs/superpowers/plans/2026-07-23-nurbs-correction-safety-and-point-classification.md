# NURBS Correction Safety and Point Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate point-classification sufficiency from KFBI correction safety, certify difficult label-changing edges with bounded extra work, and directly test analytic NURBS point containment.

**Architecture:** The surface intersector continues to return confirmed roots and ambiguous clusters. The Cartesian domain persists a confidence record for every queried edge, performs a depth-six retry only for ambiguous membership-changing edges, and exposes a strict correction-only lookup. `GridPair3D` consumes only that strict lookup. Direct containment tests exercise ray parity without the Cartesian flood-fill layer.

**Tech Stack:** C++17, Eigen, CGAL predicates, CMake/Visual Studio Release, native NURBS 3D app tests.

## Global Constraints

- Work directly on `main`; preserve unrelated changes.
- Primary local intersection depth is four; explicit targeted retry depth is six.
- Terminal certificate depth remains six.
- Geometry and deduplication tolerances remain independent of `h`.
- Endpoint component membership may decide flood parity but never, by itself, authorizes a KFBI correction crossing.
- A label-changing edge is correction-safe only when it has one certified transverse root and no ambiguity.
- Multiple roots and ambiguous clusters remain inspectable records.
- L-prism error orders and iteration counts are not acceptance gates in this plan.
- Do not run `N=256`.

---

### Task 1: Direct analytic point-containment regression

**Files:**
- Modify: `apps/native_nurbs_surface_3d_test.cpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.cpp`

**Interfaces:**
- Consumes: `NurbsSurfaceIntersector3D::containing_components(point)`.
- Produces: a geometry-independent regression table for torus, hollow-cylinder, L-prism, and two translated components.

- [ ] **Step 1: Add the analytic point table**

Add `test_direct_nurbs_point_classification()` and call it before Cartesian
domain tests. Use robust points at least `1e-3` from the analytic boundary.
For every fixture compare the returned component vector with the expected
vector:

```cpp
struct PointClassificationCase {
    Eigen::Vector3d point;
    std::vector<int> components;
    const char* description;
};

for (const auto& item : cases) {
    require(intersector.containing_components(item.point) == item.components,
            item.description);
}
```

Cover:

- torus tube material, center hole, far exterior, and an exterior point whose
  positive oblique ray crosses the torus twice;
- hollow-cylinder wall, bore, radial exterior, above-cap exterior, and
  points immediately inside/outside a cap;
- both L-prism arms, missing quadrant, outside height, and four fixed-offset
  points around the reentrant corner;
- original and translated solid components plus a point between them;
- identical answers with `use_triangle_seeds=true/false`.

Add an exact surface point and require the final message to contain:

```text
point lies on NURBS surface
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 1
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

The boundary-point assertion must fail because the current three-ray retry
replaces endpoint contact by the generic `unable to classify` message.

- [ ] **Step 3: Preserve an endpoint-contact verdict across ray retries**

In `containing_components`, distinguish:

- query point at ray parameter zero: immediately throw
  `point lies on NURBS surface`;
- degeneracy at the remote ray endpoint, tangency, overlap, or unresolved
  candidate: retry another direction.

Do not change the parity calculation for successful rays.

- [ ] **Step 4: Run GREEN and commit**

Run the focused executable and `git diff --check`, then commit:

```text
test: cover direct NURBS point classification
```

---

### Task 2: Persist per-edge confidence and strict correction lookup

**Files:**
- Modify: `src/geometry/nurbs_cartesian_domain_3d.hpp`
- Modify: `src/geometry/nurbs_cartesian_domain_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Produces:

```cpp
struct NurbsCartesianEdgeClassification3D {
    bool queried = false;
    bool has_confirmed_interface = false;
    bool changes_component_membership = false;
    bool root_count_known = true;
    bool parity_known_from_roots = true;
    bool has_near_tangent_candidate = false;
    bool used_targeted_retry = false;
    bool correction_safe = false;
    std::size_t confirmed_crossing_count = 0;
    std::size_t ambiguous_cluster_count = 0;
    int confirmed_transverse_count = 0;
};

NurbsCartesianEdgeClassification3D edge_classification_between(
    int node_a, int node_b) const;

const NurbsSurfaceCrossing3D& correction_crossing_between(
    int node_a, int node_b) const;
```

- [ ] **Step 1: Add failing domain API tests**

Extend the existing multi-component coarse-edge test:

```cpp
const auto info =
    coarse_domain.edge_classification_between(first_node, second_node);
require(info.queried
            && info.changes_component_membership
            && info.confirmed_crossing_count == 4
            && !info.correction_safe,
        "multi-crossing component change is not correction-safe");
require_throws_contains(
    [&] {
        (void)coarse_domain.correction_crossing_between(
            first_node, second_node);
    },
    "under-resolved NURBS Cartesian edge",
    "multi-crossing correction lookup fails explicitly");
```

For an ordinary L-prism label-changing edge require `correction_safe`, one
confirmed root, and reverse-order lookup identity.

- [ ] **Step 2: Verify RED**

Build the native test. Expected: compilation fails because the new record and
methods are absent.

- [ ] **Step 3: Replace the range-only map by an edge record**

Store one internal record for every queried candidate key:

```cpp
struct EdgeIntersectionRecord3D {
    EdgeCrossingRange3D range;
    NurbsCartesianEdgeClassification3D classification;
};
```

The range is valid even when its count is zero. Populate all confidence fields
from `NurbsCartesianEdgeIntersections3D` plus endpoint membership. Compute
`correction_safe` from the exact conjunction in the design document.

`edge_classification_between` returns `queried=false` for a noncandidate
neighbor edge. `correction_crossing_between` requires `correction_safe` and
otherwise formats the edge axis, node coordinates, flags, counts, and every
confirmed root.

- [ ] **Step 4: Run GREEN and commit**

Run the native test and `git diff --check`, then commit:

```text
feat: expose NURBS correction-safe edge records
```

---

### Task 3: Bound public depth and extend smooth-branch witnesses

**Files:**
- Modify: `src/geometry/nurbs_bezier_intersection_3d.hpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.hpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Changes the default element and surface local depth from 36 to 4.
- Extends close-root proof across same-patch element splits and declared G1
  connections; no public witness API is added.

- [ ] **Step 1: Add failing bounded-default and split-root tests**

Require:

```cpp
require(NurbsElementIntersectionOptions3D{}.max_subdivision_depth == 4,
        "element intersection default depth is bounded");
require(NurbsSurfaceIntersectorOptions3D{}.local_max_subdivision_depth == 4,
        "surface intersection default depth is bounded");
```

Construct the close quadratic ruled graph with query leaves split at
`u=0.5`, so the roots `0.5 +/- 1e-3` lie in adjacent elements. Require two
confirmed roots, no ambiguous cluster, and one protected root-pair
diagnostic. Repeat with the graph represented by two G1-connected patches.
Change the connection to non-G1 and require that smooth witness protection is
not used.

- [ ] **Step 2: Verify RED**

Run the focused test. The default-depth assertions and cross-element/G1
protected-pair assertions must fail.

- [ ] **Step 3: Implement smooth-branch candidate collection**

Replace the single-element condition in close-root analysis with:

1. same component;
2. same patch or connectivity through declared G1 connections;
3. candidate Bezier elements containing either root or incident to the
   matching G1 edge.

For each candidate element use seeds formed from the available root
parameters, the element midpoint, and the edge-parameter midpoint projected
into the parameter box. Accept a positive witness only when its line
parameter lies strictly between the roots, its distance exceeds
`8*geometry_tolerance`, and its tangent measure is below `1e-6`.

Do not traverse non-G1 connections. Preserve fixed-tolerance stationary
deduplication.

- [ ] **Step 4: Set bounded defaults**

Set both public default depths to four. Keep explicit targeted depth six and
terminal certificate depth six unchanged.

- [ ] **Step 5: Run GREEN and commit**

Run the focused test and `git diff --check`, then commit:

```text
feat: certify close roots across smooth NURBS elements
```

---

### Task 4: Retry ambiguous membership-changing edges to depth six

**Files:**
- Modify: `src/geometry/nurbs_cartesian_domain_3d.hpp`
- Modify: `src/geometry/nurbs_cartesian_domain_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Adds diagnostics:

```cpp
std::size_t ambiguous_label_changing_edge_count = 0;
std::size_t targeted_retry_count = 0;
std::size_t targeted_retry_resolved_count = 0;
std::size_t targeted_retry_unsafe_count = 0;
std::size_t correction_safe_edge_count = 0;
std::size_t unsafe_label_changing_edge_count = 0;
```

- [ ] **Step 1: Add a failing retry regression**

Build the production torus domain on the `[-1.5,1.5]^3`, `N=128` node grid.
This is the existing case that reports five depth-four endpoint fallbacks.
Require at least one targeted retry, zero unsafe label-changing edges, and
for every barrier require that the domain record reports:

```cpp
correction_safe == true
```

For records with `used_targeted_retry=true`, additionally require
`root_count_known` and `parity_known_from_roots`. Retain the existing
four-crossing two-component coarse fixture and require it to remain unsafe
without being collapsed to one root.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm that no retry diagnostics or retry state
exist.

- [ ] **Step 3: Add the bounded retry intersector**

Construct the normal depth-four intersector as today. Lazily construct a
second intersector over the same model and leaf extent with local depth six
only after the first ambiguous membership-changing edge is found.

For an unknown-parity edge:

1. classify and cache both endpoint component vectors;
2. if membership differs, increment the ambiguous-label-change counter;
3. query the depth-six intersector;
4. use the depth-six result as the authoritative stored result;
5. recompute endpoint fallback only if the retry still has unknown parity.

Count resolved and unsafe retries with checked arithmetic. Do not retry an
ambiguous edge whose endpoint membership is unchanged.

- [ ] **Step 4: Run GREEN and commit**

Run the focused test and `git diff --check`, then commit:

```text
feat: retry difficult NURBS barrier intersections
```

---

### Task 5: Enforce correction safety in GridPair3D

**Files:**
- Modify: `src/geometry/grid_pair_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: `NurbsCartesianDomain3D::correction_crossing_between`.
- Preserves: the legacy triangle route when no native domain is present.

- [ ] **Step 1: Add failing GridPair unsafe-edge tests**

Construct a native-domain `GridPair3D` around the existing coarse
multi-component fixture. Require construction to fail with:

```text
under-resolved NURBS Cartesian edge
```

For L-prism, require every binary label-changing edge to have
`correction_safe=true` and every P2 owner to remain an exact native owner.

- [ ] **Step 2: Verify RED**

The coarse multi-component GridPair currently catches the domain error and
reports only `label-changing edge has no native NURBS crossing`; require the
new detailed assertion to fail.

- [ ] **Step 3: Replace permissive native lookup**

Delete the catch-all wrapper around native crossing lookup. Call
`correction_crossing_between` directly in both the constructor invariant and
`p2_crossing_owner_between`. Preserve the domain's detailed exception.

- [ ] **Step 4: Run GREEN and commit**

Run the focused test and `git diff --check`, then commit:

```text
fix: reject unsafe NURBS corrections in GridPair3D
```

---

### Task 6: Diagnostics, CI, and full non-accuracy acceptance

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `.github/workflows/build-and-smoke.yml`
- Modify: `docs/superpowers/results/2026-07-23-nurbs-correction-safety-and-point-classification.md`

**Interfaces:**
- Extends console and `geometry_readiness.csv` with the six retry/safety
  counters from Task 4.

- [ ] **Step 1: Add failing output assertions**

Require the stable readiness output to include:

```text
correction_safe_edges=
unsafe_label_changing_edges=0
targeted_retries=
```

Update CI grep assertions for `unsafe_label_changing_edges=0`.

- [ ] **Step 2: Wire checked diagnostics**

Populate `ReadinessResult`, console output, per-level readiness CSV, and
combined CSV directly from `NurbsCartesianDomainDiagnostics3D`. Do not
recompute edge queries in the app.

- [ ] **Step 3: Run focused and full verification**

Run:

```powershell
cmake --build build --config Release --parallel 1
.\build\apps\Release\native_nurbs_surface_3d_test.exe
ctest --test-dir build -C Release --output-on-failure
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 32 64 128
git diff --check
git status --short --branch
```

For all nine geometry levels require:

- `label_mismatches == 0`;
- `gap_crossings == 0`;
- `triangle_fallback_crossings == 0`;
- `unsafe_label_changing_edges == 0`;
- both formulation `converged == 1`;
- root residual within geometry tolerance;
- primary depth at most four and targeted depth at most six.

Do not enforce L-prism error orders or iteration counts.

- [ ] **Step 4: Record and commit results**

The result document reports direct point cases, retry counts, safety counts,
root residuals, solver convergence, and explicitly repeats that L-prism
accuracy is deferred. Commit:

```text
test: verify NURBS correction safety and containment
```
