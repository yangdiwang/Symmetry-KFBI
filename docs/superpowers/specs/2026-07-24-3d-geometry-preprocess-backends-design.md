# 3D KFBI Geometry-Preprocess Backends Design

## Goal

Provide two optional accelerated implementations of the native-NURBS
Cartesian geometry preprocessing used by KFBI, while retaining the current
implementation as an unchanged reference backend:

1. `certified_baseline`: current BVH-driven certified intersection path.
2. `optimized_intersection`: the same certified mathematics with redundant
   seed work and repeated candidate discovery removed.
3. `hybrid`: affine-planar analytic intersection, smooth-element closest-point
   preclassification, and automatic fallback to `optimized_intersection`.

The experiment must compare both speed and accuracy on the actual data
consumed by KFBI: grid labels, barriers, interface crossings, strict
correction crossings, and `GridPair3D` crossing owners.

## Non-goals

- Do not change the PDE discretization, Cauchy stencils, restrict/spread
  formulas, GMRES tolerances, or solvers.
- Do not use analytic `exact_inside` predicates in production preprocessing.
  They are test oracles only.
- Do not accept a closest-point sign as a global inside/outside decision
  without a conservative certificate.
- Do not make performance thresholds part of normal CI. CI checks
  deterministic correctness and records timing only.

## Public selection

Add the following strategy to `NurbsCartesianDomainOptions3D`:

```cpp
enum class NurbsCartesianPreprocessStrategy3D {
    CertifiedBaseline,
    OptimizedIntersection,
    Hybrid
};

struct NurbsCartesianDomainOptions3D {
    bool use_triangle_seeds = true;
    NurbsCartesianPreprocessStrategy3D strategy =
        NurbsCartesianPreprocessStrategy3D::CertifiedBaseline;
};
```

The default remains `CertifiedBaseline` so existing applications retain
their current behavior unless a backend is explicitly selected.

## Shared invariants

All backends must:

- enumerate the same conservative set of Cartesian candidate edges;
- process every candidate surface element for an edge;
- preserve component-aware root parity;
- preserve same-patch, G1-seam, and non-G1 topology canonicalization;
- retain multi-root, tangent, overlap, endpoint-contact, and unresolved
  fail-closed behavior;
- expose the same immutable `crossings_between`,
  `edge_classification_between`, and `correction_crossing_between` API;
- require exactly one reliable transverse root before setting
  `correction_safe`;
- produce the same node labels and barrier/interface edge sets as the
  baseline.

## Backend 1: optimized certified intersection

### Stable candidate mapping

The Cartesian-domain broad phase already rasterizes every query-element AABB
onto candidate grid edges, but currently discards the element identity.
Expose stable query-element descriptors from `NurbsSurfaceIntersector3D` and
retain sorted unique `(edge_key, element_id)` incidences. The optimized and
hybrid backends pass those element IDs directly to the intersector, avoiding
a second BVH traversal per Cartesian edge.

The original single-argument `intersect_cartesian_edge` remains the baseline
BVH path.

### Early unique-root certificate

After triangle-seed Newton has produced exactly one verified element root,
apply the existing outward-rounded whole-element
`certifies_unique_transverse_root` certificate. If it succeeds, return the
element result before the 4x4 supplied-sample seeds and subdivision loop.

Certificate failure, zero roots, multiple roots, tangency, overlap, or
non-finite data must continue through the existing full path.

The certificate is element-local. Finding one certified element root must
never stop processing other candidate elements for the same Cartesian edge.

### Targeted retry

Depth-six targeted retry must use the same stable candidate IDs and the same
intersector geometry. A per-query subdivision-depth override avoids relying
on accidental ID equality between two separately constructed intersectors.

## Backend 2: hybrid geometry routing

Each candidate query element is assigned one of three routes:

### Affine-planar analytic route

The route is available only when:

- degrees are `(1, 1)`;
- all homogeneous weights are positive and equal within the geometry
  tolerance;
- the dehomogenized controls satisfy the affine residual
  `p11 - p10 - p01 + p00 == 0` within a scale-aware tolerance;
- the two parametric tangents have a nondegenerate cross product.

For a robustly nonparallel segment, solve line-plane intersection and the
two affine parameters directly. A clear miss is authoritative. A hit is
authoritative only when `t`, `(u,v)`, reconstructed point, residual,
orientation, and transversality pass the same tolerances as a certified
root. Roots close to native patch boundaries, Cartesian endpoints, parallel
or coplanar cases, and all uncertain cases fall back.

### Smooth closest-point route

For a query element that does not touch a declared non-G1 feature, run the
existing bounded element-to-segment closest-point solver before the ordinary
seed/subdivision path.

- If the closest point produces a root and the whole-element unique
  transverse-root certificate succeeds, accept it.
- If its separation vector passes the existing conservative terminal
  separation certificate over the whole element, accept an empty result.
- Otherwise retain any safe seed information and fall back to the optimized
  certified intersection.

Closest-point convergence alone is never authoritative. Endpoint signs
alone never rule out even root counts.

### Certified fallback

All other elements use `optimized_intersection`. Canonicalization and parity
analysis remain shared after all element routes finish.

## Diagnostics and timing

Extend intersection diagnostics with:

- mapped versus BVH candidate-element counts;
- maximum candidate elements on one edge;
- early unique-certificate attempts and successes;
- planar analytic hit, miss, and fallback counts;
- closest-point prefilter attempts, certified hits, certified misses, and
  fallbacks;
- certified fallback element count.

Extend Cartesian-domain diagnostics with `steady_clock` phase timings:

- intersector build;
- candidate enumeration/deduplication;
- edge intersection;
- edge-record/barrier materialization;
- flood labeling;
- representative component classification;
- invariant verification;
- total construction.

Timing fields are observational and must not alter control flow.

## Accuracy comparison

For each geometry/grid/backend:

1. Compare every node label with `certified_baseline`.
2. Compare every node label with the geometry's analytic `exact_inside`
   predicate.
3. Compare every structured edge's barrier flag, interface flag,
   root/parity-known flags, membership-change flag, correction-safe flag,
   crossing count, component, and root geometry.
4. Pair baseline and candidate crossings by canonical order and require:
   - point distance no greater than the scale-aware root tolerance;
   - edge-parameter difference no greater than the scale-aware parameter
     tolerance;
   - normal agreement for non-feature roots;
   - residual within the native geometry tolerance.
5. Build `GridPair3D` with the native domain and materialize every
   label-changing edge owner. Every such owner must report
   `ExactIntersection`.

Any mismatch is a test failure and is written with geometry, grid, axis,
indices, and both backend records.

## Performance experiment

Use the three production geometries:

- L-prism: affine planar faces, concavity, non-G1 edges and corners;
- hollow cylinder: curved walls, caps, multiple sheets and sharp rims;
- torus: curved periodic/G1 patches, nonconvexity and possible multi-roots.

Primary grid levels are `N=32,64,128`. Run one untimed warmup and three timed
repetitions per cell in Release mode, rotating backend order between
repetitions. Accuracy comparison is outside timed intervals. Report median,
minimum, maximum, and coefficient of variation.

Report separately:

- native domain construction time;
- `GridPair3D` construction/correction-owner validation time;
- combined KFBI geometry-preprocess time;
- internal phase timings and route/work counters.

The benchmark writes a raw per-run CSV and an aggregate CSV. It returns
nonzero on any accuracy mismatch. Timing is not a correctness gate.

## Acceptance

- Existing `native_nurbs_surface_3d_test` remains green with the default
  backend.
- Synthetic 0/1/2/3-root, tangent, overlap, close-root, G1, non-G1,
  periodic, and multi-component regressions remain green.
- For all three production geometries at `N=32,64,128`, both accelerated
  backends have:
  - zero baseline label mismatches;
  - zero analytic label mismatches;
  - zero barrier/interface/classification mismatches;
  - zero correction crossing/owner mismatches;
  - zero unsafe label-changing edges.
- The results document reports measured speedups without converting them
  into a brittle pass/fail threshold.
