# NURBS Correction Safety and Point Classification Design

## Goal

Finish the native-NURBS geometry route without treating an edge that is only
good enough for endpoint classification as automatically safe for a KFBI
interface correction. Add direct, analytic tests of point containment that do
not depend on Cartesian flood fill.

## Scope

This work includes:

- a persistent per-Cartesian-edge confidence record;
- bounded targeted re-intersection of ambiguous label-changing edges;
- stationary-witness searches across adjacent Bezier elements and declared G1
  patch connections;
- a strict correction-crossing API consumed by `GridPair3D`;
- dedicated analytic tests of `containing_components(point)`;
- diagnostics, CI smoke coverage, and `N=32,64,128` geometry/solver regression.

This work explicitly excludes:

- changing the Cauchy polynomial, normal restriction, or GMRES algorithms;
- requiring second-order L-prism Dirichlet-normal convergence;
- imposing an iteration-count gate on the L-prism;
- running `N=256`.

The deferred L-prism accuracy/stability work remains open and must not be
reported as completed by this implementation.

## Alternatives Considered

### Immediate fail-fast

Reject every ambiguous label-changing edge after the current depth-four
query. This is safe and simple, but it would reject difficult edges that a
small amount of local extra work can certify.

### Endpoint parity as correction authority

Continue using endpoint component membership to choose a barrier and accept a
single confirmed root even when other candidates remain unresolved. This is
adequate for flood labeling but does not prove that the root is the unique
physical interface crossing needed by the KFBI correction. This option is
rejected.

### Selected: targeted certification, then fail-safe rejection

Use depth four for every normal edge query. When endpoint component
classification says an ambiguous edge changes membership, recompute only that
edge with local depth six. A root may enter KFBI only when the final result
proves a unique transverse crossing and known parity. Remaining ambiguity or
multiple crossings produces an explicit under-resolution error at the
`GridPair3D` boundary.

## Edge Confidence Contract

`NurbsCartesianDomain3D` will expose a value record:

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
```

The domain API gains:

```cpp
NurbsCartesianEdgeClassification3D edge_classification_between(
    int node_a, int node_b) const;

const NurbsSurfaceCrossing3D& correction_crossing_between(
    int node_a, int node_b) const;
```

`correction_safe` is true only when all of the following hold:

- the edge changes component membership;
- `root_count_known` is true;
- `parity_known_from_roots` is true;
- there is exactly one confirmed crossing;
- there is exactly one confirmed transverse crossing;
- there is no near-tangent candidate or ambiguous cluster.

The ordinary `crossings_between` range remains available for geometry
inspection. The legacy `crossing_between` keeps its exact-one-record
semantics, but it is not an authority for KFBI correction.

The edge record is stored for every queried candidate edge, including edges
with only ambiguous candidates and no confirmed crossing. Reversing the node
order returns the same record and range.

## Bounded Targeted Re-intersection

The production Cartesian-domain fast path uses:

- query-leaf extent at most `2*max(h)`;
- local subdivision depth four;
- existing terminal certification depth six;
- at most four retained 4-by-4 sample seeds.

For an edge whose root parity is unknown:

1. Cache and compare the component-membership vectors at both endpoints.
2. If membership is unchanged, retain endpoint parity for flood labeling and
   do not perform correction-oriented extra work.
3. If membership changes, rerun only that edge with local depth six.
4. Replace the primary result by the deeper result when constructing the
   stored edge record.
5. If the deeper result remains ambiguous, allow domain labeling to finish
   from endpoint membership but mark the edge unsafe for correction.

The public element and surface-intersector default local depth becomes four.
Depth six must be requested explicitly by the targeted retry path. No code
path silently falls back to the legacy depth of 36.

Diagnostics record:

- ambiguous label-changing edges seen by the fast path;
- targeted edge retries;
- retries resolved to a correction-safe edge;
- retries still unsafe;
- correction-safe label-changing edges;
- unsafe label-changing edges.

## Smooth-Branch Stationary Witnesses

The existing close-root witness is extended beyond a single element.

Two roots may share a smooth branch when:

- they belong to the same solid component; and
- they are on the same patch or their patches are connected through declared
  G1 topology.

For a close root pair, collect Bezier elements containing either root and the
elements incident to the connecting G1 edge. Run bounded projected stationary
solves from:

- both root parameters;
- their edge-parameter midpoint projected into each candidate element;
- each candidate element's parameter midpoint.

A witness protects two roots only when it converges, lies strictly between
their edge parameters, has positive physical distance greater than the
geometry tolerance, and satisfies the tangent-measure threshold. Duplicate
witnesses use fixed parameter and physical tolerances.

Non-G1 connections never participate in this smooth-branch proof. Failure to
find a witness leaves an ambiguous cluster; it never forces a merge.

## GridPair3D Integration

The native route in `GridPair3D` uses
`correction_crossing_between(node,neighbor)` exclusively.

It no longer catches and replaces all domain exceptions with a generic
message. An unsafe edge error includes:

- axis and structured edge coordinates;
- endpoint component labels;
- confirmed and ambiguous counts;
- root-count and parity flags;
- every confirmed root's component, edge parameter, residual, and
  transversality;
- whether targeted retry was attempted.

A same-component even multi-crossing edge remains a geometric interface edge
without being a flood barrier. A label-changing multi-crossing edge is
classified correctly by the domain but rejected as under-resolved for the
current single-interface KFBI correction.

## Dedicated Point-Containment Tests

Tests call `NurbsSurfaceIntersector3D::containing_components` directly; they
do not construct `NurbsCartesianDomain3D` or use flood labels.

The analytic table covers:

- torus: tube interior, central hole, far exterior, and a point whose
  classification ray crosses the torus an even number of times;
- hollow cylinder: material wall, central bore, outside radius, above a cap,
  and points just inside/outside a cap;
- L-prism: both arms, missing quadrant, exterior height, and points in each
  sector near the reentrant corner while remaining a fixed positive distance
  from the boundary;
- two disjoint translated components: inside each component and between them;
- classification with triangle seeds enabled and disabled;
- points exactly on a surface, which must fail explicitly rather than receive
  an arbitrary inside/outside label.

Every expected component vector is derived from the geometry's analytic
inside predicate or the known translated fixture. Test offsets are many
orders of magnitude larger than the geometry tolerance so they do not test
floating-point coincidence accidentally.

## Regression and Acceptance

The implementation is accepted when:

- all focused native NURBS tests pass;
- direct point-containment cases return their exact component vectors;
- Release builds complete;
- torus, hollow cylinder, and L-prism complete at `N=32,64,128`;
- every level reports zero analytic node-label mismatches;
- every KFBI crossing owner is native and exact;
- gap and triangle-fallback crossing counts are zero;
- every label-changing edge consumed by `GridPair3D` is correction-safe;
- both existing GMRES formulations report convergence;
- primary local depth is at most four and targeted retry depth is at most six;
- `git diff --check` passes.

No error-order or iteration-count threshold is applied to the L-prism in this
acceptance run.
