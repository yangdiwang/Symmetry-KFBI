# 3D Nonconvex Domain-Labeling Design

## Goal

Replace the active-P2 nearest-center-normal domain label with a topology-aware
labeling route that is correct for nonconvex, multi-patch geometry, including
the L-prism reentrant corner at `N=128` and finer grids.

The labeling route must use the complete geometric surface, remain independent
of the correction/collocation edge buffers, and produce labels consistent with
the same triangle geometry used for Cartesian-edge crossings.

## Confirmed failure

The current active-P2 path labels a Cartesian node by the sign of

```text
(node - nearest expansion center) dot nearest normal.
```

At `N=128`, the L-prism grid node
`(0.0703125,-0.0703125,-0.5390625)` lies inside the lower arm but is assigned
to the vertical reentrant face at `x=0.07`. That face has normal `(1,0,0)` and
gives signed distance `+3.125e-4`, so the node is incorrectly labeled outside.
The complete unbuffered surface chooses the same face and gives the same sign;
therefore replacing correction centers with geometry centers is insufficient.

The complete triangle soup is geometrically watertight after coincident
feature-edge vertices are identified, and ray parity classifies the node as
inside. The `N=128` run has 15 wrong labels and 74 gap-fallback crossings;
these gap crossings are consequences of label-changing Cartesian edges that
do not intersect the actual surface.

## Selected architecture

Use a dedicated, watertight labeling mesh to cut the Cartesian grid graph,
then classify graph components globally:

1. Build a welded closed triangle mesh from `crossing_geometry`.
2. Find every Cartesian grid edge intersected by that mesh and mark it as a
   barrier.
3. Compute connected components of Cartesian nodes using only non-barrier
   edges.
4. Mark every component touching the embedding-box boundary as exterior.
5. Classify one representative of every remaining component with a robust
   global closed-mesh side/parity query, then assign that result to the entire
   component.

This route uses local triangle intersections only to construct barriers. The
inside/outside decision is global and never comes from one face normal.

## Alternatives considered

### Nearest normal with a larger tolerance

Rejected. The failing signed distance is `3.125e-4`, not roundoff. A tolerance
large enough to hide it would absorb genuine exterior points and remain
dependent on grid alignment.

### Complete-surface nearest normal

Rejected. Direct diagnostics show that the complete surface still selects the
wrong reentrant face. The defect is the single-half-space model, not only the
feature-edge trimming.

### Ray parity or winding at every Cartesian node

Correct but not selected as the primary route. It would perform a global mesh
query for every node, including roughly 17 million nodes at `N=256`. A global
query per connected component gives the same topological information with
lower cost.

## Labeling mesh

Create a geometry-only mesh separate from the correction interface and from
native NURBS surface DOFs.

- Input: the complete, unbuffered `crossing_geometry` triangle surface.
- Within each interface component, weld vertices whose Euclidean positions
  agree within a scale-relative tolerance independent of `h`. Do not weld
  merely because two vertices are close in units of the current grid spacing;
  reject coincident faces belonging to distinct components as ambiguous input.
- Preserve a face-to-original-geometry-panel map so crossing ownership and
  NURBS parameter recovery remain unchanged.
- Reject degenerate faces.
- Validate that every welded undirected edge belongs to exactly two faces and
  that the two face orientations oppose each other along the edge.
- Build a CGAL AABB tree on the validated mesh.

The labeling mesh may weld a non-G1 feature edge topologically. This does not
smooth its normals and does not change Cauchy patch adjacency; it only closes
the mesh used for topology queries.

## Cartesian edge barriers

Store one barrier flag for each x-, y-, and z-directed grid edge.

Avoid querying all `O(N^3)` grid edges against the AABB tree. For each surface
triangle:

1. Convert its physical bounding box to a clamped Cartesian index range,
   enlarged only by the geometric intersection tolerance.
2. Enumerate x-, y-, and z-directed grid edges in that small range.
3. Use exact-predicate segment/triangle intersection to mark the corresponding
   barrier flags.

The triangle edge length is proportional to `h`, so each triangle touches a
bounded number of Cartesian cells. Barrier construction therefore scales
primarily with surface resolution, approximately `O(h^-2)`, instead of
testing every volume edge with an AABB query.

Repeated hits from adjacent triangles are harmless. A grid edge is a barrier
when its closed segment has any proper intersection with the surface.

### Degenerate grid alignment

KFBI crossing formulas require the surface not to pass exactly through a
Cartesian node or overlap a Cartesian edge. Detect either case explicitly and
stop with a diagnostic containing the node/edge and geometry panel. Do not
silently assign a side with a tolerance. The current shifted benchmark
geometries satisfy this non-alignment requirement.

## Component labeling

Run breadth-first search over Cartesian nodes. Traversal is allowed only over
non-barrier neighbor edges.

- Components touching any box-boundary node are exterior.
- For each enclosed component, classify one representative point with a
  robust side-of-closed-mesh or ray-parity query.
- Assign the result to all nodes in that component.
- Preserve the existing `0 = exterior`, `component + 1 = interior` contract.
  For multiple surface components, perform the representative query against
  the validated closed mesh belonging to each interface component.
- Reject overlapping domain components whose Boolean ownership would be
  ambiguous under the current API.

The component query is also a consistency check: an enclosed grid component
classified as exterior can occur for cavities or nested shells and must not be
blindly relabeled as material.

## Integration

Add an internal geometry-label builder used by `GridPair3D`.

- `correction_interface` continues to provide P2 expansion centers for
  spread, restrict, projections, and local correction ownership.
- `crossing_geometry` exclusively supplies the global labeling/crossing mesh.
- Remove `build_p2_nearest_center_domain_labels` from the active labeling path;
  it may remain only as a diagnostic comparison until the new tests pass.
- Reuse the welded labeling mesh AABB tree for Cartesian-edge crossing queries
  where possible.
- Keep the public `GridPair3D::domain_label()` interface unchanged.
- Do not change Cauchy stencils, native NURBS DOFs, GMRES tolerances, spread,
  or restrict in this work.

After labels are built, verify every Cartesian neighbor pair with different
labels has a real mesh intersection. The production result must have zero
gap-fallback crossings.

## Diagnostics

Report at setup:

- input and welded vertex/face counts;
- mesh boundary/nonmanifold/orientation error counts;
- x/y/z barrier-edge counts;
- number and sizes of Cartesian connected components;
- number of box-connected exterior components;
- number of global representative queries;
- exact-on-node or edge-overlap degeneracies;
- label mismatches in manufactured geometries;
- exact/gap/endpoint crossing-owner counts.

## Validation

### Focused geometry tests

1. L-prism reentrant node
   `(0.0703125,-0.0703125,-0.5390625)` is inside.
2. A point in the missing upper-right quadrant is outside.
3. All complete meshes become watertight and consistently oriented after
   welding.
4. Torus, hollow cylinder, and L-prism labels agree with their exact predicates
   on targeted grids.
5. A deliberately surface-aligned grid node produces the required explicit
   diagnostic.
6. Multiple disjoint closed components retain distinct positive labels.

### Numerical acceptance

Run the L-prism Dirichlet normal-jump route with cubic `same_patch 32/20` at
`N=32,64,128`.

- label mismatches: `0` at every level;
- gap-fallback crossings: `0` at every level;
- Dirichlet GMRES converges at every level;
- density and interior `Linf` errors decrease on both refinements;
- `N=64` to `N=128` density and interior orders are at least two;
- constant-jump, route-mismatch, torus, cylinder, and native-NURBS regressions
  remain within their existing tolerances.

Only after `N=128` satisfies these gates should `N=256` be run as an optional
fine-grid verification.

## Out of scope

- changing the Cauchy sample-count recommendation;
- changing the L-prism manufactured solution;
- changing correction-surface edge buffers;
- defining arbitrary Boolean operations between overlapping solid components;
- running `N=256` before the `N=128` geometry gates pass.
