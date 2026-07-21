# 3D NURBS-Native Domain Labeling and Crossing Design

## Status

This design supersedes the closed-triangle-mesh labeling design. Native NURBS
patches are the geometric source of truth. Triangles may accelerate candidate
ordering and provide nonlinear-solver seeds, but they cannot accept, reject,
or define a final intersection.

## Goal

Provide correct Cartesian-grid domain labels and interface crossings for
nonconvex, multi-patch 3D geometries without requiring the patch
triangulations to be conforming or watertight.

The route must:

- remove the nearest-face-normal inside/outside decision;
- evaluate final intersections on the native NURBS surface;
- return native `(patch, u, v)` parameters directly;
- use the same NURBS intersections for graph barriers and KFBI crossing data;
- preserve the current parameter-uniform surface DOFs, P2 correction surface,
  Cauchy fitting, spread, restrict, and GMRES formulations.

## Confirmed reasons for the redesign

The nearest-expansion-center normal is not a valid global side test for a
nonconvex surface. At `N=128`, an L-prism node inside the lower arm is nearest
to the reentrant vertical face and is incorrectly marked exterior.

A second issue appears if a closed triangle mesh is made the replacement
geometric truth. Under the intended Task 1 settings, coordinate welding leaves
`144` boundary edges on the torus and `48` on the hollow cylinder. These are
real nonconforming patch traces and intra-patch hanging nodes, not roundoff.
The L-prism happens to close after welding, but that does not make triangle
closure a valid general requirement.

Repairing the triangulation would address its topology but would retain chord
geometry, triangle-based parameters, and resolution-dependent geometric
errors. The native NURBS representation already contains the continuous
geometry needed by the algorithm.

## Scope boundary

### Geometric truth

The following operations use native NURBS patches and their topology:

- Cartesian grid-edge intersection;
- global point classification;
- patch and parameter ownership of crossings;
- crossing position and normal;
- interface-component identity.

### Numerical discretization retained

The following remain discrete and are not redesigned here:

- parameter-uniform native surface DOFs;
- P2 correction/collocation triangles and feature-edge buffers;
- local cubic Cauchy fitting;
- tricubic/cubic restrict;
- spread and bulk finite differences;
- Dirichlet/Neumann GMRES equations and tolerances.

The P2 surface is therefore a numerical correction structure, not the source
of domain geometry.

## Required geometric assumptions

1. Every NURBS weight is positive.
2. Patches have a declared outward orientation.
3. Patch-edge topology forms one or more closed surface components.
4. Topology distinguishes geometric adjacency from G1 smooth adjacency.
5. The embedding box strictly contains every surface component.
6. A Cartesian grid edge contains at most one transverse interface crossing.
7. The interface does not pass through a Cartesian node, overlap a Cartesian
   edge, or touch a grid edge tangentially.

Violations of assumptions 6 and 7 produce explicit diagnostics and stop the
case. They are not silently repaired or ignored. Supporting multiple crossings
on one grid edge would require a separate multi-interface KFBI correction
design and is out of scope.

## Core surface model

Move the reusable native surface description into the geometry library. The
application may continue to provide named benchmark factories, but the query
engine consumes a core model containing:

- `std::vector<NurbsSurfacePatch3D>`;
- a component ID and outward-orientation convention for each patch;
- explicit patch-edge interval connections;
- mapped parameter intervals and orientation reversal;
- periodic identifications;
- a G1/non-G1 classification independent of geometric adjacency.

A connection is interval-based rather than only patch-based. This is required
for the L-prism, where one long patch edge can meet multiple shorter patch
edges. Every atomic patch-boundary interval in a closed component must have
exactly one geometric mate.

Topology validation checks:

- paired interval endpoints agree in physical space;
- mapped edge samples agree on the NURBS curves to geometry tolerance;
- paired orientations oppose each other around the closed surface;
- component IDs are consistent;
- no boundary interval is missing or paired more than once.

A non-G1 feature edge is welded only topologically. Its two patch normals and
smooth groups remain distinct, so existing Cauchy-neighborhood rules do not
change.

## Conservative spatial index

Split each tensor-product NURBS patch at its nonzero knot spans and represent
the spans as rational Bézier elements. Positive rational weights guarantee
that the element lies inside the Euclidean convex hull of its control points.

For each element, store:

- patch index and `(u0,u1) x (v0,v1)`;
- rational Bézier control data;
- a conservative physical AABB;
- component identity;
- optional coarse triangles tied to the same parameter element.

Build a BVH over the conservative element AABBs.

The AABB is the authoritative broad-phase test. A coarse triangle may:

- prioritize candidates;
- estimate a starting `(u,v)`;
- provide a visual/debug representation.

A triangle miss may not remove an AABB candidate. This rule prevents a flat
chord from hiding a real intersection on the curved NURBS element. Triangles
need not be conforming or watertight.

## Native segment-surface intersection

For an axis-aligned Cartesian edge

```text
x(t) = a + t (b-a),  0 <= t <= 1,
```

with direction `k`, solve the two transverse equations

```text
S_i(u,v) = a_i,
S_j(u,v) = a_j,  i,j != k,
```

and recover `t` from the longitudinal coordinate.

### Root isolation

For every Bézier element whose AABB overlaps the segment:

1. reject parameter boxes whose rational control hull cannot contain the two
   fixed transverse coordinates;
2. recursively subdivide ambiguous parameter boxes;
3. use a triangle estimate or the parameter-box center as the Newton seed;
4. solve the two equations with analytic NURBS derivatives and safeguarded
   steps constrained to the element;
5. accept only roots whose full 3D residual and segment parameter pass the
   geometry tolerances.

Recursive conservative rejection supplies completeness; Newton supplies final
accuracy. Newton failure alone never proves that no root exists.
If subdivision reaches its configured depth while a box can neither be
rejected nor resolved, report an unresolved candidate and stop. It must not be
converted into a no-intersection result.

Use an `h`-independent geometric tolerance based on the model scale, initially

```text
tol_x = max(1e-12 * geometry_diameter, 1e-14).
```

The implementation records residuals so this value can be validated rather
than hidden inside the solver.

### Crossing record

Every accepted grid crossing stores:

```text
patch_index
u, v
edge_parameter t
physical_point
outward_normal
surface_component
incident smooth-seam parameters, when applicable
```

The patch and parameters come from the NURBS solve. Triangle barycentric
recovery is not used.

### Root canonicalization

- Roots duplicated across Bézier-element boundaries on one patch are merged.
- Roots duplicated across a declared smooth or periodic patch seam are merged
  through the edge-parameter map.
- A root exactly on a non-G1 feature edge is reported as an unsupported
  grid/feature alignment because the current correction ownership is
  ambiguous there.
- Endpoint roots report `surface intersects Cartesian node`.
- A positive-length solution set reports `surface overlaps Cartesian edge`.
- A root with near-zero `normal dot edge_direction` reports a tangential
  grid-edge contact.
- Two or more distinct transverse roots on one Cartesian edge report every
  `(patch,u,v,t)` and stop with `multiple crossings on Cartesian edge`.

The last rule means “unsupported at this resolution,” not “discard all but
one.” Refinement or a grid shift is required.

## Cartesian barriers

Treat Cartesian nodes as vertices of a graph and their six-neighbor grid
edges as graph edges. Store one compact barrier flag for every x-, y-, and
z-directed edge:

```text
Bx(i,j,k): (i,j,k) <-> (i+1,j,k)
By(i,j,k): (i,j,k) <-> (i,j+1,k)
Bz(i,j,k): (i,j,k) <-> (i,j,k+1)
```

Initialize every flag to zero. Set a flag to one exactly when the native NURBS
query returns one valid transverse crossing. Store the crossing record beside
that barrier.

The element AABBs are converted to clamped Cartesian index ranges so only
nearby grid edges are queried. The work therefore scales primarily with the
surface resolution rather than testing every volume edge against every patch.

A barrier is only a topological “do not traverse” flag. It is distinct from
the crossing record used by KFBI jump correction.

## Domain labeling

Create an integer graph-component array initialized to `-1`. Run BFS over all
Cartesian nodes, traversing a six-neighbor edge only when its barrier is zero.

For each graph component, record:

- number of nodes;
- a representative node;
- whether it touches any face of the embedding box.

Every box-touching graph component is exterior and receives label `0`.

For each enclosed graph component, classify one representative point with a
global NURBS ray query. The ray solver uses the same conservative Bézier BVH,
root isolation, and seam canonicalization, but a long ray may contain many
crossings. For a general ray it solves the two components orthogonal to the ray
direction and recovers the longitudinal parameter. If one ray is tangent to a
feature, choose the next direction from a fixed deterministic list rather than
perturbing the geometry.

Count unique transverse crossings per declared surface component:

- no odd component: exterior cavity, label `0`;
- exactly one odd component: label `component + 1`;
- more than one odd component: reject as overlapping/nested solid ownership,
  which the current label API does not define.

All nodes in the graph component inherit this one result. This is equivalent
to ray-parity classification but avoids casting a ray from every volume node.

## GridPair3D integration

Introduce a NURBS-backed grid-geometry object in the geometry library. It owns
or immutably references:

- the core NURBS surface model;
- the Bézier-element index;
- Cartesian barrier arrays;
- native crossing records;
- final domain labels and diagnostics.

Add a NURBS-aware `GridPair3D` construction path while retaining the raw
triangle path for legacy callers. In the NURBS path:

- `domain_label()` reads the barrier/flood-fill labels;
- label-changing edges use the stored native crossing record;
- `P2CrossingOwner3D` gains native patch and `(u,v)` information;
- native `(patch,u,v)` selects the existing parameter-grid DOF candidates and
  correction ownership;
- `geometry_panel_index` and triangle barycentric coordinates remain legacy
  fields, not authoritative NURBS data;
- nearest P2 centers remain available only for projection, spread, restrict,
  and existing correction-support operations.

The complete triangle `geometry_interface` may remain for visualization and
regression comparisons. It no longer supplies production labels or crossings.

## Consistency checks

After construction, require:

1. every barrier has exactly one native crossing record;
2. every neighboring node pair with different binary inside/outside status is
   separated by a barrier;
3. small samples on the two sides of every transverse crossing have opposite
   NURBS parity;
4. every accepted root residual is below `tol_x`;
5. every crossing patch parameter lies inside its declared domain;
6. no production crossing uses a triangle-only fallback;
7. `gap_crossings == 0` by construction.

## Diagnostics

Report per geometry and grid level:

- NURBS patch, knot-span, and Bézier-element counts;
- topology interval counts and validation failures;
- BVH/AABB candidate edge counts;
- triangle seed hits and triangle-seed misses recovered by NURBS isolation;
- subdivision boxes, Newton attempts, iterations, and failures;
- accepted roots, same-patch deduplications, and seam deduplications;
- endpoint, overlap, tangency, and multiple-crossing degeneracies;
- x/y/z barrier counts;
- Cartesian graph-component counts and sizes;
- box-connected exterior components and representative ray queries;
- exact-predicate label mismatches for benchmark geometries;
- NURBS crossing residual maxima;
- exact/gap crossing-owner counts.

## Validation

### Geometry-query unit tests

1. A bilinear plane returns its analytic `(u,v,t)` intersection.
2. A rational quarter cylinder returns points satisfying its exact radius.
3. Torus test roots satisfy the analytic torus equation.
4. A curved case where the coarse triangle misses but the Bézier AABB contains
   a root is still found.
5. Duplicate roots on smooth and periodic seams are counted once.
6. A non-G1 feature-edge hit reports the explicit alignment error.
7. Surface-on-node, surface-on-edge, tangency, and two-crossing fixtures report
   their required diagnostics.
8. L-prism partial/one-to-many edge topology validates as closed.

### Domain-label tests

1. The known L-prism reentrant lower-arm node is inside.
2. The L-prism missing quadrant is outside.
3. Target points in the torus tube and cylinder wall are inside.
4. Target points in the torus and cylinder holes are outside.
5. Disconnected closed components retain distinct positive labels.
6. Triangle refinement or nonconforming triangle seams do not change labels.
7. Every label-changing grid edge has a stored native crossing.

### Numerical acceptance

Run the L-prism Dirichlet normal-jump formulation with cubic `same_patch 32/20`
at `N=32,64,128`:

- label mismatches are zero at every level;
- gap-fallback crossings are zero at every level;
- Dirichlet GMRES converges at every level;
- density and interior `Linf` errors decrease on both refinements;
- `N=64` to `N=128` density and interior orders are at least two;
- Neumann, torus, cylinder, constant-jump, route-mismatch, and native-NURBS
  regressions retain their existing tolerances.

Only after the `N=128` gates pass may `N=256` be run.

## Alternatives rejected

### Nearest normal

A single tangent half-space cannot represent the global side of a nonconvex
surface. Increasing its tolerance hides real geometry and does not fix the
model.

### Closed conforming triangle mesh as geometric truth

It requires repairing unrelated tessellation conformity, retains chord error,
and discards native parameter ownership. It is unnecessary when the NURBS
surface is already available.

### Triangle-only broad-phase rejection

A flat chord can miss a curved intersection. Triangles may seed or prioritize,
but only a conservative NURBS/Bézier bound may reject a candidate.

### Ray classification at every volume node

It is correct but performs a global query at `O(N^3)` points. Barrier graph
components share the same classification at much lower query count.

### Multiple crossings on one grid edge

Parity alone could label the endpoints, but the current KFBI stencil stores
one crossing and one jump correction per grid edge. Supporting a list of
crossings requires a separate discretization design.

## Out of scope

- multiple crossings on one Cartesian grid edge;
- arbitrary Boolean combinations of overlapping or nested solids;
- silent handling of grid-node, overlap, feature-edge, or tangent degeneracy;
- rewriting P2 correction patches or Cauchy/restrict/spread formulas;
- making visualization triangles conforming;
- running `N=256` before the `N=128` acceptance gates pass.
