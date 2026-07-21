# Native NURBS Surface DOFs for the 3D KFBI App

## Goal

Make the 3D interface unknowns respect the native geometric representation:
the torus, hollow cylinder, and L-prism are all represented as collections of
NURBS surface patches, and every iteration DOF is generated directly from one
of those original patches. Triangulation and surface DOFs consume the same
NURBS model but remain separate discretizations.

## Geometry Model

Introduce one native surface model containing:

- the ordered `NurbsSurfacePatch3D` collection;
- a stable name for every patch;
- the orientation implied by each patch parameterization;
- complete patch adjacency for every shared geometric edge, including
  one-to-many adjacency where native patch edge partitions do not match;
- explicit G1 edge-to-edge parameter maps, stored separately for
  parameter-based crossing ownership.

The model is the sole source for both NURBS triangulation and surface-DOF
generation. The three geometries use these native decompositions:

- Torus: exact rational quadratic tensor-product patches covering both
  periodic directions.
- Hollow cylinder: rational quadratic patches for the inner and outer walls
  and for the two annular caps.
- L-prism: the existing twelve bilinear patches: three bottom patches, three
  top patches, and six side patches.

The L-prism therefore has twelve DOF patches as well. Its three coplanar top
patches and three coplanar bottom patches remain distinct native patches; their
shared coplanar seams are smooth connections. Top/side, bottom/side, and
side/side edges are not smooth connections.

## Uniform Parameter-Space DOFs

Each native patch owns a regular tensor grid in its original parameter domain
`[u_min,u_max] x [v_min,v_max]`. The division counts are selected from cheap
estimates of the two physical patch-direction lengths and the Cartesian mesh
spacing `h`. Once the counts are selected, both parameter directions are
divided uniformly; no adaptive parameter cells are introduced.

For parameter cell `(i,j)`, place one DOF at its center:

```text
u_i = u_min + (i + 1/2) delta_u
v_j = v_min + (j + 1/2) delta_v
```

Evaluate the native patch at that point:

```text
x_q = S(u_i, v_j)
J_q = |S_u(u_i,v_j) x S_v(u_i,v_j)|
n_q = oriented(S_u x S_v) / J_q
w_q = J_q delta_u delta_v
```

Every `SurfaceDof` stores the native patch index, native `(u,v)`, tensor-grid
indices `(i,j)`, point, normal, tangential frame, and midpoint quadrature
weight. No DOF lies on a patch edge because all DOFs are parameter-cell
centers.

## Patch Connectivity

Complete patch adjacency records every native patch pair that shares a
physical edge. It is a patch graph rather than a single edge-pair array,
because one long L-prism side edge can touch two shorter cap-patch edges.
Separately, G1 connections record an explicit edge-to-edge parameter map when
the edge partitions match and the surface normals agree. This handles closed
torus and cylinder seams without interpolating raw global angles across `0`
and `2*pi`.

The two graphs have different purposes. Crossing-to-DOF association uses only
the G1 graph, so an L-prism query near a sharp or re-entrant edge remains on
its incident native patch. Cauchy stencil construction uses the complete
topological graph, so its local polynomial may gather samples across a non-G1
edge but cannot jump between geometrically close, topologically unrelated
surface sheets such as the inner and outer cylinder walls.

## Crossing-to-DOF Association

The complete geometry triangulation remains responsible for Cartesian node
labels and Cartesian-edge/interface crossings. Each triangle already carries
its source NURBS patch index and the native parameter values of its three
vertices.

For a crossing associated with a triangle:

1. Compute or reuse the approximate crossing point on that triangle.
2. Compute its triangle barycentric coordinates.
3. Interpolate the triangle vertex parameters to obtain approximate
   `(u_hat,v_hat)` on the source patch.
4. Locate the parameter cell containing `(u_hat,v_hat)`.
5. Form the enclosing `2 x 2` tensor-grid DOFs as the only initial candidates.
6. If the `2 x 2` neighborhood crosses a smooth paired edge, transform the
   query to the paired patch and add the corresponding boundary candidates.
7. Do not cross a non-G1 edge.
8. Select the owner by three-dimensional distance, using normal consistency as
   a validity filter.

The approximate intersection and approximate parameter values are sufficient
because they only generate a small candidate set. Differential traces are not
evaluated from this geometric approximation.

## Cauchy Stencils

The owner DOF remains the center of one degree-three local harmonic Cauchy
polynomial. Stencil construction always includes the source patch and its
complete first topological ring, including non-G1 edges. If that pool still
contains fewer than the requested 48 value samples, it expands by additional
whole breadth-first rings. Candidate samples within the resulting topological
neighborhood are then ranked by three-dimensional Euclidean distance; this
lets points near a patch edge use the adjacent patch even when the source
patch alone contains more than 48 DOFs. The center DOF is retained, 28 normal
samples are the nearest subset of the 48 values, and the existing polynomial
order remains unchanged.

## Validation

Add deterministic checks covering:

- one `SurfaceDofCloud` patch for every native NURBS patch;
- all DOF parameters strictly inside their native patch domains;
- DOF points, tangents, normals, and weights obtained from NURBS evaluation;
- outward normal orientation and positive finite weights;
- area-weight convergence for the torus, hollow cylinder, and L-prism;
- smooth closed seams admitted with the correct parameter direction;
- L-prism coplanar seams admitted and all sharp edges rejected;
- `2 x 2` owner candidates on patch interiors and smooth seams;
- no candidates crossing non-G1 L-prism edges;
- crossing candidates remain confined to the G1 patch component;
- L-prism Cauchy stencils near a sharp edge include samples from the
  topologically adjacent patch across that non-G1 edge;
- Cauchy stencils do not select samples from topologically unrelated patches
  merely because their Euclidean distance is small;
- successful builds and `16/32/64` runs for all three geometries;
- comparison of Neumann first-kind and Dirichlet normal second-kind errors,
  convergence orders, iteration counts, and final residuals against the
  current baseline.

## Scope

This change replaces the three geometry-specific analytic DOF builders with a
shared native-NURBS builder and changes crossing ownership/stencil topology as
described above. Crossing ownership and Cauchy sampling deliberately use
different adjacency graphs. It does not replace the triangulated geometry
used by `GridPair3D`, change the cubic Cauchy polynomial, change tricubic
restriction, or change either Krylov equation.
