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
- smooth edge-to-edge connections, including the paired edges and whether the
  neighboring edge parameter is reversed;
- non-smooth geometric edges, which are deliberately absent from the smooth
  adjacency graph.

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

## Smooth Patch Connectivity

Patch edge matching records an explicit edge-to-edge parameter transformation.
The connection is admitted only when the physical edge curves coincide and
the surface normals agree to the configured smoothness tolerance. This handles
closed torus and cylinder seams without interpolating raw global angles across
`0` and `2*pi`; the query crosses a paired NURBS edge and applies its local
parameter orientation.

Non-G1 edges never contribute neighboring candidates. In particular, an
L-prism query near a sharp or re-entrant edge remains on its incident native
patch. The same smooth adjacency graph constrains local Cauchy sampling so a
single local polynomial never gathers data across a geometric corner.

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
polynomial. Stencil construction starts from its native tensor grid and may
expand only through smooth edge connections. Candidate samples are finally
ranked in physical space. Existing value/normal sample counts and polynomial
order remain unchanged.

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
- local Cauchy stencils confined to the smooth patch component;
- successful builds and `16/32/64` runs for all three geometries;
- comparison of Neumann first-kind and Dirichlet normal second-kind errors,
  convergence orders, iteration counts, and final residuals against the
  current baseline.

## Scope

This change replaces the three geometry-specific analytic DOF builders with a
shared native-NURBS builder and changes crossing ownership/stencil topology as
described above. It does not replace the triangulated geometry used by
`GridPair3D`, change the cubic Cauchy polynomial, change tricubic restriction,
or change either Krylov equation.
