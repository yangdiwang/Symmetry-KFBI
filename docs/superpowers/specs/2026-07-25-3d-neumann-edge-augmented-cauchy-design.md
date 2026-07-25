# 3D Neumann Edge-Augmented Cauchy Design

## Goal

Add an optional 3D Neumann Cauchy reconstruction in which each non-G1
geometric edge supplies shared value-jump samples to the ordinary
patch-centered local Cauchy fits.

The method must implement the following two-stage construction during a
matrix-free operator application:

```text
current face value jumps and prescribed face normal jumps
    -> shared value jump at non-G1 edge samples
    -> edge samples added to nearby patch-centered Cauchy fits
    -> spread, FFT solve, and restrict
```

Both least-squares maps are precomputed from fixed geometry.  An operator
application therefore evaluates linear maps only; it does not repeat an
SVD or add edge unknowns to GMRES.

This is a new experiment, not a revision of the completed
`non_g1_edge_projected` density-space pilot.  The existing Cauchy route,
the density projector, and all current commands remain available.

## Motivation

The completed edge-density projection pilot enforced a common
one-sided extrapolated trace by modifying the entire value-jump density.
It reduced the measured density mismatch to roundoff but increased every
same-level error comparison.  That result indicates that global
modification of the admissible density space is too indirect and too
intrusive.

The new method places the edge information at the point where it is
needed: in the local Cauchy polynomial used to assemble the correction.
Only local fits whose support reaches a non-G1 edge are changed.  Surface
DOFs away from a feature edge retain the existing map exactly.

## Alternatives

### Recompute both fits during every operator application

This is the most literal implementation, but the geometry and design
matrices do not change during GMRES.  Repeating the weighted SVD would
waste work and introduce avoidable run-to-run numerical variation.

### Precompute two linear maps

This is the selected design.  The edge auxiliary fit produces

```text
edge_value = E_value * value_jump + E_normal * normal_jump.
```

The patch-centered fit then consumes the edge values through another
precomputed map.  The two maps remain separate at runtime so that edge
values can be diagnosed and sparsity is not destroyed by forming a
dense algebraic composition.

### Add edge values as GMRES unknowns

An explicit edge basis would require compatibility equations or a new
saddle block.  It would increase the system dimension and mix the effect
of edge sampling with a new Krylov formulation.  It is outside this
experiment.

## Geometry and Edge Sampling

Use only entries of
`NativeNurbsSurface3D::geometric_connections` with `g1 == false`.
Partial connection intervals remain distinct.  Parameter orientation is
mapped with the native interval endpoints and `reversed` flag.

For a connection of physical length `L`, use

```text
edge_sample_count = max(4, ceil(L / h))
s_q = (q + 1/2) / edge_sample_count.
```

The samples are interval midpoints, so no edge endpoint is duplicated at
a geometric vertex.  Each sample stores:

- connection and sample indices;
- both incident patch and edge IDs;
- normalized and native parameters on both sides;
- the physical point from both patches and their gap;
- the common oriented edge tangent;
- both incident outward normals;
- the IDs used by the two face-side auxiliary stencils;
- conditioning and reproduction diagnostics.

The two native surface evaluations must agree within the existing
geometry tolerance.  A mismatch is a setup failure rather than a reason
to average unrelated points.

## Symmetric Auxiliary Edge Fit

At each edge sample, fit one degree-three three-dimensional harmonic
polynomial centered at the shared physical edge point.

Use a deterministic orthonormal frame:

1. the first axis is the oriented edge tangent;
2. the second is the normalized component of the two-normal bisector
   perpendicular to the tangent;
3. the third completes the right-handed frame.

A degenerate tangent or bisector is rejected with the connection and
sample IDs in the diagnostic.

The data stencil is symmetric:

- 24 value-jump samples from the first side;
- 14 normal-jump samples from the first side;
- 24 value-jump samples from the second side;
- 14 normal-jump samples from the second side.

Each side starts from its incident patch and may expand through G1
neighbors only.  It may not cross the target non-G1 connection or any
other non-G1 connection.  Candidate DOFs are sorted by Euclidean
distance to the edge point with DOF ID as the deterministic tie breaker.
Insufficient candidates are a setup failure; the pilot does not silently
reduce the polynomial degree or use asymmetric counts.

The two derivative groups use their sample DOFs' own surface normals.
Thus a single polynomial is constrained by

```text
p_e(x_j)                    ~= value_jump_j
n_j dot grad(p_e)(x_j)      ~= normal_jump_j
```

on both incident faces.  The existing local-coordinate scaling by `h`
and the existing value/normal distance weights are retained.

Let the fixed weighted pseudoinverse be `L_e`.  Evaluating the polynomial
at its origin gives one shared scalar:

```text
edge_value_q
    = E_value(q,:)  * value_jump
    + E_normal(q,:) * normal_jump.
```

For the Neumann exterior-zero-value-trace problem, `normal_jump` is
prescribed.  Its contribution may be cached once before GMRES, while the
value-jump contribution is evaluated for every operator application.

## Adding Edge Values to Patch-Centered Cauchy Fits

The existing degree-three harmonic Cauchy fit remains the base:

- 48 G1-nearest value samples;
- 28 G1-nearest normal samples;
- no ordinary stencil traversal across a non-G1 edge.

For each patch-centered surface DOF:

1. compute the geometric radius of its existing value stencil;
2. consider only non-G1 connections incident to the same G1-connected
   side of the surface;
3. for each such connection whose samples enter that radius, select the
   nearest four edge samples;
4. append each selected edge value as a value row in the weighted
   least-squares design.

An edge row is

```text
sqrt(w_e) * p_i(x_e) = sqrt(w_e) * edge_value_e
```

with the same distance-dependent weight as an ordinary value sample.
The initial edge weight scale is exactly `1.0`; there is no penalty
parameter and the edge row is not a hard interpolation constraint.

At a geometric vertex, a center may receive four samples from each
incident non-G1 edge.  This is the first-version corner treatment.  It
does not introduce a vertex DOF, an endpoint sample, or a separate
vertex fit.

The resulting precomputed map for center `i` is

```text
coeff_i
    = M_value_i  * local_value_jump
    + M_normal_i * local_normal_jump
    + M_edge_i   * selected_edge_values.
```

Substituting the edge map is mathematically possible, but the
implementation keeps the two stages explicit:

```text
edge_values = E_value * value_jump + E_normal * normal_jump
coefficients = M_value * value_jump
             + M_normal * normal_jump
             + M_edge * edge_values.
```

This keeps both matrices sparse and exposes the edge values for
diagnostics.

## Runtime Integration

Add a Cauchy edge mode with two values:

```text
none
non_g1_auxiliary_values
```

`none` is the legacy behavior and remains the default outside the
dedicated study.  `non_g1_auxiliary_values` changes only Cauchy
coefficient construction.

The Neumann operator continues to use the patch-independent value-jump
density.  It does not apply `NeumannEdgeContinuityProjector3D`, does not
add a penalty, and does not add a GMRES unknown.

The matrix-free path is:

```text
value_jump from GMRES
normal_jump prescribed by the problem
    -> shared edge values
    -> legacy or edge-augmented Cauchy coefficients
    -> unchanged spread
    -> unchanged FFT bulk solve
    -> unchanged crossing-owner restrict
    -> exterior-zero-trace residual
```

The same geometry, surface DOFs, crossing rows, restrict-owner
preprocessing, FFT grid, right-hand side, tolerance, and GMRES cap are
used for the A/B pair.

## Code Boundaries

Create:

```text
apps/neumann_edge_augmented_cauchy_3d.hpp
apps/neumann_edge_augmented_cauchy_3d.cpp
apps/neumann_edge_augmented_cauchy_3d_test.cpp
```

The module owns:

- non-G1 edge sample construction;
- two-sided G1-restricted stencil selection;
- edge frames and weighted pseudoinverses;
- sparse `E_value` and `E_normal` maps;
- per-center edge-sample attachments;
- edge-fit diagnostics and manufactured-field checks.

`PanelCenterCauchyFit3D` remains responsible for the patch-centered
polynomial.  It gains an optional edge-augmentation input and stores the
precomputed edge columns of each local fit.  The FFT, crossing-owner, and
GMRES code do not enter the new module.

Add a dedicated command:

```text
--neumann-edge-cauchy-study [N ...]
```

Accepted level prefixes are:

```text
32
32 64
32 64 128
```

The command compares `none` with `non_g1_auxiliary_values`.  It must not
substitute the prior projected-density mode for either side.

## Failure Handling

Setup fails with connection, sample, patch, and center IDs when any of
the following occurs:

- invalid or incomplete topology metadata;
- inconsistent mapped edge points;
- a degenerate edge frame;
- insufficient symmetric face samples;
- non-finite design data or weights;
- a rank-deficient edge or augmented local fit;
- an edge sample attached to an unrelated G1 component;
- a non-G1 traversal by an ordinary face stencil.

There is no automatic degree reduction, asymmetric fill, penalty
increase, or fallback to topological-nearest sampling.

## Structural Tests

Unit tests must cover:

1. exact reproduction of every degree-three harmonic basis member by the
   auxiliary edge fit;
2. equality between direct weighted fitting and the precomputed edge
   maps;
3. linearity in both value and normal data;
4. one shared edge value used by both incident sides;
5. correct handling of `reversed` and partial intervals;
6. exactly 24/14 samples per side and no non-G1 traversal;
7. four nearest edge samples per incident edge for an eligible center;
8. unchanged legacy maps for centers outside every edge-support radius;
9. multiple incident edge groups, without duplicate endpoint samples,
   for a corner-near center;
10. rigid translation and rotation covariance;
11. expected failure for insufficient samples, invalid topology, and a
    deliberately rank-deficient stencil.

An operator-level test must verify that the legacy mode remains bitwise
unchanged and that the augmented mode performs no geometry query or SVD
during repeated applications.

## Numerical Study

The first pilot uses the native twelve-patch L-prism with:

- `N=32,64`;
- `baseline`;
- `ty_m0083`;
- `rot_axis123_17deg`;
- G1-nearest degree-three 48/28 face stencils;
- joint-tricubic crossing-owner restriction;
- `region_closest_hybrid` owner preprocessing;
- GMRES tolerance `2e-10`, restart 80, and cap 80.

Run `N=128` only if the complete `N=32,64` pilot passes.

Record for both modes:

- density and interior `Linf`/`L2` errors and adjacent orders;
- GMRES convergence, iteration count, and residual history;
- setup, edge-map, coefficient, spread, FFT, restrict, solve, and total
  times;
- edge and augmented-fit condition statistics;
- shared edge values;
- the difference between the two incident patch-centered polynomials
  evaluated at every attached edge sample;
- geometry, owner, and shared-preprocess invariants.

## Acceptance

Structural acceptance requires:

- all non-G1 connection intervals and edge samples are covered exactly
  once;
- no ordinary face stencil crosses a non-G1 edge;
- harmonic cubic reproduction, direct/precomputed-map agreement, and
  rigid-transform covariance defects are at most `1e-11`;
- all edge and augmented local fits have full harmonic-cubic rank;
- centers outside edge support are bitwise identical to legacy;
- geometry, owner, and preprocessing snapshots are unchanged during
  GMRES.

The numerical hypothesis is supported only if:

- every solve converges below the existing cap and tolerance;
- the worst augmented GMRES count does not exceed the legacy worst
  count;
- augmented density and interior errors are no more than 10 percent
  above legacy at either coarse level;
- augmented density and interior `N=32` to `N=64` orders are at least
  `1.8`;
- rigid-transform error and iteration spreads do not increase;
- the incident-polynomial edge discrepancy decreases relative to legacy
  for every pose and level.

If the coarse pilot fails, keep the mode as an experimental diagnostic,
do not run `N=128`, and do not change the weights or tolerances to force
acceptance.
