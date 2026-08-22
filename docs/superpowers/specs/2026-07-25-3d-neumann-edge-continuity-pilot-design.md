# 3D Neumann Non-G1 Edge-Continuity Pilot Design

## Goal

Add an optional pilot formulation for the 3D Neumann L-prism that
constrains the value-jump density to have a single trace across non-G1
patch connections.  The pilot must preserve the current patch-centered
surface DOFs, crossing-owner restriction, geometry preprocessing, and
physical test problem so that an A/B comparison isolates the effect of
edge continuity.

The pilot initially runs only `N=32,64` for:

- `baseline`;
- `ty_m0083`, the slowest current `N=128` translation;
- `rot_axis123_17deg`.

`N=128` is added only after the pilot passes its structural and coarse-grid
acceptance checks.

## Why This Is a Separate Neumann Mode

The Dirichlet normal-jump density is flux-like and may be represented
patchwise across a feature edge.  The Neumann value-jump density is a
physical value trace modulo a constant.  For the smooth manufactured
solution it has one value on both sides of every geometric patch
connection, including non-G1 edges.

The current `SurfaceDofCloud3D` stores an independent cell-centered tensor
grid on every patch.  `G1Nearest` couples Cauchy fitting only through
smooth patch connections; it does not constrain the value density across
feature edges.  The resulting enlarged discrete space contains
nonphysical edge-mismatch modes.

This pilot does not change Cauchy sampling across non-G1 edges.  It changes
only the admissible Neumann value-density space.  This distinction makes
the experiment causal.

## Existing Topology Reused

`NativeNurbsSurface3D::geometric_connections` already supplies everything
needed to build an edge constraint:

- the two incident patch IDs;
- the edge type on each patch;
- the active parameter interval on each side;
- orientation through `reversed`;
- the `g1` classification.

Partial L-prism edge connections are kept as separate intervals.  The
pilot processes only connections with `g1 == false`; no candidate or
constraint is allowed to jump to an unrelated topological neighbor.

## Alternatives Considered

### Penalty coupling

Adding `gamma C^T C` is the smallest code change, but the result depends on
an arbitrary penalty scale and can make the second-kind operator look
increasingly stiff as `h` decreases.  It is not suitable for deciding
whether continuity itself fixes the problem.

### Lagrange multipliers

The mortar system could append one multiplier per edge sample.  It would
enforce continuity exactly, but the current Neumann system already has an
unbalanced constant-mode multiplier.  Adding another saddle block would
confound the GMRES experiment.

### Mass-weighted projection

The recommended pilot builds the same mortar equations but enforces them
with a precomputed surface-mass projection.  It introduces no edge
multipliers into GMRES and leaves all existing density DOFs intact.

A globally shared edge basis remains the preferred long-term
representation if the pilot succeeds, but it is deliberately outside this
first experiment.

## Edge Trace Reconstruction

For each non-G1 connection, choose edge sample points at cell centers of a
uniform physical edge partition:

```text
n_edge = max(2, ceil(connection_length / h))
s_j = (j + 1/2) / n_edge
```

Map `s_j` to both native parameter intervals, applying `reversed` on the
second side.  At each side:

1. use cubic interpolation along the edge direction from the nearest four
   parameter-cell centers when available;
2. use quadratic extrapolation from the first three cell-center rows in
   the inward parameter direction;
3. reduce the order only when a coarse patch lacks enough rows, and record
   that reduction as a diagnostic.

This produces sparse row operators `E_a(j)` and `E_b(j)`.  The continuity
row is

```text
C_j = E_a(j) - E_b(j).
```

Rows are weighted by the physical edge quadrature weight.  Exact constants
must satisfy `C * 1 = 0` to roundoff.  Applying `C` to the manufactured
exact density sampled at the surface DOFs must converge at least cubically
until roundoff or corner effects dominate.

## Projection and Matrix-Free Operator

Let `W` be the diagonal matrix of surface quadrature weights.  After
dropping numerically dependent constraint rows with rank-revealing QR,
define

```text
G = C W^{-1} C^T
P = I - W^{-1} C^T G^{-1} C.
```

`G` is assembled and factorized once during setup.  `P` is applied
matrix-free.  Required invariants are:

```text
||C P x|| <= tolerance * ||x||
||P(Px) - Px|| <= tolerance * ||x||
||P 1 - 1|| <= tolerance
```

The pilot retains the existing single constant-mode augmentation so that
only edge continuity changes.  For an input `(phi, lambda)`, its operator
is

```text
top = P A(P phi) + (I - P) phi + lambda * 1
bottom = weighted_mean(P phi)
```

and the right-hand side is `(P b, 0)`.  The `(I-P)` term gives excluded
edge-mismatch modes unit eigenvalues instead of leaving an artificial
nullspace.  The returned density is projected once more before physical
field reconstruction.

A later task may replace the constant-mode saddle equation with a
fully mass-scaled rank-one stabilization, but that change is intentionally
not bundled into this pilot.

## Code Boundaries

Create a focused module:

```text
apps/neumann_edge_continuity_3d.hpp
apps/neumann_edge_continuity_3d.cpp
apps/neumann_edge_continuity_3d_test.cpp
```

It owns:

- edge interval sampling and parameter mapping;
- sparse edge-trace rows;
- constraint diagnostics;
- the factorized mass projector.

It depends on `NativeNurbsSurface3D` and `SurfaceDofCloud3D`, but not on
the FFT solver or GMRES.  The Neumann app wraps the existing
`ExteriorZeroTraceOperator3D` with the optional projector.

Add a dedicated command:

```text
--neumann-edge-continuity-study [N ...]
```

Default levels are `32 64`.  This command always runs the unconstrained and
edge-projected modes on identical preprocessed geometry before writing
checkpoints.

## Output and Diagnostics

Write under:

```text
output/neumann_edge_continuity_3d
```

The study records:

- GMRES convergence, iterations, and residual history;
- density and interior `Linf`/`L2` errors;
- maximum and weighted-RMS edge mismatch before and after projection;
- exact-density edge mismatch;
- projection idempotence and constant-preservation defects;
- constraint count, rank, reduced-order row count, and factorization time;
- all existing geometry and crossing-owner invariants.

## Acceptance

The pilot is structurally valid only if:

- every non-G1 connection interval is covered exactly once;
- no G1 or unrelated patch pair enters `C`;
- `C * 1`, projection idempotence, and projected constraint defects are at
  most `1e-11` in normalized infinity norm;
- the exact-density edge mismatch decreases by at least a factor of six
  from `N=32` to `N=64`, consistent with third-order edge reconstruction;
- geometry and crossing-owner diagnostics are unchanged.

The numerical hypothesis is supported if, relative to the unconstrained
mode:

- the worst GMRES iteration count over the three pilot poses does not
  increase and `ty_m0083` decreases;
- density and interior errors do not increase by more than 10 percent at
  either level;
- the projected numerical edge mismatch is reduced by at least four
  orders of magnitude;
- the `N=32` to `N=64` density and interior trends improve or remain
  unchanged.

Failure of the numerical gates is a useful negative result: it means edge
continuity is not the dominant remaining mechanism, and the next
experiment should target the phase-discontinuous crossing-owner blend or
the constant-mode scaling instead of expanding this constraint system.
