# 3D Neumann Cross-Edge Value-Cauchy Comparison Design

## Goal

Determine whether the unstable L-prism Neumann value-trace solve is primarily
caused by the one-sided G1-only fit of the value-jump density near non-G1
feature edges.

The comparison changes only the Cauchy sample-selection policy.  It keeps the
current crossing-owner value restrict, cubic harmonic space, tricubic Cartesian
interpolation, cubic normal-line fit, compatibility correction, bordered
Neumann equation, GMRES tolerance, and manufactured solution unchanged.

## Background

The Dirichlet crossing-owner change corrected the owner of wrong-side
continuation terms before the recovered trace was differentiated by the
`c1/h` normal restrict.  The Neumann value restrict uses `c0` with no `1/h`
factor, so the same owner error is not expected to be its dominant remaining
failure.

The current Neumann owner study still builds G1-only `48/28` Cauchy stencils.
Near an L-prism non-G1 edge, these are long one-sided stencils.  Earlier
topological sampling made them shorter and more symmetric and substantially
improved Neumann iteration counts, but that experiment changed both value and
normal sample pools at once.  The new comparison isolates the value-density
continuity hypothesis.

## Compared routes

Every route uses `JointTricubicCrossingOwner`.

| Route | Value sample pool | Normal sample pool |
|---|---|---|
| `g1_value_g1_normal` | Existing G1-nearest | Existing G1-nearest |
| `edge_value_g1_normal` | Edge-aware topology-adjacent | Existing G1-nearest |
| `edge_value_edge_normal` | Edge-aware topology-adjacent | Edge-aware topology-adjacent |

The second route is the primary candidate.  The third route determines whether
the additional angular information from adjacent-face normal data is also
needed.

All three routes use one degree-three harmonic polynomial per center DOF.
Two-sector or singular-function fits are deliberately outside this first
comparison.

## Edge-aware sample selection

For each surface DOF:

1. Compute its physical distance to native non-G1 feature edges.
2. If the distance exceeds `2h`, use the existing G1-nearest selector exactly.
3. Otherwise, admit only native patches incident to the relevant feature edge,
   together with their G1-equivalent patch pieces.  Do not admit a physically
   close but topologically unrelated patch.
4. Partition admitted patches into face sectors connected through G1 seams.
   Allocate sample quotas as evenly as possible among the sectors, then rank
   candidates within each sector by Euclidean distance and deterministic DOF
   ID.
5. Always include the center DOF.  After sector quotas are filled, use the
   globally nearest remaining admitted candidates to reach the requested
   count.

At a feature vertex, use the union of patches incident to every non-G1 edge
whose distance ties the nearest edge within geometry tolerance, and balance by
G1 sector.  Periodic seams continue to use the native topology maps.

The requested counts remain `48` value and `28` normal samples.  A route must
not silently cross an unrelated patch or silently fall back after a
rank-deficient fit.  It must instead report the center DOF, admitted sectors,
actual counts, radius, and singular values and stop that route.

## Code boundaries

The implementation should separate:

- sample-pool construction and sector balancing;
- assembly of value and normal stencil IDs;
- the existing harmonic least-squares fit;
- the existing crossing-owner restrict;
- comparison diagnostics and CSV output.

The value and normal selectors must be independently configurable so the three
routes do not require duplicate solver pipelines or conditional logic inside
the polynomial fit.

## Numerical probes

Use the L-prism at:

- `N = 32, 64, 128`;
- identity;
- 17-degree rotation about `(1,2,3)`;
- the same rotation plus the existing `t_xyz_1` translation.

This gives 27 formal solves.  Every `(pose,N,route)` records:

- physical-RHS GMRES history and final residual;
- a common, parameter-indexed, weighted-mean-zero RHS GMRES history;
- density and interior `Linf/L2` errors and adjacent orders;
- exact-density equation defect `A_h mu_exact - b_h`;
- defect and density-error norms in distance bins `<h`, `h--2h`, and `>2h`;
- stencil radius, condition number, actual value/normal counts, and sector
  balance;
- crossing-owner query count and template fingerprint before and after GMRES.

The common RHS separates a change in the discrete operator from a fortunate
change in the manufactured RHS.  Edge-distance bins determine whether any
improvement is actually caused by the non-G1-edge rows.

## Decision rules

All routes must converge below the unchanged `2e-10` tolerance without
geometry-label or owner-invariant failures.

The value-continuity hypothesis is supported when
`edge_value_g1_normal`, relative to the G1-only control:

- reduces the worst GMRES count and the pose-to-pose iteration spread;
- reduces the `N=128` `<2h` exact-equation defect;
- removes the baseline negative `N=64` to `N=128` interior/density order;
- does not increase any pose's `N=128` interior or density error by more than
  10 percent.

If only `edge_value_edge_normal` satisfies these conditions, adjacent-face
normal information is also required.  If neither edge-aware route satisfies
them, the next design should use sector-wise polynomials with shared edge
value/tangential constraints rather than further enlarging a single
polynomial stencil.

No route becomes the production default from a smoke result.  A default
change requires the complete 27-solve comparison and an audited result
summary.

## Verification

Unit tests must prove that the edge-aware selector:

- crosses the two incident faces of an L-prism non-G1 edge;
- crosses native G1 and periodic seams correctly;
- never selects an unrelated nearby sheet;
- is deterministic under equal distances;
- preserves the existing G1-only result outside the `2h` edge band.

An `N=32` application smoke verifies all three route names, complete
diagnostics, mean-zero common RHS, and unchanged crossing-owner fingerprints.
The formal run writes partial CSV output after every completed route so a
failed or interrupted comparison remains auditable.
