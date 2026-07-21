# 3D Cauchy Stencil Policy Comparison Design

## Goal

Compare two alternative Cauchy sampling policies on the native-NURBS
L-prism while holding the active cubic harmonic space and every non-stencil
algorithmic parameter fixed.

## Policies

The existing topological-nearest policy remains the control.

`same_patch` selects Euclidean-nearest DOFs owned by the center DOF's native
NURBS patch. It requests 48 value and 28 normal samples. If a coarse patch
contains fewer samples, it uses every DOF on that patch; value and normal
counts may therefore vary by stencil. The cubic 3D harmonic space has 16
coefficients, and every fitted system must be checked for full rank.

`balanced_patches` first computes the existing topological-nearest set for
the requested count. Patches represented in that set are the active patches,
which prevents a face-interior stencil from importing distant neighboring
patches. The selector then takes samples in round-robin distance order from
the active patches so their selected counts differ by at most one whenever
capacity permits. Value and normal sets are balanced independently. Within
each patch, candidates remain ordered by three-dimensional Euclidean
distance. Existing distance weights in the least-squares fit remain
unchanged.

Neither new policy changes crossing ownership. Non-G1 neighbors remain
eligible for Cauchy fitting, consistent with the current topological rule.

## Selection and output

Use `KFBIM_3D_CAUCHY_POLICY` with values `topological_nearest` (default),
`same_patch`, and `balanced_patches`. Preserve the existing command-line
format. Default output retains its existing path; non-default policies write
to a policy-named child directory so comparison runs cannot overwrite one
another.

Report the policy and the minimum/maximum per-stencil value and normal
counts, incident-patch count, face-count imbalance, radius over `h`, and
local Cauchy condition statistics (median, 95th percentile, maximum).

## Tests

Focused native-NURBS tests must establish that:

- `same_patch` never returns a DOF from another patch and returns all
  available DOFs when the request exceeds patch capacity;
- `balanced_patches` retains the center, uses exactly the requested count,
  activates only patches present in the topological-nearest control set, and
  balances active-patch counts to within one when capacity permits;
- the existing topological selector and strict crossing ownership tests
  remain unchanged.

Run the focused test, build the 3D app, and execute all three policies for
`l_prism 16 32 64` at GMRES relative tolerance `2e-10`.

## Comparison criteria

For Dirichlet normal-jump second-kind, compare GMRES iterations and relative
residual, density and interior `Linf` errors and observed orders,
exterior-normal residual, route mismatch, runtime, stencil geometry, and
Cauchy conditioning. Also check the Neumann solve as a regression guard.

A policy is preferable only if every solve converges, Dirichlet density and
interior errors decrease at both refinements, and its iteration growth and
conditioning are better than the alternatives. A lower iteration count with
lost convergence order is not accepted as more stable.
