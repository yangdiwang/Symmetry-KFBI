# 3D GMRES Run Guard Design

## Scope

The 3D harmonic-jet app keeps the existing discretization, tolerance, and
Cauchy policies. Numerical acceptance runs stop at `N=128`.

## Behavior

- Both Neumann and Dirichlet-normal solves use one configurable maximum:
  `KFBIM_3D_GMRES_MAX_ITERATIONS`.
- The default maximum is 80 iterations.
- The value must be a positive integer and is printed in the run header.
- After finishing a level, the app writes the accumulated CSV summaries.
- If either formulation did not converge, the app stops before the next
  level and reports geometry, `N`, formulation, iteration count, and final
  relative residual.

## Verification

- A normal `N=16` smoke run reports the default maximum and still converges.
- A maximum of one iteration fails at `N=16` and never starts the requested
  next level.
- Zero, negative, and non-integer maxima are rejected.
- The L-prism comparison runs `N=32,64,128` for `same_patch`, `g1_nearest`,
  and `topological_nearest`.
