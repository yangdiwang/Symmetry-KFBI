# 3D Cubic Cauchy Degree Experiment Design

## Goal

Measure the isolated effect of changing the app-local three-dimensional
harmonic Cauchy fit from degree four to degree three on the L-prism
Dirichlet normal-jump second-kind solve.

## Controlled change

- Change `PanelCenterCauchyFit3D` and its harmonic polynomial space from
  degree four to degree three. The coefficient count therefore changes from
  25 to 16.
- Keep the current topological nearest-neighbor stencil unchanged: 48 value
  samples and 28 normal-derivative samples.
- Keep the native NURBS surface DOFs, crossing ownership, spread, grid
  interpolation, normal restriction, manufactured harmonic solution, GMRES
  tolerance (`2e-10`), restart, and iteration limits unchanged.
- Preserve the existing command-line format and output files.

This is a single-variable experiment. Same-patch and face-balanced stencil
policies are explicitly deferred until the degree comparison is complete.

## Verification

Add a focused regression assertion that the 3D Cauchy fit reports 16
coefficients. Build and run the native NURBS test and the 3D app for
`l_prism 16 32 64`.

Compare the fresh degree-three results with the recorded degree-four
baseline using:

- GMRES iterations and relative residual;
- maximum local Cauchy condition number;
- density and interior `Linf` errors and their observed orders;
- exterior-normal condition residual and direct/jump route mismatch.

The degree-three route is preferable only if all solves converge, the error
sequences retain positive refinement orders, and iteration growth is no worse
than the degree-four baseline. Lower GMRES counts alone are not sufficient if
accuracy or convergence order deteriorates.
