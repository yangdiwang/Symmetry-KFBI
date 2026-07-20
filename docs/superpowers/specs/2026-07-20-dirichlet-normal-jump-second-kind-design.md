# Dirichlet Normal-Jump Second-Kind Design

## Goal

Add a third Dirichlet formulation for the L-shaped example while preserving the two existing formulations. Fix `[u]=g_D`, solve for `q=[u_n]`, and close the exterior box problem with `u_n^+=0`.

## Discrete equation

For the existing discrete potential `P_h(a,b)`, solve

`A_h q = b_h`, where

- `A_h q = R^+_{n,h} P_h(0,q)`,
- `b_h = -R^+_{n,h} P_h(g_D,0)`.

The continuous operator is the exterior normal trace of a single-layer potential, hence has the form `+/- 1/2 I + K'`.

## Normal restrict

Reuse the corrected inside/outside samples assembled by `exterior_trace`. The joint normal fit uses `t=rho/h`; row 0 of its pseudoinverse recovers the value and row 1 recovers the derivative with respect to `t`. Store row 1 as `c1_weights_` and return `c1_weights_.dot(samples)/h_`.

The production residual is the directly restricted exterior normal trace. Independently compute `u_n^- - q` and report its difference from the direct route. The first implementation supports the joint bicubic/cubic and biquadratic/quadratic restrict modes; it rejects the six-point exterior-value-only mode.

## L-shape and acceptance

Keep the existing edge-local L-shape DOFs, normals, and corner Cauchy correction. Run `normal_jump_first_kind`, `value_jump_second_kind`, and new `normal_jump_second_kind` on the same L-shape grids. Report GMRES iterations, exterior-normal residual, route mismatch, interior Dirichlet trace error, density error against the manufactured normal data, and bulk errors.

The new formulation is accepted when it builds, GMRES converges on the requested L-shape levels, the exterior-normal residual closes to the solver tolerance up to discretization effects, and the interior solution converges under refinement.
