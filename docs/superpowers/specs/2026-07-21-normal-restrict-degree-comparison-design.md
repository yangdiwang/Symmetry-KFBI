# Normal Restrict Degree Comparison Design

## Goal

Compare grid interpolation order and normal fitting order independently for the Dirichlet `normal_jump_second_kind` formulation on the L-shaped geometry.

## Restrict modes

The comparison contains exactly four modes:

| Name | Grid interpolation | Normal fit | Layers per side | Jump correction |
| --- | ---: | ---: | ---: | ---: |
| `biquadratic_quadratic` | 2 | 2 | 3 | quadratic |
| `biquadratic_cubic` | 2 | 3 | 4 | quadratic |
| `bicubic_quadratic` | 3 | 2 | 3 | full Cauchy |
| `bicubic_cubic` | 3 | 3 | 4 | full Cauchy |

Add a `degree_compare` selector for these four modes. Preserve all existing mode names and the existing `normal_compare` and `compare` selections.

## Implementation

Add two `RestrictMode` values and replace checks tied directly to `BicubicCubicNormal` with independent helpers for grid side count, normal degree, layer count, Cauchy correction degree, restrict condition reporting, and quadratic-map preparation. The normal derivative remains `c1_weights_.dot(samples) / h_`.

## Numerical comparison

Use the L-shaped cubic harmonic manufactured solution, `normal_jump_second_kind`, uniform-midpoint DOFs, cubic-harmonic spread, fixed GMRES tolerance `2e-10`, and grids `N=64,128,256,512,1024`. Report GMRES iterations, exterior-normal residual, interior Dirichlet residual, bulk `Linf/L2`, density error, conditioning, and runtime. The cubic manufactured solution may be reproduced exceptionally well by the fully cubic mode; this is identified as a polynomial-reproduction result rather than a general convergence guarantee.
