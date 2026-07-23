# 3D L-prism Neumann Cauchy-policy comparison

## Configuration

- Levels: `N=32,64,128`; no `N=256` run.
- Cauchy polynomial: degree 3, value/normal samples `48/28`.
- Restrict: tricubic grid interpolation and cubic normal fit.
- GMRES tolerance: `2e-10`.
- Default maximum GMRES iterations: 80 for both formulations.

The default stencil policy is `g1_nearest`: it may traverse
`smooth_neighbors` but never a non-G1 topological edge. `same_patch` and
`topological_nearest` remain explicit diagnostics.

## Neumann internal maximum error

| Policy | N | Interior Linf | Order | GMRES |
|---|---:|---:|---:|---:|
| `same_patch` | 32 | 1.303731216e-5 | - | 38 |
| | 64 | 5.460953214e-7 | 4.577350 | 26 |
| | 128 | 3.569001137e-7 | 0.613632 | 51 |
| `g1_nearest` | 32 | 1.299933034e-5 | - | 39 |
| | 64 | 5.019877191e-7 | 4.694641 | 26 |
| | 128 | 3.632432742e-7 | 0.466716 | 51 |
| `topological_nearest` | 32 | 2.865047313e-6 | - | 28 |
| | 64 | 6.218628046e-7 | 2.203891 | 24 |
| | 128 | 2.135845668e-7 | 1.541789 | 31 |

## Interpretation

Allowing only G1 crossings changes the L-prism maximum error very little.
The top and bottom coplanar patch seams are G1, but each vertical side is
separated by non-G1 edges. The maximum error is controlled by the reentrant
sharp-edge neighborhood, so both `same_patch` and `g1_nearest` retain a
large one-sided stencil there. Their maximum stencil radius is about `7h`
and the worst reported Cauchy condition number is about `100`, making the
maximum norm sensitive to grid/edge phase and producing the irregular
orders.

`topological_nearest` crosses sharp edges. For this manufactured Neumann
case, samples on adjacent faces still come from one smooth Cartesian
harmonic field, so the cross-edge stencil is more compact and symmetric.
Its maximum radius is about `4.18h`, worst condition number about `15`, and
the GMRES counts fall to `28,24,31`. This is a useful Neumann diagnostic,
but not a safe global default: the Dirichlet-normal formulation took
`54` steps at `N=32`, hit the new 80-step cap at `N=64`, and was deliberately
stopped at 40 steps in the isolated `N=128` diagnostic.

## Runtime guard

`KFBIM_3D_GMRES_MAX_ITERATIONS` now defaults to 80 and applies to both
formulations. Each completed level is written to CSV before convergence is
checked; a failure reports geometry, level, formulation, iterations, and
residual, then prevents the next level from starting.

## Artifacts

- `output/neumann_exterior_zero_trace_3d/analysis/neumann_policy_comparison_N32_N128.csv`
- `output/neumann_exterior_zero_trace_3d/analysis/same_patch_v48_n28_cap80_N32_N128/`
- `output/neumann_exterior_zero_trace_3d/analysis/g1_v48_n28_cap80_N32_N128/`
- `output/neumann_exterior_zero_trace_3d/analysis/topological_v48_n28_cap80_N32_N64/`
- `output/neumann_exterior_zero_trace_3d/analysis/topological_v48_n28_cap40_N128/`
