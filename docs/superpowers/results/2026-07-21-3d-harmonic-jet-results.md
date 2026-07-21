# 3D harmonic-jet KFBI results

## Configuration

- Grids: `N = 16, 32, 64`, box `[-1.5, 1.5]^3`.
- Exact harmonic field:
  `u = exp(0.35 x) cos(0.21 y) cos(0.28 z)`.
- Local Cauchy reconstruction: one topology-filtered degree-three harmonic
  polynomial, using 48 value and 28 normal samples.
- Restriction: 4x4x4 tricubic Cartesian interpolation followed by a joint
  cubic normal fit on the layers `+-{0.2, 0.6, 1.0, 1.4} h`; the recovered
  dimensionless linear coefficient is divided by `h`.
- GMRES relative tolerance: `2e-10` for both formulations.

The Neumann formulation iterates the value jump until the exterior value
trace is zero. The Dirichlet second-kind formulation prescribes the value
jump and iterates the normal jump until the exterior normal trace is zero.

## Neumann exterior-zero-value-trace formulation

| Geometry | N | Iterations | Density Linf | Density order | Interior Linf | Interior order |
|---|---:|---:|---:|---:|---:|---:|
| torus | 16 | 28 | 1.1424e-5 | - | 1.0888e-5 | - |
| torus | 32 | 25 | 1.4804e-6 | 2.95 | 1.4986e-6 | 2.86 |
| torus | 64 | 23 | 2.5978e-7 | 2.51 | 2.7685e-7 | 2.44 |
| cylinder | 16 | 24 | 7.1942e-6 | - | 6.5023e-6 | - |
| cylinder | 32 | 26 | 1.6072e-6 | 2.16 | 1.6899e-6 | 1.94 |
| cylinder | 64 | 24 | 3.1627e-7 | 2.35 | 3.5450e-7 | 2.25 |
| L prism | 16 | 26 | 9.2373e-6 | - | 9.5207e-6 | - |
| L prism | 32 | 29 | 2.4767e-6 | 1.90 | 2.8088e-6 | 1.76 |
| L prism | 64 | 23 | 5.0551e-7 | 2.29 | 6.2375e-7 | 2.17 |

## Dirichlet exterior-zero-normal-trace formulation

| Geometry | N | Iterations | Normal-jump Linf | Jump order | Interior Linf | Interior order |
|---|---:|---:|---:|---:|---:|---:|
| torus | 16 | 15 | 1.7014e-5 | - | 2.7464e-6 | - |
| torus | 32 | 15 | 3.7762e-6 | 2.17 | 4.2914e-7 | 2.68 |
| torus | 64 | 13 | 9.5701e-7 | 1.98 | 8.6492e-8 | 2.31 |
| cylinder | 16 | 97 | 1.7369e-4 | - | 5.0202e-6 | - |
| cylinder | 32 | 32 | 2.8338e-5 | 2.62 | 4.6161e-6 | 0.12 |
| cylinder | 64 | 25 | 1.4770e-6 | 4.26 | 3.1582e-7 | 3.87 |
| L prism | 16 | 31 | 2.1042e-4 | - | 1.8227e-5 | - |
| L prism | 32 | 50 | 5.9542e-5 | 1.82 | 2.3855e-6 | 2.93 |
| L prism | 64 | 81 | 4.2983e-6 | 3.79 | 4.0032e-7 | 2.58 |

All 18 solves converged. At `N=64`, the independently recovered exterior
normal trace and the value obtained from the interior trace minus the normal
jump agree to approximately `7e-14` or better. Constant-jump readiness tests
give value-trace errors around `1e-14` and normal-trace errors around `2e-13`
or smaller.

The full machine-readable results are written at runtime to
`output/neumann_exterior_zero_trace_3d/neumann_results.csv` and
`output/neumann_exterior_zero_trace_3d/dirichlet_normal_results.csv`.
