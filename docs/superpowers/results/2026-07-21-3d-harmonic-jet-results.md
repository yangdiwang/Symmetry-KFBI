# 3D native-NURBS harmonic-jet KFBI results

## Configuration

- Grids: `N = 16, 32, 64`, box `[-1.5, 1.5]^3`.
- Exact harmonic field:
  `u = exp(0.35 x) cos(0.21 y) cos(0.28 z)`.
- Geometry: native `NurbsSurfacePatch3D` models for the torus, hollow
  cylinder and L prism. Surface DOFs are the centers of uniform native
  parameter cells and remain one-to-one with their source patches.
- Crossing ownership: intersect a grid edge with the flat NURBS
  triangulation, barycentrically interpolate the triangle's native `(u,v)`,
  form exactly one `2x2` parameter-cell candidate set, then choose by physical
  distance. Periodic/G1 seams are mapped; non-G1 patch edges are not crossed.
- Local Cauchy reconstruction: one topology-filtered degree-three harmonic
  polynomial, using 48 value and 28 normal samples. Samples may cross a G1
  seam but never a non-G1 edge.
- Restriction: `4x4x4` tricubic Cartesian interpolation followed by a joint
  cubic normal fit on the layers `+-{0.2, 0.6, 1.0, 1.4} h`; the recovered
  dimensionless linear coefficient is divided by `h`.
- GMRES relative tolerance: `2e-10` for both formulations.

The Neumann formulation iterates the value jump until the exterior value
trace is zero. The Dirichlet second-kind formulation prescribes the value
jump and iterates the normal jump until the exterior normal trace is zero.

## Native-NURBS geometry readiness

| Geometry | N | Patches | DOFs | Area relative error | Exact owners | Gap fallback | Endpoint fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| torus | 16 | 16 | 224 | 1.231e-2 | 107 | 59 | 0 |
| torus | 32 | 16 | 832 | 3.033e-3 | 566 | 120 | 0 |
| torus | 64 | 16 | 2912 | 9.647e-4 | 2350 | 410 | 0 |
| cylinder | 16 | 16 | 320 | 2.478e-3 | 192 | 92 | 0 |
| cylinder | 32 | 16 | 1160 | 7.774e-4 | 926 | 234 | 0 |
| cylinder | 64 | 16 | 4200 | 2.289e-4 | 4372 | 172 | 0 |
| L prism | 16 | 12 | 394 | 4.229e-16 | 222 | 0 | 0 |
| L prism | 32 | 12 | 1050 | 2.390e-14 | 982 | 0 | 0 |
| L prism | 64 | 12 | 3926 | 2.453e-14 | 3926 | 0 | 0 |

The torus consists of 16 rational quadratic quarter patches. The cylinder
has four patches in each of its two walls and two annular caps. The L prism
has 12 bilinear patches. All levels retain the requested 48/28 Cauchy sample
counts and require no endpoint-based crossing ownership.

## Neumann exterior-zero-value-trace formulation

| Geometry | N | Iterations | Density Linf | Density order | Interior Linf | Interior order |
|---|---:|---:|---:|---:|---:|---:|
| torus | 16 | 27 | 8.3810e-6 | - | 8.5110e-6 | - |
| torus | 32 | 23 | 1.0631e-6 | 2.98 | 9.7279e-7 | 3.13 |
| torus | 64 | 22 | 2.6868e-7 | 1.98 | 2.7651e-7 | 1.81 |
| cylinder | 16 | 31 | 1.8053e-5 | - | 1.8592e-5 | - |
| cylinder | 32 | 36 | 6.4741e-6 | 1.48 | 5.9879e-6 | 1.63 |
| cylinder | 64 | 29 | 7.9017e-7 | 3.03 | 7.4712e-7 | 3.00 |
| L prism | 16 | 70 | 5.0963e-4 | - | 1.8630e-4 | - |
| L prism | 32 | 38 | 1.4903e-5 | 5.10 | 1.2563e-5 | 3.89 |
| L prism | 64 | 26 | 3.9338e-7 | 5.24 | 5.0039e-7 | 4.65 |

## Dirichlet exterior-zero-normal-trace formulation

| Geometry | N | Iterations | Normal-jump Linf | Jump order | Interior Linf | Interior order |
|---|---:|---:|---:|---:|---:|---:|
| torus | 16 | 15 | 9.3567e-6 | - | 2.1190e-6 | - |
| torus | 32 | 15 | 2.8050e-6 | 1.74 | 3.6804e-7 | 2.53 |
| torus | 64 | 13 | 7.5183e-7 | 1.90 | 6.8442e-8 | 2.43 |
| cylinder | 16 | 115 | 1.3762e-3 | - | 6.9770e-6 | - |
| cylinder | 32 | 30 | 2.8471e-5 | 5.60 | 4.5632e-6 | 0.61 |
| cylinder | 64 | 30 | 4.0201e-6 | 2.82 | 2.4115e-7 | 4.24 |
| L prism | 16 | 38 | 2.6133e-4 | - | 1.0930e-5 | - |
| L prism | 32 | 26 | 4.9589e-5 | 2.40 | 3.6970e-6 | 1.56 |
| L prism | 64 | 24 | 3.2054e-6 | 3.95 | 3.1274e-7 | 3.56 |

## Comparison with the previous 3D DOF route at N=64

| Formulation | Geometry | Old/New iterations | Old/New density Linf | Old/New interior Linf |
|---|---|---:|---:|---:|
| Neumann | torus | 23 / 22 | 2.5978e-7 / 2.6868e-7 | 2.7685e-7 / 2.7651e-7 |
| Neumann | cylinder | 24 / 29 | 3.1627e-7 / 7.9017e-7 | 3.5450e-7 / 7.4712e-7 |
| Neumann | L prism | 23 / 26 | 5.0551e-7 / 3.9338e-7 | 6.2375e-7 / 5.0039e-7 |
| Dirichlet | torus | 13 / 13 | 9.5701e-7 / 7.5183e-7 | 8.6492e-8 / 6.8442e-8 |
| Dirichlet | cylinder | 25 / 30 | 1.4770e-6 / 4.0201e-6 | 3.1582e-7 / 2.4115e-7 |
| Dirichlet | L prism | 81 / 24 | 4.2983e-6 / 3.2054e-6 | 4.0032e-7 / 3.1274e-7 |

All 18 solves converged; the largest reported GMRES relative residual was
`1.9474122e-10`. At `N=64`, the torus is essentially unchanged or improved and
both L-prism formulations improve. The cylinder density errors increase when
the circular wall/cap edge is correctly treated as non-G1, although the
Dirichlet interior error improves. Strict non-G1 filtering is most visible on
the coarse L prism: its one-face Cauchy stencil reaches about `7.9h`, so the
`N=16` error is deliberately traded for correct patch topology; refinement
then reduces the error rapidly.

The full machine-readable results are written at runtime to
`output/neumann_exterior_zero_trace_3d/neumann_results.csv` and
`output/neumann_exterior_zero_trace_3d/dirichlet_normal_results.csv`.
