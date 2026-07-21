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
  distance. Only periodic/G1 edge maps may be crossed.
- Local Cauchy reconstruction: one degree-three harmonic polynomial using 48
  value and 28 normal samples. The candidate pool always includes the first
  complete topological patch ring, including non-G1 neighbors, and expands by
  further whole rings only if needed. Physical distance ranks samples inside
  that pool, so unrelated nearby sheets cannot be selected.
- Restriction: `4x4x4` tricubic Cartesian interpolation followed by a joint
  cubic normal fit on the layers `+-{0.2, 0.6, 1.0, 1.4} h`; the recovered
  dimensionless linear coefficient is divided by `h`.
- GMRES relative tolerance: `2e-10` for both formulations.

The Neumann formulation iterates the value jump until the exterior value
trace is zero. The Dirichlet second-kind formulation prescribes the value
jump and iterates the normal jump until the exterior normal trace is zero.

## Native-NURBS geometry readiness

| Geometry | N | Patches | DOFs | Area relative error | Cauchy max radius/h | Selected patches min/max | Exact/gap/endpoint owners |
|---|---:|---:|---:|---:|---:|---:|---:|
| torus | 16 | 16 | 224 | 1.231e-2 | 4.332 | 4 / 5 | 107 / 59 / 0 |
| torus | 32 | 16 | 832 | 3.033e-3 | 3.619 | 2 / 4 | 566 / 120 / 0 |
| torus | 64 | 16 | 2912 | 9.647e-4 | 3.868 | 1 / 3 | 2350 / 410 / 0 |
| cylinder | 16 | 16 | 304 | 2.580e-3 | 4.148 | 3 / 5 | 192 / 92 / 0 |
| cylinder | 32 | 16 | 1160 | 7.774e-4 | 3.772 | 1 / 4 | 926 / 234 / 0 |
| cylinder | 64 | 16 | 4200 | 2.289e-4 | 4.032 | 1 / 3 | 4372 / 172 / 0 |
| L prism | 16 | 12 | 306 | 1.269e-15 | 3.758 | 2 / 5 | 222 / 0 / 0 |
| L prism | 32 | 12 | 1050 | 2.390e-14 | 3.962 | 1 / 4 | 982 / 0 / 0 |
| L prism | 64 | 12 | 3926 | 2.453e-14 | 4.177 | 1 / 3 | 3926 / 0 / 0 |

The torus consists of 16 rational quadratic quarter patches. The cylinder
has four patches in each of its two walls and two annular caps. The L prism
has 12 bilinear patches. All levels retain exactly 48/28 Cauchy samples and
require no endpoint-based crossing ownership. The coarse L-prism Cauchy
radius is `3.758h`; it was `7.920h` when Cauchy samples were incorrectly
confined to G1 components.

## Neumann exterior-zero-value-trace formulation

| Geometry | N | Iterations | Density Linf | Density order | Interior Linf | Interior order |
|---|---:|---:|---:|---:|---:|---:|
| torus | 16 | 27 | 9.7949e-6 | - | 9.3867e-6 | - |
| torus | 32 | 23 | 1.1208e-6 | 3.13 | 1.0353e-6 | 3.18 |
| torus | 64 | 22 | 2.6608e-7 | 2.07 | 2.8282e-7 | 1.87 |
| cylinder | 16 | 24 | 7.8737e-6 | - | 6.2779e-6 | - |
| cylinder | 32 | 25 | 3.5218e-6 | 1.16 | 3.4231e-6 | 0.88 |
| cylinder | 64 | 24 | 7.3803e-7 | 2.25 | 7.5368e-7 | 2.18 |
| L prism | 16 | 26 | 9.6601e-6 | - | 9.9130e-6 | - |
| L prism | 32 | 28 | 2.9185e-6 | 1.73 | 2.8650e-6 | 1.79 |
| L prism | 64 | 24 | 5.0373e-7 | 2.53 | 6.2186e-7 | 2.20 |

## Dirichlet exterior-zero-normal-trace formulation

| Geometry | N | Iterations | Normal-jump Linf | Jump order | Interior Linf | Interior order |
|---|---:|---:|---:|---:|---:|---:|
| torus | 16 | 17 | 1.2709e-5 | - | 1.7136e-6 | - |
| torus | 32 | 15 | 2.9812e-6 | 2.09 | 3.9820e-7 | 2.11 |
| torus | 64 | 13 | 7.5687e-7 | 1.98 | 7.2179e-8 | 2.46 |
| cylinder | 16 | 93 | 2.5515e-4 | - | 8.3289e-6 | - |
| cylinder | 32 | 31 | 2.3524e-5 | 3.44 | 4.0215e-6 | 1.05 |
| cylinder | 64 | 25 | 1.3994e-6 | 4.07 | 2.5127e-7 | 4.00 |
| L prism | 16 | 33 | 1.8034e-4 | - | 1.6981e-5 | - |
| L prism | 32 | 54 | 4.4093e-5 | 2.03 | 1.9593e-6 | 3.12 |
| L prism | 64 | 82 | 4.7520e-6 | 3.21 | 4.0178e-7 | 2.29 |

## Comparison with the previous analytic/manual DOF route at N=64

| Formulation | Geometry | Old/New iterations | Old/New density Linf | Old/New interior Linf |
|---|---|---:|---:|---:|
| Neumann | torus | 23 / 22 | 2.5978e-7 / 2.6608e-7 | 2.7685e-7 / 2.8282e-7 |
| Neumann | cylinder | 24 / 24 | 3.1627e-7 / 7.3803e-7 | 3.5450e-7 / 7.5368e-7 |
| Neumann | L prism | 23 / 24 | 5.0551e-7 / 5.0373e-7 | 6.2375e-7 / 6.2186e-7 |
| Dirichlet | torus | 13 / 13 | 9.5701e-7 / 7.5687e-7 | 8.6492e-8 / 7.2179e-8 |
| Dirichlet | cylinder | 25 / 25 | 1.4770e-6 / 1.3994e-6 | 3.1582e-7 / 2.5127e-7 |
| Dirichlet | L prism | 81 / 82 | 4.2983e-6 / 4.7520e-6 | 4.0032e-7 / 4.0178e-7 |

All 18 solves converged; the largest reported GMRES relative residual was
`1.9183131e-10`. Correctly separating the two graphs removes the coarse
L-prism failure caused by G1-only Cauchy sampling: its `N=16` Neumann density
error decreases from `5.0963e-4` to `9.6601e-6`. At `N=64`, the corrected
native-NURBS route is close to the previous analytic/manual route for the
torus and L prism. The cylinder Neumann errors remain larger, while its
Dirichlet density and interior errors improve.

The full machine-readable results are written at runtime to
`output/neumann_exterior_zero_trace_3d/neumann_results.csv` and
`output/neumann_exterior_zero_trace_3d/dirichlet_normal_results.csv`.
