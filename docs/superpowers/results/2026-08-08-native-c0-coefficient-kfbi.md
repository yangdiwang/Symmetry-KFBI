# Native C0 coefficient-KFBI on the hollow cylinder and L-prism

## Implemented route

The native NURBS Cartesian-domain and crossing-owner machinery remains the
geometry backend.  Only the boundary unknown and exterior-trace closure are
changed:

1. GMRES iterates in orthonormalized reduced density coefficients.
2. The reduction expands an iterate to C0 density coefficients.  During
   spread, every exact Cartesian-grid/surface crossing uses its own
   crossing-centered local Cauchy row, and the density values requested by that
   row are supplied directly by sparse C0 tensor-spline stencils.
3. The fixed KFBI interface problem is solved with those crossing-centered
   spread corrections.
4. Crossing-owner restriction returns exterior traces on the fixed
   patch-owned trace samples.  The panel-centered Cauchy jet is retained only
   to form the local correction in this restrict stage; it is not used to
   supply density data or spread corrections.
5. A weighted least-squares projection maps the restricted exterior trace
   back to the reduced coefficient space and closes the GMRES operator.

The density resolution is initialized without trying candidate spaces.  With
Cartesian spacing `h`, patch count `P`, target crossing coverage `tau=14`, and

```text
I_Gamma = integral_Gamma (|n_x|+|n_y|+|n_z|) dS,
```

the number of cubic knot cells is

```text
e = max(1, floor(sqrt(I_Gamma/(P*tau*h^2)))),  ncoef = e+3.
```

The executable also applies a direct observability cap from the fixed
patch-center trace samples.  This changes the cylinder `N=32` choice from the
geometry-only value 5 to 4; no failed candidate fit is used to make the
choice.

## C0 feature treatment

- Smooth G1 connection: strong C0 coefficient gluing followed by a physical
  co-normal weak-C1 mortar constraint.  The cubic mortar uses at least four
  Gauss points on every common knot span.  This is essential for the
  one-span `ncoef=4` cylinder: a three-point rule can see only three of its
  four cubic trace modes.
- Sharp/non-G1 connection for a value trace: C0 only, with no derivative
  constraint.
- Sharp/non-G1 connection for a normal trace: broken double trace, because
  `grad(u).n` changes when the face normal jumps.
- L-prism half-edge/full-edge connections: both knot partitions are mapped to
  a common seam parameter.  Four interior collocation points on every common
  cubic span impose exact C0 equality.  Reversal and the one-to-many split are
  handled explicitly.

The resulting seam counts are:

| Geometry | Value C0 seams | Value weak-C1 seams | Normal broken seams | Normal weak-C1 seams |
|---|---:|---:|---:|---:|
| Hollow cylinder | 32 | 16 | 16 | 16 |
| L-prism | 26 | 4 | 22 | 4 |

Density-space tests include the production `ncoef=4` cylinder, reject mortar
orders below four, and verify both C0 value jumps and physical co-normal jumps
of random reduced fields.  Normalized reduction residuals remain below
`2.0e-15` and constant reproduction errors below `3.1e-14`.

## Numerical results

All cases use exact-crossing-centered cubic local Cauchy spread, cubic
crossing-owner exterior restriction, GMRES tolerance `2e-10`, and automatic
density resolution.  Thus the coefficient-to-spread path is the direct C0
stencil path described above; panel-centered jets participate only in the
restrict correction.

| Geometry | N | ncoef | Value/normal K | Neumann it. / residual | Exterior value trace | Neumann interior Linf | Dirichlet it. / residual | Exterior normal trace | Dirichlet interior Linf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cylinder | 32 | 4 | 96 / 128 | 25 / 5.849168e-11 | 1.309671e-3 | 4.773119e-3 | 32 / 1.283114e-10 | 5.970871e-2 | 7.354810e-3 |
| Cylinder | 64 | 7 | 480 / 560 | 25 / 7.633151e-11 | 1.308384e-4 | 2.309519e-4 | 24 / 1.527745e-10 | 1.275841e-2 | 1.017245e-3 |
| L-prism | 32 | 5 | 170 / 260 | 29 / 1.002090e-10 | 1.192061e-6 | 1.478043e-5 | 22 / 1.660578e-10 | 6.971416e-6 | 4.741838e-6 |
| L-prism | 64 | 7 | 398 / 532 | 24 / 1.008146e-10 | 1.524878e-7 | 3.875309e-7 | 24 / 7.290485e-11 | 7.756792e-7 | 3.331874e-7 |

All projected coefficient systems reached the requested GMRES tolerance.
The raw trace at every panel-center sample is a truncation diagnostic, not the
Krylov residual, and is therefore reported separately as `sample_condition`.
The two independently reconstructed trace routes agree to below `7e-14` in
all four final runs.

The L-prism Dirichlet coefficient iteration counts are 22 and 24.  The
point-DOF baseline under the same exact-crossing-centered local Cauchy and
restrict configuration is about 24 and 23 at `N=32,64`, so the current
comparison is iteration-neutral rather than a reduction from 54 and 82.
Those 54/82 counts belong to the historical topological-nearest Cauchy
configuration and are not a like-for-like baseline for this table.  On the
cylinder, increasing from `ncoef=4` to 7 reduces the Neumann and Dirichlet
interior errors by about 20.7 and 7.2 times.  For the L-prism the corresponding
reductions are about 38.1 and 14.2 times.

The dedicated coefficient executable rewrites the machine-readable table from
the accumulated results after every completed case; it is not maintained by
hand.  For the default route it is written to
`output/kfbi_native_c0_exterior_trace_3d/coefficient_convergence_results.csv`.
Running `kfbi_native_c0_exterior_trace_3d all 32 64` reproduces all four rows.
