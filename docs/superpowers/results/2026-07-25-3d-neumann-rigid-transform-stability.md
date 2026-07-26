# 3D Neumann L-Prism Rigid-Transform Stability Results

## Configuration

The production command was:

```text
neumann_exterior_zero_trace_3d.exe --neumann-rigid-study 32 64 128
```

The study used the native twelve-patch NURBS L-prism, the eight transforms
from the Dirichlet rigid study, the transformed non-polynomial harmonic
solution, degree-3 G1-nearest Cauchy fitting with 48 value and 28 normal
conditions, tricubic Cartesian restriction,
`JointTricubicCrossingOwner`, and `RegionClosestHybrid`.

GMRES used relative tolerance `2e-10`, restart 80, and maximum 80
iterations.  The complete run took `991.312 s` wall time.

## Accuracy and GMRES

`E_N` is the shifted interior maximum error and `p` is recomputed directly
from adjacent errors.  The iteration columns are GMRES iteration counts.

| Pose | E32 | it32 | E64 | p32-64 | it64 | E128 | p64-128 | it128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.397223e-5 | 33 | 5.142635e-7 | 4.7639 | 25 | 6.153279e-7 | -0.2588 | 57 |
| tx_p0137 | 1.089511e-5 | 28 | 4.762970e-7 | 4.5157 | 25 | 3.270336e-7 | 0.5424 | 44 |
| ty_m0083 | 1.823066e-5 | 37 | 6.720805e-7 | 4.7616 | 27 | 5.806518e-7 | 0.2110 | 63 |
| tz_p0061 | 1.659129e-5 | 37 | 6.571586e-7 | 4.6580 | 26 | 2.218302e-7 | 1.5668 | 36 |
| t_xyz_1 | 1.530823e-5 | 34 | 8.293522e-7 | 4.2062 | 28 | 1.609886e-7 | 2.3650 | 28 |
| t_xyz_2 | 4.947382e-5 | 52 | 9.925315e-7 | 5.6394 | 38 | 3.434186e-7 | 1.5311 | 54 |
| rot_axis123_17deg | 4.258988e-6 | 42 | 6.098072e-7 | 2.8041 | 38 | 1.055776e-7 | 2.5300 | 42 |
| rot_axis123_17deg_t_xyz_1 | 6.414849e-6 | 39 | 9.945403e-7 | 2.6893 | 45 | 1.968610e-7 | 2.3369 | 47 |

All 24 solves converged.  Iteration counts ranged from 25 to 63; the
maximum occurred for `ty_m0083/N=128`.  The largest final relative residual
was `1.981342e-10`, below the required `2e-10`.

## Geometry and owner integrity

Every one of the 24 rows passed the execution gate:

- zero native grid-label mismatch;
- zero unsafe label-changing edge;
- zero gap or endpoint crossing;
- zero triangle ownership fallback;
- finite solve and error metrics;
- unchanged owner diagnostics and geometry-query counts across GMRES;
- bitwise equality between default and explicit legacy value-trace probes;
- finite crossing-owner probe.

The existing single-rotation `--neumann-owner-study 32` was rerun after the
implementation.  Its 18 non-timing summary fields were identical to the
pre-change baseline, and `owner_diagnostics.csv` was byte-identical.

## Timing

The following times sum all eight poses at each level:

| N | Setup | Solve | Sum of per-pose totals |
|---:|---:|---:|---:|
| 32 | 86.649 s | 4.372 s | 91.065 s |
| 64 | 161.596 s | 22.797 s | 184.547 s |
| 128 | 494.013 s | 220.117 s | 714.851 s |
| all | 742.258 s | 247.287 s | 990.463 s |

Setup accounts for `74.94%` and solve for `24.97%` of summed per-pose
time.  At N=128 the six axis-aligned poses required `39.38-57.15 s` of
setup, while the two rotated poses required about `100 s`; rotated NURBS
preprocessing remains the dominant cost.

## Acceptance

The complete study deliberately retained the Dirichlet-style strict gates:
monotone error, `p64-128 >= 1.8`, and same-level error no more than three
times baseline.

| Pose | Monotone | p64-128 >= 1.8 | <=3x baseline | Overall |
|---|---|---|---|---|
| baseline | fail | fail | pass | fail |
| tx_p0137 | pass | fail | pass | fail |
| ty_m0083 | pass | fail | pass | fail |
| tz_p0061 | pass | fail | pass | fail |
| t_xyz_1 | pass | pass | pass | pass |
| t_xyz_2 | pass | fail | fail | fail |
| rot_axis123_17deg | pass | pass | pass | pass |
| rot_axis123_17deg_t_xyz_1 | pass | pass | pass | pass |

`t_xyz_2/N=32` had the worst same-level ratio, `3.540868`; its ratios
decreased on finer grids.  The executable therefore returned 1 after
writing all 24 rows.  This is a numerical acceptance failure, not a solver,
geometry, or owner-integrity failure.

## Conclusion

The crossing-owner Neumann route is operationally stable under all tested
translations and rotations: no GMRES runaway or geometry/owner failure
occurred.  It is not uniformly second order for every axis-aligned grid
phase over only `N=32,64,128`.

The axis-aligned cases show anomalously large `p32-64` values
(`4.21-5.64`) followed by reduced or negative `p64-128`.  This is consistent
with grid-phase cancellation at N=64 rather than genuine fourth- or
fifth-order convergence.  In contrast, the two generic 17-degree rotations
remove that alignment and retain stable `p64-128` values of `2.53` and
`2.34`.  The remaining issue is therefore pose-dependent leading-error
cancellation/dispersion in the axis-aligned restriction pipeline, not
incorrect inside/outside labels, owner-template mutation, or GMRES
instability.
