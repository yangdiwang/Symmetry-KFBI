# 3D Neumann Crossing-Owner Value-Trace Results

## Implemented study route

`neumann_exterior_zero_trace_3d --neumann-owner-study [N ...]` now runs
the L-prism at the identity, 17-degree `(1,2,3)` rotation, and the same
rotation plus `t_xyz_1`.  Every `(case,N)` constructs one owner-enabled
pipeline and applies both exterior-value routes to it:

- `joint_tricubic_cauchy` (legacy center-owned correction);
- `joint_tricubic_crossing_owner` (precomputed crossing ownership).

The Neumann unknown, prescribed normal jump, compatibility correction,
mean constraint, and bordered second-kind equation are unchanged.  Both
routes use degree-three G1-nearest 48/28 Cauchy data, tricubic restriction,
GMRES tolerance `2e-10`, restart 80, and cap 80.

The study rewrites and flushes `summary.csv`, `gmres_residuals.csv`, and
`owner_diagnostics.csv` after every completed route under
`output/neumann_value_trace_crossing_owner_3d`.  A deterministic FNV-1a
fingerprint covers every trace template's grid IDs, interpolation weights,
legacy correction evaluation, owner DOFs, and owner evaluations.

## TDD and smoke evidence

Before implementation, `--neumann-owner-study 16` exited 1 with the old
geometry-selection error.  The Release target then built successfully, and
the combined `--neumann-owner-study 16 32` run exited 0.

The generated smoke matrix contained 12 unique summary rows and 12 owner
rows.  All routes converged within 80 iterations and ended below `2e-10`.
The 633 residual rows gave exactly `iterations + 1` entries for every route,
including iteration zero.  For every row:

- owner queries before and after GMRES were identical;
- the hexadecimal owner fingerprint before and after GMRES was identical;
- decision count sum equaled wrong-side count and geometry-query count;
- foreign non-G1 decisions equaled reroute audit terms;
- paired routes had identical geometry, setup, Cauchy, query, and fingerprint
  metadata because they used the same pipeline.

The `N=32` numerical comparison was:

| pose | route | GMRES | interior Linf | density Linf | exterior trace Linf |
|---|---|---:|---:|---:|---:|
| baseline | legacy | 39 | 1.299933e-5 | 1.545585e-5 | 1.393342e-6 |
| baseline | crossing owner | 33 | 1.397223e-5 | 1.749716e-5 | 1.315592e-6 |
| rot17 | legacy | 52 | 3.453142e-6 | 7.285416e-6 | 1.381749e-9 |
| rot17 | crossing owner | 42 | 4.258988e-6 | 6.348064e-6 | 1.121380e-7 |
| rot17 + translation | legacy | 50 | 5.987881e-6 | 1.459826e-5 | 1.427745e-7 |
| rot17 + translation | crossing owner | 39 | 6.414849e-6 | 8.381273e-6 | 1.633850e-7 |

At this level crossing ownership reduced GMRES by 6, 10, and 11 iterations.
Accuracy was mixed rather than uniformly better: interior error stayed of
the same scale, while density error improved for both rotated poses and
worsened modestly for the baseline.  The `N=16 -> 32` orders are present in
the CSV as a driver check; the coarse-grid values are geometry-sensitive and
are not used as the formal convergence conclusion.

## Owner preprocessing scope

The study uses the full certified NURBS intersection owner builder already
available on `main`.  The older branch's RegionClosestHybrid owner
preprocessor was not integrated because its geometry API diverges from the
current mapped Cartesian-edge/certified backend.  Hybrid owner preprocessing
therefore remains future performance work; no Hybrid timing or equivalence is
inferred here.  The shared full-certified setup costs noticeably more than a
single solve, so this is the main remaining optimization target.

The formal `N=32,64,128` matrix is intentionally left to the final full-run
stage; the same CLI and CSV gates apply without changing the implementation.
