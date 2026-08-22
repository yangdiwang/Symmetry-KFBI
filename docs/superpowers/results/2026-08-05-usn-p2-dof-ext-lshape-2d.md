# USN-P2-DOF-EXT on a two-dimensional L-shaped boundary

Date: 2026-08-05

## Scope

This experiment applies `USN-P2-DOF-EXT` to the two-dimensional L-shaped
Neumann exterior-trace BVP.  The manufactured solution is the smooth
antisymmetric harmonic cubic already used by the L-shape driver.  No corner
singularity correction is enabled, so these numbers test the transfer and
iteration machinery on a nonsmooth geometry; they are not a convergence claim
for a generic singular L-domain solution.

The cornered interface is split into six maximal smooth open branches.  Each
branch has its own arc-length spline: P3 for `phi` and P2 for `psi`.  The spline
does not wrap and no derivative continuity is imposed across the six physical
corners.  The restrict route uses one P2 spatial stencil per side and one P2
normal interpolation, while all wrong-side samples are corrected with the
current interface-DOF Cauchy polynomial.  Interior-side normal samples are
converted directly to the exterior solution before the normal interpolation.

## Axis-aligned convergence

Tolerance: `1e-8`; GMRES restart: `50`; maximum iterations: `200`.

| N | GMRES | trace L-inf | trace order | normal L-inf | normal order | interior bulk L-inf | bulk order | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30  | 11 | 3.8121e-3 | -      | 1.5338e-2 | -      | 3.7892e-3 | -      | 0.0257 |
| 60  | 12 | 1.0861e-3 | 1.8114 | 3.5363e-3 | 2.1168 | 1.0978e-3 | 1.7873 | 0.0599 |
| 120 | 13 | 2.9197e-4 | 1.8953 | 8.5807e-4 | 2.0431 | 2.9317e-4 | 1.9048 | 0.2650 |
| 240 | 13 | 7.5612e-5 | 1.9491 | 2.0801e-4 | 2.0445 | 7.5752e-5 | 1.9524 | 0.5348 |
| 480 | 15 | 1.9209e-5 | 1.9769 | 5.1452e-5 | 2.0153 | 1.9231e-5 | 1.9779 | 1.1309 |

The trace, recovered normal trace, and interior bulk solution all approach
second order.  Every augmented and physical convergence check passes on these
five levels.

At every level:

- exact crossing owners equal all corrected wrong-side nodes;
- gap fallback = 0 and endpoint fallback = 0;
- relocated unified-P2 stencil = 0;
- domain-label mismatch = 0;
- spline preprocessing reports 0 periodic components and 6 open branches.

## Comparison at N = 480

| restrict route | GMRES | trace L-inf | normal L-inf | interior bulk L-inf | seconds |
|---|---:|---:|---:|---:|---:|
| USN-P2-DOF-EXT | 15 | 1.9209e-5 | 5.1452e-5 | 1.9231e-5 | 1.1309 |
| previous USN-P2 crossing owner | 15 | 1.9217e-5 | 5.1314e-5 | 1.9240e-5 | 2.3658 |
| joint-cubic crossing owner | 15 | 1.9285e-5 | 5.2144e-6 | 1.9297e-5 | 8.0623 |

Relative to the previous USN-P2 route, the new route changes trace and bulk
errors by less than 0.05%, changes the normal error by about +0.27%, and is
2.09 times faster in this run.  It is 7.13 times faster than joint-cubic.  The
unusually small joint-cubic normal error is helped by the cubic manufactured
solution and should not be interpreted as a general-order advantage.

## Rigid-transform stress test

The following table gives the new route at `N=240`.  `physical` is the strict
unbordered physical-residual check; the bordered GMRES system converged in all
six cases.

| geometry phase | GMRES | physical | trace L-inf | normal L-inf | bulk L-inf | bulk / baseline | crossing fallback |
|---|---:|:---:|---:|---:|---:|---:|---:|
| baseline | 13 | yes | 7.5612e-5 | 2.0801e-4 | 7.5752e-5 | 1.000 | 0 |
| translate `(2h,-h)` | 17 | yes | 7.5640e-5 | 2.0800e-4 | 7.5776e-5 | 1.000 | 0 |
| fixed translation | 21 | no | 1.6045e-4 | 2.1794e-4 | 1.5010e-4 | 1.982 | 0 |
| rotate 90 degrees | 17 | yes | 7.5594e-5 | 2.0941e-4 | 7.5746e-5 | 1.000 | 0 |
| rotate 17 degrees | 21 | no | 1.2812e-4 | 1.0127e-3 | 1.2773e-4 | 1.686 | 0 |
| rotate 17 degrees + translate | 27 | yes | 2.1299e-4 | 4.6431e-4 | 1.9068e-4 | 2.517 | 0 |

The arbitrary-phase cases require relocated P2 stencils but never use a
crossing fallback and never show a domain-label mismatch.  Their nonuniform
errors, and the nonzero bordered multiplier in the two failed physical checks,
show that the unresolved limitation is the discrete corner/grid-phase
compatibility, not crossing detection.  Thus the axis-aligned result is cleanly
second order, but the present L-corner treatment is not yet fully
rotation/translation robust.

## Reproduction

```powershell
$env:KFBIM_LSHAPE_RESTRICT = 'usn_p2_dof_ext'
.\build\apps\neumann_exterior_trace_lshape_2d.exe 30 60 120 240 480
.\build\apps\neumann_exterior_trace_lshape_2d.exe --rigid-study 30 60 120 240
.\build\apps\laplace_crossing_local_polynomial_2d_test.exe
Remove-Item Env:KFBIM_LSHAPE_RESTRICT
```

Generated data:

- `output/neumann_exterior_trace_lshape_2d_usn_p2_dof_ext.csv`
- `output/neumann_exterior_trace_lshape_2d_rigid_transform_usn_p2_dof_ext.csv`
- `output/neumann_exterior_trace_lshape_2d_usn_p2.csv`
- `output/neumann_exterior_trace_lshape_2d_p2_crossing_owner_joint_cubic.csv`
