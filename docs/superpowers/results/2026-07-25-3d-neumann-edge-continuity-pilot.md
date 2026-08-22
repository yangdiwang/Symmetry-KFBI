# 3D Neumann Non-G1 Edge-Continuity Pilot Results

## Outcome

The `N=32,64` A/B pilot **rejects** the non-G1 edge projection as a
production improvement in its present form. The projection reduces the
solved edge mismatch to roundoff, by factors from `1.478849e9` to
`2.074951e10`, without a structural, geometry, owner, or solver-execution
failure. It nevertheless increases every measured same-level error by
more than the allowed 10%, does not improve the `ty_m0083` GMRES count,
and has worse refinement ratios than the patch-independent space.

This is a negative numerical result, not an implementation failure. The
complete driver wrote all 12 requested rows and then returned 1 because
the independently confirmed `gmres`, `error_guard`, and `trend`
acceptance gates failed. `N=128` was therefore not run.

The next causal experiment is **phase-continuous blending of
crossing-owner interpolation**. This is the prescribed next step when
edge continuity removes the mismatch but does not improve GMRES or
error; the result does not justify loosening gates, adding a penalty, or
presenting the projector as a production improvement.

## Configuration and command

The pilot ran on branch `codex/neumann-rigid-transform-study` from
implementation commit
`ff65201abd10be384ec5fa7f5f47c7017c6d9afb`
(`fix: verify Neumann shared preprocess snapshots`) with the Release
binary. The exact PowerShell command was:

```powershell
$env:KFBIM_3D_NEUMANN_EDGE_CONTINUITY_OUTPUT_DIR = `
    'output/neumann_edge_continuity_3d'
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --neumann-edge-continuity-study 32 64
```

The command completed the six A/B checkpoints and all 12 solves, then
exited 1 after `116.699 s` wall time because aggregate numerical
acceptance was false.

The fixed study configuration was:

- native twelve-patch NURBS L-prism;
- route `joint_tricubic_crossing_owner`;
- owner policy `region_closest_hybrid`;
- G1-nearest degree-3 Cauchy fit with 48 value and 28 normal conditions;
- relative GMRES tolerance `2e-10`, restart 80, and cap 80;
- poses `baseline`, `ty_m0083` (translation `(0,-0.083,0)`), and
  `rot_axis123_17deg` (17 degrees about normalized axis `(1,2,3)`,
  centered at `(0.07,-0.07,0.02)`);
- levels `N=32,64`, each comparing `patch_independent` with
  `non_g1_edge_projected`.

## Independent CSV audit

All CSV files were read with `Import-Csv`; the reported acceptance was
not trusted as the source of any comparison.

| CSV | Rows | Independent check |
|---|---:|---|
| `summary.csv` | 12 | Three exact cases, two exact levels, two modes per `(case,N)`, 12 unique `(case,N,density_space)` keys |
| `edge_diagnostics.csv` | 1,716 | 132 complete `(case,N,connection)` sample groups; recomputed maxima and weighted RMS values match `summary.csv` |
| `gmres_residuals.csv` | 403 | 12 contiguous histories from iteration 0 through the recorded terminal iteration; all terminal residuals match `summary.csv` |
| `owner_diagnostics.csv` | 12 | Same 12 keys; fingerprints, digests, query counts, and all before/after invariants agree |
| `acceptance.csv` | 1 | Every status agrees with independently recomputed gates |

All values needed by the execution and acceptance gates were finite.
The maximum mapped-point gap in the 1,716 edge samples was
`2.547623e-16`. No edge sample or summary row used reduced order.

Topology was identical within each level:

| N | Constraint rows | Rank | Expected/covered non-G1 connections | Duplicate intervals | G1 rows | Unrelated rows | Reduced-order rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 196 | 192 | 22/22 | 0 | 0 | 0 | 0 |
| 64 | 376 | 366 | 22/22 | 0 | 0 | 0 | 0 |

The independently observed maximum absolute projection defects were:

| Defect | Maximum |
|---|---:|
| Constant constraint | `6.800116e-16` |
| Projected constraint | `2.927392e-14` |
| Projection idempotence | `1.426647e-14` |
| Constant projection | `1.332268e-15` |

All are below `1e-11`. Every owner row had zero label mismatch, unsafe
label-changing edges, gap crossings, endpoint crossings, and triangle
fallback crossings. Workload fingerprints, output digests, wrong-side
query counts, and geometry-query counts were unchanged across GMRES and
agreed between the A/B modes. All geometry, owner, shared-preprocess,
probe, and default-route bitwise-equality flags passed.

## Errors, orders, GMRES, and solved mismatch

Orders are recomputed within one pose and one density space as
`log(E32/E64)/log(h32/h64)`. They are unavailable at the coarse level.
The requested compact review columns are shown below; `density` and
`interior` are maximum (`Linf`) errors.

| Pose | N | Density space | Density Linf | Order | Interior Linf | Order | Solved edge Linf | Iterations | Final residual |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | projected | `2.473335e-5` | -- | `1.870775e-5` | -- | `2.040792e-15` | 32 | `1.656026e-10` |
| baseline | 32 | independent | `1.749716e-5` | -- | `1.397223e-5` | -- | `1.208004e-5` | 33 | `1.975744e-10` |
| rot_axis123_17deg | 32 | projected | `1.517800e-5` | -- | `9.310988e-6` | -- | `6.734107e-16` | 36 | `9.457099e-11` |
| rot_axis123_17deg | 32 | independent | `6.348064e-6` | -- | `4.258988e-6` | -- | `1.397294e-5` | 42 | `1.353158e-10` |
| ty_m0083 | 32 | projected | `3.143116e-5` | -- | `2.280128e-5` | -- | `9.305698e-16` | 37 | `9.377339e-11` |
| ty_m0083 | 32 | independent | `2.358105e-5` | -- | `1.823066e-5` | -- | `1.198401e-5` | 37 | `1.598963e-10` |
| baseline | 64 | projected | `1.194478e-6` | 4.3720 | `9.522957e-7` | 4.2961 | `1.107358e-15` | 24 | `1.338331e-10` |
| baseline | 64 | independent | `4.145732e-7` | 5.3994 | `5.142635e-7` | 4.7639 | `1.637614e-6` | 25 | `9.413846e-11` |
| rot_axis123_17deg | 64 | projected | `2.449403e-6` | 2.6315 | `1.694225e-6` | 2.4583 | `6.410649e-16` | 35 | `1.875522e-10` |
| rot_axis123_17deg | 64 | independent | `6.487663e-7` | 3.2905 | `6.098072e-7` | 2.8041 | `1.902968e-6` | 38 | `1.280250e-10` |
| ty_m0083 | 64 | projected | `1.551684e-6` | 4.3403 | `1.352660e-6` | 4.0752 | `4.412237e-16` | 25 | `1.032075e-10` |
| ty_m0083 | 64 | independent | `5.829655e-7` | 5.3381 | `6.720805e-7` | 4.7616 | `1.672472e-6` | 27 | `1.449722e-10` |

The fine-level `L2` errors and independently recomputed orders, which
also participate in the error and trend gates, were:

| Pose | Density space | Density L2 | Order | Interior L2 | Order |
|---|---|---:|---:|---:|---:|
| baseline | projected | `3.514260e-7` | 4.2793 | `2.846752e-7` | 4.1026 |
| baseline | independent | `1.331233e-7` | 5.1476 | `1.358619e-7` | 4.6824 |
| rot_axis123_17deg | projected | `2.551003e-7` | 3.4454 | `1.227691e-7` | 3.5947 |
| rot_axis123_17deg | independent | `1.070601e-7` | 3.5682 | `8.685107e-8` | 3.1573 |
| ty_m0083 | projected | `4.543690e-7` | 4.2043 | `3.482690e-7` | 4.0872 |
| ty_m0083 | independent | `1.799527e-7` | 5.1041 | `1.592805e-7` | 4.8258 |

Every GMRES row converged below the 80-iteration cap and `2e-10`
residual threshold. All 12 residual histories decreased monotonically,
had no restart, and contained no non-finite point. The global worst count
improved from 42 independent to 37 projected, but the required
`ty_m0083` comparison tied at 37 versus 37 instead of decreasing
strictly. The GMRES gate therefore fails.

The exact trace mismatch, solved mismatch before/after projection, and
directly recomputed reduction were:

| Pose | N | Exact edge Linf | Independent solved Linf | Projected solved Linf | Reduction |
|---|---:|---:|---:|---:|---:|
| baseline | 32 | `1.244475e-5` | `1.208004e-5` | `2.040792e-15` | `5.919290e9` |
| baseline | 64 | `1.644484e-6` | `1.637614e-6` | `1.107358e-15` | `1.478849e9` |
| rot_axis123_17deg | 32 | `1.244475e-5` | `1.397294e-5` | `6.734107e-16` | `2.074951e10` |
| rot_axis123_17deg | 64 | `1.644484e-6` | `1.902968e-6` | `6.410649e-16` | `2.968448e9` |
| ty_m0083 | 32 | `1.244475e-5` | `1.198401e-5` | `9.305698e-16` | `1.287814e10` |
| ty_m0083 | 64 | `1.644484e-6` | `1.672472e-6` | `4.412237e-16` | `3.790531e9` |

The exact-density coarse/fine mismatch ratio is `7.567573` for every
pose, above the required 6. Every solved mismatch reduction is far above
`1e4`.

## Same-level error comparisons and acceptance

The projected/independent ratios for
`density_linf,density_l2,interior_linf,interior_l2` were:

| Pose | N | Density Linf | Density L2 | Interior Linf | Interior L2 |
|---|---:|---:|---:|---:|---:|
| baseline | 32 | 1.413563 | 1.446078 | 1.338924 | 1.401850 |
| baseline | 64 | 2.881224 | 2.639854 | 1.851766 | 2.095327 |
| rot_axis123_17deg | 32 | 2.390966 | 2.188267 | 2.186197 | 1.914163 |
| rot_axis123_17deg | 64 | 3.775479 | 2.382777 | 2.778297 | 1.413559 |
| ty_m0083 | 32 | 1.332899 | 1.353266 | 1.250711 | 1.310449 |
| ty_m0083 | 64 | 2.661708 | 2.524936 | 2.012646 | 2.186514 |

All 24 ratios exceed the permitted `1.10`, so `error_guard` fails.
For each of the four norms in all three poses, the projected
`E64/E32` refinement ratio was independently compared with the
independent ratio using only `64*epsilon` relative allowance. At least
one comparison fails (and the aggregate `trend` gate fails); no threshold
was changed.

| Gate | Recomputed | Recorded | Evidence |
|---|---|---|---|
| Completeness | pass | pass | 12 unique requested rows |
| Topology | pass | pass | 22/22 connections, no duplicate/G1/unrelated rows |
| Projector | pass | pass | All four defects at most `2.927392e-14` |
| Exact trace order | pass | pass | All exact mismatch ratios `7.567573 >= 6` |
| GMRES | **fail** | **fail** | Worst 37 vs 42 passes; `ty_m0083` 37 vs 37 fails strict decrease |
| Error guard | **fail** | **fail** | All projected/independent error ratios are `1.250711` to `3.775479` |
| Edge reduction | pass | pass | Reductions `1.478849e9` to `2.074951e10` |
| Refinement trend | **fail** | **fail** | Projected refinement is not no-worse for every norm and pose |
| Geometry/owner | pass | pass | All 12 rows and owner snapshots pass |
| Overall | **fail** | **fail** | Three numerical gates fail |

Because `acceptance.csv` records `overall_pass=fail`, the conditional
`N=128` command was not run.

## Timing

All timings are seconds. Geometry, pipeline, and projector setup are
shared within an A/B pair and therefore repeat in the two rows; totals are
the per-mode end-to-end values recorded by the driver.

| Pose | N | Density space | Geometry | Pipeline | Projector | Solve | Total |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline | 32 | projected | 0.119470 | 7.976195 | 0.007338 | 0.433915 | 8.536918 |
| baseline | 32 | independent | 0.119470 | 7.976195 | 0.007338 | 0.425725 | 8.528728 |
| rot_axis123_17deg | 32 | projected | 1.242473 | 15.837856 | 0.008334 | 0.463865 | 17.552528 |
| rot_axis123_17deg | 32 | independent | 1.242473 | 15.837856 | 0.008334 | 0.521051 | 17.609714 |
| ty_m0083 | 32 | projected | 0.115999 | 8.091801 | 0.010446 | 0.609953 | 8.828198 |
| ty_m0083 | 32 | independent | 0.115999 | 8.091801 | 0.010446 | 0.612712 | 8.830957 |
| baseline | 64 | projected | 0.782578 | 12.535647 | 0.029739 | 2.054171 | 15.402135 |
| baseline | 64 | independent | 0.782578 | 12.535647 | 0.029739 | 2.112516 | 15.460480 |
| rot_axis123_17deg | 64 | projected | 5.487643 | 30.488575 | 0.031035 | 3.649860 | 39.657113 |
| rot_axis123_17deg | 64 | independent | 5.487643 | 30.488575 | 0.031035 | 3.911474 | 39.918727 |
| ty_m0083 | 64 | projected | 0.754086 | 13.546892 | 0.046752 | 2.112951 | 16.460681 |
| ty_m0083 | 64 | independent | 0.754086 | 13.546892 | 0.046752 | 2.261328 | 16.609058 |

## Final verification

The prescribed final verification was run after the report was drafted:

```powershell
cmake --build build --config Release -- /m:1
.\build\apps\Release\neumann_edge_continuity_3d_test.exe
.\build\apps\Release\neumann_rigid_transform_study_3d_test.exe
.\build\apps\Release\dirichlet_rigid_transform_study_3d_test.exe
.\build\apps\Release\harmonic_trace_correction_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\restrict_owner_geometry_preprocessor_3d_test.exe
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\native_nurbs_surface_transform_3d_test.exe
git diff --check
```

The Release build succeeded, all eight executables exited 0, and
`git diff --check` was clean.
