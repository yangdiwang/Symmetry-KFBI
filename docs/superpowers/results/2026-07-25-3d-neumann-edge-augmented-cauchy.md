# 3D Neumann edge-augmented Cauchy pilot

## Decision

Keep `non_g1_auxiliary_values` experimental. The complete coarse pilot
failed the fixed error guard and rigid-pose spread gates, so the
application correctly stopped before `N=128`. No weights, thresholds, or
other algorithm parameters were changed.

The augmented route substantially reduced each of the six reported
per-case/N global incident-edge Linf discrepancies and reduced the worst
GMRES count from 42 to 27. It did not improve every finer-grained edge
comparison: 3 of 132 connection-level Linf comparisons and 72 of 1716
sample-level discrepancies increased. Its baseline `N=64` density L2
error ratio was
`1.1047788080297003`, above the fixed `1.10` guard. At `N=64`, the
augmented pose spreads were also worse for density Linf, density L2, and
interior Linf.

## Provenance and execution

- Branch: `codex/neumann-rigid-transform-study`
- Authoritative code commit:
  `e771f6799a969ae54e6cd1670f2cb0b36431297b`
- Configuration: Visual Studio x64 `Release`
- Build command: `cmake --build build --config Release -- /m:1`
- Pilot command:

  ```powershell
  $env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR = `
    'output/neumann_edge_cauchy_3d'
  .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --neumann-edge-cauchy-study 32 64 128
  ```

- Authoritative pilot result: `PILOT_EXIT=1` and
  `PILOT_WALL_SECONDS=124.706403`. The exact gated Release invocation
  completed naturally.
- Console log:
  `output/neumann_edge_cauchy_3d/final_pilot_console.log`
- The final checkpoints contain all 12 requested `N=32,64` rows.
- `N=128` did not run because the complete `N=32,64` acceptance result
  was `fail`.

This naturally completed latest-code run supersedes every earlier pilot
attempt. No earlier timeout, wrapper status, checkpoint, or timing is
used anywhere in the evidence below. The documentation update did not
rerun the pilot or tests.

## Fixed study configuration

- Native twelve-patch NURBS L-prism
- Cases: `baseline`, `ty_m0083`, `rot_axis123_17deg`
- Modes: `none`, `non_g1_auxiliary_values`
- Shared density space and geometry/FFT/crossing/owner preprocessing
- Face fit: harmonic degree 3, `G1Nearest`, 48 value and 28 normal rows
- Edge auxiliary fit: 24 value and 14 normal rows per incident side
- Each physical non-G1 connection uses
  `max(4, ceil(connection_length / h))` midpoint samples; each eligible
  local patch-centered fit attaches the nearest four samples from every
  incident connection, with soft value-row scale exactly `1.0`
- Route: `JointTricubicCrossingOwner`
- Owner policy: `RegionClosestHybrid`
- GMRES tolerance `2e-10`, restart 80, cap 80
- Gate thresholds: reproduction defect at most `1e-11`; every
  augmented/legacy error ratio at most `1.10`; every augmented coarse
  order at least `1.8`; augmented edge discrepancy strictly below
  legacy; augmented worst GMRES count no greater than legacy
- Relative comparison allowance: `64 * epsilon`

## Independent CSV audit

The audit loaded every file with PowerShell `Import-Csv` and recomputed
all derived evidence without using `acceptance.csv` as a source of truth.
It found no audit failures.

| CSV | Rows | Independent checks |
|---|---:|---|
| `summary.csv` | 12 | 12 unique coarse keys; three cases and two modes per level; finite raw metrics; every `residual_history_valid` flag true; all errors, orders, and ratios recomputed |
| `edge_values.csv` | 3432 | every connection group complete with literal `0..sample_count-1`; value differences recomputed; group Linf matches summary |
| `gmres_residuals.csv` | 367 | every summary history-valid flag true; every history finite, contiguous, and exactly `iterations + 1` rows; initial residual is one and terminal residual matches summary |
| `owner_diagnostics.csv` | 12 | keys match summary; stable/before/after/final fingerprints, digests, wrong-side queries, geometry queries, and factorization counts unchanged |
| `phase_profile.csv` | 324 | 18 phases per scope; phase sums match setup/runtime walls; legacy edge calls are zero; augmented edge calls are positive |
| `acceptance.csv` | 1 | every recorded status matches the independent recomputation |

The only non-finite CSV literals are intentional `nan` values for
inapplicable derived fields: orders on `N=32`, and legacy
augmented-to-legacy ratios.

## Linf, edge, GMRES, and wall results

Orders are reported on the fine row for the `N=32 -> 64` pair. Total
time is shared setup plus the individual mode runtime.

| Case | N | Mode | Density Linf | Order | Interior Linf | Order | Edge discrepancy | GMRES iters | Residual | Total s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | none | 1.7497e-5 | — | 1.3972e-5 | — | 3.626e-6 | 33 | 1.976e-10 | 9.721 |
| baseline | 32 | augmented | 7.8786e-6 | — | 6.9076e-6 | — | 1.129e-6 | 26 | 1.229e-10 | 9.755 |
| ty_m0083 | 32 | none | 2.3581e-5 | — | 1.8231e-5 | — | 4.110e-6 | 37 | 1.599e-10 | 10.033 |
| ty_m0083 | 32 | augmented | 8.8742e-6 | — | 7.6540e-6 | — | 1.110e-6 | 27 | 1.249e-10 | 9.937 |
| rot_axis123_17deg | 32 | none | 6.3481e-6 | — | 4.2590e-6 | — | 5.758e-6 | 42 | 1.353e-10 | 18.827 |
| rot_axis123_17deg | 32 | augmented | 2.7241e-6 | — | 2.4491e-6 | — | 1.157e-6 | 27 | 1.394e-10 | 18.679 |
| baseline | 64 | none | 4.1457e-7 | 5.399 | 5.1426e-7 | 4.764 | 1.880e-7 | 25 | 9.414e-11 | 17.621 |
| baseline | 64 | augmented | 4.4683e-7 | 4.140 | 5.4698e-7 | 3.659 | 8.697e-8 | 23 | 1.561e-10 | 17.581 |
| ty_m0083 | 64 | none | 5.8297e-7 | 5.338 | 6.7208e-7 | 4.762 | 2.445e-7 | 27 | 1.450e-10 | 18.974 |
| ty_m0083 | 64 | augmented | 5.5977e-7 | 3.987 | 6.4876e-7 | 3.560 | 9.237e-8 | 24 | 8.461e-11 | 18.825 |
| rot_axis123_17deg | 64 | none | 6.4877e-7 | 3.291 | 6.0981e-7 | 2.804 | 6.247e-7 | 38 | 1.280e-10 | 40.391 |
| rot_axis123_17deg | 64 | augmented | 3.1795e-7 | 3.099 | 3.2156e-7 | 2.929 | 8.192e-8 | 26 | 1.993e-10 | 39.850 |

## L2 errors and orders

| Case | N | Mode | Density L2 | Order | Interior L2 | Order |
|---|---:|---|---:|---:|---:|---:|
| baseline | 32 | none | 4.7188e-6 | — | 3.4886e-6 | — |
| baseline | 32 | augmented | 2.5183e-6 | — | 1.9240e-6 | — |
| ty_m0083 | 32 | none | 6.1896e-6 | — | 4.5172e-6 | — |
| ty_m0083 | 32 | augmented | 2.8547e-6 | — | 2.1502e-6 | — |
| rot_axis123_17deg | 32 | none | 1.2699e-6 | — | 7.7485e-7 | — |
| rot_axis123_17deg | 32 | augmented | 6.6206e-7 | — | 4.9472e-7 | — |
| baseline | 64 | none | 1.3312e-7 | 5.148 | 1.3586e-7 | 4.682 |
| baseline | 64 | augmented | 1.4707e-7 | 4.098 | 1.4612e-7 | 3.719 |
| ty_m0083 | 64 | none | 1.7995e-7 | 5.104 | 1.5928e-7 | 4.826 |
| ty_m0083 | 64 | augmented | 1.6907e-7 | 4.078 | 1.5842e-7 | 3.763 |
| rot_axis123_17deg | 64 | none | 1.0706e-7 | 3.568 | 8.6851e-8 | 3.157 |
| rot_axis123_17deg | 64 | augmented | 9.2074e-8 | 2.846 | 8.6766e-8 | 2.511 |

All twelve augmented orders are above `1.8`; their range is
`2.5114063423341255` to `4.140143091096661`.

## Error and edge ratios

| Case | N | Density Linf | Density L2 | Interior Linf | Interior L2 | Edge discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 32 | 0.450276 | 0.533676 | 0.494377 | 0.551505 | 0.311307 |
| baseline | 64 | 1.077800 | **1.104779** | 1.063608 | 1.075514 | 0.462723 |
| ty_m0083 | 32 | 0.376326 | 0.461217 | 0.419840 | 0.475993 | 0.270043 |
| ty_m0083 | 64 | 0.960206 | 0.939549 | 0.965305 | 0.994596 | 0.377827 |
| rot_axis123_17deg | 32 | 0.429130 | 0.521348 | 0.575053 | 0.638475 | 0.200970 |
| rot_axis123_17deg | 64 | 0.490081 | 0.860023 | 0.527312 | 0.999025 | 0.131138 |

The bold baseline `N=64` density L2 ratio is the sole individual error
guard failure. Every reported per-case/N global edge-discrepancy ratio is
strictly below one.

That global Linf improvement is not uniform at connection or sample
resolution. Independent grouping of `edge_values.csv` found three
connection-level Linf regressions:

| Case | N | Connection | Legacy Linf | Augmented Linf |
|---|---:|---:|---:|---:|
| baseline | 32 | 21 | 6.005300416112025e-7 | 6.583339985759273e-7 |
| baseline | 32 | 23 | 1.0550447701596077e-7 | 2.3727588181754466e-7 |
| ty_m0083 | 32 | 23 | 2.495042298322758e-7 | 2.687349273250428e-7 |

Thus 3 of 132 connection comparisons regress, and direct comparison of
matched physical samples finds 72 regressions among 1716 sample pairs.
The acceptance gate remains `pass` because its specified statistic is
the per-case/N global incident-edge Linf, which improves in all six
comparisons.

## Rigid-pose spreads

| N | Quantity | Legacy spread | Augmented spread | Pass |
|---:|---|---:|---:|---|
| 32 | density Linf | 3.714684 | 3.257598 | yes |
| 32 | density L2 | 4.874061 | 4.311901 | yes |
| 32 | interior Linf | 4.280514 | 3.125159 | yes |
| 32 | interior L2 | 5.829823 | 4.346227 | yes |
| 32 | GMRES iteration range | 9 | 1 | yes |
| 64 | density Linf | 1.564902 | 1.760563 | **no** |
| 64 | density L2 | 1.680857 | 1.836285 | **no** |
| 64 | interior Linf | 1.306880 | 2.017555 | **no** |
| 64 | interior L2 | 1.833950 | 1.825819 | yes |
| 64 | GMRES iteration range | 13 | 3 | yes |

## Timing

Shared setup phases, in seconds:

| Case | N | Geometry | Surface/stencils | Grid/labels | Pipeline init | Crossings | Owner preprocess | Trace templates | Other setup | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | 0.0408 | 0.0146 | 0.0583 | 1.3049 | 0.0049 | 7.7579 | 0.1312 | 0.0001 | 9.3127 |
| baseline | 64 | 0.1742 | 0.2182 | 0.3649 | 3.2328 | 0.0193 | 10.6727 | 0.4144 | 0.0003 | 15.0968 |
| ty_m0083 | 32 | 0.0520 | 0.0167 | 0.0561 | 1.2360 | 0.0049 | 7.8831 | 0.1326 | 0.0001 | 9.3816 |
| ty_m0083 | 64 | 0.1700 | 0.2133 | 0.3799 | 3.2122 | 0.0261 | 12.1574 | 0.4184 | 0.0003 | 16.5776 |
| rot_axis123_17deg | 32 | 1.1893 | 0.0147 | 0.0740 | 1.2494 | 0.0059 | 15.4626 | 0.1331 | 0.0001 | 18.1290 |
| rot_axis123_17deg | 64 | 4.7019 | 0.2415 | 0.6743 | 3.1031 | 0.0281 | 28.3673 | 0.4271 | 0.0003 | 37.5436 |

Mode runtime phases, in seconds:

| Case | N | Mode | Edge values | Cauchy | Spread | FFT | Restrict continued | Recovery | GMRES/other | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | none | 0.000000 | 0.046669 | 0.002051 | 0.272498 | 0.083564 | 0.000195 | 0.002978 | 0.407954 |
| baseline | 32 | augmented | 0.000944 | 0.107040 | 0.001971 | 0.254124 | 0.076076 | 0.000192 | 0.002348 | 0.442695 |
| ty_m0083 | 32 | none | 0.000000 | 0.055923 | 0.002583 | 0.490080 | 0.098678 | 0.000258 | 0.003463 | 0.650986 |
| ty_m0083 | 32 | augmented | 0.000915 | 0.104510 | 0.001923 | 0.366888 | 0.078778 | 0.000208 | 0.002432 | 0.555653 |
| rot_axis123_17deg | 32 | none | 0.000000 | 0.058931 | 0.003220 | 0.512769 | 0.118543 | 0.000292 | 0.003739 | 0.697494 |
| rot_axis123_17deg | 32 | augmented | 0.000901 | 0.103643 | 0.002195 | 0.358151 | 0.082657 | 0.000185 | 0.002423 | 0.550155 |
| baseline | 64 | none | 0.000000 | 0.136964 | 0.007440 | 2.158092 | 0.193910 | 0.000587 | 0.026832 | 2.523825 |
| baseline | 64 | augmented | 0.001815 | 0.249500 | 0.006781 | 2.018223 | 0.176627 | 0.000520 | 0.030680 | 2.484148 |
| ty_m0083 | 64 | none | 0.000000 | 0.162150 | 0.009358 | 1.956664 | 0.235631 | 0.000769 | 0.032096 | 2.396669 |
| ty_m0083 | 64 | augmented | 0.001887 | 0.285469 | 0.008959 | 1.721346 | 0.201194 | 0.000599 | 0.027776 | 2.247229 |
| rot_axis123_17deg | 64 | none | 0.000000 | 0.187294 | 0.012114 | 2.302183 | 0.306285 | 0.000782 | 0.038221 | 2.846880 |
| rot_axis123_17deg | 64 | augmented | 0.001882 | 0.294630 | 0.009549 | 1.745390 | 0.223641 | 0.000559 | 0.030458 | 2.306108 |

Legacy edge-value calls are exactly zero. Augmented calls are positive:
32, 33, and 33 at `N=32`, and 29, 30, and 32 at `N=64`. The measured
edge-value overhead is `0.000901--0.001887 s`. Every setup and runtime
phase sum agrees with its corresponding wall interval within the
required tolerance, and total time equals shared setup plus mode runtime.

## Structure, conditioning, and invariants

| N | Edge samples/row | Affected centers | Corner centers | Max edge condition | Max local condition |
|---:|---:|---:|---:|---:|---:|
| 32 | 196 | 990 | 758 | 44.1993730783 | 89.1166616768 |
| 64 | 376 | 2486 | 886 | 44.1993730783 | 70.3580724964 |

Every row covers all 22 expected non-G1 connections. Unrelated
sample/attachment and rank-deficient-fit counts are zero. Far centers
are bitwise legacy. The maximum harmonic cubic reproduction defect is
`4.8316906031686813e-13`.

Owner fingerprints, output digests, wrong-side query counts, geometry
query counts, and setup factorization counts are unchanged across both
GMRES solves for every pair. All geometry, owner, shared-preprocess,
exact-trace-sharing, and normal-jump-sharing flags pass.

## Recomputed acceptance

| Gate | Recomputed | Recorded | Evidence |
|---|---|---|---|
| completeness | pass | pass | 12 unique coarse keys |
| structure | pass | pass | all topology/count/bitwise conditions hold |
| reproduction | pass | pass | max defect `4.832e-13 <= 1e-11` |
| GMRES | pass | pass | all converged; max residual `1.9934253592188171e-10`; worst iterations 42 legacy, 27 augmented |
| error guard | **fail** | **fail** | baseline `N=64` density L2 ratio `1.1047788080 > 1.10` |
| order | pass | pass | all augmented orders at least `2.5114` |
| rigid spread | **fail** | **fail** | three `N=64` error spreads worsen |
| edge discrepancy | pass | pass | all six augmented per-case/N global Linf values are strictly below legacy |
| geometry/owner | pass | pass | all snapshots and flags stable |
| extended evidence | not evaluated | not evaluated | coarse failure prevented `N=128` |
| overall | **fail** | **fail** | conjunction of coarse gates |

The numerical mechanism is therefore mixed: the auxiliary edge values
improve the specified global edge-continuity metric and solver iteration
behavior, despite the disclosed local regressions, but do not meet the
fixed accuracy guard or rigid-pose robustness requirement at `N=64`.
Under the prescribed adoption rule, this is a valid coarse negative and
the mode is not adopted.
