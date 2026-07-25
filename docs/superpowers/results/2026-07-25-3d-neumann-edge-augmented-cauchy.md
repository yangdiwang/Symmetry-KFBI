# 3D Neumann edge-augmented Cauchy pilot

## Decision

Keep `non_g1_auxiliary_values` experimental. The complete coarse pilot
failed the fixed error guard and rigid-pose spread gates, so the
application correctly stopped before `N=128`. No weights, thresholds, or
other algorithm parameters were changed.

The augmented route substantially reduced every measured incident-edge
discrepancy and reduced the worst GMRES count from 42 to 27, but its
baseline `N=64` density L2 error ratio was
`1.1047788080297003`, above the fixed `1.10` guard. At `N=64`, the
augmented pose spreads were also worse for density Linf, density L2, and
interior Linf.

## Provenance and execution

- Branch: `codex/neumann-rigid-transform-study`
- Implementation commit: `7fc83d917a50541d1f5061cc0337915f4e6f538d`
- Configuration: Visual Studio x64 `Release`
- Build command: `cmake --build build --config Release -- /m:1`
- Pilot command:

  ```powershell
  $env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR = `
    'output/neumann_edge_cauchy_3d'
  .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
    --neumann-edge-cauchy-study 32 64 128
  ```

- Clean pilot result: application coarse-gate exit `1`; observed
  end-to-end launcher wall time `124.2 s`. The retry launched at
  `2026-07-25T23:10:02.7652259+08:00`; the final checkpoint was written
  at `2026-07-25T23:12:06.6106317+08:00`.
- `N=128` did not run because the complete `N=32,64` acceptance result
  was `fail`.

There was one invalid orchestration attempt before the clean run. A
60-second shell timeout returned `124` while the healthy application was
setting up `ty_m0083, N=64`, leaving eight incremental summary rows. The
solver process was confirmed gone. The exact six generated CSVs and
their timestamps were inspected, only those incomplete ignored outputs
were removed, and the identical command was rerun from a clean output
state. Exit `124` is not used as numerical evidence. On the clean retry,
the detached wrapper was reaped immediately after the application wrote
its final checkpoint and before it could persist its separate Stopwatch
metadata; the reported application exit is the evaluator's coarse-fail
exit path, corroborated by the complete 12-row checkpoint and
`overall_pass=fail`.

Before both the invalid attempt and the clean retry, a fresh full Release
build and all eleven prescribed executables exited zero. The same full
verification was repeated after writing this report.

## Fixed study configuration

- Native twelve-patch NURBS L-prism
- Cases: `baseline`, `ty_m0083`, `rot_axis123_17deg`
- Modes: `none`, `non_g1_auxiliary_values`
- Shared density space and geometry/FFT/crossing/owner preprocessing
- Face fit: harmonic degree 3, `G1Nearest`, 48 value and 28 normal rows
- Edge auxiliary fit: 24 value and 14 normal rows per incident side
- Four nearest midpoint edge samples per incident non-G1 connection,
  soft value-row scale exactly `1.0`
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
| `summary.csv` | 12 | 12 unique coarse keys; three cases and two modes per level; finite raw metrics; all orders and ratios recomputed |
| `edge_values.csv` | 3432 | every connection group complete with literal `0..sample_count-1`; value differences recomputed; group Linf matches summary |
| `gmres_residuals.csv` | 367 | every history finite and contiguous from zero through the reported terminal iteration; terminal residual matches summary |
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
| baseline | 32 | none | 1.7497e-5 | — | 1.3972e-5 | — | 3.626e-6 | 33 | 1.976e-10 | 9.960 |
| baseline | 32 | augmented | 7.8786e-6 | — | 6.9076e-6 | — | 1.129e-6 | 26 | 1.229e-10 | 9.942 |
| ty_m0083 | 32 | none | 2.3581e-5 | — | 1.8231e-5 | — | 4.110e-6 | 37 | 1.599e-10 | 9.486 |
| ty_m0083 | 32 | augmented | 8.8742e-6 | — | 7.6540e-6 | — | 1.110e-6 | 27 | 1.249e-10 | 9.419 |
| rot_axis123_17deg | 32 | none | 6.3481e-6 | — | 4.2590e-6 | — | 5.758e-6 | 42 | 1.353e-10 | 18.571 |
| rot_axis123_17deg | 32 | augmented | 2.7241e-6 | — | 2.4491e-6 | — | 1.157e-6 | 27 | 1.394e-10 | 18.424 |
| baseline | 64 | none | 4.1457e-7 | 5.399 | 5.1426e-7 | 4.764 | 1.880e-7 | 25 | 9.414e-11 | 17.102 |
| baseline | 64 | augmented | 4.4683e-7 | 4.140 | 5.4698e-7 | 3.659 | 8.697e-8 | 23 | 1.561e-10 | 17.173 |
| ty_m0083 | 64 | none | 5.8297e-7 | 5.338 | 6.7208e-7 | 4.762 | 2.445e-7 | 27 | 1.450e-10 | 18.510 |
| ty_m0083 | 64 | augmented | 5.5977e-7 | 3.987 | 6.4876e-7 | 3.560 | 9.237e-8 | 24 | 8.461e-11 | 18.232 |
| rot_axis123_17deg | 64 | none | 6.4877e-7 | 3.291 | 6.0981e-7 | 2.804 | 6.247e-7 | 38 | 1.280e-10 | 41.219 |
| rot_axis123_17deg | 64 | augmented | 3.1795e-7 | 3.099 | 3.2156e-7 | 2.929 | 8.192e-8 | 26 | 1.993e-10 | 40.493 |

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
guard failure. Every edge-discrepancy ratio is strictly below one.

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
| baseline | 32 | 0.0409 | 0.0146 | 0.0609 | 1.2803 | 0.0050 | 8.0060 | 0.1363 | 0.0001 | 9.5442 |
| baseline | 64 | 0.1712 | 0.2215 | 0.3376 | 3.2460 | 0.0191 | 10.6778 | 0.4201 | 0.0003 | 15.0937 |
| ty_m0083 | 32 | 0.0421 | 0.0152 | 0.0821 | 1.2382 | 0.0047 | 7.3976 | 0.1208 | 0.0001 | 8.9008 |
| ty_m0083 | 64 | 0.1679 | 0.2259 | 0.3624 | 3.3066 | 0.0201 | 11.6362 | 0.4405 | 0.0003 | 16.1600 |
| rot_axis123_17deg | 32 | 1.0674 | 0.0148 | 0.0785 | 1.2462 | 0.0091 | 15.4069 | 0.1412 | 0.0001 | 17.9642 |
| rot_axis123_17deg | 64 | 4.3959 | 0.2101 | 0.6135 | 3.2190 | 0.0301 | 28.6932 | 0.4756 | 0.0005 | 37.6379 |

Mode runtime phases, in seconds:

| Case | N | Mode | Edge values | Cauchy | Spread | FFT | Restrict continued | Recovery | GMRES/other | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | none | 0 | 0.044917 | 0.002069 | 0.284114 | 0.082019 | 0.000193 | 0.002757 | 0.416069 |
| baseline | 32 | augmented | 0.000759 | 0.095488 | 0.001620 | 0.229577 | 0.068006 | 0.000157 | 0.002098 | 0.397704 |
| ty_m0083 | 32 | none | 0 | 0.049913 | 0.002233 | 0.437351 | 0.092166 | 0.000236 | 0.003332 | 0.585230 |
| ty_m0083 | 32 | augmented | 0.000807 | 0.102474 | 0.001786 | 0.338287 | 0.072097 | 0.000190 | 0.002706 | 0.518347 |
| rot_axis123_17deg | 32 | none | 0 | 0.061868 | 0.003567 | 0.413175 | 0.124376 | 0.000300 | 0.003852 | 0.607136 |
| rot_axis123_17deg | 32 | augmented | 0.000836 | 0.129002 | 0.002366 | 0.246692 | 0.078399 | 0.000175 | 0.002254 | 0.459724 |
| baseline | 64 | none | 0 | 0.133055 | 0.007543 | 1.647383 | 0.192286 | 0.000570 | 0.027395 | 2.008233 |
| baseline | 64 | augmented | 0.001700 | 0.253301 | 0.007216 | 1.612253 | 0.179360 | 0.000518 | 0.025344 | 2.079691 |
| ty_m0083 | 64 | none | 0 | 0.151844 | 0.008730 | 1.938449 | 0.217615 | 0.000678 | 0.033167 | 2.350483 |
| ty_m0083 | 64 | augmented | 0.001705 | 0.252124 | 0.007075 | 1.595894 | 0.188793 | 0.000528 | 0.026290 | 2.072408 |
| rot_axis123_17deg | 64 | none | 0 | 0.189296 | 0.012390 | 3.033833 | 0.307042 | 0.000808 | 0.038241 | 3.581609 |
| rot_axis123_17deg | 64 | augmented | 0.001895 | 0.288477 | 0.009796 | 2.285715 | 0.233181 | 0.000616 | 0.035767 | 2.855447 |

Legacy edge-value calls are exactly zero. Augmented calls are positive:
32, 33, and 33 at `N=32`, and 29, 30, and 32 at `N=64`. The measured
edge-value overhead is `0.000759--0.001895 s`. Every setup and runtime
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
| edge discrepancy | pass | pass | every augmented value is strictly below legacy |
| geometry/owner | pass | pass | all snapshots and flags stable |
| extended evidence | not evaluated | not evaluated | coarse failure prevented `N=128` |
| overall | **fail** | **fail** | conjunction of coarse gates |

The numerical mechanism is therefore mixed: the auxiliary edge values
improve edge continuity and solver iteration behavior, but do not meet
the fixed accuracy guard or rigid-pose robustness requirement at
`N=64`. Under the prescribed adoption rule, this is a valid coarse
negative and the mode is not adopted.
