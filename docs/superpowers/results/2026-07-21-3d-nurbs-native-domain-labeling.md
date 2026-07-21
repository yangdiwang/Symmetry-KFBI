# 3D native-NURBS domain-label acceptance results

## Scope and configuration

- Baseline: `d25049e7943c46abdb48f4431bfdb93f525d044e`
- Post-study final-review fix:
  `622b57e740d795ddde14cf5aa05dc1f639b2bdc6` rejects an empty native NURBS
  surface model. The L-prism path always supplies 12 patches, so this guard
  does not change or require regeneration of the recorded nonempty-model
  numerical results.
- Geometry and levels: L-prism, `N=32,64,128`
- Cauchy configuration: `KFBIM_3D_CAUCHY_POLICY=same_patch`, value count
  `32`, normal count `20`
- Command: `neumann_exterior_zero_trace_3d.exe l_prism 32 64 128`
- CSV directory:
  `output/neumann_exterior_zero_trace_3d/same_patch/v32_n20`
- Complete combined application stdout/stderr:
  `.superpowers/sdd/task-8-acceptance.log` (ignored, not committed)
- Application wall time: 67.9 seconds. The log begins at 05:32:54 and the
  final CSVs and completion line were written at 05:34:02.

The application reached `Geometry checks and both GMRES formulations
completed.` and wrote all three requested levels. The enclosing capture shell
then returned nonzero only while appending post-run metadata: this host's
`Tee-Object` does not allow `-Append` with `-LiteralPath`. That command ran
after the application and after the `finally` environment cleanup, so it did
not alter the numerical run or its CSVs. No acceptance process or Cauchy
environment variable remained afterward.

## Native geometry and crossing diagnostics

The L-prism's validated explicit interval topology has 12 native NURBS
patches and 26 patch-edge connections; readiness also reports 26 feature
edges and 12 feature vertices. Every level reports two Cartesian graph
components, one box-exterior component, and one representative native point
query.

| N | NURBS patches | Bezier elements | acceleration leaves | candidate edges | seed hits | misses recovered | subdivision boxes | Newton attempts / iterations | seam dedup. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 12 | 12 | 352 | 982 | 982 | 0 | 982 | 982 / 0 | 0 |
| 64 | 12 | 12 | 1,408 | 3,926 | 3,926 | 0 | 3,926 | 3,926 / 0 | 0 |
| 128 | 12 | 12 | 5,632 | 15,122 | 15,122 | 0 | 15,122 | 15,122 / 0 | 0 |

| N | barriers x/y/z | graph / exterior / queries | max root residual | geometry tolerance | label mismatches | exact / gap / endpoint / triangle fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 364 / 364 / 254 | 2 / 1 / 1 | 4.6111025347562034e-16 | 2.1377558326431949e-12 | 0 | 982 / 0 / 0 / 0 |
| 64 | 1,456 / 1,456 / 1,014 | 2 / 1 / 1 | 5.6610488670036757e-16 | 2.1377558326431949e-12 | 0 | 3,926 / 0 / 0 / 0 |
| 128 | 5,610 / 5,610 / 3,902 | 2 / 1 / 1 | 5.6610488670036757e-16 | 2.1377558326431949e-12 | 0 | 15,122 / 0 / 0 / 0 |

All accepted roots are more than three orders of magnitude below the geometry
tolerance. No label mismatch, gap crossing, endpoint crossing, or legacy
triangle-fallback crossing occurred.

## Neumann formulation

All three Neumann solves converged. The residual columns below are the GMRES
relative residual and operator residual; time is the formulation runtime from
the CSV.

| N | iterations | time (s) | GMRES residual | operator residual | exterior trace |
|---:|---:|---:|---:|---:|---:|
| 32 | 43 | 0.4314765 | 1.2668660430966164e-10 | 3.4116754560331586e-11 | 1.1341230335583678e-06 |
| 64 | 26 | 1.8995335 | 1.1465776765471500e-10 | 2.9513950311477188e-11 | 1.9260424192858901e-07 |
| 128 | 61 | 29.7088276 | 1.8143799331567817e-10 | 2.6563086105713118e-10 | 3.8873816712430408e-08 |

| N | density Linf (order) | density L2 (order) | interior Linf (order) | interior L2 (order) |
|---:|---:|---:|---:|---:|
| 32 | 1.3226563285689656e-05 (n/a) | 3.5318150583313877e-06 (n/a) | 1.0836651590295787e-05 (n/a) | 2.5748863714697415e-06 (n/a) |
| 64 | 4.8362373947230530e-07 (4.7734093751) | 1.5195181823279841e-07 (4.5387239558) | 5.9889734027596120e-07 (4.1774665127) | 1.4886576728669879e-07 (4.1124248288) |
| 128 | 5.0257864666908986e-07 (-0.0554643126) | 6.9426613176278041e-08 (1.1300532385) | 4.1644602877255466e-07 (0.5241791902) | 5.5084990735602703e-08 (1.4342808547) |

The Task 8 acceptance gates do not threshold Neumann error orders. The
nonmonotone Neumann density Linf value at `N=128` is nevertheless retained
here without suppression.

## Dirichlet-normal formulation

All three Dirichlet-normal solves converged. The residual columns below are
the GMRES relative residual, operator residual, exterior-normal residual, and
interior-value boundary residual.

| N | iterations | time (s) | GMRES residual | operator residual | exterior normal | interior value residual |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 28 | 0.2964973 | 8.0141233264212049e-11 | 7.0184191791611283e-11 | 7.0181265723618295e-11 | 2.3722353394983742e-06 |
| 64 | 24 | 1.7787579 | 1.5123345560050000e-10 | 1.2713867458824524e-10 | 1.2717400290651288e-10 | 9.3069854711913536e-08 |
| 128 | 33 | 17.3376788 | 1.5826795690772541e-10 | 4.8867915269923401e-10 | 4.8862395913852404e-10 | 4.4100135765390291e-08 |

| N | density Linf (order) | density L2 (order) | interior Linf (order) | interior L2 (order) |
|---:|---:|---:|---:|---:|
| 32 | 3.7545069336456655e-05 (n/a) | 6.8910335487421752e-06 (n/a) | 2.3127164521241639e-06 (n/a) | 1.0029271991900759e-06 (n/a) |
| 64 | 2.7219944276901664e-06 (3.7858874375) | 7.5838635412624927e-07 (3.1837154704) | 2.5840101336438437e-07 (3.1619047640) | 1.3527379913434751e-07 (2.8902625477) |
| 128 | 8.7839262541411500e-07 (1.6317262661) | 1.8463386229469577e-07 (2.0382658346) | 6.8303936506808327e-08 (1.9195710961) | 3.5734325643434326e-08 (1.9204999662) |

The density and interior Linf errors decrease on both refinements. However,
the measured `N=64` to `N=128` Linf orders are 1.6317262661492413 and
1.9195710960942156. Independent `log2(error_64/error_128)` recomputation gives
the same values; both are below the required 2.0 threshold.

## Task 8 gates

| Gate | Result | Evidence |
|---|---|---|
| Both acceptance CSVs contain exactly three rows | PASS | geometry 3; Dirichlet 3 |
| Levels and order are exactly 32,64,128 | PASS | both CSVs: 32,64,128 |
| L-prism label mismatches are zero | PASS | 0, 0, 0 |
| Gap and triangle-fallback crossings are zero | PASS | gap 0,0,0; fallback 0,0,0 |
| Maximum native root residual is within tolerance | PASS | all residuals <= 2.1377558326431949e-12 |
| Every Dirichlet solve converged | PASS | 1, 1, 1 |
| N64-to-N128 density and interior Linf orders are at least 2.0 | **FAIL** | 1.6317262661492413; 1.9195710960942156 |
| Dirichlet density and interior Linf errors decrease on both refinements | PASS | both sequences strictly decrease |

The original gate script stops with `N64-to-N128 order gate failed`. No
threshold was changed and no failed result was omitted.

## Comparison with the former nearest-normal N=128 result

| N=128 result | label mismatches | gap crossings | Dirichlet density Linf | Dirichlet iterations |
|---|---:|---:|---:|---:|
| Former nearest-normal route | 15 | 74 | 6.2212368e-06 | 42 |
| Native NURBS barriers and crossings | 0 | 0 | 8.7839262541411500e-07 | 33 |

The native route removes the observed labeling and gap-crossing failures and
reduces the reported N=128 Dirichlet density error and iteration count, but
the required N64-to-N128 Linf order gate still does not pass.

## Verification

- Serialized Windows Release build with
  `MSBUILDDISABLENODEREUSE=1`, `CMAKE_BUILD_PARALLEL_LEVEL=1`,
  `--parallel 1`, and `/nodeReuse:false`: exit 0 in 6.495 seconds for
  `native_nurbs_surface_3d_test` and `neumann_exterior_zero_trace_3d`.
- `native_nurbs_surface_3d_test.exe`: exit 0 in 131.474 seconds, ending with
  `native NURBS model tests passed`.
- `ctest -N` listed zero registered tests. The subsequent required
  `ctest --test-dir build -C Release --output-on-failure` therefore launched
  no duplicate long test and returned native exit code 0 while reporting
  `No tests were found!!!`.

## Decision

**N=256 allowed: no.** The N64-to-N128 density and interior Linf order gate
failed, so N=256 was not run.
