# Formal 32/64/128 Neumann cross-edge value comparison

## Decision

**Reject both current candidate study routes for adoption.**  The formal
evidence is complete and every structural and mandatory numerical gate passes,
but neither candidate passes the strict near-edge acceptance gate.  With both
candidates rejected, `decision.json` selects the fallback design
`sector_polynomials_with_shared_edge_constraints`: sector-wise polynomials
with explicit shared-edge value and tangential constraints.

The shared-edge route has the best complete iteration metrics and passes its
global-error, finest-grid-order, and shared-edge-recovery gates.  It is,
however, worse than the direct route in both required `N=128` near-edge defect
norms for every pose, and it is also worse than G1 in translated-pose weighted
RMS.  The direct route is better than G1 in five of the six near-edge
comparisons, but its translated-pose weighted RMS is slightly worse than G1.
There is no schema, configuration, solver, rank, domain, ownership, sector, or
nonfinite-data blocker that calls for a rerun.

This fallback selection is a diagnostic design decision, not a production
route change.  The production Cauchy-policy default remains `g1_nearest`.

## Evidence identity and execution

- Runtime HEAD recorded immediately before the build/study:
  `1cfb90ff17a4d11324e88acdc6971ca9b6692921` on `main`.  The tracked tree was
  clean; only the two pre-existing unrelated untracked `2026-07-23` documents
  remained, and both were untouched.  The CSV schema does not embed a commit
  identifier, so this provenance comes from the recorded runtime repository
  state.
- Branch/worktree: `main`, as required by the Task 6 brief.
- Exact `Release` application-target build:
  `neumann_exterior_zero_trace_3d` reported successful/up-to-date target
  output.  The wrapper's raw exit was 1 only because of the known leaked
  MSBuild-node behavior.  Cleanup was restricted to the exact build-time
  window: 190 matching nodes were removed and 0 remained.
- Final direct regressions all exited 0:
  `harmonic_cauchy_fit_3d_test.exe` in 11.825 s,
  `native_nurbs_surface_3d_test.exe` in 56.584 s, and
  `neumann_exterior_zero_trace_3d_route_test.exe` in 206.458 s.
- The single formal application command was
  `.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-edge-cauchy-study 32 64 128`.
  It ran once from `2026-07-26T23:06:02.8184815+08:00` through
  `2026-07-26T23:29:54.4677293+08:00` (1431.649 s), exited 0, and wrote all
  27 configurations with `status=ok`.
- Preserved output:
  `output/neumann_two_level_edge_cauchy_3d/`.
- The final propagated audit invoked the final script directly:
  `& .\apps\audit_neumann_two_level_edge_cauchy_3d.ps1 -OutputDirectory .\output\neumann_two_level_edge_cauchy_3d -ExpectedLevels @(32,64,128) -DecisionJson .\output\neumann_two_level_edge_cauchy_3d\decision.json`.
  It ran from `2026-07-26T23:31:10.8522413+08:00` through
  `2026-07-26T23:31:46.8895861+08:00` (36.037 s).  Its raw exit was the
  expected 1 solely because seven acceptance predicates failed; schema,
  formal-evidence, and mandatory-numerical evaluation completed.
- Provenance note: the preserved `formal-audit.exit.txt` and
  `formal-audit.*.log` sidecars are timestamped about 23:02, before this final
  application run.  They contain the same `27/81/27` and seven-failure outcome
  but are not the final audit invocation's timing record; `decision.json`
  written at 23:31 is the final decision artifact.

## Raw audit decision and mandatory gates

| decision field | result |
|---|---|
| `schema_pass` | true |
| `formal_evidence_complete` | true |
| `expected_levels` | 32, 64, 128 |
| `summary_row_count` / unique keys | 27 / 27 |
| `bin_row_count` | 81 |
| `owner_row_count` | 27 |
| `all_status_ok` | true |
| `mandatory_numerical_pass` | true |
| `numerical_pass` (compatibility alias) | true |
| `acceptance_pass` | false |
| mandatory failed predicates | 0 |
| acceptance failed predicates | 7 |
| G1 mandatory route gate | true |
| direct mandatory route gate | true |
| shared-edge mandatory route gate | true |
| selected fallback | `sector_polynomials_with_shared_edge_constraints` |

The final artifact cardinalities are:

| artifact | data rows | audited relationship |
|---|---|
| `summary.csv` | 27 | complete 3 poses x 3 levels x 3 routes |
| `edge_distance_bins.csv` | 81 | exactly 3 bins per summary key |
| `owner_diagnostics.csv` | 27 | exactly 1 owner row per summary key |
| `gmres_residuals.csv` | 1843 | 54 complete physical/common histories |
| `dof_diagnostics.csv` | 186120 | exactly one raw row per surface DOF |
| `surface_fit_diagnostics.csv` | 186120 | exactly one fit row per surface DOF, with identical ID sets |
| `edge_point_diagnostics.csv` | 3972 | exactly the shared-route point/map total |
| `edge_fit_diagnostics.csv` | 3972 | key set exactly equals the edge-point key set |

The structural counts behind those totals are:

| N | h | patches | DOFs / surface / value / normal maps per route | shared points / edge maps |
|---:|---:|---:|---:|---:|
| 32 | 0.09375 | 12 | 1050 | 196 on shared route; 0 on controls |
| 64 | 0.046875 | 12 | 3926 | 376 on shared route; 0 on controls |
| 128 | 0.0234375 | 12 | 15704 | 752 on shared route; 0 on controls |

The strengthened mandatory audit established all of the following from the raw
CSV evidence:

| mandatory check | independently recomputed evidence |
|---|---|
| both RHS solves | 27/27 physical and 27/27 common converged |
| GMRES cap and final residual | maximum iterations 57 physical and 63 common; maximum final residuals `1.9757443820796661e-10` physical and `1.9322920800965358e-10` common, all strictly below the 80 / `2e-10` limits |
| residual histories | all 54 histories, comprising 1843 rows, are finite/nonnegative, have contiguous iterations `0..iterations`, contain exactly `iterations+1` rows, and match the summary final residual |
| common RHS | maximum absolute mean `4.719377170804526e-16`; maximum RMS deviation from 1 is `6.661338147750939e-16`; all hashes and equality flags agree |
| raw whole-surface norms | all 186120 DOF rows have finite coordinates/errors, positive weight, nonnegative edge distance, contiguous IDs, and in-range patch IDs; 27 x 4 = **108/108** recomputed density-Linf, unweighted density-L2, defect-Linf, and weighted defect-RMS values match `summary.csv` |
| raw distance-bin norms | all 81 bins are nonempty and contain 186120 rows in total; for each key the raw `<h`, `[h,2h]`, and `>2h` count/weight/density/defect aggregates match the bin CSV and the partition count matches the surface DOF count: **108/108** checks pass |
| surface-map domain and sectors | 186120/186120 rows are finite, have contiguous IDs, 48 ordinary values and 28 normals, valid positive serialized sector counts whose totals match, in-range sector patch IDs, nonnegative radii/distances, and route-valid edge counts in `[0,6]` |
| surface-map SVD and rank | all rows satisfy `sigma_max>0`, `0<sigma_min<=sigma_max`, `condition>=1`, `condition ~= sigma_max/sigma_min`, and `sigma_min>3e-12*sigma_max`; minimum `sigma_min=0.38990247752928547`, minimum rank ratio `0.009322062593572102`, maximum condition `107.27239706473715` |
| shared-point domain and identity | 3972/3972 rows have finite coordinates/values, valid open-unit fractions and native `[0,1]` coordinates, exact `error=reconstructed-exact`, and raw Linf matching the summary; maximum position mismatch `2.6631254232960932e-16`, minimum mapped tangent dot `1`, maximum frame orthogonality error `4.980559976921029e-16`, minimum determinant `0.99999999999999978` |
| shared-edge fit sectors, domain, and rank | 3972/3972 rows have exactly 24/24 value and 14/14 normal samples, finite nonnegative radii, valid SVD ordering/condition identity, and pass the same rank cutoff; minimum `sigma_min=0.61005117842337475`, minimum rank ratio `0.02262475529297439`, maximum condition `44.199373078325735` |
| owner/reference audit | 27/27 owners are available with positive query counts (`178462..2537982`); all before/after owner and Cauchy fingerprints and owner output digests are stable; all owner/summary counters and flags agree; all 27 raw G1 reference audits pass |
| preprocessing/runtime work | minimum Cauchy queries 6104 and minimum Cauchy SVD factorizations 1051; route-dependent preprocessing work is retained, while all 27 runtime geometry-query and runtime-SVD counters are zero |
| finite derived arithmetic | all audited sums, products, differences, divisions, square roots, condition ratios, error ratios, and `log2` orders remain finite; no NaN, infinity, overflow, negative-domain, or zero-denominator escape occurs |
| recorded predicate aggregates | 24525/24525 explicitly recorded mandatory predicates pass; the separate acceptance layer evaluates 58 predicates, of which 51 pass and 7 fail.  Fail-fast schema/domain assertions are additional to these recorded counts |

## Complete 27-row convergence/error/iteration evidence

| pose | N | route | status | phys it | phys residual | common it | common residual | density Linf | density L2 | interior Linf | interior L2 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | g1_value_g1_normal | ok | 33 | 1.97574438207966608e-10 | 35 | 9.44422685446321713e-11 | 1.74971626289242149e-05 | 4.71875388127092468e-06 | 1.39722297993749578e-05 | 3.48857764338931905e-06 |
| baseline | 32 | direct_cross_face_value | ok | 27 | 1.48635675602783632e-10 | 28 | 9.04266744965786205e-11 | 4.65500343047042797e-06 | 1.30028807500595497e-06 | 4.25608113818753964e-06 | 1.00334742149004007e-06 |
| baseline | 32 | edge_reconstructed_value | ok | 26 | 6.17205374436401762e-11 | 26 | 1.63048843898344052e-10 | 7.51562153583629211e-06 | 2.48883213783405026e-06 | 6.81042200201265047e-06 | 1.91835301548879482e-06 |
| baseline | 64 | g1_value_g1_normal | ok | 25 | 9.41384923117851487e-11 | 25 | 1.60967989159788117e-10 | 4.14573169377208117e-07 | 1.33123287382980814e-07 | 5.14263535533743266e-07 | 1.35861932778702079e-07 |
| baseline | 64 | direct_cross_face_value | ok | 23 | 6.95679655043237671e-11 | 24 | 8.45633520484149324e-11 | 4.90083003928942773e-07 | 1.50400930181037869e-07 | 6.06785987344871103e-07 | 1.45684651634318756e-07 |
| baseline | 64 | edge_reconstructed_value | ok | 23 | 1.41524363994498866e-10 | 24 | 1.66285043787930185e-10 | 4.31576437781577837e-07 | 1.43218174097995758e-07 | 5.29837714147163297e-07 | 1.42941738673767156e-07 |
| baseline | 128 | g1_value_g1_normal | ok | 57 | 1.70169950630461788e-10 | 63 | 1.86576784912166385e-10 | 7.84317307837324729e-07 | 8.81480570594590771e-08 | 6.15327916841934552e-07 | 6.62142536967299065e-08 |
| baseline | 128 | direct_cross_face_value | ok | 34 | 1.72215338908064541e-10 | 36 | 1.26470268724861269e-10 | 2.32689977794700575e-07 | 4.98532094050804841e-08 | 2.22300539531872232e-07 | 4.20694371091500500e-08 |
| baseline | 128 | edge_reconstructed_value | ok | 31 | 1.03608488785882067e-10 | 32 | 1.36323008125093883e-10 | 2.10301003497788408e-07 | 4.85202872425546566e-08 | 2.15493614619255425e-07 | 4.13766468151171496e-08 |
| rot_axis123_17deg | 32 | g1_value_g1_normal | ok | 42 | 1.35315788194758928e-10 | 45 | 1.22308350253902578e-10 | 6.34806372004637076e-06 | 1.26989641677387834e-06 | 4.25898798939172707e-06 | 7.74847773506473558e-07 |
| rot_axis123_17deg | 32 | direct_cross_face_value | ok | 30 | 1.57235529551270553e-10 | 31 | 1.73505858552148491e-10 | 1.38976806021706700e-06 | 4.85094058040849714e-07 | 1.47889587531757627e-06 | 4.28637912476819517e-07 |
| rot_axis123_17deg | 32 | edge_reconstructed_value | ok | 27 | 1.14817619909956356e-10 | 27 | 1.75524302437637785e-10 | 2.34079159311395557e-06 | 5.78866439257610348e-07 | 1.91205886002077108e-06 | 4.37930667939656359e-07 |
| rot_axis123_17deg | 64 | g1_value_g1_normal | ok | 38 | 1.28024948014376116e-10 | 38 | 1.71691886623376719e-10 | 6.48766290189950467e-07 | 1.07060075149425123e-07 | 6.09807224938307968e-07 | 8.68510689923892238e-08 |
| rot_axis123_17deg | 64 | direct_cross_face_value | ok | 30 | 9.20724165359578974e-11 | 30 | 1.10591936061853021e-10 | 2.86329671028706656e-07 | 8.45808536812524572e-08 | 3.17424832529944467e-07 | 8.20061520330440345e-08 |
| rot_axis123_17deg | 64 | edge_reconstructed_value | ok | 26 | 1.94484999100584110e-10 | 27 | 1.24968676455751386e-10 | 3.22085714071773666e-07 | 9.09056546556523205e-08 | 3.17098081903388618e-07 | 8.59247699059564296e-08 |
| rot_axis123_17deg | 128 | g1_value_g1_normal | ok | 42 | 1.17975859710951639e-10 | 44 | 1.31689870346434358e-10 | 1.13450215555310763e-07 | 2.38541847232333567e-08 | 1.05577623088493056e-07 | 2.21362337714112274e-08 |
| rot_axis123_17deg | 128 | direct_cross_face_value | ok | 29 | 1.82350848859524118e-10 | 30 | 1.10723769517219834e-10 | 8.18558286985737737e-08 | 2.31123369025469539e-08 | 9.97165475746797370e-08 | 2.18842137637043251e-08 |
| rot_axis123_17deg | 128 | edge_reconstructed_value | ok | 27 | 1.00198974421003799e-10 | 27 | 1.53519353755877035e-10 | 7.82356540211803786e-08 | 2.30795552293742086e-08 | 9.42011485483007505e-08 | 2.19364252270092885e-08 |
| rot_axis123_17deg_t_xyz_1 | 32 | g1_value_g1_normal | ok | 39 | 1.39153283824396627e-10 | 40 | 1.67405930269737455e-10 | 8.38127270461064278e-06 | 1.47168959251711273e-06 | 6.41484892849319976e-06 | 9.76493934380029704e-07 |
| rot_axis123_17deg_t_xyz_1 | 32 | direct_cross_face_value | ok | 30 | 1.94014838532970096e-10 | 31 | 9.74125397589857376e-11 | 1.44816184527440939e-06 | 4.42283945800163310e-07 | 1.45604177004265978e-06 | 3.90501807616676355e-07 |
| rot_axis123_17deg_t_xyz_1 | 32 | edge_reconstructed_value | ok | 27 | 1.34091547128271770e-10 | 27 | 1.54191388415460988e-10 | 3.02597934795367784e-06 | 5.92132293345452172e-07 | 1.96346069514241606e-06 | 4.35901627880129970e-07 |
| rot_axis123_17deg_t_xyz_1 | 64 | g1_value_g1_normal | ok | 45 | 1.81958366960934953e-10 | 45 | 1.93229208009653582e-10 | 1.57804097827840550e-06 | 1.46167076197361984e-07 | 9.94540251153530619e-07 | 1.05993579006234729e-07 |
| rot_axis123_17deg_t_xyz_1 | 64 | direct_cross_face_value | ok | 32 | 1.13970442154682366e-10 | 32 | 1.43769946374791108e-10 | 2.71142522657817349e-07 | 9.08465404898291529e-08 | 3.38397901034070969e-07 | 8.61961004192989694e-08 |
| rot_axis123_17deg_t_xyz_1 | 64 | edge_reconstructed_value | ok | 28 | 1.30548395957791050e-10 | 28 | 1.08743697154000363e-10 | 3.73941869441507535e-07 | 9.80722987618031574e-08 | 3.20657289942793966e-07 | 8.94708710343388376e-08 |
| rot_axis123_17deg_t_xyz_1 | 128 | g1_value_g1_normal | ok | 47 | 1.39489540560506754e-10 | 49 | 1.53634793020423695e-10 | 2.93115842903146628e-07 | 2.58210157262667466e-08 | 1.96860990175906636e-07 | 2.27025068244795001e-08 |
| rot_axis123_17deg_t_xyz_1 | 128 | direct_cross_face_value | ok | 34 | 1.36306913362763380e-10 | 34 | 1.70027112357068919e-10 | 8.90402001574308599e-08 | 2.30947081034240386e-08 | 1.03056607692053603e-07 | 2.17309262469331832e-08 |
| rot_axis123_17deg_t_xyz_1 | 128 | edge_reconstructed_value | ok | 29 | 1.79379861596592329e-10 | 30 | 9.56331432428162664e-11 | 9.19905078777816243e-08 | 2.32586673260322753e-08 | 1.04661031730124421e-07 | 2.18495594239550304e-08 |

## Iteration decision

For each RHS, `W` is the worst iteration count over all poses and levels and
`S(N)` is the pose spread at that level.

### Physical RHS

| route | W | S(32) | S(64) | S(128) |
|---|---:|---:|---:|---:|
| g1_value_g1_normal | 57 | 9 | 20 | 15 |
| direct_cross_face_value | 34 | 3 | 9 | 5 |
| edge_reconstructed_value | 31 | 1 | 5 | 4 |

### Common RHS

| route | W | S(32) | S(64) | S(128) |
|---|---:|---:|---:|---:|
| g1_value_g1_normal | 63 | 10 | 20 | 19 |
| direct_cross_face_value | 36 | 3 | 8 | 6 |
| edge_reconstructed_value | 32 | 1 | 4 | 5 |

The primary route passes the complete iteration gate:
`primary_W_no_worse=true`, `primary_W_strict=true`,
`primary_S_no_worse=true`, and `primary_S_strict=true`.
It is also strictly better than the direct route in at least one iteration
metric (`shared_better_iteration_than_direct=true`).

## Adjacent convergence orders

Orders are recomputed as `log2(error_coarse/error_fine)` and the audit confirms
that all recorded adjacent-order fields equal the recomputed values.

| pose | route | pair | density Linf | density L2 | interior Linf | interior L2 |
|---|---|---|---:|---:|---:|---:|
| baseline | g1_value_g1_normal | 32->64 | 5.3993504 | 5.1475711 | 4.7639106 | 4.6824258 |
| baseline | g1_value_g1_normal | 64->128 | -0.91981069 | 0.59476229 | -0.25884759 | 1.0369276 |
| baseline | direct_cross_face_value | 32->64 | 3.2476842 | 3.1119459 | 2.810266 | 2.7839004 |
| baseline | direct_cross_face_value | 64->128 | 1.074617 | 1.5930552 | 1.4486763 | 1.7920045 |
| baseline | edge_reconstructed_value | 32->64 | 4.1222044 | 4.1191824 | 3.6841218 | 3.7463691 |
| baseline | edge_reconstructed_value | 64->128 | 1.0371604 | 1.5615546 | 1.2979054 | 1.7885386 |
| rot_axis123_17deg | g1_value_g1_normal | 32->64 | 3.2905458 | 3.5682183 | 2.8040855 | 3.1572974 |
| rot_axis123_17deg | g1_value_g1_normal | 64->128 | 2.5156395 | 2.1661063 | 2.5300492 | 1.9721338 |
| rot_axis123_17deg | direct_cross_face_value | 32->64 | 2.279095 | 2.5198615 | 2.2200336 | 2.3859554 |
| rot_axis123_17deg | direct_cross_face_value | 64->128 | 1.8065201 | 1.871668 | 1.6705102 | 1.9058416 |
| rot_axis123_17deg | edge_reconstructed_value | 32->64 | 2.8614799 | 2.6707886 | 2.5921259 | 2.3495565 |
| rot_axis123_17deg | edge_reconstructed_value | 64->128 | 2.0415465 | 1.9777546 | 1.7511126 | 1.9697456 |
| rot_axis123_17deg_t_xyz_1 | g1_value_g1_normal | 32->64 | 2.4090347 | 3.3317831 | 2.6893136 | 3.2036342 |
| rot_axis123_17deg_t_xyz_1 | g1_value_g1_normal | 64->128 | 2.4285918 | 2.5010007 | 2.3368525 | 2.2230534 |
| rot_axis123_17deg_t_xyz_1 | direct_cross_face_value | 32->64 | 2.4170996 | 2.2834694 | 2.1052592 | 2.1796347 |
| rot_axis123_17deg_t_xyz_1 | direct_cross_face_value | 64->128 | 1.6065226 | 1.9758693 | 1.7152836 | 1.9878729 |
| rot_axis123_17deg_t_xyz_1 | edge_reconstructed_value | 32->64 | 3.0165162 | 2.5940019 | 2.6142946 | 2.2845126 |
| rot_axis123_17deg_t_xyz_1 | edge_reconstructed_value | 64->128 | 2.0232571 | 2.0760773 | 1.6153078 | 2.0338139 |

All twelve primary-route `64->128` orders are nonnegative, so
`shared_order_pass=true`.  The corresponding direct-route result also passes.
The baseline G1 `64->128` density-Linf and interior-Linf orders are respectively
`-0.91981069` and `-0.25884759`, i.e. both errors rebound; they are retained
above and do not enter the primary or direct candidate order gates.

## N=128 primary/control error ratios

Each value is
`edge_reconstructed_value error / control error`; the limit is 1.10.

| pose | control | density Linf | density L2 | interior Linf | interior L2 | all <=1.10 |
|---|---|---:|---:|---:|---:|---|
| baseline | g1_value_g1_normal | 0.26813255 | 0.5504408 | 0.35020939 | 0.62489033 | true |
| baseline | direct_cross_face_value | 0.90378196 | 0.97326306 | 0.96937963 | 0.98353222 | true |
| rot_axis123_17deg | g1_value_g1_normal | 0.6896034 | 0.96752647 | 0.8922454 | 0.99097369 | true |
| rot_axis123_17deg | direct_cross_face_value | 0.95577377 | 0.99858164 | 0.94468923 | 1.0023858 | true |
| rot_axis123_17deg_t_xyz_1 | g1_value_g1_normal | 0.3138367 | 0.900765 | 0.53164942 | 0.96242937 | true |
| rot_axis123_17deg_t_xyz_1 | direct_cross_face_value | 1.0331346 | 1.0070994 | 1.0155684 | 1.0054592 | true |

Thus `shared_n128_ratio_pass=true`; the direct-to-G1 ratio gate also passes.
Relative to the direct route, five primary ratios are slightly above 1
(`1.0023858` for rotated interior L2 and all four translated-pose metrics),
but all five remain within the required 1.10 envelope.

## Combined near-edge equation defect at N=128

The `<h` and `[h,2h]` rows are combined.  RMS uses the recorded quadrature
weights; all poses/routes have the same combined weight.

| pose | route | combined weight | defect Linf | weighted RMS |
|---|---|---:|---:|---:|
| baseline | g1_value_g1_normal | 1.539256128 | 1.210000434e-07 | 4.894929767e-08 |
| baseline | direct_cross_face_value | 1.539256128 | 1.11314681e-07 | 4.608984009e-08 |
| baseline | edge_reconstructed_value | 1.539256128 | 1.134854884e-07 | 4.709623026e-08 |
| rot_axis123_17deg | g1_value_g1_normal | 1.539256128 | 4.43950623e-08 | 1.152617982e-08 |
| rot_axis123_17deg | direct_cross_face_value | 1.539256128 | 4.141486881e-08 | 1.14168672e-08 |
| rot_axis123_17deg | edge_reconstructed_value | 1.539256128 | 4.143136144e-08 | 1.144111386e-08 |
| rot_axis123_17deg_t_xyz_1 | g1_value_g1_normal | 1.539256128 | 4.211492058e-08 | 1.129965612e-08 |
| rot_axis123_17deg_t_xyz_1 | direct_cross_face_value | 1.539256128 | 4.069890195e-08 | 1.131267154e-08 |
| rot_axis123_17deg_t_xyz_1 | edge_reconstructed_value | 1.539256128 | 4.088375213e-08 | 1.132193994e-08 |

This table is the decisive failure.  The shared route is worse than the direct
route in both defect norms for every pose, hence
`shared_near_edge_pass=false` and
`shared_better_defect_than_direct=false`.  The direct route is better than G1
except for translated-pose weighted RMS (`1.131267154e-08` versus
`1.129965612e-08`), hence `direct_near_edge_pass=false`.

## Shared-edge value recovery and conditioning

The last column below is the summary's overall route condition maximum, driven
by the surface-fit maps (`72.627...` at `N=32` and `71.218...` at finer
levels).  It is not the edge-fit condition maximum.  Across all 3972 edge fits,
the actual maximum is `44.199373078325735`.

| pose | N | shared points/maps | edge value Linf | edge weighted RMS | value radius/h | normal radius/h | edge radius/h | condition max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | 196/196 | 3.69858588644689235e-07 | 1.59396019651486097e-07 | 7.00289786922045820e+00 | 5.04931129493191122e+00 | 2.28571428571428603e+00 | 7.26272511027545278e+01 |
| baseline | 64 | 376/376 | 3.05446429538225317e-08 | 1.18748537697220365e-08 | 7.00289786922045820e+00 | 5.04931129493191388e+00 | 2.03054933292945750e+00 | 7.12180317308381063e+01 |
| baseline | 128 | 752/752 | 2.45778786212014211e-09 | 1.11706997150523402e-09 | 7.00289786922046176e+00 | 5.04931129493192099e+00 | 2.03054933292945838e+00 | 7.12180317308380495e+01 |
| rot_axis123_17deg | 32 | 196/196 | 3.69858588755711537e-07 | 1.59393341791549003e-07 | 7.00289786922045554e+00 | 5.04931129493191033e+00 | 2.28571428571428648e+00 | 7.26272511027545278e+01 |
| rot_axis123_17deg | 64 | 376/376 | 3.05446426762667755e-08 | 1.18748537790858700e-08 | 7.00289786922045820e+00 | 5.04931129493191388e+00 | 2.03054933292945838e+00 | 7.12180317308380637e+01 |
| rot_axis123_17deg | 128 | 752/752 | 2.45778805640917142e-09 | 1.11706998605053839e-09 | 7.00289786922045909e+00 | 5.04931129493191744e+00 | 2.03054933292945750e+00 | 7.12180317308381348e+01 |
| rot_axis123_17deg_t_xyz_1 | 32 | 196/196 | 3.69858588700200386e-07 | 1.59395129275721847e-07 | 7.00289786922045909e+00 | 5.04931129493191122e+00 | 2.28571428571428648e+00 | 7.26272511027544425e+01 |
| rot_axis123_17deg_t_xyz_1 | 64 | 376/376 | 3.05446428150446536e-08 | 1.18748537903572811e-08 | 7.00289786922045909e+00 | 5.04931129493191388e+00 | 2.03054933292945838e+00 | 7.12180317308381348e+01 |
| rot_axis123_17deg_t_xyz_1 | 128 | 752/752 | 2.45778800089802019e-09 | 1.11706999030356175e-09 | 7.00289786922046087e+00 | 5.04931129493191833e+00 | 2.03054933292945750e+00 | 7.12180317308386464e+01 |

Both shared-edge errors strictly decrease from `N=64` to `N=128` for all
three poses, so all six edge trend predicates pass and
`shared_edge_trend_pass=true`.

## Preprocessing, solve time, and runtime counters

`setup_seconds` is a common case/level setup value copied into each of the
three route rows.  It must not be summed across routes as wall-clock setup
time.

| pose | N | route | setup s | fit s | pipeline s | phys solve s | common solve s | neighborhood q | Cauchy q | SVD | runtime q/SVD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | g1_value_g1_normal | 1.9712537 | 0.2357166 | 7.6864135 | 0.4029564 | 0.3700299 | 6104 | 6104 | 1051 | 0/0 |
| baseline | 32 | direct_cross_face_value | 1.9712537 | 0.2141921 | 7.0698506 | 0.3082631 | 0.2914167 | 6104 | 6104 | 1051 | 0/0 |
| baseline | 32 | edge_reconstructed_value | 1.9712537 | 0.2402398 | 7.1935079 | 0.3140346 | 0.2815160 | 6104 | 6518 | 1247 | 0/0 |
| baseline | 64 | g1_value_g1_normal | 6.6363191 | 0.7801494 | 10.5360248 | 1.8981470 | 1.7067841 | 22672 | 22672 | 3927 | 0/0 |
| baseline | 64 | direct_cross_face_value | 6.6363191 | 0.9178046 | 10.7399150 | 1.7895064 | 1.7317040 | 22672 | 22672 | 3927 | 0/0 |
| baseline | 64 | edge_reconstructed_value | 6.6363191 | 0.9610231 | 10.5086126 | 1.7381957 | 1.6417060 | 22672 | 23446 | 4303 | 0/0 |
| baseline | 128 | g1_value_g1_normal | 27.7805078 | 5.6983902 | 41.2412195 | 28.9773298 | 30.2342297 | 90688 | 90688 | 15705 | 0/0 |
| baseline | 128 | direct_cross_face_value | 27.7805078 | 6.6622147 | 41.2355910 | 18.4317344 | 17.7929913 | 90688 | 90688 | 15705 | 0/0 |
| baseline | 128 | edge_reconstructed_value | 27.7805078 | 6.7550604 | 41.2802510 | 16.9781266 | 16.1027436 | 90688 | 92214 | 16457 | 0/0 |
| rot_axis123_17deg | 32 | g1_value_g1_normal | 3.1041473 | 0.1801334 | 13.7307682 | 0.4585601 | 0.4694762 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg | 32 | direct_cross_face_value | 3.1041473 | 0.2092440 | 13.8081096 | 0.3455769 | 0.3263187 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg | 32 | edge_reconstructed_value | 3.1041473 | 0.2384313 | 14.5313307 | 0.3280835 | 0.3023725 | 6104 | 6518 | 1247 | 0/0 |
| rot_axis123_17deg | 64 | g1_value_g1_normal | 12.2646032 | 0.7957079 | 27.5304985 | 3.5545877 | 3.3489214 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg | 64 | direct_cross_face_value | 12.2646032 | 0.9208679 | 27.2903870 | 2.8665254 | 2.6604308 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg | 64 | edge_reconstructed_value | 12.2646032 | 0.9575043 | 28.3914778 | 1.9520640 | 1.8449919 | 22672 | 23446 | 4303 | 0/0 |
| rot_axis123_17deg | 128 | g1_value_g1_normal | 53.4602352 | 5.6759403 | 58.7193045 | 21.9177459 | 21.5951309 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg | 128 | direct_cross_face_value | 53.4602352 | 6.6401975 | 58.2692718 | 15.8955832 | 14.9878870 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg | 128 | edge_reconstructed_value | 53.4602352 | 6.7845252 | 58.5186847 | 15.0210120 | 13.6850630 | 90688 | 92214 | 16457 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 32 | g1_value_g1_normal | 3.0864115 | 0.1920110 | 13.8865650 | 0.4379033 | 0.4220683 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 32 | direct_cross_face_value | 3.0864115 | 0.2103467 | 13.9642330 | 0.3472149 | 0.3447368 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 32 | edge_reconstructed_value | 3.0864115 | 0.2402410 | 14.1883489 | 0.3297302 | 0.2952452 | 6104 | 6518 | 1247 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 64 | g1_value_g1_normal | 12.3486805 | 0.8053838 | 27.4101283 | 3.9860328 | 3.7600822 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 64 | direct_cross_face_value | 12.3486805 | 0.9216430 | 26.9378496 | 2.9693376 | 2.7077870 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 64 | edge_reconstructed_value | 12.3486805 | 0.9600115 | 27.2380026 | 2.2202739 | 1.9013466 | 22672 | 23446 | 4303 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 128 | g1_value_g1_normal | 53.6936483 | 5.7523954 | 58.5760832 | 24.1240992 | 23.7010751 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 128 | direct_cross_face_value | 53.6936483 | 6.6527943 | 58.7950269 | 18.4009512 | 16.8523760 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 128 | edge_reconstructed_value | 53.6936483 | 6.7403982 | 59.0086749 | 16.1336548 | 15.1698695 | 90688 | 92214 | 16457 | 0/0 |

## Owner diagnostics

Values are identical across all three routes at each pose/level, as required;
the table therefore groups the three route rows without dropping any differing
value.  `stable` means the before/after values are identical.  All 27
`available`, reference, label, neighborhood, and common-RHS flags equal 1.

| pose | N | route rows | owner query count | owner fingerprint (stable) | owner output digest (stable) |
|---|---:|---:|---:|---|---|
| baseline | 32 | 3 | 179906 | 15393305584081917454 (true) | 8109028663866786082 (true) |
| baseline | 64 | 3 | 653624 | 10212258961129723804 (true) | 14103979464998803829 (true) |
| baseline | 128 | 3 | 2537982 | 13232184671465032376 (true) | 3558135855949507990 (true) |
| rot_axis123_17deg | 32 | 3 | 178462 | 1250731490842291720 (true) | 7637703388128116705 (true) |
| rot_axis123_17deg | 64 | 3 | 630118 | 8280865058472844076 (true) | 5561393217191886271 (true) |
| rot_axis123_17deg | 128 | 3 | 2441155 | 1541456893711059721 (true) | 6571750617555784707 (true) |
| rot_axis123_17deg_t_xyz_1 | 32 | 3 | 178507 | 263390229518142790 (true) | 53603338380150283 (true) |
| rot_axis123_17deg_t_xyz_1 | 64 | 3 | 629984 | 5634170801304089673 (true) | 1057097149718224891 (true) |
| rot_axis123_17deg_t_xyz_1 | 128 | 3 | 2442549 | 16612621861264341367 (true) | 11736123202450921947 (true) |

The route-specific Cauchy fingerprints below are listed in
G1/direct/shared order.  Every one is stable before/after; the common-RHS,
label, and neighborhood fingerprints are shared across the three routes at a
pose/level, and all corresponding equality flags are 1.

| pose | N | common RHS hash | Cauchy fingerprints: G1 / direct / shared | label fingerprint | neighborhood fingerprint |
|---|---:|---:|---|---:|---:|
| baseline | 32 | 16079175247052710992 | 5639469236617086693 / 8174152292102831723 / 15672802201107503855 | 1724317395460990725 | 3751049186055881515 |
| baseline | 64 | 7596787855201497449 | 14774028472547329738 / 16830639946708989538 / 17550143264517098984 | 13261685383820968469 | 1173252215149370365 |
| baseline | 128 | 6661903404197365086 | 17102680792084277165 / 5287922929961658613 / 872695794163514194 | 6618804306152956580 | 16701522626696648964 |
| rot_axis123_17deg | 32 | 10666907554377132852 | 18430049666916496517 / 6778449220741904487 / 14845843076728238251 | 7163140122960463637 | 9946039747877769581 |
| rot_axis123_17deg | 64 | 15116120696796770994 | 18154736296688221250 / 16193632218696170339 / 7100092838933945520 | 3163106246597024020 | 12098895822565158554 |
| rot_axis123_17deg | 128 | 5566324866608263457 | 5625913194432563091 / 9626182417284750524 / 7844917853188365741 | 9666602404976367300 | 3965185389810968361 |
| rot_axis123_17deg_t_xyz_1 | 32 | 6843477761456134420 | 3648505823475949746 / 15066905600092080920 / 1916397439374678753 | 2894092712341586773 | 13393888708251282612 |
| rot_axis123_17deg_t_xyz_1 | 64 | 15116120696796770994 | 17389037344772898129 / 4598739960063035452 / 15058372405914373524 | 4150622064495768277 | 10149630010630011905 |
| rot_axis123_17deg_t_xyz_1 | 128 | 5630533462738454872 | 1224561314886550725 / 2265597413797344965 / 1696808030831616507 | 15688534461193308613 | 10133139442506909948 |

## Failed gates, without omission

The seven entries in `acceptance_failed_predicates` (and therefore in
`failed_predicates`) are all **primary/shared-route** comparisons.  In the
exact order serialized by `decision.json`, they are:

1. `primary_near_linf:baseline:direct_cross_face_value`
2. `primary_near_linf:rot_axis123_17deg:direct_cross_face_value`
3. `primary_near_linf:rot_axis123_17deg_t_xyz_1:direct_cross_face_value`
4. `primary_near_rms:baseline:direct_cross_face_value`
5. `primary_near_rms:rot_axis123_17deg:direct_cross_face_value`
6. `primary_near_rms:rot_axis123_17deg_t_xyz_1:direct_cross_face_value`
7. `primary_near_rms:rot_axis123_17deg_t_xyz_1:g1_value_g1_normal`

Their recomputed values are:

| failed predicate | shared value | control value | shared/control |
|---|---:|---:|---:|
| `primary_near_linf:baseline:direct_cross_face_value` | 1.1348548841633166e-7 | 1.1131468096933772e-7 | 1.0195015376955705 |
| `primary_near_linf:rot_axis123_17deg:direct_cross_face_value` | 4.1431361438859504e-8 | 4.1414868812150729e-8 | 1.0003982296016336 |
| `primary_near_linf:rot_axis123_17deg_t_xyz_1:direct_cross_face_value` | 4.0883752133735740e-8 | 4.0698901949964839e-8 | 1.0045418960933676 |
| `primary_near_rms:baseline:direct_cross_face_value` | 4.7096230264552246e-8 | 4.6089840091175414e-8 | 1.0218354017151281 |
| `primary_near_rms:rot_axis123_17deg:direct_cross_face_value` | 1.1441113855136057e-8 | 1.1416867200523926e-8 | 1.0021237572607500 |
| `primary_near_rms:rot_axis123_17deg_t_xyz_1:direct_cross_face_value` | 1.1321939939369934e-8 | 1.1312671541388640e-8 | 1.0008192934752311 |
| `primary_near_rms:rot_axis123_17deg_t_xyz_1:g1_value_g1_normal` | 1.1321939939369934e-8 | 1.1299656115178468e-8 | 1.0019720798548490 |

No mandatory predicate failed.  Separately,
`direct_near_edge_pass=false`: the direct route's translated-pose weighted RMS
is `1.1312671541388640e-8`, versus G1's `1.1299656115178468e-8` (ratio
`1.0011518426824237`).  This derived direct-route gate is not an eighth entry
in `acceptance_failed_predicates`; the seven recorded failures above all name
the primary/shared route.  The resulting route-level decision fields are:

| gate | result |
|---|---|
| `direct_complete` | true |
| `direct_iteration_pass` | true |
| `direct_order_pass` | true |
| `direct_n128_ratio_pass` | true |
| `direct_near_edge_pass` | false |
| `direct_pass` | false |
| `shared_complete` | true |
| `primary_iteration_pass` | true |
| `shared_order_pass` | true |
| `shared_n128_ratio_pass` | true |
| `shared_near_edge_pass` | false |
| `shared_edge_trend_pass` | true |
| `shared_pass` | false |
| `shared_better_iteration_than_direct` | true |
| `shared_better_defect_than_direct` | false |

Therefore the explicit Task 6 conclusion is **reject both current candidate
routes**.  Because neither candidate passes the strict near-edge gate, the
audit selects the fallback next design **sector-wise polynomials with explicit
shared-edge value and tangential constraints**.  It does not adopt either
experimental route, and the production default remains `g1_nearest`.
