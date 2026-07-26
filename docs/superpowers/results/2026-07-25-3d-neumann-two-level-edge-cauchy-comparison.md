# Formal 32/64/128 Neumann cross-edge value comparison

## Decision

**Reject both current candidate routes for adoption.**  The formal evidence is
complete and all structural and mandatory numerical gates pass, but neither
candidate passes its complete acceptance gate.  The audited next design is
`sector_polynomials_with_shared_edge_constraints`: sector-wise polynomials
with explicit shared-edge value and tangential constraints.

The shared-edge route has the best complete iteration metrics and acceptable
global errors, orders, and shared-edge recovery.  It nevertheless fails all
six required `N=128` near-edge defect comparisons against the direct route,
plus the translated-pose weighted-RMS comparison against G1.  The direct route
also fails its translated-pose near-edge weighted-RMS comparison against G1.
There is no structural, configuration, solver, or malformed-diagnostic blocker
that calls for a rerun.

## Evidence identity and execution

- Runtime HEAD recorded immediately before the build/study:
  `f598930d55252a03f5b62300f743a401df354f12`
  (`fix: reject orphan Neumann study diagnostics`).  The tracked tree was
  clean; the two pre-existing unrelated `2026-07-23` documents were untracked
  and untouched.  The CSV schema does not itself embed a commit identifier,
  so this provenance comes from the recorded runtime repository state.
- Branch/worktree: `main`, as required by the Task 6 brief.
- Build: Visual Studio 2017 MSBuild generator, `Release`; each target was
  invoked separately with `MSBUILDDISABLENODEREUSE=1`, `/m:2`, and
  `/nr:false`.
- Targets: `harmonic_cauchy_fit_3d_test`,
  `native_nurbs_surface_3d_test`,
  `neumann_exterior_zero_trace_3d_route_test`, and
  `neumann_exterior_zero_trace_3d`.
- Each target printed its successful target/product line.  CMake subsequently
  returned 1 only because of the known leaked MSBuild-node behavior.  There
  were no `cl` or `link` processes after any invocation.  Cleanup was limited
  to each invocation's exact start-time window: 20, 20, 190, and 190 MSBuild
  nodes respectively.
- Direct tests:
  `harmonic_cauchy_fit_3d_test.exe` passed (exit 0),
  `native_nurbs_surface_3d_test.exe` passed (exit 0), and
  `neumann_exterior_zero_trace_3d_route_test.exe` passed (exit 0).
- The single formal application command was
  `.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-edge-cauchy-study 32 64 128`.
  It ran once from `2026-07-26T18:15:21.2844873+08:00` through
  `2026-07-26T18:39:11.3140863+08:00` (1430.3 s), exited 0, and wrote all
  27 configurations.
- Preserved output:
  `output/neumann_two_level_edge_cauchy_3d/`.
- Final audit used
  `-ExpectedLevels @(32,64,128)` and wrote
  `output/neumann_two_level_edge_cauchy_3d/decision.json`.  The first
  foreground audit was terminated by the executor's 600 s per-call limit, so
  the identical audit command was rerun in a monitored background PowerShell
  and allowed to finish.  The final raw audit exit was 1 solely because seven
  acceptance predicates failed.

## Raw audit decision and mandatory gates

| decision field | result |
|---|---|
| `schema_pass` | true |
| `formal_evidence_complete` | true |
| `summary_row_count` / unique keys | 27 / 27 |
| `bin_row_count` | 81 |
| `owner_row_count` | 27 |
| `all_status_ok` | true |
| `mandatory_numerical_pass` | true |
| `acceptance_pass` | false |
| mandatory failed predicates | 0 |
| acceptance failed predicates | 7 |
| G1 mandatory route gate | true |
| direct mandatory route gate | true |
| shared-edge mandatory route gate | true |
| raw selected route | `sector_polynomials_with_shared_edge_constraints` |

The mandatory audit additionally established:

| mandatory check | observed evidence |
|---|---|
| both RHS solves present and converged | 27/27 physical and 27/27 common |
| GMRES cap | maximum physical 57; maximum common 63; both below 80 |
| final relative residual | maximum physical `1.9757443820796661e-10`; maximum common `1.9322920800965358e-10`; both strictly below `2e-10` |
| residual histories | every history has exactly `iterations+1` contiguous entries and matches its summary final residual |
| preprocessing work | minimum Cauchy queries 6104 and minimum Cauchy SVD factorizations 1051; actual route-dependent work retained |
| runtime work | all runtime geometry-query and runtime-SVD counters are zero |
| fingerprints and reference data | all pre/post Cauchy and owner fingerprints/digests stable; all 27 raw G1 reference audits equal; label, neighborhood, and common-RHS checks equal |
| shared-point geometry | 3972 points; max position mismatch `2.6631254232960932e-16`; min mapped tangent dot `1`; max frame orthogonality error `4.980559976921029e-16`; min frame determinant `0.99999999999999978` |
| shared-edge fits | 3972 fits; all sector counts exactly 24/24 values and 14/14 normals; min `sigma_min=0.61005117842337475`; max edge-fit condition `44.199373078325735` |
| surface fits | 186120 maps; all have 48 ordinary values and 28 normals; min `sigma_min=0.38990247752928547` |

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
| baseline | 32 | g1_value_g1_normal | 1.7962518 | 0.1815852 | 6.9648055 | 0.3813751 | 0.3631190 | 6104 | 6104 | 1051 | 0/0 |
| baseline | 32 | direct_cross_face_value | 1.7962518 | 0.2120789 | 7.0287626 | 0.3118407 | 0.2943311 | 6104 | 6104 | 1051 | 0/0 |
| baseline | 32 | edge_reconstructed_value | 1.7962518 | 0.2393666 | 7.1228033 | 0.3074084 | 0.2871731 | 6104 | 6518 | 1247 | 0/0 |
| baseline | 64 | g1_value_g1_normal | 6.7869194 | 0.7809853 | 10.5472382 | 1.8609126 | 1.6809082 | 22672 | 22672 | 3927 | 0/0 |
| baseline | 64 | direct_cross_face_value | 6.7869194 | 0.9208587 | 10.5628558 | 1.8121178 | 1.6465772 | 22672 | 22672 | 3927 | 0/0 |
| baseline | 64 | edge_reconstructed_value | 6.7869194 | 1.0367386 | 10.6216630 | 1.7785234 | 1.7620869 | 22672 | 23446 | 4303 | 0/0 |
| baseline | 128 | g1_value_g1_normal | 27.9844398 | 5.7415057 | 41.1529517 | 28.9143946 | 30.2316977 | 90688 | 90688 | 15705 | 0/0 |
| baseline | 128 | direct_cross_face_value | 27.9844398 | 6.7345553 | 41.2297815 | 18.4068827 | 17.7749731 | 90688 | 90688 | 15705 | 0/0 |
| baseline | 128 | edge_reconstructed_value | 27.9844398 | 6.8221841 | 41.6114635 | 16.9306514 | 16.1435926 | 90688 | 92214 | 16457 | 0/0 |
| rot_axis123_17deg | 32 | g1_value_g1_normal | 3.0995192 | 0.1788833 | 14.0982303 | 0.4694957 | 0.4644838 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg | 32 | direct_cross_face_value | 3.0995192 | 0.2106933 | 13.7283725 | 0.3826404 | 0.3447506 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg | 32 | edge_reconstructed_value | 3.0995192 | 0.2398952 | 14.3331513 | 0.3319607 | 0.3150597 | 6104 | 6518 | 1247 | 0/0 |
| rot_axis123_17deg | 64 | g1_value_g1_normal | 12.2079718 | 0.7796464 | 27.4007506 | 3.6842082 | 3.4162669 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg | 64 | direct_cross_face_value | 12.2079718 | 0.9165654 | 27.3997515 | 2.9588767 | 2.6967756 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg | 64 | edge_reconstructed_value | 12.2079718 | 0.9676432 | 27.1848785 | 2.0053268 | 1.9193791 | 22672 | 23446 | 4303 | 0/0 |
| rot_axis123_17deg | 128 | g1_value_g1_normal | 53.0718334 | 5.7428480 | 57.9285420 | 22.1892286 | 21.6008665 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg | 128 | direct_cross_face_value | 53.0718334 | 6.6592221 | 57.8385197 | 16.0332556 | 15.0444478 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg | 128 | edge_reconstructed_value | 53.0718334 | 6.7645370 | 58.4978004 | 15.0760963 | 13.8521005 | 90688 | 92214 | 16457 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 32 | g1_value_g1_normal | 3.0803353 | 0.1802942 | 14.0378378 | 0.4333641 | 0.4152221 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 32 | direct_cross_face_value | 3.0803353 | 0.2104681 | 13.8426021 | 0.3466800 | 0.3317054 | 6104 | 6104 | 1051 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 32 | edge_reconstructed_value | 3.0803353 | 0.2390642 | 14.5693779 | 0.3205444 | 0.2949085 | 6104 | 6518 | 1247 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 64 | g1_value_g1_normal | 12.1545693 | 0.7795416 | 27.3555849 | 4.2716553 | 3.7894388 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 64 | direct_cross_face_value | 12.1545693 | 0.9153994 | 26.8861628 | 3.1097951 | 2.8451209 | 22672 | 22672 | 3927 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 64 | edge_reconstructed_value | 12.1545693 | 0.9561358 | 27.1549162 | 2.1405279 | 1.9732306 | 22672 | 23446 | 4303 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 128 | g1_value_g1_normal | 53.4067545 | 5.7233135 | 58.6030829 | 24.3477198 | 23.8660066 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 128 | direct_cross_face_value | 53.4067545 | 6.6630835 | 59.1027516 | 18.2979532 | 16.8975427 | 90688 | 90688 | 15705 | 0/0 |
| rot_axis123_17deg_t_xyz_1 | 128 | edge_reconstructed_value | 53.4067545 | 6.8440407 | 58.3024574 | 16.0152809 | 15.3474420 | 90688 | 92214 | 16457 | 0/0 |

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

## Failed gates, without omission

The seven failed predicates are all acceptance predicates:

1. `primary_near_linf:baseline:direct_cross_face_value`
2. `primary_near_rms:baseline:direct_cross_face_value`
3. `primary_near_linf:rot_axis123_17deg:direct_cross_face_value`
4. `primary_near_rms:rot_axis123_17deg:direct_cross_face_value`
5. `primary_near_linf:rot_axis123_17deg_t_xyz_1:direct_cross_face_value`
6. `primary_near_rms:rot_axis123_17deg_t_xyz_1:direct_cross_face_value`
7. `primary_near_rms:rot_axis123_17deg_t_xyz_1:g1_value_g1_normal`

No mandatory predicate failed.  The resulting route-level decision fields are:

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

Therefore the explicit Task 6 conclusion is **reject**, with the named next
design **sector-wise polynomials with explicit shared-edge value and
tangential constraints**.
