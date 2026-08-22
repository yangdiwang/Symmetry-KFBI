# Native C0 coefficient-KFBI: N=32--128 convergence under a rigid transform

## Configuration

The reported interior error is the maximum norm on interior Cartesian nodes.
For consecutive grids,

```text
p(N -> 2N) = log2(E_N/E_2N),
p(32 -> 128) = 0.5 log2(E_32/E_128).
```

All runs use reduced density coefficients, exact-crossing-centered cubic local
Cauchy spread, cubic crossing-owner restriction for both exterior traces,
automatic density resolution, and GMRES tolerance `2e-10` with a maximum of
80 iterations.  `Neumann` denotes the exterior-zero-value-trace formulation;
`Dirichlet-normal` denotes the exterior-zero-normal-trace formulation.

The transformed case is `rot_axis123_17deg_t_xyz_1`: rotate by 17 degrees
about the normalized axis `(1,2,3)`, around `(0.07,-0.07,0.02)`, and then
translate by `(0.137,-0.083,0.061)`.  The manufactured harmonic field is
pulled back by the inverse point transform and its gradient is rotated, so the
continuous problems before and after the transform are isometrically
equivalent.

Automatic `ncoef` choices are:

| Geometry | Baseline N=32/64/128 | Rigid N=32/64/128 |
|---|---:|---:|
| Cylinder | 4 / 7 / 11 | 4 / 7 / 12 |
| L-prism | 5 / 7 / 12 | 5 / 8 / 13 |

The difference is expected: the direct density rule uses
`integral_Gamma (|n_x|+|n_y|+|n_z|) dS`, which changes under rotation relative
to the Cartesian axes.  None of the N=128 choices is observability-cap
limited.

## Interior Linf convergence and GMRES iterations

Each error entry is `interior Linf (GMRES iterations)`.

### Cylinder

| Pose | Problem | E32 (it.) | p32-64 | E64 (it.) | p64-128 | E128 (it.) | effective p32-128 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Neumann | 4.773119e-3 (25) | 4.369 | 2.309519e-4 (25) | 3.173 | 2.561091e-5 (29) | 3.771 |
| Baseline | Dirichlet-normal | 7.354810e-3 (32) | 2.854 | 1.017245e-3 (24) | 3.678 | 7.947066e-5 (24) | 3.266 |
| Rigid | Neumann | 2.003216e-3 (27) | 2.774 | 2.928671e-4 (28) | 2.840 | 4.089484e-5 (28) | 2.807 |
| Rigid | Dirichlet-normal | 7.800536e-3 (31) | 2.695 | 1.204784e-3 (27) | 3.821 | 8.525516e-5 (25) | 3.258 |

### L-prism

| Pose | Problem | E32 (it.) | p32-64 | E64 (it.) | p64-128 | E128 (it.) | effective p32-128 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Neumann | 1.478043e-5 (29) | 5.253 | 3.875309e-7 (24) | 1.326 | 1.545838e-7 (37) | 3.290 |
| Baseline | Dirichlet-normal | 4.741838e-6 (22) | 3.831 | 3.331874e-7 (24) | 1.792 | 9.622976e-8 (21) | 2.811 |
| Rigid | Neumann | 4.998609e-6 (32) | 2.635 | 8.048692e-7 (38) | 2.296 | 1.639493e-7 (42) | 2.465 |
| Rigid | Dirichlet-normal | 2.183172e-6 (23) | 2.503 | 3.850904e-7 (22) | 2.658 | 6.103326e-8 (22) | 2.580 |

All eight error sequences decrease monotonically.  Every interval order is
positive, and all four rigid-transform effective orders exceed 2.46.

## Direct rigid/baseline comparison

Each entry is `E_rigid/E_baseline (rigid iteration delta)`.

| Geometry | Problem | N=32 | N=64 | N=128 |
|---|---|---:|---:|---:|
| Cylinder | Neumann | 0.420 (+2) | 1.268 (+3) | 1.597 (-1) |
| Cylinder | Dirichlet-normal | 1.061 (-1) | 1.184 (+3) | 1.073 (+1) |
| L-prism | Neumann | 0.338 (+3) | 2.077 (+14) | 1.061 (+5) |
| L-prism | Dirichlet-normal | 0.460 (+1) | 1.156 (-2) | 0.634 (+1) |

The transformed-to-baseline error ratios stay bounded between 0.338 and
2.077, with no common monotone growth across the four problems.  In
particular, the cylinder Neumann ratio does rise from 0.420 to 1.597, while
the other three sequences do not share that trend.  The largest iteration
count is 42, well below the limit of 80, and every projected GMRES residual is
at most `1.927e-10 < 2e-10`.

The visible outlier is L-prism Neumann at N=64: 24 baseline iterations versus
38 transformed iterations.  A rigid N=64 control with fixed `ncoef=7` needs
36 iterations, while its exterior trace and exterior-bulk errors are 2.37 and
1.86 times the automatic `ncoef=8` result.  Its interior Linf is instead
smaller (`6.323e-7` versus `8.049e-7`), so the control does not establish a
uniform improvement of every norm.  Thus rotation accounts for most of the
iteration change; the extra automatic density level costs two iterations and
materially improves the exterior-trace/exterior-bulk diagnostics.  At N=128
the transformed difference falls to +5 iterations and the error ratio to
1.061.

## N=128 geometry, crossing, and seam checks

| Geometry / pose | exact / directed ops | raw ambiguity (label-changing) | targeted resolved / unsafe | gap / triangle fallback | unique plans / SVDs |
|---|---:|---:|---:|---:|---:|
| Cylinder / baseline | 17812 / 35624 | 0 (0) | 0 / 0 | 0 / 0 | 17812 / 35624 |
| Cylinder / rigid | 19592 / 39184 | 10 (1) | 1 / 0 | 0 / 0 | 19592 / 39184 |
| L-prism / baseline | 15122 / 30244 | 0 (0) | 0 / 0 | 0 / 0 | not recorded in the pre-group snapshot |
| L-prism / rigid | 19608 / 39216 | 0 (0) | 0 / 0 | 0 / 0 | 19608 / 39216 |

The rigid cylinder has ten raw depth-4 ambiguity clusters.  Nine do not change
endpoint membership and use the endpoint-parity safety path; the only
label-changing edge is resolved by one depth-6 targeted retry.  Consequently
all 19592 correction edges are safe, with zero unsafe label-changing edges,
zero gap crossings, and zero triangle fallback crossings.  This is distinct
from the formerly failing duplicate-seed edge, whose focused production
regression now has one transverse root and no ambiguity at depth 4.

At the C0 L-prism seams, value density is continuous across 26 seams, while
normal density remains broken across the 22 sharp seams; only the four G1
seams carry the weak-C1 condition.  In the transformed N=128 run the value and
normal constraint residuals are `1.221e-15` and `1.007e-15`.  The corresponding
transformed-cylinder residuals are both `4.441e-15` (the baseline-cylinder
values are `2.665e-15`).  Neumann/Dirichlet route mismatches in all N=128
cases remain at approximately `2e-16` / `1.5e-13`.

## Stability conclusion

The rigid-transform study passes the practical stability checks:

1. every interior Linf sequence converges monotonically;
2. every transformed effective order is above 2.46;
3. all projected coefficient systems reach the requested GMRES tolerance in
   at most 42 of 80 iterations;
4. a transformed error never exceeds its baseline by more than 2.077 times;
   all ratios lie in 0.338--2.077, including 0.634--1.597 at N=128;
5. no N=128 run has an unsafe correction edge, gap crossing, or triangle
   fallback; and
6. C0/broken-normal seam constraints close at machine precision.

The CSV field `physical_converged=0` in the raw per-solve outputs is a
surface-sample approximation diagnostic.  It is not the coefficient-space
GMRES status; all reported projected systems have `converged=1`.

The machine-readable aggregate is
`output/kfbi_native_c0_exterior_trace_3d/rigid_transform_convergence_summary.csv`.
The N=32/64 accepted snapshots and the four N=128 source directories are kept
unchanged beside that aggregate.  The baseline L-prism N=128 row uses the
accepted pre-group snapshot, and that case has no root ambiguity.  The grouped
crossing-plan algebra was separately verified on the N=32 cylinder: its result
CSV was byte-identical to the per-directed-operation run, and all 2320
directed Cauchy rows passed the in-process equivalence check.  A direct grouped
rerun of the baseline N=128 L-prism was not used for this table.

## Neumann restrict stabilization addendum

The native coefficient executable now uses the following Neumann defaults:

1. all eight normal layers are converted to a common exterior branch before
   one cubic trace recovery;
2. the value-density Cauchy jet is recovered one-sided on the owning patch by
   a local 3-by-3 quadratic fit, so derivatives are never mixed across a C0
   seam;
3. reduced coefficients are expressed in trace-mass coordinates; and
4. the compatibility border is eliminated in the orthogonal complements of
   its range and mean-constraint vectors, after which the original bordered
   equation is reconstructed and audited.

The normal layers remain
`rho/h = +/-{0.2,0.6,1.0,1.4}`.  Their cubic moment weights and scaling match
the Python/reference C++ implementation; no layer-position defect was found.

For the rigid L-prism, the iteration comparison at tolerance `2e-10` is:

| N | old joint Neumann | local exterior-branch Neumann | stabilized Neumann | Dirichlet-normal |
|---:|---:|---:|---:|---:|
| 32 | 32 | 30 | **28** | 23 |
| 64 | 38 | 35 | **32** | 22 |

The stabilized interior Linf errors are `4.386106e-6` and `5.739936e-7`.
They are unchanged, to displayed precision, from the local exterior-branch
version; the iteration reduction therefore comes from the coordinate and
compatibility treatment rather than an accuracy tradeoff.  At N=32 the full
reconstructed operator and compatibility residuals are `1.756e-12` and
`7.305e-18`; at N=64 the reconstructed operator residual is `3.731e-12`,
with the mean and trace-closure audits at the `1e-16` scale.

For the rigid cylinder at N=32, stabilized Neumann takes 24 iterations versus
27 for the old Neumann route and 31 for Dirichlet-normal.  Its route mismatch
is `1.110e-16`, including the cap/side C0 seams, and its interior Linf error
improves from `2.003216e-3` to `1.356145e-3`.

Thus the cylinder reaches (and exceeds) the Dirichlet iteration level.  The
rigid L-prism gap is reduced from 9 to 5 iterations at N=32 and from 16 to 10
at N=64, but it does not reach the strict Dirichlet level.  The remaining
growth is associated with sharp-edge value-density modes: the L-prism value
space is strongly C0 across 26 seams, whereas the Dirichlet normal density is
broken across its 22 sharp seams.  A direct owner-interpolated exterior-branch
experiment was rejected because it changed N=16 from 29 to 33 iterations and
increased the interior Linf error from `1.542064e-5` to `2.457565e-5`.
