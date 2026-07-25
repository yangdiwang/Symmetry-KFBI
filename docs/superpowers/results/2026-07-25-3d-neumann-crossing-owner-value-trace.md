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

## TDD and formal verification

Before implementation, `--neumann-owner-study 16` exited 1 with the old
geometry-selection error.  A later synthetic real-app regression also first
failed on two legal center-owned exception fallbacks before the owner
partition was corrected.  The Release build, focused route test, and N=16
smoke then passed.

The formal command used `N=32,64,128`; `formal_run.exit` is `0`.  Its 18
summary rows and 18 owner-diagnostic rows are the complete Cartesian product
of three poses, three levels, and two routes.  All 18 solves converged within
80 iterations with final relative residual at most `2e-10`.

Independent CSV auditing passed all of the following checks:

- all 18 `(case_id,N,route)` keys are unique and paired by one common
  pipeline for each `(case_id,N)`;
- all 811 residual rows form complete `0..iterations` histories, exactly
  `iterations + 1` values per solve;
- owner geometry-query counts and hexadecimal fingerprints are unchanged
  before and after every GMRES solve;
- `decision_count_sum + exception_fallback_count_sum` equals both wrong-side
  count and geometry-query count, and foreign non-G1 decisions equal reroute
  audit terms;
- paired routes have identical geometry, setup, Cauchy, query, and
  fingerprint metadata;
- reported orders agree with independent adjacent-level calculations;
- the legacy N=32 deterministic baseline remains unchanged to a maximum
  absolute difference of `3.33e-16`.

`route_mismatch_linf` is an internal consistency check for one selected mode:
it compares that mode's direct exterior value trace with the exterior value
reconstructed from its interior trace and jump.  It is not the numerical
difference between legacy and crossing-owner formats.

## Formal numerical results

Here `L` is the legacy center-owned route and `O` is the crossing-owner route.
Orders use the preceding formal level for the same pose and route; the N=32
cells are therefore blank.

| pose | N | route | GMRES | interior Linf | order | density Linf | order |
|---|---:|:---:|---:|---:|---:|---:|---:|
| baseline | 32 | L | 39 | 1.299933e-5 | - | 1.545585e-5 | - |
| baseline | 32 | O | 33 | 1.397223e-5 | - | 1.749716e-5 | - |
| baseline | 64 | L | 26 | 5.019877e-7 | 4.695 | 3.948710e-7 | 5.291 |
| baseline | 64 | O | 25 | 5.142635e-7 | 4.764 | 4.145732e-7 | 5.399 |
| baseline | 128 | L | 51 | 3.632433e-7 | 0.467 | 4.259476e-7 | -0.109 |
| baseline | 128 | O | 57 | 6.153279e-7 | -0.259 | 7.843173e-7 | -0.920 |
| rot17 | 32 | L | 52 | 3.453142e-6 | - | 7.285416e-6 | - |
| rot17 | 32 | O | 42 | 4.258988e-6 | - | 6.348064e-6 | - |
| rot17 | 64 | L | 48 | 1.022363e-6 | 1.756 | 2.849892e-6 | 1.354 |
| rot17 | 64 | O | 38 | 6.098072e-7 | 2.804 | 6.487663e-7 | 3.291 |
| rot17 | 128 | L | 51 | 9.581140e-8 | 3.416 | 1.175638e-7 | 4.599 |
| rot17 | 128 | O | 42 | 1.055776e-7 | 2.530 | 1.134502e-7 | 2.516 |
| rot17 + translation | 32 | L | 50 | 5.987881e-6 | - | 1.459826e-5 | - |
| rot17 + translation | 32 | O | 39 | 6.414849e-6 | - | 8.381273e-6 | - |
| rot17 + translation | 64 | L | 46 | 6.075503e-7 | 3.301 | 8.097868e-7 | 4.172 |
| rot17 + translation | 64 | O | 45 | 9.945403e-7 | 2.689 | 1.578041e-6 | 2.409 |
| rot17 + translation | 128 | L | 62 | 1.148620e-7 | 2.403 | 1.386825e-7 | 2.546 |
| rot17 + translation | 128 | O | 47 | 1.968610e-7 | 2.337 | 2.931158e-7 | 2.429 |

Crossing ownership reduced iteration counts in all six rotated rows, although
the reduction at translated N=64 was only one step.  It did not uniformly
improve baseline convergence: at N=128 it required 57 iterations versus 51
for legacy.  The baseline N=128 errors have reached a grid/geometry-sensitive
plateau: legacy interior order is only 0.467 and its density order is
negative, while both crossing-owner orders are negative.

For rot17 at N=128, owner required 42 iterations versus 51 and the two errors
were close: owner/legacy ratios were 1.10 for interior Linf and 0.97 for
density Linf.  For rot17 plus translation, owner required 47 iterations
versus 62, but its N=128 interior and density errors were respectively 1.71
and 2.11 times the legacy errors.  The numerical conclusion is therefore
that crossing ownership often improves the Krylov behavior on rotated
geometries, while its accuracy effect is pose-dependent and is not uniformly
better.

## Timing and owner preprocessing scope

No wall-clock performance conclusion is drawn from this run: an older
worktree process overlapped the N=128 portion, so those timing rows were not
collected under isolated load.  The shared full-certified owner setup is
still visibly much more expensive than an individual solve and remains the
main preprocessing optimization target.

The study uses the full certified NURBS intersection owner builder available
on `main`.  The older branch's RegionClosestHybrid owner preprocessor was not
integrated because its geometry API diverges from the current mapped
Cartesian-edge/certified backend.  Hybrid owner preprocessing therefore
remains future performance work; no Hybrid timing or equivalence is inferred.

The corrected `correction_panels` and `crossing_panels` fields use
`Interface3D::num_panels()`, not interpolation-point counts.  Owner CSV fields
separately expose decision-only, legal exception-fallback, and total
classified-or-fallback counts.
