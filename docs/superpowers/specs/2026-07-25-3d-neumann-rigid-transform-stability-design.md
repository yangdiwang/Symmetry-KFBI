# 3D Neumann L-Prism Rigid-Transform Stability Study Design

## Goal

Add a Neumann counterpart to the existing Dirichlet L-prism
rigid-transform study.  The new study runs the accepted Neumann
crossing-owner value-trace route for the same eight rigid poses at
`N=32,64,128`, and reports accuracy, observed order, GMRES behavior,
cost, geometry integrity, owner-preprocessing integrity, and pose
sensitivity.

## Scope

The new study uses:

- the native twelve-patch NURBS L-prism;
- the existing transformed non-polynomial harmonic manufactured solution;
- degree-3 local harmonic Cauchy fitting;
- `G1Nearest` Cauchy stencils with 48 value and 28 normal conditions;
- tricubic Cartesian restriction;
- `JointTricubicCrossingOwner` exterior-value restriction;
- `RegionClosestHybrid` crossing-owner preprocessing;
- the current bordered exterior-zero-value-trace Neumann equation;
- GMRES tolerance `2e-10`, restart 80, and maximum 80 iterations.

The study does not change the Neumann equation, box solver, spread, FFT,
surface-DOF construction, geometry classifier, or Cauchy polynomial.

The existing `--neumann-owner-study` remains the focused legacy versus
crossing-owner A/B comparison.  The new rigid study runs only the accepted
crossing-owner route, so the two study modes retain distinct purposes.

## Rigid Poses

The study consumes the same catalog and ordering as the Dirichlet study:

| Case ID | Rotation | Translation |
|---|---|---|
| `baseline` | identity | `(0,0,0)` |
| `tx_p0137` | identity | `(0.137,0,0)` |
| `ty_m0083` | identity | `(0,-0.083,0)` |
| `tz_p0061` | identity | `(0,0,0.061)` |
| `t_xyz_1` | identity | `(0.137,-0.083,0.061)` |
| `t_xyz_2` | identity | `(-0.109,0.151,-0.047)` |
| `rot_axis123_17deg` | 17 degrees about normalized `(1,2,3)` | `(0,0,0)` |
| `rot_axis123_17deg_t_xyz_1` | same rotation | `(0.137,-0.083,0.061)` |

All rotations use center `(0.07,-0.07,0.02)`.  Geometry control points,
exact values, and exact gradients transform covariantly through the existing
`RigidTransform3D` implementation.

A generic L-prism rigid-case accessor will be added while preserving the
existing Dirichlet-named accessor as a compatibility wrapper.  Dirichlet and
Neumann therefore cannot silently acquire different pose definitions.

## Architecture

### Pure study evaluation

A focused `neumann_rigid_transform_study_3d` module owns data-only
measurements and deterministic derived quantities:

- observed interior maximum-error order, grouped by case and route;
- same-level error and iteration ratios against `baseline`;
- per-case completeness and acceptance statuses.

This module has no FFT, geometry, filesystem, or solver dependency and is
covered by synthetic unit tests.  The application maps production solve
metrics into these data-only measurements.

### Production driver

Add:

```text
neumann_exterior_zero_trace_3d.exe --neumann-rigid-study [N ...]
```

With no explicit levels, it runs `32 64 128`.  Explicit levels must be an
ordered subset of `32,64,128`; selecting 64 requires 32, and selecting 128
requires both 32 and 64.  This prevents reporting an incomplete convergence
sequence as a complete study.

For each level, the driver processes all eight poses.  Each `(pose,N)`:

1. constructs transformed native NURBS geometry, exact labels, surface DOFs,
   and G1-nearest Cauchy stencils;
2. constructs one owner-enabled harmonic-jet pipeline with
   `RegionClosestHybrid`;
3. verifies the default/explicit legacy probe remains bitwise identical and
   that the crossing-owner probe is finite;
4. solves only `JointTricubicCrossingOwner`;
5. verifies owner diagnostics and geometry-query counts are unchanged across
   GMRES;
6. checkpoints every CSV after the completed solve.

A failed level stops refinement before the next level, while preserving all
completed checkpoint rows.

## Outputs

Write generated files under
`output/neumann_rigid_transform_stability_3d`:

- `rigid_transform_results.csv`: one row per `(case,N)`, including full
  transform metadata, setup and solve times, DOF count, GMRES convergence,
  iteration count and residual, interior maximum/RMS errors and orders,
  density maximum/RMS errors, exterior condition, operator residual,
  baseline ratios, and row acceptance;
- `gmres_residuals.csv`: complete residual history keyed by case and level;
- `owner_diagnostics.csv`: preprocessing mode, workload fingerprint, output
  digest, wrong-side count, query counts before/after GMRES, and invariant
  flags;
- `rigid_transform_acceptance.csv`: one row per pose with completeness,
  GMRES, monotonic-error, `64->128` order, baseline-ratio, geometry, owner,
  and overall statuses.

## Acceptance

A production row passes when:

- all reported floating-point metrics are finite;
- GMRES converges in at most 80 iterations;
- final relative residual is at most `2e-10`;
- domain-label mismatches are zero;
- no unsafe label-changing edge, gap crossing, endpoint crossing, or triangle
  ownership fallback remains;
- owner diagnostics and query counts are unchanged across GMRES.

For a complete `32,64,128` run, each pose additionally requires:

- strictly decreasing interior maximum error;
- `N=64 -> 128` interior maximum-error order at least `1.8`;
- each same-level interior maximum error at most three times the baseline
  error;
- all three level rows present and individually passing.

A failed numerical acceptance is reported faithfully and makes the study
exit nonzero; it is not hidden by weakening thresholds.

## Verification

Test-first coverage will prove:

- Neumann and Dirichlet receive the identical eight-case catalog;
- level validation accepts only the required refinement prefixes;
- orders are grouped by pose and are not mixed between transformations;
- baseline ratios use the matching grid level;
- complete passing, incomplete, GMRES-failing, geometry-failing, and
  owner-invariant-failing synthetic studies receive the expected statuses;
- the existing Dirichlet catalog test and Neumann owner-study route remain
  unchanged.

Final numerical verification runs all eight poses at `N=32,64,128`, then
reports every pose's interior maximum error, observed order, GMRES iteration
count, and setup/solve/total time.
