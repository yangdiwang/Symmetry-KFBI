# 3D rotated N=128 phase-profile design

## Goal

Measure a reproducible, non-overlapping wall-clock breakdown of the current
`JointTricubicCrossingOwner` KFBI route for the rotated L-prism at `N=128`.
The result must expose the cost of NURBS/grid intersection, Cauchy setup,
spread, FFT bulk solves, crossing-owner restrict, GMRES algebra, and
diagnostic output without changing the numerical algorithm.

## Scope and isolation

- Profile only `rot_axis123_17deg` at `N=128`.
- Work in an isolated worktree based on the current `main` HEAD.
- Add a profile-only command-line entry point; keep all existing entry points
  and default behavior unchanged.
- Do not combine profiling with an optimization or numerical-method change.
- Write profile output to a dedicated directory so existing convergence data
  are not overwritten.

## Timing model

Use `std::chrono::steady_clock` and report the following mutually exclusive
leaf categories.

### Setup

1. Cartesian/NURBS domain construction, grid intersection, and
   inside/outside classification.
2. Surface degrees of freedom and Cauchy-stencil construction.
3. Grid-pair construction and label validation.
4. Fixed pipeline initialization: fit maps, FFT plans, and correction
   support, excluding the separately timed crossing and trace-template work.
5. Crossing-row construction.
6. Crossing-owner trace-template construction, split into:
   - NURBS segment-intersection queries;
   - owner selection and remaining template assembly.
7. Exact/smooth field construction and other setup.

### Operator and solve

1. Local Cauchy-coefficient construction.
2. Spread/right-hand-side assembly.
3. FFT bulk solve.
4. Continued-sample evaluation and crossing-owner restrict recovery.
5. GMRES vector algebra and remaining route/post-processing.

### Output

1. Detailed diagnostic and summary-file output.
2. Unclassified wall overhead, calculated as wall time minus all measured
   leaf categories.

Parent timers such as `pipeline_setup_seconds`, `setup_seconds`, and
`route_seconds` remain validation totals only and are never added to the leaf
categories.

## Instrumentation

- Enable detailed timers only for the profile entry point.
- Count calls as well as elapsed seconds for intersection, spread, FFT, and
  restrict so repeated GMRES applications are visible.
- Time the existing `intersect_segment` call directly in trace-template
  construction. Do not infer intersection cost from the whole pipeline
  constructor.
- Time coefficient, spread, and FFT regions inside the existing operator
  evaluation, and continued-sample/recovery regions inside the existing
  restrict route.
- Attribute the untimed remainder of a validated parent region to its named
  remainder category, preventing overlap.
- Emit a machine-readable `phase_profile.csv` with category, seconds, calls,
  percentage of algorithm time, and percentage of total wall time.

Two denominators are reported:

- **algorithm time**: all setup and operator/solve leaf categories, excluding
  diagnostic output and unclassified wall overhead;
- **wall time**: complete process time, including output and overhead.

Because per-query clocks may perturb millions of intersection calls, compare
the profiled wall time with the existing aggregate run. If measured overhead
exceeds 5%, replace per-query timing with coarser batched timing or explicitly
report the overhead and exclude it from performance conclusions.

## Numerical validation

The profiled run must reproduce the archived rotated `N=128` reference:

- interior maximum error: `5.3527160592814482e-08`;
- physical/common GMRES iterations: `25/23`;
- physical residual no larger than `8.2e-11`;
- common residual no larger than `1.9e-10`;
- no geometry/intersection query is allowed during a GMRES iteration.

Floating-point output should be bitwise identical when practical. Otherwise,
the error and residual differences must be at most a relative `1e-12`, with
the iteration counts identical.

## Deliverables

1. The raw profile CSV and run log for rotated `N=128`.
2. A table of seconds and both percentages for every leaf category.
3. Call counts and average costs for the repeated kernels.
4. Acceleration proposals ranked by measured time share and expected risk.
5. A statement of profiling overhead and numerical-equivalence checks.
