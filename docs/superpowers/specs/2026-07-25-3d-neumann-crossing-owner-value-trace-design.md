# 3D Neumann Crossing-Owner Value-Trace Study Design

## Goal

Apply the currently accepted 3D Dirichlet correction combination to the
Neumann exterior-zero-value-trace formulation and measure its effect without
changing the Neumann boundary integral equation.

The accepted combination is:

- degree-3 local harmonic Cauchy polynomial;
- G1-nearest surface stencil with 48 value and 28 normal conditions;
- tricubic Cartesian interpolation and cubic normal-line recovery;
- crossing-aware correction ownership;
- `RegionClosestHybrid` owner preprocessing.

The study compares the existing center-owned value-trace restriction with a
crossing-owned value-trace restriction on the same pipeline, geometry, grid,
right-hand side, and GMRES settings.

## Non-goals

- Do not change the first-kind Neumann equation or its mean-zero constraint.
- Do not change the box problem, spread, FFT solver, Cauchy fit, surface DOFs,
  geometry labeling, or GMRES implementation.
- Do not replace the existing Neumann route or make the new route the default
  before the comparison is accepted.
- Do not rerun the three-strategy owner-oracle benchmark: the selected owner
  workload is identical to the already accepted Dirichlet workload.

## Algorithm

Add a value-trace restriction route with two modes:

1. `JointTricubicCauchy`: the existing correction uses the center surface DOF
   for every wrong-side interpolation node.
2. `JointTricubicCrossingOwner`: each wrong-side node uses the owner selected
   by the precomputed crossing-owner template.

Both modes recover the exterior value with the existing `c0_weights_`; neither
mode divides by `h`. The normal-trace routes continue to use `c1_weights_/h`.
The internal corrected-sample assembly is shared so value and normal routes
cannot diverge in their owner logic.

Construct one `PanelCenterHarmonicJetKFBI3D` pipeline with
`RegionClosestHybrid`. It precomputes the owner templates once. Execute the
legacy and crossing-owner Neumann solves against that same pipeline:

- identical Cauchy matrices and surface data;
- identical bulk operator;
- identical correction accumulation order `q=0..63`;
- no geometry query during GMRES.

The Neumann operator remains

\[
u^+|_\Gamma = 0,\qquad [u_n]=g_N,
\]

with `[u]` and the scalar mean constraint as unknowns. Only the numerical
restriction of `u^+|_\Gamma` changes.

## Study Driver

Add a dedicated route:

```text
neumann_exterior_zero_trace_3d.exe --neumann-owner-study 32 64 128
```

The study uses the L-prism rotated by 17 degrees about the normalized
`(1,2,3)` axis. It runs `N=32`, `64`, and `128`; `N=128` starts only after both
smaller levels converge and pass route-integrity checks. GMRES is capped at
80 iterations with relative tolerance `2e-10`.

Write generated, untracked CSV files under
`output/neumann_value_trace_crossing_owner_3d`:

- `summary.csv`: one row per level and restriction mode;
- `gmres_residuals.csv`: complete residual history;
- `owner_diagnostics.csv`: preprocessing mode, query counts, and pre/post
  GMRES fingerprints/counters.

## Reported Metrics

For both restriction modes report:

- interior maximum and RMS errors after the Neumann constant shift;
- observed `N32->N64` and `N64->N128` orders;
- value-jump density maximum and RMS errors;
- exterior value-trace and operator residuals;
- direct/opposite-trace route mismatch;
- GMRES convergence, iteration count, and final residual;
- pipeline setup and solve times;
- owner geometry queries before and after GMRES.

The numerical comparison uses the existing transformed non-polynomial
manufactured harmonic solution.

## Correctness and Acceptance

Before numerical conclusions are accepted:

- existing crossing-owner, preprocessor, and phase-profile unit tests pass;
- a focused value-trace test proves that the legacy route is unchanged;
- a foreign non-G1 crossing test proves that the new route uses the selected
  foreign owner while preserving `q=0..63` accumulation order;
- both modes converge within 80 iterations at every requested level;
- final GMRES residual is at most `2e-10`;
- owner diagnostics and query counts are unchanged across GMRES;
- all reported values are finite and generated rows are complete.

The study does not require the new route to be more accurate. A deterioration
is a valid result and keeps the route experimental. Recommend it for Neumann
only if it preserves convergence order, does not destabilize GMRES, and
materially improves error or pose robustness.

## Expected Outcome

Crossing ownership should mainly alter samples near non-G1 edges and rotated
wrong-side tricubic supports. It can reduce grid-position and pose sensitivity,
but the gain should be smaller than for the Dirichlet normal trace because the
value trace has no `1/h` derivative amplification. The Neumann operator remains
first-kind, so a large reduction in GMRES iterations is not expected.
