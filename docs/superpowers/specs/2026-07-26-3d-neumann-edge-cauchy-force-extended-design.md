# 3D Neumann edge-Cauchy forced extended evidence design

## Goal

Obtain reproducible `N=128` evidence for the existing
`none` versus `non_g1_auxiliary_values` study even when the fixed
`N=32,64` adoption gate fails. The experiment must not weaken, bypass,
or reinterpret any recorded acceptance criterion.

## Command-line interface

The existing command remains unchanged:

```text
neumann_exterior_zero_trace_3d --neumann-edge-cauchy-study 32 64 128
```

It continues to stop before `N=128` when the coarse acceptance result is
`fail`.

Forced extended evidence uses an explicit option:

```text
neumann_exterior_zero_trace_3d \
  --neumann-edge-cauchy-study --force-extended 32 64 128
```

`--force-extended` is valid only for the edge-Cauchy study. It changes
only the decision to enter the `N=128` loop. Level normalization still
requires the prefix `32,64,128`.

## Runtime semantics

Before entering `N=128`, the application recomputes and checkpoints the
complete `N=32,64` acceptance exactly as it does now.

- If coarse acceptance passes, execution enters `N=128` normally.
- If coarse acceptance fails and forcing is disabled, execution stops.
- If coarse acceptance fails and forcing is enabled, execution emits a
  clear warning and continues into `N=128`.

All numerical routes, geometry data, Cauchy stencils, edge samples,
GMRES parameters, tolerances, and acceptance thresholds remain
unchanged. A forced run retains the failed coarse `overall_pass` and
therefore may finish with a nonzero process exit code even when all
`N=128` rows complete. The nonzero exit continues to mean “not accepted,”
not “the experiment failed to produce evidence.”

The study continues writing checkpoints after every A/B pair. Runtime
or numerical failure at `N=128` is represented by the existing explicit
failed rows and extended-evidence status.

## Evidence isolation

The forced run will set:

```text
KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR=
output/neumann_edge_cauchy_3d_n128_forced
```

This preserves the authoritative coarse pilot under
`output/neumann_edge_cauchy_3d`. A complete forced run should contain
18 summary rows: three rigid poses, two modes, and three grid levels.

## Implementation boundary

The gate decision will be exposed as a small pure function in
`neumann_edge_cauchy_study_3d` so its four truth-table cases can be
tested independently. The application parser will recognize the
explicit option and pass the resulting Boolean into
`run_neumann_edge_cauchy_study_3d`.

No default mode, public numerical route, or adoption decision changes.

## Verification

Testing proceeds test-first:

1. Add failing unit tests for the gate truth table: default failure
   blocks, forced failure continues, and a passing coarse gate continues
   in either mode.
2. Implement the minimal pure gate function and application wiring.
3. Run the edge-Cauchy study tests, the edge-augmented Cauchy tests, and
   a full Release build.
4. Verify the unforced command still stops before `N=128`.
5. Run the forced `32,64,128` experiment into the isolated output
   directory.
6. Independently recompute `64 -> 128` orders from raw errors and report
   the `N=128` GMRES iterations for every pose and mode.

## Reporting

The final report will show, for each rigid pose and both modes:

- internal-solution `Linf` errors at `N=64` and `N=128`;
- observed `64 -> 128` order;
- `N=128` GMRES iteration count and terminal residual;
- whether the augmented route improves error, iteration count, and
  incident-edge discrepancy relative to the baseline.

Any incomplete row, convergence failure, or degradation will be
reported directly rather than hidden by the forced execution option.
