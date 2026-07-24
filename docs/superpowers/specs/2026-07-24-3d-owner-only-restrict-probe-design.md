# 3D crossing-owner-only restrict probe design

## Goal

Add a high-resolution numerical-probe entry point that evaluates only the
`JointTricubicCrossingOwner` exterior-normal restrict route for the existing
three L-prism rigid poses.  This avoids spending `256^3` grid work on the
legacy route and the deliberately nonconvergent 160-step exterior-only
reference route.

## Interface

- Add `--restrict-probe-owner [N ...]`.
- Keep `--restrict-probe [N ...]` unchanged.
- Reuse the existing power-of-two and `N >= 16` validation.
- Write owner-only results to
  `output/dirichlet_normal_restrict_crossing_owner_3d`, so existing A/B
  results are not overwritten.

## Execution

- Build the same native NURBS geometry, surface DOFs, Cauchy stencils, and
  crossing-owner templates as the A/B probe.
- Do not construct the exterior-only restrict object.
- Execute only `JointTricubicCrossingOwner`.
- Preserve the existing physical/common GMRES tolerance and iteration cap.
- Preserve owner classification, reroute audit, residual history, setup
  timing, and localization diagnostics.
- Skip the A/B causal-hypothesis verdict because the owner-only run does not
  contain a legacy route.

## Verification

1. Before implementation, `--restrict-probe-owner 16` must fail because the
   entry point does not exist.
2. After implementation, the same command must succeed and produce exactly
   three completed rows, all named `joint_tricubic_crossing_owner`.
3. The existing `--restrict-probe 16` command must still produce all three
   routes for all three poses.
4. Run `--restrict-probe-owner 256`, then compute
   `log2(error_128 / error_256)` from the archived `N=128` results and report
   physical GMRES iterations.
