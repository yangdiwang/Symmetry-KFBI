# NURBS Correction Safety and Point Classification Results

Date: 2026-07-23

## Scope

This batch separates point-classification sufficiency from KFBI correction safety. Primary NURBS intersection work is bounded at subdivision depth 4; only ambiguous component-membership-changing Cartesian edges may receive a targeted depth-6 retry. A correction is accepted only for one certified transverse root with known count/parity and no ambiguous or near-tangent candidate.

L-prism Dirichlet-normal accuracy orders and iteration counts are intentionally not acceptance gates in this batch.

## Implemented changes

1. Added direct analytic point-containment tests for the torus, hollow cylinder, L-prism reentrant corner, and two translated components, with triangle seeds both enabled and disabled. Exact surface points now report `point lies on NURBS surface`.
2. Added a persisted confidence record for every queried Cartesian edge and public `edge_classification_between` / strict `correction_crossing_between` accessors.
3. Changed the public local subdivision defaults from 36 to 4. Close-root stationary witnesses can now use adjacent Bezier elements on the same patch and directly connected G1 patches, but never cross a non-G1 connection.
4. Added lazy depth-6 retry only when a depth-4 result is ambiguous and the endpoint component-membership vectors differ. The retry result becomes authoritative.
5. Removed GridPair3D's permissive native-crossing fallback. Every component-label-changing edge must pass the strict correction-safety lookup before correction construction.
6. Added safety/retry diagnostics to console output, readiness CSV output, and CI smoke assertions.

## Geometry acceptance

All rows have label_mismatches=0, gap_crossings=0, triangle_fallback_crossings=0, and unsafe_label_changing_edges=0.

| Geometry | N | Safe edges | Retry resolved/total | Primary/target depth | Max root residual / tolerance |
|---|---:|---:|---:|---:|---:|
| torus | 32 | 686 | 0/0 | 4/0 | 2.056e-12 / 2.159e-12 |
| cylinder | 32 | 1160 | 0/0 | 4/0 | 1.629e-12 / 2.027e-12 |
| L-prism | 32 | 982 | 0/0 | 0/0 | 3.331e-16 / 2.138e-12 |
| torus | 64 | 2758 | 0/0 | 4/0 | 2.150e-12 / 2.159e-12 |
| cylinder | 64 | 4544 | 0/0 | 4/0 | 7.080e-13 / 2.027e-12 |
| L-prism | 64 | 3926 | 0/0 | 0/0 | 2.719e-16 / 2.138e-12 |
| torus | 128 | 11404 | 2/2 | 4/6 | 2.115e-12 / 2.159e-12 |
| cylinder | 128 | 17812 | 0/0 | 4/0 | 1.775e-12 / 2.027e-12 |
| L-prism | 128 | 15122 | 0/0 | 0/0 | 2.719e-16 / 2.138e-12 |

For torus N=128, the primary pass found five ambiguous-parity edges. Two changed component membership and were both certified by targeted retry; the remaining three were membership-preserving endpoint fallbacks and cannot authorize a KFBI correction.

## Solver observations

All 18 GMRES runs reported converged=1. Errors are Linf; orders are relative to the previous level of the same geometry.

| Geometry | N | Neumann iter, density (p), interior (p) | Dirichlet-normal iter, density (p), interior (p) |
|---|---:|---|---|
| torus | 32 | 23, 1.138e-6 (-), 1.090e-6 (-) | 15, 3.030e-6 (-), 4.060e-7 (-) |
| torus | 64 | 22, 2.934e-7 (1.956), 2.882e-7 (1.919) | 13, 8.018e-7 (1.918), 7.566e-8 (2.424) |
| torus | 128 | 21, 7.211e-8 (2.025), 7.353e-8 (1.970) | 12, 1.975e-7 (2.021), 1.569e-8 (2.270) |
| cylinder | 32 | 24, 3.687e-6 (-), 3.687e-6 (-) | 31, 1.434e-5 (-), 2.303e-6 (-) |
| cylinder | 64 | 23, 7.205e-7 (2.355), 7.375e-7 (2.322) | 25, 1.239e-6 (3.533), 1.834e-7 (3.650) |
| cylinder | 128 | 26, 2.340e-7 (1.622), 2.315e-7 (1.672) | 24, 2.642e-7 (2.230), 2.921e-8 (2.650) |
| L-prism | 32 | 28, 2.918e-6 (-), 2.865e-6 (-) | 54, 4.409e-5 (-), 1.959e-6 (-) |
| L-prism | 64 | 24, 5.037e-7 (2.534), 6.219e-7 (2.204) | 82, 4.752e-6 (3.214), 4.018e-7 (2.286) |
| L-prism | 128 | 31, 1.992e-7 (1.338), 2.136e-7 (1.542) | 99, 5.865e-5 (-3.625), 1.074e-6 (-1.418) |

The last L-prism Dirichlet-normal row confirms the previously observed accuracy/iteration instability; it is recorded, not hidden, and remains deferred from this geometry-safety acceptance.

## Verification

- Full Release build completed.
- The native NURBS surface executable passed all direct containment, intersection, retry, edge-safety, and GridPair regressions.
- CTest reports that this project currently has no registered CTest tests; the native executable is run directly.
- The full application emitted all nine ready cases and its final completion message. The combined readiness CSV contains nine rows and the two convergence CSV files contain all solver rows.
- `git diff --check` is used as the final whitespace check before commit.

Generated evidence:

- `output/neumann_exterior_zero_trace_3d/geometry_readiness.csv`
- `output/neumann_exterior_zero_trace_3d/neumann_results.csv`
- `output/neumann_exterior_zero_trace_3d/dirichlet_normal_results.csv`
