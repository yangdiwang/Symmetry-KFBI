# 3D L-prism Neumann value-trace crossing-owner comparison

Date: 2026-07-24

## Configuration

- Geometry: native-NURBS L-shaped prism, identity transform.
- Grid levels: `N=32,64,128`.
- Approximation: all cubic (`3/3/3`).
- Cauchy policy and counts: `topological_nearest`, `48/28`.
- Neumann compatibility: `operator_flux`.
- GMRES: relative tolerance `2e-10`, maximum 80 iterations.
- Solve selection: Neumann only.
- Legacy route: `joint_tricubic_cauchy`.
- New route: `joint_tricubic_crossing_owner`.

The new route is used consistently by the Neumann operator, its right-hand
side, the operator-flux border column, and the final exterior value-trace
diagnostic. Geometry queries used to select owners are performed only while
the trace templates are constructed, not inside GMRES operator applications.

## Interior solution error and GMRES

The reported order at `N=64` is measured from `32 -> 64`; the order at
`N=128` is measured from `64 -> 128`.

| N | Legacy iterations | Owner iterations | Legacy interior Linf | Owner interior Linf | Owner change | Legacy order | Owner order |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 28 | 28 | 1.6672851278e-6 | 1.6746601433e-6 | +0.4423% | — | — |
| 64 | 24 | 24 | 4.6978036827e-7 | 4.6914342733e-7 | -0.1356% | 1.827443 | 1.835767 |
| 128 | 32 | 32 | 1.3115552577e-7 | 1.3125969178e-7 | +0.0794% | 1.840708 | 1.837605 |

| N | Legacy interior L2 | Owner interior L2 | Owner change | Legacy L2 order | Owner L2 order |
|---:|---:|---:|---:|---:|---:|
| 32 | 5.9321063671e-7 | 5.9543031389e-7 | +0.3742% | — | — |
| 64 | 1.3256817268e-7 | 1.3252192168e-7 | -0.0349% | 2.161810 | 2.167702 |
| 128 | 3.2683861530e-8 | 3.2734447547e-8 | +0.1548% | 2.020084 | 2.017349 |

The crossing-owner route does not change the GMRES iteration count at any
level. Its change in the interior error is below 0.5%, and both routes retain
the same approximately `1.83-1.84` Linf order and second-order L2 behavior.

## Exterior value trace and compatibility correction

| N | Legacy exterior-trace Linf | Owner exterior-trace Linf | Owner change | Legacy operator correction | Owner operator correction |
|---:|---:|---:|---:|---:|---:|
| 32 | 3.5228646652e-11 | 1.5136118582e-11 | -57.0346% | 1.4875150586e-6 | 1.4845673794e-6 |
| 64 | 4.3646835085e-11 | 2.3399332817e-11 | -46.3894% | 3.0632066752e-7 | 3.0621299540e-7 |
| 128 | 8.3250280044e-11 | 1.1803236229e-10 | +41.7801% | 9.6131758553e-8 | 9.6083607865e-8 |

At `N=32,64`, crossing-owner reduces the final exterior-trace residual, but
this improvement reverses at `N=128`. Every value is nevertheless far below
the physical trace threshold `10 * tolerance = 2e-9`, so the change does not
alter GMRES termination. The operator-flux correction changes by at most
0.20%, which is too small to explain the internal Linf convergence behavior.

## Cost

Measured end-to-end wall time:

- Legacy, all three levels in one run: approximately 149.5 seconds.
- Crossing-owner, three separate level runs: approximately
  `120.5 + 429.6 + 1217.0 = 1767.1` seconds.
- Total crossing-owner wall time is approximately 11.8 times the legacy run.

The GMRES solve times themselves remain comparable:

| N | Legacy solve seconds | Owner solve seconds |
|---:|---:|---:|
| 32 | 0.8882 | 0.8047 |
| 64 | 4.6059 | 4.3860 |
| 128 | 44.5154 | 34.6178 |

The extra time is therefore almost entirely in precomputing NURBS segment
intersections and correction owners.

## Conclusion

For the aligned L-prism Neumann manufactured problem, assigning wrong-side
trace corrections to their crossing owner is not the dominant cause of the
sub-second-order interior Linf behavior:

1. GMRES iteration counts are identical.
2. Interior errors and observed orders are effectively unchanged.
3. The operator compatibility correction is effectively unchanged.
4. Exterior-trace residual changes are non-monotone and already well below
   the stopping threshold.

The crossing-owner idea remains useful for diagnosing rotated geometries and
for local sharp-edge behavior, but the present full-surface precomputation is
too expensive to use as the default path without caching or limiting geometry
queries to degrees of freedom close to non-G1 edges.

## Raw outputs

- Legacy: `output/neumann_exterior_zero_trace_3d/analysis/neumann_value_restrict_compare_2026-07-24_legacy/neumann_results.csv`
- Crossing-owner N32:
  `output/neumann_exterior_zero_trace_3d/joint_tricubic_crossing_owner/analysis/neumann_value_restrict_compare_2026-07-24_N32/neumann_results.csv`
- Crossing-owner N64:
  `output/neumann_exterior_zero_trace_3d/joint_tricubic_crossing_owner/analysis/neumann_value_restrict_compare_2026-07-24_N64/neumann_results.csv`
- Crossing-owner N128:
  `output/neumann_exterior_zero_trace_3d/joint_tricubic_crossing_owner/analysis/neumann_value_restrict_compare_2026-07-24_N128/neumann_results.csv`
