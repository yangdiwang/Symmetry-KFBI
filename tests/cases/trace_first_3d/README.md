# Python-compatible trace-first cases

For the latest **Trace93** package, use [the Trace93 section below](#trace93-solid-cylinder-l-prism-and-u-prism),
`run_trace93.ps1` and `trace93_cases.json`. The first sections below document
the preserved, different torus benchmark; their defaults do not describe Trace93.
See the [Trace93 algorithm alignment](../../../docs/Trace93_Algorithm_Alignment_20260909.md)
and [measured phase report](../../../docs/Trace93_Numerical_Report_20260909.md).
The completed PDE scope of this phase is rotated L prism, N32, both BVPs;
the remaining long PDE runs were stopped/deferred at the user's request.

`python_torus.json` is the portable case specification for the current Python core.
The matching C++ factory is `make_trace_first_python_torus_case_3d()` in
`src/support/geometry/trace_first_case_3d.*`. It does not replace the historical
C++ torus (R=0.55, r=0.20, translated local center).

The geometry is exact degree (2,2), not bicubic. Density is independent cubic
B-splines. Patch number is `4*major_quarter+minor_quarter`, with u following the
major circle and v following the minor circle. Rigid transforms are applied to
both the geometry and the manufactured harmonic field. The analytic inside
predicate is an audit/error oracle; it must not replace native event labels.

The density allocations are nested 4x2, 8x4 and 16x8 spans per patch. They are
fixed benchmark targets, not an automatic post-constraint DOF adequacy loop.
The 8x8 boundary-mean rule on each geometry patch and the 8x8 Neumann gauge rule
on each density element are different integrals; preserve that distinction.

Numerical comparisons must match N, transform, density, trace quadrature,
center policy and coordinate chart. Python's default chart is extended tubular;
the existing C++ resource-mixed route uses physical tangent graph coordinates.
Agreement of the case specification alone does not make those two operators
mathematically identical. Runtime C++ uses the typed factory (no JSON parser
dependency); this JSON is an export specification mirrored by its unit tests.

## Run the numerical study

From the repository root, on the configured Windows build host:

```powershell
./tests/cases/trace_first_3d/run.ps1 -Levels 32,64 -Bvp both -Transform rotate
```

The default is the new trace-first route, extended tubular evaluation,
all-event **Spread**, and native whole-grid-line reuse. Trace-first Restrict
does not certify each support path. It retains Python's q-priority rule;
`event-mode=all` must not be read as a path certificate for trace Restrict.
All normal samples use the shared 3+3 reconstruction: Q27-cover3 for Neumann,
Q64-cover4 for Dirichlet. The old main application's defaults are unchanged.

Useful controlled comparisons:

```powershell
# Same mathematical operator, cached versus uncached evaluation rows.
./tests/cases/trace_first_3d/run.ps1 -Levels 32 -CompareCache -SkipBuild
# Only replace whole-line geometry queries by the old per-edge queries.
./tests/cases/trace_first_3d/run.ps1 -Levels 32 -GridLines off -SkipBuild
# Python's endpoint-event Spread selection, with local native root finding.
./tests/cases/trace_first_3d/run.ps1 -Levels 32,64 -EventMode python -SkipBuild
```

Policies are `trace`, `event`, `trace-spread` and `event-spread`. The latter
two mean trace Spread/event Restrict and event Spread/trace Restrict. Event
Restrict is an intentionally expensive native all-event reference; do not
include its setup time in a trace-first performance claim. `-Chart physical`
is a separate discretization, not Python's default extended tube.

Results go to a unique `results/<run>/` directory here. Existing runs are never
overwritten. `case.json` snapshots the case specification; `summary.json`
collects completed solves and computes observed order between completed mesh
levels, without rerunning historical data. Per-level logs survive numerical
failure. A failure or nonconverged GMRES must not be reported as a passed run.

## Density and reuse contracts

The native square C0 density is only an evaluation carrier. Exact Boehm knot
insertion embeds the coarser minor-direction spline into it. The actual trial
space has precisely the reference anisotropic DOFs: N32 D240/N239, N64
D720/N719, N128 D2448/N2447. No density fitting is used in this embedding.
The trace counts are 2048, 8192 and 32768, respectively. Cholesky observability
checks supplement the strict `trace_count > reduced_dofs` guard.

Visits `(q, side, node)` first choose a trace anchor; only then may equal
`(degree, anchor, node)` evaluation rows be shared. The old `(sheet,node)`
premerge is not used for the q-priority rule. The temporary catalog is released
after assembling fixed matrices; GMRES performs no geometric queries.

Whole-line root sets are reused only after the existing intersector's
completeness check and a node/tangency/feature exclusion guard. The guard is
not a new interval root-position certificate. Unaccepted lines fall back to
the old per-edge algorithm. The following counters must balance:

```
direct_edge_query_count + whole_line_reused_edge_count = candidate_grid_edge_count
whole_line_accepted_count + whole_line_fallback_count = whole_line_query_count
```

## Reading the diagnostics

- `interior_linf`: maximum error on all labeled interior Cartesian nodes.
- `interior_l2`: node-wise RMS (not the unnormalized volume integral norm).
- `density_linf`, `density_l2`: maximum and area-normalized RMS on trace Gauss
  points; these are not independent validation-point errors.
- N enforces the exterior value trace; D enforces the exterior normal trace.
  Both full traces and the projected residual are reported. The unused trace
  is not another boundary condition being solved.
- `projection_leakage` measures the active full trace outside the trace trial
  space; a small projected residual alone does not imply a small full trace.
- `projection_not_converged` counts local tube projections that exhausted the
  stopping rule. The projection is local, not a global closest-point proof.
- `shared_geometry_seconds` is incurred once for both BVPs at one mesh level.
  `bvp_total_seconds` includes optional cache comparisons and matrix dumps.
  Poisson/matvec/GMRES timings overlap and must not be added together.
  `whole_line_intersection_seconds` is a subset of geometry intersection time.
- In the current tube adapter, some P2 centers obtain their transfer through
  P3. `p3_centers` therefore is not a count of Spread-only centers.

The current case driver is deliberately specific to the smooth periodic torus
atlas. It does not replace the general feature-edge/vertex affine constraints
with separable torus constraints. The native physical-graph route and existing
general-geometry applications remain available as independent baselines.

## Implementation and measured results (2026-09-09)

- [中文实现说明与代码索引](../../../docs/TraceFirst_Implementation_20260909.md)
- [中文详细数值报告：N32/64、Python 对照、资源复用及验证限制](../../../docs/TraceFirst_Numerical_Report_20260909.md)

## Trace93: solid cylinder, L prism and U prism

The newer source-and-summary package is a different benchmark family. Use
`trace93_cases.json` and `run_trace93.ps1`, not the torus runner above.
All native patches are exact bicubic NURBS. The cylinder's Python angular
analysis parameters and density space remain independent of rational geometry
parameters. L/U dimensions and patch layouts also differ from the historical
same-named local cases. Old factories and the torus study are preserved.

```powershell
# Three geometries, both boundary conditions, rotated configuration.
./tests/cases/trace_first_3d/run_trace93.ps1 -Levels 32,64
# The full 72-case matrix is explicitly selectable; this command is not a
# claim that all 72 C++ cases have already been run.
./tests/cases/trace_first_3d/run_trace93.ps1 -Levels 32,64,128 `
  -Transforms translate,rotate,rotate_same_translate,rotate_translate
```

The default is event-owner validated trace-first reuse (1.75h to the event,
3.25h to the target), with Python93 last-event selection after native root-set
certification. `-EventMode all` keeps all signed event contributions;
`-Policy event` is an exact event-center reference, not a complete reproduction
of the package's older cached-crossing `baseline` center policy.
`-CompareCache` compares local fixed operators with and without row reuse.
`-GridLines off` independently disables whole-grid-line reuse.

`python_reference_trace93.json` contains the 72 archive manifest records with
explicit provenance and ZIP SHA-256. Missing source fields remain null; the
archive does not contain the complete `fresh_runs/` directories advertised by
its text. The runner records new C++ results separately, preserves failures,
and computes orders only between converged local cases of the same geometry,
BVP and transform. A missing archive trace count is not compared as zero.

See [Trace93 algorithm alignment and code design](../../../docs/Trace93_Algorithm_Alignment_20260909.md).
