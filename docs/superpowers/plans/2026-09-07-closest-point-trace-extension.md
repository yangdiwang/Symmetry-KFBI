# Closest-point trace extension implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent geometry, trace, and validation tasks. Root owns application integration.

**Goal:** Apply boundary closest-point anchored jump extension to the hollow-cylinder Q27 preprocessing failure, and measure correctness and cost.

**Architecture:** An immutable NURBS closest-point index combines feasible polar evaluations with conservative subdivision bounds. A trace helper admits a localized smooth projection on the target G1 sheet. The application evaluates its existing Cauchy jump polynomial at the support node using this anchor, then keeps the Q27 interpolation and derivative weights. Ineligible nodes use the explicit legacy all-event route; their reasons and cost are recorded.

**Tech Stack:** C++17, Eigen 3.4, existing CGAL/exact NURBS extraction, CMake, MSVC 19.16 Release.

**Spec:** User-approved design in the preceding conversation: nearest boundary point, local jump extension of matching order, side-dependent correction, retained Q27 interpolation; preserve checks for ambiguous projections, incompatible walls, and nonsmooth edges.

## Global constraints

- Preserve all pre-existing uncommitted directory reorganization and polar optimization. No commits or pushes are required.
- `KFBIM_3D_SUPPORT_PATH=closest_point` selects the new branch. Existing legacy/native_certified choices retain their meaning.
- Distance bounds do not prove uniqueness. Report localization and ties separately; the trace consumer checks eligibility.
- No analytic cylinder shortcut in production. Analytic geometry supplies independent test expectations.
- Never change Cartesian grid-edge intersection/topology construction or relax intersection certification tolerances.
- Preserve the existing Cauchy polynomial degree, curvature terms, direct coefficient density binding, and Q27 weights.
- Geometry workspaces are caller-owned and reused. Cache projections by grid node while the geometry is immutable.
- Fallback is explicit and counted. Do not discard an unresolved legacy path as successful preprocessing.
- Use serial builds in build-layout; avoid concurrent MSBuild use. Save baseline artifacts under a separate directory.

## Task 1: Bounded closest-point queries (review_core)

Files: `src/geometry/nurbs_surface_closest_point_3d.hpp/.cpp`, `tests/geometry/nurbs_surface_closest_point_3d_test.cpp`, geometry/test CMake entries.

Interface: `NurbsSurfaceClosestPointIndex3D(model).query(A, options, workspace)`. Result exposes point/normal/patch/u/v, lower and upper distance bounds, localization radius/status, separated near ties, and work counters. Only a proved distance gap earns `Bounded`; exhausted budgets remain distinguishable.

- [x] Write tests and observe RED for the missing query behavior: analytic plane/cylinder distances, the recorded A=(.5625,-.28125,-.375), equidistant walls, seams, and bounded work exhaustion.
- [x] Implement constrained local minimization with polar jets and conservative best-first bounds/subdivision. Preserve feasible surface parameters, include parameter boundaries, and reuse buffers.
- [x] Run geometry tests and inspect projection accuracy, localization, bounds, and work counts.

## Task 2: Trace anchor eligibility and manufactured extension (review_3d)

Files: `src/support/trace/closest_point_trace_extension_3d.hpp/.cpp`, `tests/trace/closest_point_trace_extension_3d_test.cpp`; root adds support/test CMake entries.

Interface: `select_closest_point_trace_anchor_3d` consumes native surface, bounded closest-point result, node A, target P/patch/normal, and a local distance limit. It returns an eligible native owner or an explicit rejection reason. Never replace the chosen owner with a nearest existing grid-line crossing.

- [x] Observe RED for smooth target-sheet selection and rejection of unrelated walls/nonsmooth owners/unlocalized results.
- [x] Implement target G1-sheet, regular normal, locality, and normal-projection checks.
- [x] Use real Cauchy reconstruction and Q27 weights to reproduce both sides of manufactured harmonic quadratic fields. Assert value and normal traces against independent analytic values, including the near-tangent failure geometry.

## Task 3: Application integration (root)

File: `apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp`.

- [x] Add the closest_point support mode and separate output directory. Reject incompatible non-cover routes.
- [x] Build one index and workspace for template construction, cache a query per grid node, and collect timing/query/fallback diagnostics.
- [x] For an eligible anchor, reuse `build_direct_crossing_plan(owner,Q,...)` and `compose_shared_quadratic_correction(slot,node,...)`; assign `continuation_sign=int(desired_inside)-int(node_inside)` and bind the true closest-point owner. Keep existing recovery formulas.
- [x] Retain legacy all-event handling for ineligible nodes and report rejection categories.
- [x] Add an explicit preprocessing-only execution option with output indicating its scope; retain required direct-density binding/validation before claiming the requested stage is complete.
- [x] Apply the minimal unique_ptr ownership correction needed to build the existing noncopyable ReducedTraceProjection3D factory under MSVC 2017. Preserve its numerical construction.

## Task 4: Application validation and review (review_validation + root)

Files: `.cache/closest-point-validation/`, `output/nurbs_closest_point_20260907/`, `docs/KFBI3D_Closest_Point_Extension_20260907.md`.

- [x] Preserve the existing failed legacy N32 result as RED evidence; exercise the new mode on the same model/grid and record completion or the next concrete blocker.
- [x] Build `kfbi_topology_affine_exterior_trace_3d` and both new tests in Release. Run relevant polar, Cauchy, trace, and endpoint regression tests.
- [x] Run full N32 preprocessing; if successful, check N64 and manufactured value/normal convergence. Do not substitute a successful single query for full preprocessing.
- [x] Report nearest-query work, cache reuse, fallback counts/timing, completed stages, and speed comparisons with explicit scope.
- [x] Obtain independent review of geometry bounds and application sign/owner integration, resolve blocking findings, and verify diffs.

## Initial validation commands

```powershell
& D:/cmake/bin/cmake.exe --build build-layout --config Release --target nurbs_surface_closest_point_3d_test closest_point_trace_extension_3d_test kfbi_topology_affine_exterior_trace_3d --parallel 1 -- /nr:false
$env:PATH='D:/CGAL/CGAL-5.2-beta1/auxiliary/gmp/lib;' + $env:PATH
& D:/cmake/bin/ctest.exe --test-dir build-layout -C Release --output-on-failure -R '^(nurbs_surface_closest_point_3d_test|closest_point_trace_extension_3d_test)$'
$env:KFBIM_3D_SUPPORT_PATH='closest_point'
$env:KFBIM_3D_SOLVE_SELECTION='neumann_only'
$env:KFBIM_3D_PREPROCESS_ONLY='1'
& ./build-layout/apps/Release/kfbi_topology_affine_exterior_trace_3d.exe hollow_cylinder 32
```

The runner records the effective environment and exact command; verify existing solve-selection variable names before use.
