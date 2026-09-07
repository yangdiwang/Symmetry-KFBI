# NURBS polar evaluation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement and review the tasks below.

**Goal:** Reduce the measured 3D segment intersection evaluation/allocation cost without weakening intersection certification.

**Architecture:** Extract homogeneous base Bézier spans once, evaluate position and first partials together using tensor-product de Casteljau, and reuse a caller-owned workspace. Intersector-owned immutable per-patch evaluators select the original knot span; subdivision boxes remain constraints and certificates. Newton retains an accepted trial jet for its next iteration.

**Tech stack:** C++17, Eigen, existing CMake/CTest, local MSVC 19.16 Release.

**Spec:** User-approved design in this conversation: homogeneous polar form, necessary reusable scratch, original base spans, original knot-side semantics, and accepted Newton jet reuse.

## Global constraints

- Work in the current user-requested workspace, preserving all existing uncommitted layout changes. No commits or pushes.
- Preserve solver tolerances, iteration limits, root polishing, certification and exact/MPFR arithmetic.
- Compare numerical results within justified tolerances; floating-point operation order changes.
- Avoid mutable shared scratch and pointers tied to movable intersector storage.
- Keep original patch evaluation as fallback for unsupported/unsafe spans and standalone API callers.
- Measure only after builds stop; retain original executable artifacts before rebuilding.

## Task 1: Immutable polar evaluator and regression tests

Files: new `src/geometry/nurbs_patch_polar_evaluator_3d.hpp/.cpp`, new `tests/geometry/nurbs_patch_polar_evaluator_3d_test.cpp`; root handles CMake registration.

Interface:
```cpp
class NurbsPatchPolarEvaluator3D {
public:
    NurbsPatchPolarEvaluator3D(const NurbsSurfacePatch3D& patch,
                              std::vector<RationalBezierElement3D> base_elements);
    NurbsSurfaceDerivatives3D evaluate_with_derivatives(
        const NurbsSurfacePatch3D& patch, double u, double v,
        NurbsPolarEvaluationWorkspace3D& workspace) const;
};
```

- [x] Write/run tests against the absent interface first, then implement.
- [x] Compare point and first derivatives with the original evaluator for rational degrees 1, 2, 3 and higher than fixed capacity, non-unit parameter spans, repeated interior knots, exact boundaries and neighboring floating-point values.
- [x] Use homogeneous last-layer differences for derivatives and rational quotient rule; no rational positive-weight validation of derivative controls.
- [x] Map original `find_span` indices to base spans, preserve clamping and endpoint tolerance behavior, and fall back for unsafe spans/nonfinite jets.
- [x] Verify common degrees allocate no evaluation heap storage and a warmed higher-degree workspace reuses storage.

## Task 2: Ordinary intersection integration

Files: `nurbs_surface_intersector_3d.hpp/.cpp`, `nurbs_bezier_intersection_3d.hpp/.cpp`, `nurbs_bezier_segment_closest_point_3d.hpp/.cpp`, CMake files.

- [x] Add optional immutable per-patch evaluator pointers to element intersection/closest-point options (null retains reference evaluator).
- [x] Build one evaluator per model patch from the already extracted base elements before acceleration subdivision. Pass the matching evaluator in all ordinary element query/certification paths.
- [x] Give each Newton/closest-point solve its own workspace. Reuse accepted full Newton trial jets without altering acceptance rules.
- [x] Propagate the evaluator into all closest-point calls; retain analytic/interval certification behavior.
- [x] Add integration comparisons for roots, ordering, owners, normals, tangent/miss/unresolved cases and subdivided leaves.

## Task 3: Verification and measured results

- [x] Build Release evaluator/integration tests and existing geometry/trace regression targets; run the relevant CTest set.
- [x] Review production changes for lifetime, thread safety, knot-side semantics and certification regressions.
- [x] Run alternating original/new uninstrumented hollow-cylinder N32/N64 benchmarks; compare topology counters and numerical residuals separately from iteration counts.
- [x] Rebuild temporary profiling driver and measure the later Q27 preprocessing phase with explicitly recorded time budgets if it still does not finish.
- [x] Save commands, hashes, raw measurements and a concise report describing remaining limits.
