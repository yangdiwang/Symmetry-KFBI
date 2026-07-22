# Grid-Scaled NURBS Intersection Implementation Plan

> **Execution:** Use test-driven development and complete each task in order.

**Goal:** Use `2h` rational Bezier query leaves, local depth `4`, and a bounded
segment-to-element closest-point terminal classifier in the native 3D NURBS
Cartesian-grid route.

**Architecture:** Add a focused closest-point module, enrich the local native
intersection outcome/diagnostics, make the intersector BVH optionally use
extent-limited leaves, and configure that option from the Cartesian domain.
Generic callers retain current defaults.

### Task 1: Closest-point solver

**Files:**
- Create: `src/geometry/nurbs_bezier_segment_closest_point_3d.hpp`
- Create: `src/geometry/nurbs_bezier_segment_closest_point_3d.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

- [x] Add planar positive-distance and curved tangent tests.
- [x] Run the tests and confirm RED because the API is absent.
- [x] Implement bounded projected Gauss-Newton with at most two seeds.
- [x] Run the focused regression and confirm GREEN.

### Task 2: Hybrid terminal classification

**Files:**
- Modify: `src/geometry/nurbs_bezier_intersection_3d.hpp`
- Modify: `src/geometry/nurbs_bezier_intersection_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

- [x] Add a forced-shallow terminal-miss test and diagnostics assertions.
- [x] Run RED.
- [x] Preserve best failed Newton state and failure classification.
- [x] Call closest assistance only for ill-conditioned Newton or terminal boxes.
- [x] Apply the `8 * geometry_tolerance` contact threshold, then require an
      adaptive convex-hull separation/single-root certificate before resolving
      a terminal box.
- [x] Add the two-root stationary-point, seed-arbitration, bounded high-degree
      work, extent-limited tangency, and certificate-depth regressions.
- [x] Run GREEN, including existing degeneracy/unresolved regressions.

### Task 3: Use `2h` leaves in the actual grid query

**Files:**
- Modify: `src/geometry/nurbs_surface_intersector_3d.hpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.cpp`
- Modify: `src/geometry/nurbs_cartesian_domain_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

- [x] Add tests that extent-limited query leaves are at most `2h`, local depth
      never exceeds `4`, labels are unchanged, and unresolved count is zero.
- [x] Run RED.
- [x] Add configurable maximum query-element extent and local depth, retaining
      infinity/depth-36 defaults.
- [x] Build the BVH over extent-limited elements and pass the configured local
      depth to every local solve.
- [x] Configure the Cartesian domain with `2 * max(h)` and depth `4`.
- [x] Run GREEN with triangle seeds enabled and disabled.

### Task 4: Verification and handoff

- [x] Build `native_nurbs_surface_3d_test` in Release and run it.
- [x] Build all Release targets.
- [x] Run CTest and `git diff --check`.
- [x] Review the final diff for accidental behavior changes.
- [x] Commit the verified implementation on `main` and report exact files,
      algorithm changes, and numerical/regression results for user review.
