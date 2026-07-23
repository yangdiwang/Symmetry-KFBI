# 3D G1-Nearest Cauchy Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make G1-only nearest Cauchy sampling the 3D default and compare it with Neumann sampling that may cross non-G1 edges.

**Architecture:** Add one selector driven exclusively by `smooth_neighbors`, expose it through a new `g1_nearest` application policy, and keep all existing policies for controlled comparisons.

**Tech Stack:** C++17, Eigen, CMake/MSVC x64, PowerShell, native NURBS topology.

## Global Constraints

- Cross G1 seams, including explicit periodic seams.
- Never cross non-G1 edges in `g1_nearest`.
- Keep Cauchy/restrict degrees `3/3/3`, counts `48/28`, and solver tolerances.
- Test before implementation and run the L-prism at `N=32,64,128,256`.

---

### Task 1: G1-only selector

**Files:**
- Modify: `apps/native_nurbs_surface_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

- [ ] Write tests proving G1 seam crossing and non-G1 rejection.
- [ ] Build the native test and confirm the new API is missing.
- [ ] Implement local G1 breadth-first candidates and Euclidean selection.
- [ ] Run the native test to green.

### Task 2: Application policy and smoke

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `.github/workflows/build-and-smoke.yml`

- [ ] Change the smoke expectation to `g1_nearest` and confirm it fails.
- [ ] Add the policy, make it default, and retain explicit old policies.
- [ ] Rebuild and run default plus explicit-policy smoke tests.

### Task 3: Numerical comparison and diagnosis

**Files:**
- Generate: `output/neumann_exterior_zero_trace_3d/g1_nearest/analysis/`
- Generate: `output/neumann_exterior_zero_trace_3d/analysis/topological_v48_n28_levels/`

- [ ] Run and archive `g1_nearest` at `N=32,64,128,256`.
- [ ] Run and archive `topological_nearest` at the same levels.
- [ ] Recompute adjacent orders from archived rows.
- [ ] Separate GMRES convergence from discretization/grid-phase error and report the comparison.
