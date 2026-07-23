# 3D GMRES Run Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound 3D GMRES work at 80 iterations by default and stop a multi-level run immediately after the first failed level.

**Architecture:** Reuse the app's positive-integer environment parser, thread one maximum through both solve paths, and check convergence after each completed readiness case. Preserve partial CSV summaries before throwing a detailed failure.

**Tech Stack:** C++17, Eigen GMRES, CMake/CTest, GitHub Actions shell smoke tests.

## Global Constraints

- Do not change the Cauchy degree/counts, restrict degrees, GMRES tolerance, or geometry algorithms.
- Default `KFBIM_3D_GMRES_MAX_ITERATIONS` is exactly 80 and applies to both formulations.
- Formal numerical runs use only `N=32,64,128`.

---

### Task 1: Configurable GMRES cap and fail-fast level loop

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `.github/workflows/build-and-smoke.yml`

**Interfaces:**
- Consumes: existing `positive_environment_integer(const char*, int)`.
- Produces: `KFBIM_3D_GMRES_MAX_ITERATIONS`, header field `gmres_max_iterations`, and detailed per-level nonconvergence error.

- [ ] **Step 1: Add failing smoke assertions**

Assert the default header contains `gmres_max_iterations=80`. Run with a
one-iteration cap and levels `16 32`; require nonzero exit, a failure naming
`N=16`, and exactly one `[ready]` record. Add invalid-value checks for `0`,
`-1`, and `abc`.

- [ ] **Step 2: Verify RED**

Run the Release app smoke commands before production changes. The default
header assertion must fail because the field does not exist.

- [ ] **Step 3: Implement the cap**

Parse the environment variable with default 80 in `main`, print it, pass it
through `run_readiness_case`, `run_neumann_case`, and
`run_dirichlet_normal_case`, and replace hard-coded maxima 200/400.

- [ ] **Step 4: Implement per-case fail-fast**

After each result is appended, write both summary CSVs. If either solve is
not converged, throw a message containing geometry, `N`, formulation,
iterations, and final residual before starting another level.

- [ ] **Step 5: Verify GREEN**

Build `neumann_exterior_zero_trace_3d`; run the default, capped, invalid,
and explicit-policy smoke cases. Run `native_nurbs_surface_3d_test`.

### Task 2: Three-policy L-prism comparison

**Files:**
- Generate: policy-specific CSVs under `output/neumann_exterior_zero_trace_3d/`.

**Interfaces:**
- Consumes: Release app with the default 80-iteration cap.
- Produces: Neumann internal maximum error, convergence order, and GMRES iterations for `N=32,64,128`.

- [ ] **Step 1: Run `same_patch`**

Run the L-prism app for `32 64 128` with the explicit policy.

- [ ] **Step 2: Run `g1_nearest`**

Run the same levels using the default policy.

- [ ] **Step 3: Run `topological_nearest`**

Run the same levels with explicit cross-non-G1 sampling.

- [ ] **Step 4: Compare**

Compute adjacent orders as `log(error_coarse/error_fine)/log(2)` and report
only Neumann internal maximum error, order, and GMRES iterations, plus the
diagnosis of any irregular order.
