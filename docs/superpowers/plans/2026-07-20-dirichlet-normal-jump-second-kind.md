# Dirichlet Normal-Jump Second-Kind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and numerically validate a normal-jump second-kind Dirichlet formulation on the L-shaped example.

**Architecture:** Extend the existing joint normal restrict to recover its first coefficient divided by `h`. Use that direct exterior normal trace in a new matrix-free GMRES operator while retaining the existing value-trace formulations.

**Tech Stack:** C++17, Eigen, zFFT, existing KFBI spread/restrict and GMRES implementation, CMake/MSBuild.

## Global Constraints

- Preserve `normal_jump_first_kind` and `value_jump_second_kind` behavior.
- Use `[u]=g_D`, unknown `q=[u_n]`, and residual `u_n^+=0`.
- Use existing edge-local L-shape normals and corner corrections.
- Reject `six_point_quadratic_exterior` for the new derivative formulation.

---

### Task 1: Failing acceptance test

**Files:**
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`

**Interfaces:**
- Consumes: `KFBIM_PYJET_DIRICHLET_FORMULATION`.
- Produces: accepted value `normal_jump_second_kind`.

- [ ] Run the current L-shape executable with `KFBIM_PYJET_DIRICHLET_FORMULATION=normal_jump_second_kind`.
- [ ] Confirm failure says the formulation name is unsupported.

### Task 2: Exterior normal restrict

**Files:**
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`

**Interfaces:**
- Produces: `PythonCompatibleHarmonicJet2D::exterior_normal_trace(...) -> Eigen::VectorXd`.

- [ ] Add `c1_weights_ = pinv.row(1).transpose()` beside `c0_weights_`.
- [ ] Add `exterior_normal_trace` using the same exterior-branch samples as `exterior_trace` and return `c1_weights_.dot(samples)/h_`.
- [ ] Add an interior-normal audit from the same joint fit and verify `u_n^+ = u_n^- - q` numerically.
- [ ] Build the Dirichlet executable and correct all compiler failures.

### Task 3: Matrix-free second-kind solve

**Files:**
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`
- Modify: `README.md`

**Interfaces:**
- Produces: `DirichletFormulation::NormalJumpSecondKind` and `ExteriorDirichletNormalJumpSecondKindOperator`.

- [ ] Extend parsing, names, compare mode, and summaries with `normal_jump_second_kind`.
- [ ] Implement `A_h q = R^+_{n,h}P_h(0,q)`.
- [ ] Build `b_h = -R^+_{n,h}P_h(g_D,0)` once, solve with GMRES, and recover `P_h(g_D,q)`.
- [ ] Record exterior-normal residual, route mismatch, interior boundary residual, and manufactured density error.
- [ ] Document the new environment option and equation.

### Task 4: L-shape numerical verification

**Files:**
- Generated: `output/dirichlet_harmonic_jet_python_compatible_2d.csv`

**Interfaces:**
- Consumes: release executable and three formulation names.
- Produces: L-shape convergence and iteration data.

- [ ] Build all targets in Release mode.
- [ ] Run the new formulation on L-shape refinement levels.
- [ ] Run compare mode on representative L-shape levels.
- [ ] Check convergence flags, finite residuals, route mismatch, density error, and bulk errors.
- [ ] Inspect `git diff --check` and repository status.
