# Normal Restrict Degree Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two mixed-order normal restrict modes and compare all four interpolation/normal-fit combinations on the L shape through `N=1024`.

**Architecture:** Keep explicit named `RestrictMode` values for compatibility, but derive grid interpolation order, normal fit order, sample count, and jump-correction order independently. Reuse the existing trace-template and joint-fit implementations.

**Tech Stack:** C++17, Eigen, zFFT, existing KFBI application and GMRES implementation, CMake/MSBuild.

## Global Constraints

- Preserve every existing restrict mode and selector.
- `degree_compare` contains exactly the four 2x2 degree combinations.
- Use fixed `KFBIM_PYJET_TOL=2e-10` for the numerical comparison.
- Do not change spread, DOF placement, geometry, or manufactured data.

---

### Task 1: Parser red-green test

**Files:**
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`

**Interfaces:**
- Produces: `bicubic_quadratic`, `biquadratic_cubic`, and `degree_compare` selections.

- [ ] Run the current executable with `KFBIM_PYJET_RESTRICT_MODE=bicubic_quadratic` and confirm the parser rejects it.
- [ ] Add both enum values, names, parser entries, and the four-mode `degree_compare` list.
- [ ] Build and rerun the parser acceptance command.

### Task 2: Degree-independent restrict construction

**Files:**
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`

**Interfaces:**
- Consumes: the two new `RestrictMode` values.
- Produces: correct grid side count, layer count, normal degree, Cauchy degree, and condition source for every mode.

- [ ] Make cubic-normal modes use four layers and degree 3.
- [ ] Make cubic-grid modes use 4x4 interpolation, full Cauchy correction, and full restrict condition data.
- [ ] Make quadratic-grid modes use 3x3 interpolation and quadratic correction.
- [ ] Keep the existing two-layer quadratic special case unchanged.
- [ ] Run all four modes on L-shape `N=64` and require successful GMRES convergence.

### Task 3: Documentation and numerical study

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/build-and-smoke.yml`
- Generated: `output/dirichlet_harmonic_jet_case_cpp_*degree_compare*.csv`

**Interfaces:**
- Consumes: `KFBIM_PYJET_RESTRICT_MODE=degree_compare`.
- Produces: four-mode L-shape comparison at `N=64,128,256,512,1024`.

- [ ] Document the two mixed modes and comparison selector.
- [ ] Extend smoke coverage to assert both new mode names and convergence.
- [ ] Run the complete Release build.
- [ ] Run the L-shape degree comparison through `N=1024` and verify 20 finite, converged rows.
- [ ] Compare errors, observed orders, iterations, conditioning, and runtime.
