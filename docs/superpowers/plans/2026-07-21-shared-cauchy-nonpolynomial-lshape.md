# Shared Cauchy Non-polynomial L-shape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the cubic-spread local Cauchy polynomial in normal restriction and recompute the L-shape study with a non-polynomial harmonic solution.

**Architecture:** Add one per-DOF coefficient builder that selects the same-edge cubic map or cross-corner cubic map. Route both cubic RHS spreading and restrict stencil corrections through it, while leaving grid interpolation and one-dimensional normal-fit degrees independent.

**Tech Stack:** C++17, Eigen, CMake/MSBuild, PowerShell integration checks.

## Global Constraints

- Preserve all existing restrict and formulation selector memberships.
- Change only the `cubic_harmonic` shared-Cauchy route; preserve other spread modes.
- Use `u=exp(0.42*x)*cos(0.42*y)` and its analytic gradient for the L-shape.
- Verify `degree_compare` at `N=64,128,256,512,1024` with tolerance `2e-10`.

---

### Task 1: Observable shared-Cauchy contract

**Files:**
- Modify: `.github/workflows/build-and-smoke.yml`
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`

**Interfaces:**
- Produces: per-run field `restrict_cauchy_source=shared_cubic_spread`.

- [ ] Add a smoke assertion requiring four L-shape runs to report the shared source.
- [ ] Run the existing executable and verify the assertion fails because the field is absent.
- [ ] Add the result/config field without changing numerical behavior.

### Task 2: Canonical per-DOF Cauchy coefficients

**Files:**
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`

**Interfaces:**
- Produces: `std::vector<Eigen::VectorXd> cubic_spread_correction_coefficients(...) const`.
- Consumes: existing same-edge cubic maps and cross-corner maps.

- [ ] Make the coefficient builder select cubic or corner coefficients per DOF.
- [ ] Route `rhs_from_jumps()` through the builder.
- [ ] Route `one_sided_samples()` through the same builder for `cubic_harmonic`.
- [ ] Size each trace-template correction from that DOF's selected polynomial degree.
- [ ] Report the selected cubic/corner map condition for restrict diagnostics.
- [ ] Build and run the L-shape four-mode smoke; require four converged shared-source rows.

### Task 3: Non-polynomial harmonic manufactured solution

**Files:**
- Modify: `apps/neumann_harmonic_jet_python_compatible_2d.cpp`
- Modify: `README.md`

**Interfaces:**
- Produces: exact L-shape value and gradient from `exp(0.42*x)*cos(0.42*y)`.

- [ ] Add a regression probe that rejects polynomial-exact L-shape behavior.
- [ ] Remove the L-shape cubic-polynomial branch from exact value and gradient.
- [ ] Build and verify the probe and smoke pass.

### Task 4: Full numerical study and result record

**Files:**
- Create: `docs/superpowers/results/2026-07-21-lshape-shared-cauchy-nonpolynomial.md`

**Interfaces:**
- Consumes: 20-row `degree_compare` CSV.

- [ ] Run Release L-shape levels `64 128 256 512 1024`.
- [ ] Assert 20 finite, converged rows and four modes with five levels each.
- [ ] Record errors, observed orders, residuals, density error, iterations, conditions, and times.
- [ ] Commit implementation and results.
- [ ] Request code review and rerun final verification after fixes.
