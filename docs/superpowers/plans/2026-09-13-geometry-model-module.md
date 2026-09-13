# Geometry Model Module Implementation Plan

> **For agentic workers:** Use subagent-driven-development for independent extraction tasks and review the integrated changes.

**Goal:** Move three-dimensional model construction into one maintainable module, with one implementation per model and compatible existing callers.

**Architecture:** `src/geometry/models3d` owns model data, model factories and geometric construction helpers. Case adapters own physical data and solver-specific sampling. Grid-dependent preprocessing stays outside the model factories.

**Tech Stack:** C++17, Eigen, existing CMake/CTest and CGAL-enabled build.

**Spec:** `docs/superpowers/specs/2026-09-13-geometry-model-module-design.md`

## Global Constraints

- Preserve all existing geometry dimensions, patch ordering, knots, weights, topology and parameter conventions.
- Preserve current uncommitted Shared Field work; work in the current feature checkout without changing Git history.
- Keep pure geometry independent of support/density/solver code and preserve standalone industrial build.
- No new PDE formulation or change to crossing certification.

## Tasks

- [x] 1. Add a geometry catalog behavior test before the API exists; verify missing API is the failure. Record existing geometry-test baseline.
- [x] 2. Extract Native model data and per-model construction; leave DOF/cloud/Cauchy-neighbor algorithms in support. Keep old headers as compatible includes.
- [x] 3. Extract Trace93 geometry-only definitions and per-model recipes; leave manufactured data, transform presets, mean quadrature and analysis evaluation in the case adapter where needed.
- [x] 4. Move industrial construction files into the module and update standalone example source paths. Extract shared analytic cap geometry from app/density duplication.
- [x] 5. Wire the module source list, public geometry include and catalog; run the new test and existing geometric/numerical regressions. Build affected callers.
- [x] 6. Document model inventory and the add-model workflow; review dependencies and migration equivalence, fix findings, record validation.

## Progress

Baseline before changes: Trace93 case and Native U geometry pass. Existing rigid-transform test fails at rotated torus N64 domain edge correction-safe certification. Several other registered targets have not yet been built in build-3d.

Both new test targets first failed because their new geometry headers did not exist. The new model catalog executable now passes. The standalone industrial gallery and test build, and its CTest passes (1/1).

Independent extraction review found unchanged Native dimensions, patch names, 57 ordered edge connections, inside predicates and rigid-transform function bodies. Trace93 support now consumes the public geometry/chart contract; it does not include private construction helpers. Analytic cap body/world mappings and atlas conventions were reviewed against the original density/app code.

Integration complete: all nine selected test targets and four affected applications build in Release. Seven tests pass; Native N128 and rigid-transform N64 certification tests fail. The N64 failure was observed before extraction; an isolated executable using the original Native factory reproduces the exact N128 failure, and its linker map confirms no new Native model objects were loaded. Standalone industrial CTest also passes (1/1). No certification algorithm was changed. Details and reproduction commands are recorded in `docs/Geometry_Model_Module_Validation.md`. Final whitespace/dependency checks and independent source review found no remaining extraction issues.

User naming feedback applied: the geometry module now uses `bicubic_nurbs`, `BicubicNurbsModel3D`, `SurfaceAnalysisChart3D`, and explicit shape enums/IDs. Model recipes and topology are unchanged. Existing case selectors are translated by their adapter. The affected study application builds, and catalog, case geometry, density layout and fixed-reference regressions all pass (4/4). The module README records the rule to name by geometric shape, mathematical representation and responsibility instead of source numbering.
