# Functional Repository Layout Implementation Plan

**Goal:** Organize existing code by responsibility without changing numerical algorithms, command-line defaults, target names, or output data.

**Architecture:** Keep the existing public core in `src/` and `include/`. Move shared application algorithms into private `src/support/{geometry,density,cauchy,trace,topology,diagnostics}`, executable entrypoints into functional `apps/` subdirectories, regression sources into `tests/`, performance programs into `benchmarks/`, and scripts into `scripts/`. Each category owns its CMake definitions; preserve executable locations under `build/apps`.

**Tech Stack:** C++17, CMake, Eigen, CGAL, zFFT, PowerShell, Python.

## Constraints

- Preserve every existing executable/library target, compile definition, optional experimental gate and numerical source body.
- Rewrite local includes and script repository-root discovery after moves.
- Keep support headers private to this checkout; do not accidentally expand the installed public package.
- Register existing regression executables with CTest and provide independent app/benchmark/test switches.
- Update executable script references and Markdown link targets; retain historical experimental records.
- Work in the user's current clean checkout; leave changes reviewable and uncommitted.

## Tasks

- [x] Capture the original file/target inventory and establish available build validation.
- [x] Generate an explicit old/new path map; move files and rewrite includes by resolving their original targets.
- [x] Split the existing CMake commands by owning target, retain conditions, and register regression tests.
- [x] Repair script, CI, source-inspection and documentation paths; add a repository navigation guide.
- [x] Configure supported build variants, attempt representative target builds, run existing tests and script self-tests, and compare the new build graph with the baseline; record any build limitations below.
- [x] Obtain independent review of the moves and integration; resolve regressions and record verification limits.

## Validation strategy

The production change is file organization. Verify original source bodies are byte-equivalent after undoing path-only substitutions. Resolve every rewritten project include and check every CMake source path. Compare the before/after target set and target-specific compile definitions. Run the available C++ regression executables, CTest discovery, and validation-script self-tests; do not introduce numerical changes to make unrelated baseline failures pass.

## Verification record (2026-09-07)

- 133 file moves recorded in `docs/architecture/file-moves.tsv`.
- Independent review confirmed unchanged numerical C++ bodies and preserved include identities, original target commands, condition stacks, and the two intentional `.cpp` includes.
- Generated-project comparison retained 52/52 baseline projects, 85 mapped compilation entries, the same project-reference graph, and 372 valid source/header/project references.
- Configured the default 3D build (32 registered tests), the 2D core-only build, and the 2D tests-only build (5 registered tests).
- Seven representative CTest cases passed: `laplace_nsc_et_bvp_2d_test`, `native_endpoint_path_3d_test`, `native_nurbs_exact_geometry_3d_test`, `restrict_crossing_feature_side_classifier_3d_test`, `restrict_crossing_path_state_3d_test`, `tensor_product_cover_restrict_3d_test`, and `direct_coefficient_cauchy_3d_test`.
- The full 2D/3D `kfbim_core`, `kfbim_3d_app_geometry`, `kfbim_2d_benchmark_geometry`, and `laplace_nsc_et_benchmark_2d` compiled successfully with local MSVC 19.16 and Eigen 3.4.0.
- Four relocated PowerShell self-tests passed; both visualization scripts compiled as Python; 35 changed-document links were checked without introducing broken links.
- Local MSBuild 15 multiprocess mode returned 1 even for successful no-op builds. Serial `/m:1 /nr:false` returned 0 for the same target; verification therefore uses `cmake --build build-layout --config Release --parallel 1 -- /nr:false`. No repository compiler defaults were changed for this environment issue.
- Building `kfbi_topology_affine_exterior_trace_3d` reached its main source but failed at line 6672 with MSVC C2280, involving the noncopyable `ReducedTraceProjection3D` member initialized through a factory. The original HEAD header, initialization block, and factory block match the relocated versions. Both a minimal example using the original HEAD helper and an Eigen-independent C++17 factory/member-initialization example reproduced C2280 with local MSVC 19.16 (exit code 2). Reproductions and logs are in the ignored `.cache/projection-repro.cpp{,.log}` and `.cache/elision-repro.cpp{,.log}`. This is an existing compiler compatibility limitation, not a changed numerical implementation; the main executable build and end-to-end 3D validation remain unverified here.
- This is relocation/integration validation, not a complete 3D PDE convergence study or a run of all experimental tests.
