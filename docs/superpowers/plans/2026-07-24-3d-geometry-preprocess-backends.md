# 3D Geometry-Preprocess Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optimized-certified and hybrid native-NURBS geometry-preprocess backends, then compare their KFBI-relevant accuracy and speed against the current implementation.

**Architecture:** Keep the current path as `CertifiedBaseline`. `OptimizedIntersection` reuses broad-phase edge-to-element incidences and performs the existing whole-element unique-root certificate before redundant sample seeds. `Hybrid` adds affine-planar analytic and smooth closest-point certified routes, with automatic fallback to the optimized certified path. A dedicated benchmark builds native domains and `GridPair3D` objects and compares every label, structured edge, crossing, and correction owner.

**Tech Stack:** C++17, Eigen 3.4, CGAL 5.6, CMake 3.20+, MSVC Release build.

## Global Constraints

- Work directly on `main`, as explicitly requested by the user and recorded in `.superpowers/sdd/progress.md`.
- Preserve the current default behavior through `CertifiedBaseline`.
- Do not modify the PDE solver, Cauchy stencils, restrict/spread formulas, or GMRES.
- Closest-point convergence or endpoint signs alone may not authorize topology or a correction crossing.
- Every fast-path uncertainty must fall back to the certified path.
- Accuracy checks use the full KFBI geometry-preprocess outputs, not a sparse sample.
- Preserve the two pre-existing untracked same-patch design/plan files.

---

### Task 1: Backend API, stable candidate workload, and diagnostics

**Files:**
- Modify: `src/geometry/nurbs_surface_intersector_3d.hpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.cpp`
- Modify: `src/geometry/nurbs_cartesian_domain_3d.hpp`
- Modify: `src/geometry/nurbs_cartesian_domain_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Produces:

```cpp
enum class NurbsCartesianPreprocessStrategy3D {
    CertifiedBaseline,
    OptimizedIntersection,
    Hybrid
};

struct NurbsQueryElementDescriptor3D {
    std::size_t id;
    NurbsAabb3D bounds;
    int patch_index;
    int component;
};

struct NurbsCartesianEdgeQueryOptions3D {
    int local_max_subdivision_depth = -1;
};

const std::vector<NurbsQueryElementDescriptor3D>&
NurbsSurfaceIntersector3D::query_elements() const noexcept;

NurbsCartesianEdgeIntersections3D
NurbsSurfaceIntersector3D::intersect_cartesian_edge(
    const NurbsCartesianEdgeQuery3D& edge,
    const std::vector<std::size_t>& candidate_element_ids,
    NurbsCartesianEdgeQueryOptions3D options = {}) const;
```

- `NurbsCartesianDomainOptions3D::strategy` defaults to
  `CertifiedBaseline`.
- Optimized and hybrid modes retain sorted unique
  `(edge_key, element_id)` incidences and call the mapped-candidate overload.

- [ ] **Step 1: Write failing API and candidate-equivalence tests**

Add tests that instantiate all three strategy enum values, assert the default
is `CertifiedBaseline`, compare mapped-candidate and BVH results on an edge
covering multiple query leaves, and reject duplicate/out-of-range candidate
IDs with a descriptive exception.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected: compile failure because the new enum, descriptors, overload, and
diagnostics do not exist.

- [ ] **Step 3: Implement the minimal public API and mapped workload**

Populate immutable descriptors from `elements_`, validate and sort mapped
candidate IDs, route them into `intersect_segment_impl`, and preserve the
single-argument BVH path. In the domain broad phase, sort/unique pairs by
`(edge_key, element_id)` and group them per edge for optimized/hybrid modes.
Use a per-query depth override for targeted retry.

- [ ] **Step 4: Add timing and route-neutral counters**

Add mapped/BVH candidate counts, maximum candidates per edge, incidence
count, and phase timings. Use `std::chrono::steady_clock`; keep all timing
outside numerical comparisons.

- [ ] **Step 5: Run tests and verify GREEN**

Run the focused executable and require `native NURBS model tests passed`.

- [ ] **Step 6: Commit**

```powershell
git add src/geometry/nurbs_surface_intersector_3d.hpp src/geometry/nurbs_surface_intersector_3d.cpp src/geometry/nurbs_cartesian_domain_3d.hpp src/geometry/nurbs_cartesian_domain_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: add selectable 3d preprocess workload"
```

### Task 2: Optimized certified-intersection backend

**Files:**
- Modify: `src/geometry/nurbs_bezier_intersection_3d.hpp`
- Modify: `src/geometry/nurbs_bezier_intersection_3d.cpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.cpp`
- Modify: `src/geometry/nurbs_cartesian_domain_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: mapped candidates and backend strategy from Task 1.
- Produces:

```cpp
bool NurbsElementIntersectionOptions3D::use_early_unique_root_certificate;

int NurbsElementIntersectionDiagnostics3D::
    early_unique_certificate_attempts;
int NurbsElementIntersectionDiagnostics3D::
    early_unique_certificate_successes;
```

- [ ] **Step 1: Write the failing early-certificate test**

For a single transverse affine element, require identical root data with the
option off/on, one successful early certificate, zero supplied-seed attempts
with the option on, and fewer Newton attempts. Add explicit tests that a
two-root ruled element, tangent root, and zero-derivative odd root are not
accepted by the early certificate.

- [ ] **Step 2: Verify RED**

Build and run `native_nurbs_surface_3d_test`; expect compile failure for the
new option/counters.

- [ ] **Step 3: Implement the early certificate**

After triangle seeds are canonicalized within the element, if exactly one
verified root exists, increment the attempt counter and invoke the existing
outward-rounded `certifies_unique_transverse_root` on the complete element.
On success increment the success counter and return before supplied seeds
and subdivision. Otherwise execute the existing code unchanged.

- [ ] **Step 4: Wire `OptimizedIntersection`**

Enable mapped candidates and the early certificate only for
`OptimizedIntersection` and `Hybrid`. Accumulate new counters through the
surface and domain diagnostics.

- [ ] **Step 5: Verify production-domain equivalence**

For torus, hollow cylinder, and L-prism at the focused test grid, compare
baseline and optimized labels, barrier/interface flags, edge
classifications, crossing geometry, and strict correction lookups.

- [ ] **Step 6: Run tests and commit**

```powershell
git add src/geometry/nurbs_bezier_intersection_3d.hpp src/geometry/nurbs_bezier_intersection_3d.cpp src/geometry/nurbs_surface_intersector_3d.cpp src/geometry/nurbs_cartesian_domain_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "perf: skip redundant certified intersection work"
```

### Task 3: Hybrid affine-planar and closest-point routes

**Files:**
- Modify: `src/geometry/nurbs_bezier_intersection_3d.hpp`
- Modify: `src/geometry/nurbs_bezier_intersection_3d.cpp`
- Modify: `src/geometry/nurbs_surface_intersector_3d.cpp`
- Modify: `src/geometry/nurbs_cartesian_domain_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: `Hybrid` strategy and early certificate.
- Produces element options:

```cpp
bool use_affine_planar_fast_path = false;
bool use_closest_point_prefilter = false;
```

- Produces counters for planar analytic hits/misses/fallbacks,
  closest-point prefilter attempts/certified hits/certified misses/fallbacks,
  and certified fallback elements.

- [ ] **Step 1: Write failing affine-planar route tests**

Cover a strict interior hit, clear miss, reversed segment, near-parallel
fallback, coplanar overlap fallback, Cartesian endpoint rejection, patch
boundary fallback, and two planar elements producing two roots.

- [ ] **Step 2: Verify affine tests RED**

Build and run the focused executable; expect missing option/counter failures.

- [ ] **Step 3: Implement affine-planar classification**

Certify degree, equal positive weights, affine residual, nondegenerate
tangents, nonparallel line, parameter bounds, patch-interior margin,
residual, normal orientation, and transversality. Return authoritative hit
or miss only when every check is decisive; otherwise fall through.

- [ ] **Step 4: Write failing closest-point route tests**

Use a quarter cylinder and torus element for a certified smooth hit and
certified miss. Require tangency, close roots, G1/non-G1 seams, hollow
cylinder rim, and the L-prism reentrant edge to fall back.

- [ ] **Step 5: Verify closest-point tests RED**

Run the focused executable and confirm failure because the prefilter route is
not implemented.

- [ ] **Step 6: Implement closest-point prefilter**

Before ordinary seeds, run bounded element-to-segment closest point with the
best existing sample seed. Accept a root only with the complete-element
unique transverse certificate. Accept a miss only with the existing
conservative terminal-separation certificate. Disable the route on elements
touching declared non-G1 features. Preserve accepted seed/root information
when falling back without changing canonicalization.

- [ ] **Step 7: Wire and verify `Hybrid`**

Enable mapped candidates, early certificate, affine planar, and safe
closest-point prefilter. Compare all three production geometries against
baseline and require nonzero planar routing for L-prism and nonzero
closest-point routing or explicit fallback diagnostics for curved targets.

- [ ] **Step 8: Run tests and commit**

```powershell
git add src/geometry/nurbs_bezier_intersection_3d.hpp src/geometry/nurbs_bezier_intersection_3d.cpp src/geometry/nurbs_surface_intersector_3d.cpp src/geometry/nurbs_cartesian_domain_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: add hybrid 3d geometry preprocessing"
```

### Task 4: KFBI geometry-preprocess benchmark and deterministic smoke test

**Files:**
- Create: `apps/nurbs_geometry_preprocess_benchmark_3d.hpp`
- Create: `apps/nurbs_geometry_preprocess_benchmark_3d.cpp`
- Create: `apps/nurbs_geometry_preprocess_benchmark_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Produces executable:

```text
nurbs_geometry_preprocess_benchmark_3d
  --backend baseline|pure|hybrid|all
  --geometry torus|cylinder|l_prism|all
  --N 32 64 128
  --warmup 1
  --reps 3
  --out <directory>
```

- Writes:
  - `nurbs_geometry_preprocess_raw.csv`
  - `nurbs_geometry_preprocess_summary.csv`
  - mismatch diagnostics on failure.

- [ ] **Step 1: Write the failing benchmark-support test**

The test runs all three backends on each production geometry at a small valid
grid, invokes full node/edge/crossing comparison, checks analytic labels,
constructs `GridPair3D`, materializes all label-changing owners, and verifies
deterministic nonzero checksums.

- [ ] **Step 2: Verify RED**

Build the new test target; expect failure because benchmark support is
absent.

- [ ] **Step 3: Implement shared workload and comparison**

Build the same native surface, triangulation options, node-layout Cartesian
grid, domain, and `GridPair3D` used by the KFBI app. Keep correctness scans
outside timed sections. Return counts/rates rather than only a pass boolean.

- [ ] **Step 4: Implement CLI, timing repetitions, and CSV**

Rotate backend order across repetitions. Record raw timing, internal phase
timing, diagnostics, accuracy counts, and a checksum. Aggregate median,
minimum, maximum, coefficient of variation, and speedup relative to
baseline. Reject malformed geometry/backend/N/repetition arguments.

- [ ] **Step 5: Build and run the smoke test**

```powershell
cmake --build build --config Release --target nurbs_geometry_preprocess_benchmark_3d_test
.\build\apps\Release\nurbs_geometry_preprocess_benchmark_3d_test.exe
```

Expected: all backend equivalence and owner checks pass.

- [ ] **Step 6: Commit**

```powershell
git add apps/nurbs_geometry_preprocess_benchmark_3d.hpp apps/nurbs_geometry_preprocess_benchmark_3d.cpp apps/nurbs_geometry_preprocess_benchmark_3d_test.cpp apps/CMakeLists.txt
git commit -m "test: add 3d geometry preprocess benchmark"
```

### Task 5: Numerical experiment and results

**Files:**
- Create: `docs/superpowers/results/2026-07-24-3d-geometry-preprocess-backends.md`
- Generated, not necessarily tracked: `output/nurbs_geometry_preprocess_benchmark/*.csv`

**Interfaces:**
- Consumes: benchmark executable from Task 4.
- Produces: reproducible raw/summary CSV and a concise results document.

- [ ] **Step 1: Run a pilot**

Run all geometries/backends at `N=32` with one warmup and one repetition.
Confirm zero mismatches and estimate the full matrix runtime.

- [ ] **Step 2: Run the primary matrix**

Run Release:

```powershell
.\build\apps\Release\nurbs_geometry_preprocess_benchmark_3d.exe --backend all --geometry all --N 32 64 128 --warmup 1 --reps 3 --out output\nurbs_geometry_preprocess_benchmark
```

If the measured pilot projects more than 25 minutes, retain all correctness
cells but reduce timed repetitions to `3,3,1` for `N=32,64,128`, recording
the actual repetitions in CSV.

- [ ] **Step 3: Audit result completeness**

Require exactly three backends for every geometry/N cell, nonzero route
checksums, finite timings, zero baseline and analytic mismatches, zero unsafe
label-changing edges, and residuals within tolerance.

- [ ] **Step 4: Write the results document**

Report per geometry/N:

- median domain, `GridPair3D`, and combined time;
- pure and hybrid speedup versus baseline;
- backend route counts and fallback rates;
- label, edge-topology, crossing, and owner accuracy;
- noisy cells with coefficient of variation above 10%;
- interpretation of where each backend helps or loses.

- [ ] **Step 5: Commit**

```powershell
git add docs/superpowers/results/2026-07-24-3d-geometry-preprocess-backends.md
git commit -m "docs: compare 3d geometry preprocess backends"
```

### Task 6: Full verification and review

**Files:**
- Modify only if review finds a defect.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified implementation and final review record.

- [ ] **Step 1: Rebuild all 3D targets**

```powershell
cmake --build build --config Release
```

- [ ] **Step 2: Run regression executables**

At minimum run:

```powershell
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\nurbs_geometry_preprocess_benchmark_3d_test.exe
```

- [ ] **Step 3: Run a KFBI application smoke**

Run the existing L-prism `N=32` production geometry/PDE app with the default
backend and confirm the output remains unchanged within existing tolerances.

- [ ] **Step 4: Review the complete diff**

Check API compatibility, certificate/fallback correctness, timing
contamination, integer overflow, deterministic sorting, CSV column/value
alignment, and preservation of unrelated user files.

- [ ] **Step 5: Fix review findings and re-run covering tests**

No Critical or Important review issue may remain open.
