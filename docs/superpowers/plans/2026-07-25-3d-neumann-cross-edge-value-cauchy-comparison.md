# 3D Neumann Cross-Edge Value-Cauchy Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and measure an edge-aware Cauchy sample pool that can cross only topology-adjacent non-G1 faces, then isolate whether applying it to the Neumann value density stabilizes the L-prism bordered second-kind solve.

**Architecture:** Add one reusable native-NURBS edge-neighborhood selector, keep value and normal Cauchy pools independently configurable, and leave the cubic harmonic fit and crossing-owner restrict unchanged. A dedicated study command runs the three approved pool combinations on three L-prism poses at `N=32,64,128`, records physical and common-RHS GMRES behavior plus edge-localized consistency defects, and never changes the production default from smoke evidence.

**Tech Stack:** C++17, Eigen, existing KFBI3D/NURBS geometry and GMRES infrastructure, CMake/MSVC x64, CSV diagnostics, PowerShell audits.

## Global Constraints

- Work directly on `main`; do not create a worktree or feature branch.
- Preserve unrelated tracked edits and the two pre-existing untracked documents.
- Keep `JointTricubicCrossingOwner` for every compared route.
- Keep harmonic degree `3`, value/normal counts `48/28`, tricubic Cartesian interpolation, cubic normal-line fitting, compatibility correction, bordered Neumann equation, GMRES tolerance `2e-10`, restart `80`, and cap `80`.
- Change only the value/normal Cauchy sample pools. Do not change weights, polynomial basis, pseudoinverse cutoff, owner construction, manufactured harmonic function, or box solver.
- Outside the closed `2h` non-G1 edge band, return the existing `nearest_g1_cauchy_dofs` ID vector in exactly the same order.
- Inside the edge band, admit only the native edge's incident patches and their transitive G1-equivalent pieces. Never use Euclidean proximity alone to cross to an unrelated sheet.
- Use `NativeNurbsSurface3D::geometric_connections`, including its patch-edge intervals, as the authoritative topology. Do not derive sample sectors from triangulator feature-edge chords.
- Geometry and owner queries are preprocessing work only; GMRES may not mutate owner query counts or fingerprints.
- A short or rank-deficient stencil is a recorded route failure with full diagnostics, not a silent fallback to a broader patch pool.
- Generated numerical files stay untracked under `output/neumann_cross_edge_value_cauchy_3d`.
- The production `KFBIM_3D_CAUCHY_POLICY` default remains `g1_nearest` until all 27 configurations pass the decision audit.

## Fixed Comparison Matrix

| Route | Value pool | Normal pool |
|---|---|---|
| `g1_value_g1_normal` | `G1Nearest` | `G1Nearest` |
| `edge_value_g1_normal` | `EdgeAwareTopologyAdjacent` | `G1Nearest` |
| `edge_value_edge_normal` | `EdgeAwareTopologyAdjacent` | `EdgeAwareTopologyAdjacent` |

The three poses are `baseline`, `rot_axis123_17deg`, and
`rot_axis123_17deg_t_xyz_1`. The formal levels are `32,64,128`. Each of the
27 `(pose,N,route)` configurations runs one physical manufactured RHS and one
common parameter-indexed RHS.

---

### Task 1: Native NURBS non-G1 edge neighborhood and balanced selector

**Files:**
- Modify: `apps/native_nurbs_surface_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Public interface:**

```cpp
struct EdgeAwareCauchyDofSelection3D {
    std::vector<int> dof_ids;
    bool edge_band_active = false;
    double nearest_feature_edge_distance =
        std::numeric_limits<double>::infinity();
    std::vector<std::vector<int>> sector_patch_ids;
    std::vector<int> sector_sample_counts;
};

[[nodiscard]] EdgeAwareCauchyDofSelection3D
edge_aware_topological_cauchy_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count,
    double h);
```

- [ ] **Step 1: Write RED selector tests**

Add `test_edge_aware_cauchy_selection()` next to the existing Cauchy/topology
tests around `apps/native_nurbs_surface_3d_test.cpp:4061`, and call it from
`main()` near the existing `test_parameter_candidates()` registration.

The test must contain these exact cases:

1. L-prism side-edge center: use `h=3.0/32.0`, patch `7`,
   `dof_index(0,nv/2)`, count `48`. Require an active edge band, center
   inclusion, patches only `{6,7}`, two sectors, and counts `24/24`.
2. L-prism split native interval: use patch `6`, `dof_index(nu/2,0)`, where
   the native parameter is `0.5`. Require both tied bottom-edge connections,
   sectors `{6}` and `{0,1,2}`, counts `24/24`, and selected DOFs on bottom
   patches `0` and `1`.
3. Feature-vertex tie: place a copied center DOF at
   `{0.67,-0.67,-0.63}`. Require sectors `{7}`, `{6}`, `{0,1,2}`, counts
   `16/16/16`, and no patch outside `{0,1,2,6,7}`.
4. Hollow-cylinder rim: choose an outer-wall DOF adjacent to the top annulus.
   Require sectors `{0,1,2,3}` and `{8,9,10,11}`, periodic-seam patches `3`
   and `11` present, and inner-wall/bottom-annulus patches `4..7,12..15`
   absent.
5. L-prism side-face interior farther than `2h`: require
   `edge_band_active == false` and element-by-element equality with
   `nearest_g1_cauchy_dofs(surface,cloud,center,48)`.
6. Repeat the split-interval and feature-vertex calls and require identical
   IDs, sector lists, and counts.

- [ ] **Step 2: Build and verify RED**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
```

Expected: compilation fails because `EdgeAwareCauchyDofSelection3D` and
`edge_aware_topological_cauchy_dofs` do not exist.

- [ ] **Step 3: Implement native interval distance**

In `native_nurbs_surface_3d.cpp`, add private helpers beside the existing
distance-stable selector helpers:

```cpp
Eigen::Vector3d patch_edge_point(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    double parameter);

double squared_distance_to_patch_edge_interval(
    const NativeNurbsSurface3D& surface,
    const geometry3d::NurbsPatchEdgeInterval3D& interval,
    const Eigen::Vector3d& point);
```

Map the interval parameter onto the fixed/varying NURBS coordinates according
to `UMin`, `UMax`, `VMin`, or `VMax`, then evaluate the original rational
NURBS patch. Minimize distance independently on every nonzero knot span
intersecting `[begin,end]`: test both endpoints and span midpoint, bracket the
best sample, and run a safeguarded one-dimensional golden-section refinement
until the parameter bracket is at most
`64*epsilon*max(1,abs(begin),abs(end))`. This handles circular NURBS edges
without replacing them by triangulator chords.

- [ ] **Step 4: Implement topology filtering and sector construction**

For a center DOF:

1. Obtain its transitive G1 component with `smooth_patch_component`.
2. Inspect only `geometric_connections` with `g1 == false` whose first or
   second patch belongs to that component.
3. Measure the first interval's physical distance; its paired interval is the
   same physical edge and must not be counted again.
4. Define the scale-aware tie tolerance as
   `max(1e-12*surface.geometry_model().control_bounds().diameter(),1e-14)`.
5. Collect every connection within that tolerance of the minimum.
6. Expand both incident patches of every tied connection through transitive
   G1 neighbors, take their union, then partition the admitted patches into
   induced G1 connected components. Sort patches inside a sector and sectors
   by their smallest patch ID.

If no incident non-G1 connection exists, or the minimum distance is strictly
greater than `2*h`, return the existing G1 vector directly and mark the edge
band inactive. Equality at `2h` is edge-aware.

- [ ] **Step 5: Implement deterministic balanced selection**

Within each admitted sector, sort candidates by
`(squared Euclidean distance to center, DOF ID)`. Allocate target quotas
`count/sector_count`, distributing the remainder first to the center sector
and then by smallest sector patch ID. Fill the center sector's quota with the
center DOF first, then its nearest remaining candidates. Fill every other
quota by its local ordering. If a sector exhausts its candidates, fill from
the globally nearest remaining admitted candidates. Sort the final IDs with
the same distance/ID comparator.

Do not add DOFs from outside admitted sectors. Return a short vector if the
admitted pool has fewer than the requested count; the application will record
and reject it.

- [ ] **Step 6: Verify GREEN and native regressions**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
.\build\apps\Release\native_nurbs_surface_3d_test.exe
```

Expected final line: `native NURBS model tests passed`.

- [ ] **Step 7: Commit**

```powershell
git add apps/native_nurbs_surface_3d.hpp apps/native_nurbs_surface_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: add edge-aware 3d Cauchy selector"
```

---

### Task 2: Independently configurable value and normal Cauchy pools

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d_route_test.cpp`

**Application interfaces:**

```cpp
enum class CauchyStencilPolicy3D {
    G1Nearest,
    TopologicalNearest,
    SamePatch,
    BalancedPatches,
    EdgeAwareTopologyAdjacent
};

struct CauchyStencilPolicies3D {
    CauchyStencilPolicy3D value = CauchyStencilPolicy3D::G1Nearest;
    CauchyStencilPolicy3D normal = CauchyStencilPolicy3D::G1Nearest;
};

enum class NeumannCauchyRoute3D {
    G1ValueG1Normal,
    EdgeValueG1Normal,
    EdgeValueEdgeNormal
};

const char* neumann_cauchy_route_name(NeumannCauchyRoute3D route);
CauchyStencilPolicies3D
neumann_cauchy_route_policies(NeumannCauchyRoute3D route);
```

- [ ] **Step 1: Write RED route and independent-pool tests**

In `neumann_exterior_zero_trace_3d_route_test.cpp`, add pure mapping assertions:

```cpp
require(neumann_cauchy_route_policies(
            NeumannCauchyRoute3D::EdgeValueG1Normal).value
        == CauchyStencilPolicy3D::EdgeAwareTopologyAdjacent,
        "edge/value route lost edge-aware value pool");
require(neumann_cauchy_route_policies(
            NeumannCauchyRoute3D::EdgeValueG1Normal).normal
        == CauchyStencilPolicy3D::G1Nearest,
        "edge/value route changed normal pool");
```

Build an L-prism `N=32` cloud and require:

- `g1_value_g1_normal` reproduces the current value and derivative ID vectors;
- at a non-G1 edge, `edge_value_g1_normal` changes value IDs but leaves
  derivative IDs exactly equal to the G1 control;
- `edge_value_edge_normal` uses edge-aware IDs for both pools;
- all three routes have exact counts `48/28` and include the center.

- [ ] **Step 2: Build and verify RED**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test --parallel 2
```

Expected: compilation fails because the route enum, paired policies, and
paired `build_cauchy_stencils` overload are absent.

- [ ] **Step 3: Add paired selection without breaking legacy callers**

Add this overload:

```cpp
CauchyStencilSet build_cauchy_stencils(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud& cloud,
    double h,
    int requested_value_count,
    int requested_derivative_count,
    CauchyStencilPolicies3D policies);
```

Keep the current single-policy overload and make it delegate with
`{policy,policy}`. `select_cauchy_dofs` must accept `h` and return both IDs and
edge diagnostics. For every center, call the edge-aware selector with the
requested value count once even for the G1 control: its IDs are used only
when the value policy is edge-aware, while its physical edge distance and
sector metadata populate route-independent diagnostics. Call it again with
the requested normal count only when the normal policy is edge-aware.
Consequently the G1 control's IDs still come directly from
`nearest_g1_cauchy_dofs`, but its edge-distance bins are populated from the
same native-edge calculation as the other routes.

Extend `CauchyStencil` with:

```cpp
double value_radius_over_h = 0.0;
double derivative_radius_over_h = 0.0;
double feature_edge_distance_over_h =
    std::numeric_limits<double>::infinity();
bool value_edge_band_active = false;
bool derivative_edge_band_active = false;
std::vector<std::vector<int>> value_sector_patch_ids;
std::vector<std::vector<int>> derivative_sector_patch_ids;
std::vector<int> value_sector_sample_counts;
std::vector<int> derivative_sector_sample_counts;
```

Retain `radius_over_h` as the maximum of the two radii so existing output and
tests remain source-compatible. Add `value_policy` and `normal_policy` to
`CauchyStencilSet`; retain `policy` for delegating single-policy callers.

- [ ] **Step 4: Reject short stencils with complete context**

Before building the fit, require exact `48/28` requested counts. The exception
must contain center DOF, value/normal actual counts, value/normal radii,
feature-edge distance, and serialized value/normal sector patch IDs/counts.
It must not substitute `TopologicalNearest` or `G1Nearest`.

- [ ] **Step 5: Preserve singular-value diagnostics**

Extend the already computed SVD result:

```cpp
struct CauchyFitDiagnostics3D {
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double condition = 0.0;
};
```

Store `singular[0]`, `singular[singular.size()-1]`, and their ratio in each
`CauchyFitMap3D`. Expose a const vector from `PanelCenterCauchyFit3D` and a
forwarding accessor from `PanelCenterHarmonicJetKFBI3D`. On rank failure,
append center DOF, actual counts, both radii, admitted sectors, `sigma_max`,
`sigma_min`, and cutoff ratio `3e-12` to the exception.

- [ ] **Step 6: Verify GREEN and unchanged legacy route**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

The existing route/owner tests and all new paired-pool assertions must exit
zero.

- [ ] **Step 7: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/neumann_exterior_zero_trace_3d_route_test.cpp
git commit -m "feat: split 3d value and normal Cauchy pools"
```

---

### Task 3: Detailed Neumann consistency and common-RHS probes

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d_route_test.cpp`

**Data interfaces:**

```cpp
struct NeumannManufacturedData3D {
    Eigen::VectorXd prescribed_normal_jump;
    Eigen::VectorXd exact_density;
};

struct NeumannRouteProbe3D {
    SolveMetrics3D physical;
    std::vector<double> physical_residuals;
    CommonRhsGmresProbe3D common;
    Eigen::VectorXd density_error;
    Eigen::VectorXd exact_equation_defect;
    double exact_mean_row_defect = 0.0;
};
```

- [ ] **Step 1: Write RED algebraic probe tests**

Add tests that:

1. build manufactured data and require both prescribed normal jump and exact
   density to have surface-weighted mean below `5e-13`;
2. build the common augmented RHS and require size `M+1`, tail exactly zero,
   weighted mean of the head below `5e-13`, and weighted RMS of the head equal
   to one within `5e-13`;
3. solve the existing `N=16` crossing-owner route and require
   `physical_residuals.size() == physical.iterations+1` and
   `common.residuals.size() == common.iterations+1`;
4. require `density_error.size() == M`,
   `exact_equation_defect.size() == M`, and finite mean-row defect.

- [ ] **Step 2: Build and verify RED**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test --parallel 2
```

Expected: compilation fails because the manufactured-data, common-RHS, and
detailed probe APIs are missing.

- [ ] **Step 3: Extract manufactured data without changing values**

Move the current `run_neumann_case` loops into
`make_neumann_manufactured_data(pipeline,transform)`. Compute the same
transformed harmonic value and normal derivative, apply the same discrete
compatibility correction to normal data, and subtract the surface-weighted
mean from exact density.

Keep `run_neumann_case` as a wrapper around the detailed implementation so all
old callers retain their behavior and default route.

- [ ] **Step 4: Add deterministic common augmented RHS**

For each DOF, normalize its native parameters to `xi,eta` in `[0,1]` using its
patch domain and set:

```cpp
head[q] =
    std::sin(2.0*kPi*xi + 0.37*(dof.patch_id + 1))
  + 0.5*std::cos(2.0*kPi*eta - 0.23*(dof.patch_id + 1))
  + 0.25*std::sin(2.0*kPi*(xi + eta));
```

Subtract the surface-weighted mean, divide by weighted RMS, and set the
bordered entry to zero. Throw if the audited weighted mean exceeds `5e-13` or
the RMS differs from one by more than `5e-13`.

Run this RHS through the same `ExteriorZeroTraceOperator3D` and GMRES settings
as the physical solve: `JointTricubicCrossingOwner`, tolerance `2e-10`,
restart `80`, cap `80`.

- [ ] **Step 5: Compute the exact-density equation defect literally**

Use the same compatibility-corrected normal data:

```cpp
ExteriorZeroTraceOperator3D op(
    pipeline,
    app3d::ExteriorValueRestrictMode3D::JointTricubicCrossingOwner);
Eigen::VectorXd exact_augmented =
    Eigen::VectorXd::Zero(op.problem_size());
exact_augmented.head(pipeline.surface_size()) = data.exact_density;
Eigen::VectorXd applied;
op.apply(exact_augmented, applied);
const Eigen::VectorXd rhs =
    op.right_hand_side(data.prescribed_normal_jump);
probe.exact_equation_defect =
    (applied - rhs).head(pipeline.surface_size());
probe.exact_mean_row_defect =
    (applied - rhs)[pipeline.surface_size()];
```

Do not remove a constant from the defect after applying the operator.

- [ ] **Step 6: Add edge-distance bin statistics**

Use the center DOF's native non-G1 edge distance recorded by the stencil. Use
exactly these disjoint bins:

- `lt_h`: `d/h < 1`;
- `h_to_2h`: `1 <= d/h <= 2`;
- `gt_2h`: `d/h > 2`.

For both density error and exact equation defect, record count, weight sum,
Linf, and weighted RMS
`sqrt(sum(weight*error^2)/sum(weight))`. Empty bins contain count/weight zero
and numeric norms zero, plus an explicit `empty=1` flag.

- [ ] **Step 7: Verify GREEN**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test --parallel 2
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

- [ ] **Step 8: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/neumann_exterior_zero_trace_3d_route_test.cpp
git commit -m "test: add Neumann edge consistency probes"
```

---

### Task 4: Dedicated three-route study, incremental CSVs, and N=32 smoke

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d_route_test.cpp`

**Command and output contract:**

```text
neumann_exterior_zero_trace_3d.exe --neumann-cauchy-study [N ...]
output/neumann_cross_edge_value_cauchy_3d/
  summary.csv
  gmres_residuals.csv
  dof_diagnostics.csv
  edge_distance_bins.csv
  stencil_diagnostics.csv
  owner_diagnostics.csv
```

- [ ] **Step 1: Add RED route-name and failure-row writer tests**

In the route test, require the exact route strings in the comparison table.
Write one successful and one failed synthetic row to a scoped test directory
under `output/`, read them back, and require:

- CSV strings with commas/quotes are escaped;
- a failed row preserves `status`, `failure_stage`, and `failure_message`;
- residual rows distinguish `physical` and `common`;
- every file contains `case_id,N,route`.

- [ ] **Step 2: Verify the CLI is RED**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-cauchy-study 32
```

Expected: nonzero exit because the command is not recognized.

- [ ] **Step 3: Implement the study driver**

Add `run_neumann_cross_edge_cauchy_study(std::vector<int> levels)`. Validate
power-of-two levels at least `16`, default this command to `32,64,128`, and
select the three fixed poses.

For each `(pose,N)`:

1. Build the grid, transformed native NURBS geometry, domain labels, surface
   DOFs, and label-mismatch audit once.
2. For each of the three routes, build its paired stencils and its own
   owner-enabled `PanelCenterHarmonicJetKFBI3D`.
3. Run only `JointTricubicCrossingOwner`.
4. Snapshot owner query count and fingerprint before both GMRES solves and
   after them.
5. Compare query counts/fingerprints across routes for the same `(pose,N)`;
   they must equal the G1-control reference.
6. Catch stencil/fit/solve exceptions per route, write a failed route row, and
   continue every other route and finer level. Never discard completed rows.

- [ ] **Step 4: Implement complete incremental writers**

Truncate all six CSVs and write their headers once at command start. After
every completed or failed route, append that route's blocks to every
applicable file and flush all streams. Emit a summary, three bin rows, and an
owner row even for a failed route; use `status`, `failure_stage`, and
`failure_message` to distinguish unavailable numeric fields. This preserves
auditable partial output without repeatedly rewriting the large per-DOF
files.

`summary.csv` must contain one row per attempted configuration with:

- case, `N`, `h`, route, value/normal pool, status/failure;
- surface/correction/crossing counts and label mismatches;
- physical/common convergence, iterations, final residual, first-to-last
  contraction, worst five-step contraction;
- common-RHS weighted mean and weighted RMS audits;
- density/interior Linf/L2 and adjacent orders;
- global exact-defect Linf/weighted-RMS and mean-row defect;
- setup, stencil, pipeline, physical solve, and common solve seconds;
- value/normal count minima/maxima, radius mean/max, sector-imbalance max;
- SVD condition median/p95/max and sigma-min minimum;
- owner query/fingerprint before/after and invariant booleans.

`gmres_residuals.csv` schema:

```text
case_id,N,route,rhs_kind,iteration,relative_residual
```

`dof_diagnostics.csv` stores patch/parameter/point/weight, edge distance over
`h`, density error, and exact equation defect.

`edge_distance_bins.csv` stores the three bins and the count/weight/Linf/RMS
statistics from Task 3.

`stencil_diagnostics.csv` stores each center's actual value/normal counts,
separate radii, edge-band flags, serialized sector patch IDs/sample counts,
`sigma_max`, `sigma_min`, and condition.

`owner_diagnostics.csv` reuses the existing aggregate owner fields and adds
reference query/fingerprint equality for the other two routes.

- [ ] **Step 5: Verify writer tests GREEN**

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

- [ ] **Step 6: Run the N=32 smoke**

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-cauchy-study 32
```

Exit code `2` is an admissible numerical result for an explicitly recorded
edge-route failure; it does not excuse missing rows or diagnostics. The G1
control must still pass as a regression guard.

Audit it independently:

```powershell
$out = 'output/neumann_cross_edge_value_cauchy_3d'
$s = Import-Csv "$out/summary.csv"
if ($s.Count -ne 9) { throw "expected 9 summary rows, got $($s.Count)" }
if (($s | Group-Object case_id,N,route | Where-Object Count -ne 1).Count) { throw 'duplicate configuration key' }
if (($s.route | Sort-Object -Unique).Count -ne 3) { throw 'missing route' }
if (($s | Where-Object { $_.status -ne 'ok' -and ([string]::IsNullOrWhiteSpace($_.failure_stage) -or [string]::IsNullOrWhiteSpace($_.failure_message)) }).Count) { throw 'unexplained route failure' }
if (($s | Where-Object { $_.route -eq 'g1_value_g1_normal' -and $_.status -ne 'ok' }).Count) { throw 'G1 control regression' }
if (($s | Where-Object { $_.status -eq 'ok' -and ($_.physical_converged -ne '1' -or $_.common_converged -ne '1') }).Count) { throw 'successful row lacks convergence' }
if (($s | Where-Object { $_.status -eq 'ok' -and ([double]$_.physical_final_residual -gt 2e-10 -or [double]$_.common_final_residual -gt 2e-10) }).Count) { throw 'successful row exceeds GMRES threshold' }
if (($s | Where-Object { $_.owner_templates_built -eq '1' -and ($_.owner_invariants_hold -ne '1' -or $_.owner_matches_reference -ne '1') }).Count) { throw 'owner invariant failure' }
if (($s | Where-Object { $_.common_rhs_built -eq '1' -and ([math]::Abs([double]$_.common_rhs_weighted_mean) -gt 5e-13 -or [math]::Abs([double]$_.common_rhs_weighted_rms - 1.0) -gt 5e-13) }).Count) { throw 'common RHS normalization failure' }
$r = Import-Csv "$out/gmres_residuals.csv"
if ((@($r.rhs_kind | Sort-Object -Unique) -join ',') -ne 'common,physical') { throw 'missing RHS history' }
$b = Import-Csv "$out/edge_distance_bins.csv"
if ($b.Count -ne 27) { throw "expected 27 bin rows, got $($b.Count)" }
```

- [ ] **Step 7: Commit**

```powershell
git add apps/neumann_exterior_zero_trace_3d.cpp apps/neumann_exterior_zero_trace_3d_route_test.cpp
git commit -m "feat: add Neumann cross-edge Cauchy study"
```

---

### Task 5: Formal 27-configuration run and numerical decision

**Files:**
- Create: `docs/superpowers/results/2026-07-25-3d-neumann-cross-edge-value-cauchy-comparison.md`
- Generated, untracked: `output/neumann_cross_edge_value_cauchy_3d/*.csv`

- [ ] **Step 1: Build the exact Release targets**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
```

- [ ] **Step 2: Run fast regressions before the long study**

```powershell
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

- [ ] **Step 3: Run all formal configurations**

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --neumann-cauchy-study 32 64 128
```

The program must attempt 27 configurations and preserve partial CSVs even if
its final exit code is `2` because a route failed a numerical criterion.

- [ ] **Step 4: Audit completeness and residual histories**

```powershell
$out = 'output/neumann_cross_edge_value_cauchy_3d'
$s = Import-Csv "$out/summary.csv"
if ($s.Count -ne 27) { throw "expected 27 summary rows, got $($s.Count)" }
if (($s | Group-Object case_id,N,route | Where-Object Count -ne 1).Count) { throw 'duplicate configuration key' }
$r = Import-Csv "$out/gmres_residuals.csv"
foreach ($row in $s | Where-Object status -eq 'ok') {
  foreach ($kind in 'physical','common') {
    $history = @($r | Where-Object { $_.case_id -eq $row.case_id -and [int]$_.N -eq [int]$row.N -and $_.route -eq $row.route -and $_.rhs_kind -eq $kind })
    $iterations = if ($kind -eq 'physical') { [int]$row.physical_iterations } else { [int]$row.common_iterations }
    if ($history.Count -ne $iterations + 1) { throw "incomplete $kind history: $($row.case_id) N=$($row.N) $($row.route)" }
  }
}
if ((Import-Csv "$out/edge_distance_bins.csv").Count -ne 81) { throw 'expected 81 edge-bin rows' }
if ((Import-Csv "$out/owner_diagnostics.csv").Count -ne 27) { throw 'expected 27 owner rows' }
```

- [ ] **Step 5: Apply the approved decision rules**

For the G1 control and `edge_value_g1_normal`, compute by route:

- worst physical and common GMRES count over all poses/levels;
- pose-to-pose iteration spread at each `N`;
- `N=128` combined `<2h` exact-defect Linf and weighted RMS;
- baseline `64->128` density/interior orders;
- each pose's `N=128` density/interior error ratio to the control.

Support the value-continuity hypothesis only if
`edge_value_g1_normal`:

1. reduces worst iteration count and pose spread;
2. reduces the `N=128` `<2h` exact-equation defect;
3. removes the baseline negative `64->128` density/interior order;
4. keeps every `N=128` density and interior error ratio at or below `1.10`;
5. converges for both RHS kinds below `2e-10` with all owner invariants true.

If only `edge_value_edge_normal` meets all five rules, conclude adjacent-face
normal information is required. If neither meets them, do not enlarge the
single-polynomial stencil further; recommend the already scoped next design:
sector-wise polynomials with shared edge value/tangential constraints.

- [ ] **Step 6: Write the result document**

The result document must contain:

- one table of all 27 physical/common iteration counts;
- one table of density/interior errors and observed orders;
- one table of `<h`, `h_to_2h`, and `>2h` exact-defect/density-error norms;
- worst count, pose spread, N=128 error ratios, and owner-invariant audit;
- a direct conclusion for each of the five rules;
- the selected next action: promote `edge_value_g1_normal`, promote
  `edge_value_edge_normal`, or proceed to sector-wise polynomials.

- [ ] **Step 7: Commit numerical evidence**

Do not add `output/`. Commit only the audited Markdown summary:

```powershell
git add docs/superpowers/results/2026-07-25-3d-neumann-cross-edge-value-cauchy-comparison.md
git commit -m "docs: report Neumann cross-edge Cauchy comparison"
```

---

### Task 6: Final verification and review

**Files:**
- Verify every source/test/document changed in Tasks 1-5.

- [ ] **Step 1: Rebuild from the current `build` directory**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d_route_test neumann_exterior_zero_trace_3d --parallel 2
```

- [ ] **Step 2: Re-run focused tests**

```powershell
.\build\apps\Release\native_nurbs_surface_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d_route_test.exe
```

- [ ] **Step 3: Re-audit the preserved formal outputs**

Do not rerun the N=32-only command here because it would replace the 27-row
formal CSVs. Reapply the completeness, residual-history, bin, and owner audits
from Task 5 to the preserved `32,64,128` output.

- [ ] **Step 4: Review the full implementation**

Use `superpowers:requesting-code-review`. Resolve every Critical or Important
finding, rerun the affected RED/GREEN test, and repeat review until none
remain. If a review change touches selector, fit, operator, or study
arithmetic, rerun the complete `32,64,128` study and update the result
document before completion.

- [ ] **Step 5: Verify repository hygiene**

```powershell
git diff --check
git status --short
```

Require no accidental build/output files staged, no production default change,
and no modification to the two pre-existing unrelated untracked documents.
