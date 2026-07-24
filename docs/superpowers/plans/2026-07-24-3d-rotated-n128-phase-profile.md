# 3D Rotated N=128 Phase Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add isolated, opt-in instrumentation and measure the current crossing-owner KFBI route for the rotated L-prism at `N=128` with mutually exclusive phase percentages.

**Architecture:** A small profiling ledger owns phase names, counters, percentage calculations, CSV output, and timer-overhead calibration. The existing 3D application receives an optional ledger pointer; when it is null, no detailed clock reads occur. A dedicated `--restrict-profile-owner` entry point selects only `rot_axis123_17deg`, while the existing owner-only and full-probe paths remain unchanged.

**Tech Stack:** C++17, `std::chrono::steady_clock`, Eigen, CMake/Visual Studio Release builds, PowerShell verification.

## Global Constraints

- Develop in an isolated worktree based on the current `main` HEAD.
- Profile only the `JointTricubicCrossingOwner` route and use `N=128` for the accepted measurement.
- Keep existing command-line behavior and numerical algorithms unchanged.
- Emit non-overlapping leaf phases; parent setup, pipeline, and route timers are validation totals only.
- Report algorithm percentages without diagnostic I/O and wall percentages with diagnostic I/O.
- Preserve the rotated reference error `5.3527160592814482e-08`, physical/common iteration counts `25/23`, physical residual at most `8.2e-11`, and common residual at most `1.9e-10`.
- Require zero new geometry queries during GMRES.
- If estimated timer overhead or the aggregate-time comparison exceeds 5%, do not use the first profile for conclusions; replace per-query timing with coarser batching and repeat.
- Do not commit generated CSV files or run logs.

---

### Task 1: Add a tested phase-profile ledger

**Files:**
- Create: `apps/kfbi_phase_profile_3d.hpp`
- Create: `apps/kfbi_phase_profile_3d.cpp`
- Create: `apps/kfbi_phase_profile_3d_test.cpp`
- Modify: `apps/CMakeLists.txt`

**Interfaces:**
- Produces: `kfbim::app3d::PhaseProfileKind3D`, `PhaseProfileRecord3D`, and `PhaseProfile3D`.
- Produces: `phase_profile_kind_count_3d`, `phase_profile_name_3d`,
  `phase_profile_is_algorithm_3d`,
  `calibrate_steady_clock_read_seconds_3d`, and
  `write_phase_profile_csv_3d`.
- Consumed by Task 2 from `apps/neumann_exterior_zero_trace_3d.cpp`.

- [ ] **Step 0: Configure the isolated x64 Release-capable build tree**

Run once from the isolated worktree root:

```powershell
cmake -S . -B build -G "Visual Studio 15 2017" -A x64 `
    -DKFBIM_BUILD_3D=ON -DKFBIM_BUILD_APPS=ON `
    -DEigen3_DIR=E:/Code/vs_code/KFBI-Corner/build/_deps/eigen3-build
```

Expected: CMake configures successfully, finds CGAL, and generates the 3D
application targets without fetching Eigen from the network.

- [ ] **Step 1: Write the failing ledger test**

Create `apps/kfbi_phase_profile_3d_test.cpp` with a local `require` helper and
the following checks:

```cpp
#include "kfbi_phase_profile_3d.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}
}

int main()
{
    using namespace kfbim::app3d;
    PhaseProfile3D profile;
    profile.add(PhaseProfileKind3D::GeometryAndDomain, 2.0, 4);
    profile.add(PhaseProfileKind3D::FftBulkSolve, 1.0, 2);
    profile.add(PhaseProfileKind3D::DiagnosticOutput, 0.5, 1);
    profile.note_timer_reads(20);
    profile.set_seconds_per_clock_read(1.0e-7);
    profile.finalize(4.0);

    require(std::abs(profile.algorithm_seconds() - 3.0) < 1.0e-14,
            "algorithm denominator is wrong");
    require(std::abs(profile.record(
                PhaseProfileKind3D::WallOverhead).seconds - 0.5) < 1.0e-14,
            "wall remainder is wrong");
    require(profile.record(
                PhaseProfileKind3D::GeometryAndDomain).calls == 4,
            "call count is wrong");
    require(std::abs(profile.estimated_timer_overhead_seconds() - 2.0e-6)
                < 1.0e-14,
            "timer overhead estimate is wrong");

    const std::filesystem::path output =
        std::filesystem::temp_directory_path()
        / "kfbim_phase_profile_3d_test.csv";
    write_phase_profile_csv_3d(output, "rot_axis123_17deg", 128, profile);
    std::ifstream input(output);
    std::string header;
    std::getline(input, header);
    require(header ==
                "case_id,N,phase,seconds,calls,average_seconds,"
                "algorithm_percent,wall_percent,timer_reads,"
                "estimated_timer_overhead_seconds,"
                "estimated_timer_overhead_percent",
            "profile CSV header is wrong");
    std::size_t rows = 0;
    std::string line;
    while (std::getline(input, line))
        ++rows;
    require(rows == phase_profile_kind_count_3d(),
            "profile CSV row count is wrong");
    std::filesystem::remove(output);
    return 0;
}
```

Add the test target to the `KFBIM_BUILD_3D` section of
`apps/CMakeLists.txt`, link it to a new `kfbim_phase_profile_3d` static
library, and link that library to `neumann_exterior_zero_trace_3d`.

- [ ] **Step 2: Build to verify the test fails**

Run:

```powershell
cmake --build build --config Release --target kfbim_phase_profile_3d_test -- /m
```

Expected: compilation fails because `kfbi_phase_profile_3d.hpp` and its
symbols do not exist yet.

- [ ] **Step 3: Implement the ledger**

Define these exact leaf phases in `apps/kfbi_phase_profile_3d.hpp`:

```cpp
enum class PhaseProfileKind3D {
    GeometryAndDomain,
    SurfaceDofsAndStencils,
    GridPairAndLabelValidation,
    PipelineFixedInitialization,
    CrossingRows,
    NurbsSegmentIntersections,
    TraceOwnerTemplateAssembly,
    ExactFieldsAndOtherSetup,
    CauchyCoefficients,
    SpreadRhsAssembly,
    FftBulkSolve,
    RestrictContinuedSamples,
    RestrictRecovery,
    GmresAndOtherRoute,
    DiagnosticOutput,
    WallOverhead,
    Count
};

struct PhaseProfileRecord3D {
    double seconds = 0.0;
    std::uint64_t calls = 0;
};

class PhaseProfile3D {
public:
    void add(PhaseProfileKind3D kind, double seconds,
             std::uint64_t calls = 1);
    void set(PhaseProfileKind3D kind, double seconds,
             std::uint64_t calls = 1);
    const PhaseProfileRecord3D& record(PhaseProfileKind3D kind) const;
    double algorithm_seconds() const;
    double measured_seconds_without_wall_overhead() const;
    void note_timer_reads(std::uint64_t count);
    void set_seconds_per_clock_read(double seconds);
    std::uint64_t timer_reads() const;
    double estimated_timer_overhead_seconds() const;
    void finalize(double wall_seconds);
    double wall_seconds() const;
};
```

Map the enum values to these exact CSV names, in the same order:
`geometry_and_domain`, `surface_dofs_and_stencils`,
`grid_pair_and_label_validation`, `pipeline_fixed_initialization`,
`crossing_rows`, `nurbs_segment_intersections`,
`trace_owner_template_assembly`, `exact_fields_and_other_setup`,
`cauchy_coefficients`, `spread_rhs_assembly`, `fft_bulk_solve`,
`restrict_continued_samples`, `restrict_recovery`,
`gmres_and_other_route`, `diagnostic_output`, and `wall_overhead`.

Implement the following invariants in `apps/kfbi_phase_profile_3d.cpp`:

- reject non-finite or negative durations;
- reject attempts to add directly to `WallOverhead`;
- `algorithm_seconds()` excludes `DiagnosticOutput` and `WallOverhead`;
- `finalize()` sets `WallOverhead` to wall time minus all other leaf
  categories;
- throw if that remainder is below
  `-max(1e-9, 1e-8 * wall_seconds)`, and clamp only round-off-sized negative
  values to zero;
- emit one CSV row per enum value in enum order;
- use zero for an algorithm percentage when the row is not an algorithm
  phase;
- use `seconds / calls` for average time, or zero when calls are zero.

Implement clock calibration as 200,000 consecutive
`std::chrono::steady_clock::now()` reads divided by 200,000. Store the final
time point in a `volatile`-observable integer duration so the loop cannot be
optimized away.

- [ ] **Step 4: Build and run the ledger test**

Run:

```powershell
cmake --build build --config Release --target kfbim_phase_profile_3d_test -- /m
.\build\apps\Release\kfbi_phase_profile_3d_test.exe
```

Expected: build exits zero and the test executable exits zero without output.

- [ ] **Step 5: Commit the ledger**

```powershell
git add -- apps/kfbi_phase_profile_3d.hpp apps/kfbi_phase_profile_3d.cpp apps/kfbi_phase_profile_3d_test.cpp apps/CMakeLists.txt
git commit -m "test: add 3d phase profiling ledger"
```

---

### Task 2: Instrument the current KFBI kernels without changing normal runs

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:1-45`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:834-1565`

**Interfaces:**
- Consumes: the Task 1 `PhaseProfile3D` API.
- Changes: `PanelCenterHarmonicJetKFBI3D` gains a final optional
  `app3d::PhaseProfile3D* phase_profile = nullptr` constructor argument.
- Produces: kernel counters used by the profile runner in Task 3.

- [ ] **Step 1: Run a source contract that must initially fail**

Run before modifying the application:

```powershell
$source = Get-Content apps\neumann_exterior_zero_trace_3d.cpp -Raw
if ($source -notmatch 'PhaseProfileKind3D::NurbsSegmentIntersections') {
    throw 'expected red test: intersection kernel is not instrumented'
}
```

Expected: the command throws
`expected red test: intersection kernel is not instrumented`.

- [ ] **Step 2: Add zero-overhead-disabled timing helpers**

Include `kfbi_phase_profile_3d.hpp`. Add a nullable `phase_profile_` member
before the expensive fit/bulk members and initialize it from the final
constructor argument.

Add this alias beside the existing application aliases:

```cpp
using PhaseProfileKind3D = app3d::PhaseProfileKind3D;
```

Add these private helpers:

```cpp
using ProfileClock3D = std::chrono::steady_clock;

ProfileClock3D::time_point profile_start() const
{
    if (phase_profile_ == nullptr)
        return {};
    phase_profile_->note_timer_reads(1);
    return ProfileClock3D::now();
}

double profile_elapsed(ProfileClock3D::time_point start) const
{
    if (phase_profile_ == nullptr)
        return 0.0;
    phase_profile_->note_timer_reads(1);
    return std::chrono::duration<double>(
        ProfileClock3D::now() - start).count();
}

void profile_add(PhaseProfileKind3D kind,
                 ProfileClock3D::time_point start,
                 std::uint64_t calls = 1) const
{
    if (phase_profile_ != nullptr)
        phase_profile_->add(kind, profile_elapsed(start), calls);
}
```

The null branch must not call `steady_clock::now()`.

- [ ] **Step 3: Instrument pipeline construction**

Wrap `build_crossing_rows()` as `CrossingRows`. Wrap the full
`build_trace_templates(&intersector)` call with a parent timer that is not
emitted. Around each existing `restrict_intersector->intersect_segment(...)`
call, record `NurbsSegmentIntersections` with one call.

After trace-template construction, calculate:

```cpp
const double owner_assembly_seconds =
    trace_template_seconds
    - (intersection_seconds_after - intersection_seconds_before);
phase_profile_->add(
    PhaseProfileKind3D::TraceOwnerTemplateAssembly,
    owner_assembly_seconds, 1);
```

Reject a negative value beyond timer round-off. This category intentionally
contains owner selection, tricubic support bookkeeping, and all remaining
template work.

- [ ] **Step 4: Instrument operator evaluation and restrict**

In both `evaluate()` and `field_from_grid_and_jumps()`, time
`fit_.coefficients(...)` as `CauchyCoefficients`.

In `evaluate()`, time only the crossing-row RHS loop as
`SpreadRhsAssembly`, and time only
`bulk_.solve(-rhs, result.potential)` as `FftBulkSolve`.

At entry/return of `continued_samples()`, record
`RestrictContinuedSamples`. At entry/return of `recover_trace()`, record
`RestrictRecovery`. Do not put a parent timer around
`exterior_normal_trace()`, because that would double-count both leaf phases.

- [ ] **Step 5: Build and run focused regressions**

Run:

```powershell
cmake --build build --config Release --target kfbim_phase_profile_3d_test neumann_exterior_zero_trace_3d crossing_owner_restrict_3d_test -- /m
.\build\apps\Release\kfbi_phase_profile_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --help
```

Expected: all commands exit zero. The help output is still the pre-profile
interface at this task boundary.

- [ ] **Step 6: Commit kernel instrumentation**

```powershell
git add -- apps/neumann_exterior_zero_trace_3d.cpp apps/kfbi_phase_profile_3d_test.cpp
git commit -m "feat: instrument 3d kfbi phase kernels"
```

---

### Task 3: Add the rotated owner-only profile runner and smoke-test it

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:3470-3650`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp:3923-4250`

**Interfaces:**
- Changes:
  `run_normal_restrict_causal_probe(std::vector<int>, bool, bool phase_profile)`
  where the third argument defaults to `false`.
- Produces: `--restrict-profile-owner [N]`, defaulting to `N=128`.
- Produces:
  `output/dirichlet_normal_restrict_crossing_owner_profile_3d/phase_profile.csv`.

- [ ] **Step 1: Verify the new CLI is absent**

Run:

```powershell
$helpText = .\build\apps\Release\neumann_exterior_zero_trace_3d.exe --help | Out-String
if ($helpText -match '--restrict-profile-owner') {
    throw 'profile command unexpectedly exists before implementation'
}
throw 'expected red test: --restrict-profile-owner is absent'
```

Expected: the command reaches the final throw, proving the new entry point is
absent.

- [ ] **Step 2: Add profile-only case and output selection**

Extend the runner with `bool phase_profile`. Enforce:

```cpp
if (phase_profile && !owner_only)
    throw std::invalid_argument(
        "phase profiling requires the crossing-owner-only route");
if (phase_profile && levels.size() != 1)
    throw std::invalid_argument(
        "phase profiling accepts exactly one grid level");
```

When profiling, select only `rot_axis123_17deg` and write to
`dirichlet_normal_restrict_crossing_owner_profile_3d`. Otherwise preserve the
current three-case lists, output directories, and route lists byte-for-byte.

Add `--restrict-profile-owner [N]` to `print_usage()`. In `main()`, give it
the default level `{128}` and dispatch:

```cpp
return run_normal_restrict_causal_probe(
    levels, true, true);
```

Existing `--restrict-probe` and `--restrict-probe-owner` calls pass
`phase_profile=false`.

- [ ] **Step 3: Add mutually exclusive outer timings**

Start the profile wall timer before the initial diagnostic writers. Time
these outer blocks directly:

- grid, transformed geometry, and `NurbsCartesianDomain3D` as
  `GeometryAndDomain`;
- native surface DOFs, validation, and Cauchy stencils as
  `SurfaceDofsAndStencils`;
- `GridPair3D` plus the full label-validation loop as
  `GridPairAndLabelValidation`.

Measure the full pipeline constructor. After construction, derive:

```cpp
pipeline_fixed =
    pipeline_total
    - crossing_rows_delta
    - intersection_delta
    - trace_owner_assembly_delta;
```

Store that positive remainder as `PipelineFixedInitialization`. It contains
the fit maps, FFT plan/buffers, correction support, and joint trace fit.

Measure the remaining setup block from diagnostics extraction through common
RHS creation. Subtract the `CauchyCoefficients` delta generated by
`field_from_grid_and_jumps()` and record the remainder as
`ExactFieldsAndOtherSetup`.

Measure the whole owner route. Subtract route-local deltas for
`CauchyCoefficients`, `SpreadRhsAssembly`, `FftBulkSolve`,
`RestrictContinuedSamples`, and `RestrictRecovery`; record the positive
remainder as `GmresAndOtherRoute`.

Wrap every existing summary, residual, localization, owner-summary, and
owner-term writer in `DiagnosticOutput`. Do not count the final tiny
`phase_profile.csv` write in algorithm time.

- [ ] **Step 4: Finalize and write the profile**

At the end of the single case:

1. set the calibrated seconds per clock read;
2. finalize with elapsed wall time;
3. write `phase_profile.csv`;
4. print algorithm seconds, wall seconds, estimated timer overhead, and its
   wall percentage;
5. throw if the phase rows do not sum to wall time within
   `max(1e-6, 1e-8 * wall_seconds)`.

The writer must emit exactly 16 phase rows and the percentages of all rows
must sum to 100% of wall time within `1e-8`.

- [ ] **Step 5: Build and verify the CLI**

Run:

```powershell
cmake --build build --config Release --target neumann_exterior_zero_trace_3d kfbim_phase_profile_3d_test -- /m
$helpText = .\build\apps\Release\neumann_exterior_zero_trace_3d.exe --help | Out-String
if ($helpText -notmatch '--restrict-profile-owner') {
    throw 'profile command missing from help'
}
```

Expected: build exits zero and help contains
`--restrict-profile-owner [N]`.

- [ ] **Step 6: Run an `N=16` integration smoke test**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-profile-owner 16
$profilePath = 'output\dirichlet_normal_restrict_crossing_owner_profile_3d\phase_profile.csv'
$summaryPath = 'output\dirichlet_normal_restrict_crossing_owner_profile_3d\summary.csv'
$phases = Import-Csv $profilePath
$summary = Import-Csv $summaryPath
if ($phases.Count -ne 16) { throw 'expected 16 profile phases' }
if ($summary.Count -ne 1) { throw 'expected one rotated owner row' }
if ($summary[0].case_id -ne 'rot_axis123_17deg') {
    throw 'profile selected the wrong rigid pose'
}
if ($summary[0].route -ne 'joint_tricubic_crossing_owner') {
    throw 'profile selected the wrong restrict route'
}
$wallPercent = ($phases | Measure-Object -Property wall_percent -Sum).Sum
if ([math]::Abs($wallPercent - 100.0) -gt 1.0e-8) {
    throw 'wall percentages do not sum to 100'
}
```

Expected: the application and all assertions exit zero.

- [ ] **Step 7: Re-run existing owner-only behavior**

Run:

```powershell
.\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-probe-owner 16
$rows = Import-Csv 'output\dirichlet_normal_restrict_crossing_owner_3d\summary.csv'
if ($rows.Count -ne 3) { throw 'owner-only regression row count changed' }
if (($rows.route | Sort-Object -Unique) -ne 'joint_tricubic_crossing_owner') {
    throw 'owner-only regression route changed'
}
```

Expected: three rigid poses and only the crossing-owner route.

- [ ] **Step 8: Commit the profile entry point**

```powershell
git add -- apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: add rotated n128 kfbi phase profile"
```

---

### Task 4: Run the accepted N=128 measurement and report it

**Files:**
- Generate, do not commit:
  `output/dirichlet_normal_restrict_crossing_owner_profile_3d/phase_profile.csv`
- Generate, do not commit:
  `output/dirichlet_normal_restrict_crossing_owner_profile_3d/summary.csv`
- Generate, do not commit:
  `output/dirichlet_normal_restrict_crossing_owner_profile_3d/run_N128.log`

**Interfaces:**
- Consumes: the Task 3 profile executable and output schema.
- Produces: the measured table and ranked acceleration recommendations in the
  final task handoff.

- [ ] **Step 1: Run the Release profile once**

Run from the isolated worktree root:

```powershell
$profileOutput = 'output\dirichlet_normal_restrict_crossing_owner_profile_3d'
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe --restrict-profile-owner 128 2>&1 |
    Tee-Object -FilePath "$profileOutput\run_N128.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: one completed rotated crossing-owner row and 16 profile phase rows.

- [ ] **Step 2: Verify numerical equivalence**

Run:

```powershell
$profileOutput = 'output\dirichlet_normal_restrict_crossing_owner_profile_3d'
$row = Import-Csv "$profileOutput\summary.csv"
if ($row.Count -ne 1) { throw 'expected one N=128 result' }
$referenceError = 5.3527160592814482e-08
$relativeError = [math]::Abs(
    ([double]$row.physical_interior_linf - $referenceError) / $referenceError)
if ($relativeError -gt 1.0e-12) { throw 'interior error changed' }
if ([int]$row.physical_iterations -ne 25) {
    throw 'physical GMRES iteration count changed'
}
if ([int]$row.common_iterations -ne 23) {
    throw 'common GMRES iteration count changed'
}
if ([double]$row.physical_final_residual -gt 8.2e-11) {
    throw 'physical residual exceeded its reference bound'
}
if ([double]$row.common_final_residual -gt 1.9e-10) {
    throw 'common residual exceeded its reference bound'
}
```

Expected: every assertion passes. The application’s existing geometry-query
invariant must also pass during the run.

- [ ] **Step 3: Verify the timing ledger and overhead**

Run:

```powershell
$profileOutput = 'output\dirichlet_normal_restrict_crossing_owner_profile_3d'
$phases = Import-Csv "$profileOutput\phase_profile.csv"
if ($phases.Count -ne 16) { throw 'profile is incomplete' }
$wallPercent = ($phases | Measure-Object wall_percent -Sum).Sum
if ([math]::Abs($wallPercent - 100.0) -gt 1.0e-8) {
    throw 'profile percentages are not exhaustive'
}
$negative = $phases | Where-Object { [double]$_.seconds -lt 0.0 }
if ($negative) { throw 'profile contains a negative phase' }
$overheadPercent =
    [double]$phases[0].estimated_timer_overhead_percent
if ($overheadPercent -gt 5.0) {
    throw 'profiling overhead exceeds the accepted 5 percent limit'
}
$phases |
    Sort-Object {[double]$_.seconds} -Descending |
    Format-Table phase,seconds,calls,average_seconds,
                 algorithm_percent,wall_percent
```

Compare the measured parent pipeline time with the archived rotated `N=128`
value `764.5994489 s`. State explicitly that the archived run also built the
exterior-only restrict object, so this comparison is a conservative
perturbation check rather than an exact control.

- [ ] **Step 4: Derive acceleration priorities from measured shares**

Apply these evidence rules:

- if `nurbs_segment_intersections` is the largest phase, prioritize a
  target/G1 local-patch fast path, global-intersector fallback only for
  ambiguous/foreign crossings, per-thread intersectors, and parallel
  trace-template construction;
- if `fft_bulk_solve` is largest among repeated kernels, prioritize plan and
  buffer reuse, a parallel FFT backend, and eliminating redundant operator
  applications before changing GMRES tolerances;
- if `spread_rhs_assembly` or either restrict phase is large, prioritize
  contiguous prepacked rows, removal of small Eigen allocations, SIMD, and
  DOF-parallel loops;
- report diagnostic aggregation as a storage/I/O optimization only, not as
  an algorithmic speedup;
- quantify every recommendation against its measured wall percentage and
  state its numerical-risk level.

- [ ] **Step 5: Final verification**

Run:

```powershell
git diff --check
git status --short
.\build\apps\Release\kfbi_phase_profile_3d_test.exe
.\build\apps\Release\crossing_owner_restrict_3d_test.exe
```

Expected: no whitespace errors; generated output remains untracked or ignored;
the isolated worktree has no uncommitted source change; the two pre-existing
unrelated untracked documents in the main workspace remain untouched; both
