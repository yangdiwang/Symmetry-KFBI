# 3D Neumann Edge-Cauchy Forced Extended Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, non-adopting `--force-extended` route and use it to obtain reproducible `N=128` Neumann edge-Cauchy A/B evidence.

**Architecture:** Keep the coarse acceptance evaluator unchanged and isolate the only new policy in a pure Boolean gate. The application parser passes an explicit force flag into the existing study driver; forced execution may enter `N=128` after coarse failure, but the recorded acceptance and process exit semantics remain unchanged.

**Tech Stack:** C++17, Eigen, CMake/Visual Studio x64 Release, PowerShell CSV audit.

## Global Constraints

- The default `--neumann-edge-cauchy-study 32 64 128` command must still stop before `N=128` when coarse acceptance fails.
- Only `--neumann-edge-cauchy-study --force-extended 32 64 128` may continue after that failure.
- Do not change numerical routes, geometry, stencils, edge samples, GMRES tolerance `2e-10`, restart `80`, cap `80`, or any acceptance threshold.
- A forced run retains the failed coarse `overall_pass`; a nonzero final process exit remains valid evidence of non-adoption.
- Write forced evidence to `output/neumann_edge_cauchy_3d_n128_forced`, never over the authoritative coarse directory.
- Do not modify `apps/CMakeLists.txt`; all required targets already exist.
- Preserve unrelated working-tree changes.

---

## File map

- `apps/neumann_edge_cauchy_study_3d.hpp`: declare the pure extended-run gate.
- `apps/neumann_edge_cauchy_study_3d.cpp`: implement the pure gate without touching acceptance calculations.
- `apps/neumann_edge_cauchy_study_3d_test.cpp`: verify the complete gate truth table.
- `apps/neumann_exterior_zero_trace_3d.cpp`: parse `--force-extended`, retain default behavior, and issue an explicit forced-run warning.
- `docs/superpowers/results/2026-07-26-3d-neumann-edge-cauchy-n128-forced.md`: record raw provenance, `64 -> 128` orders, GMRES results, and adoption status.

### Task 1: Extended-run gate policy

**Files:**
- Modify: `apps/neumann_edge_cauchy_study_3d_test.cpp`
- Modify: `apps/neumann_edge_cauchy_study_3d.hpp`
- Modify: `apps/neumann_edge_cauchy_study_3d.cpp`

**Interfaces:**
- Consumes: coarse `NeumannEdgeCauchyEvaluation3D::all_pass` and the explicit CLI force Boolean.
- Produces: `bool neumann_edge_cauchy_should_enter_n128_3d(bool coarse_all_pass, bool force_extended)`.

- [ ] **Step 1: Write the failing truth-table test**

Add this function before `test_n128_isolation()`:

```cpp
void test_extended_run_gate_policy()
{
    require(neumann_edge_cauchy_should_enter_n128_3d(true, false),
            "passing coarse evidence did not enter N=128");
    require(neumann_edge_cauchy_should_enter_n128_3d(true, true),
            "force flag blocked passing coarse evidence");
    require(!neumann_edge_cauchy_should_enter_n128_3d(false, false),
            "default route bypassed a failed coarse gate");
    require(neumann_edge_cauchy_should_enter_n128_3d(false, true),
            "explicit force did not enter N=128 after coarse failure");
}
```

Call it from `main()` immediately after `test_level_prefixes()`.

- [ ] **Step 2: Build to verify RED**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_cauchy_study_3d_test -- /m:1
```

Expected: compilation fails because
`neumann_edge_cauchy_should_enter_n128_3d` is undeclared.

- [ ] **Step 3: Declare and implement the minimal gate**

Add to `apps/neumann_edge_cauchy_study_3d.hpp` before
`process_neumann_edge_cauchy_pair_3d`:

```cpp
bool neumann_edge_cauchy_should_enter_n128_3d(
    bool coarse_all_pass,
    bool force_extended);
```

Add to `apps/neumann_edge_cauchy_study_3d.cpp` immediately before
`process_neumann_edge_cauchy_pair_3d`:

```cpp
bool neumann_edge_cauchy_should_enter_n128_3d(
    bool coarse_all_pass,
    bool force_extended)
{
    return coarse_all_pass || force_extended;
}
```

- [ ] **Step 4: Build and run to verify GREEN**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_edge_cauchy_study_3d_test -- /m:1
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
```

Expected: build exit `0`, executable exit `0`, and
`3D Neumann edge-Cauchy study tests passed`.

- [ ] **Step 5: Commit**

```powershell
git add -- `
  apps/neumann_edge_cauchy_study_3d.hpp `
  apps/neumann_edge_cauchy_study_3d.cpp `
  apps/neumann_edge_cauchy_study_3d_test.cpp
git commit -m "feat: define Neumann extended evidence gate"
```

### Task 2: Explicit CLI wiring with unchanged default gate

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`

**Interfaces:**
- Consumes: `neumann_edge_cauchy_should_enter_n128_3d(...)`.
- Produces: CLI form
  `--neumann-edge-cauchy-study --force-extended 32 64 128`.

- [ ] **Step 1: Verify the CLI is RED**

Run:

```powershell
$help = & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe --help
if ($help -match '--force-extended') { exit 0 } else { exit 1 }
```

Expected: exit `1` because the option is absent.

Also run:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
  --neumann-edge-cauchy-study --force-extended 32 64 128
```

Expected: immediate nonzero exit from parsing `--force-extended` as an
integer; no numerical pair starts.

- [ ] **Step 2: Extend the study-driver signature and gate**

Change:

```cpp
int run_neumann_edge_cauchy_study_3d(std::vector<int> levels)
```

to:

```cpp
int run_neumann_edge_cauchy_study_3d(
    std::vector<int> levels,
    bool force_extended)
```

Replace the `N == 128` rejection condition with:

```cpp
const bool enter_extended =
    app3d::neumann_edge_cauchy_should_enter_n128_3d(
        evaluation.all_pass, force_extended);
if (!enter_extended) {
    std::cerr
        << "error: N=128 gated off because the completed N=32/64 "
           "Neumann edge-Cauchy pilot did not pass\n";
    return 1;
}
if (!evaluation.all_pass) {
    std::cerr
        << "warning: forcing N=128 extended evidence after failed "
           "N=32/64 acceptance; acceptance thresholds are unchanged\n";
}
```

- [ ] **Step 3: Parse only the explicit option**

After computing `neumann_edge_cauchy_study`, add:

```cpp
const bool force_neumann_edge_cauchy_extended =
    neumann_edge_cauchy_study && argc >= 3
    && std::string(argv[2]) == "--force-extended";
const int first_level_argument =
    force_neumann_edge_cauchy_extended ? 3 : 2;
```

Replace level parsing with:

```cpp
if (argc > first_level_argument) {
    levels.clear();
    for (int argument = first_level_argument; argument < argc; ++argument)
        levels.push_back(parse_grid_level_argument(argv[argument]));
}
```

Pass the flag at dispatch:

```cpp
return run_neumann_edge_cauchy_study_3d(
    levels, force_neumann_edge_cauchy_extended);
```

Change only the edge-Cauchy usage line to:

```cpp
<< "       " << executable
<< " --neumann-edge-cauchy-study [--force-extended] [N ...]\n"
```

Add this explanation after the edge-Cauchy level description:

```cpp
<< "  --force-extended records N=128 evidence after a failed coarse "
   "gate without changing acceptance.\n"
```

- [ ] **Step 4: Build and verify the CLI is GREEN**

Run:

```powershell
cmake --build build --config Release `
  --target neumann_exterior_zero_trace_3d `
           neumann_edge_cauchy_study_3d_test -- /m:1
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
$help = & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe --help
if ($help -notmatch '--force-extended') { exit 1 }
```

Expected: build and unit test exit `0`; help contains the exact option.

Run the parser-only invalid-level check:

```powershell
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
  --neumann-edge-cauchy-study --force-extended 16
```

Expected: nonzero exit with
`Neumann edge-Cauchy study N must be 32, 64, or 128`, proving the option
was consumed before grid parsing.

- [ ] **Step 5: Commit**

```powershell
git add -- apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: expose forced Neumann extended evidence"
```

### Task 3: Verify defaults and collect `N=128` evidence

**Files:**
- Create: `docs/superpowers/results/2026-07-26-3d-neumann-edge-cauchy-n128-forced.md`
- Generated, ignored:
  `output/neumann_edge_cauchy_3d_default_gate_check/*`
- Generated, ignored:
  `output/neumann_edge_cauchy_3d_n128_forced/*`

**Interfaces:**
- Consumes: the explicit CLI implemented in Task 2.
- Produces: 18-row A/B evidence and an independently audited result report.

- [ ] **Step 1: Run fresh build and regression verification**

Run:

```powershell
cmake --build build --config Release -- /m:1
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe `
  --sample-count-boundaries
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe `
  --runtime-no-alloc
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe `
  --negative-symmetry-guard
```

Expected: build and all five test invocations exit `0`.

- [ ] **Step 2: Prove the default gate is unchanged**

Run:

```powershell
$env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR = `
  'output/neumann_edge_cauchy_3d_default_gate_check'
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
  --neumann-edge-cauchy-study 32 64 128
$defaultExit = $LASTEXITCODE
$rows = Import-Csv `
  output/neumann_edge_cauchy_3d_default_gate_check\summary.csv
if ($defaultExit -eq 0 -or @($rows).Count -ne 12 `
    -or @($rows | Where-Object N -eq '128').Count -ne 0) {
    exit 1
}
```

Expected: application exit is nonzero, exactly 12 summary rows exist,
and none has `N=128`.

- [ ] **Step 3: Run forced extended evidence**

Run:

```powershell
$env:KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR = `
  'output/neumann_edge_cauchy_3d_n128_forced'
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
  --neumann-edge-cauchy-study --force-extended 32 64 128
$forcedExit = $LASTEXITCODE
Write-Output "FORCED_EXIT=$forcedExit"
```

Expected: all requested pairs are attempted. A nonzero final exit is
expected while coarse `overall_pass` remains `fail`.

- [ ] **Step 4: Audit completeness and recompute orders**

Run:

```powershell
$rows = Import-Csv output\neumann_edge_cauchy_3d_n128_forced\summary.csv
if (@($rows).Count -ne 18) { throw 'expected 18 summary rows' }
$n128 = @($rows | Where-Object N -eq '128')
if ($n128.Count -ne 6) { throw 'expected six N=128 rows' }
if (@($n128 | Where-Object {
    $_.pair_completed -ne 'true' -or
    $_.gmres_converged -ne 'true' -or
    $_.residual_history_valid -ne 'true'
}).Count -ne 0) { throw 'incomplete N=128 evidence' }

$audit = foreach ($fine in $n128) {
    $coarse = $rows | Where-Object {
        $_.case_id -eq $fine.case_id -and
        $_.mode -eq $fine.mode -and $_.N -eq '64'
    } | Select-Object -First 1
    [pscustomobject]@{
        case_id = $fine.case_id
        mode = $fine.mode
        interior_linf_64 = [double]$coarse.interior_linf
        interior_linf_128 = [double]$fine.interior_linf
        interior_linf_order = [math]::Log(
            [double]$coarse.interior_linf /
            [double]$fine.interior_linf, 2.0)
        gmres_iterations_128 = [int]$fine.gmres_iterations
        gmres_residual_128 = [double]$fine.gmres_relative_residual
        edge_linf_128 = [double]$fine.incident_edge_discrepancy_linf
    }
}
$audit | Format-Table -AutoSize
```

Expected: six finite audited rows. Preserve the exact output for the
result report.

- [ ] **Step 5: Write the result report**

Create
`docs/superpowers/results/2026-07-26-3d-neumann-edge-cauchy-n128-forced.md`
with:

- exact commit, build command, forced command, wall time, and process exit;
- a six-row table containing the independently recomputed fields above;
- per-pose baseline-to-augmented GMRES and error ratios at `N=128`;
- all `64 -> 128` internal `Linf` orders;
- acceptance.csv statuses and a statement that forcing did not change
  coarse adoption;
- any incomplete pair or failed metric without suppression.

- [ ] **Step 6: Final verification and commit**

Run:

```powershell
cmake --build build --config Release -- /m:1
.\build\apps\Release\neumann_edge_cauchy_study_3d_test.exe
.\build\apps\Release\neumann_edge_augmented_cauchy_3d_test.exe
git diff --check
git status --short
```

Expected: build and tests exit `0`; `git diff --check` is clean; only
the intended result report is uncommitted.

Commit:

```powershell
git add -- `
  docs/superpowers/results/2026-07-26-3d-neumann-edge-cauchy-n128-forced.md
git commit -m "docs: report forced Neumann N128 evidence"
```
