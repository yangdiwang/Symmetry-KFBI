# Dirichlet Dual Formulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Preserve the stable normal-jump first-kind Dirichlet solver, add a selectable value-jump second-kind solver, and compare both on all four geometries through N=512.

**Architecture:** The shared apps executable remains the sole implementation. A DirichletFormulation enum selects which jump is the GMRES unknown; both paths reuse PythonCompatibleHarmonicJet2D, spread, restrict, trace evaluation, and reporting. StudyResult carries the formulation as a grouping key so orders never cross equations.

**Tech Stack:** C++17, Eigen, existing KFBI bulk solver and restarted GMRES, CMake, GitHub Actions shell checks, CSV.

## Global Constraints

- Default Dirichlet behavior is exactly normal_jump_first_kind.
- value_jump_second_kind solves R^- P(mu, 0) = g_D.
- KFBIM_PYJET_DIRICHLET_FORMULATION accepts normal_jump_first_kind, value_jump_second_kind, or compare.
- Neumann does not parse the Dirichlet formulation variable.
- Keep uniform_midpoint ratio 0.75, cubic_harmonic 4+3, biquadratic_quadratic_two_layer restrict, linear Taylor jump extension, and L-corner cubic harmonic 5+4 fit.
- Preserve all existing CSV columns and append dirichlet_formulation.
- Primary accuracy and order use the infinity norm.
- Preserve the existing uncommitted work in all four modified files.

## File map

- Modify .github/workflows/build-and-smoke.yml: default, second-kind, compare, and invalid-mode executable checks.
- Modify apps/neumann_harmonic_jet_python_compatible_2d.cpp: enum/parser, operators, dispatch, grouping, logging, CSV, filenames.
- Modify README.md: both equations and exact selection commands.
- Generate output/dirichlet_harmonic_jet_case_cpp_cubic_harmonic_spread_n4_d3_uniform_midpoint_dofs_biquadratic_quadratic_two_layer_restrict_compare_formulation.csv.

---

### Task 1: Add the failing second-kind smoke check

**Files:**
- Modify: .github/workflows/build-and-smoke.yml:66-85

**Interfaces:**
- Consumes: current Dirichlet executable.
- Produces: an executable contract requiring the second-kind label and finite converged result.

- [ ] **Step 1: Add this check before C++ changes**

~~~sh
second_kind_output="$(
  KFBIM_PYJET_DOF_MODE=uniform_midpoint \
  KFBIM_PYJET_RESTRICT_MODE=biquadratic_quadratic_two_layer \
  KFBIM_PYJET_DIRICHLET_FORMULATION=value_jump_second_kind \
    ./build/apps/dirichlet_harmonic_jet_python_compatible_2d circle 24
)"
printf '%s\n' "$second_kind_output"
printf '%s\n' "$second_kind_output" | grep -Fq \
  'dirichlet_formulation=value_jump_second_kind'
printf '%s\n' "$second_kind_output" | grep -Fq 'converged=yes'
second_boundary_res="$(printf '%s\n' "$second_kind_output" | sed -n 's/.* boundary_res=\([^ ]*\).*/\1/p' | head -n 1)"
second_linf="$(printf '%s\n' "$second_kind_output" | sed -n 's/.* Linf=\([^ ]*\).*/\1/p' | head -n 1)"
awk -v b="$second_boundary_res" -v e="$second_linf" \
  'BEGIN { exit !(b != "" && e != "" && b < 1e-6 && e < 1e-1) }'
~~~

- [ ] **Step 2: Verify RED locally**

~~~powershell
$env:KFBIM_PYJET_DIRICHLET_FORMULATION = 'value_jump_second_kind'
$out = .\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe circle 24 | Out-String
if ($out -notmatch 'dirichlet_formulation=value_jump_second_kind') {
    throw 'expected RED: formulation dispatch is not implemented'
}
~~~

Expected: the assertion throws because the current executable ignores the variable and prints no formulation label.

### Task 2: Implement parser, two operators, and solve dispatch

**Files:**
- Modify: apps/neumann_harmonic_jet_python_compatible_2d.cpp:49-67
- Modify: apps/neumann_harmonic_jet_python_compatible_2d.cpp:2103-2124
- Modify: apps/neumann_harmonic_jet_python_compatible_2d.cpp:2126-2423

**Interfaces:**
- Produces: enum class DirichletFormulation { NormalJumpFirstKind, ValueJumpSecondKind }.
- Produces: dirichlet_formulation_name(DirichletFormulation) returning std::string.
- Produces: parse_dirichlet_formulations(const std::string&) returning std::vector<DirichletFormulation>.
- Produces: run_one(BvpKind, DirichletFormulation, GeometryKind, ...) returning StudyResult.

- [ ] **Step 1: Add enum and strict parser**

~~~cpp
enum class DirichletFormulation {
    NormalJumpFirstKind,
    ValueJumpSecondKind
};

std::string dirichlet_formulation_name(DirichletFormulation formulation)
{
    return formulation == DirichletFormulation::ValueJumpSecondKind
        ? "value_jump_second_kind" : "normal_jump_first_kind";
}

std::vector<DirichletFormulation> parse_dirichlet_formulations(
    const std::string& name)
{
    if (name == "normal_jump_first_kind")
        return {DirichletFormulation::NormalJumpFirstKind};
    if (name == "value_jump_second_kind")
        return {DirichletFormulation::ValueJumpSecondKind};
    if (name == "compare")
        return {DirichletFormulation::NormalJumpFirstKind,
                DirichletFormulation::ValueJumpSecondKind};
    throw std::invalid_argument(
        "KFBIM_PYJET_DIRICHLET_FORMULATION must be "
        "normal_jump_first_kind, value_jump_second_kind, or compare");
}
~~~

- [ ] **Step 2: Split operators by unknown jump**

Rename InteriorDirichletTraceOperator to InteriorDirichletNormalJumpOperator without changing apply. Add:

~~~cpp
class InteriorDirichletValueJumpOperator final : public IKFBIOperator {
public:
    explicit InteriorDirichletValueJumpOperator(
        const PythonCompatibleHarmonicJet2D& solver)
        : solver_(solver), zero_(Eigen::VectorXd::Zero(solver.size())) {}
    int problem_size() const override { return solver_.size(); }
    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        if (x.size() != problem_size())
            throw std::invalid_argument("Dirichlet value-jump input size mismatch");
        const Eigen::VectorXd potential = solver_.potential(x, zero_);
        y = solver_.interior_trace(potential, x, zero_);
    }
private:
    const PythonCompatibleHarmonicJet2D& solver_;
    Eigen::VectorXd zero_;
};
~~~

- [ ] **Step 3: Preserve old algebra and add second-kind algebra**

Add formulation after bvp in run_one. Keep the current normal-jump branch byte-for-byte. The value-jump branch is:

~~~cpp
InteriorDirichletValueJumpOperator op(solver);
const Eigen::VectorXd rhs = dirichlet_data;
density = Eigen::VectorXd::Zero(ns);
GMRES gmres(max_iter, tolerance, std::min(restart, ns));
iterations = gmres.solve(op, rhs, density);
converged = gmres.converged();
potential = solver.potential(density, zero);
boundary_residual = solver.interior_trace(
    potential, density, zero, true) - dirichlet_data;
raw_exterior = solver.exterior_trace(potential, density, zero);
Eigen::VectorXd applied;
op.apply(density, applied);
operator_residual_linf = inf_norm(applied - rhs);
~~~

- [ ] **Step 4: Make diagnostics formulation-aware**

Add std::string dirichlet_formulation to StudyResult. Use not_applicable for Neumann. For first kind retain density - normal_data. For second kind set density_linf and density_l2 to quiet_NaN, retain density_mean, set flux_mean to 0 because [u_n]=0, and keep trace_linf/trace_l2 equal to the actual interior boundary residual.

- [ ] **Step 5: Build**

~~~powershell
cmake --build build --config Release --target dirichlet_harmonic_jet_python_compatible_2d
~~~

Expected: successful Release target build.

### Task 3: Add selection, grouping, output identity, and compare

**Files:**
- Modify: apps/neumann_harmonic_jet_python_compatible_2d.cpp:2435-2675
- Modify: apps/neumann_harmonic_jet_python_compatible_2d.cpp:2699-2863

**Interfaces:**
- Consumes: Task 2 parser and StudyResult field.
- Produces: formulation-safe orders, logs, appended CSV field, and non-overwriting filename.

- [ ] **Step 1: Parse only for Dirichlet**

~~~cpp
const std::string formulation_selection =
    bvp == BvpKind::Dirichlet
    ? environment_string("KFBIM_PYJET_DIRICHLET_FORMULATION",
                         "normal_jump_first_kind")
    : "not_applicable";
const std::vector<DirichletFormulation> formulations =
    bvp == BvpKind::Dirichlet
    ? parse_dirichlet_formulations(formulation_selection)
    : std::vector<DirichletFormulation>{
          DirichletFormulation::NormalJumpFirstKind};
~~~

Print dirichlet_formulation_selection for Dirichlet. Add a formulation loop inside restrict, include the name in [run], and pass it to run_one.

- [ ] **Step 2: Group all cross-row calculations**

add_orders and print_summary filter by dirichlet_formulation. Add this to matches in both comparison printers:

~~~cpp
&& candidate.dirichlet_formulation == baseline.dirichlet_formulation
~~~

- [ ] **Step 3: Identify rows and append CSV field**

print_result includes dirichlet_formulation. The CSV header ends with:

~~~text
density_linf,density_l2,dirichlet_formulation
~~~

The row ends with:

~~~cpp
<< result.density_l2 << ',' << result.dirichlet_formulation << '\n';
~~~

- [ ] **Step 4: Suffix Dirichlet files**

After the dirichlet_ prefix, insert before .csv:

~~~cpp
const std::size_t extension = csv_name.rfind(".csv");
csv_name.insert(extension, "_" + formulation_selection + "_formulation");
~~~

- [ ] **Step 5: Verify GREEN and invalid selection**

~~~powershell
$env:KFBIM_PYJET_DIRICHLET_FORMULATION = 'value_jump_second_kind'
$second = .\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe circle 24 | Out-String
if ($second -notmatch 'value_jump_second_kind' -or $second -notmatch 'converged=yes') { throw 'second-kind failure' }
Remove-Item Env:KFBIM_PYJET_DIRICHLET_FORMULATION
$default = .\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe circle 24 | Out-String
if ($default -notmatch 'normal_jump_first_kind' -or $default -notmatch 'converged=yes') { throw 'default regression' }
$env:KFBIM_PYJET_DIRICHLET_FORMULATION = 'compare'
$compare = .\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe circle 24 | Out-String
if ($compare -notmatch 'normal_jump_first_kind' -or $compare -notmatch 'value_jump_second_kind') { throw 'compare regression' }
$env:KFBIM_PYJET_DIRICHLET_FORMULATION = 'invalid'
& .\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe circle 24
if ($LASTEXITCODE -eq 0) { throw 'invalid selection must fail' }
~~~

Expected: valid cases pass; invalid exits 1 and lists the allowed values.

### Task 4: Complete CI and README

**Files:**
- Modify: .github/workflows/build-and-smoke.yml:66-105
- Modify: README.md:44-65 and environment list

**Interfaces:**
- Consumes: Tasks 2-3 executable behavior.
- Produces: maintained regression guard and documented equations.

- [ ] **Step 1: Strengthen workflow**

Keep Task 1 checks, assert default contains normal_jump_first_kind, run compare and grep both names, and require invalid mode to fail:

~~~sh
if KFBIM_PYJET_DIRICHLET_FORMULATION=invalid \
    ./build/apps/dirichlet_harmonic_jet_python_compatible_2d circle 24; then
  echo "invalid Dirichlet formulation unexpectedly succeeded" >&2
  exit 1
fi
~~~

- [ ] **Step 2: Document equations and commands**

~~~text
normal_jump_first_kind: [u]=g_D and
R^-P(0,q)=g_D-R^-P(g_D,0), q=[u_n].

value_jump_second_kind: [u_n]=0 and
R^-P(mu,0)=g_D, mu=[u], corresponding to 1/2 I + K_h.
~~~

Add KFBIM_PYJET_DIRICHLET_FORMULATION and PowerShell examples for all three values; state first kind is the default.

- [ ] **Step 3: Regression verification**

~~~powershell
cmake --build build --config Release --target neumann_harmonic_jet_python_compatible_2d dirichlet_harmonic_jet_python_compatible_2d
.\build\apps\Release\neumann_harmonic_jet_python_compatible_2d.exe ellipse 24
.\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe circle 24
ctest --test-dir build -C Release --output-on-failure
~~~

Expected: build succeeds, both runs converge, and CTest returns success with the repository's registered-test status.

### Task 5: Run and validate the N=512 comparison

**Files:**
- Generate: output/dirichlet_harmonic_jet_case_cpp_cubic_harmonic_spread_n4_d3_uniform_midpoint_dofs_biquadratic_quadratic_two_layer_restrict_compare_formulation.csv

**Interfaces:**
- Consumes: compare mode.
- Produces: 40 rows, 4 geometries x 5 levels x 2 formulations.

- [ ] **Step 1: Run agreed settings**

~~~powershell
$env:KFBIM_PYJET_DIRICHLET_FORMULATION = 'compare'
$env:KFBIM_PYJET_DOF_MODE = 'uniform_midpoint'
$env:KFBIM_PYJET_UNIFORM_DOF_RATIO = '0.75'
$env:KFBIM_PYJET_SPREAD_MODE = 'cubic_harmonic'
$env:KFBIM_PYJET_CUBIC_SPREAD_NEIGHBORS = '4'
$env:KFBIM_PYJET_CUBIC_SPREAD_DERIVATIVE_NEIGHBORS = '3'
$env:KFBIM_PYJET_RESTRICT_MODE = 'biquadratic_quadratic_two_layer'
$env:KFBIM_PYJET_CORNER_FIT_DEGREE = '3'
$env:KFBIM_PYJET_CORNER_FIT_NEIGHBORS = '5'
$env:KFBIM_PYJET_CORNER_FIT_DERIVATIVE_NEIGHBORS = '4'
$env:KFBIM_PYJET_MAX_ITER = '100'
$env:KFBIM_PYJET_RESTART = '80'
$env:KFBIM_PYJET_TOL = '2e-10'
.\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe all 32 64 128 256 512
~~~

Expected: 40 converged rows and compare-formulation CSV.

- [ ] **Step 2: Validate CSV**

~~~powershell
@'
import csv, math
from pathlib import Path
p = Path("output/dirichlet_harmonic_jet_case_cpp_cubic_harmonic_spread_n4_d3_uniform_midpoint_dofs_biquadratic_quadratic_two_layer_restrict_compare_formulation.csv")
rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
assert len(rows) == 40
assert {r["geometry"] for r in rows} == {"ellipse", "flower", "circle", "lshape"}
assert {r["dirichlet_formulation"] for r in rows} == {"normal_jump_first_kind", "value_jump_second_kind"}
assert all(r["converged"] == "1" for r in rows)
assert all(math.isfinite(float(r["global_linf"])) for r in rows)
assert all(math.isfinite(float(r["boundary_trace_res_linf"])) for r in rows)
print("validated", len(rows), "rows")
'@ | python -
~~~

Expected: validated 40 rows.

- [ ] **Step 3: Report direct Markdown tables**

For each geometry/formulation report N, DOFs, GMRES, Linf, infinity-norm order, total seconds, restrict condition maximum, spread condition maximum, and boundary residual. State whether first-kind reproduces its baseline and whether the L-shape second-kind corner density reduces convergence. Do not generate HTML.
