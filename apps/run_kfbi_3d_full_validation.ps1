<#
.SYNOPSIS
Runs the complete three-dimensional KFBI Dirichlet/Neumann validation matrix.

.DESCRIPTION
The default matrix contains seven geometries and eight rigid poses:

  * baseline and rot_axis123_17deg_t_xyz_1 at N=32,64,128;
  * the other six poses at N=32.

The topology/native executable covers torus, cylinder, l_prism, and u_prism.
The general-cap executable covers sphere, ellipsoid, and flower.  Every real
run sets KFBIM_3D_SOLVE_SELECTION=both.  Topology solver output is first
written below a short system-temporary root and then archived, while source
CSVs and stdout/stderr logs are preserved below
output/kfbi_3d_full_validation/<RunLabel>.  The two source schemas are
normalized into all_results.csv.

Use -DryRun to inspect the 24-process default plan without building or running
anything.  Use -SyntheticCsvTest to exercise both CSV adapters and the
group-wise convergence-order calculation without invoking a solver.
Use -PathSelfTest to verify the topology transient-output MAX_PATH budget.

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  apps/run_kfbi_3d_full_validation.ps1 -DryRun

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  apps/run_kfbi_3d_full_validation.ps1 -SyntheticCsvTest

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  apps/run_kfbi_3d_full_validation.ps1 -PathSelfTest

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  apps/run_kfbi_3d_full_validation.ps1 -RunLabel full_20260819 `
  -BuildDirectory build-topology-final -ContinueOnFailure

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  apps/run_kfbi_3d_full_validation.ps1 -RunLabel full_20260819 `
  -BuildDirectory build-topology-final -Resume -ContinueOnFailure
#>
[CmdletBinding()]
param(
    [ValidateSet(
        'torus', 'cylinder', 'l_prism', 'u_prism',
        'sphere', 'ellipsoid', 'flower')]
    [string[]] $Geometry = @(
        'torus', 'cylinder', 'l_prism', 'u_prism',
        'sphere', 'ellipsoid', 'flower'),

    [ValidateSet(
        'baseline',
        'tx_p0137',
        'ty_m0083',
        'tz_p0061',
        't_xyz_1',
        't_xyz_2',
        'rot_axis123_17deg',
        'rot_axis123_17deg_t_xyz_1')]
    [string[]] $RigidCase = @(
        'baseline',
        'tx_p0137',
        'ty_m0083',
        'tz_p0061',
        't_xyz_1',
        't_xyz_2',
        'rot_axis123_17deg',
        'rot_axis123_17deg_t_xyz_1'),

    [ValidateSet(16, 32, 64, 128)]
    [int[]] $Level = @(32, 64, 128),

    [ValidateSet(16, 32, 64, 128)]
    [int] $PhaseScanLevel = 32,

    [ValidateRange(0, 128)]
    [int] $DensityCoefficients = 0,

    [ValidateRange(1, 1000)]
    [int] $GmresMaxIterations = 80,

    [ValidateRange(1.0e-14, 1.0e-2)]
    [double] $GmresTolerance = 2.0e-10,

    [string] $RunLabel = (Get-Date -Format 'yyyyMMdd_HHmmss'),

    [string] $BuildDirectory = 'build-topology-final',

    [string] $TopologyTransientRoot = [IO.Path]::Combine(
        [IO.Path]::GetTempPath(), 'kfbim3v'),

    [switch] $SkipBuild,
    [switch] $Resume,
    [switch] $ContinueOnFailure,
    [switch] $DryRun,
    [switch] $SyntheticCsvTest,
    [switch] $PathSelfTest
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $repo 'output/kfbi_3d_full_validation'
$generalCapOutputRoot = Join-Path $repo 'output/kfbi_general_cap_exterior_trace_3d'
$mingwBin = 'C:\tools\msys64\mingw64\bin'
$invariantCulture = [Globalization.CultureInfo]::InvariantCulture

if ([string]::IsNullOrWhiteSpace($TopologyTransientRoot)) {
    throw 'TopologyTransientRoot must not be empty'
}
$topologySolverRoot = if ([IO.Path]::IsPathRooted($TopologyTransientRoot)) {
    [IO.Path]::GetFullPath($TopologyTransientRoot)
} else {
    [IO.Path]::GetFullPath((Join-Path $repo $TopologyTransientRoot))
}
# KFBIM_3D_OUTPUT_ROOT is the parent of the application-specific directory.
# Keeping this root outside the deeply nested repository is what removes the
# repository depth from every topology diagnostic filename.  This is a runner
# override only; invoking the executable directly still uses its legacy
# compiled/default output root.
$topologyOutputRoot = Join-Path $topologySolverRoot `
    'kfbi_topology_affine_3d'

$build = if ([IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory
} else {
    Join-Path $repo $BuildDirectory
}
$topologyExe = Join-Path $build 'apps/kfbi_topology_affine_exterior_trace_3d.exe'
$generalCapExe = Join-Path $build 'apps/kfbi_general_cap_exterior_trace_3d.exe'

$topologyGeometries = @('torus', 'cylinder', 'l_prism', 'u_prism')
$topologyAllGeometries = @('cylinder', 'l_prism', 'u_prism')
$generalCapGeometries = @('sphere', 'ellipsoid', 'flower')
$convergencePoses = @('baseline', 'rot_axis123_17deg_t_xyz_1')
$poseCodes = @{
    baseline = 'b'
    tx_p0137 = 'x'
    ty_m0083 = 'y'
    tz_p0061 = 'z'
    t_xyz_1 = '1'
    t_xyz_2 = '2'
    rot_axis123_17deg = 'r'
    rot_axis123_17deg_t_xyz_1 = 'q'
}

$managedVariables = @(
    'KFBIM_3D_CAUCHY_POLICY',
    'KFBIM_3D_CAUCHY_VALUE_COUNT',
    'KFBIM_3D_CAUCHY_NORMAL_COUNT',
    'KFBIM_3D_FEATURE_TRACE_FIT',
    'KFBIM_3D_GMRES_MAX_ITERATIONS',
    'KFBIM_3D_GMRES_TOLERANCE',
    'KFBIM_3D_NEUMANN_COMPATIBILITY',
    'KFBIM_3D_NEUMANN_TRACE_RESTRICT',
    'KFBIM_3D_NEUMANN_TRACE_PROJECTION',
    'KFBIM_3D_NEUMANN_RESTRICT_VALUE_COUNT',
    'KFBIM_3D_NEUMANN_RESTRICT_NORMAL_COUNT',
    'KFBIM_3D_NEUMANN_RESTRICT_VALUE_JET',
    'KFBIM_3D_NEUMANN_DENSITY_COORDINATES',
    'KFBIM_3D_NEUMANN_BORDER_SOLVER',
    'KFBIM_3D_NEUMANN_TRACE_SAMPLING',
    'KFBIM_3D_NEUMANN_EDGE_JUMP_JET',
    'KFBIM_3D_DIRICHLET_NORMAL_RESTRICT',
    'KFBIM_3D_DIRICHLET_JUMP_SPACE',
    'KFBIM_3D_DIRICHLET_FEATURE_COUPLING',
    'KFBIM_3D_VERIFY_CROSSING_PLAN',
    'KFBIM_3D_VERIFY_CROSSING_PLAN_ONLY',
    'KFBIM_3D_SOLVE_SELECTION',
    'KFBIM_3D_DENSITY_MODE',
    'KFBIM_3D_DENSITY_COEFFICIENTS',
    'KFBIM_3D_RIGID_CASE',
    'KFBIM_3D_OUTPUT_TAG',
    'KFBIM_3D_OUTPUT_ROOT'
)

$resultColumns = @(
    'case_id',
    'backend',
    'geometry',
    'formulation',
    'pose',
    'N',
    'h',
    'ncoef',
    'dofs',
    'pre_mean_dofs',
    'final_dofs',
    'trace_samples',
    'trace_final_dofs',
    'trace_oversampling_margin',
    'trace_oversampling_ratio',
    'iterations',
    'gmres_converged',
    'physical_converged',
    'gmres_relative_residual',
    'operator_residual_linf',
    'exterior_condition_linf',
    'raw_exterior_linf',
    'boundary_residual_linf',
    'density_linf',
    'density_weighted_mean',
    'interior_linf',
    'interior_order_linf',
    'zero_space_solver',
    'mean_free_pivot_index',
    'mean_free_pivot_moment',
    'mean_free_constant_projection_linf',
    'mean_free_intrinsic_mean',
    'seconds',
    'seconds_scope',
    'source_csv'
)

function Test-ContainsAll([string[]] $haystack, [string[]] $needles) {
    foreach ($needle in $needles) {
        if ($haystack -notcontains $needle) {
            return $false
        }
    }
    return $true
}

function Get-SafeRunLabel([string] $label) {
    $safe = ($label -replace '[^A-Za-z0-9_.-]', '_').Trim('._-')
    if ([string]::IsNullOrWhiteSpace($safe)) {
        throw 'RunLabel must contain at least one filename-safe character'
    }
    return $safe
}

function Get-ShortRunCode([string] $label) {
    $prefix = ($label -replace '[^A-Za-z0-9]', '').ToLowerInvariant()
    if ($prefix.Length -gt 4) {
        $prefix = $prefix.Substring(0, 4)
    }
    if ([string]::IsNullOrWhiteSpace($prefix)) {
        $prefix = 'run'
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($label)
        $digest = $sha.ComputeHash($bytes)
        # The readable four-character prefix plus a 48-bit digest keeps
        # concurrently running labels isolated without allowing an arbitrary
        # RunLabel length to leak into transient topology paths.
        $hash = -join ($digest[0..5] | ForEach-Object {
            $_.ToString('x2', $invariantCulture)
        })
    } finally {
        $sha.Dispose()
    }
    return $prefix + $hash
}

function Get-RigidDirectory([string] $pose) {
    if ($pose -eq 'baseline') {
        return $null
    }
    if ($pose -eq 'rot_axis123_17deg_t_xyz_1') {
        return 'rigid_r17_t1'
    }
    return "rigid_$pose"
}

function Get-TopologySourceDirectory([string] $pose, [string] $tag) {
    # With solve_selection=both the Dirichlet route contributes its own path
    # component before the compact Neumann mean-free mode component.
    $path = Join-Path $topologyOutputRoot `
        'tac_q10/dirichlet_shared_q10_cubic_gridline_cauchy/nm_pc_tm_mf/topology_affine'
    $rigidDirectory = Get-RigidDirectory $pose
    if ($null -ne $rigidDirectory) {
        $path = Join-Path $path $rigidDirectory
    }
    return Join-Path $path $tag
}

function Get-TopologyPathBudgetProbe {
    # Get-ShortRunCode is capped at 16 characters.  Together with the fixed
    # prefix, pose code, and two-character group code, this is the longest tag
    # New-RunCase can produce.
    $maximumTag = 'fvxxxx0123456789abqtn'
    $longestFilename = 'cylinder_N128_surface_panel_center_dofs.csv'
    $strongPath = Join-Path (
        Get-TopologySourceDirectory `
            'rot_axis123_17deg_t_xyz_1' $maximumTag) $longestFilename
    $catalogPaths = foreach ($pose in $poseCodes.Keys) {
        Join-Path (
            Get-TopologySourceDirectory ([string] $pose) $maximumTag) `
            $longestFilename
    }
    $worstPath = @($catalogPaths | Sort-Object Length -Descending)[0]
    return [pscustomobject]@{
        Root = $topologySolverRoot
        StrongPath = $strongPath
        StrongLength = $strongPath.Length
        WorstCatalogPath = $worstPath
        WorstCatalogLength = $worstPath.Length
        SafetyBudget = 240
        ClassicMaxPath = 260
    }
}

function Assert-TopologyTransientPathBudget {
    $probe = Get-TopologyPathBudgetProbe
    if ($probe.WorstCatalogLength -ge $probe.SafetyBudget) {
        throw (
            "Topology transient path budget failed: longest catalog path " +
            "has $($probe.WorstCatalogLength) characters (required < " +
            "$($probe.SafetyBudget)). Choose a shorter " +
            "-TopologyTransientRoot. Path: $($probe.WorstCatalogPath)")
    }
    $codeA = Get-ShortRunCode 'path_self_test_run_a'
    $codeB = Get-ShortRunCode 'path_self_test_run_b'
    if ($codeA.Length -ne 16 -or $codeB.Length -ne 16 -or
        $codeA -eq $codeB) {
        throw 'Topology output-tag isolation self-test failed'
    }
    return $probe
}

function Get-GeneralCapSourceDirectory([string] $pose) {
    return Join-Path $generalCapOutputRoot "pose_$pose/both"
}

function New-RunCase(
    [string] $backend,
    [string] $selector,
    [string[]] $expectedGeometries,
    [string] $pose,
    [int[]] $levels,
    [string] $groupCode,
    [string] $shortRunCode) {
    $poseCode = [string] $poseCodes[$pose]
    $caseId = "${poseCode}_${groupCode}"
    $tag = "fv${shortRunCode}${poseCode}${groupCode}"
    if ($backend -eq 'topology_native' -and $tag.Length -gt 21) {
        throw "Topology output tag exceeds its 21-character path budget: $tag"
    }
    $executable = if ($backend -eq 'topology_native') {
        $topologyExe
    } else {
        $generalCapExe
    }
    $arguments = [System.Collections.Generic.List[string]]::new()
    $arguments.Add($selector)
    foreach ($n in $levels) {
        if ($backend -eq 'general_cap' -and $DensityCoefficients -gt 0) {
            $arguments.Add("${n}:$DensityCoefficients")
        } else {
            $arguments.Add([string] $n)
        }
    }
    $sourceDirectory = if ($backend -eq 'topology_native') {
        Get-TopologySourceDirectory $pose $tag
    } else {
        Get-GeneralCapSourceDirectory $pose
    }
    return [pscustomobject]@{
        CaseId = $caseId
        Backend = $backend
        Selector = $selector
        ExpectedGeometries = @($expectedGeometries)
        Pose = $pose
        Levels = @($levels)
        GroupCode = $groupCode
        Tag = $tag
        Executable = $executable
        Arguments = @($arguments)
        SourceDirectory = $sourceDirectory
    }
}

function New-RunPlan {
    $cases = [System.Collections.Generic.List[object]]::new()
    $shortRunCode = Get-ShortRunCode $RunLabel
    foreach ($pose in @($RigidCase | Select-Object -Unique)) {
        $levels = if ($convergencePoses -contains $pose) {
            @($Level | Sort-Object -Unique)
        } else {
            @($PhaseScanLevel)
        }

        $selectedTopologyAll = @(
            $topologyAllGeometries | Where-Object { $Geometry -contains $_ })
        if (Test-ContainsAll $selectedTopologyAll $topologyAllGeometries) {
            $cases.Add((New-RunCase `
                'topology_native' 'all' $topologyAllGeometries `
                $pose $levels 'tn' $shortRunCode))
        } else {
            foreach ($geometryName in $selectedTopologyAll) {
                $groupCode = @{
                    cylinder = 'tc'
                    l_prism = 'tl'
                    u_prism = 'tu'
                }[$geometryName]
                $cases.Add((New-RunCase `
                    'topology_native' $geometryName @($geometryName) `
                    $pose $levels $groupCode $shortRunCode))
            }
        }
        if ($Geometry -contains 'torus') {
            $cases.Add((New-RunCase `
                'topology_native' 'torus' @('torus') `
                $pose $levels 'tt' $shortRunCode))
        }

        $selectedGeneral = @(
            $generalCapGeometries | Where-Object { $Geometry -contains $_ })
        if (Test-ContainsAll $selectedGeneral $generalCapGeometries) {
            $cases.Add((New-RunCase `
                'general_cap' 'all' $generalCapGeometries `
                $pose $levels 'gc' $shortRunCode))
        } else {
            foreach ($geometryName in $selectedGeneral) {
                $groupCode = @{
                    sphere = 'gs'
                    ellipsoid = 'ge'
                    flower = 'gf'
                }[$geometryName]
                $cases.Add((New-RunCase `
                    'general_cap' $geometryName @($geometryName) `
                    $pose $levels $groupCode $shortRunCode))
            }
        }
    }
    return @($cases)
}

function Get-CaseEnvironment($case) {
    $settings = @{
        KFBIM_3D_GMRES_MAX_ITERATIONS = [string] $GmresMaxIterations
        KFBIM_3D_GMRES_TOLERANCE = $GmresTolerance.ToString(
            'R', $invariantCulture)
        KFBIM_3D_SOLVE_SELECTION = 'both'
        KFBIM_3D_RIGID_CASE = $case.Pose
    }
    if ($case.Backend -eq 'topology_native') {
        $settings.KFBIM_3D_CAUCHY_POLICY = 'g1_nearest'
        $settings.KFBIM_3D_CAUCHY_VALUE_COUNT = '48'
        $settings.KFBIM_3D_CAUCHY_NORMAL_COUNT = '28'
        $settings.KFBIM_3D_FEATURE_TRACE_FIT = 'geometric_feature'
        $settings.KFBIM_3D_NEUMANN_COMPATIBILITY = 'trace_border_legacy'
        $settings.KFBIM_3D_NEUMANN_TRACE_RESTRICT =
            'shared_q10_cubic_gridline_cauchy'
        $settings.KFBIM_3D_NEUMANN_RESTRICT_VALUE_COUNT = '24'
        $settings.KFBIM_3D_NEUMANN_RESTRICT_NORMAL_COUNT = '12'
        $settings.KFBIM_3D_NEUMANN_RESTRICT_VALUE_JET = 'c0_one_sided'
        $settings.KFBIM_3D_NEUMANN_DENSITY_COORDINATES = 'trace_mass'
        $settings.KFBIM_3D_NEUMANN_BORDER_SOLVER =
            'mean_free_pivot_elimination'
        $settings.KFBIM_3D_NEUMANN_TRACE_SAMPLING = 'panel_centers'
        $settings.KFBIM_3D_NEUMANN_EDGE_JUMP_JET =
            'topology_affine_local_svd'
        $settings.KFBIM_3D_DIRICHLET_NORMAL_RESTRICT =
            'shared_q10_cubic_gridline_cauchy'
        $settings.KFBIM_3D_DIRICHLET_JUMP_SPACE =
            'analytic_j0_affine_j1'
        $settings.KFBIM_3D_DIRICHLET_FEATURE_COUPLING =
            'broken_sheets'
        $settings.KFBIM_3D_DENSITY_MODE = 'reduced_coefficients'
        # Remove repository depth from transient diagnostic paths.  The
        # executable's legacy default is unchanged because this override is
        # scoped to the child process and restored by this runner.
        $settings.KFBIM_3D_OUTPUT_ROOT = $topologySolverRoot
        $settings.KFBIM_3D_OUTPUT_TAG = $case.Tag
        if ($DensityCoefficients -gt 0) {
            $settings.KFBIM_3D_DENSITY_COEFFICIENTS =
                [string] $DensityCoefficients
        }
    }
    return $settings
}

function Set-ProcessEnvironment([hashtable] $values) {
    foreach ($name in $managedVariables) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    foreach ($entry in $values.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            [string] $entry.Key, [string] $entry.Value, 'Process')
    }
}

function Get-RowValue($row, [string] $name) {
    $property = $row.PSObject.Properties[$name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-FirstRowValue($row, [string[]] $names) {
    foreach ($name in $names) {
        $value = Get-RowValue $row $name
        if ($null -ne $value -and
            -not [string]::IsNullOrWhiteSpace([string] $value)) {
            return $value
        }
    }
    return $null
}

function Convert-ToDoubleOrNaN($raw) {
    if ($null -eq $raw -or [string]::IsNullOrWhiteSpace([string] $raw)) {
        return [double]::NaN
    }
    $text = ([string] $raw).Trim()
    if ($text -match '^[+-]?(nan|inf(inity)?)$') {
        return [double]::NaN
    }
    $value = 0.0
    if (-not [double]::TryParse(
            $text,
            [Globalization.NumberStyles]::Float,
            $invariantCulture,
            [ref] $value)) {
        throw "Could not parse floating-point value '$text'"
    }
    return $value
}

function Convert-ToInt($raw) {
    if ($null -eq $raw -or [string]::IsNullOrWhiteSpace([string] $raw)) {
        return 0
    }
    return [int]::Parse([string] $raw, $invariantCulture)
}

function Convert-ToNullableInt($raw) {
    if ($null -eq $raw -or [string]::IsNullOrWhiteSpace([string] $raw)) {
        return $null
    }
    return [int]::Parse([string] $raw, $invariantCulture)
}

function Convert-ToNullableBool($raw) {
    if ($null -eq $raw -or [string]::IsNullOrWhiteSpace([string] $raw)) {
        return $null
    }
    $text = ([string] $raw).Trim().ToLowerInvariant()
    if ($text -eq '1' -or $text -eq 'true') {
        return $true
    }
    if ($text -eq '0' -or $text -eq 'false') {
        return $false
    }
    throw "Could not parse Boolean value '$raw'"
}

function Test-FiniteDouble($raw) {
    $value = [double] $raw
    return -not [double]::IsNaN($value) -and
           -not [double]::IsInfinity($value)
}

function Test-RowHasColumn($row, [string] $name) {
    return $null -ne $row.PSObject.Properties[$name]
}

function Assert-TraceOversamplingDiagnostics(
    [string] $context,
    [int] $samples,
    [int] $traceFinalDofs,
    [int] $margin,
    [double] $ratio,
    $normalizedFinalDofs) {
    if ($samples -le 0 -or $traceFinalDofs -le 0) {
        throw (
            "$context has non-positive trace dimensions: " +
            "samples=$samples final_dofs=$traceFinalDofs")
    }
    if ($samples -le $traceFinalDofs) {
        throw (
            "$context trace projection is not strictly oversampled: " +
            "samples=$samples final_dofs=$traceFinalDofs")
    }
    $expectedMargin = $samples - $traceFinalDofs
    if ($margin -ne $expectedMargin) {
        throw (
            "$context has inconsistent trace oversampling margin: " +
            "source=$margin expected=$expectedMargin")
    }
    $expectedRatio = [double] $samples / [double] $traceFinalDofs
    if ([double]::IsNaN($ratio) -or [double]::IsInfinity($ratio) -or
        $ratio -le 1.0) {
        throw "$context has invalid trace oversampling ratio: $ratio"
    }
    $ratioTolerance = 1.0e-12 * [Math]::Max(1.0, [Math]::Abs($expectedRatio))
    if ([Math]::Abs($ratio - $expectedRatio) -gt $ratioTolerance) {
        throw (
            "$context has inconsistent trace oversampling ratio: " +
            "source=$ratio expected=$expectedRatio")
    }
    if ($null -ne $normalizedFinalDofs -and
        [int] $normalizedFinalDofs -ne $traceFinalDofs) {
        throw (
            "$context trace final DOFs disagree with normalized final_dofs: " +
            "trace=$traceFinalDofs normalized=$normalizedFinalDofs")
    }
}

function Get-TopologyTraceOversamplingDiagnostics(
    $row,
    $normalizedFinalDofs,
    [string] $context) {
    $sourceColumns = @(
        'topology_trace_samples',
        'topology_trace_final_dofs',
        'topology_trace_oversampling_margin',
        'topology_trace_oversampling_ratio')
    $presentCount = @($sourceColumns | Where-Object {
        Test-RowHasColumn $row $_
    }).Count
    if ($presentCount -eq 0) {
        throw (
            "$context is missing all required topology trace " +
            "oversampling columns")
    }
    if ($presentCount -ne $sourceColumns.Count) {
        throw (
            "$context has an incomplete current topology trace schema: " +
            "$presentCount/$($sourceColumns.Count) columns")
    }
    foreach ($name in $sourceColumns) {
        if ([string]::IsNullOrWhiteSpace(
                [string] (Get-RowValue $row $name))) {
            throw "$context has a blank current topology trace field '$name'"
        }
    }
    $samples = Convert-ToInt (Get-RowValue $row 'topology_trace_samples')
    $traceFinalDofs = Convert-ToInt (
        Get-RowValue $row 'topology_trace_final_dofs')
    $margin = Convert-ToInt (
        Get-RowValue $row 'topology_trace_oversampling_margin')
    $ratio = Convert-ToDoubleOrNaN (
        Get-RowValue $row 'topology_trace_oversampling_ratio')
    Assert-TraceOversamplingDiagnostics `
        $context $samples $traceFinalDofs $margin $ratio $normalizedFinalDofs
    return [pscustomobject]@{
        Samples = $samples
        FinalDofs = $traceFinalDofs
        Margin = $margin
        Ratio = $ratio
    }
}

function Get-GeneralCapTraceOversamplingDiagnostics(
    $row,
    $normalizedFinalDofs,
    [string] $context) {
    if (-not (Test-RowHasColumn $row 'trace_points') -or
        [string]::IsNullOrWhiteSpace(
            [string] (Get-RowValue $row 'trace_points'))) {
        throw "$context is missing required general-cap trace_points"
    }
    if ($null -eq $normalizedFinalDofs) {
        throw "$context is missing required general-cap final_dofs"
    }
    $samples = Convert-ToInt (Get-RowValue $row 'trace_points')
    $traceFinalDofs = [int] $normalizedFinalDofs
    $margin = $samples - $traceFinalDofs
    $ratio = [double] $samples / [double] $traceFinalDofs
    Assert-TraceOversamplingDiagnostics `
        $context $samples $traceFinalDofs $margin $ratio $normalizedFinalDofs
    return [pscustomobject]@{
        Samples = $samples
        FinalDofs = $traceFinalDofs
        Margin = $margin
        Ratio = $ratio
    }
}

function Get-SourceLabel([string] $path) {
    if ([string]::IsNullOrWhiteSpace($path)) {
        return ''
    }
    $full = [IO.Path]::GetFullPath($path)
    $repoBase = [Text.RegularExpressions.Regex]::Replace(
        [IO.Path]::GetFullPath($repo), '[\\/]+$', '')
    if ($full.Equals(
            $repoBase, [StringComparison]::OrdinalIgnoreCase)) {
        return '.'
    }
    $repoPrefix = $repoBase + [IO.Path]::DirectorySeparatorChar
    if ($full.StartsWith(
            $repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($repoPrefix.Length).Replace('\', '/')
    }
    # Transient topology source paths intentionally live outside the repo.
    # Keep such paths absolute and normalize separators for portable CSV/JSON
    # labels; do not try to manufacture a repository-relative path.
    return $full.Replace('\', '/')
}

function Convert-TopologyArchive($case, [string] $archiveDirectory) {
    $results = [System.Collections.Generic.List[object]]::new()
    $files = @(
        @{ Name = 'neumann_results.csv'; Formulation = 'neumann' },
        @{ Name = 'dirichlet_normal_results.csv'; Formulation = 'dirichlet' })
    foreach ($item in $files) {
        $path = Join-Path $archiveDirectory $item.Name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing archived topology result: $path"
        }
        foreach ($row in (Import-Csv -LiteralPath $path)) {
            $sourceDofs = Convert-ToInt (Get-RowValue $row 'dofs')
            $preMeanDofs = if ($item.Formulation -eq 'neumann') {
                Convert-ToNullableInt (Get-FirstRowValue $row @(
                    'topology_pre_mean_coordinates',
                    'pre_mean_coordinates',
                    'pre_mean_dofs'))
            } else {
                Convert-ToNullableInt (Get-FirstRowValue $row @(
                    'topology_reduced_coordinates', 'dofs'))
            }
            $finalDofs = Convert-ToNullableInt (Get-FirstRowValue $row @(
                'topology_trace_final_dofs',
                'topology_reduced_coordinates', 'final_coordinates', 'dofs'))
            $traceContext = (
                "topology case=$($case.CaseId) formulation=" +
                "$($item.Formulation) geometry=" +
                "$(Get-RowValue $row 'geometry') N=$(Get-RowValue $row 'N')")
            $traceDiagnostics = Get-TopologyTraceOversamplingDiagnostics `
                $row $finalDofs $traceContext
            $boundaryResidual = if ($item.Formulation -eq 'dirichlet') {
                Convert-ToDoubleOrNaN (Get-RowValue $row 'boundary_residual_linf')
            } else {
                [double]::NaN
            }
            $results.Add([pscustomobject][ordered]@{
                case_id = $case.CaseId
                backend = 'topology_native'
                geometry = [string] (Get-RowValue $row 'geometry')
                formulation = $item.Formulation
                pose = $case.Pose
                N = Convert-ToInt (Get-RowValue $row 'N')
                h = Convert-ToDoubleOrNaN (Get-RowValue $row 'h')
                ncoef = Convert-ToInt (
                    Get-RowValue $row 'density_coefficients_per_direction')
                dofs = $sourceDofs
                pre_mean_dofs = $preMeanDofs
                final_dofs = $finalDofs
                trace_samples = $traceDiagnostics.Samples
                trace_final_dofs = $traceDiagnostics.FinalDofs
                trace_oversampling_margin = $traceDiagnostics.Margin
                trace_oversampling_ratio = $traceDiagnostics.Ratio
                iterations = Convert-ToInt (Get-RowValue $row 'iterations')
                gmres_converged = Convert-ToNullableBool (
                    Get-RowValue $row 'converged')
                physical_converged = Convert-ToNullableBool (
                    Get-RowValue $row 'physical_converged')
                gmres_relative_residual = Convert-ToDoubleOrNaN (
                    Get-RowValue $row 'gmres_relative_residual')
                operator_residual_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row 'operator_residual_linf')
                exterior_condition_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row 'exterior_condition_linf')
                raw_exterior_linf = [double]::NaN
                boundary_residual_linf = $boundaryResidual
                density_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row 'density_linf')
                density_weighted_mean = Convert-ToDoubleOrNaN (
                    Get-RowValue $row 'density_weighted_mean')
                interior_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row 'interior_linf')
                interior_order_linf = [double]::NaN
                zero_space_solver = if ($item.Formulation -eq 'neumann') {
                    [string] (Get-RowValue $row 'border_solver')
                } else {
                    'not_applicable'
                }
                mean_free_pivot_index = if (
                    $item.Formulation -eq 'neumann') {
                    Convert-ToNullableInt (
                        Get-RowValue $row 'mean_free_pivot_index')
                } else { $null }
                mean_free_pivot_moment = if (
                    $item.Formulation -eq 'neumann') {
                    Convert-ToDoubleOrNaN (
                        Get-RowValue $row 'mean_free_pivot_moment')
                } else { [double]::NaN }
                mean_free_constant_projection_linf = if (
                    $item.Formulation -eq 'neumann') {
                    Convert-ToDoubleOrNaN (Get-RowValue $row `
                        'mean_free_constant_projection_linf')
                } else { [double]::NaN }
                mean_free_intrinsic_mean = if (
                    $item.Formulation -eq 'neumann') {
                    Convert-ToDoubleOrNaN (
                        Get-RowValue $row 'mean_free_intrinsic_mean')
                } else { [double]::NaN }
                seconds = Convert-ToDoubleOrNaN (Get-RowValue $row 'seconds')
                seconds_scope = 'formulation'
                source_csv = Get-SourceLabel $path
            })
        }
    }
    return @($results)
}

function Convert-GeneralCapArchive($case, [string] $archiveDirectory) {
    $path = Join-Path $archiveDirectory 'refinement.csv'
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing archived general-cap result: $path"
    }
    $results = [System.Collections.Generic.List[object]]::new()
    foreach ($row in (Import-Csv -LiteralPath $path)) {
        foreach ($formulation in @('neumann', 'dirichlet')) {
            if ($formulation -eq 'neumann') {
                $iterationsName = 'neumann_iterations'
                $convergedName = 'neumann_converged'
                $residualName = 'neumann_residual'
                $operatorName = if ($null -ne (
                    Get-RowValue $row 'neumann_operator_residual')) {
                    'neumann_operator_residual'
                } else {
                    'neumann_augmented_residual'
                }
                $exteriorName = 'neumann_exterior_trace'
                $rawExteriorName = 'neumann_raw_trace'
                $densityName = 'neumann_density'
                $interiorName = 'neumann_interior'
                $sourcePreMean = Get-FirstRowValue $row @(
                    'neumann_pre_mean_coordinates',
                    'neumann_pre_mean_dofs',
                    'neumann_pre_mean_reduced_dofs',
                    'pre_mean_coordinates',
                    'pre_mean_dofs',
                    'reduced_dofs')
                $sourceFinal = Get-FirstRowValue $row @(
                    'neumann_final_coordinates',
                    'neumann_final_dofs',
                    'neumann_reduced_dofs',
                    'final_coordinates',
                    'final_dofs',
                    'reduced_dofs')
            } else {
                $iterationsName = 'dirichlet_iterations'
                $convergedName = 'dirichlet_converged'
                $residualName = 'dirichlet_residual'
                # The projected final normal trace is the general-cap audit
                # closest to the topology projected operator residual.
                $operatorName = 'dirichlet_projected_trace'
                $exteriorName = 'dirichlet_exterior_normal'
                $rawExteriorName = 'dirichlet_raw_trace'
                $densityName = 'dirichlet_density'
                $interiorName = 'dirichlet_interior'
                $sourcePreMean = Get-RowValue $row 'reduced_dofs'
                $sourceFinal = Get-FirstRowValue $row @(
                    'dirichlet_final_coordinates',
                    'dirichlet_final_dofs',
                    'reduced_dofs')
            }
            $preMeanDofs = Convert-ToNullableInt $sourcePreMean
            $finalDofs = Convert-ToNullableInt $sourceFinal
            $traceContext = (
                "general-cap case=$($case.CaseId) formulation=$formulation " +
                "geometry=$(Get-RowValue $row 'shape') " +
                "N=$(Get-RowValue $row 'N')")
            $traceDiagnostics = Get-GeneralCapTraceOversamplingDiagnostics `
                $row $finalDofs $traceContext
            $results.Add([pscustomobject][ordered]@{
                case_id = $case.CaseId
                backend = 'general_cap'
                geometry = [string] (Get-RowValue $row 'shape')
                formulation = $formulation
                pose = [string] (Get-RowValue $row 'pose')
                N = Convert-ToInt (Get-RowValue $row 'N')
                h = Convert-ToDoubleOrNaN (Get-RowValue $row 'h')
                ncoef = Convert-ToInt (Get-RowValue $row 'ncoef')
                dofs = if ($null -ne $finalDofs) {
                    $finalDofs
                } else {
                    Convert-ToInt (Get-RowValue $row 'reduced_dofs')
                }
                pre_mean_dofs = $preMeanDofs
                final_dofs = $finalDofs
                trace_samples = $traceDiagnostics.Samples
                trace_final_dofs = $traceDiagnostics.FinalDofs
                trace_oversampling_margin = $traceDiagnostics.Margin
                trace_oversampling_ratio = $traceDiagnostics.Ratio
                iterations = Convert-ToInt (Get-RowValue $row $iterationsName)
                gmres_converged = Convert-ToNullableBool (
                    Get-RowValue $row $convergedName)
                # general-cap has no source physical_converged field.  Keep
                # it empty instead of manufacturing a pass/fail result.
                physical_converged = $null
                gmres_relative_residual = Convert-ToDoubleOrNaN (
                    Get-RowValue $row $residualName)
                operator_residual_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row $operatorName)
                exterior_condition_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row $exteriorName)
                raw_exterior_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row $rawExteriorName)
                boundary_residual_linf = [double]::NaN
                density_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row $densityName)
                density_weighted_mean = if ($formulation -eq 'neumann') {
                    Convert-ToDoubleOrNaN (Get-FirstRowValue $row @(
                        'neumann_density_weighted_mean',
                        'neumann_density_mean',
                        'density_weighted_mean',
                        'neumann_mean'))
                } else { [double]::NaN }
                interior_linf = Convert-ToDoubleOrNaN (
                    Get-RowValue $row $interiorName)
                interior_order_linf = [double]::NaN
                zero_space_solver = if ($formulation -eq 'neumann') {
                    $solver = Get-FirstRowValue $row @(
                        'neumann_mean_solver',
                        'neumann_border_solver',
                        'zero_space_solver')
                    if ($null -eq $solver) {
                        'mean_free_pivot_elimination'
                    }
                    else { [string] $solver }
                } else { 'not_applicable' }
                mean_free_pivot_index = if ($formulation -eq 'neumann') {
                    Convert-ToNullableInt (Get-FirstRowValue $row @(
                        'neumann_mean_free_pivot_index',
                        'neumann_mean_pivot',
                        'mean_free_pivot_index'))
                } else { $null }
                mean_free_pivot_moment = if ($formulation -eq 'neumann') {
                    Convert-ToDoubleOrNaN (Get-FirstRowValue $row @(
                        'neumann_mean_free_pivot_moment',
                        'neumann_mean_pivot_moment',
                        'mean_free_pivot_moment'))
                } else { [double]::NaN }
                mean_free_constant_projection_linf = if (
                    $formulation -eq 'neumann') {
                    Convert-ToDoubleOrNaN (Get-FirstRowValue $row @(
                        'neumann_mean_free_constant_projection_linf',
                        'neumann_P1_linf',
                        'mean_free_constant_projection_linf',
                        'neumann_constant_projection_linf'))
                } else { [double]::NaN }
                mean_free_intrinsic_mean = if ($formulation -eq 'neumann') {
                    Convert-ToDoubleOrNaN (Get-FirstRowValue $row @(
                        'neumann_mean_free_intrinsic_mean',
                        'mean_free_intrinsic_mean'))
                } else { [double]::NaN }
                # This source timer covers both formulations in a `both` run.
                seconds = Convert-ToDoubleOrNaN (
                    Get-RowValue $row 'solve_seconds')
                seconds_scope = 'both_formulations'
                source_csv = Get-SourceLabel $path
            })
        }
    }
    return @($results)
}

function Import-CaseResults($case, [string] $archiveDirectory) {
    if ($case.Backend -eq 'topology_native') {
        return @(Convert-TopologyArchive $case $archiveDirectory)
    }
    return @(Convert-GeneralCapArchive $case $archiveDirectory)
}

function Assert-ExpectedCaseRows($case, [object[]] $rows) {
    $expectedCount = 2 * $case.ExpectedGeometries.Count * $case.Levels.Count
    if ($rows.Count -ne $expectedCount) {
        throw "Case $($case.CaseId) produced $($rows.Count) normalized rows; expected $expectedCount"
    }
    foreach ($geometryName in $case.ExpectedGeometries) {
        foreach ($n in $case.Levels) {
            foreach ($formulation in @('neumann', 'dirichlet')) {
                $matches = @($rows | Where-Object {
                    $_.geometry -eq $geometryName -and
                    $_.N -eq $n -and
                    $_.formulation -eq $formulation
                })
                if ($matches.Count -ne 1) {
                    throw "Case $($case.CaseId) expected one $geometryName/$formulation/N=$n row; found $($matches.Count)"
                }
            }
        }
    }
}

function Add-RecomputedInteriorOrders([object[]] $rows) {
    $groups = $rows | Group-Object backend, geometry, formulation, pose
    foreach ($group in $groups) {
        $ordered = @($group.Group | Sort-Object N)
        $previous = $null
        foreach ($row in $ordered) {
            $row.interior_order_linf = [double]::NaN
            if ($null -ne $previous -and
                (Test-FiniteDouble $previous.interior_linf) -and
                (Test-FiniteDouble $row.interior_linf) -and
                $previous.interior_linf -gt 0.0 -and
                $row.interior_linf -gt 0.0 -and
                $previous.h -gt $row.h) {
                $row.interior_order_linf =
                    [Math]::Log($previous.interior_linf / $row.interior_linf) /
                    [Math]::Log($previous.h / $row.h)
            }
            $previous = $row
        }
    }
    return @($rows)
}

function Copy-CaseCsvs($case, [string] $archiveDirectory) {
    $sourceDirectory = [IO.Path]::GetFullPath(
        [string] $case.SourceDirectory)
    $archiveFull = [IO.Path]::GetFullPath($archiveDirectory)
    New-Item -ItemType Directory -Force -Path $archiveFull | Out-Null
    if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) {
        throw "Expected source directory was not written: $sourceDirectory"
    }
    $csvs = @(Get-ChildItem -LiteralPath $sourceDirectory -Filter '*.csv' `
        -File)
    if ($csvs.Count -eq 0) {
        throw "No source CSVs were written under $sourceDirectory"
    }
    foreach ($csv in $csvs) {
        Copy-Item -LiteralPath $csv.FullName -Destination (
            Join-Path $archiveFull $csv.Name) -Force
    }
}

function Write-CaseStatus(
    [string] $archiveDirectory,
    $case,
    [string] $status,
    [int] $exitCode,
    [string] $message) {
    New-Item -ItemType Directory -Force -Path $archiveDirectory | Out-Null
    $record = [ordered]@{
        case_id = $case.CaseId
        status = $status
        exit_code = $exitCode
        message = $message
        backend = $case.Backend
        selector = $case.Selector
        expected_geometries = @($case.ExpectedGeometries)
        pose = $case.Pose
        levels = @($case.Levels)
        command = "$($case.Executable) $($case.Arguments -join ' ')"
        output_tag = $case.Tag
        source_directory = Get-SourceLabel ([string] $case.SourceDirectory)
        completed_at = (Get-Date).ToString('o')
    }
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
        Join-Path $archiveDirectory 'case_status.json') -Encoding UTF8
}

function Test-CompletedArchive([string] $archiveDirectory) {
    $statusPath = Join-Path $archiveDirectory 'case_status.json'
    if (-not (Test-Path -LiteralPath $statusPath)) {
        return $false
    }
    try {
        $status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 |
            ConvertFrom-Json
        return $status.status -eq 'complete'
    } catch {
        return $false
    }
}

function Write-Manifest([string] $path, [object[]] $rows) {
    if ($rows.Count -eq 0) {
        'case_id,status,backend,selector,pose,levels,command,exit_code,message,stdout_log,stderr_log,archive_directory' |
            Set-Content -LiteralPath $path -Encoding UTF8
        return
    }
    $rows | Select-Object `
        case_id,status,backend,selector,pose,levels,command,exit_code,message,`
        stdout_log,stderr_log,archive_directory |
        Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
}

function Write-AllResults([string] $path, [object[]] $rows) {
    if ($rows.Count -eq 0) {
        ($resultColumns -join ',') | Set-Content -LiteralPath $path -Encoding UTF8
        return
    }
    $ordered = @($rows | Sort-Object `
        backend, geometry, formulation, pose, N)
    $ordered | Select-Object $resultColumns |
        Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
}

function Invoke-SyntheticCsvTest {
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) (
        'kfbim_full_validation_' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    try {
        $topologyArchive = Join-Path $tempRoot 'topology'
        $generalArchive = Join-Path $tempRoot 'general'
        New-Item -ItemType Directory -Force -Path $topologyArchive |
            Out-Null
        New-Item -ItemType Directory -Force -Path $generalArchive |
            Out-Null

        $topologyCommon = foreach ($n in @(32, 64, 128)) {
            $error = 0.04 * [Math]::Pow(0.25, [Math]::Log($n / 32, 2))
            [pscustomobject]@{
                geometry = 'u_prism'; N = $n; h = 3.0 / $n
                dofs = $n; density_coefficients_per_direction = 4
                topology_reduced_coordinates = $n
                topology_pre_mean_coordinates = $n + 1
                topology_trace_samples = $n + 16
                topology_trace_final_dofs = $n
                topology_trace_oversampling_margin = 16
                topology_trace_oversampling_ratio = [double] ($n + 16) / $n
                iterations = 7; converged = 1; physical_converged = 0
                gmres_relative_residual = 1.0e-11
                operator_residual_linf = 2.0e-11
                exterior_condition_linf = 3.0e-5
                boundary_residual_linf = 4.0e-6
                density_linf = 5.0e-4; interior_linf = $error
                seconds = 0.1
            }
        }
        $topologyCommon | Export-Csv -LiteralPath (
            Join-Path $topologyArchive 'neumann_results.csv') `
            -NoTypeInformation -Encoding UTF8
        $topologyCommon | ForEach-Object {
            $_.interior_linf = 2.0 * $_.interior_linf
            $_
        } | Export-Csv -LiteralPath (
            Join-Path $topologyArchive 'dirichlet_normal_results.csv') `
            -NoTypeInformation -Encoding UTF8

        $generalRows = foreach ($n in @(32, 64, 128)) {
            $error = 0.02 * [Math]::Pow(0.25, [Math]::Log($n / 32, 2))
            [pscustomobject]@{
                shape = 'sphere'; pose = 'baseline'; N = $n; h = 3.0 / $n
                ncoef = 5; reduced_dofs = 20; trace_points = 64
                solve_seconds = 0.2
                neumann_pre_mean_dofs = 20; neumann_final_dofs = 19
                neumann_mean_solver = 'mean_free_pivot_elimination'
                neumann_mean_pivot = 3
                neumann_mean_pivot_moment = 0.125
                neumann_P1_linf = 2.0e-15
                neumann_density_mean = 3.0e-16
                neumann_iterations = 8; neumann_converged = 1
                neumann_residual = 1.0e-11
                neumann_operator_residual = 1.8e-11
                neumann_augmented_residual = 2.0e-11
                neumann_exterior_trace = 3.0e-5
                neumann_raw_trace = 4.0e-5
                neumann_density = 5.0e-4; neumann_interior = $error
                dirichlet_iterations = 9; dirichlet_converged = 1
                dirichlet_residual = 1.5e-11
                dirichlet_projected_trace = 2.5e-11
                dirichlet_exterior_normal = 3.5e-5
                dirichlet_raw_trace = 4.5e-5
                dirichlet_density = 5.5e-4
                dirichlet_interior = 2.0 * $error
            }
        }
        $generalRows | Export-Csv -LiteralPath (
            Join-Path $generalArchive 'refinement.csv') `
            -NoTypeInformation -Encoding UTF8

        $topologyCase = [pscustomobject]@{
            CaseId = 'synthetic_topology'; Backend = 'topology_native'
            ExpectedGeometries = @('u_prism'); Pose = 'baseline'
            Levels = @(32, 64, 128)
        }
        $generalCase = [pscustomobject]@{
            CaseId = 'synthetic_general'; Backend = 'general_cap'
            ExpectedGeometries = @('sphere'); Pose = 'baseline'
            Levels = @(32, 64, 128)
        }
        $rows = @()
        $topologyRows = @(Import-CaseResults $topologyCase $topologyArchive)
        $generalNormalized = @(
            Import-CaseResults $generalCase $generalArchive)
        Assert-ExpectedCaseRows $topologyCase $topologyRows
        Assert-ExpectedCaseRows $generalCase $generalNormalized
        $rows += $topologyRows
        $rows += $generalNormalized
        $rows = @(Add-RecomputedInteriorOrders $rows)
        if ($rows.Count -ne 12) {
            throw "Synthetic test expected 12 rows; found $($rows.Count)"
        }
        $refined = @($rows | Where-Object { $_.N -gt 32 })
        foreach ($row in $refined) {
            if ([Math]::Abs($row.interior_order_linf - 2.0) -gt 1.0e-12) {
                throw "Synthetic order mismatch for $($row.backend)/$($row.formulation)/N=$($row.N): $($row.interior_order_linf)"
            }
        }
        $capPhysical = @($rows | Where-Object {
            $_.backend -eq 'general_cap' -and
            $null -ne $_.physical_converged
        })
        if ($capPhysical.Count -ne 0) {
            throw 'Synthetic general-cap physical_converged must remain empty'
        }
        $capNeumann = @($rows | Where-Object {
            $_.backend -eq 'general_cap' -and
            $_.formulation -eq 'neumann'
        })
        if (@($capNeumann | Where-Object {
                $_.pre_mean_dofs -ne 20 -or $_.final_dofs -ne 19 -or
                $_.dofs -ne 19 -or $_.mean_free_pivot_index -ne 3 -or
                $_.trace_samples -ne 64 -or
                $_.trace_final_dofs -ne 19 -or
                $_.trace_oversampling_margin -ne 45 -or
                [Math]::Abs(
                    $_.trace_oversampling_ratio - (64.0 / 19.0)) `
                    -gt 1.0e-12 -or
                $_.zero_space_solver -ne 'mean_free_pivot_elimination' -or
                [Math]::Abs($_.mean_free_constant_projection_linf - 2.0e-15) `
                    -gt 1.0e-24 -or
                [Math]::Abs($_.operator_residual_linf - 1.8e-11) `
                    -gt 1.0e-20
            }).Count -ne 0) {
            throw 'Synthetic general-cap mean-free diagnostics were not normalized'
        }
        $topologyCurrent = @($rows | Where-Object {
            $_.backend -eq 'topology_native'
        })
        if (@($topologyCurrent | Where-Object {
                $_.trace_samples -le $_.trace_final_dofs -or
                $_.trace_oversampling_margin -ne 16 -or
                [Math]::Abs(
                    $_.trace_oversampling_ratio -
                    ([double] $_.trace_samples / $_.trace_final_dofs)) `
                    -gt 1.0e-12
            }).Count -ne 0) {
            throw 'Synthetic current topology oversampling diagnostics were not normalized'
        }

        # Missing, partial, and non-oversampled topology diagnostics all fail
        # closed.  Historical CSVs must be regenerated before entering the
        # strict validation/report pipeline.
        $missingTopologyRejected = $false
        try {
            Get-TopologyTraceOversamplingDiagnostics `
                ([pscustomobject]@{ dofs = 7 }) 7 `
                'synthetic missing topology trace' | Out-Null
        } catch {
            if ($_.Exception.Message -like '*missing all required*') {
                $missingTopologyRejected = $true
            } else {
                throw
            }
        }
        if (-not $missingTopologyRejected) {
            throw 'Synthetic missing topology trace fields were not rejected'
        }
        $badTopologyRejected = $false
        try {
            Get-TopologyTraceOversamplingDiagnostics `
                ([pscustomobject]@{
                    topology_trace_samples = 12
                    topology_trace_final_dofs = 12
                    topology_trace_oversampling_margin = 0
                    topology_trace_oversampling_ratio = 1.0
                }) 12 'synthetic non-oversampled topology' | Out-Null
        } catch {
            if ($_.Exception.Message -like '*not strictly oversampled*') {
                $badTopologyRejected = $true
            } else {
                throw
            }
        }
        if (-not $badTopologyRejected) {
            throw 'Synthetic non-oversampled topology row was not rejected'
        }
        $badGeneralRejected = $false
        try {
            Get-GeneralCapTraceOversamplingDiagnostics `
                ([pscustomobject]@{ trace_points = 20 }) 20 `
                'synthetic non-oversampled general-cap' | Out-Null
        } catch {
            if ($_.Exception.Message -like '*not strictly oversampled*') {
                $badGeneralRejected = $true
            } else {
                throw
            }
        }
        if (-not $badGeneralRejected) {
            throw 'Synthetic non-oversampled general-cap row was not rejected'
        }
        $normalizedPath = Join-Path $tempRoot 'all_results.csv'
        Write-AllResults $normalizedPath $rows
        $roundTrip = @(Import-Csv -LiteralPath $normalizedPath)
        if ($roundTrip.Count -ne 12) {
            throw "Synthetic all_results.csv round trip found $($roundTrip.Count) rows"
        }
        foreach ($column in $resultColumns) {
            if ($roundTrip[0].PSObject.Properties.Name -notcontains $column) {
                throw "Synthetic all_results.csv is missing column '$column'"
            }
        }

        # Exercise the same external-source archive/resume route used by the
        # topology solver's system-temporary output.  Resume must depend only
        # on the durable archive, not on the transient source still existing.
        $resumeArchive = Join-Path $tempRoot 'resume_archive'
        $externalCase = [pscustomobject]@{
            SourceDirectory = $topologyArchive
        }
        Copy-CaseCsvs $externalCase $resumeArchive
        @{ status = 'complete' } | ConvertTo-Json | Set-Content -LiteralPath (
            Join-Path $resumeArchive 'case_status.json') -Encoding UTF8
        $externalLabel = Get-SourceLabel $topologyArchive
        $expectedExternalLabel = [IO.Path]::GetFullPath(
            $topologyArchive).Replace('\', '/')
        if ($externalLabel -ne $expectedExternalLabel) {
            throw (
                "External source label mismatch: '$externalLabel' != " +
                "'$expectedExternalLabel'")
        }
        Remove-Item -LiteralPath $topologyArchive -Recurse -Force
        if (-not (Test-CompletedArchive $resumeArchive)) {
            throw 'Synthetic external-source archive was not resumable'
        }
        $resumedRows = @(Import-CaseResults $topologyCase $resumeArchive)
        Assert-ExpectedCaseRows $topologyCase $resumedRows
        Write-Host (
            'Synthetic CSV normalization test: PASS ' +
            '(12 rows, p=2, strict trace oversampling, ' +
            'external source archive/resume)')
    } finally {
        $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        $resolvedTemp = [IO.Path]::GetFullPath($tempRoot)
        if ($resolvedTemp.StartsWith(
                $tempPrefix, [StringComparison]::OrdinalIgnoreCase) -and
            (Test-Path -LiteralPath $resolvedTemp)) {
            Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
        }
    }
}

$topologyPathProbe = Assert-TopologyTransientPathBudget

if ($PathSelfTest) {
    Write-Host 'Topology transient path self-test: PASS'
    Write-Host "  root=$($topologyPathProbe.Root)"
    Write-Host (
        "  strong_path_length=$($topologyPathProbe.StrongLength) " +
        "classic_headroom=" +
        "$($topologyPathProbe.ClassicMaxPath - $topologyPathProbe.StrongLength)")
    Write-Host (
        "  worst_catalog_path_length=" +
        "$($topologyPathProbe.WorstCatalogLength) " +
        "required_lt=$($topologyPathProbe.SafetyBudget)")
    Write-Host "  worst_catalog_path=$($topologyPathProbe.WorstCatalogPath)"
    exit 0
}

if ($SyntheticCsvTest) {
    Invoke-SyntheticCsvTest
    exit 0
}

$RunLabel = Get-SafeRunLabel $RunLabel
$plan = @(New-RunPlan)
if ($plan.Count -eq 0) {
    throw 'The selected geometry/pose filters produced an empty run plan'
}

if ($DryRun) {
    $geometryCases = 0
    foreach ($case in $plan) {
        $geometryCases += $case.ExpectedGeometries.Count * $case.Levels.Count
    }
    Write-Host "Dry run: processes=$($plan.Count) geometry-grid-pose_cases=$geometryCases GMRES_results=$($geometryCases * 2)"
    Write-Host (
        "Topology transient root: $topologySolverRoot " +
        "(strong_path=$($topologyPathProbe.StrongLength), " +
        "worst_catalog=$($topologyPathProbe.WorstCatalogLength), " +
        "required_lt=$($topologyPathProbe.SafetyBudget))")
    foreach ($case in $plan) {
        $settings = Get-CaseEnvironment $case
        Write-Host (
            "[$($case.CaseId)] backend=$($case.Backend) pose=$($case.Pose) " +
            "geometry=$($case.ExpectedGeometries -join '|') levels=$($case.Levels -join '|')")
        Write-Host "  command=$($case.Executable) $($case.Arguments -join ' ')"
        Write-Host "  source=$($case.SourceDirectory)"
        Write-Host (
            "  env: SOLVE_SELECTION=$($settings.KFBIM_3D_SOLVE_SELECTION) " +
            "GMRES_TOLERANCE=$($settings.KFBIM_3D_GMRES_TOLERANCE) " +
            "RIGID_CASE=$($settings.KFBIM_3D_RIGID_CASE)")
        if ($case.Backend -eq 'topology_native') {
            Write-Host (
                "  topology: BORDER_SOLVER=$($settings.KFBIM_3D_NEUMANN_BORDER_SOLVER) " +
                "NEUMANN_RESTRICT=$($settings.KFBIM_3D_NEUMANN_TRACE_RESTRICT) " +
                "DIRICHLET_RESTRICT=$($settings.KFBIM_3D_DIRICHLET_NORMAL_RESTRICT) " +
                "DIRICHLET_JUMP_SPACE=$($settings.KFBIM_3D_DIRICHLET_JUMP_SPACE) " +
                "DIRICHLET_FEATURE_COUPLING=$($settings.KFBIM_3D_DIRICHLET_FEATURE_COUPLING) " +
                "OUTPUT_ROOT=$($settings.KFBIM_3D_OUTPUT_ROOT) " +
                "OUTPUT_TAG=$($settings.KFBIM_3D_OUTPUT_TAG)")
        }
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $build)) {
    throw "Missing build directory: $build"
}

$selectedTopology = @($Geometry | Where-Object {
    $topologyGeometries -contains $_ })
$selectedGeneral = @($Geometry | Where-Object {
    $generalCapGeometries -contains $_ })
$reportDirectory = Join-Path $outputRoot $RunLabel
if ((Test-Path -LiteralPath $reportDirectory) -and -not $Resume) {
    throw "Report directory already exists; choose another RunLabel or use -Resume: $reportDirectory"
}
$savedPath = $env:Path
$savedEnvironment = @{}
foreach ($name in $managedVariables) {
    $savedEnvironment[$name] =
        [Environment]::GetEnvironmentVariable($name, 'Process')
}
$logsDirectory = Join-Path $reportDirectory 'logs'
$rawDirectory = Join-Path $reportDirectory 'raw'

$allRows = [System.Collections.Generic.List[object]]::new()
$manifestRows = [System.Collections.Generic.List[object]]::new()
$manifestPath = Join-Path $reportDirectory 'case_manifest.csv'
$allResultsPath = Join-Path $reportDirectory 'all_results.csv'

try {
    $env:Path = "$mingwBin;$savedPath"
    New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
    New-Item -ItemType Directory -Force -Path $logsDirectory | Out-Null
    New-Item -ItemType Directory -Force -Path $rawDirectory | Out-Null

    if (-not $SkipBuild) {
        $targets = [System.Collections.Generic.List[string]]::new()
        if ($selectedTopology.Count -gt 0) {
            $targets.Add('kfbi_topology_affine_exterior_trace_3d')
        }
        if ($selectedGeneral.Count -gt 0) {
            $targets.Add('kfbi_general_cap_exterior_trace_3d')
        }
        $buildArguments = @('--build', $build, '--target') + @($targets) +
            @('--', '-j', '1')
        & cmake @buildArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed with exit code $LASTEXITCODE"
        }
    }
    foreach ($requiredExe in @(
        if ($selectedTopology.Count -gt 0) { $topologyExe }
        if ($selectedGeneral.Count -gt 0) { $generalCapExe })) {
        if (-not (Test-Path -LiteralPath $requiredExe)) {
            throw "Missing executable: $requiredExe"
        }
    }

    foreach ($case in $plan) {
        $archiveDirectory = Join-Path $rawDirectory $case.CaseId
        $stdoutLog = Join-Path $logsDirectory "$($case.CaseId).stdout.log"
        $stderrLog = Join-Path $logsDirectory "$($case.CaseId).stderr.log"
        $manifestStatus = 'pending'
        $exitCode = -1
        $message = ''
        try {
            if ($Resume -and (Test-CompletedArchive $archiveDirectory)) {
                $caseRows = @(Import-CaseResults $case $archiveDirectory)
                Assert-ExpectedCaseRows $case $caseRows
                foreach ($row in $caseRows) {
                    $allRows.Add($row)
                }
                $manifestStatus = 'resumed'
                $exitCode = 0
                Write-Host "[resume/$($case.CaseId)] imported $($caseRows.Count) rows"
            } else {
                $settings = Get-CaseEnvironment $case
                Set-ProcessEnvironment $settings
                Write-Host (
                    "[run/$($case.CaseId)] $($case.Executable) " +
                    "$($case.Arguments -join ' ')")
                $process = Start-Process -FilePath $case.Executable `
                    -ArgumentList $case.Arguments -Wait -PassThru -NoNewWindow `
                    -WorkingDirectory $repo `
                    -RedirectStandardOutput $stdoutLog `
                    -RedirectStandardError $stderrLog
                $exitCode = $process.ExitCode
                if (Test-Path -LiteralPath $case.SourceDirectory) {
                    Copy-CaseCsvs $case $archiveDirectory
                }
                if ($exitCode -ne 0) {
                    throw "Solver process exited with code $exitCode"
                }
                $caseRows = @(Import-CaseResults $case $archiveDirectory)
                Assert-ExpectedCaseRows $case $caseRows
                foreach ($row in $caseRows) {
                    $allRows.Add($row)
                }
                Write-CaseStatus `
                    $archiveDirectory $case 'complete' $exitCode ''
                $manifestStatus = 'complete'
                Write-Host "[complete/$($case.CaseId)] archived $($caseRows.Count) rows"
            }
        } catch {
            $message = $_.Exception.Message
            $manifestStatus = 'failed'
            if (Test-Path -LiteralPath $case.SourceDirectory) {
                try {
                    Copy-CaseCsvs $case $archiveDirectory
                } catch {
                    $message += "; source archive failed: $($_.Exception.Message)"
                }
            }
            Write-CaseStatus `
                $archiveDirectory $case 'failed' $exitCode $message
            Write-Warning "[$($case.CaseId)] $message"
        }
        $manifestRows.Add([pscustomobject]@{
            case_id = $case.CaseId
            status = $manifestStatus
            backend = $case.Backend
            selector = $case.Selector
            pose = $case.Pose
            levels = $case.Levels -join '|'
            command = "$($case.Executable) $($case.Arguments -join ' ')"
            exit_code = $exitCode
            message = $message
            stdout_log = Get-SourceLabel $stdoutLog
            stderr_log = Get-SourceLabel $stderrLog
            archive_directory = Get-SourceLabel $archiveDirectory
        })
        Write-Manifest $manifestPath @($manifestRows)
        $withOrders = @(Add-RecomputedInteriorOrders @($allRows))
        Write-AllResults $allResultsPath $withOrders
        if ($manifestStatus -eq 'failed' -and -not $ContinueOnFailure) {
            throw "Validation stopped after failed case $($case.CaseId): $message"
        }
    }
} finally {
    foreach ($name in $managedVariables) {
        [Environment]::SetEnvironmentVariable(
            $name, $savedEnvironment[$name], 'Process')
    }
    $env:Path = $savedPath
}

$finalRows = @(Add-RecomputedInteriorOrders @($allRows))
Write-AllResults $allResultsPath $finalRows
Write-Manifest $manifestPath @($manifestRows)
$failedCases = @($manifestRows | Where-Object { $_.status -eq 'failed' })
Write-Host "Validation report: $reportDirectory"
Write-Host "  normalized_rows=$($finalRows.Count) failed_cases=$($failedCases.Count)"
Write-Host "  results=$allResultsPath"
if ($failedCases.Count -gt 0) {
    Write-Warning (
        "$($failedCases.Count) cases failed; rerun with -Resume after fixing the cause")
    exit 1
}
