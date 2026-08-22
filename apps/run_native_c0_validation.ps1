<#
.SYNOPSIS
Runs reproducible native-C0 Neumann A/B cases and writes one aggregate CSV.

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `
  "& { & 'apps/run_native_c0_validation.ps1' -Geometry @('l_prism') `
  -Level @(32) -RigidCase @('t_xyz_2') `
  -Variant @('current','feature_constrained') -RunLabel l32ab }"

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `
  "& { & 'apps/run_native_c0_validation.ps1' `
  -Geometry @('cylinder','l_prism','u_prism') -Level @(32,64,128) `
  -RigidCase @('baseline','rot_axis123_17deg_t_xyz_1') `
  -Variant @('current') -RunLabel convergence }"

Use -DryRun to inspect the complete environment without launching a solve.
The topology_affine variant selects the independent sparse-topology/local-SVD
executable and its shared-Q10 cubic trace route; existing variants keep the
legacy native executable.
#>
[CmdletBinding()]
param(
    [ValidateSet('cylinder', 'l_prism', 'u_prism')]
    [string[]] $Geometry = @('cylinder', 'l_prism'),

    [ValidateSet(16, 32, 64, 128)]
    [int[]] $Level = @(32),

    [ValidateSet(
        'baseline',
        'tx_p0137',
        'ty_m0083',
        'tz_p0061',
        't_xyz_1',
        't_xyz_2',
        'rot_axis123_17deg',
        'rot_axis123_17deg_t_xyz_1')]
    [string[]] $RigidCase = @('baseline'),

    [ValidateSet(
        'current', 'feature_constrained', 'edge_jump_jet', 'topology_affine')]
    [string[]] $Variant = @('current'),

    [ValidateRange(0, 128)]
    [int] $DensityCoefficients = 0,

    [ValidateRange(1, 1000)]
    [int] $GmresMaxIterations = 80,

    [ValidateRange(1.0e-14, 1.0e-2)]
    [double] $GmresTolerance = 2.0e-10,

    [string] $RunLabel = (Get-Date -Format 'yyyyMMdd_HHmmss'),

    [string] $BuildDirectory = 'build-3d',

    [switch] $SkipBuild,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$build = if ([IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory
} else {
    Join-Path $repo $BuildDirectory
}
$legacyExe = Join-Path $build 'apps/kfbi_native_c0_exterior_trace_3d.exe'
$topologyExe = Join-Path $build 'apps/kfbi_topology_affine_exterior_trace_3d.exe'
$outputRoot = Join-Path $repo 'output/kfbi_native_c0_exterior_trace_3d'
$topologyOutputRoot = Join-Path $repo 'output/kfbi_topology_affine_3d'
$mingwBin = 'C:\tools\msys64\mingw64\bin'

if (-not (Test-Path -LiteralPath $build)) {
    throw "Missing build directory: $build"
}
if (-not $SkipBuild -and -not $DryRun) {
    $targets = @()
    if ($Variant | Where-Object { $_ -ne 'topology_affine' }) {
        $targets += 'kfbi_native_c0_exterior_trace_3d'
    }
    if ($Variant -contains 'topology_affine') {
        $targets += 'kfbi_topology_affine_exterior_trace_3d'
    }
    & cmake --build $build --target @targets -- -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}
foreach ($requiredExe in @(
    if ($Variant | Where-Object { $_ -ne 'topology_affine' }) { $legacyExe }
    if ($Variant -contains 'topology_affine') { $topologyExe })) {
    if (-not (Test-Path -LiteralPath $requiredExe)) {
        throw "Missing executable: $requiredExe"
    }
}

# The repository executable is linked against the MinGW GMP/MPFR runtime.
# Put that runtime before Strawberry Perl's DLL directory to avoid 0xc0000139.
$savedPath = $env:Path
$env:Path = "$mingwBin;$savedPath"

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
    'KFBIM_3D_OUTPUT_TAG'
)
$savedEnvironment = @{}
foreach ($name in $managedVariables) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
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

function Get-ExpectedOutputDirectory(
    [string] $variantName,
    [string] $rigidName,
    [string] $tag) {
    if ($variantName -eq 'topology_affine') {
        $path = Join-Path $topologyOutputRoot 'tac_q10/nm_pc_tm_mf/topology_affine'
        if ($rigidName -ne 'baseline') {
            $rigidDirectory = if ($rigidName -eq 'rot_axis123_17deg_t_xyz_1') {
                'rigid_r17_t1'
            } else {
                "rigid_$rigidName"
            }
            $path = Join-Path $path $rigidDirectory
        }
        return Join-Path $path $tag
    }
    $path = Join-Path $outputRoot 'global_exterior_branch/restrict_v24_n12'
    if ($variantName -eq 'feature_constrained') {
        $path = Join-Path $path 'neumann_feature_constrained'
    }
    $path = Join-Path $path 'nm_pc_tm_proj'
    if ($variantName -eq 'edge_jump_jet') {
        $path = Join-Path $path 'edge_jj'
    }
    if ($rigidName -ne 'baseline') {
        $rigidDirectory = if ($rigidName -eq 'rot_axis123_17deg_t_xyz_1') {
            'rigid_r17_t1'
        } else {
            "rigid_$rigidName"
        }
        $path = Join-Path $path $rigidDirectory
    }
    return Join-Path $path $tag
}

$rows = [System.Collections.Generic.List[object]]::new()
try {
    foreach ($variantName in $Variant) {
        foreach ($rigidName in $RigidCase) {
            foreach ($geometryName in $Geometry) {
                # Diagnostic filenames are long and the project itself may live under
                # a long Windows path.  Keep the leaf tag deliberately below 16 chars.
                $safeLabel = ($RunLabel -replace '[^A-Za-z0-9]', '').ToLowerInvariant()
                if ($safeLabel.Length -gt 6) {
                    $safeLabel = $safeLabel.Substring(0, 6)
                }
                $variantCode = @{
                    current = 'a'
                    feature_constrained = 'b'
                    edge_jump_jet = 'c'
                    topology_affine = 'd'
                }[$variantName]
                $geometryCode = @{
                    cylinder = 'c'
                    l_prism = 'l'
                    u_prism = 'u'
                }[$geometryName]
                $rigidCode = @{
                    baseline = 'b'; tx_p0137 = 'x'; ty_m0083 = 'y'
                    tz_p0061 = 'z'; t_xyz_1 = '1'; t_xyz_2 = '2'
                    rot_axis123_17deg = 'r'
                    rot_axis123_17deg_t_xyz_1 = 'q'
                }[$rigidName]
                $tag = "ab${safeLabel}${variantCode}${geometryCode}${rigidCode}"
                $settings = @{
                    KFBIM_3D_CAUCHY_POLICY = 'g1_nearest'
                    KFBIM_3D_CAUCHY_VALUE_COUNT = '48'
                    KFBIM_3D_CAUCHY_NORMAL_COUNT = '28'
                    KFBIM_3D_FEATURE_TRACE_FIT = 'geometric_feature'
                    KFBIM_3D_GMRES_MAX_ITERATIONS = $GmresMaxIterations
                    KFBIM_3D_GMRES_TOLERANCE = $GmresTolerance.ToString(
                        'R', [Globalization.CultureInfo]::InvariantCulture)
                    KFBIM_3D_NEUMANN_COMPATIBILITY = 'operator_flux'
                    KFBIM_3D_NEUMANN_TRACE_RESTRICT =
                        'global_cubic_exterior_branch_crossing_owner'
                    KFBIM_3D_NEUMANN_RESTRICT_VALUE_COUNT = '24'
                    KFBIM_3D_NEUMANN_RESTRICT_NORMAL_COUNT = '12'
                    KFBIM_3D_NEUMANN_RESTRICT_VALUE_JET = if (
                        $variantName -eq 'feature_constrained') {
                        'feature_constrained'
                    } else {
                        'c0_one_sided'
                    }
                    KFBIM_3D_NEUMANN_DENSITY_COORDINATES = 'trace_mass'
                    KFBIM_3D_NEUMANN_BORDER_SOLVER = 'projected_elimination'
                    KFBIM_3D_NEUMANN_TRACE_SAMPLING = 'panel_centers'
                    KFBIM_3D_NEUMANN_EDGE_JUMP_JET = if (
                        $variantName -eq 'edge_jump_jet') {
                        'strong_feature_mortar'
                    } else {
                        'disabled'
                    }
                    KFBIM_3D_SOLVE_SELECTION = 'neumann_only'
                    KFBIM_3D_DENSITY_MODE = 'reduced_coefficients'
                    KFBIM_3D_RIGID_CASE = $rigidName
                    KFBIM_3D_OUTPUT_TAG = $tag
                }
                if ($variantName -eq 'topology_affine') {
                    $settings.KFBIM_3D_NEUMANN_COMPATIBILITY =
                        'trace_border_legacy'
                    $settings.KFBIM_3D_NEUMANN_TRACE_RESTRICT =
                        'shared_q10_cubic_gridline_cauchy'
                    $settings.KFBIM_3D_NEUMANN_BORDER_SOLVER =
                        'mean_free_pivot_elimination'
                    $settings.KFBIM_3D_NEUMANN_EDGE_JUMP_JET =
                        'topology_affine_local_svd'
                }
                if ($DensityCoefficients -gt 0) {
                    $settings.KFBIM_3D_DENSITY_COEFFICIENTS =
                        [string] $DensityCoefficients
                }
                Set-ProcessEnvironment $settings

                $arguments = @($geometryName) + ($Level | ForEach-Object {
                    [string] $_
                })
                $exe = if ($variantName -eq 'topology_affine') {
                    $topologyExe
                } else {
                    $legacyExe
                }
                Write-Host "[$variantName/$rigidName/$geometryName] $exe $arguments"
                if ($DryRun) {
                    $settings.GetEnumerator() | Sort-Object Key |
                        ForEach-Object { Write-Host "  $($_.Key)=$($_.Value)" }
                    continue
                }

                & $exe @arguments
                if ($LASTEXITCODE -ne 0) {
                    throw "Run failed with exit code ${LASTEXITCODE}: $tag"
                }

                $caseDirectory = Get-ExpectedOutputDirectory `
                    $variantName $rigidName $tag
                $solveCsv = Join-Path $caseDirectory 'neumann_results.csv'
                $geometryCsv = Join-Path $caseDirectory 'geometry_readiness.csv'
                if (-not (Test-Path -LiteralPath $solveCsv) -or
                    -not (Test-Path -LiteralPath $geometryCsv)) {
                    throw "Expected result CSVs were not written under $caseDirectory"
                }
                $solveResults = Import-Csv -LiteralPath $solveCsv
                $geometryResults = Import-Csv -LiteralPath $geometryCsv
                foreach ($solve in $solveResults) {
                    $ready = $geometryResults | Where-Object {
                        $_.geometry -eq $solve.geometry -and $_.N -eq $solve.N
                    } | Select-Object -First 1
                    if ($null -eq $ready) {
                        throw "No geometry row matches $($solve.geometry) N=$($solve.N)"
                    }
                    $rows.Add([pscustomobject]@{
                        variant = $variantName
                        rigid_case = $rigidName
                        geometry = $solve.geometry
                        N = [int] $solve.N
                        ncoef = [int] $solve.density_coefficients_per_direction
                        dofs = [int] $solve.dofs
                        exact_crossings = [int] $ready.exact_crossings
                        orientation_integral = [double] $ready.density_orientation_integral
                        iterations = [int] $solve.iterations
                        converged = [bool] ([int] $solve.converged)
                        physical_converged = [bool] (
                            [int] $solve.physical_converged)
                        gmres_relative_residual = [double] $solve.gmres_relative_residual
                        operator_residual_linf = [double] $solve.operator_residual_linf
                        exterior_condition_linf = [double] $solve.exterior_condition_linf
                        edge_constraint_rank = [int] $solve.edge_constraint_rank
                        edge_constraint_residual_linf = [double] `
                            $solve.edge_constraint_residual_linf
                        edge_normal_fit_linf = [double] $solve.edge_normal_fit_linf
                        edge_target_projection_linf = [double] `
                            $solve.edge_target_projection_linf
                        interior_linf = [double] $solve.interior_linf
                        value_raw = [int] $ready.value_density_raw_coefficients
                        value_c0 = [int] $ready.value_density_c0_coefficients
                        value_reduced = [int] $ready.value_density_reduced_coefficients
                        value_c0_seams = [int] $ready.value_density_c0_seams
                        value_c1_seams = [int] $ready.value_density_c1_seams
                        output_directory = $caseDirectory
                    })
                }
            }
        }
    }
} finally {
    foreach ($name in $managedVariables) {
        [Environment]::SetEnvironmentVariable(
            $name, $savedEnvironment[$name], 'Process')
    }
    $env:Path = $savedPath
}

if (-not $DryRun) {
    $summary = Join-Path $outputRoot "native_c0_ab_${RunLabel}.csv"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $summary) |
        Out-Null
    $rows | Sort-Object variant, rigid_case, geometry, N |
        Export-Csv -LiteralPath $summary -NoTypeInformation
    $rows | Sort-Object variant, rigid_case, geometry, N | Format-Table `
        variant, rigid_case, geometry, N, ncoef, dofs, exact_crossings, `
        iterations, physical_converged, operator_residual_linf, `
        edge_constraint_residual_linf, interior_linf -AutoSize
    Write-Host "A/B summary: $summary"
}
