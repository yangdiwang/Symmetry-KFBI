<#
.SYNOPSIS
Runs the independent mixed-anchor P3/P2/3+3 route. Reference/reuse differ only
in the resource engine, not in the mathematical anchor selection policy.
.EXAMPLE
./scripts/validation/run_resource_reuse_3d.ps1 -Geometry cylinder -Level 32 -CompareEngines -PreprocessOnly -SkipBuild
.EXAMPLE
./scripts/validation/run_resource_reuse_3d.ps1 -Geometry cylinder -Level 32 -Engine reuse -SkipBuild
#>
[CmdletBinding()]
param(
    [ValidateSet('cylinder', 'l_prism', 'u_prism', 'torus')]
    [string] $Geometry = 'cylinder',
    # N=16 remains available, but some geometries do not have enough surface
    # samples to observe a cubic density basis. Keep that driver guard intact.
    [ValidateSet(16, 32, 64, 128)] [int] $Level = 32,
    [ValidateSet('reference', 'reuse')] [string] $Engine = 'reuse',
    [ValidateSet('both', 'neumann_only', 'dirichlet_normal_only')]
    [string] $Solve = 'both',
    [string] $RigidCase = 'baseline',
    # Four coefficients keep the default smoke run small; zero requests the
    # driver's automatic geometry/grid-dependent density resolution.
    [ValidateRange(0, 128)] [int] $DensityCoefficients = 4,
    [ValidateRange(1, 2000)] [int] $GmresMaxIterations = 80,
    [ValidateRange(1e-14, 1e-2)] [double] $GmresTolerance = 2e-10,
    [ValidateRange(1, 1440)] [int] $TimeoutMinutes = 20,
    [string] $BuildDirectory = 'build-3d',
    [string] $OutputDirectory = '',
    [switch] $CompareEngines,
    [switch] $DumpMatrices,
    [switch] $PreprocessOnly,
    [switch] $SkipBuild,
    [switch] $DryRun
)
$ErrorActionPreference = 'Stop'
if ($DensityCoefficients -gt 0 -and $DensityCoefficients -lt 4) {
    throw 'DensityCoefficients must be zero (automatic) or at least four.'
}
$repo = [IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$build = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $repo $BuildDirectory }
$exe = Join-Path $build 'apps/kfbi_topology_affine_exterior_trace_3d.exe'
if (-not $OutputDirectory) {
    $OutputDirectory = 'o/rr_' + (Get-Date -Format 'MMddHHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0, 4)
}
$runRoot = [IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repo $OutputDirectory }))
if (-not $runRoot.StartsWith($repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputDirectory must be a new directory strictly inside the repository.'
}
if (Test-Path -LiteralPath $runRoot) { throw "Refusing to overwrite existing results: $runRoot" }
$settings = [ordered]@{
    KFBIM_3D_SUPPORT_PATH = 'near_surface_anchor_mixed'
    KFBIM_3D_RESOURCE_ENGINE = $Engine
    KFBIM_3D_RESOURCE_CHECK_AB = [string][int]$CompareEngines.IsPresent
    KFBIM_3D_RESOURCE_DUMP = [string][int]$DumpMatrices.IsPresent
    KFBIM_3D_PREPROCESS_ONLY = [string][int]$PreprocessOnly.IsPresent
    KFBIM_3D_DENSITY_MODE = 'reduced_coefficients'
    KFBIM_3D_NEUMANN_TRACE_RESTRICT = 'q27_cover3_all_event_cauchy'
    KFBIM_3D_DIRICHLET_NORMAL_RESTRICT = 'q64_cover4_all_event_cauchy'
    KFBIM_3D_NEUMANN_EDGE_JUMP_JET = 'topology_affine_local_svd'
    KFBIM_3D_NEUMANN_TRACE_SAMPLING = 'panel_centers'
    KFBIM_3D_NEUMANN_DENSITY_COORDINATES = 'trace_mass'
    KFBIM_3D_NEUMANN_BORDER_SOLVER = 'mean_free_pivot_elimination'
    KFBIM_3D_NEUMANN_COMPATIBILITY = 'trace_border_legacy'
    KFBIM_3D_DIRICHLET_JUMP_SPACE = 'analytic_j0_affine_j1'
    KFBIM_3D_DIRICHLET_FEATURE_COUPLING = 'broken_sheets'
    KFBIM_3D_SOLVE_SELECTION = $Solve
    KFBIM_3D_RIGID_CASE = $RigidCase
    KFBIM_3D_GMRES_MAX_ITERATIONS = [string]$GmresMaxIterations
    KFBIM_3D_GMRES_TOLERANCE = $GmresTolerance.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
    KFBIM_3D_OUTPUT_ROOT = $runRoot
    OMP_NUM_THREADS = '1'
    OPENBLAS_NUM_THREADS = '1'
    MKL_NUM_THREADS = '1'
}
if ($DensityCoefficients -gt 0) {
    $settings['KFBIM_3D_DENSITY_COEFFICIENTS'] = [string]$DensityCoefficients
}
if ($DryRun) {
    [pscustomobject]@{ executable=$exe; arguments=@($Geometry, $Level); environment=$settings } | ConvertTo-Json -Depth 5
    return
}
if (-not $SkipBuild) {
    $savedPath = $env:PATH
    try {
        $env:PATH = 'C:\tools\msys64\mingw64\bin;' + $savedPath
        & cmake --build $build --target kfbi_topology_affine_exterior_trace_3d --parallel 2
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }
    } finally { $env:PATH = $savedPath }
}
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing executable: $exe" }
New-Item -ItemType Directory -Path $runRoot | Out-Null
$settings | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'environment.json') -Encoding UTF8
$start = New-Object Diagnostics.ProcessStartInfo
$start.FileName = $exe
$start.Arguments = "$Geometry $Level"
$start.WorkingDirectory = $repo
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
# Sanitize only the child environment. No parent/session/global env mutation.
foreach ($name in @($start.EnvironmentVariables.Keys)) {
    if ([string]$name -like 'KFBIM_3D_*') { $start.EnvironmentVariables.Remove($name) }
}
foreach ($entry in $settings.GetEnumerator()) { $start.EnvironmentVariables[$entry.Key] = [string]$entry.Value }
$start.EnvironmentVariables['PATH'] = 'C:\tools\msys64\mingw64\bin;' + $env:PATH
$process = New-Object Diagnostics.Process
$process.StartInfo = $start
$timer = [Diagnostics.Stopwatch]::StartNew()
$timedOut = $false
$exitCode = -1
$started = $false
$stdout = $null
$stderr = $null
$runError = $null
$cleanupErrors = New-Object 'System.Collections.Generic.List[string]'
try {
    if (-not $process.Start()) { throw 'Could not launch solver.' }
    $started = $true
    Write-Host "Started PID=$($process.Id) $Geometry N=$Level engine=$Engine output=$runRoot"
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    while (-not $process.WaitForExit(1000)) {
        if ($timer.Elapsed.TotalMinutes -ge $TimeoutMinutes) {
            # Only terminate the process represented by this Start() handle.
            # It may naturally exit between the wait, HasExited and Kill.
            if (-not $process.HasExited) {
                $timedOut = $true
                try { $process.Kill() }
                catch {
                    if ($process.HasExited) { $timedOut = $false }
                    else { throw }
                }
            }
            if (-not $process.WaitForExit(10000)) {
                throw 'Owned solver process did not exit after termination request.'
            }
            break
        }
    }
    $exitCode = $process.ExitCode
} catch {
    $runError = $_
} finally {
    if ($started) {
        try {
            # On an unexpected capture/wait exception, do not leave our own
            # solver running. Never enumerate processes or kill by reused PID.
            if (-not $process.HasExited) {
                try { $process.Kill() }
                catch { if (-not $process.HasExited) { throw } }
                if (-not $process.WaitForExit(10000)) {
                    throw 'Owned solver process is still running during cleanup.'
                }
            }
            if ($process.HasExited) { $exitCode = $process.ExitCode }
        } catch { $cleanupErrors.Add($_.Exception.Message) }
    }
    # Preserve stdout/stderr even when timeout termination or another operation
    # throws. Bounded task waits also protect against an inherited open pipe.
    foreach ($capture in @(@{Name='stdout'; Task=$stdout}, @{Name='stderr'; Task=$stderr})) {
        $logPath = Join-Path $runRoot ($capture.Name + '.log')
        try {
            if ($null -eq $capture.Task) {
                [IO.File]::WriteAllText($logPath, '')
            } elseif ($capture.Task.Wait(10000)) {
                [IO.File]::WriteAllText($logPath, $capture.Task.Result)
            } else {
                $message = $capture.Name + ' capture did not finish within the cleanup deadline.'
                $cleanupErrors.Add($message)
                [IO.File]::WriteAllText($logPath, '[runner] ' + $message)
            }
        } catch { $cleanupErrors.Add($capture.Name + ': ' + $_.Exception.Message) }
    }
    $timer.Stop()
    $process.Dispose()
    [pscustomobject]@{
        geometry=$Geometry; N=$Level; engine=$Engine; solve=$Solve
        compare_engines=$CompareEngines.IsPresent; preprocess_only=$PreprocessOnly.IsPresent
        exit_code=$exitCode; timed_out=$timedOut; elapsed_seconds=$timer.Elapsed.TotalSeconds
        runner_error=$(if ($null -ne $runError) { $runError.Exception.Message } else { $null })
        cleanup_errors=@($cleanupErrors.ToArray())
        executable=$exe; source_revision=(& git -C $repo rev-parse HEAD)
        working_tree_dirty=[bool](& git -C $repo status --porcelain)
        timing_scope='entire process including optional AB comparison, diagnostics and topology'
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'run.json') -Encoding UTF8
}
if ($null -ne $runError) { throw $runError }
if ($cleanupErrors.Count -gt 0) { throw ('Runner cleanup failed: ' + ($cleanupErrors -join '; ')) }
if ($timedOut) { throw "Solver timed out after $TimeoutMinutes minutes; logs retained in $runRoot" }
if ($exitCode -ne 0) { throw "Solver failed with exit code $exitCode; see $runRoot/stderr.log" }
Write-Host "Finished in $($timer.Elapsed.TotalSeconds) seconds. Results: $runRoot"
