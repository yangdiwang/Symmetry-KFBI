[CmdletBinding()]
param(
    [ValidateSet('cylinder','l','u')] [string[]] $Geometries = @('cylinder','l','u'),
    [int[]] $Levels = @(32,64),
    [ValidateSet('translate','rotate','rotate_same_translate','rotate_translate')] [string[]] $Transforms = @('rotate'),
    [ValidateSet('both','neumann','dirichlet')] [string] $Bvp = 'both',
    [ValidateSet('trace','event')] [string] $Policy = 'trace',
    [ValidateSet('python93','all')] [string] $EventMode = 'python93',
    [ValidateSet('on','off')] [string] $GridLines = 'on',
    [ValidateRange(1,120)] [int] $TimeoutMinutes = 20,
    [string] $BuildDirectory = 'build-3d',
    [string] $RunName = '',
    [switch] $CompareCache,
    [switch] $SkipBuild
)
$ErrorActionPreference = 'Stop'
$traceRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
foreach ($traceN in $Levels) { if ($traceN -notin @(32,64,128)) { throw 'Levels must be 32, 64 or 128.' } }
if (-not $Geometries.Count -or -not $Levels.Count -or -not $Transforms.Count) { throw 'Empty study matrix.' }
foreach ($traceSelection in @(@($Geometries),@($Levels),@($Transforms))) {
    if (@($traceSelection | Select-Object -Unique).Count -ne $traceSelection.Count) { throw 'Duplicate study selections.' }
}
if (-not $RunName) { $RunName = 'trace93_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,6) }
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple unique name.' }
$traceBuild = if ([IO.Path]::IsPathRooted($BuildDirectory)) {$BuildDirectory} else {Join-Path $traceRepo $BuildDirectory}
$traceExe = Join-Path $traceBuild 'apps/kfbi_trace93_study_3d.exe'
$traceRunRoot = Join-Path $PSScriptRoot ('results/' + $RunName)
if (Test-Path -LiteralPath $traceRunRoot) { throw "Refusing to overwrite $traceRunRoot" }
if (-not $SkipBuild) {
    $traceSavedPath = $env:PATH
    try {
        $env:PATH = 'C:\tools\msys64\mingw64\bin;' + $traceSavedPath
        & cmake --build $traceBuild --target kfbi_trace93_study_3d --parallel 2
        if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
    } finally { $env:PATH = $traceSavedPath }
}
if (-not (Test-Path -LiteralPath $traceExe)) { throw "Missing executable $traceExe" }
$traceReferencePath = Join-Path $PSScriptRoot 'python_reference_trace93.json'
$traceReference = Get-Content -LiteralPath $traceReferencePath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $traceRunRoot | Out-Null
$traceRecords = [Collections.Generic.List[object]]::new()
$traceRuns = [Collections.Generic.List[object]]::new()
$traceConfig = [ordered]@{geometries=$Geometries;levels=$Levels;transforms=$Transforms;bvp=$Bvp;policy=$Policy;event_mode=$EventMode;grid_lines=$GridLines;reference_sha256=$traceReference.source_sha256}
$traceConfig | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $traceRunRoot 'study_config.json') -Encoding UTF8
foreach ($traceN in ($Levels | Sort-Object)) {
    foreach ($traceGeometry in $Geometries) {
        foreach ($traceTransform in $Transforms) {
            $traceName = "${traceGeometry}_${traceTransform}_N${traceN}"
            $traceOut = Join-Path $traceRunRoot $traceName
            $tracePsi = [Diagnostics.ProcessStartInfo]::new()
            $tracePsi.FileName = $traceExe
            $tracePsi.WorkingDirectory = $traceRepo
            $tracePsi.UseShellExecute = $false
            $tracePsi.CreateNoWindow = $true
            $tracePsi.RedirectStandardOutput = $true
            $tracePsi.RedirectStandardError = $true
            $tracePsi.Arguments = "--geometry $traceGeometry --N $traceN --transform $traceTransform --bvp $Bvp --policy $Policy --event-mode $EventMode --grid-lines $GridLines --output `"$traceOut`""
            if ($CompareCache) { $tracePsi.Arguments += ' --compare-cache' }
            foreach ($traceKey in @($tracePsi.EnvironmentVariables.Keys)) {
                if ($traceKey -like 'KFBIM_3D_*' -or $traceKey -like 'KFBI_*') { $tracePsi.EnvironmentVariables.Remove($traceKey) }
            }
            $tracePsi.EnvironmentVariables['PATH'] = 'C:\tools\msys64\mingw64\bin;' + $env:PATH
            foreach ($traceKey in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) { $tracePsi.EnvironmentVariables[$traceKey] = '1' }
            $traceProcess = [Diagnostics.Process]::new()
            $traceProcess.StartInfo = $tracePsi
            $traceStarted = $false; $traceTimeout = $false; $traceCode = -1
            $traceTimer = [Diagnostics.Stopwatch]::StartNew()
            try {
                Write-Host "Starting $traceName ($Bvp)"
                $traceStarted = $traceProcess.Start()
                if (-not $traceStarted) { throw "Could not start $traceName" }
                $traceStdout = $traceProcess.StandardOutput.ReadToEndAsync()
                $traceStderr = $traceProcess.StandardError.ReadToEndAsync()
                while (-not $traceProcess.WaitForExit(1000)) {
                    if ($traceTimer.Elapsed.TotalMinutes -ge $TimeoutMinutes) {
                        $traceTimeout = $true
                        if (-not $traceProcess.HasExited) { $traceProcess.Kill() }
                        if (-not $traceProcess.WaitForExit(10000)) { throw 'Owned solver did not stop.' }
                        break
                    }
                }
                $traceCode = $traceProcess.ExitCode
                if (-not $traceStdout.Wait(10000) -or -not $traceStderr.Wait(10000)) { throw 'Output capture timed out.' }
                [IO.File]::WriteAllText((Join-Path $traceRunRoot "${traceName}_stdout.log"),$traceStdout.Result)
                [IO.File]::WriteAllText((Join-Path $traceRunRoot "${traceName}_stderr.log"),$traceStderr.Result)
                Write-Host $traceStdout.Result
                if ($traceStderr.Result) { Write-Host $traceStderr.Result }
            } finally {
                if ($traceStarted -and -not $traceProcess.HasExited) { $traceProcess.Kill(); [void]$traceProcess.WaitForExit(10000) }
                $traceTimer.Stop(); $traceProcess.Dispose()
            }
            $traceRuns.Add([pscustomobject]@{geometry=$traceGeometry;transform=$traceTransform;N=$traceN;exit_code=$traceCode;timed_out=$traceTimeout;process_seconds=$traceTimer.Elapsed.TotalSeconds})
            foreach ($traceKind in @('neumann','dirichlet')) {
                $traceResultPath = Join-Path $traceOut "$traceKind/result.json"
                if (-not (Test-Path -LiteralPath $traceResultPath)) { continue }
                $traceResult = Get-Content -LiteralPath $traceResultPath -Raw | ConvertFrom-Json
                $tracePrior = $traceRecords | Where-Object {$_.geometry -eq $traceGeometry -and $_.transform -eq $traceTransform -and $_.bvp -eq $traceKind -and $_.N -lt $traceN} | Sort-Object N | Select-Object -Last 1
                $traceOrder = $null
                if ($tracePrior -and $tracePrior.gmres_converged -and $traceResult.gmres_converged -and $tracePrior.interior_linf -gt 0 -and $traceResult.interior_linf -gt 0) {
                    $traceOrder = [Math]::Log($tracePrior.interior_linf/$traceResult.interior_linf)/[Math]::Log($traceN/$tracePrior.N)
                }
                $traceResult | Add-Member -NotePropertyName observed_order -NotePropertyValue $traceOrder
                $traceBaseline = $traceReference.results | Where-Object {$_.geometry -eq $traceGeometry -and $_.transform -eq $traceTransform -and $_.bvp -eq $traceKind -and $_.N -eq $traceN}
                if (@($traceBaseline).Count -ne 1) { throw "Reference is missing or ambiguous: $traceName $traceKind" }
                $traceCountDelta = if ($null -eq $traceBaseline.trace_points) {$null} else {$traceResult.trace_points-$traceBaseline.trace_points}
                $traceResult | Add-Member -NotePropertyName python_reference -NotePropertyValue ([pscustomobject]@{from_archive=$true;interior_linf=$traceBaseline.interior_linf;gmres=$traceBaseline.gmres;nred=$traceBaseline.nred;trace_points=$traceBaseline.trace_points;error_ratio=($traceResult.interior_linf/$traceBaseline.interior_linf);gmres_delta=($traceResult.gmres_iterations-$traceBaseline.gmres);dof_delta=($traceResult.reduced_dofs-$traceBaseline.nred);trace_count_delta=$traceCountDelta})
                $traceRecords.Add($traceResult)
            }
            [ordered]@{source_revision=(& git -C $traceRepo rev-parse HEAD);dirty=[bool](& git -C $traceRepo status --porcelain);configuration=$traceConfig;runs=@($traceRuns.ToArray());results=@($traceRecords.ToArray())} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $traceRunRoot 'summary.json') -Encoding UTF8
            if ($traceTimeout) { throw "Timed out; preserved results at $traceRunRoot" }
        }
    }
}
Write-Host "Completed study: $traceRunRoot/summary.json"
if (@($traceRuns | Where-Object {$_.exit_code -ne 0}).Count) { throw 'One or more cases failed; inspect retained diagnostics.' }
