[CmdletBinding()]
param(
    [int[]] $Levels = @(32,64),
    [ValidateSet('both','neumann','dirichlet')] [string] $Bvp = 'both',
    [ValidateSet('translate','rotate','rotate_same_translate','rotate_translate')] [string] $Transform = 'rotate',
    [ValidateSet('trace','event','trace-spread','event-spread')] [string] $Policy = 'trace',
    [ValidateSet('extended','physical')] [string] $Chart = 'extended',
    [ValidateSet('all','python')] [string] $EventMode = 'all',
    [ValidateSet('on','off')] [string] $GridLines = 'on',
    [ValidateRange(1,5000)] [int] $MaxIterations = 1000,
    [ValidateRange(1,120)] [int] $TimeoutMinutes = 20,
    [string] $BuildDirectory = 'build-3d',
    [string] $RunName = '',
    [switch] $CompareCache,
    [switch] $SkipBuild
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
if (-not $RunName) { $RunName = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,6) }
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple unique name.' }
foreach ($n in $Levels) {
    if ($n -notin @(32,64,128)) { throw 'Levels must be 32, 64 or 128.' }
}
if (($Levels | Select-Object -Unique).Count -ne $Levels.Count) { throw 'Duplicate levels are not allowed.' }
$build = if ([IO.Path]::IsPathRooted($BuildDirectory)) {$BuildDirectory} else {Join-Path $repo $BuildDirectory}
$exe = Join-Path $build 'apps/kfbi_trace_first_study_3d.exe'
$runRoot = Join-Path $PSScriptRoot ('results/' + $RunName)
if (Test-Path -LiteralPath $runRoot) { throw "Refusing to overwrite $runRoot" }
if (-not $SkipBuild) {
    $savedPath = $env:PATH
    try {
        $env:PATH = 'C:\tools\msys64\mingw64\bin;' + $savedPath
        & cmake --build $build --target kfbi_trace_first_study_3d --parallel 2
        if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
    } finally { $env:PATH = $savedPath }
}
if (-not (Test-Path -LiteralPath $exe)) {throw "Missing executable $exe"}
New-Item -ItemType Directory -Path $runRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'python_torus.json') -Destination (Join-Path $runRoot 'case.json')
$records = [Collections.Generic.List[object]]::new()
$runs = [Collections.Generic.List[object]]::new()
foreach ($n in ($Levels | Sort-Object)) {
    $out = Join-Path $runRoot "N$n"
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $exe
    $psi.Arguments = "--N $n --bvp $Bvp --transform $Transform --policy $Policy --chart $Chart --event-mode $EventMode --grid-lines $GridLines --max-iterations $MaxIterations --output `"$out`""
    if ($CompareCache) { $psi.Arguments += ' --compare-cache' }
    $psi.WorkingDirectory=$repo; $psi.UseShellExecute=$false; $psi.CreateNoWindow=$true
    $psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
    foreach ($key in @($psi.EnvironmentVariables.Keys)) {
        if ([string]$key -like 'KFBIM_3D_*') {$psi.EnvironmentVariables.Remove($key)}
    }
    $psi.EnvironmentVariables['PATH']='C:\tools\msys64\mingw64\bin;'+$env:PATH
    foreach ($key in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) {$psi.EnvironmentVariables[$key]='1'}
    $p=[Diagnostics.Process]::new();$p.StartInfo=$psi
    $timer=[Diagnostics.Stopwatch]::StartNew();$timeout=$false;$code=-1;$started=$false
    try {
        if(-not $p.Start()){throw 'Solver launch failed'}
        $started=$true
        $stdout=$p.StandardOutput.ReadToEndAsync();$stderr=$p.StandardError.ReadToEndAsync()
        Write-Host "Started N=$n PID=$($p.Id) output=$out"
        while(-not $p.WaitForExit(1000)) {
            if($timer.Elapsed.TotalMinutes -ge $TimeoutMinutes) {
                $timeout=$true
                if(-not $p.HasExited){$p.Kill()}
                if(-not $p.WaitForExit(10000)){throw 'Owned solver did not stop'}
                break
            }
        }
        $code=$p.ExitCode
        if(-not $stdout.Wait(10000) -or -not $stderr.Wait(10000)){throw 'Output capture timed out'}
        [IO.File]::WriteAllText((Join-Path $runRoot "N${n}_stdout.log"),$stdout.Result)
        [IO.File]::WriteAllText((Join-Path $runRoot "N${n}_stderr.log"),$stderr.Result)
        Write-Host $stdout.Result
    } finally {
        if($started -and -not $p.HasExited){$p.Kill();[void]$p.WaitForExit(10000)}
        $timer.Stop();$p.Dispose()
    }
    $runs.Add([pscustomobject]@{N=$n;exit_code=$code;timed_out=$timeout;process_seconds=$timer.Elapsed.TotalSeconds})
    foreach($kind in @('neumann','dirichlet')) {
        $path=Join-Path $out "$kind/result.json"
        if(Test-Path -LiteralPath $path) {
            $r=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
            $prior=$records | Where-Object {$_.bvp -eq $kind -and $_.N -lt $n} | Sort-Object N | Select-Object -Last 1
            $order=$null
            if($prior -and $prior.gmres_converged -and $r.gmres_converged -and $prior.interior_linf -gt 0 -and $r.interior_linf -gt 0) {
                $order=[Math]::Log($prior.interior_linf/$r.interior_linf)/[Math]::Log($n/$prior.N)
            }
            $r | Add-Member -NotePropertyName observed_order -NotePropertyValue $order
            $records.Add($r)
        }
    }
    [pscustomobject]@{source_revision=(& git -C $repo rev-parse HEAD);dirty=[bool](& git -C $repo status --porcelain);runs=@($runs.ToArray());results=@($records.ToArray())} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'summary.json') -Encoding UTF8
    if($timeout -or $code -eq 1){throw "N=$n failed; retained logs at $runRoot"}
}
Write-Host "Completed study: $runRoot/summary.json"
if(@($runs | Where-Object {$_.exit_code -ne 0}).Count){throw 'One or more GMRES solves did not converge; results retained.'}
