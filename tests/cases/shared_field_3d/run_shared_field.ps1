[CmdletBinding()]
param(
    [string] $BuildDirectory = 'build-3d',
    [string] $Configuration = 'Release',
    [int[]] $Levels = @(32),
    [ValidateSet('rotate','rotate_translate')] [string[]] $Transforms = @('rotate','rotate_translate'),
    [ValidateSet('dirichlet','neumann')] [string[]] $Bvps = @('dirichlet','neumann'),
    [ValidateSet('baseline','shared_direct','shared_jump')] [string[]] $Variants = @('baseline','shared_direct','shared_jump'),
    [string] $RunName = '',
    [ValidateRange(1,720)] [int] $TimeoutMinutes = 120,
    [ValidateRange(0.1,1024)] [double] $MemoryLimitGiB = 4,
    [string] $Python = 'python',
    [switch] $SkipBuild,
    [switch] $NoSpectrum,
    [switch] $DryRun
)
$ErrorActionPreference = 'Stop'
$sharedRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
foreach ($sharedN in $Levels) { if ($sharedN -notin @(32,64,128)) { throw 'Levels must be 32, 64, 128.' } }
foreach ($sharedSelection in @(@($Levels),@($Transforms),@($Bvps),@($Variants))) {
    if (-not $sharedSelection.Count -or @($sharedSelection | Select-Object -Unique).Count -ne $sharedSelection.Count) {throw 'Empty or duplicate study selections.'}
}
if (-not $RunName) {$RunName = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)}
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') {throw 'RunName must be a simple unique name.'}
$sharedBuild = if ([IO.Path]::IsPathRooted($BuildDirectory)) {$BuildDirectory} else {Join-Path $sharedRepo $BuildDirectory}
$sharedExeCandidates = @((Join-Path $sharedBuild "apps/$Configuration/kfbi_trace93_study_3d.exe"),(Join-Path $sharedBuild 'apps/kfbi_trace93_study_3d.exe'))
$sharedExe = $sharedExeCandidates | Where-Object {Test-Path -LiteralPath $_} | Select-Object -First 1
if (-not $sharedExe) {$sharedExe=$sharedExeCandidates[0]}
$sharedRun = Join-Path $sharedRepo "output/shared_field_3d/$RunName"
if (Test-Path -LiteralPath $sharedRun) {throw "Refusing to overwrite $sharedRun"}
if (-not $SkipBuild -and -not $DryRun) {
    $sharedBuildArgs=@('--build',$sharedBuild,'--config',$Configuration,'--target','kfbi_trace93_study_3d')
    $sharedCache=Join-Path $sharedBuild 'CMakeCache.txt'
    if ((Test-Path -LiteralPath $sharedCache) -and (Select-String -LiteralPath $sharedCache -Pattern '^CMAKE_GENERATOR:INTERNAL=Visual Studio' -Quiet)) {
        $sharedBuildArgs+=@('--parallel','1','--','/p:PreferredToolArchitecture=x64','/nodeReuse:false')
    } else {$sharedBuildArgs+=@('--parallel','2')}
    & cmake @sharedBuildArgs
    if ($LASTEXITCODE -ne 0) {throw 'Build failed.'}
    $sharedExe = $sharedExeCandidates | Where-Object {Test-Path -LiteralPath $_} | Select-Object -First 1
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $sharedExe)) {throw "Missing executable: $sharedExe"}
New-Item -ItemType Directory -Path $sharedRun | Out-Null
$sharedConfig = [ordered]@{levels=$Levels;transforms=$Transforms;bvps=$Bvps;variants=$Variants;geometry='u';density_factor=8.;field_ratio=4.;field_width=4.;field_ridge=1e-12;value_weight=1.;normal_weight=1.;pde_weight=1.;restrict_mode='staged';gmres_tolerance=2e-10;true_residual_bound=3e-10;memory_limit_gib=$MemoryLimitGiB;memory_warning_gib=0.8*$MemoryLimitGiB;spectrum32=(-not $NoSpectrum);verify_replay=$true;build_configuration=$Configuration;package_sha256='23cfa03489e1d1004a57f39703e1f4237d83afb6da096fed7cc7033ecb884ce5';dry_run=[bool]$DryRun}
$sharedConfig | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $sharedRun 'study_config.json') -Encoding utf8
$sharedRecords = [Collections.Generic.List[object]]::new()
$sharedStop = $false
:levels foreach ($sharedN in ($Levels | Sort-Object)) {
    foreach ($sharedPose in $Transforms) {foreach ($sharedBvp in $Bvps) {foreach ($sharedVariant in $Variants) {
        $sharedName = "${sharedVariant}_${sharedBvp}_${sharedPose}_N${sharedN}"
        $sharedOut = Join-Path $sharedRun $sharedName
        $sharedBackend = if ($sharedVariant -eq 'baseline') {'direct_cauchy'} else {'shared_field'}
        $sharedTarget = if ($sharedVariant -eq 'shared_jump') {'input_jump_half'} else {'raw'}
        $sharedArgs = @('--geometry','u','--N',"$sharedN",'--transform',$sharedPose,'--bvp',$sharedBvp,'--correction-backend',$sharedBackend,'--exterior-target',$sharedTarget,'--dump-matrices','--verify-replay','--output',$sharedOut)
        if ($sharedN -eq 32 -and -not $NoSpectrum) {$sharedArgs += '--operator-spectrum'}
        if ($sharedVariant -eq 'baseline') {$sharedArgs += @('--policy','trace','--event-mode','python93')}
        else {$sharedArgs += @('--shared-field-ratio','4','--shared-field-width','4','--shared-field-ridge','1e-12','--shared-field-value-weight','1','--shared-field-normal-weight','1','--shared-field-pde-weight','1','--shared-field-restrict','staged')}
        $sharedRecord = [ordered]@{N=$sharedN;transform=$sharedPose;bvp=$sharedBvp;variant=$sharedVariant;arguments=$sharedArgs;result_path="$sharedName/$sharedBvp/result.json";exit_code=$null;timed_out=$false;resource_failure=$false;memory_warning=$false;peak_working_set_bytes=0;process_seconds=0.;execution='PENDING'}
        if ($DryRun) {$sharedRecord.execution='DRY_RUN';$sharedRecords.Add([pscustomobject]$sharedRecord);continue}
        $sharedPsi = [Diagnostics.ProcessStartInfo]::new()
        $sharedPsi.FileName=$sharedExe;$sharedPsi.WorkingDirectory=$sharedRepo
        $sharedPsi.UseShellExecute=$false;$sharedPsi.CreateNoWindow=$true
        $sharedPsi.RedirectStandardOutput=$true;$sharedPsi.RedirectStandardError=$true
        foreach ($sharedArg in $sharedArgs) {$sharedPsi.ArgumentList.Add($sharedArg)}
        foreach ($sharedKey in @($sharedPsi.Environment.Keys)) {if ($sharedKey -like 'KFBIM_3D_*' -or $sharedKey -like 'KFBI_*') {$sharedPsi.Environment.Remove($sharedKey) | Out-Null}}
        foreach ($sharedKey in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) {$sharedPsi.Environment[$sharedKey]='1'}
        $sharedProcess=[Diagnostics.Process]::new();$sharedProcess.StartInfo=$sharedPsi
        $sharedTimer=[Diagnostics.Stopwatch]::StartNew();$sharedStarted=$false
        try {
            Write-Host "Starting $sharedName"
            $sharedStarted=$sharedProcess.Start()
            if (-not $sharedStarted) {throw "Failed to launch $sharedName"}
            $sharedStdout=$sharedProcess.StandardOutput.ReadToEndAsync();$sharedStderr=$sharedProcess.StandardError.ReadToEndAsync()
            while (-not $sharedProcess.WaitForExit(500)) {
                $sharedProcess.Refresh()
                $sharedRecord.peak_working_set_bytes=[Math]::Max($sharedRecord.peak_working_set_bytes,$sharedProcess.PeakWorkingSet64)
                if ($sharedRecord.peak_working_set_bytes -ge .8*$MemoryLimitGiB*1GB -and -not $sharedRecord.memory_warning) {Write-Warning "$sharedName reached the memory warning threshold";$sharedRecord.memory_warning=$true}
                if ($sharedRecord.peak_working_set_bytes -ge $MemoryLimitGiB*1GB) {$sharedRecord.resource_failure=$true;$sharedStop=$true}
                if ($sharedTimer.Elapsed.TotalMinutes -ge $TimeoutMinutes) {$sharedRecord.timed_out=$true}
                if ($sharedRecord.resource_failure -or $sharedRecord.timed_out) {$sharedProcess.Kill();$sharedProcess.WaitForExit();break}
            }
            $sharedRecord.exit_code=$sharedProcess.ExitCode
            $sharedRecord.peak_working_set_bytes=[Math]::Max($sharedRecord.peak_working_set_bytes,$sharedProcess.PeakWorkingSet64)
            [IO.File]::WriteAllText((Join-Path $sharedRun "${sharedName}_stdout.log"),$sharedStdout.GetAwaiter().GetResult())
            [IO.File]::WriteAllText((Join-Path $sharedRun "${sharedName}_stderr.log"),$sharedStderr.GetAwaiter().GetResult())
            $sharedRecord.execution='COMPLETED'
        } catch {$sharedRecord.execution='ERROR';$sharedRecord.error=$_.Exception.Message}
        finally {
            if ($sharedStarted -and -not $sharedProcess.HasExited) {$sharedProcess.Kill();$sharedProcess.WaitForExit()}
            $sharedTimer.Stop();$sharedRecord.process_seconds=$sharedTimer.Elapsed.TotalSeconds;$sharedProcess.Dispose()
        }
        $sharedRecords.Add([pscustomobject]$sharedRecord)
        ConvertTo-Json -InputObject @($sharedRecords.ToArray()) -Depth 10 | Set-Content -LiteralPath (Join-Path $sharedRun 'execution_status.json') -Encoding utf8
        if ($sharedStop) {break levels}
    }}}
}
ConvertTo-Json -InputObject @($sharedRecords.ToArray()) -Depth 10 | Set-Content -LiteralPath (Join-Path $sharedRun 'execution_status.json') -Encoding utf8
if ($DryRun) {Write-Host "Wrote $($sharedRecords.Count) explicit commands to $sharedRun";return}
& $Python (Join-Path $sharedRepo 'scripts/validation/summarize_shared_field_3d.py') $sharedRun
if ($LASTEXITCODE -ne 0) {throw "Study has missing configurations or candidate failures; see $sharedRun/summary.json"}
