[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string] $InputPath
)
# Read-only reporting. This script never builds, launches a solver, or writes
# result files. Accept a run directory, an N directory, summary.json, or one
# neumann/dirichlet result.json.
$ErrorActionPreference = 'Stop'
function Read-Json([string] $Path) {
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Property-Value($Object, [string] $Name) {
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -ne $property) { return $property.Value }
    return $null
}
function File-Records([string] $Path) {
    $json = Read-Json $Path
    $results = Property-Value $json 'results'
    if ($null -ne $results) { return @($results) }
    if ($null -ne (Property-Value $json 'N') -and $null -ne (Property-Value $json 'bvp')) { return @($json) }
    throw "Not a local result.json or summary.json: $Path"
}
$reference = Read-Json (Join-Path $PSScriptRoot 'python_reference_rotate.json')
$item = Get-Item -LiteralPath $InputPath
$localRecords = @()
if (-not $item.PSIsContainer) {
    $localRecords = @(File-Records $item.FullName)
} else {
    $summary = Join-Path $item.FullName 'summary.json'
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        $localRecords = @(File-Records $summary)
    } else {
        $directories = @($item.FullName)
        $directories += @(Get-ChildItem -LiteralPath $item.FullName -Directory |
            Where-Object { $_.Name -match '^N[0-9]+$' } | ForEach-Object { $_.FullName })
        foreach ($directory in $directories) {
            foreach ($relative in @('result.json','neumann/result.json','dirichlet/result.json')) {
                $file = Join-Path $directory $relative
                if (Test-Path -LiteralPath $file -PathType Leaf) { $localRecords += @(File-Records $file) }
            }
        }
    }
}
if ($localRecords.Count -eq 0) { throw 'No completed local results were found.' }
if (@($localRecords | Group-Object -Property N,bvp | Where-Object { $_.Count -gt 1 }).Count) {
    throw 'Multiple results have the same N/BVP. Select a single run directory or one result file.'
}
$rows = [Collections.Generic.List[object]]::new()
foreach ($local in ($localRecords | Sort-Object bvp,N)) {
    $baseline = @($reference.records | Where-Object { $_.bvp -eq $local.bvp -and $_.N -eq $local.N })
    if ($baseline.Count -ne 1) { Write-Warning "No archived N=$($local.N) $($local.bvp) reference; skipped."; continue }
    $base = $baseline[0]
    $notes = [Collections.Generic.List[string]]::new()
    if ((Property-Value $local 'transform') -ne 'rotate') { $notes.Add('transform differs') }
    if ((Property-Value $local 'chart') -ne 'extended') { $notes.Add('chart differs') }
    if ((Property-Value $local 'center_policy') -ne 'trace') { $notes.Add('center policy differs') }
    if ((Property-Value $local 'density') -ne 'python_anisotropic_exact_knot_embedding') { $notes.Add('density configuration differs or is unrecorded') }
    if ((Property-Value $local 'event_mode') -eq 'all') { $notes.Add('local all-event; archive endpoint-event') }
    elseif ((Property-Value $local 'event_mode') -ne 'python') { $notes.Add('event policy unrecorded') }
    if ((Property-Value $local 'reduced_dofs') -ne $base.nred) { $notes.Add('DOF count differs') }
    if ((Property-Value $local 'trace_points') -ne $base.trace_points) { $notes.Add('trace count differs') }
    $error = Property-Value $local 'interior_linf'
    $iterations = Property-Value $local 'gmres_iterations'
    $ratio = $null; $difference = $null
    if ($null -ne $error -and [double]::IsNaN([double]$error) -eq $false -and [double]::IsInfinity([double]$error) -eq $false) {
        $ratio = [double]$error / [double]$base.interior_linf
    }
    if ($null -ne $iterations) { $difference = [int]$iterations - [int]$base.gmres }
    $rows.Add([pscustomobject]@{
        BVP = $local.bvp; N = $local.N
        LocalError = $error; ArchiveError = $base.interior_linf
        ErrorRatio = $ratio; LocalGMRES = $iterations; ArchiveGMRES = $base.gmres
        IterationDifference = $difference
        Converged = Property-Value $local 'gmres_converged'
        ComparisonNotes = ($notes -join '; ')
    })
}
if ($rows.Count -eq 0) { throw 'No matching N=32/64 archived references.' }
Write-Host 'Reference: supplied Python archive, rotate / trace-polynomial-first / extended / dyadic. Not a new local solve.'
Write-Host 'ErrorRatio = local interior maximum error / archive error; IterationDifference = local - archive.'
Write-Host 'Archive geometry cache was warm. No cross-machine whole-run speedup is inferred.'
$rows.ToArray() | Format-Table BVP,N,@{Name='Local E_inf';Expression={'{0:E8}' -f $_.LocalError}},
    @{Name='Archive E_inf';Expression={'{0:E8}' -f $_.ArchiveError}},
    @{Name='Error ratio';Expression={if($null -eq $_.ErrorRatio){'n/a'}else{'{0:F6}' -f $_.ErrorRatio}}},
    LocalGMRES,ArchiveGMRES,IterationDifference,Converged -AutoSize
foreach ($row in $rows) {
    if ($row.ComparisonNotes) { Write-Warning "$($row.BVP) N=$($row.N): $($row.ComparisonNotes)" }
}
