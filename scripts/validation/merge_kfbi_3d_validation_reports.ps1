<#
.SYNOPSIS
Merges normalized KFBI 3-D validation reports without running a solver.

.DESCRIPTION
Each input is

  output/kfbi_3d_full_validation/<RunLabel>/all_results.csv

Rows are de-duplicated by
backend/geometry/formulation/trace_restrict_mode/pose/N.  Run labels are
processed from left to right, so a row from a later run label replaces the same
row from an earlier run label.  Different trace-restrict routes are always
preserved as distinct records.  Columns absent from the replacement schema are
retained from the earlier row, and the union of all input columns is preserved.

The default, publication mode is deliberately strict.  Every source run must
have a complete manifest and complete per-case archive, its normalized primary
keys must exactly match the archived primary keys, and the merged result must be
the fixed seven-geometry/168-row validation matrix.  Use -AllowPartial only for
ad-hoc subsets or repair inspection; in that mode the historical replacement
semantics above remain available.

The following convergence-order columns are recomputed within each
backend/geometry/formulation/trace_restrict_mode/pose group, using adjacent
increasing N values and the actual h ratio
p=log(E_previous/E_current)/log(h_previous/h_current):

  * interior_order_linf
  * density_order_linf
  * exterior_condition_order_linf
  * boundary_residual_order_linf

An order is NaN when either adjacent error is missing, non-finite, or non-positive.
The original error columns are not changed.  Four RFC-compatible CSV files are
written to a new output directory:

  * merged_all_results.csv
  * baseline_results.csv
  * strong_rigid_results.csv
  * pose_N32_results.csv

The convergence-rigid pose is backend-specific: general_cap uses
rot_axis123_17deg_t_xyz_1, while topology_native uses tx_p0137.  The
strong_rigid_results.csv file combines those backend-specific rows without
rewriting their pose values.  The N=32 file contains all poses, including
baseline and both convergence-rigid poses.

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  scripts/validation/merge_kfbi_3d_validation_reports.ps1 `
  -RunLabel cap0820a,top320820a,tophi0820a,topbtor0820a,toptx640820a,toptx1280820a `
  -OutputDirectory output/kfbi_3d_full_validation/merged_final

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  scripts/validation/merge_kfbi_3d_validation_reports.ps1 `
  -RunLabel topology_part,repair_part -AllowPartial `
  -OutputDirectory output/kfbi_3d_full_validation/inspection

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  scripts/validation/merge_kfbi_3d_validation_reports.ps1 -SyntheticCsvTest
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string[]] $RunLabel = @(),

    [string] $OutputDirectory = '',

    [switch] $Force,
    [switch] $AllowPartial,
    [switch] $SyntheticCsvTest
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$validationRoot = Join-Path $repo 'output/kfbi_3d_full_validation'
$invariantCulture = [Globalization.CultureInfo]::InvariantCulture
$generalCapConvergencePose = 'rot_axis123_17deg_t_xyz_1'
$topologyConvergencePose = 'tx_p0137'
$convergencePoseByBackend = @{
    general_cap = $generalCapConvergencePose
    topology_native = $topologyConvergencePose
}
$canonicalLevels = @(32, 64, 128)
$canonicalFormulations = @('dirichlet', 'neumann')
$topologyNeumannTraceRestrict = 'q27_cover3_all_event_cauchy'
$topologyDirichletTraceRestrict = 'q64_cover4_all_event_cauchy'
$canonicalPoses = @(
    'baseline',
    'tx_p0137',
    'ty_m0083',
    'tz_p0061',
    't_xyz_1',
    't_xyz_2',
    'rot_axis123_17deg',
    $generalCapConvergencePose
)
$canonicalGeometryCatalog = @(
    [pscustomobject]@{ Backend = 'general_cap'; Geometry = 'sphere' },
    [pscustomobject]@{ Backend = 'general_cap'; Geometry = 'ellipsoid' },
    [pscustomobject]@{ Backend = 'general_cap'; Geometry = 'flower' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'torus' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'cylinder' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'l_prism' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'u_prism' }
)

$orderSpecifications = @(
    [pscustomobject]@{
        ErrorColumn = 'interior_linf'
        OrderColumn = 'interior_order_linf'
    },
    [pscustomobject]@{
        ErrorColumn = 'density_linf'
        OrderColumn = 'density_order_linf'
    },
    [pscustomobject]@{
        ErrorColumn = 'exterior_condition_linf'
        OrderColumn = 'exterior_condition_order_linf'
    },
    [pscustomobject]@{
        ErrorColumn = 'boundary_residual_linf'
        OrderColumn = 'boundary_residual_order_linf'
    }
)

function Test-SameText([string] $left, [string] $right) {
    return [string]::Equals(
        $left, $right, [StringComparison]::OrdinalIgnoreCase)
}

function Get-ConvergencePoseForBackend([string] $backend) {
    $normalized = ([string] $backend).Trim().ToLowerInvariant()
    if ($convergencePoseByBackend.ContainsKey($normalized)) {
        return [string] $convergencePoseByBackend[$normalized]
    }
    return ''
}

function Get-CanonicalTraceRestrictMode(
    [string] $backend,
    [string] $formulation) {
    if (Test-SameText $backend 'topology_native') {
        if (Test-SameText $formulation 'neumann') {
            return $topologyNeumannTraceRestrict
        }
        if (Test-SameText $formulation 'dirichlet') {
            return $topologyDirichletTraceRestrict
        }
    } elseif (Test-SameText $backend 'general_cap') {
        return 'not_applicable'
    }
    throw "Unknown backend/formulation '$backend/$formulation'"
}

function Test-IsBackendConvergenceRecord($record) {
    $backend = [string] (Get-DataValue $record 'backend')
    $expectedPose = Get-ConvergencePoseForBackend $backend
    if ([string]::IsNullOrWhiteSpace($expectedPose)) {
        return $false
    }
    return Test-SameText `
        ([string] (Get-DataValue $record 'pose')) `
        $expectedPose
}

function Test-ColumnExists(
    [System.Collections.Generic.List[string]] $columns,
    [string] $name) {
    foreach ($column in $columns) {
        if (Test-SameText $column $name) {
            return $true
        }
    }
    return $false
}

function Get-CanonicalColumn(
    [System.Collections.Generic.List[string]] $columns,
    [string] $name) {
    foreach ($column in $columns) {
        if (Test-SameText $column $name) {
            return $column
        }
    }
    return $null
}

function Ensure-ColumnAfter(
    [System.Collections.Generic.List[string]] $columns,
    [string] $name,
    [string] $afterName) {
    if (Test-ColumnExists $columns $name) {
        return
    }

    $after = Get-CanonicalColumn $columns $afterName
    if ($null -eq $after) {
        $columns.Add($name)
        return
    }

    $index = $columns.IndexOf($after)
    $columns.Insert($index + 1, $name)
}

function Get-DataKey([Collections.Specialized.OrderedDictionary] $data,
                     [string] $name) {
    foreach ($key in $data.Keys) {
        if (Test-SameText ([string] $key) $name) {
            return [string] $key
        }
    }
    return $null
}

function Get-DataValue($record, [string] $name) {
    $key = Get-DataKey $record.Data $name
    if ($null -eq $key) {
        return ''
    }
    return $record.Data[$key]
}

function Set-DataValue($record, [string] $name, $value) {
    $key = Get-DataKey $record.Data $name
    if ($null -eq $key) {
        $record.Data.Add($name, $value)
    } else {
        $record.Data[$key] = $value
    }
}

function Get-NormalizedKeyPart($value, [string] $column, [string] $source) {
    if ($null -eq $value) {
        throw "Missing key value '$column' in $source"
    }
    $text = ([string] $value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "Blank key value '$column' in $source"
    }
    return $text.ToLowerInvariant()
}

function Get-PositiveGridLevel($value, [string] $source) {
    $parsed = 0
    $text = if ($null -eq $value) { '' } else { ([string] $value).Trim() }
    $ok = [int]::TryParse(
        $text,
        [Globalization.NumberStyles]::Integer,
        $invariantCulture,
        [ref] $parsed)
    if (-not $ok -or $parsed -le 0) {
        throw "Invalid positive integer N='$text' in $source"
    }
    return $parsed
}

function Get-ObjectValue($object, [string] $name) {
    if ($null -eq $object) {
        return $null
    }
    foreach ($property in $object.PSObject.Properties) {
        if (Test-SameText $property.Name $name) {
            return $property.Value
        }
    }
    return $null
}

function New-ValidationKey(
    $backendValue,
    $geometryValue,
    $formulationValue,
    $traceRestrictModeValue,
    $poseValue,
    $nValue,
    [string] $source) {
    $backend = Get-NormalizedKeyPart $backendValue 'backend' $source
    $geometry = Get-NormalizedKeyPart $geometryValue 'geometry' $source
    $formulation = Get-NormalizedKeyPart (
        $formulationValue) 'formulation' $source
    $traceRestrictMode = Get-NormalizedKeyPart (
        $traceRestrictModeValue) 'trace_restrict_mode' $source
    $pose = Get-NormalizedKeyPart $poseValue 'pose' $source
    $n = Get-PositiveGridLevel $nValue $source
    $separator = [char] 31
    return $backend + $separator + $geometry + $separator +
        $formulation + $separator + $traceRestrictMode + $separator +
        $pose + $separator +
        $n.ToString($invariantCulture)
}

function Add-UniqueKey(
    [hashtable] $keys,
    [string] $key,
    [string] $description) {
    if ($keys.ContainsKey($key)) {
        throw "Duplicate primary key in ${description}: $($key.Replace([char] 31, '/'))"
    }
    $keys[$key] = $true
}

function Assert-KeySetsEqual(
    [hashtable] $expected,
    [hashtable] $actual,
    [string] $description) {
    $missing = @($expected.Keys | Where-Object {
        -not $actual.ContainsKey($_)
    } | Sort-Object)
    $unexpected = @($actual.Keys | Where-Object {
        -not $expected.ContainsKey($_)
    } | Sort-Object)
    if ($missing.Count -eq 0 -and $unexpected.Count -eq 0) {
        return
    }
    $formatKeys = {
        param([object[]] $values)
        if ($values.Count -eq 0) {
            return 'none'
        }
        return (@($values | Select-Object -First 5 | ForEach-Object {
            ([string] $_).Replace([char] 31, '/')
        }) -join ', ')
    }
    throw (
        "$description primary-key mismatch: expected=$($expected.Count) " +
        "actual=$($actual.Count) missing=$($missing.Count) " +
        "unexpected=$($unexpected.Count); missing sample=" +
        (& $formatKeys $missing) + '; unexpected sample=' +
        (& $formatKeys $unexpected))
}

function Import-StableCsv([string] $path, [string] $description) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing ${description}: $path"
    }
    $before = Get-Item -LiteralPath $path
    $rows = @(Import-Csv -LiteralPath $path -Encoding UTF8)
    $after = Get-Item -LiteralPath $path
    if ($before.Length -ne $after.Length -or
        $before.LastWriteTimeUtc -ne $after.LastWriteTimeUtc) {
        throw "${description} changed while it was read; retry after the run finishes: $path"
    }
    return @($rows)
}

function Import-StableJson([string] $path, [string] $description) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing ${description}: $path"
    }
    $before = Get-Item -LiteralPath $path
    $text = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    $after = Get-Item -LiteralPath $path
    if ($before.Length -ne $after.Length -or
        $before.LastWriteTimeUtc -ne $after.LastWriteTimeUtc) {
        throw "${description} changed while it was read; retry after the run finishes: $path"
    }
    try {
        return $text | ConvertFrom-Json
    } catch {
        throw "Invalid JSON in ${description} ${path}: $($_.Exception.Message)"
    }
}

function Assert-ObjectColumns(
    [object[]] $rows,
    [string[]] $required,
    [string] $source) {
    if ($rows.Count -eq 0) {
        throw "No data rows in $source"
    }
    $names = @($rows[0].PSObject.Properties.Name)
    foreach ($column in $required) {
        $found = $false
        foreach ($name in $names) {
            if (Test-SameText $name $column) {
                $found = $true
                break
            }
        }
        if (-not $found) {
            throw "Missing required column '$column' in $source"
        }
    }
}

function Get-CanonicalGeometryBackend([string] $geometry) {
    foreach ($item in $canonicalGeometryCatalog) {
        if (Test-SameText $item.Geometry $geometry) {
            return $item.Backend
        }
    }
    return $null
}

function Get-ArchivePrimaryKeys(
    [string] $backend,
    [string] $pose,
    [string] $archiveDirectory,
    [string] $caseDescription) {
    $keys = @{}
    if (Test-SameText $backend 'topology_native') {
        $files = @(
            [pscustomobject]@{
                Name = 'neumann_results.csv'
                Formulation = 'neumann'
            },
            [pscustomobject]@{
                Name = 'dirichlet_normal_results.csv'
                Formulation = 'dirichlet'
            }
        )
        foreach ($file in $files) {
            $path = Join-Path $archiveDirectory $file.Name
            $rows = @(Import-StableCsv $path "archived result for $caseDescription")
            Assert-ObjectColumns $rows @(
                'geometry', 'trace_restrict_mode', 'N') $path
            foreach ($row in $rows) {
                $key = New-ValidationKey `
                    $backend `
                    (Get-ObjectValue $row 'geometry') `
                    $file.Formulation `
                    (Get-ObjectValue $row 'trace_restrict_mode') `
                    $pose `
                    (Get-ObjectValue $row 'N') `
                    $path
                Add-UniqueKey $keys $key "archive $caseDescription"
            }
        }
        return $keys
    }

    if (-not (Test-SameText $backend 'general_cap')) {
        throw "Unsupported backend '$backend' in $caseDescription"
    }
    $path = Join-Path $archiveDirectory 'refinement.csv'
    $rows = @(Import-StableCsv $path "archived result for $caseDescription")
    Assert-ObjectColumns $rows @('shape', 'pose', 'N') $path
    foreach ($row in $rows) {
        $archivePose = [string] (Get-ObjectValue $row 'pose')
        if (-not (Test-SameText $archivePose $pose)) {
            throw "Archive pose '$archivePose' does not match '$pose' in $caseDescription"
        }
        foreach ($formulation in $canonicalFormulations) {
            $key = New-ValidationKey `
                $backend `
                (Get-ObjectValue $row 'shape') `
                $formulation `
                (Get-CanonicalTraceRestrictMode $backend $formulation) `
                $pose `
                (Get-ObjectValue $row 'N') `
                $path
            Add-UniqueKey $keys $key "archive $caseDescription"
        }
    }
    return $keys
}

function Assert-CompletedRunSource(
    [string] $runDirectory,
    [string] $allResultsPath) {
    $runDirectory = [IO.Path]::GetFullPath($runDirectory)
    $manifestPath = Join-Path $runDirectory 'case_manifest.csv'
    $manifestRows = @(Import-StableCsv $manifestPath 'run manifest')
    Assert-ObjectColumns $manifestRows @(
        'case_id', 'status', 'backend', 'pose', 'levels',
        'exit_code', 'archive_directory') $manifestPath

    $manifestByCase = @{}
    $expectedAll = @{}
    $expectedByCase = @{}
    foreach ($manifestRow in $manifestRows) {
        $caseId = ([string] (Get-ObjectValue $manifestRow 'case_id')).Trim()
        if ([string]::IsNullOrWhiteSpace($caseId) -or
            $caseId -notmatch '^[A-Za-z0-9_.-]+$' -or
            $caseId -eq '.' -or $caseId -eq '..') {
            throw "Invalid case_id '$caseId' in $manifestPath"
        }
        if ($manifestByCase.ContainsKey($caseId)) {
            throw "Duplicate manifest case_id '$caseId' in $manifestPath"
        }
        $manifestByCase[$caseId] = $manifestRow

        $manifestStatus = ([string] (
            Get-ObjectValue $manifestRow 'status')).Trim().ToLowerInvariant()
        if ($manifestStatus -ne 'complete' -and
            $manifestStatus -ne 'resumed') {
            throw (
                "Run manifest is not complete: case_id=$caseId " +
                "status='$manifestStatus' in $manifestPath")
        }
        $exitCode = 0
        if (-not [int]::TryParse(
                ([string] (Get-ObjectValue $manifestRow 'exit_code')).Trim(),
                [Globalization.NumberStyles]::Integer,
                $invariantCulture,
                [ref] $exitCode) -or $exitCode -ne 0) {
            throw "Completed manifest case '$caseId' has nonzero/invalid exit_code in $manifestPath"
        }

        $backend = ([string] (
            Get-ObjectValue $manifestRow 'backend')).Trim().ToLowerInvariant()
        $pose = ([string] (
            Get-ObjectValue $manifestRow 'pose')).Trim().ToLowerInvariant()
        if ($canonicalPoses -notcontains $pose) {
            throw "Unknown pose '$pose' in manifest case '$caseId'"
        }
        if ($backend -ne 'topology_native' -and $backend -ne 'general_cap') {
            throw "Unknown backend '$backend' in manifest case '$caseId'"
        }

        $manifestLevels = @{}
        foreach ($levelText in (([string] (
                Get-ObjectValue $manifestRow 'levels')) -split '\|')) {
            $level = Get-PositiveGridLevel $levelText $manifestPath
            Add-UniqueKey $manifestLevels ([string] $level) (
                "manifest levels for $caseId")
        }

        $archiveDirectory = Join-Path (Join-Path $runDirectory 'raw') $caseId
        $archiveDirectory = [IO.Path]::GetFullPath($archiveDirectory)
        $manifestArchiveText = ([string] (
            Get-ObjectValue $manifestRow 'archive_directory')).Trim()
        if ([string]::IsNullOrWhiteSpace($manifestArchiveText)) {
            throw "Blank archive_directory for manifest case '$caseId'"
        }
        $manifestArchive = if ([IO.Path]::IsPathRooted($manifestArchiveText)) {
            [IO.Path]::GetFullPath($manifestArchiveText)
        } else {
            [IO.Path]::GetFullPath((Join-Path $repo $manifestArchiveText))
        }
        if (-not [string]::Equals(
                $manifestArchive,
                $archiveDirectory,
                [StringComparison]::OrdinalIgnoreCase)) {
            throw (
                "Manifest archive_directory does not identify its durable " +
                "run archive for case '$caseId': $manifestArchiveText")
        }
        if (-not (Test-Path -LiteralPath $archiveDirectory -PathType Container)) {
            throw "Missing archive directory for case '$caseId': $archiveDirectory"
        }

        $statusPath = Join-Path $archiveDirectory 'case_status.json'
        $caseStatus = Import-StableJson $statusPath "case status for $caseId"
        if (-not (Test-SameText (
                [string] (Get-ObjectValue $caseStatus 'status')) 'complete')) {
            throw "Archive case '$caseId' does not have status=complete: $statusPath"
        }
        foreach ($field in @('case_id', 'backend', 'pose')) {
            $manifestValue = if ($field -eq 'case_id') {
                $caseId
            } else {
                [string] (Get-ObjectValue $manifestRow $field)
            }
            $statusValue = [string] (Get-ObjectValue $caseStatus $field)
            if (-not (Test-SameText $manifestValue $statusValue)) {
                throw "Archive field '$field' does not match manifest for case '$caseId'"
            }
        }

        $statusLevels = @{}
        foreach ($levelValue in @(Get-ObjectValue $caseStatus 'levels')) {
            $level = Get-PositiveGridLevel $levelValue $statusPath
            Add-UniqueKey $statusLevels ([string] $level) (
                "archive levels for $caseId")
        }
        Assert-KeySetsEqual $manifestLevels $statusLevels (
            "manifest/archive levels for $caseId")

        $geometries = @(
            Get-ObjectValue $caseStatus 'expected_geometries')
        if ($geometries.Count -eq 0) {
            throw "Archive case '$caseId' has no expected_geometries"
        }
        $expectedCase = @{}
        foreach ($geometryValue in $geometries) {
            $geometry = ([string] $geometryValue).Trim().ToLowerInvariant()
            $geometryBackend = Get-CanonicalGeometryBackend $geometry
            if ($null -eq $geometryBackend -or
                -not (Test-SameText $geometryBackend $backend)) {
                throw "Geometry/backend mismatch '$backend/$geometry' in archive case '$caseId'"
            }
            foreach ($levelText in $statusLevels.Keys) {
                foreach ($formulation in $canonicalFormulations) {
                    $traceRestrictMode = Get-CanonicalTraceRestrictMode `
                        $backend $formulation
                    $key = New-ValidationKey `
                        $backend $geometry $formulation $traceRestrictMode `
                        $pose $levelText $statusPath
                    Add-UniqueKey $expectedCase $key "expected archive case $caseId"
                    Add-UniqueKey $expectedAll $key "expected run $runDirectory"
                }
            }
        }
        $archiveKeys = Get-ArchivePrimaryKeys `
            $backend $pose $archiveDirectory "case $caseId"
        Assert-KeySetsEqual $expectedCase $archiveKeys "archive case $caseId"
        $expectedByCase[$caseId] = $expectedCase
    }

    $allRows = @(Import-StableCsv $allResultsPath 'normalized all_results.csv')
    Assert-ObjectColumns $allRows @(
        'case_id', 'backend', 'geometry', 'formulation',
        'trace_restrict_mode', 'pose', 'N') `
        $allResultsPath
    $actualAll = @{}
    $actualByCase = @{}
    foreach ($caseId in $manifestByCase.Keys) {
        $actualByCase[$caseId] = @{}
    }
    foreach ($row in $allRows) {
        $caseId = ([string] (Get-ObjectValue $row 'case_id')).Trim()
        if (-not $manifestByCase.ContainsKey($caseId)) {
            throw "all_results row references unknown case_id '$caseId' in $allResultsPath"
        }
        $key = New-ValidationKey `
            (Get-ObjectValue $row 'backend') `
            (Get-ObjectValue $row 'geometry') `
            (Get-ObjectValue $row 'formulation') `
            (Get-ObjectValue $row 'trace_restrict_mode') `
            (Get-ObjectValue $row 'pose') `
            (Get-ObjectValue $row 'N') `
            $allResultsPath
        Add-UniqueKey $actualAll $key "normalized run $runDirectory"
        Add-UniqueKey $actualByCase[$caseId] $key (
            "normalized case $caseId")
    }
    foreach ($caseId in $expectedByCase.Keys) {
        Assert-KeySetsEqual $expectedByCase[$caseId] $actualByCase[$caseId] (
            "archive/all_results case $caseId")
    }
    Assert-KeySetsEqual $expectedAll $actualAll (
        "manifest/archive/all_results run $runDirectory")
    return [pscustomobject]@{
        Cases = $manifestRows.Count
        Keys = $actualAll.Count
    }
}

function Try-GetFinitePositiveDouble($value, [ref] $parsed) {
    $number = 0.0
    if ($null -eq $value) {
        return $false
    }
    $text = ([string] $value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $false
    }
    $ok = [double]::TryParse(
        $text,
        [Globalization.NumberStyles]::Float,
        $invariantCulture,
        [ref] $number)
    if (-not $ok -or [double]::IsNaN($number) -or
        [double]::IsInfinity($number) -or $number -le 0.0) {
        return $false
    }
    $parsed.Value = $number
    return $true
}

function Convert-RecordToOutputObject(
    $record,
    [System.Collections.Generic.List[string]] $columns) {
    $values = [ordered]@{}
    foreach ($column in $columns) {
        $values[$column] = Get-DataValue $record $column
    }
    return [pscustomobject] $values
}

function Write-RfcCsv(
    [string] $path,
    [object[]] $records,
    [System.Collections.Generic.List[string]] $columns) {
    $objects = @($records | ForEach-Object {
        Convert-RecordToOutputObject $_ $columns
    })

    if ($objects.Count -gt 0) {
        $objects | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
        return
    }

    # Export-Csv normally writes no file for an empty input.  Generate its RFC
    # header with a one-row object, then retain only that quoted header.
    $emptyValues = [ordered]@{}
    foreach ($column in $columns) {
        $emptyValues[$column] = ''
    }
    [pscustomobject] $emptyValues |
        Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
    $header = Get-Content -LiteralPath $path -Encoding UTF8 -TotalCount 1
    $header | Set-Content -LiteralPath $path -Encoding UTF8
}

function Add-RecomputedOrders([object[]] $records) {
    $groups = @{}
    foreach ($record in $records) {
        if (-not $groups.ContainsKey($record.GroupKey)) {
            $groups[$record.GroupKey] =
                [System.Collections.Generic.List[object]]::new()
        }
        $groups[$record.GroupKey].Add($record)
    }

    foreach ($group in $groups.Values) {
        $ordered = @($group | Sort-Object NValue)
        $previous = $null
        foreach ($record in $ordered) {
            foreach ($specification in $orderSpecifications) {
                Set-DataValue $record $specification.OrderColumn 'NaN'
                if ($null -eq $previous -or
                    $record.NValue -le $previous.NValue) {
                    continue
                }

                $previousError = 0.0
                $currentError = 0.0
                $hasPrevious = Try-GetFinitePositiveDouble (
                    Get-DataValue $previous $specification.ErrorColumn) (
                    [ref] $previousError)
                $hasCurrent = Try-GetFinitePositiveDouble (
                    Get-DataValue $record $specification.ErrorColumn) (
                    [ref] $currentError)
                if (-not $hasPrevious -or -not $hasCurrent) {
                    continue
                }

                $previousH = 0.0
                $currentH = 0.0
                $hasPreviousH = Try-GetFinitePositiveDouble (
                    Get-DataValue $previous 'h') ([ref] $previousH)
                $hasCurrentH = Try-GetFinitePositiveDouble (
                    Get-DataValue $record 'h') ([ref] $currentH)
                if (-not $hasPreviousH -or -not $hasCurrentH -or
                    $previousH -le $currentH) {
                    continue
                }

                $denominator = [Math]::Log($previousH / $currentH)
                if ([double]::IsNaN($denominator) -or
                    [double]::IsInfinity($denominator) -or
                    [Math]::Abs($denominator) -le 1.0e-15) {
                    continue
                }
                $order = [Math]::Log($previousError / $currentError) /
                    $denominator
                if ([double]::IsNaN($order) -or [double]::IsInfinity($order)) {
                    continue
                }
                Set-DataValue $record $specification.OrderColumn (
                    $order.ToString('R', $invariantCulture))
            }
            $previous = $record
        }
    }
}

function Get-SortedRecords([object[]] $records) {
    return @($records | Sort-Object `
        @{ Expression = { ([string] (Get-DataValue $_ 'backend')).ToLowerInvariant() } }, `
        @{ Expression = { ([string] (Get-DataValue $_ 'geometry')).ToLowerInvariant() } }, `
        @{ Expression = { ([string] (Get-DataValue $_ 'formulation')).ToLowerInvariant() } }, `
        @{ Expression = { ([string] (Get-DataValue $_ 'trace_restrict_mode')).ToLowerInvariant() } }, `
        @{ Expression = { ([string] (Get-DataValue $_ 'pose')).ToLowerInvariant() } }, `
        @{ Expression = { $_.NValue } })
}

function Assert-SafeOutputDirectory(
    [string] $outputPath,
    [string[]] $inputPaths) {
    $resolvedOutput = [IO.Path]::GetFullPath($outputPath).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    foreach ($inputPath in $inputPaths) {
        $inputDirectory = [IO.Path]::GetFullPath(
            (Split-Path -Parent $inputPath)).TrimEnd(
                [IO.Path]::DirectorySeparatorChar,
                [IO.Path]::AltDirectorySeparatorChar)
        $inputPrefix = $inputDirectory + [IO.Path]::DirectorySeparatorChar
        if ([string]::Equals(
                $resolvedOutput,
                $inputDirectory,
                [StringComparison]::OrdinalIgnoreCase) -or
            $resolvedOutput.StartsWith(
                $inputPrefix,
                [StringComparison]::OrdinalIgnoreCase)) {
            throw "Output directory must not be an input run directory or its child: $resolvedOutput"
        }
    }
}

function Get-CanonicalMatrixKeys {
    $keys = @{}
    foreach ($geometryEntry in $canonicalGeometryCatalog) {
        $convergencePose = Get-ConvergencePoseForBackend `
            $geometryEntry.Backend
        foreach ($formulation in $canonicalFormulations) {
            foreach ($pose in $canonicalPoses) {
                $levels = if ($pose -eq 'baseline' -or
                    $pose -eq $convergencePose) {
                    $canonicalLevels
                } else {
                    @(32)
                }
                foreach ($n in $levels) {
                    $key = New-ValidationKey `
                        $geometryEntry.Backend `
                        $geometryEntry.Geometry `
                        $formulation `
                        (Get-CanonicalTraceRestrictMode `
                            $geometryEntry.Backend $formulation) `
                        $pose `
                        $n `
                        'canonical seven-geometry matrix'
                    Add-UniqueKey $keys $key 'canonical seven-geometry matrix'
                }
            }
        }
    }
    return $keys
}

function Assert-CanonicalValidationMatrix(
    [object[]] $records,
    [int] $inputDuplicateCount) {
    if ($inputDuplicateCount -ne 0) {
        throw (
            "Strict publication merge rejects duplicate input primary keys; " +
            "found $inputDuplicateCount. Use -AllowPartial only for an " +
            'intentional repair/overlay merge.')
    }
    $actual = @{}
    foreach ($record in $records) {
        $key = New-ValidationKey `
            (Get-DataValue $record 'backend') `
            (Get-DataValue $record 'geometry') `
            (Get-DataValue $record 'formulation') `
            (Get-DataValue $record 'trace_restrict_mode') `
            (Get-DataValue $record 'pose') `
            (Get-DataValue $record 'N') `
            'merged validation records'
        Add-UniqueKey $actual $key 'merged validation records'
    }
    $expected = Get-CanonicalMatrixKeys
    Assert-KeySetsEqual $expected $actual 'strict seven-geometry matrix'

    $baselineCount = @($records | Where-Object {
        Test-SameText ([string] (Get-DataValue $_ 'pose')) 'baseline'
    }).Count
    $strongCount = @($records | Where-Object {
        Test-IsBackendConvergenceRecord $_
    }).Count
    $n32Count = @($records | Where-Object {
        $_.NValue -eq 32
    }).Count
    if ($records.Count -ne 168 -or $baselineCount -ne 42 -or
        $strongCount -ne 42 -or $n32Count -ne 112) {
        throw (
            "Strict seven-geometry category counts are invalid: total=" +
            "$($records.Count)/168 baseline=$baselineCount/42 " +
            "convergence_rigid=$strongCount/42 N32_pose=$n32Count/112")
    }

    foreach ($geometryEntry in $canonicalGeometryCatalog) {
        $geometryCount = @($records | Where-Object {
            (Test-SameText `
                ([string] (Get-DataValue $_ 'backend')) `
                $geometryEntry.Backend) -and
            (Test-SameText `
                ([string] (Get-DataValue $_ 'geometry')) `
                $geometryEntry.Geometry)
        }).Count
        if ($geometryCount -ne 24) {
            throw (
                "Strict geometry coverage is invalid for " +
                "$($geometryEntry.Backend)/$($geometryEntry.Geometry): " +
                "$geometryCount/24 rows")
        }
    }
}

function Merge-ValidationCsvs(
    [string[]] $inputPaths,
    [string] $outputPath,
    [bool] $overwrite,
    [bool] $allowPartial) {
    if ($inputPaths.Count -eq 0) {
        throw 'At least one input all_results.csv is required'
    }

    Assert-SafeOutputDirectory $outputPath $inputPaths

    $columns = [System.Collections.Generic.List[string]]::new()
    $recordsByKey = @{}
    $inputRowCount = 0
    $duplicateCount = 0
    $sequence = 0

    foreach ($inputPath in $inputPaths) {
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
            throw "Missing input report: $inputPath"
        }

        # Detect a file that changed while it was being imported.  This avoids
        # silently accepting a partial snapshot of an actively written report.
        $before = Get-Item -LiteralPath $inputPath
        $rows = @(Import-Csv -LiteralPath $inputPath -Encoding UTF8)
        $after = Get-Item -LiteralPath $inputPath
        if ($before.Length -ne $after.Length -or
            $before.LastWriteTimeUtc -ne $after.LastWriteTimeUtc) {
            throw "Input report changed while it was read; retry after the run finishes: $inputPath"
        }
        if ($rows.Count -eq 0) {
            Write-Warning "Input report has no data rows and was skipped: $inputPath"
            continue
        }

        foreach ($property in $rows[0].PSObject.Properties) {
            if (-not (Test-ColumnExists $columns $property.Name)) {
                $columns.Add($property.Name)
            }
        }

        foreach ($row in $rows) {
            ++$inputRowCount
            ++$sequence
            $data = [ordered]@{}
            foreach ($property in $row.PSObject.Properties) {
                $data[$property.Name] = $property.Value
                if (-not (Test-ColumnExists $columns $property.Name)) {
                    $columns.Add($property.Name)
                }
            }

            $temporary = [pscustomobject]@{ Data = $data }
            $backend = Get-NormalizedKeyPart (
                Get-DataValue $temporary 'backend') 'backend' $inputPath
            $geometry = Get-NormalizedKeyPart (
                Get-DataValue $temporary 'geometry') 'geometry' $inputPath
            $formulation = Get-NormalizedKeyPart (
                Get-DataValue $temporary 'formulation') 'formulation' $inputPath
            $traceRestrictMode = Get-NormalizedKeyPart (
                Get-DataValue $temporary 'trace_restrict_mode') `
                'trace_restrict_mode' $inputPath
            $pose = Get-NormalizedKeyPart (
                Get-DataValue $temporary 'pose') 'pose' $inputPath
            $n = Get-PositiveGridLevel (
                Get-DataValue $temporary 'N') $inputPath
            $separator = [char] 31
            $groupKey = $backend + $separator + $geometry + $separator +
                $formulation + $separator + $traceRestrictMode + $separator +
                $pose
            $key = $groupKey + $separator +
                $n.ToString($invariantCulture)

            if ($recordsByKey.ContainsKey($key)) {
                ++$duplicateCount
                $earlier = $recordsByKey[$key]
                # Preserve values from columns that do not exist in the later
                # schema; values explicitly present in the later row win.
                foreach ($earlierKey in $earlier.Data.Keys) {
                    if ($null -eq (Get-DataKey $data ([string] $earlierKey))) {
                        $data.Add($earlierKey, $earlier.Data[$earlierKey])
                    }
                }
            }

            $recordsByKey[$key] = [pscustomobject]@{
                Data = $data
                NValue = $n
                GroupKey = $groupKey
                Sequence = $sequence
            }
        }
    }

    if ($recordsByKey.Count -eq 0) {
        throw 'The input reports contained no data rows'
    }

    foreach ($specification in $orderSpecifications) {
        Ensure-ColumnAfter `
            $columns `
            $specification.OrderColumn `
            $specification.ErrorColumn
    }

    $records = @($recordsByKey.Values)
    if (-not $allowPartial) {
        Assert-CanonicalValidationMatrix $records $duplicateCount
    }
    Add-RecomputedOrders $records
    $records = @(Get-SortedRecords $records)

    $baseline = @($records | Where-Object {
        Test-SameText ([string] (Get-DataValue $_ 'pose')) 'baseline'
    })
    $strongRigid = @($records | Where-Object {
        Test-IsBackendConvergenceRecord $_
    })
    $poseN32 = @($records | Where-Object { $_.NValue -eq 32 })

    $targetPaths = [ordered]@{
        merged = Join-Path $outputPath 'merged_all_results.csv'
        baseline = Join-Path $outputPath 'baseline_results.csv'
        strong_rigid = Join-Path $outputPath 'strong_rigid_results.csv'
        pose_N32 = Join-Path $outputPath 'pose_N32_results.csv'
    }
    if (-not $overwrite) {
        foreach ($target in $targetPaths.Values) {
            if (Test-Path -LiteralPath $target) {
                throw "Output already exists; choose another directory or pass -Force: $target"
            }
        }
    }

    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
    Write-RfcCsv $targetPaths.merged $records $columns
    Write-RfcCsv $targetPaths.baseline $baseline $columns
    Write-RfcCsv $targetPaths.strong_rigid $strongRigid $columns
    Write-RfcCsv $targetPaths.pose_N32 $poseN32 $columns

    return [pscustomobject]@{
        InputRows = $inputRowCount
        DuplicateRows = $duplicateCount
        MergedRows = $records.Count
        BaselineRows = $baseline.Count
        StrongRigidRows = $strongRigid.Count
        PoseN32Rows = $poseN32.Count
        Columns = @($columns)
        Paths = $targetPaths
    }
}

function New-SyntheticRow(
    [string] $formulation,
    [string] $pose,
    [int] $n,
    [string] $interior,
    [string] $density,
    [string] $exterior,
    [string] $boundary,
    [int] $iterations,
    [string] $note) {
    return [ordered]@{
        case_id = "fixture_${formulation}_${pose}_$n"
        backend = 'topology_native'
        geometry = 'cylinder'
        formulation = $formulation
        trace_restrict_mode = Get-CanonicalTraceRestrictMode `
            'topology_native' $formulation
        pose = $pose
        N = [string] $n
        h = (3.0 / $n).ToString('R', $invariantCulture)
        iterations = [string] $iterations
        exterior_condition_linf = $exterior
        exterior_condition_order_linf = '123'
        boundary_residual_linf = $boundary
        boundary_residual_order_linf = '123'
        density_linf = $density
        density_order_linf = '123'
        interior_linf = $interior
        interior_order_linf = '123'
        note = $note
    }
}

function Assert-NearlyEqual([double] $actual, [double] $expected,
                            [string] $description) {
    if ([Math]::Abs($actual - $expected) -gt 1.0e-12) {
        throw "$description mismatch: actual=$actual expected=$expected"
    }
}

function Assert-Throws(
    [scriptblock] $action,
    [string] $description,
    [string] $expectedMessagePart) {
    $threw = $false
    $message = ''
    try {
        & $action | Out-Null
    } catch {
        $threw = $true
        $message = $_.Exception.Message
    }
    if (-not $threw) {
        throw "Synthetic merge test expected failure: $description"
    }
    if (-not [string]::IsNullOrWhiteSpace($expectedMessagePart) -and
        $message.IndexOf(
            $expectedMessagePart,
            [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw (
            "Synthetic merge test '$description' failed for the wrong " +
            "reason: $message")
    }
}

function New-StrictSyntheticRow(
    [string] $backend,
    [string] $geometry,
    [string] $formulation,
    [string] $pose,
    [int] $n) {
    $order = if ($n -eq 32) { 'NaN' } else { '2' }
    return [pscustomobject] [ordered]@{
        case_id = "fixture_${backend}_${geometry}_${formulation}_${pose}_$n"
        backend = $backend
        geometry = $geometry
        formulation = $formulation
        trace_restrict_mode = Get-CanonicalTraceRestrictMode `
            $backend $formulation
        pose = $pose
        N = [string] $n
        h = (3.0 / $n).ToString('R', $invariantCulture)
        iterations = '9'
        gmres_converged = 'True'
        physical_converged = if ($backend -eq 'general_cap') { '' } else { 'True' }
        gmres_relative_residual = '1e-11'
        operator_residual_linf = '2e-12'
        exterior_condition_linf = '3e-4'
        exterior_condition_order_linf = $order
        boundary_residual_linf = if ($formulation -eq 'dirichlet' -and
            $backend -eq 'topology_native') { '4e-4' } else { 'NaN' }
        boundary_residual_order_linf = $order
        density_linf = '5e-4'
        density_order_linf = $order
        interior_linf = '6e-4'
        interior_order_linf = $order
    }
}

function New-FullStrictSyntheticRows {
    $rows = [Collections.Generic.List[object]]::new()
    foreach ($geometryEntry in $canonicalGeometryCatalog) {
        $convergencePose = Get-ConvergencePoseForBackend `
            $geometryEntry.Backend
        foreach ($formulation in $canonicalFormulations) {
            foreach ($pose in $canonicalPoses) {
                $levels = if ($pose -eq 'baseline' -or
                    $pose -eq $convergencePose) {
                    $canonicalLevels
                } else {
                    @(32)
                }
                foreach ($n in $levels) {
                    $rows.Add((New-StrictSyntheticRow `
                        $geometryEntry.Backend `
                        $geometryEntry.Geometry `
                        $formulation `
                        $pose `
                        $n))
                }
            }
        }
    }
    return @($rows)
}

function Write-SyntheticTopologyRun(
    [string] $runDirectory,
    [string] $manifestStatus,
    [bool] $omitArchiveDirichlet,
    [bool] $omitAllResultsDirichlet) {
    New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
    $caseId = 'b_tc'
    $archiveDirectory = Join-Path (Join-Path $runDirectory 'raw') $caseId
    $manifestRow = [pscustomobject] [ordered]@{
        case_id = $caseId
        status = $manifestStatus
        backend = 'topology_native'
        selector = 'cylinder'
        pose = 'baseline'
        levels = '32'
        command = 'synthetic'
        exit_code = if ($manifestStatus -eq 'pending') { '-1' } else { '0' }
        message = ''
        stdout_log = ''
        stderr_log = ''
        archive_directory = $archiveDirectory
    }
    $manifestRow | Export-Csv -LiteralPath (
        Join-Path $runDirectory 'case_manifest.csv') `
        -NoTypeInformation -Encoding UTF8

    $allRows = [Collections.Generic.List[object]]::new()
    $allRows.Add([pscustomobject]@{
        case_id = $caseId
        backend = 'topology_native'
        geometry = 'cylinder'
        formulation = 'neumann'
        trace_restrict_mode = $topologyNeumannTraceRestrict
        pose = 'baseline'
        N = '32'
    })
    if (-not $omitAllResultsDirichlet) {
        $allRows.Add([pscustomobject]@{
            case_id = $caseId
            backend = 'topology_native'
            geometry = 'cylinder'
            formulation = 'dirichlet'
            trace_restrict_mode = $topologyDirichletTraceRestrict
            pose = 'baseline'
            N = '32'
        })
    }
    @($allRows) | Export-Csv -LiteralPath (
        Join-Path $runDirectory 'all_results.csv') `
        -NoTypeInformation -Encoding UTF8

    if ($manifestStatus -eq 'pending') {
        return
    }
    New-Item -ItemType Directory -Force -Path $archiveDirectory | Out-Null
    [pscustomobject]@{
        geometry = 'cylinder'
        trace_restrict_mode = $topologyNeumannTraceRestrict
        N = '32'
    } |
        Export-Csv -LiteralPath (
            Join-Path $archiveDirectory 'neumann_results.csv') `
            -NoTypeInformation -Encoding UTF8
    if (-not $omitArchiveDirichlet) {
        [pscustomobject]@{
            geometry = 'cylinder'
            trace_restrict_mode = $topologyDirichletTraceRestrict
            N = '32'
        } |
            Export-Csv -LiteralPath (
                Join-Path $archiveDirectory 'dirichlet_normal_results.csv') `
                -NoTypeInformation -Encoding UTF8
    }
    [pscustomobject] [ordered]@{
        case_id = $caseId
        status = 'complete'
        exit_code = 0
        backend = 'topology_native'
        selector = 'cylinder'
        expected_geometries = @('cylinder')
        pose = 'baseline'
        levels = @(32)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
        Join-Path $archiveDirectory 'case_status.json') -Encoding UTF8
}

function Write-SyntheticGeneralCapRun([string] $runDirectory) {
    New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
    $caseId = 'b_gc'
    $archiveDirectory = Join-Path (Join-Path $runDirectory 'raw') $caseId
    New-Item -ItemType Directory -Force -Path $archiveDirectory | Out-Null
    [pscustomobject] [ordered]@{
        case_id = $caseId
        status = 'resumed'
        backend = 'general_cap'
        selector = 'sphere'
        pose = 'baseline'
        levels = '32'
        command = 'synthetic'
        exit_code = '0'
        message = ''
        stdout_log = ''
        stderr_log = ''
        archive_directory = $archiveDirectory
    } | Export-Csv -LiteralPath (
        Join-Path $runDirectory 'case_manifest.csv') `
        -NoTypeInformation -Encoding UTF8
    @(
        [pscustomobject]@{
            case_id = $caseId; backend = 'general_cap'; geometry = 'sphere'
            formulation = 'neumann'; trace_restrict_mode = 'not_applicable'
            pose = 'baseline'; N = '32'
        },
        [pscustomobject]@{
            case_id = $caseId; backend = 'general_cap'; geometry = 'sphere'
            formulation = 'dirichlet'; trace_restrict_mode = 'not_applicable'
            pose = 'baseline'; N = '32'
        }
    ) | Export-Csv -LiteralPath (
        Join-Path $runDirectory 'all_results.csv') `
        -NoTypeInformation -Encoding UTF8
    [pscustomobject]@{ shape = 'sphere'; pose = 'baseline'; N = '32' } |
        Export-Csv -LiteralPath (Join-Path $archiveDirectory 'refinement.csv') `
            -NoTypeInformation -Encoding UTF8
    [pscustomobject] [ordered]@{
        case_id = $caseId
        status = 'complete'
        exit_code = 0
        backend = 'general_cap'
        selector = 'sphere'
        expected_geometries = @('sphere')
        pose = 'baseline'
        levels = @(32)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
        Join-Path $archiveDirectory 'case_status.json') -Encoding UTF8
}

function Invoke-SyntheticCsvTest {
    $tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    $fixtureRoot = Join-Path $tempBase (
        'kfbim_merge_fixture_' + [Guid]::NewGuid().ToString('N'))
    $resolvedValidationRoot = [IO.Path]::GetFullPath($validationRoot).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    $resolvedFixtureRoot = [IO.Path]::GetFullPath($fixtureRoot)
    if ($resolvedFixtureRoot.StartsWith(
            $resolvedValidationRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Synthetic fixture unexpectedly resolved inside the formal validation directory'
    }

    New-Item -ItemType Directory -Force -Path $fixtureRoot | Out-Null
    try {
        $runA = Join-Path $fixtureRoot 'run_a'
        $runB = Join-Path $fixtureRoot 'run_b'
        $merged = Join-Path $fixtureRoot 'merged'
        New-Item -ItemType Directory -Force -Path $runA | Out-Null
        New-Item -ItemType Directory -Force -Path $runB | Out-Null

        $rowsA = @(
            [pscustomobject] (New-SyntheticRow `
                'neumann' 'baseline' 32 '.04' '.08' '.16' '.32' 11 'first'),
            [pscustomobject] (New-SyntheticRow `
                'neumann' 'baseline' 64 '.02' '.04' '.08' '.16' 99 'stale'),
            [pscustomobject] (New-SyntheticRow `
                'dirichlet' 'baseline' 32 '.2' '.3' '.4' '.5' 7 'finite'),
            [pscustomobject] (New-SyntheticRow `
                'dirichlet' 'baseline' 64 'NaN' '0' 'Infinity' '-1' 8 'invalid')
        ) | ForEach-Object {
            $_ | Add-Member -NotePropertyName legacy_only -NotePropertyValue 'retained'
            $_
        }
        $rowsA | Export-Csv -LiteralPath (
            Join-Path $runA 'all_results.csv') -NoTypeInformation -Encoding UTF8

        $replacementNote = "replacement, `"quoted`"`r`nsecond line"
        $rowsB = @(
            [pscustomobject] (New-SyntheticRow `
                'neumann' 'baseline' 64 '.01' '.02' '.04' '.08' 17 $replacementNote),
            [pscustomobject] (New-SyntheticRow `
                'neumann' 'baseline' 128 '.0025' '.005' '.01' '.02' 18 'fine'),
            [pscustomobject] (New-SyntheticRow `
                'neumann' $generalCapConvergencePose 32 '.05' '.1' '.2' '.4' 13 'cap-rigid'),
            [pscustomobject] (New-SyntheticRow `
                'neumann' 'tx_p0137' 32 '.06' '.12' '.24' '.48' 14 'pose'),
            [pscustomobject] (New-SyntheticRow `
                'neumann' 'tx_p0137' 64 '.03' '.06' '.12' '.24' 15 'bad-h')
        ) | ForEach-Object {
            $_ | Add-Member -NotePropertyName replacement_only -NotePropertyValue 'new'
            $_
        }
        # Keep h unchanged between the two translated-pose levels.  Its four
        # errors are finite and positive, so NaN orders here specifically test
        # rejection of a non-decreasing grid spacing.
        ($rowsB | Where-Object {
            $_.pose -eq 'tx_p0137' -and $_.N -eq '64'
        }).h = (3.0 / 32.0).ToString('R', $invariantCulture)
        $rowsB | Export-Csv -LiteralPath (
            Join-Path $runB 'all_results.csv') -NoTypeInformation -Encoding UTF8

        $summary = Merge-ValidationCsvs @(
            Join-Path $runA 'all_results.csv'
            Join-Path $runB 'all_results.csv') $merged $false $true

        if ($summary.InputRows -ne 9 -or $summary.DuplicateRows -ne 1 -or
            $summary.MergedRows -ne 8) {
            throw "Synthetic de-duplication counts are wrong: $($summary | ConvertTo-Json -Compress)"
        }
        if ($summary.BaselineRows -ne 5 -or
            $summary.StrongRigidRows -ne 2 -or
            $summary.PoseN32Rows -ne 4) {
            throw "Synthetic category counts are wrong: $($summary | ConvertTo-Json -Compress)"
        }

        # Identical physical cases produced by different restrict routes must
        # remain separate records; route changes are not replacement updates.
        $routeInputDirectory = Join-Path $fixtureRoot 'route_identity_input'
        New-Item -ItemType Directory -Force -Path $routeInputDirectory |
            Out-Null
        $routeInput = Join-Path $routeInputDirectory 'all_results.csv'
        $routeCurrent = [pscustomobject] (New-SyntheticRow `
            'neumann' 'baseline' 32 '.04' '.08' '.16' '.32' 11 'q27')
        $routeAlternative = [pscustomobject] (New-SyntheticRow `
            'neumann' 'baseline' 32 '.05' '.09' '.17' '.33' 12 'q10')
        $routeAlternative.trace_restrict_mode =
            'shared_q10_cubic_gridline_cauchy'
        @($routeCurrent, $routeAlternative) |
            Export-Csv -LiteralPath $routeInput -NoTypeInformation -Encoding UTF8
        $routeSummary = Merge-ValidationCsvs `
            @($routeInput) `
            (Join-Path $fixtureRoot 'route_identity_merged') `
            $false `
            $true
        if ($routeSummary.InputRows -ne 2 -or
            $routeSummary.DuplicateRows -ne 0 -or
            $routeSummary.MergedRows -ne 2) {
            throw (
                'Different trace_restrict_mode rows were incorrectly ' +
                "coalesced: $($routeSummary | ConvertTo-Json -Compress)")
        }

        $roundTrip = @(Import-Csv -LiteralPath $summary.Paths.merged -Encoding UTF8)
        $n64 = @($roundTrip | Where-Object {
            $_.formulation -eq 'neumann' -and
            $_.pose -eq 'baseline' -and $_.N -eq '64'
        })
        if ($n64.Count -ne 1 -or $n64[0].iterations -ne '17') {
            throw 'Later duplicate did not replace the synthetic N=64 row'
        }
        if ($n64[0].legacy_only -ne 'retained' -or
            $n64[0].replacement_only -ne 'new') {
            throw 'Union-schema field preservation failed for the duplicate row'
        }
        if ($n64[0].note -ne $replacementNote) {
            throw 'RFC CSV quote/newline round trip failed'
        }
        if ($n64[0].interior_linf -ne '.01') {
            throw 'An original error column was unexpectedly modified'
        }
        foreach ($orderColumn in @(
                'interior_order_linf',
                'density_order_linf',
                'exterior_condition_order_linf',
                'boundary_residual_order_linf')) {
            Assert-NearlyEqual (
                [double]::Parse($n64[0].$orderColumn, $invariantCulture)) 2.0 (
                "Synthetic $orderColumn at N=64")
        }

        $n128 = @($roundTrip | Where-Object {
            $_.formulation -eq 'neumann' -and
            $_.pose -eq 'baseline' -and $_.N -eq '128'
        })[0]
        foreach ($orderColumn in @(
                'interior_order_linf',
                'density_order_linf',
                'exterior_condition_order_linf',
                'boundary_residual_order_linf')) {
            Assert-NearlyEqual (
                [double]::Parse($n128.$orderColumn, $invariantCulture)) 2.0 (
                "Synthetic $orderColumn at N=128")
        }

        $invalid = @($roundTrip | Where-Object {
            $_.formulation -eq 'dirichlet' -and $_.N -eq '64'
        })[0]
        foreach ($orderColumn in @(
                'interior_order_linf',
                'density_order_linf',
                'exterior_condition_order_linf',
                'boundary_residual_order_linf')) {
            if ($invalid.$orderColumn -ne 'NaN') {
                throw "Invalid synthetic error did not yield NaN in $orderColumn"
            }
        }

        $nonDecreasingH = @($roundTrip | Where-Object {
            $_.pose -eq 'tx_p0137' -and $_.N -eq '64'
        })[0]
        foreach ($orderColumn in @(
                'interior_order_linf',
                'density_order_linf',
                'exterior_condition_order_linf',
                'boundary_residual_order_linf')) {
            if ($nonDecreasingH.$orderColumn -ne 'NaN') {
                throw "Non-decreasing synthetic h did not yield NaN in $orderColumn"
            }
        }

        foreach ($path in $summary.Paths.Values) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Synthetic category CSV is missing: $path"
            }
        }

        # Strict publication-mode matrix: 7 geometries x 2 formulations,
        # baseline/backend-specific convergence pose at three levels and
        # every pose at N=32.
        $strictRows = @(New-FullStrictSyntheticRows)
        $strictInput = Join-Path $fixtureRoot 'strict_input'
        New-Item -ItemType Directory -Force -Path $strictInput | Out-Null
        $strictCsv = Join-Path $strictInput 'all_results.csv'
        $strictRows | Export-Csv -LiteralPath $strictCsv `
            -NoTypeInformation -Encoding UTF8
        $strictSummary = Merge-ValidationCsvs `
            @($strictCsv) `
            (Join-Path $fixtureRoot 'strict_merged') `
            $false `
            $false
        if ($strictSummary.MergedRows -ne 168 -or
            $strictSummary.BaselineRows -ne 42 -or
            $strictSummary.StrongRigidRows -ne 42 -or
            $strictSummary.PoseN32Rows -ne 112) {
            throw (
                "Strict synthetic matrix counts are wrong: " +
                "$($strictSummary | ConvertTo-Json -Compress)")
        }

        $strictStrongRows = @(
            Import-Csv -LiteralPath $strictSummary.Paths.strong_rigid `
                -Encoding UTF8)
        $strictCapStrong = @($strictStrongRows | Where-Object {
            (Test-SameText $_.backend 'general_cap') -and
            (Test-SameText $_.pose $generalCapConvergencePose)
        }).Count
        $strictTopologyStrong = @($strictStrongRows | Where-Object {
            (Test-SameText $_.backend 'topology_native') -and
            (Test-SameText $_.pose $topologyConvergencePose)
        }).Count
        $strictWrongStrong = @($strictStrongRows | Where-Object {
            -not (((Test-SameText $_.backend 'general_cap') -and
                    (Test-SameText $_.pose $generalCapConvergencePose)) -or
                ((Test-SameText $_.backend 'topology_native') -and
                    (Test-SameText $_.pose $topologyConvergencePose)))
        }).Count
        if ($strictCapStrong -ne 18 -or
            $strictTopologyStrong -ne 24 -or
            $strictWrongStrong -ne 0) {
            throw (
                'Strict backend-specific convergence rows are wrong: ' +
                "general_cap=$strictCapStrong/18 " +
                "topology_native=$strictTopologyStrong/24 " +
                "wrong_pose=$strictWrongStrong/0")
        }

        $missingInput = Join-Path $fixtureRoot 'missing_input'
        New-Item -ItemType Directory -Force -Path $missingInput | Out-Null
        $missingGeometryCsv = Join-Path $missingInput 'all_results.csv'
        @($strictRows | Where-Object {
            -not ((Test-SameText $_.backend 'topology_native') -and
                (Test-SameText $_.geometry 'u_prism'))
        }) | Export-Csv -LiteralPath $missingGeometryCsv `
            -NoTypeInformation -Encoding UTF8
        Assert-Throws {
            Merge-ValidationCsvs `
                @($missingGeometryCsv) `
                (Join-Path $fixtureRoot 'must_not_merge_missing') `
                $false `
                $false
        } 'whole geometry missing' 'primary-key mismatch'

        $duplicateInput = Join-Path $fixtureRoot 'duplicate_input'
        New-Item -ItemType Directory -Force -Path $duplicateInput | Out-Null
        $duplicateCsv = Join-Path $duplicateInput 'all_results.csv'
        @($strictRows) + @($strictRows[0]) |
            Export-Csv -LiteralPath $duplicateCsv `
                -NoTypeInformation -Encoding UTF8
        Assert-Throws {
            Merge-ValidationCsvs `
                @($duplicateCsv) `
                (Join-Path $fixtureRoot 'must_not_merge_duplicate') `
                $false `
                $false
        } 'duplicate strict input key' 'rejects duplicate'

        # Source-run integrity checks exercise active/pending manifests and
        # both sides of the archive-vs-normalized-key comparison.
        $completeRun = Join-Path $fixtureRoot 'source_complete'
        Write-SyntheticTopologyRun $completeRun 'complete' $false $false
        $completeSource = Assert-CompletedRunSource `
            $completeRun (Join-Path $completeRun 'all_results.csv')
        if ($completeSource.Cases -ne 1 -or $completeSource.Keys -ne 2) {
            throw 'Completed source fixture validation counts are wrong'
        }

        $resumedGeneralRun = Join-Path $fixtureRoot 'source_resumed_general'
        Write-SyntheticGeneralCapRun $resumedGeneralRun
        $resumedGeneralSource = Assert-CompletedRunSource `
            $resumedGeneralRun `
            (Join-Path $resumedGeneralRun 'all_results.csv')
        if ($resumedGeneralSource.Cases -ne 1 -or
            $resumedGeneralSource.Keys -ne 2) {
            throw 'Resumed general-cap source fixture validation counts are wrong'
        }

        $pendingRun = Join-Path $fixtureRoot 'source_pending'
        Write-SyntheticTopologyRun $pendingRun 'pending' $false $false
        Assert-Throws {
            Assert-CompletedRunSource `
                $pendingRun (Join-Path $pendingRun 'all_results.csv')
        } 'partial active manifest' 'not complete'

        $missingNormalizedRun = Join-Path $fixtureRoot 'source_missing_normalized'
        Write-SyntheticTopologyRun `
            $missingNormalizedRun 'complete' $false $true
        Assert-Throws {
            Assert-CompletedRunSource `
                $missingNormalizedRun `
                (Join-Path $missingNormalizedRun 'all_results.csv')
        } 'incomplete normalized primary keys' 'primary-key mismatch'

        $missingArchiveRun = Join-Path $fixtureRoot 'source_missing_archive'
        Write-SyntheticTopologyRun `
            $missingArchiveRun 'complete' $true $false
        Assert-Throws {
            Assert-CompletedRunSource `
                $missingArchiveRun `
                (Join-Path $missingArchiveRun 'all_results.csv')
        } 'incomplete archive primary keys' 'Missing archived result'

        Write-Host (
            "Synthetic merge CSV test passed: partial_rows=" +
            "$($summary.MergedRows) strict_rows=$($strictSummary.MergedRows) " +
            "source_cases=$($completeSource.Cases + $resumedGeneralSource.Cases)")
    } finally {
        $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
        $safePrefix = $tempBase + [IO.Path]::DirectorySeparatorChar +
            'kfbim_merge_fixture_'
        if ($resolvedFixture.StartsWith(
                $safePrefix,
                [StringComparison]::OrdinalIgnoreCase) -and
            (Test-Path -LiteralPath $resolvedFixture)) {
            Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
        }
    }
}

if ($SyntheticCsvTest) {
    Invoke-SyntheticCsvTest
    exit 0
}

if ($RunLabel.Count -eq 0) {
    throw 'Pass one or more completed run labels with -RunLabel, or use -SyntheticCsvTest'
}

$inputPaths = [System.Collections.Generic.List[string]]::new()
$inputRunDirectories = [System.Collections.Generic.List[string]]::new()
$seenLabels = @{}
$expandedRunLabels = [System.Collections.Generic.List[string]]::new()
foreach ($argumentValue in $RunLabel) {
    foreach ($labelPart in ([string] $argumentValue -split ',')) {
        if (-not [string]::IsNullOrWhiteSpace($labelPart)) {
            $expandedRunLabels.Add($labelPart.Trim())
        }
    }
}
foreach ($labelValue in $expandedRunLabels) {
    if ([string]::IsNullOrWhiteSpace($labelValue) -or
        $labelValue -notmatch '^[A-Za-z0-9_.-]+$' -or
        $labelValue -eq '.' -or $labelValue -eq '..') {
        throw "Invalid RunLabel '$labelValue'; use the exact filename-safe run directory name"
    }
    if ($seenLabels.ContainsKey($labelValue)) {
        continue
    }
    $seenLabels[$labelValue] = $true
    $runDirectory = Join-Path $validationRoot $labelValue
    $inputRunDirectories.Add($runDirectory)
    $inputPaths.Add((Join-Path $runDirectory 'all_results.csv'))
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $validationRoot (
        'merged_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
} elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repo $OutputDirectory
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)

if (-not $AllowPartial) {
    for ($i = 0; $i -lt $inputPaths.Count; ++$i) {
        $sourceSummary = Assert-CompletedRunSource `
            $inputRunDirectories[$i] $inputPaths[$i]
        Write-Host (
            "Validated completed source: run=" +
            "$($inputRunDirectories[$i]) cases=$($sourceSummary.Cases) " +
            "keys=$($sourceSummary.Keys)")
    }
}

$result = Merge-ValidationCsvs `
    @($inputPaths) `
    $OutputDirectory `
    ([bool] $Force) `
    ([bool] $AllowPartial)
Write-Host (
    "Merged KFBI 3-D validation reports: input_rows=$($result.InputRows) " +
    "duplicates=$($result.DuplicateRows) merged_rows=$($result.MergedRows)")
Write-Host (
    "Categories: baseline=$($result.BaselineRows) " +
    "strong_rigid=$($result.StrongRigidRows) pose_N32=$($result.PoseN32Rows)")
foreach ($name in $result.Paths.Keys) {
    Write-Host "  $name=$($result.Paths[$name])"
}
