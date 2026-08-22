<#
.SYNOPSIS
Generates a detailed Markdown report from a merged KFBI 3-D validation CSV.

.DESCRIPTION
This is a read-only CSV-to-Markdown postprocessor.  It never starts either
solver and never modifies the input CSV.  The expected input is the
merged_all_results.csv written by merge_kfbi_3d_validation_reports.ps1.

The report contains detailed baseline and backend-specific convergence-rigid
tables for Dirichlet and Neumann formulations.  general_cap uses R17+T1 and
topology_native uses Tx.  It also contains row-wise N=32 pose details, N=32
eight-pose matrices, convergence status statistics, coverage checks, and
explicit missing/NaN diagnostics.

By default the input must be the exact fixed seven-geometry/168-row publication
matrix.  Use -AllowPartial only to inspect an intentionally incomplete ad-hoc
CSV; a report intended for publication should keep the strict default.

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  apps/generate_kfbi_3d_validation_report.ps1 `
  -InputCsv output/kfbi_3d_full_validation/merged_final/merged_all_results.csv `
  -OutputMarkdown docs/KFBI3D_Dirichlet_Neumann_Seven_Geometry_Validation.md

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  apps/generate_kfbi_3d_validation_report.ps1 -SelfTest
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $InputCsv = '',

    [Parameter(Position = 1)]
    [string] $OutputMarkdown = '',

    [switch] $Force,
    [switch] $AllowPartial,
    [switch] $SelfTest
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$invariantCulture = [Globalization.CultureInfo]::InvariantCulture
$generalCapConvergencePose = 'rot_axis123_17deg_t_xyz_1'
$topologyConvergencePose = 'tx_p0137'
$convergencePoseCatalog = @(
    [pscustomobject]@{
        Backend = 'general_cap'
        Pose = $generalCapConvergencePose
        Short = 'R17+T1'
    },
    [pscustomobject]@{
        Backend = 'topology_native'
        Pose = $topologyConvergencePose
        Short = 'Tx'
    }
)
$canonicalLevels = @(32, 64, 128)
$canonicalFormulations = @('dirichlet', 'neumann')
$canonicalGeometryCatalog = @(
    [pscustomobject]@{ Backend = 'general_cap'; Geometry = 'sphere' },
    [pscustomobject]@{ Backend = 'general_cap'; Geometry = 'ellipsoid' },
    [pscustomobject]@{ Backend = 'general_cap'; Geometry = 'flower' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'torus' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'cylinder' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'l_prism' },
    [pscustomobject]@{ Backend = 'topology_native'; Geometry = 'u_prism' }
)
$canonicalPoses = @(
    [pscustomobject]@{ Id = 'baseline'; Short = 'base' },
    [pscustomobject]@{ Id = 'tx_p0137'; Short = 'Tx' },
    [pscustomobject]@{ Id = 'ty_m0083'; Short = 'Ty' },
    [pscustomobject]@{ Id = 'tz_p0061'; Short = 'Tz' },
    [pscustomobject]@{ Id = 't_xyz_1'; Short = 'T1' },
    [pscustomobject]@{ Id = 't_xyz_2'; Short = 'T2' },
    [pscustomobject]@{ Id = 'rot_axis123_17deg'; Short = 'R17' },
    [pscustomobject]@{
        Id = 'rot_axis123_17deg_t_xyz_1'
        Short = 'R17+T1'
    }
)
$metricSpecifications = @(
    [pscustomobject]@{ Label = '内部误差'; Error = 'interior_linf'; Order = 'interior_order_linf' },
    [pscustomobject]@{ Label = '密度误差'; Error = 'density_linf'; Order = 'density_order_linf' },
    [pscustomobject]@{ Label = '外迹条件误差'; Error = 'exterior_condition_linf'; Order = 'exterior_condition_order_linf' },
    [pscustomobject]@{ Label = '边界残差'; Error = 'boundary_residual_linf'; Order = 'boundary_residual_order_linf' }
)

function Test-SameText([string] $left, [string] $right) {
    return [string]::Equals(
        $left, $right, [StringComparison]::OrdinalIgnoreCase)
}

function Get-RowValue($row, [string] $name) {
    foreach ($property in $row.PSObject.Properties) {
        if (Test-SameText $property.Name $name) {
            return $property.Value
        }
    }
    return $null
}

function Get-TrimmedText($value) {
    if ($null -eq $value) {
        return ''
    }
    return ([string] $value).Trim()
}

function Get-ConvergencePoseEntry([string] $backend) {
    foreach ($entry in $convergencePoseCatalog) {
        if (Test-SameText $entry.Backend $backend) {
            return $entry
        }
    }
    return $null
}

function Get-ConvergencePoseForBackend([string] $backend) {
    $entry = Get-ConvergencePoseEntry $backend
    if ($null -eq $entry) {
        return ''
    }
    return [string] $entry.Pose
}

function Test-IsBackendConvergenceRow($row) {
    $backend = Get-TrimmedText (Get-RowValue $row 'backend')
    $expectedPose = Get-ConvergencePoseForBackend $backend
    if ([string]::IsNullOrWhiteSpace($expectedPose)) {
        return $false
    }
    return Test-SameText `
        (Get-TrimmedText (Get-RowValue $row 'pose')) `
        $expectedPose
}

function Convert-ToMarkdownCell($value) {
    $text = Get-TrimmedText $value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return [string] $([char] 0x2014)
    }
    $text = $text.Replace('|', '\|')
    $text = $text.Replace("`r`n", '<br>')
    $text = $text.Replace("`n", '<br>')
    $text = $text.Replace("`r", '<br>')
    return $text
}

function Try-ParseDouble($value, [ref] $number) {
    $text = Get-TrimmedText $value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $false
    }
    $parsed = 0.0
    $ok = [double]::TryParse(
        $text,
        [Globalization.NumberStyles]::Float,
        $invariantCulture,
        [ref] $parsed)
    if (-not $ok) {
        return $false
    }
    $number.Value = $parsed
    return $true
}

function Format-Scientific($value) {
    $text = Get-TrimmedText $value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return [string] $([char] 0x2014)
    }
    $number = 0.0
    if (-not (Try-ParseDouble $text ([ref] $number))) {
        return Convert-ToMarkdownCell $text
    }
    if ([double]::IsNaN($number)) {
        return 'NaN'
    }
    if ([double]::IsPositiveInfinity($number)) {
        return '+Inf'
    }
    if ([double]::IsNegativeInfinity($number)) {
        return '-Inf'
    }
    return $number.ToString('0.000E+00', $invariantCulture)
}

function Format-Order($value) {
    $text = Get-TrimmedText $value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return [string] $([char] 0x2014)
    }
    $number = 0.0
    if (-not (Try-ParseDouble $text ([ref] $number))) {
        return Convert-ToMarkdownCell $text
    }
    if ([double]::IsNaN($number)) {
        return 'NaN'
    }
    if ([double]::IsInfinity($number)) {
        return if ($number -gt 0.0) { '+Inf' } else { '-Inf' }
    }
    return $number.ToString('0.000', $invariantCulture)
}

function Format-TraceRatio($value) {
    return Format-Order $value
}

function Format-Integer($value) {
    $text = Get-TrimmedText $value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return [string] $([char] 0x2014)
    }
    $number = 0
    if ([int]::TryParse(
            $text,
            [Globalization.NumberStyles]::Integer,
            $invariantCulture,
            [ref] $number)) {
        return $number.ToString($invariantCulture)
    }
    return Convert-ToMarkdownCell $text
}

function Get-TriState($value) {
    $text = (Get-TrimmedText $value).ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return -1
    }
    if ($text -eq 'true' -or $text -eq '1' -or $text -eq 'yes') {
        return 1
    }
    if ($text -eq 'false' -or $text -eq '0' -or $text -eq 'no') {
        return 0
    }
    return -1
}

function Format-TriState($value) {
    $state = Get-TriState $value
    if ($state -eq 1) {
        return '是'
    }
    if ($state -eq 0) {
        return '否'
    }
    return '未提供'
}

function Get-IntegerValue($row, [string] $name) {
    $text = Get-TrimmedText (Get-RowValue $row $name)
    $number = 0
    if ([int]::TryParse(
            $text,
            [Globalization.NumberStyles]::Integer,
            $invariantCulture,
            [ref] $number)) {
        return $number
    }
    return [int]::MaxValue
}

function Get-NameRank([string] $name, [string[]] $canonicalNames) {
    for ($i = 0; $i -lt $canonicalNames.Count; ++$i) {
        if (Test-SameText $name $canonicalNames[$i]) {
            return $i
        }
    }
    return 1000
}

function Get-SortedRows([object[]] $rows) {
    $geometryOrder = @(
        'sphere', 'ellipsoid', 'flower', 'torus',
        'cylinder', 'l_prism', 'u_prism')
    return @($rows | Sort-Object `
        @{ Expression = { Get-NameRank (Get-TrimmedText (Get-RowValue $_ 'backend')) @('topology_native', 'general_cap') } }, `
        @{ Expression = { Get-NameRank (Get-TrimmedText (Get-RowValue $_ 'geometry')) $geometryOrder } }, `
        @{ Expression = { (Get-TrimmedText (Get-RowValue $_ 'geometry')).ToLowerInvariant() } }, `
        @{ Expression = { Get-NameRank (Get-TrimmedText (Get-RowValue $_ 'formulation')) $canonicalFormulations } }, `
        @{ Expression = { Get-NameRank (Get-TrimmedText (Get-RowValue $_ 'pose')) @($canonicalPoses | ForEach-Object { $_.Id }) } }, `
        @{ Expression = { Get-IntegerValue $_ 'N' } })
}

function Get-CaseLabel($row) {
    $backend = Convert-ToMarkdownCell (Get-RowValue $row 'backend')
    $geometry = Convert-ToMarkdownCell (Get-RowValue $row 'geometry')
    return "$backend / $geometry"
}

function Get-DofText($row) {
    $pre = Get-TrimmedText (Get-RowValue $row 'pre_mean_dofs')
    $final = Get-TrimmedText (Get-RowValue $row 'final_dofs')
    $dofs = Get-TrimmedText (Get-RowValue $row 'dofs')
    if ([string]::IsNullOrWhiteSpace($pre)) {
        $pre = $dofs
    }
    if ([string]::IsNullOrWhiteSpace($final)) {
        $final = $dofs
    }
    if ([string]::IsNullOrWhiteSpace($pre) -and
        [string]::IsNullOrWhiteSpace($final)) {
        return [string] $([char] 0x2014)
    }
    return "$(Format-Integer $pre)$([char] 0x2192)$(Format-Integer $final)"
}

function Get-DirichletBoundaryResidualText($row) {
    $formulation = Get-TrimmedText (Get-RowValue $row 'formulation')
    if (-not (Test-SameText $formulation 'dirichlet')) {
        return 'N/A'
    }
    $value = Get-RowValue $row 'boundary_residual_linf'
    $text = Get-TrimmedText $value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return 'N/A'
    }
    $number = 0.0
    if ((Try-ParseDouble $text ([ref] $number)) -and
        [double]::IsNaN($number)) {
        return 'N/A'
    }
    return Format-Scientific $value
}

function Add-DetailedTable(
    [Collections.Generic.List[string]] $lines,
    [object[]] $rows,
    [string] $formulation) {
    $selected = @(Get-SortedRows @($rows | Where-Object {
        Test-SameText (Get-TrimmedText (Get-RowValue $_ 'formulation')) $formulation
    }))
    $displayName = if ($formulation -eq 'dirichlet') {
        'Dirichlet'
    } else {
        'Neumann'
    }
    $lines.Add("### $displayName")
    $lines.Add('')
    if ($selected.Count -eq 0) {
        $lines.Add('该姿态没有对应记录。')
        $lines.Add('')
        return
    }

    $lines.Add('| 后端 / 几何 | N | h | Eint | pint | Eρ | pρ | Eext | pext | Ebc | pbc | GMRES it | relres | op-res | ncoef | DOF（pre→final） | trace samples | trace final DOF | margin | ratio | 代数收敛 | physical |')
    $alignment = @('---') + @(1..19 | ForEach-Object { '---:' }) +
        @(':---:', ':---:')
    $lines.Add('| ' + ($alignment -join ' | ') + ' |')
    foreach ($row in $selected) {
        $cells = @(
            (Get-CaseLabel $row)
            (Format-Integer (Get-RowValue $row 'N'))
            (Format-Scientific (Get-RowValue $row 'h'))
            (Format-Scientific (Get-RowValue $row 'interior_linf'))
            (Format-Order (Get-RowValue $row 'interior_order_linf'))
            (Format-Scientific (Get-RowValue $row 'density_linf'))
            (Format-Order (Get-RowValue $row 'density_order_linf'))
            (Format-Scientific (Get-RowValue $row 'exterior_condition_linf'))
            (Format-Order (Get-RowValue $row 'exterior_condition_order_linf'))
            (Format-Scientific (Get-RowValue $row 'boundary_residual_linf'))
            (Format-Order (Get-RowValue $row 'boundary_residual_order_linf'))
            (Format-Integer (Get-RowValue $row 'iterations'))
            (Format-Scientific (Get-RowValue $row 'gmres_relative_residual'))
            (Format-Scientific (Get-RowValue $row 'operator_residual_linf'))
            (Format-Integer (Get-RowValue $row 'ncoef'))
            (Get-DofText $row)
            (Format-Integer (Get-RowValue $row 'trace_samples'))
            (Format-Integer (Get-RowValue $row 'trace_final_dofs'))
            (Format-Integer (Get-RowValue $row 'trace_oversampling_margin'))
            (Format-TraceRatio (
                Get-RowValue $row 'trace_oversampling_ratio'))
            (Format-TriState (Get-RowValue $row 'gmres_converged'))
            (Format-TriState (Get-RowValue $row 'physical_converged'))
        )
        $lines.Add('| ' + ($cells -join ' | ') + ' |')
    }
    $lines.Add('')
}

function Add-ConvergencePoseSection(
    [Collections.Generic.List[string]] $lines,
    [object[]] $allRows,
    [string] $heading,
    [string] $pose,
    [string] $backend = '') {
    $lines.Add("## $heading")
    $lines.Add('')
    $scope = if ([string]::IsNullOrWhiteSpace($backend)) {
        '全部后端'
    } else {
        "后端 ``$backend``"
    }
    $lines.Add("范围：$scope；姿态 ID：``$pose``。以下阶数由同一后端、几何、方程和姿态的相邻网格层计算；首个有效层没有前驱，阶通常为 `NaN`。")
    $lines.Add('')
    $poseRows = @($allRows | Where-Object {
        (Test-SameText `
            (Get-TrimmedText (Get-RowValue $_ 'pose')) $pose) -and
        ([string]::IsNullOrWhiteSpace($backend) -or
            (Test-SameText `
                (Get-TrimmedText (Get-RowValue $_ 'backend')) $backend))
    })
    foreach ($formulation in $canonicalFormulations) {
        Add-DetailedTable $lines $poseRows $formulation
    }
}

function Get-GeometryKeys([object[]] $rows) {
    $keys = @{}
    foreach ($row in $rows) {
        $backend = Get-TrimmedText (Get-RowValue $row 'backend')
        $geometry = Get-TrimmedText (Get-RowValue $row 'geometry')
        if ([string]::IsNullOrWhiteSpace($backend) -or
            [string]::IsNullOrWhiteSpace($geometry)) {
            continue
        }
        $key = $backend.ToLowerInvariant() + [char] 31 +
            $geometry.ToLowerInvariant()
        if (-not $keys.ContainsKey($key)) {
            $keys[$key] = [pscustomobject]@{
                Key = $key
                Backend = $backend
                Geometry = $geometry
            }
        }
    }
    $dummyRows = @($keys.Values | ForEach-Object {
        [pscustomobject]@{
            backend = $_.Backend
            geometry = $_.Geometry
            formulation = 'dirichlet'
            pose = 'baseline'
            N = 32
            geometry_key_record = $_
        }
    })
    return @(Get-SortedRows $dummyRows | ForEach-Object {
        $_.geometry_key_record
    })
}

function Add-PoseMatrix(
    [Collections.Generic.List[string]] $lines,
    [object[]] $allRows,
    [string] $formulation,
    [object[]] $geometryKeys) {
    $displayName = if ($formulation -eq 'dirichlet') {
        'Dirichlet'
    } else {
        'Neumann'
    }
    $lines.Add("### $displayName：Eint（GMRES it）")
    $lines.Add('')
    $header = @('后端 / 几何') + @($canonicalPoses | ForEach-Object { $_.Short })
    $alignment = @('---') + @($canonicalPoses | ForEach-Object { '---:' })
    $lines.Add('| ' + ($header -join ' | ') + ' |')
    $lines.Add('| ' + ($alignment -join ' | ') + ' |')

    # Build the N=32/formulation lookup once.  Re-filtering every input row for
    # each of the 7 x 8 table cells is disproportionately slow in Windows
    # PowerShell 5, especially for the full 168-row publication matrix.
    $rowsByCell = @{}
    foreach ($row in $allRows) {
        if ((Get-IntegerValue $row 'N') -ne 32 -or
            -not (Test-SameText `
                (Get-TrimmedText (Get-RowValue $row 'formulation')) `
                $formulation)) {
            continue
        }
        $cellKey = (
            (Get-TrimmedText (Get-RowValue $row 'backend')).ToLowerInvariant() +
            [char] 31 +
            (Get-TrimmedText (Get-RowValue $row 'geometry')).ToLowerInvariant() +
            [char] 31 +
            (Get-TrimmedText (Get-RowValue $row 'pose')).ToLowerInvariant())
        if (-not $rowsByCell.ContainsKey($cellKey)) {
            $rowsByCell[$cellKey] =
                [Collections.Generic.List[object]]::new()
        }
        $rowsByCell[$cellKey].Add($row)
    }
    foreach ($geometryKey in $geometryKeys) {
        $cells = [Collections.Generic.List[string]]::new()
        $cells.Add("$($geometryKey.Backend) / $($geometryKey.Geometry)")
        foreach ($pose in $canonicalPoses) {
            $cellKey = $geometryKey.Backend.ToLowerInvariant() +
                [char] 31 + $geometryKey.Geometry.ToLowerInvariant() +
                [char] 31 + $pose.Id.ToLowerInvariant()
            $matches = if ($rowsByCell.ContainsKey($cellKey)) {
                @($rowsByCell[$cellKey])
            } else {
                @()
            }
            if ($matches.Count -eq 0) {
                $cells.Add([string] $([char] 0x2014))
                continue
            }
            if ($matches.Count -gt 1) {
                $cells.Add("重复($($matches.Count))")
                continue
            }
            $row = $matches[0]
            $marker = if ((Get-TriState (Get-RowValue $row 'gmres_converged')) -eq 0) {
                '†'
            } else {
                ''
            }
            $cells.Add("$(Format-Scientific (Get-RowValue $row 'interior_linf')) ($(Format-Integer (Get-RowValue $row 'iterations')))$marker")
        }
        $lines.Add('| ' + (@($cells) -join ' | ') + ' |')
    }
    $lines.Add('')
}

function Add-N32PoseDetail(
    [Collections.Generic.List[string]] $lines,
    [object[]] $allRows) {
    $lines.Add('## N=32 全姿态逐行明细')
    $lines.Add('')
    $lines.Add('每条 N=32 记录单独列出；`Dirichlet boundary_residual` 对 Neumann 或未提供该指标的 Dirichlet 记录显示为 `N/A`。')
    $lines.Add('')
    $selected = @(Get-SortedRows @($allRows | Where-Object {
        (Get-IntegerValue $_ 'N') -eq 32
    }))
    if ($selected.Count -eq 0) {
        $lines.Add('没有 N=32 记录。')
        $lines.Add('')
        return
    }

    $lines.Add('| 后端 / geometry | formulation | pose | interior_linf | density_linf | exterior_condition_linf | Dirichlet boundary_residual | GMRES it | relres | operator residual | DOF（pre→final） | trace samples | trace final DOF | margin | ratio | algebraic | physical |')
    $lines.Add('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|')
    foreach ($row in $selected) {
        $cells = @(
            (Get-CaseLabel $row)
            (Convert-ToMarkdownCell (Get-RowValue $row 'formulation'))
            (Convert-ToMarkdownCell (Get-RowValue $row 'pose'))
            (Format-Scientific (Get-RowValue $row 'interior_linf'))
            (Format-Scientific (Get-RowValue $row 'density_linf'))
            (Format-Scientific (Get-RowValue $row 'exterior_condition_linf'))
            (Get-DirichletBoundaryResidualText $row)
            (Format-Integer (Get-RowValue $row 'iterations'))
            (Format-Scientific (Get-RowValue $row 'gmres_relative_residual'))
            (Format-Scientific (Get-RowValue $row 'operator_residual_linf'))
            (Get-DofText $row)
            (Format-Integer (Get-RowValue $row 'trace_samples'))
            (Format-Integer (Get-RowValue $row 'trace_final_dofs'))
            (Format-Integer (Get-RowValue $row 'trace_oversampling_margin'))
            (Format-TraceRatio (
                Get-RowValue $row 'trace_oversampling_ratio'))
            (Format-TriState (Get-RowValue $row 'gmres_converged'))
            (Format-TriState (Get-RowValue $row 'physical_converged'))
        )
        $lines.Add('| ' + ($cells -join ' | ') + ' |')
    }
    $lines.Add('')
}

function Add-StatusSummary(
    [Collections.Generic.List[string]] $lines,
    [object[]] $rows) {
    $lines.Add('## 收敛状态统计')
    $lines.Add('')
    $lines.Add('`gmres_converged` 是代数收敛的唯一统计来源；`physical_converged` 为空时记为“未提供”，不会被当作失败。')
    $lines.Add('')
    $lines.Add('| 后端 | 方程 | 行数 | 代数：是 | 代数：否 | 代数：未提供 | physical：是 | physical：否 | physical：未提供 | GMRES it 范围 |')
    $lines.Add('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    $groupKeys = @{}
    foreach ($row in $rows) {
        $backend = Get-TrimmedText (Get-RowValue $row 'backend')
        $formulation = Get-TrimmedText (Get-RowValue $row 'formulation')
        $key = $backend.ToLowerInvariant() + [char] 31 +
            $formulation.ToLowerInvariant()
        if (-not $groupKeys.ContainsKey($key)) {
            $groupKeys[$key] = [pscustomobject]@{
                Backend = $backend
                Formulation = $formulation
            }
        }
    }
    $groups = @($groupKeys.Values | Sort-Object Backend, Formulation)
    foreach ($group in $groups) {
        $selected = @($rows | Where-Object {
            (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'backend')) $group.Backend) -and
            (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'formulation')) $group.Formulation)
        })
        $algYes = @($selected | Where-Object {
            (Get-TriState (Get-RowValue $_ 'gmres_converged')) -eq 1
        }).Count
        $algNo = @($selected | Where-Object {
            (Get-TriState (Get-RowValue $_ 'gmres_converged')) -eq 0
        }).Count
        $algUnknown = $selected.Count - $algYes - $algNo
        $physicalYes = @($selected | Where-Object {
            (Get-TriState (Get-RowValue $_ 'physical_converged')) -eq 1
        }).Count
        $physicalNo = @($selected | Where-Object {
            (Get-TriState (Get-RowValue $_ 'physical_converged')) -eq 0
        }).Count
        $physicalUnknown = $selected.Count - $physicalYes - $physicalNo
        $iterations = @($selected | ForEach-Object {
            $value = Get-IntegerValue $_ 'iterations'
            if ($value -ne [int]::MaxValue) {
                $value
            }
        })
        $iterationRange = if ($iterations.Count -eq 0) {
            [string] $([char] 0x2014)
        } else {
            "$(($iterations | Measure-Object -Minimum).Minimum)$([char] 0x2013)$(($iterations | Measure-Object -Maximum).Maximum)"
        }
        $lines.Add("| $(Convert-ToMarkdownCell $group.Backend) | $(Convert-ToMarkdownCell $group.Formulation) | $($selected.Count) | $algYes | $algNo | $algUnknown | $physicalYes | $physicalNo | $physicalUnknown | $iterationRange |")
    }
    $lines.Add('')
}

function Add-CoverageSummary(
    [Collections.Generic.List[string]] $lines,
    [object[]] $rows,
    [object[]] $geometryKeys) {
    $lines.Add('## 覆盖完整性')
    $lines.Add('')
    $lines.Add('目标矩阵为：`baseline` 含 N=32/64/128；`general_cap` 的收敛刚体姿态为 `R17+T1`，`topology_native` 的收敛刚体姿态为 `Tx`，各含 N=32/64/128；全部八个姿态均含 N=32。')
    $lines.Add('')
    $lines.Add('| 后端 / 几何 | 方程 | baseline 层 | 收敛刚体姿态 | 收敛刚体层 | N=32 姿态 | 缺失组合数 |')
    $lines.Add('|---|---|---|---|---|---:|---:|')
    foreach ($geometryKey in $geometryKeys) {
        $convergenceEntry = Get-ConvergencePoseEntry $geometryKey.Backend
        $convergencePose = if ($null -eq $convergenceEntry) {
            ''
        } else {
            [string] $convergenceEntry.Pose
        }
        $convergenceLabel = if ($null -eq $convergenceEntry) {
            [string] $([char] 0x2014)
        } else {
            "$($convergenceEntry.Short) (``$convergencePose``)"
        }
        foreach ($formulation in $canonicalFormulations) {
            $selected = @($rows | Where-Object {
                (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'backend')) $geometryKey.Backend) -and
                (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'geometry')) $geometryKey.Geometry) -and
                (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'formulation')) $formulation)
            })
            $baselineLevels = @($canonicalLevels | Where-Object {
                $level = $_
                @($selected | Where-Object {
                    (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'pose')) 'baseline') -and
                    (Get-IntegerValue $_ 'N') -eq $level
                }).Count -gt 0
            })
            $strongLevels = @($canonicalLevels | Where-Object {
                $level = $_
                @($selected | Where-Object {
                    (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'pose')) $convergencePose) -and
                    (Get-IntegerValue $_ 'N') -eq $level
                }).Count -gt 0
            })
            $poseCount = @($canonicalPoses | Where-Object {
                $poseId = $_.Id
                @($selected | Where-Object {
                    (Test-SameText (Get-TrimmedText (Get-RowValue $_ 'pose')) $poseId) -and
                    (Get-IntegerValue $_ 'N') -eq 32
                }).Count -gt 0
            }).Count
            # Unique target rows per geometry/formulation: three baseline,
            # three backend-specific convergence-rigid, and six other poses
            # at N=32.
            $observedTargetKeys = @{}
            foreach ($row in $selected) {
                $pose = Get-TrimmedText (Get-RowValue $row 'pose')
                $n = Get-IntegerValue $row 'N'
                $isTarget = (($pose -eq 'baseline' -or
                    $pose -eq $convergencePose) -and
                    $canonicalLevels -contains $n) -or
                    (($canonicalPoses.Id -contains $pose) -and $n -eq 32)
                if ($isTarget) {
                    $observedTargetKeys[$pose.ToLowerInvariant() + [char] 31 + $n] = $true
                }
            }
            $missing = 12 - $observedTargetKeys.Count
            $baselineText = if ($baselineLevels.Count -eq 0) {
                [string] $([char] 0x2014)
            } else {
                $baselineLevels -join '/'
            }
            $strongText = if ($strongLevels.Count -eq 0) {
                [string] $([char] 0x2014)
            } else {
                $strongLevels -join '/'
            }
            $lines.Add("| $($geometryKey.Backend) / $($geometryKey.Geometry) | $formulation | $baselineText | $convergenceLabel | $strongText | $poseCount/8 | $missing |")
        }
    }
    $lines.Add('')
}

function Get-NumericState($value) {
    $text = Get-TrimmedText $value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return 'missing'
    }
    $number = 0.0
    if (-not (Try-ParseDouble $text ([ref] $number))) {
        return 'invalid'
    }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        return 'nonfinite'
    }
    return 'finite'
}

function Add-FailureAndNanSection(
    [Collections.Generic.List[string]] $lines,
    [object[]] $rows) {
    $lines.Add('## 失败、缺失值与 NaN')
    $lines.Add('')
    $lines.Add('- `NaN` 阶通常表示该组的首个网格层，或相邻误差/网格尺度非有限、非正，因而不能合法计算对数阶。')
    $lines.Add('- 空白（表中显示为“—”）表示源 CSV 没有给出该指标；这与数值 `NaN` 不同。general-cap 的 `physical_converged` 空值按“未提供”处理。')
    $lines.Add('- `physical_converged=False` 是源程序的物理判据未通过，不等价于 GMRES 代数未收敛；两类状态分别列出。')
    $lines.Add('')
    $lines.Add('| 指标列 | 有限 | NaN/Inf | 空白 | 无法解析 |')
    $lines.Add('|---|---:|---:|---:|---:|')
    $numericColumns = @(
        'interior_linf', 'interior_order_linf',
        'density_linf', 'density_order_linf',
        'exterior_condition_linf', 'exterior_condition_order_linf',
        'boundary_residual_linf', 'boundary_residual_order_linf',
        'gmres_relative_residual', 'operator_residual_linf',
        'trace_samples', 'trace_final_dofs',
        'trace_oversampling_margin', 'trace_oversampling_ratio')
    foreach ($column in $numericColumns) {
        $states = @($rows | ForEach-Object {
            Get-NumericState (Get-RowValue $_ $column)
        })
        $finite = @($states | Where-Object { $_ -eq 'finite' }).Count
        $nonfinite = @($states | Where-Object { $_ -eq 'nonfinite' }).Count
        $missing = @($states | Where-Object { $_ -eq 'missing' }).Count
        $invalid = @($states | Where-Object { $_ -eq 'invalid' }).Count
        $lines.Add("| ``$column`` | $finite | $nonfinite | $missing | $invalid |")
    }
    $lines.Add('')

    $problemRows = @(Get-SortedRows @($rows | Where-Object {
        (Get-TriState (Get-RowValue $_ 'gmres_converged')) -eq 0 -or
        (Get-TriState (Get-RowValue $_ 'physical_converged')) -eq 0
    }))
    $lines.Add('### 明确未通过的记录')
    $lines.Add('')
    if ($problemRows.Count -eq 0) {
        $lines.Add('没有记录到 `gmres_converged=False` 或 `physical_converged=False`。')
        $lines.Add('')
        return
    }
    $lines.Add('| case_id | 后端 / 几何 | 方程 | 姿态 | N | 代数 | physical | it | relres | op-res | Eint |')
    $lines.Add('|---|---|---|---|---:|:---:|:---:|---:|---:|---:|---:|')
    foreach ($row in $problemRows) {
        $lines.Add('| ' + (@(
            (Convert-ToMarkdownCell (Get-RowValue $row 'case_id'))
            (Get-CaseLabel $row)
            (Convert-ToMarkdownCell (Get-RowValue $row 'formulation'))
            (Convert-ToMarkdownCell (Get-RowValue $row 'pose'))
            (Format-Integer (Get-RowValue $row 'N'))
            (Format-TriState (Get-RowValue $row 'gmres_converged'))
            (Format-TriState (Get-RowValue $row 'physical_converged'))
            (Format-Integer (Get-RowValue $row 'iterations'))
            (Format-Scientific (Get-RowValue $row 'gmres_relative_residual'))
            (Format-Scientific (Get-RowValue $row 'operator_residual_linf'))
            (Format-Scientific (Get-RowValue $row 'interior_linf'))
        ) -join ' | ') + ' |')
    }
    $lines.Add('')
}

function Assert-RequiredColumns([object[]] $rows, [string] $sourcePath) {
    if ($rows.Count -eq 0) {
        throw "Input CSV contains no data rows: $sourcePath"
    }
    $names = @($rows[0].PSObject.Properties.Name)
    foreach ($required in @('backend', 'geometry', 'formulation', 'pose', 'N')) {
        $found = $false
        foreach ($name in $names) {
            if (Test-SameText $name $required) {
                $found = $true
                break
            }
        }
        if (-not $found) {
            throw "Input CSV is missing required column '$required': $sourcePath"
        }
    }
}

function Assert-TraceOversamplingRows(
    [object[]] $rows,
    [string] $sourcePath,
    [bool] $allowLegacySynthetic) {
    $columns = @(
        'trace_samples',
        'trace_final_dofs',
        'trace_oversampling_margin',
        'trace_oversampling_ratio')
    $names = @($rows[0].PSObject.Properties.Name)
    $missingColumns = @($columns | Where-Object {
        $required = $_
        -not @($names | Where-Object {
            Test-SameText $_ $required
        }).Count
    })
    if ($missingColumns.Count -gt 0) {
        $onlyLegacyTopology = $allowLegacySynthetic -and
            $missingColumns.Count -eq $columns.Count -and
            @($rows | Where-Object {
                -not (Test-SameText `
                    (Get-TrimmedText (Get-RowValue $_ 'backend')) `
                    'topology_native')
            }).Count -eq 0
        if ($onlyLegacyTopology) {
            return [pscustomobject]@{
                Validated = 0
                LegacySyntheticSkipped = $rows.Count
            }
        }
        throw (
            "Input CSV is missing trace oversampling column(s) " +
            "'$($missingColumns -join ', ')': $sourcePath")
    }

    $validated = 0
    $legacySkipped = 0
    foreach ($row in $rows) {
        $backend = Get-TrimmedText (Get-RowValue $row 'backend')
        $states = @($columns | ForEach-Object {
            Get-NumericState (Get-RowValue $row $_)
        })
        $allMissingOrNonfinite = @($states | Where-Object {
            $_ -eq 'missing' -or $_ -eq 'nonfinite'
        }).Count -eq $states.Count
        if ($allMissingOrNonfinite -and $allowLegacySynthetic -and
            (Test-SameText $backend 'topology_native')) {
            ++$legacySkipped
            continue
        }
        if (@($states | Where-Object { $_ -ne 'finite' }).Count -gt 0) {
            throw (
                "Missing, non-finite, or invalid trace oversampling data " +
                "for $(Get-CaseLabel $row), formulation=" +
                "$(Get-TrimmedText (Get-RowValue $row 'formulation')), " +
                "pose=$(Get-TrimmedText (Get-RowValue $row 'pose')), " +
                "N=$(Get-TrimmedText (Get-RowValue $row 'N'))")
        }

        $samples = 0
        $traceFinalDofs = 0
        $margin = 0
        $samplesText = Get-TrimmedText (
            Get-RowValue $row 'trace_samples')
        $traceFinalText = Get-TrimmedText (
            Get-RowValue $row 'trace_final_dofs')
        $marginText = Get-TrimmedText (
            Get-RowValue $row 'trace_oversampling_margin')
        if (-not [int]::TryParse(
                $samplesText,
                [Globalization.NumberStyles]::Integer,
                $invariantCulture,
                [ref] $samples)) {
            throw "Trace oversampling field 'trace_samples' is not an integer: '$samplesText'"
        }
        if (-not [int]::TryParse(
                $traceFinalText,
                [Globalization.NumberStyles]::Integer,
                $invariantCulture,
                [ref] $traceFinalDofs)) {
            throw "Trace oversampling field 'trace_final_dofs' is not an integer: '$traceFinalText'"
        }
        if (-not [int]::TryParse(
                $marginText,
                [Globalization.NumberStyles]::Integer,
                $invariantCulture,
                [ref] $margin)) {
            throw "Trace oversampling field 'trace_oversampling_margin' is not an integer: '$marginText'"
        }
        $ratio = 0.0
        if (-not (Try-ParseDouble `
                (Get-RowValue $row 'trace_oversampling_ratio') `
                ([ref] $ratio)) -or
            [double]::IsNaN($ratio) -or [double]::IsInfinity($ratio)) {
            throw "Trace oversampling ratio is not finite for $(Get-CaseLabel $row)"
        }
        $context = (
            "$(Get-CaseLabel $row), formulation=" +
            "$(Get-TrimmedText (Get-RowValue $row 'formulation')), " +
            "pose=$(Get-TrimmedText (Get-RowValue $row 'pose')), " +
            "N=$(Get-TrimmedText (Get-RowValue $row 'N'))")
        if ($samples -le 0 -or $traceFinalDofs -le 0) {
            throw "$context has non-positive trace dimensions: samples=$samples final_dofs=$traceFinalDofs"
        }
        if ($samples -le $traceFinalDofs) {
            throw "$context trace projection is not strictly oversampled: samples=$samples final_dofs=$traceFinalDofs"
        }
        $expectedMargin = $samples - $traceFinalDofs
        if ($margin -ne $expectedMargin) {
            throw "$context has inconsistent trace margin: source=$margin expected=$expectedMargin"
        }
        $expectedRatio = [double] $samples / [double] $traceFinalDofs
        $tolerance = 1.0e-12 * [Math]::Max(
            1.0, [Math]::Abs($expectedRatio))
        if ($ratio -le 1.0 -or
            [Math]::Abs($ratio - $expectedRatio) -gt $tolerance) {
            throw "$context has inconsistent trace ratio: source=$ratio expected=$expectedRatio"
        }
        $normalizedFinalText = Get-TrimmedText (
            Get-RowValue $row 'final_dofs')
        if (-not [string]::IsNullOrWhiteSpace($normalizedFinalText)) {
            $normalizedFinal = 0
            if (-not [int]::TryParse(
                    $normalizedFinalText,
                    [Globalization.NumberStyles]::Integer,
                    $invariantCulture,
                    [ref] $normalizedFinal) -or
                $normalizedFinal -ne $traceFinalDofs) {
                throw (
                    "$context trace_final_dofs=$traceFinalDofs disagrees " +
                    "with final_dofs='$normalizedFinalText'")
            }
        }
        ++$validated
    }
    return [pscustomobject]@{
        Validated = $validated
        LegacySyntheticSkipped = $legacySkipped
    }
}

function New-ReportValidationKey(
    $backendValue,
    $geometryValue,
    $formulationValue,
    $poseValue,
    $nValue,
    [string] $source) {
    $values = @($backendValue, $geometryValue, $formulationValue, $poseValue)
    $names = @('backend', 'geometry', 'formulation', 'pose')
    $normalized = [Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $names.Count; ++$i) {
        $text = Get-TrimmedText $values[$i]
        if ([string]::IsNullOrWhiteSpace($text)) {
            throw "Blank key value '$($names[$i])' in $source"
        }
        $normalized.Add($text.ToLowerInvariant())
    }
    $nText = Get-TrimmedText $nValue
    $n = 0
    if (-not [int]::TryParse(
            $nText,
            [Globalization.NumberStyles]::Integer,
            $invariantCulture,
            [ref] $n) -or $n -le 0) {
        throw "Invalid positive integer N='$nText' in $source"
    }
    return (@($normalized) + @($n.ToString($invariantCulture))) -join ([char] 31)
}

function Add-UniqueReportKey(
    [hashtable] $keys,
    [string] $key,
    [string] $description) {
    if ($keys.ContainsKey($key)) {
        throw "Duplicate primary key in ${description}: $($key.Replace([char] 31, '/'))"
    }
    $keys[$key] = $true
}

function Get-CanonicalReportKeys {
    $keys = @{}
    foreach ($geometryEntry in $canonicalGeometryCatalog) {
        $convergencePose = Get-ConvergencePoseForBackend `
            $geometryEntry.Backend
        foreach ($formulation in $canonicalFormulations) {
            foreach ($poseEntry in $canonicalPoses) {
                $levels = if ($poseEntry.Id -eq 'baseline' -or
                    $poseEntry.Id -eq $convergencePose) {
                    $canonicalLevels
                } else {
                    @(32)
                }
                foreach ($n in $levels) {
                    $key = New-ReportValidationKey `
                        $geometryEntry.Backend `
                        $geometryEntry.Geometry `
                        $formulation `
                        $poseEntry.Id `
                        $n `
                        'canonical seven-geometry report matrix'
                    Add-UniqueReportKey `
                        $keys $key 'canonical seven-geometry report matrix'
                }
            }
        }
    }
    return $keys
}

function Assert-CanonicalReportMatrix(
    [object[]] $rows,
    [string] $sourcePath) {
    $actual = @{}
    foreach ($row in $rows) {
        $key = New-ReportValidationKey `
            (Get-RowValue $row 'backend') `
            (Get-RowValue $row 'geometry') `
            (Get-RowValue $row 'formulation') `
            (Get-RowValue $row 'pose') `
            (Get-RowValue $row 'N') `
            $sourcePath
        Add-UniqueReportKey $actual $key $sourcePath
    }
    $expected = Get-CanonicalReportKeys
    $missing = @($expected.Keys | Where-Object {
        -not $actual.ContainsKey($_)
    } | Sort-Object)
    $unexpected = @($actual.Keys | Where-Object {
        -not $expected.ContainsKey($_)
    } | Sort-Object)
    if ($missing.Count -gt 0 -or $unexpected.Count -gt 0) {
        $missingSample = @($missing | Select-Object -First 5 | ForEach-Object {
            ([string] $_).Replace([char] 31, '/')
        }) -join ', '
        $unexpectedSample = @(
            $unexpected | Select-Object -First 5 | ForEach-Object {
                ([string] $_).Replace([char] 31, '/')
            }) -join ', '
        throw (
            "Strict seven-geometry report matrix mismatch: expected=168 " +
            "actual=$($actual.Count) missing=$($missing.Count) " +
            "unexpected=$($unexpected.Count); missing sample=$missingSample; " +
            "unexpected sample=$unexpectedSample")
    }

    $baselineCount = @($rows | Where-Object {
        Test-SameText (Get-TrimmedText (Get-RowValue $_ 'pose')) 'baseline'
    }).Count
    $strongCount = @($rows | Where-Object {
        Test-IsBackendConvergenceRow $_
    }).Count
    $n32Count = @($rows | Where-Object {
        (Get-IntegerValue $_ 'N') -eq 32
    }).Count
    if ($rows.Count -ne 168 -or $baselineCount -ne 42 -or
        $strongCount -ne 42 -or $n32Count -ne 112) {
        throw (
            "Strict report category counts are invalid: total=" +
            "$($rows.Count)/168 baseline=$baselineCount/42 " +
            "convergence_rigid=$strongCount/42 N32_pose=$n32Count/112")
    }

    foreach ($geometryEntry in $canonicalGeometryCatalog) {
        $geometryCount = @($rows | Where-Object {
            (Test-SameText `
                (Get-TrimmedText (Get-RowValue $_ 'backend')) `
                $geometryEntry.Backend) -and
            (Test-SameText `
                (Get-TrimmedText (Get-RowValue $_ 'geometry')) `
                $geometryEntry.Geometry)
        }).Count
        if ($geometryCount -ne 24) {
            throw (
                "Strict report geometry coverage is invalid for " +
                "$($geometryEntry.Backend)/$($geometryEntry.Geometry): " +
                "$geometryCount/24 rows")
        }
    }
}

function Write-ValidationReport(
    [string] $sourcePath,
    [string] $targetPath,
    [bool] $overwrite,
    [bool] $allowPartial,
    [bool] $allowLegacySyntheticTraceDiagnostics = $false) {
    $sourcePath = [IO.Path]::GetFullPath($sourcePath)
    $targetPath = [IO.Path]::GetFullPath($targetPath)
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Missing input CSV: $sourcePath"
    }
    if ([string]::Equals(
            $sourcePath, $targetPath,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Input CSV and output Markdown must be different files'
    }
    if ((Test-Path -LiteralPath $targetPath) -and -not $overwrite) {
        throw "Output already exists; choose another path or pass -Force: $targetPath"
    }

    $before = Get-Item -LiteralPath $sourcePath
    $rows = @(Import-Csv -LiteralPath $sourcePath -Encoding UTF8)
    $after = Get-Item -LiteralPath $sourcePath
    if ($before.Length -ne $after.Length -or
        $before.LastWriteTimeUtc -ne $after.LastWriteTimeUtc) {
        throw "Input CSV changed while it was read; retry after the merge finishes: $sourcePath"
    }
    Assert-RequiredColumns $rows $sourcePath
    $traceSummary = Assert-TraceOversamplingRows `
        $rows $sourcePath $allowLegacySyntheticTraceDiagnostics
    if (-not $allowPartial) {
        Assert-CanonicalReportMatrix $rows $sourcePath
    }
    $rows = @(Get-SortedRows $rows)
    $geometryKeys = @(Get-GeometryKeys $rows)

    $duplicateGroups = @($rows | Group-Object {
        (Get-TrimmedText (Get-RowValue $_ 'backend')).ToLowerInvariant() + [char] 31 +
        (Get-TrimmedText (Get-RowValue $_ 'geometry')).ToLowerInvariant() + [char] 31 +
        (Get-TrimmedText (Get-RowValue $_ 'formulation')).ToLowerInvariant() + [char] 31 +
        (Get-TrimmedText (Get-RowValue $_ 'pose')).ToLowerInvariant() + [char] 31 +
        (Get-TrimmedText (Get-RowValue $_ 'N')).ToLowerInvariant()
    } | Where-Object { $_.Count -gt 1 })

    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add('# KFBI 3D Dirichlet / Neumann 数值验证报告')
    $lines.Add('')
    $lines.Add("- 数据源：``$(Convert-ToMarkdownCell $sourcePath)``")
    $lines.Add("- 记录数：$($rows.Count)；后端/几何组合数：$($geometryKeys.Count)；重复主键组：$($duplicateGroups.Count)")
    $lines.Add(
        "- 外迹超采样校验：通过 $($traceSummary.Validated) 行；" +
        "仅自测兼容跳过 $($traceSummary.LegacySyntheticSkipped) 行")
    $lines.Add("- 生成时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')")
    $lines.Add('- 本文档仅由合并 CSV 生成；生成过程不运行求解器，也不改写源 CSV。')
    $validationMode = if ($allowPartial) {
        'AllowPartial（未强制固定目录）'
    } else {
        '严格七几何 168 行目录（已验证）'
    }
    $lines.Add("- 输入完整性模式：$validationMode")
    $lines.Add('')
    $lines.Add('## 方法与判据说明')
    $lines.Add('')
    $lines.Add('- 两类后端均求解三维 Laplace 方程 `Delta u=0`，使用同一调和制造解 `u*(x,y,z)=exp(0.35x) cos(0.21y) cos(0.28z)`；`0.35^2=0.21^2+0.28^2`。')
    $lines.Add('- Cartesian 计算盒统一为 `[-1.5,1.5]^3`，网格宽度 `h=3/N`，本报告的收敛层为 `N=32,64,128`。GMRES 相对容差为 `2e-10`，最大迭代数为 80。')
    $lines.Add('- `topology_native` 与 `general_cap` 是两条不同的几何/迹离散后端，表格保留后端标签，不把两者的自由度定义混为一谈。')
    $lines.Add('- Neumann 问题以未知值跳 `J0` 和已知法向跳 `gN` 驱动界面问题，并令外侧值迹为零；在合法零均值未知空间中直接消去一个自由度，再把外迹残差投影到同一最终空间。Dirichlet 问题以已知值跳和未知法向跳驱动界面问题，并令外侧法向迹为零；它保留完整密度坐标。')
    $lines.Add('- 三层刚体收敛序列按后端选择：`general_cap` 使用绕 `(1,2,3)` 轴旋转 17 度再平移的 `R17+T1`，`topology_native` 使用纯平移 `Tx=(0.137,0,0)`。所有七个几何在 `N=32` 仍完整测试八种姿态。')
    $lines.Add('- 姿态常量：旋转轴为 `(1,2,3)/sqrt(14)`，旋转中心为 `(0.07,-0.07,0.02)`；`Ty=(0,-0.083,0)`、`Tz=(0,0,0.061)`、`T1=(0.137,-0.083,0.061)`、`T2=(-0.109,0.151,-0.047)`。')
    $lines.Add('- `topology_native` 未把 `R17+T1` 用作既有三层收敛姿态：先前 U 柱 `N=64` 运行暴露了一条含三个已认证物理交点的 Cartesian edge，当时的 correction/spread 路线只支持单事件并按 fail-closed 停止。当前实现已改为保存有序全事件并让 spread/restrict 共享稳定 event id，且不会放宽容差或只挑一个根；但该姿态尚未用新实现重跑三层，所以既有报告仍保留已完成的 `Tx` 收敛序列。')
    $lines.Add('- 收敛阶使用同一“后端/几何/方程/姿态”内相邻网格的 `p=log(E_prev/E_cur)/log(h_prev/h_cur)`；本报告显示合并器写入的 order 字段，不自行伪造缺失阶。')
    $lines.Add('- 代数收敛严格读取 `gmres_converged`。物理收敛严格读取 `physical_converged`；其阈值由求解程序定义，CSV 中未编码阈值时本报告不反推。')
    $lines.Add('- `topology_native` 的 `physical_converged` 还要求未投影外迹条件误差通过约 `10*GMRES tolerance=2e-9` 的严格阈值；有限网格上的离散误差可令该标志为假，即使 GMRES 和投影算子残差已经收敛。`general_cap` 未输出该布尔标志，报告保留为“未提供”。')
    $lines.Add('- 误差列：`Eint=interior_linf`，`Eρ=density_linf`，`Eext=exterior_condition_linf`，`Ebc=boundary_residual_linf`；`relres` 和 `op-res` 分别为 GMRES 相对残差与算子残差。')
    $lines.Add('- 外迹离散列中，`trace samples` 是外迹采样点数，`trace final DOF` 是约束/零均值消元后的最终迭代自由度数，`margin=samples-final DOF`，`ratio=samples/final DOF`。报告生成前逐行硬校验 `samples>final DOF`，并核对 margin、ratio 与重算值一致。')
    $routes = @($rows | ForEach-Object {
        Get-TrimmedText (Get-RowValue $_ 'zero_space_solver')
    } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Sort-Object -Unique)
    if ($routes.Count -gt 0) {
        $lines.Add('- CSV 中出现的 `zero_space_solver`：' +
            (($routes | ForEach-Object { "``$(Convert-ToMarkdownCell $_)``" }) -join '、') + '。')
    }
    $lines.Add('')

    Add-ConvergencePoseSection $lines $rows '基准姿态' 'baseline'
    Add-ConvergencePoseSection `
        $lines $rows `
        'general_cap 收敛刚体变换姿态：R17+T1' `
        $generalCapConvergencePose `
        'general_cap'
    Add-ConvergencePoseSection `
        $lines $rows `
        'topology_native 收敛刚体变换姿态：Tx' `
        $topologyConvergencePose `
        'topology_native'

    Add-N32PoseDetail $lines $rows

    $lines.Add('## N=32 八姿态刚体变换矩阵')
    $lines.Add('')
    $lines.Add('每个单元为 `Eint (GMRES it)`；上标 `†` 表示 `gmres_converged=False`，破折号表示缺失记录。')
    $lines.Add('')
    $lines.Add('| 简写 | 完整姿态 ID |')
    $lines.Add('|---|---|')
    foreach ($pose in $canonicalPoses) {
        $lines.Add("| $($pose.Short) | ``$($pose.Id)`` |")
    }
    $lines.Add('')
    foreach ($formulation in $canonicalFormulations) {
        Add-PoseMatrix $lines $rows $formulation $geometryKeys
    }

    Add-StatusSummary $lines $rows
    Add-CoverageSummary $lines $rows $geometryKeys
    Add-FailureAndNanSection $lines $rows

    $lines.Add('## 解释边界')
    $lines.Add('')
    $lines.Add('该报告忠实汇总 CSV 已记录的数据。若某个误差、阶或 physical 状态未由后端输出，报告保留为 `NaN`、空白或“未提供”，不以零替代，也不据 GMRES 状态推断物理状态。')
    $lines.Add('')

    $parent = Split-Path -Parent $targetPath
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $lines | Set-Content -LiteralPath $targetPath -Encoding UTF8
    return [pscustomobject]@{
        Input = $sourcePath
        Output = $targetPath
        Rows = $rows.Count
        GeometryKeys = $geometryKeys.Count
        DuplicateGroups = $duplicateGroups.Count
        TraceRowsValidated = $traceSummary.Validated
        LegacySyntheticTraceRowsSkipped = `
            $traceSummary.LegacySyntheticSkipped
    }
}

function New-SyntheticReportRow(
    [string] $backend,
    [string] $geometry,
    [string] $formulation,
    [string] $pose,
    [int] $n,
    [bool] $algebraic,
    $physical) {
    $factor = [Math]::Pow(32.0 / $n, 2.0)
    $scale = if ($formulation -eq 'neumann') { 2.0 } else { 1.0 }
    $order = if ($n -eq 32) { 'NaN' } else { '2' }
    $finalDofs = if ($formulation -eq 'neumann') { 47 } else { 48 }
    $traceSamples = 96
    return [pscustomobject] [ordered]@{
        case_id = "fixture_${backend}_${geometry}_${formulation}_${pose}_$n"
        backend = $backend
        geometry = $geometry
        formulation = $formulation
        pose = $pose
        N = $n
        h = 3.0 / $n
        ncoef = 12
        dofs = $finalDofs
        pre_mean_dofs = 48
        final_dofs = $finalDofs
        trace_samples = $traceSamples
        trace_final_dofs = $finalDofs
        trace_oversampling_margin = $traceSamples - $finalDofs
        trace_oversampling_ratio = [double] $traceSamples / $finalDofs
        iterations = 8 + [int] ([Math]::Log($n / 32.0, 2.0))
        gmres_converged = $algebraic
        physical_converged = $physical
        gmres_relative_residual = 1.0e-11
        operator_residual_linf = 2.0e-12
        exterior_condition_linf = 5.0e-4 * $scale * $factor
        exterior_condition_order_linf = $order
        boundary_residual_linf = if ($backend -eq 'general_cap') { 'NaN' } else { 2.5e-4 * $scale * $factor }
        boundary_residual_order_linf = if ($backend -eq 'general_cap') { 'NaN' } else { $order }
        density_linf = 2.0e-3 * $scale * $factor
        density_order_linf = $order
        density_weighted_mean = if ($formulation -eq 'neumann') { 1.0e-16 } else { 'NaN' }
        interior_linf = 1.0e-3 * $scale * $factor
        interior_order_linf = $order
        zero_space_solver = if ($formulation -eq 'neumann') { 'mean_free_pivot_elimination' } else { 'not_applicable' }
    }
}

function Assert-TextContains(
    [string] $text,
    [string] $expected,
    [string] $description) {
    if ($text.IndexOf($expected, [StringComparison]::Ordinal) -lt 0) {
        throw "Self-test missing $description ('$expected')"
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
        throw "Self-test expected failure: $description"
    }
    if (-not [string]::IsNullOrWhiteSpace($expectedMessagePart) -and
        $message.IndexOf(
            $expectedMessagePart,
            [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw (
            "Self-test '$description' failed for the wrong reason: " +
            "$message")
    }
}

function New-FullSyntheticReportRows {
    $rows = [Collections.Generic.List[object]]::new()
    foreach ($geometryEntry in $canonicalGeometryCatalog) {
        $convergencePose = Get-ConvergencePoseForBackend `
            $geometryEntry.Backend
        foreach ($formulation in $canonicalFormulations) {
            foreach ($poseEntry in $canonicalPoses) {
                $levels = if ($poseEntry.Id -eq 'baseline' -or
                    $poseEntry.Id -eq $convergencePose) {
                    $canonicalLevels
                } else {
                    @(32)
                }
                foreach ($n in $levels) {
                    $physical = if ($geometryEntry.Backend -eq 'general_cap') {
                        $null
                    } else {
                        $true
                    }
                    $rows.Add((New-SyntheticReportRow `
                        $geometryEntry.Backend `
                        $geometryEntry.Geometry `
                        $formulation `
                        $poseEntry.Id `
                        $n `
                        $true `
                        $physical))
                }
            }
        }
    }
    return @($rows)
}

function Assert-MarkdownTableColumnCounts([string] $text) {
    $expectedPipes = 0
    $inTable = $false
    $tableCount = 0
    $lineNumber = 0
    foreach ($line in @($text -split "`r?`n")) {
        ++$lineNumber
        if (-not $line.TrimStart().StartsWith('|')) {
            $inTable = $false
            continue
        }
        $pipeCount = [regex]::Matches($line, '(?<!\\)\|').Count
        if (-not $inTable) {
            $expectedPipes = $pipeCount
            $inTable = $true
            ++$tableCount
        } elseif ($pipeCount -ne $expectedPipes) {
            throw (
                "Markdown table column mismatch at line ${lineNumber}: " +
                "expected $expectedPipes delimiters, found $pipeCount")
        }
    }
    if ($tableCount -eq 0) {
        throw 'Synthetic report did not contain any Markdown tables'
    }
}

function Assert-N32DetailRowCount(
    [string] $text,
    [int] $expectedRows) {
    $header = '| 后端 / geometry | formulation | pose | interior_linf | density_linf | exterior_condition_linf | Dirichlet boundary_residual | GMRES it | relres | operator residual | DOF（pre→final） | trace samples | trace final DOF | margin | ratio | algebraic | physical |'
    $reportLines = @($text -split "`r?`n")
    $headerIndex = -1
    for ($i = 0; $i -lt $reportLines.Count; ++$i) {
        if ([string]::Equals(
                $reportLines[$i], $header,
                [StringComparison]::Ordinal)) {
            $headerIndex = $i
            break
        }
    }
    if ($headerIndex -lt 0) {
        throw 'Self-test could not find the N=32 detail table header'
    }
    $actualRows = 0
    for ($i = $headerIndex + 2; $i -lt $reportLines.Count; ++$i) {
        if (-not $reportLines[$i].TrimStart().StartsWith('|')) {
            break
        }
        ++$actualRows
    }
    if ($actualRows -ne $expectedRows) {
        throw (
            "N=32 detail row count mismatch: expected $expectedRows, " +
            "found $actualRows")
    }
}

function Invoke-SelfTest {
    $tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    $fixtureRoot = Join-Path $tempBase (
        'kfbim_report_fixture_' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $fixtureRoot | Out-Null
    try {
        $syntheticCsv = Join-Path $fixtureRoot 'merged_all_results.csv'
        $syntheticMarkdown = Join-Path $fixtureRoot 'synthetic_report.md'
        $syntheticRows = [Collections.Generic.List[object]]::new()
        foreach ($formulation in $canonicalFormulations) {
            foreach ($pose in @('baseline', $topologyConvergencePose)) {
                foreach ($n in $canonicalLevels) {
                    $algebraic = -not (
                        $formulation -eq 'neumann' -and
                        $pose -eq $topologyConvergencePose -and
                        $n -eq 32)
                    $syntheticRows.Add((New-SyntheticReportRow `
                        'topology_native' 'cylinder' $formulation $pose $n `
                        $algebraic $true))
                }
            }
            foreach ($pose in $canonicalPoses) {
                if ($pose.Id -eq 'baseline' -or
                    $pose.Id -eq $topologyConvergencePose) {
                    continue
                }
                $syntheticRows.Add((New-SyntheticReportRow `
                    'topology_native' 'cylinder' $formulation $pose.Id 32 `
                    $true $true))
            }
        }
        foreach ($formulation in $canonicalFormulations) {
            $syntheticRows.Add((New-SyntheticReportRow `
                'general_cap' 'sphere' $formulation 'baseline' 32 `
                $true $null))
        }
        @($syntheticRows) | Export-Csv -LiteralPath $syntheticCsv `
            -NoTypeInformation -Encoding UTF8
        $beforeHash = (Get-FileHash -LiteralPath $syntheticCsv -Algorithm SHA256).Hash
        $summary = Write-ValidationReport `
            $syntheticCsv $syntheticMarkdown $false $true
        $afterHash = (Get-FileHash -LiteralPath $syntheticCsv -Algorithm SHA256).Hash
        if ($beforeHash -ne $afterHash) {
            throw 'Synthetic input CSV changed during report generation'
        }
        $text = Get-Content -LiteralPath $syntheticMarkdown -Raw -Encoding UTF8
        Assert-TextContains $text '## 基准姿态' 'baseline section'
        Assert-TextContains `
            $text `
            '## general_cap 收敛刚体变换姿态：R17+T1' `
            'general-cap convergence section'
        Assert-TextContains `
            $text `
            '## topology_native 收敛刚体变换姿态：Tx' `
            'topology convergence section'
        Assert-TextContains $text '## N=32 全姿态逐行明细' 'N=32 detail section'
        Assert-TextContains $text '1.000E-03' 'scientific notation'
        Assert-TextContains $text 'general_cap / sphere' 'general-cap row'
        Assert-TextContains $text '未提供' 'blank physical state'
        Assert-TextContains $text 'NaN' 'NaN rendering'
        Assert-TextContains $text '†' 'algebraic-failure marker'
        Assert-TextContains $text '逐行硬校验 `samples>final DOF`' 'strict trace oversampling explanation'
        Assert-TextContains $text '| topology_native / cylinder | neumann | tx_p0137 | 2.000E-03 | 4.000E-03 | 1.000E-03 | N/A | 8 | 1.000E-11 | 2.000E-12 | 48→47 | 96 | 47 | 49 | 2.043 | 否 | 是 |' 'Neumann N=32 detail row'
        Assert-TextContains $text '| topology_native / cylinder | dirichlet | baseline | 1.000E-03 | 2.000E-03 | 5.000E-04 | 2.500E-04 | 8 | 1.000E-11 | 2.000E-12 | 48→48 | 96 | 48 | 48 | 2.000 | 是 | 是 |' 'Dirichlet N=32 detail row'
        Assert-TextContains $text '| general_cap / sphere | dirichlet | baseline | 1.000E-03 | 2.000E-03 | 5.000E-04 | N/A | 8 | 1.000E-11 | 2.000E-12 | 48→48 | 96 | 48 | 48 | 2.000 | 是 | 未提供 |' 'missing Dirichlet boundary-residual detail row'
        foreach ($pose in $canonicalPoses) {
            Assert-TextContains $text $pose.Id "pose $($pose.Id)"
        }
        $expectedN32Rows = @($syntheticRows | Where-Object {
            (Get-IntegerValue $_ 'N') -eq 32
        }).Count
        Assert-N32DetailRowCount $text $expectedN32Rows
        Assert-MarkdownTableColumnCounts $text
        if ($summary.Rows -ne $syntheticRows.Count) {
            throw 'Synthetic report row count mismatch'
        }
        if ($summary.TraceRowsValidated -ne $syntheticRows.Count -or
            $summary.LegacySyntheticTraceRowsSkipped -ne 0) {
            throw 'Synthetic current-schema trace validation count mismatch'
        }

        # Publication mode must accept exactly the fixed seven-geometry
        # catalog.  This fixture intentionally contains legitimate NaN values
        # in first-level order and backend-inapplicable metric fields.
        $fullCsv = Join-Path $fixtureRoot 'full_merged_all_results.csv'
        $fullMarkdown = Join-Path $fixtureRoot 'full_report.md'
        $fullRows = @(New-FullSyntheticReportRows)
        $fullRows | Export-Csv -LiteralPath $fullCsv `
            -NoTypeInformation -Encoding UTF8
        $fullHashBefore = (
            Get-FileHash -LiteralPath $fullCsv -Algorithm SHA256).Hash
        $fullSummary = Write-ValidationReport `
            $fullCsv $fullMarkdown $false $false
        $fullHashAfter = (
            Get-FileHash -LiteralPath $fullCsv -Algorithm SHA256).Hash
        if ($fullHashBefore -ne $fullHashAfter) {
            throw 'Strict report generation modified its full input CSV'
        }
        if ($fullSummary.Rows -ne 168 -or
            $fullSummary.GeometryKeys -ne 7 -or
            $fullSummary.DuplicateGroups -ne 0 -or
            $fullSummary.TraceRowsValidated -ne 168 -or
            $fullSummary.LegacySyntheticTraceRowsSkipped -ne 0) {
            throw (
                "Strict synthetic report counts are wrong: " +
                "$($fullSummary | ConvertTo-Json -Compress)")
        }
        $fullText = Get-Content -LiteralPath $fullMarkdown -Raw -Encoding UTF8
        Assert-TextContains `
            $fullText `
            '严格七几何 168 行目录（已验证）' `
            'strict validation-mode marker'
        Assert-TextContains `
            $fullText `
            'general_cap` 的收敛刚体姿态为 `R17+T1' `
            'general-cap coverage pose'
        Assert-TextContains `
            $fullText `
            'topology_native` 的收敛刚体姿态为 `Tx' `
            'topology coverage pose'
        Assert-TextContains `
            $fullText `
            '| general_cap / sphere | dirichlet | 32/64/128 | R17+T1 (`rot_axis123_17deg_t_xyz_1`) | 32/64/128 | 8/8 | 0 |' `
            'general-cap coverage row'
        Assert-TextContains `
            $fullText `
            '| topology_native / cylinder | dirichlet | 32/64/128 | Tx (`tx_p0137`) | 32/64/128 | 8/8 | 0 |' `
            'topology coverage row'
        Assert-TextContains $fullText 'NaN' 'legitimate strict-mode NaN'
        Assert-N32DetailRowCount $fullText 112
        Assert-MarkdownTableColumnCounts $fullText

        foreach ($backend in @('topology_native', 'general_cap')) {
            $badTraceRow = New-SyntheticReportRow `
                $backend `
                $(if ($backend -eq 'topology_native') {
                    'cylinder'
                } else {
                    'sphere'
                }) `
                'dirichlet' 'baseline' 32 $true $true
            $badTraceRow.trace_samples = $badTraceRow.trace_final_dofs
            $badTraceRow.trace_oversampling_margin = 0
            $badTraceRow.trace_oversampling_ratio = 1.0
            $badTraceCsv = Join-Path `
                $fixtureRoot "bad_trace_${backend}.csv"
            @($badTraceRow) | Export-Csv -LiteralPath $badTraceCsv `
                -NoTypeInformation -Encoding UTF8
            Assert-Throws {
                Write-ValidationReport `
                    $badTraceCsv `
                    (Join-Path $fixtureRoot "must_not_write_${backend}.md") `
                    $false `
                    $true
            } "$backend non-oversampled trace" 'not strictly oversampled'
        }

        # Old topology normalization emits four NaNs instead of inventing
        # dimensions.  Only this explicit internal self-test switch may admit
        # such a fixture; normal report generation remains fail-closed.
        $legacyTraceRow = New-SyntheticReportRow `
            'topology_native' 'cylinder' 'dirichlet' 'baseline' 32 `
            $true $true
        foreach ($name in @(
                'trace_samples',
                'trace_final_dofs',
                'trace_oversampling_margin',
                'trace_oversampling_ratio')) {
            $legacyTraceRow.$name = 'NaN'
        }
        $legacyTraceCsv = Join-Path $fixtureRoot 'legacy_trace_nan.csv'
        $legacyTraceMarkdown = Join-Path `
            $fixtureRoot 'legacy_trace_nan.md'
        @($legacyTraceRow) | Export-Csv -LiteralPath $legacyTraceCsv `
            -NoTypeInformation -Encoding UTF8
        Assert-Throws {
            Write-ValidationReport `
                $legacyTraceCsv `
                (Join-Path $fixtureRoot 'must_not_write_legacy_trace.md') `
                $false `
                $true
        } 'legacy topology trace without explicit compatibility' `
            'Missing, non-finite, or invalid trace oversampling data'
        $legacyTraceSummary = Write-ValidationReport `
            $legacyTraceCsv $legacyTraceMarkdown $false $true $true
        if ($legacyTraceSummary.TraceRowsValidated -ne 0 -or
            $legacyTraceSummary.LegacySyntheticTraceRowsSkipped -ne 1) {
            throw 'Legacy synthetic trace compatibility count mismatch'
        }
        $legacyTraceText = Get-Content -LiteralPath $legacyTraceMarkdown `
            -Raw -Encoding UTF8
        Assert-TextContains $legacyTraceText `
            '仅自测兼容跳过 1 行' `
            'legacy synthetic trace compatibility marker'

        $missingGeometryCsv = Join-Path $fixtureRoot 'missing_geometry.csv'
        @($fullRows | Where-Object {
            -not ((Test-SameText $_.backend 'topology_native') -and
                (Test-SameText $_.geometry 'u_prism'))
        }) | Export-Csv -LiteralPath $missingGeometryCsv `
            -NoTypeInformation -Encoding UTF8
        Assert-Throws {
            Write-ValidationReport `
                $missingGeometryCsv `
                (Join-Path $fixtureRoot 'must_not_write_missing.md') `
                $false `
                $false
        } 'whole geometry missing' 'matrix mismatch'

        $duplicateCsv = Join-Path $fixtureRoot 'duplicate_key.csv'
        @($fullRows) + @($fullRows[0]) |
            Export-Csv -LiteralPath $duplicateCsv `
                -NoTypeInformation -Encoding UTF8
        Assert-Throws {
            Write-ValidationReport `
                $duplicateCsv `
                (Join-Path $fixtureRoot 'must_not_write_duplicate.md') `
                $false `
                $false
        } 'duplicate report primary key' 'Duplicate primary key'

        Write-Host (
            "Report generator self-test passed: partial_rows=$($summary.Rows) " +
            "strict_rows=$($fullSummary.Rows) geometries=" +
            "$($fullSummary.GeometryKeys)")
    } finally {
        $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
        $safePrefix = $tempBase + [IO.Path]::DirectorySeparatorChar +
            'kfbim_report_fixture_'
        if ($resolvedFixture.StartsWith(
                $safePrefix,
                [StringComparison]::OrdinalIgnoreCase) -and
            (Test-Path -LiteralPath $resolvedFixture)) {
            Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
        }
    }
}

if ($SelfTest) {
    Invoke-SelfTest
    exit 0
}

if ([string]::IsNullOrWhiteSpace($InputCsv) -or
    [string]::IsNullOrWhiteSpace($OutputMarkdown)) {
    throw 'Pass -InputCsv and -OutputMarkdown, or use -SelfTest'
}
if (-not [IO.Path]::IsPathRooted($InputCsv)) {
    $InputCsv = Join-Path $repo $InputCsv
}
if (-not [IO.Path]::IsPathRooted($OutputMarkdown)) {
    $OutputMarkdown = Join-Path $repo $OutputMarkdown
}

$result = Write-ValidationReport `
    $InputCsv $OutputMarkdown ([bool] $Force) ([bool] $AllowPartial)
Write-Host (
    "Generated KFBI 3-D validation report: rows=$($result.Rows) " +
    "geometries=$($result.GeometryKeys) duplicates=$($result.DuplicateGroups)")
Write-Host "  input=$($result.Input)"
Write-Host "  output=$($result.Output)"
