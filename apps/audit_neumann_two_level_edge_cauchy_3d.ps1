[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [int[]]$ExpectedLevels = @(32, 64, 128),
    [switch]$AllowNumericalFailure,
    [switch]$SchemaOnly,
    [string]$DecisionJson
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName Microsoft.VisualBasic

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) {
        if ([string]::IsNullOrWhiteSpace($Message)) {
            $Message = (Get-PSCallStack | Out-String)
        }
        throw $Message
    }
}

function Get-Columns($Rows) {
    if ($Rows.Count -eq 0) { return @() }
    return @($Rows[0].PSObject.Properties.Name)
}

function Assert-Columns($Columns, [string[]]$Required, [string]$Name) {
    $columns = @($Columns)
    foreach ($column in $Required) {
        Assert-True ($columns -contains $column) ("$Name missing column $column")
    }
    Assert-True ($columns.Count -ge 3) ("$Name has fewer than three columns")
    Assert-True ($columns[0] -eq 'case_id' -and $columns[1] -eq 'N' -and
        $columns[2] -eq 'route') ("$Name does not begin case_id,N,route")
}

function Assert-RfcCsvRecordWidths(
    [string]$Path, [string]$Name, [int]$ExpectedWidth) {
    $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser($Path)
    try {
        $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
        $parser.SetDelimiters(',')
        $parser.HasFieldsEnclosedInQuotes = $true
        $parser.TrimWhiteSpace = $false
        $record = 0
        while (-not $parser.EndOfData) {
            ++$record
            try {
                $fields = @($parser.ReadFields())
            } catch {
                throw ("$Name invalid RFC CSV record $record`: " +
                    $_.Exception.Message)
            }
            Assert-True ($fields.Count -eq $ExpectedWidth) `
                ("$Name record $record has width $($fields.Count); " +
                 "expected $ExpectedWidth")
        }
        Assert-True ($record -ge 1) ("$Name is empty")
    } finally {
        $parser.Close()
    }
}

function Key($Row) {
    return [string]$Row.case_id + '|' + [string]$Row.N + '|' + [string]$Row.route
}

function Assert-UniqueKeys($Rows, [string]$Name, [scriptblock]$KeyBlock) {
    $seen = @{}
    foreach ($row in $Rows) {
        $key = & $KeyBlock $row
        Assert-True (-not $seen.ContainsKey($key)) ("$Name duplicate key $key")
        $seen[$key] = $true
    }
}

function Number($Value, [string]$Context) {
    $parsed = 0.0
    Assert-True ($Value -ne 'NA' -and
        [double]::TryParse([string]$Value,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$parsed) -and -not [double]::IsNaN($parsed) -and
        -not [double]::IsInfinity($parsed)) ("$Context is not finite numeric")
    return $parsed
}

function Integer($Value, [string]$Context) {
    $parsed = 0
    Assert-True ([int]::TryParse([string]$Value, [ref]$parsed)) `
        ("$Context is not an integer")
    return $parsed
}

function Is-NA($Value) {
    return [string]::IsNullOrWhiteSpace([string]$Value) -or $Value -eq 'NA'
}

$expectedHeaders = [ordered]@{
    'summary.csv' = 'case_id,N,route,status,h,patch_count,surface_dof_count,shared_edge_point_count,physical_converged,physical_iterations,physical_final_residual,physical_contraction,common_converged,common_iterations,common_final_residual,common_contraction,common_rhs_mean,common_rhs_rms,common_rhs_hash,density_linf,density_l2,interior_linf,interior_l2,density_linf_order_32_64,density_l2_order_32_64,interior_linf_order_32_64,interior_l2_order_32_64,density_linf_order_64_128,density_l2_order_64_128,interior_linf_order_64_128,interior_l2_order_64_128,defect_linf,defect_rms,edge_value_linf,edge_value_rms,setup_seconds,fit_seconds,pipeline_seconds,physical_solve_seconds,common_solve_seconds,neighborhood_geometry_queries,cauchy_geometry_queries,cauchy_svd_factorizations,runtime_geometry_queries,runtime_svd_factorizations,cauchy_fingerprint_before,cauchy_fingerprint_after,owner_fingerprint_before,owner_fingerprint_after,label_inside_count,label_outside_count,label_fingerprint,neighborhood_fingerprint,reference_equal,label_equal,neighborhood_equal,common_rhs_hash_equal,surface_map_count,value_map_count,normal_map_count,edge_map_count,value_radius_max_over_h,normal_radius_max_over_h,edge_radius_max_over_h,condition_max,failure_stage,failure_message,failure_entity_kind,failure_entity_id,failure_connection_id,failure_incident_sectors,failure_actual_value_counts,failure_actual_normal_counts,failure_required_value_count,failure_required_normal_count,failure_actual_edge_count,failure_value_radius_over_h,failure_normal_radius_over_h,failure_edge_radius_over_h,failure_sigma_max,failure_sigma_min,failure_condition'
    'gmres_residuals.csv' = 'case_id,N,route,rhs_kind,iteration,relative_residual'
    'edge_point_diagnostics.csv' = 'case_id,N,route,connection_id,cell_id,fraction,native_parameter_0,native_parameter_1,native_u_0,native_v_0,native_u_1,native_v_1,point_x,point_y,point_z,tangent_x,tangent_y,tangent_z,sectors,position_mismatch,mapped_tangent_dot,frame_orthogonality_error,frame_determinant,exact_value,reconstructed_value,error'
    'edge_fit_diagnostics.csv' = 'case_id,N,route,connection_id,cell_id,value_sector_0_count,value_sector_1_count,normal_sector_0_count,normal_sector_1_count,value_radius_over_h,normal_radius_over_h,sigma_max,sigma_min,condition'
    'surface_fit_diagnostics.csv' = 'case_id,N,route,center_dof,ordinary_value_count,normal_count,edge_count,value_sector_patch_ids,value_sector_counts,normal_sector_patch_ids,normal_sector_counts,value_radius_over_h,normal_radius_over_h,edge_radius_over_h,edge_distance_over_h,sigma_max,sigma_min,condition'
    'dof_diagnostics.csv' = 'case_id,N,route,dof_id,patch_id,point_x,point_y,point_z,weight,edge_distance_over_h,density_error,equation_defect'
    'edge_distance_bins.csv' = 'case_id,N,route,distance_bin,status,count,weight_sum,density_linf,density_weighted_rms,defect_linf,defect_weighted_rms,empty,failure_stage,failure_message,failure_entity_kind,failure_entity_id,failure_connection_id,failure_incident_sectors,failure_actual_value_counts,failure_actual_normal_counts,failure_required_value_count,failure_required_normal_count,failure_actual_edge_count,failure_value_radius_over_h,failure_normal_radius_over_h,failure_edge_radius_over_h,failure_sigma_max,failure_sigma_min,failure_condition'
    'owner_diagnostics.csv' = 'case_id,N,route,status,available,owner_query_count,owner_fingerprint_before,owner_fingerprint_after,owner_output_digest_before,owner_output_digest_after,cauchy_geometry_queries,cauchy_svd_factorizations,runtime_geometry_queries,runtime_svd_factorizations,cauchy_fingerprint_before,cauchy_fingerprint_after,reference_equal,label_equal,neighborhood_equal,common_rhs_hash_equal,failure_stage,failure_message,failure_entity_kind,failure_entity_id,failure_connection_id,failure_incident_sectors,failure_actual_value_counts,failure_actual_normal_counts,failure_required_value_count,failure_required_normal_count,failure_actual_edge_count,failure_value_radius_over_h,failure_normal_radius_over_h,failure_edge_radius_over_h,failure_sigma_max,failure_sigma_min,failure_condition'
}
$csvNames = @($expectedHeaders.Keys)
$tables = [ordered]@{}
$columnsByName = [ordered]@{}
foreach ($name in $csvNames) {
    $path = Join-Path $OutputDirectory $name
    Assert-True (Test-Path -LiteralPath $path -PathType Leaf) ("missing $name")
    $header = Get-Content -LiteralPath $path -TotalCount 1
    Assert-True ($header -ceq $expectedHeaders[$name]) `
        ("$name header is not the exact ordered schema")
    $columnsByName[$name] = @($expectedHeaders[$name].Split(','))
    Assert-RfcCsvRecordWidths -Path $path -Name $name `
        -ExpectedWidth $columnsByName[$name].Count
    $tables[$name] = @(Import-Csv -LiteralPath $path)
}

$summary = @($tables['summary.csv'])
$residuals = @($tables['gmres_residuals.csv'])
$edgePoints = @($tables['edge_point_diagnostics.csv'])
$edgeFits = @($tables['edge_fit_diagnostics.csv'])
$surfaceFits = @($tables['surface_fit_diagnostics.csv'])
$dofs = @($tables['dof_diagnostics.csv'])
$bins = @($tables['edge_distance_bins.csv'])
$owners = @($tables['owner_diagnostics.csv'])
$failureColumns = @(
    'failure_stage', 'failure_message', 'failure_entity_kind',
    'failure_entity_id', 'failure_connection_id', 'failure_incident_sectors',
    'failure_actual_value_counts', 'failure_actual_normal_counts',
    'failure_required_value_count', 'failure_required_normal_count',
    'failure_actual_edge_count', 'failure_value_radius_over_h',
    'failure_normal_radius_over_h', 'failure_edge_radius_over_h',
    'failure_sigma_max', 'failure_sigma_min', 'failure_condition')

foreach ($name in $csvNames) {
    Assert-Columns -Columns @($columnsByName[$name]) `
        -Required @('case_id','N','route') -Name $name
}
Assert-Columns -Columns @($columnsByName['summary.csv']) `
    -Required (@('status','physical_iterations','common_iterations') +
        $failureColumns) -Name 'summary.csv'
Assert-Columns -Columns @($columnsByName['edge_distance_bins.csv']) `
    -Required (@('distance_bin','status') + $failureColumns) `
    -Name 'edge_distance_bins.csv'
Assert-Columns -Columns @($columnsByName['owner_diagnostics.csv']) `
    -Required (@('status','available') + $failureColumns) `
    -Name 'owner_diagnostics.csv'

$routes = @('g1_value_g1_normal','direct_cross_face_value',
    'edge_reconstructed_value')
Assert-UniqueKeys $summary 'summary.csv' { param($row) Key $row }
Assert-UniqueKeys $owners 'owner_diagnostics.csv' { param($row) Key $row }
Assert-UniqueKeys $bins 'edge_distance_bins.csv' {
    param($row) (Key $row) + '|' + $row.distance_bin }
Assert-UniqueKeys $surfaceFits 'surface_fit_diagnostics.csv' {
    param($row) (Key $row) + '|' + $row.center_dof }
Assert-UniqueKeys $dofs 'dof_diagnostics.csv' {
    param($row) (Key $row) + '|' + $row.dof_id }
Assert-UniqueKeys $edgePoints 'edge_point_diagnostics.csv' {
    param($row) (Key $row) + '|' + $row.connection_id + '|' + $row.cell_id +
        '|' + $row.fraction }
Assert-UniqueKeys $edgeFits 'edge_fit_diagnostics.csv' {
    param($row) (Key $row) + '|' + $row.connection_id + '|' + $row.cell_id }

$summaryByKey = @{}
foreach ($row in $summary) {
    Assert-True ($routes -contains $row.route) ("unknown route " + $row.route)
    Assert-True ($ExpectedLevels -contains (Integer $row.N 'summary N')) `
        ("unexpected level " + $row.N)
    $summaryByKey[(Key $row)] = $row
}

foreach ($key in $summaryByKey.Keys) {
    Assert-True (@($owners | Where-Object { (Key $_) -eq $key }).Count -eq 1) `
        ("missing owner row $key")
    $keyBins = @($bins | Where-Object { (Key $_) -eq $key })
    Assert-True ($keyBins.Count -eq 3) ("missing bin row $key")
    foreach ($bin in @('lt_h','h_to_2h','gt_2h')) {
        Assert-True (@($keyBins | Where-Object distance_bin -eq $bin).Count -eq 1) `
            ("missing bin $bin for $key")
    }
}

foreach ($row in $summary) {
    $key = Key $row
    $matchingOwner = @($owners | Where-Object { (Key $_) -eq $key })[0]
    $matchingBins = @($bins | Where-Object { (Key $_) -eq $key })
    if ($row.status -eq 'failed') {
        Assert-True (-not (Is-NA $row.failure_stage)) `
            ("failed summary missing failure_stage $key")
        Assert-True (-not (Is-NA $row.failure_message)) `
            ("failed summary missing failure_message $key")
        foreach ($dependent in @($matchingOwner) + $matchingBins) {
            Assert-True ($dependent.status -eq 'failed') `
                ("failed dependent row has wrong status $key")
            Assert-True (-not (Is-NA $dependent.failure_stage)) `
                ("failed dependent row missing failure_stage $key")
            Assert-True (-not (Is-NA $dependent.failure_message)) `
                ("failed dependent row missing failure_message $key")
        }
        continue
    }
    Assert-True ($row.status -eq 'ok') ("invalid status $key")
    Assert-True ($matchingOwner.status -eq 'ok') `
        ("owner status mismatch $key")
    Assert-True (@($matchingBins | Where-Object status -ne 'ok').Count -eq 0) `
        ("bin status mismatch $key")
    foreach ($kind in @('physical','common')) {
        $iterations = if ($kind -eq 'physical') {
            Integer $row.physical_iterations "$key physical_iterations"
        } else {
            Integer $row.common_iterations "$key common_iterations"
        }
        $history = @($residuals | Where-Object {
            (Key $_) -eq $key -and $_.rhs_kind -eq $kind } |
            Sort-Object { [int]$_.iteration })
        Assert-True ($history.Count -eq $iterations + 1) `
            ("residual history length mismatch $key $kind")
        for ($iteration = 0; $iteration -lt $history.Count; ++$iteration) {
            $recordedIteration = Integer $history[$iteration].iteration `
                "$key $kind iteration"
            Assert-True ($recordedIteration -eq $iteration) `
                ("residual iteration mismatch $key $kind")
            $recordedResidual = Number $history[$iteration].relative_residual `
                "$key $kind residual"
            [void]$recordedResidual
        }
        $summaryFinal = if ($kind -eq 'physical') {
            Number $row.physical_final_residual "$key physical final residual"
        } else {
            Number $row.common_final_residual "$key common final residual"
        }
        $historyFinal = Number $history[-1].relative_residual `
            "$key $kind history final residual"
        $finalTolerance = 64.0 * 2.2204460492503131e-16 *
            [Math]::Max(1.0, [Math]::Abs($summaryFinal))
        Assert-True ([Math]::Abs($summaryFinal - $historyFinal) -le
            $finalTolerance) ("final residual mismatch $key $kind")
    }
}

foreach ($row in $residuals) {
    Assert-True ($summaryByKey.ContainsKey((Key $row))) `
        ("orphan residual row " + (Key $row))
    Assert-True (@('physical','common') -contains $row.rhs_kind) `
        ("unknown rhs_kind " + $row.rhs_kind)
}
foreach ($row in @($edgePoints) + @($edgeFits)) {
    Assert-True ($row.route -eq 'edge_reconstructed_value') `
        ("control route invented shared-edge diagnostics " + (Key $row))
    Assert-True ($summaryByKey.ContainsKey((Key $row))) `
        ("orphan shared-edge diagnostic " + (Key $row))
}
foreach ($row in @($surfaceFits) + @($dofs)) {
    Assert-True ($summaryByKey.ContainsKey((Key $row))) `
        ("orphan diagnostic " + (Key $row))
}
$edgePointBaseKeys = @($edgePoints | ForEach-Object {
    (Key $_) + '|' + $_.connection_id + '|' + $_.cell_id } | Sort-Object -Unique)
$edgeFitKeys = @($edgeFits | ForEach-Object {
    (Key $_) + '|' + $_.connection_id + '|' + $_.cell_id } | Sort-Object -Unique)
Assert-True ($edgePointBaseKeys.Count -eq $edgeFitKeys.Count) `
    'edge point/fit key-set size mismatch'
for ($index = 0; $index -lt $edgePointBaseKeys.Count; ++$index) {
    Assert-True ($edgePointBaseKeys[$index] -eq $edgeFitKeys[$index]) `
        ("edge point/fit key-set mismatch at $index")
}
foreach ($row in $summary | Where-Object {
    $_.status -eq 'ok' }) {
    $key = Key $row
    $expectedDofs = Integer $row.surface_dof_count "$key surface_dof_count"
    $expectedSurfaceMaps = Integer $row.surface_map_count "$key surface_map_count"
    $expectedValueMaps = Integer $row.value_map_count "$key value_map_count"
    $expectedNormalMaps = Integer $row.normal_map_count "$key normal_map_count"
    $expectedEdgeMaps = Integer $row.edge_map_count "$key edge_map_count"
    Assert-True (@($surfaceFits | Where-Object { (Key $_) -eq $key }).Count `
        -eq $expectedSurfaceMaps) ("surface map count mismatch $key")
    Assert-True (@($dofs | Where-Object { (Key $_) -eq $key }).Count `
        -eq $expectedDofs) ("DOF diagnostic count mismatch $key")
    $surfaceIds = @($surfaceFits | Where-Object { (Key $_) -eq $key } |
        ForEach-Object { Integer $_.center_dof "$key center_dof" } | Sort-Object)
    $dofIds = @($dofs | Where-Object { (Key $_) -eq $key } |
        ForEach-Object { Integer $_.dof_id "$key dof_id" } | Sort-Object)
    for ($index = 0; $index -lt $surfaceIds.Count; ++$index) {
        Assert-True ($surfaceIds[$index] -eq $index) `
            ("surface center IDs are not contiguous $key")
    }
    for ($index = 0; $index -lt $dofIds.Count; ++$index) {
        Assert-True ($dofIds[$index] -eq $index) `
            ("DOF IDs are not contiguous $key")
    }
    Assert-True ($expectedValueMaps -eq $expectedSurfaceMaps -and
        $expectedNormalMaps -eq $expectedSurfaceMaps) `
        ("value/normal map count mismatch $key")
    if ($row.route -eq 'edge_reconstructed_value') {
        Assert-True (@($edgePoints | Where-Object { (Key $_) -eq $key }).Count `
            -eq $expectedEdgeMaps) ("shared edge point count mismatch $key")
        Assert-True (@($edgeFits | Where-Object { (Key $_) -eq $key }).Count `
            -eq $expectedEdgeMaps) ("shared edge fit count mismatch $key")
        Assert-True ($expectedEdgeMaps -gt 0) `
            ("successful shared route has no edge maps $key")
    } else {
        Assert-True ($expectedEdgeMaps -eq 0) `
            ("control route reports edge maps $key")
    }
}

$orderMetrics = @('density_linf','density_l2','interior_linf','interior_l2')
$order32Columns = @('density_linf_order_32_64','density_l2_order_32_64',
    'interior_linf_order_32_64','interior_l2_order_32_64')
$order64Columns = @('density_linf_order_64_128','density_l2_order_64_128',
    'interior_linf_order_64_128','interior_l2_order_64_128')
foreach ($row in $summary | Where-Object status -eq 'ok') {
    for ($metricIndex = 0; $metricIndex -lt $orderMetrics.Count; ++$metricIndex) {
        $metric = $orderMetrics[$metricIndex]
        foreach ($pair in @(
            [ordered]@{coarse=32; fine=64; column=$order32Columns[$metricIndex]},
            [ordered]@{coarse=64; fine=128; column=$order64Columns[$metricIndex]})) {
            $expectsOrder = $false
            if ([int]$row.N -eq $pair.fine -and
                $ExpectedLevels -contains $pair.coarse) {
                $coarseKey = "$($row.case_id)|$($pair.coarse)|$($row.route)"
                if ($summaryByKey.ContainsKey($coarseKey) -and
                    $summaryByKey[$coarseKey].status -eq 'ok') {
                    $coarseError = Number $summaryByKey[$coarseKey].$metric `
                        "$coarseKey $metric"
                    $fineError = Number $row.$metric ((Key $row) + " $metric")
                    Assert-True ($coarseError -gt 0.0 -and $fineError -gt 0.0) `
                        ("adjacent-order errors must be positive $coarseKey $metric")
                    $expectsOrder = $true
                }
            }
            $column = $pair.column
            Assert-True ((-not $expectsOrder -and (Is-NA $row.$column)) -or
                ($expectsOrder -and -not (Is-NA $row.$column))) `
                ("adjacent order NA semantics mismatch " + (Key $row) +
                 " $column")
            if ($expectsOrder) {
                [void](Number $row.$column ((Key $row) + " $column"))
            }
        }
    }
}

if ($SchemaOnly) {
    Write-Host ("schema audit passed: summary={0} bins={1} owners={2}" -f
        $summary.Count, $bins.Count, $owners.Count)
    exit 0
}

$caseIds = @('baseline','rot_axis123_17deg','rot_axis123_17deg_t_xyz_1')
foreach ($caseId in $caseIds) {
    foreach ($level in $ExpectedLevels) {
        foreach ($route in $routes) {
            $key = "$caseId|$level|$route"
            Assert-True $summaryByKey.ContainsKey($key) ("missing formal key $key")
        }
    }
}
Assert-True ($summary.Count -eq $caseIds.Count * $ExpectedLevels.Count * 3) `
    'summary row count does not equal the complete fixed matrix'

$failedPredicates = New-Object System.Collections.Generic.List[string]
$mandatoryFailedPredicates = New-Object System.Collections.Generic.List[string]
$acceptanceFailedPredicates = New-Object System.Collections.Generic.List[string]
function Record-Predicate(
    [bool]$Condition, [string]$Name, [string]$Kind = 'mandatory') {
    if (-not $Condition) {
        $failedPredicates.Add($Name)
        if ($Kind -eq 'acceptance') {
            $acceptanceFailedPredicates.Add($Name)
        } else {
            $mandatoryFailedPredicates.Add($Name)
        }
    }
}

foreach ($row in $summary) {
    $key = Key $row
    if ($row.status -ne 'ok') {
        Record-Predicate $false ("status_ok:$key")
        continue
    }
    Record-Predicate ($row.physical_converged -eq '1') `
        ("physical_converged:$key")
    Record-Predicate ($row.common_converged -eq '1') `
        ("common_converged:$key")
    Record-Predicate ((Integer $row.physical_iterations "$key physical") -le 80) `
        ("physical_iteration_cap:$key")
    Record-Predicate ((Integer $row.common_iterations "$key common") -le 80) `
        ("common_iteration_cap:$key")
    $physicalFinal = Number $row.physical_final_residual "$key physical residual"
    $commonFinal = Number $row.common_final_residual "$key common residual"
    $runtimeQueries = Integer $row.runtime_geometry_queries "$key runtime queries"
    $runtimeSvd = Integer $row.runtime_svd_factorizations "$key runtime SVD"
    Record-Predicate ($physicalFinal -lt 2.0e-10) ("physical_final_residual:$key")
    Record-Predicate ($commonFinal -lt 2.0e-10) ("common_final_residual:$key")
    Record-Predicate ($runtimeQueries -eq 0) ("runtime_geometry_queries:$key")
    Record-Predicate ($runtimeSvd -eq 0) ("runtime_svd_factorizations:$key")
    Record-Predicate ($row.cauchy_fingerprint_before -eq
        $row.cauchy_fingerprint_after) ("cauchy_fingerprint:$key")
}

$referenceAudits = New-Object System.Collections.Generic.List[object]
foreach ($caseId in $caseIds) {
    foreach ($level in $ExpectedLevels) {
        $g1Key = "$caseId|$level|g1_value_g1_normal"
        $g1Summary = $summaryByKey[$g1Key]
        $g1Owner = @($owners | Where-Object { (Key $_) -eq $g1Key })[0]
        foreach ($route in $routes) {
            $key = "$caseId|$level|$route"
            $candidate = $summaryByKey[$key]
            $candidateOwner = @($owners | Where-Object { (Key $_) -eq $key })[0]
            $equal = $g1Summary.status -eq 'ok' -and $candidate.status -eq 'ok' -and
                $candidateOwner.owner_query_count -eq $g1Owner.owner_query_count -and
                $candidateOwner.owner_fingerprint_before -eq $g1Owner.owner_fingerprint_before -and
                $candidateOwner.owner_output_digest_before -eq $g1Owner.owner_output_digest_before -and
                $candidate.label_inside_count -eq $g1Summary.label_inside_count -and
                $candidate.label_outside_count -eq $g1Summary.label_outside_count -and
                $candidate.label_fingerprint -eq $g1Summary.label_fingerprint -and
                $candidate.neighborhood_fingerprint -eq $g1Summary.neighborhood_fingerprint -and
                $candidate.common_rhs_hash -eq $g1Summary.common_rhs_hash
            $referenceAudits.Add([ordered]@{case_id=$caseId; N=$level;
                route=$route; equal_to_g1=$equal})
            Record-Predicate $equal ("raw_g1_reference:$key")
        }
    }
}

foreach ($point in $edgePoints) {
    $key = (Key $point) + '|' + $point.connection_id + '|' + $point.cell_id
    $mismatch = Number $point.position_mismatch "$key mismatch"
    $tangent = Number $point.mapped_tangent_dot "$key tangent"
    $frameError = Number $point.frame_orthogonality_error "$key frame"
    $determinant = Number $point.frame_determinant "$key determinant"
    Record-Predicate ($mismatch -le 1.0e-11) ("edge_position_mismatch:$key")
    Record-Predicate ($tangent -ge 0.9999999999) ("edge_tangent:$key")
    Record-Predicate ($frameError -le 1.0e-10) ("edge_frame_orthogonality:$key")
    Record-Predicate ($determinant -gt 0.0) ("edge_frame_determinant:$key")
}
foreach ($fit in $edgeFits) {
    $key = (Key $fit) + '|' + $fit.connection_id + '|' + $fit.cell_id
    Record-Predicate ((Integer $fit.value_sector_0_count "$key value0") -eq 24 -and
        (Integer $fit.value_sector_1_count "$key value1") -eq 24 -and
        (Integer $fit.normal_sector_0_count "$key normal0") -eq 14 -and
        (Integer $fit.normal_sector_1_count "$key normal1") -eq 14) `
        ("edge_fit_counts:$key")
    Record-Predicate ((Number $fit.sigma_min "$key sigma_min") -gt 0.0) `
        ("edge_fit_rank:$key")
}
foreach ($fit in $surfaceFits) {
    $key = (Key $fit) + '|' + $fit.center_dof
    Record-Predicate ((Integer $fit.ordinary_value_count "$key values") -eq 48 -and
        (Integer $fit.normal_count "$key normals") -eq 28) `
        ("surface_fit_counts:$key")
    Record-Predicate ((Number $fit.sigma_min "$key sigma_min") -gt 0.0) `
        ("surface_fit_rank:$key")
}

$iterationMetrics = New-Object System.Collections.Generic.List[object]
foreach ($route in $routes) {
    foreach ($kind in @('physical','common')) {
        $field = $kind + '_iterations'
        $routeRows = @($summary | Where-Object {
            $_.route -eq $route -and $_.status -eq 'ok' })
        if ($routeRows.Count -eq 0) { continue }
        $worst = ($routeRows | ForEach-Object { Integer $_.$field "$route $kind" } |
            Measure-Object -Maximum).Maximum
        foreach ($level in $ExpectedLevels) {
            $values = @($routeRows | Where-Object { [int]$_.N -eq $level } |
                ForEach-Object { Integer $_.$field "$route $kind N=$level" })
            if ($values.Count -eq 0) { continue }
            $measure = $values | Measure-Object -Minimum -Maximum
            $iterationMetrics.Add([ordered]@{
                route=$route; rhs_kind=$kind; N=$level; W=[int]$worst;
                S=[int]($measure.Maximum-$measure.Minimum) })
        }
    }
}

$orders = New-Object System.Collections.Generic.List[object]
$errorFields = @('density_linf','density_l2','interior_linf','interior_l2')
$sortedLevels = @($ExpectedLevels | Sort-Object)
for ($index = 1; $index -lt $sortedLevels.Count; ++$index) {
    $coarseN = $sortedLevels[$index - 1]
    $fineN = $sortedLevels[$index]
    foreach ($caseId in $caseIds) {
        foreach ($route in $routes) {
            $coarse = $summaryByKey["$caseId|$coarseN|$route"]
            $fine = $summaryByKey["$caseId|$fineN|$route"]
            if ($coarse.status -ne 'ok' -or $fine.status -ne 'ok') { continue }
            foreach ($field in $errorFields) {
                $coarseError = Number $coarse.$field "$caseId $route coarse $field"
                $fineError = Number $fine.$field "$caseId $route fine $field"
                Assert-True ($coarseError -gt 0.0 -and $fineError -gt 0.0) `
                    ("order errors must be positive $caseId $route $field")
                $order = [Math]::Log($coarseError / $fineError, 2.0)
                $orders.Add([ordered]@{ case_id=$caseId; route=$route;
                    coarse_N=$coarseN; fine_N=$fineN; metric=$field; order=$order })
                if ($route -eq 'edge_reconstructed_value' -and
                    $fineN -eq ($sortedLevels | Measure-Object -Maximum).Maximum) {
                    Record-Predicate ($order -ge 0.0) `
                        ("primary_nonnegative_order:$caseId`:$field") acceptance
                }
            }
        }
    }
}

$recordedOrderChecks = New-Object System.Collections.Generic.List[object]
foreach ($orderRow in $orders) {
    $suffix = if ($orderRow.coarse_N -eq 32 -and $orderRow.fine_N -eq 64) {
        'order_32_64'
    } elseif ($orderRow.coarse_N -eq 64 -and $orderRow.fine_N -eq 128) {
        'order_64_128'
    } else { $null }
    if ($null -eq $suffix) { continue }
    $fine = $summaryByKey["$($orderRow.case_id)|$($orderRow.fine_N)|$($orderRow.route)"]
    $column = $orderRow.metric + '_' + $suffix
    $recorded = Number $fine.$column "$($orderRow.case_id) $column"
    $equal = [Math]::Abs($recorded - $orderRow.order) -le
        256.0 * 2.2204460492503131e-16 *
        [Math]::Max(1.0, [Math]::Abs($orderRow.order))
    $recordedOrderChecks.Add([ordered]@{case_id=$orderRow.case_id;
        route=$orderRow.route; column=$column; recorded=$recorded;
        recomputed=$orderRow.order; equal=$equal})
    Record-Predicate $equal ("recorded_order:$($orderRow.case_id):$($orderRow.route):$column")
}

$nearEdge = New-Object System.Collections.Generic.List[object]
foreach ($row in $summary | Where-Object status -eq 'ok') {
    $key = Key $row
    $near = @($bins | Where-Object {
        (Key $_) -eq $key -and $_.distance_bin -in @('lt_h','h_to_2h') })
    $weight = 0.0
    $square = 0.0
    $linf = 0.0
    foreach ($bin in $near) {
        $w = Number $bin.weight_sum "$key near weight"
        Assert-True ($w -ge 0.0) ("negative near-edge weight $key")
        $rms = Number $bin.defect_weighted_rms "$key near rms"
        $weight += $w
        $square += $w * $rms * $rms
        $linf = [Math]::Max($linf, (Number $bin.defect_linf "$key near linf"))
    }
    Assert-True ($weight -gt 0.0) ("nonpositive near-edge combined weight $key")
    $combinedRms = if ($weight -gt 0.0) { [Math]::Sqrt($square / $weight) } else { 0.0 }
    $nearEdge.Add([ordered]@{ case_id=$row.case_id; N=[int]$row.N;
        route=$row.route; weight_sum=$weight; defect_linf=$linf;
        defect_weighted_rms=$combinedRms })
}

$ratios = New-Object System.Collections.Generic.List[object]
$finest = ($sortedLevels | Measure-Object -Maximum).Maximum
if ($sortedLevels -contains 128) {
    foreach ($caseId in $caseIds) {
        $primary = $summaryByKey["$caseId|128|edge_reconstructed_value"]
        if ($primary.status -ne 'ok') { continue }
        foreach ($controlRoute in @('g1_value_g1_normal','direct_cross_face_value')) {
            $control = $summaryByKey["$caseId|128|$controlRoute"]
            if ($control.status -ne 'ok') { continue }
            foreach ($field in $errorFields) {
                $primaryError = Number $primary.$field "$caseId primary $field"
                $controlError = Number $control.$field "$caseId control $field"
                Assert-True ($primaryError -gt 0.0 -and $controlError -gt 0.0) `
                    ("N128 ratio errors must be positive $caseId $field")
                $ratio = $primaryError / $controlError
                $ratios.Add([ordered]@{ case_id=$caseId; control=$controlRoute;
                    metric=$field; ratio=$ratio })
                Record-Predicate ($ratio -le 1.10) `
                    ("primary_n128_ratio:$caseId`:$controlRoute`:$field") acceptance
            }
            $primaryNear = @($nearEdge | Where-Object { $_.case_id -eq $caseId -and
                $_.N -eq 128 -and $_.route -eq 'edge_reconstructed_value' })[0]
            $controlNear = @($nearEdge | Where-Object { $_.case_id -eq $caseId -and
                $_.N -eq 128 -and $_.route -eq $controlRoute })[0]
            Record-Predicate ($primaryNear.defect_linf -lt $controlNear.defect_linf) `
                ("primary_near_linf:$caseId`:$controlRoute") acceptance
            Record-Predicate ($primaryNear.defect_weighted_rms -lt
                $controlNear.defect_weighted_rms) `
                ("primary_near_rms:$caseId`:$controlRoute") acceptance
        }
    }
}

$edgeTrends = New-Object System.Collections.Generic.List[object]
if ($sortedLevels -contains 64 -and $sortedLevels -contains 128) {
    foreach ($caseId in $caseIds) {
        $coarse = $summaryByKey["$caseId|64|edge_reconstructed_value"]
        $fine = $summaryByKey["$caseId|128|edge_reconstructed_value"]
        if ($coarse.status -ne 'ok' -or $fine.status -ne 'ok') { continue }
        foreach ($field in @('edge_value_linf','edge_value_rms')) {
            $decreased = (Number $fine.$field "$caseId fine edge") -lt
                (Number $coarse.$field "$caseId coarse edge")
            $edgeTrends.Add([ordered]@{case_id=$caseId; metric=$field;
                decreased=$decreased})
            Record-Predicate $decreased ("edge_error_trend:$caseId`:$field") acceptance
        }
    }
}

$iterationComparisonAvailable = $iterationMetrics.Count -gt 0 -and
    @($summary | Where-Object status -eq 'ok').Count -eq $summary.Count
$primaryWNoWorse = $iterationComparisonAvailable
$primarySNoWorse = $iterationComparisonAvailable
$primaryWStrict = $false
$primarySStrict = $false
$directIterationNoWorse = $iterationComparisonAvailable
if ($iterationComparisonAvailable) {
    foreach ($kind in @('physical','common')) {
        $primaryKind = @($iterationMetrics | Where-Object {
            $_.route -eq 'edge_reconstructed_value' -and
            $_.rhs_kind -eq $kind })[0]
        $directKind = @($iterationMetrics | Where-Object {
            $_.route -eq 'direct_cross_face_value' -and
            $_.rhs_kind -eq $kind })[0]
        $g1Kind = @($iterationMetrics | Where-Object {
            $_.route -eq 'g1_value_g1_normal' -and
            $_.rhs_kind -eq $kind })[0]
        foreach ($controlKind in @($g1Kind,$directKind)) {
            $primaryWNoWorse = $primaryWNoWorse -and
                $primaryKind.W -le $controlKind.W
            $primaryWStrict = $primaryWStrict -or
                $primaryKind.W -lt $controlKind.W
        }
        $directIterationNoWorse = $directIterationNoWorse -and
            $directKind.W -le $g1Kind.W
        foreach ($level in $ExpectedLevels) {
            $primary = @($iterationMetrics | Where-Object {
                $_.route -eq 'edge_reconstructed_value' -and
                $_.rhs_kind -eq $kind -and $_.N -eq $level })[0]
            foreach ($controlRoute in @('g1_value_g1_normal','direct_cross_face_value')) {
                $control = @($iterationMetrics | Where-Object {
                    $_.route -eq $controlRoute -and $_.rhs_kind -eq $kind -and
                    $_.N -eq $level })[0]
                $primarySNoWorse = $primarySNoWorse -and
                    $primary.S -le $control.S
                $primarySStrict = $primarySStrict -or
                    $primary.S -lt $control.S
            }
            $directAtLevel = @($iterationMetrics | Where-Object {
                $_.route -eq 'direct_cross_face_value' -and
                $_.rhs_kind -eq $kind -and $_.N -eq $level })[0]
            $g1AtLevel = @($iterationMetrics | Where-Object {
                $_.route -eq 'g1_value_g1_normal' -and
                $_.rhs_kind -eq $kind -and $_.N -eq $level })[0]
            $directIterationNoWorse = $directIterationNoWorse -and
                $directAtLevel.S -le $g1AtLevel.S
        }
    }
    Record-Predicate $primaryWNoWorse 'primary_W_no_worse' acceptance
    Record-Predicate $primarySNoWorse 'primary_S_no_worse' acceptance
    Record-Predicate $primaryWStrict 'primary_W_strict' acceptance
    Record-Predicate $primarySStrict 'primary_S_strict' acceptance
}
$primaryIterationPass = $primaryWNoWorse -and $primarySNoWorse -and
    $primaryWStrict -and $primarySStrict

$directComplete = @($summary | Where-Object {
    $_.route -eq 'direct_cross_face_value' -and $_.status -eq 'ok' }).Count -eq
    $caseIds.Count * $ExpectedLevels.Count
$sharedComplete = @($summary | Where-Object {
    $_.route -eq 'edge_reconstructed_value' -and $_.status -eq 'ok' }).Count -eq
    $caseIds.Count * $ExpectedLevels.Count
$directOrdersAtFinest = @($orders | Where-Object {
    $_.route -eq 'direct_cross_face_value' -and $_.fine_N -eq $finest })
$sharedOrdersAtFinest = @($orders | Where-Object {
    $_.route -eq 'edge_reconstructed_value' -and $_.fine_N -eq $finest })
$directOrderPass = $directOrdersAtFinest.Count -eq $caseIds.Count * 4 -and
    @($directOrdersAtFinest | Where-Object order -lt 0.0).Count -eq 0
$sharedOrderPass = $sharedOrdersAtFinest.Count -eq $caseIds.Count * 4 -and
    @($sharedOrdersAtFinest | Where-Object order -lt 0.0).Count -eq 0
$directRatioPass = $false
$directNearEdgePass = $false
$sharedNearEdgePass = $false
if ($ExpectedLevels -contains 128) {
    $directRatioPass = $true
    $directNearEdgePass = $true
    $sharedNearEdgePass = $true
    foreach ($caseId in $caseIds) {
        $g1 = $summaryByKey["$caseId|128|g1_value_g1_normal"]
        $direct = $summaryByKey["$caseId|128|direct_cross_face_value"]
        $shared = $summaryByKey["$caseId|128|edge_reconstructed_value"]
        if ($g1.status -ne 'ok' -or $direct.status -ne 'ok') {
            $directRatioPass = $false
            $directNearEdgePass = $false
            $sharedNearEdgePass = $false
            continue
        }
        foreach ($field in $errorFields) {
            $directError = Number $direct.$field "$caseId direct $field"
            $g1Error = Number $g1.$field "$caseId g1 $field"
            Assert-True ($directError -gt 0.0 -and $g1Error -gt 0.0) `
                ("direct ratio errors must be positive $caseId $field")
            $directRatioPass = $directRatioPass -and
                $directError / $g1Error -le 1.10
        }
        $g1Near = @($nearEdge | Where-Object { $_.case_id -eq $caseId -and
            $_.N -eq 128 -and $_.route -eq 'g1_value_g1_normal' })[0]
        $directNear = @($nearEdge | Where-Object { $_.case_id -eq $caseId -and
            $_.N -eq 128 -and $_.route -eq 'direct_cross_face_value' })[0]
        $directNearEdgePass = $directNearEdgePass -and
            $directNear.defect_linf -lt $g1Near.defect_linf -and
            $directNear.defect_weighted_rms -lt $g1Near.defect_weighted_rms
        if ($shared.status -ne 'ok') {
            $sharedNearEdgePass = $false
            continue
        }
        $sharedNear = @($nearEdge | Where-Object { $_.case_id -eq $caseId -and
            $_.N -eq 128 -and $_.route -eq 'edge_reconstructed_value' })[0]
        foreach ($controlNear in @($g1Near,$directNear)) {
            $sharedNearEdgePass = $sharedNearEdgePass -and
                $sharedNear.defect_linf -lt $controlNear.defect_linf -and
                $sharedNear.defect_weighted_rms -lt
                    $controlNear.defect_weighted_rms
        }
    }
}
$sharedRatioPass = $ratios.Count -eq $caseIds.Count * 2 * 4 -and
    @($ratios | Where-Object ratio -gt 1.10).Count -eq 0
$sharedEdgeTrendPass = $edgeTrends.Count -eq $caseIds.Count * 2 -and
    @($edgeTrends | Where-Object decreased -ne $true).Count -eq 0
$routeGatePass = [ordered]@{}
foreach ($route in $routes) {
    $pass = $true
    foreach ($row in $summary | Where-Object route -eq $route) {
        $key = Key $row
        $owner = @($owners | Where-Object { (Key $_) -eq $key })[0]
        $reference = @($referenceAudits | Where-Object {
            $_.case_id -eq $row.case_id -and $_.N -eq [int]$row.N -and
            $_.route -eq $route })[0]
        $rowPass = $false
        if ($row.status -eq 'ok') {
            $rowPass = $row.physical_converged -eq '1' -and
                $row.common_converged -eq '1' -and
                (Number $row.physical_final_residual "$key physical") -lt 2.0e-10 -and
                (Number $row.common_final_residual "$key common") -lt 2.0e-10 -and
                (Integer $owner.owner_query_count "$key owner query") -gt 0 -and
                (Integer $row.cauchy_geometry_queries "$key cauchy query") -gt 0 -and
                (Integer $row.cauchy_svd_factorizations "$key cauchy svd") -gt 0 -and
                (Integer $row.runtime_geometry_queries "$key runtime query") -eq 0 -and
                (Integer $row.runtime_svd_factorizations "$key runtime svd") -eq 0 -and
                $row.cauchy_fingerprint_before -eq $row.cauchy_fingerprint_after -and
                $owner.owner_fingerprint_before -eq $owner.owner_fingerprint_after -and
                $owner.owner_output_digest_before -eq $owner.owner_output_digest_after -and
                $reference.equal_to_g1
        }
        Record-Predicate $rowPass ("route_mandatory_gate:$key")
        $pass = $pass -and $rowPass
    }
    $routeGatePass[$route] = $pass
}
$directPass = $directComplete -and $routeGatePass['g1_value_g1_normal'] -and
    $routeGatePass['direct_cross_face_value'] -and $directIterationNoWorse -and
    $directOrderPass -and $directRatioPass -and $directNearEdgePass
$sharedPass = $sharedComplete -and
    $routeGatePass['g1_value_g1_normal'] -and
    $routeGatePass['direct_cross_face_value'] -and
    $routeGatePass['edge_reconstructed_value'] -and $primaryIterationPass -and
    $sharedOrderPass -and $sharedRatioPass -and $sharedNearEdgePass -and
    $sharedEdgeTrendPass
$sharedBetterIterationThanDirect = $false
foreach ($primary in $iterationMetrics | Where-Object route -eq 'edge_reconstructed_value') {
    $direct = @($iterationMetrics | Where-Object {
        $_.route -eq 'direct_cross_face_value' -and
        $_.rhs_kind -eq $primary.rhs_kind -and $_.N -eq $primary.N })
    if ($direct.Count -eq 1 -and
        ($primary.W -lt $direct[0].W -or $primary.S -lt $direct[0].S)) {
        $sharedBetterIterationThanDirect = $true
    }
}
$sharedBetterDefectThanDirect = $false
foreach ($primary in $nearEdge | Where-Object route -eq 'edge_reconstructed_value') {
    $direct = @($nearEdge | Where-Object { $_.case_id -eq $primary.case_id -and
        $_.N -eq $primary.N -and $_.route -eq 'direct_cross_face_value' })
    if ($direct.Count -eq 1 -and $primary.defect_linf -lt $direct[0].defect_linf -and
        $primary.defect_weighted_rms -lt $direct[0].defect_weighted_rms) {
        $sharedBetterDefectThanDirect = $true
    }
}

$formalEvidenceComplete = $sortedLevels.Count -eq 3 -and
    $sortedLevels[0] -eq 32 -and $sortedLevels[1] -eq 64 -and
    $sortedLevels[2] -eq 128
$mandatoryNumericalPass = $mandatoryFailedPredicates.Count -eq 0
$acceptancePass = $acceptanceFailedPredicates.Count -eq 0
$selectedRoute = 'insufficient_evidence'
if ($formalEvidenceComplete) {
    $selectedRoute = 'rerun_after_numerical_failure'
    if ($mandatoryNumericalPass) {
        $selectedRoute = 'sector_polynomials_with_shared_edge_constraints'
        if ($directPass) { $selectedRoute = 'direct_cross_face_value' }
        if ($sharedPass -and $sharedBetterIterationThanDirect -and
            $sharedBetterDefectThanDirect) {
            $selectedRoute = 'edge_reconstructed_value'
        } elseif (-not $directPass -and $sharedPass) {
            $selectedRoute = 'edge_reconstructed_value'
        }
    }
}

$decision = [ordered]@{
    schema_pass = $true
    formal_evidence_complete = $formalEvidenceComplete
    mandatory_numerical_pass = $mandatoryNumericalPass
    acceptance_pass = $acceptancePass
    expected_levels = @($ExpectedLevels | Sort-Object)
    summary_row_count = $summary.Count
    bin_row_count = $bins.Count
    owner_row_count = $owners.Count
    all_status_ok = (@($summary | Where-Object status -ne 'ok').Count -eq 0)
    iteration_metrics = @($iterationMetrics | ForEach-Object { $_ })
    adjacent_orders = @($orders | ForEach-Object { $_ })
    recorded_order_checks = @($recordedOrderChecks | ForEach-Object { $_ })
    near_edge_norms = @($nearEdge | ForEach-Object { $_ })
    n128_ratios = @($ratios | ForEach-Object { $_ })
    edge_error_trends = @($edgeTrends | ForEach-Object { $_ })
    raw_g1_reference_audits = @($referenceAudits | ForEach-Object { $_ })
    route_gate_pass = $routeGatePass
    primary_iteration_pass = $primaryIterationPass
    primary_W_no_worse = $primaryWNoWorse
    primary_W_strict = $primaryWStrict
    primary_S_no_worse = $primarySNoWorse
    primary_S_strict = $primarySStrict
    direct_complete = $directComplete
    direct_iteration_pass = $directIterationNoWorse
    direct_order_pass = $directOrderPass
    direct_n128_ratio_pass = $directRatioPass
    direct_near_edge_pass = $directNearEdgePass
    direct_pass = $directPass
    shared_complete = $sharedComplete
    shared_order_pass = $sharedOrderPass
    shared_n128_ratio_pass = $sharedRatioPass
    shared_near_edge_pass = $sharedNearEdgePass
    shared_edge_trend_pass = $sharedEdgeTrendPass
    shared_pass = $sharedPass
    shared_better_iteration_than_direct = $sharedBetterIterationThanDirect
    shared_better_defect_than_direct = $sharedBetterDefectThanDirect
    selected_route = $selectedRoute
    failed_predicates = @($failedPredicates | Sort-Object)
    mandatory_failed_predicates = @($mandatoryFailedPredicates | Sort-Object)
    acceptance_failed_predicates = @($acceptanceFailedPredicates | Sort-Object)
    numerical_pass = $mandatoryNumericalPass
}

if ([string]::IsNullOrWhiteSpace($DecisionJson)) {
    $DecisionJson = Join-Path $OutputDirectory 'decision.json'
}
$json = $decision | ConvertTo-Json -Depth 12
[IO.File]::WriteAllText($DecisionJson, $json + [Environment]::NewLine,
    (New-Object Text.UTF8Encoding($false)))
Write-Host ("audit complete: summary={0} bins={1} owners={2} failed={3}" -f
    $summary.Count, $bins.Count, $owners.Count, $failedPredicates.Count)
if ($failedPredicates.Count -gt 0 -and -not $AllowNumericalFailure) { exit 1 }
exit 0
