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

$domainAuditSource = @'
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Microsoft.VisualBasic.FileIO;

public sealed class NeumannDofBinAggregate {
    public string Key;
    public string Bin;
    public int Count;
    public double WeightSum;
    public double DensityLinf;
    public double DensitySquare;
    public double DefectLinf;
    public double DefectSquare;
}

public sealed class NeumannDetailKeyAggregate {
    public string Key;
    public int Count;
    public int MinId = Int32.MaxValue;
    public int MaxId = -1;
    public int MaxPatchId = -1;
    public double ValueRadiusMax;
    public double NormalRadiusMax;
    public double EdgeRadiusMax;
    public double ConditionMax;
    public double WeightSum;
    public double DensityLinf;
    public double DensitySquare;
    public double DefectLinf;
    public double DefectSquare;
}

public sealed class NeumannSurfaceAuditResult {
    public readonly List<NeumannDetailKeyAggregate> Keys =
        new List<NeumannDetailKeyAggregate>();
    public readonly List<string> MandatoryFailures = new List<string>();
}

public sealed class NeumannDofAuditResult {
    public readonly List<NeumannDetailKeyAggregate> Keys =
        new List<NeumannDetailKeyAggregate>();
    public readonly List<NeumannDofBinAggregate> Bins =
        new List<NeumannDofBinAggregate>();
}

public static class NeumannEvidenceDomainAudit {
    private static bool Finite(double value) {
        return !Double.IsNaN(value) && !Double.IsInfinity(value);
    }

    private static bool NearlyEqual(double left, double right) {
        if (!Finite(left) || !Finite(right)) return false;
        double scale = Math.Max(1.0, Math.Max(Math.Abs(left), Math.Abs(right)));
        return Math.Abs(left - right) <= 1.0e-12 * scale;
    }

    private static double FiniteResult(double value, string context) {
        if (!Finite(value))
            throw new InvalidDataException(context + " overflowed");
        return value;
    }

    private static Dictionary<string, int> Header(string[] fields) {
        Dictionary<string, int> result = new Dictionary<string, int>();
        for (int index = 0; index < fields.Length; ++index)
            result.Add(fields[index], index);
        return result;
    }

    private static string Key(string[] row, Dictionary<string, int> columns) {
        return row[columns["case_id"]] + "|" + row[columns["N"]] + "|"
            + row[columns["route"]];
    }

    private static double Number(string text, string context) {
        double value;
        if (text == "NA" || !double.TryParse(text, NumberStyles.Float,
                CultureInfo.InvariantCulture, out value)
            || Double.IsNaN(value) || Double.IsInfinity(value))
            throw new InvalidDataException(context + " is not finite numeric");
        return value;
    }

    private static int Integer(string text, string context) {
        int value;
        if (!Int32.TryParse(text, out value))
            throw new InvalidDataException(context + " is not an integer");
        return value;
    }

    private static int NonnegativeIntegerToken(
            string text, string context) {
        if (String.IsNullOrEmpty(text))
            throw new InvalidDataException(context + " is empty");
        foreach (char value in text) {
            if (value < '0' || value > '9')
                throw new InvalidDataException(
                    context + " is not an unsigned decimal integer");
        }
        int parsed;
        if (!Int32.TryParse(text, NumberStyles.None,
                CultureInfo.InvariantCulture, out parsed))
            throw new InvalidDataException(context + " exceeds Int32");
        return parsed;
    }

    private static List<int> PositiveCounts(string text, string context) {
        if (text == null || text.Length < 3 || text[0] != '{'
                || text[text.Length - 1] != '}')
            throw new InvalidDataException(
                context + " has invalid count-vector syntax");
        string body = text.Substring(1, text.Length - 2);
        if (body.Length == 0)
            throw new InvalidDataException(context + " has no counts");
        List<int> result = new List<int>();
        foreach (string token in body.Split(';')) {
            int value = NonnegativeIntegerToken(token, context + " count");
            if (value <= 0)
                throw new InvalidDataException(
                    context + " contains a nonpositive count");
            result.Add(value);
        }
        return result;
    }

    private static List<List<int>> SectorPatchIds(
            string text, string context) {
        if (text == null || text.Length < 4 || text[0] != '{'
                || text[text.Length - 1] != '}')
            throw new InvalidDataException(
                context + " has invalid nested-sector syntax");
        List<List<int>> result = new List<List<int>>();
        int position = 1;
        while (position < text.Length - 1) {
            if (text[position] != '{')
                throw new InvalidDataException(
                    context + " has invalid sector-group syntax");
            int close = text.IndexOf('}', position + 1);
            if (close < 0 || close == position + 1)
                throw new InvalidDataException(
                    context + " has an empty or unterminated sector group");
            string body = text.Substring(position + 1, close - position - 1);
            List<int> group = new List<int>();
            foreach (string token in body.Split(';'))
                group.Add(NonnegativeIntegerToken(
                    token, context + " patch ID"));
            result.Add(group);
            position = close + 1;
            if (position == text.Length - 1) break;
            if (text[position] != ';')
                throw new InvalidDataException(
                    context + " has invalid sector-group separator");
            ++position;
            if (position == text.Length - 1)
                throw new InvalidDataException(
                    context + " has a trailing sector separator");
        }
        if (position != text.Length - 1 || result.Count == 0)
            throw new InvalidDataException(
                context + " has invalid nested-sector cardinality");
        return result;
    }

    private static int ValidateSectorCounts(
            string sectorText, string countText, int expectedTotal,
            string context) {
        List<List<int>> sectors = SectorPatchIds(sectorText, context);
        List<int> counts = PositiveCounts(countText, context);
        if (sectors.Count != counts.Count)
            throw new InvalidDataException(
                context + " sector/count group cardinality mismatch");
        long total = 0;
        int maxPatchId = -1;
        for (int group = 0; group < sectors.Count; ++group) {
            total += counts[group];
            foreach (int patchId in sectors[group])
                maxPatchId = Math.Max(maxPatchId, patchId);
        }
        if (total != expectedTotal)
            throw new InvalidDataException(
                context + " count total mismatch");
        return maxPatchId;
    }

    private static TextFieldParser Parser(string path) {
        TextFieldParser parser = new TextFieldParser(path);
        parser.TextFieldType = FieldType.Delimited;
        parser.SetDelimiters(",");
        parser.HasFieldsEnclosedInQuotes = true;
        parser.TrimWhiteSpace = false;
        return parser;
    }

    public static void ValidateStrictCsv(string path, int expectedWidth) {
        const int StartField = 0;
        const int InUnquoted = 1;
        const int InQuoted = 2;
        const int AfterQuote = 3;
        int state = StartField;
        int fieldCount = 1;
        int record = 1;
        bool atRecordStart = true;
        bool skipLf = false;
        using (StreamReader reader = new StreamReader(path, true)) {
            char[] buffer = new char[65536];
            int length;
            while ((length = reader.Read(buffer, 0, buffer.Length)) != 0) {
                for (int index = 0; index < length; ++index) {
                    char value = buffer[index];
                    if (skipLf) {
                        skipLf = false;
                        if (value == '\n') continue;
                    }
                    if (state == InQuoted) {
                        atRecordStart = false;
                        if (value == '"') state = AfterQuote;
                        continue;
                    }
                    if (state == AfterQuote && value == '"') {
                        state = InQuoted;
                        atRecordStart = false;
                        continue;
                    }
                    if (value == ',') {
                        if (state == AfterQuote || state == InUnquoted
                            || state == StartField) {
                            ++fieldCount;
                            state = StartField;
                            atRecordStart = false;
                            continue;
                        }
                    }
                    if (value == '\r' || value == '\n') {
                        if (state != AfterQuote && state != InUnquoted
                            && state != StartField)
                            throw new InvalidDataException(
                                "CSV record " + record + " has invalid quote state");
                        if (fieldCount != expectedWidth)
                            throw new InvalidDataException(
                                "CSV record " + record + " has width "
                                + fieldCount + "; expected " + expectedWidth);
                        ++record;
                        fieldCount = 1;
                        state = StartField;
                        atRecordStart = true;
                        skipLf = value == '\r';
                        continue;
                    }
                    if (state == StartField) {
                        atRecordStart = false;
                        state = value == '"' ? InQuoted : InUnquoted;
                        continue;
                    }
                    if (state == InUnquoted && value == '"')
                        throw new InvalidDataException(
                            "CSV record " + record
                            + " has a bare quote in an unquoted field");
                    if (state == AfterQuote)
                        throw new InvalidDataException(
                            "CSV record " + record
                            + " has data after a closing quote");
                    atRecordStart = false;
                }
            }
        }
        if (state == InQuoted)
            throw new InvalidDataException(
                "CSV record " + record + " ends inside a quoted field");
        if (!atRecordStart) {
            if (fieldCount != expectedWidth)
                throw new InvalidDataException(
                    "CSV record " + record + " has width " + fieldCount
                    + "; expected " + expectedWidth);
            ++record;
        }
        if (record <= 1)
            throw new InvalidDataException("CSV file is empty");
    }

    public static NeumannSurfaceAuditResult AuditSurfaceFits(
            string path, string[] allKeys, string[] successfulKeys) {
        HashSet<string> successful = new HashSet<string>(
            successfulKeys, StringComparer.Ordinal);
        Dictionary<string, NeumannDetailKeyAggregate> states =
            new Dictionary<string, NeumannDetailKeyAggregate>(
                StringComparer.Ordinal);
        Dictionary<string, HashSet<int>> ids =
            new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);
        foreach (string key in allKeys) {
            NeumannDetailKeyAggregate aggregate =
                new NeumannDetailKeyAggregate();
            aggregate.Key = key;
            states.Add(key, aggregate);
            ids.Add(key, new HashSet<int>());
        }
        NeumannSurfaceAuditResult result = new NeumannSurfaceAuditResult();
        using (TextFieldParser parser = Parser(path)) {
            Dictionary<string, int> columns = Header(parser.ReadFields());
            while (!parser.EndOfData) {
                string[] row = parser.ReadFields();
                string key = Key(row, columns);
                if (!states.ContainsKey(key))
                    throw new InvalidDataException(
                        "orphan surface diagnostic " + key);
                int center = Integer(row[columns["center_dof"]],
                    key + " center_dof");
                if (center < 0 || !ids[key].Add(center))
                    throw new InvalidDataException(
                        "invalid or duplicate surface center ID "
                        + key + "|" + center);
                string entity = key + "|" + center;
                NeumannDetailKeyAggregate state = states[key];
                ++state.Count;
                state.MinId = Math.Min(state.MinId, center);
                state.MaxId = Math.Max(state.MaxId, center);
                int values = Integer(row[columns["ordinary_value_count"]],
                    entity + " ordinary_value_count");
                int normals = Integer(row[columns["normal_count"]],
                    entity + " normal_count");
                int edgeCount = Integer(row[columns["edge_count"]],
                    entity + " edge_count");
                if (values < 0 || normals < 0 || edgeCount < 0)
                    throw new InvalidDataException(
                        "negative surface fit count " + entity);
                string route = row[columns["route"]];
                if ((route == "edge_reconstructed_value"
                            && edgeCount > 6)
                        || (route != "edge_reconstructed_value"
                            && edgeCount != 0))
                    throw new InvalidDataException(
                        "surface edge count is outside its route domain "
                        + entity);
                int valueMaxPatchId = ValidateSectorCounts(
                    row[columns["value_sector_patch_ids"]],
                    row[columns["value_sector_counts"]], values,
                    entity + " value sectors");
                int normalMaxPatchId = ValidateSectorCounts(
                    row[columns["normal_sector_patch_ids"]],
                    row[columns["normal_sector_counts"]], normals,
                    entity + " normal sectors");
                state.MaxPatchId = Math.Max(state.MaxPatchId,
                    Math.Max(valueMaxPatchId, normalMaxPatchId));
                double valueRadius = Number(
                    row[columns["value_radius_over_h"]],
                    entity + " value_radius_over_h");
                double normalRadius = Number(
                    row[columns["normal_radius_over_h"]],
                    entity + " normal_radius_over_h");
                double edgeRadius = Number(
                    row[columns["edge_radius_over_h"]],
                    entity + " edge_radius_over_h");
                double edgeDistance = Number(
                    row[columns["edge_distance_over_h"]],
                    entity + " edge_distance_over_h");
                if (valueRadius < 0.0 || normalRadius < 0.0
                        || edgeRadius < 0.0 || edgeDistance < 0.0)
                    throw new InvalidDataException(
                        "negative surface fit radius " + entity);
                state.ValueRadiusMax = Math.Max(
                    state.ValueRadiusMax, valueRadius);
                state.NormalRadiusMax = Math.Max(
                    state.NormalRadiusMax, normalRadius);
                state.EdgeRadiusMax = Math.Max(
                    state.EdgeRadiusMax, edgeRadius);
                double sigmaMax = Number(row[columns["sigma_max"]],
                    entity + " sigma_max");
                double sigmaMin = Number(row[columns["sigma_min"]],
                    entity + " sigma_min");
                double condition = Number(row[columns["condition"]],
                    entity + " condition");
                if (!(sigmaMax > 0.0 && sigmaMin >= 0.0
                        && sigmaMin <= sigmaMax && condition >= 1.0))
                    throw new InvalidDataException(
                        "invalid surface fit SVD domain " + entity);
                state.ConditionMax = Math.Max(state.ConditionMax, condition);
                if (!successful.Contains(key)) continue;
                if (!(sigmaMin > 0.0)
                        || !NearlyEqual(condition, FiniteResult(
                            sigmaMax / sigmaMin,
                            entity + " surface condition ratio")))
                    throw new InvalidDataException(
                        "surface fit condition ratio mismatch " + entity);
                if (values != 48 || normals != 28)
                    result.MandatoryFailures.Add("surface_fit_counts:" + entity);
                if (!(sigmaMin > 3.0e-12 * sigmaMax))
                    result.MandatoryFailures.Add("surface_fit_rank:" + entity);
            }
        }
        foreach (NeumannDetailKeyAggregate state in states.Values) {
            if (state.Count != 0
                    && (state.MinId != 0 || state.MaxId != state.Count - 1))
                throw new InvalidDataException(
                    "surface center IDs are not contiguous " + state.Key);
            result.Keys.Add(state);
        }
        return result;
    }

    public static NeumannDofAuditResult AuditDofs(
            string path, string[] allKeys, string[] successfulKeys) {
        HashSet<string> successful = new HashSet<string>(
            successfulKeys, StringComparer.Ordinal);
        Dictionary<string, NeumannDetailKeyAggregate> states =
            new Dictionary<string, NeumannDetailKeyAggregate>(
                StringComparer.Ordinal);
        Dictionary<string, HashSet<int>> ids =
            new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);
        Dictionary<string, NeumannDofBinAggregate> bins =
            new Dictionary<string, NeumannDofBinAggregate>(
                StringComparer.Ordinal);
        foreach (string key in allKeys) {
            NeumannDetailKeyAggregate aggregate =
                new NeumannDetailKeyAggregate();
            aggregate.Key = key;
            states.Add(key, aggregate);
            ids.Add(key, new HashSet<int>());
            if (!successful.Contains(key)) continue;
            foreach (string bin in new string[] { "lt_h", "h_to_2h", "gt_2h" }) {
                NeumannDofBinAggregate binAggregate =
                    new NeumannDofBinAggregate();
                binAggregate.Key = key;
                binAggregate.Bin = bin;
                bins.Add(key + "|" + bin, binAggregate);
            }
        }
        using (TextFieldParser parser = Parser(path)) {
            Dictionary<string, int> columns = Header(parser.ReadFields());
            while (!parser.EndOfData) {
                string[] row = parser.ReadFields();
                string key = Key(row, columns);
                if (!states.ContainsKey(key))
                    throw new InvalidDataException("orphan diagnostic " + key);
                int dof = Integer(row[columns["dof_id"]], key + " dof_id");
                if (dof < 0 || !ids[key].Add(dof))
                    throw new InvalidDataException(
                        "invalid or duplicate DOF ID " + key + "|" + dof);
                string entity = key + "|" + dof;
                NeumannDetailKeyAggregate state = states[key];
                ++state.Count;
                state.MinId = Math.Min(state.MinId, dof);
                state.MaxId = Math.Max(state.MaxId, dof);
                int patchId = Integer(row[columns["patch_id"]],
                    entity + " patch_id");
                if (patchId < 0)
                    throw new InvalidDataException(
                        "negative DOF patch ID " + entity);
                foreach (string field in new string[] {
                        "point_x", "point_y", "point_z" })
                    Number(row[columns[field]], entity + " " + field);
                double weight = Number(
                    row[columns["weight"]], entity + " weight");
                double distance = Number(
                    row[columns["edge_distance_over_h"]],
                    entity + " edge_distance_over_h");
                if (!(weight > 0.0))
                    throw new InvalidDataException(
                        "nonpositive DOF weight " + entity);
                if (distance < 0.0)
                    throw new InvalidDataException(
                        "negative DOF edge distance " + entity);
                string densityText = row[columns["density_error"]];
                string defectText = row[columns["equation_defect"]];
                bool densityMissing = densityText == "NA";
                bool defectMissing = defectText == "NA";
                if (densityMissing != defectMissing)
                    throw new InvalidDataException(
                        "DOF post-solve value availability mismatch " + entity);
                if (densityText != "NA")
                    Number(densityText, entity + " density_error");
                if (defectText != "NA")
                    Number(defectText, entity + " equation_defect");
                if (!successful.Contains(key)) continue;
                if (densityText == "NA" || defectText == "NA")
                    throw new InvalidDataException(
                        "successful DOF lacks post-solve values " + entity);
                double density = Number(
                    densityText, entity + " density_error");
                double defect = Number(defectText, entity + " equation_defect");
                state.WeightSum = FiniteResult(state.WeightSum + weight,
                    entity + " total weight sum");
                state.DensityLinf = Math.Max(
                    state.DensityLinf, Math.Abs(density));
                double totalDensityTerm = FiniteResult(
                    weight * density * density,
                    entity + " total density weighted square");
                state.DensitySquare = FiniteResult(
                    state.DensitySquare + totalDensityTerm,
                    entity + " total density square sum");
                state.DefectLinf = Math.Max(
                    state.DefectLinf, Math.Abs(defect));
                double totalDefectTerm = FiniteResult(
                    weight * defect * defect,
                    entity + " total defect weighted square");
                state.DefectSquare = FiniteResult(
                    state.DefectSquare + totalDefectTerm,
                    entity + " total defect square sum");
                string bin = distance < 1.0 ? "lt_h"
                    : (distance <= 2.0 ? "h_to_2h" : "gt_2h");
                NeumannDofBinAggregate aggregate = bins[key + "|" + bin];
                ++aggregate.Count;
                aggregate.WeightSum = FiniteResult(
                    aggregate.WeightSum + weight,
                    entity + " bin weight sum");
                aggregate.DensityLinf = Math.Max(
                    aggregate.DensityLinf, Math.Abs(density));
                double densityTerm = FiniteResult(
                    weight * density * density,
                    entity + " density weighted square");
                aggregate.DensitySquare = FiniteResult(
                    aggregate.DensitySquare + densityTerm,
                    entity + " density square sum");
                aggregate.DefectLinf = Math.Max(
                    aggregate.DefectLinf, Math.Abs(defect));
                double defectTerm = FiniteResult(
                    weight * defect * defect,
                    entity + " defect weighted square");
                aggregate.DefectSquare = FiniteResult(
                    aggregate.DefectSquare + defectTerm,
                    entity + " defect square sum");
            }
        }
        NeumannDofAuditResult result = new NeumannDofAuditResult();
        foreach (NeumannDetailKeyAggregate state in states.Values) {
            if (state.Count != 0
                    && (state.MinId != 0 || state.MaxId != state.Count - 1))
                throw new InvalidDataException(
                    "DOF IDs are not contiguous " + state.Key);
            if (successful.Contains(state.Key)
                    && (!Finite(state.WeightSum)
                        || !Finite(state.DensityLinf)
                        || !Finite(state.DensitySquare)
                        || !Finite(state.DefectLinf)
                        || !Finite(state.DefectSquare)))
                throw new InvalidDataException(
                    "nonfinite raw DOF aggregate " + state.Key);
            result.Keys.Add(state);
        }
        foreach (NeumannDofBinAggregate aggregate in bins.Values) {
            if (!Finite(aggregate.WeightSum)
                    || !Finite(aggregate.DensityLinf)
                    || !Finite(aggregate.DensitySquare)
                    || !Finite(aggregate.DefectLinf)
                    || !Finite(aggregate.DefectSquare))
                throw new InvalidDataException(
                    "nonfinite raw bin aggregate "
                    + aggregate.Key + "|" + aggregate.Bin);
            if (aggregate.Count == 0
                    && (aggregate.WeightSum != 0.0
                        || aggregate.DensityLinf != 0.0
                        || aggregate.DensitySquare != 0.0
                        || aggregate.DefectLinf != 0.0
                        || aggregate.DefectSquare != 0.0))
                throw new InvalidDataException(
                    "nonzero empty raw bin "
                    + aggregate.Key + "|" + aggregate.Bin);
            if (aggregate.WeightSum > 0.0) {
                FiniteResult(Math.Sqrt(
                    aggregate.DensitySquare / aggregate.WeightSum),
                    aggregate.Key + "|" + aggregate.Bin
                    + " density RMS");
                FiniteResult(Math.Sqrt(
                    aggregate.DefectSquare / aggregate.WeightSum),
                    aggregate.Key + "|" + aggregate.Bin
                    + " defect RMS");
            }
            result.Bins.Add(aggregate);
        }
        return result;
    }
}
'@
if (-not ('NeumannEvidenceDomainAudit' -as [type])) {
    Add-Type -TypeDefinition $domainAuditSource -Language CSharp `
        -ReferencedAssemblies Microsoft.VisualBasic.dll
}

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

function UnsignedInteger($Value, [string]$Context) {
    $parsed = [uint64]0
    Assert-True ([uint64]::TryParse([string]$Value, [ref]$parsed)) `
        ("$Context is not an unsigned integer")
    return $parsed
}

function Is-NA($Value) {
    return [string]::IsNullOrWhiteSpace([string]$Value) -or $Value -eq 'NA'
}

function Add-IndexedRow($Index, [string]$IndexKey, $Row) {
    if (-not $Index.ContainsKey($IndexKey)) {
        $Index[$IndexKey] = New-Object System.Collections.ArrayList
    }
    [void]$Index[$IndexKey].Add($Row)
}

function Indexed-Rows($Index, [string]$IndexKey) {
    if ($Index.ContainsKey($IndexKey)) { return @($Index[$IndexKey]) }
    return @()
}

function Nearly-Equal([double]$Left, [double]$Right) {
    if ([double]::IsNaN($Left) -or [double]::IsInfinity($Left) -or
        [double]::IsNaN($Right) -or [double]::IsInfinity($Right)) {
        return $false
    }
    $scale = [Math]::Max(1.0, [Math]::Max([Math]::Abs($Left),
        [Math]::Abs($Right)))
    return [Math]::Abs($Left - $Right) -le 1.0e-12 * $scale
}

function Finite-Result([double]$Value, [string]$Context) {
    Assert-True (-not [double]::IsNaN($Value) -and
        -not [double]::IsInfinity($Value)) ("$Context is not finite")
    return $Value
}

function Finite-Add(
    [double]$Left, [double]$Right, [string]$Context) {
    return Finite-Result ($Left + $Right) $Context
}

function Finite-Multiply(
    [double]$Left, [double]$Right, [string]$Context) {
    return Finite-Result ($Left * $Right) $Context
}

function Finite-Divide(
    [double]$Numerator, [double]$Denominator, [string]$Context) {
    Assert-True ($Denominator -ne 0.0) ("$Context divides by zero")
    return Finite-Result ($Numerator / $Denominator) $Context
}

function Finite-Subtract(
    [double]$Left, [double]$Right, [string]$Context) {
    return Finite-Result ($Left - $Right) $Context
}

function Finite-SquareRoot([double]$Value, [string]$Context) {
    Assert-True ($Value -ge 0.0) ("$Context has a negative radicand")
    return Finite-Result ([Math]::Sqrt($Value)) $Context
}

function Finite-Log2([double]$Value, [string]$Context) {
    Assert-True ($Value -gt 0.0) ("$Context has a nonpositive argument")
    return Finite-Result ([Math]::Log($Value, 2.0)) $Context
}

function Assert-SectorSetSerialization($Value, [string]$Context) {
    $text = [string]$Value
    Assert-True ([regex]::IsMatch($text,
        '^\{\{[0-9]+(?:;[0-9]+)*\}(?:;\{[0-9]+(?:;[0-9]+)*\})*\}\z')) `
        ("$Context has invalid nested-sector syntax")
    foreach ($match in [regex]::Matches($text, '[0-9]+')) {
        [void](Integer $match.Value "$Context patch ID")
    }
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
$streamedNames = @('surface_fit_diagnostics.csv', 'dof_diagnostics.csv')
foreach ($name in $csvNames) {
    $path = Join-Path $OutputDirectory $name
    Assert-True (Test-Path -LiteralPath $path -PathType Leaf) ("missing $name")
    $header = Get-Content -LiteralPath $path -TotalCount 1
    Assert-True ($header -ceq $expectedHeaders[$name]) `
        ("$name header is not the exact ordered schema")
    $columnsByName[$name] = @($expectedHeaders[$name].Split(','))
    [NeumannEvidenceDomainAudit]::ValidateStrictCsv(
        $path, $columnsByName[$name].Count)
    if ($streamedNames -contains $name) {
        $tables[$name] = @()
    } else {
        $tables[$name] = @(Import-Csv -LiteralPath $path)
    }
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
Assert-UniqueKeys $residuals 'gmres_residuals.csv' {
    param($row)
    (Key $row) + '|' + $row.rhs_kind + '|' +
        (Integer $row.iteration ((Key $row) + ' residual iteration'))
}
Assert-UniqueKeys $owners 'owner_diagnostics.csv' { param($row) Key $row }
Assert-UniqueKeys $bins 'edge_distance_bins.csv' {
    param($row) (Key $row) + '|' + $row.distance_bin }
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
$allKeys = [string[]]@($summaryByKey.Keys)
$successfulKeys = [string[]]@($summary | Where-Object status -eq 'ok' |
    ForEach-Object { Key $_ })
$surfaceDomainAudit = [NeumannEvidenceDomainAudit]::AuditSurfaceFits(
    (Join-Path $OutputDirectory 'surface_fit_diagnostics.csv'),
    $allKeys, $successfulKeys)
$dofDomainAudit = [NeumannEvidenceDomainAudit]::AuditDofs(
    (Join-Path $OutputDirectory 'dof_diagnostics.csv'),
    $allKeys, $successfulKeys)
$surfaceAuditByKey = @{}
foreach ($aggregate in $surfaceDomainAudit.Keys) {
    $surfaceAuditByKey[$aggregate.Key] = $aggregate
}
$dofAuditByKey = @{}
foreach ($aggregate in $dofDomainAudit.Keys) {
    $dofAuditByKey[$aggregate.Key] = $aggregate
}

$ownerByKey = @{}
foreach ($row in $owners) { $ownerByKey[(Key $row)] = $row }
$binsByKey = @{}
foreach ($row in $bins) { Add-IndexedRow $binsByKey (Key $row) $row }
$residualsByKeyKind = @{}
foreach ($row in $residuals) {
    Add-IndexedRow $residualsByKeyKind ((Key $row) + '|' + $row.rhs_kind) $row
}
$edgePointsByKey = @{}
foreach ($row in $edgePoints) { Add-IndexedRow $edgePointsByKey (Key $row) $row }
$edgeFitsByKey = @{}
foreach ($row in $edgeFits) { Add-IndexedRow $edgeFitsByKey (Key $row) $row }

foreach ($key in $summaryByKey.Keys) {
    Assert-True ($ownerByKey.ContainsKey($key)) `
        ("missing owner row $key")
    $keyBins = @(Indexed-Rows $binsByKey $key)
    Assert-True ($keyBins.Count -eq 3) ("missing bin row $key")
    foreach ($bin in @('lt_h','h_to_2h','gt_2h')) {
        Assert-True (@($keyBins | Where-Object distance_bin -eq $bin).Count -eq 1) `
            ("missing bin $bin for $key")
    }
}
Assert-True ($owners.Count -eq $summary.Count) `
    'owner row count does not equal summary row count'
Assert-True ($bins.Count -eq 3 * $summary.Count) `
    'bin row count does not equal three rows per summary'
foreach ($row in $owners) {
    Assert-True ($summaryByKey.ContainsKey((Key $row))) `
        ("orphan owner row " + (Key $row))
}
foreach ($row in $bins) {
    Assert-True ($summaryByKey.ContainsKey((Key $row))) `
        ("orphan bin row " + (Key $row))
}

foreach ($row in $summary) {
    $key = Key $row
    $matchingOwner = $ownerByKey[$key]
    $matchingBins = @(Indexed-Rows $binsByKey $key)
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
    foreach ($field in $failureColumns) {
        Assert-True ([string]$row.$field -ceq 'NA') `
            ("successful summary carries failure metadata $key $field")
        Assert-True ([string]$matchingOwner.$field -ceq 'NA') `
            ("successful owner carries failure metadata $key $field")
        foreach ($binRow in $matchingBins) {
            Assert-True ([string]$binRow.$field -ceq 'NA') `
                ("successful bin carries failure metadata $key $field")
        }
    }
    Assert-True ((Number $row.h "$key h") -gt 0.0) `
        ("nonpositive mesh spacing $key")
    foreach ($field in @('patch_count','surface_dof_count','surface_map_count',
            'value_map_count','normal_map_count')) {
        Assert-True ((Integer $row.$field "$key $field") -gt 0) `
            ("nonpositive successful count $key $field")
    }
    foreach ($field in @('shared_edge_point_count','edge_map_count',
            'physical_iterations','common_iterations',
            'neighborhood_geometry_queries','cauchy_geometry_queries',
            'cauchy_svd_factorizations','runtime_geometry_queries',
            'runtime_svd_factorizations','label_inside_count',
            'label_outside_count')) {
        Assert-True ((Integer $row.$field "$key $field") -ge 0) `
            ("negative successful counter $key $field")
    }
    foreach ($field in @('physical_converged','common_converged',
            'reference_equal','label_equal','neighborhood_equal',
            'common_rhs_hash_equal')) {
        $flag = Integer $row.$field "$key $field"
        Assert-True ($flag -eq 0 -or $flag -eq 1) `
            ("invalid successful flag $key $field")
    }
    foreach ($field in @('physical_contraction','common_contraction',
            'density_linf','density_l2','interior_linf','interior_l2',
            'defect_linf','defect_rms','setup_seconds','fit_seconds',
            'pipeline_seconds','physical_solve_seconds',
            'common_solve_seconds')) {
        Assert-True ((Number $row.$field "$key $field") -ge 0.0) `
            ("negative successful numeric evidence $key $field")
    }
    $rhsMean = Number $row.common_rhs_mean "$key common_rhs_mean"
    $rhsRms = Number $row.common_rhs_rms "$key common_rhs_rms"
    Assert-True ([Math]::Abs($rhsMean) -le 5.0e-13) `
        ("common RHS mean tolerance exceeded $key")
    Assert-True ([Math]::Abs($rhsRms - 1.0) -le 5.0e-13) `
        ("common RHS RMS normalization mismatch $key")
    foreach ($field in @('common_rhs_hash','cauchy_fingerprint_before',
            'cauchy_fingerprint_after','owner_fingerprint_before',
            'owner_fingerprint_after','label_fingerprint',
            'neighborhood_fingerprint')) {
        [void](UnsignedInteger $row.$field "$key $field")
    }
    $ownerAvailable = Integer $matchingOwner.available "$key owner available"
    Assert-True ($ownerAvailable -eq 0 -or $ownerAvailable -eq 1) `
        ("invalid successful owner availability flag $key")
    foreach ($field in @('owner_query_count','cauchy_geometry_queries',
            'cauchy_svd_factorizations','runtime_geometry_queries',
            'runtime_svd_factorizations')) {
        Assert-True ((Integer $matchingOwner.$field "$key owner $field") -ge 0) `
            ("negative successful owner counter $key $field")
    }
    foreach ($field in @('reference_equal','label_equal',
            'neighborhood_equal','common_rhs_hash_equal')) {
        $flag = Integer $matchingOwner.$field "$key owner $field"
        Assert-True ($flag -eq 0 -or $flag -eq 1) `
            ("invalid successful owner flag $key $field")
    }
    foreach ($field in @('owner_fingerprint_before','owner_fingerprint_after',
            'owner_output_digest_before','owner_output_digest_after',
            'cauchy_fingerprint_before','cauchy_fingerprint_after')) {
        [void](UnsignedInteger $matchingOwner.$field "$key owner $field")
    }
    if ($row.route -eq 'edge_reconstructed_value') {
        $sharedPointCount = Integer $row.shared_edge_point_count `
            "$key shared_edge_point_count"
        Assert-True ($sharedPointCount -gt 0) `
            ("successful shared route has no edge points $key")
        foreach ($field in @('edge_value_linf','edge_value_rms')) {
            Assert-True ((Number $row.$field "$key $field") -ge 0.0) `
                ("negative shared edge error $key $field")
        }
    } else {
        $sharedPointCount = Integer $row.shared_edge_point_count `
            "$key shared_edge_point_count"
        Assert-True ($sharedPointCount -eq 0) `
            ("control route reports shared edge points $key")
        Assert-True ([string]$row.edge_value_linf -ceq 'NA' -and
            [string]$row.edge_value_rms -ceq 'NA') `
            ("control route reports edge error evidence $key")
    }
    foreach ($kind in @('physical','common')) {
        $iterations = if ($kind -eq 'physical') {
            Integer $row.physical_iterations "$key physical_iterations"
        } else {
            Integer $row.common_iterations "$key common_iterations"
        }
        $history = @(Indexed-Rows $residualsByKeyKind "$key|$kind" |
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
            Assert-True ($recordedResidual -ge 0.0) `
                ("negative residual $key $kind iteration=$iteration")
        }
        $summaryFinal = if ($kind -eq 'physical') {
            Number $row.physical_final_residual "$key physical final residual"
        } else {
            Number $row.common_final_residual "$key common final residual"
        }
        Assert-True ($summaryFinal -ge 0.0) `
            ("negative final residual $key $kind")
        $historyFinal = Number $history[-1].relative_residual `
            "$key $kind history final residual"
        $finalToleranceFactor = Finite-Multiply 64.0 `
            2.2204460492503131e-16 "$key $kind residual tolerance factor"
        $finalTolerance = Finite-Multiply $finalToleranceFactor `
            ([Math]::Max(1.0, [Math]::Abs($summaryFinal))) `
            "$key $kind residual tolerance"
        $finalDifference = [Math]::Abs((Finite-Subtract $summaryFinal `
            $historyFinal "$key $kind final residual difference"))
        Assert-True ($finalDifference -le
            $finalTolerance) ("final residual mismatch $key $kind")
    }
}

foreach ($row in $residuals) {
    Assert-True ($summaryByKey.ContainsKey((Key $row))) `
        ("orphan residual row " + (Key $row))
    Assert-True (@('physical','common') -contains $row.rhs_kind) `
        ("unknown rhs_kind " + $row.rhs_kind)
}
foreach ($row in $summary | Where-Object status -eq 'failed') {
    $key = Key $row
    foreach ($kind in @('physical','common')) {
        $history = @(Indexed-Rows $residualsByKeyKind "$key|$kind" |
            Sort-Object { [int]$_.iteration })
        for ($iteration = 0; $iteration -lt $history.Count; ++$iteration) {
            $recordedIteration = Integer $history[$iteration].iteration `
                "$key $kind iteration"
            Assert-True ($recordedIteration -eq $iteration) `
                ("residual iteration mismatch $key $kind")
            $failedResidual = Number $history[$iteration].relative_residual `
                "$key $kind residual"
            Assert-True ($failedResidual -ge 0.0) `
                ("negative residual $key $kind iteration=$iteration")
        }
        $iterationValue = if ($kind -eq 'physical') {
            $row.physical_iterations
        } else {
            $row.common_iterations
        }
        $finalValue = if ($kind -eq 'physical') {
            $row.physical_final_residual
        } else {
            $row.common_final_residual
        }
        $hasIterations = -not (Is-NA $iterationValue)
        $hasFinal = -not (Is-NA $finalValue)
        Assert-True ($hasIterations -eq $hasFinal) `
            ("summary solve availability mismatch $key $kind")
        if ($hasIterations) {
            Assert-True ($hasIterations -and $hasFinal) `
                ("complete residual summary unavailable $key $kind")
            $iterations = Integer $iterationValue "$key $kind iterations"
            Assert-True ($iterations -ge 0) `
                ("negative iteration count $key $kind")
            Assert-True ($history.Count -eq $iterations + 1) `
                ("residual history length mismatch $key $kind")
            $summaryFinal = Number $finalValue "$key $kind final residual"
            Assert-True ($summaryFinal -ge 0.0) `
                ("negative final residual $key $kind")
            $historyFinal = Number $history[-1].relative_residual `
                "$key $kind history final residual"
            $finalToleranceFactor = Finite-Multiply 64.0 `
                2.2204460492503131e-16 `
                "$key $kind residual tolerance factor"
            $finalTolerance = Finite-Multiply $finalToleranceFactor `
                ([Math]::Max(1.0, [Math]::Abs($summaryFinal))) `
                "$key $kind residual tolerance"
            $finalDifference = [Math]::Abs((Finite-Subtract $summaryFinal `
                $historyFinal "$key $kind final residual difference"))
            Assert-True ($finalDifference -le
                $finalTolerance) ("final residual mismatch $key $kind")
        } else {
            Assert-True ($history.Count -eq 0) `
                ("residual history exists without summary evidence $key $kind")
        }
    }
}
foreach ($row in @($edgePoints) + @($edgeFits)) {
    Assert-True ($row.route -eq 'edge_reconstructed_value') `
        ("control route invented shared-edge diagnostics " + (Key $row))
    Assert-True ($summaryByKey.ContainsKey((Key $row))) `
        ("orphan shared-edge diagnostic " + (Key $row))
}
$rawEdgeLinfByKey = @{}
foreach ($point in $edgePoints) {
    $summaryKey = Key $point
    $key = $summaryKey + '|' + $point.connection_id + '|' + $point.cell_id
    Assert-True ((Integer $point.connection_id "$key connection_id") -ge 0) `
        ("negative edge connection ID $key")
    Assert-True ((Integer $point.cell_id "$key cell_id") -ge 0) `
        ("negative edge cell ID $key")
    $fraction = Number $point.fraction "$key fraction"
    Assert-True ($fraction -gt 0.0 -and $fraction -lt 1.0) `
        ("edge fraction is outside the open unit interval $key")
    foreach ($field in @('native_parameter_0','native_parameter_1',
            'native_u_0','native_v_0','native_u_1','native_v_1')) {
        $native = Number $point.$field "$key $field"
        Assert-True ($native -ge 0.0 -and $native -le 1.0) `
            ("edge native coordinate is outside [0,1] $key $field")
    }
    foreach ($field in @('point_x','point_y','point_z','tangent_x','tangent_y',
            'tangent_z','mapped_tangent_dot','frame_determinant')) {
        [void](Number $point.$field "$key $field")
    }
    $mismatch = Number $point.position_mismatch "$key position_mismatch"
    $frameError = Number $point.frame_orthogonality_error `
        "$key frame_orthogonality_error"
    Assert-True ($mismatch -ge 0.0) ("negative position mismatch $key")
    Assert-True ($frameError -ge 0.0) ("negative frame error $key")
    Assert-SectorSetSerialization $point.sectors "$key sectors"
    $hasExact = -not (Is-NA $point.exact_value)
    $hasReconstructed = -not (Is-NA $point.reconstructed_value)
    $hasError = -not (Is-NA $point.error)
    Assert-True ($hasExact -eq $hasReconstructed -and
        $hasExact -eq $hasError) `
        ("edge value availability mismatch $key")
    if ($summaryByKey[$summaryKey].status -eq 'ok') {
        Assert-True $hasExact ("successful edge point lacks values $key")
    }
    if ($hasExact) {
        $exact = Number $point.exact_value "$key exact_value"
        $reconstructed = Number $point.reconstructed_value `
            "$key reconstructed_value"
        $edgeError = Number $point.error "$key error"
        $recomputedEdgeError = Finite-Subtract $reconstructed $exact `
            "$key recomputed edge error"
        Assert-True (Nearly-Equal $edgeError $recomputedEdgeError) `
            ("edge error identity mismatch $key")
        $absoluteError = [Math]::Abs($edgeError)
        if (-not $rawEdgeLinfByKey.ContainsKey($summaryKey)) {
            $rawEdgeLinfByKey[$summaryKey] = 0.0
        }
        $rawEdgeLinfByKey[$summaryKey] = [Math]::Max(
            $rawEdgeLinfByKey[$summaryKey], $absoluteError)
    }
}
$edgeFitMaxByKey = @{}
foreach ($fit in $edgeFits) {
    $summaryKey = Key $fit
    $key = $summaryKey + '|' + $fit.connection_id + '|' + $fit.cell_id
    Assert-True ((Integer $fit.connection_id "$key connection_id") -ge 0) `
        ("negative edge fit connection ID $key")
    Assert-True ((Integer $fit.cell_id "$key cell_id") -ge 0) `
        ("negative edge fit cell ID $key")
    foreach ($field in @('value_sector_0_count','value_sector_1_count',
            'normal_sector_0_count','normal_sector_1_count')) {
        Assert-True ((Integer $fit.$field "$key $field") -ge 0) `
            ("negative edge fit count $key $field")
    }
    foreach ($field in @('value_radius_over_h','normal_radius_over_h')) {
        $radius = Number $fit.$field "$key $field"
        Assert-True ($radius -ge 0.0) ("negative edge fit radius $key $field")
    }
    $sigmaMax = Number $fit.sigma_max "$key sigma_max"
    $sigmaMin = Number $fit.sigma_min "$key sigma_min"
    $condition = Number $fit.condition "$key condition"
    Assert-True ($sigmaMax -gt 0.0 -and $sigmaMin -ge 0.0 -and
        $sigmaMin -le $sigmaMax -and $condition -ge 1.0) `
        ("invalid edge fit SVD domain $key")
    if ($summaryByKey[$summaryKey].status -eq 'ok') {
        $conditionRatio = if ($sigmaMin -gt 0.0) {
            Finite-Divide $sigmaMax $sigmaMin "$key edge condition ratio"
        } else { 0.0 }
        Assert-True ($sigmaMin -gt 0.0 -and
            (Nearly-Equal $condition $conditionRatio)) `
            ("edge fit condition ratio mismatch $key")
    }
    if (-not $edgeFitMaxByKey.ContainsKey($summaryKey)) {
        $edgeFitMaxByKey[$summaryKey] = @{
            value_radius=0.0; normal_radius=0.0; condition=0.0 }
    }
    $edgeFitMaxByKey[$summaryKey].value_radius = [Math]::Max(
        $edgeFitMaxByKey[$summaryKey].value_radius,
        (Number $fit.value_radius_over_h "$key value_radius_over_h"))
    $edgeFitMaxByKey[$summaryKey].normal_radius = [Math]::Max(
        $edgeFitMaxByKey[$summaryKey].normal_radius,
        (Number $fit.normal_radius_over_h "$key normal_radius_over_h"))
    $edgeFitMaxByKey[$summaryKey].condition = [Math]::Max(
        $edgeFitMaxByKey[$summaryKey].condition, $condition)
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
    $keySurfaceAudit = $surfaceAuditByKey[$key]
    $keyDofAudit = $dofAuditByKey[$key]
    $keyEdgePoints = @(Indexed-Rows $edgePointsByKey $key)
    $keyEdgeFits = @(Indexed-Rows $edgeFitsByKey $key)
    $patchCount = Integer $row.patch_count "$key patch_count"
    Assert-True ($keySurfaceAudit.Count -eq $expectedSurfaceMaps) `
        ("surface map count mismatch $key")
    Assert-True ($keyDofAudit.Count -eq $expectedDofs) `
        ("DOF diagnostic count mismatch $key")
    Assert-True ($expectedSurfaceMaps -eq $expectedDofs -and
        $keySurfaceAudit.Count -eq $keyDofAudit.Count) `
        ("surface/DOF ID-set mismatch $key")
    Assert-True ($expectedValueMaps -eq $expectedSurfaceMaps -and
        $expectedNormalMaps -eq $expectedSurfaceMaps) `
        ("value/normal map count mismatch $key")
    Assert-True ($keySurfaceAudit.MaxPatchId -ge 0 -and
        $keySurfaceAudit.MaxPatchId -lt $patchCount) `
        ("surface sector patch ID is outside patch_count $key")
    Assert-True ($ownerByKey[$key].available -eq '1') `
        ("successful owner is unavailable $key")
    foreach ($field in @('value_radius_max_over_h','normal_radius_max_over_h',
            'edge_radius_max_over_h')) {
        $radius = Number $row.$field "$key $field"
        Assert-True ($radius -ge 0.0) ("negative summary radius $key $field")
    }
    $rawValueRadiusMax = $keySurfaceAudit.ValueRadiusMax
    $rawNormalRadiusMax = $keySurfaceAudit.NormalRadiusMax
    $rawConditionMax = $keySurfaceAudit.ConditionMax
    if ($edgeFitMaxByKey.ContainsKey($key)) {
        $rawValueRadiusMax = [Math]::Max($rawValueRadiusMax,
            $edgeFitMaxByKey[$key].value_radius)
        $rawNormalRadiusMax = [Math]::Max($rawNormalRadiusMax,
            $edgeFitMaxByKey[$key].normal_radius)
        $rawConditionMax = [Math]::Max($rawConditionMax,
            $edgeFitMaxByKey[$key].condition)
    }
    $recordedValueRadiusMax = Number $row.value_radius_max_over_h `
        "$key value_radius_max_over_h"
    $recordedNormalRadiusMax = Number $row.normal_radius_max_over_h `
        "$key normal_radius_max_over_h"
    $recordedEdgeRadiusMax = Number $row.edge_radius_max_over_h `
        "$key edge_radius_max_over_h"
    Assert-True (Nearly-Equal $recordedValueRadiusMax $rawValueRadiusMax) `
        ("summary value radius maximum mismatch $key")
    Assert-True (Nearly-Equal $recordedNormalRadiusMax $rawNormalRadiusMax) `
        ("summary normal radius maximum mismatch $key")
    Assert-True (Nearly-Equal $recordedEdgeRadiusMax `
        $keySurfaceAudit.EdgeRadiusMax) `
        ("summary edge radius maximum mismatch $key")
    $summaryCondition = Number $row.condition_max "$key condition_max"
    Assert-True ($summaryCondition -ge 1.0 -and
        (Nearly-Equal $summaryCondition $rawConditionMax)) `
        ("summary condition maximum mismatch $key")
    Assert-True ($keyDofAudit.WeightSum -gt 0.0) `
        ("nonpositive raw DOF total weight $key")
    $rawDefectMeanSquare = Finite-Divide $keyDofAudit.DefectSquare `
        $keyDofAudit.WeightSum "$key raw defect mean square"
    $rawDefectRms = Finite-SquareRoot $rawDefectMeanSquare `
        "$key raw defect RMS"
    $summaryDefectLinf = Number $row.defect_linf "$key defect_linf"
    $summaryDefectRms = Number $row.defect_rms "$key defect_rms"
    Assert-True (Nearly-Equal $summaryDefectLinf $keyDofAudit.DefectLinf) `
        ("summary defect Linf does not match raw DOFs $key")
    Assert-True (Nearly-Equal $summaryDefectRms $rawDefectRms) `
        ("summary defect RMS does not match raw DOFs $key")
    if ($row.route -eq 'edge_reconstructed_value') {
        $expectedSharedPoints = Integer $row.shared_edge_point_count `
            "$key shared_edge_point_count"
        Assert-True ($expectedSharedPoints -eq $expectedEdgeMaps) `
            ("shared edge point/map count mismatch $key")
        Assert-True ($keyEdgePoints.Count `
            -eq $expectedEdgeMaps) ("shared edge point count mismatch $key")
        Assert-True ($keyEdgeFits.Count `
            -eq $expectedEdgeMaps) ("shared edge fit count mismatch $key")
        Assert-True ($expectedEdgeMaps -gt 0) `
            ("successful shared route has no edge maps $key")
        $summaryEdgeLinf = Number $row.edge_value_linf "$key edge_value_linf"
        Assert-True ($rawEdgeLinfByKey.ContainsKey($key) -and
            (Nearly-Equal $summaryEdgeLinf $rawEdgeLinfByKey[$key])) `
            ("summary edge Linf does not match raw points $key")
        # The edge-point CSV has no quadrature weight, so RMS cannot be
        # reconstructed here; its finite/nonnegative domain and refinement
        # trend are checked from the summary evidence.
    } else {
        Assert-True ($expectedEdgeMaps -eq 0) `
            ("control route reports edge maps $key")
    }
}

$rawBinsByKey = @{}
foreach ($aggregate in $dofDomainAudit.Bins) {
    if (-not $rawBinsByKey.ContainsKey($aggregate.Key)) {
        $rawBinsByKey[$aggregate.Key] = @{}
    }
    $rawBinsByKey[$aggregate.Key][$aggregate.Bin] = [pscustomobject]@{
        count=$aggregate.Count; weight_sum=$aggregate.WeightSum;
        density_linf=$aggregate.DensityLinf;
        density_square=$aggregate.DensitySquare;
        defect_linf=$aggregate.DefectLinf;
        defect_square=$aggregate.DefectSquare }
}
$binAggregationChecks = New-Object System.Collections.Generic.List[object]
foreach ($row in $summary | Where-Object status -eq 'ok') {
    $key = Key $row
    $totalCount = 0
    foreach ($bin in Indexed-Rows $binsByKey $key) {
        $count = Integer $bin.count "$key $($bin.distance_bin) count"
        $weight = Number $bin.weight_sum "$key $($bin.distance_bin) weight"
        $densityLinf = Number $bin.density_linf "$key $($bin.distance_bin) density Linf"
        $densityRms = Number $bin.density_weighted_rms `
            "$key $($bin.distance_bin) density RMS"
        $defectLinf = Number $bin.defect_linf "$key $($bin.distance_bin) defect Linf"
        $defectRms = Number $bin.defect_weighted_rms `
            "$key $($bin.distance_bin) defect RMS"
        $empty = Integer $bin.empty "$key $($bin.distance_bin) empty"
        Assert-True ($count -ge 0 -and $weight -ge 0.0 -and
            $densityLinf -ge 0.0 -and $densityRms -ge 0.0 -and
            $defectLinf -ge 0.0 -and $defectRms -ge 0.0) `
            ("invalid successful bin norm domain $key|$($bin.distance_bin)")
        Assert-True (($empty -eq 1 -and $count -eq 0 -and $weight -eq 0.0) -or
            ($empty -eq 0 -and $count -gt 0 -and $weight -gt 0.0)) `
            ("bin empty/count/weight mismatch $key|$($bin.distance_bin)")
        if ($empty -eq 1) {
            Assert-True ($densityLinf -eq 0.0 -and $densityRms -eq 0.0 -and
                $defectLinf -eq 0.0 -and $defectRms -eq 0.0) `
                ("empty bin has nonzero norms $key|$($bin.distance_bin)")
        }
        $totalCount += $count
        $raw = $rawBinsByKey[$key][$bin.distance_bin]
        $rawDensityRms = if ($raw.weight_sum -gt 0.0) {
            $meanSquare = Finite-Divide $raw.density_square $raw.weight_sum `
                "$key $($bin.distance_bin) raw density mean square"
            Finite-SquareRoot $meanSquare `
                "$key $($bin.distance_bin) raw density RMS"
        } else { 0.0 }
        $rawDefectRms = if ($raw.weight_sum -gt 0.0) {
            $meanSquare = Finite-Divide $raw.defect_square $raw.weight_sum `
                "$key $($bin.distance_bin) raw defect mean square"
            Finite-SquareRoot $meanSquare `
                "$key $($bin.distance_bin) raw defect RMS"
        } else { 0.0 }
        $equal = $count -eq $raw.count -and
            (Nearly-Equal $weight $raw.weight_sum) -and
            (Nearly-Equal $densityLinf $raw.density_linf) -and
            (Nearly-Equal $densityRms $rawDensityRms) -and
            (Nearly-Equal $defectLinf $raw.defect_linf) -and
            (Nearly-Equal $defectRms $rawDefectRms)
        $binAggregationChecks.Add([ordered]@{ key=$key;
            distance_bin=$bin.distance_bin; equal=$equal })
    }
    $binAggregationChecks.Add([ordered]@{ key=$key;
        distance_bin='partition_total';
        equal=($totalCount -eq (Integer $row.surface_dof_count `
            "$key surface_dof_count")) })
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

foreach ($check in $binAggregationChecks) {
    Record-Predicate $check.equal `
        ("raw_bin_aggregation:$($check.key):$($check.distance_bin)")
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
    $owner = $ownerByKey[$key]
    $runtimeQueries = Integer $row.runtime_geometry_queries "$key runtime queries"
    $runtimeSvd = Integer $row.runtime_svd_factorizations "$key runtime SVD"
    Record-Predicate ($physicalFinal -lt 2.0e-10) ("physical_final_residual:$key")
    Record-Predicate ($commonFinal -lt 2.0e-10) ("common_final_residual:$key")
    Record-Predicate ($runtimeQueries -eq 0) ("runtime_geometry_queries:$key")
    Record-Predicate ($runtimeSvd -eq 0) ("runtime_svd_factorizations:$key")
    Record-Predicate ($row.cauchy_fingerprint_before -eq
        $row.cauchy_fingerprint_after) ("cauchy_fingerprint:$key")
    Record-Predicate ($owner.available -eq '1') ("owner_available:$key")
    Record-Predicate ((Integer $owner.owner_query_count "$key owner queries") -gt 0) `
        ("owner_queries:$key")
    Record-Predicate ((Integer $row.cauchy_geometry_queries "$key cauchy queries") -gt 0 -and
        (Integer $row.cauchy_svd_factorizations "$key cauchy SVD") -gt 0) `
        ("cauchy_preprocess_work:$key")
    Record-Predicate ($row.cauchy_geometry_queries -eq $owner.cauchy_geometry_queries -and
        $row.cauchy_svd_factorizations -eq $owner.cauchy_svd_factorizations -and
        $row.runtime_geometry_queries -eq $owner.runtime_geometry_queries -and
        $row.runtime_svd_factorizations -eq $owner.runtime_svd_factorizations) `
        ("summary_owner_counters:$key")
    Record-Predicate ($row.cauchy_fingerprint_before -eq
            $owner.cauchy_fingerprint_before -and
        $row.cauchy_fingerprint_after -eq $owner.cauchy_fingerprint_after -and
        $row.owner_fingerprint_before -eq $owner.owner_fingerprint_before -and
        $row.owner_fingerprint_after -eq $owner.owner_fingerprint_after) `
        ("summary_owner_fingerprints:$key")
    Record-Predicate ($owner.cauchy_fingerprint_before -eq
            $owner.cauchy_fingerprint_after -and
        $owner.owner_fingerprint_before -eq $owner.owner_fingerprint_after -and
        $owner.owner_output_digest_before -eq $owner.owner_output_digest_after) `
        ("owner_before_after:$key")
    Record-Predicate ($row.reference_equal -eq '1' -and $row.label_equal -eq '1' -and
        $row.neighborhood_equal -eq '1' -and $row.common_rhs_hash_equal -eq '1' -and
        $owner.reference_equal -eq '1' -and $owner.label_equal -eq '1' -and
        $owner.neighborhood_equal -eq '1' -and
        $owner.common_rhs_hash_equal -eq '1') ("owner_equal_flags:$key")
    Record-Predicate ($row.reference_equal -eq $owner.reference_equal -and
        $row.label_equal -eq $owner.label_equal -and
        $row.neighborhood_equal -eq $owner.neighborhood_equal -and
        $row.common_rhs_hash_equal -eq $owner.common_rhs_hash_equal) `
        ("summary_owner_equal_flags:$key")
}

$referenceAudits = New-Object System.Collections.Generic.List[object]
foreach ($caseId in $caseIds) {
    foreach ($level in $ExpectedLevels) {
        $g1Key = "$caseId|$level|g1_value_g1_normal"
        $g1Summary = $summaryByKey[$g1Key]
        $g1Owner = $ownerByKey[$g1Key]
        foreach ($route in $routes) {
            $key = "$caseId|$level|$route"
            $candidate = $summaryByKey[$key]
            $candidateOwner = $ownerByKey[$key]
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
    if ($summaryByKey[(Key $point)].status -ne 'ok') { continue }
    $key = (Key $point) + '|' + $point.connection_id + '|' + $point.cell_id
    $mismatch = Number $point.position_mismatch "$key mismatch"
    $tangent = Number $point.mapped_tangent_dot "$key tangent"
    $frameError = Number $point.frame_orthogonality_error "$key frame"
    $determinant = Number $point.frame_determinant "$key determinant"
    Record-Predicate ($mismatch -ge 0.0 -and $mismatch -le 1.0e-11) `
        ("edge_position_mismatch:$key")
    Record-Predicate ($tangent -ge 0.9999999999 -and
        $tangent -le 1.0 + 64.0 * 2.2204460492503131e-16) `
        ("edge_tangent:$key")
    Record-Predicate ($frameError -ge 0.0 -and $frameError -le 1.0e-10) `
        ("edge_frame_orthogonality:$key")
    Record-Predicate ($determinant -gt 0.0) ("edge_frame_determinant:$key")
}
foreach ($fit in $edgeFits) {
    if ($summaryByKey[(Key $fit)].status -ne 'ok') { continue }
    $key = (Key $fit) + '|' + $fit.connection_id + '|' + $fit.cell_id
    Record-Predicate ((Integer $fit.value_sector_0_count "$key value0") -eq 24 -and
        (Integer $fit.value_sector_1_count "$key value1") -eq 24 -and
        (Integer $fit.normal_sector_0_count "$key normal0") -eq 14 -and
        (Integer $fit.normal_sector_1_count "$key normal1") -eq 14) `
        ("edge_fit_counts:$key")
    $sigmaMax = Number $fit.sigma_max "$key sigma_max"
    $sigmaMin = Number $fit.sigma_min "$key sigma_min"
    Record-Predicate ($sigmaMin -gt 3.0e-12 * $sigmaMax) `
        ("edge_fit_rank:$key")
}
foreach ($failure in $surfaceDomainAudit.MandatoryFailures) {
    Record-Predicate $false $failure
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
                $errorRatio = Finite-Divide $coarseError $fineError `
                    "$caseId $route $field order ratio"
                $order = Finite-Log2 $errorRatio `
                    "$caseId $route $field order"
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
    $recordedDifference = [Math]::Abs((Finite-Subtract $recorded `
        $orderRow.order "$($orderRow.case_id) $column order difference"))
    $orderToleranceFactor = Finite-Multiply 256.0 `
        2.2204460492503131e-16 `
        "$($orderRow.case_id) $column order tolerance factor"
    $orderTolerance = Finite-Multiply $orderToleranceFactor `
        ([Math]::Max(1.0, [Math]::Abs($orderRow.order))) `
        "$($orderRow.case_id) $column order tolerance"
    $equal = $recordedDifference -le $orderTolerance
    $recordedOrderChecks.Add([ordered]@{case_id=$orderRow.case_id;
        route=$orderRow.route; column=$column; recorded=$recorded;
        recomputed=$orderRow.order; equal=$equal})
    Record-Predicate $equal ("recorded_order:$($orderRow.case_id):$($orderRow.route):$column")
}

$nearEdge = New-Object System.Collections.Generic.List[object]
foreach ($row in $summary | Where-Object status -eq 'ok') {
    $key = Key $row
    $near = @(Indexed-Rows $binsByKey $key | Where-Object {
        $_.distance_bin -in @('lt_h','h_to_2h') })
    $weight = 0.0
    $square = 0.0
    $linf = 0.0
    foreach ($bin in $near) {
        $w = Number $bin.weight_sum "$key near weight"
        Assert-True ($w -ge 0.0) ("negative near-edge weight $key")
        $rms = Number $bin.defect_weighted_rms "$key near rms"
        $weight = Finite-Add $weight $w "$key near-edge weight sum"
        $weightedRms = Finite-Multiply $w $rms `
            "$key near-edge weighted RMS factor"
        $weightedSquare = Finite-Multiply $weightedRms $rms `
            "$key near-edge weighted square"
        $square = Finite-Add $square $weightedSquare `
            "$key near-edge square sum"
        $linf = [Math]::Max($linf, (Number $bin.defect_linf "$key near linf"))
    }
    Assert-True ($weight -gt 0.0) ("nonpositive near-edge combined weight $key")
    $combinedMeanSquare = Finite-Divide $square $weight `
        "$key near-edge mean square"
    $combinedRms = Finite-SquareRoot $combinedMeanSquare `
        "$key near-edge combined RMS"
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
                $ratio = Finite-Divide $primaryError $controlError `
                    "$caseId primary/control $field ratio"
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
            $directRatio = Finite-Divide $directError $g1Error `
                "$caseId direct/g1 $field ratio"
            $directRatioPass = $directRatioPass -and
                $directRatio -le 1.10
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
        $owner = $ownerByKey[$key]
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
    raw_bin_aggregation_checks = @($binAggregationChecks | ForEach-Object { $_ })
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
