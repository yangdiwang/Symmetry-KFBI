#!/usr/bin/env python3
"""Summarize an explicitly selected shared-field run matrix, preserving failures."""
import argparse
import itertools
import json
import math
from pathlib import Path


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


def key(row):
    return int(row["N"]), row["transform"], row["bvp"], row["variant"]


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def valid_saved_matrix(path):
    try:
        with path.open(encoding="utf-8") as stream:
            rows, cols = map(int, stream.readline().split())
            values = [float(v) for line in stream for v in line.split()]
        return rows > 0 and cols > 0 and len(values) == rows*cols and all(map(math.isfinite, values))
    except (OSError, ValueError):
        return False


def summarize(root, reference):
    config = json.loads((root / "study_config.json").read_text(encoding="utf-8-sig"))
    execution = json.loads((root / "execution_status.json").read_text(encoding="utf-8-sig"))
    expected = set(itertools.product(config["levels"], config["transforms"], config["bvps"], config["variants"]))
    old = {key(r): r for r in json.loads(reference.read_text(encoding="utf-8"))["cases"]}
    records = []
    seen = set()
    for run in execution:
        identity = key(run)
        if identity in seen or identity not in expected:
            raise ValueError(f"duplicate or unrequested configuration: {identity}")
        seen.add(identity)
        row = dict(run)
        path = root / run["result_path"]
        issues = []
        if not path.is_file():
            row.update(numerical_acceptance=False, status="MISSING_RESULT", issues=["result.json absent"])
            records.append(row)
            continue
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        if not finite_tree(result):
            issues.append("NaN or infinity")
        residual = result.get("projected_relative_residual")
        iterations = result.get("gmres_iterations")
        error = result.get("interior_linf")
        if not isinstance(residual, (int, float)) or not math.isfinite(residual) or residual >= 3e-10:
            issues.append("true projected relative residual fails 3e-10 certification")
        if result.get("gmres_converged") is not True:
            issues.append("GMRES convergence flag false or absent")
        if error is None or not isinstance(error, (float, int)) or not math.isfinite(error):
            issues.append("physical interior error absent or nonfinite")
        if run.get("timed_out") or run.get("resource_failure"):
            issues.append("process timeout or resource limit")
        if run.get("exit_code") != 0:
            issues.append("process exit code is nonzero or missing")
        if config.get("verify_replay", False) and result.get("replay_verified") is not True:
            issues.append("saved coefficient one-forward replay was not verified")
        variant = run["variant"]
        expected_backend = "direct_cauchy" if variant == "baseline" else "shared_field"
        expected_target = "input_jump_half" if variant == "shared_jump" else "raw"
        if result.get("correction_backend") != expected_backend or result.get("exterior_target") != expected_target:
            issues.append("resolved backend/target missing or differs from explicit configuration")
        if variant != "baseline" and result.get("shared_field_restrict") != "staged":
            issues.append("resolved restrict mode missing or differs")
        for diagnostic in ("raw_value_linf", "raw_normal_linf", "equation_value_linf", "equation_normal_linf"):
            value = result.get(diagnostic)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                issues.append(f"missing/nonfinite separate trace diagnostic: {diagnostic}")
        saved = ["density_coefficients", "reduced_coordinates", "raw_value_trace", "raw_normal_trace", "equation_value_trace", "equation_normal_trace"]
        if variant != "baseline":
            saved += ["field_coefficients", "requested_value_jump", "requested_normal_jump", "fitted_value_jump", "fitted_normal_jump"]
        for name in saved:
            if not valid_saved_matrix(path.parent / (name + ".txt")):
                issues.append(f"missing/malformed/nonfinite saved array: {name}")
        if run["N"] == 32 and config.get("spectrum32", False):
            for name in ("reduced_operator", "density_nullspace", "operator_singular_values"):
                if not valid_saved_matrix(path.parent / (name + ".txt")):
                    issues.append(f"missing/malformed/nonfinite spectrum array: {name}")
            condition = result.get("operator_condition")
            if not isinstance(condition, (float, int)) or not math.isfinite(condition):
                issues.append("operator condition unavailable or nonfinite")
            elif variant == "shared_jump" and condition >= 100:
                issues.append("candidate fails N32 condition<100 coarse screen")
        # Direct coarse-grid examples are retained as failed physical models even
        # when their projected equation happens to meet the GMRES tolerance.
        known_direct_failure = variant == "shared_direct" and run["N"] == 32
        if known_direct_failure:
            issues.append("known N32 direct research failure; requires physical/spectral review")
        prior = old.get(identity)
        investigation = []
        if prior and variant != "baseline":
            if isinstance(error, (int, float)) and math.isfinite(error):
                relative_error_difference = abs(error - prior["interior_linf"]) / max(prior["interior_linf"], 1e-300)
                row["package_interior_relative_difference"] = relative_error_difference
                if relative_error_difference > .05:
                    investigation.append("interior error differs from package by more than 5 percent")
            if isinstance(iterations, (int, float)):
                row["package_gmres_iteration_difference"] = iterations - prior["gmres"]
                if abs(iterations - prior["gmres"]) > 2:
                    investigation.append("GMRES iteration count differs from package by more than two")
        row.update(numerical_acceptance=not issues, issues=issues, investigation=investigation,
                   status="KNOWN_DIRECT_RESEARCH_FAILURE" if known_direct_failure else ("PASS" if not issues else "FAIL"),
                   gmres_iteration_tolerance=2e-10, true_residual_certification_bound=3e-10,
                   true_relative_residual=residual, gmres_iterations=iterations, interior_linf=error,
                   replay_verified=result.get("replay_verified", False))
        row["timings"] = {k: v for k, v in result.items() if k.endswith("_seconds")}
        row["field_and_trace_diagnostics"] = {k: v for k, v in result.items() if k.startswith(("field_", "raw_", "equation_"))}
        row["operator_diagnostics"] = {k: v for k, v in result.items() if k.startswith("operator_")}
        records.append(row)
    missing = sorted(expected - seen)
    rates = []
    for pose, bvp, variant in itertools.product(config["transforms"], config["bvps"], config["variants"]):
        series = sorted((r for r in records if (r["transform"], r["bvp"], r["variant"]) == (pose, bvp, variant)), key=lambda r: r["N"])
        for low, high in zip(series, series[1:]):
            e0, e1 = low.get("interior_linf"), high.get("interior_linf")
            valid = isinstance(e0, (int, float)) and isinstance(e1, (int, float)) and e0 > 0 and e1 > 0 and math.isfinite(e0/e1)
            rate = math.log(e0/e1)/math.log(high["N"]/low["N"]) if valid else None
            rates.append(dict(transform=pose, bvp=bvp, variant=variant, interval=f"{low['N']}->{high['N']}", order=rate))
    candidate = [r for r in records if r["variant"] == "shared_jump"]
    complete = not missing and len(records) == len(expected) and all((root / r["result_path"]).is_file() for r in records)
    candidate_requested = "shared_jump" in config["variants"]
    candidate_acceptance = (bool(candidate) and all(r["numerical_acceptance"] for r in candidate)
                            and not any(k[3] == "shared_jump" for k in missing)) if candidate_requested else None
    return json_safe({"scope": config, "expected_cases": len(expected), "recorded_cases": len(records),
            "missing": missing, "configuration_matrix_complete": complete,
            "candidate_numerical_acceptance": candidate_acceptance,
            "selected_numerical_acceptance": complete and bool(records) and all(r["numerical_acceptance"] for r in records),
            "all_saved_replays_verified": bool(records) and all(r.get("replay_verified", False) for r in records),
            "baseline_reference_identity": "C++ baseline is a distinct discretization until physical/density/operator alignment is established",
            "rates": rates, "cases": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[2] / "tests/cases/shared_field_3d/package_reference_summary.json")
    args = parser.parse_args()
    summary = summarize(args.run_dir, args.reference)
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["| N | pose | BVP | method | status | GMRES | true residual | interior Linf |", "|---:|---|---|---|---|---:|---:|---:|"]
    for r in summary["cases"]:
        lines.append(f"| {r['N']} | {r['transform']} | {r['bvp']} | {r['variant']} | {r['status']} | {r.get('gmres_iterations', '')} | {r.get('true_relative_residual', '')} | {r.get('interior_linf', '')} |")
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cases", "rates", "scope")}, indent=2))
    # A control-only scope has no candidate to accept or reject. Preserve the
    # established candidate-focused full-matrix status, but require every
    # selected control row to pass when no shared_jump case was requested.
    acceptance = (summary["selected_numerical_acceptance"] if summary["candidate_numerical_acceptance"] is None
                  else summary["candidate_numerical_acceptance"])
    return 0 if summary["configuration_matrix_complete"] and acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
