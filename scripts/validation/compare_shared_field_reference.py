#!/usr/bin/env python3
"""Compare hash-pinned fixture trees or reduced operators after basis alignment.

Never compares differently ordered physical/raw data silently. Fixture ordering is
checked before response data, and differing reduced bases are compared by ZZ^T.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows, columns = map(int, stream.readline().split())
        matrix = np.loadtxt(stream, ndmin=2)
    if matrix.shape != (rows, columns) or not np.isfinite(matrix).all():
        raise ValueError(f"nonfinite or malformed matrix: {path}")
    return matrix


def norm_inf(matrix):
    return float(np.max(np.sum(np.abs(matrix), axis=1)))


def difference(actual, expected, tolerance):
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    delta = np.abs(actual - expected)
    relative = norm_inf(actual - expected) / max(1., norm_inf(actual), norm_inf(expected))
    return {"passed": bool(relative <= tolerance), "relative_induced_inf": relative,
            "max_absolute": float(delta.max()), "absolute_quantiles": np.quantile(delta, [0., .5, .9, .99, 1.]).tolist(),
            "tolerance": tolerance}


def fixtures(reference, candidate, tolerance):
    manifest = json.loads((reference / "cases.json").read_text(encoding="utf-8"))
    provenance = json.loads((reference / "provenance.json").read_text(encoding="utf-8"))
    records = []
    for name, info in manifest["files"].items():
        expected_path, actual_path = reference / name, candidate / name
        with expected_path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != provenance["fixture_files_sha256"][name]:
            raise ValueError(f"reference fixture SHA256 mismatch: {name}")
        if not actual_path.is_file():
            records.append({"file": name, "passed": False, "error": "missing candidate file"})
            continue
        try:
            a, b = load(actual_path), load(expected_path)
        except (OSError, ValueError) as error:
            records.append({"file": name, "passed": False, "stage": "matrix loading", "error": str(error)})
            continue
        stem = expected_path.stem
        if stem == "Z":
            if a.shape != b.shape:
                records.append({"file": name, "passed": False, "error": "raw/reduced dimensions differ"})
            else:
                records.append({"file": name, "quantity": "ZZ^T", **difference(a@a.T, b@b.T, tolerance)})
            continue
        if stem == "fit_coefficients" and a.shape == b.shape:
            records.append({"file": name, "quantity": "lift and column scale", **difference(a[:, 1:3], b[:, 1:3], tolerance),
                            "coefficient_diagnostics_not_a_gate": difference(a[:, [0, 3]], b[:, [0, 3]], tolerance)})
            continue
        limit = 0. if info["integer"] else tolerance
        if stem in ("surface", "trace_points", "volume_samples", "raw_direction", "probe_points"):
            limit = 2e-14
        scale = 1.
        if stem.startswith("basis_d"):
            scale = (2./32)**(1 if len(stem) == len("basis_dx") else 2)
            limit = 2e-13
        records.append({"file": name, **difference(scale*a, scale*b, limit)})
    return {"mode": "fixture-tree", "metric": "induced infinity norm; derivatives scaled by h or h^2",
            "passed": all(r["passed"] for r in records), "comparisons": records}


def operators(reference, candidate, reference_z, candidate_z, tolerance, ill_conditioned):
    ar, ac, zr, zc = map(load, (reference, candidate, reference_z, candidate_z))
    if zr.shape != zc.shape or ar.shape != (zr.shape[1],)*2 or ac.shape != ar.shape:
        raise ValueError("raw dimensions/reduced dimensions/operator shapes differ")
    # Q converts reference reduced coordinates into candidate reduced coordinates.
    ident = np.eye(zr.shape[1])
    for label, z in (("reference", zr), ("candidate", zc)):
        if norm_inf(z.T@z - ident) > 1e-10:
            raise ValueError(f"{label} Z is not Euclidean orthonormal; provide its coordinate metric explicitly")
    subspace = difference(zc@zc.T, zr@zr.T, tolerance)
    if not subspace["passed"]:
        return {"mode": "operator", "passed": False, "stage": "constraint subspace ZZ^T", "subspace": subspace}
    q = zc.T @ zr
    aligned = q.T @ ac @ q
    matrix = difference(aligned, ar, tolerance)
    sr, sc = np.linalg.svd(ar, compute_uv=False), np.linalg.svd(aligned, compute_uv=False)
    def finite_condition(singular_values):
        if singular_values[-1] == 0.:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            value = singular_values[0] / singular_values[-1]
        return float(value) if np.isfinite(value) else None

    cr, cc = finite_condition(sr), finite_condition(sc)
    cond_delta = None
    if cr is not None and cc is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            delta = abs(np.float64(cc)/np.float64(cr) - 1.)
        if np.isfinite(delta):
            cond_delta = float(delta)
    condition_pass = bool(ill_conditioned or (cond_delta is not None and cond_delta <= .01))
    return {"mode": "operator", "passed": bool(matrix["passed"] and condition_pass), "subspace": subspace,
            "aligned_matrix": matrix, "matrix_perturbation_2norm": float(np.linalg.norm(aligned-ar, 2)),
            "reference_sigma_min": float(sr[-1]), "candidate_sigma_min": float(sc[-1]),
            "reference_sigma_max": float(sr[0]), "candidate_sigma_max": float(sc[0]),
            "reference_exactly_singular": bool(sr[-1] == 0.), "candidate_exactly_singular": bool(sc[-1] == 0.),
            "reference_condition": cr, "candidate_condition": cc,
            "condition_relative_difference": cond_delta, "condition_one_percent_gate_applied": not ill_conditioned,
            "candidate_condition_under_100_screen": bool(cc is not None and cc < 100),
            "note": "Direct-variant comparison success certifies aligned matrix agreement, not invertibility or conditioning. Undefined/nonfinite condition diagnostics are null; exact singularity is reported separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--reference-z", type=Path)
    parser.add_argument("--candidate-z", type=Path)
    parser.add_argument("--ill-conditioned-direct", action="store_true")
    args = parser.parse_args()
    if args.tolerance <= 0 or not np.isfinite(args.tolerance):
        parser.error("tolerance must be finite and positive")
    try:
        if args.reference.is_dir():
            report = fixtures(args.reference, args.candidate, args.tolerance)
        else:
            if not args.reference_z or not args.candidate_z:
                parser.error("operator comparison requires both raw-coordinate Z matrices")
            report = operators(args.reference, args.candidate, args.reference_z, args.candidate_z,
                               args.tolerance, args.ill_conditioned_direct)
    except (OSError, ValueError, KeyError) as error:
        report = {"passed": False, "stage": "input/provenance/alignment validation", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
