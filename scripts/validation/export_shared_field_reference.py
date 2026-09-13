#!/usr/bin/env python3
"""Export small, hash-pinned Python reference data; Python is never a C++ dependency.

The explicit source directory must contain the reviewed source from the specified
archive. Source modules are imported only after their hashes and archive membership
have been checked. No historical solution coefficients are used to generate fits.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import zipfile

VERSION = "1.1.0"
PACKAGE_SHA256 = "23cfa03489e1d1004a57f39703e1f4237d83afb6da096fed7cc7033ecb884ce5"
SOURCE_SHA256 = {
    "kfbi_shared.py": "77c4a682f1fa016a43bae63a1ad01667826d9ea7948cb82ed928f250b321f44d",
    "shared_field_base.py": "754ebea429426415b01e9142dbccc736d0063797fb8faf08e0077df75b74d6fc",
    "planar_neumann.py": "f6e9cb6deda85dd134afb3b09d5988bcf453780fe7f8e40d1d3ced485d8613f7",
    "planar_dirichlet.py": "ab5fcec342d3293671400f7911827abb60588ca4e50bed60f4a282dd55fc4f97",
    "polar_planar_experiment.py": "03567241fe3c38023b42b843f2a6ef084c39447dfa6509c993402434c4192ba5",
    "polar_feature_constraints.py": "fd4d11fbdbd00b8fa91bf25af31c9127c90e964c7b51b9cd79a6b3fa1465db2d",
}
ALLOWED_RESULTS = {"baseline_fresh", "baseline_pose2", "validated_small", "raw", "remaining128"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_source(source, archive):
    if sha256(archive) != PACKAGE_SHA256:
        raise ValueError("source archive SHA256 does not match reviewed package")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name, expected in SOURCE_SHA256.items():
            local = source / "src" / name
            matching = [n for n in names if n.endswith("/src/" + name) and "/upstream/" not in n and "/prototype/" not in n]
            if len(matching) != 1 or sha256(local) != expected:
                raise ValueError(f"unreviewed source file: {name}")
            if hashlib.sha256(zf.read(matching[0])).hexdigest() != expected:
                raise ValueError(f"archive source differs: {name}")


def export_primary_records(source, output):
    table = source / "results/summary/all_cases.csv"
    records = []
    used = {table.relative_to(source).as_posix(): sha256(table)}
    for row in csv.DictReader(table.read_text(encoding="utf-8").splitlines()):
        relative = Path(row["source"])
        if len(relative.parts) < 4 or ".." in relative.parts or relative.parts[0] != "results" or relative.parts[1] not in ALLOWED_RESULTS:
            raise ValueError(f"non-primary source in primary table: {relative}")
        file = source / relative
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("moment_constraint") or data.get("operators", {}).get("extension", {}).get("ratio", 4) != 4:
            raise ValueError("moment/H2 record is not a primary reference")
        keep = ("geometry", "bvp", "transform", "N", "nfull", "nred", "gmres", "info",
                "relative_true_residual", "interior_linf", "core_linf", "density_linf", "status")
        records.append({k: data[k] for k in keep} | {"variant": row["variant"], "source": relative.as_posix()})
        used[relative.as_posix()] = sha256(file)
    keys = {(r["N"], r["transform"], r["bvp"], r["variant"]) for r in records}
    if len(records) != 36 or len(keys) != 36:
        raise ValueError("primary table must contain exactly 36 unique configurations")
    write_json(output / "package_reference_summary.json", {"identity": "package-recorded, not local rerun", "cases": records})
    return used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--N", type=int, default=32, choices=[32])
    parser.add_argument("--transforms", nargs="+", choices=["rotate", "rotate_translate"], default=["rotate"])
    parser.add_argument("--bvps", nargs="+", choices=["dirichlet", "neumann"], default=["dirichlet", "neumann"])
    parser.add_argument("--full", action="store_true", help="include complete Z and transfer observations for runtime comparison")
    parser.add_argument("--run-cases", action="store_true", help="also independently rerun 12 N32 PDE cases; no larger grids")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    verify_source(source, args.source_archive)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be a new directory; refusing to overwrite provenance")
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "1"
    # Pin optional upstream knobs; do not inherit experiment settings.
    os.environ.update(KFBI_CENTER_POLICY="trace_first", KFBI_EVENT_POLICY="current",
                      KFBI_EDGE_STAR_WEIGHT="0.1", KFBI_VERTEX_STAR_WEIGHT="0.0", KFBI_VERTEX_STAR_POWER="0.0",
                      KFBI_TRACE_FIRST_EVENT_RADIUS="1.75", KFBI_TRACE_FIRST_TARGET_RADIUS="3.25")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "src"))
    import numpy as np
    import scipy
    import pandas
    import kfbi_shared as ks
    from shared_field_base import polynomial, POW

    args.output.mkdir(parents=True, exist_ok=True)
    used = export_primary_records(source, args.output)
    manifest = {}

    def save(directory, name, array, integer=False):
        a = np.asarray(array)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or not np.isfinite(a).all():
            raise ValueError(f"invalid fixture matrix {name}")
        path = directory / (name + ".txt")
        np.savetxt(path, a, fmt="%d" if integer else "%.17e", header=f"{a.shape[0]} {a.shape[1]}", comments="")
        manifest[path.relative_to(args.output).as_posix()] = {"shape": list(a.shape), "integer": integer}

    cases = []
    for pose in args.transforms:
        for bvp in args.bvps:
            mod = ks.pn if bvp == "neumann" else ks.pd
            mod.CENTER_POLICY = "trace_first"
            geom = mod.PlanarPrism("u", mod.TRANSFORMS[pose])
            grid = mod.Grid(-1, 1, args.N)
            exact = mod.ExactHarmonic(geom.T)
            exact.shift = mod.boundary_mean(geom, exact)
            density = mod.Space(geom, grid.h, exact, bvp, 8.)
            policy = ks.polar_policy_only(density, exact) if bvp == "neumann" else {
                "cp": density.cp, "Z": density.Z, "C": density.C, "d": density.d}
            traces = mod.traces(density)
            ext = ks.LinearExtension(geom, grid, density, exact, bvp)
            ops = ks.SharedOperators(mod, grid, geom, density, traces, ext, bvp, "staged")
            sf = ext.sf
            directory = args.output / "fixtures" / f"u_{bvp}_{pose}_N{args.N}"
            directory.mkdir(parents=True)
            rng = np.random.default_rng(51209)
            reduced_direction = rng.normal(size=policy["Z"].shape[1])
            raw = policy["Z"] @ reduced_direction
            alpha = ext.solve(raw, known=False)
            a, b = ext.data(raw, False)
            lift = ext.lift_lattice @ (ext.lift_map @ a)
            q = np.r_[ext.wa * a, ext.wb * b, np.zeros(len(ext.wp))]
            z = (alpha - lift) * ext.scale
            target = q - ext.A @ (ext.scale * lift)
            optimality = ext.A.T @ (ext.A @ z - target) + ext.ridge * z
            rhs, cv, cn = ops.evaluate(alpha)
            u = mod.Poisson(grid).solve(rhs)
            x = np.array([t.x for t in traces])
            normal = np.array([t.n for t in traces])
            trace_local = (x - geom.T.t) @ geom.T.R
            local_normals = normal @ geom.T.R
            Tv = sf.matrix(trace_local)
            Tn = sum(sf.matrix(trace_local, np.eye(3, dtype=int)[j]).multiply(local_normals[:, j, None]) for j in range(3))
            Btrace = ks.density_sampling_matrix(density, x, np.array([t.pid for t in traces]))
            requested_v = Btrace @ raw if bvp == "neumann" else np.zeros(len(traces))
            requested_n = Btrace @ raw if bvp == "dirichlet" else np.zeros(len(traces))
            raw_v, raw_n = ops.Rg_value @ u + cv, ops.Rg_normal @ u + cn
            # An actual fitted harmonic P2, separate from direct lift injection.
            terms = {(0, 0, 0): 1., (1, 0, 0): .2, (0, 1, 0): .1, (0, 0, 1): -.3,
                     (1, 1, 0): .4, (2, 0, 0): .2, (0, 2, 0): -.2}
            pc = np.array([terms.get(tuple(p), 0.) for p in POW])
            p2a = polynomial(sf.sx) @ pc
            ns = np.vstack([sf.g.rects[k].normal() for k in sf.so])
            p2b = sum((polynomial(sf.sx, np.eye(3, dtype=int)[j]) @ pc) * ns[:, j] for j in range(3))
            p2lift = ext.lift_lattice @ (ext.lift_map @ p2a)
            p2q = np.r_[ext.wa * p2a, ext.wb * p2b, np.zeros(len(ext.wp))]
            p2target = p2q - ext.A @ (ext.scale * p2lift)
            p2z = ext.lu.solve(ext.A.T @ p2target)
            for _ in range(2):
                p2z += ext.lu.solve(ext.A.T @ (p2target - ext.A @ p2z) - ext.ridge * p2z)
            p2fit = p2lift + p2z / ext.scale
            p2rhs, p2cv, p2cn = ops.evaluate(p2fit)
            p2u = mod.Poisson(grid).solve(p2rhs)
            p2check = {"field_coefficient_linf": float(np.max(abs(p2fit - ext.lift_lattice @ pc))),
                       "raw_value_linf": float(np.max(abs(ops.Rg_value @ p2u + p2cv))),
                       "raw_normal_linf": float(np.max(abs(ops.Rg_normal @ p2u + p2cn)))}
            if p2check["raw_value_linf"] > 2e-12 or p2check["raw_normal_linf"] > 2e-11:
                raise AssertionError(f"actual P2 fit transfer failed: {p2check}")

            save(directory, "field_indices", sf.indices, True)
            save(directory, "active_cells", sf.cells, True)
            save(directory, "surface", np.c_[sf.sx, ns, sf.sw, sf.so])
            vi = np.unique(np.linspace(0, len(sf.vx) - 1, 32, dtype=int))
            save(directory, "volume_samples", np.c_[vi, sf.vx[vi], sf.vw[vi]])
            save(directory, "raw_direction", raw)
            save(directory, "particular", policy["cp"])
            save(directory, "requested_surface", np.c_[a, b, p2a, p2b])
            save(directory, "fit_coefficients", np.c_[alpha, lift, ext.scale, p2fit])
            qi = np.unique(np.linspace(0, len(q) - 1, 32, dtype=int))
            save(directory, "fit_rhs_samples", np.c_[qi, q[qi], target[qi]])
            # Stable probes include negative cells and off-node fractional coordinates.
            pi = np.unique(np.linspace(0, len(sf.cells) - 1, 16, dtype=int))
            probe = (sf.cells[pi] + np.array([.193, .417, .731])) * sf.H
            ids, values = sf.support(probe)
            save(directory, "probe_points", probe)
            save(directory, "probe_support", ids, True)
            save(directory, "basis_value", values)
            for name, der in [("dx", (1, 0, 0)), ("dy", (0, 1, 0)), ("dz", (0, 0, 1)),
                              ("dxx", (2, 0, 0)), ("dxy", (1, 1, 0)), ("dzz", (0, 0, 2))]:
                save(directory, "basis_" + name, sf.support(probe, der)[1])
            save(directory, "probe_fit", np.c_[sf.matrix(probe) @ alpha, sf.matrix(probe) @ p2fit])
            ti = np.arange(len(traces)) if args.full else np.unique(np.linspace(0, len(traces) - 1, 32, dtype=int))
            # Complete cover starts encode every tensor node, including box
            # boundary/zero-weight nodes that do not survive in the Rg matrix.
            cover_rows = []
            kcover = 3 if bvp == "neumann" else 4
            for side in (-1, 1):
                queries = x[:, None, :] + side*np.array([.5, .75, 1.5])[None, :, None]*grid.h*normal[:, None, :]
                starts = ks.cover_starts(queries, grid, kcover)
                cover_method = grid.cover3 if kcover == 3 else grid.cover4
                scalar_starts = np.array([[cover_method(query[:, axis])[0] for axis in range(3)] for query in queries])
                if not np.array_equal(starts, scalar_starts):
                    raise AssertionError("source vectorized and scalar cover indices differ")
                cover_rows.append(np.c_[np.arange(len(traces)), np.full(len(traces), side), starts])
            save(directory, "cover_starts", np.vstack(cover_rows), True)
            observer_rows = []
            for trace_id in ti:
                rv, rn = ops.Rg_value.getrow(trace_id), ops.Rg_normal.getrow(trace_id)
                ids = np.union1d(rv.indices, rn.indices)
                izyx = np.array(np.unravel_index(ids, (grid.ni,)*3)).T + 1
                full_ids = np.ravel_multi_index(izyx.T, (grid.N+1,)*3)
                observer_rows.append(np.c_[np.full(len(ids), trace_id), full_ids, rv[:, ids].toarray().ravel(), rn[:, ids].toarray().ravel()])
            save(directory, "observer_rows", np.vstack(observer_rows))
            save(directory, "trace_points", np.c_[ti, x[ti], normal[ti], [traces[i].w for i in ti], [traces[i].pid for i in ti]])
            save(directory, "trace_outputs", np.c_[cv[ti], cn[ti], raw_v[ti], raw_n[ti],
                (Tv @ alpha)[ti], (Tn @ alpha)[ti], requested_v[ti], requested_n[ti],
                (raw_v + .5 * (Tv @ alpha - requested_v))[ti], (raw_n + .5 * (Tn @ alpha - requested_n))[ti]])
            nonzero = np.flatnonzero(rhs)
            izyx = np.array(np.unravel_index(nonzero, (grid.ni,) * 3)).T + 1
            full_ids = np.ravel_multi_index(izyx.T, (grid.N + 1,) * 3)
            save(directory, "spread_rhs", np.c_[full_ids, -rhs[nonzero]])
            ui = np.arange(len(ops.unique)) if args.full else np.unique(np.linspace(0, len(ops.unique) - 1, 32, dtype=int))
            union_ids = ops.unique[ui]
            union_xyz = grid.lo + grid.h * np.array(np.unravel_index(union_ids, (grid.N + 1,) * 3)).T[:, ::-1]
            save(directory, "union_samples", np.c_[union_ids, union_xyz, ops.labels.ravel()[union_ids], (ops.E @ alpha)[ui]])
            inside_ids = np.flatnonzero(ops.labels.ravel())
            save(directory, "inside_full_grid_ids", inside_ids, True)
            if args.full:
                save(directory, "Z", policy["Z"])
                save(directory, "constraints", policy["C"])
                save(directory, "constraint_rhs", policy["d"])
                save(directory, "reduced_direction", reduced_direction)
            case = {"fixture": directory.relative_to(args.output).as_posix(), "geometry": "u", "N": args.N,
                "bvp": bvp, "transform": pose, "R": geom.T.R.tolist(), "translation": geom.T.t.tolist(),
                "geometry_dimensions": {"a": geom.a, "inner": geom.inner, "notch": geom.notch, "z0": geom.z0, "z1": geom.z1},
                "density_factor": 8., "ratio": 4., "width": 4., "ridge": 1e-12,
                "value_weight": 1., "normal_weight": 1., "pde_weight": 1., "restrict_mode": "staged",
                "exterior_targets": ["raw", "input_jump_half"], "seed": 51209,
                "raw_dofs": density.nfull, "reduced_dofs": policy["Z"].shape[1],
                "field_dofs": sf.ncoef, "surface_samples": len(sf.sx), "volume_samples": len(sf.vx),
                "trace_count": len(traces), "constraint_direction_linf": float(np.max(abs(policy["C"] @ raw))),
                "optimality_linf": float(np.max(abs(optimality))), "actual_fitted_P2": p2check,
                "extension": ext.info, "transfer": ops.info}
            write_json(directory / "case.json", case)
            cases.append(case)
            print(json.dumps({"fixture": case["fixture"], "optimality": case["optimality_linf"], "P2": p2check}), flush=True)
    write_json(args.output / "cases.json", {"format_version": 1, "cases": cases, "files": manifest})
    if args.run_cases:
        for pose in args.transforms:
            for bvp in args.bvps:
                for variant in ("baseline", "shared_direct", "shared_jump"):
                    case_args = ks.parser().parse_args(["--geometry", "u", "--N", "32", "--transform", pose,
                        "--bvp", bvp, "--method", "baseline" if variant == "baseline" else "shared",
                        "--jump-identity", ".5" if variant == "shared_jump" else "0", "--output", str(args.output / "rerun")])
                    ks.run_case(case_args)
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.show_config()
    used.update({"src/" + name: digest for name, digest in SOURCE_SHA256.items()})
    files = {p.relative_to(args.output).as_posix(): sha256(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output / "provenance.json", {"format_version": 1, "package_sha256": PACKAGE_SHA256,
        "exporter_version": VERSION, "exporter_sha256": sha256(__file__), "source_files_sha256": used,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "pandas": pandas.__version__, "blas": blas.getvalue()},
        "conventions": {"matrices": "rows cols header; row-major whitespace; %.17e; indices zero based",
            "coordinates": "physical xyz; body = R^T(world-translation); normals are outward",
            "field_order": "lexicographic integer xyz lattice coordinates; floor for negative cells",
            "grid_id": "x-fast: (z*(N+1)+y)*(N+1)+x; complete box, including boundary",
            "labels": "source analytic U labels; native C++ labels must be compared independently",
            "spread_rhs": "actual rhs_for_delta = negative Python -Delta RHS; omitted entries zero",
            "cover_starts": "every trace and each side (-1 inside, +1 outside): trace ID, side, Cartesian start xyz; all k^3 nodes including boundary/zero weights, x fastest",
            "observer_rows": "selected trace ID, full Cartesian grid ID, Rg_value, Rg_normal; merged interior-only sparse support, omitted coefficients zero; not a complete cover inventory",
            "density": "patch pid increasing, within patch u fastest; full raw vectors saved",
            "basis_alignment": "compare subspaces ZZ^T after raw/physical alignment, never unmatched reduced coordinates",
            "fit": "actual fixed LU fit with cubic lift, ridge1e-12, exactly two refinements; no exact unknown samples"},
        "fixture_files_sha256": files})


if __name__ == "__main__":
    main()
