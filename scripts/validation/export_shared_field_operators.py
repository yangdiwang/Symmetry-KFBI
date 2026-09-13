#!/usr/bin/env python3
"""Capture the reviewed source's actual projected homogeneous N32 operators.

Source run_case performs its unmodified geometry/space/fit/transfer/projection
setup. At the scipy.gmres boundary this exporter materializes the supplied
LinearOperator, saves its actual Z, and stops before iterative solution. Thus no
second KFBI forward implementation or historical operator is substituted.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import export_shared_field_reference as reference


class OperatorCaptured(Exception):
    """Intentional stop at the outer solve boundary, after saving the operator."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transforms", nargs="+", choices=["rotate", "rotate_translate"], default=["rotate", "rotate_translate"])
    parser.add_argument("--bvps", nargs="+", choices=["dirichlet", "neumann"], default=["dirichlet", "neumann"])
    parser.add_argument("--variants", nargs="+", choices=["baseline", "shared_direct", "shared_jump"], default=["baseline", "shared_direct", "shared_jump"])
    args = parser.parse_args()
    reference.verify_source(args.source_dir.resolve(), args.source_archive)
    if args.output.exists():
        raise ValueError("output must be a new directory")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(KFBI_CENTER_POLICY="trace_first", KFBI_EVENT_POLICY="current", KFBI_EDGE_STAR_WEIGHT="0.1",
                      KFBI_VERTEX_STAR_WEIGHT="0.0", KFBI_VERTEX_STAR_POWER="0.0", KFBI_TRACE_FIRST_EVENT_RADIUS="1.75",
                      KFBI_TRACE_FIRST_TARGET_RADIUS="3.25")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_dir.resolve() / "src"))
    import numpy as np
    import scipy
    import pandas
    import kfbi_shared as ks

    args.output.mkdir(parents=True)
    original_projector, original_extension, original_gmres = ks.Projector, ks.LinearExtension, ks.gmres
    state = {}
    records = []

    def save(path, values):
        array = np.asarray(values)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"invalid exported matrix: {path}")
        np.savetxt(path, array, fmt="%.17e", header=f"{array.shape[0]} {array.shape[1]}", comments="")

    class CapturingProjector(original_projector):
        def __init__(self, mod, space, trace_points, bvp, Z):
            super().__init__(mod, space, trace_points, bvp, Z)
            state.update(projector=self, space=space, trace_points=trace_points, mod=mod)

    class CapturingExtension(original_extension):
        def __init__(self, *params, **kwargs):
            super().__init__(*params, **kwargs)
            state["extension"] = self

    def capture_gmres(operator, rhs, **kwargs):
        start = time.perf_counter()
        n = operator.shape[1]
        dense = np.empty((n, n))
        direction = np.zeros(n)
        for column in range(n):
            direction.fill(0.)
            direction[column] = 1.
            dense[:, column] = operator @ direction
        singular_values = np.linalg.svd(dense, compute_uv=False)
        directory = state["directory"]
        projector = state["projector"]
        space = state["space"]
        traces = state["trace_points"]
        rng = np.random.default_rng(3901)
        probe = rng.normal(size=n)
        # Independent additional action ensures the saved columns act exactly
        # like the supplied matrix-free operator in the same coordinates.
        probe_action = operator @ probe
        error = float(np.max(abs(dense @ probe - probe_action)) / max(1., np.max(abs(probe_action))))
        if error > 1e-8:
            raise AssertionError(f"matrix-free/columns mismatch: {error}")
        save(directory / "reduced_operator.txt", dense)
        save(directory / "density_nullspace.txt", projector.Z)
        save(directory / "operator_singular_values.txt", singular_values)
        save(directory / "projected_rhs.txt", rhs)
        save(directory / "reduced_probe.txt", probe)
        save(directory / "raw_probe.txt", projector.Z @ probe)
        save(directory / "probe_action.txt", probe_action)
        save(directory / "trace_points.txt", np.array([np.r_[q.x, q.n, q.w, q.pid] for q in traces]))
        patches = [{"pid": p.pid, "offset": p.off, "ncu": p.ncu, "ncv": p.ncv,
                    "body_origin": p.o0.tolist(), "body_eu": p.eu0.tolist(), "body_ev": p.ev0.tolist(),
                    "Lu": p.Lu, "Lv": p.Lv, "u_knots": p.bu.knots.tolist(), "v_knots": p.bv.knots.tolist()}
                   for p in space.patches]
        reference.write_json(directory / "raw_layout.json", {"order": "patch-major; offset+j*ncu+i, u fastest", "patches": patches})
        config = state["config"] | {"R": space.geometry.T.R.tolist(), "translation": space.geometry.T.t.tolist(),
            "geometry_dimensions": {"a": space.geometry.a, "inner": space.geometry.inner, "notch": space.geometry.notch,
                                     "z0": space.geometry.z0, "z1": space.geometry.z1},
            "raw_dofs": space.nfull, "reduced_dofs": n, "trace_count": len(traces),
            "homogeneous_operator": "Source run_case supplied scipy LinearOperator at the gmres boundary; no prescribed data in its matvec",
            "coordinate_metric": "source Euclidean-orthonormal raw nullspace Z; exact source Projector including edge terms",
            "matrix_free_probe_relative_inf": error, "operator_sigma_min": float(singular_values[-1]),
            "operator_sigma_max": float(singular_values[0]), "operator_condition": float(singular_values[0]/singular_values[-1]),
            "assembly_seconds": time.perf_counter()-start, "iterative_solve_executed": False,
            "source_gmres_configuration": {k: v for k, v in kwargs.items() if k in ("rtol", "atol", "restart", "maxiter")},
            "density_layout_sha256": reference.sha256(directory / "raw_layout.json"),
            "trace_layout_sha256": reference.sha256(directory / "trace_points.txt")}
        if "extension" in state:
            config["extension"] = state["extension"].info
            config["actual_field_fit_calls"] = state["extension"].calls
        reference.write_json(directory / "case.json", config)
        records.append(config)
        raise OperatorCaptured()

    ks.Projector, ks.LinearExtension, ks.gmres = CapturingProjector, CapturingExtension, capture_gmres
    try:
        for pose in args.transforms:
            for bvp in args.bvps:
                for variant in args.variants:
                    name = f"{variant}_{bvp}_{pose}_N32"
                    directory = args.output / name
                    directory.mkdir()
                    config = {"case_id": name, "N": 32, "geometry": "u", "transform": pose, "bvp": bvp, "variant": variant,
                        "correction_backend": "direct_cauchy" if variant == "baseline" else "shared_field",
                        "exterior_target": "input_jump_half" if variant == "shared_jump" else "raw", "density_factor": 8.,
                        "field_ratio": 4., "field_width": 4., "field_ridge": 1e-12, "value_weight": 1., "normal_weight": 1.,
                        "pde_weight": 1., "restrict_mode": "staged", "seed": 3901}
                    state.clear()
                    state.update(directory=directory, config=config)
                    driver_args = ks.parser().parse_args(["--geometry", "u", "--N", "32", "--transform", pose,
                        "--bvp", bvp, "--method", "baseline" if variant == "baseline" else "shared",
                        "--jump-identity", ".5" if variant == "shared_jump" else "0", "--output", str(directory / "source_setup")])
                    with (directory / "source_setup.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        try:
                            ks.run_case(driver_args)
                        except OperatorCaptured:
                            pass
                        else:
                            raise RuntimeError("source run did not reach the intercepted GMRES boundary")
                    print(json.dumps({k: records[-1][k] for k in ("case_id", "reduced_dofs", "operator_condition", "matrix_free_probe_relative_inf")}), flush=True)
    finally:
        ks.Projector, ks.LinearExtension, ks.gmres = original_projector, original_extension, original_gmres
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.show_config()
    reference.write_json(args.output / "cases.json", {"cases": records, "expected_count": len(args.transforms)*len(args.bvps)*len(args.variants)})
    reference.write_json(args.output / "provenance.json", {"format_version": 1, "package_sha256": reference.PACKAGE_SHA256,
        "source_files_sha256": {"src/"+k: v for k, v in reference.SOURCE_SHA256.items()},
        "exporter_sha256": reference.sha256(__file__), "verification_helper_sha256": reference.sha256(reference.__file__),
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
                    "platform": platform.platform(), "blas": blas.getvalue()},
        "files_sha256": {p.relative_to(args.output).as_posix(): reference.sha256(p) for p in sorted(args.output.rglob("*")) if p.is_file()},
        "method": "Unmodified reviewed run_case builds exact source LinearOperator. Exporter intercepts gmres, applies every canonical reduced direction, and stops without iterative solve.",
        "source_directory_is_runtime_dependency": False})


if __name__ == "__main__":
    main()
