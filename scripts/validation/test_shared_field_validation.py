"""Small negative tests for validation gates, not copies of numerical algorithms."""
import json
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np

import compare_shared_field_reference as compare
import summarize_shared_field_3d as summary


class ValidationGates(unittest.TestCase):
    def output_directory(self):
        # Python 3.14's restrictive Windows TemporaryDirectory ACL can exclude
        # the desktop sandbox identity. Keep tiny diagnostic artifacts in the
        # explicitly writable, ignored output directory instead.
        root = Path(__file__).resolve().parents[2] / "output/shared_field_3d/validation_tests" / uuid.uuid4().hex
        root.mkdir(parents=True)
        return root

    def test_localized_vector_error_cannot_average_away(self):
        ref = np.ones((10000, 1))
        actual = ref.copy()
        actual[7] += 1e-7
        self.assertFalse(compare.difference(actual, ref, 1e-8)["passed"])

    def test_subspace_rotation_aligns_operator(self):
        root = self.output_directory()
        q = np.array([[0., -1.], [1., 0.]])
        operator = np.array([[2., .3], [-.1, 1.]])
        for name, m in [("ar", operator), ("ac", q.T@operator@q), ("zr", np.eye(2)), ("zc", q)]:
            np.savetxt(root/name, m, header=f"{m.shape[0]} {m.shape[1]}", comments="")
        report = compare.operators(root/"ar", root/"ac", root/"zr", root/"zc", 1e-8, False)
        self.assertTrue(report["passed"])

    def test_exactly_singular_operator_writes_strict_json_report(self):
        root = self.output_directory()
        for name, matrix in [("operator", np.diag([1., 0.])), ("zero", np.zeros((2, 2))), ("Z", np.eye(2))]:
            np.savetxt(root/name, matrix, header="2 2", comments="")
        for name in ("operator", "zero"):
            for direct in (False, True):
                with self.subTest(matrix=name, direct=direct):
                    destination = root/f"{name}_{direct}.json"
                    argv = ["compare", "--reference", str(root/name), "--candidate", str(root/name),
                            "--reference-z", str(root/"Z"), "--candidate-z", str(root/"Z"), "--output", str(destination)]
                    if direct:
                        argv.append("--ill-conditioned-direct")
                    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                        status = compare.main()
                    report = json.loads(destination.read_text(encoding="utf-8"))
                    json.dumps(report, allow_nan=False)
                    self.assertEqual(status, 0 if direct else 1)
                    self.assertEqual(report["passed"], direct)
                    self.assertTrue(report["reference_exactly_singular"])
                    self.assertTrue(report["candidate_exactly_singular"])
                    self.assertIsNone(report["reference_condition"])
                    self.assertIsNone(report["candidate_condition"])
                    self.assertIsNone(report["condition_relative_difference"])
                    self.assertFalse(report["candidate_condition_under_100_screen"])
                    self.assertEqual(report["matrix_perturbation_2norm"], 0.)

    def evaluate(self, result=None, include_execution=True, variant="shared_jump"):
        root = self.output_directory()
        config = dict(levels=[32], transforms=["rotate"], bvps=["neumann"], variants=[variant])
        execution = [dict(N=32, transform="rotate", bvp="neumann", variant=variant, result_path="result.json", exit_code=0)] if include_execution else []
        for name, value in [("study_config.json", config), ("execution_status.json", execution), ("reference.json", {"cases": []})]:
            (root/name).write_text(json.dumps(value), encoding="utf-8")
        if result is not None:
            (root/"result.json").write_text(json.dumps(result), encoding="utf-8")
            for name in ["density_coefficients", "reduced_coordinates", "raw_value_trace", "raw_normal_trace", "equation_value_trace", "equation_normal_trace", "field_coefficients", "requested_value_jump", "requested_normal_jump", "fitted_value_jump", "fitted_normal_jump"]:
                (root/(name+".txt")).write_text("1 1\n0\n", encoding="utf-8")
        return summary.summarize(root, root/"reference.json")

    @staticmethod
    def valid():
        return dict(projected_relative_residual=1e-11, gmres_converged=True, gmres_iterations=20,
                    interior_linf=1e-5, correction_backend="shared_field", exterior_target="input_jump_half", shared_field_restrict="staged",
                    raw_value_linf=1e-3, raw_normal_linf=1e-3, equation_value_linf=1e-5, equation_normal_linf=1e-5)

    def run_summary_cli(self, level, requested_variants, actual_results):
        root = self.output_directory()
        config = dict(levels=[level], transforms=["rotate"], bvps=["neumann"], variants=requested_variants)
        execution = []
        for variant, result in actual_results.items():
            case = root/variant
            case.mkdir()
            (case/"result.json").write_text(json.dumps(result), encoding="utf-8")
            for name in ["density_coefficients", "reduced_coordinates", "raw_value_trace", "raw_normal_trace", "equation_value_trace", "equation_normal_trace", "field_coefficients", "requested_value_jump", "requested_normal_jump", "fitted_value_jump", "fitted_normal_jump"]:
                (case/(name+".txt")).write_text("1 1\n0\n", encoding="utf-8")
            execution.append(dict(N=level, transform="rotate", bvp="neumann", variant=variant,
                                  result_path=f"{variant}/result.json", exit_code=0))
        for name, value in [("study_config.json", config), ("execution_status.json", execution), ("reference.json", {"cases": []})]:
            (root/name).write_text(json.dumps(value), encoding="utf-8")
        argv = ["summarize", str(root), "--reference", str(root/"reference.json")]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            status = summary.main()
        return status, json.loads((root/"summary.json").read_text(encoding="utf-8"))

    def test_successful_control_only_scope_has_candidate_not_applicable(self):
        baseline = self.valid() | {"correction_backend": "direct_cauchy", "exterior_target": "raw"}
        direct = self.valid() | {"exterior_target": "raw"}
        for level in (64, 128):
            with self.subTest(level=level):
                status, report = self.run_summary_cli(level, ["baseline", "shared_direct"], {"baseline": baseline, "shared_direct": direct})
                self.assertEqual(status, 0)
                self.assertIsNone(report["candidate_numerical_acceptance"])
                self.assertTrue(report["selected_numerical_acceptance"])
                self.assertTrue(report["configuration_matrix_complete"])

    def test_failed_control_only_scope_and_known_coarse_failure_exit_nonzero(self):
        baseline = self.valid() | {"correction_backend": "direct_cauchy", "exterior_target": "raw"}
        for level, direct in [(64, self.valid() | {"exterior_target": "raw", "projected_relative_residual": 1e-3}),
                              (32, self.valid() | {"exterior_target": "raw"})]:
            with self.subTest(level=level):
                status, report = self.run_summary_cli(level, ["baseline", "shared_direct"], {"baseline": baseline, "shared_direct": direct})
                self.assertEqual(status, 1)
                self.assertIsNone(report["candidate_numerical_acceptance"])
                self.assertFalse(report["selected_numerical_acceptance"])
                if level == 32:
                    direct_row = next(r for r in report["cases"] if r["variant"] == "shared_direct")
                    self.assertEqual(direct_row["status"], "KNOWN_DIRECT_RESEARCH_FAILURE")
                    self.assertFalse(direct_row["numerical_acceptance"])

    def test_missing_requested_candidate_is_failure_not_not_applicable(self):
        baseline = self.valid() | {"correction_backend": "direct_cauchy", "exterior_target": "raw"}
        status, report = self.run_summary_cli(64, ["baseline", "shared_jump"], {"baseline": baseline})
        self.assertEqual(status, 1)
        self.assertIs(report["candidate_numerical_acceptance"], False)
        self.assertFalse(report["configuration_matrix_complete"])

    def test_candidate_scope_keeps_failed_direct_rows_without_changing_exit_semantics(self):
        status, report = self.run_summary_cli(32, ["shared_direct", "shared_jump"],
            {"shared_direct": self.valid() | {"exterior_target": "raw"}, "shared_jump": self.valid()})
        self.assertEqual(status, 0)
        self.assertTrue(report["candidate_numerical_acceptance"])
        self.assertFalse(report["selected_numerical_acceptance"])
        direct_row = next(r for r in report["cases"] if r["variant"] == "shared_direct")
        self.assertFalse(direct_row["numerical_acceptance"])

    def test_true_residual_overrides_recursive_flag(self):
        result = self.valid()
        result["projected_relative_residual"] = 1e-3
        self.assertFalse(self.evaluate(result)["candidate_numerical_acceptance"])

    def test_nan_is_preserved_as_failure_and_report_serializes(self):
        result = self.valid()
        result["projected_relative_residual"] = float("nan")
        report = self.evaluate(result)
        self.assertFalse(report["candidate_numerical_acceptance"])
        json.dumps(report, allow_nan=False)

    def test_missing_config_does_not_pass(self):
        report = self.evaluate(include_execution=False)
        self.assertFalse(report["configuration_matrix_complete"])
        self.assertFalse(report["candidate_numerical_acceptance"])

    def test_saved_coefficients_are_not_verified_replay(self):
        report = self.evaluate(self.valid())
        self.assertTrue(report["candidate_numerical_acceptance"])
        self.assertFalse(report["all_saved_replays_verified"])

    def test_direct_coarse_equation_convergence_not_physical_acceptance(self):
        result = self.valid()
        result["exterior_target"] = "raw"
        report = self.evaluate(result, variant="shared_direct")
        self.assertFalse(report["cases"][0]["numerical_acceptance"])
        self.assertEqual(report["cases"][0]["status"], "KNOWN_DIRECT_RESEARCH_FAILURE")


if __name__ == "__main__":
    unittest.main()
