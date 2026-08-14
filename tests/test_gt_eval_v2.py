from __future__ import annotations

import math
import unittest
from typing import Any, Optional

import torch
from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors


def _make_mock_solver_results(
    *,
    reflection: Optional[torch.Tensor] = None,
    transmission: Optional[torch.Tensor] = None,
    reflection_diffraction: Optional[torch.Tensor] = None,
    transmission_diffraction: Optional[torch.Tensor] = None,
) -> Any:
    default_reflection = torch.tensor([0.2], dtype=torch.float32)
    default_transmission = torch.tensor([0.8], dtype=torch.float32)
    default_reflection_diffraction = torch.tensor([[[0.1, 0.05], [0.03, 0.02]]], dtype=torch.float32)
    default_transmission_diffraction = torch.tensor([[[0.6, 0.1], [0.05, 0.05]]], dtype=torch.float32)

    vec = torch.zeros(1, dtype=torch.float32)
    mat = torch.zeros(1, 1, 1, dtype=torch.float32)
    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
    wave = WaveVectors(kx=vec.clone(), ky=vec.clone(), kinc=vec.clone(), kzref=vec.clone(), kztrn=vec.clone())

    return SolverResults(
        reflection=reflection if reflection is not None else default_reflection,
        transmission=transmission if transmission is not None else default_transmission,
        reflection_diffraction=(
            reflection_diffraction
            if reflection_diffraction is not None
            else default_reflection_diffraction
        ),
        transmission_diffraction=(
            transmission_diffraction
            if transmission_diffraction is not None
            else default_transmission_diffraction
        ),
        reflection_field=field,
        transmission_field=field,
        structure_matrix=smat,
        wave_vectors=wave,
    )


# ---------------------------------------------------------------------------
# Tests for validate_gt_eval (new schema)
# ---------------------------------------------------------------------------
class TestValidateGtEval(unittest.TestCase):
    def _validate(self, value):
        from evo_metaoptics.meta_design.gt_eval import validate_gt_eval
        return validate_gt_eval(value)

    def test_valid_minimal(self):
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        result = self._validate(payload)
        self.assertEqual(result["wavelength_um"], [1.55])
        self.assertEqual(len(result["criteria"]), 1)
        self.assertEqual(result["criteria"][0]["expr"], "r.transmission[0].item()")
        self.assertEqual(result["criteria"][0]["operation"], ">=")
        self.assertAlmostEqual(result["criteria"][0]["target"], 0.8)

    def test_valid_multi_criteria(self):
        payload = {
            "wavelength_um": [1.55, 1.31],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8},
                {"expr": "r.reflection[0].item()", "operation": "<=", "target": 0.15},
            ],
        }
        result = self._validate(payload)
        self.assertEqual(len(result["criteria"]), 2)

    def test_rejects_loss_expr(self):
        payload = {
            "wavelength_um": [1.55],
            "loss_expr": "1 - r.transmission[0].item()",
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_missing_criteria(self):
        payload = {
            "wavelength_um": [1.55],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_empty_criteria_list(self):
        payload = {
            "wavelength_um": [1.55],
            "criteria": [],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_criteria_missing_expr(self):
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_criteria_invalid_operation(self):
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": "==", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_criteria_missing_target(self):
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">="}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_missing_wavelength_um(self):
        payload = {
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_empty_wavelength_um(self):
        payload = {
            "wavelength_um": [],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_rejects_old_loss_function_key(self):
        payload = {
            "wavelength_um": [1.55],
            "loss_function": "1 - T_total(wl=1.55)",
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_rejects_old_metrics_key_in_criteria(self):
        """Old schema used 'metrics'; new schema uses 'expr'."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"metrics": "T_total(wl=1.55)", "operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_not_a_mapping(self):
        with self.assertRaises(ValueError):
            self._validate("not a mapping")

    def test_rejects_budget_key(self):
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
            "budget": 10,
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_valid_all_operations(self):
        """All five operations: >, <, >=, <=, close_to."""
        for op in [">", "<", ">=", "<="]:
            payload = {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": op, "target": 0.5}
                ],
            }
            result = self._validate(payload)
            self.assertEqual(result["criteria"][0]["operation"], op)

        # close_to requires tolerance
        payload_ct = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": "close_to", "target": 0.5, "tolerance": 0.1}
            ],
        }
        result_ct = self._validate(payload_ct)
        self.assertEqual(result_ct["criteria"][0]["operation"], "close_to")
        self.assertAlmostEqual(result_ct["criteria"][0]["tolerance"], 0.1)

    def test_valid_close_to_operation(self):
        """close_to with tolerance validates successfully."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "expr": "r.transmission[0].item()",
                    "operation": "close_to",
                    "target": 170.0,
                    "tolerance": 8.5,
                }
            ],
        }
        result = self._validate(payload)
        self.assertEqual(result["criteria"][0]["operation"], "close_to")
        self.assertAlmostEqual(result["criteria"][0]["target"], 170.0)
        self.assertAlmostEqual(result["criteria"][0]["tolerance"], 8.5)

    def test_close_to_missing_tolerance_rejected(self):
        """close_to without tolerance raises ValueError."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": "close_to", "target": 170.0}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_close_to_negative_tolerance_rejected(self):
        """Negative tolerance raises ValueError."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": "close_to", "target": 170.0, "tolerance": -1.0}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_close_to_zero_tolerance_rejected(self):
        """Zero tolerance raises ValueError."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": "close_to", "target": 170.0, "tolerance": 0.0}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_tolerance_on_non_close_to_ignored(self):
        """tolerance key on >= operation is silently ignored."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8, "tolerance": 5.0}
            ],
        }
        result = self._validate(payload)
        self.assertEqual(result["criteria"][0]["operation"], ">=")
        # tolerance should NOT appear in normalized output for non-close_to
        self.assertNotIn("tolerance", result["criteria"][0])

# ---------------------------------------------------------------------------
# Tests for evaluate_gt_eval (new eval()-based evaluation)
# ---------------------------------------------------------------------------
class TestEvaluateGtEval(unittest.TestCase):
    def _evaluate(self, gt_eval, solver_results, *, compile_ok=True, solver_ok=True):
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval
        return evaluate_gt_eval(gt_eval, solver_results, compile_ok=compile_ok, solver_ok=solver_ok)

    def test_success_case(self):
        results = _make_mock_solver_results(transmission=torch.tensor([0.9]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, results)

        self.assertTrue(out["compile_ok"])
        self.assertTrue(out["solver_ok"])
        self.assertTrue(out["success_exec"])
        self.assertEqual(len(out["criteria"]), 1)
        self.assertTrue(out["criteria"][0]["passed"])
        self.assertAlmostEqual(out["criteria"][0]["value"], 0.9, places=5)
        self.assertTrue(out["meets_gt_criteria"])
        self.assertTrue(out["success_goal"])

    def test_failure_case(self):
        """Transmission = 0.5, target >= 0.8 → fails."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.5]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, results)

        self.assertTrue(out["success_exec"])
        self.assertFalse(out["criteria"][0]["passed"])
        self.assertAlmostEqual(out["criteria"][0]["value"], 0.5, places=5)
        self.assertFalse(out["meets_gt_criteria"])
        self.assertFalse(out["success_goal"])

    def test_compile_not_ok(self):
        """compile_ok=False → early return with success_exec=False."""
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, None, compile_ok=False)

        self.assertFalse(out["compile_ok"])
        self.assertTrue(out["solver_ok"])
        self.assertFalse(out["success_exec"])
        self.assertEqual(out["criteria"], [])
        self.assertFalse(out["meets_gt_criteria"])
        self.assertFalse(out["success_goal"])

    def test_solver_not_ok(self):
        """solver_ok=False → early return."""
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, None, solver_ok=False)

        self.assertFalse(out["success_exec"])
        self.assertFalse(out["solver_ok"])
        self.assertFalse(out["success_goal"])

    def test_multi_criteria_partial_pass(self):
        """Two criteria: one passes, one fails → meets_gt_criteria=False."""
        results = _make_mock_solver_results(
            transmission=torch.tensor([0.85]),
            reflection=torch.tensor([0.15]),
        )
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8},
                {"expr": "r.reflection[0].item()", "operation": "<=", "target": 0.1},
            ],
        }
        out = self._evaluate(gt, results)

        self.assertTrue(out["success_exec"])
        self.assertTrue(out["criteria"][0]["passed"])  # 0.85 >= 0.8
        self.assertFalse(out["criteria"][1]["passed"])  # 0.15 <= 0.1 → False
        self.assertFalse(out["meets_gt_criteria"])
        self.assertFalse(out["success_goal"])

    def test_multi_criteria_all_pass(self):
        """Both criteria pass → success_goal=True."""
        results = _make_mock_solver_results(
            transmission=torch.tensor([0.9]),
            reflection=torch.tensor([0.08]),
        )
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8},
                {"expr": "r.reflection[0].item()", "operation": "<=", "target": 0.1},
            ],
        }
        out = self._evaluate(gt, results)

        self.assertTrue(out["success_exec"])
        self.assertTrue(out["criteria"][0]["passed"])
        self.assertTrue(out["criteria"][1]["passed"])
        self.assertTrue(out["meets_gt_criteria"])
        self.assertTrue(out["success_goal"])

    def test_criterion_margin_and_violation(self):
        """Criteria rows should include margin and violation values."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.7]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        self.assertIn("margin", row)
        self.assertIn("violation", row)
        self.assertAlmostEqual(row["margin"], -0.1, places=5)  # 0.7 - 0.8
        self.assertAlmostEqual(row["violation"], 0.1, places=5)

    def test_criterion_margin_pass(self):
        """When criterion passes, violation = 0."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.9]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        self.assertAlmostEqual(row["margin"], 0.1, places=5)  # 0.9 - 0.8
        self.assertAlmostEqual(row["violation"], 0.0, places=5)

    def test_less_than_operation(self):
        """Test < operation."""
        results = _make_mock_solver_results(reflection=torch.tensor([0.05]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.reflection[0].item()", "operation": "<", "target": 0.1}
            ],
        }
        out = self._evaluate(gt, results)

        self.assertTrue(out["criteria"][0]["passed"])
        # For <, margin = target - value = 0.1 - 0.05 = 0.05
        self.assertAlmostEqual(out["criteria"][0]["margin"], 0.05, places=5)

    def test_greater_than_operation(self):
        """Test > operation."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.9]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, results)

        self.assertTrue(out["criteria"][0]["passed"])
        self.assertAlmostEqual(out["criteria"][0]["margin"], 0.1, places=5)

    def test_less_equal_operation(self):
        """Test <= operation with value clearly below target."""
        results = _make_mock_solver_results(reflection=torch.tensor([0.08]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.reflection[0].item()", "operation": "<=", "target": 0.1}
            ],
        }
        out = self._evaluate(gt, results)

        self.assertTrue(out["criteria"][0]["passed"])

    def test_return_schema_keys(self):
        """Return dict must have exactly these keys."""
        results = _make_mock_solver_results()
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, results)

        expected_keys = {
            "compile_ok", "solver_ok", "success_exec",
            "criteria", "meets_gt_criteria", "success_goal",
        }
        self.assertEqual(set(out.keys()), expected_keys)

    def test_eval_context_supports_import_dependent_expression(self):
        results = _make_mock_solver_results()
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(__import__('math').exp(0.0))", "operation": "==", "target": 1.0}
            ],
        }

        with self.assertRaises(ValueError):
            self._evaluate(gt, results)

        gt["criteria"][0]["operation"] = ">="
        out = self._evaluate(gt, results)
        self.assertTrue(out["criteria"][0]["passed"])
        self.assertAlmostEqual(out["criteria"][0]["value"], 1.0, places=6)

    def test_close_to_pass_exact_match(self):
        """value == target → distance=0, margin=tolerance, passed=True."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.5]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(r.transmission[0])", "operation": "close_to", "target": 0.5, "tolerance": 0.1}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["value"], 0.5, places=5)
        self.assertAlmostEqual(row["margin"], 0.1, places=5)  # tolerance - 0 = 0.1
        self.assertAlmostEqual(row["violation"], 0.0, places=5)
        self.assertAlmostEqual(row["tolerance"], 0.1, places=5)

    def test_close_to_pass_within_tolerance(self):
        """value near target within tolerance → passed=True."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.55]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(r.transmission[0])", "operation": "close_to", "target": 0.5, "tolerance": 0.1}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["margin"], 0.05, places=5)  # 0.1 - 0.05 = 0.05

    def test_close_to_fail_outside_tolerance(self):
        """value far from target → passed=False, margin<0."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.8]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(r.transmission[0])", "operation": "close_to", "target": 0.5, "tolerance": 0.1}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        self.assertFalse(row["passed"])
        self.assertAlmostEqual(row["margin"], -0.2, places=5)  # 0.1 - 0.3 = -0.2
        self.assertAlmostEqual(row["violation"], 0.2, places=5)

    def test_close_to_wrapping_near_zero(self):
        """target=5, value=355 → distance=10 (not 350), wrapping works."""
        # Use a transmission value of 355/1000 = 0.355 and eval expression that scales
        # We need a custom expression for this — use inline math
        results = _make_mock_solver_results(transmission=torch.tensor([355.0]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(r.transmission[0])", "operation": "close_to", "target": 5.0, "tolerance": 15.0}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        # distance = min(350 % 360, 360 - 350 % 360) = min(350, 10) = 10
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["margin"], 5.0, places=5)  # 15 - 10 = 5

    def test_close_to_wrapping_near_360(self):
        """target=355, value=5 → distance=10, wrapping works."""
        results = _make_mock_solver_results(transmission=torch.tensor([5.0]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(r.transmission[0])", "operation": "close_to", "target": 355.0, "tolerance": 15.0}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        # distance = min(350 % 360, 360 - 350 % 360) = min(350, 10) = 10
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["margin"], 5.0, places=5)  # 15 - 10 = 5

    def test_close_to_margin_semantics(self):
        """Verify margin = tolerance - distance, positive = passing."""
        results = _make_mock_solver_results(transmission=torch.tensor([173.0]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(r.transmission[0])", "operation": "close_to", "target": 170.0, "tolerance": 8.5}
            ],
        }
        out = self._evaluate(gt, results)

        row = out["criteria"][0]
        self.assertTrue(row["passed"])
        # distance = 3.0, margin = 8.5 - 3.0 = 5.5
        self.assertAlmostEqual(row["margin"], 5.5, places=5)

    def test_rejects_duck_typed_solver_results(self):
        class FakeResults:
            def __init__(self):
                self.transmission = torch.tensor([0.9])
                self.reflection = torch.tensor([0.1])

        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._evaluate(gt, FakeResults())


# ---------------------------------------------------------------------------
# Tests for summarize_normalized_margins (kept from old API)
# ---------------------------------------------------------------------------
class TestSummarizeNormalizedMargins(unittest.TestCase):
    """Ensure summarize_normalized_margins still works with new criteria row format."""

    def _summarize(self, criteria_rows):
        from evo_metaoptics.meta_design.gt_eval import summarize_normalized_margins
        return summarize_normalized_margins(criteria_rows)

    def test_empty(self):
        result = self._summarize([])
        self.assertIsNone(result["best_margin"])
        violation_norm = result["criteria_violation_norm"]
        self.assertIsNotNone(violation_norm)
        assert violation_norm is not None
        self.assertAlmostEqual(violation_norm, 0.0)

    def test_all_passing(self):
        rows = [
            {"margin": 0.1, "target": 0.8, "operation": ">="},
            {"margin": 0.05, "target": 0.1, "operation": "<="},
        ]
        result = self._summarize(rows)
        self.assertIsNotNone(result["best_margin"])
        violation_norm = result["criteria_violation_norm"]
        self.assertIsNotNone(violation_norm)
        assert violation_norm is not None
        self.assertAlmostEqual(violation_norm, 0.0)

    def test_violation(self):
        rows = [
            {"margin": -0.1, "target": 0.8, "operation": ">="},
        ]
        result = self._summarize(rows)
        best_margin = result["best_margin"]
        violation_norm = result["criteria_violation_norm"]
        self.assertIsNotNone(best_margin)
        self.assertIsNotNone(violation_norm)
        assert best_margin is not None
        assert violation_norm is not None
        self.assertLess(best_margin, 0.0)
        self.assertGreater(violation_norm, 0.0)

    def test_close_to_margin_scale_uses_tolerance(self):
        """For close_to, normalization scale = tolerance (not abs(target))."""
        rows = [
            {"margin": 5.0, "target": 170.0, "operation": "close_to", "tolerance": 8.5},
        ]
        result = self._summarize(rows)
        best_margin = result["best_margin"]
        self.assertIsNotNone(best_margin)
        assert best_margin is not None
        # normalized = margin / tolerance = 5.0 / 8.5 ≈ 0.5882
        self.assertAlmostEqual(best_margin, 5.0 / 8.5, places=4)

    def test_close_to_normalized_margin_violation(self):
        """For close_to with negative margin, violation uses tolerance scale."""
        rows = [
            {"margin": -2.0, "target": 170.0, "operation": "close_to", "tolerance": 8.5},
        ]
        result = self._summarize(rows)
        best_margin = result["best_margin"]
        violation_norm = result["criteria_violation_norm"]
        self.assertIsNotNone(best_margin)
        self.assertIsNotNone(violation_norm)
        assert best_margin is not None
        assert violation_norm is not None
        # normalized margin = -2.0 / 8.5
        self.assertAlmostEqual(best_margin, -2.0 / 8.5, places=4)
        self.assertGreater(violation_norm, 0.0)

# ---------------------------------------------------------------------------
# Tests for v8 metric criteria (schema v8 dispatch)
# ---------------------------------------------------------------------------


def _make_mock_solver_results_with_fields(
    *,
    transmission: torch.Tensor | None = None,
    reflection: torch.Tensor | None = None,
    harmonics: tuple[int, int] = (3, 3),
    tx_complex: complex | None = None,
    ty_complex: complex | None = None,
) -> SolverResults:
    """Build SolverResults with realistic field components for phase/amplitude tests."""
    h0, h1 = harmonics
    n_freqs = 1
    default_transmission = torch.tensor([0.8], dtype=torch.float32)
    default_reflection = torch.tensor([0.2], dtype=torch.float32)

    # Build transmission field with known complex values at zero-order center
    field_shape = (n_freqs, h0, h1)
    tx_field = torch.zeros(field_shape, dtype=torch.complex64)
    ty_field = torch.zeros(field_shape, dtype=torch.complex64)
    tz_field = torch.zeros(field_shape, dtype=torch.complex64)
    center_x, center_y = h0 // 2, h1 // 2
    if tx_complex is not None:
        tx_field[0, center_x, center_y] = tx_complex
    if ty_complex is not None:
        ty_field[0, center_x, center_y] = ty_complex

    tf = FieldComponents(x=tx_field, y=ty_field, z=tz_field)
    rf = FieldComponents(
        x=torch.zeros(field_shape, dtype=torch.complex64),
        y=torch.zeros(field_shape, dtype=torch.complex64),
        z=torch.zeros(field_shape, dtype=torch.complex64),
    )

    td = torch.zeros(n_freqs, h0, h1, dtype=torch.float32)
    rd = torch.zeros(n_freqs, h0, h1, dtype=torch.float32)

    vec = torch.zeros(1, dtype=torch.float32)
    smat = ScatteringMatrix(
        S11=torch.zeros(1, 1, 1), S12=torch.zeros(1, 1, 1),
        S21=torch.zeros(1, 1, 1), S22=torch.zeros(1, 1, 1),
    )
    wave = WaveVectors(kx=vec, ky=vec, kinc=vec, kzref=vec, kztrn=vec)

    return SolverResults(
        reflection=reflection if reflection is not None else default_reflection,
        transmission=transmission if transmission is not None else default_transmission,
        reflection_diffraction=rd,
        transmission_diffraction=td,
        reflection_field=rf,
        transmission_field=tf,
        structure_matrix=smat,
        wave_vectors=wave,
    )


class TestMetricCriteriaDispatch(unittest.TestCase):
    """Test that criteria with 'metric' key dispatch to the registry."""

    def _evaluate(self, gt_eval, solver_results, *, compile_ok=True, solver_ok=True):
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval
        return evaluate_gt_eval(gt_eval, solver_results, compile_ok=compile_ok, solver_ok=solver_ok)

    def test_metric_criterion_dispatches_to_registry(self):
        """{'metric': 'total_transmission', 'params': {...}} → uses registry."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.9]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "metric": "total_transmission",
                    "params": {"wavelength_index": 0},
                    "operation": ">=",
                    "target": 0.8,
                }
            ],
        }
        out = self._evaluate(gt, results)
        self.assertTrue(out["success_exec"])
        self.assertTrue(out["criteria"][0]["passed"])
        self.assertAlmostEqual(out["criteria"][0]["value"], 0.9, places=5)
        self.assertTrue(out["meets_gt_criteria"])
        self.assertTrue(out["success_goal"])

    def test_expr_criterion_uses_legacy_eval(self):
        """{'expr': '...'} still works via legacy eval() path."""
        results = _make_mock_solver_results(transmission=torch.tensor([0.9]))
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "float(r.transmission[0])", "operation": ">=", "target": 0.8}
            ],
        }
        out = self._evaluate(gt, results)
        self.assertTrue(out["criteria"][0]["passed"])
        self.assertAlmostEqual(out["criteria"][0]["value"], 0.9, places=5)

    def test_mixed_criteria(self):
        """One metric + one expr in same criteria list."""
        results = _make_mock_solver_results(
            transmission=torch.tensor([0.85]),
            reflection=torch.tensor([0.12]),
        )
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "metric": "total_transmission",
                    "params": {"wavelength_index": 0},
                    "operation": ">=",
                    "target": 0.8,
                },
                {"expr": "float(r.reflection[0])", "operation": "<=", "target": 0.15},
            ],
        }
        out = self._evaluate(gt, results)
        self.assertTrue(out["criteria"][0]["passed"])  # 0.85 >= 0.8
        self.assertTrue(out["criteria"][1]["passed"])  # 0.12 <= 0.15
        self.assertTrue(out["meets_gt_criteria"])

    def test_unknown_metric_gives_clear_error(self):
        """Bad metric name → criteria row with error info, not an exception."""
        results = _make_mock_solver_results()
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "metric": "nonexistent_metric",
                    "params": {"wavelength_index": 0},
                    "operation": ">=",
                    "target": 0.8,
                }
            ],
        }
        # Should not crash — should produce a failed criterion with error info
        out = self._evaluate(gt, results)
        self.assertFalse(out["criteria"][0]["passed"])
        self.assertIn("error", out["criteria"][0])

    def test_metric_with_bad_params_gives_clear_error(self):
        """Missing required param → failed criterion with error info."""
        results = _make_mock_solver_results()
        gt = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "metric": "total_transmission",
                    "params": {},  # missing wavelength_index
                    "operation": ">=",
                    "target": 0.8,
                }
            ],
        }
        out = self._evaluate(gt, results)
        self.assertFalse(out["criteria"][0]["passed"])
        self.assertIn("error", out["criteria"][0])


class TestV8SchemaValidation(unittest.TestCase):
    """Test validate_gt_eval accepts v8 metric criteria."""

    def _validate(self, value):
        from evo_metaoptics.meta_design.gt_eval import validate_gt_eval
        return validate_gt_eval(value)

    def test_validate_metric_criterion(self):
        """metric + params + operation + target validates successfully."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "metric": "total_transmission",
                    "params": {"wavelength_index": 0},
                    "operation": ">=",
                    "target": 0.8,
                }
            ],
        }
        result = self._validate(payload)
        self.assertEqual(result["criteria"][0]["metric"], "total_transmission")
        self.assertEqual(result["criteria"][0]["params"]["wavelength_index"], 0)
        self.assertEqual(result["criteria"][0]["operation"], ">=")

    def test_validate_metric_with_tolerance(self):
        """close_to operation with metric + tolerance."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "metric": "zero_order_transmission_phase_deg",
                    "params": {"component": "x", "wavelength_index": 0},
                    "operation": "close_to",
                    "target": 170.0,
                    "tolerance": 8.5,
                }
            ],
        }
        result = self._validate(payload)
        self.assertEqual(result["criteria"][0]["metric"], "zero_order_transmission_phase_deg")
        self.assertAlmostEqual(result["criteria"][0]["tolerance"], 8.5)

    def test_reject_criterion_without_metric_or_expr(self):
        """Neither 'metric' nor 'expr' → error."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {"operation": ">=", "target": 0.8}
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)

    def test_reject_criterion_with_both_metric_and_expr(self):
        """Both 'metric' and 'expr' → ambiguous → error."""
        payload = {
            "wavelength_um": [1.55],
            "criteria": [
                {
                    "metric": "total_transmission",
                    "params": {"wavelength_index": 0},
                    "expr": "float(r.transmission[0])",
                    "operation": ">=",
                    "target": 0.8,
                }
            ],
        }
        with self.assertRaises(ValueError):
            self._validate(payload)


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing v7 specs still work after v8 changes."""

    def _evaluate(self, gt_eval, solver_results):
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval
        return evaluate_gt_eval(gt_eval, solver_results)

    def test_existing_v7_specs_still_work(self):
        """All 5 expression patterns from the codebase."""
        results = _make_mock_solver_results_with_fields(
            transmission=torch.tensor([0.9]),
            reflection=torch.tensor([0.1]),
            tx_complex=complex(1, 1),  # phase = 45 deg
            ty_complex=complex(-1, 0),  # phase = 180 deg
        )

        # Pattern 1: float(r.transmission[0])
        gt1 = {
            "wavelength_um": [1.55],
            "criteria": [{"expr": "float(r.transmission[0])", "operation": ">=", "target": 0.5}],
        }
        out1 = self._evaluate(gt1, results)
        self.assertTrue(out1["criteria"][0]["passed"])

        # Pattern 2: float(r.reflection[0])
        gt2 = {
            "wavelength_um": [1.55],
            "criteria": [{"expr": "float(r.reflection[0])", "operation": "<=", "target": 0.5}],
        }
        out2 = self._evaluate(gt2, results)
        self.assertTrue(out2["criteria"][0]["passed"])

        # Pattern 3: phase via get_zero_order_transmission()[0] (tx → x component)
        gt3 = {
            "wavelength_um": [1.55],
            "criteria": [{
                "expr": "float(torch.rad2deg(torch.angle(r.get_zero_order_transmission()[0][0])))",
                "operation": "close_to", "target": 45.0, "tolerance": 5.0,
            }],
        }
        out3 = self._evaluate(gt3, results)
        self.assertTrue(out3["criteria"][0]["passed"])

        # Pattern 4: phase via get_zero_order_transmission()[1] (ty → y component)
        gt4 = {
            "wavelength_um": [1.55],
            "criteria": [{
                "expr": "float(torch.rad2deg(torch.angle(r.get_zero_order_transmission()[1][0])))",
                "operation": "close_to", "target": 180.0, "tolerance": 5.0,
            }],
        }
        out4 = self._evaluate(gt4, results)
        self.assertTrue(out4["criteria"][0]["passed"])

    def test_v8_metric_gives_same_result_as_v7_expr(self):
        """Parity check: metric and expr produce identical values."""
        results = _make_mock_solver_results_with_fields(
            transmission=torch.tensor([0.85]),
            tx_complex=complex(1, 1),  # phase = 45 deg
        )

        # v7 expr for total_transmission
        gt_v7 = {
            "wavelength_um": [1.55],
            "criteria": [{"expr": "float(r.transmission[0])", "operation": ">=", "target": 0.5}],
        }
        out_v7 = self._evaluate(gt_v7, results)

        # v8 metric for total_transmission
        gt_v8 = {
            "wavelength_um": [1.55],
            "criteria": [{
                "metric": "total_transmission",
                "params": {"wavelength_index": 0},
                "operation": ">=",
                "target": 0.5,
            }],
        }
        out_v8 = self._evaluate(gt_v8, results)

        self.assertAlmostEqual(
            out_v7["criteria"][0]["value"],
            out_v8["criteria"][0]["value"],
            places=6,
        )
        self.assertEqual(out_v7["criteria"][0]["passed"], out_v8["criteria"][0]["passed"])
        self.assertAlmostEqual(
            out_v7["criteria"][0]["margin"],
            out_v8["criteria"][0]["margin"],
            places=6,
        )

    def test_v8_phase_parity(self):
        """Parity check for phase metric."""
        results = _make_mock_solver_results_with_fields(
            tx_complex=complex(1, 1),  # phase = 45 deg
        )

        # v7 expr
        gt_v7 = {
            "wavelength_um": [1.55],
            "criteria": [{
                "expr": "float(torch.rad2deg(torch.angle(r.get_zero_order_transmission()[0][0])))",
                "operation": "close_to", "target": 45.0, "tolerance": 5.0,
            }],
        }
        out_v7 = self._evaluate(gt_v7, results)

        # v8 metric
        gt_v8 = {
            "wavelength_um": [1.55],
            "criteria": [{
                "metric": "zero_order_transmission_phase_deg",
                "params": {"component": "x", "wavelength_index": 0},
                "operation": "close_to", "target": 45.0, "tolerance": 5.0,
            }],
        }
        out_v8 = self._evaluate(gt_v8, results)

        self.assertAlmostEqual(
            out_v7["criteria"][0]["value"],
            out_v8["criteria"][0]["value"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
