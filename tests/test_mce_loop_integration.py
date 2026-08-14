"""M7: MCE loop integration tests for the code-gen environment.

Verifies that MetaopticsInverseDesignEnvironment is fully compatible
with the MCE loop contracts: registry lookup, batch_evaluate result
structure, compute_avg_metrics, format_result_for_training, and
aggregate_iteration_results.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from evo_metaoptics.mce_env.base import EnvironmentResult, Sample
from evo_metaoptics.mce_env.registry import EnvironmentRegistry
from evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment import (
    MetaopticsInverseDesignEnvironment,
)
from evo_metaoptics.mce.utils import (
    compute_avg_metrics,
    _compute_train_failure_summary,
    _collect_train_rollouts,
    aggregate_iteration_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_success_metrics() -> Dict[str, float]:
    """Return metric dict matching a successful aevaluate() call."""
    return {
        "compile_ok": 1.0,
        "solver_ok": 1.0,
        "success_exec": 1.0,
        "success_goal": 1.0,
        "best_margin": 0.05,
        "criteria_pass_fraction": 1.0,
        "criteria_violation_norm": 0.0,
        "execution_time_s": 3.2,
        "attempt_count": 1.0,
    }


def _make_failure_metrics() -> Dict[str, float]:
    """Return metric dict matching a failed aevaluate() call."""
    return {
        "compile_ok": 0.0,
        "solver_ok": 0.0,
        "success_exec": 0.0,
        "success_goal": 0.0,
        "best_margin": 0.0,
        "criteria_pass_fraction": 0.0,
        "criteria_violation_norm": 0.0,
        "execution_time_s": 0.0,
        "attempt_count": 0.0,
    }


def _make_batch_evaluate_result(
    metrics: Dict[str, float],
    sample_id: int = 0,
) -> Dict[str, Any]:
    """Build a single result dict matching batch_evaluate's output format."""
    return {
        "sample": {
            "id": sample_id,
            "question": "Design a high-transmission metasurface.",
            "context": "",
            "ground_truth": None,
            "gt_eval": {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
                ],
            },
        },
        "evaluation": {
            "feedback": "codegen run finished (attempt=1, success_exec=True, success_goal=True).",
            "ground_truth": None,
            "metrics": metrics,
            "trajectory": [
                {
                    "step": "attempt_success",
                    "attempt": 1,
                    "execution_time_s": metrics.get("execution_time_s", 0.0),
                    "code_path": "/tmp/solution.py",
                    "success_exec": True,
                    "success_goal": True,
                }
            ],
        },
    }


def _make_train_json_payload(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a train.json payload for _collect_train_rollouts."""
    return {"detailed_results": rollouts}


# ---------------------------------------------------------------------------
# Test: Registry
# ---------------------------------------------------------------------------


class TestRegistryIntegration(unittest.TestCase):
    """Verify environment is registered and discoverable."""

    def test_environment_registered(self) -> None:
        env = EnvironmentRegistry.get("metaoptics_inverse_design")
        self.assertIsInstance(env, MetaopticsInverseDesignEnvironment)

    def test_environment_listed(self) -> None:
        names = EnvironmentRegistry.list_environments()
        self.assertIn("metaoptics_inverse_design", names)

    def test_primary_metric_name(self) -> None:
        env = EnvironmentRegistry.get("metaoptics_inverse_design")
        self.assertEqual(env.get_primary_metric_name(), "success_goal")

    def test_interface_signatures_empty(self) -> None:
        """Code-gen env returns no interfaces (routing removed in task-195)."""
        env = EnvironmentRegistry.get("metaoptics_inverse_design")
        sigs = env.get_interface_signatures()
        self.assertEqual(len(sigs), 0)
    def test_task_instruction_nonempty(self) -> None:
        env = EnvironmentRegistry.get("metaoptics_inverse_design")
        instruction = env.get_task_instruction()
        self.assertIn("solve_inverse_design", instruction)


# ---------------------------------------------------------------------------
# Test: compute_avg_metrics compatibility
# ---------------------------------------------------------------------------


class TestComputeAvgMetricsCodegen(unittest.TestCase):
    """Verify compute_avg_metrics handles our metric names correctly."""

    def test_all_success(self) -> None:
        results = [
            _make_batch_evaluate_result(_make_success_metrics(), sample_id=0),
            _make_batch_evaluate_result(_make_success_metrics(), sample_id=1),
        ]
        avg = compute_avg_metrics(results)

        # All metrics should be present
        expected_keys = {
            "compile_ok", "solver_ok", "success_exec", "success_goal",
            "best_margin",
            "criteria_pass_fraction", "criteria_violation_norm",
            "execution_time_s", "attempt_count",
        }
        self.assertTrue(expected_keys.issubset(avg.keys()), f"Missing keys: {expected_keys - avg.keys()}")

        # Values should match (identical inputs → same average)
        self.assertAlmostEqual(avg["success_goal"], 1.0)

    def test_mixed_success_failure(self) -> None:
        results = [
            _make_batch_evaluate_result(_make_success_metrics(), sample_id=0),
            _make_batch_evaluate_result(_make_failure_metrics(), sample_id=1),
        ]
        avg = compute_avg_metrics(results)

        # Average of 1.0 and 0.0
        self.assertAlmostEqual(avg["success_goal"], 0.5)
        self.assertAlmostEqual(avg["success_exec"], 0.5)
        self.assertAlmostEqual(avg["compile_ok"], 0.5)

    def test_empty_results(self) -> None:
        avg = compute_avg_metrics([])
        self.assertEqual(avg, {})


# ---------------------------------------------------------------------------
# Test: format_result_for_training contract
# ---------------------------------------------------------------------------


class TestFormatResultForTraining(unittest.TestCase):
    """Verify format_result_for_training produces expected keys."""

    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()

    def test_success_rollout_keys(self) -> None:
        item = _make_batch_evaluate_result(_make_success_metrics(), sample_id=42)
        rollout = self.env.format_result_for_training(item)

        required_keys = {
            "id", "question", "gt_eval",
            "compile_ok", "solver_ok", "success_exec", "success_goal",
            "best_margin", "criteria_pass_fraction", "criteria_violation_norm",
            "execution_time_s", "code_error_type", "code_error_message",
            "code_error_class", "code_traceback_tail", "attempt_error_types",
            "error_reason", "attempt_count", "round_count", "total_attempt_count", "code_hash",
            "fallback_attempted", "fallback_solution_present_before_recovery",
            "fallback_solution_present_after_recovery", "fallback_code_extracted_from_messages",
            "fallback_codegen_executed", "fallback_execution_success", "fallback_no_code_generated",
            "mandatory_guard_retrigger_count", "mandatory_guard_satisfied", "mandatory_guard_exhausted",
        }
        self.assertEqual(required_keys, set(rollout.keys()))

    def test_success_rollout_values(self) -> None:
        item = _make_batch_evaluate_result(_make_success_metrics(), sample_id=42)
        rollout = self.env.format_result_for_training(item)

        self.assertEqual(rollout["id"], 42)
        self.assertTrue(rollout["compile_ok"])
        self.assertTrue(rollout["solver_ok"])
        self.assertTrue(rollout["success_exec"])
        self.assertTrue(rollout["success_goal"])
        self.assertEqual(rollout["attempt_count"], 1)
        self.assertIsNone(rollout["code_error_type"])
        self.assertFalse(rollout["fallback_attempted"])
        self.assertFalse(rollout["fallback_codegen_executed"])

    def test_failure_rollout_values(self) -> None:
        item = _make_batch_evaluate_result(_make_failure_metrics(), sample_id=99)
        # Simulate error trajectory
        item["evaluation"]["trajectory"] = [
            {
                "step": "attempt_error",
                "attempt": 3,
                "error_type": "syntax_error",
                "execution_time_s": 0.0,
            }
        ]
        rollout = self.env.format_result_for_training(item)

        self.assertFalse(rollout["compile_ok"])
        self.assertFalse(rollout["success_exec"])
        self.assertFalse(rollout["success_goal"])
        self.assertEqual(rollout["code_error_type"], "syntax_error")
        self.assertEqual(rollout["attempt_count"], 3)

    def test_format_result_preserves_gt_eval(self) -> None:
        item = _make_batch_evaluate_result(_make_success_metrics(), sample_id=0)
        rollout = self.env.format_result_for_training(item)
        self.assertIsInstance(rollout["gt_eval"], dict)


# ---------------------------------------------------------------------------
# Test: _compute_train_failure_summary compatibility
# ---------------------------------------------------------------------------


class TestComputeTrainFailureSummaryCodegen(unittest.TestCase):

    def test_success_rollouts(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        items = [
            _make_batch_evaluate_result(_make_success_metrics(), sample_id=i)
            for i in range(3)
        ]
        rollouts = [env.format_result_for_training(item) for item in items]
        train_metrics = compute_avg_metrics(items)

        summary = _compute_train_failure_summary(rollouts, train_metrics=train_metrics)

        # Core keys must be present
        required_summary_keys = {
            "error_reason_counts_top3",
            "validator_signature_counts_top5",
            "precheck_ok_rate",
            "avg_precheck_attempts",
            "near_miss_rate",
            "criteria_pass_fraction_avg",
            "criteria_violation_norm_avg",
            "best_margin_avg",
            "optimizer_limited_rate",
            "spec_limited_rate",
            "skill_hash_counts_top3",
        }
        self.assertTrue(
            required_summary_keys.issubset(summary.keys()),
            f"Missing: {required_summary_keys - summary.keys()}",
        )

        # With all-success rollouts, criteria_pass_fraction should be 1.0
        self.assertAlmostEqual(summary["criteria_pass_fraction_avg"], 1.0)
        # precheck_ok should be 0.0 (field absent in code-gen rollouts)
        self.assertAlmostEqual(summary["precheck_ok_rate"], 0.0)
    def test_failure_rollouts(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        items = [
            _make_batch_evaluate_result(_make_failure_metrics(), sample_id=i)
            for i in range(2)
        ]
        rollouts = [env.format_result_for_training(item) for item in items]
        train_metrics = compute_avg_metrics(items)

        summary = _compute_train_failure_summary(rollouts, train_metrics=train_metrics)

        self.assertAlmostEqual(summary["criteria_pass_fraction_avg"], 0.0)
        # spec_limited_rate: all failures with success_exec=False → classified as "exec_failed"
        # so spec_limited_rate should be 0 (only counted when exec succeeded but goal failed)
        self.assertAlmostEqual(summary["spec_limited_rate"], 0.0)

    def test_empty_rollouts_uses_train_metrics_fallback(self) -> None:
        """When no rollouts available, summary falls back to train_metrics."""
        train_metrics: Dict[str, Any] = {
            "success_goal": 0.5,
            "success_exec": 0.8,
            "criteria_pass_fraction": 0.6,
        }
        summary = _compute_train_failure_summary([], train_metrics=train_metrics)
        self.assertAlmostEqual(summary["criteria_pass_fraction_avg"], 0.6)
        self.assertAlmostEqual(summary["near_miss_rate"], 0.0)


# ---------------------------------------------------------------------------
# Test: aggregate_iteration_results end-to-end
# ---------------------------------------------------------------------------


class TestAggregateIterationResultsCodegen(unittest.TestCase):
    """Verify aggregate_iteration_results writes evaluations.json correctly
    using code-gen environment metrics.
    """

    def test_aggregate_writes_evaluations_json(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        success_metrics = _make_success_metrics()
        items = [
            _make_batch_evaluate_result(success_metrics, sample_id=i)
            for i in range(3)
        ]
        avg = compute_avg_metrics(items)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            meta_dir = workspace / "meta_agent"
            meta_dir.mkdir()
            (meta_dir / "evaluations.json").write_text("{}", encoding="utf-8")

            # Create sub-iteration folder with train.json
            sub_folder_name = "iter1_sub1"
            sub_folder = workspace / sub_folder_name
            data_dir = sub_folder / "data"
            data_dir.mkdir(parents=True)
            rollouts = [env.format_result_for_training(item) for item in items]
            train_payload = _make_train_json_payload(rollouts)
            (data_dir / "train.json").write_text(
                json.dumps(train_payload), encoding="utf-8"
            )

            # Create skills dir so skill copy doesn't fail
            skills_dir = sub_folder / ".agents" / "skills" / "learning-context"
            skills_dir.mkdir(parents=True)
            (skills_dir / "SKILL.md").write_text(
                "---\nname: learning-context\n---\n\n# Skill\nContent.",
                encoding="utf-8",
            )

            sub_iterations = [
                {
                    "folder": str(sub_folder),
                    "batch_size": 3,
                    "batch_train_primary_metric": avg.get("success_goal", 0.0),
                    "batch_train_metrics": avg,
                }
            ]

            aggregate_iteration_results(
                workspace_base=workspace,
                iteration=1,
                sub_iterations=sub_iterations,
                val_primary_metric=avg.get("success_goal", 0.0),
                val_metrics=avg,
                val_total=3,
                cumulative_rollouts=3,
                num_sub_iters=1,
                last_sub_folder_name=sub_folder_name,
                environment=env,
                skill_guard=None,
                logger=MagicMock(),
            )

            # Verify evaluations.json was written with iter1
            evals = json.loads((meta_dir / "evaluations.json").read_text(encoding="utf-8"))
            self.assertIn("iter1", evals)
            summary = evals["iter1"]

            self.assertEqual(summary["primary_metric_name"], "success_goal")
            self.assertAlmostEqual(summary["train_metrics"]["success_goal"], 1.0)
            self.assertAlmostEqual(summary["val_metrics"]["success_goal"], 1.0)
            self.assertAlmostEqual(summary["train_metrics"]["success_exec"], 1.0)
            self.assertAlmostEqual(summary["val_metrics"]["success_exec"], 1.0)

            self.assertAlmostEqual(summary["precheck_ok_rate"], 0.0)
            self.assertNotIn("promotion_decision", summary)

    def test_aggregate_handles_all_failures(self) -> None:
        """Verify aggregate works when all rollouts are failures."""
        env = MetaopticsInverseDesignEnvironment()
        failure_metrics = _make_failure_metrics()
        items = [
            _make_batch_evaluate_result(failure_metrics, sample_id=i)
            for i in range(2)
        ]
        avg = compute_avg_metrics(items)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            meta_dir = workspace / "meta_agent"
            meta_dir.mkdir()
            (meta_dir / "evaluations.json").write_text("{}", encoding="utf-8")

            sub_folder_name = "iter1_sub1"
            sub_folder = workspace / sub_folder_name
            data_dir = sub_folder / "data"
            data_dir.mkdir(parents=True)
            rollouts = [env.format_result_for_training(item) for item in items]
            train_payload = _make_train_json_payload(rollouts)
            (data_dir / "train.json").write_text(
                json.dumps(train_payload), encoding="utf-8"
            )

            skills_dir = sub_folder / ".agents" / "skills" / "learning-context"
            skills_dir.mkdir(parents=True)
            (skills_dir / "SKILL.md").write_text(
                "---\nname: learning-context\n---\n\n# Skill\nContent.",
                encoding="utf-8",
            )

            sub_iterations = [
                {
                    "folder": str(sub_folder),
                    "batch_size": 2,
                    "batch_train_primary_metric": 0.0,
                    "batch_train_metrics": avg,
                }
            ]

            # loss=0.0 with success_exec=0 → need to handle the val_loss assertion
            # The function requires val_loss to be a finite float when
            # val_success_exec > 0 — but here success_exec=0, so it's OK.
            aggregate_iteration_results(
                workspace_base=workspace,
                iteration=1,
                sub_iterations=sub_iterations,
                val_primary_metric=0.0,
                val_metrics=avg,
                val_total=2,
                cumulative_rollouts=2,
                num_sub_iters=1,
                last_sub_folder_name=sub_folder_name,
                environment=env,
                skill_guard=None,
                logger=MagicMock(),
            )

            evals = json.loads((meta_dir / "evaluations.json").read_text(encoding="utf-8"))
            self.assertIn("iter1", evals)
            summary = evals["iter1"]
            self.assertAlmostEqual(summary["train_metrics"]["success_goal"], 0.0)
            self.assertAlmostEqual(summary["val_metrics"]["success_goal"], 0.0)


# ---------------------------------------------------------------------------
# Test: main.py reads the expected metric keys
# ---------------------------------------------------------------------------


class TestMainLoopMetricKeys(unittest.TestCase):
    """Verify the metric keys that main.py reads from val_summary are produced."""

    def test_required_metric_keys_in_avg_metrics(self) -> None:
        results = [_make_batch_evaluate_result(_make_success_metrics())]
        avg = compute_avg_metrics(results)

        for key in ("success_goal", "success_exec"):
            self.assertIn(key, avg, f"main.py expects avg_metrics['{key}'] but it's missing")

    def test_primary_metric_in_avg_metrics(self) -> None:
        """batch_evaluate reads primary_metric_name from environment,
        then looks it up in avg_metrics.
        """
        env = MetaopticsInverseDesignEnvironment()
        results = [_make_batch_evaluate_result(_make_success_metrics())]
        avg = compute_avg_metrics(results)
        primary = env.get_primary_metric_name()
        self.assertIn(primary, avg)
        self.assertAlmostEqual(avg[primary], 1.0)


# ---------------------------------------------------------------------------
# Test: _collect_train_rollouts reads format_result_for_training output
# ---------------------------------------------------------------------------


class TestCollectTrainRolloutsCodegen(unittest.TestCase):
    """Verify _collect_train_rollouts can read rollouts from train.json
    written using format_result_for_training output.
    """

    def test_collect_from_disk(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        item = _make_batch_evaluate_result(_make_success_metrics(), sample_id=0)
        rollout = env.format_result_for_training(item)

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_folder = Path(tmpdir) / "iter1_sub1"
            data_dir = sub_folder / "data"
            data_dir.mkdir(parents=True)
            payload = _make_train_json_payload([rollout])
            (data_dir / "train.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            sub_iterations = [{"folder": str(sub_folder), "batch_size": 1}]
            collected = _collect_train_rollouts(sub_iterations)

            self.assertEqual(len(collected), 1)
            self.assertEqual(collected[0]["id"], 0)
            self.assertTrue(collected[0]["success_goal"])


if __name__ == "__main__":
    unittest.main()
