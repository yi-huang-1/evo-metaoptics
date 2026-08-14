"""Tests for M5: End-to-End Offline Pipeline Verification.

Verifies the complete offline pipeline produces all expected artifacts
when start_iter=0 and test_data are configured.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from evo_metaoptics.mce.main import run_iteration
from evo_metaoptics.mce.skills import learning_context_skill_host_path


def _make_mock_env(primary_metric_name: str = "criteria_pass_fraction"):
    env = MagicMock()
    env.get_primary_metric_name.return_value = primary_metric_name
    env.get_task_instruction.return_value = "test instruction"
    env.get_interface_signatures.return_value = []
    env.get_required_context_files.return_value = []
    env.get_min_context_file_chars.return_value = 0
    env.load_samples.return_value = [
        {"id": "sample_1", "query": "q1"},
        {"id": "sample_2", "query": "q2"},
    ]
    env.format_result_for_training.side_effect = lambda item: item
    return env


def _make_batch_eval_return(**kwargs):
    defaults = {
        "primary_metric": "criteria_pass_fraction",
        "primary_metric_value": 0.3,
        "metrics": {
            "criteria_pass_fraction": 0.3,
            "success_goal": 0.2,
            "success_exec": 0.8,
            "best_margin": -0.5,
            "criteria_violation_norm": 0.4,
        },
        "total": 2,
        "errors": 0,
    }
    defaults.update(kwargs)
    return {
        "summary": defaults,
        "results": [
            {"id": 0, "success_exec": True, "success_goal": False},
        ],
    }


def _make_meta_success():
    return {"success": True, "skill_md": "# skill", "skill_guard": None}


def _make_base_success():
    return {"success": True}


class TestFullPipelineArtifacts(unittest.TestCase):
    """T5.1: Full pipeline with start_iter=0 produces all expected artifacts."""

    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.workspace = Path(self._tmpdir.name) / "workspace"
        self.workspace.mkdir()
        self.run_dir = Path(self._tmpdir.name) / "logs" / "run_test"
        self.run_dir.mkdir(parents=True)
        self.logger = MagicMock()

    def tearDown(self):
        self._tmpdir.cleanup()

    @patch("evo_metaoptics.mce.main.EnvironmentRegistry")
    @patch("evo_metaoptics.mce.main.batch_evaluate", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_meta_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_base_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.aggregate_iteration_results")
    @patch("evo_metaoptics.mce.main.run_iteration_review")
    @patch("evo_metaoptics.mce.main.load_interfaces")
    def test_iter0_produces_baseline_artifact(
        self,
        mock_load_ifaces,
        mock_review,
        mock_aggregate,
        mock_base_agent,
        mock_meta_agent,
        mock_batch_eval,
        mock_registry,
    ):
        env = _make_mock_env()
        mock_registry.get.return_value = env
        mock_batch_eval.return_value = _make_batch_eval_return()

        result = asyncio.run(
            run_iteration(
                workspace_base=self.workspace,
                iteration=0,
                env_name="metaoptics_inverse_design",
                val_data_path="dummy_val.jsonl",
                train_data_path="dummy_train.jsonl",
                train_limit=2,
                val_limit=2,
                model="test-model",
                logger=self.logger,
                run_dir=self.run_dir,
            )
        )

        assert (self.workspace / "iter0_sub0").exists()
        assert (self.workspace / "meta_agent" / "baseline_evaluation.json").exists()

        mock_meta_agent.assert_not_called()
        mock_base_agent.assert_not_called()
        mock_aggregate.assert_not_called()

        self.assertEqual(result["iteration"], 0)
        self.assertEqual(result["cumulative_rollouts"], 0)

    @patch("evo_metaoptics.mce.main.EnvironmentRegistry")
    @patch("evo_metaoptics.mce.main.batch_evaluate", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_meta_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_base_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.aggregate_iteration_results")
    @patch("evo_metaoptics.mce.main.run_iteration_review")
    @patch("evo_metaoptics.mce.main.load_interfaces")
    def test_iter1_after_iter0_produces_training_artifacts(
        self,
        mock_load_ifaces,
        mock_review,
        mock_aggregate,
        mock_base_agent,
        mock_meta_agent,
        mock_batch_eval,
        mock_registry,
    ):
        env = _make_mock_env()
        mock_registry.get.return_value = env
        mock_batch_eval.return_value = _make_batch_eval_return()
        mock_meta_agent.return_value = _make_meta_success()
        mock_base_agent.return_value = _make_base_success()
        mock_load_ifaces.return_value = {}

        asyncio.run(
            run_iteration(
                workspace_base=self.workspace,
                iteration=0,
                env_name="metaoptics_inverse_design",
                val_data_path="dummy_val.jsonl",
                train_data_path="dummy_train.jsonl",
                train_limit=2,
                val_limit=2,
                model="test-model",
                logger=self.logger,
                run_dir=self.run_dir,
            )
        )

        result = asyncio.run(
            run_iteration(
                workspace_base=self.workspace,
                iteration=1,
                env_name="metaoptics_inverse_design",
                val_data_path="dummy_val.jsonl",
                train_data_path="dummy_train.jsonl",
                train_limit=2,
                val_limit=2,
                model="test-model",
                logger=self.logger,
                run_dir=self.run_dir,
            )
        )

        assert (self.workspace / "meta_agent" / "baseline_evaluation.json").exists()
        assert (self.workspace / "iter1_sub0").exists()
        self.assertEqual(result["iteration"], 1)
        assert result["cumulative_rollouts"] > 0

        mock_meta_agent.assert_called_once()
        mock_aggregate.assert_called_once()


class TestPipelineWithoutTestData(unittest.TestCase):
    """T5.3: Pipeline without test_data still works (no regression)."""

    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.workspace = Path(self._tmpdir.name) / "workspace"
        self.workspace.mkdir()
        self.run_dir = Path(self._tmpdir.name) / "logs" / "run_test"
        self.run_dir.mkdir(parents=True)
        self.logger = MagicMock()

    def tearDown(self):
        self._tmpdir.cleanup()

    @patch("evo_metaoptics.mce.main.EnvironmentRegistry")
    @patch("evo_metaoptics.mce.main.batch_evaluate", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_meta_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_base_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.aggregate_iteration_results")
    @patch("evo_metaoptics.mce.main.run_iteration_review")
    @patch("evo_metaoptics.mce.main.load_interfaces")
    def test_iter1_without_test_data_works(
        self,
        mock_load_ifaces,
        mock_review,
        mock_aggregate,
        mock_base_agent,
        mock_meta_agent,
        mock_batch_eval,
        mock_registry,
    ):
        env = _make_mock_env()
        mock_registry.get.return_value = env
        mock_batch_eval.return_value = _make_batch_eval_return()
        mock_meta_agent.return_value = _make_meta_success()
        mock_base_agent.return_value = _make_base_success()
        mock_load_ifaces.return_value = {}

        result = asyncio.run(
            run_iteration(
                workspace_base=self.workspace,
                iteration=1,
                env_name="metaoptics_inverse_design",
                val_data_path="dummy_val.jsonl",
                train_data_path="dummy_train.jsonl",
                train_limit=2,
                val_limit=2,
                model="test-model",
                logger=self.logger,
                run_dir=self.run_dir,
            )
        )

        self.assertEqual(result["iteration"], 1)
        self.assertFalse(
            (self.workspace / "meta_agent" / "baseline_evaluation.json").exists()
        )
        self.assertFalse(
            (self.workspace / "meta_agent" / "final_test_evaluation.json").exists()
        )


class TestPipelineWithStartIter1(unittest.TestCase):
    """T5.4: Pipeline with start_iter=1 still works (no regression)."""

    def setUp(self):
        self._tmpdir = TemporaryDirectory()
        self.workspace = Path(self._tmpdir.name) / "workspace"
        self.workspace.mkdir()
        self.run_dir = Path(self._tmpdir.name) / "logs" / "run_test"
        self.run_dir.mkdir(parents=True)
        self.logger = MagicMock()

    def tearDown(self):
        self._tmpdir.cleanup()

    @patch("evo_metaoptics.mce.main.EnvironmentRegistry")
    @patch("evo_metaoptics.mce.main.batch_evaluate", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_meta_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.run_base_agent", new_callable=AsyncMock)
    @patch("evo_metaoptics.mce.main.aggregate_iteration_results")
    @patch("evo_metaoptics.mce.main.run_iteration_review")
    @patch("evo_metaoptics.mce.main.load_interfaces")
    def test_start_iter_1_skips_baseline(
        self,
        mock_load_ifaces,
        mock_review,
        mock_aggregate,
        mock_base_agent,
        mock_meta_agent,
        mock_batch_eval,
        mock_registry,
    ):
        env = _make_mock_env()
        mock_registry.get.return_value = env
        mock_batch_eval.return_value = _make_batch_eval_return()
        mock_meta_agent.return_value = _make_meta_success()
        mock_base_agent.return_value = _make_base_success()
        mock_load_ifaces.return_value = {}

        result = asyncio.run(
            run_iteration(
                workspace_base=self.workspace,
                iteration=1,
                env_name="metaoptics_inverse_design",
                val_data_path="dummy_val.jsonl",
                train_data_path="dummy_train.jsonl",
                train_limit=2,
                val_limit=2,
                model="test-model",
                logger=self.logger,
                run_dir=self.run_dir,
            )
        )

        self.assertEqual(result["iteration"], 1)
        self.assertFalse(
            (self.workspace / "meta_agent" / "baseline_evaluation.json").exists()
        )
        self.assertFalse((self.workspace / "iter0_sub0").exists())


if __name__ == "__main__":
    unittest.main()
