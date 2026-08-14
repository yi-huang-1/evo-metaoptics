"""Tests for YAML run config schema and validation.

Tests cover:
- Required sections and field types
- Mode section behavior without baseline flag
- env_vars support
- Bundle settings
- Invalid-config tests with clear field-level error expectations
- One-run-per-YAML constraint
- workspace and log_dir should NOT be user-authored YAML fields
- name field is required and participates in derived folder naming
"""

import importlib
import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_RUN_CONFIG = importlib.import_module("evo_metaoptics.mce.run_config")
RunConfig = _RUN_CONFIG.RunConfig
RunConfigError = _RUN_CONFIG.RunConfigError


class TestRunConfigSchema(unittest.TestCase):
    """Tests for YAML run config schema validation."""

    def test_valid_minimal_config_loads(self):
        """Minimal valid config with required fields should load."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        config = RunConfig.from_dict(config_dict)
        self.assertEqual(config.name, "test-run")

    def test_valid_full_config_with_all_fields(self):
        """Full valid config with all optional fields should load."""
        config_dict = {
            "name": "mini-e2e-2-2-1",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
                "test_data": "meta_design_tasks/splits_mini/iid_test.jsonl",
                "split": "iid",
            },
            "execution": {
                "iterations": 2,
                "start_iter": 1,
                "train_limit": 2,
                "val_limit": 2,
                "test_limit": 1,
                "train_batch_size": 2,
                "codegen_rounds": 3,
                "codegen_inner_attempts": 2,
                "pi_timeout_s": 300,
            },
            "mode": {
                "skill_path": None,
                "no_meta_agent": False,
            },
            "traces": {
                "pi_session_traces": True,
            },
            "env_vars": {
                "MCE_LEARNING_MIN_TRAIN_SAMPLES": "1",
                "MCE_VALIDATION_ATTEMPTS": "3",
            },
            "bundle": {
                "enabled": True,
            },
        }
        config = RunConfig.from_dict(config_dict)
        self.assertEqual(config.name, "mini-e2e-2-2-1")
        self.assertFalse(config.mode.no_meta_agent)
        self.assertEqual(config.execution.iterations, 2)
        self.assertEqual(config.execution.codegen_rounds, 3)
        self.assertEqual(config.execution.codegen_inner_attempts, 2)
        self.assertEqual(config.execution.pi_timeout_s, 300.0)

    def test_mode_use_subagents_is_rejected(self):
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
                "use_subagents": False,
            },
        }

        with self.assertRaises(RunConfigError):
            RunConfig.from_dict(config_dict)

    def test_name_field_is_required(self):
        """Config without 'name' field should raise validation error."""
        config_dict = {
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("name", str(ctx.exception).lower())

    def test_experiment_section_required(self):
        """Config without 'experiment' section should raise validation error."""
        config_dict = {
            "name": "test-run",
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("experiment", str(ctx.exception).lower())

    def test_experiment_env_field_required(self):
        """experiment.env field is required."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("env", str(ctx.exception).lower())

    def test_experiment_model_field_required(self):
        """experiment.model field is required."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("model", str(ctx.exception).lower())

    def test_data_section_required(self):
        """Config without 'data' section should raise validation error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("data", str(ctx.exception).lower())

    def test_data_train_data_required(self):
        """data.train_data field is required."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("train_data", str(ctx.exception).lower())

    def test_data_val_data_required(self):
        """data.val_data field is required."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("val_data", str(ctx.exception).lower())

    def test_execution_section_required(self):
        """Config without 'execution' section should raise validation error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("execution", str(ctx.exception).lower())

    def test_execution_iterations_required(self):
        """execution.iterations field is required."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {},
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("iterations", str(ctx.exception).lower())

    def test_mode_section_required(self):
        """Config without 'mode' section should raise validation error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("mode", str(ctx.exception).lower())

    def test_empty_mode_section_is_valid(self):
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {},
        }
        config = RunConfig.from_dict(config_dict)
        self.assertIsNotNone(config)
        self.assertIsNone(config.mode.skill_path)
        self.assertFalse(config.mode.no_meta_agent)

    def test_baseline_field_rejected_as_unknown_mode_key(self):
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
                "baseline": True,
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("baseline", str(ctx.exception).lower())

    def test_env_vars_section_optional(self):
        """env_vars section is optional."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        config = RunConfig.from_dict(config_dict)
        # Should load without env_vars section
        self.assertIsNotNone(config)

    def test_env_vars_dict_accepted(self):
        """env_vars as dict of string key-value pairs should be accepted."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
            "env_vars": {
                "MCE_LEARNING_MIN_TRAIN_SAMPLES": "1",
                "MCE_VALIDATION_ATTEMPTS": "3",
            },
        }
        config = RunConfig.from_dict(config_dict)
        self.assertEqual(config.env_vars.get("MCE_LEARNING_MIN_TRAIN_SAMPLES"), "1")
        self.assertEqual(config.env_vars.get("MCE_VALIDATION_ATTEMPTS"), "3")

    def test_traces_section_optional(self):
        """traces section is optional."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        config = RunConfig.from_dict(config_dict)
        self.assertIsNotNone(config)

    def test_traces_pi_session_traces_boolean(self):
        """traces.pi_session_traces should be boolean."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
            "traces": {
                "pi_session_traces": True,
            },
        }
        config = RunConfig.from_dict(config_dict)
        self.assertTrue(config.traces.pi_session_traces)

    def test_bundle_section_optional(self):
        """bundle section is optional."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        config = RunConfig.from_dict(config_dict)
        self.assertIsNotNone(config)

    def test_bundle_enabled_boolean(self):
        """bundle.enabled should be boolean."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
            "bundle": {
                "enabled": True,
            },
        }
        config = RunConfig.from_dict(config_dict)
        self.assertTrue(config.bundle.enabled)

    def test_bundle_unknown_field_rejected(self):
        """bundle should reject fields outside the active schema."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
            "bundle": {
                "enabled": True,
                "format": "zip",
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("bundle", str(ctx.exception).lower())

    def test_report_section_rejected(self):
        """report section was removed from the active launcher schema."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
            "report": {
                "generate_from_bundle": True,
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("report", str(ctx.exception).lower())

    def test_workspace_not_user_authored(self):
        """workspace field should NOT be accepted in YAML (launcher-generated)."""
        config_dict = {
            "name": "test-run",
            "workspace": "/some/path",  # Invalid: should not be in YAML
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("workspace", str(ctx.exception).lower())

    def test_log_dir_not_user_authored(self):
        """log_dir field should NOT be accepted in YAML (launcher-generated)."""
        config_dict = {
            "name": "test-run",
            "log_dir": "/some/path",  # Invalid: should not be in YAML
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("log_dir", str(ctx.exception).lower())

    def test_one_run_per_yaml_constraint(self):
        """Config represents exactly one run, not a campaign or matrix."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        config = RunConfig.from_dict(config_dict)
        # Config should be a single run object, not a list or campaign
        self.assertIsInstance(config, RunConfig)
        self.assertFalse(isinstance(config, list))

    def test_name_participates_in_folder_naming(self):
        """name field should be used for derived folder naming."""
        config_dict = {
            "name": "my-special-run-name",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        config = RunConfig.from_dict(config_dict)
        # name should be accessible and used for folder naming
        self.assertEqual(config.name, "my-special-run-name")

    def test_invalid_iterations_type(self):
        """execution.iterations with non-integer value should raise error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": "not-an-int",  # Invalid
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("iterations", str(ctx.exception).lower())

    def test_invalid_train_limit_type(self):
        """execution.train_limit with non-integer value should raise error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
                "train_limit": "not-an-int",  # Invalid
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("train_limit", str(ctx.exception).lower())

    def test_invalid_env_field_type(self):
        """experiment.env with non-string value should raise error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": 123,  # Invalid: should be string
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("env", str(ctx.exception).lower())

    def test_invalid_model_field_type(self):
        """experiment.model with non-string value should raise error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": 456,  # Invalid: should be string
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("model", str(ctx.exception).lower())

    def test_invalid_train_data_type(self):
        """data.train_data with non-string value should raise error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": 789,  # Invalid: should be string
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("train_data", str(ctx.exception).lower())

    def test_invalid_name_type(self):
        """name with non-string value should raise error."""
        config_dict = {
            "name": 123,  # Invalid: should be string
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("name", str(ctx.exception).lower())

    def test_extra_unknown_fields_rejected(self):
        """Config with unknown top-level fields should raise error."""
        config_dict = {
            "name": "test-run",
            "unknown_field": "should-fail",  # Invalid: unknown field
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("unknown", str(ctx.exception).lower())

    def test_empty_name_rejected(self):
        """Config with empty name string should raise error."""
        config_dict = {
            "name": "",  # Invalid: empty string
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("name", str(ctx.exception).lower())

    def test_zero_iterations_rejected(self):
        """execution.iterations with zero value should raise error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 0,  # Invalid: must be >= 1
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("iterations", str(ctx.exception).lower())

    def test_negative_iterations_rejected(self):
        """execution.iterations with negative value should raise error."""
        config_dict = {
            "name": "test-run",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "openai-codex/gpt-5.3-codex",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": -1,  # Invalid: must be >= 1
            },
            "mode": {
            },
        }
        with self.assertRaises(RunConfigError) as ctx:
            RunConfig.from_dict(config_dict)
        self.assertIn("iterations", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
