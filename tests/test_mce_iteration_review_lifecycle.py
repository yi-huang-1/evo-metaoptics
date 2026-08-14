from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


AgentResponse = importlib.import_module("evo_metaoptics.mce.agent_session").AgentResponse
Sample = importlib.import_module("evo_metaoptics.mce_env.base").Sample
run_iteration = importlib.import_module("evo_metaoptics.mce.main").run_iteration
env_module = importlib.import_module(
    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
)
MetaopticsInverseDesignEnvironment = env_module.MetaopticsInverseDesignEnvironment
SYSTEM_PROMPT = env_module._SYSTEM_PROMPT


VALID_SKILL = """---
name: learning-context
description: Lifecycle test skill.
---

## Skill Overview
Use aggregate diagnostics plus advisory review artifacts to refine bounded search guidance.

## Methodology
1. Read aggregate metrics before changing guidance.
2. Treat the latest review and durable summary as soft evidence.
3. Keep context updates concise and reusable.
"""


class _FakeEnv:
    def get_task_instruction(self) -> str:
        return "task"

    def get_interface_signatures(self) -> list:
        return []

    def get_primary_metric_name(self) -> str:
        return "success_goal"

    def get_required_context_files(self) -> list[str]:
        return []

    def get_min_context_file_chars(self) -> int:
        return 30

    def load_samples(
        self,
        path: str,
        limit: int,
        random_sample: bool = False,
        shuffle: bool = False,
    ) -> list[Sample]:
        del path, random_sample, shuffle
        return [
            Sample(
                id=idx,
                question=f"question-{idx}",
                extras={
                    "gt_eval": {
                        "wavelength_um": [1.55],
                        "criteria": [
                            {
                                "expr": "r.transmission[0].item()",
                                "operation": ">=",
                                "target": 0.8,
                            }
                        ],
                    }
                },
            )
            for idx in range(1, max(0, limit) + 1)
        ]

    def format_result_for_training(self, item):
        return item


class _CapturingMetaSession:
    def __init__(self, *, iter_dir: Path, prompts: list[str]) -> None:
        self._iter_dir = iter_dir
        self._prompts = prompts

    async def send_message(self, prompt: str) -> AgentResponse:
        self._prompts.append(prompt)
        skill_path = self._iter_dir / ".agents" / "skills" / "learning-context" / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(VALID_SKILL, encoding="utf-8")
        return AgentResponse(
            content="done",
            error=None,
        )

    async def close(self) -> None:
        return None


class TestIterationReviewLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_retry_prompt_keeps_strategy_family_during_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter2_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            prompt_env = MetaopticsInverseDesignEnvironment()
            prompt = prompt_env._build_initial_prompt(
                "design task",
                iter_dir,
                prev_feedback="criteria_pass_fraction improved; keep tightening margins",
            )

            self.assertIn("Keep the existing optimization family when it is still improving.", prompt)
            self.assertIn("Only switch families on stagnation", prompt)
            self.assertIn("# Strategy: <family> - <why>", prompt)

    async def test_closed_loop_writes_review_and_feeds_next_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "iter0" / "context").mkdir(parents=True, exist_ok=True)
            run_dir = Path(tmp) / "logs"
            run_dir.mkdir(parents=True, exist_ok=True)

            meta_prompts: dict[int, list[str]] = {1: [], 2: []}

            async def _fake_batch_evaluate(*, samples, **_kwargs):
                value = 1.0 if samples else 0.0
                metrics = {
                    "success_goal": value,
                    "success_exec": value,
                    "criteria_pass_fraction": value,
                    "criteria_violation_norm": 0.0,
                    "best_margin": 0.125 if samples else 0.0,
                }
                return {
                    "summary": {
                        "primary_metric": "success_goal",
                        "primary_metric_value": value,
                        "metrics": metrics,
                        "total": len(samples),
                        "errors": 0,
                    },
                    "results": [
                        {
                            "sample": {"id": sample.id, "question": sample.question},
                            "evaluation": {
                                "metrics": metrics,
                                "feedback": "ok",
                                "trajectory": [],
                            },
                        }
                        for sample in samples
                    ],
                }

            def _fake_aggregate(*, workspace_base: Path, iteration: int, last_sub_folder_name: str, **_kwargs) -> None:
                meta_dir = workspace_base / "meta_agent"
                meta_dir.mkdir(parents=True, exist_ok=True)
                eval_path = meta_dir / "evaluations.json"
                evaluations = {}
                if eval_path.exists():
                    evaluations = json.loads(eval_path.read_text(encoding="utf-8"))
                evaluations[f"iter{iteration}"] = {
                    "primary_metric_name": "success_goal",
                    "train_metrics": {
                        "success_goal": 1.0,
                        "success_exec": 1.0,
                        "criteria_pass_fraction": 1.0,
                        "criteria_violation_norm": 0.0,
                        "best_margin": 0.125,
                    },
                    "val_metrics": {
                        "success_goal": 1.0,
                        "success_exec": 1.0,
                        "criteria_pass_fraction": 1.0,
                        "criteria_violation_norm": 0.0,
                        "best_margin": 0.125,
                    },
                    "criteria_pass_fraction_avg": 1.0,
                    "criteria_violation_norm_avg": 0.0,
                    "best_margin_avg": 0.125,
                    "total_rollouts": 1,
                    "num_sub_iters": 1,
                    "last_sub_folder": last_sub_folder_name,
                }
                eval_path.write_text(json.dumps(evaluations), encoding="utf-8")

                archived_skill = meta_dir / "skills" / f"learning-context-iter{iteration}" / "SKILL.md"
                archived_skill.parent.mkdir(parents=True, exist_ok=True)
                source_skill = (
                    workspace_base
                    / last_sub_folder_name
                    / ".agents"
                    / "skills"
                    / "learning-context"
                    / "SKILL.md"
                )
                archived_skill.write_text(source_skill.read_text(encoding="utf-8"), encoding="utf-8")

            def _fake_create_meta_session(
                workspace_base: Path,
                model: str | None,
                system_prompt: str,
                skill_database: str,
                run_dir: Path | None = None,
                timeout_s: float | None = None,
                session_traces_enabled: bool | None = None,
                iteration: int | None = None,
            ) -> _CapturingMetaSession:
                del model, system_prompt, skill_database, run_dir, timeout_s, session_traces_enabled
                assert iteration is not None
                return _CapturingMetaSession(
                    iter_dir=workspace_base / f"iter{iteration}_sub0",
                    prompts=meta_prompts[iteration],
                )

            def _fake_review_session(
                *,
                workspace_base: Path,
                prompt: str,
                model: str | None = None,
                run_dir: Path | None = None,
                timeout_s: float | None = None,
                session_traces_enabled: bool | None = None,
            ) -> AgentResponse:
                del model, run_dir, timeout_s, session_traces_enabled
                self.assertIn("# Iteration Review Writer", prompt)
                reviews_dir = workspace_base / "meta_agent" / "iteration_reviews"
                manifests = sorted(reviews_dir.glob("iter*_manifest.md"))
                self.assertTrue(manifests)
                current_iteration = int(manifests[-1].stem.removeprefix("iter").removesuffix("_manifest"))
                review_path = reviews_dir / f"iter{current_iteration}.md"
                summary_path = workspace_base / "meta_agent" / "strategy_summary.md"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                review_path.write_text(
                    (
                        "# Iteration Review\n\n"
                        f"- Iteration {current_iteration} review keeps bounded global search evidence.\n"
                    ),
                    encoding="utf-8",
                )
                summary_path.write_text(
                    (
                        "# Strategy Summary\n\n"
                        f"- Durable lesson from iter{current_iteration}: prefer bounded global screening before local refinement.\n"
                    ),
                    encoding="utf-8",
                )
                return AgentResponse(
                    content="done",
                    error=None,
                )

            with (
                patch("evo_metaoptics.mce.main.EnvironmentRegistry.get", return_value=_FakeEnv()),
                patch("evo_metaoptics.mce.main.batch_evaluate", new=AsyncMock(side_effect=_fake_batch_evaluate)),
                patch("evo_metaoptics.mce.main.run_base_agent", new=AsyncMock(return_value={"success": True})),
                patch("evo_metaoptics.mce.main.resolve_learning_min_train_samples_from_env", return_value=1),
                patch("evo_metaoptics.mce.main.load_interfaces", return_value={}),
                patch("evo_metaoptics.mce.main.aggregate_iteration_results", side_effect=_fake_aggregate),
                patch("evo_metaoptics.mce.meta_agent._create_meta_pi_session", side_effect=_fake_create_meta_session),
                patch("evo_metaoptics.mce.iteration_review._run_iteration_review_session", side_effect=_fake_review_session),
            ):
                logger1 = MagicMock()
                iter1_result = await run_iteration(
                    workspace_base=workspace,
                    iteration=1,
                    env_name="metaoptics_inverse_design",
                    val_data_path="val.jsonl",
                    train_data_path="train.jsonl",
                    train_limit=1,
                    val_limit=1,
                    model="mock-model",
                    logger=logger1,
                    run_dir=run_dir,
                    train_batch_size=1,
                )

                manifest_path = workspace / "meta_agent" / "iteration_reviews" / "iter1_manifest.md"
                review_path = workspace / "meta_agent" / "iteration_reviews" / "iter1.md"
                durable_summary_path = workspace / "meta_agent" / "strategy_summary.md"
                mirrored_summary_path = workspace / "context" / "strategy_summary.md"
                self.assertEqual(1, iter1_result["val_total"])
                self.assertTrue(manifest_path.is_file())
                self.assertTrue(review_path.is_file())
                self.assertTrue(durable_summary_path.is_file())
                self.assertEqual(
                    durable_summary_path.read_text(encoding="utf-8"),
                    mirrored_summary_path.read_text(encoding="utf-8"),
                )
                iter1_summary_text = durable_summary_path.read_text(encoding="utf-8")

                logger2 = MagicMock()
                iter2_result = await run_iteration(
                    workspace_base=workspace,
                    iteration=2,
                    env_name="metaoptics_inverse_design",
                    val_data_path="val.jsonl",
                    train_data_path="train.jsonl",
                    train_limit=1,
                    val_limit=1,
                    model="mock-model",
                    logger=logger2,
                    run_dir=run_dir,
                    train_batch_size=1,
                )

            self.assertEqual(1, iter2_result["val_total"])
            iter2_prompt = meta_prompts[2][0]
            self.assertIn("meta_agent/iteration_reviews/iter1.md", iter2_prompt)
            self.assertIn("Iteration 1 review keeps bounded global search evidence.", iter2_prompt)
            self.assertIn("## Durable Lessons", iter2_prompt)
            self.assertIn("meta_agent/strategy_summary.md", iter2_prompt)

            iter2_context_summary = workspace / "iter2_sub0" / "context" / "strategy_summary.md"
            self.assertTrue(iter2_context_summary.is_file())
            self.assertEqual(iter1_summary_text, iter2_context_summary.read_text(encoding="utf-8"))

            prompt_env = MetaopticsInverseDesignEnvironment()
            coding_prompt = prompt_env._build_initial_prompt("design task", workspace / "iter2_sub0")
            self.assertIn("context/strategy_summary.md", SYSTEM_PROMPT)
            self.assertIn("cross-iteration memory", SYSTEM_PROMPT)
            self.assertIn("# Strategy:", SYSTEM_PROMPT)
            self.assertIn("context/strategy_summary.md", coding_prompt)
            self.assertIn("cross-iteration memory", coding_prompt)
            self.assertIn("# Strategy:", coding_prompt)

    async def test_review_failure_is_advisory_for_run_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "iter0" / "context").mkdir(parents=True, exist_ok=True)
            run_dir = Path(tmp) / "logs"
            run_dir.mkdir(parents=True, exist_ok=True)

            async def _fake_batch_evaluate(*, samples, **_kwargs):
                return {
                    "summary": {
                        "primary_metric": "success_goal",
                        "primary_metric_value": 1.0 if samples else 0.0,
                        "metrics": {"success_goal": 1.0, "success_exec": 1.0},
                        "total": len(samples),
                        "errors": 0,
                    },
                    "results": [],
                }

            def _fake_aggregate(*, workspace_base: Path, iteration: int, last_sub_folder_name: str, **_kwargs) -> None:
                meta_dir = workspace_base / "meta_agent"
                meta_dir.mkdir(parents=True, exist_ok=True)
                (meta_dir / "evaluations.json").write_text(
                    json.dumps(
                        {
                            f"iter{iteration}": {
                                "primary_metric_name": "success_goal",
                                "train_metrics": {"success_goal": 1.0, "success_exec": 1.0},
                                "val_metrics": {"success_goal": 1.0, "success_exec": 1.0},
                                "last_sub_folder": last_sub_folder_name,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                archived_skill = meta_dir / "skills" / f"learning-context-iter{iteration}" / "SKILL.md"
                archived_skill.parent.mkdir(parents=True, exist_ok=True)
                archived_skill.write_text(VALID_SKILL, encoding="utf-8")

            def _fake_create_meta_session(
                workspace_base: Path,
                model: str | None,
                system_prompt: str,
                skill_database: str,
                run_dir: Path | None = None,
                timeout_s: float | None = None,
                session_traces_enabled: bool | None = None,
                iteration: int | None = None,
            ) -> _CapturingMetaSession:
                del model, system_prompt, skill_database, run_dir, timeout_s, session_traces_enabled
                assert iteration is not None
                return _CapturingMetaSession(
                    iter_dir=workspace_base / f"iter{iteration}_sub0",
                    prompts=[],
                )

            logger = MagicMock()
            review_exists = False
            summary_exists = False
            with (
                patch("evo_metaoptics.mce.main.EnvironmentRegistry.get", return_value=_FakeEnv()),
                patch("evo_metaoptics.mce.main.batch_evaluate", new=AsyncMock(side_effect=_fake_batch_evaluate)),
                patch("evo_metaoptics.mce.main.run_base_agent", new=AsyncMock(return_value={"success": True})),
                patch("evo_metaoptics.mce.main.resolve_learning_min_train_samples_from_env", return_value=1),
                patch("evo_metaoptics.mce.main.load_interfaces", return_value={}),
                patch("evo_metaoptics.mce.main.aggregate_iteration_results", side_effect=_fake_aggregate),
                patch("evo_metaoptics.mce.meta_agent._create_meta_pi_session", side_effect=_fake_create_meta_session),
                patch(
                    "evo_metaoptics.mce.iteration_review._run_iteration_review_session",
                    side_effect=RuntimeError("review failed"),
                ),
            ):
                result = await run_iteration(
                    workspace_base=workspace,
                    iteration=1,
                    env_name="metaoptics_inverse_design",
                    val_data_path="val.jsonl",
                    train_data_path="train.jsonl",
                    train_limit=1,
                    val_limit=1,
                    model="mock-model",
                    logger=logger,
                    run_dir=run_dir,
                    train_batch_size=1,
                )
                review_exists = (workspace / "meta_agent" / "iteration_reviews" / "iter1.md").is_file()
                summary_exists = (workspace / "meta_agent" / "strategy_summary.md").is_file()

        self.assertEqual(1, result["val_total"])
        self.assertTrue(review_exists)
        self.assertTrue(summary_exists)
        self.assertFalse(logger.exception.called)


if __name__ == "__main__":
    unittest.main()
