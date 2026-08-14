from __future__ import annotations

import tempfile
import unittest
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

AgentResponse = importlib.import_module("evo_metaoptics.mce.agent_session").AgentResponse
learning_context_skill_host_path = importlib.import_module(
    "evo_metaoptics.mce.skills"
).learning_context_skill_host_path
SkillBundle = importlib.import_module("evo_metaoptics.mce.skills").SkillBundle
Sample = importlib.import_module("evo_metaoptics.mce_env.base").Sample
RunnerResult = importlib.import_module(
    "evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner"
).RunnerResult
_env_mod = importlib.import_module(
    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
)
MetaopticsInverseDesignEnvironment = _env_mod.MetaopticsInverseDesignEnvironment


MOD = "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment."


def _sample() -> Sample:
    return Sample(
        id=7,
        question="Design a high-transmission device.",
        extras={
            "gt_eval": {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
                ],
            }
        },
    )


def _bundle() -> SkillBundle:
    return SkillBundle(
        selected_sources=["---\nname: learning-context\n---\n# Skill\n"],
        excluded_sources=[],
        overlay_included=True,
        bundle_hash="0" * 64,
    )


def _latest_round_dir(iter_dir: Path) -> Path:
    round_dirs = [
        child
        for child in iter_dir.iterdir()
        if child.is_dir() and child.name.startswith("round_")
    ]
    if not round_dirs:
        raise AssertionError(f"No round directories found under {iter_dir}")
    return max(round_dirs, key=lambda path: int(path.name.split("_", 1)[1]))


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class TestMetaopticsPiEnvironment(unittest.IsolatedAsyncioTestCase):
    async def test_source_has_no_langchain_or_langgraph_imports(self) -> None:
        env_path = Path("src/evo_metaoptics/mce_env/metaoptics_inverse_design/metaoptics_inverse_design_environment.py")
        text = env_path.read_text(encoding="utf-8")
        self.assertNotIn("langchain", text)
        self.assertNotIn("langgraph", text)
        self.assertNotIn("create_mce_deep_agent", text)
        self.assertNotIn("invoke_mce_deep_agent", text)

    async def test_aevaluate_cold_start_uses_pi_and_deterministic_pipeline(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        sample = _sample()
        session = _FakeSession()
        run_result = RunnerResult(MagicMock(), None, None, 0.25, None)

        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td)
            iter_dir = log_dir / "iter_7"

            def _invoke_side_effect(_session: _FakeSession, prompt: str) -> AgentResponse:
                (_latest_round_dir(iter_dir) / "solution.py").write_text(
                    "def solve_inverse_design(*, device: str = \"cuda\"):\n    return None\n",
                    encoding="utf-8",
                )
                # Contract: prompt must not advertise the old signature.
                assert "solve_inverse_design(query: str)" not in prompt, \
                    "Prompt should not advertise old signature with query parameter"
                # Contract: prompt must still contain the query text.
                assert "Design a high-transmission device" in prompt, \
                    "Prompt should still contain the original query text"
                return AgentResponse(content="done")

            with (
                patch(MOD + "compose_skill_bundle", return_value=_bundle()),
                patch(MOD + "create_pi_session", return_value=session) as create_session_mock,
                patch(MOD + "invoke_pi_session", side_effect=_invoke_side_effect) as invoke_mock,
                patch(MOD + "run_codegen", return_value=run_result) as run_codegen_mock,
                patch(MOD + "materialize_progressive_reference_subtree"),
                patch(
                    MOD + "evaluate_gt_eval",
                    return_value={
                        "success_exec": True,
                        "success_goal": True,
                        "criteria": [{"passed": True, "violation": 0.0, "margin": 0.1}],
                        "best_margin": 0.1,
                    },
                ),
            ):
                result = await env.aevaluate(
                    sample=sample,
                    interfaces={},
                    model="openai/gpt-4.1-mini",
                    context_dir=None,
                    log_dir=log_dir,
                )

                self.assertFalse((iter_dir / "context").exists())
                self.assertTrue((iter_dir / "AGENTS.md").is_file())
                self.assertTrue((iter_dir / "gt_eval.json").is_file())

        self.assertEqual(result.metrics["success_goal"], 1.0)
        self.assertEqual(run_codegen_mock.call_count, 1)
        self.assertEqual(run_codegen_mock.call_args.kwargs["device"], "cpu")
        self.assertNotIn("query", run_codegen_mock.call_args.kwargs)
        self.assertEqual(invoke_mock.call_count, 1)
        create_session_mock.assert_called_once()
        kwargs = create_session_mock.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-4.1-mini")
        self.assertEqual([], kwargs["skills"])
        self.assertEqual(
            [str(learning_context_skill_host_path(iter_dir / "round_0").parent)],
            kwargs["skill_paths"],
        )
        self.assertTrue(session.closed)

    async def test_aevaluate_copies_context_when_available(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        sample = _sample()
        session = _FakeSession()
        run_result = RunnerResult(MagicMock(), None, None, 0.25, None)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs"
            context_dir = root / "seed_context"
            context_dir.mkdir(parents=True)
            (context_dir / "rules.txt").write_text("use deterministic search", encoding="utf-8")
            iter_dir = log_dir / "iter_7"

            def _invoke_side_effect(_session: _FakeSession, prompt: str) -> AgentResponse:
                (_latest_round_dir(iter_dir) / "solution.py").write_text(
                    "def solve_inverse_design(*, device: str = \"cuda\"):\n    return None\n",
                    encoding="utf-8",
                )
                # Contract: prompt must not advertise the old signature.
                assert "solve_inverse_design(query: str)" not in prompt, \
                    "Prompt should not advertise old signature with query parameter"
                # Contract: prompt must still contain the query text.
                assert "Design a high-transmission device" in prompt, \
                    "Prompt should still contain the original query text"
                return AgentResponse(content="done")

            with (
                patch(MOD + "compose_skill_bundle", return_value=_bundle()),
                patch(MOD + "create_pi_session", return_value=session),
                patch(MOD + "invoke_pi_session", side_effect=_invoke_side_effect),
                patch(MOD + "run_codegen", return_value=run_result) as run_codegen_mock,
                patch(MOD + "materialize_progressive_reference_subtree"),
                patch(
                    MOD + "evaluate_gt_eval",
                    return_value={
                        "success_exec": True,
                        "success_goal": True,
                        "criteria": [{"passed": True, "violation": 0.0, "margin": 0.1}],
                        "best_margin": 0.1,
                    },
                ),
            ):
                result = await env.aevaluate(
                    sample=sample,
                    interfaces={},
                    model="openai/gpt-4.1-mini",
                    context_dir=context_dir,
                    log_dir=log_dir,
                )

            self.assertEqual(result.metrics["success_goal"], 1.0)
            self.assertEqual(
                [call.kwargs["device"] for call in run_codegen_mock.call_args_list],
                ["cpu"],
            )
            self.assertTrue(all("query" not in call.kwargs for call in run_codegen_mock.call_args_list))
            self.assertEqual(
                (iter_dir / "round_0" / "context" / "rules.txt").read_text(encoding="utf-8"),
                "use deterministic search",
            )
            agents_text = (iter_dir / "round_0" / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("Read the `context/` directory", agents_text)

    async def test_aevaluate_retries_with_error_feedback(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        sample = _sample()
        session = _FakeSession()
        prompts: list[str] = []
        failure = RunnerResult(None, "bad syntax", "syntax_error", 0.1, "Traceback")
        success = RunnerResult(MagicMock(), None, None, 0.2, None)

        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td)
            iter_dir = log_dir / "iter_7"

            def _invoke_side_effect(_session: _FakeSession, prompt: str) -> AgentResponse:
                prompts.append(prompt)
                (_latest_round_dir(iter_dir) / "solution.py").write_text(
                    "def solve_inverse_design(*, device: str = \"cuda\"):\n    return None\n",
                    encoding="utf-8",
                )
                # Contract: all prompts must not advertise the old signature.
                assert "solve_inverse_design(query: str)" not in prompt, \
                    "Prompt should not advertise old signature with query parameter"
                # Contract: all prompts must still contain the query text.
                assert "Design a high-transmission device" in prompt, \
                    "Prompt should still contain the original query text"
                return AgentResponse(content="updated")

            with (
                patch(MOD + "compose_skill_bundle", return_value=_bundle()),
                patch(MOD + "create_pi_session", return_value=session),
                patch(MOD + "invoke_pi_session", side_effect=_invoke_side_effect),
                patch(MOD + "run_codegen", side_effect=[failure, success]) as run_codegen_mock,
                patch(MOD + "materialize_progressive_reference_subtree"),
                patch(
                    MOD + "evaluate_gt_eval",
                    return_value={
                        "success_exec": True,
                        "success_goal": True,
                        "criteria": [{"passed": True, "violation": 0.0, "margin": 0.1}],
                        "best_margin": 0.1,
                    },
                ),
            ):
                result = await env.aevaluate(sample=sample, interfaces={}, model=None, log_dir=log_dir)

        self.assertEqual(result.metrics["success_goal"], 1.0)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(
            [call.kwargs["device"] for call in run_codegen_mock.call_args_list],
            ["cpu", "cpu"],
        )
        self.assertTrue(all("query" not in call.kwargs for call in run_codegen_mock.call_args_list))
        self.assertIn("EXECUTION FAILED", prompts[1])
        self.assertIn("syntax_error", prompts[1])

    async def test_aevaluate_retries_when_exec_succeeds_but_goal_fails(self) -> None:
        env = MetaopticsInverseDesignEnvironment()
        sample = _sample()
        session = _FakeSession()
        prompts: list[str] = []
        first = RunnerResult(MagicMock(), None, None, 0.1, None)
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td)
            iter_dir = log_dir / "iter_7"

            def _invoke_side_effect(_session: _FakeSession, prompt: str) -> AgentResponse:
                prompts.append(prompt)
                (_latest_round_dir(iter_dir) / "solution.py").write_text(
                    "def solve_inverse_design(*, device: str = \"cuda\"):\n    return None\n",
                    encoding="utf-8",
                )
                assert "solve_inverse_design(query: str)" not in prompt, \
                    "Prompt should not advertise old signature with query parameter"
                assert "Design a high-transmission device" in prompt, \
                    "Prompt should still contain the original query text"
                return AgentResponse(content="updated")

            with (
                patch(MOD + "compose_skill_bundle", return_value=_bundle()),
                patch(MOD + "create_pi_session", return_value=session),
                patch(MOD + "invoke_pi_session", side_effect=_invoke_side_effect),
                patch(MOD + "run_codegen", return_value=first) as run_codegen_mock,
                patch(MOD + "materialize_progressive_reference_subtree"),
                patch(
                    MOD + "evaluate_gt_eval",
                    return_value={
                        "success_exec": True,
                        "success_goal": False,
                        "criteria": [{"passed": False, "violation": 0.3, "margin": -0.3}],
                        "best_margin": -0.3,
                    },
                ),
                patch.object(_env_mod, "_NUM_ROUNDS", 1),
                patch.object(_env_mod, "_INNER_MAX_ATTEMPTS", 1),
            ):
                result = await env.aevaluate(sample=sample, interfaces={}, model=None, log_dir=log_dir)

        self.assertEqual(result.metrics["success_goal"], 0.0)
        self.assertEqual(result.metrics["success_exec"], 1.0)
        self.assertEqual(len(prompts), 1)
        self.assertIn("Design a high-transmission device", prompts[0])


if __name__ == "__main__":
    unittest.main()
