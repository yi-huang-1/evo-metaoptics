import importlib
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

_base_mod = importlib.import_module("evo_metaoptics.mce_env.base")
EnvironmentResult = _base_mod.EnvironmentResult
EnvironmentRuntimeConfig = _base_mod.EnvironmentRuntimeConfig
MetaopticsInverseDesignEnvironment = importlib.import_module(
    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
).MetaopticsInverseDesignEnvironment
_env_mod = importlib.import_module(
    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
)


def _session_root(*args: Any, **kwargs: Any) -> Path:
    if args:
        return Path(args[0])
    iter_dir = kwargs.get("iter_dir")
    if iter_dir is not None:
        return Path(iter_dir)
    cwd = kwargs.get("cwd")
    if cwd is not None:
        return Path(cwd)
    raise AssertionError("create_pi_session test double missing iter_dir/cwd")


def _make_result(
    cpf: float,
    margin: float,
    round_index: int,
    *,
    trajectory: list[dict] | None = None,
) -> EnvironmentResult:
    return EnvironmentResult(
        feedback=f"round {round_index}",
        ground_truth=None,
        metrics={
            "criteria_pass_fraction": cpf,
            "best_margin": margin,
            "round_index": float(round_index),
        },
        trajectory=trajectory or [],
    )


class TestConstants(unittest.TestCase):
    def test_default_constants(self) -> None:
        mod = importlib.import_module(
            "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
        )
        self.assertEqual(getattr(mod, "_NUM_ROUNDS", None), 5)
        self.assertEqual(getattr(mod, "_INNER_MAX_ATTEMPTS", None), 5)

    def test_runtime_config_overrides_rounds(self) -> None:
        env = MetaopticsInverseDesignEnvironment(
            runtime_config=EnvironmentRuntimeConfig(codegen_rounds=3)
        )
        self.assertEqual(env._resolve_num_rounds(), 3)

    def test_runtime_config_overrides_inner_attempts(self) -> None:
        env = MetaopticsInverseDesignEnvironment(
            runtime_config=EnvironmentRuntimeConfig(codegen_inner_attempts=2)
        )
        self.assertEqual(env._resolve_inner_max_attempts(), 2)

    def test_runtime_config_minimum_is_enforced_by_schema_not_environment(self) -> None:
        env = MetaopticsInverseDesignEnvironment(
            runtime_config=EnvironmentRuntimeConfig(codegen_rounds=1)
        )
        self.assertEqual(env._resolve_num_rounds(), 1)


class TestSelectBestRound(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()

    def test_select_best_round_picks_highest_cpf(self) -> None:
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        results = [
            _make_result(0.3, 0.1, 0),
            _make_result(0.8, 0.0, 1),
            _make_result(0.5, 0.5, 2),
        ]
        best = select_best_round(results, combined_trajectory=[])
        self.assertEqual(best.metrics.get("criteria_pass_fraction"), 0.8)

    def test_select_best_round_tiebreak_by_margin(self) -> None:
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        results = [
            _make_result(0.7, 0.1, 0),
            _make_result(0.7, 0.6, 1),
            _make_result(0.7, 0.2, 2),
        ]
        best = select_best_round(results, combined_trajectory=[])
        self.assertEqual(best.metrics.get("best_margin"), 0.6)

    def test_select_best_round_tiebreak_by_round_index(self) -> None:
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        results = [
            _make_result(0.7, 0.4, 0),
            _make_result(0.7, 0.4, 1),
            _make_result(0.7, 0.4, 2),
        ]
        best = select_best_round(results, combined_trajectory=[])
        self.assertEqual(best.metrics.get("round_index"), 0.0)

    def test_select_best_round_all_failed(self) -> None:
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        results = [
            _make_result(0.0, -0.3, 0),
            _make_result(0.0, 0.1, 1),
            _make_result(0.0, -0.1, 2),
        ]
        best = select_best_round(results, combined_trajectory=[])
        self.assertEqual(best.metrics.get("best_margin"), 0.1)

    def test_select_best_round_single_result(self) -> None:
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        only = _make_result(0.4, 0.2, 0)
        best = select_best_round([only], combined_trajectory=[])
        self.assertEqual(best.metrics.get("criteria_pass_fraction"), 0.4)
        self.assertEqual(best.metrics.get("best_margin"), 0.2)
        self.assertEqual(best.feedback, only.feedback)

    def test_select_best_round_merges_trajectory(self) -> None:
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        combined_trajectory = [
            {"step": "round_start", "round": 1},
            {"step": "attempt_evaluated", "round": 1},
            {"step": "round_start", "round": 2},
            {"step": "attempt_error", "round": 2, "error_type": "runtime_error"},
        ]
        results = [
            _make_result(0.2, -0.1, 0, trajectory=[{"step": "attempt_evaluated", "round": 0}]),
            _make_result(0.9, 0.5, 1, trajectory=[{"step": "attempt_evaluated", "round": 1}]),
        ]
        best = select_best_round(results, combined_trajectory=combined_trajectory)
        self.assertEqual(best.trajectory, combined_trajectory)


class TestFormatRoundFeedback(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()

    def test_format_round_feedback_score_failure(self) -> None:
        self.assertTrue(hasattr(self.env, "_format_round_feedback"))
        format_round_feedback = getattr(self.env, "_format_round_feedback")
        result = EnvironmentResult(
            feedback="round failed",
            ground_truth=None,
            metrics={"criteria_pass_fraction": 0.5, "best_margin": 0.1},
            trajectory=[
                {
                    "step": "attempt_evaluated",
                    "success_goal": False,
                    "criteria": [
                        {"passed": True, "operation": ">=", "target": 0.8, "value": 0.9},
                        {"passed": False, "operation": "<=", "target": 0.2, "value": 0.6},
                    ],
                }
            ],
        )
        text = format_round_feedback(result)
        self.assertIn("[PASS]", text)
        self.assertIn("[FAIL]", text)
        self.assertIn("Score: 1/2 criteria passed", text)

    def test_format_round_feedback_execution_error(self) -> None:
        self.assertTrue(hasattr(self.env, "_format_round_feedback"))
        format_round_feedback = getattr(self.env, "_format_round_feedback")
        result = EnvironmentResult(
            feedback="execution failed",
            ground_truth=None,
            metrics={},
            trajectory=[
                {
                    "step": "attempt_error",
                    "error_type": "runtime_error",
                    "error": "NameError: solver is not defined",
                }
            ],
        )
        text = format_round_feedback(result)
        self.assertIn("runtime_error", text)
        self.assertIn("NameError", text)

    def test_format_round_feedback_no_solution(self) -> None:
        self.assertTrue(hasattr(self.env, "_format_round_feedback"))
        format_round_feedback = getattr(self.env, "_format_round_feedback")
        result = EnvironmentResult(
            feedback="no file",
            ground_truth=None,
            metrics={},
            trajectory=[
                {
                    "step": "attempt_error",
                    "error_type": "no_code_generated",
                    "error": "solution.py was not produced by the agent.",
                }
            ],
        )
        text = format_round_feedback(result)
        self.assertIn("did not produce solution.py", text)


class TestBuildInitialPrompt(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()

    def test_initial_prompt_fresh_contains_absolute_path(self) -> None:
        self.assertTrue(hasattr(self.env, "_build_initial_prompt"))
        build_initial_prompt = getattr(self.env, "_build_initial_prompt")
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp).resolve()
            prompt = build_initial_prompt("design query", round_dir, prev_feedback=None)
            self.assertIn(str(round_dir / "solution.py"), prompt)
            self.assertIn("Write", prompt)

    def test_initial_prompt_carry_forward_contains_absolute_path(self) -> None:
        self.assertTrue(hasattr(self.env, "_build_initial_prompt"))
        build_initial_prompt = getattr(self.env, "_build_initial_prompt")
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp).resolve()
            prompt = build_initial_prompt(
                "design query",
                round_dir,
                prev_feedback="some feedback",
            )
            self.assertIn(str(round_dir / "solution.py"), prompt)

    def test_initial_prompt_carry_forward_contains_feedback(self) -> None:
        self.assertTrue(hasattr(self.env, "_build_initial_prompt"))
        build_initial_prompt = getattr(self.env, "_build_initial_prompt")
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp).resolve()
            prompt = build_initial_prompt(
                "design query",
                round_dir,
                prev_feedback="Score: 1/2 criteria passed.",
            )
            self.assertIn("[FEEDBACK FROM PREVIOUS ROUND]", prompt)

    def test_initial_prompt_carry_forward_instructs_edit_file(self) -> None:
        self.assertTrue(hasattr(self.env, "_build_initial_prompt"))
        build_initial_prompt = getattr(self.env, "_build_initial_prompt")
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp).resolve()
            prompt = build_initial_prompt(
                "design query",
                round_dir,
                prev_feedback="some feedback",
            )
            self.assertIn("edit_file", prompt)


class _RoundSession:
    def __init__(self, cwd: Path, on_send: Any = None, on_close: Any = None) -> None:
        self.cwd = cwd
        self._on_send = on_send
        self._on_close = on_close

    def send_message_sync(self, prompt: str) -> Any:
        if callable(self._on_send):
            self._on_send(self.cwd, prompt)
        solution = self.cwd / "solution.py"
        if not solution.exists():
            solution.write_text(
                "from torchrdit.results import SolverResults\n"
                "def solve_inverse_design(*, device: str = \"cuda\") -> SolverResults:\n"
                "    raise RuntimeError('placeholder')\n",
                encoding="utf-8",
            )
        return SimpleNamespace(error=None)

    async def send_message(self, prompt: str) -> Any:
        del prompt
        raise AssertionError("send_message should not be called")

    def close_sync(self) -> None:
        if callable(self._on_close):
            self._on_close(self.cwd)

    async def close(self) -> None:
        self.close_sync()


class TestMultistartLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()

    def _sample(self) -> Any:
        Sample = importlib.import_module("evo_metaoptics.mce_env.base").Sample
        return Sample(
            id=0,
            question="Design a metasurface",
            extras={
                "gt_eval": {
                    "wavelength_um": [1.55],
                    "criteria": [
                        {
                            "metric": "total_transmission",
                            "params": {"wavelength_index": 0},
                            "operation": ">=",
                            "target": 0.5,
                        }
                    ],
                }
            },
        )

    def _runner_result(self) -> MagicMock:
        result = MagicMock()
        result.error = None
        result.error_type = None
        result.execution_time_s = 1.0
        result.solver_results = MagicMock()
        result.traceback = None
        return result

    def _eval_result(
        self,
        *,
        cpf: float,
        margin: float,
        success_goal: bool,
        success_exec: bool | None = None,
    ) -> dict[str, Any]:
        total = 10
        passed_count = max(0, min(total, int(round(cpf * total))))
        criteria = []
        for i in range(total):
            passed = i < passed_count
            criteria.append(
                {
                    "passed": passed,
                    "margin": margin if passed else margin - 1.0,
                    "violation": 0.0 if passed else 1.0,
                    "operation": ">=",
                    "target": 0.5,
                    "value": 0.8 if passed else 0.2,
                }
            )
        effective_success_exec = success_goal if success_exec is None else success_exec
        return {
            "success_exec": effective_success_exec,
            "success_goal": success_goal,
            "criteria": criteria,
            "best_margin": margin,
        }

    def test_multistart_creates_fresh_session_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)
            created_dirs: list[Path] = []

            def _create_session(*args: Any, **kwargs: Any) -> _RoundSession:
                cwd = _session_root(*args, **kwargs)
                created_dirs.append(cwd)
                return _RoundSession(cwd)

            eval_calls = {"count": 0}

            def _evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                eval_calls["count"] += 1
                return self._eval_result(cpf=0.0, margin=float(eval_calls["count"]), success_goal=False)

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 3),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=_create_session,
                ) as create_session_mock,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_evaluate,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                result = asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(create_session_mock.call_count, 3)
            self.assertEqual(len({str(p) for p in created_dirs}), create_session_mock.call_count)
            self.assertEqual(result.metrics.get("round_count"), 3.0)

    def test_multistart_carries_forward_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)
            checks: list[bool] = []

            def _on_send(cwd: Path, prompt: str) -> None:
                del prompt
                if cwd.name == "round_1":
                    checks.append((cwd / "solution.py").is_file())

            def _create_session(*args: Any, **kwargs: Any) -> _RoundSession:
                return _RoundSession(_session_root(*args, **kwargs), on_send=_on_send)

            eval_calls = {"count": 0}

            def _evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                idx = eval_calls["count"]
                eval_calls["count"] += 1
                if idx == 0:
                    return self._eval_result(cpf=0.0, margin=0.2, success_goal=False)
                return self._eval_result(cpf=1.0, margin=0.3, success_goal=True)

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 2),
                patch.object(_env_mod, "_INNER_MAX_ATTEMPTS", 1),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=_create_session,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_evaluate,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertIn(True, checks)

    def test_multistart_carries_from_best_not_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)
            carried_content: list[str] = []

            def _create_session(*args: Any, **kwargs: Any) -> _RoundSession:
                cwd = _session_root(*args, **kwargs)

                def _on_send(cur_cwd: Path, prompt: str) -> None:
                    del prompt
                    if cur_cwd.name == "round_0":
                        (cur_cwd / "solution.py").write_text("round0", encoding="utf-8")
                    elif cur_cwd.name == "round_1":
                        (cur_cwd / "solution.py").write_text("round1", encoding="utf-8")
                    elif cur_cwd.name == "round_2":
                        carried_content.append((cur_cwd / "solution.py").read_text(encoding="utf-8"))

                return _RoundSession(cwd, on_send=_on_send)

            def _runner_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
                code_path = kwargs.get("code_path") if "code_path" in kwargs else args[0]
                if code_path is None:
                    raise AssertionError("run_codegen missing code_path")
                rr = self._runner_result()
                rr.solver_results = {"round": Path(code_path).parent.name}
                return rr

            def _eval_side_effect(gt_eval: Any, solver_results: Any, **kwargs: Any) -> dict[str, Any]:
                del gt_eval, kwargs
                round_name = solver_results["round"]
                if round_name == "round_0":
                    return self._eval_result(cpf=0.6, margin=0.6, success_goal=False)
                if round_name == "round_1":
                    return self._eval_result(cpf=0.4, margin=0.4, success_goal=False)
                return self._eval_result(cpf=1.0, margin=0.7, success_goal=True)

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 3),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=_create_session,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    side_effect=_runner_side_effect,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_eval_side_effect,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(carried_content[-1], "round0")

    def test_multistart_round0_uses_fresh_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp)
            prompts: list[str] = []

            def _retry(*args: Any, **kwargs: Any) -> EnvironmentResult:
                prompts.append(str(kwargs.get("initial_prompt")))
                return _make_result(1.0, 0.1, 0, trajectory=[{"step": "attempt_evaluated", "attempt": 1}])

            with (
                patch.object(self.env, "_run_codegen_retry_loop", side_effect=_retry),
                patch.object(self.env, "_setup_round_workspace", return_value=(_RoundSession(iter_dir / "round_0"), True)),
            ):
                self.env._run_multistart_loop(
                    query="q",
                    device="cpu",
                    iter_dir=iter_dir,
                    gt_eval={"wavelength_um": [1.55], "criteria": []},
                    skill_content="# SKILL\n",
                    skill_sources=["skill1"],
                    context_dir=None,
                    model="",
                )

            self.assertIn("Write", prompts[0])
            self.assertNotIn("[FEEDBACK FROM PREVIOUS ROUND]", prompts[0])

    def test_multistart_round1_uses_carry_forward_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp)
            prompts: list[str] = []
            calls = {"count": 0}

            def _retry(*args: Any, **kwargs: Any) -> EnvironmentResult:
                prompts.append(str(kwargs.get("initial_prompt")))
                idx = calls["count"]
                calls["count"] += 1
                if idx == 0:
                    return _make_result(0.0, 0.0, 0, trajectory=[{"step": "attempt_error", "error_type": "runtime_error", "error": "x", "attempt": 1}])
                return _make_result(1.0, 0.1, 1, trajectory=[{"step": "attempt_evaluated", "attempt": 1}])

            def _setup(*, round_dir: Path, **kwargs: Any) -> tuple[_RoundSession, bool]:
                del kwargs
                round_dir.mkdir(parents=True, exist_ok=True)
                return _RoundSession(round_dir), False

            with (
                patch.object(self.env, "_run_codegen_retry_loop", side_effect=_retry),
                patch.object(self.env, "_setup_round_workspace", side_effect=_setup),
            ):
                self.env._run_multistart_loop(
                    query="q",
                    device="cpu",
                    iter_dir=iter_dir,
                    gt_eval={"wavelength_um": [1.55], "criteria": []},
                    skill_content="# SKILL\n",
                    skill_sources=["skill1"],
                    context_dir=None,
                    model="",
                )

            self.assertGreaterEqual(len(prompts), 2)
            self.assertIn("[FEEDBACK FROM PREVIOUS ROUND]", prompts[1])

    def test_multistart_early_exit_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)

            def _evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                _evaluate.count += 1
                if _evaluate.count == 1:
                    return self._eval_result(cpf=0.0, margin=0.1, success_goal=False)
                return self._eval_result(cpf=1.0, margin=0.9, success_goal=True)

            _evaluate.count = 0

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 5),
                patch.object(_env_mod, "_INNER_MAX_ATTEMPTS", 1),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=lambda *a, **k: _RoundSession(_session_root(*a, **k)),
                ) as create_session_mock,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_evaluate,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(create_session_mock.call_count, 2)

    def test_multistart_no_early_exit_on_success_exec_without_success_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 5),
                patch.object(_env_mod, "_INNER_MAX_ATTEMPTS", 1),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=lambda *a, **k: _RoundSession(_session_root(*a, **k)),
                ) as create_session_mock,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    return_value=self._eval_result(cpf=0.0, margin=0.1, success_goal=False, success_exec=True),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                result = asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(create_session_mock.call_count, 5)
            self.assertEqual(result.metrics.get("success_exec"), 1.0)
            self.assertEqual(result.metrics.get("success_goal"), 0.0)

    def test_multistart_all_rounds_fail_returns_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)

            def _runner_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
                code_path = kwargs.get("code_path") if "code_path" in kwargs else args[0]
                if code_path is None:
                    raise AssertionError("run_codegen missing code_path")
                rr = self._runner_result()
                rr.solver_results = {"round": Path(code_path).parent.name}
                return rr

            def _eval_side_effect(gt_eval: Any, solver_results: Any, **kwargs: Any) -> dict[str, Any]:
                del gt_eval, kwargs
                round_name = solver_results["round"]
                if round_name == "round_0":
                    return self._eval_result(cpf=0.2, margin=0.1, success_goal=False)
                if round_name == "round_1":
                    return self._eval_result(cpf=0.7, margin=0.2, success_goal=False)
                if round_name == "round_2":
                    return self._eval_result(cpf=0.4, margin=0.3, success_goal=False)
                if round_name == "round_3":
                    return self._eval_result(cpf=0.1, margin=0.0, success_goal=False)
                return self._eval_result(cpf=0.3, margin=-0.1, success_goal=False)

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 5),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=lambda *a, **k: _RoundSession(_session_root(*a, **k)),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    side_effect=_runner_side_effect,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_eval_side_effect,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                result = asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(result.metrics.get("criteria_pass_fraction"), 0.7)

    def test_multistart_single_round_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 1),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=lambda *a, **k: _RoundSession(_session_root(*a, **k)),
                ) as create_session_mock,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    return_value=self._eval_result(cpf=1.0, margin=0.5, success_goal=True),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                result = asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(create_session_mock.call_count, 1)
            self.assertEqual(result.metrics.get("round_count"), 1.0)

    def test_multistart_round_dirs_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)

            def _evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                _evaluate.count += 1
                if _evaluate.count == 1:
                    return self._eval_result(cpf=0.0, margin=0.1, success_goal=False)
                return self._eval_result(cpf=1.0, margin=0.2, success_goal=True)

            _evaluate.count = 0

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 2),
                patch.object(_env_mod, "_INNER_MAX_ATTEMPTS", 1),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=lambda *a, **k: _RoundSession(_session_root(*a, **k)),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_evaluate,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertTrue((log_dir / "iter_0" / "round_0").is_dir())
            self.assertTrue((log_dir / "iter_0" / "round_1").is_dir())

    def test_multistart_session_closed_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)
            close_calls: list[Path] = []

            def _create_session(*args: Any, **kwargs: Any) -> _RoundSession:
                return _RoundSession(
                    _session_root(*args, **kwargs),
                    on_close=lambda p: close_calls.append(p),
                )

            def _evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                _evaluate.count += 1
                if _evaluate.count == 1:
                    return self._eval_result(cpf=0.0, margin=0.1, success_goal=False)
                return self._eval_result(cpf=1.0, margin=0.2, success_goal=True)

            _evaluate.count = 0

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 2),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=_create_session,
                ) as create_session_mock,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_evaluate,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(len(close_calls), create_session_mock.call_count)

    def test_multistart_trajectory_has_round_num(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)

            def _evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                _evaluate.count += 1
                if _evaluate.count == 1:
                    return self._eval_result(cpf=0.0, margin=0.1, success_goal=False)
                return self._eval_result(cpf=1.0, margin=0.2, success_goal=True)

            _evaluate.count = 0

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 2),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=lambda *a, **k: _RoundSession(_session_root(*a, **k)),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=self._runner_result(),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    side_effect=_evaluate,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                result = asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertGreater(len(result.trajectory), 0)
            self.assertTrue(all("round_num" in step for step in result.trajectory if isinstance(step, dict)))


class TestMetricsAndTrainingFormat(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()

    def test_metrics_contain_round_count(self) -> None:
        """Test that _select_best_round adds round_count metric."""
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        results = [
            _make_result(0.5, 0.2, 0),
            _make_result(0.7, 0.4, 1),
            _make_result(0.6, 0.3, 2),
        ]
        best = select_best_round(results, combined_trajectory=[])
        self.assertEqual(best.metrics.get("round_count"), 3.0)

    def test_metrics_contain_total_attempt_count(self) -> None:
        """Test that _select_best_round sums attempt_count across results."""
        self.assertTrue(hasattr(self.env, "_select_best_round"))
        select_best_round = getattr(self.env, "_select_best_round")
        results = [
            _make_result(0.5, 0.2, 0),
            _make_result(0.7, 0.4, 1),
            _make_result(0.6, 0.3, 2),
        ]
        # Manually add attempt_count to metrics
        results[0].metrics["attempt_count"] = 3
        results[1].metrics["attempt_count"] = 2
        results[2].metrics["attempt_count"] = 3
        best = select_best_round(results, combined_trajectory=[])
        self.assertEqual(best.metrics.get("total_attempt_count"), 8)

    def test_get_max_precheck_attempts_matches_budget(self) -> None:
        """Test that get_max_precheck_attempts returns _NUM_ROUNDS * _INNER_MAX_ATTEMPTS."""
        self.assertTrue(hasattr(self.env, "get_max_precheck_attempts"))
        get_max_precheck_attempts = getattr(self.env, "get_max_precheck_attempts")
        result = get_max_precheck_attempts()
        expected = _env_mod._NUM_ROUNDS * _env_mod._INNER_MAX_ATTEMPTS
        self.assertEqual(result, expected)

    def test_format_result_for_training_includes_round_count(self) -> None:
        """Test that format_result_for_training includes round_count in output."""
        self.assertTrue(hasattr(self.env, "format_result_for_training"))
        format_result_for_training = getattr(self.env, "format_result_for_training")
        # Test with round_count present
        item = {
            "sample": {"id": 0, "question": "test", "gt_eval": {}},
            "evaluation": {
                "metrics": {
                    "compile_ok": 1.0,
                    "solver_ok": 1.0,
                    "success_exec": 1.0,
                    "success_goal": 1.0,
                    "best_margin": 0.5,
                    "criteria_pass_fraction": 1.0,
                    "criteria_violation_norm": 0.0,
                    "round_count": 3.0,
                    "total_attempt_count": 8,
                },
                "trajectory": [{"step": "attempt_evaluated", "attempt": 2, "code_hash": "abc123"}],
            },
        }
        result = format_result_for_training(item)
        self.assertEqual(result["round_count"], 3)
        self.assertIsInstance(result["round_count"], int)
        # Test default: item with NO round_count in metrics
        item_no_round = {
            "sample": {"id": 0, "question": "test", "gt_eval": {}},
            "evaluation": {
                "metrics": {
                    "compile_ok": 1.0,
                    "solver_ok": 1.0,
                    "success_exec": 1.0,
                    "success_goal": 1.0,
                    "best_margin": 0.5,
                    "criteria_pass_fraction": 1.0,
                    "criteria_violation_norm": 0.0,
                },
                "trajectory": [{"step": "attempt_evaluated", "attempt": 2, "code_hash": "abc123"}],
            },
        }
        result_default = format_result_for_training(item_no_round)
        self.assertEqual(result_default["round_count"], 1)

_shutdown_mod = importlib.import_module("evo_metaoptics.mce.shutdown")


class TestShutdownInMultistartLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()
        _shutdown_mod.reset()

    def tearDown(self) -> None:
        _shutdown_mod.reset()

    def _sample(self) -> Any:
        Sample = importlib.import_module("evo_metaoptics.mce_env.base").Sample
        return Sample(
            id=0,
            question="Design a metasurface",
            extras={
                "gt_eval": {
                    "wavelength_um": [1.55],
                    "criteria": [
                        {
                            "metric": "total_transmission",
                            "params": {"wavelength_index": 0},
                            "operation": ">=",
                            "target": 0.5,
                        }
                    ],
                }
            },
        )

    def test_multistart_breaks_immediately_on_shutdown(self) -> None:
        _shutdown_mod.request_shutdown()
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._sample()
            log_dir = Path(tmp)

            with (
                patch.object(_env_mod, "_NUM_ROUNDS", 5),
                patch.object(_env_mod, "_INNER_MAX_ATTEMPTS", 1),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=lambda *a, **k: _RoundSession(_session_root(*a, **k)),
                ) as create_session_mock,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                    return_value=MagicMock(selected_sources=["skill1"]),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    return_value=MagicMock(
                        solver_results=MagicMock(), error=None, error_type=None,
                        execution_time_s=0.5, traceback=None,
                    ),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    return_value={
                        "success_exec": True, "success_goal": True,
                        "criteria": [{"passed": True, "violation": 0.0, "margin": 0.5}],
                        "best_margin": 0.5,
                    },
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                result = asyncio.run(
                    self.env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir)
                )

            self.assertEqual(create_session_mock.call_count, 0)
            self.assertIn("shutdown", result.feedback)


if __name__ == "__main__":
    unittest.main()
