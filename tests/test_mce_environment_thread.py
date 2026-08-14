import asyncio
import importlib
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

AgentResponse = importlib.import_module("evo_metaoptics.mce.agent_session").AgentResponse
Sample = importlib.import_module("evo_metaoptics.mce_env.base").Sample
MetaopticsInverseDesignEnvironment = importlib.import_module(
    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
).MetaopticsInverseDesignEnvironment


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


class _ThreadRecordingSession:
    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd
        self.threads: list[threading.Thread] = []

    @property
    def cwd(self) -> Path:
        return self._cwd

    def send_message_sync(self, prompt: str) -> AgentResponse:
        self.threads.append(threading.current_thread())
        solution = self._cwd / "solution.py"
        solution.write_text(
            "from torchrdit.results import SolverResults\n"
            "def solve_inverse_design(*, device: str = \"cuda\") -> SolverResults:\n"
            "    pass\n",
            encoding="utf-8",
        )
        # Contract: prompt must not advertise the old signature.
        assert "solve_inverse_design(query: str)" not in prompt, \
            "Prompt should not advertise old signature with query parameter"
        # Contract: prompt must still contain the query text.
        assert "Design a metasurface" in prompt, \
            "Prompt should still contain the original query text"
        return AgentResponse(content="done")

    async def send_message(self, prompt: str) -> AgentResponse:
        del prompt
        raise AssertionError("async send_message should NOT be called in sync retry loop")

    def close_sync(self) -> None:
        pass

    async def close(self) -> None:
        pass


class TestEnvironmentThreadModel(unittest.TestCase):
    def test_retry_loop_runs_on_single_non_main_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            env = MetaopticsInverseDesignEnvironment()
            codegen_threads: list[threading.Thread] = []
            sessions: list[_ThreadRecordingSession] = []

            sample = Sample(
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

            mock_runner_result = MagicMock()
            mock_runner_result.error = None
            mock_runner_result.error_type = None
            mock_runner_result.execution_time_s = 1.0
            mock_runner_result.solver_results = MagicMock()
            mock_runner_result.traceback = None

            mock_eval = {
                "success_exec": True,
                "success_goal": True,
                "criteria": [
                    {
                        "passed": True,
                        "margin": 0.1,
                        "violation": 0.0,
                        "operation": ">=",
                        "target": 0.5,
                        "value": 0.6,
                    }
                ],
                "best_margin": 0.1,
            }

            def _run_codegen_side_effect(*args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                codegen_threads.append(threading.current_thread())
                return mock_runner_result

            def _create_session(*args: Any, **kwargs: Any) -> _ThreadRecordingSession:
                session = _ThreadRecordingSession(cwd=_session_root(*args, **kwargs))
                sessions.append(session)
                return session

            with (
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.run_codegen",
                    side_effect=_run_codegen_side_effect,
                ) as run_codegen_mock,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.evaluate_gt_eval",
                    return_value=mock_eval,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.create_pi_session",
                    side_effect=_create_session,
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_preloaded_template_skill",
                    return_value="# SKILL\n\n## Description\nTest skill\n",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.validate_skill_markdown",
                    return_value=(True, None),
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.compose_skill_bundle",
                ) as mock_bundle,
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.materialize_progressive_reference_subtree",
                ),
                patch(
                    "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment.write_agents_md",
                ),
            ):
                mock_bundle.return_value = MagicMock(selected_sources=["skill1"])

                result = asyncio.run(env.aevaluate(sample=sample, interfaces={}, log_dir=log_dir))

            self.assertEqual(result.metrics["success_goal"], 1.0)
            self.assertEqual(run_codegen_mock.call_count, 1)
            self.assertEqual(run_codegen_mock.call_args.kwargs["device"], "cpu")
            self.assertNotIn("query", run_codegen_mock.call_args.kwargs)
            self.assertGreaterEqual(len(sessions), 1)
            session = sessions[0]
            self.assertGreaterEqual(
                len(session.threads), 1, "send_message_sync should have been called"
            )

            unique_threads = set(id(thread_obj) for thread_obj in session.threads)
            self.assertEqual(len(unique_threads), 1, "All Pi invocations should be on the same thread")
            self.assertGreaterEqual(
                len(codegen_threads), 1,
                "run_codegen should have been called",
            )
            self.assertEqual(
                codegen_threads[0],
                session.threads[0],
                "Pi invoke and run_codegen should run on the same thread",
            )

            main_thread = threading.main_thread()
            self.assertNotEqual(
                session.threads[0],
                main_thread,
                "Retry loop should NOT run on the main thread",
            )


if __name__ == "__main__":
    unittest.main()
