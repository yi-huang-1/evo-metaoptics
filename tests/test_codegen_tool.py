from __future__ import annotations

# pyright: reportMissingImports=false

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import RunnerResult
from evo_metaoptics.mce_env.metaoptics_inverse_design import metaoptics_inverse_design_environment as env_mod


class TestMakeRunCodegenTool(unittest.TestCase):
    def _runtime_error(self, message: str = "boom") -> RunnerResult:
        return RunnerResult(
            solver_results=None,
            error=message,
            error_type="runtime_error",
            execution_time_s=0.1,
            traceback="tb",
        )

    def test_factory_returns_tool_and_empty_result_ref(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tool, result_ref = env_mod.make_run_codegen_tool(
                iter_dir=Path(td),
                device="cpu",
            )

        self.assertIsNotNone(tool)
        self.assertIsInstance(result_ref, dict)
        self.assertIn("runner_result", result_ref)
        self.assertIsNone(result_ref["runner_result"])

    def test_missing_solution_py_returns_error_and_keeps_result_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tool, result_ref = env_mod.make_run_codegen_tool(
                iter_dir=Path(td),
                device="cpu",
            )
            out = tool.invoke({})

        self.assertIn("EXECUTION FAILED", out)
        self.assertIn("solution.py not found", out)
        self.assertIsNone(result_ref["runner_result"])

    def test_success_response_and_result_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text(
                "def solve_inverse_design(query: str):\n    return None\n",
                encoding="utf-8",
            )
            ok = RunnerResult(
                solver_results=object(),
                error=None,
                error_type=None,
                execution_time_s=0.2,
                traceback=None,
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=ok):
                tool, result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("EXECUTION SUCCESS", out)
        self.assertIs(result_ref["runner_result"], ok)

    def test_error_response_contains_type_message_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="boom",
                error_type="runtime_error",
                execution_time_s=0.3,
                traceback="Traceback: boom",
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("EXECUTION FAILED", out)
        self.assertIn("Error type: runtime_error", out)
        self.assertIn("Error: boom", out)
        self.assertIn("Traceback", out)
        self.assertIs(result_ref["runner_result"], err)

    def test_timeout_response_contains_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="Execution timed out after 120s.",
                error_type="timeout",
                execution_time_s=120.0,
                traceback=None,
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("EXECUTION FAILED", out)
        self.assertIn("Error type: timeout", out)
        self.assertIn("timed out", out)
        self.assertIs(result_ref["runner_result"], err)

    def test_error_feedback_prefers_edit_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="boom",
                error_type="runtime_error",
                execution_time_s=0.1,
                traceback="tb",
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, _result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("edit_file", out)

    def test_missing_file_feedback_includes_write_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            tool, _result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
            out = tool.invoke({})

        self.assertIn("write_file", out)

    def test_multiple_calls_keep_last_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            first = RunnerResult(None, "e1", "runtime_error", 0.1, "tb1")
            second = RunnerResult(object(), None, None, 0.2, None)
            with patch.object(env_mod.codegen_runner, "run_codegen", side_effect=[first, second]):
                tool, result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                tool.invoke({})
                tool.invoke({})

        self.assertIs(result_ref["runner_result"], second)

    def test_consecutive_same_error_triggers_guard_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = self._runtime_error("boom")
            with patch.object(
                env_mod.codegen_runner,
                "run_codegen",
                side_effect=[err, err, err, err],
            ):
                tool, _result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out1 = tool.invoke({})
                out2 = tool.invoke({})
                out3 = tool.invoke({})
                out4 = tool.invoke({})

        self.assertIn("EXECUTION FAILED", out1)
        self.assertIn("EXECUTION FAILED", out2)
        self.assertIn("EXECUTION FAILED", out3)
        self.assertIn("LOOP GUARD ACTIVATED", out4)
        self.assertIn("edit_file", out4)

    def test_guard_resets_on_different_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            boom = self._runtime_error("boom")
            different = self._runtime_error("different_boom")
            with patch.object(
                env_mod.codegen_runner,
                "run_codegen",
                side_effect=[boom, boom, different, different],
            ):
                tool, _result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                tool.invoke({})
                tool.invoke({})
                out3 = tool.invoke({})
                out4 = tool.invoke({})

        self.assertIn("EXECUTION FAILED", out3)
        self.assertNotIn("LOOP GUARD ACTIVATED", out3)
        self.assertIn("EXECUTION FAILED", out4)
        self.assertNotIn("LOOP GUARD ACTIVATED", out4)

    def test_guard_resets_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            boom = self._runtime_error("boom")
            success = RunnerResult(
                solver_results=object(),
                error=None,
                error_type=None,
                execution_time_s=0.1,
                traceback=None,
            )
            with patch.object(
                env_mod.codegen_runner,
                "run_codegen",
                side_effect=[boom, boom, success, boom, boom],
            ):
                tool, _result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                tool.invoke({})
                tool.invoke({})
                out3 = tool.invoke({})
                out4 = tool.invoke({})
                out5 = tool.invoke({})

        self.assertIn("EXECUTION SUCCESS", out3)
        self.assertIn("EXECUTION FAILED", out4)
        self.assertNotIn("LOOP GUARD ACTIVATED", out4)
        self.assertIn("EXECUTION FAILED", out5)
        self.assertNotIn("LOOP GUARD ACTIVATED", out5)

    def test_guard_message_does_not_call_runner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = self._runtime_error("boom")
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err) as run_mock:
                tool, _result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                tool.invoke({})
                tool.invoke({})
                tool.invoke({})
                out4 = tool.invoke({})

        self.assertIn("LOOP GUARD ACTIVATED", out4)
        self.assertEqual(run_mock.call_count, 3)

    def test_guard_keeps_firing_without_file_modification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = self._runtime_error("boom")
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err) as run_mock:
                tool, _ = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                tool.invoke({})
                tool.invoke({})
                tool.invoke({})
                out4 = tool.invoke({})
                out5 = tool.invoke({})

        self.assertIn("LOOP GUARD ACTIVATED", out4)
        self.assertIn("LOOP GUARD ACTIVATED", out5)
        self.assertEqual(run_mock.call_count, 3)

    def test_guard_resets_on_file_modification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = self._runtime_error("boom")
            err2 = self._runtime_error("new_error")
            with patch.object(
                env_mod.codegen_runner,
                "run_codegen",
                side_effect=[err, err, err, err2],
            ) as run_mock:
                tool, _ = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                tool.invoke({})
                tool.invoke({})
                tool.invoke({})
                out4 = tool.invoke({})
                (iter_dir / "solution.py").write_text("fixed code", encoding="utf-8")
                out5 = tool.invoke({})

        self.assertIn("LOOP GUARD ACTIVATED", out4)
        self.assertIn("EXECUTION FAILED", out5)
        self.assertIn("new_error", out5)
        self.assertNotIn("LOOP GUARD ACTIVATED", out5)
        self.assertEqual(run_mock.call_count, 4)

    def test_guard_resets_after_activation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            boom = self._runtime_error("boom")
            different = self._runtime_error("different_boom")
            with patch.object(
                env_mod.codegen_runner,
                "run_codegen",
                side_effect=[boom, boom, boom, different],
            ):
                tool, _result_ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                tool.invoke({})
                tool.invoke({})
                tool.invoke({})
                out4 = tool.invoke({})
                (iter_dir / "solution.py").write_text("modified", encoding="utf-8")
                out5 = tool.invoke({})

        self.assertIn("LOOP GUARD ACTIVATED", out4)
        self.assertIn("EXECUTION FAILED", out5)
        self.assertIn("different_boom", out5)
        self.assertNotIn("LOOP GUARD ACTIVATED", out5)



class TestErrorAugmentedApiHints(unittest.TestCase):
    """M3: Error-augmented API hints in run_codegen error output."""

    def test_import_error_torchrdit_structures_includes_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="module 'torchrdit' has no attribute 'structures'",
                error_type="import_error",
                execution_time_s=0.1,
                traceback="AttributeError: module 'torchrdit' has no attribute 'structures'",
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, _ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("API HINTS", out)
        self.assertIn("torchrdit.structures does NOT exist", out)
        self.assertIn("get_solver_builder", out)

    def test_import_error_torchrdit_sources_includes_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="cannot import name 'sources' from 'torchrdit'",
                error_type="import_error",
                execution_time_s=0.1,
                traceback="ImportError: cannot import name 'sources' from 'torchrdit'",
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, _ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("API HINTS", out)
        self.assertIn("torchrdit.sources does NOT exist", out)

    def test_attribute_error_torchrdit_materials_includes_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="module 'torchrdit' has no attribute 'materials'",
                error_type="runtime_error",
                execution_time_s=0.1,
                traceback="AttributeError: module 'torchrdit' has no attribute 'materials'",
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, _ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("API HINTS", out)
        self.assertIn("torchrdit.materials does NOT exist", out)
        self.assertIn("create_material", out)

    def test_cannot_import_name_includes_generic_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="cannot import name 'FooBar' from 'torchrdit'",
                error_type="import_error",
                execution_time_s=0.1,
                traceback="ImportError: cannot import name 'FooBar' from 'torchrdit'",
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, _ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertIn("API HINTS", out)
        self.assertIn("Check the skill templates", out)

    def test_unrelated_error_has_no_api_hints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            (iter_dir / "solution.py").write_text("bad", encoding="utf-8")
            err = RunnerResult(
                solver_results=None,
                error="ZeroDivisionError: division by zero",
                error_type="runtime_error",
                execution_time_s=0.1,
                traceback="ZeroDivisionError: division by zero",
            )
            with patch.object(env_mod.codegen_runner, "run_codegen", return_value=err):
                tool, _ref = env_mod.make_run_codegen_tool(iter_dir=iter_dir, device="cpu")
                out = tool.invoke({})

        self.assertNotIn("API HINTS", out)


if __name__ == "__main__":
    unittest.main()
