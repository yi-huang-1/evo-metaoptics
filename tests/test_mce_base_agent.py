"""Tests for base_agent Pi session migration.

Validates:
- Zero framework imports (no LangGraph, no LangChain, no Deep Agents patterns)
- Pi session creation with correct skill_paths and cwd
- Validation/retry loop preserved
- Context validation preserved
- Interface validation preserved
"""
from __future__ import annotations

import asyncio
import ast
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from evo_metaoptics.mce.agent_session import AgentResponse


# ---------------------------------------------------------------------------
# 1. Zero-framework-import gate
# ---------------------------------------------------------------------------
class TestNoFrameworkImports(unittest.TestCase):
    """Verify base_agent.py has zero LangGraph/LangChain/Deep-Agents imports."""

    def _get_source(self) -> str:
        src = Path(__file__).resolve().parent.parent / "src" / "evo_metaoptics" / "mce" / "base_agent.py"
        return src.read_text(encoding="utf-8")

    def test_no_langgraph_imports(self):
        source = self._get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = ".".join(alias.name for alias in node.names)
                self.assertNotIn("langgraph", module.lower(),
                                 f"Found langgraph import: {module}")

    def test_no_langchain_imports(self):
        source = self._get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = ".".join(alias.name for alias in node.names)
                self.assertNotIn("langchain", module.lower(),
                                 f"Found langchain import: {module}")

    def test_no_deep_agents_symbols(self):
        source = self._get_source()
        banned = [
            "close_mce_deep_agent",
            "MCE_FORCE_TOOL_CHOICE_MARKER",
            "_flatten_message_content",
            "build_iter_path_sanitizer_middleware",
            "GraphRecursionError",
        ]
        for symbol in banned:
            self.assertNotIn(symbol, source,
                             f"Found banned symbol: {symbol}")

    def test_no_agent_tools_import(self):
        """Pi agent writes files via its own write tool; no custom tool builders needed."""
        source = self._get_source()
        self.assertNotIn("build_write_context_file_tool", source)
        self.assertNotIn("build_write_primary_context_tool", source)

    def test_uses_pi_session_imports(self):
        """Must import create_pi_session and invoke_pi_session (or equivalent)."""
        source = self._get_source()
        self.assertIn("create_pi_session", source,
                       "Missing create_pi_session import")


# ---------------------------------------------------------------------------
# 2. Validation logic preserved
# ---------------------------------------------------------------------------
class TestValidationLogicPreserved(unittest.TestCase):
    """Ensure validation helpers are still importable and correct."""

    def test_validate_context_outputs_importable(self):
        from evo_metaoptics.mce.base_agent import _validate_context_outputs
        self.assertTrue(callable(_validate_context_outputs))

    def test_collect_interface_syntax_errors_importable(self):
        from evo_metaoptics.mce.base_agent import _collect_interface_syntax_errors
        self.assertTrue(callable(_collect_interface_syntax_errors))

    def test_validate_interfaces_used(self):
        """validate_interfaces from validation.py must still be imported."""
        source = Path(__file__).resolve().parent.parent / "src" / "evo_metaoptics" / "mce" / "base_agent.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("validate_interfaces", text)

    def test_format_validation_feedback_used(self):
        source = Path(__file__).resolve().parent.parent / "src" / "evo_metaoptics" / "mce" / "base_agent.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("format_validation_feedback", text)


# ---------------------------------------------------------------------------
# 3. Pi session integration
# ---------------------------------------------------------------------------
def _make_mock_session(responses: list[AgentResponse]) -> MagicMock:
    """Create a mock AgentSession that returns responses in sequence."""
    session = MagicMock()
    session.cwd = Path("/tmp/test_iter")
    call_idx = {"i": 0}

    async def _send(prompt: str) -> AgentResponse:
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(responses):
            return responses[idx]
        return responses[-1]

    session.send_message = _send
    session.close = AsyncMock()
    return session


class TestRunBaseAgentPiSession(unittest.TestCase):
    """Test run_base_agent creates and uses Pi sessions correctly."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.iter_dir = Path(self._tmpdir) / "iter1_sub0"
        self.iter_dir.mkdir(parents=True)
        # Create required directories
        (self.iter_dir / "data").mkdir()
        (self.iter_dir / "data" / "train.json").write_text(
            json.dumps({"summary": {}, "detailed_results": []}),
            encoding="utf-8",
        )
        # Skill dir
        skill_dir = self.iter_dir / ".agents" / "skills" / "learning-context"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: learning-context\n---\n# Skill\n## Skill Overview\nTest", encoding="utf-8")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _create_context_files(self):
        """Helper: write valid context files so validation passes."""
        ctx = self.iter_dir / "context"
        ctx.mkdir(exist_ok=True)
        (ctx / "rules.txt").write_text(
            "Rule 1: If solver timeout, reduce steps.\n" * 10,
            encoding="utf-8",
        )

    def _create_interface_files(self, name="extract_facts_ir"):
        """Helper: write valid interface implementation."""
        ifaces = self.iter_dir / "interfaces"
        ifaces.mkdir(exist_ok=True)
        (ifaces / "__init__.py").write_text(
            f"from .{name} import {name}\n__all__ = ['{name}']\n",
            encoding="utf-8",
        )
        (ifaces / f"{name}.py").write_text(
            f"def {name}(query: str) -> str:\n    return query\n",
            encoding="utf-8",
        )

    @patch("evo_metaoptics.mce.base_agent.create_pi_session")
    def test_context_only_success(self, mock_create):
        """Context-only run (no interfaces) succeeds when context is valid."""
        from evo_metaoptics.mce.base_agent import run_base_agent

        iter_dir = self.iter_dir

        async def _send(prompt: str) -> AgentResponse:
            # Simulate Pi agent writing context files to disk
            ctx = iter_dir / "context"
            ctx.mkdir(exist_ok=True)
            (ctx / "rules.txt").write_text(
                "Rule 1: If solver timeout, reduce steps.\n" * 10,
                encoding="utf-8",
            )
            return AgentResponse(content="Wrote context/rules.txt")

        mock_session = MagicMock()
        mock_session.cwd = iter_dir
        mock_session.send_message = _send
        mock_session.close = AsyncMock()
        mock_create.return_value = mock_session

        result = asyncio.run(run_base_agent(
            iter_dir=self.iter_dir,
            task_instruction="Learn from training data",
            interface_signatures=[],
            model="test-model",
            required_context_files=["rules.txt"],
            context_min_chars=30,
        ))
        self.assertTrue(result["success"])
        mock_session.close.assert_awaited()

    @patch("evo_metaoptics.mce.base_agent.create_pi_session")
    def test_context_retry_on_validation_failure(self, mock_create):
        """If context validation fails, retry with feedback prompt."""
        from evo_metaoptics.mce.base_agent import run_base_agent

        call_count = {"n": 0}

        async def _send(prompt: str) -> AgentResponse:
            call_count["n"] += 1
            if call_count["n"] >= 2:
                # On retry, create valid context
                ctx = self.iter_dir / "context"
                ctx.mkdir(exist_ok=True)
                (ctx / "rules.txt").write_text("Rule 1: fix things properly " * 5, encoding="utf-8")
            return AgentResponse(content="Done")

        mock_session = MagicMock()
        mock_session.cwd = self.iter_dir
        mock_session.send_message = _send
        mock_session.close = AsyncMock()
        mock_create.return_value = mock_session

        result = asyncio.run(run_base_agent(
            iter_dir=self.iter_dir,
            task_instruction="Learn from training data",
            interface_signatures=[],
            model="test-model",
            required_context_files=["rules.txt"],
            context_min_chars=30,
            max_validation_attempts=3,
        ))
        self.assertTrue(result["success"])
        self.assertGreater(call_count["n"], 1, "Should have retried at least once")

    @patch("evo_metaoptics.mce.base_agent.create_pi_session")
    def test_interface_validation_success(self, mock_create):
        """Interface validation succeeds when interfaces are implemented."""
        from evo_metaoptics.mce.base_agent import run_base_agent
        from evo_metaoptics.mce_env.base import InterfaceSignature

        iter_dir = self.iter_dir

        async def _send(prompt: str) -> AgentResponse:
            # Simulate Pi agent writing context + interfaces to disk
            ctx = iter_dir / "context"
            ctx.mkdir(exist_ok=True)
            (ctx / "rules.txt").write_text(
                "Rule 1: If solver timeout, reduce steps.\n" * 10,
                encoding="utf-8",
            )
            ifaces = iter_dir / "interfaces"
            ifaces.mkdir(exist_ok=True)
            (ifaces / "__init__.py").write_text(
                "from .extract_facts_ir import extract_facts_ir\n__all__ = ['extract_facts_ir']\n",
                encoding="utf-8",
            )
            (ifaces / "extract_facts_ir.py").write_text(
                "def extract_facts_ir(query: str) -> str:\n    return query\n",
                encoding="utf-8",
            )
            return AgentResponse(content="Done")

        mock_session = MagicMock()
        mock_session.cwd = iter_dir
        mock_session.send_message = _send
        mock_session.close = AsyncMock()
        mock_create.return_value = mock_session

        sig = InterfaceSignature(
            name="extract_facts_ir",
            inputs=[("query", "str", "The input query")],
            output=("str", "Extracted facts"),
            description="Extract facts from query",
        )
        result = asyncio.run(run_base_agent(
            iter_dir=self.iter_dir,
            task_instruction="Learn from training data",
            interface_signatures=[sig],
            model="test-model",
            required_context_files=["rules.txt"],
            context_min_chars=30,
        ))
        self.assertTrue(result["success"])

    @patch("evo_metaoptics.mce.base_agent.create_pi_session")
    def test_pi_session_created_with_skill_paths(self, mock_create):
        """create_pi_session called with correct skill_paths for learning-context."""
        from evo_metaoptics.mce.base_agent import run_base_agent

        iter_dir = self.iter_dir

        async def _send(prompt: str) -> AgentResponse:
            ctx = iter_dir / "context"
            ctx.mkdir(exist_ok=True)
            (ctx / "rules.txt").write_text("Rule: reduce steps on timeout\n" * 5, encoding="utf-8")
            return AgentResponse(content="Done")

        mock_session = MagicMock()
        mock_session.cwd = iter_dir
        mock_session.send_message = _send
        mock_session.close = AsyncMock()
        mock_create.return_value = mock_session

        asyncio.run(run_base_agent(
            iter_dir=self.iter_dir,
            task_instruction="Learn",
            interface_signatures=[],
            model="test-model",
            required_context_files=["rules.txt"],
            context_min_chars=30,
        ))

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        skill_paths = call_kwargs.kwargs.get("skill_paths", [])
        self.assertTrue(
            any("learning-context" in str(p) for p in skill_paths),
            f"skill_paths should include learning-context dir, got: {skill_paths}"
        )

    @patch("evo_metaoptics.mce.base_agent.create_pi_session")
    def test_session_closed_on_exception(self, mock_create):
        """Pi session is closed even when an exception occurs."""
        from evo_metaoptics.mce.base_agent import run_base_agent

        async def _send_raise(prompt: str) -> AgentResponse:
            raise RuntimeError("Agent exploded")

        mock_session = MagicMock()
        mock_session.cwd = self.iter_dir
        mock_session.send_message = _send_raise
        mock_session.close = AsyncMock()
        mock_create.return_value = mock_session

        result = asyncio.run(run_base_agent(
            iter_dir=self.iter_dir,
            task_instruction="Learn",
            interface_signatures=[],
            model="test-model",
        ))
        self.assertFalse(result["success"])
        mock_session.close.assert_awaited()

    @patch("evo_metaoptics.mce.base_agent.create_pi_session")
    def test_max_attempts_exhausted_returns_failure(self, mock_create):
        """When all validation attempts fail, returns failure result."""
        from evo_metaoptics.mce.base_agent import run_base_agent

        mock_session = _make_mock_session([
            AgentResponse(content="Did nothing useful"),
        ])
        mock_create.return_value = mock_session

        result = asyncio.run(run_base_agent(
            iter_dir=self.iter_dir,
            task_instruction="Learn",
            interface_signatures=[],
            model="test-model",
            required_context_files=["rules.txt"],
            context_min_chars=30,
            max_validation_attempts=2,
        ))
        self.assertFalse(result["success"])
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# 4. Context validation helpers (preserved from original)
# ---------------------------------------------------------------------------
class TestContextValidation(unittest.TestCase):
    """Test _validate_context_outputs still works correctly."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.iter_dir = Path(self._tmpdir) / "iter1_sub0"
        self.iter_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_missing_context_dir(self):
        from evo_metaoptics.mce.base_agent import _validate_context_outputs
        ok, errors, _ = _validate_context_outputs(
            self.iter_dir, required_context_files=[], min_context_chars=30
        )
        self.assertFalse(ok)

    def test_empty_context_dir(self):
        from evo_metaoptics.mce.base_agent import _validate_context_outputs
        (self.iter_dir / "context").mkdir()
        ok, errors, _ = _validate_context_outputs(
            self.iter_dir, required_context_files=[], min_context_chars=30
        )
        self.assertFalse(ok)

    def test_valid_context(self):
        from evo_metaoptics.mce.base_agent import _validate_context_outputs
        ctx = self.iter_dir / "context"
        ctx.mkdir()
        (ctx / "rules.txt").write_text("Some meaningful content here " * 10, encoding="utf-8")
        ok, errors, _ = _validate_context_outputs(
            self.iter_dir, required_context_files=["rules.txt"], min_context_chars=30
        )
        self.assertTrue(ok, f"Expected valid, got errors: {errors}")

    def test_required_file_too_short(self):
        from evo_metaoptics.mce.base_agent import _validate_context_outputs
        ctx = self.iter_dir / "context"
        ctx.mkdir()
        (ctx / "rules.txt").write_text("x", encoding="utf-8")
        ok, errors, _ = _validate_context_outputs(
            self.iter_dir, required_context_files=["rules.txt"], min_context_chars=30
        )
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 5. Interface syntax error collection (preserved)
# ---------------------------------------------------------------------------
class TestCollectInterfaceSyntaxErrors(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.iter_dir = Path(self._tmpdir) / "iter1_sub0"
        self.iter_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_no_interfaces_dir(self):
        from evo_metaoptics.mce.base_agent import _collect_interface_syntax_errors
        errors = _collect_interface_syntax_errors(self.iter_dir)
        self.assertEqual(errors, [])

    def test_valid_python(self):
        from evo_metaoptics.mce.base_agent import _collect_interface_syntax_errors
        ifaces = self.iter_dir / "interfaces"
        ifaces.mkdir()
        (ifaces / "foo.py").write_text("def foo(): return 1\n", encoding="utf-8")
        errors = _collect_interface_syntax_errors(self.iter_dir)
        self.assertEqual(errors, [])

    def test_syntax_error_detected(self):
        from evo_metaoptics.mce.base_agent import _collect_interface_syntax_errors
        ifaces = self.iter_dir / "interfaces"
        ifaces.mkdir()
        (ifaces / "bad.py").write_text("def bad( return 1\n", encoding="utf-8")
        errors = _collect_interface_syntax_errors(self.iter_dir)
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
