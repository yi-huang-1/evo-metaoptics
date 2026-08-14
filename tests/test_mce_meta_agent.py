"""Tests for Pi-based meta_agent implementation.

Covers:
- No LangGraph/LangChain imports in meta_agent.py
- Preserved validation/degeneracy helpers
- run_meta_agent with Pi sessions (mocked)
- Retry loop behavior
- AGENTS.md skill_history embedding
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_agent_session_module = importlib.import_module("evo_metaoptics.mce.agent_session")
_meta_agent_module = importlib.import_module("evo_metaoptics.mce.meta_agent")
_skills_module = importlib.import_module("evo_metaoptics.mce.skills")

AgentResponse = _agent_session_module.AgentResponse
_apply_structural_skill_antiregression_guard = _meta_agent_module._apply_structural_skill_antiregression_guard
_build_skill_validation_retry_prompt = _meta_agent_module._build_skill_validation_retry_prompt
_create_meta_pi_session = _meta_agent_module._create_meta_pi_session
_detect_structural_skill_degeneracy = _meta_agent_module._detect_structural_skill_degeneracy
_verify_meta_agent_outputs = _meta_agent_module._verify_meta_agent_outputs
run_meta_agent = _meta_agent_module.run_meta_agent
build_learning_context_skill_seed = _skills_module.build_learning_context_skill_seed


VALID_SKILL = """---
name: learning-context
description: Skill description.
---

## Skill Overview
Use rollout evidence to iteratively refine context updates and remove repeated error patterns.

## Methodology
1. Group failures by symptom overlap and contradictory cues.
2. Record disambiguation boundaries for frequently confused diagnoses.
3. Add concise negative evidence checks to reduce false positives.
4. Preserve high-performing rules and only replace weak or conflicting guidance.

## Skill Delta (This Iteration)

### ADD
- Add one compact validation-rule note.

### UPDATE
- Update one ambiguous guidance rule for clarity.

### REMOVE
- Remove one stale rule with no recent evidence.
"""


class _SkillWriterSession:
    """Mock AgentSession that writes SKILL.md to iter_dir on send_message."""

    def __init__(
        self,
        iter_dir: Path,
        skill_md: str | None = None,
        *,
        fail_first: int = 0,
        raise_on_attempt: int | None = None,
    ) -> None:
        self._iter_dir = iter_dir
        self._skill_md = skill_md
        self._fail_first = fail_first
        self._raise_on_attempt = raise_on_attempt
        self._call_count = 0
        self._prompts: list[str] = []
        self._cwd = iter_dir.parent

    @property
    def cwd(self) -> Path:
        return self._cwd

    async def send_message(self, prompt: str) -> AgentResponse:
        self._call_count += 1
        self._prompts.append(prompt)

        if self._raise_on_attempt is not None and self._call_count == self._raise_on_attempt:
            raise RuntimeError("Pi subprocess crashed")

        if self._call_count <= self._fail_first:
            # Simulate attempt without writing SKILL.md
            return AgentResponse(
                content="Working on it...",
                error=None,
            )

        if self._skill_md is not None:
            skill_path = (
                self._iter_dir / ".agents" / "skills" / "learning-context" / "SKILL.md"
            )
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(self._skill_md, encoding="utf-8")
            return AgentResponse(
                content="Done! Wrote SKILL.md",
                error=None,
            )

        return AgentResponse(
            content="Done",
            error=None,
        )

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 1. No framework imports
# ---------------------------------------------------------------------------


class TestNoFrameworkImports(unittest.TestCase):
    """meta_agent.py must not import LangGraph/LangChain patterns."""

    def _source(self) -> str:
        return inspect.getsource(_meta_agent_module)

    def test_no_langgraph_import(self) -> None:
        self.assertNotIn("langgraph", self._source())

    def test_no_langchain_import(self) -> None:
        self.assertNotIn("langchain", self._source())

    def test_no_in_memory_saver(self) -> None:
        self.assertNotIn("InMemorySaver", self._source())

    def test_no_create_mce_deep_agent(self) -> None:
        self.assertNotIn("create_mce_deep_agent", self._source())

    def test_no_invoke_mce_deep_agent(self) -> None:
        self.assertNotIn("invoke_mce_deep_agent", self._source())

    def test_no_close_mce_deep_agent(self) -> None:
        self.assertNotIn("close_mce_deep_agent", self._source())

    def test_no_flatten_message_content(self) -> None:
        self.assertNotIn("_flatten_message_content", self._source())

    def test_no_path_sanitizer_middleware(self) -> None:
        self.assertNotIn("build_workspace_path_sanitizer_middleware", self._source())


# ---------------------------------------------------------------------------
# 2. Preserved helper functions
# ---------------------------------------------------------------------------


class TestVerifyMetaAgentOutputs(unittest.TestCase):
    def test_valid_skill_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_path = iter_dir / ".agents" / "skills" / "learning-context" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(VALID_SKILL, encoding="utf-8")

            logger = MagicMock()
            result = _verify_meta_agent_outputs(iter_dir, logger)
            self.assertTrue(result["success"])
            self.assertIsNotNone(result["skill_md"])

    def test_valid_skill_with_reference_subtree_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_dir = iter_dir / ".agents" / "skills" / "learning-context"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
            reference_dir = skill_dir / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            (reference_dir / "setup.md").write_text("setup reference\n", encoding="utf-8")

            logger = MagicMock()
            result = _verify_meta_agent_outputs(iter_dir, logger)
            self.assertTrue(result["success"])

    def test_missing_skill_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            logger = MagicMock()
            result = _verify_meta_agent_outputs(iter_dir, logger)
            self.assertFalse(result["success"])
            self.assertIn("did not generate SKILL.md", result["error"])

    def test_unexpected_variant_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_dir = iter_dir / ".agents" / "skills" / "learning-context"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
            (skill_dir / "SKILL-evolved.md").write_text(VALID_SKILL, encoding="utf-8")

            logger = MagicMock()
            result = _verify_meta_agent_outputs(iter_dir, logger)
            self.assertFalse(result["success"])
            self.assertIn("Unexpected alternate skill file", result["error"])


class TestDetectStructuralSkillDegeneracy(unittest.TestCase):
    def test_empty_is_degenerate(self) -> None:
        is_deg, reason = _detect_structural_skill_degeneracy("")
        self.assertTrue(is_deg)
        self.assertEqual("skill_empty", reason)

    def test_seed_unchanged_is_degenerate(self) -> None:
        seed = build_learning_context_skill_seed()
        is_deg, reason = _detect_structural_skill_degeneracy(seed)
        self.assertTrue(is_deg)
        self.assertEqual("seed_scaffold_unchanged", reason)

    def test_seed_placeholder_retained_is_degenerate(self) -> None:
        text = """## Skill Overview
Draft this section with the core learning strategy.

## Methodology
1. Review prior context and training outcomes.
2. Identify recurring failure patterns.
3. Update context and interfaces with focused, testable improvements.
"""
        is_deg, reason = _detect_structural_skill_degeneracy(text)
        self.assertTrue(is_deg)
        self.assertEqual("seed_placeholder_retained", reason)

    def test_too_short_is_degenerate(self) -> None:
        is_deg, reason = _detect_structural_skill_degeneracy("Short.")
        self.assertTrue(is_deg)
        self.assertEqual("skill_too_short", reason)

    def test_valid_content_is_not_degenerate(self) -> None:
        is_deg, reason = _detect_structural_skill_degeneracy(VALID_SKILL)
        self.assertFalse(is_deg)
        self.assertEqual("", reason)


class TestApplySkillAntiregression(unittest.TestCase):
    def test_non_degenerate_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            skill_path = iter_dir / ".agents" / "skills" / "learning-context" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(VALID_SKILL, encoding="utf-8")

            result = _apply_structural_skill_antiregression_guard(
                iter_dir=iter_dir,
                workspace_base=workspace,
                iteration=1,
            )
            self.assertFalse(result["skill_guard_triggered"])
            self.assertEqual("accept_generated", result["guard_action"])

    def test_degenerate_triggers_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            skill_path = iter_dir / ".agents" / "skills" / "learning-context" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("Short.", encoding="utf-8")

            result = _apply_structural_skill_antiregression_guard(
                iter_dir=iter_dir,
                workspace_base=workspace,
                iteration=1,
            )
            self.assertTrue(result["skill_guard_triggered"])


class TestBuildSkillValidationRetryPrompt(unittest.TestCase):
    def test_uses_virtual_path(self) -> None:
        prompt = _build_skill_validation_retry_prompt(
            validation_error="Invalid SKILL.md",
            expected_virtual_path="/iter1_sub0/.agents/skills/learning-context/SKILL.md",
        )
        self.assertIn("/iter1_sub0/.agents/skills/learning-context/SKILL.md", prompt)
        self.assertIn("VALIDATION ERROR", prompt)


class TestCreateMetaPiSession(unittest.TestCase):
    def test_uses_shared_runtime_trace_plumbing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_base = Path(tmp) / "workspace"
            workspace_base.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tmp) / "run"
            fake_client = MagicMock()
            fake_session = MagicMock()

            with patch("evo_metaoptics.mce.meta_agent.write_agents_md") as write_agents_md, patch(
                "evo_metaoptics.mce.meta_agent.start_pi_session_client",
                return_value=fake_client,
            ) as start_client, patch(
                "evo_metaoptics.mce.meta_agent.wrap_pi_session_client_as_session",
                return_value=fake_session,
            ) as wrap_session:
                session = _create_meta_pi_session(
                    workspace_base=workspace_base,
                    model="model-A",
                    system_prompt="system",
                    skill_database="skill-db",
                    run_dir=run_dir,
                    iteration=7,
                )

        write_agents_md.assert_called_once_with(
            iter_dir=workspace_base,
            system_prompt="system",
            skill_guidance=None,
        )
        start_client.assert_called_once_with(
            cwd=workspace_base,
            model="model-A",
            skill_paths=[],
            run_dir=run_dir,
            timeout_s=None,
            session_traces_enabled=None,
        )
        wrap_session.assert_called_once_with(fake_client, cwd=workspace_base)
        self.assertIs(session, fake_session)


# ---------------------------------------------------------------------------
# 3. run_meta_agent with Pi sessions
# ---------------------------------------------------------------------------


class TestRunMetaAgentPiSession(unittest.TestCase):
    """Test run_meta_agent using mocked Pi session."""

    def _run(self, session: _SkillWriterSession, **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_base = kwargs.pop("workspace_base", Path(tmp) / "workspace")
            iter_dir = kwargs.pop("iter_dir", workspace_base / "iter1_sub0")
            iter_dir.mkdir(parents=True, exist_ok=True)

            with patch(
                "evo_metaoptics.mce.meta_agent._create_meta_pi_session",
                return_value=session,
            ):
                return asyncio.run(
                    run_meta_agent(
                        iter_dir=iter_dir,
                        task_instruction="Evolve learning skill",
                        interface_signatures=[],
                        iteration=kwargs.pop("iteration", 1),
                        workspace_base=workspace_base,
                        **kwargs,
                    )
                )

    def test_successful_skill_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            session = _SkillWriterSession(iter_dir, VALID_SKILL)

            result = self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
            )
            self.assertTrue(result["success"])
            self.assertIsNotNone(result["skill_md"])
            self.assertIn("skill_guard", result)
            self.assertIn("skill_bundle", result)

    def test_retry_on_missing_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            # First attempt: no SKILL.md written; second: SKILL.md written
            session = _SkillWriterSession(iter_dir, VALID_SKILL, fail_first=1)

            result = self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
                max_validation_attempts=3,
            )
            self.assertTrue(result["success"])
            self.assertEqual(2, session._call_count)
            # Second prompt should be the retry prompt
            self.assertIn("VALIDATION ERROR", session._prompts[1])

    def test_max_retries_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            # Never write SKILL.md
            session = _SkillWriterSession(iter_dir, skill_md=None)

            result = self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
                max_validation_attempts=2,
            )
            self.assertFalse(result["success"])
            self.assertIn("failed to generate SKILL.md", result["error"])
            self.assertEqual(2, session._call_count)

    def test_execution_error_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            session = _SkillWriterSession(
                iter_dir, VALID_SKILL, raise_on_attempt=1
            )

            result = self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
            )
            self.assertFalse(result["success"])
            self.assertIn("Pi subprocess crashed", result["error"])

    def test_attempt_timings_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            session = _SkillWriterSession(iter_dir, VALID_SKILL)

            result = self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
            )
            self.assertIn("attempt_timings", result)
            self.assertIn("attempt_timing_summary", result)
            self.assertGreaterEqual(len(result["attempt_timings"]), 1)
            self.assertTrue(result["attempt_timings"][0]["success"])

    def test_skill_bundle_sidecar_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            run_dir = Path(tmp) / "run"
            iter_dir = workspace / "iter1_sub0"
            session = _SkillWriterSession(iter_dir, VALID_SKILL)

            result = self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
                run_dir=run_dir,
            )
            self.assertTrue(result["success"])
            sidecar = run_dir / "skill_bundle_provenance.jsonl"
            self.assertTrue(sidecar.exists())
            rows = [
                json.loads(line)
                for line in sidecar.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual("meta", rows[-1].get("agent_type"))

    def test_session_closed_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            session = _SkillWriterSession(iter_dir, VALID_SKILL)
            close_called = []
            original_close = session.close

            async def tracking_close():
                close_called.append(True)
                await original_close()

            session.close = tracking_close

            self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
            )
            self.assertTrue(close_called)

    def test_session_closed_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            iter_dir = workspace / "iter1_sub0"
            session = _SkillWriterSession(
                iter_dir, VALID_SKILL, raise_on_attempt=1
            )
            close_called = []
            original_close = session.close

            async def tracking_close():
                close_called.append(True)
                await original_close()

            session.close = tracking_close

            self._run(
                session,
                workspace_base=workspace,
                iter_dir=iter_dir,
            )
            self.assertTrue(close_called)


# ---------------------------------------------------------------------------
# 4. Coding-agent strategy prompts and meta-agent review consumption
# ---------------------------------------------------------------------------


class TestCodingAgentStrategyPrompts(unittest.TestCase):
    """Tests for coding-agent strategy prompt contracts.
    
    These tests verify that:
    1. _SYSTEM_PROMPT requires a lightweight # Strategy: note in solution.py
    2. _build_initial_prompt explicitly tells agent to read context/strategy_summary.md
    3. _build_initial_prompt explicitly chooses a global optimization family
    4. _build_retry_prompt_for_score_failure encourages family switching only on stagnation
    """

    def test_system_prompt_requires_strategy_note(self) -> None:
        """SYSTEM_PROMPT must document that solution.py needs a # Strategy: note."""
        environment_module = importlib.import_module(
            "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
        )
        _SYSTEM_PROMPT = environment_module._SYSTEM_PROMPT

        # Contract: _SYSTEM_PROMPT should mention the # Strategy: note.
        self.assertIn(
            "# Strategy:",
            _SYSTEM_PROMPT,
            "SYSTEM_PROMPT must require a lightweight '# Strategy:' note in solution.py",
        )

    def test_build_initial_prompt_reads_strategy_summary(self) -> None:
        environment_module = importlib.import_module(
            "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
        )
        MetaopticsInverseDesignEnvironment = environment_module.MetaopticsInverseDesignEnvironment

        env = MetaopticsInverseDesignEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            prompt = env._build_initial_prompt(
                query="Design a metasurface for phase control",
                iter_dir=iter_dir,
            )

            # Contract: _build_initial_prompt should mention strategy_summary.md.
            self.assertIn(
                "context/strategy_summary.md",
                prompt,
                "_build_initial_prompt must explicitly tell agent to read context/strategy_summary.md when present",
            )
            self.assertIn(
                "cross-iteration memory",
                prompt,
                "_build_initial_prompt must explain that strategy_summary is cross-iteration memory",
            )

    def test_build_initial_prompt_chooses_global_family(self) -> None:
        """_build_initial_prompt must explicitly choose a global optimization family."""
        environment_module = importlib.import_module(
            "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
        )
        MetaopticsInverseDesignEnvironment = environment_module.MetaopticsInverseDesignEnvironment

        env = MetaopticsInverseDesignEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            prompt = env._build_initial_prompt(
                query="Design a metasurface for phase control",
                iter_dir=iter_dir,
            )

            # Contract: _build_initial_prompt should mention the optimization family choice.
            self.assertIn(
                "global",
                prompt.lower(),
                "_build_initial_prompt must explicitly choose a global optimization family (e.g., 'global screening', 'multistart')",
            )

    def test_build_retry_prompt_conditional_family_switching(self) -> None:
        """_build_retry_prompt_for_score_failure must encourage family switching only on stagnation."""
        environment_module = importlib.import_module(
            "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
        )
        MetaopticsInverseDesignEnvironment = environment_module.MetaopticsInverseDesignEnvironment

        env = MetaopticsInverseDesignEnvironment()

        prompt = env._build_retry_prompt_for_score_failure(
            query="Design a metasurface for phase control",
            attempt=2,
            max_attempts=5,
            criteria_rows=[
                {
                    "passed": False,
                    "margin": -0.05,
                    "violation": 0.05,
                    "operation": ">=",
                    "target": 0.8,
                    "value": 0.75,
                }
            ],
            criteria_pass_fraction=0.0,
            criteria_violation_norm=0.05,
            best_margin=-0.05,
        )

        # Contract: retry guidance should mention stagnation and family switching.
        self.assertIn(
            "stagnation",
            prompt.lower(),
            "_build_retry_prompt_for_score_failure must mention stagnation as trigger for family switching",
        )
        self.assertIn(
            "family",
            prompt.lower(),
            "_build_retry_prompt_for_score_failure must mention optimization family switching",
        )
        self.assertIn(
            "Previous evaluated inner attempt: 2/5",
            prompt,
            "_build_retry_prompt_for_score_failure must label the failed inner attempt clearly",
        )
        self.assertIn(
            "Next inner retry attempt: 3/5",
            prompt,
            "_build_retry_prompt_for_score_failure must label the upcoming inner retry clearly",
        )

    def test_build_retry_prompt_for_execution_error_labels_next_attempt(self) -> None:
        environment_module = importlib.import_module(
            "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
        )
        MetaopticsInverseDesignEnvironment = environment_module.MetaopticsInverseDesignEnvironment

        env = MetaopticsInverseDesignEnvironment()

        prompt = env._build_retry_prompt_for_execution_error(
            query="Design a metasurface for phase control",
            attempt=1,
            max_attempts=5,
            error_type="runtime_error",
            error_message="bad call",
            traceback_text="TB",
        )

        self.assertIn("Previous failed inner attempt: 1/5", prompt)
        self.assertIn("Next inner retry attempt: 2/5", prompt)

    def test_build_initial_prompt_labels_outer_round_feedback(self) -> None:
        environment_module = importlib.import_module(
            "evo_metaoptics.mce_env.metaoptics_inverse_design.metaoptics_inverse_design_environment"
        )
        MetaopticsInverseDesignEnvironment = environment_module.MetaopticsInverseDesignEnvironment

        env = MetaopticsInverseDesignEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0" / "round_1"
            iter_dir.mkdir(parents=True, exist_ok=True)

            prompt = env._build_initial_prompt(
                query="Design a metasurface for phase control",
                iter_dir=iter_dir,
                prev_feedback="Previous round scored criteria_pass_fraction=0.25.",
                round_num=1,
                total_rounds=5,
            )

        self.assertIn("Outer round: 2/5.", prompt)
        self.assertIn("distinct from inner retry attempts", prompt)


class TestMetaAgentPriorReviewConsumption(unittest.TestCase):
    """Tests for meta-agent consumption of prior iteration reviews.
    
    These tests verify that:
    1. Latest prior review is embedded in meta-agent prompt
    2. Missing prior review is handled gracefully
    3. Review content is truncated/bounded before embedding
    4. Only the latest prior review is included, not all historical reviews
    """

    def test_meta_agent_prompt_includes_latest_review(self) -> None:
        """Meta-agent prompt must include the latest prior iteration review."""
        build_meta_agent_prompt = importlib.import_module(
            "evo_metaoptics.mce.prompts.meta_agent"
        ).build_meta_agent_prompt

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            iter_dir = workspace / "iter3_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Create prior iteration reviews
            reviews_dir = workspace / "meta_agent" / "iteration_reviews"
            reviews_dir.mkdir(parents=True, exist_ok=True)
            (reviews_dir / "iter1.md").write_text(
                "# Iteration 1 Review\nFound pattern X in failures.\n",
                encoding="utf-8",
            )
            (reviews_dir / "iter2.md").write_text(
                "# Iteration 2 Review\nPattern X improved, found pattern Y.\n",
                encoding="utf-8",
            )

            # Create evaluations file for iteration 3
            meta_agent_dir = workspace / "meta_agent"
            meta_agent_dir.mkdir(parents=True, exist_ok=True)
            (meta_agent_dir / "evaluations.json").write_text(
                json.dumps({"iter1": {}, "iter2": {}}),
                encoding="utf-8",
            )

            prompt = build_meta_agent_prompt(
                task_instruction="Evolve skill",
                interface_signatures=[],
                iter_dir=str(iter_dir),
                workspace_base=str(workspace),
            )

            # Contract: build_meta_agent_prompt should embed prior reviews.
            self.assertIn(
                "iter2",
                prompt,
                "Meta-agent prompt must include the latest prior iteration review (iter2 for iteration 3)",
            )

    def test_meta_agent_prompt_handles_missing_review_gracefully(self) -> None:
        """Meta-agent prompt must handle missing prior review gracefully."""
        build_meta_agent_prompt = importlib.import_module(
            "evo_metaoptics.mce.prompts.meta_agent"
        ).build_meta_agent_prompt

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            iter_dir = workspace / "iter1_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # No reviews directory or reviews
            prompt = build_meta_agent_prompt(
                task_instruction="Evolve skill",
                interface_signatures=[],
                iter_dir=str(iter_dir),
                workspace_base=str(workspace),
            )

            # Contract: build_meta_agent_prompt should tolerate missing reviews.
            # The test passes if no exception is raised
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)

    def test_meta_agent_prompt_truncates_review_content(self) -> None:
        """Meta-agent prompt must truncate/bound review content before embedding."""
        build_meta_agent_prompt = importlib.import_module(
            "evo_metaoptics.mce.prompts.meta_agent"
        ).build_meta_agent_prompt

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            iter_dir = workspace / "iter2_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Create a very large review
            reviews_dir = workspace / "meta_agent" / "iteration_reviews"
            reviews_dir.mkdir(parents=True, exist_ok=True)
            large_review = "# Iteration 1 Review\n" + ("X" * 50000)
            (reviews_dir / "iter1.md").write_text(large_review, encoding="utf-8")

            # Create evaluations file for iteration 2
            meta_agent_dir = workspace / "meta_agent"
            meta_agent_dir.mkdir(parents=True, exist_ok=True)
            (meta_agent_dir / "evaluations.json").write_text(
                json.dumps({"iter1": {}}),
                encoding="utf-8",
            )

            prompt = build_meta_agent_prompt(
                task_instruction="Evolve skill",
                interface_signatures=[],
                iter_dir=str(iter_dir),
                workspace_base=str(workspace),
            )

            # Contract: build_meta_agent_prompt should bound review content.
            # The prompt should be reasonable in size (not 50KB+)
            self.assertLess(
                len(prompt),
                100000,
                "Meta-agent prompt must truncate/bound review content to prevent excessive prompt size",
            )

    def test_meta_agent_prompt_includes_only_latest_review(self) -> None:
        """Meta-agent prompt must include only the latest prior review, not all historical reviews."""
        build_meta_agent_prompt = importlib.import_module(
            "evo_metaoptics.mce.prompts.meta_agent"
        ).build_meta_agent_prompt

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            iter_dir = workspace / "iter4_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Create multiple prior iteration reviews
            reviews_dir = workspace / "meta_agent" / "iteration_reviews"
            reviews_dir.mkdir(parents=True, exist_ok=True)
            (reviews_dir / "iter1.md").write_text(
                "# Iteration 1 Review\nOld pattern A.\n",
                encoding="utf-8",
            )
            (reviews_dir / "iter2.md").write_text(
                "# Iteration 2 Review\nPattern B found.\n",
                encoding="utf-8",
            )
            (reviews_dir / "iter3.md").write_text(
                "# Iteration 3 Review\nLatest pattern C.\n",
                encoding="utf-8",
            )

            # Create evaluations file for iteration 4
            meta_agent_dir = workspace / "meta_agent"
            meta_agent_dir.mkdir(parents=True, exist_ok=True)
            (meta_agent_dir / "evaluations.json").write_text(
                json.dumps({"iter1": {}, "iter2": {}, "iter3": {}}),
                encoding="utf-8",
            )

            prompt = build_meta_agent_prompt(
                task_instruction="Evolve skill",
                interface_signatures=[],
                iter_dir=str(iter_dir),
                workspace_base=str(workspace),
            )

            # Contract: build_meta_agent_prompt should not include every review verbatim.
            # Count occurrences of "Iteration" to verify only latest is included
            iteration_count = prompt.count("Iteration")
            self.assertLessEqual(
                iteration_count,
                2,  # At most 2: one in task instruction, one in latest review
                "Meta-agent prompt must include only the latest prior review, not all historical reviews",
            )

    def test_meta_agent_prompt_includes_durable_strategy_summary(self) -> None:
        """Meta-agent prompt must include bounded durable lessons from strategy_summary.md."""
        build_meta_agent_prompt = importlib.import_module(
            "evo_metaoptics.mce.prompts.meta_agent"
        ).build_meta_agent_prompt

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            iter_dir = workspace / "iter2_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            meta_agent_dir = workspace / "meta_agent"
            meta_agent_dir.mkdir(parents=True, exist_ok=True)
            (meta_agent_dir / "evaluations.json").write_text(
                json.dumps({"iter1": {}}),
                encoding="utf-8",
            )
            strategy_summary = "# Durable Summary\n" + ("Lesson A\n" * 400)
            (meta_agent_dir / "strategy_summary.md").write_text(
                strategy_summary,
                encoding="utf-8",
            )

            prompt = build_meta_agent_prompt(
                task_instruction="Evolve skill",
                interface_signatures=[],
                iter_dir=str(iter_dir),
                workspace_base=str(workspace),
            )

            self.assertIn("## Durable Lessons", prompt)
            self.assertIn("meta_agent/strategy_summary.md", prompt)
            self.assertIn("cross-iteration memory", prompt)
            self.assertIn("[truncated]", prompt)
            self.assertLess(len(prompt), 100000)

    def test_meta_agent_prompt_handles_missing_strategy_summary_gracefully(self) -> None:
        """Meta-agent prompt must keep the durable-lessons section even when summary is missing."""
        build_meta_agent_prompt = importlib.import_module(
            "evo_metaoptics.mce.prompts.meta_agent"
        ).build_meta_agent_prompt

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            iter_dir = workspace / "iter1_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            prompt = build_meta_agent_prompt(
                task_instruction="Evolve skill",
                interface_signatures=[],
                iter_dir=str(iter_dir),
                workspace_base=str(workspace),
            )

            self.assertIn("## Durable Lessons", prompt)
            self.assertIn("Read cross-iteration memory from `meta_agent/strategy_summary.md` when present.", prompt)
            self.assertIn("(none)", prompt)


if __name__ == "__main__":
    unittest.main()
