"""Tests for AGENTS.md format adapter (agents_md.py).

Covers:
- write_agents_md: AGENTS.md generation with correct content structure
- write_skills_to_workspace: skill file placement in .agents/skills/
- Context-available conditional: instruction present/absent
- Skill loading instruction inclusion
- No duplication of repo root AGENTS.md info
"""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

_agents_md = importlib.import_module("evo_metaoptics.mce.agents_md")
write_agents_md = _agents_md.write_agents_md
write_skills_to_workspace = _agents_md.write_skills_to_workspace


class TestWriteAgentsMd(unittest.TestCase):
    """Tests for write_agents_md function."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.iter_dir = Path(self._tmpdir.name) / "iter1_sub0"
        self.iter_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # --- Core file creation ---

    def test_creates_agents_md_file(self) -> None:
        """write_agents_md creates AGENTS.md in iter_dir."""
        path = write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="You are a test agent.",
        )
        self.assertEqual(path, self.iter_dir / "AGENTS.md")
        self.assertTrue(path.exists())

    def test_returns_path_to_agents_md(self) -> None:
        """Return value is the path to the written AGENTS.md."""
        result = write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt here.",
        )
        self.assertIsInstance(result, Path)
        self.assertEqual(result.name, "AGENTS.md")

    # --- Content structure ---

    def test_contains_system_prompt(self) -> None:
        """AGENTS.md contains the provided system prompt."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="You are a code-generation agent for inverse design.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertIn("You are a code-generation agent for inverse design.", content)

    def test_contains_task_prompt_when_provided(self) -> None:
        """AGENTS.md contains task prompt when provided."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
            task_prompt="Design a metasurface that achieves 90% transmission.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertIn(
            "Design a metasurface that achieves 90% transmission.", content
        )

    def test_no_task_section_when_task_prompt_none(self) -> None:
        """AGENTS.md omits task section when task_prompt is None."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
            task_prompt=None,
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        # Should not contain a "## Task" heading when no task is given
        self.assertNotIn("## Task\n", content)

    # --- Context-available conditional ---

    def test_context_available_true_includes_read_instruction(self) -> None:
        """When context_available=True, AGENTS.md instructs to read context/ directory."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
            context_available=True,
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        # Should have the Prior Context section with read instruction
        self.assertIn("## Prior Context", content)
        self.assertIn("Read the `context/` directory", content)

    def test_context_available_false_omits_context_instruction(self) -> None:
        """When context_available=False, AGENTS.md does NOT mention context/."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
            context_available=False,
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        # Must NOT contain Prior Context section
        self.assertNotIn("## Prior Context", content)
        self.assertNotIn("Read the `context/` directory", content)

    def test_context_available_defaults_to_false(self) -> None:
        """Default context_available=False means no context instruction."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertNotIn("## Prior Context", content)
        self.assertNotIn("Read the `context/` directory", content)

    # --- Skill loading instruction ---

    def test_includes_skill_loading_instruction(self) -> None:
        """AGENTS.md includes instruction to read SKILL.md."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertIn(
            "Read the learning-context skill package in `.agents/skills/learning-context/`: start with `SKILL.md`, then consult `.agents/skills/learning-context/reference/` for supporting examples and API notes.",
            content,
        )

    def test_skill_guidance_can_be_omitted(self) -> None:
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
            skill_guidance=None,
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertNotIn("## Skill Guidance", content)
        self.assertNotIn("learning-context skill package", content)

    # --- Walk-up awareness ---

    def test_does_not_duplicate_repo_root_agents_md(self) -> None:
        """Per-iteration AGENTS.md should NOT contain repo-root-level content markers."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        # Should NOT contain repo-level sections like "Project Structure"
        self.assertNotIn("## Project Structure", content)
        self.assertNotIn("## Build, Test, and Development Commands", content)

    # --- Creates parent directories ---

    def test_creates_iter_dir_if_missing(self) -> None:
        """write_agents_md creates iter_dir if it doesn't exist."""
        new_iter_dir = Path(self._tmpdir.name) / "new_iter" / "sub"
        path = write_agents_md(
            iter_dir=new_iter_dir,
            system_prompt="System prompt.",
        )
        self.assertTrue(path.exists())

    # --- Markdown structure ---

    def test_agents_md_starts_with_heading(self) -> None:
        """Generated AGENTS.md starts with a markdown heading."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertTrue(content.startswith("#"), f"Content starts with: {content[:50]!r}")

    # --- Meta-agent skill history embedding ---

    def test_skill_history_embedded_when_provided(self) -> None:
        """When skill_history is provided, it appears in AGENTS.md."""
        history = "### Iteration 1\n- Train: 50% | Val: 45%\n- Skill: Direct curation"
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="You are a meta-level agent.",
            skill_history=history,
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertIn("### Iteration 1", content)
        self.assertIn("Train: 50%", content)

    def test_skill_history_absent_by_default(self) -> None:
        """Without skill_history, no iteration history section appears."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="System prompt.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertNotIn("## Skill Database", content)

    def test_agents_md_contains_new_device_parameter_signature(self) -> None:
        """RED TEST: generated AGENTS.md must document new device-explicit signature."""
        system_prompt = (
            "You are a code-generation agent for inverse design.\n\n"
            "Function signature: def solve_inverse_design(*, device: str = \"cuda\") -> SolverResults"
        )
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt=system_prompt,
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertIn("def solve_inverse_design(*, device: str = \"cuda\") -> SolverResults", content)

    def test_agents_md_no_device_agnostic_guidance(self) -> None:
        """RED TEST: generated AGENTS.md must NOT contain device-agnostic guidance."""
        write_agents_md(
            iter_dir=self.iter_dir,
            system_prompt="You are a code-generation agent for inverse design.",
        )
        content = (self.iter_dir / "AGENTS.md").read_text()
        self.assertNotIn("device-agnostic", content)
        self.assertNotIn("keep solver setup device-agnostic", content)
        self.assertNotIn("Do not set the compute device manually", content)


class TestWriteSkillsToWorkspace(unittest.TestCase):
    """Tests for write_skills_to_workspace function."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.iter_dir = Path(self._tmpdir.name) / "iter1_sub0"
        self.iter_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_writes_single_skill(self) -> None:
        """Single skill source is written to .agents/skills/ directory."""
        skill_content = "---\nname: learning-context\ndescription: Test skill\n---\n## Skill Overview\nTest."
        paths = write_skills_to_workspace(
            iter_dir=self.iter_dir,
            skill_sources=[skill_content],
        )
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].exists())
        self.assertEqual(paths[0].read_text(), skill_content)

    def test_skill_placed_in_agents_skills_dir(self) -> None:
        """Skill file is placed under .agents/skills/ directory structure."""
        skill_content = "---\nname: learning-context\ndescription: Test\n---\n## Skill Overview\nTest."
        paths = write_skills_to_workspace(
            iter_dir=self.iter_dir,
            skill_sources=[skill_content],
        )
        # Should be under .agents/skills/learning-context/SKILL.md
        rel = paths[0].relative_to(self.iter_dir)
        self.assertEqual(
            str(rel),
            ".agents/skills/learning-context/SKILL.md",
        )

    def test_writes_multiple_skills(self) -> None:
        """Multiple skill sources produce multiple files."""
        skills = [
            "---\nname: skill-a\ndescription: A\n---\n## Skill Overview\nA.",
            "---\nname: skill-b\ndescription: B\n---\n## Skill Overview\nB.",
        ]
        paths = write_skills_to_workspace(
            iter_dir=self.iter_dir,
            skill_sources=skills,
        )
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(p.exists())

    def test_creates_directory_structure(self) -> None:
        """Creates .agents/skills/ directory tree if it doesn't exist."""
        skill_content = "---\nname: learning-context\ndescription: Test\n---\n## Skill Overview\nTest."
        paths = write_skills_to_workspace(
            iter_dir=self.iter_dir,
            skill_sources=[skill_content],
        )
        self.assertTrue((self.iter_dir / ".agents" / "skills").is_dir())

    def test_empty_skill_sources_returns_empty(self) -> None:
        """Empty skill_sources list returns empty list."""
        paths = write_skills_to_workspace(
            iter_dir=self.iter_dir,
            skill_sources=[],
        )
        self.assertEqual(paths, [])

    def test_skill_name_extracted_for_directory(self) -> None:
        """Skill directory name comes from the frontmatter name field."""
        skill_content = "---\nname: my-custom-skill\ndescription: Custom\n---\n## Skill Overview\nCustom."
        paths = write_skills_to_workspace(
            iter_dir=self.iter_dir,
            skill_sources=[skill_content],
        )
        rel = paths[0].relative_to(self.iter_dir)
        self.assertEqual(
            str(rel),
            ".agents/skills/my-custom-skill/SKILL.md",
        )

    def test_skill_without_name_uses_fallback(self) -> None:
        """Skill without frontmatter name uses fallback directory name."""
        skill_content = "## Skill Overview\nPlain skill content."
        paths = write_skills_to_workspace(
            iter_dir=self.iter_dir,
            skill_sources=[skill_content],
        )
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].exists())
        # Should use fallback name
        rel = paths[0].relative_to(self.iter_dir)
        self.assertIn(".agents/skills/", str(rel))


if __name__ == "__main__":
    unittest.main()
