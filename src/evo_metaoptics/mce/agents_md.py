"""AGENTS.md format adapter for Pi harness.

Converts existing MCE prompt content into AGENTS.md files that Pi loads
as system context when cwd is the iteration directory.

Skills are written to .agents/skills/ (loaded by Pi via --skill <path>).
"""

from __future__ import annotations

import re
from pathlib import Path

# Canonical skill filename reused from skills module
AGENTS_MD_FILENAME = "AGENTS.md"
_SKILL_FILENAME = "SKILL.md"
_SKILLS_DIR = ".agents/skills"
_DEFAULT_SKILL_GUIDANCE = (
    "Read the learning-context skill package in `.agents/skills/learning-context/`: "
    "start with `SKILL.md`, then consult `.agents/skills/learning-context/reference/` "
    "for supporting examples and API notes."
)

# Regex to extract name from YAML frontmatter
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"(?m)^name:\s*(.+?)\s*$")


def write_agents_md(
    iter_dir: Path,
    system_prompt: str,
    task_prompt: str | None = None,
    context_available: bool = False,
    skill_history: str | None = None,
    skill_guidance: str | None = _DEFAULT_SKILL_GUIDANCE,
) -> Path:
    """Write an AGENTS.md file to iter_dir for Pi system context.

    The generated AGENTS.md is what Pi loads when cwd is iter_dir.
    It wraps the system prompt and optional task/context instructions
    in a markdown document that does NOT duplicate repo root AGENTS.md.

    Args:
        iter_dir: Iteration directory (e.g. workspace/iter1_sub0).
        system_prompt: Full system prompt content for the agent.
        task_prompt: Optional task-specific query/instructions.
        context_available: If True, include instruction to read context/.
            If False (cold start), omit entirely.
        skill_history: Optional skill database summary (iteration metrics
            + skill overviews) to embed directly. Used for meta-agent.
        skill_guidance: Optional guidance about on-disk skill packages.
            Pass None when the session does not load a native skill package.

    Returns:
        Path to the written AGENTS.md file.
    """
    iter_dir = Path(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []

    # --- Header ---
    sections.append("# Agent Instructions")
    sections.append("")

    # --- System prompt (core content) ---
    sections.append("## System Prompt")
    sections.append("")
    sections.append(system_prompt.rstrip())
    sections.append("")

    # --- Task prompt (optional) ---
    if task_prompt is not None:
        sections.append("## Task")
        sections.append("")
        sections.append(task_prompt.rstrip())
        sections.append("")

    # --- Skill loading instruction ---
    if skill_guidance is not None:
        sections.append("## Skill Guidance")
        sections.append("")
        sections.append(skill_guidance.rstrip())
        sections.append("")

    # --- Context availability (conditional) ---
    if context_available:
        sections.append("## Prior Context")
        sections.append("")
        sections.append(
            "Read the `context/` directory for prior learned context artifacts. "
            "Build upon existing knowledge rather than starting from scratch."
        )
        sections.append("")

    # --- Skill history / database (meta-agent) ---
    if skill_history is not None:
        sections.append("## Skill Database (Iteration History)")
        sections.append("")
        sections.append(skill_history.rstrip())
        sections.append("")

    content = "\n".join(sections)
    agents_md_path = iter_dir / AGENTS_MD_FILENAME
    agents_md_path.write_text(content, encoding="utf-8")
    return agents_md_path


def write_skills_to_workspace(
    iter_dir: Path,
    skill_sources: list[str],
) -> list[Path]:
    """Write skill markdown sources to .agents/skills/ directory.

    Each skill source is a markdown string (with optional YAML frontmatter).
    The skill name is extracted from frontmatter ``name:`` field; if absent,
    a fallback name ``skill-{index}`` is used.

    Skills are written as SKILL.md files in subdirectories named after the
    skill: ``iter_dir/.agents/skills/<skill-name>/SKILL.md``.

    Args:
        iter_dir: Iteration directory.
        skill_sources: List of skill markdown content strings.

    Returns:
        List of paths to written skill files.
    """
    if not skill_sources:
        return []

    iter_dir = Path(iter_dir)
    skills_base = iter_dir / _SKILLS_DIR
    written: list[Path] = []

    for idx, source in enumerate(skill_sources):
        name = _extract_skill_name(source)
        if name is None:
            name = f"skill-{idx}"

        skill_dir = skills_base / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / _SKILL_FILENAME
        skill_path.write_text(source, encoding="utf-8")
        written.append(skill_path)

    return written


def _extract_skill_name(content: str) -> str | None:
    """Extract skill name from YAML frontmatter ``name:`` field.

    Returns None if no frontmatter or no name field found.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None
    frontmatter = match.group(1)
    name_match = _NAME_RE.search(frontmatter)
    if not name_match:
        return None
    name = name_match.group(1).strip()
    # Strip surrounding quotes
    if (name.startswith('"') and name.endswith('"')) or (
        name.startswith("'") and name.endswith("'")
    ):
        name = name[1:-1].strip()
    return name or None
