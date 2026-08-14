"""Skill artifact validation and source-resolution helpers for MCE."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Tuple

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"(?m)^name:\s*(.+?)\s*$")
_DESCRIPTION_RE = re.compile(r"(?m)^description:\s*(.+?)\s*$")
_SKILL_OVERVIEW_RE = re.compile(r"(?im)^##\s*Skill\s+Overview\s*$")

SKILLS_ROOT_DIRNAME = ".agents"
SKILLS_DIRNAME = "skills"
LEARNING_CONTEXT_DIRNAME = "learning-context"
LEARNING_CONTEXT_REFERENCE_DIRNAME = "reference"
SKILL_FILENAME = "SKILL.md"


@dataclass(frozen=True)
class SkillBundle:
    selected_sources: list[str]
    excluded_sources: list[str]
    overlay_included: bool
    bundle_hash: str
    hash_algo: str = "sha256"
    bundle_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_hash": self.bundle_hash,
            "selected_sources": list(self.selected_sources),
            "excluded_sources": list(self.excluded_sources),
            "overlay_included": bool(self.overlay_included),
            "hash_algo": self.hash_algo,
            "bundle_version": self.bundle_version,
        }


def _iter_virtual_root(iter_folder_name: str) -> PurePosixPath:
    """Return virtual-root iteration path used by Deep Agents tools."""
    return PurePosixPath("/") / iter_folder_name


def learning_context_skill_relative_path() -> Path:
    """Return iteration-relative path for the canonical learning-context skill file."""
    return (
        Path(SKILLS_ROOT_DIRNAME)
        / SKILLS_DIRNAME
        / LEARNING_CONTEXT_DIRNAME
        / SKILL_FILENAME
    )


def learning_context_skill_relative_dir() -> Path:
    return Path(SKILLS_ROOT_DIRNAME) / SKILLS_DIRNAME / LEARNING_CONTEXT_DIRNAME


def learning_context_reference_relative_dir() -> Path:
    return learning_context_skill_relative_dir() / LEARNING_CONTEXT_REFERENCE_DIRNAME


def learning_context_skills_relative_dir() -> Path:
    """Return iteration-relative path for the canonical skills directory."""
    return Path(SKILLS_ROOT_DIRNAME) / SKILLS_DIRNAME


def learning_context_skill_host_path(iter_dir: Path) -> Path:
    """Return host path to canonical learning-context skill file for an iteration."""
    return Path(iter_dir) / learning_context_skill_relative_path()


def learning_context_skills_host_dir(iter_dir: Path) -> Path:
    """Return host path to canonical skills directory for an iteration."""
    return Path(iter_dir) / learning_context_skills_relative_dir()


def learning_context_skill_host_dir(iter_dir: Path) -> Path:
    return Path(iter_dir) / learning_context_skill_relative_dir()


def learning_context_reference_host_dir(iter_dir: Path) -> Path:
    return Path(iter_dir) / learning_context_reference_relative_dir()


def build_learning_context_skill_seed(
    *,
    description: str = "Scaffold for iterative context-learning skill evolution.",
) -> str:
    """Build a minimal valid learning-context skill document.

    The seed is intentionally simple: it guarantees valid frontmatter + required
    section so small models can safely edit content instead of creating format
    from scratch.
    """
    normalized_description = _normalize_yaml_scalar(description) or (
        "Scaffold for iterative context-learning skill evolution."
    )
    return f"""---
name: learning-context
description: {normalized_description}
---

## Skill Overview
Draft this section with the core learning strategy.

## Methodology
1. Review prior context and training outcomes.
2. Identify recurring failure patterns.
3. Update context and interfaces with focused, testable improvements.
"""


def ensure_learning_context_skill_seed(
    iter_dir: Path,
    *,
    overwrite: bool = False,
    description: str = "Scaffold for iterative context-learning skill evolution.",
) -> tuple[Path, bool]:
    """Ensure a valid scaffold file exists for learning-context skill editing.

    Returns:
        tuple[path, created_or_overwritten]
    """
    skill_file = learning_context_skill_host_path(iter_dir)
    if skill_file.exists() and not overwrite:
        return skill_file, False

    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(
        build_learning_context_skill_seed(description=description),
        encoding="utf-8",
    )
    return skill_file, True


def find_unexpected_skill_variants(iter_dir: Path) -> list[Path]:
    """Find alternate SKILL markdown files beside canonical SKILL.md.

    Examples considered unexpected: `SKILL-evolved.md`, `SKILL_backup.md`.
    """
    skill_dir = learning_context_skill_host_path(iter_dir).parent
    if not skill_dir.exists():
        return []

    variants: list[Path] = []
    for candidate in skill_dir.glob("SKILL*.md"):
        if candidate.name == SKILL_FILENAME:
            continue
        variants.append(candidate)
    return sorted(variants)


def learning_context_skill_virtual_path(iter_folder_name: str) -> str:
    """Return virtual path for learning-context SKILL.md in an iteration folder."""
    return str(
        _iter_virtual_root(iter_folder_name)
        / SKILLS_ROOT_DIRNAME
        / SKILLS_DIRNAME
        / LEARNING_CONTEXT_DIRNAME
        / SKILL_FILENAME
    )


def learning_context_skill_virtual_dir(iter_folder_name: str) -> str:
    """Return virtual path for the current iteration skills directory."""
    return str(_iter_virtual_root(iter_folder_name) / SKILLS_ROOT_DIRNAME / SKILLS_DIRNAME)


def backend_root_learning_context_skill_virtual_dir() -> str:
    """Return skill dir path when backend root is the active iteration directory."""
    return str(PurePosixPath("/") / SKILLS_ROOT_DIRNAME / SKILLS_DIRNAME)


def meta_agent_skills_virtual_dir() -> str:
    """Return virtual path to historical skill archive directory."""
    return "/meta_agent/skills"


def find_mirrored_host_skill_artifacts(
    *,
    workspace_base: Path,
    iter_folder_name: str,
) -> list[Path]:
    """Find mirrored host-absolute skill artifacts written under workspace root.

    In `FilesystemBackend(..., virtual_mode=True)`, host-absolute write attempts can
    appear under `${workspace_base}/Users/...`. These are invalid for normal MCE flow.
    """
    mirror_root = Path(workspace_base) / "Users"
    if not mirror_root.exists():
        return []

    expected_suffix = (
        Path(iter_folder_name)
        / SKILLS_ROOT_DIRNAME
        / SKILLS_DIRNAME
        / LEARNING_CONTEXT_DIRNAME
        / SKILL_FILENAME
    )
    suffix_parts = expected_suffix.parts

    artifacts: list[Path] = []
    for path in mirror_root.rglob("SKILL.md"):
        parts = path.parts
        if len(parts) >= len(suffix_parts) and parts[-len(suffix_parts) :] == suffix_parts:
            artifacts.append(path)

    return sorted(set(artifacts))


def _normalize_yaml_scalar(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1].strip()
    return value


_YAML_PLAIN_SAFE_RE = re.compile(r"^[A-Za-z0-9_.\-/ ]+$")


def _yaml_quote_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_emit_scalar(value: str) -> str:
    candidate = " ".join(value.strip().split())
    if not candidate:
        return '""'
    if (
        _YAML_PLAIN_SAFE_RE.fullmatch(candidate)
        and not candidate.startswith(("-", "?", ":", "{", "[", ",", "&", "*", "#", "!", "|", ">", "@", "`"))
        and ": " not in candidate
    ):
        return candidate
    return _yaml_quote_scalar(candidate)


def _derive_skill_description(content: str) -> str:
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = " ".join(line.split())
        if not line:
            continue
        if len(line) > 96:
            line = line[:96].rstrip()
        return line
    return "Evolved learning-context skill."


def normalize_learning_context_skill_markdown(
    content: str,
    *,
    expected_name: str = "learning-context",
) -> str:
    """Normalize SKILL markdown into canonical frontmatter form.

    This normalizer is intentionally narrow for hard-cut MCE flow:
    - Ensures top YAML frontmatter exists.
    - Ensures `name` and `description` are present and non-empty.
    - Preserves body text order and content.
    """
    if not isinstance(content, str):
        return ""

    raw = content
    if not raw.strip():
        return raw

    name_value = expected_name.strip() or "learning-context"
    description_value = _derive_skill_description(raw)

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        body = raw.lstrip()
        return (
            "---\n"
            f"name: {_yaml_emit_scalar(name_value)}\n"
            f"description: {_yaml_emit_scalar(description_value)}\n"
            "---\n\n"
            f"{body}"
        )

    frontmatter = match.group(1)
    remainder = raw[match.end() :].lstrip()

    name_match = _NAME_RE.search(frontmatter)
    if name_match and _normalize_yaml_scalar(name_match.group(1)):
        updated_frontmatter = _NAME_RE.sub(
            f"name: {_yaml_emit_scalar(name_value)}",
            frontmatter,
            count=1,
        )
    else:
        updated_frontmatter = (
            f"name: {_yaml_emit_scalar(name_value)}\n{frontmatter}"
        ).strip()

    desc_match = _DESCRIPTION_RE.search(updated_frontmatter)
    if desc_match and _normalize_yaml_scalar(desc_match.group(1)):
        return f"---\n{updated_frontmatter}\n---\n\n{remainder}"

    updated_frontmatter = (
        f"description: {_yaml_emit_scalar(description_value)}\n{updated_frontmatter}"
    ).strip()
    return f"---\n{updated_frontmatter}\n---\n\n{remainder}"

def validate_skill_markdown(
    content: str,
    *,
    expected_name: str | None = None,
    require_skill_overview: bool = True,
) -> Tuple[bool, str]:
    """Validate SKILL.md against Agent Skills + MCE minimum contract."""
    if not content.strip():
        return False, "SKILL.md is empty."

    match = _FRONTMATTER_RE.match(content)
    if not match:
        return False, "SKILL.md must begin with YAML frontmatter."

    frontmatter = match.group(1)

    name_match = _NAME_RE.search(frontmatter)
    if not name_match:
        return False, "SKILL.md frontmatter missing required field: name."
    name = _normalize_yaml_scalar(name_match.group(1))
    if not name:
        return False, "SKILL.md frontmatter field 'name' cannot be empty."

    desc_match = _DESCRIPTION_RE.search(frontmatter)
    if not desc_match:
        return False, "SKILL.md frontmatter missing required field: description."
    description = _normalize_yaml_scalar(desc_match.group(1))
    if not description:
        return False, "SKILL.md frontmatter field 'description' cannot be empty."

    if expected_name and name != expected_name:
        return (
            False,
            f"SKILL.md frontmatter name '{name}' must match expected '{expected_name}'.",
        )

    if require_skill_overview and not _SKILL_OVERVIEW_RE.search(content):
        return False, "SKILL.md must include a '## Skill Overview' section."

    return True, ""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_iteration_dir(*, workspace_base: Path, iter_folder_name: str, backend_scope: str) -> Path:
    base = Path(workspace_base)
    if backend_scope == "workspace":
        return base / iter_folder_name
    candidate = base / iter_folder_name
    if candidate.exists():
        return candidate
    return base


def _hash_skill_dir(skill_dir: Path) -> str | None:
    if not skill_dir.exists() or not skill_dir.is_dir():
        return None

    entries: list[dict[str, str]] = []
    for skill_file in sorted(path for path in skill_dir.rglob("*") if path.is_file()):
        rel_parts = skill_file.relative_to(skill_dir).parts
        if "__pycache__" in rel_parts:
            continue
        rel_path = Path(*rel_parts).as_posix()
        content_hash = _sha256_bytes(skill_file.read_bytes())
        entries.append({"path": rel_path, "content_sha256": content_hash})

    if not entries:
        return None

    payload = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _source_content_sha256(
    *,
    source_path: str,
    workspace_base: Path,
    iter_folder_name: str,
    backend_scope: str,
) -> str | None:
    workspace = Path(workspace_base)
    iter_dir = _resolve_iteration_dir(
        workspace_base=workspace,
        iter_folder_name=iter_folder_name,
        backend_scope=backend_scope,
    )

    if source_path == meta_agent_skills_virtual_dir():
        return _hash_skill_dir(workspace / "meta_agent" / SKILLS_DIRNAME)

    if source_path == learning_context_skill_virtual_dir(iter_folder_name):
        return _hash_skill_dir(workspace / iter_folder_name / learning_context_skills_relative_dir())

    if source_path == backend_root_learning_context_skill_virtual_dir():
        return _hash_skill_dir(iter_dir / learning_context_skills_relative_dir())

    return None


def compose_skill_bundle(
    *,
    workspace_base: Path,
    iter_folder_name: str,
    include_history: bool = True,
    include_current: bool = True,
    backend_scope: Literal["workspace", "iteration"] = "workspace",
) -> SkillBundle:
    if backend_scope not in {"workspace", "iteration"}:
        raise ValueError(
            f"Invalid backend_scope '{backend_scope}'. Expected 'workspace' or 'iteration'."
        )

    workspace = Path(workspace_base)
    selected_sources: list[str] = []
    excluded_sources: list[str] = []

    if include_history:
        selected_sources.append(meta_agent_skills_virtual_dir())

    current_source: str | None = None
    if include_current:
        if backend_scope == "workspace":
            current_source = learning_context_skill_virtual_dir(iter_folder_name)
            current_skill_file = workspace / iter_folder_name / learning_context_skill_relative_path()
        else:
            current_source = backend_root_learning_context_skill_virtual_dir()
            current_iter_dir = _resolve_iteration_dir(
                workspace_base=workspace,
                iter_folder_name=iter_folder_name,
                backend_scope=backend_scope,
            )
            current_skill_file = learning_context_skill_host_path(current_iter_dir)

        if current_skill_file.exists():
            content = current_skill_file.read_text(encoding="utf-8")
            is_valid, _ = validate_skill_markdown(content, expected_name="learning-context")
            if is_valid:
                selected_sources.append(current_source)
            else:
                excluded_sources.append(current_source)
        else:
            selected_sources.append(current_source)

    hash_entries: list[dict[str, str | None]] = []
    for source in selected_sources:
        hash_entries.append(
            {
                "path": source,
                "content_sha256": _source_content_sha256(
                    source_path=source,
                    workspace_base=workspace,
                    iter_folder_name=iter_folder_name,
                    backend_scope=backend_scope,
                ),
            }
        )
    hash_payload = json.dumps(hash_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    bundle_hash = _sha256_text(hash_payload)

    return SkillBundle(
        selected_sources=selected_sources,
        excluded_sources=excluded_sources,
        overlay_included=(current_source is not None and current_source in selected_sources),
        bundle_hash=bundle_hash,
    )
