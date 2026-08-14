"""Immutable Template Registry for TorchRDIT code templates.

Provides read-only access to code templates extracted from
``reference/torchrdit-mcp/server.py``.  Templates are static ``ρ``
components in MCE terminology — their content never evolves.

Template content is verified against SHA-256 hashes on every retrieval
to guarantee immutability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evo_metaoptics.mce.skill_shard_registry import SkillShardRegistry, render_shard_markdown


_TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent
    / "mce_env"
    / "metaoptics_inverse_design"
    / "skills"
    / "templates"
)
_MANIFEST_FILENAME = "templates_manifest.json"

_REFERENCE_DOC_SPECS: tuple[dict[str, Any], ...] = (
    {
        "filename": "setup.md",
        "title": "Setup",
        "overview": "Canonical imports, units, and builder setup for cold-start-safe TorchRDIT runs.",
        "templates": (
            ("Required imports", "basic_imports"),
            ("Unit helpers", "unit_setup"),
            ("Builder setup", "solver_setup"),
        ),
        "shards": (("Builder reference", "solver_setup"),),
    },
    {
        "filename": "materials_and_layers.md",
        "title": "Materials And Layers",
        "overview": "Material registration, boundary media, and stack-order patterns.",
        "templates": (
            ("Material creation", "material_creation"),
            ("Material API guardrails", "material_api"),
            ("Layer ordering", "layer_order"),
            ("Layer stack", "layer_stack"),
        ),
        "shards": (("Materials and layer guidance", "materials_layers"),),
    },
    {
        "filename": "patterning.md",
        "title": "Patterning",
        "overview": "Patternable-layer setup, masks, and ShapeGenerator operations.",
        "templates": (
            ("Patterned layers", "patterned_layer"),
            ("Shape operations", "shape_operations"),
        ),
        "shards": (("Patterning reference", "patterning_shapes"),),
    },
    {
        "filename": "sources_and_solving.md",
        "title": "Sources And Solving",
        "overview": "Single-source and batched-source execution patterns plus result access basics.",
        "templates": (
            ("Source setup", "source_setup"),
            ("Solve and inspect", "solve_and_analyze"),
        ),
        "shards": (("Source and solve reference", "source_solve"),),
    },
    {
        "filename": "phase_and_orders.md",
        "title": "Phase And Orders",
        "overview": "Zero-order phase extraction and diffraction-order access patterns.",
        "templates": (("Solve and inspect", "solve_and_analyze"),),
        "shards": (("Phase extraction", "postprocess_phase"),),
    },
    {
        "filename": "amplitude_and_efficiency.md",
        "title": "Amplitude And Efficiency",
        "overview": "Amplitude, total transmission/reflection, and diffraction-efficiency retrieval.",
        "templates": (("Solve and inspect", "solve_and_analyze"),),
        "shards": (("Amplitude and efficiency", "postprocess_amplitude"),),
    },
    {
        "filename": "optimization.md",
        "title": "Optimization",
        "overview": "Deterministic local and global optimization patterns for inverse design.",
        "templates": (
            ("Basic optimization", "optimization_basic"),
            ("Gradient-based workflow", "gradient_based"),
            ("Full differentiable pipeline", "gradient_full_pipeline"),
            ("Phase-target optimization", "gradient_phase_target"),
            ("Multi-angle optimization", "gradient_multiangle"),
            ("Multi-objective loss", "multi_objective"),
            ("Common design patterns", "common_patterns"),
        ),
        "shards": (("Global optimization strategies", "optimization_patterns"),),
    },
    {
        "filename": "pitfalls.md",
        "title": "Pitfalls",
        "overview": "High-frequency mistakes and the shortest reliable fixes.",
        "templates": (("Common mistakes", "common_mistakes"),),
        "shards": (("Common pitfalls", "common_pitfalls"),),
    },
    {
        "filename": "context_usage.md",
        "title": "Context Usage",
        "overview": "How to use copied `context/` artifacts without treating them as ground truth.",
        "body": (
            "## Reading context\n\n"
            "- Treat `context/` as optional learned guidance from prior iterations.\n"
            "- Reuse stable rules or examples when they fit the current query and `gt_eval`.\n"
            "- Prefer concise synthesis over copying large context fragments verbatim.\n\n"
            "## Safe usage pattern\n\n"
            "1. Read `SKILL.md` first for the active contract and strategy.\n"
            "2. Read only the reference files needed for the current failure mode or design goal.\n"
            "3. Read `context/` artifacts last, then adapt them to the current sample.\n"
            "4. Keep generated code deterministic and bounded even if context suggests broader search.\n"
        ),
    },
)


@dataclass(frozen=True)
class TemplateEntry:
    """Metadata for a single template."""

    id: str
    category: str
    sha256: str
    tags: tuple[str, ...]
    summary: str
    est_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "sha256": self.sha256,
            "tags": list(self.tags),
            "summary": self.summary,
            "est_tokens": self.est_tokens,
        }


class TemplateIntegrityError(Exception):
    """Raised when template content does not match its manifest SHA-256."""


class TemplateNotFoundError(KeyError):
    """Raised when a requested template ID does not exist."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TemplateRegistry:
    """Read-only registry of immutable TorchRDIT code templates.

    Loads templates from disk on construction and verifies content integrity
    on every retrieval via SHA-256 hashes from the manifest.

    Parameters
    ----------
    templates_dir : Path, optional
        Override the default templates directory (mainly for testing).
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        self._templates_dir = Path(templates_dir) if templates_dir else _TEMPLATES_DIR
        self._entries: dict[str, TemplateEntry] = {}
        self._content_cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        manifest_path = self._templates_dir / _MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Template manifest not found: {manifest_path}"
            )

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry_dict in raw:
            tid = entry_dict["id"]
            self._entries[tid] = TemplateEntry(
                id=tid,
                category=entry_dict["category"],
                sha256=entry_dict["sha256"],
                tags=tuple(entry_dict["tags"]),
                summary=entry_dict["summary"],
                est_tokens=entry_dict["est_tokens"],
            )
            # Eagerly load and verify content
            template_path = self._templates_dir / f"{tid}.py.jinja"
            if not template_path.exists():
                raise FileNotFoundError(
                    f"Template file missing for '{tid}': {template_path}"
                )
            content = template_path.read_text(encoding="utf-8")
            actual_hash = _sha256(content)
            if actual_hash != entry_dict["sha256"]:
                raise TemplateIntegrityError(
                    f"Template '{tid}' content hash mismatch: "
                    f"manifest={entry_dict['sha256']}, actual={actual_hash}"
                )
            self._content_cache[tid] = content

    def manifest(self) -> list[dict[str, Any]]:
        """Return sorted list of template metadata dicts.

        Each dict contains: id, category, sha256, tags, summary, est_tokens.
        The list is sorted by ``id`` for deterministic ordering.
        """
        return [
            self._entries[tid].to_dict()
            for tid in sorted(self._entries)
        ]

    def get_template(self, template_id: str) -> str:
        """Return template content string, verified by SHA-256.

        Raises
        ------
        TemplateNotFoundError
            If the template_id does not exist.
        TemplateIntegrityError
            If content no longer matches its manifest hash (should not happen
            with frozen files, but guards against runtime tampering).
        """
        if template_id not in self._entries:
            raise TemplateNotFoundError(
                f"Unknown template '{template_id}'. "
                f"Available: {sorted(self._entries)}"
            )
        content = self._content_cache[template_id]
        # Verify on every access
        actual_hash = _sha256(content)
        expected_hash = self._entries[template_id].sha256
        if actual_hash != expected_hash:
            raise TemplateIntegrityError(
                f"Template '{template_id}' content integrity check failed"
            )
        return content

    def get_templates(self, template_ids: list[str]) -> dict[str, str]:
        """Batch-retrieve templates by ID.

        Returns dict mapping template_id -> content string.
        Raises TemplateNotFoundError for any unknown ID.
        """
        result: dict[str, str] = {}
        for tid in template_ids:
            result[tid] = self.get_template(tid)
        return result

    def get_entry(self, template_id: str) -> TemplateEntry:
        """Return the TemplateEntry metadata for a template."""
        if template_id not in self._entries:
            raise TemplateNotFoundError(
                f"Unknown template '{template_id}'. "
                f"Available: {sorted(self._entries)}"
            )
        return self._entries[template_id]

    @property
    def template_ids(self) -> list[str]:
        """Return sorted list of all template IDs."""
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._entries


def render_template_markdown(
    template_id: str,
    *,
    heading: str,
    registry: TemplateRegistry | None = None,
) -> str:
    active_registry = registry or TemplateRegistry()
    content = active_registry.get_template(template_id).strip()
    return f"## {heading}\n\n```python\n{content}\n```"


def build_progressive_reference_documents(
    *,
    template_registry: TemplateRegistry | None = None,
    shard_registry: SkillShardRegistry | None = None,
) -> dict[str, str]:
    active_template_registry = template_registry or TemplateRegistry()
    active_shard_registry = shard_registry or SkillShardRegistry()
    documents: dict[str, str] = {}

    for spec in _REFERENCE_DOC_SPECS:
        sections = [f"# {spec['title']}", "", spec["overview"]]
        for heading, template_id in spec.get("templates", ()):
            sections.extend([
                "",
                render_template_markdown(
                    template_id,
                    heading=heading,
                    registry=active_template_registry,
                ),
            ])
        for heading, shard_id in spec.get("shards", ()):
            sections.extend([
                "",
                render_shard_markdown(
                    shard_id,
                    heading=heading,
                    registry=active_shard_registry,
                ),
            ])
        if "body" in spec:
            sections.extend(["", spec["body"].rstrip()])
        documents[spec["filename"]] = "\n".join(sections).rstrip() + "\n"

    return documents


def materialize_progressive_reference_subtree(
    target_dir: Path,
    *,
    template_registry: TemplateRegistry | None = None,
    shard_registry: SkillShardRegistry | None = None,
) -> list[Path]:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    documents = build_progressive_reference_documents(
        template_registry=template_registry,
        shard_registry=shard_registry,
    )
    written_paths: list[Path] = []
    for filename, content in documents.items():
        path = target_dir / filename
        path.write_text(content, encoding="utf-8")
        written_paths.append(path)
    return written_paths
