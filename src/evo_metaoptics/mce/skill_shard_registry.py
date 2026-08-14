from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHARDS_DIR = (
    Path(__file__).resolve().parent.parent
    / "mce_env"
    / "metaoptics_inverse_design"
    / "skills"
    / "shards"
)
_MANIFEST_FILENAME = "shards_manifest.json"
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class ShardEntry:
    id: str
    sha256: str
    tags: tuple[str, ...]
    summary: str
    est_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "tags": list(self.tags),
            "summary": self.summary,
            "est_tokens": self.est_tokens,
        }


class ShardIntegrityError(Exception):
    pass


class ShardNotFoundError(KeyError):
    pass


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SkillShardRegistry:
    """Read-only registry of skill shards split from the monolithic TorchRDIT skill."""

    def __init__(self, shards_dir: Path | None = None) -> None:
        self._shards_dir = Path(shards_dir) if shards_dir else _SHARDS_DIR
        self._entries: dict[str, ShardEntry] = {}
        self._content_cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        manifest_path = self._shards_dir / _MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Shard manifest not found: {manifest_path}")

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry_dict in raw:
            shard_id = entry_dict["id"]
            self._entries[shard_id] = ShardEntry(
                id=shard_id,
                sha256=entry_dict["sha256"],
                tags=tuple(entry_dict["tags"]),
                summary=entry_dict["summary"],
                est_tokens=entry_dict["est_tokens"],
            )

            shard_path = self._shards_dir / f"{shard_id}.md"
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Shard file missing for '{shard_id}': {shard_path}"
                )

            content = shard_path.read_text(encoding="utf-8")
            actual_hash = _sha256(content)
            if actual_hash != entry_dict["sha256"]:
                raise ShardIntegrityError(
                    f"Shard '{shard_id}' content hash mismatch: "
                    f"manifest={entry_dict['sha256']}, actual={actual_hash}"
                )
            self._content_cache[shard_id] = content

    def manifest(self) -> list[dict[str, Any]]:
        return [self._entries[shard_id].to_dict() for shard_id in sorted(self._entries)]

    def get_shard(self, shard_id: str) -> str:
        if shard_id not in self._entries:
            raise ShardNotFoundError(
                f"Unknown shard '{shard_id}'. Available: {sorted(self._entries)}"
            )

        content = self._content_cache[shard_id]
        actual_hash = _sha256(content)
        expected_hash = self._entries[shard_id].sha256
        if actual_hash != expected_hash:
            raise ShardIntegrityError(
                f"Shard '{shard_id}' content integrity check failed"
            )
        return content

    def get_shards(self, shard_ids: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for shard_id in shard_ids:
            result[shard_id] = self.get_shard(shard_id)
        return result

    def get_entry(self, shard_id: str) -> ShardEntry:
        if shard_id not in self._entries:
            raise ShardNotFoundError(
                f"Unknown shard '{shard_id}'. Available: {sorted(self._entries)}"
            )
        return self._entries[shard_id]

    @property
    def shard_ids(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, shard_id: object) -> bool:
        return shard_id in self._entries


def strip_shard_frontmatter(content: str) -> str:
    return _FRONTMATTER_RE.sub("", content, count=1).lstrip()


def render_shard_markdown(
    shard_id: str,
    *,
    heading: str | None = None,
    registry: SkillShardRegistry | None = None,
) -> str:
    active_registry = registry or SkillShardRegistry()
    body = strip_shard_frontmatter(active_registry.get_shard(shard_id)).strip()
    if heading is None:
        return body
    return f"## {heading}\n\n{body}"
