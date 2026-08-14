"""Material DB settings for deterministic inverse-design runtime.

This migrated module intentionally avoids archived Hydra config dependencies and
uses environment/default-path resolution only.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from .paths import default_db_path, default_source_path

_ENV_DB_PATH = "EVO_METAOPTICS_MATERIAL_DB_PATH"
_ENV_SOURCE_ROOT = "EVO_METAOPTICS_MATERIAL_DB_SOURCE_ROOT"
_ENV_AUTO_DOWNLOAD = "EVO_METAOPTICS_MATERIAL_DB_AUTO_DOWNLOAD"


@dataclass(frozen=True)
class MaterialDbSettings:
    db_path: Path
    source_root: Path
    auto_download: bool = True

    @staticmethod
    def from_mapping(
        config: Mapping[str, Any] | None,
        *,
        workspace_root: Path | None = None,
    ) -> "MaterialDbSettings":
        payload = dict(config) if isinstance(config, Mapping) else {}
        db_path = _resolve_path(payload.get("path"), default_db_path(), workspace_root)
        source_root = _resolve_path(
            payload.get("source_root"),
            default_source_path(),
            workspace_root,
        )
        auto_download = _parse_bool(payload.get("auto_download"), default=True)
        return MaterialDbSettings(
            db_path=db_path,
            source_root=source_root,
            auto_download=auto_download,
        )

    @staticmethod
    def from_config(
        config: Any | None,
        *,
        workspace_root: Path | None = None,
    ) -> "MaterialDbSettings":
        if isinstance(config, Mapping):
            return MaterialDbSettings.from_mapping(config, workspace_root=workspace_root)
        if config is None:
            return MaterialDbSettings.from_mapping(None, workspace_root=workspace_root)
        mapped = {
            "path": getattr(config, "path", None),
            "source_root": getattr(config, "source_root", None),
            "auto_download": getattr(config, "auto_download", None),
        }
        return MaterialDbSettings.from_mapping(mapped, workspace_root=workspace_root)


def load_material_db_settings(config_dir: str | Path | None = None) -> MaterialDbSettings:
    del config_dir
    db_path = _resolve_path(os.getenv(_ENV_DB_PATH), default_db_path(), workspace_root=None)
    source_root = _resolve_path(
        os.getenv(_ENV_SOURCE_ROOT),
        default_source_path(),
        workspace_root=None,
    )
    auto_download = _parse_bool(os.getenv(_ENV_AUTO_DOWNLOAD), default=True)
    return MaterialDbSettings(
        db_path=db_path,
        source_root=source_root,
        auto_download=auto_download,
    )


def _resolve_path(value: Any, fallback: Path, workspace_root: Path | None) -> Path:
    if value is None:
        return fallback
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return fallback
        path = Path(raw).expanduser()
    elif isinstance(value, Path):
        path = value.expanduser()
    else:
        return fallback
    if not path.is_absolute():
        base = workspace_root or Path.cwd()
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
