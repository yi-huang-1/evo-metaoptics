from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3] / "resources" / "material-db"


def default_source_path() -> Path:
    return repo_root() / "database" / "refractiveindex.info-database"


def default_db_path() -> Path:
    return repo_root() / "database" / "nk_index.sqlite"
