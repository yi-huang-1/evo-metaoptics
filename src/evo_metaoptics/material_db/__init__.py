"""Deterministic material-index/search helpers for inverse-design runtime."""

from .download import ensure_refractiveindex_db, ensure_material_db_ready
from .index_db import BuildStats, build_index
from .material_index import MaterialIndex
from .search import (
    MaterialMatch,
    MaterialPageRef,
    MaterialSearchResult,
    get_page_by_id,
    get_page_by_ref,
    search_materials,
)
from .settings import MaterialDbSettings, load_material_db_settings
from .tooling import search_material_candidates
from .runtime import get_material_nk

__all__ = [
    "BuildStats",
    "MaterialIndex",
    "MaterialMatch",
    "MaterialPageRef",
    "MaterialSearchResult",
    "build_index",
    "ensure_material_db_ready",
    "ensure_refractiveindex_db",
    "get_page_by_id",
    "get_material_nk",
    "get_page_by_ref",
    "load_material_db_settings",
    "search_materials",
    "MaterialDbSettings",
    "search_material_candidates",
]
