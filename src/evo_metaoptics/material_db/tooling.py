from __future__ import annotations

from typing import Sequence

from typing_extensions import TypedDict

from .material_index import MaterialIndex
from .search import MaterialMatch, MaterialPage


class MaterialCandidate(TypedDict, total=False):
    name: str
    page_id: int
    score: float | None


def search_material_candidates(
    query: str,
    *,
    wavelengths_um: Sequence[float] | None = None,
    limit: int = 3,
    index: MaterialIndex | None = None,
) -> list[MaterialCandidate]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    index = index or MaterialIndex.default()
    min_wavelength_um, max_wavelength_um = _min_max(wavelengths_um)
    search_result = index.search(
        query,
        limit=limit,
        min_wavelength_um=min_wavelength_um,
        max_wavelength_um=max_wavelength_um,
    )
    candidates: list[MaterialCandidate] = []
    for match in search_result.matches:
        page_id = _select_page_id(match)
        if page_id is None:
            continue
        candidates.append(
            {
                "name": _display_name(match),
                "page_id": page_id,
                "score": match.score,
            }
        )
    return candidates


def _min_max(values: Sequence[float] | None) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    floats = [float(v) for v in values]
    return min(floats), max(floats)


def _display_name(match: MaterialMatch) -> str:
    for attr in ("display_name", "title", "name_plain", "book"):
        value = getattr(match, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return str(match.book)


def _select_page_id(match: MaterialMatch) -> int | None:
    pages = [page for page in match.pages if page.page_id is not None]
    if not pages:
        return None
    pages = sorted(pages, key=lambda p: str(p.page))
    for page in pages:
        if _has_nk(page):
            return int(page.page_id)
    return int(pages[0].page_id)


def _has_nk(page: MaterialPage) -> bool:
    return bool(page.has_n) and bool(page.has_k)
