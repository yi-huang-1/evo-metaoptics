"""Material resolver used by deterministic inverse-design evaluation in MCE."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

from evo_metaoptics.material_db import MaterialIndex, get_page_by_id, load_material_db_settings
from evo_metaoptics.material_db.paths import default_source_path
from evo_metaoptics.material_db.search import MaterialMatch, MaterialSearchResult
from evo_metaoptics.material_db.text_normalize import normalize_query

try:  # pragma: no cover - dependency is exercised in integration runs.
    from refractiveindex import RefractiveIndexMaterial
except Exception:  # pragma: no cover - handled by injection in tests.
    RefractiveIndexMaterial = None

try:  # pragma: no cover - dependency is exercised in integration runs.
    from refractiveindex.refractiveindex import NoExtinctionCoefficient
except Exception:  # pragma: no cover - handled by injection in tests.
    NoExtinctionCoefficient = None


@dataclass(frozen=True)
class MaterialChoice:
    shelf: str
    book: str
    page: str


@dataclass(frozen=True)
class MaterialSelection:
    shelf: str
    book: str
    page: str
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class ResolvedMaterial:
    selection: MaterialSelection
    wavelengths_um: tuple[float, ...]
    n: tuple[float, ...]
    k: tuple[float, ...]

    def to_material_spec(self) -> dict[str, object]:
        if not self.wavelengths_um:
            raise ValueError("Resolved material missing wavelengths_um.")
        if len(self.wavelengths_um) == 1:
            n0 = float(self.n[0])
            k0 = float(self.k[0])
            return {"n": n0, "k": k0}
        return {
            "dielectric_dispersion": True,
            "dispersion": {
                "wavelengths_um": list(float(wl) for wl in self.wavelengths_um),
                "n": list(float(n) for n in self.n),
                "k": list(float(k) for k in self.k),
            },
        }


PickerFn = Callable[[str, MaterialSearchResult, Sequence[float]], MaterialChoice]


class MaterialResolver:
    def __init__(
        self,
        *,
        index: MaterialIndex | None = None,
        database_path: Path | None = None,
        picker: PickerFn | None = None,
        nk_factory: Callable[..., object] | None = None,
    ) -> None:
        settings = None
        if index is None or database_path is None:
            settings = load_material_db_settings()
        self._index = (
            index
            or (MaterialIndex(db_path=settings.db_path) if settings else MaterialIndex.default())
        )
        if database_path is None:
            self._database_path = settings.source_root if settings else default_source_path()
        else:
            self._database_path = database_path
        self._picker = picker or self._default_picker
        self._nk_factory = nk_factory or RefractiveIndexMaterial
        if self._nk_factory is None:
            raise RuntimeError("Missing refractiveindex dependency; install `refractiveindex`.")
        self._cache: dict[tuple[str, str, str, str], ResolvedMaterial] = {}

    @property
    def db_path(self) -> Path:
        return self._index.db_path

    @property
    def source_root(self) -> Path:
        return self._database_path

    def resolve(
        self,
        ref: dict[str, str],
        wavelengths_um: Sequence[float],
        context: dict | None = None,
    ) -> ResolvedMaterial:
        del context
        ref_type = ref.get("type")
        value = ref.get("value")
        if ref_type not in {"name", "query", "page_id", "page"}:
            raise ValueError(f"Unsupported ref.type: {ref_type!r}")
        if ref_type in {"name", "query"}:
            if not value:
                raise ValueError("ref.value must be a non-empty string")
        if ref_type == "page_id":
            page_id = _parse_page_id(value)
            resolved = self._resolve_by_page_id(page_id, wavelengths_um)
            return resolved
        if ref_type == "page":
            resolved = self._resolve_by_page_ref(value, wavelengths_um)
            return resolved

        search_result = self._index.search(value)
        match_count = len(search_result.matches)
        if not search_result.matches:
            raise ValueError(f"No material matches for {value!r} (candidates={match_count})")

        try:
            choice, selection = self._select_choice(ref_type, value, search_result, wavelengths_um)
        except ValueError as exc:
            context_str = _format_search_context(ref_type, value, search_result)
            raise ValueError(f"Material selection failed ({context_str}): {exc}") from exc
        return self._resolve_choice(choice, selection, wavelengths_um)

    def _resolve_by_page_id(self, page_id: int, wavelengths_um: Sequence[float]) -> ResolvedMaterial:
        page_ref = get_page_by_id(page_id, db_path=self._index.db_path)
        if page_ref is None:
            raise ValueError(f"Unknown material page id: {page_id}")
        choice = MaterialChoice(shelf=page_ref.shelf, book=page_ref.book, page=page_ref.page)
        selection = MaterialSelection(
            shelf=page_ref.shelf,
            book=page_ref.book,
            page=page_ref.page,
            reason="page_id",
            score=None,
        )
        return self._resolve_choice(choice, selection, wavelengths_um)

    def _resolve_by_page_ref(self, value: object, wavelengths_um: Sequence[float]) -> ResolvedMaterial:
        if not isinstance(value, dict):
            raise ValueError("ref.value must be an object with shelf/book/page for type='page'.")
        shelf = value.get("shelf")
        book = value.get("book")
        page = value.get("page")
        if not all(isinstance(v, str) and v for v in (shelf, book, page)):
            raise ValueError("ref.value must include non-empty shelf, book, and page strings.")
        choice = MaterialChoice(shelf=shelf, book=book, page=page)
        selection = MaterialSelection(
            shelf=shelf,
            book=book,
            page=page,
            reason="page_ref",
            score=None,
        )
        return self._resolve_choice(choice, selection, wavelengths_um)

    def _resolve_choice(
        self,
        choice: MaterialChoice,
        selection: MaterialSelection,
        wavelengths_um: Sequence[float],
    ) -> ResolvedMaterial:
        grid_hash = _grid_hash(wavelengths_um)
        cache_key = (choice.shelf, choice.book, choice.page, grid_hash)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        n_values, k_values = _fetch_nk(
            self._nk_factory,
            choice=choice,
            wavelengths_um=wavelengths_um,
            database_path=self._database_path,
        )
        resolved = ResolvedMaterial(
            selection=selection,
            wavelengths_um=tuple(float(w) for w in wavelengths_um),
            n=n_values,
            k=k_values,
        )
        self._cache[cache_key] = resolved
        return resolved

    def _select_choice(
        self,
        ref_type: str,
        query: str,
        search_result: MaterialSearchResult,
        wavelengths_um: Sequence[float],
    ) -> tuple[MaterialChoice, MaterialSelection]:
        if ref_type == "query":
            choice = self._picker(query, search_result, wavelengths_um)
            return choice, MaterialSelection(
                shelf=choice.shelf,
                book=choice.book,
                page=choice.page,
                reason="picker",
            )

        query_norm = normalize_query(query)
        for match in search_result.matches:
            if _is_exact_match(query_norm, match):
                choice, reason = _pick_page(match, query, search_result, wavelengths_um, self._picker)
                selection = MaterialSelection(
                    shelf=choice.shelf,
                    book=choice.book,
                    page=choice.page,
                    reason=reason,
                    score=match.score,
                )
                return choice, selection

        choice = self._picker(query, search_result, wavelengths_um)
        selection = MaterialSelection(
            shelf=choice.shelf,
            book=choice.book,
            page=choice.page,
            reason="picker",
        )
        return choice, selection

    @staticmethod
    def _default_picker(
        query: str,
        search_result: MaterialSearchResult,
        wavelengths_um: Sequence[float],
    ) -> MaterialChoice:
        del query
        match = search_result.matches[0]
        page = _best_page(match.pages, wavelengths_um)
        return MaterialChoice(shelf=match.shelf, book=match.book, page=page)


def _grid_hash(wavelengths_um: Sequence[float]) -> str:
    payload = ",".join(f"{float(w):.9g}" for w in wavelengths_um)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _parse_page_id(value: object) -> int:
    try:
        page_id = int(value)
    except (TypeError, ValueError):
        raise ValueError("ref.value must be an integer page_id for type='page_id'.") from None
    if page_id <= 0:
        raise ValueError("ref.value must be a positive integer page_id for type='page_id'.")
    return page_id


def _format_search_context(
    ref_type: str | None, value: str | None, search_result: MaterialSearchResult
) -> str:
    return (
        f"ref_type={ref_type!r} value={value!r} candidates={len(search_result.matches)}"
    )


def _is_exact_match(query_norm: str, match: MaterialMatch) -> bool:
    candidates = [match.book, match.name_plain or "", match.title or ""]
    candidates.extend(list(match.other_names))
    for candidate in candidates:
        if normalize_query(candidate) == query_norm:
            return True
    return False


def _pick_page(
    match: MaterialMatch,
    query: str,
    search_result: MaterialSearchResult,
    wavelengths_um: Sequence[float],
    picker: PickerFn,
) -> tuple[MaterialChoice, str]:
    if not match.pages:
        raise ValueError(f"No pages available for material {match.book!r}")
    if len(match.pages) == 1:
        return (
            MaterialChoice(shelf=match.shelf, book=match.book, page=match.pages[0].page),
            "exact_match",
        )

    narrowed = MaterialSearchResult(
        query=search_result.query,
        normalized_query=search_result.normalized_query,
        matches=(match,),
    )
    choice = picker(query, narrowed, wavelengths_um)
    if (choice.shelf, choice.book) != (match.shelf, match.book):
        raise ValueError("Picker must choose a page for the matched material.")
    return choice, "picker"


def _first_page(pages: Iterable) -> str:
    sorted_pages = sorted(pages, key=lambda p: p.page)
    if not sorted_pages:
        raise ValueError("Material has no dataset pages.")
    return sorted_pages[0].page


def _best_page(pages: Iterable, wavelengths_um: Sequence[float]) -> str:
    candidates = list(pages)
    if not candidates:
        raise ValueError("Material has no dataset pages.")
    if not wavelengths_um:
        return candidates[0].page
    if not _has_coverage_candidate(candidates, wavelengths_um):
        raise ValueError("No material dataset pages cover the requested wavelengths.")
    wl_min = min(float(w) for w in wavelengths_um)
    wl_max = max(float(w) for w in wavelengths_um)

    best = None
    best_key = None
    for idx, page in enumerate(candidates):
        has_nk = bool(getattr(page, "has_n", False)) and bool(getattr(page, "has_k", False))
        range_um = _page_range(page)
        coverage = _coverage_rank(range_um, wl_min, wl_max)
        key = (coverage, 1 if has_nk else 0, -idx)
        if best_key is None or key > best_key:
            best = page
            best_key = key

    if best is None:
        raise ValueError("Material has no dataset pages.")
    return best.page


def _has_coverage_candidate(pages: Iterable, wavelengths_um: Sequence[float]) -> bool:
    wl_min = min(float(w) for w in wavelengths_um)
    wl_max = max(float(w) for w in wavelengths_um)
    any_range = False
    for page in pages:
        range_um = _page_range(page)
        if range_um is None:
            continue
        any_range = True
        low, high = range_um
        if low <= wl_min and wl_max <= high:
            return True
    return not any_range


def _page_range(page: object) -> tuple[float, float] | None:
    low = getattr(page, "coverage_min", None)
    high = getattr(page, "coverage_max", None)
    if low is None or high is None:
        return None
    return float(low), float(high)


def _coverage_rank(range_um: tuple[float, float] | None, wl_min: float, wl_max: float) -> int:
    if range_um is None:
        return 1
    low, high = range_um
    if low <= wl_min and wl_max <= high:
        return 2
    return 0


def _fetch_nk(
    nk_factory: Callable[..., object],
    *,
    choice: MaterialChoice,
    wavelengths_um: Sequence[float],
    database_path: Path,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    material = nk_factory(
        shelf=choice.shelf,
        book=choice.book,
        page=choice.page,
        databasePath=database_path,
    )

    n_values: list[float] = []
    k_values: list[float] = []
    for wl_um in wavelengths_um:
        wl_nm = float(wl_um) * 1000.0
        n = float(material.get_refractive_index(wl_nm))
        try:
            k_raw = material.get_extinction_coefficient(wl_nm)
        except Exception as exc:
            if NoExtinctionCoefficient is not None and isinstance(exc, NoExtinctionCoefficient):
                k_raw = None
            elif NoExtinctionCoefficient is None and exc.__class__.__name__ == "NoExtinctionCoefficient":
                k_raw = None
            else:
                raise
        k = 0.0 if k_raw is None else float(k_raw)
        _validate_nk(n, k, wl_um)
        n_values.append(n)
        k_values.append(k)

    return tuple(n_values), tuple(k_values)


def _validate_nk(n: float, k: float, wl_um: float) -> None:
    if not math.isfinite(n):
        raise ValueError(f"Invalid refractive index at {wl_um} um: {n!r}")
    if not math.isfinite(k):
        raise ValueError(f"Invalid extinction coefficient at {wl_um} um: {k!r}")
