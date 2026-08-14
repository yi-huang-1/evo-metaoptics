"""Material resolver used by deterministic inverse-design evaluation in MCE.

Task-111 M3 note:
- legacy resolver code is preserved in `material_resolver_legacy_base.py`.
- this module keeps the existing `resolve(...)` API and adds
  `resolve_facts_ir_materials(...)` for typed-IR instance-indexed resolution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import threading
from typing import Any

import evo_metaoptics.meta_design.material_resolver_legacy_base as _legacy
from evo_metaoptics.meta_design.material_resolver_legacy_base import *  # noqa: F401,F403
from evo_metaoptics.meta_design.material_resolver_legacy_base import (
    MaterialChoice,
    MaterialResolver as _LegacyMaterialResolver,
    MaterialSelection,
)

get_page_by_id = _legacy.get_page_by_id


def _get_page_by_id_proxy(*args: Any, **kwargs: Any) -> Any:
    return get_page_by_id(*args, **kwargs)


_legacy.get_page_by_id = _get_page_by_id_proxy


class MaterialResolveError(ValueError):
    def __init__(self, code: str, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.code = str(code).strip() or "material_resolve_error"
        self.details = details


def _extract_wavelength_grid(canonical_ir: Mapping[str, Any]) -> tuple[float, ...]:
    sim = canonical_ir.get("sim")
    if not isinstance(sim, Mapping):
        raise MaterialResolveError(
            "wavelength_grid_invalid",
            "facts_ir.sim must be an object with wavelength_um.",
        )
    wavelengths = sim.get("wavelength_um")
    if not isinstance(wavelengths, Sequence) or isinstance(wavelengths, (str, bytes)):
        raise MaterialResolveError(
            "wavelength_grid_invalid",
            "facts_ir.sim.wavelength_um must be a non-empty numeric array.",
        )
    values: list[float] = []
    for raw in wavelengths:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise MaterialResolveError(
                "wavelength_grid_invalid",
                "facts_ir.sim.wavelength_um entries must be finite positive numbers.",
            )
        values.append(value)
    if not values:
        raise MaterialResolveError(
            "wavelength_grid_invalid",
            "facts_ir.sim.wavelength_um must not be empty.",
        )
    return tuple(values)


def _coerce_material_instances(raw_mats: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_mats, list):
        raise ValueError("facts_ir.mats must be a JSON array.")
    instances: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in raw_mats:
        if not isinstance(raw, Mapping):
            raise ValueError("facts_ir.mats entries must be JSON objects.")
        payload = dict(raw)
        idx_raw = payload.get("idx")
        if isinstance(idx_raw, bool):
            raise ValueError("facts_ir.mats[].idx must be an integer >= 0.")
        if not isinstance(idx_raw, (int, float, str)):
            raise ValueError("facts_ir.mats[].idx must be an integer >= 0.")
        idx = int(idx_raw)
        if idx < 0:
            raise ValueError("facts_ir.mats[].idx must be an integer >= 0.")
        if idx in seen:
            raise ValueError("facts_ir.mats[].idx values must be unique.")
        seen.add(idx)

        opt = payload.get("opt")
        if not isinstance(opt, Mapping):
            raise ValueError(f"facts_ir.mats[{idx}].opt must be an object.")

        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"facts_ir.mats[{idx}].name must be a non-empty string.")
        payload["idx"] = idx
        payload["name"] = name.strip()
        payload["opt"] = dict(opt)
        instances.append(payload)

    instances.sort(key=lambda item: int(item["idx"]))
    expected = list(range(len(instances)))
    observed = [int(item["idx"]) for item in instances]
    if observed != expected:
        raise ValueError("facts_ir.mats[].idx must be contiguous and range from 0..len(mats)-1")
    return instances


def _material_signature(material: Mapping[str, Any]) -> str:
    payload = {
        "idx": int(material.get("idx", -1)),
        "name": str(material.get("name", "")),
        "opt": material.get("opt"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _nk_table_values(opt: Mapping[str, Any], wavelengths_um: Sequence[float]) -> tuple[list[float], list[float]]:
    raw_points = opt.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("nk_table requires a non-empty points array.")

    points: list[tuple[float, float, float]] = []
    for item in raw_points:
        if not isinstance(item, Mapping):
            raise ValueError("nk_table points entries must be objects.")
        raw_wl = item.get("wavelength_um")
        raw_n = item.get("n")
        if not isinstance(raw_wl, (int, float, str)) or not isinstance(raw_n, (int, float, str)):
            raise ValueError("nk_table points entries must include numeric wavelength_um and n values.")
        wl = float(raw_wl)
        n = float(raw_n)
        k_raw = item.get("k", 0.0)
        k = float(0.0 if k_raw is None else k_raw)
        if not math.isfinite(wl) or wl <= 0.0:
            raise ValueError("nk_table wavelength_um values must be finite positive numbers.")
        if not math.isfinite(n) or not math.isfinite(k):
            raise ValueError("nk_table n/k values must be finite numbers.")
        points.append((wl, n, k))

    points.sort(key=lambda value: value[0])
    for i in range(1, len(points)):
        if points[i][0] <= points[i - 1][0]:
            raise ValueError("nk_table wavelength_um values must be strictly increasing.")

    n_values: list[float] = []
    k_values: list[float] = []
    min_wl = points[0][0]
    max_wl = points[-1][0]
    for wl in wavelengths_um:
        target = float(wl)
        if target < min_wl or target > max_wl:
            raise MaterialResolveError(
                "nk_table_coverage_missing",
                "nk_table points do not cover requested wavelength grid; provide wider table or use db_lookup."
            )
        if len(points) == 1:
            n_values.append(points[0][1])
            k_values.append(points[0][2])
            continue
        for left, right in zip(points[:-1], points[1:]):
            l_wl, l_n, l_k = left
            r_wl, r_n, r_k = right
            if l_wl <= target <= r_wl:
                if abs(r_wl - l_wl) <= 1.0e-15:
                    alpha = 0.0
                else:
                    alpha = (target - l_wl) / (r_wl - l_wl)
                n_values.append(float(l_n + alpha * (r_n - l_n)))
                k_values.append(float(l_k + alpha * (r_k - l_k)))
                break
    return n_values, k_values


def _match_candidate_page(match: Any, wavelengths_um: Sequence[float]) -> Any:
    pages = list(getattr(match, "pages", ()) or ())
    if not pages:
        raise ValueError(f"No dataset pages found for material {getattr(match, 'book', '<unknown>')!r}")
    page_name = _legacy._best_page(pages, wavelengths_um)
    for page in pages:
        if str(getattr(page, "page", "")) == str(page_name):
            return page
    return sorted(pages, key=lambda item: str(getattr(item, "page", "")))[0]


class MaterialResolver(_LegacyMaterialResolver):
    """Legacy resolver + typed-IR material-instance resolver for M3."""

    def __init__(
        self,
        *,
        index=None,
        database_path=None,
        picker=None,
        nk_factory=None,
    ) -> None:
        super().__init__(
            index=index,
            database_path=database_path,
            picker=picker,
            nk_factory=nk_factory,
        )
        self._ir_cache_lock = threading.Lock()
        self._ir_cache: dict[tuple[int, str, str], dict[str, Any]] = {}

    def resolve_facts_ir_materials(
        self,
        canonical_ir: Mapping[str, Any],
        *,
        max_workers: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve `facts_ir.mats` into deterministic index-keyed material payloads.

        Returns `(resolved_materials, resolver_diagnostics)` where:
        - `resolved_materials.by_index` is keyed by stringified material index.
        - `resolver_diagnostics` reports cache hit/miss and candidate ranking reasons.
        """

        if not isinstance(canonical_ir, Mapping):
            raise ValueError("facts_ir payload must be a JSON object.")
        wavelengths_um = _extract_wavelength_grid(canonical_ir)
        materials = _coerce_material_instances(canonical_ir.get("mats"))

        if not materials:
            return (
                {
                    "resolved_count": 0,
                    "entries": [],
                    "by_index": {},
                },
                {
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "entries": [],
                },
            )

        resolved_rows: dict[int, dict[str, Any]] = {}
        diag_rows: dict[int, dict[str, Any]] = {}

        worker_count = len(materials)
        if max_workers is not None:
            worker_count = max(1, min(len(materials), int(max_workers)))

        if worker_count <= 1 or len(materials) == 1:
            for material in materials:
                idx = int(material["idx"])
                entry, diag = self._resolve_material_instance(material, wavelengths_um)
                resolved_rows[idx] = entry
                diag_rows[idx] = diag
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_by_idx = {
                    executor.submit(self._resolve_material_instance, material, wavelengths_um): int(
                        material["idx"]
                    )
                    for material in materials
                }
                for future in as_completed(future_by_idx):
                    idx = future_by_idx[future]
                    entry, diag = future.result()
                    resolved_rows[idx] = entry
                    diag_rows[idx] = diag

        ordered_indices = sorted(resolved_rows.keys())
        entries = [resolved_rows[idx] for idx in ordered_indices]
        by_index = {str(idx): resolved_rows[idx] for idx in ordered_indices}
        diag_entries = [diag_rows[idx] for idx in ordered_indices]

        cache_hits = sum(1 for row in diag_entries if bool(row.get("cache_hit")))
        cache_misses = len(diag_entries) - cache_hits

        resolved_materials = {
            "resolved_count": len(entries),
            "entries": entries,
            "by_index": by_index,
        }
        diagnostics = {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "entries": diag_entries,
        }
        return resolved_materials, diagnostics

    def _resolve_material_instance(
        self,
        material: Mapping[str, Any],
        wavelengths_um: Sequence[float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        idx = int(material["idx"])
        name = str(material["name"])
        opt = dict(material["opt"]) if isinstance(material.get("opt"), Mapping) else {}
        source_kind = str(opt.get("kind") or "")
        if not source_kind:
            raise ValueError(f"facts_ir.mats[{idx}].opt.kind is required.")

        cache_key = (idx, _legacy._grid_hash(wavelengths_um), _material_signature(material))
        with self._ir_cache_lock:
            cached = self._ir_cache.get(cache_key)
        if cached is not None:
            cached_entry = copy.deepcopy(cached["entry"])
            cached_diag = copy.deepcopy(cached["diag_template"])
            cached_diag["cache_hit"] = True
            return cached_entry, cached_diag

        if source_kind == "nk_const":
            entry, diag_template = self._resolve_inline_nk_const(idx, name, opt, wavelengths_um)
        elif source_kind == "nk_table":
            entry, diag_template = self._resolve_inline_nk_table(idx, name, opt, wavelengths_um)
        elif source_kind == "db_lookup":
            entry, diag_template = self._resolve_db_lookup(idx, name, opt, wavelengths_um)
        else:
            raise MaterialResolveError(
                "material_source_kind_unsupported",
                f"facts_ir.mats[{idx}].opt.kind={source_kind!r} is unsupported; expected nk_const|nk_table|db_lookup."
            )

        with self._ir_cache_lock:
            self._ir_cache[cache_key] = {
                "entry": copy.deepcopy(entry),
                "diag_template": copy.deepcopy(diag_template),
            }
        miss_diag = copy.deepcopy(diag_template)
        miss_diag["cache_hit"] = False
        return entry, miss_diag

    def _resolve_inline_nk_const(
        self,
        idx: int,
        name: str,
        opt: Mapping[str, Any],
        wavelengths_um: Sequence[float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        n_raw = opt.get("n")
        if not isinstance(n_raw, (int, float, str)):
            raise ValueError(f"facts_ir.mats[{idx}] nk_const.n must be a finite number.")
        n = float(n_raw)
        k_raw = opt.get("k", 0.0)
        k = float(0.0 if k_raw is None else k_raw)
        if not math.isfinite(n) or not math.isfinite(k):
            raise ValueError(f"facts_ir.mats[{idx}] nk_const n/k must be finite numbers.")
        if k < 0.0:
            raise ValueError(f"facts_ir.mats[{idx}] nk_const.k must be >= 0.")

        n_values = [n for _ in wavelengths_um]
        k_values = [k for _ in wavelengths_um]
        entry = {
            "idx": idx,
            "name": name,
            "source_kind": "inline_nk_const",
            "wavelength_um": [float(wl) for wl in wavelengths_um],
            "n": n_values,
            "k": k_values,
            "selected_page_id": None,
            "selected_record_id": None,
            "selection": None,
        }
        diag = {
            "idx": idx,
            "source_kind": "inline_nk_const",
            "cache_hit": False,
            "candidate_count": 0,
            "selected_page_id": None,
            "selected_record_id": None,
            "selection_reason": "inline_nk_const",
            "ranking_reasons": [],
        }
        return entry, diag

    def _resolve_inline_nk_table(
        self,
        idx: int,
        name: str,
        opt: Mapping[str, Any],
        wavelengths_um: Sequence[float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        n_values, k_values = _nk_table_values(opt, wavelengths_um)
        entry = {
            "idx": idx,
            "name": name,
            "source_kind": "inline_nk_table",
            "wavelength_um": [float(wl) for wl in wavelengths_um],
            "n": n_values,
            "k": k_values,
            "selected_page_id": None,
            "selected_record_id": None,
            "selection": None,
        }
        diag = {
            "idx": idx,
            "source_kind": "inline_nk_table",
            "cache_hit": False,
            "candidate_count": 0,
            "selected_page_id": None,
            "selected_record_id": None,
            "selection_reason": "inline_nk_table",
            "ranking_reasons": [],
        }
        return entry, diag

    def _resolve_db_lookup(
        self,
        idx: int,
        name: str,
        opt: Mapping[str, Any],
        wavelengths_um: Sequence[float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        db_key = str(opt.get("db_key") or "").strip()
        if not db_key:
            raise MaterialResolveError(
                "db_lookup_key_missing",
                f"facts_ir.mats[{idx}] db_lookup requires non-empty db_key.",
                details={"idx": idx},
            )

        search_result = self._index.search(db_key)
        matches = list(getattr(search_result, "matches", ()) or ())
        if not matches:
            raise MaterialResolveError(
                "db_lookup_no_match",
                f"No material matches for db_lookup key {db_key!r}.",
                details={"idx": idx, "db_key": db_key},
            )

        wl_min = min(float(value) for value in wavelengths_um)
        wl_max = max(float(value) for value in wavelengths_um)
        query_norm = _legacy.normalize_query(db_key)

        ranked: list[dict[str, Any]] = []
        for match in matches:
            page = _match_candidate_page(match, wavelengths_um)
            page_id_raw = getattr(page, "page_id", None)
            page_id = int(page_id_raw) if isinstance(page_id_raw, int) else None
            coverage_rank = _legacy._coverage_rank(
                _legacy._page_range(page),
                wl_min,
                wl_max,
            )
            exact_name_match = 1 if _legacy._is_exact_match(query_norm, match) else 0
            quality_score = float(getattr(match, "score", 0.0) or 0.0)
            stable_id = page_id if page_id is not None else 2**31 - 1
            sort_key = (
                -coverage_rank,
                -exact_name_match,
                -quality_score,
                stable_id,
                int(getattr(match, "material_id", 2**31 - 1)),
                str(getattr(page, "page", "")),
            )
            ranked.append(
                {
                    "sort_key": sort_key,
                    "shelf": str(getattr(match, "shelf", "")),
                    "book": str(getattr(match, "book", "")),
                    "page": str(getattr(page, "page", "")),
                    "page_id": page_id,
                    "score": quality_score,
                    "coverage_rank": int(coverage_rank),
                    "exact_name_match": int(exact_name_match),
                }
            )

        ranked.sort(key=lambda row: row["sort_key"])
        chosen = ranked[0]

        selection = MaterialSelection(
            shelf=str(chosen["shelf"]),
            book=str(chosen["book"]),
            page=str(chosen["page"]),
            reason="db_lookup_ranked",
            score=float(chosen["score"]),
        )
        resolved = self._resolve_choice(
            MaterialChoice(
                shelf=str(chosen["shelf"]),
                book=str(chosen["book"]),
                page=str(chosen["page"]),
            ),
            selection,
            wavelengths_um,
        )

        ranking_reasons: list[dict[str, Any]] = []
        for rank, candidate in enumerate(ranked):
            page_id = candidate["page_id"]
            record_id = (
                f"page_id:{int(page_id)}"
                if isinstance(page_id, int)
                else f"ref:{candidate['shelf']}/{candidate['book']}/{candidate['page']}"
            )
            ranking_reasons.append(
                {
                    "rank": rank,
                    "record_id": record_id,
                    "page_id": page_id,
                    "coverage_rank": int(candidate["coverage_rank"]),
                    "exact_name_match": int(candidate["exact_name_match"]),
                    "quality_score": float(candidate["score"]),
                    "shelf": str(candidate["shelf"]),
                    "book": str(candidate["book"]),
                    "page": str(candidate["page"]),
                }
            )

        selected_page_id = int(chosen["page_id"]) if isinstance(chosen["page_id"], int) else None
        selected_record_id = (
            f"page_id:{selected_page_id}"
            if selected_page_id is not None
            else f"ref:{chosen['shelf']}/{chosen['book']}/{chosen['page']}"
        )
        entry = {
            "idx": idx,
            "name": name,
            "source_kind": "db_lookup",
            "wavelength_um": [float(wl) for wl in resolved.wavelengths_um],
            "n": [float(v) for v in resolved.n],
            "k": [float(v) for v in resolved.k],
            "selected_page_id": selected_page_id,
            "selected_record_id": selected_record_id,
            "selection": {
                "shelf": resolved.selection.shelf,
                "book": resolved.selection.book,
                "page": resolved.selection.page,
                "score": resolved.selection.score,
                "reason": resolved.selection.reason,
            },
        }
        diag = {
            "idx": idx,
            "source_kind": "db_lookup",
            "cache_hit": False,
            "candidate_count": len(ranked),
            "selected_page_id": selected_page_id,
            "selected_record_id": selected_record_id,
            "selection_reason": resolved.selection.reason,
            "ranking_reasons": ranking_reasons,
        }
        return entry, diag


__all__ = [
    "MaterialChoice",
    "MaterialSelection",
    "ResolvedMaterial",
    "MaterialResolver",
]
