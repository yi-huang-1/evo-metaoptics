"""Material database lookup tool for the code-gen agent.

Provides a callable tool that searches the refractiveindex.info database
and returns a ``page_id`` plus metadata.  The agent uses the page_id with
``get_material_nk()`` in solution.py to fetch n/k at runtime.
"""

from __future__ import annotations

import json
import math
from typing import Any

from evo_metaoptics.material_db import (
    MaterialIndex,
    load_material_db_settings,
)
from evo_metaoptics.material_db.paths import default_source_path
from evo_metaoptics.material_db.search import MaterialMatch


def build_material_lookup_tool() -> Any:
    """Create a deterministic search-only tool for material lookup.

    Returns a callable ``lookup_material_nk(material_name, wavelengths_um)``
    that searches the refractiveindex.info database and returns a page_id
    plus metadata as a JSON string.  The agent then uses the page_id in
    ``get_material_nk(page_id, wavelengths_um)`` at runtime.
    """

    # Lazy-initialized shared state (initialized on first call, cached after).
    _state: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized() -> tuple[MaterialIndex | None, str | None]:
        """Initialize material DB access on first call."""
        if "index" in _state:
            return _state["index"], _state.get("init_error")
        try:
            settings = load_material_db_settings()
            index = MaterialIndex(db_path=settings.db_path)
            _state.update(index=index, init_error=None)
            return index, None
        except Exception as exc:
            _state.update(index=None, init_error=str(exc))
            return None, str(exc)

    def _select_best_page(match: MaterialMatch) -> Any:
        """Pick the best page from a match, preferring pages with n+k data."""
        pages = list(match.pages)
        if not pages:
            return None
        for page in pages:
            if page.has_n and page.has_k:
                return page
        return pages[0]

    def _parse_wavelengths(raw: str) -> tuple[list[float], str | None]:
        """Parse wavelengths from a JSON-array or comma-separated string."""
        text = str(raw or "").strip()
        if not text:
            return [], "wavelengths_um must not be empty."
        wavelengths: list[float] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (int, float)):
                wavelengths = [float(parsed)]
            elif isinstance(parsed, list):
                wavelengths = [float(v) for v in parsed]
            else:
                return [], "Expected a number or array of numbers."
        except (json.JSONDecodeError, ValueError, TypeError):
            # Comma-separated fallback
            for part in text.replace("[", "").replace("]", "").split(","):
                part = part.strip()
                if part:
                    try:
                        wavelengths.append(float(part))
                    except ValueError:
                        return [], f"Cannot parse wavelength: {part!r}."

        if not wavelengths:
            return [], "No valid wavelength values found."
        for wl in wavelengths:
            if not math.isfinite(wl) or wl <= 0:
                return [], f"Wavelength {wl} is invalid; must be positive (in um)."
        return wavelengths, None

    # ------------------------------------------------------------------
    # Public tool function
    # ------------------------------------------------------------------

    def lookup_material_nk(material_name: str, wavelengths_um: str) -> str:
        """Search for a material and return its page_id plus metadata.

        Searches the refractiveindex.info database for the given material,
        filtered by wavelength coverage.  Returns a JSON string with
        ``page_id``, shelf/book/page identifiers, wavelength coverage,
        and alternative matches.

        The agent should use the returned ``page_id`` with
        ``get_material_nk(page_id, wavelengths_um)`` in solution.py
        to fetch n/k data at runtime.

        Args:
            material_name: Material name or chemical formula
                (e.g. ``"SiO2"``, ``"TiO2"``, ``"Gold"``, ``"Si3N4"``).
            wavelengths_um: Wavelengths in micrometers as a JSON array string
                (e.g. ``"[1.55]"`` or ``"[0.4, 0.5, 0.6, 0.7]"``).

        Returns:
            JSON string with page_id, metadata, and alternatives.
        """
        # --- validate inputs ---
        name = str(material_name or "").strip()
        if not name:
            return json.dumps({"status": "error", "error": "material_name must be non-empty."})

        wavelengths, wl_error = _parse_wavelengths(wavelengths_um)
        if wl_error:
            return json.dumps({"status": "error", "error": wl_error})

        # --- initialize DB ---
        index, init_error = _ensure_initialized()
        if index is None:
            return json.dumps(
                {"status": "error", "error": f"Material DB unavailable: {init_error}"}
            )

        # --- search ---
        wl_min, wl_max = min(wavelengths), max(wavelengths)
        try:
            result = index.search(
                name,
                limit=3,
                min_wavelength_um=wl_min,
                max_wavelength_um=wl_max,
            )
        except Exception as exc:
            return json.dumps({"status": "error", "error": f"Search failed: {exc}"})

        matches = list(result.matches)
        if not matches:
            # Retry without wavelength filter for a better error message.
            try:
                broad = index.search(name, limit=3)
                broad_names = [m.book for m in broad.matches]
            except Exception:
                broad_names = []
            msg = f"No matches for '{name}' at {wavelengths} um."
            if broad_names:
                msg += f" Found without wavelength filter: {broad_names}."
            else:
                msg += " Try: SiO2, Si, TiO2, GaN, Si3N4, Au, Ag, Al2O3."
            return json.dumps({"status": "error", "error": msg})

        # --- select best match and page ---
        best_match = matches[0]
        best_page = _select_best_page(best_match)
        if best_page is None:
            return json.dumps(
                {"status": "error", "error": f"'{best_match.book}' has no data pages."}
            )

        # --- build search-only response ---
        coverage = [best_page.coverage_min, best_page.coverage_max]
        alternatives = [
            {
                "name": m.book,
                "page_id": (_select_best_page(m).page_id if _select_best_page(m) else None),
                "score": m.score,
            }
            for m in matches[1:]
        ]

        return json.dumps(
            {
                "status": "ok",
                "page_id": best_page.page_id,
                "shelf": best_match.shelf,
                "book": best_match.book,
                "page": best_page.page,
                "wavelength_coverage_um": coverage,
                "has_n": best_page.has_n,
                "has_k": best_page.has_k,
                "alternatives": alternatives,
            }
        )

    return lookup_material_nk


__all__ = ["build_material_lookup_tool"]
