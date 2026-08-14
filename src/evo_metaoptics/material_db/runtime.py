"""Runtime n/k lookup helper for agent-generated solution.py code.

This module provides ``get_material_nk``, a thin helper that solution.py
imports to fetch refractive-index data at *execution time* (not at
tool-call time).  The agent obtains a ``page_id`` from the
``lookup_material_nk`` tool during planning and embeds it in the
generated code; at runtime the helper reads n/k from the YAML database.

Usage in solution.py::

    from evo_metaoptics.material_db.runtime import get_material_nk

    nk = get_material_nk(page_id=123, wavelengths_um=[1.55])
    # nk == {"n": [1.444], "k": [0.0], "book": "SiO2", ...}
"""

from __future__ import annotations

import math
from typing import Any

from .search import get_page_by_id
from .settings import load_material_db_settings


def get_material_nk(page_id: int, wavelengths_um: list[float]) -> dict[str, Any]:
    """Fetch n/k for a material page at the given wavelengths.

    Parameters
    ----------
    page_id:
        Integer page ID from the material-database index (obtained via the
        ``lookup_material_nk`` tool during planning).
    wavelengths_um:
        Wavelengths in **micrometers** at which to evaluate n and k.

    Returns
    -------
    dict with keys:
        ``n``  – list[float] of refractive-index values
        ``k``  – list[float] of extinction coefficients (0.0 when absent)
        ``book``, ``page``, ``shelf`` – material identifiers
        ``page_id`` – echo of the input page_id

    Raises
    ------
    TypeError
        If *page_id* is not an int or *wavelengths_um* is not a list.
    ValueError
        If *page_id* ≤ 0, *wavelengths_um* is empty, or any wavelength is
        non-positive / non-finite.  Also raised if the requested wavelengths
        are outside the material's valid range (propagated from the
        ``refractiveindex`` package).
    FileNotFoundError
        If the material-database index or YAML source files cannot be found.
    RuntimeError
        If the ``refractiveindex`` package is not installed, or if the
        page_id does not correspond to a known material page.
    """

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not isinstance(page_id, int):
        raise TypeError(f"page_id must be int, got {type(page_id).__name__}")
    if page_id <= 0:
        raise ValueError(f"page_id must be > 0, got {page_id}")

    if not isinstance(wavelengths_um, list):
        raise TypeError(
            f"wavelengths_um must be a list of floats, got {type(wavelengths_um).__name__}"
        )
    if not wavelengths_um:
        raise ValueError("wavelengths_um must be non-empty")
    for wl in wavelengths_um:
        if not isinstance(wl, (int, float)):
            raise TypeError(f"Each wavelength must be a number, got {type(wl).__name__}")
        if not math.isfinite(wl) or wl <= 0:
            raise ValueError(f"Wavelength must be positive and finite, got {wl}")

    # ------------------------------------------------------------------
    # 2. Resolve page via material-db index
    # ------------------------------------------------------------------
    settings = load_material_db_settings()

    page_ref = get_page_by_id(page_id, db_path=settings.db_path)
    if page_ref is None:
        raise RuntimeError(
            f"No material page found for page_id={page_id}. "
            "The page_id may be invalid or the index may need rebuilding."
        )

    # ------------------------------------------------------------------
    # 3. Load the refractiveindex material object
    # ------------------------------------------------------------------
    try:
        from refractiveindex import RefractiveIndexMaterial
    except ImportError as exc:
        raise RuntimeError(
            "The 'refractiveindex' package is required for runtime n/k lookup. "
            "Install it with: pip install refractiveindex"
        ) from exc

    try:
        from refractiveindex.refractiveindex import NoExtinctionCoefficient
    except ImportError:
        NoExtinctionCoefficient = None  # type: ignore[assignment,misc]

    source_root = settings.source_root
    try:
        material_obj = RefractiveIndexMaterial(
            shelf=page_ref.shelf,
            book=page_ref.book,
            page=page_ref.page,
            databasePath=source_root,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load material data for page_id={page_id} "
            f"({page_ref.shelf}/{page_ref.book}/{page_ref.page}): {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 4. Fetch n/k at each wavelength (um → nm for the package)
    # ------------------------------------------------------------------
    n_values: list[float] = []
    k_values: list[float] = []

    for wl_um in wavelengths_um:
        wl_nm = float(wl_um) * 1000.0

        # Refractive index (n)
        n = float(material_obj.get_refractive_index(wl_nm))
        if not math.isfinite(n):
            raise ValueError(
                f"Invalid refractive index at {wl_um} um: n={n!r} "
                f"(page_id={page_id}, {page_ref.book}/{page_ref.page})"
            )

        # Extinction coefficient (k) — fallback to 0.0 when absent
        try:
            k_raw = material_obj.get_extinction_coefficient(wl_nm)
        except Exception as exc:
            if NoExtinctionCoefficient is not None and isinstance(
                exc, NoExtinctionCoefficient
            ):
                k_raw = None
            elif (
                NoExtinctionCoefficient is None
                and exc.__class__.__name__ == "NoExtinctionCoefficient"
            ):
                k_raw = None
            else:
                raise
        k = 0.0 if k_raw is None else float(k_raw)
        if not math.isfinite(k):
            raise ValueError(
                f"Invalid extinction coefficient at {wl_um} um: k={k!r} "
                f"(page_id={page_id}, {page_ref.book}/{page_ref.page})"
            )

        n_values.append(n)
        k_values.append(k)

    # ------------------------------------------------------------------
    # 5. Return plain dict
    # ------------------------------------------------------------------
    return {
        "n": n_values,
        "k": k_values,
        "book": page_ref.book,
        "page": page_ref.page,
        "shelf": page_ref.shelf,
        "page_id": page_id,
    }


__all__ = ["get_material_nk"]
