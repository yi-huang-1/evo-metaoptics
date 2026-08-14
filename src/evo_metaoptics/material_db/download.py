from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable
import urllib.request

try:  # pragma: no cover - exercised via mocks in tests.
    from refractiveindex import RefractiveIndex  # type: ignore
except Exception:  # pragma: no cover
    try:  # fallback for versions that only expose the class in the submodule
        from refractiveindex.refractiveindex import RefractiveIndex  # type: ignore
    except Exception:
        RefractiveIndex = None

from .index_db import build_index
from .settings import load_material_db_settings


class MaterialDbError(RuntimeError):
    pass


LOGGER = logging.getLogger(__name__)


__all__ = [
    "ensure_refractiveindex_db",
    "ensure_material_db_ready",
    "MaterialDbError",
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _wrap_urlretrieve(
    printer: Callable[[str, Path, str], None],
) -> Callable[..., tuple[str, object]]:
    original = urllib.request.urlretrieve

    def _wrapped(
        url: str,
        filename: str | None = None,
        reporthook=None,
        data=None,
    ):
        result = original(url, filename, reporthook, data)
        file_path = Path(filename or result[0])
        sha256 = _sha256_file(file_path)
        printer(url, file_path, sha256)
        return result

    return _wrapped


def ensure_refractiveindex_db(root: Path, *, auto_download: bool = True) -> Path:
    """Ensure the refractiveindex.info YAML database exists under root.

    Args:
        root: Target database directory (contains catalog-nk.yml and data-nk/).
        auto_download: If True, allow refractiveindex to download the DB when missing.

    Returns:
        The resolved database root path.

    Raises:
        FileNotFoundError: When the DB is missing and auto_download is False.
        MaterialDbError: When the DB cannot be validated or dependency is missing.
    """
    root = Path(root).expanduser().resolve()
    catalog_path = root / "catalog-nk.yml"
    data_root = root / "data-nk"

    missing = not root.exists() or not catalog_path.exists() or not data_root.exists()
    if missing:
        if not auto_download:
            raise FileNotFoundError(
                f"Missing refractiveindex DB at {root} (set auto_download=True to fetch)."
            )
        if RefractiveIndex is None:  # pragma: no cover - exercised via mock.
            raise MaterialDbError(
                "Missing refractiveindex dependency; install `refractiveindex`."
            )
        kwargs = {"databasePath": str(root), "auto_download": True}
        if root.exists() and (not catalog_path.exists() or not data_root.exists()):
            # Force refresh when directory exists but is incomplete.
            kwargs["update_database"] = True

        def _log_download(url: str, file_path: Path, sha256: str) -> None:
            url_name = Path(url).name
            LOGGER.info(
                "Downloaded %s -> %s (sha256=%s)",
                url_name,
                file_path.name,
                sha256,
            )

        original = urllib.request.urlretrieve
        urllib.request.urlretrieve = _wrap_urlretrieve(_log_download)
        try:
            RefractiveIndex(**kwargs)
        finally:
            urllib.request.urlretrieve = original

    if not catalog_path.exists() or not data_root.exists():
        raise MaterialDbError(
            f"Invalid refractiveindex DB at {root}: missing catalog-nk.yml or data-nk/."
        )

    return root


def ensure_material_db_ready() -> None:
    """Ensure the material database SQLite index is built and ready.

    This function composes the following steps:
    1. Load material DB settings (respects env vars and defaults)
    2. Check if SQLite index already exists (early exit if present)
    3. Ensure refractiveindex YAML database is available
    4. Build the SQLite index from the YAML database

    Raises:
        FileNotFoundError: When the YAML database is missing and auto_download is False.
        MaterialDbError: When the database cannot be validated or dependency is missing.
    """
    settings = load_material_db_settings()

    # Early exit if SQLite index already exists (prevents wasteful rebuild)
    if settings.db_path.exists():
        return

    # Ensure YAML database is available
    ensure_refractiveindex_db(settings.source_root, auto_download=settings.auto_download)

    # Build SQLite index from YAML database
    LOGGER.info("Building material DB index: %s", settings.db_path)
    build_index(output_db=settings.db_path, source_root=settings.source_root)
