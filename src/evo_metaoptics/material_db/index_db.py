import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .catalog_nk import NkMaterial, load_catalog_nk, iter_nk_materials
from .html_extract import extract_material_info
from .paths import default_db_path, default_source_path
from .text_normalize import normalize_query, strip_inline_html


SCHEMA_VERSION = 2


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS materials (
  material_id INTEGER PRIMARY KEY,
  shelf TEXT NOT NULL,
  book TEXT NOT NULL,
  display_name TEXT,
  name_plain TEXT,
  info_relpath TEXT,
  title TEXT,
  description TEXT,
  name_norm TEXT,
  UNIQUE(shelf, book)
);

CREATE TABLE IF NOT EXISTS aliases (
  material_id INTEGER NOT NULL,
  alias_raw TEXT NOT NULL,
  alias_norm TEXT NOT NULL,
  PRIMARY KEY(material_id, alias_norm),
  FOREIGN KEY(material_id) REFERENCES materials(material_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pages (
  page_id INTEGER PRIMARY KEY,
  material_id INTEGER NOT NULL,
  page TEXT NOT NULL,
  page_name TEXT,
  data_path TEXT,
  has_n INTEGER NOT NULL,
  has_k INTEGER NOT NULL,
  points_n INTEGER,
  points_k INTEGER,
  n_range_min REAL,
  n_range_max REAL,
  k_range_min REAL,
  k_range_max REAL,
  coverage_min REAL,
  coverage_max REAL,
  UNIQUE(material_id, page),
  FOREIGN KEY(material_id) REFERENCES materials(material_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_materials_book ON materials(book);
CREATE INDEX IF NOT EXISTS idx_materials_name_norm ON materials(name_norm);
CREATE INDEX IF NOT EXISTS idx_aliases_alias_norm ON aliases(alias_norm);
CREATE INDEX IF NOT EXISTS idx_pages_coverage ON pages(coverage_min, coverage_max);
"""


@dataclass(frozen=True)
class BuildStats:
    materials: int
    pages: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _material_info(
    source_root: Path, info_relpath: Optional[str]
) -> Tuple[Optional[str], Optional[str], Tuple[str, ...]]:
    if not info_relpath:
        return None, None, ()
    info_path = source_root / "info" / info_relpath
    if not info_path.exists():
        return None, None, ()
    info = extract_material_info(_read_text(info_path))
    title = info.title or None
    description = info.description or None
    other_names = tuple(info.other_names) if info.other_names else ()
    return title, description, other_names


def build_index(
    *,
    output_db: Optional[Path] = None,
    source_root: Optional[Path] = None,
    overwrite: bool = False,
) -> BuildStats:
    source_root = (source_root or default_source_path()).resolve()
    catalog_path = source_root / "catalog-nk.yml"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing catalog: {catalog_path}")

    output_db = (output_db or default_db_path()).resolve()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        if not overwrite:
            raise FileExistsError(f"Output DB already exists: {output_db} (use --overwrite to replace)")
        output_db.unlink()

    catalog = load_catalog_nk(catalog_path)

    conn = sqlite3.connect(str(output_db))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)

        built_at = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", ("schema_version", str(SCHEMA_VERSION)))
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", ("built_at_utc", built_at))
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", ("source_root", str(source_root)))
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", ("catalog", "catalog-nk.yml"))

        material_count = 0
        page_count = 0

        for material in iter_nk_materials(catalog):
            material_id, other_names = _insert_material(conn, source_root, material)
            _insert_aliases(conn, material_id, other_names)
            material_count += 1
            page_count += _insert_pages(conn, source_root, material_id, material)

        conn.commit()
        return BuildStats(materials=material_count, pages=page_count)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _insert_material(
    conn: sqlite3.Connection, source_root: Path, material: NkMaterial
) -> tuple[int, Tuple[str, ...]]:
    display_name = material.display_name
    name_plain = strip_inline_html(display_name) if display_name else None
    title, description, other_names = _material_info(source_root, material.info_relpath)
    name_norm = normalize_query(name_plain or display_name or material.book)
    cur = conn.execute(
        """
        INSERT INTO materials(shelf, book, display_name, name_plain, info_relpath, title, description, name_norm)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            material.shelf,
            material.book,
            display_name,
            name_plain,
            material.info_relpath,
            title,
            description,
            name_norm,
        ),
    )
    return int(cur.lastrowid), other_names


def _insert_aliases(
    conn: sqlite3.Connection,
    material_id: int,
    other_names: Tuple[str, ...],
) -> None:
    if not other_names:
        return
    rows = []
    for alias in other_names:
        alias_raw = str(alias).strip()
        if not alias_raw:
            continue
        alias_norm = normalize_query(alias_raw)
        if not alias_norm:
            continue
        rows.append((material_id, alias_raw, alias_norm))
    if rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO aliases(material_id, alias_raw, alias_norm)
            VALUES(?, ?, ?)
            """,
            rows,
        )


@dataclass(frozen=True)
class PageStats:
    has_n: bool
    has_k: bool
    points_n: Optional[int]
    points_k: Optional[int]
    n_range: Optional[Tuple[float, float]]
    k_range: Optional[Tuple[float, float]]
    coverage: Optional[Tuple[float, float]]


def _insert_pages(
    conn: sqlite3.Connection, source_root: Path, material_id: int, material: NkMaterial
) -> int:
    if not material.pages:
        return 0
    rows = []
    for p in material.pages:
        stats = _page_stats(source_root, p.data_path)
        rows.append(
            (
                material_id,
                p.page,
                p.page_name,
                p.data_path,
                1 if stats.has_n else 0,
                1 if stats.has_k else 0,
                stats.points_n,
                stats.points_k,
                stats.n_range[0] if stats.n_range else None,
                stats.n_range[1] if stats.n_range else None,
                stats.k_range[0] if stats.k_range else None,
                stats.k_range[1] if stats.k_range else None,
                stats.coverage[0] if stats.coverage else None,
                stats.coverage[1] if stats.coverage else None,
            )
        )
    conn.executemany(
        """
        INSERT INTO pages(
            material_id, page, page_name, data_path,
            has_n, has_k, points_n, points_k,
            n_range_min, n_range_max, k_range_min, k_range_max,
            coverage_min, coverage_max
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _page_stats(source_root: Path, data_path: Optional[str]) -> PageStats:
    if not data_path:
        return PageStats(False, False, None, None, None, None, None)
    yaml_path = source_root / "data-nk" / data_path
    if not yaml_path.exists():
        return PageStats(False, False, None, None, None, None, None)
    payload = _load_yaml(yaml_path)
    data_entries = payload.get("DATA") if isinstance(payload, dict) else None
    if not isinstance(data_entries, list):
        return PageStats(False, False, None, None, None, None, None)
    n_ranges: list[Tuple[float, float]] = []
    k_ranges: list[Tuple[float, float]] = []
    points_n: Optional[int] = None
    points_k: Optional[int] = None
    for entry in data_entries:
        if not isinstance(entry, dict):
            continue
        type_raw = str(entry.get("type", "")).strip()
        if not type_raw:
            continue
        type_parts = type_raw.split()
        kind = type_parts[0].lower()
        if kind == "tabulated":
            wl = _parse_tabulated_wavelengths(entry.get("data"))
            if not wl:
                continue
            wl_min = min(wl)
            wl_max = max(wl)
            suffix = type_parts[1].lower() if len(type_parts) > 1 else "n"
            if suffix in {"n", "nk"}:
                n_ranges.append((wl_min, wl_max))
                points_n = _max_optional(points_n, len(wl))
            if suffix in {"k", "nk"}:
                k_ranges.append((wl_min, wl_max))
                points_k = _max_optional(points_k, len(wl))
        elif kind == "formula":
            range_tuple = _parse_formula_range(entry)
            if range_tuple is None:
                continue
            n_ranges.append(range_tuple)

    n_range = _combine_ranges(n_ranges)
    k_range = _combine_ranges(k_ranges)
    has_n = n_range is not None
    has_k = k_range is not None
    coverage = _coverage_range(n_range, k_range, has_n, has_k)
    return PageStats(
        has_n=has_n,
        has_k=has_k,
        points_n=points_n,
        points_k=points_k,
        n_range=n_range,
        k_range=k_range,
        coverage=coverage,
    )


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for YAML parsing. Install with `uv add pyyaml` (or `pip install pyyaml`)."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _parse_tabulated_wavelengths(data_blob: object) -> list[float]:
    if not isinstance(data_blob, str):
        return []
    wavelengths: list[float] = []
    for line in data_blob.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            wavelengths.append(float(parts[0]))
        except ValueError:
            continue
    return wavelengths


def _parse_formula_range(entry: dict) -> Optional[Tuple[float, float]]:
    for key in ("wavelength_range", "range"):
        if key in entry:
            raw = entry.get(key)
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                try:
                    low = float(raw[0])
                    high = float(raw[1])
                    return (low, high)
                except (TypeError, ValueError):
                    return None
            if isinstance(raw, str):
                parts = raw.split()
                if len(parts) >= 2:
                    try:
                        return (float(parts[0]), float(parts[1]))
                    except (TypeError, ValueError):
                        return None
    return None


def _combine_ranges(ranges: list[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not ranges:
        return None
    lows = [r[0] for r in ranges]
    highs = [r[1] for r in ranges]
    return (min(lows), max(highs))


def _coverage_range(
    n_range: Optional[Tuple[float, float]],
    k_range: Optional[Tuple[float, float]],
    has_n: bool,
    has_k: bool,
) -> Optional[Tuple[float, float]]:
    if has_n and has_k and n_range and k_range:
        low = max(n_range[0], k_range[0])
        high = min(n_range[1], k_range[1])
        if low <= high:
            return (low, high)
        return None
    if has_n and n_range:
        return n_range
    return None


def _max_optional(current: Optional[int], candidate: int) -> int:
    if current is None:
        return candidate
    return max(current, candidate)
