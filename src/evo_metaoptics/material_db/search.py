import sqlite3
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .paths import default_db_path
from .text_normalize import normalize_query

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None


@dataclass(frozen=True)
class MaterialPage:
    page: str
    page_name: Optional[str]
    data_path: Optional[str]
    page_id: Optional[int] = None
    coverage_min: Optional[float] = None
    coverage_max: Optional[float] = None
    has_n: Optional[bool] = None
    has_k: Optional[bool] = None


@dataclass(frozen=True)
class MaterialMatch:
    score: float
    material_id: int
    shelf: str
    book: str
    display_name: Optional[str]
    name_plain: Optional[str]
    title: Optional[str]
    description: Optional[str]
    other_names: Tuple[str, ...]
    alias_hits: Tuple[str, ...]
    pages: Tuple[MaterialPage, ...]


@dataclass(frozen=True)
class MaterialSearchResult:
    query: str
    normalized_query: str
    matches: Tuple[MaterialMatch, ...]


@dataclass(frozen=True)
class MaterialPageRef:
    page_id: int
    material_id: int
    shelf: str
    book: str
    page: str
    page_name: Optional[str]
    data_path: Optional[str]
    coverage_min: Optional[float]
    coverage_max: Optional[float]
    has_n: Optional[bool] = None
    has_k: Optional[bool] = None


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        return float(fuzz.WRatio(a, b))

    import difflib  # local import for fallback only

    return difflib.SequenceMatcher(a=a, b=b).ratio() * 100.0


_SPLIT_RE = re.compile(r"[(),;/]|\s+-\s+")


def _candidate_terms_from_name_plain(name_plain: str) -> List[str]:
    if not name_plain:
        return []
    parts = []
    for piece in _SPLIT_RE.split(name_plain):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) == 1:
            continue
        parts.append(piece)
    return parts


def _score_material(query_norm: str, row: Dict[str, Any], aliases: Sequence[str]) -> float:
    book_raw = str(row.get("book", ""))
    book_norm = normalize_query(book_raw)

    name_plain_raw = str(row.get("name_plain") or "")
    name_plain_norm = normalize_query(name_plain_raw)

    title_raw = str(row.get("title") or "")
    title_norm = normalize_query(title_raw)

    candidate_norms: List[str] = []
    for c in (book_norm, title_norm, name_plain_norm):
        if c:
            candidate_norms.append(c)

    for piece in _candidate_terms_from_name_plain(name_plain_raw):
        c = normalize_query(piece)
        if c:
            candidate_norms.append(c)

    for alias in aliases:
        c = normalize_query(alias)
        if c:
            candidate_norms.append(c)

    deduped: List[str] = []
    seen = set()
    for c in candidate_norms:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)

    best = 0.0
    if query_norm == book_norm and book_norm:
        best = 100.0
    for c in deduped:
        best = max(best, _similarity(query_norm, c))
    return best


def _alias_hits(query_norm: str, aliases: Sequence[str]) -> Tuple[str, ...]:
    if not aliases or not query_norm:
        return ()
    hits: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        alias_norm = normalize_query(alias)
        if not alias_norm:
            continue
        if query_norm in alias_norm or alias_norm in query_norm:
            if alias in seen:
                continue
            seen.add(alias)
            hits.append(alias)
    return tuple(hits)


def search_materials(
    query: str,
    *,
    db_path: Optional[Path] = None,
    limit: int = 3,
    min_wavelength_um: float | None = None,
    max_wavelength_um: float | None = None,
) -> MaterialSearchResult:
    db_path = (db_path or default_db_path()).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path} (build it first)")
    if limit <= 0:
        raise ValueError("limit must be > 0")

    query_norm = normalize_query(query)
    if not query_norm:
        raise ValueError("query must be non-empty")
    if min_wavelength_um is not None and max_wavelength_um is not None:
        if float(min_wavelength_um) > float(max_wavelength_um):
            raise ValueError("min_wavelength_um must be <= max_wavelength_um")
    conn = sqlite3.connect(str(db_path))
    try:
        candidate_ids = _prefilter_material_ids(conn, query, query_norm)
        if not candidate_ids:
            return MaterialSearchResult(query=query, normalized_query=query_norm, matches=())
        base_by_id = _load_material_details(
            conn,
            candidate_ids,
            min_wavelength_um=min_wavelength_um,
            max_wavelength_um=max_wavelength_um,
        )
        scored: List[Tuple[float, int]] = []
        alias_hits_by_id: Dict[int, Tuple[str, ...]] = {}
        for material_id, base in base_by_id.items():
            row = base.to_row(material_id)
            score = _score_material(query_norm, row, base.other_names)
            scored.append((score, material_id))
            alias_hits_by_id[material_id] = _alias_hits(query_norm, base.other_names)
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = scored[:limit]
        if not top:
            return MaterialSearchResult(query=query, normalized_query=query_norm, matches=())
        return MaterialSearchResult(
            query=query,
            normalized_query=query_norm,
            matches=tuple(
                base_by_id[material_id].with_score(
                    material_id,
                    score,
                    alias_hits_by_id.get(material_id, ()),
                )
                for score, material_id in top
                if material_id in base_by_id
            ),
        )
    finally:
        conn.close()


def get_page_by_id(page_id: int, *, db_path: Optional[Path] = None) -> Optional[MaterialPageRef]:
    db_path = (db_path or default_db_path()).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path} (build it first)")
    if page_id <= 0:
        raise ValueError("page_id must be > 0")
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT p.page_id, p.material_id, m.shelf, m.book, p.page, p.page_name, p.data_path,
                   p.coverage_min, p.coverage_max, p.has_n, p.has_k
            FROM pages p
            JOIN materials m ON m.material_id = p.material_id
            WHERE p.page_id = ?
            """,
            (int(page_id),),
        ).fetchone()
        if not row:
            return None
        return MaterialPageRef(
            page_id=int(row[0]),
            material_id=int(row[1]),
            shelf=str(row[2]),
            book=str(row[3]),
            page=str(row[4]),
            page_name=row[5],
            data_path=row[6],
            coverage_min=row[7],
            coverage_max=row[8],
            has_n=bool(row[9]) if row[9] is not None else None,
            has_k=bool(row[10]) if row[10] is not None else None,
        )
    finally:
        conn.close()


def get_page_by_ref(
    shelf: str,
    book: str,
    page: str,
    *,
    db_path: Optional[Path] = None,
) -> Optional[MaterialPageRef]:
    db_path = (db_path or default_db_path()).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path} (build it first)")
    if not shelf or not book or not page:
        raise ValueError("shelf, book, and page must be non-empty strings")
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT p.page_id, p.material_id, m.shelf, m.book, p.page, p.page_name, p.data_path,
                   p.coverage_min, p.coverage_max, p.has_n, p.has_k
            FROM pages p
            JOIN materials m ON m.material_id = p.material_id
            WHERE m.shelf = ? AND m.book = ? AND p.page = ?
            """,
            (str(shelf), str(book), str(page)),
        ).fetchone()
        if not row:
            return None
        return MaterialPageRef(
            page_id=int(row[0]),
            material_id=int(row[1]),
            shelf=str(row[2]),
            book=str(row[3]),
            page=str(row[4]),
            page_name=row[5],
            data_path=row[6],
            coverage_min=row[7],
            coverage_max=row[8],
            has_n=bool(row[9]) if row[9] is not None else None,
            has_k=bool(row[10]) if row[10] is not None else None,
        )
    finally:
        conn.close()


def _prefilter_material_ids(conn: sqlite3.Connection, query: str, query_norm: str) -> List[int]:
    query_fold = query.casefold().strip()
    like_norm = f"%{query_norm}%"
    like_fold = f"%{query_fold}%"
    rows = conn.execute(
        """
        SELECT DISTINCT m.material_id
        FROM materials m
        LEFT JOIN aliases a ON a.material_id = m.material_id
        WHERE m.name_norm = ?
           OR m.name_norm LIKE ?
           OR a.alias_norm = ?
           OR a.alias_norm LIKE ?
           OR LOWER(m.book) = ?
           OR LOWER(m.book) LIKE ?
        ORDER BY m.material_id
        """,
        (
            query_norm,
            like_norm,
            query_norm,
            like_norm,
            query_fold,
            like_fold,
        ),
    ).fetchall()
    return [int(r[0]) for r in rows]


def _load_material_details(
    conn: sqlite3.Connection,
    material_ids: Sequence[int],
    *,
    min_wavelength_um: float | None,
    max_wavelength_um: float | None,
) -> Dict[int, "_MatchBase"]:
    placeholders = ",".join(["?"] * len(material_ids))

    material_rows = conn.execute(
        f"""
        SELECT material_id, shelf, book, display_name, name_plain, title, description
        FROM materials
        WHERE material_id IN ({placeholders})
        """,
        list(material_ids),
    ).fetchall()

    aliases_rows = conn.execute(
        f"""
        SELECT material_id, alias_raw
        FROM aliases
        WHERE material_id IN ({placeholders})
        ORDER BY material_id, alias_raw
        """,
        list(material_ids),
    ).fetchall()

    pages_rows = _load_pages_rows(
        conn,
        material_ids,
        min_wavelength_um=min_wavelength_um,
        max_wavelength_um=max_wavelength_um,
    )

    aliases_by_material: Dict[int, List[str]] = {mid: [] for mid in material_ids}
    for r in aliases_rows:
        aliases_by_material[int(r[0])].append(str(r[1]))

    pages_by_material: Dict[int, List[MaterialPage]] = {mid: [] for mid in material_ids}
    for r in pages_rows:
        pages_by_material[int(r[0])].append(
            MaterialPage(
                page=str(r[1]),
                page_name=r[2],
                data_path=r[3],
                page_id=int(r[4]) if r[4] is not None else None,
                coverage_min=r[5],
                coverage_max=r[6],
                has_n=bool(r[7]) if r[7] is not None else None,
                has_k=bool(r[8]) if r[8] is not None else None,
            )
        )

    out: Dict[int, _MatchBase] = {}
    for r in material_rows:
        material_id = int(r[0])
        pages = tuple(pages_by_material.get(material_id, []))
        if (min_wavelength_um is not None or max_wavelength_um is not None) and not pages:
            continue
        out[material_id] = _MatchBase(
            shelf=str(r[1]),
            book=str(r[2]),
            display_name=r[3],
            name_plain=r[4],
            title=r[5],
            description=r[6],
            other_names=tuple(aliases_by_material.get(material_id, [])),
            pages=pages,
        )
    return out


def _load_pages_rows(
    conn: sqlite3.Connection,
    material_ids: Sequence[int],
    *,
    min_wavelength_um: float | None,
    max_wavelength_um: float | None,
) -> list[tuple]:
    placeholders = ",".join(["?"] * len(material_ids))
    clauses = [f"material_id IN ({placeholders})"]
    params: list[object] = list(material_ids)
    if min_wavelength_um is not None or max_wavelength_um is not None:
        clauses.append("coverage_min IS NOT NULL")
        clauses.append("coverage_max IS NOT NULL")
        if min_wavelength_um is not None:
            clauses.append("coverage_min <= ? AND coverage_max >= ?")
            params.extend([float(min_wavelength_um), float(min_wavelength_um)])
        if max_wavelength_um is not None:
            clauses.append("coverage_min <= ? AND coverage_max >= ?")
            params.extend([float(max_wavelength_um), float(max_wavelength_um)])
    where_sql = " AND ".join(clauses)
    return conn.execute(
        f"""
        SELECT material_id, page, page_name, data_path, page_id, coverage_min, coverage_max, has_n, has_k
        FROM pages
        WHERE {where_sql}
        ORDER BY material_id, page
        """,
        params,
    ).fetchall()


@dataclass(frozen=True)
class _MatchBase:
    shelf: str
    book: str
    display_name: Optional[str]
    name_plain: Optional[str]
    title: Optional[str]
    description: Optional[str]
    other_names: Tuple[str, ...]
    pages: Tuple[MaterialPage, ...]

    def with_score(
        self,
        material_id: int,
        score: float,
        alias_hits: Tuple[str, ...],
    ) -> MaterialMatch:
        return MaterialMatch(
            score=score,
            material_id=material_id,
            shelf=self.shelf,
            book=self.book,
            display_name=self.display_name,
            name_plain=self.name_plain,
            title=self.title,
            description=self.description,
            other_names=self.other_names,
            alias_hits=alias_hits,
            pages=self.pages,
        )

    def to_row(self, material_id: int) -> Dict[str, Any]:
        return {
            "material_id": material_id,
            "shelf": self.shelf,
            "book": self.book,
            "display_name": self.display_name,
            "name_plain": self.name_plain,
            "title": self.title,
            "description": self.description,
        }
