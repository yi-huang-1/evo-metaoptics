from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Dict


@dataclass(frozen=True)
class NkPage:
    page: str
    page_name: Optional[str]
    data_path: Optional[str]


@dataclass(frozen=True)
class NkMaterial:
    shelf: str
    book: str
    display_name: Optional[str]
    info_relpath: Optional[str]
    pages: Tuple[NkPage, ...]


def _iter_pages(items: Optional[List[Dict[str, Any]]]) -> Iterable[NkPage]:
    if not items:
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        if "PAGE" in item:
            yield NkPage(
                page=str(item["PAGE"]),
                page_name=item.get("name"),
                data_path=item.get("data"),
            )
        nested = item.get("content")
        if isinstance(nested, list):
            yield from _iter_pages(nested)


def iter_nk_materials(catalog_nk: list[dict[str, Any]]) -> Iterable[NkMaterial]:
    for shelf_node in catalog_nk:
        if not isinstance(shelf_node, dict) or "SHELF" not in shelf_node:
            continue
        shelf = str(shelf_node["SHELF"])
        content = shelf_node.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict) or "BOOK" not in item:
                continue
            book = str(item["BOOK"])
            content = item.get("content")
            pages = tuple(_iter_pages(content if isinstance(content, list) else None))
            yield NkMaterial(
                shelf=shelf,
                book=book,
                display_name=item.get("name"),
                info_relpath=item.get("info"),
                pages=pages,
            )


def load_catalog_nk(path: Path) -> List[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for YAML parsing. Install with `uv add pyyaml` (or `pip install pyyaml`)."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected YAML root type in {path}: expected list, got {type(data).__name__}")
    return data
