from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .download import ensure_refractiveindex_db
from .index_db import BuildStats, build_index
from .paths import default_db_path
from .search import MaterialSearchResult, search_materials
from .settings import load_material_db_settings


@dataclass(frozen=True)
class MaterialIndex:
    db_path: Path = default_db_path()
    default_limit: int = 10

    @classmethod
    def default(cls) -> "MaterialIndex":
        settings = load_material_db_settings()
        return cls(db_path=settings.db_path)

    def build(
        self,
        *,
        source_root: Optional[Path] = None,
        overwrite: bool = False,
        auto_download: Optional[bool] = None,
    ) -> BuildStats:
        settings = load_material_db_settings()
        root = source_root or settings.source_root
        allow_download = settings.auto_download if auto_download is None else auto_download
        root = ensure_refractiveindex_db(root, auto_download=allow_download)
        return build_index(output_db=self.db_path, source_root=root, overwrite=overwrite)

    def search(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        min_wavelength_um: float | None = None,
        max_wavelength_um: float | None = None,
    ) -> MaterialSearchResult:
        return search_materials(
            query,
            db_path=self.db_path,
            limit=limit or self.default_limit,
            min_wavelength_um=min_wavelength_um,
            max_wavelength_um=max_wavelength_um,
        )
