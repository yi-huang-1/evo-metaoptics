# Material DB (SQLite-first)

This module builds and queries a SQLite index over the refractiveindex.info YAML database.
It is designed to be usable on its own (no agent/evaluator dependencies), while still
supporting downstream runtime resolution of n/k profiles.

## What it provides
- SQLite index builder for names, aliases, and wavelength coverage metadata.
- Search API (SQL prefilter + optional Python scoring) returning material/page IDs.
- Page lookup helpers for `page_id` or `(shelf, book, page)`.
- Auto-download wiring via the `refractiveindex` package.

## Default paths
By default, paths are rooted in the repo:
- YAML DB: `resources/material-db/database/refractiveindex.info-database`
- SQLite index: `resources/material-db/database/nk_index.sqlite`

These can be overridden in config:
```
material_db:
  path: null            # override SQLite index path
  source_root: null     # override YAML DB root path
  auto_download: true   # allow auto-download when missing
```

## Build the index
Programmatic:
```python
from evo_metaoptics.material_db import MaterialIndex

index = MaterialIndex.default()
index.build(overwrite=True)
```

CLI demos:
```
uv run external_scrpits/material_db_build_demo.py --overwrite
```

## Search for materials
```python
from evo_metaoptics.material_db import search_materials

result = search_materials(
    "SiO2",
    limit=5,
    min_wavelength_um=0.5,
    max_wavelength_um=0.7,
)

for match in result.matches:
    print(match.material_id, match.book, match.alias_hits)
    for page in match.pages:
        print("  page_id=", page.page_id, "page=", page.page)
```

CLI demo:
```
uv run external_scrpits/material_db_search_demo.py SiO2 --min-wavelength-um 0.5 --max-wavelength-um 0.7
```

## Use page IDs (preferred)
The search results include `page_id` and `(shelf, book, page)` references.
Downstream systems should prefer `page_id` when possible:
- `{"type": "page_id", "value": 123}` (preferred)
- `{"type": "page", "value": {"shelf": "main", "book": "SiO2", "page": "Malitson"}}`
- `{"type": "name", "value": "SiO2"}` (will re-search and pick a page)

## Page lookups
```python
from evo_metaoptics.material_db import get_page_by_id, get_page_by_ref

page = get_page_by_id(123)
page = get_page_by_ref("main", "SiO2", "Malitson")
```

## Testing
```
uv run pytest tests/test_material_db_search_name_alias.py
uv run pytest tests/test_material_db_search_range.py
uv run pytest tests/test_material_db_download_guard.py
```

## Notes
- The SQLite DB stores metadata only; n/k data remain in the YAML database.
- Coverage ranges are computed from YAML data entries, not page-name regexes.
- If you change `material_db.source_root`, rebuild the index so coverage metadata stays consistent.
- Auto-download requires the `refractiveindex` package. If it fails to download, check network/SSL.
