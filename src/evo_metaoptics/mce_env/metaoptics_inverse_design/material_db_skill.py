"""Material database skill content for the code-gen agent.

This module provides a markdown skill section that teaches the agent how to use
the material database CLI tool to search for materials and get a ``page_id``,
then call ``get_material_nk(page_id, wavelengths_um)`` in solution.py to fetch n/k
at runtime.

The content is appended to the main TorchRDIT API skill (which has the YAML
frontmatter) and does NOT include its own frontmatter.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Skill section: Material Database Lookup
# ---------------------------------------------------------------------------

MATERIAL_DB_SKILL = """\

---

## 10. Material Database Lookup via CLI

When the design query mentions materials by name (e.g. "SiO2", "TiO2", "gold")
without providing explicit refractive index values, use the **two-step material
workflow**: (1) call the material database CLI to search for the material
and get a ``page_id``, then (2) call ``get_material_nk(page_id, wavelengths_um)``
inside ``solution.py`` to fetch accurate n/k data at runtime.

### When to use the material database

- **Use it**: Query says "use SiO2" or "silicon nitride substrate" without n/k values.
- **Skip it**: Query provides explicit n, k, or permittivity values.
- **Skip it**: Material is Air or vacuum — use ``create_material(name="Air", permittivity=1.0)``.

### Step 1: CLI search — Get page_id (during planning)

Use the material database CLI to search for the material and get a ``page_id``:

```bash
uv run python -m evo_metaoptics.material_db search "SiO2" --wavelengths 1.55
uv run python -m evo_metaoptics.material_db search "TiO2" --wavelengths 0.4,0.5,0.6,0.7
uv run python -m evo_metaoptics.material_db search "Gold" --wavelengths 0.5,0.8,1.0
```

The CLI returns JSON with the following fields:

| Field | Description |
|-------|-------------|
| ``page_id`` | Integer ID to use with ``get_material_nk()`` in solution.py |
| ``shelf`` | Database shelf (e.g. ``"main"``) |
| ``book`` | Material name (e.g. ``"SiO2"``) |
| ``page`` | Data page name (e.g. ``"Malitson"``) |
| ``coverage_min`` | Minimum wavelength coverage in um |
| ``coverage_max`` | Maximum wavelength coverage in um |
| ``has_n`` | Whether this page has refractive index data |
| ``has_k`` | Whether this page has extinction coefficient data |

Example CLI output:

```json
{
  "query": "SiO2",
  "matches": [
    {
      "page_id": 123,
      "shelf": "main",
      "book": "SiO2",
      "page": "Malitson",
      "coverage_min": 0.2,
      "coverage_max": 3.0,
      "has_n": true,
      "has_k": false
    }
  ]
}
```
### Step 2: Runtime n/k lookup — In solution.py (during execution)

Import ``get_material_nk`` and call it with the ``page_id`` from Step 1:

```python
from evo_metaoptics.material_db.runtime import get_material_nk

nk = get_material_nk(page_id=<PAGE_ID>, wavelengths_um=[1.55])
# nk = {"n": [1.444024], "k": [0.0], "book": "SiO2", "page": "Malitson",
#        "shelf": "main", "page_id": <PAGE_ID>}
```

This fetches n/k from the YAML database at runtime — values are always accurate
for the exact wavelengths you request.

### How to use the n/k result in ``create_material()``

**Single wavelength, lossless material** (k ≈ 0):

```python
from evo_metaoptics.material_db.runtime import get_material_nk
from torchrdit.materials import create_material

nk = get_material_nk(page_id=123, wavelengths_um=[1.55])
sio2 = create_material(name="SiO2", permittivity=nk["n"][0] ** 2)
```

**Multiple wavelengths (dispersive)**:

```python
from evo_metaoptics.material_db.runtime import get_material_nk
from torchrdit.materials import create_material

wavelengths = [0.5, 0.7, 1.0]
nk = get_material_nk(page_id=456, wavelengths_um=wavelengths)
tio2 = create_material(
    name="TiO2",
    dielectric_dispersion=True,
    user_dielectric_wavelengths_um=wavelengths,
    user_dielectric_n=nk["n"],
    user_dielectric_k=nk["k"],
)
```

**Lossy material (metal)** — always use dispersive form:

```python
from evo_metaoptics.material_db.runtime import get_material_nk
from torchrdit.materials import create_material

wavelengths = [0.5]
nk = get_material_nk(page_id=789, wavelengths_um=wavelengths)
gold = create_material(
    name="Au",
    dielectric_dispersion=True,
    user_dielectric_wavelengths_um=wavelengths,
    user_dielectric_n=nk["n"],
    user_dielectric_k=nk["k"],
)
```

### Complete workflow example

```python
# Inside solve_inverse_design(*, device: str = "cpu"):

# 1. The tool returned page_id=123 for SiO2 and page_id=456 for TiO2
#    during the planning phase.

from evo_metaoptics.material_db.runtime import get_material_nk
from torchrdit.materials import create_material

wavelengths = [1.55]

# 2. Fetch n/k at runtime
nk_sio2 = get_material_nk(page_id=123, wavelengths_um=wavelengths)
nk_tio2 = get_material_nk(page_id=456, wavelengths_um=wavelengths)

# 3. Create materials
sio2 = create_material(name="SiO2", permittivity=nk_sio2["n"][0] ** 2)
tio2 = create_material(name="TiO2", permittivity=nk_tio2["n"][0] ** 2)

# 4. Register with builder and proceed with design
# builder.add_material(sio2)
# builder.add_material(tio2)
```

### Common materials quick reference

For quick prototyping when the tool is unavailable, use these approximate values.
**Note**: page_ids may vary depending on the database build; always use the tool
to get the correct page_id for your environment.

| Material | Approx. n | Wavelength range | ``permittivity`` |
|----------|----------|------------------|-----------------|
| Air | 1.0 | all | 1.0 |
| SiO2 | 1.45 | 0.2–3 um | 2.1025 |
| Si | 3.48 | >1 um (NIR) | 12.1104 |
| TiO2 | 2.40 | 0.4–1.5 um | 5.76 |
| Si3N4 | 2.00 | 0.3–5 um | 4.0 |
| GaN | 2.40 | 0.4–1 um | 5.76 |
| Al2O3 (Sapphire) | 1.76 | 0.2–5 um | 3.0976 |
| AlN | 2.12 | 0.4–5 um | 4.4944 |

**Important**: These are rough single-wavelength approximations. For accurate
multi-wavelength designs or materials with significant dispersion/absorption,
always use the ``lookup_material_nk`` tool + ``get_material_nk()`` runtime helper.
"""


__all__ = ["MATERIAL_DB_SKILL"]
