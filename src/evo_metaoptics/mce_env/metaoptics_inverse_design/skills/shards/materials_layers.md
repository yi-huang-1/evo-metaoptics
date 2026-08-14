---
name: materials_layers
description: "Material creation APIs, layer-stack boundaries, and practical layer-construction examples."
---

## Skill Overview

This shard focuses on creating materials, loading dispersion, and assembling finite layers with proper boundary media.

## Creating Materials

### Non-dispersive materials

```python
air = create_material(name="Air", permittivity=1.0)
film = create_material(name="Film", permittivity=4.0)
substrate = create_material(name="Substrate", permittivity=2.25)
```

### Dispersive materials (in-memory n/k vectors)

```python
disp = create_material(
    name="DispMat",
    dielectric_dispersion=True,
    user_dielectric_wavelengths_um=[1.0, 1.5, 2.0],
    user_dielectric_n=[2.2, 2.1, 2.0],
    user_dielectric_k=[0.02, 0.01, 0.01],
)
```

### Dispersive materials (from file)

```python
metal = create_material(
    name="Metal",
    dielectric_dispersion=True,
    user_dielectric_file="material_data.txt",
    data_format="freq-eps",  # or "wl-eps", "wl-nk"
    data_unit="thz",         # or "um"
)
```

### Quick n/k constructor

```python
from torchrdit.materials import MaterialClass

mat = MaterialClass.from_nk_data(name="Proxy", n=2.0, k=0.01)
```

## Adding Materials and Layers to the Solver

```python
solver.add_materials([air, film, substrate])
solver.add_layer(material_name="Film", thickness=0.5, is_homogeneous=False, is_optimize=True)
solver.update_ref_material(ref_material="Air")
solver.update_trn_material(trn_material="Substrate")
```

## Layer Stack Ordering

`ref_material -> Layer 0 -> Layer 1 -> ... -> Layer N -> trn_material`

- Layer 0 is the first finite layer adjacent to `ref_material`.
- Always add material definitions before referencing material names in layers.

## Example A: Light incident from lower boundary medium

```python
air = create_material(name="Air", permittivity=1.0)
base = create_material(name="Base", permittivity=2.9)
core = create_material(name="Core", permittivity=5.2)
cap = create_material(name="Cap", permittivity=2.1)

solver = (
    get_solver_builder()
    .with_materials([air, base, core, cap])
    .with_ref_material("Base")
    .with_trn_material("Air")
    .add_layer({"material": "Core", "thickness": 0.8, "is_homogeneous": False})
    .add_layer({"material": "Cap", "thickness": 0.1, "is_homogeneous": True})
    .build()
)
```

## Example B: Light incident from upper boundary medium

```python
air = create_material(name="Air", permittivity=1.0)
glass = create_material(name="Glass", permittivity=2.25)
high_index = create_material(name="HighIndex", permittivity=5.8)
spacer = create_material(name="Spacer", permittivity=2.1)

solver = (
    get_solver_builder()
    .with_materials([air, glass, high_index, spacer])
    .with_ref_material("Air")
    .with_trn_material("Glass")
    .add_layer({"material": "HighIndex", "thickness": 0.6, "is_homogeneous": False})
    .add_layer({"material": "Spacer", "thickness": 0.2, "is_homogeneous": True})
    .build()
)
```
