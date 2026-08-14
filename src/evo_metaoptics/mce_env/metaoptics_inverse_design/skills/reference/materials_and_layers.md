# Materials And Layers

Material registration, boundary media, and stack-order patterns.

## Material creation

```python
# Create materials

# Method 1: Using permittivity directly
air = create_material(name='air', permittivity=1.0)
silicon = create_material(name='silicon', permittivity=11.7)  # at 1.55um

# Method 2: Using refractive index (n)
# permittivity = n^2
glass = create_material(name='glass', permittivity=1.5**2)

# Method 3: Complex permittivity (for lossy materials)
gold = create_material(name='gold', permittivity=complex(-100, 10))

# Method 4: Dispersive material from file
# dispersive_si = create_material(
#     name='silicon_dispersive',
#     dielectric_dispersion=True,
#     user_dielectric_file='Si_data.txt',
#     data_format='wl-eps',  # wavelength-permittivity format
#     data_unit='um'
# )

# Add materials to solver
solver.add_materials(material_list=[air, silicon])
# Add other materials as needed
```

## Material API guardrails

```python
# Material-related API clarification

CORRECT API calls:
- solver.update_ref_material('material_name')  # Set bottom/incident material
- solver.update_trn_material('material_name')  # Set top material
- builder.with_ref_material(material_object)   # During building
- builder.with_trn_material(material_object)   # During building

INCORRECT (these don't exist):
- solver.update_inc_material()  ❌
- builder.with_inc_material()   ❌
- solver.set_incident_material() ❌

Remember: The incident medium is the ref_material (reflection/bottom layer)
```

## Layer ordering

```python
# CRITICAL: Understanding Layer Order in TorchRDIT

The layer stack in TorchRDIT follows this structure:

    ↑ z-direction (upward)
    |
    |  Transmission region (top) - semi-infinite
    |  ================================
    |  Layer N-1 (last added layer)
    |  --------------------------------
    |  Layer N-2
    |  --------------------------------
    |  ...
    |  --------------------------------
    |  Layer 1 (second added layer)
    |  --------------------------------
    |  Layer 0 (first added layer)
    |  ================================
    |  Reflection region (bottom) - semi-infinite
    |
    ↓ Light incident from here

Key points:
1. Light is incident from the BOTTOM (reflection region)
2. The reflection region is also the incident medium
3. Layers are numbered in the order they are added (0, 1, 2, ...)
4. NO 'incident' material - use 'ref_material' for the incident medium
```

## Layer stack

```python
# IMPORTANT: Layer Stack Structure
# The layer ordering in TorchRDIT is:
# 
# TOP (Transmission side): trn_material
#            ↑
#        Layer N-1
#            ↑
#          ...
#            ↑
#        Layer 1
#            ↑
#        Layer 0
#            ↑
# BOTTOM (Reflection/Incident side): ref_material
#
# Light is incident from the bottom (ref_material side)
#
# CRITICAL: add_layer() is a ONE-TIME setup call. Each call permanently
# appends a layer to the solver. There is no clear_layers() or
# remove_layer(). NEVER call add_layer() inside an optimization loop.
# To change layer properties during optimization, modify them in-place:
#   solver.layers[0].thickness = new_thickness
# Set the bottom material (where light comes from)
solver.update_ref_material('air')  # or your substrate material

# Add layers from bottom to top
solver.add_layer(
    material_name='silicon',
    thickness=torch.tensor(0.22, dtype=torch.float64),  # in um
    is_homogeneous=True  # False for patterned layers
)

# Add more layers as needed...

# Set the top material
solver.update_trn_material('air')  # or your superstrate material

# NOTE: There is NO 'inc_material' or 'update_inc_material' in TorchRDIT!
# The incident medium is the ref_material (reflection/bottom layer)
```

## Materials and layer guidance

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
