---
name: core_rules
description: "Core TorchRDIT function contract, imports, units, and stack-order conventions."
---

## Skill Overview

This shard defines the baseline rules every inverse-design solution should follow.

## Function Contract

Always implement this callable:

```python
def solve_inverse_design(*, device: str = "cpu") -> SolverResults:
```

The `device` parameter is **keyword-only** and specifies the compute device (e.g., `"cpu"`). Return the direct result from `solver.solve(source)` whenever possible.

## Required Imports

```python
import numpy as np
import torch

from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator
from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material
from torchrdit.results import SolverResults
```

## Unit Conventions

- Lengths are in `um` by default.
- Angles passed to `add_source(theta, phi, ...)` are radians.
- For lossless media, `permittivity = n**2`.
- Wavelength arrays passed to `.with_wavelengths(...)` are interpreted in the configured length unit.

## Layer Ordering Reminder

TorchRDIT physical ordering is:

`ref_material -> Layer 0 -> ... -> Layer N -> trn_material`

- `ref_material` and `trn_material` are semi-infinite boundary media.
- Do not add boundaries with `add_layer(...)`.
- Layer 0 is the first finite layer adjacent to `ref_material`.

## Minimal Skeleton

```python
def solve_inverse_design(*, device: str = "cpu") -> SolverResults:
    air = create_material(name="Air", permittivity=1.0)
    substrate = create_material(name="Substrate", permittivity=2.25)
    film = create_material(name="Film", permittivity=4.0)

    solver = (
        get_solver_builder()
        .with_device(device)  # Explicitly set compute device
        .with_algorithm(Algorithm.RCWA)
        .with_precision(Precision.SINGLE)
        .with_length_unit("um")
        .with_wavelengths(np.array([1.55]))
        .with_materials([air, substrate, film])
        .with_ref_material("Air")
        .with_trn_material("Substrate")
        .add_layer({"material": "Film", "thickness": 0.4, "is_homogeneous": False})
        .build()
    )

    src = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
    return solver.solve(src)
```
