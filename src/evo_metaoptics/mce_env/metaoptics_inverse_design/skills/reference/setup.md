# Setup

Canonical imports, units, and builder setup for cold-start-safe TorchRDIT runs.

## Required imports

```python
# Basic imports for TorchRDIT
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import get_solver_builder
from torchrdit.shapes import ShapeGenerator
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm, Precision
from torchrdit.results import SolverResults
from torchrdit.viz import plot_layer, display_fitted_permittivity
```

## Unit helpers

```python
# Define units (all calculations will use these units)
um = 1  # micrometers as base unit
nm = 1e-3 * um  # nanometers
mm = 1e3 * um  # millimeters
degrees = np.pi / 180  # convert degrees to radians
```

## Builder setup

```python
# Create solver using builder pattern
builder = get_solver_builder()

# Essential configuration
builder.with_algorithm(Algorithm.RCWA)  # or Algorithm.RDIT
builder.with_precision(Precision.DOUBLE)  # SINGLE or DOUBLE
builder.with_real_dimensions([512, 512])  # Real space grid resolution
builder.with_k_dimensions([9, 9])  # Fourier space harmonics
builder.with_wavelengths(np.array([1.55]))  # Wavelengths in um
builder.with_length_unit('um')

# For periodic structures, define lattice vectors
period = 0.5  # um
t1 = torch.tensor([[period, 0]])
t2 = torch.tensor([[0, period]])
builder.with_lattice_vectors(t1, t2)

# Device is controlled by execution environment; do not manually override here.

# For RDIT algorithm only: set order (typically 8-15)
# builder.with_rdit_order(10)

# Build the solver
solver = builder.build()
```

## Builder reference

## Skill Overview

This shard covers configuring a solver instance with the fluent `get_solver_builder()` API.

## Builder Pattern

```python
builder = get_solver_builder()

solver = (
    builder
    .with_algorithm(Algorithm.RCWA)      # or Algorithm.RDIT
    .with_precision(Precision.SINGLE)    # or Precision.DOUBLE
    .with_wavelengths(np.array([1.55]))
    .with_length_unit("um")
    .with_real_dimensions([512, 512])
    .with_k_dimensions([5, 5])
    .with_lattice_vectors(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
    )
    .with_materials([air, film, substrate])
    .with_ref_material("Air")
    .with_trn_material("Substrate")
    .add_layer(
        {
            "material": "Film",
            "thickness": 0.5,
            "is_homogeneous": False,
            "is_optimize": False,
        }
    )
    .build()
)
```

## Builder Methods Reference

| Method | Description |
|--------|-------------|
| `.with_algorithm(Algorithm.RCWA/RDIT)` | Solver algorithm |
| `.with_precision(Precision.SINGLE/DOUBLE)` | Float precision |
| `.with_wavelengths(float or np.ndarray)` | Operating wavelengths |
| `.with_length_unit("um")` | Length unit |
| `.with_real_dimensions([Nx, Ny])` | Spatial grid size |
| `.with_k_dimensions([Hx, Hy])` | Fourier harmonics count |
| `.with_lattice_vectors(t1, t2)` | Unit-cell lattice vectors (`torch.Tensor`) |
| `.with_materials([mat1, mat2, ...])` | Bulk-add materials |
| `.add_material(mat)` | Add single material |
| `.with_trn_material(name_or_mat)` | Transmission-side material |
| `.with_ref_material(name_or_mat)` | Reflection-side material |
| `.with_fff(True/False)` | Fast Fourier factorization |
| `.with_rdit_order(int)` | R-DIT expansion order |
| `.add_layer(dict)` | Add a finite layer |
| `.from_config(dict)` | Configure from a settings dictionary |
| `.build()` | Build and return solver |

## Layer Dict Keys

```python
{
    "material": "Film",
    "thickness": 0.5,
    "is_homogeneous": False,
    "is_optimize": False,
    "slice_count": 1,
}
```

## Enum Notes

- Algorithm choices: `Algorithm.RCWA`, `Algorithm.RDIT`
- Precision choices: `Precision.SINGLE`, `Precision.DOUBLE`
