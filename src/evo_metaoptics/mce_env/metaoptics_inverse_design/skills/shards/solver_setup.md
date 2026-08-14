---
name: solver_setup
description: "TorchRDIT solver builder pattern, configuration methods, and layer dictionary schema."
---

## Skill Overview

This shard covers configuring a solver instance with the fluent `get_solver_builder()` API.

## Builder Pattern

```python
builder = get_solver_builder()

solver = (
    builder
    .with_device(device)                 # Explicitly set compute device (e.g., "cpu")
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

> **Device Placement**: Always call `.with_device(device)` early in the builder chain to ensure all solver tensors are placed on the correct compute device. The `device` parameter is passed from the function signature and should be propagated explicitly.

## Builder Methods Reference

| Method | Description |
|--------|-------------|
| `.with_device(device)` | **Set compute device** (e.g., `"cpu"`); call early in chain |
| `.with_algorithm(Algorithm.RCWA/RDIT)` | Solver algorithm |
| `.with_precision(Precision.SINGLE/DOUBLE)` | Float precision |
| `.with_wavelengths(float or np.ndarray)` | Operating wavelengths |
| `.with_length_unit("um")` | Length unit |
| `.with_real_dimensions([Nx, Ny])` | Spatial grid size |
| `.with_k_dimensions([Hx, Hy])` | Fourier harmonics count |
| `.with_lattice_vectors(t1, t2)` | Unit-cell lattice vectors (`torch.Tensor`) |
| `.with_materials([mat1, mat2, ...])` | Bulk-add materials |
| `.with_ref_material(name)` | Set reference/incident material |
| `.with_trn_material(name)` | Set transmission/output material |
| `.with_fff(periods)` | Set fast Fourier factorization periods |
| `.with_rdit_order(n)` | Set RDIT expansion order (RDIT only) |
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
