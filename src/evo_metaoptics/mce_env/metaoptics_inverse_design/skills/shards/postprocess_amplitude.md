---
name: postprocess_amplitude
description: "Transmission/reflection efficiency APIs, diffraction-order utilities, and results serialization."
---

## Skill Overview

This shard covers amplitude/efficiency observables and advanced `SolverResults` data structures.

## Core Efficiency Attributes

```python
total_t = results.transmission
total_r = results.reflection

per_order_t = results.transmission_diffraction
per_order_r = results.reflection_diffraction
```

- `transmission` and `reflection` are per-wavelength totals.
- `*_diffraction` tensors provide order-resolved values.

## Diffraction-Order Methods

```python
t00 = results.get_order_transmission_efficiency(m=0, n=0)
r10 = results.get_order_reflection_efficiency(m=1, n=0)

orders = results.get_all_diffraction_orders()
propagating = results.get_propagating_orders(wavelength_idx=0)
```

## Structural Result Objects

- `results.transmission_field` and `results.reflection_field` are `FieldComponents`.
- `results.structure_matrix` is `ScatteringMatrix` (`S11`, `S12`, `S21`, `S22`).
- `results.wave_vectors` is `WaveVectors` (`kx`, `ky`, `kinc`, `kzref`, `kztrn`).

## Serialization

```python
payload = results.to_dict()
restored = SolverResults.from_dict(payload)
```
