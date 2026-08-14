# Amplitude And Efficiency

Amplitude, total transmission/reflection, and diffraction-efficiency retrieval.

## Solve and inspect

```python
# Solve the electromagnetic problem
result = solver.solve(source)

# Access simulation results
# Overall efficiencies (summed over all diffraction orders)
transmission_total = result.transmission  # Shape: (n_wavelengths,)
reflection_total = result.reflection    # Shape: (n_wavelengths,)

# Print results
for i, wavelength in enumerate(solver.lam0):
    print(f"λ = {wavelength*1000:.1f} nm:")
    print(f"  Transmission: {transmission_total[i]*100:.2f}%")
    print(f"  Reflection: {reflection_total[i]*100:.2f}%")
    print(f"  Absorption: {(1-transmission_total[i]-reflection_total[i])*100:.2f}%")

# Access field components (zero order)
tx, ty, tz = result.get_zero_order_transmission()
rx, ry, rz = result.get_zero_order_reflection()

# Calculate phase
phase_t = torch.angle(tx[0])  # Phase of x-component, first wavelength
amplitude_t = torch.abs(tx[0])  # Amplitude

# Access diffraction efficiencies for specific orders
# efficiency = result.get_order_transmission_efficiency(order_x=1, order_y=0)
```

## Amplitude and efficiency

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
