# Phase And Orders

Zero-order phase extraction and diffraction-order access patterns.

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

## Phase extraction

## Skill Overview

This shard explains extracting complex field components and converting them into phase/amplitude scalars.

## Zero-Order Field Components

```python
tx, ty, tz = results.get_zero_order_transmission()
rx, ry, rz = results.get_zero_order_reflection()
```

Each component is a complex tensor indexed by wavelength.

## Phase and Amplitude

```python
phase_tx = torch.angle(tx[0])
phase_rx = torch.angle(rx[0])

amp_tx = torch.abs(tx[0])
amp_rx = torch.abs(rx[0])
```

- `torch.angle(...)` returns phase in radians.
- `torch.abs(...)` returns complex magnitude.

## Scalar Extraction

```python
phase_scalar = phase_tx.item()
amp_scalar = amp_tx.item()
```

Use `.item()` when a Python float is required by logging, scoring, or JSON serialization.
