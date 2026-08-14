# Sources And Solving

Single-source and batched-source execution patterns plus result access basics.

## Source setup

```python
# Define the light source

# Create source with specified properties
source = solver.add_source(
    theta=0 * degrees,  # Incident angle from normal
    phi=0 * degrees,    # Azimuthal angle
    pte=1.0,           # TE polarization amplitude
    ptm=0.0            # TM polarization amplitude
)

# For angled incidence:
# source = solver.add_source(theta=30*degrees, phi=0, pte=1, ptm=0)

# For unpolarized light (equal TE and TM):
# source = solver.add_source(theta=0, phi=0, pte=1/np.sqrt(2), ptm=1/np.sqrt(2))

# --- Dual-Polarization Batched Sources (for TE/TM separation) ---
# When the goal is "maximize TE reflection, minimize TM reflection" (or similar),
# create two sources with pure polarizations and batch-solve:
#
# source_te = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
# source_tm = solver.add_source(theta=0, phi=0, pte=0.0, ptm=1.0)
# results = solver.solve([source_te, source_tm])
#   results.reflection[0] = R_TE, results.reflection[1] = R_TM
#   results.transmission[0] = T_TE, results.transmission[1] = T_TM
```

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

## Source and solve reference

## Skill Overview

This shard covers illumination setup and execution with `solver.solve(...)`.

## Single Source and Solve

```python
source = solver.add_source(
    theta=0.0,
    phi=0.0,
    pte=1.0,
    ptm=0.0,
)

results = solver.solve(source)
```

- `theta` and `phi` are radians.
- Return type is `SolverResults`.

## Batched Sources

```python
source_0 = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
source_1 = solver.add_source(theta=0.5, phi=0.0, pte=1.0, ptm=0.0)

batched = solver.solve([source_0, source_1])
```

Common batched behaviors:

- `batched.transmission.shape` is `(n_sources, n_freqs)`.
- `batched[0]` returns a per-source `SolverResults` view.
- `batched.reflection.shape` is `(n_sources, n_freqs)`.
- `batched.reflection[0]` = reflection for source 0, `batched.reflection[1]` = reflection for source 1.

## Result Access Patterns

```python
t = results.transmission
r = results.reflection
t0 = float(results.transmission[0].item())
r0 = float(results.reflection[0].item())
```

## Dual-Polarization Sources (TE/TM Separation)

When the design goal involves **TE vs TM polarization separation** (e.g., "maximize TE
reflection while minimizing TM reflection"), create **two separate sources** with pure
polarizations and batch-solve:

```python
# Two sources: pure TE and pure TM
source_te = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
source_tm = solver.add_source(theta=0.0, phi=0.0, pte=0.0, ptm=1.0)

results = solver.solve([source_te, source_tm])
# results.reflection shape: (2, n_freqs)
#   results.reflection[0] = R_TE  (reflection under TE illumination)
#   results.reflection[1] = R_TM  (reflection under TM illumination)
```

**Trigger phrases**: "TE reflection high / TM reflection low",
"polarization-selective", "TE/TM separation", "polarizer",
"high TE, low TM" or vice versa.

**Key rule**: A single source with `pte=1, ptm=0` produces one combined reflection
scalar per wavelength. To get **separate TE and TM** reflection values, you must
use two sources and batch-solve.
