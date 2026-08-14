---
name: source_solve
description: "Source construction, single and batched solves, and SolverResults access patterns."
---

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

### CPU Transfer for Export and NumPy Consumption

When exporting results to NumPy arrays or scalar values for downstream processing, explicitly transfer tensors to CPU:

```python
# For NumPy export
transmission_np = results.transmission.detach().cpu().numpy()
reflection_np = results.reflection.detach().cpu().numpy()

# For scalar extraction (when device is not CPU)
t_scalar = float(results.transmission[0].detach().cpu().item())
r_scalar = float(results.reflection[0].detach().cpu().item())
```

This ensures compatibility across environments and prevents device mismatch errors when consuming results outside the solver context.

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
