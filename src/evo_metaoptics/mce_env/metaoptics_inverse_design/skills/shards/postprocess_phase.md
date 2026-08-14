---
name: postprocess_phase
description: "Complex-field retrieval and phase extraction from zero-order transmission/reflection outputs."
---

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
