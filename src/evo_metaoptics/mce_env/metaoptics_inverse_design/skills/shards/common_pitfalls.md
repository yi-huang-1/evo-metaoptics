---
name: common_pitfalls
description: "Frequent TorchRDIT setup mistakes and reliable corrections for inverse-design runs."
---

## Skill Overview

This shard lists high-impact mistakes that commonly break correctness, gradients, or runtime budgets.

## Layer Ordering Mistakes

- Mistake: adding finite layers in the wrong physical order.
- Fix: build stack as `ref_material -> Layer 0 -> ... -> trn_material`.
- Rule: the first `add_layer(...)` call defines Layer 0 (bottom-most finite layer).

## Incident-Medium API Confusion

- Mistake: searching for `update_inc_material`.
- Fix: use `update_ref_material(...)` for the incident/reflection-side medium.
- Use `update_trn_material(...)` for the transmission-side medium.

## Material Registration Errors

- Mistake: referencing a material name before adding the material object.
- Fix: call `.with_materials([...])` or `solver.add_materials([...])` first.

## Layer Indexing Errors

- Mistake: patterning the wrong layer index in `solver.update_er_with_mask(...)`.
- Fix: verify index mapping against the order of `add_layer(...)` calls.

## Broken Gradient Flow

- Mistake: optimizing tensors without `requires_grad=True`.
- Fix: initialize design parameters with gradient tracking and call `loss.backward()`.

## Timeout-Prone Optimization Loops

- Mistake: unbounded local/global loops.
- Fix: bound iteration counts, use coarse screening, then refine only top candidates.

## Layer Accumulation Inside Loops (CRITICAL)

- Mistake: calling `solver.add_layer(...)` inside an optimization or multistart loop.
  Each call **appends** a new layer to the solver — layers are never removed automatically.
  Doing this inside a loop creates hundreds of layers, exhausting memory and time.
- Rule: `add_layer(...)` is a **one-time setup** call. There is no `clear_layers()` or
  `remove_layer()` method. Once a layer is added, it persists for the lifetime of the
  solver instance.
- Fix: call `add_layer(...)` **once** before any loop. Inside the loop, modify layer
  properties in-place:
  ```python
  # ✓ CORRECT — add layer once, modify in-place inside loop
  solver.add_layer(material_name="Film", thickness=t0, is_homogeneous=False, is_optimize=True)
  for seed in range(80):
      solver.layers[0].thickness = new_thickness   # in-place update
      solver.update_er_with_mask(mask=mask, layer_index=0)
      result = solver.solve(source)
  ```
  ```python
  # ❌ WRONG — adds a new layer every iteration (120+ layers!)
  for seed in range(80):
      for epoch in range(30):
          solver.add_layer(...)   # NEVER do this inside a loop
  ```

## Missing SolverResults Import

- Mistake: annotating return type as `-> SolverResults` without importing the class.
  Python evaluates annotations at import time, causing a `NameError`.
- Fix: always include `from torchrdit.results import SolverResults` in your imports.
