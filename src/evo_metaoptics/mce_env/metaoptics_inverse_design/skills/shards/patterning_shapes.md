---
name: patterning_shapes
description: "Shape generation, boolean mask operations, and patterned-permittivity updates."
---

## Skill Overview

This shard contains shape and mask APIs used for geometric patterning in finite layers.

## Shape Generator

```python
shape_gen = ShapeGenerator.from_solver(solver)
```

## Primitive Masks

```python
circle = shape_gen.generate_circle_mask(
    center=(0.0, 0.0),
    radius=0.25,
    soft_edge=0.001,
)

rect = shape_gen.generate_rectangle_mask(
    center=(0.0, 0.0),
    x_size=0.4,
    y_size=0.3,
    angle=45,
    soft_edge=0.001,
)

poly = shape_gen.generate_polygon_mask(
    polygon_points=[(0.0, 0.0), (0.5, 0.0), (0.25, 0.5)],
    center=(0.0, 0.0),
    angle=0,
    soft_edge=0.001,
)
```

## Combining Masks

```python
u = shape_gen.combine_masks(circle, rect, operation="union")
i = shape_gen.combine_masks(circle, rect, operation="intersection")
d = shape_gen.combine_masks(circle, rect, operation="difference")
s = shape_gen.combine_masks(circle, rect, operation="subtract")
```

Supported `operation` values: `"union"`, `"intersection"`, `"difference"`, `"subtract"`.

## Applying a Mask to a Layer

```python
solver.update_er_with_mask(
    mask=u,
    layer_index=0,
    bg_material="Air",
)
```

- `layer_index` selects the finite layer to pattern.
- `bg_material` is the matrix/background medium used where mask values are off.
