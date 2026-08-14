# Patterning

Patternable-layer setup, masks, and ShapeGenerator operations.

## Patterned layers

```python
# Creating patterned (non-homogeneous) layers

# First, add a non-homogeneous layer
solver.add_layer(
    material_name='foreground_material',  # foreground material, the name of the material should be added to the solver materials list first
    thickness=torch.tensor(0.5, dtype=torch.float64),
    is_homogeneous=False,  # This makes it patternable
    is_optimize=True  # Optional: mark for optimization
)

# Create shape generator
shapegen = ShapeGenerator.from_solver(solver)

# Generate shapes (example: circle)
mask = shapegen.generate_circle_mask(
    center=[0.0, 0.0],  # Center position
    radius=0.3,  # Radius in um
    soft_edge=0.001  # Soft edge width (0 for hard edge)
)

# Apply mask to the layer (layer_index starts from 0)
solver.update_er_with_mask(mask=mask, 
                           layer_index=0, 
                           bg_material='air', # background material if other than air (optional)
                           )

# For multiple shapes, combine them:
# mask1 = shapegen.generate_circle_mask(...)
# mask2 = shapegen.generate_rectangle_mask(...)
# combined = shapegen.combine_masks(mask1, mask2, operation='union')
```

## Shape operations

```python
# Available shape generation functions

# Circle
circle = shapegen.generate_circle_mask(
    center=[x, y],
    radius=r,
    soft_edge=0.001
)

# Rectangle
rectangle = shapegen.generate_rectangle_mask(
    center=[x, y],
    x_size=w,
    y_size=h,
    angle=theta,  # Rotation angle in degrees
    soft_edge=0.001
)

# Polygon
vertices = [[x1, y1], [x2, y2], [x3, y3]]  # List of vertices
polygon = shapegen.generate_polygon_mask(
    polygon_points=vertices,
    center=[x, y],
    angle=theta,
    soft_edge=0.001
)

# Combine shapes using boolean operations
union = shapegen.combine_masks(mask1, mask2, operation='union')
intersection = shapegen.combine_masks(mask1, mask2, operation='intersection')
difference = shapegen.combine_masks(mask1, mask2, operation='difference')
subtract = shapegen.combine_masks(mask1, mask2, operation='subtract')
```

## Patterning reference

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
