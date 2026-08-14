# Optimization

Deterministic local and global optimization patterns for inverse design.

## Basic optimization

```python
# Basic optimization setup for inverse design

# Define parameter to optimize (example: circle radius)
radius = torch.tensor(0.2, requires_grad=True)  # Initial value

# Define objective function
def objective(solver, source, radius):
    # Update device geometry
    mask = shapegen.generate_circle_mask(center=[0, 0], radius=radius)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Solve
    result = solver.solve(source)
    
    # Define loss (example: maximize transmission at first wavelength)
    loss = -result.transmission[0]  # Negative because we minimize
    
    return loss

# Setup optimizer
optimizer = torch.optim.Adam([radius], lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = objective(solver, source, radius)
    loss.backward()
    optimizer.step()
    
    # Optional: Apply constraints
    with torch.no_grad():
        radius.data = torch.clamp(radius.data, 0.05, 0.45)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Radius = {radius.item():.3f}")
```

## Gradient-based workflow

```python
# Gradient-based optimization workflow

# 1. Define optimizable parameters
params = {
    'radius': torch.tensor(0.2, requires_grad=True),
    'thickness': torch.tensor(0.3, requires_grad=True)
}

# 2. Define objective function
def objective_function(solver, source, params):
    # Update geometry based on parameters
    update_device_with_params(solver, params)
    
    # Solve
    result = solver.solve(source)
    
    # Calculate loss (example: target specific transmission)
    target_transmission = 0.95
    loss = (result.transmission[0] - target_transmission)**2
    
    return loss

# 3. Optimization loop with Adam
optimizer = torch.optim.Adam(params.values(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    loss = objective_function(solver, source, params)
    loss.backward()
    optimizer.step()
    
    # Apply physical constraints
    with torch.no_grad():
        params['radius'].clamp_(0.05, 0.45)
        params['thickness'].clamp_(0.1, 1.0)
```

## Full differentiable pipeline

```python
# Complete inverse design pipeline with MULTISTART global optimization
# This is a self-contained example showing every step from solver setup through
# optimization.  Adapt the parameter type (radius, mask, thickness) to your task.
#
# IMPORTANT: TorchRDIT's solver provides LOCAL gradient optimization only.
# A single random initialization almost always gets stuck in a local minimum.
# You MUST use a global optimization strategy. This example shows multistart;
# see the "Global Optimization Patterns" skill section for alternatives
# (differential evolution, simulated annealing, particle swarm, basin-hopping).

import numpy as np
import torch
from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator

# --- 1. Materials ----------------------------------------------------------
air = create_material(name="air", permittivity=1.0)
high_index = create_material(name="high_index", permittivity=2.1**2)  # e.g. Ta2O5
substrate = create_material(name="substrate", permittivity=1.5**2)    # e.g. BK7

# --- 2. Solver build (builder pattern) ------------------------------------
builder = get_solver_builder()
builder.with_algorithm(Algorithm.RCWA)
builder.with_precision(Precision.DOUBLE)
builder.with_real_dimensions([256, 256])
builder.with_k_dimensions([5, 5])
builder.with_wavelengths(np.array([0.632]))       # wavelength in um
builder.with_length_unit("um")

period = 0.45  # um
t1 = torch.tensor([[period, 0.0]])
t2 = torch.tensor([[0.0, period]])
builder.with_lattice_vectors(t1, t2)
builder.with_materials([air, high_index, substrate])
builder.with_ref_material("air")         # incident side
builder.with_trn_material("substrate")   # transmission side

solver = builder.build()

# --- 3. Layer stack --------------------------------------------------------
# CRITICAL: add_layer() is called ONCE here. Each call permanently appends a
# layer to the solver (there is no clear_layers/remove_layer). NEVER call
# add_layer() inside an optimization loop — it would create hundreds of layers.
# To change thickness during optimization, update in-place: solver.layers[0].thickness = ...
thickness = torch.tensor(0.2, dtype=torch.float64)
solver.add_layer(
    material_name="high_index",
    thickness=thickness,
    is_homogeneous=False,   # patterned layer
    is_optimize=True,       # mark for optimization
)
# --- 4. Source -------------------------------------------------------------
source = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)

# --- 5. Objective function -------------------------------------------------
# IMPORTANT: ShapeGenerator, update_er_with_mask, and solver.solve ALL
# preserve gradients.  The key rule: every tensor in the chain from the
# optimizable parameter to the loss must be a torch differentiable op.
def objective(solver, source, radius):
    shapegen = ShapeGenerator(solver)  # or ShapeGenerator.from_solver(solver)
    mask = shapegen.generate_circle_mask(center=[0, 0], radius=radius)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    result = solver.solve(source)
    loss = -result.transmission[0]      # maximise transmission
    return loss

# --- 6. MULTISTART global exploration + local refinement -------------------
# Sweep 50-100 random seeds, each with a short gradient refinement,
# then polish the best candidate with more steps.
NUM_SEEDS = 80          # global exploration: 50-100 random starts
SHORT_EPOCHS = 20       # quick local refinement per seed

best_radius = None
best_loss = float("inf")

for seed in range(NUM_SEEDS):
    torch.manual_seed(seed)
    # Random initialization within physical bounds
    radius = torch.tensor(
        0.05 + torch.rand(1).item() * (period / 2 - 0.06),
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([radius], lr=0.01)

    for epoch in range(SHORT_EPOCHS):
        optimizer.zero_grad()
        loss = objective(solver, source, radius)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            radius.data = torch.clamp(radius.data, 0.05, period / 2 - 0.01)

    current_loss = float(loss.detach().item())
    if current_loss < best_loss:
        best_loss = current_loss
        best_radius = radius.detach().clone()

# --- 7. Refine the best candidate further ---------------------------------
radius = best_radius.clone().requires_grad_(True)
optimizer = torch.optim.Adam([radius], lr=0.005)
for epoch in range(80):   # deeper local polish: 50-100 steps
    optimizer.zero_grad()
    loss = objective(solver, source, radius)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        radius.data = torch.clamp(radius.data, 0.05, period / 2 - 0.01)

# --- 8. Final result -------------------------------------------------------
with torch.no_grad():
    shapegen = ShapeGenerator(solver)
    mask = shapegen.generate_circle_mask(center=[0, 0], radius=radius)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    final_result = solver.solve(source)
    # final_result is your SolverResults — return it from solve_inverse_design()
```

## Phase-target optimization

```python
# Phase-target gradient-based optimization
# Optimise a structure to hit a specific transmitted phase AND amplitude target.
# Key pattern: extract phase via torch.angle() on zero-order field components.

import numpy as np
import torch
from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator

# --- Build solver (abbreviated — see full pipeline template for details) ---
# ... builder setup, materials, lattice, build ...

# --- Differentiable parameter(s) ------------------------------------------
radius = torch.tensor(0.20, requires_grad=True)
thickness = torch.tensor(0.30, requires_grad=True)

optimizer = torch.optim.Adam([radius, thickness], lr=0.01)

# --- Phase + amplitude targets --------------------------------------------
target_phase_deg = 170.0                                  # target phase in degrees
target_phase_rad = torch.tensor(target_phase_deg * np.pi / 180.0)
target_transmission = 0.8                                 # minimum transmission

for epoch in range(80):
    optimizer.zero_grad()

    # Update geometry with current parameters
    solver.update_layer_thickness(layer_index=0, thickness=thickness)
    shapegen = ShapeGenerator(solver)
    mask = shapegen.generate_circle_mask(center=[0, 0], radius=radius)
    solver.update_er_with_mask(mask=mask, layer_index=0)

    result = solver.solve(source)

    # --- Extract phase from ZERO-ORDER transmitted field -------------------
    # get_zero_order_transmission() returns (tx, ty, tz) complex tensors
    # For TE: tx component;  For TM: ty component
    tx, ty, tz = result.get_zero_order_transmission()
    complex_coeff = tx[0]                    # first wavelength, TE x-component
    phase_rad = torch.angle(complex_coeff)   # phase in radians  (differentiable!)

    # --- Circular phase distance (handles wrap-around at +/-pi) ------------
    phase_diff = phase_rad - target_phase_rad
    # Wrap to [-pi, pi] — use atan2 for differentiable wrapping
    circular_dist = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = circular_dist ** 2

    # --- Amplitude loss: penalise low transmission -------------------------
    amplitude_loss = torch.clamp(target_transmission - result.transmission[0], min=0.0) ** 2

    # --- Combined loss -----------------------------------------------------
    loss = phase_loss + 2.0 * amplitude_loss
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        radius.data.clamp_(0.05, 0.45)
        thickness.data.clamp_(0.05, 1.0)

# --- Final evaluation ------------------------------------------------------
with torch.no_grad():
    solver.update_layer_thickness(layer_index=0, thickness=thickness)
    shapegen = ShapeGenerator(solver)
    mask = shapegen.generate_circle_mask(center=[0, 0], radius=radius)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    final_result = solver.solve(source)
    # Return final_result from solve_inverse_design()
```

## Multi-angle optimization

```python
# Multi-angle / multi-source gradient-based optimization
# Optimise a structure for uniform performance across several incident angles.
# Uses batched solver.solve(sources) for efficiency.

import numpy as np
import torch
from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator

# --- Build solver (abbreviated — see full pipeline template for details) ---
# ... builder setup, materials, lattice, build ...

# --- Create sources for multiple angles ------------------------------------
deg = np.pi / 180
target_angles = [0.0, 5.0 * deg, 15.0 * deg]   # angles in radians
sources = [
    solver.add_source(theta=angle, phi=0.0, pte=1.0, ptm=0.0)
    for angle in target_angles
]

# --- Differentiable parameter(s) ------------------------------------------
radius = torch.tensor(0.20, requires_grad=True)

# --- Optimiser -------------------------------------------------------------
optimizer = torch.optim.Adam([radius], lr=0.02)

for epoch in range(50):
    optimizer.zero_grad()

    # Regenerate mask from current radius
    shapegen = ShapeGenerator(solver)
    mask = shapegen.generate_circle_mask(center=[0, 0], radius=radius)
    solver.update_er_with_mask(mask=mask, layer_index=0)

    # Batch-solve for ALL angles in one call (preserves gradients)
    results = solver.solve(sources)

    # results.transmission has shape (n_sources, n_wavelengths)
    trans_per_angle = results.transmission[:, 0]     # first wavelength
    avg_trans = trans_per_angle.mean()
    variance  = trans_per_angle.var()

    # Loss: maximise average transmission, minimise angle-dependent variance
    loss = -avg_trans + 0.5 * variance
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        radius.data.clamp_(0.05, 0.40)

# --- Final multi-angle solve ----------------------------------------------
with torch.no_grad():
    shapegen = ShapeGenerator(solver)
    mask = shapegen.generate_circle_mask(center=[0, 0], radius=radius)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    final_result = solver.solve(sources)
    # final_result.reflection  shape: (n_sources, n_wavelengths)
    # final_result.transmission  shape: (n_sources, n_wavelengths)
```

## Multi-objective loss

```python
# Multi-objective optimization

def multi_objective(solver, source, params):
    result = solver.solve(source)
    
    # Multiple objectives
    obj1 = (result.transmission[0] - 0.9)**2  # Target 90% at λ1
    obj2 = (result.transmission[1] - 0.1)**2  # Target 10% at λ2
    obj3 = torch.abs(result.reflection[0] - 0.05)  # Target 5% reflection
    
    # Weighted sum
    weights = [1.0, 1.0, 0.5]
    total_loss = weights[0]*obj1 + weights[1]*obj2 + weights[2]*obj3
    
    return total_loss

# Or use separate losses for monitoring
losses = {'transmission': obj1, 'blocking': obj2, 'reflection': obj3}
```

## Common design patterns

```python
# Common Design Patterns

# 1. Multilayer Stack (Bragg reflector example)
n_periods = 10
for i in range(n_periods):
    solver.add_layer(material_name='high_index', thickness=torch.tensor(d1))
    solver.add_layer(material_name='low_index', thickness=torch.tensor(d2))

# 2. Metasurface with periodic array
# Set up lattice vectors for periodic boundary conditions
period = 0.5  # um
t1 = torch.tensor([[period, 0]])
t2 = torch.tensor([[0, period]])
builder.with_lattice_vectors(t1, t2)

# 3. Parameter sweep
thicknesses = np.linspace(0.1, 0.5, 50)
transmissions = []
for t in thicknesses:
    solver_temp = create_new_solver_with_thickness(t)
    result = solver_temp.solve(source)
    transmissions.append(result.transmission[0].item())

# 4. Multi-wavelength optimization
# Define wavelengths spanning your range of interest
wavelengths = np.linspace(1.4, 1.7, 10)  # um
builder.with_wavelengths(wavelengths)
```

## Global optimization strategies

## Skill Overview

TorchRDIT's solver provides **local** gradient-based optimization only.  A single
gradient run from one random start almost always gets stuck in a local minimum.
You MUST wrap local solves inside a **global** optimization strategy.

This shard provides ready-to-use patterns.  Pick the one that fits your problem;
all are compatible with `solver.solve(source)` → loss → optimize.

## Pattern 1: Multistart Gradient Descent (recommended default)

Sweep many random initializations, do a short gradient polish each, keep the best.

```python
best_param = None
best_loss = float("inf")

for seed in range(80):                    # 50-100 random starts
    torch.manual_seed(seed)
    param = torch.tensor(
        lo + torch.rand(1).item() * (hi - lo),
        requires_grad=True,
    )
    opt = torch.optim.Adam([param], lr=0.01)
    for _ in range(20):                   # short local refinement
        opt.zero_grad()
        loss = objective(param)           # calls solver.solve internally
        loss.backward()
        opt.step()
        with torch.no_grad():
            param.data.clamp_(lo, hi)
    cur = float(loss.detach())
    if cur < best_loss:
        best_loss = cur
        best_param = param.detach().clone()

# Polish the winner with more steps
param = best_param.clone().requires_grad_(True)
opt = torch.optim.Adam([param], lr=0.005)
for _ in range(80):                       # 50-100 refinement steps
    opt.zero_grad()
    loss = objective(param)
    loss.backward()
    opt.step()
    with torch.no_grad():
        param.data.clamp_(lo, hi)
```

## Pattern 2: Differential Evolution (scipy — gradient-free)

Good when the search space is large or the loss landscape is very rugged.

```python
import numpy as np
from scipy.optimize import differential_evolution

def scipy_objective(x):
    """x is a numpy array of design variables."""
    with torch.no_grad():
        radius = torch.tensor(x[0])
        # ... build mask, solver.update_er_with_mask, solver.solve ...
        loss = 1.0 - float(result.transmission[0])
    return loss

bounds = [(0.05, period / 2 - 0.01)]     # bounds per variable
result = differential_evolution(
    scipy_objective,
    bounds,
    maxiter=100,                          # 50-100 generations
    seed=42,
    tol=1e-6,
    polish=False,                         # we do our own gradient polish
)
best_x = result.x

# Optional: gradient polish from the DE winner
param = torch.tensor(best_x[0], requires_grad=True)
opt = torch.optim.Adam([param], lr=0.005)
for _ in range(80):
    opt.zero_grad()
    loss = objective(param)
    loss.backward()
    opt.step()
    with torch.no_grad():
        param.data.clamp_(bounds[0][0], bounds[0][1])
```

## Pattern 3: Dual Annealing (scipy — simulated annealing variant)

Effective for problems with many local minima and moderate dimensionality.

```python
from scipy.optimize import dual_annealing

result = dual_annealing(
    scipy_objective,
    bounds=[(0.05, 0.20), (0.1, 0.5)],   # e.g. [radius, thickness]
    maxiter=100,
    seed=42,
)
best_x = result.x
# ... gradient polish from best_x (same as Pattern 2) ...
```

## Pattern 4: Particle Swarm Optimization (pure torch)

Maintain a swarm of candidates; update velocities toward personal and global best.

```python
N_PARTICLES = 40
N_ITERS = 100
dim = 1           # number of design variables

lo_t = torch.tensor([0.05])
hi_t = torch.tensor([period / 2 - 0.01])

positions = lo_t + torch.rand(N_PARTICLES, dim) * (hi_t - lo_t)
velocities = torch.zeros_like(positions)
p_best_pos = positions.clone()
p_best_val = torch.full((N_PARTICLES,), float("inf"))
g_best_pos = positions[0].clone()
g_best_val = float("inf")

w, c1, c2 = 0.5, 1.5, 1.5               # inertia, cognitive, social

for _ in range(N_ITERS):
    for i in range(N_PARTICLES):
        with torch.no_grad():
            val = float(objective_no_grad(positions[i]))
        if val < p_best_val[i]:
            p_best_val[i] = val
            p_best_pos[i] = positions[i].clone()
        if val < g_best_val:
            g_best_val = val
            g_best_pos = positions[i].clone()
    r1, r2 = torch.rand(N_PARTICLES, dim), torch.rand(N_PARTICLES, dim)
    velocities = (w * velocities
                  + c1 * r1 * (p_best_pos - positions)
                  + c2 * r2 * (g_best_pos - positions))
    positions = torch.clamp(positions + velocities, lo_t, hi_t)

# Gradient polish from g_best_pos
param = g_best_pos.clone().requires_grad_(True)
# ... Adam loop 50-100 steps (same as Pattern 1 refinement) ...
```

## Pattern 5: Basin-Hopping (scipy)

Repeated local minimization from random perturbations of the best-so-far.

```python
from scipy.optimize import basinhopping

result = basinhopping(
    scipy_objective,
    x0=np.array([0.15]),
    niter=100,
    minimizer_kwargs={"method": "Nelder-Mead",
                      "options": {"maxiter": 50}},
    seed=42,
)
best_x = result.x
# ... gradient polish (same as Pattern 2) ...
```

## Gradient-Based Mask Optimization (local only — wrap in global)

For structured masks, use `ShapeGenerator.from_solver(solver)` and `generate_circle_mask` to create parameterized geometry, then optimize the parameters.

```python
mask = torch.rand(256, 256, requires_grad=True)
optimizer = torch.optim.Adam([mask], lr=0.01)

for _ in range(80):
    optimizer.zero_grad()
    solver.update_er_with_mask(mask=mask, layer_index=0, bg_material="Air")
    src = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
    results = solver.solve(src)
    loss = 1.0 - results.transmission[0]
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        mask.clamp_(0.0, 1.0)
```

## Multi-Objective Loss Pattern

```python
loss = (
    0.7 * (1.0 - results.transmission[0])
    + 0.3 * results.reflection[0]
)
```

## Practical Tips

- Use `torch.optim.Adam` for stable gradient refinement; try `torch.optim.LBFGS` for tighter local minima.
- Use 50-100 optimization steps per local refinement run.
- For multistart, sweep 50-100 random initializations.
- `scipy.optimize` is available — use `differential_evolution`, `dual_annealing`, or `basinhopping` for gradient-free global search.
- Use `torch.sigmoid` or explicit clamping when optimization variables must stay bounded.
- Prefer a coarse global sweep first, then spend compute refining the top candidate.
