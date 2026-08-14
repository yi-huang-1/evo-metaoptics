---
name: optimization_patterns
description: "Global and local optimization recipes for TorchRDIT inverse-design workflows."
---

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
