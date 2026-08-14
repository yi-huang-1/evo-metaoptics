---
name: learning-context
description: Code-generation context for metaoptics inverse-design MCE runs.
---

## Skill Overview
Use rollout telemetry to improve inverse-design code-generation context and reduce repeated failures.

## Function Contract
The agent writes a Python function:

```python
def solve_inverse_design(*, device: str = "cuda") -> SolverResults:
```

- Input: keyword-only `device` parameter (string, defaults to "cuda") specifying compute device placement.
- Output: `torchrdit.results.SolverResults` object from `solver.solve(source)`.
- The function calls TorchRDIT raw solver/builder APIs directly.
- The query context is provided in the prompt but not passed as a function parameter.

## Working Rules
1. The agent produces a complete `.py` file via the `write_file` tool.
2. The host runner imports `solve_inverse_design` from the written file and executes it with an explicit `device` parameter.
3. When executing Python scripts or modules in this project, always use `uv run python ...` instead of bare `python ...` so commands run inside the repo-managed environment.
4. Scoring is deterministic: `evaluate_gt_eval()` evaluates Python lambda expressions against the returned `SolverResults`.
5. Treat `success_goal` as primary metric; use `success_exec` to isolate runtime failures.
6. Always call `.with_device(device)` early in the builder chain to ensure all solver tensors are placed on the correct compute device.
7. TorchRDIT provides local gradient-based optimization; for global optima, the agent should write explicit global exploration code (for example multistart or population screening) and then locally refine top candidates.
8. Keep context concise and actionable; remove duplicated or contradictory guidance.

## Improvement Loop
1. Group failures by error type (import errors, shape mismatches, solver crashes, goal misses).
2. Add solver stability hints when execution succeeds but goal metrics are not met.
3. Preserve patterns associated with `success_goal=true` and avoid broad rewrites.
4. Keep guidance minimal and executable; avoid references to nonexistent utilities or helpers.

## Working with Learned Context

The `context/` folder contains accumulated knowledge from previous training iterations:
- `rules.txt` - Actionable rules for handling common failure patterns
- `analysis.md` - Root cause analysis from training data
- `examples.json` - Examples of successful patterns

Before implementing interfaces:
1. Check what context files exist
2. Read relevant guidance
3. Build upon existing knowledge rather than starting from scratch
4. Update context based on your analysis of train.json
