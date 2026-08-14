from __future__ import annotations

_REFERENCE_FILES: tuple[tuple[str, str], ...] = (
    ("reference/setup.md", "imports, units, and builder configuration"),
    ("reference/materials_and_layers.md", "materials, boundary media, and stack ordering"),
    ("reference/patterning.md", "mask construction and patterned layers"),
    ("reference/sources_and_solving.md", "source setup, batched solves, and execution"),
    ("reference/phase_and_orders.md", "phase extraction and diffraction-order reads"),
    ("reference/amplitude_and_efficiency.md", "amplitude and efficiency metrics"),
    ("reference/optimization.md", "deterministic optimization recipes"),
    ("reference/pitfalls.md", "high-frequency failure modes and fixes"),
    ("reference/context_usage.md", "how to use copied context artifacts"),
)


_FRONTMATTER = (
    "---\n"
    "name: learning-context\n"
    "description: Minimal TorchRDIT bootstrap plus navigation for inverse design.\n"
    "---\n"
)

_HEADER = "# TorchRDIT Inverse Design Bootstrap"

_SKILL_OVERVIEW = (
    "## Skill Overview\n\n"
    "Start from the minimal TorchRDIT contract in this file. Keep imports canonical, "
    "build a solver with `get_solver_builder()`, create the source with `solver.add_source(...)`, "
    "and return the direct `solver.solve(...)` result. Read `reference/*.md` only when the query "
    "or a runtime error requires deeper context."
)

_BOOTSTRAP_RULES = (
    "## Bootstrap Rules\n\n"
    "- Implement `def solve_inverse_design(*, device: str = \"cpu\") -> SolverResults`.\n"
    "- Import TorchRDIT symbols from the documented paths in this file; do not invent modules.\n"
    "- Register materials before they are referenced by name in layers or boundary media.\n"
    "- Add finite layers once during setup, then modify them in place inside optimization loops.\n"
    "- Always call `.with_device(device)` early in the builder chain to ensure all solver tensors are placed on the correct compute device.\n"
    "- Build the source with `solver.add_source(...)` and return the direct result of `solver.solve(source)`.\n"
)

_REFERENCE_MAP = "## Reference Map\n\n" + "\n".join(
    f"- `{path}` - {description}." for path, description in _REFERENCE_FILES
)

_OPTIONAL_CONTEXT = (
    "## Optional Context\n\n"
    "If a `context/` directory exists, treat it as optional learned guidance. Read `context/strategy_summary.md` "
    "first when present. It is the cross-iteration memory mirrored from `meta_agent/strategy_summary.md`, so use it "
    "to anchor durable strategy choices before consulting batch-local context. Then read only the most relevant "
    "remaining context files after `SKILL.md` and the needed "
    "`reference/*.md` files."
)

_STRATEGY_PORTFOLIO = (
    "## Strategy Portfolio\n\n"
    "Pick one bounded global search family before coding, then refine locally only after a promising candidate appears. "
    "Good default families are: multistart/global screening, small population-style candidate search, or coarse-to-fine "
    "parameter sweeps. Keep the first line of `solution.py` as `# Strategy: <family> - <why>` so retries can preserve or "
    "change the family deliberately instead of thrashing."
)

_REQUIRED_IMPORTS = """import numpy as np
import torch

from torchrdit.constants import Algorithm, Precision
from torchrdit.results import SolverResults
from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material"""
_BUILDER_SETUP = """builder = get_solver_builder()
builder.with_device(device)
builder.with_algorithm(Algorithm.RCWA)
builder.with_precision(Precision.DOUBLE)
builder.with_wavelengths(np.array([1.55]))
builder.with_length_unit(\"um\")
builder.with_real_dimensions([512, 512])
builder.with_k_dimensions([9, 9])
solver = builder.build()"""

_MATERIALS_AND_LAYERS = """air = create_material(name=\"air\", permittivity=1.0)
film = create_material(name=\"film\", permittivity=11.7)
solver.add_materials(material_list=[air, film])
solver.update_ref_material(\"air\")
solver.add_layer(
     material_name=\"film\",\n    thickness=torch.tensor(0.22, dtype=torch.float64, device=device),\n    is_homogeneous=True,\n)
solver.update_trn_material(\"air\")"""

_SOURCE_AND_SOLVE = """degrees = np.pi / 180
source = solver.add_source(theta=0 * degrees, phi=0 * degrees, pte=1.0, ptm=0.0)
result = solver.solve(source)
transmission_total = result.transmission
reflection_total = result.reflection
tx, ty, tz = result.get_zero_order_transmission()"""


def compose_preloaded_template_skill(query: str) -> str:
    # query is used for context in prompt building but not passed to generated function
    sections = [
        _FRONTMATTER,
        _HEADER,
        _SKILL_OVERVIEW,
        _BOOTSTRAP_RULES,
        "## Required Imports\n\n```python\n" + _REQUIRED_IMPORTS + "\n```",
        "## Builder Setup\n\n```python\n" + _BUILDER_SETUP + "\n```",
        "## Materials And Layers\n\n```python\n" + _MATERIALS_AND_LAYERS + "\n```",
        "## Source And Solve Pattern\n\n```python\n" + _SOURCE_AND_SOLVE + "\n```",
        _REFERENCE_MAP,
        _STRATEGY_PORTFOLIO,
        _OPTIONAL_CONTEXT,
    ]
    return "\n\n".join(sections) + "\n"



__all__ = ["compose_preloaded_template_skill", "_HEADER", "_REFERENCE_FILES"]
