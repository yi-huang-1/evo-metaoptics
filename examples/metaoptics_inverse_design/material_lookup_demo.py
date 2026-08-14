#!/usr/bin/env python3
"""Demo: Agent material workflow — tool search → runtime n/k → create_material.

This script demonstrates the full two-phase material workflow end-to-end
without requiring an LLM or MCE loop.  It is purely deterministic.

Phase 1 (Planning): Call ``lookup_material_nk`` tool → get page_id + metadata.
Phase 2 (Execution): Call ``get_material_nk(page_id, wavelengths)`` in solution.py
                       → get n/k → call ``create_material()``.

Note: Materials are scalar permittivity values (independent of device placement).
Device placement happens in solve_inverse_design(*, device: str = "cuda") via
builder.with_device(device) in the solver setup phase.

Usage:
    uv run python examples/metaoptics_inverse_design/material_lookup_demo.py
"""

from __future__ import annotations

import json
import sys


def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _try_create_material(name: str, **kwargs) -> None:
    """Try to create a TorchRDIT material; skip gracefully if unavailable."""
    try:
        from torchrdit.materials import create_material

        mat = create_material(name=name, **kwargs)
        print(f"  Material object: {mat}")
    except ImportError:
        print("  [SKIP] create_material not available in this TorchRDIT version.")
        print(f"  Would call: create_material(name={name!r}, ...)")
    except Exception as exc:
        print(f"  [SKIP] create_material raised: {exc}")
        print(f"  Would call: create_material(name={name!r}, ...)")


def _demo_single_wavelength_lossless() -> None:
    """SiO2 at 1.55 um — single wavelength, lossless."""
    _separator("Example 1: SiO2 at 1.55 um (single wavelength, lossless)")

    # --- Phase 1: Tool call (planning) ---
    print("[Phase 1] Calling lookup_material_nk tool...")
    from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
        build_material_lookup_tool,
    )

    tool = build_material_lookup_tool()
    raw = tool("SiO2", "[1.55]")
    response = json.loads(raw)
    print(f"  Tool response: {json.dumps(response, indent=2)}")

    if response["status"] != "ok":
        print(f"  ERROR: {response['error']}")
        return

    page_id = response["page_id"]
    print(f"  → Got page_id={page_id} ({response['book']}/{response['page']})")

    # --- Phase 2: Runtime n/k lookup (execution) ---
    print(f"\n[Phase 2] Calling get_material_nk(page_id={page_id}, wavelengths_um=[1.55])...")
    from evo_metaoptics.material_db.runtime import get_material_nk

    nk = get_material_nk(page_id=page_id, wavelengths_um=[1.55])
    print(f"  Runtime n/k: {nk}")

    # --- Phase 3: Create material ---
    permittivity = nk["n"][0] ** 2
    print(f"\n[Phase 3] Creating material (permittivity={permittivity:.6f})...")
    _try_create_material("SiO2", permittivity=permittivity)


def _demo_dispersive() -> None:
    """TiO2 at multiple wavelengths — dispersive."""
    _separator("Example 2: TiO2 at [0.5, 0.7, 1.0] um (dispersive)")

    # --- Phase 1: Tool call ---
    print("[Phase 1] Calling lookup_material_nk tool...")
    from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
        build_material_lookup_tool,
    )

    tool = build_material_lookup_tool()
    raw = tool("TiO2", "[0.5, 0.7, 1.0]")
    response = json.loads(raw)
    print(f"  Tool response: {json.dumps(response, indent=2)}")

    if response["status"] != "ok":
        print(f"  ERROR: {response['error']}")
        return

    page_id = response["page_id"]
    print(f"  → Got page_id={page_id}")

    # --- Phase 2: Runtime n/k ---
    wavelengths = [0.5, 0.7, 1.0]
    print(f"\n[Phase 2] Calling get_material_nk(page_id={page_id}, wavelengths_um={wavelengths})...")
    from evo_metaoptics.material_db.runtime import get_material_nk

    nk = get_material_nk(page_id=page_id, wavelengths_um=wavelengths)
    print(f"  n values: {nk['n']}")
    print(f"  k values: {nk['k']}")

    # --- Phase 3: Create material (dispersive) ---
    print("\n[Phase 3] Creating dispersive material...")
    _try_create_material(
        "TiO2",
        dielectric_dispersion=True,
        user_dielectric_wavelengths_um=wavelengths,
        user_dielectric_n=nk["n"],
        user_dielectric_k=nk["k"],
    )


def _demo_lossy_metal() -> None:
    """Gold at visible wavelengths — lossy metal."""
    _separator("Example 3: Gold at [0.5, 0.6, 0.7] um (lossy metal)")

    # --- Phase 1: Tool call ---
    print("[Phase 1] Calling lookup_material_nk tool...")
    from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
        build_material_lookup_tool,
    )

    tool = build_material_lookup_tool()
    raw = tool("Gold", "[0.5, 0.6, 0.7]")
    response = json.loads(raw)
    print(f"  Tool response: {json.dumps(response, indent=2)}")

    if response["status"] != "ok":
        print(f"  ERROR: {response['error']}")
        return

    page_id = response["page_id"]
    print(f"  → Got page_id={page_id}")

    # --- Phase 2: Runtime n/k ---
    wavelengths = [0.5, 0.6, 0.7]
    print(f"\n[Phase 2] Calling get_material_nk(page_id={page_id}, wavelengths_um={wavelengths})...")
    from evo_metaoptics.material_db.runtime import get_material_nk

    nk = get_material_nk(page_id=page_id, wavelengths_um=wavelengths)
    print(f"  n values: {nk['n']}")
    print(f"  k values: {nk['k']}")

    # --- Phase 3: Create material (lossy) ---
    print("\n[Phase 3] Creating lossy metal material...")
    _try_create_material(
        "Au",
        dielectric_dispersion=True,
        user_dielectric_wavelengths_um=wavelengths,
        user_dielectric_n=nk["n"],
        user_dielectric_k=nk["k"],
    )


def main() -> int:
    print("Material Lookup Demo — Two-Phase Agent Workflow")
    print("=" * 60)
    print()
    print("This demo shows the agent material workflow:")
    print("  1. Tool call (lookup_material_nk) → page_id + metadata")
    print("  2. Runtime helper (get_material_nk) → n/k values")
    print("  3. Create material (create_material) → TorchRDIT material")

    try:
        _demo_single_wavelength_lossless()
        _demo_dispersive()
        _demo_lossy_metal()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    _separator("All examples completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
