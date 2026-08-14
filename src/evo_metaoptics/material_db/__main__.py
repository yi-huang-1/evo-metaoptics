"""CLI tool for material database lookup.

Enables: python -m evo_metaoptics.material_db <subcommand> [args]

Subcommands:
  search <material_name> [--wavelengths W1,W2,...] [--limit N]
  get-page <page_id>
  nk <page_id> --wavelengths W1,W2,...
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .runtime import get_material_nk
from .search import get_page_by_id, search_materials


def _output_json(data: dict[str, Any]) -> None:
    """Output data as JSON to stdout."""
    print(json.dumps(data))


def _error_json(message: str, exit_code: int = 1) -> None:
    """Output error as JSON and exit."""
    _output_json({"error": message})
    sys.exit(exit_code)


def _parse_wavelengths(wavelengths_str: str) -> list[float]:
    """Parse comma-separated wavelengths string to list of floats."""
    if not wavelengths_str.strip():
        raise ValueError("wavelengths string is empty")
    
    try:
        parts = wavelengths_str.split(",")
        return [float(p.strip()) for p in parts if p.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid wavelength format: {e}") from e


def cmd_search(args: argparse.Namespace) -> None:
    """Handle 'search' subcommand."""
    try:
        material_name = args.material_name
        limit = args.limit
        
        # Parse wavelengths if provided
        min_wavelength_um = None
        max_wavelength_um = None
        if args.wavelengths:
            wavelengths = _parse_wavelengths(args.wavelengths)
            if wavelengths:
                min_wavelength_um = min(wavelengths)
                max_wavelength_um = max(wavelengths)
        
        # Search
        result = search_materials(
            material_name,
            limit=limit,
            min_wavelength_um=min_wavelength_um,
            max_wavelength_um=max_wavelength_um,
        )
        
        # Format output
        matches = []
        for match in result.matches:
            for page in match.pages:
                matches.append({
                    "page_id": page.page_id,
                    "material_id": match.material_id,
                    "shelf": match.shelf,
                    "book": match.book,
                    "page": page.page,
                    "page_name": page.page_name,
                    "display_name": match.display_name,
                    "score": match.score,
                    "coverage_min": page.coverage_min,
                    "coverage_max": page.coverage_max,
                    "has_n": page.has_n,
                    "has_k": page.has_k,
                })
        
        _output_json({
            "query": material_name,
            "matches": matches,
        })
    
    except Exception as e:
        _error_json(f"Search failed: {e}")


def cmd_get_page(args: argparse.Namespace) -> None:
    """Handle 'get-page' subcommand."""
    try:
        page_id = args.page_id
        
        # Validate page_id
        if not isinstance(page_id, int) or page_id <= 0:
            _error_json(f"Invalid page_id: {page_id}")
        
        # Lookup page
        page_ref = get_page_by_id(page_id)
        if page_ref is None:
            _error_json(f"Page not found: page_id={page_id}", exit_code=1)
        
        # Format output
        _output_json({
            "page_id": page_ref.page_id,
            "material_id": page_ref.material_id,
            "shelf": page_ref.shelf,
            "book": page_ref.book,
            "page": page_ref.page,
            "page_name": page_ref.page_name,
            "data_path": page_ref.data_path,
            "coverage_min": page_ref.coverage_min,
            "coverage_max": page_ref.coverage_max,
            "has_n": page_ref.has_n,
            "has_k": page_ref.has_k,
        })
    
    except Exception as e:
        _error_json(f"Get page failed: {e}")


def cmd_nk(args: argparse.Namespace) -> None:
    """Handle 'nk' subcommand."""
    try:
        page_id = args.page_id
        wavelengths_str = args.wavelengths
        
        # Validate page_id
        if not isinstance(page_id, int) or page_id <= 0:
            _error_json(f"Invalid page_id: {page_id}")
        
        # Parse wavelengths
        if not wavelengths_str:
            _error_json("--wavelengths is required")
        
        wavelengths = _parse_wavelengths(wavelengths_str)
        if not wavelengths:
            _error_json("No valid wavelengths provided")
        
        # Fetch n/k
        result = get_material_nk(page_id, wavelengths)
        
        # Format output
        _output_json(result)
    
    except Exception as e:
        _error_json(f"NK lookup failed: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m evo_metaoptics.material_db",
        description="Material database CLI tool for Pi agent material lookup",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search for materials by name",
    )
    search_parser.add_argument(
        "material_name",
        help="Material name to search for (e.g., 'SiO2', 'TiO2')",
    )
    search_parser.add_argument(
        "--wavelengths",
        default=None,
        help="Comma-separated wavelengths in micrometers (e.g., '1.55,0.5')",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of results (default: 3)",
    )
    search_parser.set_defaults(func=cmd_search)
    
    # get-page subcommand
    get_page_parser = subparsers.add_parser(
        "get-page",
        help="Get page details by page_id",
    )
    get_page_parser.add_argument(
        "page_id",
        type=int,
        help="Page ID from search results",
    )
    get_page_parser.set_defaults(func=cmd_get_page)
    
    # nk subcommand
    nk_parser = subparsers.add_parser(
        "nk",
        help="Get n/k values for a material page at given wavelengths",
    )
    nk_parser.add_argument(
        "page_id",
        type=int,
        help="Page ID from search results",
    )
    nk_parser.add_argument(
        "--wavelengths",
        required=True,
        help="Comma-separated wavelengths in micrometers (e.g., '1.55,0.5')",
    )
    nk_parser.set_defaults(func=cmd_nk)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute subcommand
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except Exception as e:
        _error_json(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
