"""Write a run_manifest.json to both workspace and log directories.

The manifest provides a single source of truth for correlating workspace
artifacts with their corresponding log outputs during post-processing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


_MANIFEST_FILENAME = "run_manifest.json"


def write_run_manifest(
    *,
    workspace_base: str | Path,
    run_dir: str | Path,
    run_id: str,
    args: Dict[str, Any],
) -> tuple[Path, Path]:
    """Write ``run_manifest.json`` to *workspace_base* and *run_dir*.

    Both files contain identical JSON so that either directory can be used
    as the entry-point for post-processing scripts.

    Args:
        workspace_base: Workspace root directory.
        run_dir: Log run directory (e.g. ``logs/run_20260306_143000``).
        run_id: Shared run identifier (same value used in both directory names).
        args: CLI arguments dict — keys should match the main.py argparse namespace
            (``env``, ``model``, ``train_data``, ``val_data``, ``test_data``,
            ``iterations``, ``start_iter``, ``train_limit``, ``val_limit``,
            ``train_batch_size``, ``skill_path``, ``no_meta_agent``,
            ``pi_session_traces``).

    Returns:
        Tuple of (workspace_manifest_path, log_dir_manifest_path).
    """
    workspace_base = Path(workspace_base).resolve()
    run_dir = Path(run_dir).resolve()

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "workspace": str(workspace_base),
        "log_dir": str(run_dir),
        "env": args.get("env"),
        "model": args.get("model"),
        "train_data": args.get("train_data"),
        "val_data": args.get("val_data"),
        "test_data": args.get("test_data"),
        "iterations": args.get("iterations"),
        "start_iter": args.get("start_iter"),
        "train_limit": args.get("train_limit"),
        "val_limit": args.get("val_limit"),
        "train_batch_size": args.get("train_batch_size"),
        "skill_path": args.get("skill_path"),
        "no_meta_agent": args.get("no_meta_agent"),
        "pi_session_traces": args.get("pi_session_traces", True),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    ws_path = workspace_base / _MANIFEST_FILENAME
    ws_path.write_text(payload, encoding="utf-8")

    ld_path = run_dir / _MANIFEST_FILENAME
    ld_path.write_text(payload, encoding="utf-8")

    return ws_path, ld_path
