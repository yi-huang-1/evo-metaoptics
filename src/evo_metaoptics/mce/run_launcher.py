"""Unified YAML-driven MCE launcher.

CLI entrypoint that loads a YAML run config, renders a launch plan,
captures a GPU snapshot, and executes (or dry-runs) the plan sequentially.

Usage (real run)::

    uv run python -m evo_metaoptics.mce.run_launcher configs/runs/my-run.yaml

Usage (dry-run)::

    MCE_LAUNCHER_DRY_RUN=1 uv run python -m evo_metaoptics.mce.run_launcher configs/runs/my-run.yaml

"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .run_bundle import create_run_bundle
from .run_config import RunConfig
from .run_launch_plan import LaunchPlanRenderer, capture_startup_gpu_snapshot
from .run_preflight import validate_data_files, validate_model_configured


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dry_run() -> bool:
    return os.environ.get("MCE_LAUNCHER_DRY_RUN", "").strip() in ("1", "true", "yes")


def _log(msg: str) -> None:
    print(f"[mce-launcher] {msg}", flush=True)


def _bundle_output_path(config: RunConfig) -> Path:
    return Path("bundles") / f"{config.run_slug}.zip"


def _format_argv(argv: list[str]) -> str:
    """Format argv list into a human-readable command string."""
    parts: list[str] = []
    for arg in argv:
        if " " in arg or not arg:
            parts.append(f'"{arg}"')
        else:
            parts.append(arg)
    return " ".join(parts)


def _format_env(env: dict[str, str]) -> str:
    """Format env dict into KEY=value lines for display."""
    lines = [f"  {k}={v}" for k, v in sorted(env.items())]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------

def load_config_from_yaml(yaml_path: Path) -> RunConfig:
    """Load and validate a YAML run config file."""
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")
    return RunConfig.from_dict(raw)


def run_preflight(config: RunConfig) -> None:
    """Validate data files exist and model is configured."""
    _log("preflight: validating data files...")
    kwargs: dict[str, Any] = {
        "train_data": config.data.train_data,
        "val_data": config.data.val_data,
    }
    if config.data.test_data:
        kwargs["test_data"] = config.data.test_data
    validate_data_files(**kwargs)

    _log("preflight: validating model...")
    validate_model_configured(model=config.experiment.model)

    _log("preflight: OK")


def execute_plan(
    config: RunConfig,
    *,
    dry_run: bool = False,
) -> int:
    """Render and execute (or dry-run) a launch plan.

    Returns the process exit code (0 on success).
    """
    plan = LaunchPlanRenderer.render(config)

    _log(f"run name: {config.name}")
    _log(f"run slug: {config.run_slug}")
    _log(f"workspace: {config.workspace}")
    _log(f"log dir: {config.log_dir}")
    if config.bundle.enabled:
        _log(f"bundle path: {_bundle_output_path(config)}")
    _log(f"phases: {len(plan.invocations)}")

    # --- GPU snapshot ---
    gpu_snapshot_path: str | None = None
    if plan.invocations:
        gpu_snapshot_path = plan.invocations[0].get("gpu_snapshot_path")

    if gpu_snapshot_path:
        _log(f"gpu snapshot target: {gpu_snapshot_path}")

    if dry_run:
        _log("mode: DRY-RUN (will print commands and exit)")
    else:
        _log("mode: EXECUTE")

    # Capture GPU snapshot before any execution
    if not dry_run and gpu_snapshot_path:
        _log("capturing GPU snapshot...")
        snapshot = capture_startup_gpu_snapshot(config, output_path=Path(gpu_snapshot_path))
        gpu_ok = snapshot.get("probe_success", False)
        gpu_count = snapshot.get("gpu_count", 0)
        _log(f"gpu snapshot written: {gpu_snapshot_path} (probe_success={gpu_ok}, gpu_count={gpu_count})")
    elif dry_run and gpu_snapshot_path:
        _log(f"[DRY-RUN] would capture GPU snapshot -> {gpu_snapshot_path}")

    # --- Execute each phase ---
    for i, invocation in enumerate(plan.invocations):
        phase = invocation.get("phase", f"phase-{i}")
        argv = invocation["argv"]
        env = invocation["env"]

        _log(f"--- phase {i + 1}/{len(plan.invocations)}: {phase} ---")

        cmd = ["uv", "run", "python", "-m", "evo_metaoptics.mce.main"] + argv
        cmd_str = _format_argv(cmd)

        if dry_run:
            _log(f"[DRY-RUN] env:\n{_format_env(env)}")
            _log(f"[DRY-RUN] command:\n  {cmd_str}")
            continue

        _log(f"executing: {cmd_str}")

        # Merge invocation env into current process env
        run_env = dict(os.environ)
        run_env.update(env)

        result = subprocess.run(
            cmd,
            env=run_env,
            check=False,
        )

        if result.returncode != 0:
            _log(f"FAILED: phase '{phase}' exited with code {result.returncode}")
            return result.returncode

        _log(f"phase '{phase}' completed successfully")

    if dry_run:
        _log("dry-run complete — no commands were executed")
        return 0

    _log("all phases completed successfully")

    # --- Bundle creation ---
    if config.bundle.enabled:
        _log("bundle creation enabled — creating archive...")
        bundle_path = _bundle_output_path(config)
        try:
            create_run_bundle(
                workspace_root=config.workspace,
                log_dir=config.log_dir,
                output_path=bundle_path,
            )
            _log(f"bundle created: {bundle_path}")
        except Exception as exc:
            _log(f"WARNING: bundle creation failed: {exc}")
            # Don't fail the run if bundling fails; just warn
    else:
        _log("bundle creation disabled (config.bundle.enabled=false)")

    return 0


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the unified launcher.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 on success).
    """
    args = argv if argv is not None else sys.argv[1:]

    if not args:
        print("Usage: python -m evo_metaoptics.mce.run_launcher <config.yaml>", file=sys.stderr)
        return 2

    yaml_path = Path(args[0]).resolve()
    dry_run = _is_dry_run()

    _log(f"config: {yaml_path}")
    _log(f"timestamp: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}")

    # Load and validate config
    try:
        config = load_config_from_yaml(yaml_path)
    except Exception as exc:
        _log(f"ERROR: failed to load config: {exc}")
        return 1

    # Preflight checks
    if not dry_run:
        try:
            run_preflight(config)
        except Exception as exc:
            _log(f"ERROR: preflight failed: {exc}")
            return 1
    else:
        _log("[DRY-RUN] skipping preflight checks")

    # Execute plan
    try:
        return execute_plan(config, dry_run=dry_run)
    except Exception as exc:
        _log(f"ERROR: execution failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
