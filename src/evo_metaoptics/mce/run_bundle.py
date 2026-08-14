"""Create a raw run ZIP bundle from workspace/ and logs/ trees.

The bundle archives full raw trees for a single MCE run, including
required artifacts (run_manifest.json, evaluations.json) and optional
artifacts (timing_report_check.json, pi_session_traces/, gpu_status_snapshot.json).

Archive structure::

    <bundle>.zip/
        workspace/          # Full copy of workspace_root tree
            run_manifest.json
            meta_agent/
                evaluations.json
            iter1_sub0/
                ...
        logs/<run_id>/      # Full copy of log_dir tree
            run_manifest.json
            timing_report_check.json   (optional)
            gpu_status_snapshot.json   (optional)
            pi_session_traces/         (optional)
                session_001.jsonl
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Collection
from pathlib import Path


def create_run_bundle(
    workspace_root: str | Path,
    log_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a raw run ZIP bundle from workspace and log trees.

    Archives the full *workspace_root* tree under ``workspace/`` and
    the full *log_dir* tree under ``logs/<log_dir.name>/`` inside the
    output ZIP file.  Optional artifacts are included when present and
    silently skipped when absent.

    Args:
        workspace_root: Path to the workspace root directory.
        log_dir: Path to the log run directory (e.g. ``logs/run_20260310_120000``).
        output_path: Destination path for the ZIP bundle.

    Returns:
        The resolved *output_path* as a :class:`~pathlib.Path`.
    """
    workspace_root = Path(workspace_root).resolve()
    log_dir = Path(log_dir).resolve()
    output_path = Path(output_path).resolve()

    # Ensure parent directory exists for the output file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_paths = {output_path}

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        _add_tree_to_zip(zf, workspace_root, arcname_prefix="workspace", exclude_paths=excluded_paths)

        log_prefix = f"logs/{log_dir.name}"
        _add_tree_to_zip(zf, log_dir, arcname_prefix=log_prefix, exclude_paths=excluded_paths)

    return output_path


def _add_tree_to_zip(
    zf: zipfile.ZipFile,
    root: Path,
    arcname_prefix: str,
    *,
    exclude_paths: Collection[Path] | None = None,
) -> None:
    if not root.is_dir():
        return

    excluded = {path.resolve() for path in (exclude_paths or ())}

    for child in sorted(root.rglob("*")):
        if child.is_file():
            if child.resolve() in excluded:
                continue
            arcname = f"{arcname_prefix}/{child.relative_to(root)}"
            _add_file_to_zip(zf, child, arcname=arcname)


def _add_file_to_zip(
    zf: zipfile.ZipFile,
    source_path: Path,
    arcname: str,
) -> None:
    zinfo = zipfile.ZipInfo.from_file(source_path, arcname)
    zinfo.compress_type = zf.compression
    with source_path.open("rb") as source_handle:
        with zf.open(zinfo, mode="w", force_zip64=True) as bundle_handle:
            shutil.copyfileobj(source_handle, bundle_handle)
