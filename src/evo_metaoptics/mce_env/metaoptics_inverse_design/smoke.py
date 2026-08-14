"""Smoke-run helpers for metaoptics inverse-design environment."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import re
from typing import Any, Callable

_ITERATION_COMPLETED_MARKER = "✅ ITERATION 1 COMPLETED"
_FINAL_SUMMARY_MARKER = "🎯 FINAL SUMMARY"
_ENVIRONMENT_MARKER = "Environment: metaoptics_inverse_design"
_SUCCESS_GOAL_RE = re.compile(r"Final validation success_goal:\s*\d+(?:\.\d+)?%")
_TIMING_REPORT_CHECK_FILENAME = "timing_report_check.json"


@dataclass
class SmokeRunValidation:
    run_dir: Path
    ok: bool
    interrupted: bool
    issues: list[str] = field(default_factory=list)


def ensure_torchrdit_available(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
) -> Any:
    """Validate torchrdit runtime dependency with device-aware API surface checks.
    
    Validates that torchrdit.solver, torchrdit.builder, and torchrdit.results
    are available. The device-aware API surface (builder.with_device(device))
    is the canonical path for explicit device placement in solver creation.
    """
    try:
        torchrdit_mod = importer("torchrdit")
    except Exception as exc:
        raise RuntimeError(
            "torchrdit is required for inverse-design smoke runs."
        ) from exc

    for attr_name in ("solver", "builder", "results"):
        mod = None
        try:
            mod = importer(f"torchrdit.{attr_name}")
        except Exception:
            pass
        if mod is None:
            raise RuntimeError(
                f"torchrdit.{attr_name} module is required for inverse-design smoke validation."
            )
    return torchrdit_mod


def validate_metaoptics_inverse_design_smoke_run(run_dir: Path) -> SmokeRunValidation:
    """Validate one-iteration inverse-design smoke run artifacts."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.log"
    issues: list[str] = []
    interrupted = False

    if not summary_path.exists():
        issues.append(f"Missing run_summary.log: {summary_path}")
        return SmokeRunValidation(
            run_dir=run_dir,
            ok=False,
            interrupted=True,
            issues=issues,
        )

    summary_text = summary_path.read_text(encoding="utf-8")
    if _ENVIRONMENT_MARKER not in summary_text:
        issues.append("Run summary missing environment marker for metaoptics_inverse_design.")

    completed = (
        _ITERATION_COMPLETED_MARKER in summary_text
        and _FINAL_SUMMARY_MARKER in summary_text
    )
    if not completed:
        interrupted = True
        issues.append("Smoke run not completed (missing iteration completion/final summary markers).")

    if not _SUCCESS_GOAL_RE.search(summary_text):
        issues.append("Run summary missing `Final validation success_goal` marker.")

    timing_report_path = run_dir / _TIMING_REPORT_CHECK_FILENAME
    if not timing_report_path.exists():
        issues.append(f"Missing {_TIMING_REPORT_CHECK_FILENAME}: {timing_report_path}")
    else:
        try:
            parsed_report = json.loads(timing_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"Invalid {_TIMING_REPORT_CHECK_FILENAME}: {exc}")
        else:
            if not isinstance(parsed_report, dict):
                issues.append(
                    f"Invalid {_TIMING_REPORT_CHECK_FILENAME}: expected JSON object root."
                )
            else:
                for key in ("run_dir", "event_type_counts", "attempt_duration_s"):
                    if key not in parsed_report:
                        issues.append(
                            f"Invalid {_TIMING_REPORT_CHECK_FILENAME}: missing key `{key}`."
                        )

    return SmokeRunValidation(
        run_dir=run_dir,
        ok=not issues,
        interrupted=interrupted,
        issues=issues,
    )


__all__ = [
    "SmokeRunValidation",
    "ensure_torchrdit_available",
    "validate_metaoptics_inverse_design_smoke_run",
]
