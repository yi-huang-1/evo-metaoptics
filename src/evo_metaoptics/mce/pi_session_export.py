from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path


def _is_pi_session_jsonl(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                return isinstance(payload, dict) and payload.get("type") == "session"
    except (OSError, json.JSONDecodeError):
        return False
    return False


def export_pi_session_html_reports(
    *,
    session_dir: str | Path | None,
    logger: logging.Logger,
) -> list[Path]:
    if session_dir is None:
        return []

    session_root = Path(session_dir)
    if not session_root.exists() or not session_root.is_dir():
        return []

    exported_paths: list[Path] = []
    for session_path in sorted(session_root.glob("*.jsonl")):
        if not _is_pi_session_jsonl(session_path):
            logger.warning(
                "Skipping non-session JSONL during Pi session HTML export: %s",
                session_path,
            )
            continue

        html_path = session_path.with_suffix(".html")
        try:
            subprocess.run(
                ["pi", "--export", str(session_path), str(html_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            logger.warning("Skipping Pi session HTML export because `pi` CLI is unavailable")
            return exported_paths
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            logger.warning(
                "Failed to export Pi session HTML for %s: %s",
                session_path,
                detail,
            )
            continue
        except OSError as exc:
            logger.warning(
                "Failed to export Pi session HTML for %s: %s",
                session_path,
                str(exc),
            )
            continue
        exported_paths.append(html_path)

    return exported_paths


__all__ = ["export_pi_session_html_reports"]
