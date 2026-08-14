from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

MAX_REVIEW_CHARS = 3200
_MAX_MANIFEST_CHARS = 12000
_MAX_SUMMARY_CHARS = 2400
_MAX_NOTE_CHARS = 240
_MAX_NOTES = 8


def build_iteration_review_prompt(
    *,
    iteration: int,
    iteration_manifest_markdown: str,
    prior_strategy_summary_markdown: str | None = None,
    exemplar_strategy_notes: Sequence[str] | None = None,
) -> str:
    manifest_block = truncate_for_prompt(
        iteration_manifest_markdown,
        max_chars=_MAX_MANIFEST_CHARS,
    )
    summary_block = truncate_for_prompt(
        prior_strategy_summary_markdown or "",
        max_chars=_MAX_SUMMARY_CHARS,
    ) or "(none)"
    notes = _format_strategy_notes(exemplar_strategy_notes or [])
    review_path = f"meta_agent/iteration_reviews/iter{iteration}.md"

    return f"""# Iteration Review Writer

Review the completed iteration using the manifest first, then only drill into exemplar details when the manifest suggests it is necessary.

## Inputs

- Primary input: iteration manifest for `iter{iteration}`.
- Cross-iteration memory: prior `meta_agent/strategy_summary.md` when present. Treat it as the durable memory carried forward across iterations, then update it only when this iteration adds genuinely portable lessons.
- Optional exemplar hints: lightweight `# Strategy:` notes copied from a small number of exemplar `solution.py` files.

## Review Method

1. Start from the manifest and aggregate metrics before making conclusions.
2. Cite evidence with concrete artifact names, metric fields, or exemplar filenames.
3. Open at most a bounded set of exemplars; do not expand into an unbounded trace audit.
4. Keep strategy guidance advisory and durable. Do not impose rigid round-by-round rotation or new telemetry requirements.

## Output Contract

- Write the iteration-specific review to `{review_path}`.
- Update the durable cross-iteration memory at `meta_agent/strategy_summary.md`.
- Mirror that cross-iteration memory to `context/strategy_summary.md` for the next coding agent.
- Separate durable guidance from iteration-specific observations.

## Prior Durable Summary

{summary_block}

## Exemplar Strategy Notes

{notes}

## Iteration Manifest

{manifest_block}
"""


def load_latest_prior_review_excerpt(
    *,
    workspace_base: str,
    current_iteration: int,
    max_chars: int = MAX_REVIEW_CHARS,
) -> str:
    if current_iteration <= 1:
        return ""

    reviews_dir = Path(workspace_base) / "meta_agent" / "iteration_reviews"
    if not reviews_dir.exists():
        return ""

    for review_iteration in range(current_iteration - 1, 0, -1):
        review_path = reviews_dir / f"iter{review_iteration}.md"
        if review_path.is_file():
            review_text = review_path.read_text(encoding="utf-8")
            bounded = truncate_for_prompt(review_text, max_chars=max_chars)
            return (
                "## Latest Prior Review\n\n"
                f"Read the latest prior review first: `meta_agent/iteration_reviews/iter{review_iteration}.md`.\n\n"
                f"{bounded}"
            )
    return ""


def truncate_for_prompt(text: str, *, max_chars: int) -> str:
    normalized = _normalize_prompt_text(text)
    if len(normalized) <= max_chars:
        return normalized
    marker = "\n\n[truncated]"
    keep = max(0, max_chars - len(marker))
    return normalized[:keep].rstrip() + marker


def _format_strategy_notes(notes: Sequence[str]) -> str:
    lines: list[str] = []
    for note in notes[:_MAX_NOTES]:
        bounded = truncate_for_prompt(note, max_chars=_MAX_NOTE_CHARS)
        if bounded:
            lines.append(f"- {bounded}")
    if not lines:
        return "- (none)"
    return "\n".join(lines)


def _normalize_prompt_text(text: str) -> str:
    collapsed = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed


__all__ = [
    "build_iteration_review_prompt",
    "load_latest_prior_review_excerpt",
    "truncate_for_prompt",
    "MAX_REVIEW_CHARS",
]
