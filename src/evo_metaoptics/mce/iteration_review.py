from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from evo_metaoptics.mce.agent_runtime import invoke_pi_session, start_pi_session_client, wrap_pi_session_client_as_session
from evo_metaoptics.mce.agent_session import AgentResponse, AgentSession
from evo_metaoptics.mce.agents_md import write_agents_md
from evo_metaoptics.mce.prompts.iteration_review import (
    MAX_REVIEW_CHARS,
    build_iteration_review_prompt,
    load_latest_prior_review_excerpt,
    truncate_for_prompt,
)

_MAX_MANIFEST_SECTION_CHARS = 2400
_MAX_REVIEW_ARTIFACT_CHARS = MAX_REVIEW_CHARS
_MAX_SUMMARY_ARTIFACT_CHARS = 2400
_MAX_SKILL_EXCERPT_CHARS = 1200
_MAX_EXEMPLARS = 4
_STRATEGY_NOTE_PATTERN = re.compile(r"^\s*#\s*Strategy\s*:\s*(.+?)\s*$", re.IGNORECASE)
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
_ITERATION_REVIEW_SYSTEM_PROMPT = (
    "You are the MCE iteration reviewer. Review the completed iteration using the provided manifest "
    "and bounded workspace context, then write the required markdown artifacts exactly where requested."
)


def iteration_review_dir(workspace_base: str | Path) -> Path:
    return Path(workspace_base) / "meta_agent" / "iteration_reviews"


def iteration_review_path(workspace_base: str | Path, iteration: int) -> Path:
    return iteration_review_dir(workspace_base) / f"iter{iteration}.md"


def iteration_review_manifest_path(workspace_base: str | Path, iteration: int) -> Path:
    return iteration_review_dir(workspace_base) / f"iter{iteration}_manifest.md"


def strategy_summary_path(workspace_base: str | Path) -> Path:
    return Path(workspace_base) / "meta_agent" / "strategy_summary.md"


def context_strategy_summary_path(workspace_base: str | Path) -> Path:
    return Path(workspace_base) / "context" / "strategy_summary.md"


def archived_iteration_skill_path(workspace_base: str | Path, iteration: int) -> Path:
    return (
        Path(workspace_base)
        / "meta_agent"
        / "skills"
        / f"learning-context-iter{iteration}"
        / "SKILL.md"
    )


def find_latest_prior_review_path(
    workspace_base: str | Path,
    *,
    current_iteration: int,
) -> Path | None:
    if current_iteration <= 1:
        return None
    reviews_dir = iteration_review_dir(workspace_base)
    for review_iteration in range(current_iteration - 1, 0, -1):
        candidate = reviews_dir / f"iter{review_iteration}.md"
        if candidate.is_file():
            return candidate
    return None


def find_latest_strategy_summary_path(workspace_base: str | Path) -> Path | None:
    candidates = [
        strategy_summary_path(workspace_base),
        context_strategy_summary_path(workspace_base),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def extract_strategy_note_from_solution(solution_path: str | Path) -> str | None:
    path = Path(solution_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _ in range(8):
                line = handle.readline()
                if not line:
                    break
                match = _STRATEGY_NOTE_PATTERN.match(line)
                if match:
                    return truncate_for_prompt(match.group(1).strip(), max_chars=240)
                if line.strip() and not line.lstrip().startswith("#"):
                    break
    except OSError:
        return None
    return None


def sanitize_review_artifact_markdown(markdown: str) -> str:
    return _sanitize_markdown_artifact(
        markdown,
        max_chars=_MAX_REVIEW_ARTIFACT_CHARS,
        default_heading="# Iteration Review",
        empty_fallback="No review content was produced.",
    )


def sanitize_strategy_summary_markdown(markdown: str) -> str:
    return _sanitize_markdown_artifact(
        markdown,
        max_chars=_MAX_SUMMARY_ARTIFACT_CHARS,
        default_heading="# Strategy Summary",
        empty_fallback="- No durable strategy guidance recorded yet.",
    )


def select_exemplar_references(
    *,
    iteration_dir: str | Path,
    train_payload: Mapping[str, Any] | None = None,
    val_payload: Mapping[str, Any] | None = None,
    max_exemplars: int = _MAX_EXEMPLARS,
) -> list[dict[str, str]]:
    iter_dir = Path(iteration_dir)
    references: list[dict[str, str]] = []
    seen_paths: set[Path] = set()

    payload_specs: Sequence[tuple[str, Mapping[str, Any] | None]] = (
        ("codegen", train_payload),
        ("codegen_val", val_payload),
    )
    for artifact_dir_name, payload in payload_specs:
        if len(references) >= max_exemplars:
            break
        detailed_results = payload.get("detailed_results", []) if isinstance(payload, Mapping) else []
        if not isinstance(detailed_results, list):
            detailed_results = []
        artifact_dir = iter_dir / artifact_dir_name
        for index, item in enumerate(detailed_results):
            if len(references) >= max_exemplars:
                break
            solution_path = artifact_dir / f"solution_{index}.py"
            if solution_path in seen_paths or not solution_path.is_file():
                continue
            seen_paths.add(solution_path)
            sample_id = None
            if isinstance(item, Mapping) and item.get("id") is not None:
                sample_id = str(item.get("id"))
            references.append(
                {
                    "id": sample_id or f"{artifact_dir_name}:{index}",
                    "path": str(solution_path),
                }
            )

    if len(references) < max_exemplars:
        for artifact_dir_name in ("codegen", "codegen_val"):
            artifact_dir = iter_dir / artifact_dir_name
            for solution_path in sorted(artifact_dir.glob("solution_*.py")):
                if len(references) >= max_exemplars:
                    break
                if solution_path in seen_paths:
                    continue
                seen_paths.add(solution_path)
                references.append(
                    {
                        "id": f"{artifact_dir_name}:{solution_path.stem}",
                        "path": str(solution_path),
                    }
                )
    return references


def synthesize_current_iteration_inputs(
    *,
    workspace_base: str | Path,
    iteration: int,
    last_sub_folder_name: str | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_base)
    meta_agent_dir = workspace / "meta_agent"
    evaluations_path = meta_agent_dir / "evaluations.json"
    evaluations_payload = _read_json_file(evaluations_path)
    iteration_key = f"iter{iteration}"
    iteration_summary = {}
    if isinstance(evaluations_payload, Mapping):
        raw_summary = evaluations_payload.get(iteration_key, {})
        if isinstance(raw_summary, Mapping):
            iteration_summary = dict(raw_summary)

    resolved_last_sub = last_sub_folder_name
    if not resolved_last_sub:
        last_from_eval = iteration_summary.get("last_sub_folder")
        if isinstance(last_from_eval, str) and last_from_eval.strip():
            resolved_last_sub = last_from_eval.strip()
        else:
            resolved_last_sub = f"iter{iteration}_sub0"

    iteration_dir = workspace / resolved_last_sub
    train_path = iteration_dir / "data" / "train.json"
    val_path = iteration_dir / "data" / "val.json"
    train_payload = _read_json_file(train_path)
    val_payload = _read_json_file(val_path)
    archived_skill_path = archived_iteration_skill_path(workspace, iteration)
    prior_review_path = find_latest_prior_review_path(
        workspace,
        current_iteration=iteration,
    )
    prior_summary_path = find_latest_strategy_summary_path(workspace)
    exemplar_refs = select_exemplar_references(
        iteration_dir=iteration_dir,
        train_payload=train_payload if isinstance(train_payload, Mapping) else None,
        val_payload=val_payload if isinstance(val_payload, Mapping) else None,
    )

    exemplar_notes: list[dict[str, str]] = []
    for ref in exemplar_refs:
        solution_path = Path(ref["path"])
        note = extract_strategy_note_from_solution(solution_path)
        if note:
            exemplar_notes.append(
                {
                    "id": ref["id"],
                    "path": ref["path"],
                    "note": note,
                }
            )

    return {
        "workspace_base": str(workspace),
        "iteration": iteration,
        "iteration_key": iteration_key,
        "iteration_dir": iteration_dir,
        "last_sub_folder_name": resolved_last_sub,
        "evaluations_path": evaluations_path,
        "evaluations": evaluations_payload,
        "iteration_summary": iteration_summary,
        "train_path": train_path,
        "train_payload": train_payload,
        "val_path": val_path,
        "val_payload": val_payload,
        "archived_skill_path": archived_skill_path,
        "prior_review_path": prior_review_path,
        "prior_strategy_summary_path": prior_summary_path,
        "exemplar_references": exemplar_refs,
        "exemplar_strategy_notes": exemplar_notes,
    }


def build_iteration_manifest_markdown(assembled: Mapping[str, Any]) -> str:
    iteration = int(assembled.get("iteration", 0) or 0)
    iteration_dir = Path(assembled.get("iteration_dir", ""))
    lines = [
        f"# Iteration {iteration} Manifest",
        "",
        "## Artifact Paths",
        f"- Iteration directory: `{iteration_dir}`",
        f"- Train artifact: `{assembled.get('train_path')}`",
        f"- Val artifact: `{assembled.get('val_path')}`",
        f"- Evaluations: `{assembled.get('evaluations_path')}`",
        f"- Archived skill: `{assembled.get('archived_skill_path')}`",
    ]

    prior_review_path = assembled.get("prior_review_path")
    prior_summary_path = assembled.get("prior_strategy_summary_path")
    if prior_review_path:
        lines.append(f"- Latest prior review: `{prior_review_path}`")
    if prior_summary_path:
        lines.append(f"- Latest strategy summary: `{prior_summary_path}`")

    lines.extend(
        [
            "",
            "## Iteration Summary Snapshot",
            _json_block(assembled.get("iteration_summary", {})),
            "",
            "## Train Artifact Snapshot",
            _json_block(_artifact_snapshot(assembled.get("train_payload"))),
            "",
            "## Val Artifact Snapshot",
            _json_block(_artifact_snapshot(assembled.get("val_payload"))),
            "",
            "## Archived Skill Excerpt",
            _skill_excerpt(Path(assembled.get("archived_skill_path", ""))),
            "",
            "## Exemplar References",
        ]
    )

    exemplar_refs = assembled.get("exemplar_references", [])
    if isinstance(exemplar_refs, list) and exemplar_refs:
        for ref in exemplar_refs:
            if not isinstance(ref, Mapping):
                continue
            lines.append(f"- `{ref.get('id', 'unknown')}` -> `{ref.get('path', '')}`")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Exemplar Strategy Notes"])
    exemplar_notes = assembled.get("exemplar_strategy_notes", [])
    if isinstance(exemplar_notes, list) and exemplar_notes:
        for note_entry in exemplar_notes:
            if not isinstance(note_entry, Mapping):
                continue
            note_text = truncate_for_prompt(str(note_entry.get("note", "")), max_chars=240)
            lines.append(
                f"- `{note_entry.get('id', 'unknown')}`: {note_text}"
            )
    else:
        lines.append("- (none)")

    return "\n".join(lines).strip() + "\n"


def run_iteration_review(
    *,
    workspace_base: str | Path,
    iteration: int,
    last_sub_folder_name: str | None = None,
    model: str | None = None,
    run_dir: Path | None = None,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if logger is None:
        logger = logging.getLogger(__name__)

    assembled = synthesize_current_iteration_inputs(
        workspace_base=workspace_base,
        iteration=iteration,
        last_sub_folder_name=last_sub_folder_name,
    )
    review_dir = iteration_review_dir(workspace_base)
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = iteration_review_manifest_path(workspace_base, iteration)
    review_path = iteration_review_path(workspace_base, iteration)
    durable_summary_path = strategy_summary_path(workspace_base)
    mirrored_summary_path = context_strategy_summary_path(workspace_base)

    manifest_markdown = build_iteration_manifest_markdown(assembled)
    manifest_path.write_text(manifest_markdown, encoding="utf-8")

    prompt = _build_iteration_review_session_prompt(
        assembled=assembled,
        manifest_markdown=manifest_markdown,
    )

    response: AgentResponse | None = None
    session_error: str | None = None
    try:
        response = _run_iteration_review_session(
            workspace_base=Path(workspace_base),
            prompt=prompt,
            model=model,
            run_dir=run_dir,
            timeout_s=timeout_s,
            session_traces_enabled=session_traces_enabled,
        )
        if response.error:
            session_error = str(response.error)
            logger.warning(
                "Iteration review Pi session returned error for iter%s: %s",
                iteration,
                session_error,
            )
    except Exception as exc:
        session_error = str(exc)
        logger.warning(
            "Iteration review Pi execution failed for iter%s: %s",
            iteration,
            session_error,
        )

    review_source_text = _select_review_artifact_text(
        review_path=review_path,
        response=response,
    )
    review_fallback_reason = None
    if review_source_text is None:
        review_fallback_reason = session_error or "Reviewer produced no usable iteration review markdown."
        logger.warning(
            "Iteration review missing or empty for iter%s; writing deterministic placeholder review. Reason: %s",
            iteration,
            review_fallback_reason,
        )
        review_source_text = _build_stub_review_markdown(
            assembled,
            manifest_path=manifest_path,
            failure_note=review_fallback_reason,
        )
    review_markdown = sanitize_review_artifact_markdown(review_source_text)
    review_path.write_text(review_markdown, encoding="utf-8")

    strategy_summary_source_text = _select_strategy_summary_artifact_text(
        durable_summary_path=durable_summary_path,
        mirrored_summary_path=mirrored_summary_path,
    )
    summary_fallback_reason = None
    if strategy_summary_source_text is None:
        summary_fallback_reason = session_error or "Reviewer produced no usable durable strategy summary markdown."
        logger.warning(
            "Strategy summary missing or empty for iter%s; writing deterministic placeholder summary. Reason: %s",
            iteration,
            summary_fallback_reason,
        )
        strategy_summary_source_text = _build_stub_strategy_summary_markdown(
            assembled,
            failure_note=summary_fallback_reason,
        )
    strategy_summary_markdown = sanitize_strategy_summary_markdown(
        strategy_summary_source_text
    )
    durable_summary_path.parent.mkdir(parents=True, exist_ok=True)
    durable_summary_path.write_text(strategy_summary_markdown, encoding="utf-8")
    mirrored_summary_path.parent.mkdir(parents=True, exist_ok=True)
    mirrored_summary_path.write_text(strategy_summary_markdown, encoding="utf-8")

    logger.info(
        "✅ Iteration review artifacts written: %s, %s, %s",
        manifest_path,
        review_path,
        durable_summary_path,
    )
    review_fallback_used = review_fallback_reason is not None
    summary_fallback_used = summary_fallback_reason is not None
    return {
        "status": "ok" if session_error is None else "error",
        "manifest_path": manifest_path,
        "review_path": review_path,
        "strategy_summary_path": durable_summary_path,
        "context_strategy_summary_path": mirrored_summary_path,
        "prompt_chars": len(prompt),
        "response_error": response.error if response is not None else None,
        "error": session_error,
        "review_fallback_used": review_fallback_used,
        "strategy_summary_fallback_used": summary_fallback_used,
        "assembled": assembled,
    }


def _build_stub_review_markdown(
    assembled: Mapping[str, Any],
    *,
    manifest_path: Path,
    failure_note: str | None = None,
) -> str:
    iteration = int(assembled.get("iteration", 0) or 0)
    summary = assembled.get("iteration_summary", {})
    primary_metric_name = None
    if isinstance(summary, Mapping):
        raw_primary_metric_name = summary.get("primary_metric_name")
        if isinstance(raw_primary_metric_name, str) and raw_primary_metric_name:
            primary_metric_name = raw_primary_metric_name
    val_metrics = summary.get("val_metrics", {}) if isinstance(summary, Mapping) else {}
    primary_metric_value = None
    if primary_metric_name and isinstance(val_metrics, Mapping):
        primary_metric_value = val_metrics.get(primary_metric_name)

    lines = [
        f"# Iteration {iteration} Review",
        "",
        "Deterministic fallback review generated from bounded iteration artifacts.",
        "",
        "## Inputs",
        f"- Manifest: `{manifest_path}`",
        f"- Iteration dir: `{assembled.get('iteration_dir')}`",
    ]
    if failure_note:
        lines.append(f"- Pi review fallback reason: {truncate_for_prompt(failure_note, max_chars=240)}")
    if primary_metric_name is not None:
        lines.append(
            f"- Validation `{primary_metric_name}`: {_format_metric(primary_metric_value)}"
        )
    prior_review_path = assembled.get("prior_review_path")
    if prior_review_path:
        lines.append(f"- Prior review consulted: `{prior_review_path}`")

    lines.extend(["", "## Exemplar Strategy Notes"])
    exemplar_notes = assembled.get("exemplar_strategy_notes", [])
    if isinstance(exemplar_notes, list) and exemplar_notes:
        for note_entry in exemplar_notes:
            if not isinstance(note_entry, Mapping):
                continue
            lines.append(
                f"- Strategy `{note_entry.get('id', 'unknown')}`: {note_entry.get('note', '')}"
            )
    else:
        lines.append("- Strategy notes unavailable for this iteration.")

    lines.extend(
        [
            "",
            "## Durable Guidance Boundary",
            "- Keep this review manifest-first and advisory; prefer manifest and summary evidence over raw trace spelunking.",
            "- Preserve exemplar references as paths only; do not copy iteration artifacts.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_stub_strategy_summary_markdown(
    assembled: Mapping[str, Any],
    failure_note: str | None = None,
) -> str:
    summary = assembled.get("iteration_summary", {})
    primary_metric_name = "primary metric"
    primary_metric_value: Any = None
    if isinstance(summary, Mapping):
        raw_metric_name = summary.get("primary_metric_name")
        if isinstance(raw_metric_name, str) and raw_metric_name:
            primary_metric_name = raw_metric_name
        val_metrics = summary.get("val_metrics", {})
        if isinstance(val_metrics, Mapping):
            primary_metric_value = val_metrics.get(primary_metric_name)

    lines = [
        "# Strategy Summary",
        "",
        "Cross-iteration memory for future iterations and the next coding agent.",
        "",
        f"- Track validation `{primary_metric_name}` at {_format_metric(primary_metric_value)} when available.",
        "- Read `context/strategy_summary.md` before planning as cross-iteration memory, but treat it as soft guidance rather than a rigid checklist.",
    ]
    if failure_note:
        lines.append(
            f"- Iteration review fell back to deterministic synthesis because Pi review failed: {truncate_for_prompt(failure_note, max_chars=240)}"
        )

    exemplar_notes = assembled.get("exemplar_strategy_notes", [])
    if isinstance(exemplar_notes, list) and exemplar_notes:
        for note_entry in exemplar_notes[:2]:
            if not isinstance(note_entry, Mapping):
                continue
            lines.append(
                f"- Reuse strategy family from `{note_entry.get('id', 'unknown')}` when it matches the current failure mode: {note_entry.get('note', '')}"
            )
    else:
        lines.append("- No exemplar `# Strategy:` notes were available; prefer conservative iteration-on-evidence changes.")

    return "\n".join(lines).strip() + "\n"


def _build_iteration_review_session_prompt(
    *,
    assembled: Mapping[str, Any],
    manifest_markdown: str,
) -> str:
    prior_summary_markdown = _read_optional_text(
        assembled.get("prior_strategy_summary_path")
    )
    exemplar_strategy_notes = []
    raw_notes = assembled.get("exemplar_strategy_notes", [])
    if isinstance(raw_notes, list):
        for entry in raw_notes:
            if not isinstance(entry, Mapping):
                continue
            exemplar_strategy_notes.append(
                f"{entry.get('id', 'unknown')}: {entry.get('note', '')}"
            )
    prompt = build_iteration_review_prompt(
        iteration=int(assembled.get("iteration", 0) or 0),
        iteration_manifest_markdown=manifest_markdown,
        prior_strategy_summary_markdown=prior_summary_markdown,
        exemplar_strategy_notes=exemplar_strategy_notes,
    )
    prior_review_excerpt = load_latest_prior_review_excerpt(
        workspace_base=str(assembled.get("workspace_base", "")),
        current_iteration=int(assembled.get("iteration", 0) or 0),
    )
    if prior_review_excerpt:
        prompt = f"{prompt}\n\n{prior_review_excerpt}"
    return prompt


def _run_iteration_review_session(
    *,
    workspace_base: Path,
    prompt: str,
    model: str | None = None,
    run_dir: Path | None = None,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
) -> AgentResponse:
    write_agents_md(
        iter_dir=workspace_base,
        system_prompt=_ITERATION_REVIEW_SYSTEM_PROMPT,
        skill_guidance=None,
        context_available=(workspace_base / "context").exists(),
    )
    client = start_pi_session_client(
        cwd=workspace_base,
        model=model or "",
        skill_paths=[],
        run_dir=run_dir,
        timeout_s=timeout_s,
        session_traces_enabled=session_traces_enabled,
    )
    session: AgentSession = wrap_pi_session_client_as_session(client, cwd=workspace_base)
    try:
        return invoke_pi_session(session, prompt)
    finally:
        _close_agent_session_sync(session)


def _close_agent_session_sync(session: AgentSession) -> None:
    close_sync = getattr(session, "close_sync", None)
    if callable(close_sync):
        close_sync()
        return
    asyncio.run(session.close())


def _select_review_artifact_text(
    *,
    review_path: Path,
    response: AgentResponse | None,
) -> str | None:
    on_disk = _read_optional_text(review_path)
    if on_disk:
        return on_disk
    if response is not None and response.content.strip():
        return response.content
    return None


def _select_strategy_summary_artifact_text(
    *,
    durable_summary_path: Path,
    mirrored_summary_path: Path,
) -> str | None:
    durable_text = _read_optional_text(durable_summary_path)
    if durable_text:
        return durable_text
    mirrored_text = _read_optional_text(mirrored_summary_path)
    if mirrored_text:
        return mirrored_text
    return None


def _sanitize_markdown_artifact(
    markdown: str,
    *,
    max_chars: int,
    default_heading: str,
    empty_fallback: str,
) -> str:
    bounded = truncate_for_prompt(markdown or "", max_chars=max_chars)
    normalized = bounded.strip()
    if not normalized:
        return f"{default_heading}\n\n{empty_fallback}"
    first_line = normalized.splitlines()[0].strip() if normalized else ""
    if not _HEADING_PATTERN.search(normalized):
        normalized = f"{default_heading}\n\n{normalized}"
    elif not first_line.startswith("# "):
        normalized = f"{default_heading}\n\n{normalized}"
    final_text = truncate_for_prompt(normalized, max_chars=max_chars)
    if not final_text.strip():
        return f"{default_heading}\n\n{empty_fallback}"
    return final_text


def _read_optional_text(path_like: Any) -> str | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    stripped = text.strip()
    return stripped or None


def _artifact_snapshot(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"status": "missing"}
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    detailed_results = payload.get("detailed_results", [])
    result_ids: list[str] = []
    if isinstance(detailed_results, list):
        for item in detailed_results[:5]:
            if not isinstance(item, Mapping):
                continue
            raw_id = item.get("id")
            if raw_id is not None:
                result_ids.append(str(raw_id))
    return {
        "summary": summary,
        "detailed_result_count": len(detailed_results) if isinstance(detailed_results, list) else 0,
        "sample_ids": result_ids,
    }


def _json_block(payload: Any) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    bounded = truncate_for_prompt(text, max_chars=_MAX_MANIFEST_SECTION_CHARS)
    return f"```json\n{bounded}\n```"


def _skill_excerpt(skill_path: Path) -> str:
    if not skill_path.is_file():
        return "(missing)"
    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError:
        return "(unreadable)"
    bounded = truncate_for_prompt(content, max_chars=_MAX_SKILL_EXCERPT_CHARS)
    return f"Path: `{skill_path}`\n\n```md\n{bounded}\n```"


def _read_json_file(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return "1.0000" if value else "0.0000"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


__all__ = [
    "build_iteration_manifest_markdown",
    "archived_iteration_skill_path",
    "context_strategy_summary_path",
    "extract_strategy_note_from_solution",
    "find_latest_prior_review_path",
    "find_latest_strategy_summary_path",
    "iteration_review_dir",
    "iteration_review_manifest_path",
    "iteration_review_path",
    "run_iteration_review",
    "sanitize_review_artifact_markdown",
    "sanitize_strategy_summary_markdown",
    "select_exemplar_references",
    "strategy_summary_path",
    "synthesize_current_iteration_inputs",
]
