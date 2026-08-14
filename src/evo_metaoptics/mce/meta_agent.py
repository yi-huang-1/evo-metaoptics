"""Meta-agent implementation using Pi sessions."""

from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from evo_metaoptics.mce.agent_runtime import start_pi_session_client, wrap_pi_session_client_as_session
from evo_metaoptics.mce.agent_session import AgentSession
from evo_metaoptics.mce.agents_md import write_agents_md
from evo_metaoptics.mce.logging_utils import setup_logger
from evo_metaoptics.mce.prompts.meta_agent import build_meta_agent_prompt
from evo_metaoptics.mce.prompts.meta_agent import _build_skill_database
from evo_metaoptics.mce.skills import (
    find_unexpected_skill_variants,
    find_mirrored_host_skill_artifacts,
    build_learning_context_skill_seed,
    compose_skill_bundle,
    normalize_learning_context_skill_markdown,
    learning_context_skill_host_path,
    learning_context_skill_virtual_path,
    validate_skill_markdown,
)
from evo_metaoptics.mce.utils import (
    _find_best_iteration,
    archived_meta_skill_host_path,
    cleanup_irrelevant_files,
)

try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

load_dotenv(override=True)

_MAX_SKILL_RETRY_ERROR_CHARS = 900


def _verify_meta_agent_outputs(
    iter_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Verify required meta-agent artifacts and return SKILL.md content."""
    skill_file = learning_context_skill_host_path(iter_dir)

    if not skill_file.exists():
        logger.error(f"Meta-agent did not generate SKILL.md at {skill_file}")
        return {
            "success": False,
            "error": f"Meta-agent did not generate SKILL.md at {skill_file}",
            "skill_md": None,
        }

    unexpected_files = find_unexpected_skill_variants(iter_dir)
    if unexpected_files:
        unexpected_names = ", ".join(path.name for path in unexpected_files)
        error = (
            "Unexpected alternate skill file(s) found in learning-context directory: "
            f"{unexpected_names}. Only SKILL.md is allowed."
        )
        logger.error(error)
        return {
            "success": False,
            "error": error,
            "skill_md": None,
        }

    skill_md = skill_file.read_text(encoding="utf-8")
    normalized_skill_md = normalize_learning_context_skill_markdown(
        skill_md,
        expected_name="learning-context",
    )
    if normalized_skill_md != skill_md:
        skill_file.write_text(normalized_skill_md, encoding="utf-8")
        skill_md = normalized_skill_md

    is_valid, validation_error = validate_skill_markdown(
        skill_md,
        expected_name="learning-context",
    )
    if not is_valid:
        error = f"Invalid SKILL.md at {skill_file}: {validation_error}"
        logger.error(error)
        return {
            "success": False,
            "error": error,
            "skill_md": None,
        }

    logger.info(f"✓ Generated SKILL.md ({len(skill_md)} chars)")

    return {
        "success": True,
        "skill_md": skill_md,
        "error": None,
    }


def _detect_structural_skill_degeneracy(skill_md: str) -> tuple[bool, str]:
    """Detect scaffold-like or structurally degenerate skill outputs."""
    text = skill_md.strip()
    if not text:
        return True, "skill_empty"

    seed_text = build_learning_context_skill_seed().strip()
    if text == seed_text:
        return True, "seed_scaffold_unchanged"

    seed_markers = [
        "Draft this section with the core learning strategy.",
        "1. Review prior context and training outcomes.",
        "2. Identify recurring failure patterns.",
        "3. Update context and interfaces with focused, testable improvements.",
    ]
    retained_markers = sum(1 for marker in seed_markers if marker in text)
    if retained_markers >= 3:
        return True, "seed_placeholder_retained"

    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if len(text) < 120 and len(non_empty_lines) <= 5:
        return True, "skill_too_short"

    return False, ""


def _resolve_best_prior_skill_path(workspace_base: Path, iteration: int) -> Path | None:
    if iteration <= 1:
        return None
    meta_agent_dir = workspace_base / "meta_agent"
    best_info = _find_best_iteration(
        workspace_base=workspace_base,
        current_iteration=iteration,
        env=None,
    )
    if not isinstance(best_info, Mapping):
        return None
    best_iteration_raw = best_info.get("iteration")
    if isinstance(best_iteration_raw, bool) or not isinstance(best_iteration_raw, int):
        return None
    best_iteration = int(best_iteration_raw)
    if best_iteration <= 0:
        return None
    prior_path = archived_meta_skill_host_path(meta_agent_dir, best_iteration)
    if prior_path.exists():
        return prior_path
    return None


def _apply_structural_skill_antiregression_guard(
    *,
    iter_dir: Path,
    workspace_base: Path,
    iteration: int,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Classify structural degeneracy without penalty-based replacement."""
    if logger is None:
        logger = logging.getLogger(__name__)

    skill_path = learning_context_skill_host_path(iter_dir)
    if not skill_path.exists():
        decision = {
            "skill_guard_triggered": False,
            "guard_action": "accept_generated",
            "degeneracy_type": "none",
        }
        logger.warning(
            "skill_guard action=%s triggered=%s degeneracy_type=%s reason=skill_missing",
            decision["guard_action"],
            decision["skill_guard_triggered"],
            decision["degeneracy_type"],
        )
        return decision

    generated = skill_path.read_text(encoding="utf-8")
    is_degenerate, reason = _detect_structural_skill_degeneracy(generated)
    if not is_degenerate:
        decision = {
            "skill_guard_triggered": False,
            "guard_action": "accept_generated",
            "degeneracy_type": "none",
        }
        logger.info(
            "skill_guard action=%s triggered=%s degeneracy_type=%s",
            decision["guard_action"],
            decision["skill_guard_triggered"],
            decision["degeneracy_type"],
        )
        return decision

    degeneracy_type = reason if reason else "skill_empty"
    decision = {
        "skill_guard_triggered": True,
        "guard_action": "accept_generated",
        "degeneracy_type": degeneracy_type,
    }
    logger.warning(
        "skill_guard action=%s triggered=%s degeneracy_type=%s",
        decision["guard_action"],
        decision["skill_guard_triggered"],
        decision["degeneracy_type"],
    )
    return decision


def _build_skill_validation_retry_prompt(
    validation_error: str,
    expected_virtual_path: str,
) -> str:
    """Build retry prompt when SKILL.md validation fails."""
    error_excerpt = str(validation_error or "").strip()
    if len(error_excerpt) > _MAX_SKILL_RETRY_ERROR_CHARS:
        error_excerpt = error_excerpt[: _MAX_SKILL_RETRY_ERROR_CHARS - 3].rstrip() + "..."
    return f"""
⚠️ VALIDATION ERROR

Your SKILL.md file failed validation:
Validation error excerpt:
{error_excerpt}

Please create or update SKILL.md at this EXACT path:
{expected_virtual_path}

Required:
1. Write to path: {expected_virtual_path}
2. Include ## Skill Overview section
3. Provide complete learning methodology
4. Write SKILL.md using your file write capability.
5. HARD RULE: do not create alternate files like SKILL-evolved.md. Only SKILL.md is allowed.

Repair workflow:
1. Read current SKILL.md content at {expected_virtual_path}.
2. Write the updated content to the SAME file path.
3. Keep the filename exactly `SKILL.md`.
4. Replace placeholder text with concrete evolved instructions and actionable methodology.
5. Do NOT create alternate files (Do NOT create `SKILL-evolved.md`, `SKILL_new.md`, etc).
6. Frontmatter normalization is handled automatically after write.

Please create the SKILL.md file now.
"""


def _warn_on_mirrored_host_skill_artifacts(
    workspace_base: Path,
    iter_folder_name: str,
    logger: logging.Logger,
) -> None:
    """Warn when host-absolute writes are mirrored under workspace root."""
    artifacts = find_mirrored_host_skill_artifacts(
        workspace_base=workspace_base,
        iter_folder_name=iter_folder_name,
    )
    if not artifacts:
        return

    preview = ", ".join(str(path) for path in artifacts[:2])
    extra = "" if len(artifacts) <= 2 else f" (+{len(artifacts) - 2} more)"
    logger.warning(
        "⚠️ Detected host-path mirror skill artifact(s) under workspace root; "
        f"check virtual path usage: {preview}{extra}"
    )


def _summarize_attempt_timings(attempt_timings: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [
        float(item.get("duration_s", 0.0))
        for item in attempt_timings
        if isinstance(item, dict)
    ]
    success_count = sum(
        1
        for item in attempt_timings
        if isinstance(item, dict) and bool(item.get("success"))
    )
    total_duration = sum(durations)
    attempt_count = len(attempt_timings)
    return {
        "attempt_count": attempt_count,
        "success_count": success_count,
        "failure_count": max(0, attempt_count - success_count),
        "total_duration_s": round(total_duration, 6),
        "avg_duration_s": round(total_duration / attempt_count, 6)
        if attempt_count > 0
        else 0.0,
        "max_duration_s": round(max(durations), 6) if durations else 0.0,
    }


def _record_attempt_timing(
    *,
    logger: logging.Logger,
    timings: list[dict[str, Any]],
    phase: str,
    attempt: int,
    max_attempts: int,
    attempt_start: float,
    success: bool,
    error: str | None = None,
    usage: Mapping[str, Any] | None = None,
) -> None:
    attempt_end = time.time()
    record = {
        "phase": phase,
        "attempt": int(attempt),
        "max_attempts": int(max_attempts),
        "start_time": attempt_start,
        "end_time": attempt_end,
        "duration_s": round(max(0.0, attempt_end - attempt_start), 6),
        "success": bool(success),
        "error": error,
    }
    if isinstance(usage, Mapping):
        for key in ("message_count", "steps_used", "runtime_s"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                record[str(key)] = value
    timings.append(record)
    logger.info(
        "attempt_timing "
        f"phase={phase} attempt={attempt}/{max_attempts} "
        f"duration_s={record['duration_s']:.6f} success={success}"
    )


def _append_attempt_timing_sidecar(
    *,
    run_dir: Path | None,
    iteration: int,
    attempt_timings: list[dict[str, Any]],
) -> None:
    if run_dir is None or not attempt_timings:
        return
    path = Path(run_dir) / "agent_attempt_timings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in attempt_timings:
            payload = {
                "agent_type": "meta",
                "iteration": int(iteration),
                **item,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_skill_bundle_sidecar(
    *,
    run_dir: Path | None,
    iteration: int,
    skill_bundle: Mapping[str, Any],
) -> None:
    if run_dir is None:
        return
    path = Path(run_dir) / "skill_bundle_provenance.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_type": "meta",
        "iteration": int(iteration),
        **dict(skill_bundle),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Pi session creation
# ---------------------------------------------------------------------------

_META_SYSTEM_PROMPT = (
    "You are the MCE meta-agent. Execute the user request faithfully. "
    "Use tools to inspect workspace history and produce required files."
)


def _create_meta_pi_session(
    workspace_base: Path,
    model: str | None,
    system_prompt: str,
    skill_database: str,
    run_dir: Path | None = None,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
    iteration: int | None = None,
) -> AgentSession:
    """Create Pi session for meta-agent.

    - cwd = workspace_base (read access to all prior iterations)
    - AGENTS.md carries only system-level meta-agent instructions
    - No skill_paths (meta-agent doesn't use loaded skills)
    """
    write_agents_md(
        iter_dir=workspace_base,
        system_prompt=system_prompt,
        skill_guidance=None,
    )

    client = start_pi_session_client(
        cwd=workspace_base,
        model=model or "",
        skill_paths=[],
        run_dir=run_dir,
        timeout_s=timeout_s,
        session_traces_enabled=session_traces_enabled,
    )
    return wrap_pi_session_client_as_session(client, cwd=workspace_base)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_meta_agent(
    iter_dir: Path,
    task_instruction: str,
    interface_signatures: list,
    iteration: int,
    model: str | None = None,
    workspace_base: Path | None = None,
    run_dir: Path | None = None,
    max_validation_attempts: int = 3,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
) -> Dict[str, Any]:
    """Run meta-agent to evolve skills for base-level learning.

    Uses a Pi session with cwd=workspace_base. The Pi agent writes SKILL.md
    via its native write tool. Python validates the output from disk.
    """
    workspace_base = Path(iter_dir.parent) if workspace_base is None else Path(workspace_base)

    logger = setup_logger(
        name=f"meta_iter{iteration}",
        run_dir=run_dir,
        agent_type="meta",
        iteration=iteration,
        minimal_console=True,
    )

    meta_prompt = build_meta_agent_prompt(
        task_instruction=task_instruction,
        interface_signatures=interface_signatures,
        iter_dir=str(iter_dir),
        workspace_base=str(workspace_base),
    )

    logger.info("📝 META-AGENT PROMPT:")
    logger.info(f"\n{meta_prompt}\n")

    # Compose skill bundle for telemetry.
    skill_bundle = compose_skill_bundle(
        workspace_base=workspace_base,
        iter_folder_name=iter_dir.name,
        include_history=True,
        include_current=True,
    )
    if skill_bundle.excluded_sources:
        logger.warning(
            "Excluded invalid overlay skill source(s): "
            + ", ".join(skill_bundle.excluded_sources)
        )
    logger.info(
        "Resolved skill bundle: hash=%s sources=%s",
        skill_bundle.bundle_hash,
        skill_bundle.selected_sources,
    )

    # Build skill database for AGENTS.md embedding.
    skill_database = _build_skill_database(str(workspace_base), iteration)

    # Create Pi session: cwd=workspace_base, no skills.
    session = _create_meta_pi_session(
        workspace_base=workspace_base,
        model=model,
        system_prompt=_META_SYSTEM_PROMPT,
        skill_database=skill_database,
        run_dir=run_dir,
        timeout_s=timeout_s,
        session_traces_enabled=session_traces_enabled,
        iteration=iteration,
    )

    pending_prompt = meta_prompt
    last_validation_error: str | None = None
    attempt_timings: list[dict[str, Any]] = []

    def _finalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(result)
        payload["attempt_timings"] = list(attempt_timings)
        payload["attempt_timing_summary"] = _summarize_attempt_timings(attempt_timings)
        payload["skill_bundle"] = skill_bundle.to_dict()
        _append_attempt_timing_sidecar(
            run_dir=run_dir,
            iteration=iteration,
            attempt_timings=attempt_timings,
        )
        _append_skill_bundle_sidecar(
            run_dir=run_dir,
            iteration=iteration,
            skill_bundle=skill_bundle.to_dict(),
        )
        return payload

    try:
        for attempt in range(max_validation_attempts):
            logger.info(f"\n--- Validation attempt {attempt + 1}/{max_validation_attempts} ---")
            attempt_start = time.time()

            try:
                response = await session.send_message(pending_prompt)
            except Exception as exc:
                logger.error(f"Meta-agent execution failed: {exc}", exc_info=True)
                _record_attempt_timing(
                    logger=logger,
                    timings=attempt_timings,
                    phase="validation",
                    attempt=attempt + 1,
                    max_attempts=max_validation_attempts,
                    attempt_start=attempt_start,
                    success=False,
                    error=str(exc),
                )
                cleanup_irrelevant_files(iter_dir, agent_type="meta", logger=logger)
                return _finalize_result({
                    "success": False,
                    "error": str(exc),
                    "skill_md": None,
                })

            # Log Pi response.
            if response.error:
                logger.warning(f"Pi session returned error: {response.error}")
            if response.content:
                snippet = (
                    response.content
                    if len(response.content) <= 1500
                    else f"{response.content[:1500]} ..."
                )
                logger.info(f"Pi response ({len(response.content)} chars): {snippet}")

            _warn_on_mirrored_host_skill_artifacts(
                workspace_base=workspace_base,
                iter_folder_name=iter_dir.name,
                logger=logger,
            )

            verification_result = _verify_meta_agent_outputs(iter_dir, logger)
            if verification_result["success"]:
                skill_guard = _apply_structural_skill_antiregression_guard(
                    iter_dir=iter_dir,
                    workspace_base=workspace_base,
                    iteration=iteration,
                    logger=logger,
                )
                verification_result["skill_guard"] = skill_guard
                verification_result["skill_md"] = learning_context_skill_host_path(iter_dir).read_text(
                    encoding="utf-8"
                )
                _record_attempt_timing(
                    logger=logger,
                    timings=attempt_timings,
                    phase="validation",
                    attempt=attempt + 1,
                    max_attempts=max_validation_attempts,
                    attempt_start=attempt_start,
                    success=True,
                )
                cleanup_irrelevant_files(iter_dir, agent_type="meta", logger=logger)
                return _finalize_result(verification_result)

            last_validation_error = verification_result["error"]
            logger.warning(f"❌ SKILL.md validation failed: {verification_result['error']}")
            _record_attempt_timing(
                logger=logger,
                timings=attempt_timings,
                phase="validation",
                attempt=attempt + 1,
                max_attempts=max_validation_attempts,
                attempt_start=attempt_start,
                success=False,
                error=str(verification_result["error"] or ""),
            )

            if attempt + 1 >= max_validation_attempts:
                logger.error(f"Max validation attempts ({max_validation_attempts}) exceeded")
                break

            expected_host_path = learning_context_skill_host_path(iter_dir)
            expected_virtual_path = learning_context_skill_virtual_path(iter_dir.name)
            validation_error = verification_result["error"].replace(
                str(expected_host_path),
                expected_virtual_path,
            )
            pending_prompt = _build_skill_validation_retry_prompt(
                validation_error=validation_error,
                expected_virtual_path=expected_virtual_path,
            )

        cleanup_irrelevant_files(iter_dir, agent_type="meta", logger=logger)
        final_error = f"Meta-agent failed to generate SKILL.md after {max_validation_attempts} attempts"
        if last_validation_error:
            final_error = f"{final_error}. Last error: {last_validation_error}"
        return _finalize_result({
            "success": False,
            "error": final_error,
            "skill_md": None,
        })
    finally:
        await session.close()
