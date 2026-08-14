"""Base-agent implementation using Pi sessions."""

from __future__ import annotations

import importlib
import hashlib
import json
import logging
import py_compile
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping

from evo_metaoptics.mce.agent_runtime import create_pi_session
from evo_metaoptics.mce.logging_utils import setup_logger
from evo_metaoptics.mce.prompts.base_agent import build_base_agent_prompt
from evo_metaoptics.mce.skills import compose_skill_bundle, learning_context_skill_host_path
from evo_metaoptics.mce.utils import cleanup_irrelevant_files
from evo_metaoptics.mce.validation import (
    ValidationResult,
    format_validation_feedback,
    validate_interfaces,
)
from evo_metaoptics.mce_env.base import InterfaceSignature

try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


load_dotenv(override=True)

_BASE_SYSTEM_PROMPT = (
    "You are the MCE base-agent. Produce required interface implementations "
    "inside the workspace and follow user instructions exactly."
)


# ---------------------------------------------------------------------------
# Pure helpers — validation, hashing, timing  (preserved from original)
# ---------------------------------------------------------------------------

def _normalize_required_context_paths(required_context_files: list[str]) -> list[str]:
    """Normalize required context-file paths into safe relative POSIX strings."""
    normalized: list[str] = []
    for raw_path in required_context_files:
        candidate = PurePosixPath(str(raw_path).strip())
        if not candidate.parts:
            continue
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                f"Invalid required context path '{raw_path}': must be relative and cannot contain '..'."
            )
        normalized.append(candidate.as_posix())
    return normalized


def _compute_required_context_contract_hash(
    *,
    iter_dir: Path,
    required_context_files: list[str],
) -> str | None:
    """Compute deterministic hash over required context artifacts only."""
    normalized = _normalize_required_context_paths(required_context_files)
    if not normalized:
        return None

    context_root = Path(iter_dir) / "context"
    rows: list[dict[str, Any]] = []
    for rel_path in sorted(normalized):
        host_path = context_root / Path(*PurePosixPath(rel_path).parts)
        row: dict[str, Any] = {"path": rel_path}
        if not host_path.exists() or not host_path.is_file():
            row["state"] = "missing"
        else:
            try:
                raw = host_path.read_bytes()
            except Exception as exc:
                row["state"] = "read_error"
                row["error"] = str(exc)
            else:
                row["state"] = "file"
                row["size"] = len(raw)
                row["sha256"] = hashlib.sha256(raw).hexdigest()
        rows.append(row)

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_context_validation_feedback(
    errors: list[str],
    *,
    required_context_files: list[str],
    min_context_chars: int,
    error_classes: list[str],
    retry_attempt: int,
    max_attempts: int,
) -> str:
    """Build retry guidance when context-only output quality checks fail."""
    bullets = "\n".join(f"- {error}" for error in errors)
    required_block = ""
    if required_context_files:
        required_lines = "\n".join(
            f"- context/{path} (>= {min_context_chars} chars)"
            for path in required_context_files
        )
        update_lines = "\n".join(
            f"- Edit or append in place: context/{path}"
            for path in required_context_files
        )
        required_block = f"""
Required artifact contract:
{required_lines}

Canonical update behavior:
{update_lines}
"""
    format_guidance_block = ""
    if required_context_files:
        format_guidance_block = """
Context format guidance (format-agnostic):
- Use concise, actionable lines that are easy to distill.
- Any text structure is acceptable; strict tag formatting is not required.
- Optional `context/examples.json` can be used when helpful.
"""

    attempt_label = f"Retry attempt {retry_attempt}/{max_attempts}"
    class_guidance_map = {
        "tool_not_called": "Write at least one context file now under context/.",
        "tool_wrong_args": "Use only allowed relative paths under context/.",
        "model_refused": "Do not refuse. Execute the write action directly.",
        "timeout": "Write one concise context file first, then stop.",
        "stagnation": "Change strategy from previous attempt and update file content.",
        "validation_error": "Fix the reported validation errors directly.",
    }
    class_guidance = "\n".join(
        f"- {klass}: {class_guidance_map.get(klass, class_guidance_map['validation_error'])}"
        for klass in error_classes
    )
    if not class_guidance:
        class_guidance = f"- validation_error: {class_guidance_map['validation_error']}"
    few_shot_block = """
Few-shot examples:
- Write `context/analysis.md` with root causes and fixes from train.json
- Write `context/rules.txt` with rules like "If solver timeout then reduce stage steps"
"""

    if retry_attempt >= max_attempts:
        action_block = """
Single required action now:
1. Write one valid context artifact under `context/`.
2. The file must contain task-relevant content.
3. Do not return without writing a file.
"""
    else:
        action_block = """
Please update files under `context/` now.
Requirements:
1. Write at least one non-empty, task-relevant context file in `context/`.
2. Do not write mirrored host-absolute artifacts like `context/Users/...`.
3. Keep the output practical and concise for downstream diagnosis usage.
4. Use canonical schema examples aligned with current runtime contracts.
"""

    return f"""
⚠️ CONTEXT OUTPUT VALIDATION FAILED ({attempt_label})

The current run did not produce acceptable context artifacts:
{bullets}
{required_block}
{format_guidance_block}
Error classes and recovery strategy:
{class_guidance}
{few_shot_block}
{action_block}
"""


def _validate_context_outputs(
    iter_dir: Path,
    *,
    required_context_files: list[str],
    min_context_chars: int,
) -> tuple[bool, list[str], list[str]]:
    """Validate that context-only runs produce meaningful, clean artifacts."""
    context_dir = Path(iter_dir) / "context"
    errors: list[str] = []
    warnings: list[str] = []

    if not context_dir.exists():
        return False, [f"Missing context directory: {context_dir}"], warnings

    mirrored_root = context_dir / "Users"
    mirrored_artifacts = [p for p in mirrored_root.rglob("*") if p.is_file()] if mirrored_root.exists() else []
    if mirrored_artifacts:
        errors.append("Detected mirrored host-path artifacts under context/Users.")

    files = [p for p in context_dir.rglob("*") if p.is_file()]
    if not files:
        errors.append("No context files were generated.")
        return False, errors, warnings

    required_rel_paths = _normalize_required_context_paths(required_context_files)
    if required_rel_paths:
        for rel_path in required_rel_paths:
            target = context_dir / Path(*PurePosixPath(rel_path).parts)
            if not target.exists() or not target.is_file():
                errors.append(f"Missing required context file: {rel_path}")
                continue

            try:
                text = target.read_text(encoding="utf-8").strip()
            except Exception:
                errors.append(f"Failed reading required context file: {rel_path}")
                continue

            if len(text) < min_context_chars:
                errors.append(
                    f"Required context file too short: {rel_path} "
                    f"(found {len(text)} chars, need >= {min_context_chars})"
                )
        warnings.extend(
            _collect_nonfatal_context_format_warnings(
                context_dir=context_dir,
                required_rel_paths=required_rel_paths,
            )
        )
    else:
        meaningful_files = []
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if len(text) >= min_context_chars:
                meaningful_files.append(file_path)

        if not meaningful_files:
            errors.append(
                f"All context files are empty or too short to be useful (need >= {min_context_chars} chars)."
            )

    return len(errors) == 0, errors, warnings


def _collect_nonfatal_context_format_warnings(
    *,
    context_dir: Path,
    required_rel_paths: list[str],
) -> list[str]:
    del context_dir, required_rel_paths
    return []


def _is_cold_start(iteration: int | None) -> bool:
    """Return True if iteration is None or iteration <= 1 (cold start)."""
    return iteration is None or iteration <= 1


def _has_any_context_files(iter_dir: Path) -> bool:
    """Check if any context files exist in iter_dir/context/."""
    context_dir = Path(iter_dir) / "context"
    if not context_dir.exists():
        return False
    files = [p for p in context_dir.rglob("*") if p.is_file()]
    return len(files) > 0


def _write_default_context(
    iter_dir: Path,
    logger: logging.Logger,
    *,
    required_context_files: list[str],
    min_context_chars: int,
) -> list[str]:
    base_context_text = (
        "# Bootstrap Context (auto-generated)\n\n"
        "No training patterns have been identified yet.\n"
        "This is the initial iteration. Context will be populated after analyzing training results.\n\n"
        "## Guidelines for Next Iteration\n\n"
        "- Review `data/train.json`\n"
        "- Identify common failure patterns\n"
        "- Write actionable rules in `context/` artifacts\n"
    )

    context_dir = Path(iter_dir) / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[str] = []

    bootstrap_path = context_dir / "bootstrap.md"
    bootstrap_path.write_text(base_context_text, encoding="utf-8")
    written_paths.append("bootstrap.md")

    normalized_required = _normalize_required_context_paths(required_context_files)
    for rel_path in normalized_required:
        target = context_dir / Path(*PurePosixPath(rel_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        required_text = (
            f"{base_context_text}\n"
            f"Target artifact: context/{rel_path}\n"
            "Placeholder content generated by cold-start fallback.\n"
        )
        if len(required_text.strip()) < min_context_chars:
            pad = "placeholder " * max(1, ((min_context_chars - len(required_text)) // 12) + 2)
            required_text = f"{required_text}{pad}"
        target.write_text(required_text.strip(), encoding="utf-8")
        written_paths.append(rel_path)

    logger.warning(
        "Cold-start fallback wrote default context artifacts: "
        + ", ".join(written_paths)
    )
    return written_paths


def _all_training_results_zero_success(iter_dir: Path) -> bool:
    train_path = Path(iter_dir) / "data" / "train.json"
    if not train_path.is_file():
        return False

    try:
        payload = json.loads(train_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False

    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("success_goal", "train_success_goal"):
            value = summary.get(key)
            if isinstance(value, (int, float)) and float(value) > 0.0:
                return False

        metrics = summary.get("train_metrics")
        if isinstance(metrics, dict):
            value = metrics.get("success_goal")
            if isinstance(value, (int, float)) and float(value) > 0.0:
                return False

    detailed = payload.get("detailed_results")
    if isinstance(detailed, list):
        saw_success_field = False
        for row in detailed:
            if not isinstance(row, dict):
                continue
            if "success_goal" not in row:
                continue
            saw_success_field = True
            if bool(row.get("success_goal")):
                return False
        if saw_success_field:
            return True

    if isinstance(summary, dict):
        metrics = summary.get("train_metrics")
        if isinstance(metrics, dict):
            value = metrics.get("success_goal")
            if isinstance(value, (int, float)):
                return float(value) <= 0.0

    return False


def _is_lenient_context_error(error: str) -> bool:
    markers = (
        "Missing context directory",
        "No context files were generated",
        "All context files are empty or too short",
        "Missing required context file",
        "Required context file too short",
    )
    return any(marker in error for marker in markers)


def _should_lenient_accept_context(iter_dir: Path, errors: list[str]) -> bool:
    if not errors:
        return False
    if not _all_training_results_zero_success(iter_dir):
        return False
    return all(_is_lenient_context_error(error) for error in errors)


def _classify_context_error(error: str) -> str:
    lower = error.lower()
    if "no context files were generated" in lower:
        return "tool_not_called"
    if "missing context directory" in lower:
        return "tool_not_called"
    if "missing required context file" in lower:
        return "tool_not_called"
    if "path is not allowed by context output contract" in lower:
        return "tool_wrong_args"
    if "path must be relative" in lower:
        return "tool_wrong_args"
    if "path cannot contain parent-directory traversal" in lower:
        return "tool_wrong_args"
    if "i cannot" in lower or "i'm sorry" in lower or "i am sorry" in lower:
        return "model_refused"
    if "timeout" in lower:
        return "timeout"
    if "hash delta = 0" in lower:
        return "stagnation"
    return "validation_error"


def _classify_context_errors(errors: list[str]) -> list[str]:
    classes: list[str] = []
    for error in errors:
        klass = _classify_context_error(error)
        if klass not in classes:
            classes.append(klass)
    return classes


def _run_ruff_fix_on_interfaces(iter_dir: Path, logger: logging.Logger) -> None:
    """Best-effort auto-fix pass on generated interface Python files."""
    interfaces_dir = Path(iter_dir) / "interfaces"
    if not interfaces_dir.exists() or not interfaces_dir.is_dir():
        return

    ruff_bin = shutil.which("ruff")
    if ruff_bin:
        cmd = [ruff_bin, "check", "--fix", str(interfaces_dir)]
    else:
        cmd = [sys.executable, "-m", "ruff", "check", "--fix", str(interfaces_dir)]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        logger.warning(f"⚠️ Ruff auto-fix skipped: {exc}")
        return

    if result.returncode == 0:
        return

    stderr_text = (result.stderr or "").strip()
    stdout_text = (result.stdout or "").strip()
    snippet = stderr_text or stdout_text
    if len(snippet) > 220:
        snippet = snippet[:220] + "..."
    if snippet:
        logger.warning(f"⚠️ Ruff auto-fix reported issues: {snippet}")


def _collect_interface_syntax_errors(iter_dir: Path) -> list[str]:
    """Return syntax precheck errors for interface Python modules."""
    interfaces_dir = Path(iter_dir) / "interfaces"
    if not interfaces_dir.exists() or not interfaces_dir.is_dir():
        return []

    errors: list[str] = []
    for path in sorted(interfaces_dir.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            detail = str(exc).strip().replace("\n", " ")
            errors.append(
                f"Syntax precheck failed in interfaces/{path.name}: {detail}"
            )
        except Exception as exc:
            errors.append(
                f"Syntax precheck failed in interfaces/{path.name}: {exc}"
            )
    return errors


# ---------------------------------------------------------------------------
# Timing / telemetry helpers (preserved from original)
# ---------------------------------------------------------------------------

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
    attempt_count = len(attempt_timings)
    total_duration = sum(durations)
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
    iteration: int | None,
    sub_iteration: int | None,
    attempt_timings: list[dict[str, Any]],
) -> None:
    if run_dir is None or not attempt_timings:
        return
    path = Path(run_dir) / "agent_attempt_timings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in attempt_timings:
            payload = {
                "agent_type": "base",
                "iteration": iteration,
                "sub_iteration": sub_iteration,
                **item,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_skill_bundle_sidecar(
    *,
    run_dir: Path | None,
    iteration: int | None,
    sub_iteration: int | None,
    skill_bundle: Mapping[str, Any],
) -> None:
    if run_dir is None:
        return
    path = Path(run_dir) / "skill_bundle_provenance.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_type": "base",
        "iteration": iteration,
        "sub_iteration": sub_iteration,
        **dict(skill_bundle),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main entry point — Pi session based
# ---------------------------------------------------------------------------

async def create_mce_deep_agent(
    backend_root: Path,
    task_instruction: str = "",
    interface_signatures: list | None = None,
    model: str = "",
    *,
    filesystem_write_prefixes: list[str] | None = None,
    overwrite_allowed_write_paths: list[str] | None = None,
    middleware: list | None = None,
    run_dir: Path | None = None,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
    **kwargs,
) -> Any:
    del task_instruction, interface_signatures, filesystem_write_prefixes
    del overwrite_allowed_write_paths, middleware, kwargs
    skill_paths = [str(Path(backend_root) / ".agents" / "skills" / "learning-context")]
    session = create_pi_session(
        iter_dir=Path(backend_root),
        system_prompt=_BASE_SYSTEM_PROMPT,
        skills=[],
        model=model,
        skill_paths=skill_paths,
        run_dir=run_dir,
        timeout_s=timeout_s,
        session_traces_enabled=session_traces_enabled,
    )
    session.backend_root = Path(backend_root)  # type: ignore[attr-defined]
    return session


async def invoke_mce_deep_agent(
    agent: Any,
    prompt: str,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    if hasattr(agent, "send_message") and callable(agent.send_message):
        response = await agent.send_message(prompt)
        return {
            "messages": [
                {"role": "assistant", "content": response.content or ""}
            ]
        }
    if hasattr(agent, "invoke") and callable(agent.invoke):
        result = agent.invoke(prompt, config={"thread_id": thread_id})
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, Mapping):
            messages = result.get("messages")
            if isinstance(messages, list):
                return {"messages": messages}
        return {"messages": [{"role": "assistant", "content": str(result) if result is not None else ""}]}
    raise AttributeError("agent does not support send_message or invoke")


async def run_base_agent(
    iter_dir: Path,
    task_instruction: str,
    interface_signatures: List[InterfaceSignature],
    model: str | None = None,
    workspace_base: Path | None = None,
    log_dir: str = "logs",
    run_dir: Path | None = None,
    iteration: int | None = None,
    initial_prompt: str | None = None,
    max_validation_attempts: int = 3,
    required_context_files: List[str] | None = None,
    context_min_chars: int = 30,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
) -> Dict[str, Any]:
    """Run base-agent with validation/retry loop using Pi sessions."""
    sub_iteration = None
    iter_dir_name = Path(iter_dir).name
    if "_sub" in iter_dir_name:
        sub_iteration = int(iter_dir_name.split("_sub")[1])

    if run_dir and iteration is not None:
        logger = setup_logger(
            name=(
                f"base_iter{iteration}_sub{sub_iteration}"
                if sub_iteration is not None
                else f"base_iter{iteration}"
            ),
            run_dir=run_dir,
            agent_type="base",
            iteration=iteration,
            sub_iteration=sub_iteration,
            minimal_console=True,
        )
    else:
        logger = setup_logger(name="base_agent", log_dir=log_dir, console_colors=True)

    logger.info("\n🤖 BASE-AGENT: Learning context")
    logger.info(f"  Iteration directory: {iter_dir}")
    logger.info(f"  Required interfaces: {[s.name for s in interface_signatures]}")

    if workspace_base is None:
        workspace_base = iter_dir.parent
    workspace_base = Path(workspace_base)
    normalized_required_context_paths = _normalize_required_context_paths(
        required_context_files or []
    )
    context_hash_before = _compute_required_context_contract_hash(
        iter_dir=iter_dir,
        required_context_files=normalized_required_context_paths,
    )
    logger.info(
        "  Required context files: "
        f"{normalized_required_context_paths if normalized_required_context_paths else '[any meaningful file]'}"
    )
    if isinstance(context_hash_before, str):
        logger.info(f"  Required-context hash (before): {context_hash_before}")

    full_prompt = build_base_agent_prompt(
        task_instruction=task_instruction,
        interface_signatures=interface_signatures,
        iter_dir=str(iter_dir),
        workspace_base=str(workspace_base),
        initial_prompt=initial_prompt,
        required_context_files=normalized_required_context_paths,
        min_context_chars=context_min_chars,
    )

    logger.info("\n" + "=" * 80)
    logger.info("📝 BASE-AGENT PROMPT")
    logger.info("=" * 80)
    logger.info(f"\n{full_prompt}\n")
    logger.info("=" * 80 + "\n")

    # Compose skill bundle for provenance tracking
    skill_bundle = compose_skill_bundle(
        workspace_base=workspace_base,
        iter_folder_name=iter_dir.name,
        include_history=False,
        include_current=True,
        backend_scope="iteration",
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

    session = await create_mce_deep_agent(
        backend_root=iter_dir,
        task_instruction=task_instruction,
        interface_signatures=interface_signatures,
        model=model or "",
        filesystem_write_prefixes=["/context/", "/interfaces/", "/.agents/", "/data/"],
        overwrite_allowed_write_paths=[
            f"/context/{p}" for p in normalized_required_context_paths
        ],
        middleware=[{"type": "path_sanitizer"}],
        run_dir=run_dir,
        timeout_s=timeout_s,
        session_traces_enabled=session_traces_enabled,
    )

    pending_prompt = full_prompt
    attempt_timings: list[dict[str, Any]] = []

    def _context_hash_telemetry(*, warn_if_unchanged: bool) -> dict[str, Any]:
        context_hash_after = _compute_required_context_contract_hash(
            iter_dir=iter_dir,
            required_context_files=normalized_required_context_paths,
        )
        context_hash_changed: bool | None = None
        if isinstance(context_hash_before, str) and isinstance(context_hash_after, str):
            context_hash_changed = context_hash_before != context_hash_after
            logger.info(f"  Required-context hash (after): {context_hash_after}")
            if warn_if_unchanged and not context_hash_changed:
                logger.warning(
                    "⚠️ Required context artifacts unchanged (hash delta = 0). "
                    "This is non-fatal but indicates low context-learning progress."
                )
        return {
            "context_hash_scope": list(normalized_required_context_paths),
            "context_hash_before": context_hash_before,
            "context_hash_after": context_hash_after,
            "context_hash_changed": context_hash_changed,
        }

    def _finalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(result)
        payload["attempt_timings"] = list(attempt_timings)
        payload["attempt_timing_summary"] = _summarize_attempt_timings(attempt_timings)
        payload["skill_bundle"] = skill_bundle.to_dict()
        _append_attempt_timing_sidecar(
            run_dir=run_dir,
            iteration=iteration,
            sub_iteration=sub_iteration,
            attempt_timings=attempt_timings,
        )
        _append_skill_bundle_sidecar(
            run_dir=run_dir,
            iteration=iteration,
            sub_iteration=sub_iteration,
            skill_bundle=skill_bundle.to_dict(),
        )
        return payload

    def _coerce_legacy_response(raw_response: Any, invoke_error: str | None = None) -> Any:
        content = ""
        if isinstance(raw_response, Mapping):
            messages = raw_response.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, Mapping):
                    candidate = last.get("content")
                    if isinstance(candidate, str):
                        content = candidate
        return type(
            "_Resp",
            (),
            {
                "content": content,
                "error": invoke_error,
            },
        )()

    def _maybe_extract_response_to_context(content: str) -> None:
        if normalized_required_context_paths:
            return
        if _has_any_context_files(iter_dir):
            return
        trimmed = content.strip()
        if len(trimmed) < 120:
            return
        context_dir = Path(iter_dir) / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        (context_dir / "analysis.md").write_text(trimmed, encoding="utf-8")

    try:
        if not interface_signatures:
            # ---------------------------------------------------------------
            # Context-only path (no interface signatures)
            # ---------------------------------------------------------------
            validation_errors: list[str] = []
            for attempt in range(max_validation_attempts):
                logger.info(
                    f"\n--- Context validation attempt {attempt + 1}/{max_validation_attempts} ---"
                )
                attempt_start = time.time()

                invoke_error: str | None = None
                try:
                    raw_response = await invoke_mce_deep_agent(
                        session,
                        pending_prompt,
                        thread_id=None,
                    )
                except Exception as exc:
                    invoke_error = str(exc)
                    raw_response = {
                        "messages": [
                            {"role": "assistant", "content": invoke_error}
                        ]
                    }
                response = _coerce_legacy_response(raw_response, invoke_error)
                _maybe_extract_response_to_context(response.content)
                if response.error:
                    _record_attempt_timing(
                        logger=logger,
                        timings=attempt_timings,
                        phase="context_validation",
                        attempt=attempt + 1,
                        max_attempts=max_validation_attempts,
                        attempt_start=attempt_start,
                        success=False,
                        error=response.error,
                    )
                    validation_errors = [response.error]
                    if attempt + 1 >= max_validation_attempts:
                        break
                    pending_prompt = _build_context_validation_feedback(
                        validation_errors,
                        required_context_files=normalized_required_context_paths,
                        min_context_chars=context_min_chars,
                        error_classes=_classify_context_errors(validation_errors),
                        retry_attempt=attempt + 2,
                        max_attempts=max_validation_attempts,
                    )
                    logger.info("📤 Sending context validation feedback to agent...")
                    continue

                logger.info("Agent completed")

                # Validate context outputs on disk
                context_ok, validation_errors, context_warnings = _validate_context_outputs(
                    iter_dir,
                    required_context_files=normalized_required_context_paths,
                    min_context_chars=context_min_chars,
                )
                for warning in context_warnings:
                    logger.warning(f"⚠️ {warning}")
                if context_ok:
                    context_hash_after = _compute_required_context_contract_hash(
                        iter_dir=iter_dir,
                        required_context_files=normalized_required_context_paths,
                    )
                    context_hash_changed: bool | None = None
                    if isinstance(context_hash_before, str) and isinstance(context_hash_after, str):
                        context_hash_changed = context_hash_before != context_hash_after
                    if (
                        normalized_required_context_paths
                        and context_hash_changed is False
                    ):
                        unchanged_error = (
                            "Required context artifacts unchanged (hash delta = 0). "
                            "Update at least one required context file in-place."
                        )
                        validation_errors = [unchanged_error]
                        logger.warning(f"❌ {unchanged_error}")
                        _record_attempt_timing(
                            logger=logger,
                            timings=attempt_timings,
                            phase="context_validation",
                            attempt=attempt + 1,
                            max_attempts=max_validation_attempts,
                            attempt_start=attempt_start,
                            success=False,
                            error=unchanged_error,
                        )
                        if attempt + 1 >= max_validation_attempts:
                            break
                        pending_prompt = _build_context_validation_feedback(
                            validation_errors,
                            required_context_files=normalized_required_context_paths,
                            min_context_chars=context_min_chars,
                            error_classes=_classify_context_errors(validation_errors),
                            retry_attempt=attempt + 2,
                            max_attempts=max_validation_attempts,
                        )
                        logger.info("📤 Sending context hash-change feedback to agent...")
                        continue
                    _record_attempt_timing(
                        logger=logger,
                        timings=attempt_timings,
                        phase="context_validation",
                        attempt=attempt + 1,
                        max_attempts=max_validation_attempts,
                        attempt_start=attempt_start,
                        success=True,
                    )
                    cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
                    return _finalize_result({
                        "success": True,
                        "interfaces": {},
                        "message_count": 0,
                        "validation_attempts": attempt + 1,
                        **_context_hash_telemetry(warn_if_unchanged=True),
                    })

                if (
                    _is_cold_start(iteration)
                    and not normalized_required_context_paths
                    and _should_lenient_accept_context(iter_dir, validation_errors)
                ):
                    logger.warning(
                        "⚠️ Cold-start lenient acceptance: empty context accepted "
                        "because training results show 0 success."
                    )
                    _record_attempt_timing(
                        logger=logger,
                        timings=attempt_timings,
                        phase="context_validation",
                        attempt=attempt + 1,
                        max_attempts=max_validation_attempts,
                        attempt_start=attempt_start,
                        success=True,
                        error="lenient_acceptance_zero_success",
                    )
                    cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
                    return _finalize_result({
                        "success": True,
                        "interfaces": {},
                        "message_count": 0,
                        "validation_attempts": attempt + 1,
                        "context_lenient_acceptance": True,
                        **_context_hash_telemetry(warn_if_unchanged=True),
                    })

                logger.warning(
                    "❌ Context output validation failed with "
                    f"{len(validation_errors)} error(s):"
                )
                for error in validation_errors:
                    logger.warning(f"  - {error}")
                _record_attempt_timing(
                    logger=logger,
                    timings=attempt_timings,
                    phase="context_validation",
                    attempt=attempt + 1,
                    max_attempts=max_validation_attempts,
                    attempt_start=attempt_start,
                    success=False,
                    error="; ".join(validation_errors[:3]) if validation_errors else None,
                )

                if attempt + 1 >= max_validation_attempts:
                    break

                pending_prompt = _build_context_validation_feedback(
                    validation_errors,
                    required_context_files=normalized_required_context_paths,
                    min_context_chars=context_min_chars,
                    error_classes=_classify_context_errors(validation_errors),
                    retry_attempt=attempt + 2,
                    max_attempts=max_validation_attempts,
                )
                logger.info("📤 Sending context validation feedback to agent...")

            if _is_cold_start(iteration) and _all_training_results_zero_success(iter_dir):
                written_paths = _write_default_context(
                    iter_dir,
                    logger,
                    required_context_files=normalized_required_context_paths,
                    min_context_chars=context_min_chars,
                )
                context_ok, _, _ = _validate_context_outputs(
                    iter_dir,
                    required_context_files=normalized_required_context_paths,
                    min_context_chars=context_min_chars,
                )
                if context_ok:
                    logger.info("✅ Cold-start: default context written, proceeding.")
                    cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
                    return _finalize_result({
                        "success": True,
                        "interfaces": {},
                        "message_count": 0,
                        "validation_attempts": max_validation_attempts,
                        "cold_start_fallback": True,
                        "cold_start_written_paths": written_paths,
                        **_context_hash_telemetry(warn_if_unchanged=True),
                    })

            cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
            return _finalize_result({
                "success": False,
                "error": (
                    "Context validation failed after "
                    f"{max_validation_attempts} attempt(s)"
                ),
                "last_errors": validation_errors,
                "error_classes": _classify_context_errors(validation_errors),
                "message_count": 0,
                **_context_hash_telemetry(warn_if_unchanged=False),
            })

        # -------------------------------------------------------------------
        # Interface + context path
        # -------------------------------------------------------------------
        validation_result = None
        last_context_errors: list[str] = []
        for attempt in range(max_validation_attempts):
            logger.info(f"\n--- Validation attempt {attempt + 1}/{max_validation_attempts} ---")
            attempt_start = time.time()

            invoke_error: str | None = None
            try:
                raw_response = await invoke_mce_deep_agent(
                    session,
                    pending_prompt,
                    thread_id=None,
                )
            except Exception as exc:
                invoke_error = str(exc)
                raw_response = {
                    "messages": [
                        {"role": "assistant", "content": invoke_error}
                    ]
                }
            response = _coerce_legacy_response(raw_response, invoke_error)
            if response.error:
                _record_attempt_timing(
                    logger=logger,
                    timings=attempt_timings,
                    phase="validation",
                    attempt=attempt + 1,
                    max_attempts=max_validation_attempts,
                    attempt_start=attempt_start,
                    success=False,
                    error=response.error,
                )
                validation_result = ValidationResult(
                    success=False,
                    errors=[f"Agent invocation error: {response.error}"],
                    interfaces={},
                )
                if attempt + 1 >= max_validation_attempts:
                    logger.error(f"Max validation attempts ({max_validation_attempts}) exceeded")
                    break
                pending_prompt = format_validation_feedback(validation_result)
                logger.info("📤 Sending validation feedback to agent...")
                continue

                logger.info("Agent completed")

            _run_ruff_fix_on_interfaces(iter_dir, logger)
            syntax_errors = _collect_interface_syntax_errors(iter_dir)
            if syntax_errors:
                validation_result = ValidationResult(
                    success=False,
                    errors=syntax_errors,
                    interfaces={},
                )
            else:
                validation_result = validate_interfaces(iter_dir, interface_signatures)
            if validation_result.success:
                context_ok, context_errors, context_warnings = _validate_context_outputs(
                    iter_dir,
                    required_context_files=normalized_required_context_paths,
                    min_context_chars=context_min_chars,
                )
                for warning in context_warnings:
                    logger.warning(f"⚠️ {warning}")
                if not context_ok:
                    last_context_errors = context_errors
                    if (
                        _is_cold_start(iteration)
                        and not normalized_required_context_paths
                        and _should_lenient_accept_context(iter_dir, context_errors)
                    ):
                        logger.warning(
                            "⚠️ Cold-start lenient acceptance: empty context accepted "
                            "because training results show 0 success."
                        )
                        _record_attempt_timing(
                            logger=logger,
                            timings=attempt_timings,
                            phase="validation",
                            attempt=attempt + 1,
                            max_attempts=max_validation_attempts,
                            attempt_start=attempt_start,
                            success=True,
                            error="lenient_acceptance_zero_success",
                        )
                        cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
                        return _finalize_result({
                            "success": True,
                            "interfaces": validation_result.interfaces,
                            "message_count": 0,
                            "validation_attempts": attempt + 1,
                            "context_lenient_acceptance": True,
                            **_context_hash_telemetry(warn_if_unchanged=True),
                        })

                    logger.warning(
                        "❌ Context output validation failed with "
                        f"{len(context_errors)} error(s):"
                    )
                    for error in context_errors:
                        logger.warning(f"  - {error}")
                    _record_attempt_timing(
                        logger=logger,
                        timings=attempt_timings,
                        phase="validation",
                        attempt=attempt + 1,
                        max_attempts=max_validation_attempts,
                        attempt_start=attempt_start,
                        success=False,
                        error="; ".join(context_errors[:3]) if context_errors else None,
                    )
                    if attempt + 1 >= max_validation_attempts:
                        logger.error(
                            f"Max validation attempts ({max_validation_attempts}) exceeded"
                        )
                        break
                    pending_prompt = _build_context_validation_feedback(
                        context_errors,
                        required_context_files=normalized_required_context_paths,
                        min_context_chars=context_min_chars,
                        error_classes=_classify_context_errors(context_errors),
                        retry_attempt=attempt + 2,
                        max_attempts=max_validation_attempts,
                    )
                    logger.info("📤 Sending context validation feedback to agent...")
                    continue

                _record_attempt_timing(
                    logger=logger,
                    timings=attempt_timings,
                    phase="validation",
                    attempt=attempt + 1,
                    max_attempts=max_validation_attempts,
                    attempt_start=attempt_start,
                    success=True,
                )
                logger.info(
                    f"✅ All {len(interface_signatures)} interfaces validated successfully"
                )
                cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
                return _finalize_result({
                    "success": True,
                    "interfaces": validation_result.interfaces,
                    "message_count": 0,
                    "validation_attempts": attempt + 1,
                    **_context_hash_telemetry(warn_if_unchanged=True),
                })

            logger.warning(
                f"❌ Validation failed with {len(validation_result.errors)} errors:"
            )
            for error in validation_result.errors:
                logger.warning(f"  - {error}")
            _record_attempt_timing(
                logger=logger,
                timings=attempt_timings,
                phase="validation",
                attempt=attempt + 1,
                max_attempts=max_validation_attempts,
                attempt_start=attempt_start,
                success=False,
                error="; ".join(validation_result.errors[:3]),
            )

            if attempt + 1 >= max_validation_attempts:
                logger.error(f"Max validation attempts ({max_validation_attempts}) exceeded")
                break

            pending_prompt = format_validation_feedback(validation_result)
            logger.info("📤 Sending validation feedback to agent...")

        cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
        return _finalize_result({
            "success": False,
            "error": f"Validation failed after {max_validation_attempts} attempts",
            "last_errors": (
                list(last_context_errors)
                if last_context_errors
                else (validation_result.errors if validation_result else [])
            ),
            "error_classes": (
                _classify_context_errors(last_context_errors)
                if last_context_errors
                else []
            ),
            "message_count": 0,
            **_context_hash_telemetry(warn_if_unchanged=False),
        })

    except Exception as exc:
        logger.error(f"Base-agent execution failed: {exc}", exc_info=True)
        cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
        return _finalize_result({
            "success": False,
            "error": str(exc),
            "last_errors": [],
            "message_count": 0,
            **_context_hash_telemetry(warn_if_unchanged=False),
        })
    finally:
        if hasattr(session, "close"):
            await session.close()
if __name__ == "__main__":
    import argparse
    import asyncio

    async def main():
        parser = argparse.ArgumentParser(description="Run base-agent to learn context")
        parser.add_argument("iter_dir", type=str, help="Iteration directory")
        parser.add_argument("--env", type=str, required=True, help="Environment name")
        parser.add_argument("--iteration", type=int, default=None, help="Iteration number")
        args = parser.parse_args()

        from evo_metaoptics.mce_env.registry import EnvironmentRegistry

        env = EnvironmentRegistry.get(args.env)
        task_instruction = env.get_task_instruction()
        interface_signatures = env.get_interface_signatures()

        iter_dir = Path(args.iter_dir).resolve()

        skill_path = learning_context_skill_host_path(iter_dir)
        if skill_path.exists():
            print(f"✓ Found SKILL.md at {skill_path}")
        else:
            print(f"⚠ SKILL.md not found at {skill_path}")
            print("  Continuing without a local skill file (native skills fallback mode).")

        result = await run_base_agent(
            iter_dir=iter_dir,
            task_instruction=task_instruction,
            interface_signatures=interface_signatures,
            iteration=args.iteration,
        )

        if result["success"]:
            print("\n✓ Base-agent completed successfully")
            print(f"  Validated interfaces: {list(result['interfaces'].keys())}")
        else:
            print(f"\n✗ Base-agent failed: {result['error']}")

    asyncio.run(main())
