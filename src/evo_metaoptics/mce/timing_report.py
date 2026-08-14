from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_TYPED_IR_NODE_NAMES = frozenset(
    {
        "typed_ir_graph",
        "extract_facts_ir",
        "canonicalize_ir",
        "resolve_materials",
        "compile_ir",
        "validate_compile",
        "optional_repair_ir",
        "parallel_fallback_candidates",
        "finalize",
    }
)


def build_timing_report(run_dir: str | Path, *, top_k: int = 10) -> dict[str, Any]:
    run_path = _resolve_run_dir(Path(run_dir))
    events_path = run_path / "events.jsonl"
    attempt_timings_path = run_path / "agent_attempt_timings.jsonl"
    detailed_dir = run_path / "detailed_llm_logs"

    events = _read_jsonl_objects(events_path) if events_path.exists() else []
    attempt_rows = _read_jsonl_objects(attempt_timings_path) if attempt_timings_path.exists() else []
    llm_calls = _collect_llm_calls(run_path=run_path, detailed_dir=detailed_dir, events=events)

    return {
        "run_dir": str(run_path),
        "files": {
            "events": events_path.exists(),
            "agent_attempt_timings": attempt_timings_path.exists(),
            "detailed_llm_logs": detailed_dir.exists(),
        },
        "event_type_counts": _build_event_type_counts(events),
        "invoke_done_counts": _build_invoke_done_counts(events),
        "node_duration_summary": _build_node_duration_summary(events),
        "typed_ir_node_duration_summary": _build_typed_ir_node_duration_summary(events),
        "attempt_duration_s": _build_attempt_duration_map(attempt_rows),
        "llm_call_counters": _build_llm_call_counters(llm_calls),
        "role_llm_summary": _build_role_llm_summary(llm_calls),
        "slowest_calls": _build_slowest_calls(llm_calls, top_k=top_k),
        "retry_hotspots": _build_retry_hotspots(llm_calls),
    }


def write_timing_report_check(
    run_dir: str | Path,
    *,
    top_k: int = 10,
    output_name: str = "timing_report_check.json",
) -> Path:
    run_path = _resolve_run_dir(Path(run_dir))
    report = build_timing_report(run_path, top_k=top_k)
    output_path = run_path / output_name
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _resolve_run_dir(path: Path) -> Path:
    if path.name == "detailed_llm_logs":
        return path.parent
    return path


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _read_llm_calls(detailed_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(detailed_dir.glob("*.json")):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(parsed, Mapping):
            continue
        role = parsed.get("role")
        call_id = parsed.get("call_id")
        if not isinstance(role, str) or not role.strip():
            continue
        if role == "timing":
            continue
        if not isinstance(call_id, str) or not call_id.strip():
            continue
        rows.append(dict(parsed))
    return rows


def _collect_llm_calls(
    *,
    run_path: Path,
    detailed_dir: Path,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate LLM call rows from all run artifacts with deterministic precedence."""
    detailed_rows = _read_llm_calls(detailed_dir) if detailed_dir.exists() else []
    root_rows = _read_llm_calls_from_run_root(run_path)
    event_rows = _read_llm_calls_from_events(events)
    return _merge_llm_call_rows(detailed_rows, root_rows, event_rows)


def _read_llm_calls_from_run_root(run_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_path.glob("*.json")):
        if path.name == "timing_report_check.json":
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(parsed, Mapping):
            continue
        role = parsed.get("role")
        call_id = parsed.get("call_id")
        if not isinstance(role, str) or not role.strip():
            continue
        if role == "timing":
            continue
        if not isinstance(call_id, str) or not call_id.strip():
            continue
        rows.append(dict(parsed))
    return rows


def _read_llm_calls_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type") or "").strip()
        if event_type not in {"llm_call_succeeded", "llm_call_failed"}:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        row_like = payload.get("call_info")
        if not isinstance(row_like, Mapping):
            row_like = payload
        role = row_like.get("role")
        call_id = row_like.get("call_id")
        if not isinstance(role, str) or not role.strip():
            continue
        if role == "timing":
            continue
        if not isinstance(call_id, str) or not call_id.strip():
            continue
        rows.append(dict(row_like))
    return rows


def _merge_llm_call_rows(*sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rows in sources:
        for row in rows:
            role = str(row.get("role") or "").strip()
            call_id = str(row.get("call_id") or "").strip()
            if not role or not call_id or role == "timing":
                continue
            existing = merged.get(call_id)
            if existing is None:
                merged[call_id] = dict(row)
                continue
            for key, value in row.items():
                if key not in existing or existing[key] in ("", None):
                    existing[key] = value
    return list(merged.values())


def _build_node_duration_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_node: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if str(event.get("event_type") or "") != "timing_recorded":
            continue
        node = event.get("node")
        if not isinstance(node, str) or not node.strip():
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        duration = _coerce_float(payload.get("duration_s"))
        if duration is None:
            duration = _coerce_float(payload.get("duration"))
        if duration is None:
            continue
        per_node[node].append(max(0.0, duration))

    summary: list[dict[str, Any]] = []
    for node, durations in per_node.items():
        if not durations:
            continue
        ordered = sorted(durations)
        count = len(ordered)
        total = sum(ordered)
        summary.append(
            {
                "node": node,
                "count": count,
                "p50_s": _nearest_rank_percentile(ordered, 50),
                "p90_s": _nearest_rank_percentile(ordered, 90),
                "avg_s": round(total / count, 6),
                "max_s": round(ordered[-1], 6),
            }
        )
    return sorted(summary, key=lambda row: (str(row["node"])))


def _build_typed_ir_node_duration_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in _build_node_duration_summary(events)
        if _is_typed_ir_node_name(str(row.get("node") or ""))
    ]


def _build_event_type_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        event_type = str(event.get("event_type") or "").strip()
        if event_type:
            counts[event_type] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _build_invoke_done_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        if str(event.get("event_type") or "") != "timing_recorded":
            continue
        node = event.get("node")
        payload = event.get("payload")
        if not isinstance(node, str) or not node.strip():
            continue
        if not isinstance(payload, Mapping):
            continue
        phase = str(payload.get("phase") or "").strip()
        if phase == "invoke_done":
            counts[node] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _build_attempt_duration_map(attempt_rows: list[dict[str, Any]]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for row in attempt_rows:
        agent_type = str(row.get("agent_type") or "").strip()
        phase = str(row.get("phase") or "").strip()
        attempt = _coerce_int(row.get("attempt"))
        duration_s = _coerce_float(row.get("duration_s"))
        if not agent_type or not phase or attempt is None or duration_s is None:
            continue
        sub_iteration = _coerce_int(row.get("sub_iteration"))
        if sub_iteration is None:
            key = f"{agent_type}.{phase}.attempt_{attempt}"
        else:
            key = f"{agent_type}.{phase}.sub_{sub_iteration}.attempt_{attempt}"
        durations[key] = round(max(0.0, duration_s), 6)
    return dict(sorted(durations.items(), key=lambda item: item[0]))


def _build_llm_call_counters(llm_calls: list[dict[str, Any]]) -> dict[str, Any]:
    per_role: dict[str, int] = defaultdict(int)
    retry_total = 0
    timeout_total = 0
    for row in llm_calls:
        role = str(row.get("role") or "").strip()
        if role:
            per_role[role] += 1
        retry_count = max(0, _coerce_int(row.get("retry_count")) or 0)
        timeout_count = max(0, _coerce_int(row.get("timeout_count")) or 0)
        retry_total += retry_count
        timeout_total += timeout_count

    by_role = dict(sorted(per_role.items(), key=lambda item: item[0]))
    return {
        "llm_call_total": len(llm_calls),
        "llm_calls_by_role": by_role,
        "patch_loop_calls": int(by_role.get("patch_agent", 0)),
        "retry_count_total": retry_total,
        "timeout_count_total": timeout_total,
    }


def _build_role_llm_summary(llm_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in llm_calls:
        role = str(row.get("role") or "").strip()
        if role:
            grouped[role].append(row)

    summary: list[dict[str, Any]] = []
    for role, rows in grouped.items():
        durations = [_llm_call_duration(row) for row in rows]
        prompt_total = 0
        response_total = 0
        total_tokens_total = 0
        for row in rows:
            prompt_tokens = _coerce_int(row.get("prompt_num_tokens")) or 0
            response_tokens = _coerce_int(row.get("response_num_tokens")) or 0
            total_tokens = _coerce_int(row.get("total_num_tokens"))
            if total_tokens is None:
                total_tokens = prompt_tokens + response_tokens
            prompt_total += prompt_tokens
            response_total += response_tokens
            total_tokens_total += total_tokens
        call_count = len(rows)
        summary.append(
            {
                "role": role,
                "call_count": call_count,
                "prompt_tokens_total": prompt_total,
                "response_tokens_total": response_total,
                "total_tokens_total": total_tokens_total,
                "avg_duration_s": round(sum(durations) / call_count, 6)
                if call_count > 0
                else 0.0,
                "max_duration_s": round(max(durations), 6) if durations else 0.0,
            }
        )
    return sorted(summary, key=lambda row: (str(row["role"])))


def _build_slowest_calls(llm_calls: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    capped = max(1, int(top_k))
    rows: list[dict[str, Any]] = []
    for row in llm_calls:
        rows.append(
            {
                "role": str(row.get("role") or ""),
                "call_id": str(row.get("call_id") or ""),
                "status": str(row.get("status") or "unknown"),
                "duration_s": _llm_call_duration(row),
                "retry_count": _coerce_int(row.get("retry_count")) or 0,
                "timeout_count": _coerce_int(row.get("timeout_count")) or 0,
            }
        )
    rows.sort(key=lambda item: float(item.get("duration_s") or 0.0), reverse=True)
    return rows[:capped]


def _build_retry_hotspots(llm_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in llm_calls:
        role = str(row.get("role") or "").strip()
        if role:
            grouped[role].append(row)

    hotspots: list[dict[str, Any]] = []
    for role, rows in grouped.items():
        retry_values = [max(0, _coerce_int(row.get("retry_count")) or 0) for row in rows]
        timeout_values = [max(0, _coerce_int(row.get("timeout_count")) or 0) for row in rows]
        retried_calls = sum(1 for value in retry_values if value > 0)
        timeout_calls = sum(1 for value in timeout_values if value > 0)
        call_count = len(rows)
        hotspots.append(
            {
                "role": role,
                "call_count": call_count,
                "retried_calls": retried_calls,
                "timeout_calls": timeout_calls,
                "avg_retry_count": round(sum(retry_values) / call_count, 6)
                if call_count > 0
                else 0.0,
                "max_retry_count": max(retry_values) if retry_values else 0,
                "max_timeout_count": max(timeout_values) if timeout_values else 0,
            }
        )
    hotspots.sort(
        key=lambda row: (
            -int(row["retried_calls"]),
            -int(row["timeout_calls"]),
            str(row["role"]),
        )
    )
    return hotspots


def _is_typed_ir_node_name(node_name: str) -> bool:
    name = str(node_name or "").strip()
    if not name:
        return False
    if name in _TYPED_IR_NODE_NAMES:
        return True
    return name.startswith("typed_ir_")


def _llm_call_duration(row: Mapping[str, Any]) -> float:
    for key in ("call_time", "total_time", "response_time"):
        value = _coerce_float(row.get(key))
        if value is not None:
            return round(max(0.0, value), 6)
    return 0.0


def _nearest_rank_percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    p = max(0, min(100, int(percentile)))
    rank = max(1, math.ceil((p / 100.0) * len(values)))
    return round(float(values[rank - 1]), 6)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return None
