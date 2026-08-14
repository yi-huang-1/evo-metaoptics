"""
Workspace management utilities for MCE.

Handles creation and setup of iteration workspaces.
"""

import os
import shutil
import json
import logging
import importlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Any, List, Mapping

from evo_metaoptics.mce.skills import (
    learning_context_skill_host_path,
    learning_context_skill_host_dir,
    learning_context_skills_host_dir,
)
from evo_metaoptics.mce_env.base import InterfaceSignature

try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

load_dotenv(override=True)

logger = logging.getLogger(__name__)
_TOP_K_ERROR_REASONS = 3
_TOP_K_ERROR_FINGERPRINTS = 5
_TOP_K_VALIDATOR_SIGNATURES = 5
_TOP_K_SKILL_HASHES = 3
_REMOVED_CONTEXT_ARTIFACT_FILENAMES = {"execution_plan.json"}
_CANONICAL_SUMMARY_METRIC_NAMES = {
    "success_exec",
    "success_goal",
    "best_margin",
    "criteria_pass_fraction",
    "criteria_violation_norm",
    "attempt_count",
    "round_count",
    "total_attempt_count",
}


def archived_meta_skill_dir_name(iteration: int) -> str:
    """Return spec-compliant archived skill directory name for an iteration."""
    return f"learning-context-iter{iteration}"


def archived_meta_skill_host_path(meta_agent_dir: Path, iteration: int) -> Path:
    """Return host path for archived skill file of an iteration."""
    return (
        Path(meta_agent_dir)
        / "skills"
        / archived_meta_skill_dir_name(iteration)
        / "SKILL.md"
    )


def archived_meta_skill_host_dir(meta_agent_dir: Path, iteration: int) -> Path:
    return archived_meta_skill_host_path(meta_agent_dir, iteration).parent


def _rewrite_frontmatter_name(skill_md: str, expected_name: str) -> str:
    """Rewrite YAML frontmatter `name:` field to expected archived skill name."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", skill_md, re.DOTALL)
    if not match:
        return skill_md

    frontmatter = match.group(1)
    if re.search(r"(?m)^name:\s*.+$", frontmatter):
        updated_frontmatter = re.sub(
            r"(?m)^name:\s*.+$",
            f"name: {expected_name}",
            frontmatter,
            count=1,
        )
    else:
        updated_frontmatter = f"name: {expected_name}\n{frontmatter}".strip()

    remainder = skill_md[match.end() :]
    return f"---\n{updated_frontmatter}\n---\n\n{remainder.lstrip()}"


def compute_avg_metrics(successful_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute average metrics from successful evaluation results.
    
    Args:
        successful_results: List of result dicts with "evaluation.metrics" field
    
    Returns:
        Dictionary of averaged metrics
    """
    # Collect all metric values
    all_metrics_values: dict[str, list[Any]] = {}
    for r in successful_results:
        metrics = r.get("evaluation", {}).get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if metric_name not in all_metrics_values:
                all_metrics_values[metric_name] = []
            all_metrics_values[metric_name].append(metric_value)
    
    # Average using only numeric values; skip None/non-numeric entries.
    avg_metrics = {}
    for metric_name, raw_values in all_metrics_values.items():
        numeric_values: list[float] = []
        for value in raw_values:
            if isinstance(value, bool):
                numeric_values.append(float(value))
            elif isinstance(value, (int, float)):
                numeric_values.append(float(value))
        if numeric_values:
            avg_metrics[metric_name] = sum(numeric_values) / len(numeric_values)

    return avg_metrics


# Helper function to ignore __pycache__
def ignore_pycache(directory, contents):
    return ['__pycache__'] if '__pycache__' in contents else []


def prepare_meta_agent_reference(
    workspace_base: Path,
    train_data_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Set up meta-agent reference data at the beginning of MCE loop.
    
    Creates meta_agent/ directory and bootstraps deterministic evaluation summary store.
    Hard-cut policy: do not copy raw training dataset into meta_agent/.
    
    Args:
        workspace_base: Base workspace directory (e.g., workspace/finer)
        train_data_path: Path to training data file
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    workspace_base = Path(workspace_base)
    meta_agent_dir = workspace_base / "meta_agent"
    meta_agent_dir.mkdir(exist_ok=True)
    
    # Create skills directory
    skills_dir = meta_agent_dir / "skills"
    skills_dir.mkdir(exist_ok=True)

    # Bootstrap evaluations summary store so iter1 meta-agent reads are deterministic.
    evaluations_file = meta_agent_dir / "evaluations.json"
    if not evaluations_file.exists():
        evaluations_file.write_text("{}", encoding="utf-8")

    train_data_src = Path(train_data_path)
    if not train_data_src.exists():
        logger.warning(f"⚠️  Training data not found at {train_data_path}")
    logger.info("✅ Data-scope hard-cut active: skipped raw meta-agent dataset copy")


def aggregate_iteration_results(
    workspace_base: Path,
    iteration: int,
    sub_iterations: list,
    val_primary_metric: float,
    val_metrics: dict,
    val_total: int,
    cumulative_rollouts: int,
    num_sub_iters: int,
    last_sub_folder_name: str,
    environment,
    skill_guard: Dict[str, Any] | None = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Aggregate iteration results to meta_agent/evaluations.json and copy skill.
    
    Aggregates metrics for recording in evaluations.json. The primary metric name
    is used for logging and metric naming, but selection uses criteria-first
    validation ranking (see _extract_iteration_selection_rank()).
    
    Args:
        workspace_base: Base workspace directory
        iteration: Iteration number
        sub_iterations: List of sub-iteration results with metrics
        val_primary_metric: Final validation primary metric value
        val_metrics: All validation metrics
        val_total: Total validation samples
        cumulative_rollouts: Total rollouts across all sub-iterations
        num_sub_iters: Number of sub-iterations
        last_sub_folder_name: Name of the last sub-iteration folder (e.g., "iter1_sub3")
        environment: TaskEnvironment instance
        skill_guard: Optional skill-guard telemetry payload from meta-agent
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    workspace_base = Path(workspace_base)
    meta_agent_dir = workspace_base / "meta_agent"
    meta_agent_dir.mkdir(exist_ok=True)
    
    # Compute aggregated train metrics
    # Weight by batch size to get proper average
    total_weighted = sum(sr['batch_train_primary_metric'] * sr['batch_size'] for sr in sub_iterations)
    total_samples = sum(sr['batch_size'] for sr in sub_iterations)
    train_primary_metric = total_weighted / total_samples if total_samples > 0 else 0.0
    
    # Aggregate all train metrics (weighted average across batches)
    train_metrics = {}
    if sub_iterations and 'batch_train_metrics' in sub_iterations[0]:
        # Get all metric names from first sub-iteration
        all_metric_names = sub_iterations[0]['batch_train_metrics'].keys()
        for metric_name in all_metric_names:
            weighted_sum = sum(
                sr['batch_train_metrics'].get(metric_name, 0.0) * sr['batch_size']
                for sr in sub_iterations
            )
            train_metrics[metric_name] = weighted_sum / total_samples if total_samples > 0 else 0.0
    
    # Get primary metric name from environment
    primary_metric_name = environment.get_primary_metric_name()
    train_metrics = _canonicalize_summary_metrics(train_metrics)
    # Build compact training failure summary from per-sub-iteration train rollouts.
    train_rollouts = _collect_train_rollouts(sub_iterations)
    failure_summary = _compute_train_failure_summary(
        train_rollouts,
        train_metrics=train_metrics,
    )
    val_metrics = _canonicalize_summary_metrics(
        dict(val_metrics) if isinstance(val_metrics, Mapping) else {}
    )
    normalized_skill_guard = _normalize_skill_guard_payload(skill_guard)
    
    # Load existing evaluations or create new
    evaluations_file = meta_agent_dir / "evaluations.json"
    if evaluations_file.exists():
        with open(evaluations_file, 'r') as f:
            evaluations = json.load(f)
    else:
        evaluations = {}
    
    val_primary_metric_value = val_metrics.get(primary_metric_name, val_primary_metric)
    iteration_summary = {
        "primary_metric_name": primary_metric_name,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "total_rollouts": cumulative_rollouts,
        "skill_guard_triggered": normalized_skill_guard["skill_guard_triggered"],
        "degeneracy_type": normalized_skill_guard["degeneracy_type"],
        "num_sub_iters": num_sub_iters,
        "last_sub_folder": last_sub_folder_name,
        "error_reason_counts_top3": failure_summary["error_reason_counts_top3"],
        "error_fingerprint_counts_top5": failure_summary["error_fingerprint_counts_top5"],
        "validator_signature_counts_top5": failure_summary["validator_signature_counts_top5"],
        "precheck_ok_rate": failure_summary["precheck_ok_rate"],
        "avg_precheck_attempts": failure_summary["avg_precheck_attempts"],
        "near_miss_rate": failure_summary["near_miss_rate"],
        "criteria_pass_fraction_avg": failure_summary["criteria_pass_fraction_avg"],
        "criteria_violation_norm_avg": failure_summary["criteria_violation_norm_avg"],
        "best_margin_avg": failure_summary["best_margin_avg"],
        "optimizer_limited_rate": failure_summary["optimizer_limited_rate"],
        "spec_limited_rate": failure_summary["spec_limited_rate"],
        "skill_hash_counts_top3": failure_summary["skill_hash_counts_top3"],
    }
    evaluations[f"iter{iteration}"] = iteration_summary
    
    # Save summary evaluations
    with open(evaluations_file, 'w') as f:
        json.dump(evaluations, f, indent=2)
    
    logger.info(
        f"✅ Aggregated iter{iteration} results to meta_agent/evaluations.json: "
        f"train_metrics.{primary_metric_name}={float(train_metrics.get(primary_metric_name, train_primary_metric)):.2%}, "
        f"val_metrics.{primary_metric_name}={float(val_primary_metric_value):.2%}, "
        f"criteria_pass_fraction_avg={failure_summary.get('criteria_pass_fraction_avg', 0.0):.4f}, "
        f"criteria_violation_norm_avg={failure_summary.get('criteria_violation_norm_avg', 0.0):.4f}, "
        f"best_margin_avg={failure_summary.get('best_margin_avg', 0.0):.6f}"
    )
    
    # Copy skill from last sub-iteration to meta_agent/skills/ (spec-compliant archive naming)
    if iteration >= 1:
        last_sub_folder = workspace_base / last_sub_folder_name
        source_skill_dir = learning_context_skill_host_dir(last_sub_folder)
        source_skill = learning_context_skill_host_path(last_sub_folder)
        target_skill_dir = archived_meta_skill_host_dir(meta_agent_dir, iteration)
        target_skill = target_skill_dir / "SKILL.md"
        target_skill_dir.parent.mkdir(parents=True, exist_ok=True)

        if source_skill.exists() and source_skill_dir.exists():
            if target_skill_dir.exists():
                shutil.rmtree(target_skill_dir)
            shutil.copytree(source_skill_dir, target_skill_dir)
            skill_content = target_skill.read_text(encoding="utf-8")
            target_name = archived_meta_skill_dir_name(iteration)
            rewritten = _rewrite_frontmatter_name(skill_content, expected_name=target_name)
            target_skill.write_text(rewritten, encoding="utf-8")
            logger.info(f"✅ Copied skill package to meta_agent/skills/{target_name}/")
        else:
            logger.warning(f"⚠️  No skill found at {source_skill}")

def _to_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _normalize_skill_guard_payload(skill_guard: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return stable skill-guard fields for iteration summaries."""
    if not isinstance(skill_guard, dict):
        return {
            "skill_guard_triggered": False,
            "guard_action": "accept_generated",
            "degeneracy_type": "none",
        }

    triggered_raw = skill_guard.get("skill_guard_triggered", False)
    if isinstance(triggered_raw, bool):
        triggered = triggered_raw
    else:
        triggered = bool(triggered_raw)

    action_raw = skill_guard.get("guard_action", "accept_generated")
    action = action_raw.strip() if isinstance(action_raw, str) else "accept_generated"
    if action != "accept_generated":
        action = "accept_generated"

    degeneracy_type_raw = skill_guard.get("degeneracy_type", "none")
    degeneracy_type = (
        degeneracy_type_raw.strip()
        if isinstance(degeneracy_type_raw, str) and degeneracy_type_raw.strip()
        else "none"
    )

    return {
        "skill_guard_triggered": triggered,
        "degeneracy_type": degeneracy_type,
    }


def _canonicalize_summary_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key in sorted(_CANONICAL_SUMMARY_METRIC_NAMES):
        if key not in metrics:
            continue
        normalized[key] = float(_to_float(metrics.get(key), default=0.0))
    return normalized


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _collect_train_rollouts(sub_iterations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rollouts: List[Dict[str, Any]] = []
    for sub_iter in sub_iterations:
        folder = sub_iter.get("folder")
        if not isinstance(folder, str) or not folder.strip():
            continue
        train_file = Path(folder) / "data" / "train.json"
        if not train_file.exists():
            continue
        try:
            payload = json.loads(train_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        detailed_results = payload.get("detailed_results")
        if not isinstance(detailed_results, list):
            continue
        for item in detailed_results:
            if isinstance(item, dict):
                rollouts.append(item)
    return rollouts


def _extract_validator_signatures(rollout: Dict[str, Any]) -> List[str]:
    explicit = rollout.get("validator_signatures")
    if isinstance(explicit, list):
        deduped = {
            str(item).strip()
            for item in explicit
            if isinstance(item, str) and item.strip()
        }
        if deduped:
            return sorted(deduped)

    signatures: set[str] = set()
    validation_errors = rollout.get("validation_errors")
    if isinstance(validation_errors, list):
        for item in validation_errors:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", "") or "").strip()
            path = str(item.get("path", "") or "").strip()
            if code and path:
                signatures.add(f"{code}:{path}")
    return sorted(signatures)


def _criteria_pass_fraction(rollout: Dict[str, Any]) -> float:
    explicit = rollout.get("criteria_pass_fraction")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        return _clip01(float(explicit))

    criteria_total = int(rollout.get("criteria_total", 0) or 0)
    criteria_passed = rollout.get("criteria_passed")
    if isinstance(criteria_passed, (int, float)) and not isinstance(criteria_passed, bool):
        return _clip01(float(criteria_passed) / float(max(criteria_total, 1)))

    if bool(rollout.get("success_goal")):
        return 1.0
    return 0.0


def _criteria_violation_norm(rollout: Dict[str, Any]) -> float:
    explicit = rollout.get("criteria_violation_norm")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        return _clip01(float(explicit))

    criteria_total = int(rollout.get("criteria_total", 0) or 0)
    criteria_violation_sum = _to_float(rollout.get("criteria_violation_sum"), default=0.0)
    if criteria_total > 0:
        return _clip01(criteria_violation_sum / float(criteria_total))

    if bool(rollout.get("success_goal")):
        return 0.0
    return 1.0


def _near_miss_flag(rollout: Dict[str, Any]) -> float:
    explicit = rollout.get("near_miss")
    if isinstance(explicit, bool):
        return 1.0 if explicit else 0.0

    success_exec = bool(rollout.get("success_exec"))
    success_goal = bool(rollout.get("success_goal"))
    criteria_total = int(rollout.get("criteria_total", 0) or 0)
    if not success_exec or success_goal or criteria_total <= 0:
        return 0.0
    return 1.0 if _criteria_violation_norm(rollout) <= 0.15 else 0.0


def _opt_diagnosis_classification(rollout: Dict[str, Any]) -> str:
    payload = rollout.get("opt_diagnosis")
    if isinstance(payload, dict):
        value = payload.get("classification")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if not bool(rollout.get("success_exec")):
        return "exec_failed"
    if bool(rollout.get("success_goal")):
        return "goal_met"
    return "spec_limited"


def _sorted_topk(counter: Counter[str], *, top_k: int, field_name: str) -> List[Dict[str, Any]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [{field_name: key, "count": int(count)} for key, count in items]


def _sorted_topk_with_rate(
    counter: Counter[str],
    *,
    top_k: int,
    field_name: str,
    total_count: float,
) -> List[Dict[str, Any]]:
    if total_count <= 0.0:
        return []
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    rows: list[dict[str, Any]] = []
    for key, count in items:
        rows.append(
            {
                field_name: key,
                "count": int(count),
                "rate": float(count) / float(total_count),
            }
        )
    return rows


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compute_train_failure_summary(
    rollouts: List[Dict[str, Any]],
    *,
    train_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if not rollouts:
        return {
            "error_reason_counts_top3": [],
            "error_fingerprint_counts_top5": [],
            "validator_signature_counts_top5": [],
            "precheck_ok_rate": _clip01(_to_float(train_metrics.get("precheck_ok"), default=0.0)),
            "avg_precheck_attempts": _to_float(train_metrics.get("precheck_attempts"), default=0.0),
            "near_miss_rate": _clip01(_to_float(train_metrics.get("near_miss"), default=0.0)),
            "criteria_pass_fraction_avg": _clip01(
                _to_float(train_metrics.get("criteria_pass_fraction"), default=0.0)
            ),
            "criteria_violation_norm_avg": _clip01(
                _to_float(train_metrics.get("criteria_violation_norm"), default=0.0)
            ),
            "best_margin_avg": _to_float(train_metrics.get("best_margin"), default=0.0),
            "optimizer_limited_rate": _clip01(
                _to_float(train_metrics.get("optimizer_limited"), default=0.0)
            ),
            "spec_limited_rate": _clip01(
                _to_float(train_metrics.get("spec_limited"), default=0.0)
            ),
            "skill_hash_counts_top3": [],
        }

    total = float(len(rollouts))
    reason_counter: Counter[str] = Counter()
    fingerprint_counter: Counter[str] = Counter()
    signature_counter: Counter[str] = Counter()
    skill_hash_counter: Counter[str] = Counter()

    goal_sum = 0.0
    exec_sum = 0.0
    precheck_ok_sum = 0.0
    precheck_attempts_sum = 0.0
    near_miss_sum = 0.0
    criteria_pass_fraction_sum = 0.0
    criteria_violation_norm_sum = 0.0
    best_margin_sum = 0.0
    best_margin_count = 0
    optimizer_limited_sum = 0.0
    spec_limited_sum = 0.0
    for rollout in rollouts:
        goal_sum += 1.0 if bool(rollout.get("success_goal")) else 0.0
        exec_sum += 1.0 if bool(rollout.get("success_exec")) else 0.0
        precheck_ok_sum += 1.0 if bool(rollout.get("precheck_ok")) else 0.0
        precheck_attempts_sum += max(0.0, _to_float(rollout.get("precheck_attempts"), default=0.0))
        near_miss_sum += _near_miss_flag(rollout)
        criteria_pass_fraction_sum += _criteria_pass_fraction(rollout)
        criteria_violation_norm_sum += _criteria_violation_norm(rollout)
        best_margin = rollout.get("best_margin")
        if isinstance(best_margin, (int, float)) and not isinstance(best_margin, bool):
            best_margin_sum += float(best_margin)
            best_margin_count += 1

        classification = _opt_diagnosis_classification(rollout)
        if classification == "optimizer_limited":
            optimizer_limited_sum += 1.0
        if classification == "spec_limited":
            spec_limited_sum += 1.0

        raw_skill_hash = rollout.get("skill_hash")
        if isinstance(raw_skill_hash, str) and raw_skill_hash.strip():
            skill_hash_counter[raw_skill_hash.strip()] += 1

        reason = rollout.get("error_reason")
        if isinstance(reason, str) and reason.strip():
            reason_counter[reason.strip()] += 1

        raw_error_message = rollout.get("code_error_message")
        if isinstance(raw_error_message, str) and raw_error_message.strip():
            fingerprint = _normalize_error_fingerprint(raw_error_message)
            if fingerprint:
                fingerprint_counter[fingerprint] += 1

        for signature in _extract_validator_signatures(rollout):
            signature_counter[signature] += 1

    goal_rate = goal_sum / total
    exec_rate = exec_sum / total
    precheck_ok_rate = precheck_ok_sum / total
    avg_precheck_attempts = precheck_attempts_sum / total
    near_miss_rate = near_miss_sum / total
    criteria_pass_fraction_avg = criteria_pass_fraction_sum / total
    criteria_violation_norm_avg = criteria_violation_norm_sum / total
    best_margin_avg = best_margin_sum / float(best_margin_count) if best_margin_count > 0 else 0.0
    optimizer_limited_rate = optimizer_limited_sum / total
    spec_limited_rate = spec_limited_sum / total
    return {
        "error_reason_counts_top3": _sorted_topk(
            reason_counter,
            top_k=_TOP_K_ERROR_REASONS,
            field_name="error_reason",
        ),
        "error_fingerprint_counts_top5": _sorted_topk(
            fingerprint_counter,
            top_k=_TOP_K_ERROR_FINGERPRINTS,
            field_name="error_fingerprint",
        ),
        "validator_signature_counts_top5": _sorted_topk(
            signature_counter,
            top_k=_TOP_K_VALIDATOR_SIGNATURES,
            field_name="signature",
        ),
        "precheck_ok_rate": precheck_ok_rate,
        "avg_precheck_attempts": avg_precheck_attempts,
        "near_miss_rate": near_miss_rate,
        "criteria_pass_fraction_avg": criteria_pass_fraction_avg,
        "criteria_violation_norm_avg": criteria_violation_norm_avg,
        "best_margin_avg": best_margin_avg,
        "optimizer_limited_rate": optimizer_limited_rate,
        "spec_limited_rate": spec_limited_rate,
        "skill_hash_counts_top3": _sorted_topk_with_rate(
            skill_hash_counter,
            top_k=_TOP_K_SKILL_HASHES,
            field_name="skill_hash",
            total_count=total,
        ),
    }


def _normalize_error_fingerprint(message: str) -> str:
    text = str(message).strip().lower()
    if not text:
        return ""
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"[A-Za-z]:\\[^\s:]+|/[^\s:]+", "<path>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:160]




def get_sub_iteration_folder_name(iteration: int, sub_iter: Optional[int] = None) -> str:
    """
    Get folder name for iteration or sub-iteration.
    
    Examples:
        iter=0, sub_iter=None  → "iter0"
        iter=1, sub_iter=None  → "iter1"
        iter=1, sub_iter=0     → "iter1_sub0" 
        iter=1, sub_iter=3     → "iter1_sub3"
    """
    if sub_iter is None:
        return f"iter{iteration}"
    else:
        return f"iter{iteration}_sub{sub_iter}"


def create_iteration_workspace(
    workspace_base: Path,
    iteration: int,
    sub_iter: Optional[int] = None,
) -> Path:
    """
    Create workspace folder for a specific iteration or sub-iteration.

    Args:
        workspace_base: Base workspace directory (e.g., workspace/finer)
        iteration: Iteration number (0+)
        sub_iter: Optional sub-iteration number. When omitted, create ``iterN``;
            otherwise create ``iterN_subM``.
    
    Returns:
        Path to created iteration directory
    
    Raises:
        FileExistsError: If iteration directory already exists
    """
    workspace_base = Path(workspace_base)
    folder_name = get_sub_iteration_folder_name(iteration, sub_iter)
    iter_folder = workspace_base / folder_name
    
    logger.debug(f"Creating iteration workspace: {iter_folder}")

    # Check if iteration folder already exists
    if iter_folder.exists():
        error_msg = (
            f"Iteration folder already exists at {iter_folder}. "
            "Please remove it or use a different iteration number."
        )
        logger.error(error_msg)
        raise FileExistsError(error_msg)

    # Create iteration folder
    iter_folder.mkdir(parents=True, exist_ok=True)
    
    # Create .agents/skills directory for iter >= 1 (when meta-agent is used)
    if iteration >= 1:
        skills_dir = learning_context_skill_host_path(iter_folder).parent
        skills_dir.mkdir(parents=True, exist_ok=True)

    return iter_folder


def setup_base_agent_workspace(
    workspace_base: Path,
    iter_folder: Path,
    iteration: int,
    env,
    logger: Optional[logging.Logger] = None,
    source_folder: Optional[Path] = None,
    interface_signatures: Optional[List[InterfaceSignature]] = None,
) -> None:
    """
    Set up base agent workspace for the current iteration or sub-iteration.
    
    For iteration >= 1:
    1. Copy context/ and interfaces/ from source folder (best previous iteration or previous sub-iteration)
    
    Note: Training data (train.json) is NOT copied here. It will be generated by evaluating
    the current batch BEFORE running base-agent.
    
    Args:
        workspace_base: Base workspace directory (e.g., workspace/finer)
        iter_folder: Current iteration folder path
        iteration: Current iteration number
        env: Environment instance
        logger: Optional logger instance
        source_folder: Explicit source folder to copy from (for sub-iterations). If None, finds best previous iteration.
        interface_signatures: Optional required signatures for bootstrap stub seeding.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if iteration < 1:
        return
    
    workspace_base = Path(workspace_base)
    
    # Determine source folder
    if source_folder is not None:
        source_iter_dir = Path(source_folder)
        source_name = source_iter_dir.name
        logger.info(f"Copying from {source_name}")
    else:
        if iteration == 1:
            source_iter_dir = None
            source_name = "bootstrap"
            logger.info("Bootstrapping empty context/interfaces for the first learning iteration")
        else:
            best_iter_info = _find_best_iteration(workspace_base, iteration, env)
            if best_iter_info:
                source_iter_num = best_iter_info['iteration']
                best_metric_value = best_iter_info['primary_metric_value']
                last_sub_folder = best_iter_info.get('last_sub_folder', f'iter{source_iter_num}')
                source_iter_dir = workspace_base / last_sub_folder
                source_name = last_sub_folder
                logger.info(f"Copying from {source_name} (iter{source_iter_num} val_primary_metric: {best_metric_value:.2%})")
            else:
                raise ValueError(f"No previous iterations found for iteration {iteration}")
    
    # Copy context/ directory
    source_context = source_iter_dir / "context" if source_iter_dir is not None else None
    target_context = iter_folder / "context"
    if source_context is not None and source_context.exists() and not target_context.exists():
        shutil.copytree(source_context, target_context, ignore=ignore_pycache)
        logger.info(f"✅ Copied context/ from {source_name}")
    elif not target_context.exists():
        target_context.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Created empty context/ folder")
    removed_context_artifacts = _remove_removed_context_artifacts(target_context)
    if removed_context_artifacts:
        logger.info(
            "🧹 Removed deprecated context artifacts: "
            + ", ".join(f"context/{rel}" for rel in removed_context_artifacts)
        )
    _overlay_canonical_strategy_summary(
        workspace_base=workspace_base,
        target_context=target_context,
        logger=logger,
    )
    
    # Copy interfaces/ directory (new interface-based system)
    source_interfaces = source_iter_dir / "interfaces" if source_iter_dir is not None else None
    target_interfaces = iter_folder / "interfaces"
    if source_interfaces is not None and source_interfaces.exists() and not target_interfaces.exists():
        shutil.copytree(source_interfaces, target_interfaces, ignore=ignore_pycache)
        logger.info(f"✅ Copied interfaces/ from {source_name}")
    elif not target_interfaces.exists():
        target_interfaces.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Created empty interfaces/ folder")
        _seed_bootstrap_interfaces(
            target_interfaces=target_interfaces,
            interface_signatures=interface_signatures or [],
            logger=logger,
        )
    
    # Create data directory (training data will be written after batch evaluation)
    data_dir = iter_folder / "data"
    data_dir.mkdir(exist_ok=True)


def _overlay_canonical_strategy_summary(
    *,
    workspace_base: Path,
    target_context: Path,
    logger: logging.Logger,
) -> None:
    canonical_summary = Path(workspace_base) / "meta_agent" / "strategy_summary.md"
    if not canonical_summary.is_file():
        return

    target_context.mkdir(parents=True, exist_ok=True)
    target_summary = target_context / "strategy_summary.md"
    shutil.copyfile(canonical_summary, target_summary)
    logger.info("✅ Overlayed canonical strategy_summary.md into context/")


def _seed_bootstrap_interfaces(
    *,
    target_interfaces: Path,
    interface_signatures: List[InterfaceSignature],
    logger: logging.Logger,
) -> None:
    """Seed deterministic signature-compliant interface stubs for bootstrap iterations."""
    if not interface_signatures:
        return

    target_interfaces.mkdir(parents=True, exist_ok=True)
    exported_names: List[str] = []
    for signature in interface_signatures:
        exported_names.append(signature.name)
        module_path = target_interfaces / f"{signature.name}.py"
        if module_path.exists():
            continue
        module_path.write_text(
            _render_bootstrap_interface_module(signature),
            encoding="utf-8",
        )

    init_path = target_interfaces / "__init__.py"
    if not init_path.exists():
        imports = [f"from .{name} import {name}" for name in exported_names]
        export_line = "__all__ = [" + ", ".join(repr(name) for name in exported_names) + "]"
        init_path.write_text("\n".join([*imports, "", export_line, ""]), encoding="utf-8")

    logger.info(
        "✅ Seeded bootstrap interface stubs for required signatures: "
        + ", ".join(exported_names)
    )


def _render_bootstrap_interface_module(signature: InterfaceSignature) -> str:
    """Render a minimal signature-compliant bootstrap interface module."""
    params = ", ".join(f"{name}: {typ}" for name, typ, _ in signature.inputs)
    return_type = signature.output[0]
    return_stmt = 'return "{}"' if return_type == "str" else "return None"
    lines = [
        f'"""Bootstrap stub for `{signature.name}` generated by workspace setup."""',
        "",
        f"def {signature.name}({params}) -> {return_type}:",
        "    \"\"\"Return a minimal placeholder value for first-pass bootstrap.\"\"\"",
        f"    {return_stmt}",
        "",
    ]
    return "\n".join(lines)


def copy_skills_to_sub_iteration(
    source_folder: Path,
    target_folder: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Copy .agents/skills from source to target folder.
    
    Args:
        source_folder: Source iteration folder (with skills)
        target_folder: Target sub-iteration folder
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    source_skills = learning_context_skills_host_dir(source_folder)
    target_skills = learning_context_skills_host_dir(target_folder)

    # Ensure parent directory exists
    target_skills.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy with dirs_exist_ok=True to overwrite existing directory
    shutil.copytree(source_skills, target_skills, ignore=ignore_pycache, dirs_exist_ok=True)
    logger.info(f"✅ Copied skills from {source_folder.name}")



def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _extract_iteration_selection_rank(
    iter_data: Mapping[str, Any],
    iteration_index: int,
) -> tuple[float, float, float, float, float, float, int]:
    """
    Build canonical iteration rank tuple.

    Criteria-first selector contract:
      1) success and criteria diagnostics,
      2) execution stability,
      3) cost (lower is better),
      4) recency (latest iteration).
    """
    val_metrics = iter_data.get("val_metrics")
    val_metrics_map = val_metrics if isinstance(val_metrics, Mapping) else {}

    criteria_pass_fraction = _clip01(
        _to_float(
            val_metrics_map.get("criteria_pass_fraction"),
            default=_to_float(iter_data.get("criteria_pass_fraction_avg"), default=0.0),
        )
    )
    criteria_violation_norm = _clip01(
        _to_float(
            val_metrics_map.get("criteria_violation_norm"),
            default=_to_float(iter_data.get("criteria_violation_norm_avg"), default=1.0),
        )
    )
    criteria_best_margin = _to_float(
        val_metrics_map.get("best_margin"),
        default=float("-inf"),
    )

    val_success_goal = _clip01(
        _to_float(val_metrics_map.get("success_goal"), default=0.0)
    )
    val_success_exec = _clip01(
        _to_float(val_metrics_map.get("success_exec"), default=0.0)
    )

    raw_cost = val_metrics_map.get("solver_calls_total")
    if raw_cost is None:
        raw_cost = val_metrics_map.get("opt_steps_total")
    cost_value = _finite_float(raw_cost)
    cost_rank = -cost_value if cost_value is not None else float("-inf")

    return (
        float(val_success_goal),
        float(criteria_pass_fraction),
        float(-criteria_violation_norm),
        float(criteria_best_margin),
        float(val_success_exec),
        float(cost_rank),
        int(iteration_index),
    )


def _find_best_iteration(workspace_base: Path, current_iteration: int, env) -> Optional[Dict[str, Any]]:
    """
    Select best iteration using criteria-first validation ranking.

    Ranking tuple (lexicographic order):
    1. success_goal (higher better) - whether criteria fully satisfied
    2. criteria_pass_fraction (higher better) - fraction of criteria passing
    3. -criteria_violation_norm (higher better, i.e., lower norm) - aggregate violation
    4. best_margin (higher better) - minimum margin across criteria
    5. success_exec (higher better) - execution success flag
    6. -cost (higher better, i.e., lower cost) - computational cost
    7. recency (higher better) - iteration number (prefer later)

    This repo uses criteria-first selection unlike reference MCE which uses
    validation primary metric only.
    
    Args:
        workspace_base: Base workspace directory
        current_iteration: Upper bound (exclusive) of iterations to search (i.e. searches iter0..iter{current_iteration-1})
        env: Environment instance
    
    Returns:
        Dictionary with 'iteration', 'primary_metric_value', and 'last_sub_folder' keys, or None if no iterations found
    """
    workspace_base = Path(workspace_base)
    evaluations_file = workspace_base / "meta_agent" / "evaluations.json"
    
    if current_iteration <= 0:
        return None

    if not evaluations_file.exists():
        logger.warning(f"No evaluations file found at {evaluations_file}")
        return None
    
    with open(evaluations_file, 'r') as f:
        evaluations = json.load(f)
    
    best_iter = None
    best_last_sub_folder = None
    best_rank: tuple[float, float, float, float, float, float, int] = (
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        -1,
    )
    best_primary_metric_value = 0.0

    for i in range(0, current_iteration):
        iter_key = f"iter{i}"
        if iter_key not in evaluations:
            logger.warning(f"No evaluation found for {iter_key}")
            continue
        
        iter_data = evaluations[iter_key]
        
        candidate_rank = _extract_iteration_selection_rank(iter_data, i)
        if candidate_rank > best_rank:
            best_rank = candidate_rank
            best_iter = i
            val_metrics = iter_data.get("val_metrics")
            val_metrics_map = val_metrics if isinstance(val_metrics, Mapping) else {}
            primary_metric_name = "criteria_pass_fraction"
            if env is not None and hasattr(env, "get_primary_metric_name"):
                primary_metric_name = str(env.get_primary_metric_name())
            elif isinstance(iter_data.get("primary_metric_name"), str):
                primary_metric_name = str(iter_data.get("primary_metric_name"))
            best_primary_metric_value = _clip01(
                _to_float(
                    val_metrics_map.get(primary_metric_name),
                    default=_to_float(iter_data.get("criteria_pass_fraction_avg"), default=0.0),
                )
            )
            best_last_sub_folder = iter_data.get('last_sub_folder', f'iter{i}')
    
    if best_iter is not None:
        return {
            'iteration': best_iter,
            'primary_metric_value': best_primary_metric_value,
            'last_sub_folder': best_last_sub_folder,
        }
    
    return None


def write_final_test_evaluation(
    *,
    workspace_base: Path,
    evaluated_iteration: int,
    primary_metric_name: str,
    test_metrics: Mapping[str, Any],
    test_total: int,
    selected_iteration: int,
) -> Path:
    workspace_base = Path(workspace_base)
    meta_agent_dir = workspace_base / "meta_agent"
    meta_agent_dir.mkdir(parents=True, exist_ok=True)
    output_path = meta_agent_dir / "final_test_evaluation.json"
    payload = {
        "evaluated_iteration": int(evaluated_iteration),
        "primary_metric_name": str(primary_metric_name),
        "test_total": int(test_total),
        "test_metrics": _canonicalize_summary_metrics(test_metrics),
        "selection_context": {
            "selected_iteration": int(selected_iteration),
            "ranking_split": "val",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def cleanup_irrelevant_files(
    iter_dir: Path,
    agent_type: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Clean up irrelevant files generated by agents.
    
    Meta-agent should only generate:
    - .agents/skills/learning-context/SKILL.md
    
    Base-agent should only generate:
    - retrieve_context.py
    - context/ directory (with markdown files)
    
    Args:
        iter_dir: Iteration directory
        agent_type: "meta" or "base"
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    iter_dir = Path(iter_dir)
    
    # Files/directories that should always be preserved
    protected_paths = {
        "data",
        "__pycache__",
        ".agents",
        "context",
        "interfaces",  # New interface-based system
        "codegen",  # Per-sample code generation logs
    }
    
    deleted_files = []
    
    if agent_type == "meta":
        # Meta-agent: Only keep .agents/skills/learning-context/SKILL.md
        # Delete everything else at root level (except protected paths)
        for item in iter_dir.iterdir():
            item_name = item.name
            
            # Skip protected paths
            if item_name in protected_paths:
                continue
            
            # Skip hidden files/directories (except .agents which is protected)
            if item_name.startswith("."):
                continue
            
            # Delete everything else
            try:
                if item.is_file():
                    item.unlink()
                    deleted_files.append(item_name)
                    logger.debug(f"Deleted irrelevant file: {item_name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_files.append(f"{item_name}/")
                    logger.debug(f"Deleted irrelevant directory: {item_name}/")
            except Exception as e:
                logger.warning(f"Failed to delete {item_name}: {e}")
    
    elif agent_type == "base":
        # Base-agent: Only keep context/ and interfaces/
        # Delete other files at root level (except protected paths)
        for item in iter_dir.iterdir():
            item_name = item.name
            
            # Skip protected paths
            if item_name in protected_paths:
                continue
            
            # Skip hidden files/directories
            if item_name.startswith("."):
                continue
            
            # Skip context/ and interfaces/ (they are protected)
            if item_name in ["context", "interfaces"]:
                continue
            
            # Delete everything else
            try:
                if item.is_file():
                    item.unlink()
                    deleted_files.append(item_name)
                    logger.debug(f"Deleted irrelevant file: {item_name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_files.append(f"{item_name}/")
                    logger.debug(f"Deleted irrelevant directory: {item_name}/")
            except Exception as e:
                logger.warning(f"Failed to delete {item_name}: {e}")

    context_dir = iter_dir / "context"
    removed_context_artifacts = _remove_removed_context_artifacts(context_dir)
    if removed_context_artifacts:
        deleted_files.extend(f"context/{rel}" for rel in removed_context_artifacts)
    
    if deleted_files:
        logger.info(f"🧹 Cleaned up {len(deleted_files)} irrelevant file(s): {', '.join(deleted_files)}")
    else:
        logger.debug("No irrelevant files found to clean up")


def _remove_removed_context_artifacts(context_dir: Path) -> list[str]:
    """Remove deprecated context artifacts and return removed relative paths."""
    removed: list[str] = []
    if not context_dir.exists() or not context_dir.is_dir():
        return removed

    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name not in _REMOVED_CONTEXT_ARTIFACT_FILENAMES:
            continue
        try:
            rel = path.relative_to(context_dir).as_posix()
        except ValueError:
            rel = path.name
        try:
            path.unlink()
        except OSError:
            continue
        removed.append(rel)

    return removed
