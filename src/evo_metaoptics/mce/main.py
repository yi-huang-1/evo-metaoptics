"""
MCE Main Loop: Orchestrate multiple iterations of context engineering.
"""

import os
import json
import asyncio
import argparse
import importlib
import logging
import signal
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from evo_metaoptics.mce.timing_report import write_timing_report_check
from evo_metaoptics.mce.pi_session_export import export_pi_session_html_reports
from evo_metaoptics.mce.utils import (
    create_iteration_workspace, 
    setup_base_agent_workspace, 
    prepare_meta_agent_reference,
    copy_skills_to_sub_iteration,
    get_sub_iteration_folder_name,
    aggregate_iteration_results,
    _find_best_iteration,
    write_final_test_evaluation,
)
from evo_metaoptics.mce.eval import batch_evaluate, load_interfaces
from evo_metaoptics.mce.iteration_review import run_iteration_review
from evo_metaoptics.mce.logging_utils import setup_logger, setup_run_logger
from evo_metaoptics.mce.run_manifest import write_run_manifest
from evo_metaoptics.mce.meta_agent import _resolve_best_prior_skill_path, run_meta_agent
from evo_metaoptics.mce.base_agent import run_base_agent
from evo_metaoptics.mce.skills import (
    ensure_learning_context_skill_seed,
    learning_context_skills_host_dir,
    learning_context_skill_host_path,
    validate_skill_markdown,
)
from evo_metaoptics.mce_env.base import EnvironmentRuntimeConfig
from evo_metaoptics.mce_env.registry import EnvironmentRegistry

def load_dotenv(*args: Any, **kwargs: Any) -> bool:
    try:
        dotenv_module = importlib.import_module("dotenv")
    except ModuleNotFoundError:
        return False
    return bool(dotenv_module.load_dotenv(*args, **kwargs))

load_dotenv(override=True)

_LEARNING_MIN_TRAIN_SAMPLES_ENV = "MCE_LEARNING_MIN_TRAIN_SAMPLES"
_DEFAULT_LEARNING_MIN_TRAIN_SAMPLES = 2
_VALIDATION_ATTEMPTS_ENV = "MCE_VALIDATION_ATTEMPTS"
_DEFAULT_VALIDATION_ATTEMPTS = 3


def _validate_pre_evolved_skill_dir(skill_source: Path) -> None:
    """Validate pre-evolved skill directory against native skills contract."""
    skill_file = skill_source / "learning-context" / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(
            f"Pre-evolved skill missing required file: {skill_file}"
        )

    skill_md = skill_file.read_text(encoding="utf-8")
    is_valid, error = validate_skill_markdown(
        skill_md,
        expected_name="learning-context",
    )
    if not is_valid:
        raise ValueError(f"Invalid pre-evolved skill at {skill_file}: {error}")



def _emit_timing_report_check(run_dir: Path, logger: logging.Logger) -> None:
    try:
        report_path = write_timing_report_check(run_dir, top_k=20)
    except Exception:
        logger.error(
            "Failed to generate timing_report_check.json",
            exc_info=True,
        )
        return
    logger.info(f"🧾 Timing report check written: {report_path}")


def _emit_pi_session_html_exports(run_dir: Path, logger: logging.Logger) -> None:
    try:
        exported_paths = export_pi_session_html_reports(
            session_dir=run_dir / "pi_session_traces",
            logger=logger,
        )
    except Exception:
        logger.error(
            "Failed to export Pi session HTML reports",
            exc_info=True,
        )
        return
    if exported_paths:
        logger.info(
            "🧾 Pi session HTML exports written: %d file(s) under %s",
            len(exported_paths),
            run_dir / "pi_session_traces",
        )


def _write_eval_split_artifact(
    *,
    env: Any,
    data_dir: Path,
    file_name: str,
    split_name: str,
    batch_data: Dict[str, Any],
    primary_metric_value: float,
    retry_artifact_dir: str,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    primary_metric_name = env.get_primary_metric_name()
    formatted_results = [
        env.format_result_for_training(item)
        for item in batch_data.get("results", [])
    ]
    summary = {
        f"{split_name}_{primary_metric_name}": primary_metric_value,
        f"{split_name}_metrics": batch_data["summary"]["metrics"],
        f"{split_name}_total": batch_data["summary"]["total"],
        f"{split_name}_errors": batch_data["summary"]["errors"],
        "retry_artifact_dir": retry_artifact_dir,
    }
    if extra_summary:
        summary.update(extra_summary)
    payload = {
        "summary": summary,
        "detailed_results": formatted_results,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = data_dir / file_name
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return artifact_path


def resolve_learning_min_train_samples_from_env(
    *,
    env_var: str = _LEARNING_MIN_TRAIN_SAMPLES_ENV,
    default_min_samples: int = _DEFAULT_LEARNING_MIN_TRAIN_SAMPLES,
) -> int:
    raw_value = os.getenv(env_var)
    if raw_value is None or not str(raw_value).strip():
        return int(max(1, default_min_samples))
    try:
        parsed = int(str(raw_value).strip())
    except ValueError as exc:
        raise ValueError(
            f"{env_var} must be a positive integer; got {raw_value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(f"{env_var} must be >= 1; got {parsed}.")
    return int(parsed)


def validate_learning_sample_count(
    *,
    effective_train_samples: int,
    min_train_samples: int,
) -> None:
    if int(effective_train_samples) >= int(min_train_samples):
        return
    raise ValueError(
        "learning_dataset_too_small: "
        f"effective train samples={int(effective_train_samples)} is below required minimum="
        f"{int(min_train_samples)}. "
        "Set MCE_LEARNING_MIN_TRAIN_SAMPLES to an explicit lower value only when degraded "
        "single-sample learning mode is intentional."
    )


def resolve_validation_attempts_from_env(
    *,
    env_var: str = _VALIDATION_ATTEMPTS_ENV,
    default_attempts: int = _DEFAULT_VALIDATION_ATTEMPTS,
) -> int:
    raw_value = os.getenv(env_var)
    if raw_value is None or not str(raw_value).strip():
        return int(max(1, default_attempts))
    try:
        parsed = int(str(raw_value).strip())
    except ValueError as exc:
        raise ValueError(
            f"{env_var} must be a positive integer; got {raw_value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(f"{env_var} must be >= 1; got {parsed}.")
    return int(parsed)


def _cleanup_failed_meta_artifacts(iter_dir: Path, logger: logging.Logger) -> None:
    skill_path = learning_context_skill_host_path(iter_dir)
    if skill_path.exists():
        skill_path.unlink()
        logger.info(f"Cleaned up failed skill artifact: {skill_path}")


def _apply_meta_agent_fallback(
    *,
    workspace_base: Path,
    iter_dir: Path,
    iteration: int,
    logger: logging.Logger,
) -> None:
    if iteration > 1:
        best_prior = _resolve_best_prior_skill_path(workspace_base, iteration)
        if best_prior and best_prior.exists():
            target = learning_context_skill_host_path(iter_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_prior, target)
            logger.warning(f"Meta-agent fallback: using best prior skill from {best_prior}")
            return
    ensure_learning_context_skill_seed(iter_dir)
    logger.warning("Meta-agent fallback: using seed scaffold skill")


async def run_iteration(
    workspace_base: Path,
    iteration: int,
    env_name: str,
    val_data_path: str,
    train_data_path: str,
    train_limit: int,
    val_limit: int,
    model: str,
    logger: logging.Logger,
    run_dir: Path,
    train_batch_size: Optional[int] = None,
    skill_path: Optional[str] = None,
    no_meta_agent: bool = False,
    test_data_path: Optional[str] = None,
    codegen_rounds: int = 5,
    codegen_inner_attempts: int = 5,
    pi_timeout_s: float = 300.0,
    pi_session_traces: bool = True,
    meta_agent_hard_retries: int = 2,
) -> dict:
    """
    Run a single iteration of the MCE loop with optional sub-iterations.
    
    When train_batch_size is specified, the iteration is split into sub-iterations:
    - Each sub-iteration processes a batch of training samples
    - Base-agent learns incrementally from each batch
    - Final validation is done at the end of all sub-iterations
    
    Args:
        workspace_base: Base workspace directory (absolute path)
        iteration: Iteration number
        env_name: Environment name
        val_data_path: Path to validation data file
        train_data_path: Path to training data file
        train_limit: Number of training samples to evaluate
        val_limit: Number of validation samples to evaluate
        model: LLM model to use
        logger: Logger instance
        run_dir: Run directory for organized logging
        train_batch_size: Batch size for sub-iterations (None = process all at once)
        skill_path: Path to pre-evolved skill directory (None = run meta-agent)
        no_meta_agent: Skip meta-agent entirely (no skills will be used)
    
    Returns:
        Dictionary with iteration results (accuracy, errors, etc.)
    """
    logger.info(f"\n🔄 ITERATION {iteration}")

    runtime_config = EnvironmentRuntimeConfig(
        codegen_rounds=codegen_rounds,
        codegen_inner_attempts=codegen_inner_attempts,
        run_dir=run_dir,
        pi_timeout_s=pi_timeout_s,
        pi_session_traces_enabled=pi_session_traces,
    )
    env = EnvironmentRegistry.get(env_name, runtime_config=runtime_config)

    # ---------------------------------------------------------------
    # BASELINE ITERATION 0 — Zero-shot evaluation (empty interfaces)
    # Use test_data when available; fall back to val_data.
    # ---------------------------------------------------------------
    if iteration == 0:
        baseline_data_path = test_data_path or val_data_path
        baseline_label = "test" if test_data_path else "val"
        logger.info(f"  📊 Baseline iteration 0: evaluating {baseline_label} set with empty interfaces")
        iter_folder = create_iteration_workspace(workspace_base, iteration, sub_iter=0)

        val_samples = env.load_samples(path=baseline_data_path, limit=val_limit, random_sample=False)
        logger.info(f"  Evaluating {len(val_samples)} {baseline_label} samples (zero-shot)...")
        val_data = await batch_evaluate(
            interfaces={},
            samples=val_samples,
            env_name=env_name,
            llm=model,
            iter_dir=iter_folder,
            log_dir=iter_folder / "codegen_val",
            environment=env,
        )
        val_summary = val_data["summary"]
        val_primary_metric = val_summary["primary_metric_value"]

        # Write baseline artifact (separate from evaluations.json)
        baseline_artifact = {
            "iteration": 0,
            "label": "baseline_zero_shot",
            "val_metrics": val_summary["metrics"],
            "primary_metric_name": env.get_primary_metric_name(),
            "val_total": val_summary["total"],
        }
        baseline_path = workspace_base / "meta_agent" / "baseline_evaluation.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_artifact, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✅ Baseline written: {baseline_path}")
        logger.info(f"  📊 Baseline val {val_summary['primary_metric']}: {val_primary_metric:.2%}")

        return {
            "iteration": 0,
            "val_primary_metric": val_primary_metric,
            "cumulative_rollouts": 0,
            "train_primary_metric": 0.0,
            "train_total": 0,
            "val_total": val_summary["total"],
            "num_sub_iters": 0,
            "sub_iterations": [],
            "val_criteria_pass_fraction": float(val_summary["metrics"].get("criteria_pass_fraction", 0.0) or 0.0),
            "val_best_margin": float(val_summary["metrics"].get("best_margin", 0.0) or 0.0),
        }

    task_instruction = env.get_task_instruction()
    interface_signatures = env.get_interface_signatures()
    
    llm = model

    # Load all training samples upfront
    train_samples = env.load_samples(path=train_data_path, limit=train_limit, random_sample=True, shuffle=True)
    validation_attempts = resolve_validation_attempts_from_env()

    effective_train_limit = len(train_samples) if train_limit is None else min(train_limit, len(train_samples))
    if effective_train_limit > 0:
        min_train_samples = resolve_learning_min_train_samples_from_env()
        validate_learning_sample_count(
            effective_train_samples=effective_train_limit,
            min_train_samples=min_train_samples,
        )
        logger.info(
            "  🧪 Learning sample guard: effective_train_samples=%s min_required=%s",
            effective_train_limit,
            min_train_samples,
        )
    
    # Calculate number of sub-iterations.
    # Zero training samples means skip batch/base-agent loop and only run final validation.
    if effective_train_limit <= 0:
        num_sub_iters = 0
        train_batch_size = 0
    elif train_batch_size is None or train_batch_size <= 0 or train_batch_size >= effective_train_limit:
        num_sub_iters = 1
        train_batch_size = effective_train_limit
    else:
        num_sub_iters = (effective_train_limit + train_batch_size - 1) // train_batch_size

    logger.info(f"  📊 Sub-iterations: {num_sub_iters} (batch_size={train_batch_size}, total={effective_train_limit})")
    logger.info(f"  🔁 Agent validation attempts: {validation_attempts}")
    
    # Step 1: Create first sub-iteration folder and setup skills
    first_sub_folder = create_iteration_workspace(workspace_base, iteration, sub_iter=0)
    
    # Conditional: use pre-evolved skill, skip meta-agent, or run meta-agent
    if no_meta_agent:
        logger.info("\n🧠 STEP 1: Skipping Meta-Agent (no skills will be used)")
        meta_result = {'success': True}
    elif skill_path:
        logger.info("\n🧠 STEP 1: Using Pre-Evolved Skill (skipping meta-agent)")
        logger.info(f"  Copying skills from: {skill_path}")

        # Copy skills from provided path to first sub-iteration
        skill_source = Path(skill_path)
        _validate_pre_evolved_skill_dir(skill_source)
        target_skills = learning_context_skills_host_dir(first_sub_folder)
        target_skills.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(skill_source, target_skills, dirs_exist_ok=True)
        logger.info("✅ Successfully copied pre-evolved skills")
        meta_result = {'success': True}
    else:
        meta_result = None
        for hard_attempt in range(1, meta_agent_hard_retries + 1):
            logger.info(
                f"\n🧠 STEP 1: META-AGENT (session {hard_attempt}/{meta_agent_hard_retries})"
            )
            meta_result = await run_meta_agent(
                iter_dir=first_sub_folder,
                task_instruction=task_instruction,
                interface_signatures=interface_signatures,
                iteration=iteration,
                model=model,
                workspace_base=workspace_base,
                run_dir=run_dir,
                max_validation_attempts=validation_attempts,
                timeout_s=pi_timeout_s,
                session_traces_enabled=pi_session_traces,
            )
            if meta_result["success"]:
                break
            logger.warning(
                f"Meta-agent session {hard_attempt}/{meta_agent_hard_retries} failed: "
                f"{meta_result.get('error', 'unknown')}. "
                f"{'Restarting with fresh session...' if hard_attempt < meta_agent_hard_retries else 'Exhausted hard retries.'}"
            )
            _cleanup_failed_meta_artifacts(first_sub_folder, logger)

        if meta_result is None:
            meta_result = {"success": False, "error": "meta-agent did not run"}
        if not meta_result["success"]:
            _apply_meta_agent_fallback(
                workspace_base=workspace_base,
                iter_dir=first_sub_folder,
                iteration=iteration,
                logger=logger,
            )
    
    # Step 2: Run sub-iterations
    logger.info(f"\n🔄 STEP 2: SUB-ITERATIONS - Batch Learning ({num_sub_iters} batches)")
    
    intermediate_results = []
    cumulative_rollouts = 0
    current_folder = first_sub_folder
    
    for sub_iter in range(num_sub_iters):
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 SUB-ITERATION {iteration}.{sub_iter} ({sub_iter + 1}/{num_sub_iters})")
        logger.info(f"{'='*60}")

        # Create sub-iteration folder (first one already created for meta-agent)
        if sub_iter == 0:
            sub_iter_folder = first_sub_folder
            setup_base_agent_workspace(
                workspace_base,
                sub_iter_folder,
                iteration,
                env,
                logger,
                interface_signatures=interface_signatures,
            )
        else:
            sub_iter_folder = create_iteration_workspace(workspace_base, iteration, sub_iter=sub_iter)
            # Copy skills and context from previous sub-iteration
            prev_sub_folder = workspace_base / get_sub_iteration_folder_name(iteration, sub_iter - 1)
            copy_skills_to_sub_iteration(prev_sub_folder, sub_iter_folder, logger)
            setup_base_agent_workspace(
                workspace_base, sub_iter_folder, iteration, env, logger,
                source_folder=prev_sub_folder,
                interface_signatures=interface_signatures,
            )
        
        # Calculate batch range
        start_idx = sub_iter * train_batch_size
        end_idx = min(start_idx + train_batch_size, len(train_samples))
        batch_samples = train_samples[start_idx:end_idx]
        batch_size = len(batch_samples)
        cumulative_rollouts += batch_size
        
        logger.info(f"  📊 Batch: samples [{start_idx}:{end_idx}] ({batch_size} samples)")
        logger.info(f"  📊 Cumulative rollouts: {cumulative_rollouts}")
        
        # Evaluate current batch
        logger.info(f"\n📊 Evaluating batch {sub_iter}...")
        
        # Load interfaces (or use empty if first run)
        try:
            interfaces = load_interfaces(sub_iter_folder, interface_signatures)
        except FileNotFoundError:
            interfaces = {}  # Empty interfaces for first evaluation
        
        batch_data = await batch_evaluate(
            interfaces=interfaces,
            samples=batch_samples,
            env_name=env_name,
            llm=llm,
            iter_dir=sub_iter_folder,
            log_dir=sub_iter_folder / "codegen",
            environment=env,
        )

        batch_summary = batch_data["summary"]
        primary_metric_value = batch_summary["primary_metric_value"]
        
        logger.info(f"  ✅ Batch {batch_summary['primary_metric']}: {primary_metric_value:.2%}")
        
        # Save batch evaluation results as train.json for base-agent to learn from
        data_dir = sub_iter_folder / "data"
        _write_eval_split_artifact(
            env=env,
            data_dir=data_dir,
            file_name="train.json",
            split_name="train",
            batch_data=batch_data,
            primary_metric_value=primary_metric_value,
            retry_artifact_dir="codegen",
            extra_summary={
                "batch_idx": sub_iter,
                "cumulative_rollouts": cumulative_rollouts,
            },
        )
    
        # Run base-agent to learn from this batch
        logger.info(f"\n🤖 BASE-AGENT: Learning from batch {sub_iter}...")
        base_result = await run_base_agent(
            iter_dir=sub_iter_folder,
            task_instruction=task_instruction,
            interface_signatures=interface_signatures,
            model=model,
            workspace_base=workspace_base,
            run_dir=run_dir,
            iteration=iteration,
            max_validation_attempts=validation_attempts,
            required_context_files=env.get_required_context_files(),
            context_min_chars=env.get_min_context_file_chars(),
            timeout_s=pi_timeout_s,
            session_traces_enabled=pi_session_traces,
        )
        
        if not base_result['success']:
            logger.error(f"Base-agent failed: {base_result['error']}")
            raise Exception(f"Base-agent failed at sub-iter {sub_iter}: {base_result['error']}")
        
        logger.info(f"✅ Base-agent completed for sub-iter {sub_iter}")
        
        # Record intermediate result
        intermediate_results.append({
            "sub_iter": sub_iter,
            "folder": str(sub_iter_folder),
            "batch_start": start_idx,
            "batch_end": end_idx,
            "batch_size": batch_size,
            "cumulative_rollouts": cumulative_rollouts,
            "batch_train_primary_metric": primary_metric_value,
            "batch_train_metrics": batch_summary["metrics"],
        })
        
        current_folder = sub_iter_folder
    
    # Step 3: Final validation evaluation
    final_folder = current_folder  # Use the last sub-iteration folder directly
    
    logger.info("\n📊 STEP 3: FINAL VALIDATION")

    interfaces = load_interfaces(final_folder, interface_signatures)
    val_samples = env.load_samples(path=val_data_path, limit=val_limit, random_sample=False)
    logger.info(f"Evaluating {len(val_samples)} validation samples...")
    val_data = await batch_evaluate(
        interfaces=interfaces,
        samples=val_samples,
        env_name=env_name,
        llm=llm,
        iter_dir=final_folder,
        log_dir=final_folder / "codegen_val",
        environment=env,
    )

    val_summary = val_data["summary"]
    val_primary_metric = val_summary["primary_metric_value"]
    val_artifact = _write_eval_split_artifact(
        env=env,
        data_dir=final_folder / "data",
        file_name="val.json",
        split_name="val",
        batch_data=val_data,
        primary_metric_value=val_primary_metric,
        retry_artifact_dir="codegen_val",
    )
    logger.info("✅ Validation detailed results written: %s", val_artifact)
    
    # Aggregate results to meta_agent/ for meta-agent review
    # (No longer saving to individual sub-iteration folders)
    skill_guard_payload: Dict[str, Any] | None = None
    if isinstance(meta_result, Mapping):
        raw_skill_guard = meta_result.get("skill_guard")
        if isinstance(raw_skill_guard, dict):
            skill_guard_payload = dict(raw_skill_guard)

    aggregate_iteration_results(
        workspace_base=workspace_base,
        iteration=iteration,
        sub_iterations=intermediate_results,
        val_primary_metric=val_primary_metric,
        val_metrics=val_summary["metrics"],
        val_total=val_summary["total"],
        cumulative_rollouts=cumulative_rollouts,
        num_sub_iters=num_sub_iters,
        last_sub_folder_name=current_folder.name,
        environment=env,
        skill_guard=skill_guard_payload,
        logger=logger,
    )
    if not no_meta_agent and not skill_path:
        try:
            run_iteration_review(
                workspace_base=workspace_base,
                iteration=iteration,
                last_sub_folder_name=current_folder.name,
                model=model,
                run_dir=run_dir,
                timeout_s=pi_timeout_s,
                session_traces_enabled=pi_session_traces,
                logger=logger,
            )
        except Exception:
            logger.exception(
                "⚠️  Iteration review failed for iter%s; continuing without blocking.",
                iteration,
            )
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ITERATION {iteration} COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"  📊 Total rollouts: {cumulative_rollouts}")
    logger.info(f"  📊 Sub-iterations: {num_sub_iters}")
    logger.info(f"  ✅ Final validation {val_summary['primary_metric']}: {val_primary_metric:.2%}")
    logger.info(
        "  📈 Final validation quality: "
        f"criteria_pass_fraction={float(val_summary['metrics'].get('criteria_pass_fraction', 0.0) or 0.0):.4f}, "
        f"best_margin={float(val_summary['metrics'].get('best_margin', 0.0) or 0.0):.6f}, "
        f"criteria_violation_norm={float(val_summary['metrics'].get('criteria_violation_norm', 0.0) or 0.0):.4f}"
    )
    
    # Log sub-iteration summary
    logger.info("\n  📈 Sub-iteration results:")
    for sr in intermediate_results:
        logger.info(f"    - Sub {sr['sub_iter']}: batch_metric={sr['batch_train_primary_metric']:.2%}, rollouts={sr['cumulative_rollouts']}")
    
    return {
        "iteration": iteration,
        "train_primary_metric": intermediate_results[-1]["batch_train_primary_metric"] if intermediate_results else 0.0,
        "val_primary_metric": val_primary_metric,
        "train_total": cumulative_rollouts,
        "val_total": val_summary["total"],
        "cumulative_rollouts": cumulative_rollouts,
        "num_sub_iters": num_sub_iters,
        "sub_iterations": intermediate_results,
        "val_criteria_pass_fraction": float(val_summary["metrics"].get("criteria_pass_fraction", 0.0) or 0.0),
        "val_best_margin": float(val_summary["metrics"].get("best_margin", 0.0) or 0.0),
    }


async def main():
    """Main entry point for MCE loop."""
    parser = argparse.ArgumentParser(
        description="Run MCE main loop: iterate, evaluate, and log results"
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path to workspace base directory (e.g., workspace/finer)"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment type"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data file"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data file"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to held-out test data file"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Number of held-out test samples to evaluate (0 = all available)"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=50,
        help="Training batch size for sub-iterations (default: 50). "
             "Values <= 0 or >= train-limit collapse to one batch per iteration."
    )
    parser.add_argument(
        "--meta-agent-hard-retries",
        type=int,
        default=2,
        help="Number of hard restart attempts for meta-agent (default: 2)",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Number of samples in training (default: all available)"
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Number of samples in validation (default: all available)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat-v3.1",
        help="LLM model used in context eval (default: deepseek/deepseek-chat-v3.1)"
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=1,
        help="Starting iteration number (default: 1)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)"
    )
    parser.add_argument(
        "--skill-path",
        type=str,
        default=None,
        help="Path to pre-evolved skill directory. "
             "If provided, meta-agent will be skipped and this skill will be used for all iterations"
    )
    parser.add_argument(
        "--no-meta-agent",
        action="store_true",
        default=False,
        help="Skip meta-agent entirely (no skills will be used). "
             "This is mutually exclusive with --skill-path"
    )
    parser.add_argument(
        "--codegen-rounds",
        type=int,
        default=5,
        help="Number of outer codegen rounds for inverse-design environments (default: 5)",
    )
    parser.add_argument(
        "--codegen-inner-attempts",
        type=int,
        default=5,
        help="Number of inner codegen attempts per round for inverse-design environments (default: 5)",
    )
    parser.add_argument(
        "--pi-timeout-s",
        type=float,
        default=300.0,
        help="Pi subprocess timeout in seconds (default: 300)",
    )
    
    # Pi session persistence control (default: ON)
    session_traces_group = parser.add_mutually_exclusive_group()
    session_traces_group.add_argument(
        "--pi-session-traces",
        action="store_true",
        dest="pi_session_traces",
        help="Enable Pi session persistence (default: enabled)"
    )
    session_traces_group.add_argument(
        "--no-pi-session-traces",
        action="store_false",
        dest="pi_session_traces",
        help="Disable Pi session persistence"
    )
    parser.set_defaults(pi_session_traces=True)
    
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run identifier shared between workspace and log directory. "
             "When omitted, a YYYYMMDD_HHMMSS timestamp is auto-generated."
    )

    args = parser.parse_args()

    from evo_metaoptics.mce.shutdown import request_shutdown

    def _shutdown_signal_handler(signum: int, frame: Any) -> None:
        request_shutdown()
        for t in asyncio.all_tasks():
            t.cancel()

    signal.signal(signal.SIGINT, _shutdown_signal_handler)
    signal.signal(signal.SIGTERM, _shutdown_signal_handler)

    # Validate mutually exclusive flags
    if args.skill_path and args.no_meta_agent:
        parser.error("--skill-path and --no-meta-agent are mutually exclusive")
    
    # Setup run directory for organized logging
    run_dir = setup_run_logger(log_base_dir=args.log_dir, run_id=args.run_id)
    
    # Setup main logger with run directory
    logger = setup_logger(
        name="mce_main",
        run_dir=run_dir,
        agent_type="run_summary",
        console_colors=True,
        minimal_console=False
    )

    # Print run start message
    timestamp = run_dir.name.replace("run_", "")
    print(f"[RUN {timestamp}] Starting MCE with {args.iterations} iteration(s) (iter{args.start_iter}-iter{args.start_iter + args.iterations - 1})")
    
    # Resolve workspace path
    workspace_base = Path(args.workspace).resolve()
    if not workspace_base.exists():
        logger.info(f"Creating workspace directory: {workspace_base}")
        workspace_base.mkdir(parents=True, exist_ok=True)

    # Write run manifest to both workspace and log directory
    manifest_args = {
        "env": args.env,
        "model": args.model,
        "train_data": args.train_data,
        "val_data": args.val_data,
        "test_data": args.test_data,
        "iterations": args.iterations,
        "start_iter": args.start_iter,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "train_batch_size": args.train_batch_size,
        "meta_agent_hard_retries": args.meta_agent_hard_retries,
        "skill_path": args.skill_path,
        "no_meta_agent": args.no_meta_agent,
        "pi_session_traces": args.pi_session_traces,
    }
    run_id = run_dir.name.replace("run_", "")
    ws_manifest, ld_manifest = write_run_manifest(
        workspace_base=workspace_base,
        run_dir=run_dir,
        run_id=run_id,
        args=manifest_args,
    )
    logger.info(f"  Run manifest: {ws_manifest}")
    logger.info(f"  Run manifest: {ld_manifest}")
    
    # Validate skill path if provided
    if args.skill_path:
        skill_path = Path(args.skill_path).resolve()
        if not skill_path.exists() or not skill_path.is_dir():
            logger.error(f"Skill path not found or not a directory: {skill_path}")
            return
        try:
            _validate_pre_evolved_skill_dir(skill_path)
        except Exception as e:
            logger.error(f"Invalid pre-evolved skill path: {e}")
            return
        logger.info(f"⚠️  Using pre-evolved skill from: {skill_path}")
        logger.info("⚠️  Meta-agent will be SKIPPED for all iterations")
    
    logger.info("\n🚀 MCE MAIN LOOP")
    logger.info(f"  Workspace: {workspace_base}")
    logger.info(f"  Environment: {args.env}")
    logger.info(f"  Iterations: {args.iterations} (starting from iter{args.start_iter})")
    logger.info(f"  Training data: {args.train_data}")
    logger.info(f"  Validation data: {args.val_data}")
    logger.info(f"  Held-out test data: {args.test_data}")
    logger.info(f"  Train samples: {args.train_limit if args.train_limit is not None else 'all'}")
    logger.info(f"  Val samples: {args.val_limit if args.val_limit is not None else 'all'}")
    if args.test_data:
        logger.info(f"  Test samples: {args.test_limit if args.test_limit > 0 else 'all'}")
    if args.train_batch_size:
        if args.train_limit is None:
            logger.info(
                f"  Train batch size: {args.train_batch_size} (sub-iterations depend on loaded training sample count)"
            )
        else:
            num_sub_iters = (args.train_limit + args.train_batch_size - 1) // args.train_batch_size
            logger.info(f"  Train batch size: {args.train_batch_size} ({num_sub_iters} sub-iterations per iteration)")
    else:
        logger.info("  Train batch size: None (process all samples at once)")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Meta-agent hard retries: {args.meta_agent_hard_retries}")
    if args.no_meta_agent:
        logger.info("  Meta-agent: DISABLED (no skills)")
    elif args.skill_path:
        logger.info(f"  Pre-evolved skill: {args.skill_path}")
        logger.info("  Meta-agent: DISABLED (using pre-evolved skill)")
    else:
        logger.info("  Meta-agent: ENABLED")

    # Setup meta-agent reference data (copy training data once at the beginning)
    if args.train_data:
        prepare_meta_agent_reference(workspace_base, args.train_data, logger)

    # Run iterations
    results = []
    try:
        for i in range(args.start_iter, args.start_iter + args.iterations):
            result = await run_iteration(
                workspace_base=workspace_base,
                iteration=i,
                env_name=args.env,
                val_data_path=args.val_data,
                train_data_path=args.train_data,
                train_limit=args.train_limit,
                val_limit=args.val_limit,
                model=args.model,
                logger=logger,
                run_dir=run_dir,
                train_batch_size=args.train_batch_size,
                skill_path=args.skill_path,
                no_meta_agent=args.no_meta_agent,
                test_data_path=getattr(args, "test_data", None),
                codegen_rounds=args.codegen_rounds,
                codegen_inner_attempts=args.codegen_inner_attempts,
                pi_timeout_s=args.pi_timeout_s,
                pi_session_traces=args.pi_session_traces,
                meta_agent_hard_retries=args.meta_agent_hard_retries,
            )
            results.append(result)

        # Print summary
        logger.info("\n🎯 FINAL SUMMARY")
        logger.info(f"Completed {len(results)} iteration(s):")

        total_rollouts = 0
        for result in results:
            iter_num = result["iteration"]
            if "error" in result:
                logger.error(f"  ❌ iter{iter_num}: ERROR - {result['error']}")
            else:
                val_metric = result.get("val_primary_metric", result.get("val_accuracy", 0.0))
                rollouts = result.get("cumulative_rollouts", result.get("train_total", 0))
                num_sub_iters = result.get("num_sub_iters", 1)
                total_rollouts += rollouts

                if num_sub_iters > 1:
                    logger.info(f"  📊 iter{iter_num}: Val={val_metric:.2%}, Rollouts={rollouts} ({num_sub_iters} sub-iters)")
                else:
                    train_metric = result.get("train_primary_metric", result.get("train_accuracy", 0.0))
                    val_criteria_pass_fraction = float(result.get("val_criteria_pass_fraction", 0.0) or 0.0)
                    val_best_margin = float(result.get("val_best_margin", 0.0) or 0.0)
                    logger.info(
                        f"  📊 iter{iter_num}: Train={train_metric:.2%}, "
                        f"Val={val_metric:.2%}, Rollouts={rollouts}, "
                        f"criteria_pass_fraction={val_criteria_pass_fraction:.4f}, "
                        f"best_margin={val_best_margin:.6f}"
                    )

        logger.info(f"\n📊 Total rollouts: {total_rollouts}")

        # Find best iteration using the same selector used in workspace setup.
        best_info = _find_best_iteration(
            workspace_base=workspace_base,
            current_iteration=args.start_iter + args.iterations,
            env=None,
        )
        if best_info and isinstance(best_info.get("iteration"), int):
            best_iter = int(best_info["iteration"])
            eval_file = workspace_base / "meta_agent" / "evaluations.json"
            iter_key = f"iter{best_iter}"
            details = None
            if eval_file.exists():
                try:
                    evaluations = json.loads(eval_file.read_text(encoding="utf-8"))
                    candidate = evaluations.get(iter_key)
                    if isinstance(candidate, dict):
                        details = candidate
                except Exception:
                    details = None

            if details is None:
                logger.info(
                    f"\n🏆 Best iteration selector: iter{best_iter}"
                )
            else:
                def _coerce_float(value: Any, *, fallback: float) -> float:
                    if value is None:
                        return float(fallback)
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return float(fallback)

                val_metrics = details.get("val_metrics")
                val_metrics_map = val_metrics if isinstance(val_metrics, dict) else {}
                val_success_goal = _coerce_float(
                    val_metrics_map.get("success_goal"),
                    fallback=0.0,
                )
                val_success_exec = _coerce_float(
                    val_metrics_map.get("success_exec"),
                    fallback=0.0,
                )
                val_criteria_pass_fraction = _coerce_float(
                    val_metrics_map.get("criteria_pass_fraction"),
                    fallback=0.0,
                )
                val_criteria_violation_norm = _coerce_float(
                    val_metrics_map.get("criteria_violation_norm"),
                    fallback=1.0,
                )
                val_best_margin = _coerce_float(
                    val_metrics_map.get("best_margin"),
                    fallback=float("-inf"),
                )
                val_cost_mean = _coerce_float(
                    val_metrics_map.get("solver_calls_total"),
                    fallback=_coerce_float(
                        val_metrics_map.get("opt_steps_total"),
                        fallback=float("inf"),
                    ),
                )
                logger.info(
                    "\n🏆 Best iteration selector: "
                    f"iter{best_iter} | "
                    f"val_criteria_pass_fraction={val_criteria_pass_fraction:.4f}, "
                    f"val_criteria_violation_norm={val_criteria_violation_norm:.4f}, "
                    f"val_best_margin={val_best_margin:.6f}, "
                    f"val_success_goal={val_success_goal:.2%}, "
                    f"val_success_exec={val_success_exec:.2%}, "
                    f"val_cost_mean={val_cost_mean:.2f}"
                )

                if args.test_data:
                    runtime_config = EnvironmentRuntimeConfig(
                        codegen_rounds=args.codegen_rounds,
                        codegen_inner_attempts=args.codegen_inner_attempts,
                        run_dir=run_dir,
                        pi_timeout_s=args.pi_timeout_s,
                        pi_session_traces_enabled=args.pi_session_traces,
                    )
                    env = EnvironmentRegistry.get(args.env, runtime_config=runtime_config)
                    target_folder = workspace_base / str(
                        details.get("last_sub_folder") or f"iter{best_iter}"
                    )
                    interfaces = load_interfaces(target_folder, env.get_interface_signatures())
                    test_limit = args.test_limit if args.test_limit > 0 else 0
                    test_samples = env.load_samples(
                        path=args.test_data,
                        limit=test_limit,
                        random_sample=False,
                    )
                    logger.info("\n🧪 HELD-OUT TEST EVALUATION")
                    logger.info("Evaluating %s held-out test samples...", len(test_samples))
                    test_data = await batch_evaluate(
                        interfaces=interfaces,
                        samples=test_samples,
                        env_name=args.env,
                        llm=args.model,
                        iter_dir=target_folder,
                        log_dir=target_folder / "codegen_test",
                        environment=env,
                    )
                    test_summary = test_data["summary"]
                    primary_metric_name = str(
                        details.get("primary_metric_name") or env.get_primary_metric_name()
                    )
                    test_primary_metric_value = float(
                        test_summary.get("primary_metric_value", 0.0) or 0.0
                    )
                    _write_eval_split_artifact(
                        env=env,
                        data_dir=target_folder / "data",
                        file_name="test.json",
                        split_name="test",
                        batch_data=test_data,
                        primary_metric_value=test_primary_metric_value,
                        retry_artifact_dir="codegen_test",
                    )
                    artifact_path = write_final_test_evaluation(
                        workspace_base=workspace_base,
                        evaluated_iteration=best_iter,
                        primary_metric_name=primary_metric_name,
                        test_metrics=test_summary["metrics"],
                        test_total=test_summary["total"],
                        selected_iteration=best_iter,
                    )
                    logger.info(
                        "✅ Held-out test artifact written: %s | test_metrics.%s=%.2f%%",
                        artifact_path,
                        primary_metric_name,
                        float(test_summary["metrics"].get(primary_metric_name, 0.0) or 0.0) * 100.0,
                    )
        else:
            has_valid = any("error" not in r for r in results)
            if has_valid:
                logger.warning(
                    "Best-iteration selector returned no result "
                    "(evaluations.json may be missing or empty) - Logs: %s",
                    run_dir,
                )
            else:
                logger.error("All iterations failed - Logs: %s", run_dir)
    finally:
        _emit_pi_session_html_exports(run_dir=run_dir, logger=logger)
        _emit_timing_report_check(run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
