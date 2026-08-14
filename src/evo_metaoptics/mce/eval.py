"""
Evaluate learned interfaces on samples with async and concurrent execution.
"""

import asyncio
import importlib
import json
import logging
import shutil
import tempfile
from typing import Callable, Dict, List, Any, Optional
from evo_metaoptics.mce_env.base import EnvironmentRuntimeConfig, Sample, InterfaceSignature, TaskEnvironment
from evo_metaoptics.mce_env.registry import EnvironmentRegistry
from pathlib import Path
from evo_metaoptics.mce.skills import learning_context_skill_host_path
from evo_metaoptics.mce.utils import compute_avg_metrics
from evo_metaoptics.mce.validation import (
    format_validation_feedback,
    load_interfaces_from_init,
    validate_interfaces,
)

def load_dotenv(*args: Any, **kwargs: Any) -> bool:
    try:
        dotenv_module = importlib.import_module("dotenv")
    except ModuleNotFoundError:
        return False
    return bool(dotenv_module.load_dotenv(*args, **kwargs))


load_dotenv(override=True)

logger = logging.getLogger(__name__)

MAX_CONCURRENCY = 8
_OPTIONAL_HELPER_INTERFACE_NAMES = {"build_extractor_hints"}



def _load_skill_guidance_from_iter_dir(iter_dir: Optional[Path]) -> str | None:
    if iter_dir is None:
        return None
    try:
        skill_path = learning_context_skill_host_path(Path(iter_dir))
        content = skill_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return content or None


def load_interfaces(iter_dir: Path, signatures: List[InterfaceSignature]) -> Dict[str, Callable]:
    """
    Load interfaces from an iteration directory.
    
    First tries to load from interfaces/__init__.py (new format).
    
    Args:
        iter_dir: Iteration directory
        signatures: Required interface signatures
        
    Returns:
        Dict mapping interface names to callables
    """
    # No required signatures: load additive helper interfaces if present.
    if not signatures:
        interfaces_dir = Path(iter_dir) / "interfaces"
        if not interfaces_dir.exists():
            logger.info("No interface signatures required, skipping interface loading")
            return {}
        try:
            exported = load_interfaces_from_init(Path(iter_dir))
        except Exception:
            logger.info(
                "No interface signatures required and optional helper exports unavailable"
            )
            return {}
        optional_helpers = {
            name: fn
            for name, fn in exported.items()
            if name in _OPTIONAL_HELPER_INTERFACE_NAMES and callable(fn)
        }
        if optional_helpers:
            logger.info(
                "Loaded optional helper interfaces: "
                f"{sorted(optional_helpers.keys())}"
            )
        else:
            logger.info("No interface signatures required, skipping interface loading")
        return optional_helpers
    
    iter_dir = Path(iter_dir)
    
    logger.info(f"Loading interfaces from {iter_dir / 'interfaces'}")

    validation_result = validate_interfaces(iter_dir, signatures)
    if not validation_result.success:
        raise ValueError(format_validation_feedback(validation_result))

    return validation_result.interfaces


async def batch_evaluate(
    interfaces: Dict[str, Callable],
    samples: List[Sample],
    env_name: str,
    llm: Optional[str] = None,
    iter_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    environment: TaskEnvironment | None = None,
) -> Dict[str, Any]:
    """
    Evaluate samples using loaded interfaces.
    
    Args:
        interfaces: Dict mapping interface names to callables
        samples: List of Sample objects to evaluate
        env_name: Environment name
        llm: Optional LLM client for environments that need inference
        iter_dir: Optional iteration directory. When set, `iter_dir/context` is copied
            per-sample and passed to environments as `context_dir` (context root).
        log_dir: Optional log directory (for storing agent trajectories)
    
    Returns:
        Dictionary with evaluation results
    """
    environment = environment or EnvironmentRegistry.get(env_name)
    runtime_config_getter = getattr(environment, "get_runtime_config", None)
    runtime_config = runtime_config_getter() if callable(runtime_config_getter) else None
    # Prepare canonical context directory for agent environments.
    # Contract: `context_dir` passed into environments always points to a context root.
    canonical_context_dir = None
    if iter_dir:
        candidate = Path(iter_dir) / "context"
        if candidate.exists():
            canonical_context_dir = candidate
    skill_guidance_text = _load_skill_guidance_from_iter_dir(Path(iter_dir) if iter_dir else None)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    completed_count = 0
    total_count = len(samples)
    
    async def evaluate_single(sample: Sample) -> Dict[str, Any]:
        """Evaluate a single sample using interfaces."""
        nonlocal completed_count
        async with semaphore:
            temp_dir_obj = None
            sample_context_dir = None
            try:
                # Isolate context per sample to prevent cross-sample mutation.
                # Each sample gets its own `<tmp>/context` root directory.
                if canonical_context_dir is not None:
                    temp_dir_obj = tempfile.TemporaryDirectory(
                        prefix=f"mce_eval_ctx_{sample.id}_"
                    )
                    sample_context_dir = Path(temp_dir_obj.name) / "context"
                    shutil.copytree(canonical_context_dir, sample_context_dir)

                eval_sample = sample
                if isinstance(skill_guidance_text, str) and skill_guidance_text:
                    existing = sample.extras.get("skill_guidance")
                    if not (isinstance(existing, str) and existing.strip()):
                        extras = dict(sample.extras)
                        extras["skill_guidance"] = skill_guidance_text
                        eval_sample = Sample(
                            id=sample.id,
                            question=sample.question,
                            context=sample.context,
                            ground_truth=sample.ground_truth,
                            extras=extras,
                        )

                # Call environment's evaluate with interfaces
                result = await environment.aevaluate(
                    sample=eval_sample,
                    interfaces=interfaces,
                    llm_client=llm,
                    context_dir=sample_context_dir,
                    log_dir=log_dir,
                )
                
                completed_count += 1
                metrics = result.metrics or {}
                s_exec = metrics.get("success_exec", 0.0) == 1.0
                s_goal = metrics.get("success_goal", 0.0) == 1.0
                status = "goal" if s_goal else ("exec" if s_exec else "FAIL")
                print(
                    f"\rProgress: {completed_count}/{total_count}"
                    f" ({completed_count/total_count*100:.1f}%)"
                    f" [sample={sample.id} {status}]",
                    end="", flush=True,
                )
                
                return {
                    "sample": eval_sample.to_dict(),
                    "evaluation": {
                        "feedback": result.feedback,
                        "ground_truth": result.ground_truth,
                        "metrics": result.metrics,
                        "trajectory": result.trajectory,
                    }
                }
            except Exception as e:
                completed_count += 1
                logger.error(f"Error evaluating sample {sample.id}: {e}")
                return {
                    "sample": sample.to_dict(),
                    "error": str(e),
                    "evaluation": {
                        "feedback": f"Error: {e}",
                        "ground_truth": sample.ground_truth,
                        "metrics": {},
                    }
                }
            finally:
                if temp_dir_obj is not None:
                    temp_dir_obj.cleanup()
    
    logger.info(f"Processing {total_count} samples with max concurrency of {MAX_CONCURRENCY}...")
    tasks = [asyncio.create_task(evaluate_single(sample)) for sample in samples]
    try:
        results = list(await asyncio.gather(*tasks))
    except (asyncio.CancelledError, KeyboardInterrupt):
        for task in tasks:
            if not task.done():
                task.cancel()
        results = []
        for task in tasks:
            if task.done() and not task.cancelled():
                try:
                    results.append(task.result())
                except Exception:
                    pass
    print()

    eval_errors = [r for r in results if "error" in r]
    successful = [r for r in results if "error" not in r]
    
    # Compute average metrics
    avg_metrics = compute_avg_metrics(successful)
    
    primary_metric_name = environment.get_primary_metric_name()
    primary_metric_value = avg_metrics.get(primary_metric_name, 0.0)
    
    log_data = {
        "summary": {
            "metrics": avg_metrics,
            "primary_metric": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "total": len(results),
            "successful": len(successful),
            "errors": len(eval_errors),
            "environment": env_name,
        },
        "errors": eval_errors,
        "results": successful,
    }
    
    logger.info("Evaluation Summary:")
    logger.info(f"  Primary Metric ({primary_metric_name}): {primary_metric_value:.2%}")
    logger.info(
        "  Quality Metrics: "
        f"criteria_pass_fraction={float(avg_metrics.get('criteria_pass_fraction', 0.0) or 0.0):.4f}, "
        f"best_margin={float(avg_metrics.get('best_margin', 0.0) or 0.0):.6f}"
    )
    logger.info(f"  Total: {len(results)} ({len(successful)} successful, {len(eval_errors)} errors)")

    return log_data


async def main():
    """Standalone evaluation script."""
    import argparse
    import time
    from evo_metaoptics.mce.logging_utils import setup_logger
    
    parser = argparse.ArgumentParser(
        description="Evaluate learned interfaces from a workspace folder"
    )
    parser.add_argument(
        "--iter_dir",
        type=str,
        default="",
        help="Path to the iteration directory"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat-v3.1",
        help="LLM model to use"
    )
    parser.add_argument(
        "--save-results-to",
        type=str,
        required=True,
        help="Directory to save results to"
    )
    args = parser.parse_args()
    
    # Setup log directory - use iter_dir/codegen if available, else fallback
    if args.iter_dir:
        iter_dir = Path(args.iter_dir).resolve()
        if not iter_dir.exists() or not iter_dir.is_dir():
            raise ValueError(f"Iteration directory error: {iter_dir}")
        log_dir = iter_dir / "codegen"
    else:
        iter_dir = None
        log_dir = Path("logs") / "eval_standalone"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_logger = setup_logger(name="eval", log_dir="logs", console_colors=True)

    eval_logger.info("="*80)
    eval_logger.info("EVALUATION SETUP")
    eval_logger.info("="*80)
    eval_logger.info(f"Iteration directory: {iter_dir}")
    eval_logger.info(f"Environment: {args.env}")
    eval_logger.info(f"Data: {args.data}")
    eval_logger.info(f"Sample limit: {args.limit}")
    
    environment = EnvironmentRegistry.get(args.env)
    signatures = environment.get_interface_signatures()
    
    # Load interfaces
    if iter_dir:
        eval_logger.info("Loading interfaces...")
        interfaces = load_interfaces(iter_dir, signatures)
        eval_logger.info(f"✓ Loaded interfaces: {list(interfaces.keys())}")
    else:
        interfaces = {}
        eval_logger.info("No iteration directory - using empty interfaces")
    
    # Initialize LLM if needed
    eval_logger.info(f"Initializing LLM: {args.model}")
    llm = args.model
    eval_logger.info("✓ LLM initialized")
    
    eval_logger.info("="*80)
    eval_logger.info("STARTING EVALUATION")
    eval_logger.info("="*80)
    
    samples = environment.load_samples(path=args.data, limit=args.limit, random_sample=False)
    eval_logger.info(f"📦 Loaded {len(samples)} samples from: {args.data}")
    
    start_time = time.time()
    
    results = await batch_evaluate(
        interfaces=interfaces,
        samples=samples,
        env_name=args.env,
        llm=llm,
        iter_dir=iter_dir,
        log_dir=log_dir,
        environment=environment,
    )
    
    elapsed = time.time() - start_time
    summary = results["summary"]

    print(f"\n✓ Completed: {summary['primary_metric_value']:.2%} {summary['primary_metric']} ({summary['total']} samples) in {elapsed:.0f}s")
    
    save_dir = Path(args.save_results_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "evaluation.json"
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    eval_logger.info(f"📁 Results saved to: {log_path}")
    eval_logger.info("="*80)
    

if __name__ == "__main__":
    asyncio.run(main())
