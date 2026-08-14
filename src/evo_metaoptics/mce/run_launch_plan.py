from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .run_config import RunConfig
from .run_preflight import capture_gpu_snapshot


@dataclass(frozen=True)
class LaunchPlan:
    invocations: list[dict[str, Any]]


class LaunchPlanRenderer:
    @staticmethod
    def render(config: RunConfig | Mapping[str, Any]) -> LaunchPlan:
        run_config = _coerce_run_config(config)
        invocations = [_render_invocation(phase) for phase in _phase_configs(run_config)]
        return LaunchPlan(invocations=invocations)


def normalize_env_for_learning(config: RunConfig | Mapping[str, Any]) -> dict[str, str]:
    run_config = _coerce_run_config(config)
    return _base_env(run_config)


def generate_launch_plans(config: RunConfig | Mapping[str, Any]) -> list[dict[str, Any]]:
    return LaunchPlanRenderer.render(config).invocations


def capture_startup_gpu_snapshot(
    config: RunConfig | Mapping[str, Any],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    run_config = _coerce_run_config(config)
    target_path = output_path or (run_config.log_dir / "gpu_status_snapshot.json")
    return capture_gpu_snapshot(run_name=run_config.name, output_path=target_path)


def _coerce_run_config(config: RunConfig | Mapping[str, Any]) -> RunConfig:
    if isinstance(config, RunConfig):
        return config
    return RunConfig.from_dict(dict(config))


def _phase_configs(run_config: RunConfig) -> list[dict[str, Any]]:
    return [
        {
            "workspace": str(run_config.workspace),
            "log_dir": str(run_config.log_dir),
            "run_id": run_config.run_slug,
            "env_name": run_config.experiment.env,
            "model": run_config.experiment.model,
            "train_data": run_config.data.train_data,
            "val_data": run_config.data.val_data,
            "test_data": run_config.data.test_data,
            "test_limit": run_config.execution.test_limit,
            "val_limit": run_config.execution.val_limit,
            "train_batch_size": run_config.execution.train_batch_size,
            "meta_agent_hard_retries": run_config.execution.meta_agent_hard_retries,
            "codegen_rounds": run_config.execution.codegen_rounds,
            "codegen_inner_attempts": run_config.execution.codegen_inner_attempts,
            "pi_timeout_s": run_config.execution.pi_timeout_s,
            "skill_path": run_config.mode.skill_path,
            "traces": run_config.traces,
            "iterations": run_config.execution.iterations,
            "start_iter": run_config.execution.start_iter,
            "train_limit": run_config.execution.train_limit,
            "no_meta_agent": run_config.mode.no_meta_agent,
            "env": normalize_env_for_learning(run_config),
            "phase": "training",
        }
    ]


def _render_invocation(phase_config: Mapping[str, Any]) -> dict[str, Any]:
    argv = [
        "--workspace",
        str(phase_config["workspace"]),
        "--env",
        str(phase_config["env_name"]),
        "--train-data",
        str(phase_config["train_data"]),
        "--val-data",
        str(phase_config["val_data"]),
        "--model",
        str(phase_config["model"]),
        "--iterations",
        str(phase_config["iterations"]),
        "--start-iter",
        str(phase_config["start_iter"]),
        "--log-dir",
        str(phase_config["log_dir"]),
        "--run-id",
        str(phase_config["run_id"]),
    ]
    _append_optional_arg(argv, "--test-data", phase_config.get("test_data"))
    _append_optional_arg(argv, "--test-limit", phase_config.get("test_limit"))
    _append_optional_arg(argv, "--train-limit", phase_config.get("train_limit"))
    _append_optional_arg(argv, "--val-limit", phase_config.get("val_limit"))
    _append_optional_arg(argv, "--train-batch-size", phase_config.get("train_batch_size"))
    _append_optional_arg(argv, "--meta-agent-hard-retries", phase_config.get("meta_agent_hard_retries"))
    _append_optional_arg(argv, "--codegen-rounds", phase_config.get("codegen_rounds"))
    _append_optional_arg(argv, "--codegen-inner-attempts", phase_config.get("codegen_inner_attempts"))
    _append_optional_arg(argv, "--pi-timeout-s", phase_config.get("pi_timeout_s"))
    if phase_config.get("skill_path") and not phase_config.get("no_meta_agent"):
        argv.extend(["--skill-path", str(phase_config["skill_path"])])
    if phase_config.get("no_meta_agent"):
        argv.append("--no-meta-agent")
    argv.extend(_trace_flags(phase_config["traces"].pi_session_traces))
    return {
        "argv": argv,
        "env": dict(phase_config["env"]),
        "phase": phase_config["phase"],
        "gpu_snapshot_path": str(Path(phase_config["log_dir"]) / "gpu_status_snapshot.json"),
    }


def _append_optional_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _trace_flags(pi_session_traces: bool) -> list[str]:
    return [
        "--pi-session-traces" if pi_session_traces else "--no-pi-session-traces",
    ]


def _base_env(run_config: RunConfig) -> dict[str, str]:
    return dict(run_config.env_vars)


__all__ = [
    "LaunchPlan",
    "LaunchPlanRenderer",
    "capture_startup_gpu_snapshot",
    "generate_launch_plans",
    "normalize_env_for_learning",
]
