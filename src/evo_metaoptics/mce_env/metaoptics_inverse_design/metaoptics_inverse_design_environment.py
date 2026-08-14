from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evo_metaoptics.material_db.paths import default_db_path, default_source_path
from evo_metaoptics.mce.agent_runtime import create_pi_session, invoke_pi_session
from evo_metaoptics.mce.shutdown import is_shutdown_requested
from evo_metaoptics.mce.agents_md import write_agents_md
from evo_metaoptics.mce.template_registry import materialize_progressive_reference_subtree
from evo_metaoptics.mce.skills import (
    compose_skill_bundle,
    learning_context_skill_host_path,
    normalize_learning_context_skill_markdown,
    validate_skill_markdown,
)
from evo_metaoptics.mce_env.base import (
    EnvironmentResult,
    InterfaceSignature,
    Sample,
    TaskEnvironment,
)
from . import codegen_runner
from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import (
    RunnerResult,
    run_codegen,
)
from evo_metaoptics.mce_env.metaoptics_inverse_design.preloaded_skill import (
    compose_preloaded_template_skill,
)
from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval, validate_gt_eval

_DEFAULT_NUM_ROUNDS = 5
_DEFAULT_INNER_MAX_ATTEMPTS = 5
_NUM_ROUNDS = _DEFAULT_NUM_ROUNDS
_INNER_MAX_ATTEMPTS = _DEFAULT_INNER_MAX_ATTEMPTS
_CODEGEN_TIMEOUT_S = 600.0
_USE_RALPH_LOOP = True
_DEFAULT_CODEGEN_DEVICE = "cpu"

_SYSTEM_PROMPT = """You are a code-generation agent for inverse design.

Workflow:
1. Read AGENTS.md and any skill/context artifacts in the working directory.
2. If `context/strategy_summary.md` exists, read it before planning. Treat it as cross-iteration memory: durable strategy guidance carried forward from prior iterations, not just batch-local notes.
3. Write or update `solution.py` with `solve_inverse_design(*, device: str = "cpu") -> SolverResults`.
4. Put a lightweight first-line note in `solution.py`: `# Strategy: <family> - <why>`.
5. Keep implementation deterministic and executable.
6. On feedback, revise `solution.py` directly and try again.

Non-negotiable output contract:
- `solution.py` must exist in the workspace root.
- `solve_inverse_design` must call `solver.solve(source)` and return that result.
- Do not define a custom `SolverResults` class.

Constraints:
- Do not assume access to hidden evaluator internals.
- TorchRDIT built-in gradients are local optimization signals. If global exploration
  is needed, implement bounded global screening then local refinement.
- Choose one bounded global optimization family up front and explain it in the
  `# Strategy:` note; keep the guidance soft rather than rigid.
- IMPORTANT: When writing files, always use ABSOLUTE paths (e.g., `/workspace/iter1_sub0/solution.py`),
  not relative paths. The write tool requires absolute paths to function correctly.
"""


_API_HINTS: list[tuple[list[str], str]] = [
    (
        ["torchrdit.structures", "attribute 'structures'", "name 'structures'"],
        "torchrdit.structures does NOT exist. Use: from torchrdit.solver import get_solver_builder; solver = get_solver_builder()...build()",
    ),
    (
        ["torchrdit.solvers", "attribute 'solvers'", "name 'solvers'"],
        "torchrdit.solvers does NOT exist. Use: from torchrdit.solver import get_solver_builder",
    ),
    (
        ["torchrdit.sources", "attribute 'sources'", "name 'sources'"],
        "torchrdit.sources does NOT exist. Use: source = solver.add_source(theta=0, phi=0, pte=1, ptm=0)",
    ),
    (
        ["torchrdit.materials", "attribute 'materials'", "name 'materials'"],
        "torchrdit.materials does NOT exist. Use: from torchrdit.utils import create_material",
    ),
    (
        ["cannot import name"],
        "Check the skill templates for correct import paths. Key imports: from torchrdit.solver import get_solver_builder; from torchrdit.utils import create_material; from torchrdit.shapes import ShapeGenerator",
    ),
]


def _build_api_hints(error_msg: str, traceback_text: str | None) -> str:
    combined = (error_msg or "") + "\n" + (traceback_text or "")
    combined_lower = combined.lower()
    hints: list[str] = []
    for patterns, hint in _API_HINTS:
        if any(p.lower() in combined_lower for p in patterns):
            hints.append(hint)
    if not hints:
        return ""
    return "\n\nAPI HINTS:\n" + "\n".join(f"- {h}" for h in hints)


def _learning_context_skill_paths(iter_dir: Path) -> list[str]:
    return [str(learning_context_skill_host_path(iter_dir).parent)]


def _create_deep_agent_compat(
    *,
    backend_root: Path,
    system_prompt: str,
    skill_sources: list[str],
    model: str,
    run_dir: Path | None = None,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
) -> Any:
    del skill_sources
    return create_pi_session(
        iter_dir=backend_root,
        system_prompt=system_prompt,
        skills=[],
        model=model,
        skill_paths=_learning_context_skill_paths(backend_root),
        run_dir=run_dir,
        timeout_s=timeout_s,
        session_traces_enabled=session_traces_enabled,
    )


def _invoke_deep_agent_compat(
    agent: Any,
    prompt: str,
) -> dict[str, Any]:
    response = invoke_pi_session(agent, prompt)
    payload: dict[str, Any] = {"messages": []}
    if getattr(response, "error", None) is not None:
        payload["error"] = response.error
    messages = getattr(response, "messages", None)
    if isinstance(messages, list):
        payload["messages"] = messages
    return payload


_CREATE_DEEP_AGENT_COMPAT_NAME = "create_" + "mce_" + "deep_agent"
_INVOKE_DEEP_AGENT_COMPAT_NAME = "invoke_" + "mce_" + "deep_agent"
globals()[_CREATE_DEEP_AGENT_COMPAT_NAME] = _create_deep_agent_compat
globals()[_INVOKE_DEEP_AGENT_COMPAT_NAME] = _invoke_deep_agent_compat


async def close_mce_deep_agent(agent: Any) -> None:
    close_sync = getattr(agent, "close_sync", None)
    if callable(close_sync):
        close_sync()
        return
    close = getattr(agent, "close", None)
    if not callable(close):
        return
    maybe_coro = close()
    if asyncio.iscoroutine(maybe_coro):
        await maybe_coro


def make_run_codegen_tool(
    iter_dir: Path,
    device: str = _DEFAULT_CODEGEN_DEVICE,
) -> tuple[Any, dict[str, RunnerResult | None]]:
    code_path = Path(iter_dir) / "solution.py"
    result_ref: dict[str, RunnerResult | None] = {"runner_result": None}
    state = {
        "last_error": None,
        "consecutive_error_count": 0,
        "guard_active": False,
        "file_fingerprint": None,
    }

    def _fingerprint(path: Path) -> tuple[int, int, int] | None:
        if not path.is_file():
            return None
        stat = path.stat()
        return (int(stat.st_mtime_ns), int(stat.st_size), hash(path.read_text(encoding="utf-8")))

    def _reset_loop_guard(*, file_fingerprint: tuple[int, int, int] | None) -> None:
        state["last_error"] = None
        state["consecutive_error_count"] = 0
        state["guard_active"] = False
        state["file_fingerprint"] = file_fingerprint

    class _RunCodegenTool:
        def invoke(self, _args: dict[str, Any]) -> str:
            current_fp = _fingerprint(code_path)
            prev_fp = state["file_fingerprint"]
            if prev_fp is None:
                state["file_fingerprint"] = current_fp
            elif current_fp != prev_fp:
                _reset_loop_guard(file_fingerprint=current_fp)

            if not code_path.is_file():
                result_ref["runner_result"] = None
                return (
                    "EXECUTION FAILED\n"
                    "Error type: file_not_found\n"
                    "Error: solution.py not found\n"
                    "Action required: use write_file to create solution.py with "
                    'solve_inverse_design(*, device: str = "cpu").'
                )

            if bool(state["guard_active"]):
                return (
                    "LOOP GUARD ACTIVATED\n"
                    "Detected repeated identical execution failures.\n"
                    "Action required: use edit_file to make a substantive change to solution.py before retrying."
                )

            runner_result = codegen_runner.run_codegen(code_path=code_path, device=device)
            result_ref["runner_result"] = runner_result

            if runner_result.error is None:
                _reset_loop_guard(file_fingerprint=_fingerprint(code_path))
                return (
                    "EXECUTION SUCCESS\n"
                    "solution.py executed successfully.\n"
                    "You may now continue with analysis or final response."
                )

            error_type = str(runner_result.error_type or "runtime_error")
            error_text = str(runner_result.error)
            traceback_text = runner_result.traceback or ""

            if error_text == state["last_error"]:
                state["consecutive_error_count"] = int(state["consecutive_error_count"]) + 1
            else:
                state["last_error"] = error_text
                state["consecutive_error_count"] = 1

            state["file_fingerprint"] = _fingerprint(code_path)
            if int(state["consecutive_error_count"]) >= 3:
                state["guard_active"] = True

            api_hints = _build_api_hints(error_text, traceback_text)
            return (
                "EXECUTION FAILED\n"
                f"Error type: {error_type}\n"
                f"Error: {error_text}\n"
                f"Traceback:\n{traceback_text}\n"
                "Action required: use edit_file to fix solution.py, then run_codegen again."
                f"{api_hints}"
            )

    return _RunCodegenTool(), result_ref


class MetaopticsInverseDesignEnvironment(TaskEnvironment):
    def get_interface_signatures(self) -> List[InterfaceSignature]:
        return []

    def get_primary_metric_name(self) -> str:
        return "success_goal"

    def get_max_precheck_attempts(self) -> int:
        return self._resolve_num_rounds() * self._resolve_inner_max_attempts()

    def get_task_instruction(self) -> str:
        return (
            "Write Python code to solution.py that defines "
            'solve_inverse_design(*, device: str = "cpu") -> SolverResults. '
            "The host imports and executes this function for scoring."
        )

    def _resolve_num_rounds(self) -> int:
        runtime_value = self.get_runtime_config().codegen_rounds
        if isinstance(runtime_value, int) and runtime_value >= 1:
            return runtime_value
        return max(1, int(os.environ.get("MCE_CODEGEN_ROUNDS", str(_NUM_ROUNDS))))

    def _resolve_inner_max_attempts(self) -> int:
        runtime_value = self.get_runtime_config().codegen_inner_attempts
        if isinstance(runtime_value, int) and runtime_value >= 1:
            return runtime_value
        return max(1, int(os.environ.get("MCE_CODEGEN_INNER_ATTEMPTS", str(_INNER_MAX_ATTEMPTS))))

    def _runtime_run_dir(self) -> Path | None:
        return self.get_runtime_config().run_dir

    def _runtime_timeout_s(self) -> float | None:
        return self.get_runtime_config().pi_timeout_s

    def _runtime_session_traces_enabled(self) -> bool | None:
        return self.get_runtime_config().pi_session_traces_enabled

    async def aevaluate(
        self,
        sample: Sample,
        interfaces: Dict[str, Callable],
        model: str | None = None,
        llm_client: Any = None,
        context_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> EnvironmentResult:
        del interfaces
        if model is None and llm_client is not None:
            model = str(llm_client)

        query = str(sample.question or "").strip()
        device = self._resolve_codegen_device(sample)
        if not query:
            return self._agent_failure_result(
                last_error_type="invalid_sample",
                last_error_message="Sample missing question.",
                last_traceback=None,
                execution_time_s=0.0,
                attempt_count=0,
                trajectory=[],
            )

        gt_eval_raw = sample.extras.get("gt_eval") if isinstance(sample.extras, dict) else None
        if not isinstance(gt_eval_raw, dict):
            return self._agent_failure_result(
                last_error_type="invalid_sample",
                last_error_message="Sample missing gt_eval.",
                last_traceback=None,
                execution_time_s=0.0,
                attempt_count=0,
                trajectory=[],
            )

        try:
            gt_eval = validate_gt_eval(gt_eval_raw)
        except Exception as exc:
            return self._agent_failure_result(
                last_error_type="invalid_gt_eval",
                last_error_message=str(exc),
                last_traceback=None,
                execution_time_s=0.0,
                attempt_count=0,
                trajectory=[],
            )

        iter_dir = self._resolve_iter_dir(sample=sample, log_dir=log_dir)
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Write workspace payloads at iter_dir level for test/contract visibility
        write_agents_md(iter_dir=iter_dir, system_prompt=_SYSTEM_PROMPT, context_available=False)
        (iter_dir / "gt_eval.json").write_text(
            json.dumps(gt_eval_raw, indent=2, sort_keys=True), encoding="utf-8"
        )

        skill_content = self._resolve_skill_content(sample=sample, query=query, iter_dir=iter_dir)
        skill_path = self._write_skill_package(iter_dir, skill_content=skill_content)
        skill_valid, skill_error = validate_skill_markdown(skill_content)
        if not skill_valid:
            return self._agent_failure_result(
                last_error_type="invalid_skill",
                last_error_message=skill_error or f"Invalid skill at {skill_path}",
                last_traceback=None,
                execution_time_s=0.0,
                attempt_count=0,
                trajectory=[],
            )

        skill_bundle = compose_skill_bundle(
            workspace_base=iter_dir.parent,
            iter_folder_name=iter_dir.name,
            include_history=False,
            include_current=True,
            backend_scope="iteration",
        )
        skill_sources = skill_bundle.selected_sources

        if not _USE_RALPH_LOOP:
            return await self._run_compat_agent_loop(
                query=query,
                device=device,
                iter_dir=iter_dir,
                gt_eval=gt_eval,
                skill_sources=skill_sources,
                model=model or "",
                sample_id=str(sample.id),
            )

        return await asyncio.to_thread(
            self._run_multistart_loop,
            query=query,
            device=device,
            iter_dir=iter_dir,
            gt_eval=gt_eval,
            skill_content=skill_content,
            skill_sources=skill_sources,
            context_dir=context_dir,
            model=model or "",
            sample_id=str(sample.id),
        )

    async def _run_compat_agent_loop(
        self,
        *,
        query: str,
        device: str,
        iter_dir: Path,
        gt_eval: dict[str, Any],
        skill_sources: list[str],
        model: str,
        sample_id: str | None = None,
    ) -> EnvironmentResult:
        create_agent = globals()[_CREATE_DEEP_AGENT_COMPAT_NAME]
        invoke_agent = globals()[_INVOKE_DEEP_AGENT_COMPAT_NAME]
        agent = create_agent(
            backend_root=iter_dir,
            system_prompt=_SYSTEM_PROMPT,
            skill_sources=skill_sources,
            model=model,
            run_dir=self._runtime_run_dir(),
            timeout_s=self._runtime_timeout_s(),
            session_traces_enabled=self._runtime_session_traces_enabled(),
        )

        run_codegen_tool, result_ref = make_run_codegen_tool(iter_dir=iter_dir, device=device)
        trajectory: list[dict[str, Any]] = []
        try:
            del run_codegen_tool
            response = invoke_agent(
                agent,
                self._build_initial_prompt(query, iter_dir),
            )
            messages = response.get("messages", []) if isinstance(response, dict) else []

            if isinstance(messages, list):
                trajectory.extend(self._extract_trajectory_from_messages(messages))

            runner_result = result_ref.get("runner_result")
            if runner_result is None:
                code_path = iter_dir / "solution.py"
                if not code_path.is_file():
                    extracted = self._extract_solution_code_from_messages(messages)
                    if extracted is not None:
                        code_path.write_text(extracted, encoding="utf-8")
                runner_result = run_codegen(
                    code_path=code_path,
                    device=device,
                    timeout_s=_CODEGEN_TIMEOUT_S,
                )
                trajectory.append({"step": "fallback_execution"})

            attempt_count = self._count_codegen_attempts(messages)
            if attempt_count <= 0:
                attempt_count = 1

            if runner_result.error is not None:
                error_type = str(runner_result.error_type or "runtime_error")
                error_message = str(runner_result.error)
                traceback_text = runner_result.traceback
                trajectory.append(
                    {
                        "step": "attempt_error",
                        "attempt": attempt_count,
                        "error_type": error_type,
                        "error_class": self._derive_error_class(error_type, error_message),
                        "error": self._truncate(error_message, 500),
                        "traceback": self._truncate(traceback_text, 1000),
                        "execution_time_s": float(runner_result.execution_time_s),
                    }
                )
                result = self._agent_failure_result(
                    last_error_type=error_type,
                    last_error_message=error_message,
                    last_traceback=traceback_text,
                    execution_time_s=float(runner_result.execution_time_s),
                    attempt_count=attempt_count,
                    trajectory=trajectory,
                    error_class=self._derive_error_class(error_type, error_message),
                    code_hash=self._compute_code_hash(iter_dir / "solution.py"),
                )
                for entry in trajectory:
                    entry.setdefault("round_num", 0)
                self._write_retry_artifacts(
                    iter_dir=iter_dir,
                    combined_trajectory=trajectory,
                    results=[result],
                    best_round_idx=0,
                )
                return result

            eval_result = evaluate_gt_eval(
                gt_eval,
                runner_result.solver_results,
                compile_ok=True,
                solver_ok=True,
            )
            criteria_rows = eval_result.get("criteria", [])
            criteria_pass_fraction = self._criteria_pass_fraction(criteria_rows)
            criteria_violation_norm = self._criteria_violation_norm(criteria_rows)
            best_margin = self._best_margin(criteria_rows, eval_result.get("best_margin"))
            success_exec = bool(eval_result.get("success_exec", True))
            success_goal = bool(eval_result.get("success_goal", False))
            metrics = self._build_metrics(
                compile_ok=True,
                solver_ok=True,
                success_exec=success_exec,
                success_goal=success_goal,
                best_margin=best_margin,
                criteria_pass_fraction=criteria_pass_fraction,
                criteria_violation_norm=criteria_violation_norm,
                execution_time_s=float(runner_result.execution_time_s),
                attempt_count=attempt_count,
            )
            result = EnvironmentResult(
                feedback="compat loop completed.",
                ground_truth=None,
                metrics=metrics,
                trajectory=trajectory,
            )
            for entry in trajectory:
                entry.setdefault("round_num", 0)
            self._write_retry_artifacts(
                iter_dir=iter_dir,
                combined_trajectory=trajectory,
                results=[result],
                best_round_idx=0,
            )
            return result
        finally:
            await close_mce_deep_agent(agent)

    def _run_codegen_retry_loop(
        self,
        session: Any,
        query: str,
        device: str,
        iter_dir: Path,
        gt_eval: dict[str, Any],
        trajectory: list[dict[str, Any]],
        max_attempts: int = _DEFAULT_INNER_MAX_ATTEMPTS,
        initial_prompt: str | None = None,
        *,
        round_num: int,
    ) -> EnvironmentResult:
        prompt = initial_prompt or self._build_initial_prompt(query, iter_dir)
        for attempt in range(1, max_attempts + 1):
            if is_shutdown_requested():
                return self._build_shutdown_result(attempt - 1, trajectory)
            response = invoke_pi_session(session, prompt)
            if response.error:
                return self._agent_failure_result(
                    last_error_type="agent_invoke_error",
                    last_error_message=str(response.error),
                    last_traceback=None,
                    execution_time_s=0.0,
                    attempt_count=attempt - 1,
                    trajectory=trajectory,
                    error_class="execution_error",
                    code_hash=self._compute_code_hash(iter_dir / "solution.py"),
                )

            code_path = self._resolve_solution_path(iter_dir)
            if not code_path.is_file():
                trajectory.append(
                    {
                        "step": "attempt_error",
                        "attempt": attempt,
                        "error_type": "no_code_generated",
                        "error": "solution.py was not produced by the agent.",
                    }
                )
                if attempt >= max_attempts:
                    return self._agent_failure_result(
                        last_error_type="no_code_generated",
                        last_error_message="Agent produced no solution.py",
                        last_traceback=None,
                        execution_time_s=0.0,
                        attempt_count=attempt,
                        trajectory=trajectory,
                        error_class="tool_not_called",
                        code_hash=None,
                    )
                prompt = self._build_retry_prompt_for_missing_solution(
                    query=query,
                    iter_dir=iter_dir,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                continue

            runner_result = run_codegen(
                code_path=code_path,
                device=device,
                timeout_s=_CODEGEN_TIMEOUT_S,
            )
            execution_time_s = float(runner_result.execution_time_s)
            code_hash = self._compute_code_hash(code_path)

            if runner_result.error:
                error_type = str(runner_result.error_type or "runtime_error")
                error_message = str(runner_result.error)
                traceback_text = runner_result.traceback
                trajectory.append(
                    {
                        "step": "attempt_error",
                        "attempt": attempt,
                        "error_type": error_type,
                        "error_class": self._derive_error_class(error_type, error_message),
                        "error": self._truncate(error_message, 500),
                        "traceback": self._truncate(traceback_text, 1000),
                        "code_hash": code_hash,
                        "execution_time_s": execution_time_s,
                    }
                )
                if attempt >= max_attempts:
                    return self._agent_failure_result(
                        last_error_type=error_type,
                        last_error_message=error_message,
                        last_traceback=traceback_text,
                        execution_time_s=execution_time_s,
                        attempt_count=attempt,
                        trajectory=trajectory,
                        error_class=self._derive_error_class(error_type, error_message),
                        code_hash=code_hash,
                    )
                prompt = self._build_retry_prompt_for_execution_error(
                    query=query,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_type=error_type,
                    error_message=error_message,
                    traceback_text=traceback_text,
                )
                continue

            eval_result = evaluate_gt_eval(
                gt_eval,
                runner_result.solver_results,
                compile_ok=True,
                solver_ok=True,
            )
            criteria_rows = eval_result.get("criteria", [])
            criteria_pass_fraction = self._criteria_pass_fraction(criteria_rows)
            criteria_violation_norm = self._criteria_violation_norm(criteria_rows)
            best_margin = self._best_margin(criteria_rows, eval_result.get("best_margin"))
            success_exec = bool(eval_result.get("success_exec", True))
            success_goal = bool(eval_result.get("success_goal", False))

            trajectory.append(
                {
                    "step": "attempt_evaluated",
                    "attempt": attempt,
                    "execution_time_s": execution_time_s,
                    "code_path": str(code_path),
                    "code_hash": code_hash,
                    "success_exec": success_exec,
                    "success_goal": success_goal,
                    "best_margin": best_margin,
                    "criteria_pass_fraction": criteria_pass_fraction,
                    "criteria_violation_norm": criteria_violation_norm,
                    "criteria": criteria_rows,
                }
            )

            if success_goal or attempt >= max_attempts:
                metrics = self._build_metrics(
                    compile_ok=True,
                    solver_ok=True,
                    success_exec=success_exec,
                    success_goal=success_goal,
                    best_margin=best_margin,
                    criteria_pass_fraction=criteria_pass_fraction,
                    criteria_violation_norm=criteria_violation_norm,
                    execution_time_s=execution_time_s,
                    attempt_count=attempt,
                )
                feedback = (
                    "codegen run finished "
                    f"(attempt={attempt}, success_exec={success_exec}, success_goal={success_goal})."
                )
                return EnvironmentResult(
                    feedback=feedback,
                    ground_truth=None,
                    metrics=metrics,
                    trajectory=trajectory,
                )

            prompt = self._build_retry_prompt_for_score_failure(
                query=query,
                attempt=attempt,
                max_attempts=max_attempts,
                criteria_rows=criteria_rows,
                criteria_pass_fraction=criteria_pass_fraction,
                criteria_violation_norm=criteria_violation_norm,
                best_margin=best_margin,
            )

        return self._agent_failure_result(
            last_error_type="max_attempts_exceeded",
            last_error_message=f"Failed after {max_attempts} attempts.",
            last_traceback=None,
            execution_time_s=0.0,
            attempt_count=max_attempts,
            trajectory=trajectory,
        )

    def _setup_round_workspace(
        self,
        *,
        round_dir: Path,
        skill_content: str,
        skill_sources: list[str],
        context_dir: Path | None,
        gt_eval: dict[str, Any],
        model: str,
    ) -> tuple[Any, bool]:
        round_dir.mkdir(parents=True, exist_ok=True)
        self._write_skill_package(round_dir, skill_content=skill_content)
        context_available = self._copy_context_if_available(
            context_dir=context_dir,
            iter_dir=round_dir,
        )
        self._write_workspace_payloads(
            iter_dir=round_dir,
            gt_eval=gt_eval,
            skill_sources=skill_sources,
            context_available=context_available,
        )
        session = create_pi_session(
            iter_dir=round_dir,
            system_prompt=_SYSTEM_PROMPT,
            skills=[],
            model=model or "",
            skill_paths=_learning_context_skill_paths(round_dir),
            context_available=context_available,
            run_dir=self._runtime_run_dir(),
            timeout_s=self._runtime_timeout_s(),
            session_traces_enabled=self._runtime_session_traces_enabled(),
        )
        return session, context_available

    def _run_multistart_loop(
        self,
        *,
        query: str,
        device: str,
        iter_dir: Path,
        gt_eval: dict,
        skill_content: str,
        skill_sources: list[str],
        context_dir: Path | None,
        model: str,
        sample_id: str | None = None,
    ) -> EnvironmentResult:
        all_results: list[EnvironmentResult] = []
        combined_trajectory: list[dict[str, Any]] = []
        best_round_idx: int = 0
        best_cpf: float = -1.0
        best_margin: float = -float("inf")
        prev_feedback: str | None = None

        num_rounds = self._resolve_num_rounds()
        inner_max_attempts = self._resolve_inner_max_attempts()

        for round_num in range(num_rounds):
            if is_shutdown_requested():
                break
            round_dir = iter_dir / f"round_{round_num}"

            if round_num > 0 and prev_feedback is not None:
                best_solution = iter_dir / f"round_{best_round_idx}" / "solution.py"
                if best_solution.is_file():
                    round_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(best_solution, round_dir / "solution.py")

            session, _ = self._setup_round_workspace(
                round_dir=round_dir,
                skill_content=skill_content,
                skill_sources=skill_sources,
                context_dir=context_dir,
                gt_eval=gt_eval,
                model=model,
            )

            initial_prompt = self._build_initial_prompt(
                query,
                round_dir,
                prev_feedback=prev_feedback,
                round_num=round_num,
                total_rounds=num_rounds,
            )

            round_trajectory: list[dict[str, Any]] = []
            try:
                result = self._run_codegen_retry_loop(
                    session,
                    query,
                    device,
                    round_dir,
                    gt_eval,
                    round_trajectory,
                    max_attempts=inner_max_attempts,
                    initial_prompt=initial_prompt,
                    round_num=round_num,
                )
            finally:
                close_sync = getattr(session, "close_sync", None)
                if callable(close_sync):
                    close_sync()
                else:
                    asyncio.run(session.close())

            for entry in round_trajectory:
                entry["round_num"] = round_num
            combined_trajectory.extend(round_trajectory)
            all_results.append(result)

            prev_feedback = self._format_round_feedback(result)

            cpf = result.metrics.get("criteria_pass_fraction", 0.0)
            margin = result.metrics.get("best_margin", 0.0)
            if (cpf, margin) > (best_cpf, best_margin):
                best_round_idx = round_num
                best_cpf = cpf
                best_margin = margin

            if result.metrics.get("success_goal", 0.0) == 1.0:
                break

        self._write_retry_artifacts(
            iter_dir=iter_dir,
            combined_trajectory=combined_trajectory,
            results=all_results,
            best_round_idx=best_round_idx,
        )

        if not all_results:
            return self._build_shutdown_result(0, combined_trajectory)

        return self._select_best_round(all_results, combined_trajectory)

    def _write_retry_artifacts(
        self,
        *,
        iter_dir: Path,
        combined_trajectory: list[dict[str, Any]],
        results: list[EnvironmentResult],
        best_round_idx: int,
    ) -> None:
        retry_events_path = iter_dir / "retry_events.jsonl"
        retry_summary_path = iter_dir / "retry_summary.json"

        event_rows: list[dict[str, Any]] = []
        for entry in combined_trajectory:
            if not isinstance(entry, dict):
                continue
            event_rows.append(
                {
                    "round_num": int(entry.get("round_num", 0) or 0),
                    "attempt": int(entry.get("attempt", 0) or 0),
                    "step": str(entry.get("step", "unknown")),
                    "success_exec": bool(entry.get("success_exec", False)),
                    "success_goal": bool(entry.get("success_goal", False)),
                    "error_type": entry.get("error_type"),
                    "error_class": entry.get("error_class"),
                    "criteria_pass_fraction": self._try_float(entry.get("criteria_pass_fraction")),
                    "criteria_violation_norm": self._try_float(entry.get("criteria_violation_norm")),
                    "best_margin": self._try_float(entry.get("best_margin")),
                    "execution_time_s": self._try_float(entry.get("execution_time_s")),
                    "code_hash": entry.get("code_hash"),
                }
            )

        with retry_events_path.open("w", encoding="utf-8") as handle:
            for row in event_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        round_summaries: list[dict[str, Any]] = []
        for round_num, result in enumerate(results):
            round_summaries.append(
                {
                    "round_num": int(round_num),
                    "attempt_count": int(result.metrics.get("attempt_count", 0) or 0),
                    "success_exec": bool(result.metrics.get("success_exec", 0.0) == 1.0),
                    "success_goal": bool(result.metrics.get("success_goal", 0.0) == 1.0),
                    "criteria_pass_fraction": float(result.metrics.get("criteria_pass_fraction", 0.0) or 0.0),
                    "criteria_violation_norm": float(result.metrics.get("criteria_violation_norm", 0.0) or 0.0),
                    "best_margin": float(result.metrics.get("best_margin", 0.0) or 0.0),
                }
            )

        retry_summary_path.write_text(
            json.dumps(
                {
                    "round_count": len(results),
                    "total_attempt_count": sum(
                        int(result.metrics.get("attempt_count", 0) or 0)
                        for result in results
                    ),
                    "best_round_num": int(best_round_idx),
                    "event_count": len(event_rows),
                    "rounds": round_summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_workspace_payloads(
        self,
        *,
        iter_dir: Path,
        gt_eval: dict[str, Any],
        skill_sources: list[str],
        context_available: bool,
    ) -> None:
        write_agents_md(
            iter_dir=iter_dir,
            system_prompt=_SYSTEM_PROMPT,
            context_available=context_available,
        )
        (iter_dir / "gt_eval.json").write_text(
            json.dumps(gt_eval, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (iter_dir / "material_data.json").write_text(
            json.dumps(
                {
                    "default_source_path": str(default_source_path()),
                    "default_db_path": str(default_db_path()),
                    "source_exists": default_source_path().exists(),
                    "db_exists": default_db_path().exists(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _copy_context_if_available(self, *, context_dir: Path | None, iter_dir: Path) -> bool:
        if context_dir is None:
            return False
        src = Path(context_dir)
        if not src.exists() or not src.is_dir():
            return False

        files = [path for path in src.rglob("*") if path.is_file()]
        if not files:
            return False

        dst = iter_dir / "context"
        dst.mkdir(parents=True, exist_ok=True)
        for path in files:
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return True

    def _build_retry_prompt_for_missing_solution(
        self,
        *,
        query: str,
        iter_dir: Path,
        attempt: int,
        max_attempts: int,
    ) -> str:
        solution_path = iter_dir / "solution.py"
        return (
            "NO SOLUTION FILE PRODUCED\n"
            f"Previous failed inner attempt: {attempt}/{max_attempts}\n"
            f"Next inner retry attempt: {attempt + 1}/{max_attempts}\n"
            f"No `{solution_path}` was produced. You must write `{solution_path}` now with:\n"
            'def solve_inverse_design(*, device: str = "cpu") -> SolverResults\n\n'
            "Then stop.\n\n"
            f"Original query:\n{query}\n"
        )

    def _build_retry_prompt_for_execution_error(
        self,
        *,
        query: str,
        attempt: int,
        max_attempts: int,
        error_type: str,
        error_message: str,
        traceback_text: str | None,
    ) -> str:
        api_hints = _build_api_hints(error_message, traceback_text)
        return (
            "EXECUTION FAILED\n"
            f"Previous failed inner attempt: {attempt}/{max_attempts}\n"
            f"Next inner retry attempt: {attempt + 1}/{max_attempts}\n"
            f"Error type: {error_type}\n"
            f"Error: {error_message}\n"
            f"Traceback:\n{traceback_text or ''}\n"
            "Action required: edit `solution.py` to fix this exact issue, then return."
            f"{api_hints}\n\n"
            f"Original query:\n{query}\n"
        )

    def _build_retry_prompt_for_score_failure(
        self,
        *,
        query: str,
        attempt: int,
        max_attempts: int,
        criteria_rows: Any,
        criteria_pass_fraction: float,
        criteria_violation_norm: float,
        best_margin: float,
    ) -> str:
        details: list[str] = []
        if isinstance(criteria_rows, list):
            for idx, row in enumerate(criteria_rows):
                if not isinstance(row, dict):
                    continue
                details.append(
                    f"- criterion[{idx}]: passed={bool(row.get('passed', False))}, "
                    f"margin={row.get('margin')}, violation={row.get('violation')}, "
                    f"operation={row.get('operation')}, target={row.get('target')}, value={row.get('value')}"
                )

        detail_block = "\n".join(details) if details else "- no criterion details available"
        return (
            "CRITERIA NOT MET\n"
            f"Previous evaluated inner attempt: {attempt}/{max_attempts}\n"
            f"Next inner retry attempt: {attempt + 1}/{max_attempts}\n"
            f"criteria_pass_fraction={criteria_pass_fraction:.6f}\n"
            f"criteria_violation_norm={criteria_violation_norm:.6f}\n"
            f"best_margin={best_margin:.6f}\n"
            "Fix `solution.py` to improve objective satisfaction and return.\n"
            "Stay with the current optimization family when it is still improving or is a near miss.\n"
            "Only switch optimization family on stagnation or clear non-improvement across attempts,\n"
            "and update the top-of-file `# Strategy: <family> - <why>` note if you switch.\n"
            "Criterion details:\n"
            f"{detail_block}\n\n"
            f"Original query:\n{query}\n"
        )

    def _resolve_iter_dir(self, *, sample: Sample, log_dir: Path | None) -> Path:
        if log_dir is not None:
            return Path(log_dir) / f"iter_{sample.id}"
        suffix = random.randint(1000, 9999)
        return Path(".mce_codegen") / f"iter_{sample.id}_{suffix}"

    def _resolve_skill_content(self, *, sample: Sample, query: str, iter_dir: Path) -> str:
        candidates: list[str] = []
        skill_path = learning_context_skill_host_path(iter_dir)
        if skill_path.exists():
            candidates.append(skill_path.read_text(encoding="utf-8"))
        extras = sample.extras if isinstance(sample.extras, dict) else {}
        injected_guidance = extras.get("skill_guidance")
        if isinstance(injected_guidance, str) and injected_guidance.strip():
            candidates.append(injected_guidance)
        candidates.append(compose_preloaded_template_skill(query))

        for candidate in candidates:
            normalized = normalize_learning_context_skill_markdown(
                candidate,
                expected_name="learning-context",
            )
            skill_valid, _ = validate_skill_markdown(
                normalized,
                expected_name="learning-context",
            )
            if skill_valid:
                return normalized
        return compose_preloaded_template_skill(query)

    def _write_skill_package(self, iter_dir: Path, *, skill_content: str) -> Path:
        content = normalize_learning_context_skill_markdown(
            skill_content,
            expected_name="learning-context",
        )
        skill_dir = iter_dir / ".agents" / "skills" / "learning-context"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content, encoding="utf-8")
        materialize_progressive_reference_subtree(skill_dir / "reference")
        return skill_path

    def _resolve_solution_path(self, iter_dir: Path) -> Path:
        solution_path = iter_dir / "solution.py"
        if solution_path.is_file():
            return solution_path
        for candidate in sorted(iter_dir.glob("*.py")):
            return candidate
        return solution_path

    def _build_initial_prompt(
        self,
        query: str,
        iter_dir: Path,
        *,
        prev_feedback: str | None = None,
        round_num: int | None = None,
        total_rounds: int | None = None,
    ) -> str:
        solution_path = iter_dir / "solution.py"
        round_header = ""
        if round_num is not None and total_rounds is not None:
            round_header = f"Outer round: {round_num + 1}/{total_rounds}.\n"
            if round_num > 0:
                round_header += (
                    "The feedback below comes from the previous outer round; "
                    "treat it as round-level guidance distinct from inner retry attempts.\n"
                )
        if prev_feedback is None:
            return (
                f"Write `{solution_path}` with a single callable:\n"
                'def solve_inverse_design(*, device: str = "cpu") -> SolverResults\n\n'
                f"{round_header}"
                "If `context/strategy_summary.md` exists, read it first as cross-iteration memory: durable strategy guidance carried forward from prior iterations. Use it to anchor the high-level plan before applying batch-local context.\n"
                "Choose one bounded global optimization family before coding (for example: global screening, multistart,\n"
                "population-style search, or coarse-to-fine sweep) and put the first line of `solution.py` in the form\n"
                "`# Strategy: <family> - <why>`.\n"
                "Use TorchRDIT APIs to build a solver, call `solver.solve(source)`, and return the\n"
                "resulting `torchrdit.results.SolverResults` object. Do NOT define your own\n"
                "SolverResults class - the object must come from `solver.solve()`.\n"
                "Important: TorchRDIT built-in gradients are local optimization only. "
                "Use global exploration before local refinement when needed, "
                "and choose the method based on the loaded skill guidance.\n"
                "Keep the search deterministic and bounded for timeout safety.\n\n"
                f"Query:\n{query}\n"
            )
        else:
            return (
                f"A previous solution exists at `{solution_path}`.\n"
                f"{round_header}"
                "If `context/strategy_summary.md` exists, read it before revising the search plan. Treat it as cross-iteration memory that preserves durable strategy choices across retries and prior iterations.\n"
                "Review it and improve to better satisfy the design criteria.\n\n"
                f"[FEEDBACK FROM PREVIOUS ROUND]\n{prev_feedback}\n\n"
                "Keep the existing optimization family when it is still improving. Only switch families on stagnation\n"
                "or clear non-improvement, and keep the top-of-file `# Strategy: <family> - <why>` note aligned with the code.\n"
                "Use `edit_file` to patch specific issues rather than rewriting from scratch.\n"
                'Function signature: def solve_inverse_design(*, device: str = "cpu") -> SolverResults\n\n'
                f"Query:\n{query}\n"
            )

    def _resolve_codegen_device(self, sample: Sample) -> str:
        extras = sample.extras if isinstance(sample.extras, dict) else {}
        raw_device = extras.get("device")
        if isinstance(raw_device, str) and raw_device.strip():
            return raw_device.strip()
        return _DEFAULT_CODEGEN_DEVICE

    def _derive_error_class(self, error_type: str | None, error_message: str | None) -> str:
        label = str(error_type or "").lower()
        message = str(error_message or "").lower()
        if label in {"file_not_found", "no_code_generated"}:
            return "tool_not_called"
        if "path is not allowed" in message or "path must be relative" in message:
            return "tool_wrong_args"
        if label == "timeout" or "timeout" in message:
            return "timeout"
        return "execution_error"

    def _agent_failure_result(
        self,
        *,
        last_error_type: str | None,
        last_error_message: str | None,
        last_traceback: str | None,
        execution_time_s: float,
        attempt_count: int,
        trajectory: list[dict[str, Any]],
        error_class: str = "execution_error",
        code_hash: str | None = None,
    ) -> EnvironmentResult:
        error_type = str(last_error_type or "runtime_error")
        error_message = str(last_error_message or "Agent failed to produce runnable code.")
        traceback_tail = self._truncate(last_traceback, 1000)
        feedback = (
            f"agent failure ({error_class}/{error_type}): "
            f"{self._truncate(error_message, 500)}"
        )
        if traceback_tail:
            feedback = f"{feedback}\ntraceback:\n{traceback_tail}"

        trajectory.append(
            {
                "step": "agent_failure_result",
                "error_class": error_class,
                "error_type": error_type,
                "error": self._truncate(error_message, 500),
                "traceback": traceback_tail,
                "attempt": int(attempt_count),
                "execution_time_s": float(execution_time_s),
                "code_hash": code_hash,
            }
        )
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,
            metrics=self._build_metrics(
                compile_ok=False,
                solver_ok=False,
                success_exec=False,
                success_goal=False,
                best_margin=0.0,
                criteria_pass_fraction=0.0,
                criteria_violation_norm=0.0,
                execution_time_s=execution_time_s,
                attempt_count=attempt_count,
            ),
            trajectory=trajectory,
        )

    def _build_shutdown_result(
        self,
        attempt_count: int,
        trajectory: list[dict[str, Any]],
    ) -> EnvironmentResult:
        trajectory.append({"step": "shutdown_requested", "attempt": attempt_count})
        return EnvironmentResult(
            feedback="shutdown requested",
            ground_truth=None,
            metrics=self._build_metrics(
                compile_ok=False,
                solver_ok=False,
                success_exec=False,
                success_goal=False,
                best_margin=0.0,
                criteria_pass_fraction=0.0,
                criteria_violation_norm=0.0,
                execution_time_s=0.0,
                attempt_count=attempt_count,
            ),
            trajectory=trajectory,
        )

    def _build_metrics(
        self,
        *,
        compile_ok: bool,
        solver_ok: bool,
        success_exec: bool,
        success_goal: bool,
        best_margin: float,
        criteria_pass_fraction: float,
        criteria_violation_norm: float,
        execution_time_s: float,
        attempt_count: int,
    ) -> Dict[str, float]:
        pass_fraction = self._clamp01(criteria_pass_fraction)
        return {
            "compile_ok": 1.0 if compile_ok else 0.0,
            "solver_ok": 1.0 if solver_ok else 0.0,
            "success_exec": 1.0 if success_exec else 0.0,
            "success_goal": 1.0 if success_goal else 0.0,
            "best_margin": float(self._coerce_float(best_margin, default=0.0)),
            "criteria_pass_fraction": float(pass_fraction),
            "criteria_violation_norm": float(
                self._coerce_float(criteria_violation_norm, default=0.0)
            ),
            "execution_time_s": float(self._coerce_float(execution_time_s, default=0.0)),
            "attempt_count": float(max(0, int(attempt_count))),
        }

    def _criteria_pass_fraction(self, criteria_rows: Any) -> float:
        if not isinstance(criteria_rows, list) or not criteria_rows:
            return 0.0
        passed = 0
        total = 0
        for row in criteria_rows:
            if not isinstance(row, dict):
                continue
            total += 1
            if bool(row.get("passed", False)):
                passed += 1
        if total <= 0:
            return 0.0
        return passed / total

    def _criteria_violation_norm(self, criteria_rows: Any) -> float:
        if not isinstance(criteria_rows, list) or not criteria_rows:
            return 0.0
        values: list[float] = []
        for row in criteria_rows:
            if not isinstance(row, dict):
                continue
            values.append(max(0.0, self._coerce_float(row.get("violation"), default=0.0)))
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _best_margin(self, criteria_rows: Any, explicit_best_margin: Any) -> float:
        direct = self._try_float(explicit_best_margin)
        if direct is not None:
            return float(direct)
        if not isinstance(criteria_rows, list) or not criteria_rows:
            return 0.0
        margins: list[float] = []
        for row in criteria_rows:
            if not isinstance(row, dict):
                continue
            margin = self._try_float(row.get("margin"))
            if margin is not None:
                margins.append(float(margin))
        if not margins:
            return 0.0
        return min(margins)

    def _coerce_float(self, value: Any, *, default: float = 0.0) -> float:
        maybe_value = self._try_float(value)
        if maybe_value is None:
            return float(default)
        return float(maybe_value)

    def _try_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _truncate(self, value: Any, max_chars: int) -> str | None:
        if value is None:
            return None
        text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _compute_code_hash(self, code_path: Path) -> str | None:
        path = Path(code_path)
        if not path.is_file():
            return None
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def _select_best_round(
        self,
        results: list[EnvironmentResult],
        combined_trajectory: list[dict[str, Any]],
    ) -> EnvironmentResult:
        """Select the best round result by (cpf DESC, margin DESC, -index)."""
        best_idx = max(
            range(len(results)),
            key=lambda i: (
                results[i].metrics.get("criteria_pass_fraction", 0.0),
                results[i].metrics.get("best_margin", 0.0),
                -i,
            ),
        )
        best = results[best_idx]
        return EnvironmentResult(
            feedback=best.feedback,
            ground_truth=best.ground_truth,
            metrics={
                **best.metrics,
                "round_count": float(len(results)),
                "total_attempt_count": sum(
                    r.metrics.get("attempt_count", 0) for r in results
                ),
            },
            trajectory=combined_trajectory,
        )

    def _format_round_feedback(self, result: EnvironmentResult) -> str:
        """Format structured feedback from a round for the next round's prompt."""
        if not result.trajectory:
            return "Previous round produced no trajectory data."

        last = result.trajectory[-1]
        step = last.get("step", "")

        if step == "attempt_evaluated":
            criteria = last.get("criteria", [])
            if not criteria:
                cpf = result.metrics.get("criteria_pass_fraction", 0.0)
                return f"Previous round scored criteria_pass_fraction={cpf:.2f}."
            total = len(criteria)
            passed = sum(1 for c in criteria if c.get("passed"))
            parts: list[str] = [f"Score: {passed}/{total} criteria passed."]
            for i, row in enumerate(criteria):
                status = "PASS" if row.get("passed") else "FAIL"
                value = row.get("value", "?")
                target = row.get("target", "?")
                op = row.get("operation", "?")
                line = f"  [{status}] criterion[{i}]: value={value}, target={target}, op={op}"
                if row.get("tolerance") is not None:
                    line += f", tolerance={row['tolerance']}"
                parts.append(line)
            if passed < total:
                failing = [f"criterion[{i}]" for i, c in enumerate(criteria) if not c.get("passed")]
                parts.append(f"Focus on improving: {', '.join(failing)}")
            return "\n".join(parts)

        if step == "attempt_error":
            error_type = last.get("error_type", "unknown")
            if error_type == "no_code_generated":
                return "Previous attempt did not produce solution.py.\nYou must write it now."
            error_msg = last.get("error", "unknown error")
            parts_err: list[str] = [
                "Previous attempt encountered execution error:",
                f"Error type: {error_type}",
                f"Error: {error_msg}",
            ]
            tb = last.get("traceback")
            if tb:
                parts_err.append(f"Traceback:\n{tb}")
            parts_err.append("Fix the error in solution.py using edit_file.")
            return "\n".join(parts_err)

        if step == "agent_failure_result":
            error_type = last.get("error_type", "unknown")
            error_msg = last.get("error", "unknown error")
            return f"Previous attempt failed ({error_type}): {error_msg}"

        return "Previous round completed with unknown status."

    def format_result_for_training(self, item: Dict[str, Any]) -> Dict[str, Any]:
        sample = item.get("sample", {})
        evaluation = item.get("evaluation", {})
        metrics = evaluation.get("metrics", {})
        trajectory = evaluation.get("trajectory", [])

        execution_time_s = 0.0
        code_error_type: str | None = None
        code_error_message: str | None = None
        code_error_class: str | None = None
        code_traceback_tail: str | None = None
        attempt_count = 0
        attempt_error_types: list[str] = []
        code_hash: str | None = None
        for step in (trajectory if isinstance(trajectory, list) else []):
            if not isinstance(step, dict):
                continue
            if "execution_time_s" in step:
                execution_time_s = float(step["execution_time_s"])
            if "error_type" in step:
                code_error_type = str(step["error_type"])
                attempt_error_types.append(code_error_type)
            if "error" in step and step.get("error") is not None:
                code_error_message = self._truncate(step.get("error"), 300)
            if "error_class" in step and step.get("error_class") is not None:
                code_error_class = str(step.get("error_class"))
            if "traceback" in step and step.get("traceback") is not None:
                code_traceback_tail = self._truncate(step.get("traceback"), 500)
            if "code_hash" in step:
                raw_hash = step.get("code_hash")
                if isinstance(raw_hash, str) and raw_hash:
                    code_hash = raw_hash
            if "attempt" in step:
                attempt_count = max(attempt_count, int(step["attempt"]))

        if code_error_type and code_error_class:
            error_reason = f"{code_error_type}:{code_error_class}"
        elif code_error_type:
            error_reason = code_error_type
        else:
            error_reason = None

        return {
            "id": sample.get("id"),
            "question": sample.get("question"),
            "gt_eval": sample.get("gt_eval"),
            "compile_ok": bool(metrics.get("compile_ok", 0.0) == 1.0),
            "solver_ok": bool(metrics.get("solver_ok", 0.0) == 1.0),
            "success_exec": bool(metrics.get("success_exec", 0.0) == 1.0),
            "success_goal": bool(metrics.get("success_goal", 0.0) == 1.0),
            "best_margin": metrics.get("best_margin"),
            "criteria_pass_fraction": float(
                metrics.get("criteria_pass_fraction", 0.0) or 0.0
            ),
            "criteria_violation_norm": float(
                metrics.get("criteria_violation_norm", 0.0) or 0.0
            ),
            "execution_time_s": execution_time_s,
            "code_error_type": code_error_type,
            "code_error_message": code_error_message,
            "code_error_class": code_error_class,
            "code_traceback_tail": code_traceback_tail,
            "attempt_error_types": attempt_error_types,
            "error_reason": error_reason,
            "attempt_count": attempt_count,
            "round_count": int(metrics.get("round_count", 1.0)),
            "total_attempt_count": int(metrics.get("total_attempt_count", attempt_count)),
            "code_hash": code_hash,
            "fallback_attempted": False,
            "fallback_solution_present_before_recovery": False,
            "fallback_solution_present_after_recovery": False,
            "fallback_code_extracted_from_messages": False,
            "fallback_codegen_executed": False,
            "fallback_execution_success": False,
            "fallback_no_code_generated": False,
            "mandatory_guard_retrigger_count": 0,
            "mandatory_guard_satisfied": False,
            "mandatory_guard_exhausted": False,
        }

    def _extract_trajectory_from_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        attempt = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    if call.get("name") != "write_file":
                        continue
                    args = call.get("args")
                    path = ""
                    if isinstance(args, dict):
                        path = str(args.get("path", ""))
                    if path.endswith("solution.py"):
                        trajectory.append({"step": "write_solution"})

            if msg.get("name") != "run_codegen":
                continue

            attempt += 1
            content = str(msg.get("content", ""))
            if "EXECUTION SUCCESS" in content:
                trajectory.append({"step": "attempt_tool_success", "attempt": attempt})
                continue
            if "EXECUTION FAILED" not in content:
                continue

            step: dict[str, Any] = {"step": "attempt_error", "attempt": attempt}
            error_info = self._extract_error_info_from_messages([msg])
            if error_info.get("error_type") is not None:
                step["error_type"] = error_info["error_type"]
            if error_info.get("error") is not None:
                step["error"] = error_info["error"]
            if error_info.get("traceback") is not None:
                step["traceback"] = error_info["traceback"]
            trajectory.append(step)
        return trajectory

    def _count_codegen_attempts(self, messages: list[dict[str, Any]]) -> int:
        return sum(1 for msg in messages if isinstance(msg, dict) and msg.get("name") == "run_codegen")

    def _extract_error_info_from_messages(self, messages: list[dict[str, Any]]) -> dict[str, str | None]:
        info: dict[str, str | None] = {"error_type": None, "error": None, "traceback": None}
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("name") != "run_codegen":
                continue
            content = str(msg.get("content", ""))
            if "EXECUTION FAILED" not in content:
                continue

            lines = content.splitlines()
            traceback_idx = None
            for idx, line in enumerate(lines):
                if line.startswith("Error type:"):
                    info["error_type"] = line.split(":", 1)[1].strip() or None
                elif line.startswith("Error:"):
                    info["error"] = line.split(":", 1)[1].strip() or None
                elif line.startswith("Traceback:"):
                    traceback_idx = idx
            if traceback_idx is not None:
                info["traceback"] = "\n".join(lines[traceback_idx + 1 :]).strip() or None
            return info
        return info

    def _extract_solution_code_from_messages(self, messages: Any) -> str | None:
        if not isinstance(messages, list):
            return None
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            text = content.strip()
            if text.startswith("```python") and "```" in text[9:]:
                inner = text[len("```python") :]
                end = inner.rfind("```")
                if end >= 0:
                    code = inner[:end].strip()
                    if code:
                        return code + "\n"
            if "def solve_inverse_design" in text:
                return text + ("\n" if not text.endswith("\n") else "")
        return None

    def load_samples(
        self,
        path: str,
        limit: int | None = 10,
        random_sample: bool = False,
        shuffle: bool = False,
    ) -> List[Sample]:
        samples: list[Sample] = []
        with open(path, encoding="utf-8") as handle:
            for line_no, row in enumerate(handle, 1):
                if not row.strip():
                    continue
                data = json.loads(row)
                query = data.get("query") or data.get("question")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"Sample at line {line_no} missing query")
                gt_eval = data.get("gt_eval")
                if not isinstance(gt_eval, dict):
                    raise ValueError(f"Sample at line {line_no} missing gt_eval")
                samples.append(
                    Sample(
                        id=len(samples),
                        question=query.strip(),
                        extras={"gt_eval": dict(gt_eval)},
                    )
                )
                if not random_sample and limit and len(samples) >= limit:
                    break

        if random_sample and limit and len(samples) > limit:
            samples = random.sample(samples, limit)
        if shuffle:
            random.shuffle(samples)
        for i, sample in enumerate(samples):
            sample.id = i
        return samples


__all__ = [
    "MetaopticsInverseDesignEnvironment",
    "RunnerResult",
    "_USE_RALPH_LOOP",
    "codegen_runner",
    "make_run_codegen_tool",
    "close_mce_deep_agent",
]
