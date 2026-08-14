"""Deterministic code-gen runner for agent-written inverse-design functions.

Imports and executes agent-written Python modules containing a
``solve_inverse_design(*, device: str = "cpu") -> SolverResults`` function.

The runner is deterministic and does NOT invoke any LLM.  It is called by the
environment's ``aevaluate()`` after the agent writes its code via the filesystem
backend.

Design decisions (from task-176):
- D9:  In-process importlib with threading-based timeout (no subprocess isolation).
- D10: Agent writes ``.py`` via filesystem backend; runner reads from backend root.
- D14: Function returns raw ``SolverResults`` object.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_FUNCTION_NAME = "solve_inverse_design"
_SMOKE_SKIP_SOLVE_ENV_KEY = "EVO_METAOPTICS_SMOKE_SKIP_SOLVE"
_SMOKE_SKIP_SOLVE_LOCK = threading.Lock()
_THREAD_LIMIT_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _apply_thread_limit_env_defaults() -> None:
    for key, value in _THREAD_LIMIT_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _get_active_codegen_logger() -> logging.Logger | None:
    """Find an already-configured logger for codegen events.

    During MCE smoke runs, 'mce_main' logger is configured by setup_logger()
    with both console and file handlers. During standalone eval, 'eval' logger
    is configured. Returns None if no active logger found.
    """
    for name in ("mce_main", "eval"):
        logger = logging.getLogger(name)
        if logger.level != logging.NOTSET and logger.handlers:
            return logger
    return None


@dataclass
class RunnerResult:
    """Result of running an agent-written ``solve_inverse_design`` function.

    Attributes:
        solver_results: Raw ``SolverResults`` object on success, ``None`` on failure.
        error: Human-readable error message, ``None`` on success.
        error_type: Structured error category for retry feedback.
            One of: ``syntax_error``, ``import_error``, ``runtime_error``,
            ``timeout``, ``invalid_output``, ``missing_function``, ``file_not_found``.
        execution_time_s: Wall-clock time for the entire run (import + execute).
        traceback: Full traceback string on failure, ``None`` on success.
    """

    solver_results: Any | None
    error: str | None
    error_type: str | None
    execution_time_s: float
    traceback: str | None



def run_codegen(
    *,
    code_path: Path,
    device: str,
    timeout_s: float = 600.0,
) -> RunnerResult:
    """Import and execute an agent-written module, returning structured results.

    Args:
        code_path: Path to the ``.py`` file containing ``solve_inverse_design``.
        device: Compute device keyword argument to pass to the function.
        timeout_s: Maximum wall-clock seconds for import + execution.

    Returns:
        A ``RunnerResult`` with either ``solver_results`` or error details.
    """
    t0 = time.monotonic()
    _apply_thread_limit_env_defaults()

    # --- Check file exists ---
    code_path = Path(code_path)
    if not code_path.is_file():
        return RunnerResult(
            solver_results=None,
            error=f"Code file not found: {code_path}",
            error_type="file_not_found",
            execution_time_s=time.monotonic() - t0,
            traceback=None,
        )

    smoke_skip_device = _resolve_smoke_skip_solve_device()
    # --- Import the module ---
    try:
        module = _import_module(code_path)
    except SyntaxError as exc:
        return RunnerResult(
            solver_results=None,
            error=f"Syntax error in agent code: {exc}",
            error_type="syntax_error",
            execution_time_s=time.monotonic() - t0,
            traceback=traceback.format_exc(),
        )
    except ImportError as exc:
        return RunnerResult(
            solver_results=None,
            error=f"Import error in agent code: {exc}",
            error_type="import_error",
            execution_time_s=time.monotonic() - t0,
            traceback=traceback.format_exc(),
        )
    except Exception as exc:
        return RunnerResult(
            solver_results=None,
            error=f"Error loading agent code: {exc}",
            error_type="import_error",
            execution_time_s=time.monotonic() - t0,
            traceback=traceback.format_exc(),
        )

    # --- Check function exists ---
    func = getattr(module, _FUNCTION_NAME, None)
    if func is None or not callable(func):
        return RunnerResult(
            solver_results=None,
            error=f"Module does not define callable '{_FUNCTION_NAME}'.",
            error_type="missing_function",
            execution_time_s=time.monotonic() - t0,
            traceback=None,
        )

    # --- Execute with timeout ---
    try:
        try:
            result = _execute_with_timeout(
                func,
                device,
                timeout_s=timeout_s,
                smoke_skip_device=smoke_skip_device,
            )
        except FuturesTimeoutError:
            return RunnerResult(
                solver_results=None,
                error=f"Execution timed out after {timeout_s}s.",
                error_type="timeout",
                execution_time_s=time.monotonic() - t0,
                traceback=None,
            )
        except Exception as exc:
            return RunnerResult(
                solver_results=None,
                error=f"Runtime error: {exc}",
                error_type="runtime_error",
                execution_time_s=time.monotonic() - t0,
                traceback=traceback.format_exc(),
            )

        # --- Validate return type ---
        if not _is_solver_results(result):
            return RunnerResult(
                solver_results=None,
                error=(
                    f"Function returned {type(result).__module__}.{type(result).__qualname__} instead of "
                    f"torchrdit.results.SolverResults. "
                    f"The SolverResults object is returned by solver.solve(source). "
                    f"Do NOT define your own SolverResults class. "
                    f"Call solver.solve(source) and return its result directly."
                ),
                error_type="invalid_output",
                execution_time_s=time.monotonic() - t0,
                traceback=None,
            )

        device_str = _get_solver_results_device(result)
        msg = f"run_codegen: requested_device={device}, solver_results_device={device_str}"

        active_logger = _get_active_codegen_logger()
        if active_logger is not None:
            active_logger.info(msg)
        else:
            logging.getLogger(__name__).info(msg)

        return RunnerResult(
            solver_results=result,
            error=None,
            error_type=None,
            execution_time_s=time.monotonic() - t0,
            traceback=None,
        )
    finally:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_module(code_path: Path) -> Any:
    """Dynamically import a Python module from a file path.

    Uses a unique module name to avoid cache collisions when the same file is
    re-imported with different content (e.g., agent retry iterations).
    """
    # Use a unique module name to avoid importlib caching stale modules.
    module_name = f"_codegen_{id(code_path)}_{int(time.monotonic() * 1e6)}"

    spec = importlib.util.spec_from_file_location(module_name, str(code_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {code_path}")

    module = importlib.util.module_from_spec(spec)
    # Temporarily add to sys.modules so relative imports within the module work.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Clean up on failure.
        sys.modules.pop(module_name, None)
        raise
    finally:
        # Always clean up — we don't want these ephemeral modules lingering.
        sys.modules.pop(module_name, None)

    return module


def _execute_with_timeout(
    func: Any,
    device: str,
    *,
    timeout_s: float,
    smoke_skip_device: str | None = None,
) -> Any:
    """Execute ``func(device=device)`` with a wall-clock timeout via ThreadPoolExecutor.

    Uses ``shutdown(wait=False, cancel_futures=True)`` to avoid blocking on the
    worker thread after timeout.
    """
    def _run_guarded() -> Any:
        with _maybe_smoke_skip_solve(smoke_skip_device):
            return func(device=device)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_run_guarded)
        return future.result(timeout=timeout_s)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _get_solver_results_device(result: Any) -> str:
    """Extract device string from solver results for logging."""
    try:
        if hasattr(result, "transmission") and hasattr(result.transmission, "device"):
            return str(result.transmission.device)
    except Exception:
        pass
    return "unknown"


def _resolve_smoke_skip_solve_device() -> str | None:
    if os.environ.get(_SMOKE_SKIP_SOLVE_ENV_KEY) != "1":
        return None
    return "cpu"


@contextmanager
def _maybe_smoke_skip_solve(device_str: str | None) -> Any:
    if device_str is None:
        yield
        return

    classes_to_patch = _torchrdit_solver_classes()
    if not classes_to_patch:
        yield
        return

    originals: list[tuple[type[Any], Any]] = []
    with _SMOKE_SKIP_SOLVE_LOCK:
        try:
            for cls in classes_to_patch:
                originals.append((cls, getattr(cls, "solve")))
                cls.solve = _build_smoke_skip_solve(device_str)  # type: ignore[assignment]
            yield
        finally:
            for cls, original in originals:
                cls.solve = original  # type: ignore[assignment]


def _build_smoke_skip_solve(device_str: str) -> Any:
    def _smoke_skip_solve(self: Any, source: Any, **kwargs: Any) -> Any:
        del self, source, kwargs
        return _build_synthetic_solver_results(device_str)

    return _smoke_skip_solve


def _build_synthetic_solver_results(device_str: str) -> Any:
    import torch
    from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

    device = torch.device(device_str)
    scalar = torch.float32
    field_shape = (1, 1, 1)
    vector = torch.zeros(1, device=device, dtype=scalar)
    field_tensor = torch.zeros(field_shape, device=device, dtype=scalar)
    scattering_tensor = torch.zeros((1, 1, 1), device=device, dtype=scalar)

    field = FieldComponents(
        x=field_tensor.clone(),
        y=field_tensor.clone(),
        z=field_tensor.clone(),
        mag_x=field_tensor.clone(),
        mag_y=field_tensor.clone(),
        mag_z=field_tensor.clone(),
    )
    structure_matrix = ScatteringMatrix(
        S11=scattering_tensor.clone(),
        S12=scattering_tensor.clone(),
        S21=scattering_tensor.clone(),
        S22=scattering_tensor.clone(),
    )
    wave_vectors = WaveVectors(
        kx=vector.clone(),
        ky=vector.clone(),
        kinc=vector.clone(),
        kzref=vector.clone(),
        kztrn=vector.clone(),
    )
    return SolverResults(
        reflection=torch.zeros(1, device=device, dtype=scalar),
        transmission=torch.ones(1, device=device, dtype=scalar),
        reflection_diffraction=torch.zeros(field_shape, device=device, dtype=scalar),
        transmission_diffraction=torch.zeros(field_shape, device=device, dtype=scalar),
        reflection_field=field,
        transmission_field=field,
        structure_matrix=structure_matrix,
        wave_vectors=wave_vectors,
        raw_data={"smoke_skip_solve": True},
        solver_config={"device": device_str, "smoke_skip_solve": True},
    )


def _is_solver_results(obj: Any) -> bool:
    if obj is None:
        return False
    solver_results_type = _solver_results_type()
    if solver_results_type is None:
        return False
    return isinstance(obj, solver_results_type)


@lru_cache(maxsize=1)
def _torchrdit_solver_classes() -> tuple[type[Any], ...]:
    try:
        from torchrdit.solver import FourierBaseSolver, RCWASolver, RDITSolver
    except Exception:
        return ()
    return (FourierBaseSolver, RCWASolver, RDITSolver)


@lru_cache(maxsize=1)
def _solver_results_type() -> type[Any] | None:
    try:
        from torchrdit.results import SolverResults
    except Exception:
        return None
    return SolverResults
