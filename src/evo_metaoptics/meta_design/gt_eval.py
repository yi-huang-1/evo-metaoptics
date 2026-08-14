"""Canonical `gt_eval` specification and evaluator for `meta_design`.

This docstring is the authoritative dataset contract for `gt_eval` in this repo.

Schema (v8 — metric-based, with v7 legacy fallback)::

    # v8 declarative metric criteria (preferred)
    {
      "wavelength_um": [1.55],
      "criteria": [
        {"metric": "total_transmission", "params": {"wavelength_index": 0}, "operation": ">=", "target": 0.8},
        {"metric": "zero_order_transmission_phase_deg", "params": {"component": "x", "wavelength_index": 0},
         "operation": "close_to", "target": 170.0, "tolerance": 8.5}
      ]
    }

    # v7 legacy eval-based criteria (still supported)
    {
      "wavelength_um": [1.55],
      "criteria": [
        {"expr": "float(r.transmission[0])", "operation": ">=", "target": 0.8}
      ]
    }

Top-level keys:
- ``wavelength_um`` (required, non-empty list[number])
- ``criteria`` (required, non-empty list of criterion rows)

Each criterion row must have EITHER ``metric`` (v8) or ``expr`` (v7), not both.

Criterion operations: ``>``, ``<``, ``>=``, ``<=``, ``close_to``.

``close_to`` requires a ``tolerance`` key (positive finite float). It computes the
circular distance (mod 360°) between the evaluated value and the target, and
passes when the distance is within the tolerance. The ``tolerance`` key is
silently ignored for other operations.

v7 Eval context:
- ``r``: the raw ``SolverResults`` object from ``solver.solve()``
- ``torch``: the ``torch`` module
- ``math``: the ``math`` module
- ``cmath``: the ``cmath`` module (complex number math)
- ``np``: the ``numpy`` module

Execution semantics:
- ``success_exec = compile_ok and solver_ok``
- ``success_goal = success_exec and all(criteria_i.passed)``
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Any, Mapping, Sequence

import torch
import numpy as np

_ALLOWED_CRITERION_OPERATIONS = {">", "<", ">=", "<=", "close_to"}

# Eval context: restricted builtins + torch/math/cmath/numpy + the SolverResults object.
_SAFE_BUILTINS = {
    "abs": abs, "max": max, "min": min, "round": round, "sum": sum,
    "len": len, "float": float, "int": int, "bool": bool, "complex": complex,
    "range": range, "enumerate": enumerate, "zip": zip,
    "True": True, "False": False, "None": None,
    "__import__": __import__,
}
_EVAL_GLOBALS = {"__builtins__": _SAFE_BUILTINS, "torch": torch, "math": math, "cmath": cmath, "np": np}


def evaluate_gt_eval(
    gt_eval: Mapping[str, Any],
    solver_results: Any | None,
    *,
    compile_ok: bool = True,
    solver_ok: bool = True,
) -> dict[str, Any]:
    """Evaluate canonical gt_eval criteria contract against solver results.

    This is a **runtime** evaluator invoked after solver execution.
    Returns scoring and criteria pass/fail results.

    Args:
        gt_eval: Validated gt_eval payload (schema v7).
        solver_results: Raw ``SolverResults`` object from ``solver.solve()``.
            May be ``None`` when ``compile_ok`` or ``solver_ok`` is ``False``.
        compile_ok: Whether the agent code compiled successfully.
        solver_ok: Whether the solver ran without error.

    Return schema::

        compile_ok, solver_ok, success_exec, criteria,
        meets_gt_criteria, success_goal
    """
    normalized = validate_gt_eval(gt_eval)
    success_exec = bool(compile_ok and solver_ok)

    if not success_exec:
        return {
            "compile_ok": bool(compile_ok),
            "solver_ok": bool(solver_ok),
            "success_exec": False,
            "criteria": [],
            "meets_gt_criteria": False,
            "success_goal": False,
        }

    if solver_results is None:
        raise ValueError("solver_results must not be None when compile_ok and solver_ok are true.")
    if not _is_solver_results_instance(solver_results):
        raise ValueError(
            "solver_results must be an instance of torchrdit.results.SolverResults when compile_ok and solver_ok are true."
        )

    criteria_rows = _evaluate_criteria_rows(
        criteria=normalized["criteria"],
        solver_results=solver_results,
    )
    meets_criteria = bool(all(bool(row.get("passed", False)) for row in criteria_rows))
    return {
        "compile_ok": True,
        "solver_ok": True,
        "success_exec": True,
        "criteria": criteria_rows,
        "meets_gt_criteria": meets_criteria,
        "success_goal": meets_criteria,
    }


def summarize_normalized_margins(
    criteria_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Summarize normalized signed margins across evaluated criteria rows."""
    normalized_margins: list[float] = []
    for row in criteria_rows:
        normalized = _normalized_signed_margin(row)
        if isinstance(normalized, float):
            normalized_margins.append(normalized)

    if not normalized_margins:
        return {
            "best_margin": None,
            "criteria_violation_norm": 0.0,
        }

    violation_terms = [max(0.0, -value) for value in normalized_margins]
    return {
        "best_margin": float(min(normalized_margins)),
        "criteria_violation_norm": float(sum(violation_terms) / len(violation_terms)),
    }


def validate_gt_eval(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize gt_eval payload (schema v7 and v8)."""
    if not isinstance(value, Mapping):
        raise ValueError("gt_eval must be an object.")
    if "budget" in value:
        raise ValueError("gt_eval.budget is no longer supported.")
    if "loss_function" in value:
        raise ValueError("gt_eval.loss_function is no longer supported (removed in schema v6).")
    if "loss_expr" in value:
        raise ValueError(
            "gt_eval.loss_expr is no longer supported (removed in schema v6). "
            "Criteria-derived margins replace loss_expr for ranking and selection."
        )

    wavelength_um = _validate_number_list(value.get("wavelength_um"), "gt_eval.wavelength_um")
    criteria = _validate_criteria_rows(value.get("criteria"))
    return {
        "wavelength_um": wavelength_um,
        "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_criteria_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("gt_eval.criteria must be a non-empty list.")
    if len(value) <= 0:
        raise ValueError("gt_eval.criteria must be a non-empty list.")

    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(value):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"gt_eval.criteria[{index}] must be an object.")
        operation_path = f"gt_eval.criteria[{index}].operation"
        target_path = f"gt_eval.criteria[{index}].target"

        has_metric = "metric" in raw_row
        has_expr = "expr" in raw_row

        # Reject old schema key.
        if "metrics" in raw_row and not has_expr and not has_metric:
            raise ValueError(
                f"gt_eval.criteria[{index}].metrics is no longer supported. "
                f"Use 'metric' with params, or 'expr' with a Python expression."
            )

        # Reject ambiguous criteria with both keys.
        if has_metric and has_expr:
            raise ValueError(
                f"gt_eval.criteria[{index}] has both 'metric' and 'expr'. "
                f"Use one or the other, not both."
            )

        # Must have at least one.
        if not has_metric and not has_expr:
            raise ValueError(
                f"gt_eval.criteria[{index}] must have either 'metric' or 'expr' key."
            )

        # Validate operation (shared for both paths).
        operation = str(raw_row.get("operation") or "").strip()
        if operation not in _ALLOWED_CRITERION_OPERATIONS:
            raise ValueError(
                f"{operation_path} must be one of: >, <, >=, <=, close_to."
            )

        # Validate target (shared for both paths).
        target = _as_float(raw_row.get("target"), target_path)
        if not math.isfinite(target):
            raise ValueError(f"{target_path} must be finite.")

        if has_metric:
            # v8 metric-based path.
            metric_name = raw_row["metric"]
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ValueError(f"gt_eval.criteria[{index}].metric must be a non-empty string.")
            params = raw_row.get("params", {})
            if not isinstance(params, Mapping):
                raise ValueError(f"gt_eval.criteria[{index}].params must be an object.")
            row_dict: dict[str, Any] = {
                "metric": metric_name.strip(),
                "params": dict(params),
                "operation": operation,
                "target": float(target),
            }
        else:
            # v7 legacy expr-based path.
            expr_path = f"gt_eval.criteria[{index}].expr"
            expr_str = raw_row.get("expr")
            if not isinstance(expr_str, str) or not expr_str.strip():
                raise ValueError(f"{expr_path} must be a non-empty string.")
            row_dict = {
                "expr": expr_str.strip(),
                "operation": operation,
                "target": float(target),
            }

        # close_to requires a positive finite tolerance.
        if operation == "close_to":
            tolerance_path = f"gt_eval.criteria[{index}].tolerance"
            raw_tolerance = raw_row.get("tolerance")
            if raw_tolerance is None:
                raise ValueError(f"{tolerance_path} is required for close_to operation.")
            tolerance = _as_float(raw_tolerance, tolerance_path)
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(f"{tolerance_path} must be a positive finite number.")
            row_dict["tolerance"] = float(tolerance)

        rows.append(row_dict)
    return rows


# ---------------------------------------------------------------------------
# Eval-based evaluation (replaces torchrdit.expr)
# ---------------------------------------------------------------------------

def _make_eval_context(solver_results: Any) -> dict[str, Any]:
    """Build the restricted eval context dict."""
    return {**_EVAL_GLOBALS, "r": solver_results}


def _evaluate_criteria_rows(
    *,
    criteria: Sequence[Mapping[str, Any]],
    solver_results: Any,
) -> list[dict[str, Any]]:
    ctx = _make_eval_context(solver_results)
    rows: list[dict[str, Any]] = []
    for index, criterion in enumerate(criteria):
        operation = str(criterion["operation"])
        target = float(criterion["target"])
        tolerance: float | None = None
        if operation == "close_to":
            tolerance = float(criterion["tolerance"])

        if "metric" in criterion:
            # v8 metric-based dispatch
            row_dict = _evaluate_metric_criterion(
                index=index,
                criterion=criterion,
                solver_results=solver_results,
                operation=operation,
                target=target,
                tolerance=tolerance,
            )
        else:
            # v7 legacy eval-based dispatch
            row_dict = _evaluate_expr_criterion(
                index=index,
                criterion=criterion,
                ctx=ctx,
                operation=operation,
                target=target,
                tolerance=tolerance,
            )

        rows.append(row_dict)
    return rows


def _evaluate_expr_criterion(
    *,
    index: int,
    criterion: Mapping[str, Any],
    ctx: dict[str, Any],
    operation: str,
    target: float,
    tolerance: float | None,
) -> dict[str, Any]:
    """Evaluate a v7 legacy expr-based criterion via eval()."""
    expr_str = str(criterion["expr"])
    raw_value = eval(expr_str, ctx)  # noqa: S307
    value = _coerce_scalar_float(
        value=raw_value,
        label=f"gt_eval.criteria[{index}].value",
    )
    passed, margin = _evaluate_criterion(
        value=value, operation=operation, target=target, tolerance=tolerance,
    )
    violation = 0.0 if passed else max(0.0, -margin)
    row_dict: dict[str, Any] = {
        "expr": expr_str,
        "operation": operation,
        "target": float(target),
        "value": float(value),
        "passed": bool(passed),
        "margin": float(margin),
        "violation": float(violation),
    }
    if tolerance is not None:
        row_dict["tolerance"] = float(tolerance)
    return row_dict


def _evaluate_metric_criterion(
    *,
    index: int,
    criterion: Mapping[str, Any],
    solver_results: Any,
    operation: str,
    target: float,
    tolerance: float | None,
) -> dict[str, Any]:
    """Evaluate a v8 metric-based criterion via the metric registry."""
    from evo_metaoptics.meta_design.gt_eval_metrics import MetricError, extract_metric

    metric_name = str(criterion["metric"])
    params = dict(criterion.get("params", {}))

    try:
        value = extract_metric(solver_results, metric_name, params)
    except (KeyError, MetricError, TypeError) as exc:
        # Metric extraction failed — return a failed criterion with error info
        # rather than crashing the entire evaluation.
        return {
            "metric": metric_name,
            "params": params,
            "operation": operation,
            "target": float(target),
            "value": None,
            "passed": False,
            "margin": float("-inf"),
            "violation": float("inf"),
            "error": str(exc),
            **(({"tolerance": float(tolerance)} if tolerance is not None else {})),
        }

    passed, margin = _evaluate_criterion(
        value=value, operation=operation, target=target, tolerance=tolerance,
    )
    violation = 0.0 if passed else max(0.0, -margin)
    row_dict: dict[str, Any] = {
        "metric": metric_name,
        "params": params,
        "operation": operation,
        "target": float(target),
        "value": float(value),
        "passed": bool(passed),
        "margin": float(margin),
        "violation": float(violation),
    }
    if tolerance is not None:
        row_dict["tolerance"] = float(tolerance)
    return row_dict


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criterion(
    *, value: float, operation: str, target: float, tolerance: float | None = None,
) -> tuple[bool, float]:
    if operation == ">=":
        margin = float(value - target)
        return bool(value >= target), margin
    if operation == ">":
        margin = float(value - target)
        return bool(value > target), margin
    if operation == "<=":
        margin = float(target - value)
        return bool(value <= target), margin
    if operation == "<":
        margin = float(target - value)
        return bool(value < target), margin
    if operation == "close_to":
        if tolerance is None or tolerance <= 0:
            raise ValueError("close_to requires positive tolerance")
        diff = abs(value - target) % 360.0
        distance = min(diff, 360.0 - diff)
        margin = float(tolerance - distance)
        return bool(margin >= 0.0), margin
    raise ValueError(f"Unsupported criterion operation: {operation!r}")


# ---------------------------------------------------------------------------
# Scalar coercion (unchanged — handles torch Tensors, numpy, plain floats)
# ---------------------------------------------------------------------------

def _coerce_scalar_float(*, value: Any, label: str) -> float:
    try:
        if hasattr(value, "detach"):
            detached = value.detach()
            if hasattr(detached, "cpu"):
                detached = detached.cpu()
            if hasattr(detached, "item"):
                return float(detached.item())
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception as exc:
        raise ValueError(f"{label} must evaluate to a scalar float.") from exc


def _is_solver_results_instance(value: Any) -> bool:
    solver_results_type = _solver_results_type()
    if solver_results_type is None:
        return False
    return isinstance(value, solver_results_type)


@lru_cache(maxsize=1)
def _solver_results_type() -> type[Any] | None:
    try:
        from torchrdit.results import SolverResults
    except Exception:
        return None
    return SolverResults


# ---------------------------------------------------------------------------
# Margin normalization helpers
# ---------------------------------------------------------------------------

def _normalized_signed_margin(row: Mapping[str, Any]) -> float | None:
    margin = _as_float_or_none(row.get("margin"))
    if not isinstance(margin, float):
        margin = _derive_margin_from_row(row)
    if not isinstance(margin, float):
        return None

    scale = _criterion_margin_scale(row)
    if not isinstance(scale, float) or scale <= 0.0:
        scale = 1.0
    return margin / scale


def _derive_margin_from_row(row: Mapping[str, Any]) -> float | None:
    value = _as_float_or_none(row.get("value"))
    if not isinstance(value, float):
        return None

    operation = row.get("operation")
    target = _as_float_or_none(row.get("target"))
    if isinstance(operation, str) and isinstance(target, float):
        if operation in {">", ">="}:
            return value - target
        if operation in {"<", "<="}:
            return target - value
        if operation == "close_to":
            tolerance = _as_float_or_none(row.get("tolerance"))
            if isinstance(tolerance, float) and tolerance > 0:
                diff = abs(value - target) % 360.0
                distance = min(diff, 360.0 - diff)
                return float(tolerance - distance)
            return None
        return None

    return None


def _criterion_margin_scale(row: Mapping[str, Any]) -> float:
    operation = row.get("operation")
    if operation == "close_to":
        tolerance = _as_float_or_none(row.get("tolerance"))
        if isinstance(tolerance, float) and tolerance > 0.0:
            return tolerance
        return 1.0
    target = _as_float_or_none(row.get("target"))
    if isinstance(target, float) and abs(target) > 0.0:
        return abs(target)
    return 1.0


# ---------------------------------------------------------------------------
# Number helpers (unchanged)
# ---------------------------------------------------------------------------

def _validate_number_list(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list.")
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    return [_as_float(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    if isinstance(value, complex):
        if abs(value.imag) > 1e-9:
            raise ValueError(f"{label} must be real-valued.")
        return float(value.real)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, complex):
        if abs(value.imag) > 1e-9:
            return None
        return float(value.real)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
