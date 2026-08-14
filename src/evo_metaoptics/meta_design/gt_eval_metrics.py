"""Semantic Metric Registry for gt_eval criteria evaluation.

Replaces raw ``eval()`` expression strings with named, parameterized metric
types dispatched through a registry. Each metric is a pure function that
extracts a scalar float from a ``SolverResults`` object.

Metric registry design:
- **Declare WHAT, not HOW** — Criteria say ``"metric": "total_transmission"``
  with ``"wavelength_index": 0``, not ``"float(r.transmission[0])"``.
- **Shape-safe by construction** — Extractors validate tensor shapes and give
  clear errors.
- **Source-aware** — Handle both single-source and batched-source results via
  explicit ``source_index`` parameter.

Usage::

    from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric

    value = extract_metric(
        solver_results=r,
        metric="total_transmission",
        params={"wavelength_index": 0},
    )
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MetricError(Exception):
    """Raised when a metric extraction fails with a clear, actionable message."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps metric name → extractor function(r, **params) -> float
_METRIC_REGISTRY: dict[str, Callable[..., float]] = {}


def register_metric(name: str) -> Callable:
    """Decorator to register a metric extractor function."""
    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        if name in _METRIC_REGISTRY:
            raise ValueError(f"Metric {name!r} is already registered.")
        _METRIC_REGISTRY[name] = fn
        return fn
    return decorator


def get_metric(name: str) -> Callable[..., float]:
    """Look up a registered metric extractor by name.

    Raises:
        KeyError: If the metric name is not registered.
    """
    if name not in _METRIC_REGISTRY:
        registered = sorted(_METRIC_REGISTRY.keys())
        raise KeyError(
            f"Unknown metric {name!r}. Registered metrics: {registered}"
        )
    return _METRIC_REGISTRY[name]


def list_metrics() -> list[str]:
    """Return sorted list of all registered metric names."""
    return sorted(_METRIC_REGISTRY.keys())


def extract_metric(
    solver_results: Any,
    metric: str,
    params: dict[str, Any] | None = None,
) -> float:
    """Extract a scalar metric value from SolverResults.

    Args:
        solver_results: A ``torchrdit.results.SolverResults`` instance.
        metric: Registered metric name (e.g. ``"total_transmission"``).
        params: Metric-specific parameters (e.g. ``{"wavelength_index": 0}``).

    Returns:
        Scalar float value.

    Raises:
        KeyError: If metric name is not registered.
        MetricError: If extraction fails (shape mismatch, index OOB, etc.).
    """
    fn = get_metric(metric)
    resolved_params = _resolve_params(params or {})
    try:
        return fn(solver_results, **resolved_params)
    except MetricError:
        raise
    except (IndexError, ValueError, TypeError, RuntimeError) as exc:
        raise MetricError(
            f"Failed to extract metric {metric!r} with params {resolved_params}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Parameter resolution (aliases)
# ---------------------------------------------------------------------------

# Polarization → field component alias
_POLARIZATION_TO_COMPONENT = {
    "TM": "x",
    "TE": "y",
}


def _resolve_params(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve parameter aliases (e.g. polarization → component)."""
    resolved = dict(params)
    if "polarization" in resolved:
        pol = resolved.pop("polarization")
        if pol not in _POLARIZATION_TO_COMPONENT:
            raise MetricError(
                f"Unknown polarization {pol!r}. Must be one of: {sorted(_POLARIZATION_TO_COMPONENT.keys())}"
            )
        if "component" in resolved:
            raise MetricError(
                "Cannot specify both 'polarization' and 'component'. Use one or the other."
            )
        resolved["component"] = _POLARIZATION_TO_COMPONENT[pol]
    return resolved


# ---------------------------------------------------------------------------
# Source resolution helper
# ---------------------------------------------------------------------------

def _resolve_source(r: Any, source_index: int | None) -> Any:
    """Extract single-source result if source_index is specified.

    Args:
        r: SolverResults object.
        source_index: Optional source index for batched results.

    Returns:
        SolverResults (unbatched) for the selected source.

    Raises:
        MetricError: If source_index is required but missing, or out of range.
    """
    if source_index is None:
        if r.is_batched:
            raise MetricError(
                f"SolverResults has {r.n_sources} sources but no source_index specified. "
                f"Use source_index to select which source to evaluate."
            )
        return r
    # Validate source_index range
    if source_index < 0 or source_index >= r.n_sources:
        raise MetricError(
            f"source_index={source_index} is out of range for {r.n_sources} source(s)."
        )
    return r[source_index]


# ---------------------------------------------------------------------------
# Wavelength index validation
# ---------------------------------------------------------------------------

def _validate_wavelength_index(tensor: torch.Tensor, wavelength_index: int, metric_name: str) -> None:
    """Validate that wavelength_index is within tensor bounds.

    Args:
        tensor: 1-D tensor of shape ``(n_freqs,)``.
        wavelength_index: Index to validate.
        metric_name: For error messages.

    Raises:
        MetricError: If index is out of bounds.
    """
    n_freqs = tensor.shape[0]
    if wavelength_index < 0 or wavelength_index >= n_freqs:
        raise MetricError(
            f"wavelength_index={wavelength_index} is out of bounds for "
            f"{metric_name} with {n_freqs} wavelength(s)."
        )


# ---------------------------------------------------------------------------
# Component resolution
# ---------------------------------------------------------------------------

_COMPONENT_MAP = {"x": 0, "y": 1, "z": 2}


def _resolve_component_index(component: str) -> int:
    """Map component name to tuple index.

    Args:
        component: One of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        Integer index (0, 1, or 2).

    Raises:
        MetricError: If component is invalid.
    """
    if component not in _COMPONENT_MAP:
        raise MetricError(
            f"Invalid component {component!r}. Must be one of: {sorted(_COMPONENT_MAP.keys())}"
        )
    return _COMPONENT_MAP[component]


# ---------------------------------------------------------------------------
# Scalar coercion (detach from autograd, move to CPU)
# ---------------------------------------------------------------------------

def _to_scalar_float(value: torch.Tensor) -> float:
    """Convert a scalar tensor to Python float, safely detaching from autograd."""
    detached = value.detach()
    if hasattr(detached, "cpu"):
        detached = detached.cpu()
    return float(detached.item())


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------

@register_metric("total_transmission")
def _extract_total_transmission(
    r: Any,
    *,
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract total transmission efficiency at a specific wavelength.

    ``r.transmission`` has shape ``(n_freqs,)`` for single source,
    or ``(n_sources, n_freqs)`` for batched. Source resolution handles
    the batched case.
    """
    r = _resolve_source(r, source_index)
    _validate_wavelength_index(r.transmission, wavelength_index, "total_transmission")
    return _to_scalar_float(r.transmission[wavelength_index])


@register_metric("total_reflection")
def _extract_total_reflection(
    r: Any,
    *,
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract total reflection efficiency at a specific wavelength."""
    r = _resolve_source(r, source_index)
    _validate_wavelength_index(r.reflection, wavelength_index, "total_reflection")
    return _to_scalar_float(r.reflection[wavelength_index])


@register_metric("zero_order_transmission_phase_deg")
def _extract_zero_order_transmission_phase_deg(
    r: Any,
    *,
    component: str,
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract zero-order transmission phase in degrees.

    Uses ``r.get_zero_order_transmission()`` which returns ``(tx, ty, tz)``
    tuple. The ``component`` parameter selects which field component.
    """
    r = _resolve_source(r, source_index)
    comp_idx = _resolve_component_index(component)
    fields = r.get_zero_order_transmission()  # (tx, ty, tz)
    field = fields[comp_idx]
    _validate_wavelength_index(field, wavelength_index, "zero_order_transmission_phase_deg")
    phase_rad = torch.angle(field[wavelength_index])
    phase_deg = torch.rad2deg(phase_rad)
    return _to_scalar_float(phase_deg)


@register_metric("zero_order_transmission_amplitude")
def _extract_zero_order_transmission_amplitude(
    r: Any,
    *,
    component: str,
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract zero-order transmission amplitude (|E| magnitude).

    Uses ``r.get_zero_order_transmission()`` which returns ``(tx, ty, tz)``
    tuple. The ``component`` parameter selects which field component.
    """
    r = _resolve_source(r, source_index)
    comp_idx = _resolve_component_index(component)
    fields = r.get_zero_order_transmission()
    field = fields[comp_idx]
    _validate_wavelength_index(field, wavelength_index, "zero_order_transmission_amplitude")
    amplitude = torch.abs(field[wavelength_index])
    return _to_scalar_float(amplitude)


@register_metric("order_transmission_efficiency")
def _extract_order_transmission_efficiency(
    r: Any,
    *,
    order: Sequence[int],
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract transmission diffraction efficiency for a specific order.

    Uses ``r.get_order_transmission_efficiency(order_x, order_y)`` which
    internally resolves the (m,n) order to array indices via
    ``get_diffraction_order_indices()``.

    Args:
        order: ``[order_x, order_y]`` diffraction order.
    """
    r = _resolve_source(r, source_index)
    if not isinstance(order, (list, tuple)) or len(order) != 2:
        raise MetricError(
            f"order must be a list/tuple of [order_x, order_y], got {order!r}"
        )
    order_x, order_y = int(order[0]), int(order[1])
    eff = r.get_order_transmission_efficiency(order_x, order_y)
    _validate_wavelength_index(eff, wavelength_index, "order_transmission_efficiency")
    return _to_scalar_float(eff[wavelength_index])


@register_metric("order_reflection_efficiency")
def _extract_order_reflection_efficiency(
    r: Any,
    *,
    order: Sequence[int],
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract reflection diffraction efficiency for a specific order.

    Uses ``r.get_order_reflection_efficiency(order_x, order_y)``.

    Args:
        order: ``[order_x, order_y]`` diffraction order.
    """
    r = _resolve_source(r, source_index)
    if not isinstance(order, (list, tuple)) or len(order) != 2:
        raise MetricError(
            f"order must be a list/tuple of [order_x, order_y], got {order!r}"
        )
    order_x, order_y = int(order[0]), int(order[1])
    eff = r.get_order_reflection_efficiency(order_x, order_y)
    _validate_wavelength_index(eff, wavelength_index, "order_reflection_efficiency")
    return _to_scalar_float(eff[wavelength_index])


@register_metric("zero_order_reflection_phase_deg")
def _extract_zero_order_reflection_phase_deg(
    r: Any,
    *,
    component: str,
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract zero-order reflection phase in degrees.

    Uses ``r.get_zero_order_reflection()`` which returns ``(rx, ry, rz)``
    tuple. The ``component`` parameter selects which field component.
    """
    r = _resolve_source(r, source_index)
    comp_idx = _resolve_component_index(component)
    fields = r.get_zero_order_reflection()  # (rx, ry, rz)
    field = fields[comp_idx]
    _validate_wavelength_index(field, wavelength_index, "zero_order_reflection_phase_deg")
    phase_rad = torch.angle(field[wavelength_index])
    phase_deg = torch.rad2deg(phase_rad)
    return _to_scalar_float(phase_deg)


@register_metric("zero_order_reflection_amplitude")
def _extract_zero_order_reflection_amplitude(
    r: Any,
    *,
    component: str,
    wavelength_index: int,
    source_index: int | None = None,
) -> float:
    """Extract zero-order reflection amplitude (|E| magnitude).

    Uses ``r.get_zero_order_reflection()`` which returns ``(rx, ry, rz)``
    tuple. The ``component`` parameter selects which field component.
    """
    r = _resolve_source(r, source_index)
    comp_idx = _resolve_component_index(component)
    fields = r.get_zero_order_reflection()
    field = fields[comp_idx]
    _validate_wavelength_index(field, wavelength_index, "zero_order_reflection_amplitude")
    amplitude = torch.abs(field[wavelength_index])
    return _to_scalar_float(amplitude)


# ---------------------------------------------------------------------------
# Generation-time validation
# ---------------------------------------------------------------------------

# Required params for each metric type
_METRIC_REQUIRED_PARAMS: dict[str, list[str]] = {
    "total_transmission": ["wavelength_index"],
    "total_reflection": ["wavelength_index"],
    "zero_order_transmission_phase_deg": ["wavelength_index"],  # component or polarization also required
    "zero_order_transmission_amplitude": ["wavelength_index"],  # component or polarization also required
    "order_transmission_efficiency": ["wavelength_index", "order"],
    "order_reflection_efficiency": ["wavelength_index", "order"],
    "zero_order_reflection_phase_deg": ["wavelength_index"],  # component or polarization also required
    "zero_order_reflection_amplitude": ["wavelength_index"],  # component or polarization also required
}

# Metrics that require component or polarization
_METRICS_REQUIRING_COMPONENT = {
    "zero_order_transmission_phase_deg",
    "zero_order_transmission_amplitude",
    "zero_order_reflection_phase_deg",
    "zero_order_reflection_amplitude",
}


def validate_metric_criterion(
    criterion: dict[str, Any],
    wavelength_um: list[float] | None = None,
) -> None:
    """Validate a v8 metric criterion at generation time.

    Checks:
    - Metric name is registered.
    - Required params are present.
    - Component/polarization is valid (for field metrics).
    - wavelength_index is within bounds (if wavelength_um provided).

    Args:
        criterion: A single criterion dict with 'metric' and 'params' keys.
        wavelength_um: Optional wavelength list for bounds checking.

    Raises:
        MetricError: If validation fails.
    """
    metric_name = criterion.get("metric")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise MetricError("Criterion must have a non-empty 'metric' string.")
    metric_name = metric_name.strip()

    # Check metric is registered.
    if metric_name not in _METRIC_REGISTRY:
        registered = sorted(_METRIC_REGISTRY.keys())
        raise MetricError(
            f"Unknown metric {metric_name!r}. Registered metrics: {registered}"
        )

    params = criterion.get("params", {})
    if not isinstance(params, dict):
        raise MetricError("Criterion 'params' must be a dict.")

    # Resolve aliases (polarization → component) for validation.
    resolved = _resolve_params(params)

    # Check required params.
    required = _METRIC_REQUIRED_PARAMS.get(metric_name, [])
    for key in required:
        if key not in resolved:
            raise MetricError(
                f"Metric {metric_name!r} requires param '{key}' but it was not provided."
            )

    # Check component/polarization for field metrics.
    if metric_name in _METRICS_REQUIRING_COMPONENT:
        if "component" not in resolved:
            raise MetricError(
                f"Metric {metric_name!r} requires 'component' or 'polarization' param."
            )
        comp = resolved["component"]
        if comp not in _COMPONENT_MAP:
            raise MetricError(
                f"Invalid component {comp!r} for metric {metric_name!r}. "
                f"Must be one of: {sorted(_COMPONENT_MAP.keys())}"
            )

    # Check wavelength_index bounds.
    wl_idx = resolved.get("wavelength_index")
    if isinstance(wl_idx, int) and wavelength_um is not None:
        if wl_idx < 0 or wl_idx >= len(wavelength_um):
            raise MetricError(
                f"wavelength_index={wl_idx} is out of bounds for "
                f"{len(wavelength_um)} wavelength(s) in wavelength_um."
            )
