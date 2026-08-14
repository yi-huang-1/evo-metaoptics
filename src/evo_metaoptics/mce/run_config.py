from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping


class RunConfigError(ValueError):
    pass


_FORBIDDEN_TOP_LEVEL_KEYS = {"workspace", "log_dir", "campaign", "matrix", "runs"}
_RUN_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ExperimentConfig:
    env: str
    model: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        _reject_unknown_keys(payload, {"env", "model"}, section="experiment")
        return cls(
            env=_require_non_empty_string(payload, "experiment.env"),
            model=_require_non_empty_string(payload, "experiment.model"),
        )


@dataclass(frozen=True)
class DataConfig:
    train_data: str
    val_data: str
    test_data: str | None = None
    split: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataConfig":
        _reject_unknown_keys(payload, {"train_data", "val_data", "test_data", "split"}, section="data")
        return cls(
            train_data=_require_non_empty_string(payload, "data.train_data"),
            val_data=_require_non_empty_string(payload, "data.val_data"),
            test_data=_optional_string(payload.get("test_data"), "data.test_data"),
            split=_optional_string(payload.get("split"), "data.split"),
        )


@dataclass(frozen=True)
class ExecutionConfig:
    iterations: int
    start_iter: int = 1
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None
    train_batch_size: int | None = None
    meta_agent_hard_retries: int = 2
    codegen_rounds: int = 5
    codegen_inner_attempts: int = 5
    pi_timeout_s: float = 300.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionConfig":
        _reject_unknown_keys(
            payload,
            {
                "iterations",
                "start_iter",
                "train_limit",
                "val_limit",
                "test_limit",
                "train_batch_size",
                "meta_agent_hard_retries",
                "codegen_rounds",
                "codegen_inner_attempts",
                "pi_timeout_s",
            },
            section="execution",
        )
        return cls(
            iterations=_require_positive_int(payload, "execution.iterations", minimum=1),
            start_iter=_optional_positive_int_with_default(
                payload.get("start_iter"),
                "execution.start_iter",
                default=1,
                minimum=0,
            ),
            train_limit=_optional_positive_int_or_zero(payload.get("train_limit"), "execution.train_limit"),
            val_limit=_optional_positive_int_or_zero(payload.get("val_limit"), "execution.val_limit"),
            test_limit=_optional_positive_int_or_zero(payload.get("test_limit"), "execution.test_limit"),
            train_batch_size=_optional_positive_int(payload.get("train_batch_size"), "execution.train_batch_size", minimum=1),
            meta_agent_hard_retries=_optional_positive_int_with_default(
                payload.get("meta_agent_hard_retries"),
                "execution.meta_agent_hard_retries",
                default=2,
                minimum=1,
            ),
            codegen_rounds=_optional_positive_int_with_default(
                payload.get("codegen_rounds"),
                "execution.codegen_rounds",
                default=5,
                minimum=1,
            ),
            codegen_inner_attempts=_optional_positive_int_with_default(
                payload.get("codegen_inner_attempts"),
                "execution.codegen_inner_attempts",
                default=5,
                minimum=1,
            ),
            pi_timeout_s=_optional_positive_float_with_default(
                payload.get("pi_timeout_s"),
                "execution.pi_timeout_s",
                default=300.0,
                minimum=0.001,
            ),
        )


@dataclass(frozen=True)
class ModeConfig:
    skill_path: str | None = None
    no_meta_agent: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModeConfig":
        _reject_unknown_keys(
            payload,
            {"skill_path", "no_meta_agent"},
            section="mode",
        )
        return cls(
            skill_path=_optional_string(payload.get("skill_path"), "mode.skill_path"),
            no_meta_agent=_optional_bool(payload.get("no_meta_agent"), "mode.no_meta_agent", default=False),
        )


@dataclass(frozen=True)
class TracesConfig:
    pi_session_traces: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TracesConfig":
        mapping = _coerce_section_mapping(payload, section="traces", required=False)
        _reject_unknown_keys(mapping, {"pi_session_traces"}, section="traces")
        return cls(
            pi_session_traces=_optional_bool(mapping.get("pi_session_traces"), "traces.pi_session_traces", default=True),
        )


@dataclass(frozen=True)
class BundleConfig:
    enabled: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "BundleConfig":
        mapping = _coerce_section_mapping(payload, section="bundle", required=False)
        _reject_unknown_keys(mapping, {"enabled"}, section="bundle")
        return cls(
            enabled=_optional_bool(mapping.get("enabled"), "bundle.enabled", default=True),
        )


@dataclass(frozen=True)
class RunConfig:
    name: str
    experiment: ExperimentConfig
    data: DataConfig
    execution: ExecutionConfig
    mode: ModeConfig
    traces: TracesConfig = field(default_factory=TracesConfig)
    bundle: BundleConfig = field(default_factory=BundleConfig)
    env_vars: dict[str, str] = field(default_factory=dict)
    run_slug: str = field(init=False)
    workspace: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        run_slug = _slugify_run_name(self.name)
        object.__setattr__(self, "run_slug", run_slug)
        object.__setattr__(self, "workspace", Path("workspace") / run_slug)
        object.__setattr__(self, "log_dir", Path("logs") / f"run_{run_slug}")

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "RunConfig":
        payload = _coerce_section_mapping(config, section="config", required=True)
        if any(key in payload for key in _FORBIDDEN_TOP_LEVEL_KEYS):
            forbidden = sorted(key for key in _FORBIDDEN_TOP_LEVEL_KEYS if key in payload)
            if len(forbidden) == 1:
                raise RunConfigError(f"Field '{forbidden[0]}' is launcher-generated and must not appear in YAML")
            joined = ", ".join(forbidden)
            raise RunConfigError(f"Fields {joined} are launcher-generated and must not appear in YAML")
        _reject_unknown_keys(
            payload,
            {"name", "experiment", "data", "execution", "mode", "traces", "bundle", "env_vars"},
            section="config",
        )
        return cls(
            name=_require_non_empty_string(payload, "name"),
            experiment=ExperimentConfig.from_dict(_coerce_section_mapping(payload.get("experiment"), section="experiment", required=True)),
            data=DataConfig.from_dict(_coerce_section_mapping(payload.get("data"), section="data", required=True)),
            execution=ExecutionConfig.from_dict(
                _coerce_section_mapping(payload.get("execution"), section="execution", required=True)
            ),
            mode=ModeConfig.from_dict(_coerce_section_mapping(payload.get("mode"), section="mode", required=True)),
            traces=TracesConfig.from_dict(payload.get("traces")),
            bundle=BundleConfig.from_dict(payload.get("bundle")),
            env_vars=_parse_env_vars(payload.get("env_vars")),
        )


def _coerce_section_mapping(value: Any, *, section: str, required: bool) -> dict[str, Any]:
    if value is None:
        if required:
            raise RunConfigError(f"Missing required section '{section}'")
        return {}
    if not isinstance(value, Mapping):
        if section == "config":
            raise RunConfigError("Config must describe exactly one run as a mapping")
        raise RunConfigError(f"Section '{section}' must be a mapping")
    return dict(value)


def _reject_unknown_keys(payload: Mapping[str, Any], allowed_keys: set[str], *, section: str) -> None:
    unknown_keys = sorted(key for key in payload if key not in allowed_keys)
    if not unknown_keys:
        return
    if len(unknown_keys) == 1:
        raise RunConfigError(f"Unknown field '{unknown_keys[0]}' in section '{section}'")
    unknown_display = ", ".join(unknown_keys)
    raise RunConfigError(f"Unknown fields in section '{section}': {unknown_display}")


def _require_non_empty_string(payload: Mapping[str, Any], field_name: str) -> str:
    key = field_name.rsplit(".", 1)[-1]
    if key not in payload:
        raise RunConfigError(f"Missing required field '{field_name}'")
    value = payload[key]
    if not isinstance(value, str):
        raise RunConfigError(f"Field '{field_name}' must be a string")
    normalized = value.strip()
    if not normalized:
        raise RunConfigError(f"Field '{field_name}' must not be empty")
    return normalized


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunConfigError(f"Field '{field_name}' must be a string or null")
    normalized = value.strip()
    return normalized or None

def _optional_bool(value: Any, field_name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RunConfigError(f"Field '{field_name}' must be a boolean")
    return value


def _require_positive_int(payload: Mapping[str, Any], field_name: str, *, minimum: int) -> int:
    key = field_name.rsplit(".", 1)[-1]
    if key not in payload:
        raise RunConfigError(f"Missing required field '{field_name}'")
    return _validate_int(payload[key], field_name, minimum=minimum)


def _optional_positive_int(value: Any, field_name: str, *, minimum: int, default: int | None = None) -> int | None:
    if value is None:
        return default
    return _validate_int(value, field_name, minimum=minimum)


def _optional_positive_int_with_default(value: Any, field_name: str, *, minimum: int, default: int) -> int:
    if value is None:
        return default
    return _validate_int(value, field_name, minimum=minimum)


def _optional_positive_int_or_zero(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _validate_int(value, field_name, minimum=0)


def _optional_positive_float_with_default(value: Any, field_name: str, *, minimum: float, default: float) -> float:
    if value is None:
        return default
    return _validate_float(value, field_name, minimum=minimum)


def _validate_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunConfigError(f"Field '{field_name}' must be an integer")
    if value < minimum:
        qualifier = f">= {minimum}"
        raise RunConfigError(f"Field '{field_name}' must be {qualifier}")
    return value


def _validate_float(value: Any, field_name: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunConfigError(f"Field '{field_name}' must be a number")
    normalized = float(value)
    if normalized < minimum:
        raise RunConfigError(f"Field '{field_name}' must be >= {minimum}")
    return normalized


def _parse_env_vars(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RunConfigError("Section 'env_vars' must be a mapping")
    env_vars: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise RunConfigError("Field 'env_vars' keys must be non-empty strings")
        if isinstance(raw_value, (str, int, float, bool)):
            env_vars[key] = str(raw_value)
            continue
        if raw_value is None:
            raise RunConfigError(f"Field 'env_vars.{key}' must not be null")
        raise RunConfigError(f"Field 'env_vars.{key}' must be a scalar value")
    return env_vars


def _slugify_run_name(name: str) -> str:
    slug = _RUN_NAME_SANITIZE_RE.sub("-", name.strip()).strip("-.")
    if not slug:
        raise RunConfigError("Field 'name' does not produce a usable folder name")
    return slug
