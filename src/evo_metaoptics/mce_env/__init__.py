"""Environment interfaces and lazy registry access."""

from typing import TYPE_CHECKING

from evo_metaoptics.mce_env.base import EnvironmentResult, InterfaceSignature, Sample, TaskEnvironment

if TYPE_CHECKING:
    from evo_metaoptics.mce_env.registry import EnvironmentRegistry as EnvironmentRegistry


def __getattr__(name: str):
    if name == "EnvironmentRegistry":
        from evo_metaoptics.mce_env.registry import EnvironmentRegistry

        return EnvironmentRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EnvironmentResult",
    "InterfaceSignature",
    "Sample",
    "TaskEnvironment",
    "EnvironmentRegistry",
]
