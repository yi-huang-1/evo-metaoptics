"""Environment registry for migrated MCE environments."""

from __future__ import annotations

from typing import Dict, Type

from evo_metaoptics.mce_env.base import EnvironmentRuntimeConfig, TaskEnvironment
from evo_metaoptics.mce_env.metaoptics_inverse_design import MetaopticsInverseDesignEnvironment


class EnvironmentRegistry:
    """Registry for task environments used by migrated MCE modules."""

    _environments: Dict[str, Type[TaskEnvironment]] = {}

    @classmethod
    def register(cls, name: str, env_class: Type[TaskEnvironment]) -> None:
        cls._environments[name] = env_class

    @classmethod
    def get(
        cls,
        name: str,
        *,
        runtime_config: EnvironmentRuntimeConfig | None = None,
    ) -> TaskEnvironment:
        if name not in cls._environments:
            available = ", ".join(cls._environments.keys()) or "(none)"
            raise ValueError(f"Environment '{name}' not found. Available: {available}")
        env = cls._environments[name]()
        env.configure_runtime(runtime_config)
        return env

    @classmethod
    def list_environments(cls) -> list[str]:
        return list(cls._environments.keys())

    @classmethod
    def get_task_instruction(cls, name: str) -> str:
        return cls.get(name).get_task_instruction()


EnvironmentRegistry.register("metaoptics_inverse_design", MetaopticsInverseDesignEnvironment)
