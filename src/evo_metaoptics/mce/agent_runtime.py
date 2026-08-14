from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from evo_metaoptics.mce.agent_session import AgentResponse, AgentSession, PiAgentSession
from evo_metaoptics.mce.agents_md import write_agents_md, write_skills_to_workspace
from evo_metaoptics.mce.pi_print_client import PiPrintClient


_RUN_DIR_ENV = "EVO_METAOPTICS_RUN_DIR"
_SESSION_TRACES_ENV = "EVO_METAOPTICS_SESSION_TRACES"
_PI_TIMEOUT_ENV = "EVO_METAOPTICS_PI_TIMEOUT_S"
_LOGGER = logging.getLogger(__name__)


def resolve_model_name(explicit_model: str | None = None) -> str:
    if explicit_model:
        return explicit_model

    value = os.getenv("EVO_METAOPTICS_MODEL")
    if value:
        return value

    return "openai/gpt-4.1-mini"


class _PiPrintSessionClient:
    def __init__(self, client: PiPrintClient) -> None:
        self._client = client

    async def invoke(self, prompt: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._invoke_sync, prompt)

    def _invoke_sync(self, prompt: str) -> dict[str, Any]:
        return self._client.invoke_sync(prompt)

    def invoke_sync(self, prompt: str) -> dict[str, Any]:
        return self._invoke_sync(prompt)

    async def close(self) -> None:
        await asyncio.to_thread(self._client.stop)

    def close_sync(self) -> None:
        self._client.stop()


def _parse_optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_optional_float(value: object, *, default: float, minimum: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str)):
        return default
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return default
    if normalized < minimum:
        return default
    return normalized


def set_run_dir_env(run_dir: str | Path | None) -> None:
    if run_dir is None:
        os.environ.pop(_RUN_DIR_ENV, None)
        return
    os.environ[_RUN_DIR_ENV] = str(Path(run_dir).resolve())


def resolve_session_traces_enabled(explicit_value: bool | None = None) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    return _parse_optional_bool(
        os.getenv(_SESSION_TRACES_ENV),
        default=True,
    )


def resolve_pi_timeout_seconds(explicit_value: float | None = None) -> float:
    if explicit_value is not None:
        return _parse_optional_float(explicit_value, default=300.0, minimum=0.001)
    return _parse_optional_float(os.getenv(_PI_TIMEOUT_ENV), default=300.0, minimum=0.001)


def _resolve_pi_session_traces_dir(
    run_dir: str | Path | None = None,
    *,
    session_traces_enabled: bool | None = None,
) -> Path | None:
    if not resolve_session_traces_enabled(session_traces_enabled):
        return None
    candidate = run_dir
    if candidate is None:
        candidate = os.getenv(_RUN_DIR_ENV)
    if candidate is None or not str(candidate).strip():
        return None
    return Path(candidate).resolve() / "pi_session_traces"


def start_pi_session_client(
    *,
    cwd: str | Path,
    model: str,
    skill_paths: list[str] | None = None,
    run_dir: str | Path | None = None,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
) -> PiPrintClient:
    session_dir = _resolve_pi_session_traces_dir(
        run_dir,
        session_traces_enabled=session_traces_enabled,
    )

    client = PiPrintClient(
        model=resolve_model_name(model),
        skill_paths=list(skill_paths or []),
        timeout_seconds=resolve_pi_timeout_seconds(timeout_s),
        cwd=Path(cwd),
        session_dir=session_dir,
    )
    client.start()
    if session_dir is not None:
        _LOGGER.info("Pi session persistence enabled: %s", session_dir)
    return client


def create_pi_session(
    iter_dir: str | Path,
    system_prompt: str,
    skills: list[str],
    model: str,
    skill_paths: list[str] | None = None,
    context_available: bool = False,
    run_dir: str | Path | None = None,
    timeout_s: float | None = None,
    session_traces_enabled: bool | None = None,
) -> AgentSession:
    iter_path = Path(iter_dir)
    write_agents_md(iter_dir=iter_path, system_prompt=system_prompt, context_available=context_available)
    write_skills_to_workspace(iter_dir=iter_path, skill_sources=skills)
    client = start_pi_session_client(
        cwd=iter_path,
        model=model,
        skill_paths=list(skill_paths or []),
        run_dir=run_dir,
        timeout_s=timeout_s,
        session_traces_enabled=session_traces_enabled,
    )

    return PiAgentSession(_PiPrintSessionClient(client), cwd=iter_path)


def invoke_pi_session(session: AgentSession, prompt: str) -> AgentResponse:
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    send_message_sync = getattr(session, "send_message_sync", None)
    if callable(send_message_sync):
        return send_message_sync(prompt)
    return asyncio.run(session.send_message(prompt))


def wrap_pi_session_client_as_session(
    client: PiPrintClient,
    cwd: Path,
) -> AgentSession:
    return PiAgentSession(_PiPrintSessionClient(client), cwd=cwd)


__all__ = [
    "resolve_model_name",
    "resolve_session_traces_enabled",
    "resolve_pi_timeout_seconds",
    "set_run_dir_env",
    "start_pi_session_client",
    "create_pi_session",
    "invoke_pi_session",
    "wrap_pi_session_client_as_session",
]
