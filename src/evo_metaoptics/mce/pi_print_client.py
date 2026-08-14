from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

from evo_metaoptics.mce.shutdown import is_shutdown_requested, wait_unless_shutdown

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.5


class _ShutdownInterrupt(Exception):
    pass


def _is_auth_error(detail: str) -> bool:
    """Detect authentication/credential errors from Pi subprocess output."""
    d = detail.lower()
    return (
        "authentication failed" in d
        or ("auth" in d and "expired" in d)
        or "credentials" in d
        or "re-authenticate" in d
        or "unauthorized" in d
    )


class PiPrintClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        skill_paths: list[str] | None = None,
        timeout_seconds: float = 300.0,
        cwd: str | os.PathLike[str] | None = None,
        session_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._explicit_model = model
        self._skill_paths = list(skill_paths or [])
        self._timeout_seconds = timeout_seconds
        self._cwd: str | None = str(cwd) if cwd is not None else None
        self._session_dir = Path(session_dir) if session_dir is not None else None
        self._started = False

    def _resolve_model(self) -> str | None:
        if self._explicit_model:
            return self._explicit_model
        return os.getenv("EVO_METAOPTICS_PI_MODEL")

    def _build_command(self, prompt: str) -> list[str]:
        command = ["pi", "--print", "--thinking", "high"]
        if self._session_dir is None:
            command.append("--no-session")
        else:
            command.extend(["--session-dir", str(self._session_dir)])
        command.append("--no-skills")
        model = self._resolve_model()
        if model:
            command.extend(["--model", model])
        for path in self._skill_paths:
            command.extend(["--skill", path])
        command.append(prompt)
        return command

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Pi print client already started")
        if self._session_dir is not None:
            logger.debug("Session persistence enabled: %s", self._session_dir)
            self._session_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.debug("Session persistence disabled (--no-session)")
        self._started = True

    def is_healthy(self) -> bool:
        return self._started

    def stop(self) -> None:
        self._started = False

    async def invoke(self, prompt: str) -> dict[str, str | None]:
        return await asyncio.to_thread(self.invoke_sync, prompt)

    def _build_timeout_warning(self) -> str:
        timeout_value = format(float(self._timeout_seconds), "g")
        return (
            f"WARNING: Pi subprocess timed out after {timeout_value}s. "
            "Consider increasing execution.pi_timeout_s or --pi-timeout-s if longer Pi runs are expected."
        )

    def invoke_sync(self, prompt: str) -> dict[str, str | None]:
        if not self._started:
            raise RuntimeError("Pi print client is not started")

        max_retries = 99
        sleep_minutes = 1

        for attempt in range(max_retries + 1):
            if is_shutdown_requested():
                raise RuntimeError("Shutdown requested")

            proc = subprocess.Popen(
                self._build_command(prompt),
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = self._wait_or_terminate(
                    proc, self._timeout_seconds
                )
            except _ShutdownInterrupt:
                raise RuntimeError(
                    "Shutdown requested — Pi subprocess terminated"
                )

            if proc.returncode != 0:
                detail = (
                    (stderr or "").strip()
                    or (stdout or "").strip()
                    or f"pi exited with code {proc.returncode}"
                )
                if "429" in detail and attempt < max_retries:
                    logger.warning(
                        "Hit 429 rate limit (attempt %d/%d), sleeping for %d minute(s) and retrying...",
                        attempt + 1,
                        max_retries,
                        sleep_minutes,
                    )
                    if wait_unless_shutdown(sleep_minutes * 60):
                        raise RuntimeError(
                            "Shutdown requested during 429 backoff"
                        )
                    continue
                if _is_auth_error(detail) and attempt < max_retries:
                    logger.warning(
                        "Authentication failure detected (attempt %d/%d). "
                        "Please re-authenticate and press Enter to retry...",
                        attempt + 1,
                        max_retries,
                    )
                    try:
                        input(">>> Press Enter to retry after re-authenticating...")
                    except EOFError:
                        raise RuntimeError(
                            "Non-interactive environment — cannot prompt for auth retry"
                        )
                    if is_shutdown_requested():
                        raise RuntimeError(
                            "Shutdown requested during auth retry"
                        )
                    continue
                raise RuntimeError(detail)
            return {
                "content": (stdout or "").rstrip("\n"),
                "error": None,
            }

        raise RuntimeError("Exhausted retries")  # pragma: no cover

    def _wait_or_terminate(
        self, proc: subprocess.Popen[str], timeout_s: float
    ) -> tuple[str, str]:
        elapsed = 0.0
        while proc.poll() is None:
            if is_shutdown_requested():
                self._terminate_process(proc)
                raise _ShutdownInterrupt()
            if elapsed >= timeout_s:
                self._terminate_process(proc)
                warning = self._build_timeout_warning()
                logger.warning(warning)
                raise RuntimeError(warning)
            time.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        return stdout, stderr

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[str]) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


__all__ = ["PiPrintClient"]
