from __future__ import annotations

import logging
import threading

_logger = logging.getLogger(__name__)
_shutdown_event = threading.Event()


def request_shutdown() -> None:
    if not _shutdown_event.is_set():
        _logger.warning(
            "Shutdown requested — stopping after in-flight operations complete."
        )
    _shutdown_event.set()


def is_shutdown_requested() -> bool:
    return _shutdown_event.is_set()


def wait_unless_shutdown(seconds: float) -> bool:
    """Sleep for up to *seconds*, returning ``True`` if shutdown was requested."""
    return _shutdown_event.wait(timeout=seconds)


def reset() -> None:
    """Clear the shutdown flag.  Intended for test isolation only."""
    _shutdown_event.clear()
