"""
AgentSession protocol and implementations.

Defines the abstraction for agent communication used by MCE consumers
(base_agent, meta_agent, task environments).
"""
from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class AgentResponse:
    """Response from an agent invocation.
    
    Attributes:
        content: Final assistant text response.
        error: Error message if invocation failed, None otherwise.
    """

    content: str
    error: str | None = None


@runtime_checkable
class AgentSession(Protocol):
    """Protocol for agent communication sessions.
    
    Defines the interface that all agent session implementations must follow.
    Used by base_agent, meta_agent, and task environments to communicate
    with agents in a backend-agnostic way.
    """

    @property
    def cwd(self) -> Path:
        """Current working directory for the session."""
        ...

    async def send_message(self, prompt: str) -> AgentResponse:
        """Send a message to the agent and get a response.
        
        Args:
            prompt: The user prompt to send to the agent.
            
        Returns:
            AgentResponse with content and error.
        """
        ...

    async def close(self) -> None:
        """Close the session and clean up resources."""
        ...


class MockAgentSession:
    """Mock implementation of AgentSession for testing.
    
    Simulates agent behavior without requiring a real backend.
    Writes files to a temporary directory.
    """

    def __init__(self, cwd: Path | None = None) -> None:
        """Initialize MockAgentSession.
        
        Args:
            cwd: Working directory for the session. If None, uses a temp directory.
        """
        if cwd is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self._cwd = Path(self._temp_dir.name)
        else:
            self._temp_dir = None
            self._cwd = cwd

    @property
    def cwd(self) -> Path:
        """Return the current working directory."""
        return self._cwd

    async def send_message(self, prompt: str) -> AgentResponse:
        """Send a message and return a mock response.
        
        Args:
            prompt: The user prompt.
            
        Returns:
            AgentResponse with simulated content.
        """
        # Simulate some async work
        await asyncio.sleep(0.01)

        # Generate a simple mock response
        content = f"Mock response to: {prompt[:50]}..."
        # Simulate file writing if prompt mentions it
        if "write" in prompt.lower() or "file" in prompt.lower():
            test_file = self._cwd / "mock_output.py"
            test_file.write_text("# Mock generated file\npass\n")

        return AgentResponse(
            content=content,
            error=None,
        )

    async def close(self) -> None:
        """Close the session and clean up resources."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()


class PiAgentSession:
    """AgentSession implementation wrapping a Pi subprocess client."""

    def __init__(self, client: Any, cwd: Path) -> None:
        """Initialize PiAgentSession.
        
        Args:
            client: Pi subprocess client instance (must have invoke() and close() methods).
            cwd: Working directory for the session.
        """
        self._client = client
        self._cwd = cwd

    @property
    def cwd(self) -> Path:
        """Return the current working directory."""
        return self._cwd

    async def send_message(self, prompt: str) -> AgentResponse:
        """Send a message via Pi and return the response.
        
        Args:
            prompt: The user prompt.
            
        Returns:
            AgentResponse with content from Pi agent.
        """
        try:
            result = await self._client.invoke(prompt)
            content = result.get("content", "")
            error = result.get("error")

            return AgentResponse(
                content=content,
                error=error,
            )
        except Exception as e:
            return AgentResponse(
                content="",
                error=str(e),
            )

    async def close(self) -> None:
        """Close the session and clean up resources."""
        if hasattr(self._client, "close"):
            await self._client.close()

    def send_message_sync(self, prompt: str) -> AgentResponse:
        """Send a message synchronously via Pi.

        Same as send_message but without async overhead.
        Used by the environment's retry loop running on a dedicated thread.
        """
        try:
            result = self._client.invoke_sync(prompt)
            content = result.get("content", "")
            error = result.get("error")
            return AgentResponse(
                content=content,
                error=error,
            )
        except Exception as e:
            return AgentResponse(
                content="",
                error=str(e),
            )

    def close_sync(self) -> None:
        """Close the session synchronously."""
        if hasattr(self._client, "close_sync"):
            self._client.close_sync()
