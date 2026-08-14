"""Tests for AgentSession protocol and implementations."""
from __future__ import annotations

import asyncio
import importlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

agent_session_module = importlib.import_module("evo_metaoptics.mce.agent_session")
AgentResponse = agent_session_module.AgentResponse
AgentSession = agent_session_module.AgentSession
MockAgentSession = agent_session_module.MockAgentSession
PiAgentSession = agent_session_module.PiAgentSession


class TestAgentResponse(unittest.TestCase):
    """Test AgentResponse dataclass."""

    def test_agent_response_creation(self) -> None:
        """Test creating an AgentResponse."""
        response = AgentResponse(
            content="Hello, world!",
            error=None,
        )
        self.assertEqual(response.content, "Hello, world!")
        self.assertIsNone(response.error)

    def test_agent_response_with_error(self) -> None:
        """Test AgentResponse with error."""
        response = AgentResponse(
            content="",
            error="Something went wrong",
        )
        self.assertEqual(response.error, "Something went wrong")

    def test_agent_response_defaults(self) -> None:
        """Test AgentResponse with default values."""
        response = AgentResponse(content="test")
        self.assertEqual(response.content, "test")
        self.assertIsNone(response.error)


class TestMockAgentSession(unittest.TestCase):
    """Test MockAgentSession implementation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_mock_session_creation(self) -> None:
        """Test creating a MockAgentSession."""
        session = MockAgentSession(cwd=self.cwd)
        self.assertEqual(session.cwd, self.cwd)

    def test_mock_session_send_message(self) -> None:
        """Test sending a message to MockAgentSession."""
        session = MockAgentSession(cwd=self.cwd)
        response = asyncio.run(session.send_message("Hello"))
        self.assertIsInstance(response, AgentResponse)
        self.assertIsNotNone(response.content)

    def test_mock_session_file_writing(self) -> None:
        """Test MockAgentSession can write files."""
        session = MockAgentSession(cwd=self.cwd)
        # Send a message that triggers file writing
        response = asyncio.run(session.send_message("Write a file"))
        self.assertIn("Mock response", response.content)

    def test_mock_session_close(self) -> None:
        """Test closing a MockAgentSession."""
        session = MockAgentSession(cwd=self.cwd)
        # Should not raise
        asyncio.run(session.close())

    def test_mock_session_protocol_conformance(self) -> None:
        """Test that MockAgentSession conforms to AgentSession protocol."""
        session = MockAgentSession(cwd=self.cwd)
        # Check that it has the required methods
        self.assertTrue(hasattr(session, "send_message"))
        self.assertTrue(hasattr(session, "close"))
        self.assertTrue(hasattr(session, "cwd"))
        # Check that methods are callable
        self.assertTrue(callable(session.send_message))
        self.assertTrue(callable(session.close))


class TestPiAgentSession(unittest.TestCase):
    """Test PiAgentSession implementation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_pi_session_creation(self) -> None:
        """Test creating a PiAgentSession."""
        mock_client = MockPiPrintClient()
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        self.assertEqual(session.cwd, self.cwd)

    def test_pi_session_send_message(self) -> None:
        """Test sending a message to PiAgentSession."""
        mock_client = MockPiPrintClient()
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        response = asyncio.run(session.send_message("Hello"))
        self.assertIsInstance(response, AgentResponse)

    def test_pi_session_close(self) -> None:
        """Test closing a PiAgentSession."""
        mock_client = MockPiPrintClient()
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        # Should not raise
        asyncio.run(session.close())

    def test_pi_session_send_message_sync(self) -> None:
        """Test sync message sending on PiAgentSession."""
        mock_client = MockPiPrintClient()
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        response = session.send_message_sync("Hello sync")
        self.assertIsInstance(response, AgentResponse)
        self.assertIn("Hello sync", response.content)
        self.assertIsNone(response.error)

    def test_pi_session_send_message_sync_error(self) -> None:
        """Test sync message sending handles errors."""
        mock_client = MockPiPrintClient(error=RuntimeError("test error"))
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        response = session.send_message_sync("Hello")
        self.assertEqual(response.error, "test error")

    def test_pi_session_close_sync(self) -> None:
        """Test sync close on PiAgentSession."""
        mock_client = MockPiPrintClient()
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        # Should not raise
        session.close_sync()

    def test_pi_session_protocol_conformance(self) -> None:
        """Test that PiAgentSession conforms to AgentSession protocol."""
        mock_client = MockPiPrintClient()
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        # Check that it has the required methods
        self.assertTrue(hasattr(session, "send_message"))
        self.assertTrue(hasattr(session, "close"))
        self.assertTrue(hasattr(session, "cwd"))


class TestAgentSessionProtocol(unittest.TestCase):
    """Test AgentSession protocol contract."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_mock_session_implements_protocol(self) -> None:
        """Test that MockAgentSession implements AgentSession protocol."""
        session = MockAgentSession(cwd=self.cwd)
        # Protocol check: can we use it as AgentSession?
        self._verify_agent_session(session)

    def test_pi_session_implements_protocol(self) -> None:
        """Test that PiAgentSession implements AgentSession protocol."""
        mock_client = MockPiPrintClient()
        session = PiAgentSession(client=mock_client, cwd=self.cwd)
        # Protocol check: can we use it as AgentSession?
        self._verify_agent_session(session)

    def _verify_agent_session(self, session: AgentSession) -> None:
        """Verify that an object implements AgentSession protocol."""
        # Check cwd property
        self.assertIsInstance(session.cwd, Path)
        # Check send_message is async callable
        self.assertTrue(callable(session.send_message))
        # Check close is async callable
        self.assertTrue(callable(session.close))


class MockPiPrintClient:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def invoke_sync(self, prompt: str) -> dict[str, Any]:
        """Mock sync invoke method."""
        if self._error:
            raise self._error
        return {
            "content": f"Response to: {prompt}",
            "error": None,
        }

    async def invoke(self, prompt: str) -> dict[str, Any]:
        """Mock async invoke method."""
        return self.invoke_sync(prompt)

    def close_sync(self) -> None:
        """Mock sync close method."""
        pass

    async def close(self) -> None:
        """Mock async close method."""
        pass


if __name__ == "__main__":
    unittest.main()
