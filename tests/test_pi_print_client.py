from __future__ import annotations

import io
import os
import tempfile
import unittest
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

_pi_mod = importlib.import_module("evo_metaoptics.mce.pi_print_client")
PiPrintClient = _pi_mod.PiPrintClient
_shutdown_mod = importlib.import_module("evo_metaoptics.mce.shutdown")

_MOD = "evo_metaoptics.mce.pi_print_client"


def _make_mock_popen(
    *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> MagicMock:
    proc = MagicMock()
    proc.poll.side_effect = [None, returncode]
    proc.returncode = returncode
    proc.stdout = io.StringIO(stdout)
    proc.stderr = io.StringIO(stderr)
    proc.wait.return_value = returncode
    return proc


class TestPiPrintClient(unittest.TestCase):
    def setUp(self) -> None:
        _shutdown_mod.reset()

    def tearDown(self) -> None:
        _shutdown_mod.reset()

    @patch(f"{_MOD}.subprocess.Popen")
    def test_start_marks_client_healthy(self, popen_mock: MagicMock) -> None:
        client = PiPrintClient()
        client.start()
        self.assertTrue(client.is_healthy())
        client.stop()
        self.assertFalse(client.is_healthy())
        popen_mock.assert_not_called()

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_uses_required_default_flags(
        self, popen_mock: MagicMock, _sleep: MagicMock
    ) -> None:
        popen_mock.return_value = _make_mock_popen(stdout="OK\n")

        client = PiPrintClient()
        client.start()
        result = client.invoke_sync("hello")

        args = popen_mock.call_args.args[0]
        self.assertEqual(
            ["pi", "--print", "--thinking", "high", "--no-session", "--no-skills", "hello"],
            args,
        )
        self.assertEqual({"content": "OK", "error": None}, result)

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_with_session_dir_uses_persistent_sessions(
        self, popen_mock: MagicMock, _sleep: MagicMock
    ) -> None:
        popen_mock.return_value = _make_mock_popen(stdout="OK\n")

        client = PiPrintClient(session_dir=Path("/tmp/pi-sessions"))
        client.start()
        client.invoke_sync("hello")

        args = popen_mock.call_args.args[0]
        self.assertEqual(
            [
                "pi", "--print", "--thinking", "high",
                "--session-dir", "/tmp/pi-sessions",
                "--no-skills", "hello",
            ],
            args,
        )

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_adds_model_and_explicit_skills(
        self, popen_mock: MagicMock, _sleep: MagicMock
    ) -> None:
        popen_mock.return_value = _make_mock_popen(stdout="OK\n")

        client = PiPrintClient(model="openai/gpt-5", skill_paths=["/tmp/a", "/tmp/b"])
        client.start()
        client.invoke_sync("hello")

        args = popen_mock.call_args.args[0]
        self.assertEqual(
            [
                "pi", "--print", "--thinking", "high",
                "--no-session", "--no-skills",
                "--model", "openai/gpt-5",
                "--skill", "/tmp/a",
                "--skill", "/tmp/b",
                "hello",
            ],
            args,
        )

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_uses_environment_model_when_explicit_model_missing(
        self, popen_mock: MagicMock, _sleep: MagicMock
    ) -> None:
        popen_mock.return_value = _make_mock_popen(stdout="OK\n")

        with patch.dict(os.environ, {"EVO_METAOPTICS_PI_MODEL": "anthropic/claude-sonnet"}, clear=False):
            client = PiPrintClient()
            client.start()
            client.invoke_sync("hello")

        args = popen_mock.call_args.args[0]
        self.assertIn("--model", args)
        model_idx = args.index("--model")
        self.assertEqual("anthropic/claude-sonnet", args[model_idx + 1])

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_raises_on_timeout(
        self, popen_mock: MagicMock, sleep_mock: MagicMock
    ) -> None:
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.return_value = None
        popen_mock.return_value = proc

        client = PiPrintClient(timeout_seconds=0.1)
        client.start()

        with self.assertRaisesRegex(RuntimeError, "Pi subprocess timed out after 0.1s"):
            client.invoke_sync("hello")

        proc.terminate.assert_called_once()

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_raises_on_subprocess_error(
        self, popen_mock: MagicMock, _sleep: MagicMock
    ) -> None:
        popen_mock.return_value = _make_mock_popen(returncode=1, stderr="fatal")

        client = PiPrintClient()
        client.start()

        with self.assertRaisesRegex(RuntimeError, "fatal"):
            client.invoke_sync("hello")

    @patch(f"{_MOD}.subprocess.Popen")
    def test_session_dir_is_created_on_start(self, popen_mock: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions"
            client = PiPrintClient(session_dir=session_dir)
            client.start()
            self.assertTrue(session_dir.exists())

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_raises_immediately_on_shutdown(
        self, popen_mock: MagicMock, _sleep: MagicMock
    ) -> None:
        _shutdown_mod.request_shutdown()
        client = PiPrintClient()
        client.start()

        with self.assertRaisesRegex(RuntimeError, "Shutdown requested"):
            client.invoke_sync("hello")

        popen_mock.assert_not_called()

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.subprocess.Popen")
    def test_invoke_sync_terminates_process_on_shutdown_during_poll(
        self, popen_mock: MagicMock, sleep_mock: MagicMock
    ) -> None:
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.return_value = None
        popen_mock.return_value = proc

        def _trigger_shutdown(*_args: object, **_kw: object) -> None:
            _shutdown_mod.request_shutdown()

        sleep_mock.side_effect = _trigger_shutdown

        client = PiPrintClient(timeout_seconds=300)
        client.start()

        with self.assertRaisesRegex(RuntimeError, "Shutdown requested"):
            client.invoke_sync("hello")

        proc.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
