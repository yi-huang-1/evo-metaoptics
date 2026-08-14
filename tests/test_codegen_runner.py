"""TDD tests for codegen_runner — importlib-based execution of agent-written code.

Decision D9: In-process importlib with threading-based timeout.
Decision D10: Agent writes .py via filesystem backend; runner reads from backend root.
Decision D2/D14: Function contract: solve_inverse_design(*, device: str = "cpu") -> SolverResults.

codegen_runner responsibilities:
1. Read agent-written .py file from a given path
2. Import the module dynamically via importlib
3. Call solve_inverse_design(device=device) with threading timeout
4. Validate return type is SolverResults
5. Capture structured errors for agent feedback
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import io
import contextlib
import logging
import sys
import textwrap
import tempfile
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


_VALID_RESULTS_BODY = '''
import torch
from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors


def _build_results(reflection_val: float, transmission_val: float):
    vec = torch.zeros(1)
    mat = torch.zeros(1, 1, 1)
    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
    wave = WaveVectors(
        kx=vec.clone(),
        ky=vec.clone(),
        kinc=vec.clone(),
        kzref=vec.clone(),
        kztrn=vec.clone(),
    )
    return SolverResults(
        reflection=torch.tensor([reflection_val], dtype=torch.float32),
        transmission=torch.tensor([transmission_val], dtype=torch.float32),
        reflection_diffraction=torch.zeros(1, 3, 3),
        transmission_diffraction=torch.zeros(1, 3, 3),
        reflection_field=field,
        transmission_field=field,
        structure_matrix=smat,
        wave_vectors=wave,
    )
'''


@contextlib.contextmanager
def _isolated_codegen_loggers() -> Iterator[None]:
    states = []
    for name in ("mce_main", "eval"):
        logger = logging.getLogger(name)
        states.append((logger, logger.level, list(logger.handlers), logger.propagate))
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = False
    try:
        yield
    finally:
        for logger, level, handlers, propagate in states:
            logger.handlers.clear()
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


def _write_temp_module(content: str, *, tmpdir: str, filename: str = "solve_inverse_design.py") -> Path:
    """Write a temporary Python module file and return its path."""
    path = Path(tmpdir) / filename
    path.write_text(textwrap.dedent(content))
    return path


def _build_guarded_module(*, prelude: str = "", function_body: str) -> str:
    parts = ["import os", _VALID_RESULTS_BODY.strip()]
    if prelude:
        parts.append(textwrap.dedent(prelude).strip())
    body = textwrap.indent(textwrap.dedent(function_body).strip() + "\n", "    ").rstrip()
    parts.append(f"def solve_inverse_design(*, device: str = \"cuda\"):\n{body}")
    return "\n\n".join(part for part in parts if part) + "\n"


class TestCodegenRunnerImport(unittest.TestCase):
    """Test that runner can import and call agent-written modules."""

    def _run(self, code_path, device="cpu", timeout_s=10.0):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        return run_codegen(code_path=code_path, device=device, timeout_s=timeout_s)


class TestCodegenRunnerExecution(unittest.TestCase):
    def _run(self, code_path, device="cpu", timeout_s=10.0):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        return run_codegen(code_path=code_path, device=device, timeout_s=timeout_s)

    def test_valid_function_returns_result(self):
        """A valid solve_inverse_design function should produce a successful RunnerResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=torch.zeros(1, 3, 3),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8)
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNone(result.error)
            self.assertIsNone(result.error_type)
            self.assertIsNotNone(result.solver_results)
            self.assertGreater(result.execution_time_s, 0.0)

    def test_syntax_error(self):
        """Syntax errors in agent code should be caught and reported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                def solve_inverse_design(*, device: str = "cpu")
                    return None  # missing colon above
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "syntax_error")
            self.assertIsNone(result.solver_results)
            self.assertIsNotNone(result.traceback)

    def test_import_error(self):
        """Import errors in agent code should be caught and reported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                from nonexistent_module import something

                def solve_inverse_design(*, device: str = "cpu"):
                    return something(device)
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "import_error")
            self.assertIsNone(result.solver_results)

    def test_runtime_error(self):
        """Runtime exceptions during function execution should be caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                def solve_inverse_design(*, device: str = "cpu"):
                    raise RuntimeError("Solver exploded")
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "runtime_error")
            self.assertIn("Solver exploded", result.error or "")
            self.assertIsNone(result.solver_results)

    def test_missing_function(self):
        """Module without solve_inverse_design should report missing function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                def some_other_function(*, device: str = "cpu"):
                    return None
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "missing_function")
            self.assertIsNone(result.solver_results)

    def test_timeout(self):
        """Functions that take too long should be terminated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import time

                def solve_inverse_design(*, device: str = "cpu"):
                    time.sleep(60)  # way too long
                    return None
            """, tmpdir=tmpdir)

            result = self._run(code_path, timeout_s=1.0)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "timeout")
            self.assertIsNone(result.solver_results)
            # Should not take much longer than the timeout
            self.assertLess(result.execution_time_s, 5.0)

    def test_wrong_return_type(self):
        """Function returning wrong type should report invalid_output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                def solve_inverse_design(*, device: str = "cpu"):
                    return {"transmission": [0.8]}  # should be SolverResults, not dict
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "invalid_output")
            self.assertIsNone(result.solver_results)

    def test_duck_typed_object_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                class FakeResults:
                    def __init__(self):
                        self.reflection = [0.1]
                        self.transmission = [0.9]

                def solve_inverse_design(*, device: str = "cpu"):
                    return FakeResults()
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "invalid_output")
            self.assertIsNone(result.solver_results)

    def test_invalid_output_error_includes_module_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                class SolverResults:
                    pass

                def solve_inverse_design(*, device: str = "cpu"):
                    return SolverResults()
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertEqual(result.error_type, "invalid_output")
            self.assertIsNotNone(result.error)
            self.assertIn("_codegen_", result.error or "")
            self.assertIn(".SolverResults", result.error or "")

    def test_invalid_output_error_mentions_solver_solve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                class SolverResults:
                    pass

                def solve_inverse_design(*, device: str = "cpu"):
                    return SolverResults()
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertEqual(result.error_type, "invalid_output")
            self.assertIsNotNone(result.error)
            self.assertIn("solver.solve", result.error or "")

    def test_returns_none(self):
        """Function returning None should report invalid_output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                def solve_inverse_design(*, device: str = "cpu"):
                    return None
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error_type, "invalid_output")
            self.assertIsNone(result.solver_results)

    def test_nonexistent_file(self):
        """Missing code file should report file_not_found."""
        result = self._run(Path("/nonexistent/solve_inverse_design.py"))

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error_type, "file_not_found")
        self.assertIsNone(result.solver_results)
    def test_valid_function_logs_solver_results_device(self):
        """Runner logs solver_results_device for successful execution."""
        import io

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=torch.zeros(1, 3, 3),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8)
            """, tmpdir=tmpdir)

            log_capture = io.StringIO()
            module_logger = logging.getLogger("evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner")
            module_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.INFO)
            module_logger.addHandler(handler)
            try:
                result = self._run(code_path)
            finally:
                module_logger.removeHandler(handler)

            self.assertIsNone(result.error)
            self.assertIsNone(result.error_type)
            self.assertIsNotNone(result.solver_results)

            output = log_capture.getvalue()
            self.assertIn("run_codegen: requested_device=cpu, solver_results_device=", output)


class TestCodegenRunnerLogging(unittest.TestCase):
    """Verify logger emission for durable smoke evidence."""

    def _run(self, code_path, device="cpu", timeout_s=10.0):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        return run_codegen(code_path=code_path, device=device, timeout_s=timeout_s)

    def test_valid_function_logs_to_active_logger(self):
        """Device log is emitted via active logger discovery mechanism."""
        with tempfile.TemporaryDirectory() as tmpdir, _isolated_codegen_loggers():
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=torch.zeros(1, 3, 3),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8)
            """, tmpdir=tmpdir)

            log_capture = io.StringIO()
            test_logger = logging.getLogger("mce_main")
            test_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.INFO)
            test_logger.addHandler(handler)

            result = self._run(code_path)

            self.assertIsNone(result.error)
            self.assertIsNone(result.error_type)
            self.assertIsNotNone(result.solver_results)

            log_output = log_capture.getvalue()
            self.assertIn("run_codegen: requested_device=cpu, solver_results_device=", log_output)

    def test_module_logger_fallback_when_no_active_logger(self):
        """Device log goes to module-level logger when no active logger is configured."""
        with tempfile.TemporaryDirectory() as tmpdir, _isolated_codegen_loggers():
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=torch.zeros(1, 3, 3),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8)
            """, tmpdir=tmpdir)

            log_capture = io.StringIO()
            module_logger = logging.getLogger("evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner")
            module_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.INFO)
            module_logger.addHandler(handler)
            try:
                result = self._run(code_path)
            finally:
                module_logger.removeHandler(handler)

            self.assertIsNone(result.error)
            output = log_capture.getvalue()
            self.assertIn("run_codegen: requested_device=cpu, solver_results_device=", output)


class TestCodegenRunnerSmokeSkipSolve(unittest.TestCase):
    def _run(self, code_path, device="cpu", timeout_s=10.0):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        return run_codegen(code_path=code_path, device=device, timeout_s=timeout_s)

    def test_smoke_skip_solve_flag_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module(
                _build_guarded_module(
                    function_body="return _build_results(reflection_val=0.3, transmission_val=0.7)",
                ),
                tmpdir=tmpdir,
            )

            result = self._run(code_path)

            self.assertIsNone(result.error)
            self.assertIsNone(result.error_type)
            self.assertIsNotNone(result.solver_results)
            solver_results = result.solver_results
            assert solver_results is not None
            self.assertAlmostEqual(solver_results.transmission[0].item(), 0.7, places=6)
            self.assertEqual(solver_results.transmission.device.type, "cpu")
            self.assertNotIn("smoke_skip_solve", solver_results.raw_data)

    def test_smoke_skip_solve_returns_results_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = Path(tmpdir) / "initialized.txt"
            code_path = _write_temp_module(
                f'''\
                    import numpy as np
                    import torch
                    from pathlib import Path

                    from torchrdit.constants import Algorithm, Precision
                    from torchrdit.solver import FourierBaseSolver, get_solver_builder
                    from torchrdit.utils import create_material

                    def solve_inverse_design(*, device: str = "cpu"):
                        del device
                        builder = get_solver_builder()
                        builder.with_algorithm(Algorithm.RCWA)
                        builder.with_precision(Precision.SINGLE)
                        builder.with_wavelengths(np.array([1.55]))
                        builder.with_length_unit("um")
                        builder.with_real_dimensions([8, 8])
                        builder.with_k_dimensions([3, 3])
                        solver = builder.build()
                        air = create_material(name="air", permittivity=1.0)
                        film = create_material(name="film", permittivity=11.7)
                        solver.add_materials(material_list=[air, film])
                        solver.update_ref_material("air")
                        solver.add_layer(
                            material_name="film",
                            thickness=torch.tensor(0.22, dtype=torch.float32),
                            is_homogeneous=True,
                        )
                        solver.update_trn_material("air")
                        Path(r"{marker_path}").write_text("initialized")
                        source = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
                        assert isinstance(solver, FourierBaseSolver)
                        return solver.solve(source)
                ''',
                tmpdir=tmpdir,
            )

            with patch.dict(
                "os.environ",
                {
                    "EVO_METAOPTICS_SMOKE_SKIP_SOLVE": "1",
                },
                clear=False,
            ):
                result = self._run(code_path)

            self.assertIsNone(result.error, msg=f"unexpected error: {result.error}")
            self.assertIsNone(result.error_type)
            self.assertTrue(marker_path.is_file())
            self.assertEqual(marker_path.read_text(), "initialized")
            self.assertIsNotNone(result.solver_results)
            solver_results = result.solver_results
            assert solver_results is not None
            self.assertEqual(solver_results.transmission.device.type, "cpu")
            self.assertEqual(solver_results.reflection.device.type, "cpu")
            self.assertEqual(solver_results.transmission_diffraction.device.type, "cpu")
            self.assertTrue(solver_results.raw_data.get("smoke_skip_solve"))


class TestRunnerResultDataclass(unittest.TestCase):
    """Verify the RunnerResult dataclass structure."""

    def test_runner_result_fields(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import RunnerResult

        result = RunnerResult(
            solver_results=None,
            error="test error",
            error_type="runtime_error",
            execution_time_s=1.5,
            traceback="Traceback ...",
        )
        self.assertIsNone(result.solver_results)
        self.assertEqual(result.error, "test error")
        self.assertEqual(result.error_type, "runtime_error")
        self.assertAlmostEqual(result.execution_time_s, 1.5)
        self.assertEqual(result.traceback, "Traceback ...")

    def test_runner_result_success(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import RunnerResult

        mock_results = MagicMock()
        result = RunnerResult(
            solver_results=mock_results,
            error=None,
            error_type=None,
            execution_time_s=2.3,
            traceback=None,
        )
        self.assertIs(result.solver_results, mock_results)
        self.assertIsNone(result.error)
        self.assertIsNone(result.error_type)


class TestCleanBreakDeviceContract(unittest.TestCase):

    def _run(self, code_path, device="cpu", timeout_s=10.0):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        return run_codegen(code_path=code_path, device=device, timeout_s=timeout_s)

    def test_keyword_only_device_signature_with_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=torch.zeros(1, 3, 3),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8)
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNone(result.error, msg=f"Expected success but got error: {result.error}")
            self.assertIsNone(result.error_type)
            self.assertIsNotNone(result.solver_results)

    def test_device_parameter_cpu_safe_propagation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float, device: str):
                    dev = torch.device(device)
                    vec = torch.zeros(1, device=dev)
                    mat = torch.zeros(1, 1, 1, device=dev)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32, device=dev),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32, device=dev),
                        reflection_diffraction=torch.zeros(1, 3, 3, device=dev),
                        transmission_diffraction=torch.zeros(1, 3, 3, device=dev),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8, device=device)
            """, tmpdir=tmpdir)

            result = self._run(code_path)

            self.assertIsNone(result.error, msg=f"Expected success but got error: {result.error}")
            self.assertIsNone(result.error_type)
            self.assertIsNotNone(result.solver_results)
            solver_results = result.solver_results
            assert solver_results is not None
            self.assertEqual(solver_results.transmission.device.type, "cpu")

    def test_explicit_cpu_device_override_cpu_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float, device: str):
                    dev = torch.device(device)
                    vec = torch.zeros(1, device=dev)
                    mat = torch.zeros(1, 1, 1, device=dev)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32, device=dev),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32, device=dev),
                        reflection_diffraction=torch.zeros(1, 3, 3, device=dev),
                        transmission_diffraction=torch.zeros(1, 3, 3, device=dev),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8, device=device)
            """, tmpdir=tmpdir)

            result = self._run(code_path, device="cpu")

            self.assertIsNone(result.error, msg=f"Expected success but got error: {result.error}")
            self.assertIsNone(result.error_type)
            self.assertIsNotNone(result.solver_results)
            solver_results = result.solver_results
            assert solver_results is not None
            self.assertEqual(solver_results.transmission.device.type, "cpu")


class TestModuleIsolation(unittest.TestCase):
    """Test that runner properly isolates imported modules."""

    def _run(self, code_path, device="cpu", timeout_s=10.0):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        return run_codegen(code_path=code_path, device=device, timeout_s=timeout_s)

    def test_repeated_imports_different_code(self):
        """Running different code files should produce different results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First module: transmission = 0.8
            code_path_1 = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=torch.zeros(1, 3, 3),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.2, transmission_val=0.8)
            """, tmpdir=tmpdir, filename="module_a.py")

            # Second module: transmission = 0.5
            code_path_2 = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(
                        kx=vec.clone(),
                        ky=vec.clone(),
                        kinc=vec.clone(),
                        kzref=vec.clone(),
                        kztrn=vec.clone(),
                    )
                    return SolverResults(
                        reflection=torch.tensor([reflection_val], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=torch.zeros(1, 3, 3),
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(reflection_val=0.5, transmission_val=0.5)
            """, tmpdir=tmpdir, filename="module_b.py")

            result_1 = self._run(code_path_1)
            result_2 = self._run(code_path_2)

            self.assertIsNone(result_1.error)
            self.assertIsNone(result_2.error)
            solver_results_1 = result_1.solver_results
            solver_results_2 = result_2.solver_results
            self.assertIsNotNone(solver_results_1)
            self.assertIsNotNone(solver_results_2)
            assert solver_results_1 is not None
            assert solver_results_2 is not None
            # Results should differ
            self.assertAlmostEqual(
                solver_results_1.transmission[0].item(), 0.8, places=5
            )
            self.assertAlmostEqual(
                solver_results_2.transmission[0].item(), 0.5, places=5
            )


if __name__ == "__main__":
    unittest.main()
