"""End-to-end test: hand-written solve_inverse_design → runner → gt_eval → metrics.

This is the Stage 0 gate test. It proves the full deterministic pipeline:
  1. Agent writes solve_inverse_design.py (simulated by a temp file)
  2. codegen_runner imports and executes it
  3. gt_eval scores the SolverResults with eval()-based lambda expressions
  4. Correct metrics come out

No LLM, no network, no real TorchRDIT solver — uses synthetic real SolverResults objects.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import torch


def _write_temp_module(content: str, *, tmpdir: str, filename: str = "solve_inverse_design.py") -> Path:
    path = Path(tmpdir) / filename
    path.write_text(textwrap.dedent(content))
    return path


class TestEndToEndRunnerGtEval(unittest.TestCase):
    """End-to-end: runner + gt_eval scoring."""

    def test_full_pipeline_success(self):
        """Happy path: agent code → runner → gt_eval → success_goal=True."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(kx=vec.clone(), ky=vec.clone(), kinc=vec.clone(), kzref=vec.clone(), kztrn=vec.clone())
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
                    return _build_results(reflection_val=0.1, transmission_val=0.9)
            """, tmpdir=tmpdir)

            runner_result = run_codegen(code_path=code_path, device="cpu")

            self.assertIsNone(runner_result.error)
            self.assertIsNotNone(runner_result.solver_results)

            # Step 2: gt_eval scores.
            gt_eval_spec = {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8},
                    {"expr": "r.reflection[0].item()", "operation": "<=", "target": 0.15},
                ],
            }
            gt_result = evaluate_gt_eval(
                gt_eval_spec,
                runner_result.solver_results,
                compile_ok=True,
                solver_ok=True,
            )

            # Step 3: Verify metrics.
            self.assertTrue(gt_result["success_exec"])
            self.assertTrue(gt_result["success_goal"])
            self.assertTrue(gt_result["meets_gt_criteria"])
            self.assertEqual(len(gt_result["criteria"]), 2)
            self.assertTrue(gt_result["criteria"][0]["passed"])  # T >= 0.8
            self.assertTrue(gt_result["criteria"][1]["passed"])  # R <= 0.15

    def test_full_pipeline_criteria_failure(self):
        """Agent code runs but criteria not met → success_goal=False."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(kx=vec.clone(), ky=vec.clone(), kinc=vec.clone(), kzref=vec.clone(), kztrn=vec.clone())
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
                    return _build_results(reflection_val=0.4, transmission_val=0.6)
            """, tmpdir=tmpdir)

            runner_result = run_codegen(code_path=code_path, device="cpu")
            self.assertIsNone(runner_result.error)

            gt_eval_spec = {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8},
                ],
            }
            gt_result = evaluate_gt_eval(
                gt_eval_spec,
                runner_result.solver_results,
                compile_ok=True,
                solver_ok=True,
            )

            self.assertTrue(gt_result["success_exec"])
            self.assertFalse(gt_result["success_goal"])
            self.assertFalse(gt_result["criteria"][0]["passed"])

    def test_full_pipeline_runtime_error(self):
        """Agent code crashes → runner captures error → gt_eval with compile_ok=False."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                def solve_inverse_design(*, device: str = "cpu"):
                    raise ValueError("Simulation failed")
            """, tmpdir=tmpdir)

            runner_result = run_codegen(code_path=code_path, device="cpu")
            self.assertIsNotNone(runner_result.error)
            self.assertEqual(runner_result.error_type, "runtime_error")

            gt_eval_spec = {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8},
                ],
            }
            # When runner fails, we pass compile_ok=False and solver_ok=False.
            gt_result = evaluate_gt_eval(
                gt_eval_spec,
                None,
                compile_ok=False,
                solver_ok=False,
            )

            self.assertFalse(gt_result["success_exec"])
            self.assertFalse(gt_result["success_goal"])

    def test_full_pipeline_diffraction_expr(self):
        """Use diffraction order expressions end-to-end."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(transmission_val: float, diffraction_value: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(kx=vec.clone(), ky=vec.clone(), kinc=vec.clone(), kzref=vec.clone(), kztrn=vec.clone())
                    td = torch.zeros(1, 3, 3)
                    td[0, 0, 1] = diffraction_value
                    return SolverResults(
                        reflection=torch.tensor([0.1], dtype=torch.float32),
                        transmission=torch.tensor([transmission_val], dtype=torch.float32),
                        reflection_diffraction=torch.zeros(1, 3, 3),
                        transmission_diffraction=td,
                        reflection_field=field,
                        transmission_field=field,
                        structure_matrix=smat,
                        wave_vectors=wave,
                    )

                def solve_inverse_design(*, device: str = "cpu"):
                    return _build_results(transmission_val=0.85, diffraction_value=0.7)
            """, tmpdir=tmpdir)

            runner_result = run_codegen(code_path=code_path, device="cpu")
            self.assertIsNone(runner_result.error)

            gt_eval_spec = {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission_diffraction[0, 0, 1].item()", "operation": ">=", "target": 0.6},
                ],
            }
            gt_result = evaluate_gt_eval(
                gt_eval_spec,
                runner_result.solver_results,
            )

            self.assertTrue(gt_result["success_exec"])
            self.assertTrue(gt_result["success_goal"])

    def test_margin_summary_end_to_end(self):
        """summarize_normalized_margins works with runner + gt_eval output."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval, summarize_normalized_margins

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(kx=vec.clone(), ky=vec.clone(), kinc=vec.clone(), kzref=vec.clone(), kztrn=vec.clone())
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
                    return _build_results(reflection_val=0.15, transmission_val=0.75)
            """, tmpdir=tmpdir)

            runner_result = run_codegen(code_path=code_path, device="cpu")
            gt_eval_spec = {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8},
                ],
            }
            gt_result = evaluate_gt_eval(gt_eval_spec, runner_result.solver_results)
            margins = summarize_normalized_margins(gt_result["criteria"])

            best_margin = margins["best_margin"]
            self.assertIsNotNone(best_margin)
            assert best_margin is not None
            self.assertLess(best_margin, 0.0)
            violation_norm = margins["criteria_violation_norm"]
            self.assertIsNotNone(violation_norm)
            assert violation_norm is not None
            self.assertGreater(violation_norm, 0.0)

    def test_cpu_device_explicit_contract(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.codegen_runner import run_codegen
        from evo_metaoptics.meta_design.gt_eval import evaluate_gt_eval

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = _write_temp_module("""\
                import torch
                from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors

                def _build_results(reflection_val: float, transmission_val: float):
                    vec = torch.zeros(1)
                    mat = torch.zeros(1, 1, 1)
                    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
                    smat = ScatteringMatrix(S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone())
                    wave = WaveVectors(kx=vec.clone(), ky=vec.clone(), kinc=vec.clone(), kzref=vec.clone(), kztrn=vec.clone())
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
                    if device == "cpu":
                        return _build_results(reflection_val=0.05, transmission_val=0.95)
                    else:
                        return _build_results(reflection_val=0.1, transmission_val=0.9)
            """, tmpdir=tmpdir)

            runner_result = run_codegen(code_path=code_path, device="cpu")

            self.assertIsNone(runner_result.error)
            self.assertIsNotNone(runner_result.solver_results)

            gt_eval_spec = {
                "wavelength_um": [1.55],
                "criteria": [
                    {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.9},
                    {"expr": "r.reflection[0].item()", "operation": "<=", "target": 0.1},
                ],
            }
            gt_result = evaluate_gt_eval(
                gt_eval_spec,
                runner_result.solver_results,
                compile_ok=True,
                solver_ok=True,
            )

            self.assertTrue(gt_result["success_exec"])
            self.assertTrue(gt_result["success_goal"])


if __name__ == "__main__":
    unittest.main()
