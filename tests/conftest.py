"""Shared pytest fixtures for MCE tests.

Provides:
- mock_pi_session: MockAgentSession fixture for testing agent interactions
- make_solver_results: Helper to construct SolverResults objects
- make_sample: Helper to construct Sample objects
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure top-level repo package imports work when pytest executes from tests/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest
import torch

from evo_metaoptics.mce.agent_session import MockAgentSession
from evo_metaoptics.mce_env.base import Sample


@pytest.fixture
def mock_pi_session() -> MockAgentSession:
    """Fixture: MockAgentSession for testing Pi agent interactions.
    
    Returns a MockAgentSession with a temporary working directory.
    The session simulates Pi agent behavior without requiring a real backend.
    
    Yields:
        MockAgentSession: Mock agent session instance.
    """
    temp_dir = tempfile.TemporaryDirectory()
    session = MockAgentSession(cwd=Path(temp_dir.name))
    yield session
    # Cleanup
    temp_dir.cleanup()


def make_solver_results(
    reflection_val: float = 0.1,
    transmission_val: float = 0.9,
) -> Any:
    """Helper: Construct a SolverResults object for testing.
    
    Creates a minimal but valid SolverResults object with the given
    reflection and transmission values. Reuses the pattern from
    test_codegen_runner_e2e.py.
    
    Args:
        reflection_val: Reflection efficiency value (default 0.1).
        transmission_val: Transmission efficiency value (default 0.9).
        
    Returns:
        SolverResults: A synthetic SolverResults object.
    """
    from torchrdit.results import (
        FieldComponents,
        ScatteringMatrix,
        SolverResults,
        WaveVectors,
    )

    vec = torch.zeros(1)
    mat = torch.zeros(1, 1, 1)
    field = FieldComponents(x=vec.clone(), y=vec.clone(), z=vec.clone())
    smat = ScatteringMatrix(
        S11=mat.clone(), S12=mat.clone(), S21=mat.clone(), S22=mat.clone()
    )
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


def make_sample(
    sample_id: int = 7,
    question: str = "Design a high-transmission device.",
    gt_eval: dict[str, Any] | None = None,
) -> Sample:
    """Helper: Construct a Sample object for testing.
    
    Creates a Sample with the given parameters. Reuses the pattern from
    test_metaoptics_env_codegen.py.
    
    Args:
        sample_id: Sample ID (default 7).
        question: Question/query string (default "Design a high-transmission device.").
        gt_eval: Ground-truth evaluation spec dict. If None, uses a default v7 spec.
        
    Returns:
        Sample: A Sample object with the given parameters.
    """
    if gt_eval is None:
        gt_eval = {
            "wavelength_um": [1.55],
            "criteria": [
                {"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.8}
            ],
        }

    return Sample(
        id=sample_id,
        question=question,
        extras={"gt_eval": gt_eval},
    )
