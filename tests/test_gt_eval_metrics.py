"""Tests for gt_eval_metrics.py — Semantic Metric Registry (Milestone 1).

Uses real torchrdit.results.SolverResults objects to test metric extraction
against the actual tensor shapes and API conventions.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Optional

import torch
from torchrdit.results import FieldComponents, ScatteringMatrix, SolverResults, WaveVectors


# ---------------------------------------------------------------------------
# Test fixtures — realistic SolverResults objects
# ---------------------------------------------------------------------------

def _make_field_components(
    *,
    n_freqs: int = 1,
    harmonics: tuple[int, int] = (3, 3),
    x_vals: Optional[torch.Tensor] = None,
    y_vals: Optional[torch.Tensor] = None,
    z_vals: Optional[torch.Tensor] = None,
) -> FieldComponents:
    """Build FieldComponents with shape (n_freqs, H0, H1) for diffraction fields.

    If x_vals/y_vals/z_vals are provided, they should be complex tensors of shape
    (n_freqs, H0, H1). Otherwise zeros are used.
    """
    h0, h1 = harmonics
    shape = (n_freqs, h0, h1)
    return FieldComponents(
        x=x_vals if x_vals is not None else torch.zeros(shape, dtype=torch.complex64),
        y=y_vals if y_vals is not None else torch.zeros(shape, dtype=torch.complex64),
        z=z_vals if z_vals is not None else torch.zeros(shape, dtype=torch.complex64),
    )


def _make_solver_results(
    *,
    n_freqs: int = 1,
    harmonics: tuple[int, int] = (3, 3),
    transmission: Optional[torch.Tensor] = None,
    reflection: Optional[torch.Tensor] = None,
    transmission_diffraction: Optional[torch.Tensor] = None,
    reflection_diffraction: Optional[torch.Tensor] = None,
    transmission_field: Optional[FieldComponents] = None,
    reflection_field: Optional[FieldComponents] = None,
    n_sources: int = 1,
    is_batched: Optional[bool] = None,
) -> SolverResults:
    """Build a SolverResults for testing with correct shapes.

    For single source:
        transmission/reflection: (n_freqs,)
        diffraction: (n_freqs, H0, H1)
        field components: (n_freqs, H0, H1) complex

    For batched (n_sources > 1):
        transmission/reflection: (n_sources, n_freqs)
        diffraction: (n_sources, n_freqs, H0, H1)
        field components: (n_sources, n_freqs, H0, H1) complex
    """
    h0, h1 = harmonics
    actual_n_sources = n_sources

    if actual_n_sources == 1 and not is_batched:
        # Single source shapes
        t = transmission if transmission is not None else torch.tensor([0.8] * n_freqs)
        r = reflection if reflection is not None else torch.tensor([0.2] * n_freqs)
        td = transmission_diffraction if transmission_diffraction is not None else torch.zeros(n_freqs, h0, h1)
        rd = reflection_diffraction if reflection_diffraction is not None else torch.zeros(n_freqs, h0, h1)
        tf = transmission_field if transmission_field is not None else _make_field_components(n_freqs=n_freqs, harmonics=harmonics)
        rf = reflection_field if reflection_field is not None else _make_field_components(n_freqs=n_freqs, harmonics=harmonics)
    else:
        # Batched shapes
        t = transmission if transmission is not None else torch.zeros(actual_n_sources, n_freqs) + 0.8
        r = reflection if reflection is not None else torch.zeros(actual_n_sources, n_freqs) + 0.2
        td = transmission_diffraction if transmission_diffraction is not None else torch.zeros(actual_n_sources, n_freqs, h0, h1)
        rd = reflection_diffraction if reflection_diffraction is not None else torch.zeros(actual_n_sources, n_freqs, h0, h1)
        if transmission_field is not None:
            tf = transmission_field
        else:
            shape = (actual_n_sources, n_freqs, h0, h1)
            tf = FieldComponents(
                x=torch.zeros(shape, dtype=torch.complex64),
                y=torch.zeros(shape, dtype=torch.complex64),
                z=torch.zeros(shape, dtype=torch.complex64),
            )
        if reflection_field is not None:
            rf = reflection_field
        else:
            shape = (actual_n_sources, n_freqs, h0, h1)
            rf = FieldComponents(
                x=torch.zeros(shape, dtype=torch.complex64),
                y=torch.zeros(shape, dtype=torch.complex64),
                z=torch.zeros(shape, dtype=torch.complex64),
            )

    vec = torch.zeros(1)
    smat = ScatteringMatrix(
        S11=torch.zeros(1, 1, 1),
        S12=torch.zeros(1, 1, 1),
        S21=torch.zeros(1, 1, 1),
        S22=torch.zeros(1, 1, 1),
    )
    wave = WaveVectors(kx=vec, ky=vec, kinc=vec, kzref=vec, kztrn=vec)

    return SolverResults(
        reflection=r,
        transmission=t,
        reflection_diffraction=rd,
        transmission_diffraction=td,
        reflection_field=rf,
        transmission_field=tf,
        structure_matrix=smat,
        wave_vectors=wave,
        n_sources=actual_n_sources,
        _is_batched=is_batched,
    )


def _make_batched_two_sources(
    *,
    n_freqs: int = 1,
    harmonics: tuple[int, int] = (3, 3),
    transmission_s0: Optional[torch.Tensor] = None,
    transmission_s1: Optional[torch.Tensor] = None,
    reflection_s0: Optional[torch.Tensor] = None,
    reflection_s1: Optional[torch.Tensor] = None,
) -> SolverResults:
    """Build a batched SolverResults with 2 sources (e.g. TE + TM)."""
    t0 = transmission_s0 if transmission_s0 is not None else torch.tensor([0.8] * n_freqs)
    t1 = transmission_s1 if transmission_s1 is not None else torch.tensor([0.6] * n_freqs)
    r0 = reflection_s0 if reflection_s0 is not None else torch.tensor([0.2] * n_freqs)
    r1 = reflection_s1 if reflection_s1 is not None else torch.tensor([0.4] * n_freqs)

    transmission = torch.stack([t0, t1])  # (2, n_freqs)
    reflection = torch.stack([r0, r1])

    h0, h1 = harmonics
    td = torch.zeros(2, n_freqs, h0, h1)
    rd = torch.zeros(2, n_freqs, h0, h1)

    # Build field components with known complex values for phase/amplitude testing
    shape = (2, n_freqs, h0, h1)
    tf = FieldComponents(
        x=torch.zeros(shape, dtype=torch.complex64),
        y=torch.zeros(shape, dtype=torch.complex64),
        z=torch.zeros(shape, dtype=torch.complex64),
    )
    rf = FieldComponents(
        x=torch.zeros(shape, dtype=torch.complex64),
        y=torch.zeros(shape, dtype=torch.complex64),
        z=torch.zeros(shape, dtype=torch.complex64),
    )

    vec = torch.zeros(1)
    smat = ScatteringMatrix(
        S11=torch.zeros(1, 1, 1),
        S12=torch.zeros(1, 1, 1),
        S21=torch.zeros(1, 1, 1),
        S22=torch.zeros(1, 1, 1),
    )
    wave = WaveVectors(kx=vec, ky=vec, kinc=vec, kzref=vec, kztrn=vec)

    return SolverResults(
        reflection=reflection,
        transmission=transmission,
        reflection_diffraction=rd,
        transmission_diffraction=td,
        reflection_field=rf,
        transmission_field=tf,
        structure_matrix=smat,
        wave_vectors=wave,
        n_sources=2,
        _is_batched=True,
    )


# ---------------------------------------------------------------------------
# TestMetricRegistry
# ---------------------------------------------------------------------------

class TestMetricRegistry(unittest.TestCase):
    """Test registry registration, lookup, and listing."""

    def test_register_and_lookup(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import get_metric
        fn = get_metric("total_transmission")
        self.assertIsNotNone(fn)
        self.assertTrue(callable(fn))

    def test_unknown_metric_raises(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import get_metric
        with self.assertRaises(KeyError) as ctx:
            get_metric("nonexistent_metric")
        self.assertIn("nonexistent_metric", str(ctx.exception))

    def test_list_registered_metrics(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import list_metrics
        metrics = list_metrics()
        expected = [
            "order_reflection_efficiency",
            "order_transmission_efficiency",
            "total_reflection",
            "total_transmission",
            "zero_order_reflection_amplitude",
            "zero_order_reflection_phase_deg",
            "zero_order_transmission_amplitude",
            "zero_order_transmission_phase_deg",
        ]
        self.assertEqual(metrics, expected)


# ---------------------------------------------------------------------------
# TestSourceResolution
# ---------------------------------------------------------------------------

class TestSourceResolution(unittest.TestCase):
    """Test _resolve_source helper."""

    def _resolve(self, r, source_index):
        from evo_metaoptics.meta_design.gt_eval_metrics import _resolve_source
        return _resolve_source(r, source_index)

    def test_single_source_no_index(self):
        """Unbatched result with source_index=None → returns self."""
        r = _make_solver_results()
        result = self._resolve(r, None)
        self.assertIs(result, r)

    def test_single_source_index_zero(self):
        """Unbatched result with source_index=0 → returns self."""
        r = _make_solver_results()
        result = self._resolve(r, 0)
        self.assertIs(result, r)  # SolverResults.__getitem__(0) returns self for single

    def test_batched_no_index_raises(self):
        """Batched result without source_index → MetricError."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = _make_batched_two_sources()
        with self.assertRaises(MetricError) as ctx:
            self._resolve(r, None)
        self.assertIn("2 sources", str(ctx.exception))
        self.assertIn("source_index", str(ctx.exception))

    def test_batched_with_index(self):
        """Batched result with valid source_index → unbatched result."""
        r = _make_batched_two_sources(
            reflection_s0=torch.tensor([0.9]),
            reflection_s1=torch.tensor([0.1]),
        )
        r0 = self._resolve(r, 0)
        r1 = self._resolve(r, 1)
        self.assertFalse(r0.is_batched)
        self.assertFalse(r1.is_batched)
        self.assertAlmostEqual(float(r0.reflection[0]), 0.9, places=5)
        self.assertAlmostEqual(float(r1.reflection[0]), 0.1, places=5)

    def test_source_index_out_of_range(self):
        """source_index >= n_sources → MetricError."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = _make_batched_two_sources()
        with self.assertRaises(MetricError) as ctx:
            self._resolve(r, 2)
        self.assertIn("out of range", str(ctx.exception))

    def test_source_index_negative_raises(self):
        """Negative source_index → MetricError."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = _make_batched_two_sources()
        with self.assertRaises(MetricError) as ctx:
            self._resolve(r, -1)
        self.assertIn("out of range", str(ctx.exception))


# ---------------------------------------------------------------------------
# TestTotalTransmission
# ---------------------------------------------------------------------------

class TestTotalTransmission(unittest.TestCase):
    """Test total_transmission metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "total_transmission", params)

    def test_single_wavelength(self):
        r = _make_solver_results(transmission=torch.tensor([0.85]))
        value = self._extract(r, wavelength_index=0)
        self.assertAlmostEqual(value, 0.85, places=5)

    def test_multi_wavelength(self):
        r = _make_solver_results(
            n_freqs=3,
            transmission=torch.tensor([0.7, 0.8, 0.9]),
        )
        self.assertAlmostEqual(self._extract(r, wavelength_index=0), 0.7, places=5)
        self.assertAlmostEqual(self._extract(r, wavelength_index=1), 0.8, places=5)
        self.assertAlmostEqual(self._extract(r, wavelength_index=2), 0.9, places=5)

    def test_wavelength_index_oob(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = _make_solver_results(transmission=torch.tensor([0.8]))
        with self.assertRaises(MetricError) as ctx:
            self._extract(r, wavelength_index=1)
        self.assertIn("out of bounds", str(ctx.exception))

    def test_batched_with_source_index(self):
        r = _make_batched_two_sources(
            transmission_s0=torch.tensor([0.9]),
            transmission_s1=torch.tensor([0.3]),
        )
        self.assertAlmostEqual(
            self._extract(r, wavelength_index=0, source_index=0), 0.9, places=5
        )
        self.assertAlmostEqual(
            self._extract(r, wavelength_index=0, source_index=1), 0.3, places=5
        )

    def test_batched_without_source_index_raises(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = _make_batched_two_sources()
        with self.assertRaises(MetricError):
            self._extract(r, wavelength_index=0)


# ---------------------------------------------------------------------------
# TestTotalReflection
# ---------------------------------------------------------------------------

class TestTotalReflection(unittest.TestCase):
    """Test total_reflection metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "total_reflection", params)

    def test_single_wavelength(self):
        r = _make_solver_results(reflection=torch.tensor([0.15]))
        value = self._extract(r, wavelength_index=0)
        self.assertAlmostEqual(value, 0.15, places=5)

    def test_batched_two_sources(self):
        """The exact bug scenario: two-polarization with source_index selection."""
        r = _make_batched_two_sources(
            reflection_s0=torch.tensor([0.92]),
            reflection_s1=torch.tensor([0.08]),
        )
        self.assertAlmostEqual(
            self._extract(r, source_index=0, wavelength_index=0), 0.92, places=5
        )
        self.assertAlmostEqual(
            self._extract(r, source_index=1, wavelength_index=0), 0.08, places=5
        )


# ---------------------------------------------------------------------------
# TestZeroOrderPhaseDeg
# ---------------------------------------------------------------------------

class TestZeroOrderPhaseDeg(unittest.TestCase):
    """Test zero_order_transmission_phase_deg metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "zero_order_transmission_phase_deg", params)

    def _make_with_field(self, complex_val: complex, component: str = "x") -> SolverResults:
        """Create SolverResults with a known complex value at zero-order for given component."""
        # harmonics (3,3) → center at (1,1)
        n_freqs = 1
        harmonics = (3, 3)
        h0, h1 = harmonics

        field_data = {c: torch.zeros(n_freqs, h0, h1, dtype=torch.complex64) for c in ("x", "y", "z")}
        # Set the zero-order (center) element
        field_data[component][0, h0 // 2, h1 // 2] = complex_val

        tf = FieldComponents(
            x=field_data["x"],
            y=field_data["y"],
            z=field_data["z"],
        )
        return _make_solver_results(
            n_freqs=n_freqs,
            harmonics=harmonics,
            transmission_field=tf,
        )

    def test_component_x(self):
        """Phase of Ex at zero order."""
        # angle of (1+1j) = 45 degrees
        r = self._make_with_field(complex(1, 1), component="x")
        value = self._extract(r, component="x", wavelength_index=0)
        self.assertAlmostEqual(value, 45.0, places=3)

    def test_component_y(self):
        """Phase of Ey at zero order."""
        # angle of (-1+0j) = 180 degrees
        r = self._make_with_field(complex(-1, 0), component="y")
        value = self._extract(r, component="y", wavelength_index=0)
        self.assertAlmostEqual(value, 180.0, places=3)

    def test_component_z(self):
        """Phase of Ez at zero order."""
        # angle of (0+1j) = 90 degrees
        r = self._make_with_field(complex(0, 1), component="z")
        value = self._extract(r, component="z", wavelength_index=0)
        self.assertAlmostEqual(value, 90.0, places=3)

    def test_polarization_alias_tm(self):
        """polarization='TM' → component='x'."""
        r = self._make_with_field(complex(1, 1), component="x")
        value = self._extract(r, polarization="TM", wavelength_index=0)
        self.assertAlmostEqual(value, 45.0, places=3)

    def test_polarization_alias_te(self):
        """polarization='TE' → component='y'."""
        r = self._make_with_field(complex(-1, 0), component="y")
        value = self._extract(r, polarization="TE", wavelength_index=0)
        self.assertAlmostEqual(value, 180.0, places=3)

    def test_phase_range(self):
        """Phase should be in [-180, 180] range."""
        # angle of (0-1j) = -90 degrees
        r = self._make_with_field(complex(0, -1), component="x")
        value = self._extract(r, component="x", wavelength_index=0)
        self.assertAlmostEqual(value, -90.0, places=3)
        self.assertGreaterEqual(value, -180.0)
        self.assertLessEqual(value, 180.0)

    def test_invalid_component_raises(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = self._make_with_field(complex(1, 0), component="x")
        with self.assertRaises(MetricError) as ctx:
            self._extract(r, component="w", wavelength_index=0)
        self.assertIn("Invalid component", str(ctx.exception))

    def test_unknown_polarization_raises(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = self._make_with_field(complex(1, 0), component="x")
        with self.assertRaises(MetricError) as ctx:
            self._extract(r, polarization="INVALID", wavelength_index=0)
        self.assertIn("Unknown polarization", str(ctx.exception))

    def test_both_polarization_and_component_raises(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = self._make_with_field(complex(1, 0), component="x")
        with self.assertRaises(MetricError) as ctx:
            self._extract(r, polarization="TM", component="x", wavelength_index=0)
        self.assertIn("Cannot specify both", str(ctx.exception))


# ---------------------------------------------------------------------------
# TestZeroOrderAmplitude
# ---------------------------------------------------------------------------

class TestZeroOrderAmplitude(unittest.TestCase):
    """Test zero_order_transmission_amplitude metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "zero_order_transmission_amplitude", params)

    def _make_with_field(self, complex_val: complex, component: str = "x") -> SolverResults:
        n_freqs = 1
        harmonics = (3, 3)
        h0, h1 = harmonics
        field_data = {c: torch.zeros(n_freqs, h0, h1, dtype=torch.complex64) for c in ("x", "y", "z")}
        field_data[component][0, h0 // 2, h1 // 2] = complex_val
        tf = FieldComponents(x=field_data["x"], y=field_data["y"], z=field_data["z"])
        return _make_solver_results(n_freqs=n_freqs, harmonics=harmonics, transmission_field=tf)

    def test_component_x(self):
        """Amplitude |Ex| at zero order."""
        # |3+4j| = 5.0
        r = self._make_with_field(complex(3, 4), component="x")
        value = self._extract(r, component="x", wavelength_index=0)
        self.assertAlmostEqual(value, 5.0, places=4)

    def test_component_y(self):
        """Amplitude |Ey| at zero order."""
        r = self._make_with_field(complex(0, 2), component="y")
        value = self._extract(r, component="y", wavelength_index=0)
        self.assertAlmostEqual(value, 2.0, places=4)


# ---------------------------------------------------------------------------
# TestOrderTransmissionEfficiency
# ---------------------------------------------------------------------------

class TestOrderTransmissionEfficiency(unittest.TestCase):
    """Test order_transmission_efficiency metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "order_transmission_efficiency", params)

    def _make_with_diffraction(self, values_2d: torch.Tensor) -> SolverResults:
        """Create SolverResults with specific transmission_diffraction values.

        values_2d: shape (H0, H1), will be expanded to (1, H0, H1).
        """
        h0, h1 = values_2d.shape
        td = values_2d.unsqueeze(0)  # (1, H0, H1)
        return _make_solver_results(
            n_freqs=1,
            harmonics=(h0, h1),
            transmission_diffraction=td,
        )

    def test_zero_order(self):
        """order=[0,0] extracts the center element (specular transmission)."""
        # 3x3 grid: center is (1,1)
        td = torch.zeros(3, 3)
        td[1, 1] = 0.75  # zero-order
        r = self._make_with_diffraction(td)
        value = self._extract(r, order=[0, 0], wavelength_index=0)
        self.assertAlmostEqual(value, 0.75, places=5)

    def test_first_order(self):
        """order=[1,0] extracts the (1,0) diffraction order."""
        td = torch.zeros(3, 3)
        # order (1, 0): ix = center_x + 1 = 2, iy = center_y + 0 = 1
        # Access: transmission_diffraction[:, ix, iy] = td[:, 2, 1]
        td[2, 1] = 0.12
        r = self._make_with_diffraction(td)
        # get_order_transmission_efficiency uses get_diffraction_order_indices which
        # maps (order_x, order_y) → (center_y + order_y, center_x + order_x)
        # Actually: ix = center_x + order_x, iy = center_y + order_y
        # Then returns transmission_diffraction[:, ix, iy]
        # With harmonics (3,3): center_x = 1, center_y = 1
        # order (1, 0): ix = 1+1=2, iy = 1+0=1
        # So we need td[iy, ix] = td[1, 2]... wait, get_diffraction_order_indices returns (ix, iy)
        # and then get_order_transmission_efficiency does self.transmission_diffraction[:, ix, iy]
        value = self._extract(r, order=[1, 0], wavelength_index=0)
        self.assertAlmostEqual(value, 0.12, places=5)

    def test_order_out_of_bounds(self):
        """Order exceeding harmonics → ValueError from torchrdit."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = _make_solver_results(harmonics=(3, 3))
        with self.assertRaises((MetricError, ValueError)):
            self._extract(r, order=[5, 0], wavelength_index=0)

    def test_invalid_order_format(self):
        """Non-list order → MetricError."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        r = _make_solver_results()
        with self.assertRaises(MetricError):
            self._extract(r, order=1, wavelength_index=0)


# ---------------------------------------------------------------------------
# TestOrderReflectionEfficiency
# ---------------------------------------------------------------------------

class TestOrderReflectionEfficiency(unittest.TestCase):
    """Test order_reflection_efficiency metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "order_reflection_efficiency", params)

    def test_zero_order(self):
        """order=[0,0] extracts the center element."""
        rd = torch.zeros(3, 3)
        rd[1, 1] = 0.55  # zero-order center
        r = _make_solver_results(
            n_freqs=1,
            harmonics=(3, 3),
            reflection_diffraction=rd.unsqueeze(0),
        )
        value = self._extract(r, order=[0, 0], wavelength_index=0)
        self.assertAlmostEqual(value, 0.55, places=5)

    def test_first_order(self):
        """order=[1,0] extracts the (1,0) diffraction order."""
        rd = torch.zeros(3, 3)
        # order (1,0): ix = center_x + 1 = 2, iy = center_y + 0 = 1
        # Access: reflection_diffraction[:, ix, iy] = rd[:, 2, 1]
        rd[2, 1] = 0.03
        r = _make_solver_results(
            n_freqs=1,
            harmonics=(3, 3),
            reflection_diffraction=rd.unsqueeze(0),
        )
        value = self._extract(r, order=[1, 0], wavelength_index=0)
        self.assertAlmostEqual(value, 0.03, places=5)


# ---------------------------------------------------------------------------
# TestZeroOrderReflectionPhaseDeg
# ---------------------------------------------------------------------------

class TestZeroOrderReflectionPhaseDeg(unittest.TestCase):
    """Test zero_order_reflection_phase_deg metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "zero_order_reflection_phase_deg", params)

    def _make_with_reflection_field(self, complex_val: complex, component: str = "x") -> SolverResults:
        n_freqs = 1
        harmonics = (3, 3)
        h0, h1 = harmonics
        field_data = {c: torch.zeros(n_freqs, h0, h1, dtype=torch.complex64) for c in ("x", "y", "z")}
        field_data[component][0, h0 // 2, h1 // 2] = complex_val
        rf = FieldComponents(x=field_data["x"], y=field_data["y"], z=field_data["z"])
        return _make_solver_results(n_freqs=n_freqs, harmonics=harmonics, reflection_field=rf)

    def test_component_x(self):
        """Phase of reflected Ex at zero order."""
        r = self._make_with_reflection_field(complex(1, 1), component="x")
        value = self._extract(r, component="x", wavelength_index=0)
        self.assertAlmostEqual(value, 45.0, places=3)

    def test_component_y(self):
        r = self._make_with_reflection_field(complex(0, -1), component="y")
        value = self._extract(r, component="y", wavelength_index=0)
        self.assertAlmostEqual(value, -90.0, places=3)


# ---------------------------------------------------------------------------
# TestZeroOrderReflectionAmplitude
# ---------------------------------------------------------------------------

class TestZeroOrderReflectionAmplitude(unittest.TestCase):
    """Test zero_order_reflection_amplitude metric extractor."""

    def _extract(self, r, **params):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        return extract_metric(r, "zero_order_reflection_amplitude", params)

    def _make_with_reflection_field(self, complex_val: complex, component: str = "x") -> SolverResults:
        n_freqs = 1
        harmonics = (3, 3)
        h0, h1 = harmonics
        field_data = {c: torch.zeros(n_freqs, h0, h1, dtype=torch.complex64) for c in ("x", "y", "z")}
        field_data[component][0, h0 // 2, h1 // 2] = complex_val
        rf = FieldComponents(x=field_data["x"], y=field_data["y"], z=field_data["z"])
        return _make_solver_results(n_freqs=n_freqs, harmonics=harmonics, reflection_field=rf)

    def test_component_x(self):
        """Amplitude |Rx| at zero order."""
        r = self._make_with_reflection_field(complex(3, 4), component="x")
        value = self._extract(r, component="x", wavelength_index=0)
        self.assertAlmostEqual(value, 5.0, places=4)


# ---------------------------------------------------------------------------
# TestExtractMetricAPI
# ---------------------------------------------------------------------------

class TestExtractMetricAPI(unittest.TestCase):
    """Test the top-level extract_metric function."""

    def test_unknown_metric_raises_key_error(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        r = _make_solver_results()
        with self.assertRaises(KeyError):
            extract_metric(r, "bogus_metric", {"wavelength_index": 0})

    def test_missing_required_param_raises(self):
        """Calling total_transmission without wavelength_index → TypeError wrapped in MetricError."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError, extract_metric
        r = _make_solver_results()
        with self.assertRaises((MetricError, TypeError)):
            extract_metric(r, "total_transmission", {})

    def test_params_none_treated_as_empty(self):
        """params=None → empty dict, still requires wavelength_index."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError, extract_metric
        r = _make_solver_results()
        with self.assertRaises((MetricError, TypeError)):
            extract_metric(r, "total_transmission", None)

    def test_returns_python_float(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        r = _make_solver_results(transmission=torch.tensor([0.75]))
        value = extract_metric(r, "total_transmission", {"wavelength_index": 0})
        self.assertIsInstance(value, float)


# ---------------------------------------------------------------------------
# TestDiffractionOrderIndexing (sanity check against torchrdit)
# ---------------------------------------------------------------------------

class TestDiffractionOrderIndexing(unittest.TestCase):
    """Verify that our order parameter correctly maps through torchrdit's indexing."""

    def test_negative_order(self):
        """order=[-1, 0] should work for 3x3 harmonics."""
        from evo_metaoptics.meta_design.gt_eval_metrics import extract_metric
        td = torch.zeros(3, 3)
        # order (-1, 0): ix = center_x + (-1) = 0, iy = center_y + 0 = 1
        # Access: transmission_diffraction[:, ix, iy] = td[:, 0, 1]
        td[0, 1] = 0.07
        r = _make_solver_results(
            n_freqs=1,
            harmonics=(3, 3),
            transmission_diffraction=td.unsqueeze(0),
        )
        value = extract_metric(r, "order_transmission_efficiency", {"order": [-1, 0], "wavelength_index": 0})
        self.assertAlmostEqual(value, 0.07, places=5)


# ---------------------------------------------------------------------------
# TestGenerationTimeValidation
# ---------------------------------------------------------------------------

class TestGenerationTimeValidation(unittest.TestCase):
    """Test validate_metric_criterion for generation-time checks."""

    def _validate(self, criterion, wavelength_um=None):
        from evo_metaoptics.meta_design.gt_eval_metrics import validate_metric_criterion
        return validate_metric_criterion(criterion, wavelength_um)

    def test_validate_criterion_valid_metric(self):
        """Valid criterion passes without error."""
        self._validate({
            "metric": "total_transmission",
            "params": {"wavelength_index": 0},
        }, wavelength_um=[1.55])

    def test_validate_criterion_unknown_metric(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        with self.assertRaises(MetricError) as ctx:
            self._validate({"metric": "bogus", "params": {}})
        self.assertIn("Unknown metric", str(ctx.exception))

    def test_validate_criterion_missing_required_param(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        with self.assertRaises(MetricError) as ctx:
            self._validate({
                "metric": "total_transmission",
                "params": {},  # missing wavelength_index
            })
        self.assertIn("wavelength_index", str(ctx.exception))

    def test_validate_criterion_wavelength_oob(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        with self.assertRaises(MetricError) as ctx:
            self._validate({
                "metric": "total_transmission",
                "params": {"wavelength_index": 2},
            }, wavelength_um=[1.55])
        self.assertIn("out of bounds", str(ctx.exception))

    def test_validate_criterion_invalid_component(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        with self.assertRaises(MetricError) as ctx:
            self._validate({
                "metric": "zero_order_transmission_phase_deg",
                "params": {"component": "w", "wavelength_index": 0},
            })
        self.assertIn("Invalid component", str(ctx.exception))

    def test_validate_criterion_missing_component(self):
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        with self.assertRaises(MetricError) as ctx:
            self._validate({
                "metric": "zero_order_transmission_phase_deg",
                "params": {"wavelength_index": 0},  # missing component/polarization
            })
        self.assertIn("component", str(ctx.exception))

    def test_validate_criterion_polarization_alias(self):
        """Polarization alias should be resolved and validated."""
        self._validate({
            "metric": "zero_order_transmission_phase_deg",
            "params": {"polarization": "TM", "wavelength_index": 0},
        })

    def test_validate_criterion_order_metric(self):
        """order_transmission_efficiency requires 'order' param."""
        from evo_metaoptics.meta_design.gt_eval_metrics import MetricError
        with self.assertRaises(MetricError):
            self._validate({
                "metric": "order_transmission_efficiency",
                "params": {"wavelength_index": 0},  # missing order
            })


if __name__ == "__main__":
    unittest.main()
