"""Tests for the runtime n/k lookup helper (material_db/runtime.py).

Validates:
- Input validation (type errors, value errors)
- page_id lookup delegation to get_page_by_id
- n/k fetch via RefractiveIndexMaterial with NoExtinctionCoefficient fallback
- Return dict structure and content
- Error propagation for missing packages, invalid pages, bad data
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class _NoExtCoeff(Exception):
    """Stub for refractiveindex.refractiveindex.NoExtinctionCoefficient."""

    pass


def _make_page_ref(
    *,
    page_id: int = 42,
    shelf: str = "main",
    book: str = "SiO2",
    page: str = "Malitson",
) -> SimpleNamespace:
    return SimpleNamespace(
        page_id=page_id,
        material_id=1,
        shelf=shelf,
        book=book,
        page=page,
        page_name=None,
        data_path=None,
        coverage_min=0.21,
        coverage_max=6.7,
        has_n=True,
        has_k=False,
    )


def _make_settings(source_root: str = "/fake/db") -> SimpleNamespace:
    return SimpleNamespace(
        db_path=Path("/fake/index.sqlite"),
        source_root=Path(source_root),
        auto_download=True,
    )


class TestGetMaterialNkInputValidation(unittest.TestCase):
    """Input validation should reject bad types/values before any DB access."""

    def test_page_id_must_be_int(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(TypeError, msg="page_id must be int"):
            get_material_nk(page_id="42", wavelengths_um=[1.55])  # type: ignore[arg-type]

    def test_page_id_must_be_positive(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(ValueError, msg="page_id must be > 0"):
            get_material_nk(page_id=0, wavelengths_um=[1.55])

    def test_page_id_negative(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(ValueError):
            get_material_nk(page_id=-1, wavelengths_um=[1.55])

    def test_wavelengths_must_be_list(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(TypeError, msg="wavelengths_um must be a list"):
            get_material_nk(page_id=1, wavelengths_um=1.55)  # type: ignore[arg-type]

    def test_wavelengths_must_be_nonempty(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(ValueError, msg="wavelengths_um must be non-empty"):
            get_material_nk(page_id=1, wavelengths_um=[])

    def test_wavelengths_must_be_positive(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(ValueError, msg="Wavelength must be positive"):
            get_material_nk(page_id=1, wavelengths_um=[-1.0])

    def test_wavelengths_must_be_finite(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(ValueError):
            get_material_nk(page_id=1, wavelengths_um=[float("inf")])

    def test_wavelengths_nan_rejected(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(ValueError):
            get_material_nk(page_id=1, wavelengths_um=[float("nan")])

    def test_wavelengths_zero_rejected(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(ValueError):
            get_material_nk(page_id=1, wavelengths_um=[0.0])

    def test_wavelength_element_must_be_numeric(self):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(TypeError):
            get_material_nk(page_id=1, wavelengths_um=["abc"])  # type: ignore[list-item]


class TestGetMaterialNkPageLookup(unittest.TestCase):
    """Page lookup delegation and error handling."""

    @patch("evo_metaoptics.material_db.runtime.get_page_by_id", return_value=None)
    @patch(
        "evo_metaoptics.material_db.runtime.load_material_db_settings",
        return_value=_make_settings(),
    )
    def test_unknown_page_id_raises_runtime_error(self, _mock_settings, _mock_get):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(RuntimeError, msg="No material page found"):
            get_material_nk(page_id=99999, wavelengths_um=[1.55])

    @patch("evo_metaoptics.material_db.runtime.get_page_by_id", return_value=None)
    @patch(
        "evo_metaoptics.material_db.runtime.load_material_db_settings",
        return_value=_make_settings(),
    )
    def test_error_message_includes_page_id(self, _mock_settings, _mock_get):
        from evo_metaoptics.material_db.runtime import get_material_nk

        with self.assertRaises(RuntimeError) as ctx:
            get_material_nk(page_id=12345, wavelengths_um=[1.55])
        self.assertIn("12345", str(ctx.exception))


class TestGetMaterialNkFetch(unittest.TestCase):
    """n/k fetch via mocked RefractiveIndexMaterial."""

    def _run_with_mock_material(
        self,
        page_ref=None,
        n_values=None,
        k_values=None,
        k_raises=None,
        wavelengths_um=None,
    ):
        """Helper: mock DB access and refractiveindex, run get_material_nk."""
        from evo_metaoptics.material_db.runtime import get_material_nk

        if page_ref is None:
            page_ref = _make_page_ref()
        if wavelengths_um is None:
            wavelengths_um = [1.55]
        if n_values is None:
            n_values = [1.444024] * len(wavelengths_um)
        if k_values is None:
            k_values = [0.0] * len(wavelengths_um)

        # Build a mock material object
        mock_material = MagicMock()
        n_iter = iter(n_values)
        k_iter = iter(k_values) if k_raises is None else None

        mock_material.get_refractive_index.side_effect = lambda wl_nm: next(n_iter)

        if k_raises is not None:
            mock_material.get_extinction_coefficient.side_effect = k_raises
        else:
            k_list = list(k_values)
            k_call_idx = [0]

            def _get_k(wl_nm):
                val = k_list[k_call_idx[0]]
                k_call_idx[0] += 1
                return val

            mock_material.get_extinction_coefficient.side_effect = _get_k

        mock_factory = MagicMock(return_value=mock_material)

        with (
            patch(
                "evo_metaoptics.material_db.runtime.load_material_db_settings",
                return_value=_make_settings(),
            ),
            patch(
                "evo_metaoptics.material_db.runtime.get_page_by_id",
                return_value=page_ref,
            ),
            patch.dict(
                "sys.modules",
                {
                    "refractiveindex": MagicMock(
                        RefractiveIndexMaterial=mock_factory
                    ),
                    "refractiveindex.refractiveindex": MagicMock(
                        NoExtinctionCoefficient=_NoExtCoeff
                    ),
                },
            ),
        ):
            result = get_material_nk(
                page_id=page_ref.page_id, wavelengths_um=wavelengths_um
            )

        return result, mock_factory, mock_material

    def test_single_wavelength_lossless(self):
        result, _, _ = self._run_with_mock_material(
            n_values=[1.444024], k_values=[0.0], wavelengths_um=[1.55]
        )
        self.assertEqual(result["n"], [1.444024])
        self.assertEqual(result["k"], [0.0])
        self.assertEqual(result["book"], "SiO2")
        self.assertEqual(result["page"], "Malitson")
        self.assertEqual(result["shelf"], "main")
        self.assertEqual(result["page_id"], 42)

    def test_multiple_wavelengths(self):
        wls = [0.5, 0.7, 1.0]
        ns = [2.49, 2.43, 2.38]
        ks = [0.0, 0.0, 0.0]
        result, _, _ = self._run_with_mock_material(
            page_ref=_make_page_ref(book="TiO2", page="Devore"),
            n_values=ns,
            k_values=ks,
            wavelengths_um=wls,
        )
        self.assertEqual(result["n"], ns)
        self.assertEqual(result["k"], ks)
        self.assertEqual(result["book"], "TiO2")

    def test_lossy_material(self):
        result, _, _ = self._run_with_mock_material(
            page_ref=_make_page_ref(book="Au", page="Johnson", shelf="main"),
            n_values=[0.47],
            k_values=[2.42],
            wavelengths_um=[0.5],
        )
        self.assertEqual(result["n"], [0.47])
        self.assertEqual(result["k"], [2.42])

    def test_wavelength_converted_to_nm(self):
        """Verify the material object receives wavelengths in nm."""
        _, _, mock_mat = self._run_with_mock_material(wavelengths_um=[1.55])
        mock_mat.get_refractive_index.assert_called_once_with(1550.0)

    def test_no_extinction_coefficient_fallback(self):
        """k should be 0.0 when NoExtinctionCoefficient is raised."""
        result, _, _ = self._run_with_mock_material(
            k_raises=_NoExtCoeff("no k data"),
            wavelengths_um=[1.55],
        )
        self.assertEqual(result["k"], [0.0])

    def test_return_dict_keys(self):
        result, _, _ = self._run_with_mock_material()
        expected_keys = {"n", "k", "book", "page", "shelf", "page_id"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_invalid_n_raises_value_error(self):
        """Non-finite n should raise ValueError."""
        from evo_metaoptics.material_db.runtime import get_material_nk

        mock_material = MagicMock()
        mock_material.get_refractive_index.return_value = float("nan")

        with (
            patch(
                "evo_metaoptics.material_db.runtime.load_material_db_settings",
                return_value=_make_settings(),
            ),
            patch(
                "evo_metaoptics.material_db.runtime.get_page_by_id",
                return_value=_make_page_ref(),
            ),
            patch.dict(
                "sys.modules",
                {
                    "refractiveindex": MagicMock(
                        RefractiveIndexMaterial=MagicMock(return_value=mock_material)
                    ),
                    "refractiveindex.refractiveindex": MagicMock(
                        NoExtinctionCoefficient=_NoExtCoeff
                    ),
                },
            ),
        ):
            with self.assertRaises(ValueError, msg="Invalid refractive index"):
                get_material_nk(page_id=42, wavelengths_um=[1.55])

    def test_invalid_k_raises_value_error(self):
        """Non-finite k should raise ValueError."""
        from evo_metaoptics.material_db.runtime import get_material_nk

        mock_material = MagicMock()
        mock_material.get_refractive_index.return_value = 1.5
        mock_material.get_extinction_coefficient.return_value = float("inf")

        with (
            patch(
                "evo_metaoptics.material_db.runtime.load_material_db_settings",
                return_value=_make_settings(),
            ),
            patch(
                "evo_metaoptics.material_db.runtime.get_page_by_id",
                return_value=_make_page_ref(),
            ),
            patch.dict(
                "sys.modules",
                {
                    "refractiveindex": MagicMock(
                        RefractiveIndexMaterial=MagicMock(return_value=mock_material)
                    ),
                    "refractiveindex.refractiveindex": MagicMock(
                        NoExtinctionCoefficient=_NoExtCoeff
                    ),
                },
            ),
        ):
            with self.assertRaises(ValueError, msg="Invalid extinction"):
                get_material_nk(page_id=42, wavelengths_um=[1.55])

    def test_integer_wavelength_accepted(self):
        """Integer wavelengths (e.g., 1 instead of 1.0) should work."""
        result, _, _ = self._run_with_mock_material(
            n_values=[1.5], k_values=[0.0], wavelengths_um=[1]
        )
        self.assertEqual(result["n"], [1.5])

    def test_factory_receives_correct_args(self):
        """RefractiveIndexMaterial should be called with shelf/book/page/databasePath."""
        page = _make_page_ref(shelf="3d", book="Ag", page="Hagemann")
        _, mock_factory, _ = self._run_with_mock_material(
            page_ref=page,
            n_values=[0.05],
            k_values=[3.42],
            wavelengths_um=[0.5],
        )
        mock_factory.assert_called_once_with(
            shelf="3d",
            book="Ag",
            page="Hagemann",
            databasePath=Path("/fake/db"),
        )


class TestGetMaterialNkMissingPackage(unittest.TestCase):
    """Verify clear error when refractiveindex is not installed."""

    @patch(
        "evo_metaoptics.material_db.runtime.load_material_db_settings",
        return_value=_make_settings(),
    )
    @patch(
        "evo_metaoptics.material_db.runtime.get_page_by_id",
        return_value=_make_page_ref(),
    )
    def test_import_error_gives_runtime_error(self, _mock_get, _mock_settings):
        """RuntimeError should mention pip install refractiveindex."""
        import sys

        import evo_metaoptics.material_db.runtime as runtime_mod

        # Remove cached refractiveindex modules so the lazy import re-fires
        saved = {}
        for key in list(sys.modules):
            if key.startswith("refractiveindex"):
                saved[key] = sys.modules.pop(key)

        orig_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith("refractiveindex"):
                raise ImportError("no such module")
            return orig_import(name, *args, **kwargs)

        try:
            # Reload the module to clear any cached imports
            with patch("builtins.__import__", side_effect=_fake_import):
                with self.assertRaises(RuntimeError) as ctx:
                    runtime_mod.get_material_nk(page_id=42, wavelengths_um=[1.55])
                self.assertIn("refractiveindex", str(ctx.exception))
        finally:
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
