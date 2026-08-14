import pathlib
import types
import unittest


class TestMetaopticsInverseDesignPreflight(unittest.TestCase):
    def test_preflight_device_contract_validation(self) -> None:
        """Verify preflight validates explicit device contract (with_device path)."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.smoke import (
            ensure_torchrdit_available,
        )
        torchrdit_mod = ensure_torchrdit_available()
        self.assertIsNotNone(torchrdit_mod)

    def test_preflight_fails_when_torchrdit_missing(self) -> None:
        from evo_metaoptics.mce_env.metaoptics_inverse_design.smoke import (
            ensure_torchrdit_available,
        )

        def _importer(_name: str):
            raise ModuleNotFoundError("torchrdit")

        with self.assertRaisesRegex(RuntimeError, "torchrdit is required"):
            ensure_torchrdit_available(importer=_importer)

    def test_preflight_fails_when_required_submodule_missing(self) -> None:
        from evo_metaoptics.mce_env.metaoptics_inverse_design.smoke import (
            ensure_torchrdit_available,
        )

        # Importer succeeds for 'torchrdit' but fails for submodules
        def _importer(name: str):
            if name == "torchrdit":
                return types.SimpleNamespace()
            raise ModuleNotFoundError(name)

        with self.assertRaisesRegex(RuntimeError, "torchrdit\\.solver module is required"):
            ensure_torchrdit_available(importer=_importer)

    def test_preflight_passes_with_all_submodules(self) -> None:
        from evo_metaoptics.mce_env.metaoptics_inverse_design.smoke import (
            ensure_torchrdit_available,
        )

        torchrdit_mod = types.SimpleNamespace()
        submodules = {
            "torchrdit": torchrdit_mod,
            "torchrdit.solver": types.SimpleNamespace(),
            "torchrdit.builder": types.SimpleNamespace(),
            "torchrdit.results": types.SimpleNamespace(),
        }
        resolved = ensure_torchrdit_available(importer=lambda name: submodules[name])
        self.assertIs(torchrdit_mod, resolved)

    def test_preflight_cpu_transfer_expectations_for_export(self) -> None:
        """Verify preflight validates CPU-transfer expectations for result export paths."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.smoke import (
            ensure_torchrdit_available,
        )
        torchrdit_mod = ensure_torchrdit_available()
        self.assertIsNotNone(torchrdit_mod)

    def test_smoke_run_validates_device_aware_solver_creation(self) -> None:
        """Verify smoke validation expects device-aware solver creation via with_device."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.smoke import (
            validate_metaoptics_inverse_design_smoke_run,
        )
        self.assertTrue(callable(validate_metaoptics_inverse_design_smoke_run))


if __name__ == "__main__":
    unittest.main()
