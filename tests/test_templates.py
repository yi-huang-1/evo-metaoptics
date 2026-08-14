"""Tests for M0: Template Extraction, API Validation, and Jinja Management.

Validates:
- All 17 template files exist at the expected path.
- Template content uses correct TorchRDIT API calls (verified against actual API).
- Template content round-trips through Jinja rendering unchanged.
- API discrepancy fixes are applied (x_size/y_size, subtract labeling).
- templates_manifest.json is consistent with actual template files.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "evo_metaoptics"
    / "mce_env"
    / "metaoptics_inverse_design"
    / "skills"
    / "templates"
)

# All expected template IDs (20 total)
EXPECTED_TEMPLATE_IDS = sorted([
    # Basic (12)
    "basic_imports",
    "unit_setup",
    "solver_setup",
    "material_creation",
    "layer_stack",
    "patterned_layer",
    "source_setup",
    "solve_and_analyze",
    "optimization_basic",
    "shape_operations",
    "visualization",
    "common_patterns",
    # Optimization (5)
    "gradient_based",
    "multi_objective",
    "gradient_full_pipeline",
    "gradient_multiangle",
    "gradient_phase_target",
    # Clarifications (3)
    "layer_order",
    "material_api",
    "common_mistakes",
])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_manifest() -> list[dict]:
    manifest_path = TEMPLATES_DIR / "templates_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_template(template_id: str) -> str:
    path = TEMPLATES_DIR / f"{template_id}.py.jinja"
    return path.read_text(encoding="utf-8")


class TestTemplateFilesExist(unittest.TestCase):
    """M0 Red #1: Each template file exists at the expected path."""

    def test_templates_dir_exists(self):
        self.assertTrue(TEMPLATES_DIR.exists(), f"Templates dir missing: {TEMPLATES_DIR}")

    def test_all_template_files_exist(self):
        for tid in EXPECTED_TEMPLATE_IDS:
            with self.subTest(template_id=tid):
                path = TEMPLATES_DIR / f"{tid}.py.jinja"
                self.assertTrue(path.exists(), f"Template file missing: {path}")

    def test_manifest_exists(self):
        manifest_path = TEMPLATES_DIR / "templates_manifest.json"
        self.assertTrue(manifest_path.exists(), f"Manifest missing: {manifest_path}")

    def test_no_unexpected_template_files(self):
        """Only expected templates + manifest + __init__.py should be present."""
        expected_files = {f"{tid}.py.jinja" for tid in EXPECTED_TEMPLATE_IDS}
        expected_files.add("templates_manifest.json")
        expected_files.add("__init__.py")
        expected_files.add("__pycache__")
        actual_files = {p.name for p in TEMPLATES_DIR.iterdir()}
        unexpected = actual_files - expected_files
        self.assertEqual(unexpected, set(), f"Unexpected files: {unexpected}")


class TestTemplateAPICorrectness(unittest.TestCase):
    """M0 Red #2: Templates use only valid TorchRDIT API calls."""

    def test_basic_imports_valid_imports(self):
        content = _load_template("basic_imports")
        self.assertIn("from torchrdit.solver import get_solver_builder", content)
        self.assertIn("from torchrdit.shapes import ShapeGenerator", content)
        self.assertIn("from torchrdit.utils import create_material", content)
        self.assertIn("from torchrdit.constants import Algorithm, Precision", content)

    def test_solver_setup_uses_builder_pattern(self):
        content = _load_template("solver_setup")
        self.assertIn("get_solver_builder()", content)
        self.assertIn("builder.with_algorithm(", content)
        self.assertIn("builder.with_precision(", content)
        self.assertIn("builder.with_real_dimensions(", content)
        self.assertIn("builder.with_k_dimensions(", content)
        self.assertIn("builder.with_wavelengths(", content)
        self.assertIn("builder.build()", content)

    def test_material_creation_uses_create_material(self):
        content = _load_template("material_creation")
        self.assertIn("create_material(", content)
        self.assertIn("permittivity=", content)

    def test_layer_stack_uses_correct_methods(self):
        content = _load_template("layer_stack")
        self.assertIn("solver.update_ref_material(", content)
        self.assertIn("solver.update_trn_material(", content)
        self.assertIn("solver.add_layer(", content)

    def test_patterned_layer_uses_shape_generator(self):
        content = _load_template("patterned_layer")
        self.assertIn("ShapeGenerator.from_solver(solver)", content)
        self.assertIn("update_er_with_mask(", content)
        self.assertIn("is_homogeneous=False", content)

    def test_source_setup_uses_add_source(self):
        content = _load_template("source_setup")
        self.assertIn("solver.add_source(", content)
        self.assertIn("theta=", content)
        self.assertIn("phi=", content)
        self.assertIn("pte=", content)
        self.assertIn("ptm=", content)

    def test_solve_and_analyze_correct_api(self):
        content = _load_template("solve_and_analyze")
        self.assertIn("solver.solve(source)", content)
        self.assertIn("result.transmission", content)
        self.assertIn("result.reflection", content)
        self.assertIn("result.get_zero_order_transmission()", content)
        self.assertIn("result.get_zero_order_reflection()", content)

    def test_shape_operations_correct_param_names(self):
        """M0 Red #4: generate_rectangle_mask uses x_size/y_size, not width/height."""
        content = _load_template("shape_operations")
        self.assertIn("generate_circle_mask(", content)
        self.assertIn("generate_rectangle_mask(", content)
        self.assertIn("generate_polygon_mask(", content)
        self.assertIn("combine_masks(", content)
        # API fix: must use x_size/y_size
        self.assertIn("x_size=w", content)
        self.assertIn("y_size=h", content)
        # Must NOT use width/height
        self.assertNotIn("width=w", content)
        self.assertNotIn("height=h", content)

    def test_shape_operations_subtract_not_labeled_xor(self):
        """API fix: subtract operation should not be misleadingly labeled as xor."""
        content = _load_template("shape_operations")
        # The variable name should be 'subtract', not 'xor'
        self.assertNotIn("xor = shapegen.combine_masks", content)
        self.assertIn("subtract = shapegen.combine_masks", content)

    def test_layer_order_documents_convention(self):
        content = _load_template("layer_order")
        self.assertIn("Reflection", content)
        self.assertIn("Transmission", content)
        self.assertIn("ref_material", content)
        self.assertIn("Layer 0", content)

    def test_material_api_warns_about_inc_material(self):
        content = _load_template("material_api")
        self.assertIn("update_ref_material", content)
        self.assertIn("update_trn_material", content)
        # Must warn about non-existent functions
        self.assertIn("update_inc_material", content)

    def test_common_mistakes_lists_errors(self):
        content = _load_template("common_mistakes")
        self.assertIn("Layer ordering", content)
        self.assertIn("requires_grad", content)

    def test_gradient_based_optimization_pattern(self):
        content = _load_template("gradient_based")
        self.assertIn("requires_grad=True", content)
        self.assertIn("torch.optim.Adam", content)
        self.assertIn("optimizer.zero_grad()", content)
        self.assertIn("loss.backward()", content)
        self.assertIn("optimizer.step()", content)

    def test_multi_objective_pattern(self):
        content = _load_template("multi_objective")
        self.assertIn("result.transmission", content)
        self.assertIn("weights", content)
        self.assertIn("total_loss", content)

    def test_visualization_template(self):
        content = _load_template("visualization")
        self.assertIn("plot_layer", content)
        self.assertIn("plt", content)


class TestTemplateJinjaRoundtrip(unittest.TestCase):
    """M0 Red #3: Template content round-trips through Jinja rendering unchanged.

    Since templates are immutable and contain no Jinja variables, rendering them
    through Jinja should produce identical output.
    """

    def test_all_templates_roundtrip_unchanged(self):
        try:
            import jinja2
        except ImportError:
            self.skipTest("jinja2 not installed")

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            keep_trailing_newline=True,
            undefined=jinja2.StrictUndefined,
        )

        for tid in EXPECTED_TEMPLATE_IDS:
            with self.subTest(template_id=tid):
                original = _load_template(tid)
                template = env.get_template(f"{tid}.py.jinja")
                rendered = template.render()
                self.assertEqual(
                    original,
                    rendered,
                    f"Template {tid} changed after Jinja rendering — "
                    "templates must not contain Jinja variables.",
                )


class TestTemplatesManifest(unittest.TestCase):
    """Validate templates_manifest.json consistency with actual template files."""

    def test_manifest_is_valid_json(self):
        manifest = _load_manifest()
        self.assertIsInstance(manifest, list)

    def test_manifest_has_all_templates(self):
        manifest = _load_manifest()
        manifest_ids = sorted(e["id"] for e in manifest)
        self.assertEqual(manifest_ids, EXPECTED_TEMPLATE_IDS)

    def test_manifest_sha256_matches_files(self):
        manifest = _load_manifest()
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                content = _load_template(entry["id"])
                expected_hash = _sha256(content)
                self.assertEqual(
                    entry["sha256"],
                    expected_hash,
                    f"SHA256 mismatch for {entry['id']}: manifest says {entry['sha256']}, "
                    f"file has {expected_hash}",
                )

    def test_manifest_entries_have_required_fields(self):
        manifest = _load_manifest()
        required_fields = {"id", "category", "sha256", "tags", "summary", "est_tokens"}
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                missing = required_fields - set(entry.keys())
                self.assertEqual(missing, set(), f"Missing fields in {entry['id']}: {missing}")

    def test_manifest_categories_valid(self):
        manifest = _load_manifest()
        valid_categories = {"basic", "optimization", "clarifications"}
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                self.assertIn(
                    entry["category"],
                    valid_categories,
                    f"Invalid category '{entry['category']}' for {entry['id']}",
                )

    def test_manifest_tags_are_nonempty_lists(self):
        manifest = _load_manifest()
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                self.assertIsInstance(entry["tags"], list)
                self.assertGreater(len(entry["tags"]), 0, f"Empty tags for {entry['id']}")

    def test_manifest_est_tokens_positive(self):
        manifest = _load_manifest()
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                self.assertGreater(entry["est_tokens"], 0)

    def test_manifest_sorted_by_id(self):
        manifest = _load_manifest()
        ids = [e["id"] for e in manifest]
        self.assertEqual(ids, sorted(ids), "Manifest must be sorted by id for determinism")

    def test_template_content_not_empty(self):
        """Each template file must have non-trivial content."""
        for tid in EXPECTED_TEMPLATE_IDS:
            with self.subTest(template_id=tid):
                content = _load_template(tid)
                self.assertGreater(
                    len(content.strip()), 20,
                    f"Template {tid} is too short",
                )


if __name__ == "__main__":
    unittest.main()
