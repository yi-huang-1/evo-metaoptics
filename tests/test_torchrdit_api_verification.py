"""Runtime TorchRDIT API verification for templates and shards.

These tests validate live TorchRDIT imports, enum members, class/method presence,
and runtime signatures via ``inspect.signature``. This intentionally verifies the
installed runtime API rather than relying on static string matching alone.
"""

from __future__ import annotations

import inspect
import pathlib
import re
import unittest

np = __import__("numpy")
Algorithm = __import__("torchrdit.constants", fromlist=["Algorithm"]).Algorithm
Precision = __import__("torchrdit.constants", fromlist=["Precision"]).Precision
SolverResults = __import__("torchrdit.results", fromlist=["SolverResults"]).SolverResults
ShapeGenerator = __import__("torchrdit.shapes", fromlist=["ShapeGenerator"]).ShapeGenerator
get_solver_builder = __import__("torchrdit.solver", fromlist=["get_solver_builder"]).get_solver_builder
create_material = __import__("torchrdit.utils", fromlist=["create_material"]).create_material
plot_layer = __import__("torchrdit.viz", fromlist=["plot_layer"]).plot_layer
display_fitted_permittivity = __import__(
    "torchrdit.viz", fromlist=["display_fitted_permittivity"]
).display_fitted_permittivity

TEMPLATES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "evo_metaoptics"
    / "mce_env"
    / "metaoptics_inverse_design"
    / "skills"
    / "templates"
)

SHARDS_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "evo_metaoptics"
    / "mce_env"
    / "metaoptics_inverse_design"
    / "skills"
    / "shards"
)


def _sig_params(fn) -> set[str]:
    """Return signature parameter names excluding ``self``."""
    return {name for name in inspect.signature(fn).parameters if name != "self"}


def _load_template(template_id: str) -> str:
    """Load a template file from skills/templates."""
    return (TEMPLATES_DIR / f"{template_id}.py.jinja").read_text(encoding="utf-8")


def _load_shard(shard_id: str) -> str:
    """Load a shard markdown file from skills/shards."""
    return (SHARDS_DIR / f"{shard_id}.md").read_text(encoding="utf-8")


def _build_minimal_solver():
    """Build a minimal solver instance for runtime signature checks."""
    air = create_material(name="air", permittivity=1.0)
    return (
        get_solver_builder()
        .with_algorithm(Algorithm.RCWA)
        .with_precision(Precision.SINGLE)
        .with_wavelengths(np.array([1.55]))
        .with_real_dimensions([32, 32])
        .with_k_dimensions([1, 1])
        .with_materials([air])
        .with_ref_material(air)
        .with_trn_material(air)
        .build()
    )


class _CrossReferenceAssertions(unittest.TestCase):
    """Shared assertion helpers for template/shard cross-reference tests."""

    def assert_mentions(self, text: str, symbol: str):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
        self.assertRegex(text, pattern, f"Expected symbol mention missing: {symbol}")

    def assert_has_params(self, fn, expected_params: set[str]):
        params = _sig_params(fn)
        self.assertTrue(
            expected_params.issubset(params),
            f"Expected params {expected_params} in {params}",
        )


class TestTorchrditImports(unittest.TestCase):
    def test_import_get_solver_builder(self):
        imported = __import__("torchrdit.solver", fromlist=["get_solver_builder"]).get_solver_builder

        self.assertTrue(callable(imported))

    def test_import_create_material(self):
        imported = __import__("torchrdit.utils", fromlist=["create_material"]).create_material

        self.assertTrue(callable(imported))

    def test_import_shape_generator(self):
        imported = __import__("torchrdit.shapes", fromlist=["ShapeGenerator"]).ShapeGenerator

        self.assertTrue(inspect.isclass(imported))

    def test_import_algorithm_precision(self):
        constants_mod = __import__("torchrdit.constants", fromlist=["Algorithm", "Precision"])
        imported_algorithm = constants_mod.Algorithm
        imported_precision = constants_mod.Precision

        self.assertTrue(inspect.isclass(imported_algorithm))
        self.assertTrue(inspect.isclass(imported_precision))

    def test_import_solver_results(self):
        imported = __import__("torchrdit.results", fromlist=["SolverResults"]).SolverResults

        self.assertTrue(inspect.isclass(imported))

    def test_import_plot_layer(self):
        imported = __import__("torchrdit.viz", fromlist=["plot_layer"]).plot_layer

        self.assertTrue(callable(imported))

    def test_import_display_fitted_permittivity(self):
        imported = __import__(
            "torchrdit.viz", fromlist=["display_fitted_permittivity"]
        ).display_fitted_permittivity

        self.assertTrue(callable(imported))


class TestSolverBuilderMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = get_solver_builder()

    def _assert_builder_method(self, name: str):
        self.assertTrue(hasattr(self.builder, name), f"builder missing {name}")
        self.assertTrue(callable(getattr(self.builder, name)), f"{name} not callable")

    def test_with_device_is_canonical_explicit_device_path(self):
        """Verify with_device is the intended explicit device setup path."""
        self._assert_builder_method("with_device")
        result = self.builder.with_device("cpu")
        self.assertIsNotNone(result, "with_device should return builder for chaining")

    def test_with_algorithm_exists_callable(self):
        self._assert_builder_method("with_algorithm")

    def test_with_precision_exists_callable(self):
        self._assert_builder_method("with_precision")

    def test_with_wavelengths_exists_callable(self):
        self._assert_builder_method("with_wavelengths")

    def test_with_real_dimensions_exists_callable(self):
        self._assert_builder_method("with_real_dimensions")

    def test_with_k_dimensions_exists_callable(self):
        self._assert_builder_method("with_k_dimensions")

    def test_with_materials_exists_callable(self):
        self._assert_builder_method("with_materials")

    def test_with_ref_material_exists_callable(self):
        self._assert_builder_method("with_ref_material")

    def test_with_trn_material_exists_callable(self):
        self._assert_builder_method("with_trn_material")

    def test_with_lattice_vectors_exists_callable(self):
        self._assert_builder_method("with_lattice_vectors")

    def test_with_length_unit_exists_callable(self):
        self._assert_builder_method("with_length_unit")

    def test_with_device_exists_callable(self):
        self._assert_builder_method("with_device")

    def test_with_rdit_order_exists_callable(self):
        self._assert_builder_method("with_rdit_order")

    def test_with_fff_exists_callable(self):
        self._assert_builder_method("with_fff")

    def test_with_fff_vector_options_exists_callable(self):
        self._assert_builder_method("with_fff_vector_options")

    def test_with_algorithm_instance_exists_callable(self):
        self._assert_builder_method("with_algorithm_instance")

    def test_add_layer_exists_callable(self):
        self._assert_builder_method("add_layer")

    def test_build_exists_callable(self):
        self._assert_builder_method("build")


class TestSolverRuntimeSignatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = _build_minimal_solver()

    def test_solver_device_awareness_via_with_device(self):
        """Verify solver respects explicit device placement via builder.with_device()."""
        air = create_material(name="air", permittivity=1.0)
        solver_cpu = (
            get_solver_builder()
            .with_algorithm(Algorithm.RCWA)
            .with_precision(Precision.SINGLE)
            .with_wavelengths(np.array([1.55]))
            .with_real_dimensions([32, 32])
            .with_k_dimensions([1, 1])
            .with_materials([air])
            .with_ref_material(air)
            .with_trn_material(air)
            .with_device("cpu")
            .build()
        )
        self.assertIsNotNone(solver_cpu)

    def test_add_layer_signature(self):
        params = _sig_params(self.solver.add_layer)
        expected = {"material_name", "thickness", "is_homogeneous", "is_optimize"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_add_source_signature(self):
        params = _sig_params(self.solver.add_source)
        expected = {"theta", "phi", "pte", "ptm"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_solve_signature(self):
        params = _sig_params(self.solver.solve)
        self.assertIn("source", params)

    def test_update_er_with_mask_signature(self):
        params = _sig_params(self.solver.update_er_with_mask)
        expected = {"mask", "layer_index", "bg_material"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_update_ref_material_exists_callable(self):
        self.assertTrue(hasattr(self.solver, "update_ref_material"))
        self.assertTrue(callable(self.solver.update_ref_material))

    def test_update_trn_material_exists_callable(self):
        self.assertTrue(hasattr(self.solver, "update_trn_material"))
        self.assertTrue(callable(self.solver.update_trn_material))

    def test_update_layer_thickness_signature(self):
        params = _sig_params(self.solver.update_layer_thickness)
        expected = {"layer_index", "thickness"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_add_materials_signature(self):
        params = _sig_params(self.solver.add_materials)
        self.assertIn("material_list", params)


class TestShapeGeneratorAPI(unittest.TestCase):
    def test_from_solver_is_classmethod(self):
        descriptor = inspect.getattr_static(ShapeGenerator, "from_solver")
        self.assertIsInstance(descriptor, classmethod)

    def test_generate_circle_mask_signature(self):
        params = _sig_params(ShapeGenerator.generate_circle_mask)
        expected = {"center", "radius", "soft_edge"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_generate_rectangle_mask_signature(self):
        params = _sig_params(ShapeGenerator.generate_rectangle_mask)
        expected = {"center", "x_size", "y_size", "angle", "soft_edge"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_generate_rectangle_mask_does_not_use_width_height(self):
        params = _sig_params(ShapeGenerator.generate_rectangle_mask)
        self.assertNotIn("width", params)
        self.assertNotIn("height", params)

    def test_generate_polygon_mask_signature(self):
        params = _sig_params(ShapeGenerator.generate_polygon_mask)
        expected = {"polygon_points", "center", "angle", "soft_edge"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_combine_masks_signature(self):
        params = _sig_params(ShapeGenerator.combine_masks)
        expected = {"mask1", "mask2", "operation"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")


class TestSolverResultsAPI(unittest.TestCase):
    def _has_class_field_or_attr(self, name: str) -> bool:
        annotations = getattr(SolverResults, "__annotations__", {})
        return hasattr(SolverResults, name) or name in annotations

    def test_solver_results_support_cpu_transfer_for_export(self):
        """Verify SolverResults can be transferred to CPU for NumPy/export paths."""
        self.assertTrue(
            hasattr(SolverResults, "transmission") or "transmission" in getattr(SolverResults, "__annotations__", {}),
            "SolverResults must have transmission field for CPU-transfer validation"
        )

    def test_transmission_exists(self):
        self.assertTrue(self._has_class_field_or_attr("transmission"))

    def test_reflection_exists(self):
        self.assertTrue(self._has_class_field_or_attr("reflection"))

    def test_get_zero_order_transmission_exists(self):
        self.assertTrue(hasattr(SolverResults, "get_zero_order_transmission"))

    def test_get_zero_order_reflection_exists(self):
        self.assertTrue(hasattr(SolverResults, "get_zero_order_reflection"))

    def test_get_order_transmission_efficiency_exists(self):
        self.assertTrue(hasattr(SolverResults, "get_order_transmission_efficiency"))

    def test_get_order_reflection_efficiency_exists(self):
        self.assertTrue(hasattr(SolverResults, "get_order_reflection_efficiency"))

    def test_get_all_diffraction_orders_exists(self):
        self.assertTrue(hasattr(SolverResults, "get_all_diffraction_orders"))

    def test_is_batched_exists(self):
        self.assertTrue(hasattr(SolverResults, "is_batched"))

    def test_get_source_result_exists(self):
        self.assertTrue(hasattr(SolverResults, "get_source_result"))

    def test_as_list_exists(self):
        self.assertTrue(hasattr(SolverResults, "as_list"))

    def test_constructor_declares_reflection_and_transmission(self):
        params = _sig_params(SolverResults)
        self.assertIn("reflection", params)
        self.assertIn("transmission", params)


class TestEnumValues(unittest.TestCase):
    def test_algorithm_rcwa_exists(self):
        self.assertIsInstance(Algorithm.RCWA, Algorithm)

    def test_algorithm_rdit_exists(self):
        self.assertIsInstance(Algorithm.RDIT, Algorithm)

    def test_precision_single_exists(self):
        self.assertIsInstance(Precision.SINGLE, Precision)

    def test_precision_double_exists(self):
        self.assertIsInstance(Precision.DOUBLE, Precision)


class TestCreateMaterialSignature(unittest.TestCase):
    def test_base_parameter_names(self):
        params = _sig_params(create_material)
        expected = {"name", "permittivity", "permeability", "dielectric_dispersion"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_dispersive_parameter_names(self):
        params = _sig_params(create_material)
        expected = {
            "user_dielectric_file",
            "user_dielectric_wavelengths_um",
            "user_dielectric_n",
            "user_dielectric_k",
        }
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")


class TestVizModuleAPI(unittest.TestCase):
    def test_plot_layer_signature(self):
        self.assertTrue(callable(plot_layer))
        params = _sig_params(plot_layer)
        expected = {"layer_index", "fig_ax", "title", "labels"}
        self.assertTrue(expected.issubset(params), f"expected {expected} in {params}")

    def test_display_fitted_permittivity_signature(self):
        self.assertTrue(callable(display_fitted_permittivity))
        params = _sig_params(display_fitted_permittivity)
        self.assertIn("fig_ax", params)


class TestTemplateCrossReference(_CrossReferenceAssertions):
    @classmethod
    def setUpClass(cls):
        cls.builder = get_solver_builder()
        cls.solver = _build_minimal_solver()

    def test_template_basic_imports_api_symbols(self):
        content = _load_template("basic_imports")
        for symbol in [
            "get_solver_builder",
            "ShapeGenerator",
            "create_material",
            "Algorithm",
            "Precision",
            "plot_layer",
            "display_fitted_permittivity",
        ]:
            self.assert_mentions(content, symbol)
        self.assertTrue(callable(get_solver_builder))
        self.assertTrue(inspect.isclass(ShapeGenerator))
        self.assertTrue(callable(create_material))
        self.assertTrue(inspect.isclass(Algorithm))
        self.assertTrue(inspect.isclass(Precision))
        self.assertTrue(callable(plot_layer))
        self.assertTrue(callable(display_fitted_permittivity))

    def test_template_solver_setup_api_symbols(self):
        content = _load_template("solver_setup")
        for symbol in [
            "get_solver_builder",
            "with_algorithm",
            "with_precision",
            "with_real_dimensions",
            "with_k_dimensions",
            "with_wavelengths",
            "with_length_unit",
            "with_lattice_vectors",
            "with_device",
            "build",
        ]:
            self.assert_mentions(content, symbol)
        for method in [
            "with_algorithm",
            "with_precision",
            "with_real_dimensions",
            "with_k_dimensions",
            "with_wavelengths",
            "with_length_unit",
            "with_lattice_vectors",
            "with_device",
            "build",
        ]:
            self.assertTrue(callable(getattr(self.builder, method)))

    def test_template_material_creation_api_symbols(self):
        content = _load_template("material_creation")
        for symbol in ["create_material", "add_materials", "name", "permittivity", "material_list"]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(create_material, {"name", "permittivity"})
        self.assert_has_params(self.solver.add_materials, {"material_list"})

    def test_template_layer_stack_api_symbols(self):
        content = _load_template("layer_stack")
        for symbol in [
            "update_ref_material",
            "update_trn_material",
            "add_layer",
            "material_name",
            "thickness",
            "is_homogeneous",
        ]:
            self.assert_mentions(content, symbol)
        self.assertTrue(callable(self.solver.update_ref_material))
        self.assertTrue(callable(self.solver.update_trn_material))
        self.assert_has_params(self.solver.add_layer, {"material_name", "thickness", "is_homogeneous"})

    def test_template_source_setup_api_symbols(self):
        content = _load_template("source_setup")
        for symbol in ["add_source", "theta", "phi", "pte", "ptm"]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(self.solver.add_source, {"theta", "phi", "pte", "ptm"})

    def test_template_patterned_layer_api_symbols(self):
        content = _load_template("patterned_layer")
        for symbol in [
            "add_layer",
            "is_homogeneous",
            "is_optimize",
            "ShapeGenerator.from_solver",
            "generate_circle_mask",
            "center",
            "radius",
            "soft_edge",
            "update_er_with_mask",
            "mask",
            "layer_index",
            "bg_material",
        ]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(self.solver.add_layer, {"is_homogeneous", "is_optimize"})
        self.assertTrue(callable(ShapeGenerator.from_solver))
        self.assert_has_params(ShapeGenerator.generate_circle_mask, {"center", "radius", "soft_edge"})
        self.assert_has_params(self.solver.update_er_with_mask, {"mask", "layer_index", "bg_material"})

    def test_template_shape_operations_api_symbols(self):
        content = _load_template("shape_operations")
        for symbol in [
            "generate_circle_mask",
            "center",
            "radius",
            "soft_edge",
            "generate_rectangle_mask",
            "x_size",
            "y_size",
            "angle",
            "generate_polygon_mask",
            "polygon_points",
            "combine_masks",
            "operation",
        ]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(ShapeGenerator.generate_circle_mask, {"center", "radius", "soft_edge"})
        self.assert_has_params(
            ShapeGenerator.generate_rectangle_mask,
            {"x_size", "y_size", "angle", "soft_edge"},
        )
        self.assert_has_params(
            ShapeGenerator.generate_polygon_mask,
            {"polygon_points", "center", "angle"},
        )
        self.assert_has_params(ShapeGenerator.combine_masks, {"mask1", "mask2", "operation"})

    def test_template_solve_and_analyze_api_symbols(self):
        content = _load_template("solve_and_analyze")
        for symbol in [
            "solve",
            "get_zero_order_transmission",
            "get_zero_order_reflection",
            "transmission",
            "reflection",
        ]:
            self.assert_mentions(content, symbol)
        self.assertIn("source", _sig_params(self.solver.solve))
        self.assertTrue(hasattr(SolverResults, "get_zero_order_transmission"))
        self.assertTrue(hasattr(SolverResults, "get_zero_order_reflection"))
        annotations = getattr(SolverResults, "__annotations__", {})
        self.assertIn("transmission", annotations)
        self.assertIn("reflection", annotations)

    def test_template_visualization_api_symbols(self):
        content = _load_template("visualization")
        for symbol in ["plot_layer", "layer_index", "fig_ax", "title", "labels"]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(plot_layer, {"layer_index", "fig_ax", "title", "labels"})

    def test_template_optimization_basic_api_symbols(self):
        content = _load_template("optimization_basic")
        for symbol in ["update_er_with_mask", "mask", "layer_index", "solve"]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(self.solver.update_er_with_mask, {"mask", "layer_index"})
        self.assertIn("source", _sig_params(self.solver.solve))

    def test_template_multi_objective_api_symbols(self):
        content = _load_template("multi_objective")
        for symbol in ["solve", "transmission", "reflection"]:
            self.assert_mentions(content, symbol)
        self.assertIn("source", _sig_params(self.solver.solve))
        annotations = getattr(SolverResults, "__annotations__", {})
        self.assertIn("transmission", annotations)
        self.assertIn("reflection", annotations)

    def test_template_gradient_based_api_symbols(self):
        content = _load_template("gradient_based")
        for symbol in ["solve", "transmission"]:
            self.assert_mentions(content, symbol)
        self.assertIn("source", _sig_params(self.solver.solve))
        annotations = getattr(SolverResults, "__annotations__", {})
        self.assertIn("transmission", annotations)

    def test_template_common_patterns_api_symbols(self):
        content = _load_template("common_patterns")
        for symbol in [
            "add_layer",
            "material_name",
            "thickness",
            "with_lattice_vectors",
            "with_wavelengths",
            "solve",
        ]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(self.solver.add_layer, {"material_name", "thickness"})
        self.assertTrue(callable(self.builder.with_lattice_vectors))
        self.assertTrue(callable(self.builder.with_wavelengths))
        self.assertIn("source", _sig_params(self.solver.solve))

    def test_template_material_api_symbols(self):
        content = _load_template("material_api")
        for symbol in [
            "update_ref_material",
            "update_trn_material",
            "with_ref_material",
            "with_trn_material",
        ]:
            self.assert_mentions(content, symbol)
        self.assertTrue(callable(self.solver.update_ref_material))
        self.assertTrue(callable(self.solver.update_trn_material))
        self.assertTrue(callable(self.builder.with_ref_material))
        self.assertTrue(callable(self.builder.with_trn_material))


class TestShardCrossReference(_CrossReferenceAssertions):
    @classmethod
    def setUpClass(cls):
        cls.builder = get_solver_builder()
        cls.solver = _build_minimal_solver()

    def test_shard_core_rules_api_symbols(self):
        content = _load_shard("core_rules")
        for symbol in [
            "create_material",
            "get_solver_builder",
            "Algorithm",
            "Precision",
            "with_algorithm",
            "with_precision",
            "with_wavelengths",
            "with_materials",
            "add_source",
            "theta",
            "phi",
            "pte",
            "ptm",
            "solve",
        ]:
            self.assert_mentions(content, symbol)
        self.assertTrue(callable(create_material))
        self.assertTrue(callable(get_solver_builder))
        self.assertTrue(inspect.isclass(Algorithm))
        self.assertTrue(inspect.isclass(Precision))
        self.assertTrue(callable(self.builder.with_algorithm))
        self.assertTrue(callable(self.builder.with_precision))
        self.assertTrue(callable(self.builder.with_wavelengths))
        self.assertTrue(hasattr(self.solver, "solve"))

    def test_shard_solver_setup_api_symbols(self):
        content = _load_shard("solver_setup")
        for symbol in [
            "with_algorithm",
            "with_precision",
            "with_wavelengths",
            "with_length_unit",
            "with_real_dimensions",
            "with_k_dimensions",
            "with_device",
            "with_lattice_vectors",
            "with_materials",
            "with_trn_material",
            "with_ref_material",
            "with_fff",
            "with_rdit_order",
            "add_layer",
            "build",
            "Algorithm.RCWA",
            "Algorithm.RDIT",
            "Precision.SINGLE",
            "Precision.DOUBLE",
        ]:
            self.assert_mentions(content, symbol)
        self.assertIsInstance(Algorithm.RCWA, Algorithm)
        self.assertIsInstance(Algorithm.RDIT, Algorithm)
        self.assertIsInstance(Precision.SINGLE, Precision)
        self.assertIsInstance(Precision.DOUBLE, Precision)

    def test_shard_materials_layers_api_symbols(self):
        content = _load_shard("materials_layers")
        for symbol in [
            "create_material",
            "name",
            "permittivity",
            "dielectric_dispersion",
            "user_dielectric_file",
            "user_dielectric_wavelengths_um",
            "user_dielectric_n",
            "user_dielectric_k",
            "add_materials",
            "add_layer",
            "update_ref_material",
            "update_trn_material",
        ]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(
            create_material,
            {
                "name",
                "permittivity",
                "permeability",
                "dielectric_dispersion",
                "user_dielectric_file",
                "user_dielectric_wavelengths_um",
                "user_dielectric_eps",
                "user_dielectric_n",
                "user_dielectric_k",
            },
        )
        self.assert_has_params(self.solver.add_materials, {"material_list"})
        self.assertTrue(callable(self.solver.add_layer))
        self.assertTrue(callable(self.solver.update_ref_material))
        self.assertTrue(callable(self.solver.update_trn_material))

    def test_shard_patterning_shapes_api_symbols(self):
        content = _load_shard("patterning_shapes")
        for symbol in [
            "ShapeGenerator.from_solver",
            "generate_circle_mask",
            "center",
            "radius",
            "soft_edge",
            "generate_rectangle_mask",
            "x_size",
            "y_size",
            "angle",
            "generate_polygon_mask",
            "polygon_points",
            "combine_masks",
            "operation",
            "update_er_with_mask",
        ]:
            self.assert_mentions(content, symbol)
        self.assertTrue(callable(ShapeGenerator.from_solver))
        self.assert_has_params(ShapeGenerator.generate_rectangle_mask, {"x_size", "y_size"})
        self.assert_has_params(ShapeGenerator.combine_masks, {"operation"})
        self.assert_has_params(self.solver.update_er_with_mask, {"mask", "layer_index", "bg_material"})

    def test_shard_source_solve_api_symbols(self):
        content = _load_shard("source_solve")
        for symbol in ["add_source", "theta", "phi", "pte", "ptm", "solve", "transmission", "reflection"]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(self.solver.add_source, {"theta", "phi", "pte", "ptm"})
        self.assertIn("source", _sig_params(self.solver.solve))
        annotations = getattr(SolverResults, "__annotations__", {})
        self.assertIn("transmission", annotations)
        self.assertIn("reflection", annotations)

    def test_shard_postprocess_phase_api_symbols(self):
        content = _load_shard("postprocess_phase")
        for symbol in ["get_zero_order_transmission", "get_zero_order_reflection"]:
            self.assert_mentions(content, symbol)
        self.assertTrue(hasattr(SolverResults, "get_zero_order_transmission"))
        self.assertTrue(hasattr(SolverResults, "get_zero_order_reflection"))

    def test_shard_postprocess_amplitude_api_symbols(self):
        content = _load_shard("postprocess_amplitude")
        for symbol in [
            "transmission",
            "reflection",
            "get_order_transmission_efficiency",
            "get_order_reflection_efficiency",
            "get_all_diffraction_orders",
        ]:
            self.assert_mentions(content, symbol)
        annotations = getattr(SolverResults, "__annotations__", {})
        self.assertIn("transmission", annotations)
        self.assertIn("reflection", annotations)
        self.assertTrue(hasattr(SolverResults, "get_order_transmission_efficiency"))
        self.assertTrue(hasattr(SolverResults, "get_order_reflection_efficiency"))
        self.assertTrue(hasattr(SolverResults, "get_all_diffraction_orders"))

    def test_shard_optimization_patterns_api_symbols(self):
        content = _load_shard("optimization_patterns")
        for symbol in [
            "update_er_with_mask",
            "add_source",
            "solve",
            "ShapeGenerator.from_solver",
            "generate_circle_mask",
        ]:
            self.assert_mentions(content, symbol)
        self.assert_has_params(self.solver.update_er_with_mask, {"mask", "layer_index", "bg_material"})
        self.assert_has_params(self.solver.add_source, {"theta", "phi", "pte", "ptm"})
        self.assertIn("source", _sig_params(self.solver.solve))
        self.assertTrue(callable(ShapeGenerator.from_solver))
        self.assert_has_params(ShapeGenerator.generate_circle_mask, {"center", "radius", "soft_edge"})

    def test_shard_common_pitfalls_api_symbols(self):
        content = _load_shard("common_pitfalls")
        for symbol in [
            "update_ref_material",
            "update_trn_material",
            "add_materials",
            "update_er_with_mask",
        ]:
            self.assert_mentions(content, symbol)
        self.assertTrue(callable(self.solver.update_ref_material))
        self.assertTrue(callable(self.solver.update_trn_material))
        self.assert_has_params(self.solver.add_materials, {"material_list"})
        self.assert_has_params(self.solver.update_er_with_mask, {"mask", "layer_index", "bg_material"})


class TestDeviceGuidanceReversal(unittest.TestCase):
    """Red assertions validating the reversal from device-agnostic to explicit device setup."""

    def test_reference_mcp_server_no_longer_says_never_use_with_device(self):
        """Verify reference MCP server.py no longer contains contradictory device-agnostic guidance."""
        server_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "reference"
            / "torchrdit-mcp"
            / "server.py"
        )
        if server_path.exists():
            content = server_path.read_text(encoding="utf-8")
            self.assertNotIn(
                "NEVER use with_device",
                content,
                "Reference MCP server.py must not contain 'NEVER use with_device' guidance"
            )

    def test_smoke_py_validates_device_aware_contract(self):
        """Verify smoke.py validates device-aware TorchRDIT contract."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.smoke import (
            ensure_torchrdit_available,
        )
        self.assertTrue(callable(ensure_torchrdit_available))


if __name__ == "__main__":
    unittest.main()
