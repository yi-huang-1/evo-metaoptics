# pyright: reportMissingImports=false

from __future__ import annotations

import unittest

from evo_metaoptics.mce.skills import validate_skill_markdown
from evo_metaoptics.mce_env.metaoptics_inverse_design.preloaded_skill import (
    _HEADER,
    _REFERENCE_FILES,
    compose_preloaded_template_skill,
)


class TestPreloadedSkillBootstrap(unittest.TestCase):
    def test_contains_header_and_skill_overview(self) -> None:
        doc = compose_preloaded_template_skill("Design a simple device.")
        self.assertIn(_HEADER, doc)
        self.assertIn("## Skill Overview", doc)

    def test_keeps_cold_start_safe_imports(self) -> None:
        doc = compose_preloaded_template_skill("Design a simple device.")
        self.assertIn("from torchrdit.solver import get_solver_builder", doc)
        self.assertIn("from torchrdit.utils import create_material", doc)
        self.assertIn("from torchrdit.results import SolverResults", doc)

    def test_keeps_builder_source_and_solve_patterns(self) -> None:
        doc = compose_preloaded_template_skill("Design a simple device.")
        self.assertIn("get_solver_builder()", doc)
        self.assertIn("builder.build()", doc)
        self.assertIn("solver.add_source(", doc)
        self.assertIn("solver.solve(source)", doc)

    def test_keeps_material_and_layer_bootstrap(self) -> None:
        doc = compose_preloaded_template_skill("Design a simple device.")
        self.assertIn("create_material", doc)
        self.assertIn("solver.add_materials", doc)
        self.assertIn("solver.add_layer(", doc)
        self.assertIn("solver.update_ref_material", doc)
        self.assertIn("solver.update_trn_material", doc)

    def test_lists_progressive_disclosure_reference_files(self) -> None:
        doc = compose_preloaded_template_skill("Design a simple device.")
        for ref_path, _description in _REFERENCE_FILES:
            with self.subTest(ref_path=ref_path):
                self.assertIn(ref_path, doc)

    def test_query_does_not_change_bootstrap(self) -> None:
        simple = compose_preloaded_template_skill("Design a simple filter")
        advanced = compose_preloaded_template_skill(
            "Optimize transmitted phase at multiple angles with a grating"
        )
        self.assertEqual(simple, advanced)

    def test_no_query_specific_tier_two_sections_remain(self) -> None:
        doc = compose_preloaded_template_skill(
            "Optimize transmitted phase at multiple angles with a grating"
        )
        forbidden_headings = [
            "Shape & Patterning",
            "Phase Analysis",
            "Amplitude & Efficiency",
            "Multi-Objective Optimization",
            "Advanced Solver Configuration",
            "Multi-Angle / Batched Sources",
        ]
        for heading in forbidden_headings:
            with self.subTest(heading=heading):
                self.assertNotIn(heading, doc)

    def test_bootstrap_token_budget_is_tight(self) -> None:
        doc = compose_preloaded_template_skill("Design a simple device.")
        estimated_tokens = len(doc) / 4
        self.assertLess(estimated_tokens, 1500)

    def test_output_format_remains_valid(self) -> None:
        doc = compose_preloaded_template_skill("Design a simple device.")
        valid, error = validate_skill_markdown(doc, expected_name="learning-context")
        self.assertTrue(valid, error)
        self.assertTrue(doc.startswith("---\n"))
        self.assertTrue(doc.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
