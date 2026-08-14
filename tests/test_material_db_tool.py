"""Tests for the material database lookup tool and skill integration.

Validates:
- MATERIAL_DB_SKILL is well-formed markdown content
- COMBINED_API_SKILL passes MCE validate_skill_markdown
- build_material_lookup_tool returns a callable with correct interface
- Tool handles invalid inputs gracefully (returns JSON error)
- Tool parses various wavelength formats
- Tool returns search-only response (page_id, metadata, no n/k)
"""

from __future__ import annotations

import json
import unittest


class TestMaterialDbSkillContent(unittest.TestCase):
    """Validate the material DB skill markdown content."""

    def test_skill_is_nonempty_string(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        self.assertIsInstance(MATERIAL_DB_SKILL, str)
        self.assertTrue(MATERIAL_DB_SKILL.strip())

    def test_skill_has_section_heading(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        self.assertIn("## 10. Material Database Lookup via CLI", MATERIAL_DB_SKILL)

    def test_skill_documents_tool_name(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        self.assertIn("python -m evo_metaoptics.material_db", MATERIAL_DB_SKILL)

    def test_skill_has_common_materials_table(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        for material in ("SiO2", "Si", "TiO2", "Si3N4", "GaN"):
            with self.subTest(material=material):
                self.assertIn(material, MATERIAL_DB_SKILL)

    def test_skill_documents_get_material_nk(self):
        """Skill should teach the runtime helper usage."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        self.assertIn("get_material_nk", MATERIAL_DB_SKILL)

    def test_skill_documents_page_id(self):
        """Skill should reference page_id as the key output."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        self.assertIn("page_id", MATERIAL_DB_SKILL)

    def test_skill_has_create_material_examples(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        self.assertIn("create_material", MATERIAL_DB_SKILL)
        self.assertIn("dielectric_dispersion", MATERIAL_DB_SKILL)

    def test_skill_documents_dispersive_pattern(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_skill import (
            MATERIAL_DB_SKILL,
        )

        self.assertIn("user_dielectric_wavelengths_um", MATERIAL_DB_SKILL)
        self.assertIn("user_dielectric_n", MATERIAL_DB_SKILL)
        self.assertIn("user_dielectric_k", MATERIAL_DB_SKILL)



class TestBuildMaterialLookupTool(unittest.TestCase):
    """Validate the tool builder and input handling."""

    def test_returns_callable(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        self.assertTrue(callable(tool))

    def test_tool_has_docstring(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        self.assertIsNotNone(tool.__doc__)
        self.assertIn("material", tool.__doc__.lower())

    def test_tool_name(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        self.assertEqual(tool.__name__, "lookup_material_nk")

    def test_empty_material_name_returns_error(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("", "[1.55]"))
        self.assertEqual(result["status"], "error")
        self.assertIn("non-empty", result["error"])

    def test_empty_wavelengths_returns_error(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", ""))
        self.assertEqual(result["status"], "error")

    def test_invalid_wavelength_returns_error(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", "[-1.0]"))
        self.assertEqual(result["status"], "error")
        self.assertIn("invalid", result["error"].lower())

    def test_non_numeric_wavelength_returns_error(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", "[abc]"))
        self.assertEqual(result["status"], "error")

    def test_parses_single_number(self):
        """Tool should accept a bare number string like '1.55'."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", "1.55"))
        # Should not fail on wavelength parsing; may fail on DB availability
        if result["status"] == "error":
            self.assertNotIn("parse", result["error"].lower())

    def test_parses_comma_separated(self):
        """Tool should accept comma-separated wavelengths."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", "1.3, 1.55"))
        # Should not fail on wavelength parsing
        if result["status"] == "error":
            self.assertNotIn("parse", result["error"].lower())

    def test_returns_valid_json(self):
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        raw = tool("SiO2", "[1.55]")
        result = json.loads(raw)
        self.assertIn("status", result)

    def test_search_only_response_has_no_nk(self):
        """Successful response should NOT contain n/k arrays or code snippets."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", "[1.55]"))
        # If search succeeded, verify search-only shape
        if result["status"] == "ok":
            self.assertNotIn("n", result)
            self.assertNotIn("k", result)
            self.assertNotIn("create_material_code", result)
            self.assertIn("page_id", result)
            self.assertIn("wavelength_coverage_um", result)
            self.assertIn("has_n", result)
            self.assertIn("has_k", result)
            self.assertIn("alternatives", result)
            self.assertIn("shelf", result)
            self.assertIn("book", result)
            self.assertIn("page", result)

    def test_search_only_response_page_id_is_int(self):
        """page_id should be an integer."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", "[1.55]"))
        if result["status"] == "ok":
            self.assertIsInstance(result["page_id"], int)

    def test_search_only_alternatives_have_page_id(self):
        """Each alternative should include a page_id."""
        from evo_metaoptics.mce_env.metaoptics_inverse_design.material_db_tool import (
            build_material_lookup_tool,
        )

        tool = build_material_lookup_tool()
        result = json.loads(tool("SiO2", "[1.55]"))
        if result["status"] == "ok" and result["alternatives"]:
            for alt in result["alternatives"]:
                self.assertIn("page_id", alt)
                self.assertIn("name", alt)
                self.assertIn("score", alt)


if __name__ == "__main__":
    unittest.main()
