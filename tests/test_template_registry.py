"""Tests for M1: Immutable Template Registry.

Validates:
- Registry loads all templates and returns manifest with IDs, hashes, tags, est_tokens.
- Registry rejects modification of template content (frozen dataclass).
- Hash verification detects tampered content.
- Batch retrieval works correctly.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evo_metaoptics.mce.template_registry import (
    TemplateEntry,
    TemplateIntegrityError,
    TemplateNotFoundError,
    TemplateRegistry,
)


class TestRegistryLoading(unittest.TestCase):
    """Registry loads templates and returns manifest."""

    def setUp(self):
        self.registry = TemplateRegistry()

    def test_registry_loads_all_templates(self):
        self.assertEqual(len(self.registry), 20)

    def test_manifest_returns_sorted_list(self):
        manifest = self.registry.manifest()
        self.assertIsInstance(manifest, list)
        ids = [e["id"] for e in manifest]
        self.assertEqual(ids, sorted(ids))

    def test_manifest_entries_have_required_fields(self):
        manifest = self.registry.manifest()
        required = {"id", "category", "sha256", "tags", "summary", "est_tokens"}
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                self.assertEqual(required - set(entry.keys()), set())

    def test_manifest_hashes_are_sha256(self):
        manifest = self.registry.manifest()
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                self.assertEqual(len(entry["sha256"]), 64)
                # Should be valid hex
                int(entry["sha256"], 16)

    def test_manifest_tags_are_lists(self):
        manifest = self.registry.manifest()
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                self.assertIsInstance(entry["tags"], list)
                self.assertGreater(len(entry["tags"]), 0)

    def test_manifest_est_tokens_positive(self):
        manifest = self.registry.manifest()
        for entry in manifest:
            with self.subTest(template_id=entry["id"]):
                self.assertGreater(entry["est_tokens"], 0)

    def test_template_ids_property(self):
        ids = self.registry.template_ids
        self.assertEqual(len(ids), 20)
        self.assertEqual(ids, sorted(ids))

    def test_contains(self):
        self.assertIn("basic_imports", self.registry)
        self.assertNotIn("nonexistent", self.registry)


class TestRegistryRetrieval(unittest.TestCase):
    """Registry retrieves content correctly."""

    def setUp(self):
        self.registry = TemplateRegistry()

    def test_get_template_returns_string(self):
        content = self.registry.get_template("basic_imports")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_get_template_content_matches_hash(self):
        for tid in self.registry.template_ids:
            with self.subTest(template_id=tid):
                content = self.registry.get_template(tid)
                entry = self.registry.get_entry(tid)
                actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                self.assertEqual(actual_hash, entry.sha256)

    def test_get_template_unknown_raises(self):
        with self.assertRaises(TemplateNotFoundError):
            self.registry.get_template("nonexistent")

    def test_get_templates_batch(self):
        ids = ["basic_imports", "solver_setup", "layer_stack"]
        result = self.registry.get_templates(ids)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(ids))
        for tid in ids:
            self.assertIsInstance(result[tid], str)

    def test_get_templates_empty_list(self):
        result = self.registry.get_templates([])
        self.assertEqual(result, {})

    def test_get_templates_unknown_raises(self):
        with self.assertRaises(TemplateNotFoundError):
            self.registry.get_templates(["basic_imports", "nonexistent"])

    def test_get_entry_returns_template_entry(self):
        entry = self.registry.get_entry("basic_imports")
        self.assertIsInstance(entry, TemplateEntry)
        self.assertEqual(entry.id, "basic_imports")
        self.assertEqual(entry.category, "basic")
        self.assertIsInstance(entry.tags, tuple)

    def test_get_entry_unknown_raises(self):
        with self.assertRaises(TemplateNotFoundError):
            self.registry.get_entry("nonexistent")


class TestRegistryImmutability(unittest.TestCase):
    """Registry content is immutable."""

    def test_template_entry_is_frozen(self):
        registry = TemplateRegistry()
        entry = registry.get_entry("basic_imports")
        with self.assertRaises(AttributeError):
            entry.id = "modified"  # type: ignore[misc]

    def test_manifest_returns_copy(self):
        """Modifying returned manifest doesn't affect registry."""
        registry = TemplateRegistry()
        manifest1 = registry.manifest()
        manifest1[0]["id"] = "TAMPERED"
        manifest2 = registry.manifest()
        self.assertNotEqual(manifest2[0]["id"], "TAMPERED")


class TestRegistryIntegrityDetection(unittest.TestCase):
    """Hash verification detects tampered content."""

    def test_tampered_content_detected_on_load(self):
        """Registry detects tampered template file at load time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a minimal valid manifest
            manifest = [{
                "id": "test_template",
                "category": "basic",
                "sha256": hashlib.sha256(b"original content\n").hexdigest(),
                "tags": ["test"],
                "summary": "Test template",
                "est_tokens": 10,
            }]
            (tmpdir / "templates_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            # Write tampered content
            (tmpdir / "test_template.py.jinja").write_text(
                "TAMPERED content\n", encoding="utf-8"
            )

            with self.assertRaises(TemplateIntegrityError):
                TemplateRegistry(templates_dir=tmpdir)

    def test_missing_template_file_detected(self):
        """Registry detects missing template file referenced in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest = [{
                "id": "missing_template",
                "category": "basic",
                "sha256": "abc123",
                "tags": ["test"],
                "summary": "Missing template",
                "est_tokens": 10,
            }]
            (tmpdir / "templates_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(FileNotFoundError):
                TemplateRegistry(templates_dir=tmpdir)

    def test_missing_manifest_detected(self):
        """Registry raises if manifest file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                TemplateRegistry(templates_dir=Path(tmpdir))


class TestRegistryDeterminism(unittest.TestCase):
    """Registry output is deterministic."""

    def test_manifest_deterministic(self):
        r1 = TemplateRegistry()
        r2 = TemplateRegistry()
        self.assertEqual(r1.manifest(), r2.manifest())

    def test_template_ids_deterministic(self):
        r1 = TemplateRegistry()
        r2 = TemplateRegistry()
        self.assertEqual(r1.template_ids, r2.template_ids)


if __name__ == "__main__":
    unittest.main()
