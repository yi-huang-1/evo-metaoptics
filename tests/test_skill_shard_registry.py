# pyright: reportMissingImports=false

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evo_metaoptics.mce.skill_shard_registry import (
    ShardEntry,
    ShardIntegrityError,
    ShardNotFoundError,
    SkillShardRegistry,
)
from evo_metaoptics.mce.skills import validate_skill_markdown


class TestRegistryLoading(unittest.TestCase):
    def setUp(self):
        self.registry = SkillShardRegistry()

    def test_registry_loads_all_shards(self):
        self.assertEqual(len(self.registry), 9)

    def test_manifest_returns_sorted_list(self):
        manifest = self.registry.manifest()
        self.assertIsInstance(manifest, list)
        ids = [e["id"] for e in manifest]
        self.assertEqual(ids, sorted(ids))

    def test_manifest_entries_have_required_fields(self):
        manifest = self.registry.manifest()
        required = {"id", "sha256", "tags", "summary", "est_tokens"}
        for entry in manifest:
            with self.subTest(shard_id=entry["id"]):
                self.assertEqual(required - set(entry.keys()), set())

    def test_manifest_hashes_are_sha256(self):
        manifest = self.registry.manifest()
        for entry in manifest:
            with self.subTest(shard_id=entry["id"]):
                self.assertEqual(len(entry["sha256"]), 64)
                int(entry["sha256"], 16)

    def test_manifest_tags_are_lists(self):
        manifest = self.registry.manifest()
        for entry in manifest:
            with self.subTest(shard_id=entry["id"]):
                self.assertIsInstance(entry["tags"], list)
                self.assertGreater(len(entry["tags"]), 0)

    def test_manifest_est_tokens_positive(self):
        manifest = self.registry.manifest()
        for entry in manifest:
            with self.subTest(shard_id=entry["id"]):
                self.assertGreater(entry["est_tokens"], 0)

    def test_shard_ids_property(self):
        ids = self.registry.shard_ids
        self.assertEqual(len(ids), 9)
        self.assertEqual(ids, sorted(ids))

    def test_contains(self):
        self.assertIn("core_rules", self.registry)
        self.assertNotIn("nonexistent", self.registry)


class TestRegistryRetrieval(unittest.TestCase):
    def setUp(self):
        self.registry = SkillShardRegistry()

    def test_get_shard_returns_string(self):
        content = self.registry.get_shard("core_rules")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_get_shard_content_matches_hash(self):
        for shard_id in self.registry.shard_ids:
            with self.subTest(shard_id=shard_id):
                content = self.registry.get_shard(shard_id)
                entry = self.registry.get_entry(shard_id)
                actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                self.assertEqual(actual_hash, entry.sha256)

    def test_get_shard_unknown_raises(self):
        with self.assertRaises(ShardNotFoundError):
            self.registry.get_shard("nonexistent")

    def test_get_shards_batch(self):
        ids = ["core_rules", "solver_setup", "materials_layers"]
        result = self.registry.get_shards(ids)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(ids))
        for shard_id in ids:
            self.assertIsInstance(result[shard_id], str)

    def test_get_shards_empty_list(self):
        result = self.registry.get_shards([])
        self.assertEqual(result, {})

    def test_get_shards_unknown_raises(self):
        with self.assertRaises(ShardNotFoundError):
            self.registry.get_shards(["core_rules", "nonexistent"])

    def test_get_entry_returns_shard_entry(self):
        entry = self.registry.get_entry("core_rules")
        self.assertIsInstance(entry, ShardEntry)
        self.assertEqual(entry.id, "core_rules")
        self.assertIsInstance(entry.tags, tuple)

    def test_get_entry_unknown_raises(self):
        with self.assertRaises(ShardNotFoundError):
            self.registry.get_entry("nonexistent")


class TestRegistryImmutability(unittest.TestCase):
    def test_shard_entry_is_frozen(self):
        registry = SkillShardRegistry()
        entry = registry.get_entry("core_rules")
        with self.assertRaises(AttributeError):
            entry.id = "modified"  # type: ignore[misc]

    def test_manifest_returns_copy(self):
        registry = SkillShardRegistry()
        manifest1 = registry.manifest()
        manifest1[0]["id"] = "TAMPERED"
        manifest2 = registry.manifest()
        self.assertNotEqual(manifest2[0]["id"], "TAMPERED")


class TestRegistryIntegrityDetection(unittest.TestCase):
    def test_tampered_content_detected_on_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest = [{
                "id": "test_shard",
                "sha256": hashlib.sha256(b"original content\n").hexdigest(),
                "tags": ["test"],
                "summary": "Test shard",
                "est_tokens": 10,
            }]
            (tmpdir / "shards_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (tmpdir / "test_shard.md").write_text("TAMPERED content\n", encoding="utf-8")

            with self.assertRaises(ShardIntegrityError):
                SkillShardRegistry(shards_dir=tmpdir)

    def test_missing_shard_file_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest = [{
                "id": "missing_shard",
                "sha256": "abc123",
                "tags": ["test"],
                "summary": "Missing shard",
                "est_tokens": 10,
            }]
            (tmpdir / "shards_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(FileNotFoundError):
                SkillShardRegistry(shards_dir=tmpdir)

    def test_missing_manifest_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                SkillShardRegistry(shards_dir=Path(tmpdir))


class TestRegistryDeterminism(unittest.TestCase):
    def test_manifest_deterministic(self):
        r1 = SkillShardRegistry()
        r2 = SkillShardRegistry()
        self.assertEqual(r1.manifest(), r2.manifest())

    def test_shard_ids_deterministic(self):
        r1 = SkillShardRegistry()
        r2 = SkillShardRegistry()
        self.assertEqual(r1.shard_ids, r2.shard_ids)


class TestShardContentValidation(unittest.TestCase):
    def setUp(self):
        self.registry = SkillShardRegistry()

    def test_all_shards_pass_validate_skill_markdown(self):
        for shard_id in self.registry.shard_ids:
            with self.subTest(shard_id=shard_id):
                content = self.registry.get_shard(shard_id)
                valid, reason = validate_skill_markdown(content, expected_name=shard_id)
                self.assertTrue(valid, reason)

    def test_shard_content_covers_api_surface(self):
        full_text = "\n".join(
            self.registry.get_shard(shard_id) for shard_id in self.registry.shard_ids
        )
        required_terms = [
            "get_solver_builder",
            "create_material",
            "ShapeGenerator",
            "add_source",
            "solver.solve",
            "transmission",
            "reflection",
            "torch.angle",
            "backward()",
            "torch.optim",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, full_text)

    def test_no_device_specific_content(self):
        forbidden_terms = ["pbte", "caf2", "5.2 um"]
        for shard_id in self.registry.shard_ids:
            with self.subTest(shard_id=shard_id):
                content = self.registry.get_shard(shard_id).lower()
                for term in forbidden_terms:
                    self.assertNotIn(term, content)

    def test_core_rules_has_new_device_parameter_signature(self):
        """core_rules must declare new device-explicit signature."""
        content = self.registry.get_shard("core_rules")
        self.assertIn("def solve_inverse_design(*, device: str = \"cpu\") -> SolverResults", content)

    def test_core_rules_no_device_agnostic_guidance(self):
        content = self.registry.get_shard("core_rules")
        self.assertNotIn("device-agnostic", content)
        self.assertNotIn("keep solver setup device-agnostic", content)
        self.assertNotIn("Do not set the compute device manually", content)


if __name__ == "__main__":
    unittest.main()
