from pathlib import Path
import unittest
from unittest.mock import patch


class _FakeNkMaterial:
    _N_BY_PAGE = {
        "Malitson": 2.0,
        "Li": 3.0,
        "A": 1.7,
        "B": 1.8,
    }

    def __init__(self, **kwargs) -> None:
        self._page = str(kwargs.get("page", ""))

    def get_refractive_index(self, _wavelength_nm: float) -> float:
        return float(self._N_BY_PAGE.get(self._page, 2.0))

    def get_extinction_coefficient(self, _wavelength_nm: float) -> float:
        return 0.1


class _FakeIndex:
    def __init__(self, search_result) -> None:
        self.db_path = Path("/tmp/fake-index.sqlite")
        self._search_result = search_result
        self.search_calls = 0

    def search(self, _query: str):
        self.search_calls += 1
        return self._search_result


class _FakeLookupIndex:
    def __init__(self, mapping) -> None:
        self.db_path = Path("/tmp/fake-index.sqlite")
        self._mapping = dict(mapping)
        self.search_queries: list[str] = []

    def search(self, query: str):
        self.search_queries.append(str(query))
        result = self._mapping.get(str(query))
        if result is None:
            from evo_metaoptics.material_db.search import MaterialSearchResult

            return MaterialSearchResult(query=str(query), normalized_query=str(query), matches=())
        return result


class TestInverseMaterialResolver(unittest.TestCase):
    def test_resolve_by_name_prefers_exact_match(self) -> None:
        from evo_metaoptics.material_db.search import (
            MaterialMatch,
            MaterialPage,
            MaterialSearchResult,
        )
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        search_result = MaterialSearchResult(
            query="SiO2",
            normalized_query="sio2",
            matches=(
                MaterialMatch(
                    score=100.0,
                    material_id=1,
                    shelf="main",
                    book="SiO2",
                    display_name="Silicon Dioxide",
                    name_plain="SiO2",
                    title="SiO2",
                    description="",
                    other_names=(),
                    alias_hits=(),
                    pages=(
                        MaterialPage(
                            page="Malitson",
                            page_name="Malitson",
                            data_path="main/SiO2/Malitson.yml",
                            page_id=123,
                            coverage_min=0.2,
                            coverage_max=5.0,
                            has_n=True,
                            has_k=True,
                        ),
                    ),
                ),
            ),
        )

        resolver = MaterialResolver(
            index=_FakeIndex(search_result),
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )
        resolved = resolver.resolve({"type": "name", "value": "SiO2"}, [1.55], context={})
        self.assertEqual("exact_match", resolved.selection.reason)
        self.assertEqual("Malitson", resolved.selection.page)
        material_spec = resolved.to_material_spec()
        self.assertIn("n", material_spec)
        self.assertIn("k", material_spec)
        self.assertNotIn("permittivity", material_spec)

    def test_resolve_by_page_id_does_not_search(self) -> None:
        from evo_metaoptics.material_db.search import (
            MaterialPageRef,
            MaterialSearchResult,
        )
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        index = _FakeIndex(
            MaterialSearchResult(query="", normalized_query="", matches=()),
        )
        resolver = MaterialResolver(
            index=index,
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )

        page_ref = MaterialPageRef(
            page_id=88,
            material_id=2,
            shelf="main",
            book="Si",
            page="Li",
            page_name="Li",
            data_path="main/Si/Li.yml",
            coverage_min=1.0,
            coverage_max=2.0,
            has_n=True,
            has_k=True,
        )
        with patch(
            "evo_metaoptics.meta_design.material_resolver.get_page_by_id",
            return_value=page_ref,
        ):
            resolved = resolver.resolve({"type": "page_id", "value": 88}, [1.55], context={})

        self.assertEqual("page_id", resolved.selection.reason)
        self.assertEqual(0, index.search_calls)

    def test_resolve_facts_ir_materials_prefers_inline_nk_over_db(self) -> None:
        from evo_metaoptics.material_db.search import MaterialSearchResult
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        index = _FakeIndex(MaterialSearchResult(query="", normalized_query="", matches=()))
        resolver = MaterialResolver(
            index=index,
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )

        canonical_ir = {
            "sim": {"wavelength_um": [1.55]},
            "mats": [
                {"idx": 0, "name": "Si", "opt": {"kind": "nk_const", "n": 3.4, "k": 0.02}},
            ],
        }
        resolved, diagnostics = resolver.resolve_facts_ir_materials(canonical_ir)

        self.assertEqual(0, index.search_calls)
        self.assertEqual(1, resolved["resolved_count"])
        self.assertEqual("inline_nk_const", resolved["by_index"]["0"]["source_kind"])
        self.assertEqual([3.4], resolved["by_index"]["0"]["n"])
        self.assertEqual([0.02], resolved["by_index"]["0"]["k"])
        self.assertEqual(0, diagnostics["cache_hits"])
        self.assertEqual(1, diagnostics["cache_misses"])

    def test_resolve_facts_ir_materials_uses_db_lookup_when_needed(self) -> None:
        from evo_metaoptics.material_db.search import (
            MaterialMatch,
            MaterialPage,
            MaterialSearchResult,
        )
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        search_result = MaterialSearchResult(
            query="SiO2",
            normalized_query="sio2",
            matches=(
                MaterialMatch(
                    score=95.0,
                    material_id=11,
                    shelf="main",
                    book="SiO2",
                    display_name="Silicon Dioxide",
                    name_plain="SiO2",
                    title="SiO2",
                    description="",
                    other_names=(),
                    alias_hits=(),
                    pages=(
                        MaterialPage(
                            page="Malitson",
                            page_name="Malitson",
                            data_path="main/SiO2/Malitson.yml",
                            page_id=123,
                            coverage_min=0.2,
                            coverage_max=5.0,
                            has_n=True,
                            has_k=True,
                        ),
                    ),
                ),
            ),
        )
        index = _FakeLookupIndex({"sio2": search_result, "SiO2": search_result})
        resolver = MaterialResolver(
            index=index,
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )
        canonical_ir = {
            "sim": {"wavelength_um": [1.55]},
            "mats": [
                {"idx": 0, "name": "SiO2", "opt": {"kind": "db_lookup", "db_key": "sio2"}},
            ],
        }
        resolved, diagnostics = resolver.resolve_facts_ir_materials(canonical_ir)

        self.assertEqual(["sio2"], index.search_queries)
        self.assertEqual("db_lookup", resolved["by_index"]["0"]["source_kind"])
        self.assertEqual(123, resolved["by_index"]["0"]["selected_page_id"])
        self.assertEqual(1, diagnostics["entries"][0]["candidate_count"])
        self.assertEqual("page_id:123", diagnostics["entries"][0]["selected_record_id"])

    def test_same_base_name_instances_remain_distinct(self) -> None:
        from evo_metaoptics.material_db.search import MaterialSearchResult
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        resolver = MaterialResolver(
            index=_FakeIndex(MaterialSearchResult(query="", normalized_query="", matches=())),
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )
        canonical_ir = {
            "sim": {"wavelength_um": [5.2]},
            "mats": [
                {"idx": 0, "name": "PbTe", "opt": {"kind": "nk_const", "n": 4.8, "k": 0.0}},
                {"idx": 1, "name": "PbTe", "opt": {"kind": "nk_const", "n": 5.4, "k": 0.01}},
            ],
        }
        resolved, diagnostics = resolver.resolve_facts_ir_materials(canonical_ir)

        self.assertEqual(2, resolved["resolved_count"])
        self.assertEqual([4.8], resolved["by_index"]["0"]["n"])
        self.assertEqual([5.4], resolved["by_index"]["1"]["n"])
        self.assertEqual(2, diagnostics["cache_misses"])

    def test_db_lookup_tiebreak_is_deterministic_and_cache_hits_on_repeat(self) -> None:
        from evo_metaoptics.material_db.search import (
            MaterialMatch,
            MaterialPage,
            MaterialSearchResult,
        )
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        search_result = MaterialSearchResult(
            query="si",
            normalized_query="si",
            matches=(
                MaterialMatch(
                    score=90.0,
                    material_id=20,
                    shelf="main",
                    book="Si",
                    display_name="Silicon",
                    name_plain="Si",
                    title="Silicon",
                    description="",
                    other_names=(),
                    alias_hits=(),
                    pages=(
                        MaterialPage(
                            page="B",
                            page_name="B",
                            data_path="main/Si/B.yml",
                            page_id=500,
                            coverage_min=1.0,
                            coverage_max=2.0,
                            has_n=True,
                            has_k=True,
                        ),
                    ),
                ),
                MaterialMatch(
                    score=90.0,
                    material_id=21,
                    shelf="main",
                    book="Si",
                    display_name="Silicon",
                    name_plain="Si",
                    title="Silicon",
                    description="",
                    other_names=(),
                    alias_hits=(),
                    pages=(
                        MaterialPage(
                            page="A",
                            page_name="A",
                            data_path="main/Si/A.yml",
                            page_id=100,
                            coverage_min=1.0,
                            coverage_max=2.0,
                            has_n=True,
                            has_k=True,
                        ),
                    ),
                ),
            ),
        )
        index = _FakeLookupIndex({"si": search_result})
        resolver = MaterialResolver(
            index=index,
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )
        canonical_ir = {
            "sim": {"wavelength_um": [1.55]},
            "mats": [
                {"idx": 0, "name": "Si", "opt": {"kind": "db_lookup", "db_key": "si"}},
            ],
        }

        first_resolved, first_diag = resolver.resolve_facts_ir_materials(canonical_ir)
        second_resolved, second_diag = resolver.resolve_facts_ir_materials(canonical_ir)

        self.assertEqual(100, first_resolved["by_index"]["0"]["selected_page_id"])
        self.assertEqual(0, first_diag["cache_hits"])
        self.assertEqual(1, first_diag["cache_misses"])
        self.assertEqual(1, second_diag["cache_hits"])
        self.assertEqual(0, second_diag["cache_misses"])
        self.assertEqual(first_resolved, second_resolved)

    def test_only_db_lookup_materials_use_fuzzy_search(self) -> None:
        from evo_metaoptics.material_db.search import (
            MaterialMatch,
            MaterialPage,
            MaterialSearchResult,
        )
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        search_result = MaterialSearchResult(
            query="sio2",
            normalized_query="sio2",
            matches=(
                MaterialMatch(
                    score=95.0,
                    material_id=11,
                    shelf="main",
                    book="SiO2",
                    display_name="SiO2",
                    name_plain="SiO2",
                    title="SiO2",
                    description="",
                    other_names=(),
                    alias_hits=(),
                    pages=(
                        MaterialPage(
                            page="Malitson",
                            page_name="Malitson",
                            data_path="main/SiO2/Malitson.yml",
                            page_id=123,
                            coverage_min=0.2,
                            coverage_max=5.0,
                            has_n=True,
                            has_k=True,
                        ),
                    ),
                ),
            ),
        )
        index = _FakeLookupIndex({"sio2": search_result})
        resolver = MaterialResolver(
            index=index,
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )
        canonical_ir = {
            "sim": {"wavelength_um": [1.55]},
            "mats": [
                {"idx": 0, "name": "Air", "opt": {"kind": "nk_const", "n": 1.0, "k": 0.0}},
                {"idx": 1, "name": "SiO2", "opt": {"kind": "db_lookup", "db_key": "sio2"}},
            ],
        }
        resolver.resolve_facts_ir_materials(canonical_ir)

        self.assertEqual(["sio2"], index.search_queries)

    def test_resolve_facts_ir_materials_rejects_nk_table_without_wavelength_coverage(self) -> None:
        from evo_metaoptics.material_db.search import MaterialSearchResult
        from evo_metaoptics.meta_design.material_resolver import MaterialResolver

        resolver = MaterialResolver(
            index=_FakeIndex(MaterialSearchResult(query="", normalized_query="", matches=())),
            database_path=Path("/tmp/fake-db-root"),
            nk_factory=_FakeNkMaterial,
        )
        canonical_ir = {
            "sim": {"wavelength_um": [5.2]},
            "mats": [
                {
                    "idx": 0,
                    "name": "tabulated",
                    "opt": {
                        "kind": "nk_table",
                        "points": [
                            {"wavelength_um": 1.0, "n": 2.0, "k": 0.0},
                            {"wavelength_um": 2.0, "n": 2.1, "k": 0.0},
                        ],
                    },
                },
            ],
        }

        with self.assertRaisesRegex(ValueError, "do not cover requested wavelength grid"):
            resolver.resolve_facts_ir_materials(canonical_ir)


if __name__ == "__main__":
    unittest.main()
