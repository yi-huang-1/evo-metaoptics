import json
import tempfile
import unittest
from pathlib import Path

from evo_metaoptics.mce_env.metaoptics_inverse_design import (
    MetaopticsInverseDesignEnvironment,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")


class TestMetaopticsEnvLoadSamples(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MetaopticsInverseDesignEnvironment()

    def test_load_samples_normalizes_query_and_stores_gt_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "train.jsonl"
            _write_jsonl(
                data_path,
                [
                    {
                        "query": "Design a metasurface",
                        "gt_eval": {
                            "wavelength_um": [1.55],
                            "criteria": [
                                {
                                    "expr": "r.transmission[0].item()",
                                    "operation": ">=",
                                    "target": 0.8,
                                }
                            ],
                        },
                    }
                ],
            )

            samples = self.env.load_samples(str(data_path), limit=10)
            self.assertEqual(1, len(samples))
            self.assertEqual("Design a metasurface", samples[0].question)
            self.assertIn("gt_eval", samples[0].extras)
            gt_eval = samples[0].extras["gt_eval"]
            self.assertEqual([1.55], gt_eval["wavelength_um"])

    def test_load_samples_requires_gt_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "train.jsonl"
            _write_jsonl(
                data_path,
                [{"query": "Missing gt_eval payload"}],
            )

            with self.assertRaisesRegex(ValueError, "gt_eval"):
                self.env.load_samples(str(data_path), limit=10)

    def test_load_samples_accepts_question_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "train.jsonl"
            _write_jsonl(
                data_path,
                [
                    {
                        "question": "A question-keyed sample",
                        "gt_eval": {
                            "wavelength_um": [0.632],
                            "criteria": [
                                {
                                    "expr": "r.reflection[0].item()",
                                    "operation": ">=",
                                    "target": 0.5,
                                }
                            ],
                        },
                    }
                ],
            )

            samples = self.env.load_samples(str(data_path), limit=10)
            self.assertEqual(1, len(samples))
            self.assertEqual("A question-keyed sample", samples[0].question)

    def test_load_samples_respects_limit(self) -> None:
        rows = [
            {
                "query": f"Sample {i}",
                "gt_eval": {
                    "wavelength_um": [1.55],
                    "criteria": [{"expr": "r.transmission[0].item()", "operation": ">=", "target": 0.5}],
                },
            }
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "train.jsonl"
            _write_jsonl(data_path, rows)

            samples = self.env.load_samples(str(data_path), limit=2)
            self.assertEqual(2, len(samples))

if __name__ == "__main__":
    unittest.main()
