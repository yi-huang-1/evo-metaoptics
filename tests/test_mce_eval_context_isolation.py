from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evo_metaoptics.mce.eval import batch_evaluate
from evo_metaoptics.mce_env.base import EnvironmentResult, Sample


class _MutatingEvalEnvironment:
    def __init__(self) -> None:
        self.received_context_dirs: list[Path] = []

    def get_primary_metric_name(self) -> str:
        return "accuracy"

    async def aevaluate(self, sample, interfaces, llm_client=None, context_dir=None, log_dir=None):
        assert context_dir is not None
        context_dir = Path(context_dir)
        self.received_context_dirs.append(context_dir)
        (context_dir / f"mutation_{sample.id}.txt").write_text("mutated", encoding="utf-8")
        return EnvironmentResult(
            feedback="ok",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0},
            trajectory=[],
        )


class TestMCEEvalContextIsolation(unittest.TestCase):
    def test_batch_eval_must_not_mutate_canonical_context_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            context_dir = iter_dir / "context"
            context_dir.mkdir(parents=True, exist_ok=True)
            (context_dir / "seed.txt").write_text("seed", encoding="utf-8")

            samples = [
                Sample(id=1, question="q1", ground_truth="a1"),
                Sample(id=2, question="q2", ground_truth="a2"),
            ]
            fake_env = _MutatingEvalEnvironment()

            with patch("evo_metaoptics.mce.eval.EnvironmentRegistry.get", return_value=fake_env):
                asyncio.run(
                    batch_evaluate(
                        interfaces={},
                        samples=samples,
                        env_name="metaoptics_inverse_design",
                        iter_dir=iter_dir,
                        log_dir=iter_dir / "logs",
                    )
                )

            mutated = sorted(p.name for p in context_dir.glob("mutation_*.txt"))
            self.assertEqual(
                [],
                mutated,
                "Evaluation should use isolated/read-only context and leave canonical context untouched.",
            )

            self.assertTrue(
                all(path != context_dir for path in fake_env.received_context_dirs),
                "Environment should not receive canonical context directory directly.",
            )


if __name__ == "__main__":
    unittest.main()
