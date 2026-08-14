from __future__ import annotations

import asyncio
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

batch_evaluate = importlib.import_module("evo_metaoptics.mce.eval").batch_evaluate
_BASE = importlib.import_module("evo_metaoptics.mce_env.base")
EnvironmentResult = _BASE.EnvironmentResult
EnvironmentRuntimeConfig = _BASE.EnvironmentRuntimeConfig
Sample = _BASE.Sample


class _ConcurrencyEvalEnvironment:
    def __init__(self) -> None:
        self._runtime_config = EnvironmentRuntimeConfig()
        self.active = 0
        self.max_active = 0

    def configure_runtime(self, runtime_config: EnvironmentRuntimeConfig | None) -> None:
        self._runtime_config = runtime_config or EnvironmentRuntimeConfig()

    def get_runtime_config(self) -> EnvironmentRuntimeConfig:
        return self._runtime_config

    def get_primary_metric_name(self) -> str:
        return "accuracy"

    async def aevaluate(self, sample, interfaces, llm_client=None, context_dir=None, log_dir=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return EnvironmentResult(
            feedback=f"ok-{sample.id}",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0},
            trajectory=[],
        )


class TestMCEEvalConcurrency(unittest.TestCase):
    def test_batch_eval_runs_samples_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            iter_dir.mkdir(parents=True, exist_ok=True)

            samples = [
                Sample(id=1, question="q1", ground_truth="a1"),
                Sample(id=2, question="q2", ground_truth="a2"),
                Sample(id=3, question="q3", ground_truth="a3"),
                Sample(id=4, question="q4", ground_truth="a4"),
            ]
            fake_env = _ConcurrencyEvalEnvironment()

            asyncio.run(
                batch_evaluate(
                    interfaces={},
                    samples=samples,
                    env_name="metaoptics_inverse_design",
                    iter_dir=iter_dir,
                    log_dir=iter_dir / "logs",
                    environment=fake_env,
                )
            )

            self.assertGreater(fake_env.max_active, 1)


if __name__ == "__main__":
    unittest.main()
