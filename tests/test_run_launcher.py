from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RunConfig = importlib.import_module("evo_metaoptics.mce.run_config").RunConfig
LaunchPlan = importlib.import_module("evo_metaoptics.mce.run_launch_plan").LaunchPlan
run_launcher = importlib.import_module("evo_metaoptics.mce.run_launcher")


def _make_config() -> RunConfig:
    return RunConfig.from_dict(
        {
            "name": "launcher-bundle-smoke",
            "experiment": {
                "env": "metaoptics_inverse_design",
                "model": "test-model",
            },
            "data": {
                "train_data": "meta_design_tasks/splits_mini/iid_train.jsonl",
                "val_data": "meta_design_tasks/splits_mini/iid_val.jsonl",
            },
            "execution": {
                "iterations": 1,
            },
            "mode": {
            },
        }
    )


def test_execute_plan_uses_bundles_directory_for_auto_bundle() -> None:
    config = _make_config()
    plan = LaunchPlan(invocations=[])

    with patch.object(run_launcher.LaunchPlanRenderer, "render", return_value=plan), patch.object(
        run_launcher, "create_run_bundle"
    ) as create_run_bundle:
        exit_code = run_launcher.execute_plan(config, dry_run=False)

    assert exit_code == 0
    create_run_bundle.assert_called_once()
    assert create_run_bundle.call_args.kwargs["output_path"] == Path("bundles") / f"{config.run_slug}.zip"


def test_execute_plan_dry_run_reports_bundle_path() -> None:
    config = _make_config()
    plan = LaunchPlan(invocations=[])
    stdout = io.StringIO()

    with patch.object(run_launcher.LaunchPlanRenderer, "render", return_value=plan), patch.object(
        run_launcher, "create_run_bundle"
    ) as create_run_bundle, redirect_stdout(stdout):
        exit_code = run_launcher.execute_plan(config, dry_run=True)

    assert exit_code == 0
    assert f"bundle path: bundles/{config.run_slug}.zip" in stdout.getvalue()
    create_run_bundle.assert_not_called()
