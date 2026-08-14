# evo-metaoptics

Public code for **A Self-Evolving Agentic Framework for Metasurface Inverse
Design**.

## Associated publication

Yi Huang, Bowen Zheng, Yunxi Dong, Hong Tang, Huan Zhao, S. M. Rakibul Hasan
Shawon, and Hualiang Zhang, “A Self-Evolving Agentic Framework for Metasurface
Inverse Design,” *Laser & Photonics Reviews*, e71739 (2026).
[Publisher page](https://onlinelibrary.wiley.com/doi/10.1002/lpor.71739) ·
[DOI: 10.1002/lpor.71739](https://doi.org/10.1002/lpor.71739)

This repository contains the core MCE runtime, the metasurface inverse-design
environment, deterministic `gt_eval` scoring, one IID dataset, and one example
run configuration. The instructions below show how to install the project and
run that included example. Plotting, publication-report generation, and the
paper's complete experiment suite are intentionally outside the public scope.

## Included example

The bundled configuration uses the complete public IID split:

```text
meta_design_tasks/splits_50_15_50/iid_train.jsonl   50 tasks
meta_design_tasks/splits_50_15_50/iid_val.jsonl     15 tasks
meta_design_tasks/splits_50_15_50/iid_test.jsonl    50 tasks
```

During a run, the coding agent writes a `solve_inverse_design()` Python
function, TorchRDIT executes the generated design code, and the deterministic
evaluator checks the solver results against each task's criteria. The
meta-agent uses rollout feedback to update the explicit skill/context
artifacts between iterations.

The included configuration runs all three splits for five iterations and can
make many external model calls. It requires meaningful compute time and may
incur model-provider charges; it is an example experiment, not an instant smoke
test.

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- [TorchRDIT](https://github.com/yi-huang-1/torchrdit)
- [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
  installed as `pi`
- A model provider and credentials configured for Pi

For TorchRDIT installation and configuration, follow the instructions in the
[TorchRDIT repository](https://github.com/yi-huang-1/torchrdit).

## 1. Install

```bash
git clone https://github.com/yi-huang-1/evo-metaoptics.git
cd evo-metaoptics
uv sync
```

Confirm that Pi and the Python package are available:

```bash
pi --version
uv run python -m compileall -q src/evo_metaoptics
```

## 2. Configure the model

Follow Pi's upstream instructions to configure your provider and credentials.
The launcher also reads provider environment variables from a local `.env`
file when one is present; do not commit that file.

The included YAML currently specifies `claude-sonnet-4-6`. If that model is not
available through your Pi setup, edit `experiment.model` in:

```text
configs/runs/data_collect-50-15-50-iid-c46.yaml
```

## 3. Preview the command

Dry-run mode validates the configuration and prints the rendered launch command
without executing the experiment:

```bash
MCE_LAUNCHER_DRY_RUN=1 \
  ./scripts/run_mce_from_config.sh \
  configs/runs/data_collect-50-15-50-iid-c46.yaml
```

## 4. Run the included example

Run this command from the repository root:

```bash
./scripts/run_mce_from_config.sh \
  configs/runs/data_collect-50-15-50-iid-c46.yaml
```

For a long run, use a persistent terminal session such as `tmux` so the process
continues if the SSH connection closes.

## Outputs

The launcher creates local run artifacts under:

```text
workspace/data_collect-50-15-50-iid-c46_*/
logs/run_*/
bundles/data_collect-50-15-50-iid-c46_*.zip
```

The workspace contains generated `solution.py` programs, learned skill/context
artifacts, and evaluation results. Logs contain launcher and agent execution
records. The ZIP bundle collects the run artifacts for later inspection. These
runtime directories are ignored by Git.

## Dataset format

Each JSONL row contains:

- `query` or `question`: the natural-language design task
- `gt_eval`: deterministic evaluation criteria
- optional `execution_plan`: stage-only execution guidance
- `metadata`: dataset provenance and template information

Only the IID train, validation, and test files are included in this public
repository.

## Core repository map

```text
src/evo_metaoptics/mce/                         MCE training and skill evolution
src/evo_metaoptics/mce_env/metaoptics_inverse_design/  inverse-design environment
src/evo_metaoptics/meta_design/                 deterministic evaluation
src/evo_metaoptics/material_db/                 material lookup helpers
examples/metaoptics_inverse_design/             starter skill and lookup demo
configs/runs/data_collect-50-15-50-iid-c46.yaml included example experiment
meta_design_tasks/splits_50_15_50/              public IID data
```

## Acknowledgements

The MCE learning architecture in this repository builds on the ideas and
implementation of [Meta Context Engineering via Agentic Skill
Evolution](https://github.com/metaevo-ai/meta-context-engineering). We thank its
authors for making their work available under the MIT License.

Metasurface simulation and differentiable optimization use
[TorchRDIT](https://github.com/yi-huang-1/torchrdit). See that repository for
installation and configuration instructions.

## Citation

If you use this code, please cite the associated article. GitHub's **Cite this
repository** menu also provides citation metadata from `CITATION.cff`.

```bibtex
@article{huang2026selfevolving,
  author  = {Huang, Yi and Zheng, Bowen and Dong, Yunxi and Tang, Hong and
             Zhao, Huan and Shawon, S. M. Rakibul Hasan and Zhang, Hualiang},
  title   = {A Self-Evolving Agentic Framework for Metasurface Inverse Design},
  journal = {Laser \& Photonics Reviews},
  year    = {2026},
  pages   = {e71739},
  doi     = {10.1002/lpor.71739},
  url     = {https://doi.org/10.1002/lpor.71739}
}
```

This public repository intentionally excludes internal manuscript/reviewer
materials, dataset-generation tooling, plotting code, publication-report code,
local workspaces, and private run traces.
