"""
Meta agent prompt builder.

The meta-agent evolves skills across iterations. It needs to understand
the interface signatures that the base-agent must implement.
"""

from pathlib import Path, PurePosixPath
import json
import re
from typing import List, Mapping, Optional

from evo_metaoptics.mce.prompts.iteration_review import (
    load_latest_prior_review_excerpt,
    truncate_for_prompt,
)
from evo_metaoptics.mce.skills import learning_context_skill_relative_path
from evo_metaoptics.mce_env.base import InterfaceSignature


_MAX_STRATEGY_SUMMARY_CHARS = 2400

def build_meta_agent_prompt(
    task_instruction: str,
    interface_signatures: List[InterfaceSignature],
    iter_dir: str,
    workspace_base: str,
) -> str:
    """
    Build the meta agent prompt.
    
    Args:
        task_instruction: Task-specific instruction from env
        interface_signatures: Required interface signatures
        iter_dir: iteration directory path
        workspace_base: Workspace base directory path
        
    Returns:
        Formatted prompt string
    """
    iter_name = Path(iter_dir).name
    iter_part = iter_name.split("iter")[1]
    current_iteration = int(iter_part.split("_")[0])
    
    skill_database = _build_skill_database(workspace_base, current_iteration)
    latest_review_section = _build_latest_review_advisory_section(
        workspace_base=workspace_base,
        current_iteration=current_iteration,
    )
    strategy_summary_section = _build_strategy_summary_section(workspace_base)
    interface_section = _build_interface_section(interface_signatures)
    
    skill_output_path = f"{iter_name}/{learning_context_skill_relative_path()}"
    skill_output_dir = str(PurePosixPath(skill_output_path).parent)
    
    return f"""# Meta-Level Agent: Skill Evolution for Context Engineering

## Task Overview

Your job is to evolve the next learning-context skill for the downstream task below.

### Downstream Task

{task_instruction}

{interface_section}

## Your Role

You are a **meta-level agent** that evolves context engineering skills across iterations. Your goal is to design self-contained skills that teach a base agent how to learn optimal task-specific context from training data.

Each skill you create should be a complete learning procedure that can be understood and executed independently.

## Architecture

**Meta-Level (You)**:
- Analyze iteration history (skills → implementations → results)
- Perform agentic crossover to evolve better skills
- Output: `{skill_output_path}`

**Base-Level (Context Engineer)**:
- Receives your skill + training data + prior best context
- Executes the skill to implement required interfaces
- Output: `interfaces/` functions + `context/` files

**Key Flow**: Base-agent starts with the BEST context from previous iterations and UPDATES it based on your skill's instructions.

## Runtime Skills Behavior

Historical iterations are embedded in the Skill Database section of this prompt.
This meta-agent session does not load prior iterations from separate native skill directories.
Write the next `SKILL.md` entrypoint; supporting reference files are materialized later for base-agent sessions.

## Write Helper

Tool available: Pi's built-in Write tool
- Purpose: write final markdown directly to canonical `{skill_output_path}`.
- Use this tool to create or update SKILL.md instead of only describing file contents.

## Working Directory

**Working Directory**: `/`

```
/                                          # Virtual workspace root for tool calls
  meta_agent/                                # Reference data (read-only)
    evaluations.json                         # Aggregated metrics per iteration
    skills/learning-context-iter*/SKILL.md   # Archived skills from each iteration
    iteration_reviews/iter*.md               # Prior iteration review observations
    strategy_summary.md                      # Durable lessons across iterations
  {iter_name}/                               # YOUR OUTPUT DIRECTORY
    .agents/skills/learning-context/SKILL.md # <-- WRITE YOUR SKILL HERE
    context/                                  # Static resources (base-agent writes)
      strategy_summary.md                     # Mirrored from meta_agent/strategy_summary.md
    interfaces/                               # Implemented interfaces (base-agent writes)
    data/train.json                           # Training results
```

**Write Access**: Only `{skill_output_dir}`
**IMPORTANT**: Write SKILL.md to `{skill_output_path}`
**HARD RULE**: Edit `{skill_output_path}` in-place. Do NOT create alternate files such as `SKILL-evolved.md`, `SKILL_v2.md`, or backup copies.

## Skill Database (History)

{skill_database}

{latest_review_section}

{strategy_summary_section}

## Your Task

1. **Review history**: 
   - Read `meta_agent/evaluations.json` for performance metrics.
   - Use criteria metrics as primary optimization signals: `criteria_pass_fraction_avg`, `criteria_violation_norm_avg`, `best_margin_avg`, and `val_metrics.success_goal`.
   - Treat `criteria_pass_fraction_avg` and `criteria_violation_norm_avg` as objective-aligned constraint diagnostics.
   - Read compact failure summaries first: `error_reason_counts_top3`, `error_fingerprint_counts_top5`, `validator_signature_counts_top5`, `precheck_ok_rate`, `avg_precheck_attempts`, `near_miss_rate`, `criteria_pass_fraction_avg`, `criteria_violation_norm_avg`.
   - Treat `optimizer_limited_rate` as a first-class signal for local-minimum stagnation; if elevated, evolve skills that force explicit global exploration before local refinement.
   - Do not copy raw per-sample traces into SKILL.md; use rollout traces only to validate top signatures after aggregate review.
   - Review archived skill strategy patterns under `meta_agent/skills/` for reference.
   - Analyze: What strategies worked? What failed? Which top signatures repeat across iterations?
   - **Overfitting Check**: Is train success_goal >> val success_goal?
   - **Underfitting Check**: Are both success_goal values low?
   - Use detailed rollout files (`iter*/data/train.json`) only after identifying top failure signatures.

2. **Agentic Crossover**:
   - Combine successful elements, address failures, innovate.
   - Convert advisory evidence from the latest prior review and durable lessons from `meta_agent/strategy_summary.md` into soft strategy guidance inside `SKILL.md`.
   - Do not turn those artifacts into a rigid round-by-round schedule, mandatory rotation, or fixed checklist that overrides fresh evidence.

3. **Evolve Skill**: Design a skill that guides the base-agent

## Objective Contract (Criteria-First, MCE-Aligned)

- Treat inverse-design optimization as satisfying authoritative dataset-owned criteria constraints.
- Do not change objective weights in generated specs to game metrics; criteria authority is outside the agent.
- Keep ground-truth metrics and aggregate failure signatures as the primary evidence; treat the prior review and durable summary as advisory context only.
- For evolution ranking, prioritize signals in this order:
  1. higher `val_metrics.success_goal` (binary: all criteria pass)
  2. higher `criteria_pass_fraction_avg` (partial credit across criteria)
  3. lower `criteria_violation_norm_avg` (severity of violations)
  4. higher `best_margin_avg` (worst-case distance to satisfaction)
  5. higher `val_metrics.success_exec` (execution stability)
- Keep strategy aligned with the paper's constraint-satisfaction view by improving all criteria margins through stable margin improvement, not single-metric overfitting.

## Skill Examples

### Example Skill A: Direct Agentic Curation

```markdown
## Skill Overview
Directly analyze training data and curate context in a fully agentic manner.

## Methodology
1. **Load prior context**: Read existing `context/` files
2. **Scan evaluation results**: Load `data/train.json`
3. **Analyze incorrect patterns**: Group by mistake type
4. **Update context incrementally**: keep changes targeted and evidence-based
5. **Implement interfaces**: Create functions in `interfaces/`

## Key Principles
- Build upon existing context
- Prioritize high-impact patterns
- Focus on generalizable patterns
```

### Example Skill B: LLM-Assisted Reflection

```markdown
## Skill Overview
Use structured reflection on incorrect predictions.

## Methodology
1. **Load existing context**
2. **Load training results**
3. **Reflect on errors**: Analyze each incorrect sample and extract missing constraints
4. **Curate insights incrementally**
5. **Implement interfaces** based on learned patterns
```

## Output Requirements

**Write SKILL.md to**: `{skill_output_path}`

Requirements:
- MUST include `## Skill Overview` section
- Keep guidance concise and grounded in aggregate diagnostics.
- Describe a complete learning procedure
- NO iteration-specific references
- Include clear methodology and implementation guidance
- Use Pi's built-in Write tool to create/update the canonical SKILL.md path.
- HARD RULE: only one skill file is allowed in this folder, named exactly `SKILL.md`.

**Before finishing, verify**:
- SKILL.md exists at `{skill_output_path}`
- SKILL.md has a clear `## Skill Overview` section

Begin by analyzing the skill database and evolving the next generation skill.
"""


def _build_interface_section(signatures: List[InterfaceSignature]) -> str:
    """Build interface signatures section for meta-agent."""
    if not signatures:
        return "## Required Interfaces\n\nNo specific interfaces defined."
    
    lines = [
        "## Required Interfaces",
        "",
        "The base-agent must implement these interfaces:",
        "",
    ]
    
    for sig in signatures:
        inputs_str = ", ".join(f"{name}: {typ}" for name, typ, _ in sig.inputs)
        lines.append(f"- `{sig.name}({inputs_str}) -> {sig.output[0]}`: {sig.description}")
    
    lines.append("")
    lines.append("Your skill should guide the base-agent in implementing these effectively.")
    
    return "\n".join(lines)


def _build_latest_review_advisory_section(
    workspace_base: str,
    current_iteration: int,
) -> str:
    excerpt = load_latest_prior_review_excerpt(
        workspace_base=workspace_base,
        current_iteration=current_iteration,
    )
    body = excerpt or "(none)"
    return (
        "## Advisory Review Artifact\n\n"
        "Use only the latest prior iteration review as bounded advisory context. "
        "Do not infer a broader review history beyond what is embedded here.\n\n"
        f"{body}"
    )


def _build_strategy_summary_section(workspace_base: str) -> str:
    summary_path = Path(workspace_base) / "meta_agent" / "strategy_summary.md"
    if summary_path.is_file():
        summary_text = summary_path.read_text(encoding="utf-8")
        bounded = truncate_for_prompt(
            summary_text,
            max_chars=_MAX_STRATEGY_SUMMARY_CHARS,
        ) or "(none)"
    else:
        bounded = "(none)"

    return (
        "## Durable Lessons\n\n"
        "Treat `meta_agent/strategy_summary.md` as cross-iteration memory: the durable strategy lessons carried forward from prior iterations. "
        "Extract portable heuristics from that memory, but keep the evolved `SKILL.md` flexible and evidence-led rather than a rigid schedule.\n\n"
        "Read cross-iteration memory from `meta_agent/strategy_summary.md` when present.\n\n"
        f"{bounded}"
    )


def _build_skill_database(workspace_base: str, current_iteration: int) -> str:
    """Build a summary of the skill database (history of all previous iterations)."""
    if current_iteration <= 1:
        return "No previous iterations are available yet. Design an initial skill based on the task."
    
    workspace_root = Path(workspace_base)
    meta_agent_dir = workspace_root / "meta_agent"
    
    evaluations_file = meta_agent_dir / "evaluations.json"
    if not evaluations_file.exists():
        raise FileNotFoundError(
            f"Evaluations file not found at {evaluations_file}. "
            "This file should have been created by previous iterations."
        )
    
    with open(evaluations_file) as f:
        evaluations = json.load(f)
    
    database_entries = []
    
    for i in range(1, current_iteration):
        iter_key = f"iter{i}"
        
        if iter_key not in evaluations:
            continue
        
        iter_data = evaluations[iter_key]
        
        primary_metric_name = _pick_history_metric_name(iter_data)
        train_value = _extract_history_metric_value(
            iter_data,
            metric_name=primary_metric_name,
            split="train",
        )
        val_value = _extract_history_metric_value(
            iter_data,
            metric_name=primary_metric_name,
            split="val",
        )
        train_str = _format_metric_percent(train_value)
        val_str = _format_metric_percent(val_value)
        
        metrics_display = f"**Train**: {train_str} | **Val**: {val_str}"
        criteria_pass_fraction_avg = _coerce_optional_float(
            iter_data.get("criteria_pass_fraction_avg")
        )
        criteria_violation_norm_avg = _coerce_optional_float(
            iter_data.get("criteria_violation_norm_avg")
        )
        objective_display = (
            "**Objective Signals (criteria)**: "
            f"criteria_pass_fraction_avg={_format_percent_or_na(criteria_pass_fraction_avg)}, "
            f"criteria_violation_norm_avg={_format_float_or_na(criteria_violation_norm_avg)}, "
            f"best_margin_avg={_format_float_or_na(_coerce_optional_float(iter_data.get('best_margin_avg')))}, "
            f"val_success_goal={_format_percent_or_na(_extract_history_metric_value(iter_data, metric_name='success_goal', split='val'))}"
        )
        
        skill_file = (
            meta_agent_dir / "skills" / f"learning-context-iter{i}" / "SKILL.md"
        )
        skill_overview = _extract_skill_overview(skill_file)
        
        num_sub_iters = iter_data.get('num_sub_iters', 1)
        total_rollouts = iter_data.get('total_rollouts', 0)
        last_sub_folder = iter_data.get('last_sub_folder', f'iter{i}')
        failure_summary = _format_iteration_failure_summary(iter_data)

        entry = f"""### iter{i}
- {objective_display}
- {metrics_display}
- **Rollouts**: {total_rollouts} ({num_sub_iters} sub-iteration{"s" if num_sub_iters > 1 else ""})
{failure_summary}
- **Skill Overview**:
{skill_overview}
- **Files**: `meta_agent/skills/learning-context-iter{i}/SKILL.md`, `{last_sub_folder}/`"""
        database_entries.append(entry)
    
    if not database_entries:
        return "No previous iterations available."
    
    return "\n\n".join(database_entries)


def _pick_history_metric_name(iter_data: dict) -> str:
    """Pick a stable primary metric name from an iteration history record."""
    explicit = iter_data.get("primary_metric_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    train_metrics = iter_data.get("train_metrics") or {}
    val_metrics = iter_data.get("val_metrics") or {}

    # Prefer canonical metric names when present in either top-level or metric dicts.
    for preferred in ("success_goal", "accuracy"):
        if _extract_history_metric_value(
            iter_data, metric_name=preferred, split="train"
        ) is not None and _extract_history_metric_value(
            iter_data, metric_name=preferred, split="val"
        ) is not None:
            return preferred

    # Then prefer shared metric dict keys.
    common_metric_keys = sorted(set(train_metrics) & set(val_metrics))
    if common_metric_keys:
        return common_metric_keys[0]

    # Deterministic fallback that keeps prompt generation resilient.
    for key in sorted(val_metrics):
        return key
    for key in sorted(train_metrics):
        return key
    return "accuracy"


def _extract_history_metric_value(
    iter_data: dict,
    metric_name: str,
    split: str,
) -> Optional[float]:
    metrics = iter_data.get(f"{split}_metrics") or {}
    metric_value = metrics.get(metric_name)
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return None


def _format_metric_percent(value: Optional[float]) -> str:
    """Format metric values for prompt display."""
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def _format_percent_or_na(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def _format_float_or_na(value: Optional[float], *, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _format_bool_or_na(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "N/A"


def _coerce_optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_topk_counts(
    raw_items: object,
    *,
    key_name: str,
) -> str:
    if not isinstance(raw_items, list):
        return "  - (none)"

    lines: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        key = item.get(key_name)
        count = item.get("count")
        if isinstance(key, str) and key.strip() and isinstance(count, (int, float)):
            rate = item.get("rate")
            if isinstance(rate, (int, float)):
                lines.append(
                    f"  - {key.strip()}: {int(count)} "
                    f"(rate={_format_percent_or_na(float(rate))})"
                )
            else:
                lines.append(f"  - {key.strip()}: {int(count)}")
    if not lines:
        return "  - (none)"
    return "\n".join(lines)


def _format_iteration_failure_summary(iter_data: dict) -> str:
    reason_lines = _format_topk_counts(
        iter_data.get("error_reason_counts_top3"),
        key_name="error_reason",
    )
    fingerprint_lines = _format_topk_counts(
        iter_data.get("error_fingerprint_counts_top5"),
        key_name="error_fingerprint",
    )
    signature_lines = _format_topk_counts(
        iter_data.get("validator_signature_counts_top5"),
        key_name="signature",
    )

    precheck_ok_rate = _coerce_optional_float(iter_data.get("precheck_ok_rate"))
    avg_precheck_attempts = _coerce_optional_float(iter_data.get("avg_precheck_attempts"))
    near_miss_rate = _coerce_optional_float(iter_data.get("near_miss_rate"))
    criteria_pass_fraction_avg = _coerce_optional_float(
        iter_data.get("criteria_pass_fraction_avg")
    )
    criteria_violation_norm_avg = _coerce_optional_float(
        iter_data.get("criteria_violation_norm_avg")
    )
    best_margin_avg = _coerce_optional_float(iter_data.get("best_margin_avg"))
    optimizer_limited_rate = _coerce_optional_float(iter_data.get("optimizer_limited_rate"))
    spec_limited_rate = _coerce_optional_float(
        iter_data.get("spec_limited_rate")
    )
    skill_guard_triggered = iter_data.get("skill_guard_triggered")
    degeneracy_type_raw = iter_data.get("degeneracy_type")
    degeneracy_type = (
        degeneracy_type_raw.strip()
        if isinstance(degeneracy_type_raw, str) and degeneracy_type_raw.strip()
        else "N/A"
    )
    return (
        "- **Failure Summary**:\n"
        "  - Top Error Reasons:\n"
        f"{reason_lines}\n"
        "  - Top Error Fingerprints:\n"
        f"{fingerprint_lines}\n"
        "  - Top Validator Signatures:\n"
        f"{signature_lines}\n"
        "  - Diagnostics: "
        f"precheck_ok_rate={_format_percent_or_na(precheck_ok_rate)}, "
        f"avg_precheck_attempts={_format_float_or_na(avg_precheck_attempts, digits=2)}, "
        f"near_miss_rate={_format_percent_or_na(near_miss_rate)}\n"
        "  - Criteria: "
        f"criteria_pass_fraction_avg={_format_percent_or_na(criteria_pass_fraction_avg)}, "
        f"criteria_violation_norm_avg={_format_float_or_na(criteria_violation_norm_avg)}\n"
        "  - Optimization Classification: "
        f"optimizer_limited_rate={_format_percent_or_na(optimizer_limited_rate)}, "
        f"spec_limited_rate={_format_percent_or_na(spec_limited_rate)}\n"
        "  - Runtime Signals: "
        f"best_margin_avg={_format_float_or_na(best_margin_avg)}\n"
        "  - Skill Guard: "
        f"skill_guard_triggered={_format_bool_or_na(skill_guard_triggered)}, "
        f"degeneracy_type={degeneracy_type}"
    )


def _extract_skill_overview(skill_path: Path) -> str:
    """Extract the '## Skill Overview' section from SKILL.md."""
    if not skill_path.exists():
        return "  (SKILL.md not found)"
    
    try:
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"  (error reading file: {e})"
    
    pattern = r'^##\s*Skill\s+Overview\s*$'
    match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
    
    if not match:
        return "  (no '## Skill Overview' section found)"
    
    start_pos = match.end()
    next_match = re.search(r'\n##\s+[^#]', content[start_pos:])
    
    if next_match:
        overview_content = content[start_pos:start_pos + next_match.start()].strip()
    else:
        overview_content = content[start_pos:].strip()
    
    if not overview_content:
        return "  (Skill Overview section is empty)"
    
    return "\n".join(f"  {line}" if line.strip() else "" for line in overview_content.split("\n"))


if __name__ == "__main__":
    # Example usage
    from evo_metaoptics.mce_env.base import InterfaceSignature
    
    sigs = [
        InterfaceSignature(
            name="get_context",
            inputs=[("question", "str", "The question")],
            output=("str", "Context string"),
            description="Return relevant context."
        )
    ]
    
    print(build_meta_agent_prompt(
        task_instruction="Example task",
        interface_signatures=sigs,
        iter_dir="/workspace/iter2",
        workspace_base="/workspace"
    ))
