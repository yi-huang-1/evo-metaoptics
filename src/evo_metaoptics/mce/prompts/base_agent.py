"""
Base agent prompt builder.

Injects interface signatures into the prompt for the agent to implement.
"""

from pathlib import Path
from typing import List

from evo_metaoptics.mce_env.base import InterfaceSignature


def build_base_agent_prompt(
    task_instruction: str,
    interface_signatures: List[InterfaceSignature],
    iter_dir: str,
    workspace_base: str | None = None,
    initial_prompt: str | None = None,
    required_context_files: List[str] | None = None,
    min_context_chars: int = 30,
) -> str:
    """
    Build the base agent prompt with interface signatures.
    
    Args:
        task_instruction: Task-specific instruction from env
        interface_signatures: List of interfaces to implement
        iter_dir: Iteration directory path
        workspace_base: Base workspace directory
        initial_prompt: Optional additional prompt
        
    Returns:
        Formatted prompt string
    """
    iter_name = Path(iter_dir).name
    virtual_iter_dir = f"/{iter_name}"
    
    # Build interface section
    interface_section = _build_interface_section(interface_signatures)
    required_context_files = required_context_files or []
    required_context_section = ""
    if required_context_files:
        required_lines = "\n".join(
            f"- `context/{rel_path}` (at least {min_context_chars} characters)"
            for rel_path in required_context_files
        )
        required_context_section = f"""
## Required Context Artifacts

You MUST produce these canonical context files before finishing:
{required_lines}

If you write context files, the path value must match one of the required paths exactly.
"""
        if any(Path(rel_path).name == "rules.txt" for rel_path in required_context_files):
            required_context_section += """

Context Guidance Style:
- `context/rules.txt` is loaded verbatim into extractor context guidance.
- Keep rules concise, actionable, and canonical for current meta_design contracts.
- Prefer plain deterministic statements that are directly executable by the extractor.
"""

    deterministic_write_helper = """Tool available: Pi's built-in Write tool
- Writes files within the iteration workspace (`/{iter_name}/`).
- Use this tool to create context artifacts in `context/` directory instead of only describing intended writes.
- Example: write to `context/rules.txt` or `context/analysis.md`."""
    
    # Build directory structure based on whether interfaces are required
    if interface_signatures:
        dir_structure = f'''```
{iter_name}/
  .agents/skills/learning-context/SKILL.md  # Optional skill artifact (may be loaded natively)
  context/                                   # Write static resources here (knowledge, rules, etc.)
  interfaces/                                # Implement required interfaces here
    __init__.py                              # Export all functions
    <function_name>.py                       # One file per interface
  data/
    train.json                               # Training results to learn from
```'''
    else:
        dir_structure = f'''```
{iter_name}/
  .agents/skills/learning-context/SKILL.md  # Optional skill artifact (may be loaded natively)
  context/                                   # Write static resources here (knowledge, rules, etc.)
  data/
    train.json                               # Training results to learn from
```

**NOTE**: This task has NO interface requirements. Focus on curating high-quality context files in `context/`.'''
    
    prompt = f'''# Context Engineer

## Task Overview

Your job is to learn from training results and improve the iteration-local context and interfaces for the downstream task below.

### Downstream Task

{task_instruction}

{interface_section}

## Working Directory

**Working Directory**: `{virtual_iter_dir}`

Your directory contains:
{dir_structure}

**File Access**:
- Read/Write: Only files within `{virtual_iter_dir}/`
- You CANNOT access other directories

## Skill Guidance

Pi loads native skill packages from explicit `--skill <path>` directories.
If `.agents/skills/learning-context/` is present, read `SKILL.md` first and use `.agents/skills/learning-context/reference/` for supporting examples and API details.
If no skill package is present, continue with direct training-data analysis and context curation.

## Deterministic Write Helper

{deterministic_write_helper}

{required_context_section}

## Core Objective: Learn from Training Data

**CRITICAL**: Analyze `data/train.json` and curate context to fix incorrect predictions.

**IMPORTANT**: In most cases, high-quality context files are MORE impactful than complex interface logic.

### Schema Consistency Contract

- Treat local gradient optimization as insufficient for global optima; encode when to use explicit global exploration (multistart/population screening) before local refinement.
- After each batch, summarize durable learning rules into `context/rules.txt` using concrete failure signatures and outcome metrics.

- **Context files** (`context/`) provide knowledge, patterns, and guidelines that directly improve LLM reasoning'''
    
    if interface_signatures:
        prompt += '''
- **Interface functions** (`interfaces/`) can often be simple and naive while still achieving good results'''
    
    prompt += '''

### Training Data Analysis

1. **Load and inspect** `data/train.json`:
    - `summary`: Overall metrics
    - `detailed_results`: List of rollouts

2. **Analyze predictions**:
   - **Incorrect**: Why did it fail? What knowledge/pattern was missing?
   - **Correct**: What patterns led to success? How to reinforce?

3. '''
    
    if interface_signatures:
        prompt += '''**Curate context and implement interfaces**:
   - Write context files in `context/` directory
   - Implement required functions in `interfaces/` directory
   - **IMPORTANT**: Use iteration-root-relative paths for context reads in interface code:
     ```python
     # Good: iteration-relative path
     context_path = "context/knowledge.md"
     
     # Also acceptable: virtual absolute path within current iteration root
     context_path = "/context/knowledge.md"

     # Bad: host absolute path (outside virtual root)
     context_path = "/Users/name/workspace/.../context/knowledge.md"
     ```'''
    else:
        prompt += '''**Curate context**:
   - Write context files in `context/` directory'''
    
    prompt += '''

## Environment

Use `uv run python ...` for all Python execution.
'''
    
    if interface_signatures:
        prompt += '''
## Validation

The system will automatically validate your interface implementations.
If validation fails, you'll receive specific error messages to fix.
Just keep working until all interfaces are valid.
'''
    
    prompt += '''
Work efficiently: focus on impactful changes, avoid over-analysis, finish promptly.
'''
    
    if initial_prompt:
        prompt += f"\n\n## Additional Instructions\n\n{initial_prompt}"
    
    return prompt


def _build_interface_section(signatures: List[InterfaceSignature]) -> str:
    """Build the interface requirements section of the prompt."""
    if not signatures:
        return "## Required Interfaces\n\nNo specific interfaces required."
    
    lines = [
        "## Required Interfaces",
        "",
        "You MUST implement these interfaces in the `interfaces/` directory:",
        "",
    ]
    
    for sig in signatures:
        lines.append(sig.to_prompt())
        lines.append("")

    lines.extend([
        "### Hard Contract",
        "",
        "- Function name must match exactly.",
        "- Parameter names must match exactly and in the declared order.",
        "- Do not add/remove required parameters.",
        "- Do not use `*args` or `**kwargs` wrappers for required interfaces.",
        "",
    ])

    if any(sig.name == "extract_facts_ir" for sig in signatures):
        lines.extend([
            "### `extract_facts_ir` Output Contract",
            "",
            "- Output must be JSON string only (no markdown fences, no prose).",
            "- For canonical optimize/solve payloads, include required top-level keys:",
            "  - `stack`",
            "  - `design`",
            "  - `execution_plan`",
            "- Optional objective shaping is only through `execution_plan.compiler_knobs`:",
            "  - `penalty_type`",
            "  - `criterion_weights`",
            "",
        ])

    lines.extend([
        "### Output Structure",
        "",
        "```",
        "interfaces/",
        "  __init__.py          # Export all functions",
    ])
    
    for sig in signatures:
        lines.append(f"  {sig.name}.py         # Implementation of {sig.name}")
    
    lines.extend([
        "```",
        "",
        "**Example `interfaces/__init__.py`**:",
        "",
        "```python",
    ])
    
    for sig in signatures:
        lines.append(f"from .{sig.name} import {sig.name}")
    
    lines.append("")
    lines.append(f"__all__ = [{', '.join(repr(s.name) for s in signatures)}]")
    lines.append("```")
    
    return "\n".join(lines)
