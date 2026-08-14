from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from evo_metaoptics.mce.meta_agent import _verify_meta_agent_outputs
from evo_metaoptics.mce.skills import ensure_learning_context_skill_seed

_VALID_DELTA_SECTION = """## Skill Delta (This Iteration)
### ADD
- Add one deterministic schema reset rule for repeated top-level mismatch signatures.

### UPDATE
- Update policy guidance to require full bounds coverage before multistart execution.

### REMOVE
- Remove stale anti-pattern entries that did not recur for two iterations.
"""


class TestMCESkillEvolutionGate(unittest.TestCase):
    def test_scaffold_only_skill_is_accepted_under_loose_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            ensure_learning_context_skill_seed(iter_dir, overwrite=True)

            result = _verify_meta_agent_outputs(
                iter_dir=iter_dir,
                logger=logging.getLogger("test_skill_evolution_gate"),
            )

            self.assertTrue(
                result["success"],
                "Scaffold-only SKILL.md should be accepted under loose policy.",
            )

    def test_trivial_edit_skill_is_accepted_under_loose_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_file, _ = ensure_learning_context_skill_seed(iter_dir, overwrite=True)
            skill_file.write_text(
                """---
name: learning-context
description: Scaffold for iterative context-learning skill evolution.
---

## Skill Overview
Core strategy for learning.

## Methodology
1. Review outcomes.
2. Update context.
3. Improve accuracy.
""",
                encoding="utf-8",
            )

            result = _verify_meta_agent_outputs(
                iter_dir=iter_dir,
                logger=logging.getLogger("test_skill_evolution_gate"),
            )

            self.assertTrue(
                result["success"],
                "Trivial SKILL.md edits should be accepted under loose policy.",
            )

    def test_substantive_skill_edit_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_file, _ = ensure_learning_context_skill_seed(iter_dir, overwrite=True)
            skill_file.write_text(
                """---
name: learning-context
description: Iteratively improve context quality from rollout evidence.
---

## Skill Overview
Use rollout-level error patterns to update context in a targeted way.
Focus on reducing repeated confusion clusters and preserving high-performing rules.

## Methodology
1. Extract failure clusters from the latest train and validation traces.
2. Group errors by symptom overlap, contradictory cues, and missing negative evidence.
3. Write explicit disambiguation rules and attach concrete symptom boundaries.
4. Add short counterexamples for commonly confused disease pairs.
5. Keep updates incremental and retain proven rules from previous iterations.

## Update Rules
- Replace vague rules with deterministic, testable criteria.
- Prioritize guidance that changes model decisions on recurring error groups.
- Remove stale or conflicting rules only when replaced with stronger evidence-backed alternatives.

## Skill Delta (This Iteration)
### ADD
- Add one deterministic schema reset rule for repeated top-level mismatch signatures.

### UPDATE
- Update policy guidance to require full bounds coverage before multistart execution.

### REMOVE
- Remove stale anti-pattern entries that did not recur for two iterations.
""",
                encoding="utf-8",
            )

            result = _verify_meta_agent_outputs(
                iter_dir=iter_dir,
                logger=logging.getLogger("test_skill_evolution_gate"),
            )

            self.assertTrue(
                result["success"],
                "Substantive SKILL.md edits should pass the meta-agent gate.",
            )

    def test_skill_with_many_anti_pattern_items_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_file, _ = ensure_learning_context_skill_seed(iter_dir, overwrite=True)
            anti_items = "\n".join(f"- anti-pattern {idx}" for idx in range(1, 13))
            skill_file.write_text(
                f"""---
name: learning-context
description: Iteratively improve context quality from rollout evidence.
---

## Skill Overview
Use rollout-level error patterns to update context in a targeted way.
Focus on reducing repeated confusion clusters and preserving high-performing rules.

## Methodology
1. Extract failure clusters from latest traces.
2. Update context and interfaces incrementally.
3. Preserve high-value rules and prune stale weak rules.

## Anti-Pattern Catalog
{anti_items}

{_VALID_DELTA_SECTION}
""",
                encoding="utf-8",
            )

            result = _verify_meta_agent_outputs(
                iter_dir=iter_dir,
                logger=logging.getLogger("test_skill_evolution_gate"),
            )

            self.assertTrue(result["success"])

    def test_skill_with_many_example_items_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_file, _ = ensure_learning_context_skill_seed(iter_dir, overwrite=True)
            example_items = "\n".join(f"- Example {idx}: mapping case" for idx in range(1, 7))
            skill_file.write_text(
                f"""---
name: learning-context
description: Iteratively improve context quality from rollout evidence.
---

## Skill Overview
Use rollout-level error patterns to update context in a targeted way.
Focus on reducing repeated confusion clusters and preserving high-performing rules.

## Methodology
1. Extract failure clusters from latest traces.
2. Update context and interfaces incrementally.
3. Preserve high-value rules and prune stale weak rules.

## Minimal Examples
{example_items}

{_VALID_DELTA_SECTION}
""",
                encoding="utf-8",
            )

            result = _verify_meta_agent_outputs(
                iter_dir=iter_dir,
                logger=logging.getLogger("test_skill_evolution_gate"),
            )

            self.assertTrue(result["success"])

    def test_skill_missing_delta_section_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_file, _ = ensure_learning_context_skill_seed(iter_dir, overwrite=True)
            skill_file.write_text(
                """---
name: learning-context
description: Iteratively improve context quality from rollout evidence.
---

## Skill Overview
Use rollout-level error patterns to update context in a targeted way.
Focus on reducing repeated confusion clusters and preserving high-performing rules.

## Methodology
1. Extract failure clusters from the latest train and validation traces.
2. Group errors by symptom overlap and missing boundary cues.
3. Add deterministic disambiguation rules and remove stale contradictory rules.
4. Keep updates incremental and retain proven guidance.
""",
                encoding="utf-8",
            )

            result = _verify_meta_agent_outputs(
                iter_dir=iter_dir,
                logger=logging.getLogger("test_skill_evolution_gate"),
            )

            self.assertTrue(result["success"])

    def test_skill_with_many_delta_items_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "iter1_sub0"
            skill_file, _ = ensure_learning_context_skill_seed(iter_dir, overwrite=True)
            add_items = "\n".join(f"- Add delta rule {idx}" for idx in range(1, 7))
            skill_file.write_text(
                f"""---
name: learning-context
description: Iteratively improve context quality from rollout evidence.
---

## Skill Overview
Use rollout-level error patterns to update context in a targeted way.
Focus on reducing repeated confusion clusters and preserving high-performing rules.

## Methodology
1. Extract failure clusters from latest traces.
2. Update context and interfaces incrementally.
3. Preserve high-value rules and prune stale weak rules.

## Skill Delta (This Iteration)
### ADD
{add_items}

### UPDATE
- Update one ambiguous rule with explicit bounds semantics.

### REMOVE
- Remove one stale rule that no longer appears in top signatures.
""",
                encoding="utf-8",
            )

            result = _verify_meta_agent_outputs(
                iter_dir=iter_dir,
                logger=logging.getLogger("test_skill_evolution_gate"),
            )

            self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
