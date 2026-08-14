from __future__ import annotations

import unittest

from evo_metaoptics.mce.validation import ValidationResult, format_validation_feedback


class TestMCEValidationFeedbackCompactness(unittest.TestCase):
    def test_feedback_includes_stable_signatures_and_is_bounded(self) -> None:
        errors = [
            "[extract_facts_ir] File not found: interfaces/extract_facts_ir.py",
            "[extract_facts_ir] Parameter mismatch. Expected: (query), Got: (text)",
        ]
        result = ValidationResult(success=False, errors=errors, interfaces={})

        feedback = format_validation_feedback(result)

        self.assertIn("Validation signatures:", feedback)
        self.assertIn("missing_file:/interfaces/extract_facts_ir.py", feedback)
        self.assertIn("param_mismatch:/interfaces/extract_facts_ir.py", feedback)
        self.assertLessEqual(len(feedback), 1600)


if __name__ == "__main__":
    unittest.main()
