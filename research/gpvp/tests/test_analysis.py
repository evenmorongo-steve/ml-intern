from __future__ import annotations

import unittest

from research.gpvp.analysis import (
    detect_memoryless_lockin,
    holm_adjust,
    krippendorff_alpha_nominal,
    paired_bootstrap_mean_difference,
    response_logit_summaries,
)


class AnalysisTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_bounded(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted.values()))

    def test_paired_bootstrap_uses_only_complete_pairs(self) -> None:
        result = paired_bootstrap_mean_difference(
            {"p1": 2.0, "p2": 3.0, "unmatched-treatment": 20.0},
            {"p1": 1.0, "p2": 1.0, "unmatched-control": -20.0},
            seed=11,
            resamples=200,
        )
        self.assertEqual(result["matched_n"], 2)
        self.assertAlmostEqual(result["mean_difference"], 1.5)
        self.assertEqual(result["pair_ids"], ["p1", "p2"])

    def test_nominal_krippendorff_alpha(self) -> None:
        self.assertEqual(krippendorff_alpha_nominal({"a": [1, 1], "b": [2, 2]}), 1.0)
        self.assertAlmostEqual(
            krippendorff_alpha_nominal({"a": [1, 1], "b": [0, 1]}), 0.0
        )
        self.assertIsNone(krippendorff_alpha_nominal({"a": [1], "b": [None]}))

    def test_lockin_rule_detects_two_consecutive_high_overlap_transitions(self) -> None:
        repeated = "A language model selects tokens based on the current context and learned probabilities."
        result = detect_memoryless_lockin(
            [
                "Initial distinct sentence with enough words here.",
                repeated,
                repeated,
                repeated,
                repeated,
            ]
        )
        self.assertTrue(result["onset_found"])
        self.assertEqual(result["onset_step"], 4)
        self.assertIsNone(result["right_censored_at_step"])

    def test_no_lockin_is_right_censored(self) -> None:
        outputs = [
            "A red fox runs quickly through empty fields.",
            "Computers calculate finite values using carefully chosen arithmetic methods.",
            "Ocean tides respond to gravitational force from the Moon.",
            "Ancient manuscripts preserve ideas across many different cultures.",
            "Bicycles turn through streets when riders adjust their handlebars.",
        ]
        result = detect_memoryless_lockin(outputs)
        self.assertFalse(result["onset_found"])
        self.assertEqual(result["right_censored_at_step"], 5)

    def test_missing_token_trace_remains_null(self) -> None:
        result = response_logit_summaries({"record_id": "empty", "token_trace": []})
        self.assertEqual(result["n_tokens_measured"], 0)
        self.assertIsNone(result["mean_entropy_nats"])
        self.assertIsNone(result["mean_top1_probability"])


if __name__ == "__main__":
    unittest.main()
