from __future__ import annotations

import unittest

from research.gpvp.power import (
    minimum_n_for_power,
    normal_approx_paired_power,
    power_plan,
)


class PowerPlanTests(unittest.TestCase):
    def test_registered_design_assumptions_are_transparent(self) -> None:
        result = power_plan(n=60, effect=0.5, tests=6)
        self.assertFalse(result["is_empirical_result"])
        self.assertEqual(result["paired_n"], 60)
        self.assertAlmostEqual(result["conservative_per_test_alpha"], 0.05 / 6)
        self.assertAlmostEqual(
            result["approximate_power_at_n"], normal_approx_paired_power(60)
        )
        self.assertGreaterEqual(result["approximate_power_at_n"], 0.8)

    def test_minimum_n_reaches_target_under_the_same_assumptions(self) -> None:
        n = minimum_n_for_power(
            0.8, standardized_paired_effect=0.5, number_of_primary_tests=6
        )
        self.assertGreaterEqual(normal_approx_paired_power(n), 0.8)
        self.assertLess(normal_approx_paired_power(n - 1), 0.8)


if __name__ == "__main__":
    unittest.main()
