"""Transparent, assumption-driven normal-approximation power calculation.

This is a design calculation, not an estimate from model observations. It does
not account for clustering, missingness, family heterogeneity, or non-normal
outcomes; separate power analyses are required for those estimands.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any


def normal_approx_paired_power(
    n: int,
    *,
    standardized_paired_effect: float = 0.5,
    family_alpha: float = 0.05,
    number_of_primary_tests: int = 6,
) -> float:
    if n < 2:
        raise ValueError("n must be at least 2.")
    if standardized_paired_effect <= 0:
        raise ValueError("standardized_paired_effect must be positive.")
    if not 0 < family_alpha < 1:
        raise ValueError("family_alpha must be in (0, 1).")
    if number_of_primary_tests < 1:
        raise ValueError("number_of_primary_tests must be positive.")
    normal = NormalDist()
    per_test_alpha = family_alpha / number_of_primary_tests
    critical = normal.inv_cdf(1 - per_test_alpha / 2)
    noncentrality = standardized_paired_effect * math.sqrt(n)
    return (
        1 - normal.cdf(critical - noncentrality) + normal.cdf(-critical - noncentrality)
    )


def minimum_n_for_power(
    target_power: float = 0.8,
    *,
    standardized_paired_effect: float = 0.5,
    family_alpha: float = 0.05,
    number_of_primary_tests: int = 6,
    maximum_n: int = 100_000,
) -> int:
    if not 0 < target_power < 1:
        raise ValueError("target_power must be in (0, 1).")
    for n in range(2, maximum_n + 1):
        if (
            normal_approx_paired_power(
                n,
                standardized_paired_effect=standardized_paired_effect,
                family_alpha=family_alpha,
                number_of_primary_tests=number_of_primary_tests,
            )
            >= target_power
        ):
            return n
    raise ValueError("Required sample size exceeded maximum_n.")


def power_plan(
    n: int = 60,
    *,
    effect: float = 0.5,
    tests: int = 6,
    family_alpha: float = 0.05,
    target_power: float = 0.8,
) -> dict[str, Any]:
    per_test_alpha = family_alpha / tests
    return {
        "calculation_type": "analytical normal approximation for paired standardized mean difference",
        "is_empirical_result": False,
        "paired_n": n,
        "assumed_standardized_paired_effect": effect,
        "family_alpha": family_alpha,
        "primary_test_count": tests,
        "conservative_per_test_alpha": per_test_alpha,
        "two_sided": True,
        "approximate_power_at_n": normal_approx_paired_power(
            n,
            standardized_paired_effect=effect,
            family_alpha=family_alpha,
            number_of_primary_tests=tests,
        ),
        "minimum_n_for_target_power_under_same_assumptions": minimum_n_for_power(
            target_power,
            standardized_paired_effect=effect,
            family_alpha=family_alpha,
            number_of_primary_tests=tests,
        ),
        "target_power": target_power,
        "limitations": [
            "Normal approximation; not a noncentral-t calculation.",
            "Assumes independent paired units and a known standardized effect.",
            "Does not account for clustering by prompt, model family, or participant.",
            "Does not establish power for binary, classification, survival, or correlation tests.",
            "No study observations are consumed by this calculation.",
        ],
    }
