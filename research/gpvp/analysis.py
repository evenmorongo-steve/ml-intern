"""Small analysis primitives; no analysis is run unless raw ratings/data are supplied.

Clustering requires scikit-learn and an independently generated, version-pinned
embedding file. This module never converts missing measurements into zeros.
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Any


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed by registered test name."""
    if not p_values:
        return {}
    for name, p_value in p_values.items():
        if not math.isfinite(p_value) or not 0 <= p_value <= 1:
            raise ValueError(f"Invalid p-value for {name!r}: {p_value!r}")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * p_value)
        adjusted[name] = min(1.0, running_max)
    return adjusted


def paired_bootstrap_mean_difference(
    treatment: dict[str, float],
    control: dict[str, float],
    *,
    seed: int = 20261009,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired mean difference and percentile bootstrap CI over shared pair IDs."""
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be in (0, 1).")
    pair_ids = sorted(set(treatment) & set(control))
    if len(pair_ids) < 2:
        raise ValueError("At least two complete matched pairs are required.")
    differences = [
        float(treatment[pair_id]) - float(control[pair_id]) for pair_id in pair_ids
    ]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("Paired differences must be finite numbers.")
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    lower = _quantile(estimates, (1 - confidence) / 2)
    upper = _quantile(estimates, 1 - (1 - confidence) / 2)
    return {
        "matched_n": len(differences),
        "pair_ids": pair_ids,
        "mean_difference": observed,
        "confidence": confidence,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a quantile of an empty sequence.")
    index = (len(sorted_values) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(sorted_values[low])
    fraction = index - low
    return float(sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction)


def krippendorff_alpha_nominal(
    ratings: dict[str, Sequence[str | int | None]],
) -> float | None:
    """Nominal Krippendorff alpha for units -> ratings, with missing values omitted."""
    usable: dict[str, list[str | int]] = {}
    values: list[str | int] = []
    for unit, raw_values in ratings.items():
        unit_values = [value for value in raw_values if value is not None]
        if unit_values:
            usable[unit] = unit_values
            values.extend(unit_values)
    if len(values) < 2:
        return None
    category_counts = Counter(values)
    total = len(values)
    observed_disagreement = 0.0
    coincidence_pairs = 0
    for unit_values in usable.values():
        n = len(unit_values)
        if n < 2:
            continue
        unit_counts = Counter(unit_values)
        for first, first_count in unit_counts.items():
            for second, second_count in unit_counts.items():
                if first != second:
                    observed_disagreement += first_count * second_count
        coincidence_pairs += n * (n - 1)
    if coincidence_pairs == 0 or total < 2:
        return None
    observed_disagreement /= coincidence_pairs
    expected_disagreement = 1 - sum(
        count * (count - 1) for count in category_counts.values()
    ) / (total * (total - 1))
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1 - observed_disagreement / expected_disagreement


def response_logit_summaries(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize only measurements present in the supplied token trace."""
    trace = record.get("token_trace")
    if not isinstance(trace, list) or not trace:
        return {
            "record_id": record.get("record_id"),
            "n_tokens_measured": 0,
            "mean_entropy_nats": None,
            "mean_logit_margin": None,
            "mean_selected_rank": None,
            "mean_top1_probability": None,
        }
    entropies = [
        float(row["entropy_nats"])
        for row in trace
        if isinstance(row, dict) and row.get("entropy_nats") is not None
    ]
    margins = [
        float(row["logit_margin"])
        for row in trace
        if isinstance(row, dict) and row.get("logit_margin") is not None
    ]
    ranks = [
        float(row["selected_rank_in_returned_top_k"])
        for row in trace
        if isinstance(row, dict)
        and row.get("selected_rank_in_returned_top_k") is not None
    ]
    top1_probabilities: list[float] = []
    for row in trace:
        if not isinstance(row, dict):
            continue
        raw_top = row.get("base_top_k")
        if (
            isinstance(raw_top, list)
            and raw_top
            and raw_top[0].get("logprob_temperature_1") is not None
        ):
            top1_probabilities.append(
                math.exp(float(raw_top[0]["logprob_temperature_1"]))
            )
    return {
        "record_id": record.get("record_id"),
        "n_tokens_measured": len(trace),
        "mean_entropy_nats": sum(entropies) / len(entropies) if entropies else None,
        "mean_logit_margin": sum(margins) / len(margins) if margins else None,
        "mean_selected_rank": sum(ranks) / len(ranks) if ranks else None,
        "mean_top1_probability": sum(top1_probabilities) / len(top1_probabilities)
        if top1_probabilities
        else None,
    }


def _word_ngram_set(text: str, n: int = 5) -> set[tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"\w+", normalized)
    return {
        tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))
    }


def _jaccard(first: set[Any], second: set[Any]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def detect_memoryless_lockin(
    outputs: Sequence[str], *, threshold: float = 0.80, ngram_size: int = 5
) -> dict[str, Any]:
    """Apply the preregistered two-consecutive-transition H6 onset rule."""
    if not 0 < threshold <= 1 or ngram_size < 1:
        raise ValueError("threshold must be in (0, 1] and ngram_size must be positive.")
    ngrams = [_word_ngram_set(text, n=ngram_size) for text in outputs]
    similarities: list[dict[str, Any]] = []
    for index in range(1, len(ngrams)):
        similarities.append(
            {
                "step": index + 1,
                "jaccard_to_previous": _jaccard(ngrams[index], ngrams[index - 1]),
            }
        )
    onset = None
    for index in range(1, len(similarities)):
        previous_transition = similarities[index - 1]
        current_transition = similarities[index]
        if (
            previous_transition["step"] >= 4
            and previous_transition["jaccard_to_previous"] >= threshold
            and current_transition["jaccard_to_previous"] >= threshold
        ):
            onset = previous_transition["step"]
            break
    return {
        "output_count": len(outputs),
        "ngram_size": ngram_size,
        "threshold": threshold,
        "similarities": similarities,
        "onset_step": onset,
        "right_censored_at_step": len(outputs) if onset is None else None,
        "onset_found": onset is not None,
    }


def _cosine_cluster_report(
    vectors: Sequence[Sequence[float]],
    condition_labels: Sequence[str],
    *,
    k: int = 3,
    seed: int = 20261009,
    permutations: int = 1000,
    bootstraps: int = 100,
    stability_fraction: float = 0.8,
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError as exc:
        raise RuntimeError(
            "Clustering requires numpy and scikit-learn; install requirements-analysis.txt."
        ) from exc

    matrix = np.asarray(vectors, dtype=float)
    labels = np.asarray(condition_labels, dtype=object)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError(
            "Embeddings must be a 2D matrix with one condition label per row."
        )
    if matrix.shape[0] < max(5, k + 1) or k < 2 or k >= matrix.shape[0]:
        raise ValueError("Need at least max(5, k+1) rows and 2 <= k < n_rows.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains non-finite values.")
    if len(set(labels.tolist())) < 2:
        raise ValueError(
            "At least two condition labels are required for the shuffled-label comparison."
        )

    def make_estimators(random_state: int):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        try:
            agglomerative = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            )
        except TypeError:  # scikit-learn before `metric` replaced `affinity`
            agglomerative = AgglomerativeClustering(
                n_clusters=k, affinity="cosine", linkage="average"
            )
        return (("kmeans", kmeans), ("agglomerative_average_cosine", agglomerative))

    rng = random.Random(seed)
    report: dict[str, Any] = {
        "n": int(matrix.shape[0]),
        "embedding_dimension": int(matrix.shape[1]),
        "k": k,
        "random_seed": seed,
        "permutations": permutations,
        "bootstraps": bootstraps,
        "methods": {},
        "interpretation_gate": "no cluster interpretation unless all registered validity gates pass",
    }
    for method_name, estimator in make_estimators(seed):
        cluster_labels = estimator.fit_predict(matrix)
        silhouette = float(silhouette_score(matrix, cluster_labels, metric="cosine"))
        observed_ari = float(adjusted_rand_score(labels, cluster_labels))
        permutation_values: list[float] = []
        for _ in range(permutations):
            shuffled = labels.tolist()
            rng.shuffle(shuffled)
            permutation_values.append(
                float(adjusted_rand_score(shuffled, cluster_labels))
            )
        shuffle_p = (1 + sum(value >= observed_ari for value in permutation_values)) / (
            permutations + 1
        )

        stability_values: list[float] = []
        silhouette_values: list[float] = []
        sample_size = max(k + 1, round(matrix.shape[0] * stability_fraction))
        sample_size = min(sample_size, matrix.shape[0] - 1)
        for bootstrap_index in range(bootstraps):
            indices = sorted(rng.sample(range(matrix.shape[0]), sample_size))
            _, bootstrap_estimator = make_estimators(seed + bootstrap_index)[
                0 if method_name == "kmeans" else 1
            ]
            bootstrap_cluster_labels = bootstrap_estimator.fit_predict(matrix[indices])
            stability_values.append(
                float(
                    adjusted_rand_score(
                        cluster_labels[indices], bootstrap_cluster_labels
                    )
                )
            )
            if len(set(bootstrap_cluster_labels)) > 1:
                silhouette_values.append(
                    float(
                        silhouette_score(
                            matrix[indices], bootstrap_cluster_labels, metric="cosine"
                        )
                    )
                )
        stability_values.sort()
        silhouette_values.sort()
        median_stability = (
            _quantile(stability_values, 0.5) if stability_values else None
        )
        stability_lower = (
            _quantile(stability_values, 0.025) if stability_values else None
        )
        stability_upper = (
            _quantile(stability_values, 0.975) if stability_values else None
        )
        silhouette_interval = (
            [_quantile(silhouette_values, 0.025), _quantile(silhouette_values, 0.975)]
            if silhouette_values
            else None
        )
        report["methods"][method_name] = {
            "silhouette_cosine": silhouette,
            "silhouette_bootstrap_95_ci": silhouette_interval,
            "bootstrap_stability_median_ari": median_stability,
            "bootstrap_stability_95_ci": [stability_lower, stability_upper],
            "condition_cluster_ari": observed_ari,
            "shuffled_label_p_value": shuffle_p,
            "validity_gates_pass": bool(
                silhouette >= 0.15
                and median_stability is not None
                and median_stability >= 0.60
                and shuffle_p <= 0.05
            ),
        }
    adjusted = holm_adjust(
        {
            name: float(value["shuffled_label_p_value"])
            for name, value in report["methods"].items()
        }
    )
    for name, value in report["methods"].items():
        value["shuffled_label_p_holm"] = adjusted[name]
        value["validity_gates_pass"] = bool(
            value["validity_gates_pass"] and adjusted[name] <= 0.05
        )
    return report


def analyze_embedding_manifest(value: dict[str, Any]) -> dict[str, Any]:
    """Run the registered two-method cluster check on a frozen embedding manifest."""
    if not value.get("embedding_model_id") or not value.get("embedding_model_version"):
        raise ValueError(
            "Embedding manifest must include an independent model ID and immutable version."
        )
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Embedding manifest must contain a non-empty rows array.")
    record_ids = [row.get("record_id") for row in rows]
    if any(not record_id for record_id in record_ids) or len(set(record_ids)) != len(
        record_ids
    ):
        raise ValueError("Embedding rows must have unique record_id values.")
    return {
        "analysis_type": "two-method unsupervised cluster validation",
        "embedding_model_id": value["embedding_model_id"],
        "embedding_model_version": value["embedding_model_version"],
        "corpus_or_model_family_split": value.get("split_definition"),
        **_cosine_cluster_report(
            [row["embedding"] for row in rows],
            [str(row["condition_label"]) for row in rows],
            k=int(value.get("k", 3)),
            seed=int(value.get("seed", 20261009)),
            permutations=int(value.get("permutations", 1000)),
            bootstraps=int(value.get("bootstraps", 100)),
        ),
    }
