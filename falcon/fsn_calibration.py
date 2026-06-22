"""Calibration utilities for FSN value and accepted-probability heads."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Sequence


def collect_calibration_rows(
    payloads: Sequence[tuple[str, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source, payload in payloads:
        for record in payload.get("candidate_records") or []:
            generator = str(record.get("generator_type") or "").lower()
            if "fsn" not in generator:
                continue
            candidate = record.get("candidate") or {}
            metadata = candidate.get("metadata") or {}
            difficulty = record.get("difficulty_result") or {}
            predicted_value = _number(metadata.get("predicted_value_score"))
            accepted_probability = _number(
                metadata.get("predicted_accepted_probability")
            )
            final_value = _number(difficulty.get("final_value_score"))
            if (
                predicted_value is None
                or accepted_probability is None
                or final_value is None
            ):
                continue
            rows.append(
                {
                    "source": source,
                    "failure_id": record.get("failure_id"),
                    "seed": record.get("seed"),
                    "round_id": record.get("round_id"),
                    "generator_type": record.get("generator_type"),
                    "scenario_id": candidate.get("scenario_id"),
                    "predicted_value_score": predicted_value,
                    "accepted_probability": accepted_probability,
                    "final_value_score": final_value,
                    "accepted": bool(
                        difficulty.get("accepted_into_curriculum_pool")
                    ),
                }
            )
    return rows


def calibrate(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> Dict[str, Any]:
    labels = [1 if row.get("accepted") else 0 for row in rows]
    value_scores = [float(row["predicted_value_score"]) for row in rows]
    probability_scores = [float(row["accepted_probability"]) for row in rows]
    final_values = [float(row["final_value_score"]) for row in rows]
    probability_threshold = _best_threshold(probability_scores, labels)
    value_threshold = _best_threshold(value_scores, labels)
    return {
        "schema_version": "falcon.fsn_acceptance_calibration.v1",
        "num_samples": len(rows),
        "accepted_count": sum(labels),
        "rejected_count": len(rows) - sum(labels),
        "accepted_rate": _rate(sum(labels), len(rows)),
        "source_counts": dict(
            sorted(Counter(str(row.get("source")) for row in rows).items())
        ),
        "predicted_value_score_stats": _distribution(value_scores),
        "accepted_probability_stats": _distribution(probability_scores),
        "final_value_score_stats": _distribution(final_values),
        "predicted_value_vs_final_value_pearson": _pearson(
            value_scores, final_values
        ),
        "predicted_value_accepted_auc": _auc(value_scores, labels),
        "accepted_probability_auc": _auc(probability_scores, labels),
        "best_predicted_value_threshold": value_threshold,
        "best_accepted_probability_threshold": probability_threshold,
        "predicted_value_metrics_at_best_threshold": _classification_metrics(
            value_scores, labels, value_threshold
        ),
        "accepted_probability_metrics_at_best_threshold": (
            _classification_metrics(
                probability_scores, labels, probability_threshold
            )
        ),
        "accepted_probability_brier_score": _brier(
            probability_scores, labels
        ),
        "accepted_probability_expected_calibration_error": (
            _expected_calibration_error(probability_scores, labels, bins)
        ),
        "accepted_probability_calibration_curve": _calibration_curve(
            probability_scores, labels, bins
        ),
        "predicted_value_calibration_curve": _calibration_curve(
            value_scores, labels, bins
        ),
        "warnings": _warnings(value_scores, probability_scores),
    }


def calibration_curve_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for score_name, key in (
        ("accepted_probability", "accepted_probability_calibration_curve"),
        ("predicted_value_score", "predicted_value_calibration_curve"),
    ):
        for item in summary.get(key) or []:
            rows.append({"score_name": score_name, **dict(item)})
    return rows


def _best_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.5
    thresholds = sorted(set(scores))
    candidates = [thresholds[0] - 1e-9, *thresholds, thresholds[-1] + 1e-9]
    best = max(
        candidates,
        key=lambda threshold: (
            _classification_metrics(scores, labels, threshold)["f1"],
            _classification_metrics(scores, labels, threshold)["precision"],
            _classification_metrics(scores, labels, threshold)["recall"],
            threshold,
        ),
    )
    return round(float(best), 8)


def _classification_metrics(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> Dict[str, Any]:
    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for prediction, label in zip(predictions, labels) if prediction and label)
    fp = sum(1 for prediction, label in zip(predictions, labels) if prediction and not label)
    fn = sum(1 for prediction, label in zip(predictions, labels) if not prediction and label)
    tn = sum(1 for prediction, label in zip(predictions, labels) if not prediction and not label)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = _rate(2 * precision * recall, precision + recall)
    accuracy = _rate(tp + tn, len(labels))
    return {
        "threshold": round(float(threshold), 8),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _calibration_curve(
    scores: Sequence[float], labels: Sequence[int], bins: int
) -> List[Dict[str, Any]]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    span = max(high - low, 1e-12)
    groups: List[List[tuple[float, int]]] = [[] for _index in range(bins)]
    for score, label in zip(scores, labels):
        index = min(int((score - low) / span * bins), bins - 1)
        groups[index].append((score, label))
    rows = []
    for index, group in enumerate(groups):
        if not group:
            continue
        rows.append(
            {
                "bin_index": index,
                "bin_lower": round(low + span * index / bins, 8),
                "bin_upper": round(low + span * (index + 1) / bins, 8),
                "count": len(group),
                "mean_predicted_score": round(
                    statistics.fmean(score for score, _label in group), 8
                ),
                "empirical_accepted_rate": _rate(
                    sum(label for _score, label in group), len(group)
                ),
            }
        )
    return rows


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 6)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    return round(numerator / denominator, 6) if denominator > 1e-12 else None


def _brier(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    return round(
        statistics.fmean(
            (min(max(score, 0.0), 1.0) - label) ** 2
            for score, label in zip(scores, labels)
        ),
        8,
    )


def _expected_calibration_error(
    scores: Sequence[float], labels: Sequence[int], bins: int
) -> float | None:
    curve = _calibration_curve(scores, labels, bins)
    if not scores:
        return None
    return round(
        sum(
            item["count"]
            / len(scores)
            * abs(
                item["mean_predicted_score"]
                - item["empirical_accepted_rate"]
            )
            for item in curve
        ),
        8,
    )


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 8),
        "std": round(statistics.pstdev(values), 8),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
        "unique_count": len(set(values)),
    }


def _warnings(
    value_scores: Sequence[float], probability_scores: Sequence[float]
) -> List[str]:
    warnings = []
    if len(set(value_scores)) < max(len(value_scores) // 10, 2):
        warnings.append("Predicted value scores have low uniqueness.")
    if len(set(probability_scores)) < max(len(probability_scores) // 10, 2):
        warnings.append("Accepted probabilities have low uniqueness.")
    if probability_scores and max(probability_scores) < 0.1:
        warnings.append(
            "Accepted probabilities are strongly under-confident and require calibration."
        )
    return warnings


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        denominator = float(denominator)
        return round(float(numerator) / denominator, 6) if denominator > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0
