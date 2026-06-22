"""Double-boundary difficulty evaluator for FALCON curriculum candidates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .trajectory_recorder import SCENARIO_VECTOR_KEYS

DIFFICULTY_SCHEMA_VERSION = "falcon.difficulty_evaluation.v1"

DEFAULT_CONFIG: Dict[str, Any] = {
    "eps": 1e-8,
    "tau_easy": 0.75,
    "tau_solve": 0.40,
    "tau_diversity": 0.20,
    "default_diversity_without_pool": 0.50,
    "priority_thresholds": {
        "high": 0.70,
        "medium": 0.40,
    },
    "weights": {
        "current_policy_weakness": 0.25,
        "historical_solvability": 0.20,
        "learning_potential": 0.25,
        "scenario_diversity": 0.15,
        "failure_mode_match": 0.15,
    },
    "scenario_vector_scales": {
        "team_center_distance": 10000.0,
        "own_formation_spread": 3000.0,
        "opponent_formation_spread": 3000.0,
        "altitude_difference": 2000.0,
        "velocity_difference": 150.0,
        "heading_difference": math.pi,
        "approximate_aspect_angle": math.pi,
        "own_center_x": 10000.0,
        "own_center_y": 10000.0,
        "own_center_z": 8000.0,
        "opponent_center_x": 10000.0,
        "opponent_center_y": 10000.0,
        "opponent_center_z": 8000.0,
    },
    "diversity_distance_normalization": 1.0,
}


class DifficultyEvaluator:
    """Evaluate candidate scenarios against FALCON's double-boundary filter."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))

    def evaluate_candidate(
        self,
        candidate: Mapping[str, Any],
        current_policy_eval: Mapping[str, Any],
        best_policy_eval: Mapping[str, Any],
        pool_stats: Optional[Mapping[str, Any]],
        failure_summary: Optional[Mapping[str, Any]],
        constraint_result: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        rejection_reasons: List[str] = []
        scenario_id = str(candidate.get("scenario_id", "unknown"))
        scenario_vector = candidate.get("scenario_vector") if isinstance(candidate.get("scenario_vector"), MappingABC) else {}
        current_win_rate = _clamp01(current_policy_eval.get("win_rate"))
        best_win_rate = _clamp01(best_policy_eval.get("win_rate"))
        tau_easy = _float(self.config["tau_easy"])
        tau_solve = _float(self.config["tau_solve"])
        tau_diversity = _float(self.config["tau_diversity"])

        current_policy_weakness = _clamp01(1.0 - current_win_rate)
        historical_solvability = _clamp01((best_win_rate - tau_solve) / max(1.0 - tau_solve, self.config["eps"]))
        learning_potential = _clamp01(best_win_rate - current_win_rate)
        scenario_diversity, diversity_warnings = self._scenario_diversity(scenario_vector, pool_stats)
        warnings.extend(diversity_warnings)
        failure_mode_match = self._failure_mode_match(candidate, failure_summary)
        constraint_validity = 1.0 if bool((constraint_result or {}).get("is_valid", False)) else 0.0

        if constraint_validity < 1.0:
            rejection_reasons.extend((constraint_result or {}).get("rejection_reasons") or ["constraint_invalid"])
        if current_win_rate > tau_easy:
            rejection_reasons.append("too_easy_for_current_policy")
        if best_win_rate < tau_solve:
            rejection_reasons.append("not_solvable_by_historical_best_policy")
        if scenario_diversity < tau_diversity:
            rejection_reasons.append("insufficient_scenario_diversity")

        components = {
            "current_policy_weakness": current_policy_weakness,
            "historical_solvability": historical_solvability,
            "learning_potential": learning_potential,
            "scenario_diversity": scenario_diversity,
            "failure_mode_match": failure_mode_match,
        }
        final_value_score = _weighted_sum(components, self.config["weights"])
        hard_filter_passed = len(rejection_reasons) == 0
        accepted = hard_filter_passed
        return {
            "schema_version": DIFFICULTY_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "current_policy_weakness": _round01(current_policy_weakness),
            "historical_solvability": _round01(historical_solvability),
            "learning_potential": _round01(learning_potential),
            "scenario_diversity": _round01(scenario_diversity),
            "failure_mode_match": _round01(failure_mode_match),
            "constraint_validity": _round01(constraint_validity),
            "hard_filter_passed": hard_filter_passed,
            "rejection_reasons": sorted(set(rejection_reasons)),
            "final_value_score": _round01(final_value_score),
            "accepted_into_curriculum_pool": accepted,
            "sampling_weight": 0.0,
            "priority_level": self._priority_level(final_value_score),
            "warnings": sorted(set(warnings)),
            "metadata": {
                "current_policy_eval": dict(current_policy_eval or {}),
                "best_policy_eval": dict(best_policy_eval or {}),
                "constraint_result": dict(constraint_result or {}),
                "target_failure_modes": list(candidate.get("target_failure_modes") or []),
            },
        }

    def evaluate_batch(
        self,
        candidates: Sequence[Mapping[str, Any]],
        current_policy_evals: Union[Sequence[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]],
        best_policy_evals: Union[Sequence[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]],
        pool_stats: Optional[Mapping[str, Any]],
        failure_summary: Optional[Mapping[str, Any]],
        constraint_results: Union[Sequence[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        evaluated = []
        for idx, candidate in enumerate(candidates):
            scenario_id = str(candidate.get("scenario_id", idx))
            evaluated.append(
                self.evaluate_candidate(
                    candidate,
                    _select_eval(current_policy_evals, scenario_id, idx),
                    _select_eval(best_policy_evals, scenario_id, idx),
                    pool_stats,
                    failure_summary,
                    _select_eval(constraint_results, scenario_id, idx),
                )
            )
        return self.compute_sampling_weights(evaluated)

    def compute_sampling_weights(self, evaluated_scenarios: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        accepted_scores = [
            _float(item.get("final_value_score"))
            for item in evaluated_scenarios
            if item.get("accepted_into_curriculum_pool")
        ]
        total = sum(accepted_scores)
        output = []
        for item in evaluated_scenarios:
            item = dict(item)
            if item.get("accepted_into_curriculum_pool") and total > self.config["eps"]:
                item["sampling_weight"] = _round01(_float(item.get("final_value_score")) / total)
            else:
                item["sampling_weight"] = 0.0
            output.append(item)
        return output

    def save_batch(self, evaluated_scenarios: Sequence[Mapping[str, Any]], output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "falcon.difficulty_evaluation_batch.v1",
                    "evaluated_scenarios": list(evaluated_scenarios),
                },
                f,
                indent=2,
                sort_keys=True,
            )

    def _scenario_diversity(
        self,
        scenario_vector: Mapping[str, Any],
        pool_stats: Optional[Mapping[str, Any]],
    ) -> Tuple[float, List[str]]:
        warnings: List[str] = []
        vectors = _pool_vectors(pool_stats)
        if not vectors:
            default = _clamp01(self.config["default_diversity_without_pool"])
            warnings.append(f"scenario_diversity defaulted to {default} because pool_stats has no scenario vectors.")
            return default, warnings
        distances = [
            _normalized_vector_distance(
                scenario_vector,
                vector,
                self.config["scenario_vector_scales"],
            )
            for vector in vectors
        ]
        distances = [distance for distance in distances if distance is not None]
        if not distances:
            default = _clamp01(self.config["default_diversity_without_pool"])
            warnings.append(f"scenario_diversity defaulted to {default} because no overlapping scenario_vector fields were available.")
            return default, warnings
        normalizer = max(_float(self.config["diversity_distance_normalization"]), self.config["eps"])
        return _clamp01(min(distances) / normalizer), warnings

    def _failure_mode_match(
        self,
        candidate: Mapping[str, Any],
        failure_summary: Optional[Mapping[str, Any]],
    ) -> float:
        target_modes = set(candidate.get("target_failure_modes") or [])
        if not target_modes or not isinstance(failure_summary, MappingABC):
            return 0.0
        primary = set(failure_summary.get("primary_failure_modes") or [])
        secondary = set(failure_summary.get("secondary_failure_modes") or [])
        if target_modes & primary:
            return 1.0
        if target_modes & secondary:
            return 0.6
        return 0.0

    def _priority_level(self, score: float) -> str:
        if score >= _float(self.config["priority_thresholds"]["high"]):
            return "high"
        if score >= _float(self.config["priority_thresholds"]["medium"]):
            return "medium"
        return "low"


def _select_eval(source: Union[Sequence[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]], scenario_id: str, idx: int) -> Mapping[str, Any]:
    if isinstance(source, MappingABC):
        value = source.get(scenario_id) or source.get(str(idx)) or {}
        return value if isinstance(value, MappingABC) else {}
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)) and idx < len(source):
        value = source[idx]
        return value if isinstance(value, MappingABC) else {}
    return {}


def _pool_vectors(pool_stats: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not isinstance(pool_stats, MappingABC):
        return []
    for key in ("scenario_vectors", "vector_pool", "training_scenario_vectors"):
        vectors = pool_stats.get(key)
        if isinstance(vectors, list):
            return [vector for vector in vectors if isinstance(vector, MappingABC)]
    return []


def _normalized_vector_distance(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    scales: Mapping[str, float],
) -> Optional[float]:
    parts = []
    for key in SCENARIO_VECTOR_KEYS:
        if a.get(key) is None or b.get(key) is None:
            continue
        scale = max(_float(scales.get(key, 1.0)), 1e-8)
        parts.append(((_float(a.get(key)) - _float(b.get(key))) / scale) ** 2)
    if not parts:
        return None
    return math.sqrt(sum(parts) / len(parts))


def _weighted_sum(values: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return _clamp01(sum(_float(values.get(key)) * _float(weight) for key, weight in weights.items()))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _round01(value: Any) -> float:
    return round(_clamp01(value), 6)


def _clamp01(value: Any) -> float:
    value = _float(value)
    return max(0.0, min(1.0, value))


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value
