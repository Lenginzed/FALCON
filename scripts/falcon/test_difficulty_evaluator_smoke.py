#!/usr/bin/env python
"""Smoke test for the FALCON double-boundary difficulty evaluator."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402


BASE_VECTOR = {
    "team_center_distance": 10000.0,
    "own_formation_spread": 1000.0,
    "opponent_formation_spread": 1000.0,
    "altitude_difference": 0.0,
    "velocity_difference": 0.0,
    "heading_difference": 3.14159,
    "approximate_aspect_angle": 0.5,
    "own_center_x": 0.0,
    "own_center_y": 0.0,
    "own_center_z": 6000.0,
    "opponent_center_x": 10000.0,
    "opponent_center_y": 0.0,
    "opponent_center_z": 6000.0
}


def shifted(distance, altitude):
    vector = dict(BASE_VECTOR)
    vector["team_center_distance"] = distance
    vector["opponent_center_x"] = distance
    vector["altitude_difference"] = altitude
    vector["opponent_center_z"] = 6000.0 + altitude
    return vector


def main():
    evaluator = DifficultyEvaluator()
    candidates = [
        {"scenario_id": "too_easy", "scenario_vector": shifted(11000.0, 100.0), "target_failure_modes": ["coordination_failure"], "scenario_parameters": {}},
        {"scenario_id": "too_hard", "scenario_vector": shifted(18000.0, 2000.0), "target_failure_modes": ["coordination_failure"], "scenario_parameters": {}},
        {"scenario_id": "accepted", "scenario_vector": shifted(22000.0, 3500.0), "target_failure_modes": ["coordination_failure"], "scenario_parameters": {}}
    ]
    current_policy_evals = {
        "too_easy": {"win_rate": 0.90, "mean_return": 300.0, "num_eval_episodes": 8},
        "too_hard": {"win_rate": 0.10, "mean_return": -300.0, "num_eval_episodes": 8},
        "accepted": {"win_rate": 0.25, "mean_return": -150.0, "num_eval_episodes": 8}
    }
    best_policy_evals = {
        "too_easy": {"win_rate": 0.95, "mean_return": 500.0, "num_eval_episodes": 8},
        "too_hard": {"win_rate": 0.20, "mean_return": -200.0, "num_eval_episodes": 8},
        "accepted": {"win_rate": 0.80, "mean_return": 350.0, "num_eval_episodes": 8}
    }
    pool_stats = {"scenario_vectors": [BASE_VECTOR]}
    failure_summary = {
        "primary_failure_modes": ["coordination_failure"],
        "secondary_failure_modes": [],
        "failure_scores": {"coordination_failure": 0.8}
    }
    constraint_results = {
        "too_easy": {"is_valid": True, "validity_score": 1.0, "rejection_reasons": []},
        "too_hard": {"is_valid": True, "validity_score": 1.0, "rejection_reasons": []},
        "accepted": {"is_valid": True, "validity_score": 1.0, "rejection_reasons": []}
    }
    evaluated = evaluator.evaluate_batch(
        candidates,
        current_policy_evals,
        best_policy_evals,
        pool_stats,
        failure_summary,
        constraint_results,
    )
    by_id = {item["scenario_id"]: item for item in evaluated}
    if by_id["too_easy"]["accepted_into_curriculum_pool"]:
        raise AssertionError("too_easy should be rejected")
    if by_id["too_hard"]["accepted_into_curriculum_pool"]:
        raise AssertionError("too_hard should be rejected")
    if not by_id["accepted"]["accepted_into_curriculum_pool"]:
        raise AssertionError("accepted scenario should pass")
    if by_id["accepted"]["sampling_weight"] <= 0:
        raise AssertionError("accepted scenario should have positive sampling weight")
    if by_id["too_easy"]["sampling_weight"] != 0 or by_id["too_hard"]["sampling_weight"] != 0:
        raise AssertionError("rejected scenarios must have zero sampling weight")

    output = {"schema_version": "falcon.difficulty_evaluator_smoke.v1", "evaluated_scenarios": evaluated}
    out_path = ROOT_DIR / "tests" / "tmp_falcon_trajectories" / "difficulty_evaluation_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
