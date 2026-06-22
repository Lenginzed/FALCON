#!/usr/bin/env python
"""End-to-end smoke test for FALCON scenario pipeline interfaces."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.random_scenario_generator import RandomScenarioGenerator  # noqa: E402
from falcon.scenario_adapter import (  # noqa: E402
    apply_initial_config_to_yaml,
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    load_base_scenario_config,
    save_scenario_yaml,
    trajectory_to_replay_scenario,
)


def main():
    base_config_path = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
    output_dir = ROOT_DIR / "tests" / "tmp_falcon_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_base_scenario_config(base_config_path)
    initial_config = extract_initial_config_from_yaml(base_config)
    vector_result = initial_config_to_scenario_vector(initial_config)

    failure_summary = {
        "schema_version": "falcon.failure_analysis.v1",
        "source_trajectory": "scenario_pipeline_smoke",
        "primary_failure_modes": ["coordination_failure"],
        "secondary_failure_modes": ["target_assignment_confusion"],
        "failure_scores": {"coordination_failure": 0.8, "target_assignment_confusion": 0.5},
    }
    generator = RandomScenarioGenerator({"seed": 11})
    candidates = generator.generate_from_failure_summary(failure_summary, base_config, 3)
    candidate_validations = [validate_candidate_schema(candidate) for candidate in candidates]

    checker = ConstraintChecker()
    constraint_results = checker.validate_batch(candidates)

    generated_yaml_paths = []
    yaml_validation_results = {}
    for candidate, constraint in zip(candidates, constraint_results):
        if not constraint["is_valid"]:
            continue
        yaml_config = apply_initial_config_to_yaml(base_config, candidate["initial_config"])
        yaml_path = output_dir / f"{candidate['scenario_id']}.yaml"
        save_scenario_yaml(yaml_config, yaml_path)
        generated_yaml_paths.append(str(yaml_path))
        yaml_validation_results[candidate["scenario_id"]] = checker.validate_yaml_config(yaml_config)

    replay_source = ROOT_DIR / "tests" / "fixtures" / "falcon_coordination_failure_v2.json"
    replay_yaml_path = output_dir / "replay_from_trajectory_smoke.yaml"
    replay_result = None
    if replay_source.exists():
        replay_result = trajectory_to_replay_scenario(replay_source, replay_yaml_path, base_config_path=base_config_path)

    current_policy_eval = {
        candidate["scenario_id"]: {"win_rate": 0.25 + idx * 0.15, "mean_return": -120.0 + idx * 30.0, "num_eval_episodes": 4}
        for idx, candidate in enumerate(candidates)
    }
    best_policy_eval = {
        candidate["scenario_id"]: {"win_rate": 0.82 - idx * 0.08, "mean_return": 260.0 - idx * 20.0, "num_eval_episodes": 4}
        for idx, candidate in enumerate(candidates)
    }
    pool_stats = {"scenario_vectors": [vector_result["scenario_vector"]]}
    difficulty_results = DifficultyEvaluator().evaluate_batch(
        candidates,
        current_policy_eval,
        best_policy_eval,
        pool_stats,
        failure_summary,
        {item["scenario_id"]: item for item in constraint_results},
    )

    output = {
        "schema_version": "falcon.scenario_pipeline_smoke.v1",
        "base_config_path": str(base_config_path),
        "initial_config": initial_config,
        "scenario_vector_result": vector_result,
        "candidate_validations": candidate_validations,
        "candidates": candidates,
        "constraint_results": constraint_results,
        "generated_yaml_paths": generated_yaml_paths,
        "yaml_validation_results": yaml_validation_results,
        "replay_result": {
            "output_yaml_path": replay_result["output_yaml_path"],
            "schema_version": replay_result["schema_version"],
        } if replay_result else None,
        "difficulty_results": difficulty_results,
    }
    out_path = output_dir / "scenario_pipeline_smoke.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))

    if not all(item["is_valid"] for item in candidate_validations):
        raise AssertionError("Candidate schema validation failed.")
    if not generated_yaml_paths:
        raise AssertionError("No valid scenario YAML was generated.")
    if not difficulty_results:
        raise AssertionError("Difficulty evaluator returned no results.")


if __name__ == "__main__":
    main()
