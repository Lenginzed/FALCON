#!/usr/bin/env python
"""Smoke test for FALCON policy evaluation bridge.

This does not train MAPPO and does not modify MAPPO. It attempts real offline
evaluation only when actor checkpoints are present; otherwise it uses the
MockPolicyEvaluator to keep the difficulty-evaluator bridge testable.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.failure_analyzer import FailureAnalyzer  # noqa: E402
from falcon.policy_evaluator import MockPolicyEvaluator, PolicyEvaluator, discover_policy_checkpoints  # noqa: E402
from falcon.scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config  # noqa: E402
from falcon.trajectory_recorder import extract_scenario_vector, load_trajectory, summarize_episode  # noqa: E402


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _load_candidates(output_dir):
    candidate_path = output_dir / "ollama_qwen8b_candidate_scenarios.json"
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = data.get("valid_candidates") or []
        if candidates:
            return candidates, str(candidate_path), []
    warnings = [f"No qwen3:8b candidate JSON found at {candidate_path}; using generated YAML paths if present."]
    candidates = []
    for yaml_path in sorted(output_dir.glob("ollama_qwen8b_generated_*.yaml")):
        candidates.append({"scenario_id": yaml_path.stem, "yaml_path": str(yaml_path), "target_failure_modes": ["coordination_failure"]})
    return candidates, str(candidate_path), warnings


def _candidate_yaml_config(candidate, base_config):
    if candidate.get("yaml_path"):
        return candidate["yaml_path"]
    if candidate.get("initial_config"):
        yaml_config = apply_initial_config_to_yaml(base_config, candidate["initial_config"])
        yaml_config["scenario_id"] = candidate.get("scenario_id")
        return yaml_config
    yaml_config = dict(base_config)
    yaml_config["scenario_id"] = candidate.get("scenario_id", "fallback_candidate")
    return yaml_config


def _checkpoint_config(discovery):
    current = os.environ.get("FALCON_CURRENT_CHECKPOINT") or discovery.get("current_checkpoint")
    best = os.environ.get("FALCON_BEST_CHECKPOINT") or discovery.get("best_checkpoint")
    return current, best


def _failure_stage(summary):
    if summary["num_candidate_scenarios"] <= 0:
        return "candidate_loading"
    if summary["num_scenarios_env_load_success"] <= 0:
        return "env_load"
    if summary["real_policy_eval_available"]:
        if summary["num_policy_eval_success"] <= 0:
            return "real_policy_eval"
    elif summary.get("evaluator_used") == "PolicyEvaluator" and summary["current_checkpoint_found"] and summary["best_checkpoint_found"]:
        if not summary.get("actor_loaded"):
            return "actor_loading"
        return "real_policy_eval"
    elif not summary["current_checkpoint_found"] or not summary["best_checkpoint_found"]:
        return "checkpoint_loading_not_available"
    if not summary["difficulty_evaluator_consumed"]:
        return "difficulty_evaluator"
    return None


def main():
    candidate_dir = Path(os.environ.get("FALCON_POLICY_BRIDGE_CANDIDATE_DIR", ROOT_DIR / "tests" / "tmp_falcon_trajectories"))
    output_dir = Path(os.environ.get("FALCON_POLICY_BRIDGE_OUTPUT_DIR", candidate_dir))
    output_file = os.environ.get("FALCON_POLICY_BRIDGE_OUTPUT_FILE", "policy_evaluation_bridge_smoke.json")
    max_candidates = int(os.environ.get("FALCON_POLICY_BRIDGE_MAX_CANDIDATES", "0") or 0)
    num_eval_episodes = int(os.environ.get("FALCON_POLICY_BRIDGE_NUM_EPISODES", "1") or 1)
    allow_mock_fallback = os.environ.get("FALCON_POLICY_BRIDGE_ALLOW_MOCK_FALLBACK", "true").lower() not in {"0", "false", "no"}
    base_config_path = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
    fixture_path = ROOT_DIR / "tests" / "fixtures" / "falcon_coordination_failure_v2.json"
    base_config = load_base_scenario_config(base_config_path)
    candidates, candidate_source, warnings = _load_candidates(candidate_dir)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    discovery = discover_policy_checkpoints(ROOT_DIR / "results")
    warnings.extend(discovery.get("warnings", []))
    current_checkpoint, best_checkpoint = _checkpoint_config(discovery)
    current_checkpoint_found = bool(current_checkpoint and Path(current_checkpoint).exists())
    best_checkpoint_found = bool(best_checkpoint and Path(best_checkpoint).exists())

    constraint_checker = ConstraintChecker({"enable_env_load_check": True})
    env_load_results = []
    env_load_success = 0
    for idx, candidate in enumerate(candidates):
        scenario_yaml = _candidate_yaml_config(candidate, base_config)
        if isinstance(scenario_yaml, str):
            with Path(scenario_yaml).open("r", encoding="utf-8") as f:
                import yaml

                yaml_config = yaml.safe_load(f)
        else:
            yaml_config = scenario_yaml
        env_result = constraint_checker.validate_yaml_config(
            yaml_config,
            enable_env_load_check=(idx == 0),
            temp_config_name=f"policy_bridge_{idx:04d}",
        )
        env_load_results.append(env_result)
        if env_result.get("is_valid"):
            env_load_success += 1
        warnings.extend(env_result.get("warnings", []))

    real_results = []
    real_policy_eval_available = False
    evaluator = PolicyEvaluator({"base_config_path": str(base_config_path)})
    if current_checkpoint_found and best_checkpoint_found:
        real_results = evaluator.evaluate_current_and_best(current_checkpoint, best_checkpoint, candidates, num_episodes=num_eval_episodes)
        real_policy_eval_available = all(item.get("real_policy_eval_available") for item in real_results)
        for item in real_results:
            warnings.extend(item.get("warnings", []))

    if real_policy_eval_available:
        policy_eval_results = real_results
        evaluator_used = "PolicyEvaluator"
    elif allow_mock_fallback:
        mock = MockPolicyEvaluator()
        policy_eval_results = mock.evaluate_current_and_best(
            {"policy_id": "mock_current"},
            {"policy_id": "mock_best"},
            candidates,
            num_episodes=5,
        )
        evaluator_used = "MockPolicyEvaluator"
        if current_checkpoint_found and best_checkpoint_found:
            warnings.append("Real checkpoints were found but real evaluation did not complete; used MockPolicyEvaluator for smoke continuity.")
        else:
            warnings.append("Real MAPPO checkpoints were not found; used MockPolicyEvaluator for bridge smoke test.")
    else:
        policy_eval_results = real_results
        evaluator_used = "PolicyEvaluator"
        warnings.append("Mock fallback disabled; reporting real policy evaluation results only.")

    trajectory = load_trajectory(fixture_path)
    trajectory["_source_trajectory"] = str(fixture_path)
    failure_summary = FailureAnalyzer().analyze_trajectory(trajectory, success_stats={"mean_success_team_reward": 500.0})
    failure_summary["scenario_vector"] = extract_scenario_vector(trajectory)
    failure_summary["episode_summary"] = summarize_episode(trajectory)
    current_policy_evals = [item["current_policy_eval"] for item in policy_eval_results]
    best_policy_evals = [item["best_policy_eval"] for item in policy_eval_results]
    constraint_results = [
        result if result.get("is_valid") else {"schema_version": "1.0", "scenario_id": result.get("scenario_id"), "is_valid": False, "rejection_reasons": result.get("rejection_reasons", [])}
        for result in env_load_results
    ]
    difficulty_results = DifficultyEvaluator().evaluate_batch(
        candidates,
        current_policy_evals,
        best_policy_evals,
        {"scenario_vectors": [failure_summary["scenario_vector"]]},
        failure_summary,
        constraint_results,
    )

    num_policy_eval_success = sum(
        1
        for item in policy_eval_results
        if len(item.get("current_policy_eval", {}).get("episode_results", [])) > 0
        and len(item.get("best_policy_eval", {}).get("episode_results", [])) > 0
    )
    num_real_policy_eval_success = sum(
        1
        for item in policy_eval_results
        if item.get("current_policy_eval", {}).get("real_policy_eval_available")
        and item.get("best_policy_eval", {}).get("real_policy_eval_available")
        and len(item.get("current_policy_eval", {}).get("episode_results", [])) > 0
        and len(item.get("best_policy_eval", {}).get("episode_results", [])) > 0
    )
    actor_loaded = any(
        item.get("current_policy_eval", {}).get("actor_loaded") or item.get("best_policy_eval", {}).get("actor_loaded")
        for item in policy_eval_results
    )
    summary = {
        "schema_version": "falcon.policy_evaluation_bridge_smoke_summary.v1",
        "real_policy_eval_available": bool(real_policy_eval_available),
        "evaluator_used": evaluator_used,
        "current_checkpoint_found": current_checkpoint_found,
        "best_checkpoint_found": best_checkpoint_found,
        "actor_loaded": bool(actor_loaded),
        "current_checkpoint": current_checkpoint,
        "best_checkpoint": best_checkpoint,
        "checkpoint_discovery": discovery,
        "candidate_source": candidate_source,
        "num_candidate_scenarios": len(candidates),
        "num_scenarios_env_load_success": env_load_success,
        "num_policy_eval_success": num_policy_eval_success,
        "num_real_policy_eval_success": num_real_policy_eval_success,
        "num_eval_episodes": num_eval_episodes,
        "difficulty_evaluator_consumed": bool(len(difficulty_results) == len(candidates) and len(candidates) > 0),
        "failure_stage": None,
        "warnings": sorted(set(warnings)),
    }
    summary["failure_stage"] = _failure_stage(summary)
    output = {
        "schema_version": "falcon.policy_evaluation_bridge_smoke.v1",
        "summary": summary,
        "candidates": candidates,
        "env_load_results": env_load_results,
        "policy_eval_results": policy_eval_results,
        "difficulty_results": difficulty_results,
    }
    out_path = output_dir / output_file
    _write_json(out_path, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
