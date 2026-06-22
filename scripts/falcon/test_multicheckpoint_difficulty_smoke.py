#!/usr/bin/env python
"""Multi-checkpoint policy-eval and difficulty smoke for FALCON.

This script evaluates the same candidate scenarios with two different MAPPO
actor checkpoints and checks whether the double-boundary difficulty metrics
become non-trivial. It does not train MAPPO or modify MAPPO.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.curriculum_pool import CurriculumPool  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.failure_analyzer import FailureAnalyzer  # noqa: E402
from falcon.llm_scenario_generator import QwenScenarioGenerator  # noqa: E402
from falcon.policy_evaluator import PolicyEvaluator  # noqa: E402
from falcon.random_scenario_generator import RandomScenarioGenerator  # noqa: E402
from falcon.scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config, save_scenario_yaml  # noqa: E402
from falcon.trajectory_recorder import extract_scenario_vector, load_trajectory, summarize_episode  # noqa: E402


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _checkpoint_paths(output_dir: Path) -> Tuple[Optional[str], Optional[str], List[str]]:
    warnings: List[str] = []
    train_summary = _load_json(output_dir / "multicheckpoint_train_summary.json") or {}
    early = os.environ.get("FALCON_EARLY_CHECKPOINT") or train_summary.get("early_checkpoint_path")
    later = os.environ.get("FALCON_LATER_CHECKPOINT") or train_summary.get("later_checkpoint_path")
    if not early:
        early = str(output_dir / "mappo_2v2_multicheckpoint" / "actor_early.pt")
    if not later:
        later = str(output_dir / "mappo_2v2_multicheckpoint" / "actor_later.pt")
    if not Path(early).exists():
        warnings.append(f"Early checkpoint not found: {early}. Run scripts/falcon/train_mappo_2v2_multicheckpoint_smoke.py first.")
        early = None
    if not Path(later).exists():
        warnings.append(f"Later checkpoint not found: {later}. Run scripts/falcon/train_mappo_2v2_multicheckpoint_smoke.py first.")
        later = None
    return early, later, warnings


def _failure_summary(fixture_path: Path) -> Dict[str, Any]:
    trajectory = load_trajectory(fixture_path)
    trajectory["_source_trajectory"] = str(fixture_path)
    failure_summary = FailureAnalyzer().analyze_trajectory(
        trajectory,
        success_stats={"mean_success_team_reward": 500.0},
    )
    failure_summary["scenario_vector"] = extract_scenario_vector(trajectory)
    failure_summary["episode_summary"] = summarize_episode(trajectory)
    return failure_summary


def _load_existing_qwen_candidates() -> List[Dict[str, Any]]:
    candidate_sources = [
        ROOT_DIR / "tests" / "tmp_falcon_miniloop" / "falcon_miniloop_validated_candidates.json",
        ROOT_DIR / "tests" / "tmp_falcon_trajectories" / "ollama_qwen8b_candidate_scenarios.json",
    ]
    for path in candidate_sources:
        data = _load_json(path)
        if not data:
            continue
        candidates = data.get("valid_candidates") or data.get("candidates") or []
        candidates = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
        if candidates:
            return candidates[:3]
    return []


def _generate_qwen_candidates(failure_summary: Mapping[str, Any], base_config: Mapping[str, Any], warnings: List[str]) -> List[Dict[str, Any]]:
    generator = QwenScenarioGenerator(
        {
            "provider": "ollama",
            "provider_mode": "ollama_native",
            "model_name": "qwen3:8b",
            "think": False,
            "stream": False,
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 4096,
            "timeout": 180.0,
            "num_retries": 2,
        }
    )
    health = generator.check_llm_server()
    warnings.extend(health.get("warnings", []))
    if not health.get("server_reachable") or not health.get("model_available"):
        warnings.append("Could not regenerate Qwen candidates because Ollama/qwen3:8b is unavailable.")
        return []
    candidates = generator.generate_from_failure_summary(failure_summary, base_config, num_scenarios=3)
    warnings.extend(generator.last_result.get("warnings", []))
    return candidates[:3]


def _prepare_candidates(
    output_dir: Path,
    base_config: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    warnings: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    qwen_candidates = _load_existing_qwen_candidates()
    if not qwen_candidates:
        qwen_candidates = _generate_qwen_candidates(failure_summary, base_config, warnings)
    random_candidates = RandomScenarioGenerator({"seed": 17}).generate_from_failure_summary(
        failure_summary,
        base_config,
        num_scenarios=3,
    )
    for candidate in qwen_candidates:
        candidate.setdefault("metadata", {})["multicheckpoint_source"] = "qwen_existing_or_generated"
    for candidate in random_candidates:
        candidate.setdefault("metadata", {})["multicheckpoint_source"] = "random_baseline"
    all_candidates = [dict(candidate) for candidate in qwen_candidates[:3] + random_candidates[:3]]

    checker = ConstraintChecker()
    prepared: List[Dict[str, Any]] = []
    schema_validations: List[Dict[str, Any]] = []
    constraint_results: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(all_candidates):
        validation = {"scenario_id": candidate.get("scenario_id"), **validate_candidate_schema(candidate)}
        schema_validations.append(validation)
        if not validation.get("is_valid"):
            constraint_results.append(
                {
                    "schema_version": "1.0",
                    "scenario_id": candidate.get("scenario_id", f"candidate_{idx:04d}"),
                    "is_valid": False,
                    "rejection_reasons": ["candidate_schema_invalid"],
                    "missing_fields": validation.get("missing_fields", []),
                    "warnings": validation.get("warnings", []),
                }
            )
            continue
        constraint = checker.validate_candidate(candidate)
        constraint_results.append(constraint)
        if not constraint.get("is_valid"):
            continue
        yaml_config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
        yaml_config["scenario_id"] = candidate.get("scenario_id")
        yaml_path = output_dir / f"multicheckpoint_candidate_{idx:04d}.yaml"
        save_scenario_yaml(yaml_config, yaml_path)
        enriched = dict(candidate)
        enriched["yaml_path"] = str(yaml_path)
        enriched["scenario_yaml_path"] = str(yaml_path)
        prepared.append(enriched)
    return prepared, qwen_candidates[:3], random_candidates[:3], schema_validations, constraint_results


def _policy_rows(
    candidates: Sequence[Mapping[str, Any]],
    policy_eval_results: Sequence[Mapping[str, Any]],
    difficulty_results: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    difficulty_by_id = {str(item.get("scenario_id")): item for item in difficulty_results}
    for idx, result in enumerate(policy_eval_results):
        scenario_id = str(result.get("scenario_id", candidates[idx].get("scenario_id") if idx < len(candidates) else idx))
        current = result.get("current_policy_eval") or {}
        best = result.get("best_policy_eval") or {}
        difficulty = difficulty_by_id.get(scenario_id, {})
        rows.append(
            {
                "schema_version": "falcon.multicheckpoint_policy_eval_row.v1",
                "scenario_id": scenario_id,
                "generator_type": candidates[idx].get("generator_type") if idx < len(candidates) else None,
                "W_t": current.get("win_rate"),
                "W_best": best.get("win_rate"),
                "R_t": current.get("mean_return"),
                "R_best": best.get("mean_return"),
                "current_policy_weakness": difficulty.get("current_policy_weakness"),
                "historical_solvability": difficulty.get("historical_solvability"),
                "learning_potential": difficulty.get("learning_potential"),
                "win_rate_gap_abs": round(abs(_float(best.get("win_rate")) - _float(current.get("win_rate"))), 6),
                "return_gap": round(_float(best.get("mean_return")) - _float(current.get("mean_return")), 6),
                "final_value_score": difficulty.get("final_value_score"),
                "accepted_into_curriculum_pool": difficulty.get("accepted_into_curriculum_pool"),
                "rejection_reasons": difficulty.get("rejection_reasons", []),
            }
        )
    return rows


def _failure_stage(summary: Mapping[str, Any]) -> Optional[str]:
    if not summary.get("early_checkpoint_found") or not summary.get("later_checkpoint_found"):
        return "checkpoint"
    if not summary.get("current_best_are_different"):
        return "checkpoint_same_path"
    if summary.get("num_policy_eval_success", 0) <= 0:
        return "policy_eval"
    if summary.get("num_difficulty_evaluated", 0) <= 0:
        return "difficulty"
    return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    output_dir = Path(os.environ.get("FALCON_MULTICHECKPOINT_OUTPUT_DIR", ROOT_DIR / "tests" / "tmp_falcon_multicheckpoint"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
    fixture_path = ROOT_DIR / "tests" / "fixtures" / "falcon_coordination_failure_v2.json"
    warnings: List[str] = []

    early_checkpoint, later_checkpoint, checkpoint_warnings = _checkpoint_paths(output_dir)
    warnings.extend(checkpoint_warnings)
    base_config = load_base_scenario_config(base_config_path)
    failure_summary = _failure_summary(fixture_path)
    candidates, qwen_candidates, random_candidates, schema_validations, constraint_results = _prepare_candidates(
        output_dir,
        base_config,
        failure_summary,
        warnings,
    )

    policy_eval_results: List[Dict[str, Any]] = []
    difficulty_results: List[Dict[str, Any]] = []
    current_best_are_different = bool(
        early_checkpoint
        and later_checkpoint
        and Path(early_checkpoint).resolve() != Path(later_checkpoint).resolve()
    )
    if early_checkpoint and later_checkpoint and candidates:
        evaluator = PolicyEvaluator({"base_config_path": str(base_config_path)})
        policy_eval_results = evaluator.evaluate_current_and_best(
            early_checkpoint,
            later_checkpoint,
            candidates,
            num_episodes=int(os.environ.get("FALCON_MULTICHECKPOINT_EVAL_EPISODES", "1") or 1),
        )
        for item in policy_eval_results:
            warnings.extend(item.get("warnings", []))
        current_policy_evals = [item.get("current_policy_eval", {}) for item in policy_eval_results]
        best_policy_evals = [item.get("best_policy_eval", {}) for item in policy_eval_results]
        valid_constraints = [result for result in constraint_results if result.get("is_valid")]
        difficulty_results = DifficultyEvaluator().evaluate_batch(
            candidates,
            current_policy_evals,
            best_policy_evals,
            {"scenario_vectors": [failure_summary["scenario_vector"]]},
            failure_summary,
            valid_constraints,
        )
    elif not candidates:
        warnings.append("No schema+constraint valid qwen/random candidates were available for policy evaluation.")
    else:
        warnings.append("Policy evaluation skipped because early/later checkpoints were unavailable.")

    rows = _policy_rows(candidates, policy_eval_results, difficulty_results)
    learning_values = [_float(row.get("learning_potential")) for row in rows]
    win_gaps = [_float(row.get("win_rate_gap_abs")) for row in rows]
    learning_nontrivial = any(gap >= 0.2 for gap in win_gaps)
    if rows and not learning_nontrivial:
        warnings.append("Different checkpoints did not produce meaningful performance separation; longer training may be required.")

    pool = CurriculumPool()
    pool.add_batch(candidates, difficulty_results, source="multicheckpoint_smoke")
    pool_path = output_dir / "multicheckpoint_curriculum_pool.json"
    pool.save(pool_path)

    num_policy_eval_success = sum(
        1
        for item in policy_eval_results
        if item.get("current_policy_eval", {}).get("real_policy_eval_available")
        and item.get("best_policy_eval", {}).get("real_policy_eval_available")
        and item.get("current_policy_eval", {}).get("episode_results")
        and item.get("best_policy_eval", {}).get("episode_results")
    )
    rejected_reasons = [reason for item in difficulty_results for reason in item.get("rejection_reasons", [])]
    summary = {
        "schema_version": "falcon.multicheckpoint_summary.v1",
        "early_checkpoint_found": bool(early_checkpoint),
        "later_checkpoint_found": bool(later_checkpoint),
        "early_checkpoint_path": early_checkpoint,
        "later_checkpoint_path": later_checkpoint,
        "current_best_are_different": current_best_are_different,
        "num_qwen_candidates": len(qwen_candidates),
        "num_random_candidates": len(random_candidates),
        "num_schema_valid": sum(1 for item in schema_validations if item.get("is_valid")),
        "num_constraint_valid": sum(1 for item in constraint_results if item.get("is_valid")),
        "num_policy_eval_success": num_policy_eval_success,
        "num_difficulty_evaluated": len(difficulty_results),
        "num_accepted_into_pool": sum(1 for item in difficulty_results if item.get("accepted_into_curriculum_pool")),
        "num_rejected_too_easy": sum(1 for reason in rejected_reasons if reason == "too_easy_for_current_policy"),
        "num_rejected_not_solvable": sum(1 for reason in rejected_reasons if reason == "not_solvable_by_historical_best_policy"),
        "mean_learning_potential": round(sum(learning_values) / len(learning_values), 6) if learning_values else 0.0,
        "max_learning_potential": round(max(learning_values), 6) if learning_values else 0.0,
        "learning_potential_nontrivial": learning_nontrivial,
        "failure_stage": None,
        "warnings": sorted(set(warnings)),
    }
    summary["failure_stage"] = _failure_stage(summary)

    _write_json(
        output_dir / "multicheckpoint_policy_eval.json",
        {
            "schema_version": "falcon.multicheckpoint_policy_eval.v1",
            "early_checkpoint_path": early_checkpoint,
            "later_checkpoint_path": later_checkpoint,
            "candidates": candidates,
            "policy_eval_results": policy_eval_results,
            "policy_eval_rows": rows,
            "warnings": sorted(set(warnings)),
        },
    )
    _write_json(
        output_dir / "multicheckpoint_difficulty_eval.json",
        {
            "schema_version": "falcon.multicheckpoint_difficulty_eval.v1",
            "difficulty_results": difficulty_results,
            "policy_eval_rows": rows,
            "warnings": sorted(set(warnings)),
        },
    )
    _write_json(output_dir / "multicheckpoint_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
