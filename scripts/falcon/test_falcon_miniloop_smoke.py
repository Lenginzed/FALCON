#!/usr/bin/env python
"""Mini FALCON outer-loop smoke.

Pipeline:
trajectory fixture -> failure summary -> Ollama qwen3:8b candidates -> schema /
constraint -> YAML -> real policy evaluation -> difficulty evaluation ->
curriculum pool -> scheduler sampling plan.

This is an interface smoke only. It does not train MAPPO, modify MAPPO, use FSN,
or use Qwen3-14B/32B.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.curriculum_pool import CurriculumPool  # noqa: E402
from falcon.curriculum_scheduler import CurriculumScheduler  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.failure_analyzer import FailureAnalyzer  # noqa: E402
from falcon.llm_scenario_generator import QwenScenarioGenerator  # noqa: E402
from falcon.policy_evaluator import PolicyEvaluator  # noqa: E402
from falcon.scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config, save_scenario_yaml  # noqa: E402
from falcon.trajectory_recorder import extract_scenario_vector, load_trajectory, summarize_episode  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _default_checkpoint() -> Path:
    return ROOT_DIR / "tests" / "tmp_falcon_policy_smoke" / "mappo_2v2_smoke" / "actor_latest.pt"


def _checkpoint_paths() -> tuple[str | None, str | None, list[str]]:
    warnings: list[str] = []
    default = _default_checkpoint()
    current = os.environ.get("FALCON_CURRENT_CHECKPOINT") or str(default)
    best = os.environ.get("FALCON_BEST_CHECKPOINT") or str(default)
    if not Path(current).exists():
        warnings.append(f"Current checkpoint not found: {current}. Run scripts/falcon/train_minimal_mappo_2v2_smoke.py first.")
    if not Path(best).exists():
        warnings.append(f"Best checkpoint not found: {best}. Run scripts/falcon/train_minimal_mappo_2v2_smoke.py first.")
    if Path(current).exists() and Path(best).exists() and Path(current).resolve() == Path(best).resolve():
        warnings.append("Learning potential is not meaningful because current and best checkpoints are identical.")
    return current if Path(current).exists() else None, best if Path(best).exists() else None, warnings


def _failure_stage(summary: dict) -> str | None:
    if not summary["qwen_generation_success"]:
        return "qwen_generation"
    if summary["num_schema_valid"] <= 0:
        return "schema"
    if summary["num_constraint_valid"] <= 0:
        return "constraint"
    if summary["num_yaml_generated"] <= 0:
        return "yaml"
    if not summary["real_policy_eval_available"] or summary["num_policy_eval_success"] <= 0:
        return "policy_eval"
    if summary["num_difficulty_evaluated"] <= 0:
        return "difficulty"
    if not summary["sampling_plan_generated"]:
        return "scheduler"
    return None


def _schema_valid_count(schema_validations: list[dict]) -> int:
    return sum(1 for item in schema_validations if item.get("is_valid"))


def _candidate_source() -> str:
    return "llm_qwen8b"


def main() -> None:
    output_dir = Path(os.environ.get("FALCON_MINILOOP_OUTPUT_DIR", ROOT_DIR / "tests" / "tmp_falcon_miniloop"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
    fixture_path = ROOT_DIR / "tests" / "fixtures" / "falcon_coordination_failure_v2.json"
    warnings: list[str] = []

    trajectory = load_trajectory(fixture_path)
    trajectory["_source_trajectory"] = str(fixture_path)
    failure_summary = FailureAnalyzer().analyze_trajectory(
        trajectory,
        success_stats={"mean_success_team_reward": 500.0},
    )
    failure_summary["scenario_vector"] = extract_scenario_vector(trajectory)
    failure_summary["episode_summary"] = summarize_episode(trajectory)
    _write_json(output_dir / "falcon_miniloop_failure_summary.json", failure_summary)

    base_config = load_base_scenario_config(base_config_path)
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
    candidates: list[dict] = []
    generation_result = {
        "schema_version": "falcon.qwen_generation_result.v1",
        "health": health,
        "candidates": [],
        "warnings": warnings,
    }
    if health.get("server_reachable") and health.get("model_available"):
        candidates = generator.generate_from_failure_summary(failure_summary, base_config, num_scenarios=3)
        generation_result = generator.last_result
        warnings.extend(generation_result.get("warnings", []))
    else:
        warnings.append("Skipped qwen3:8b generation because Ollama health check did not pass.")

    _write_json(
        output_dir / "falcon_miniloop_candidates.json",
        {
            "schema_version": "falcon.miniloop_candidates.v1",
            "health": health,
            "generation_result": generation_result,
            "candidates": candidates,
        },
    )

    schema_validations = [{"scenario_id": candidate.get("scenario_id"), **validate_candidate_schema(candidate)} for candidate in candidates]
    checker = ConstraintChecker()
    constraint_results = checker.validate_batch(candidates)
    valid_candidates = []
    valid_constraint_results = []
    generated_yaml_paths = []
    for idx, (candidate, constraint) in enumerate(zip(candidates, constraint_results)):
        if not schema_validations[idx].get("is_valid"):
            continue
        if not constraint.get("is_valid"):
            continue
        yaml_config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
        yaml_config["scenario_id"] = candidate.get("scenario_id")
        yaml_path = output_dir / f"falcon_miniloop_generated_{idx:04d}.yaml"
        save_scenario_yaml(yaml_config, yaml_path)
        enriched = dict(candidate)
        enriched["yaml_path"] = str(yaml_path)
        enriched["scenario_yaml_path"] = str(yaml_path)
        valid_candidates.append(enriched)
        valid_constraint_results.append(constraint)
        generated_yaml_paths.append(str(yaml_path))

    _write_json(
        output_dir / "falcon_miniloop_validated_candidates.json",
        {
            "schema_version": "falcon.miniloop_validated_candidates.v1",
            "schema_validations": schema_validations,
            "constraint_results": constraint_results,
            "valid_candidates": valid_candidates,
            "generated_yaml_paths": generated_yaml_paths,
            "warnings": sorted(set(warnings)),
        },
    )

    current_checkpoint, best_checkpoint, checkpoint_warnings = _checkpoint_paths()
    warnings.extend(checkpoint_warnings)
    current_and_best_same = bool(
        current_checkpoint
        and best_checkpoint
        and Path(current_checkpoint).resolve() == Path(best_checkpoint).resolve()
    )

    policy_eval_results = []
    real_policy_eval_available = False
    num_policy_eval_success = 0
    if current_checkpoint and best_checkpoint and valid_candidates:
        evaluator = PolicyEvaluator({"base_config_path": str(base_config_path)})
        policy_eval_results = evaluator.evaluate_current_and_best(
            current_checkpoint,
            best_checkpoint,
            valid_candidates,
            num_episodes=int(os.environ.get("FALCON_MINILOOP_EVAL_EPISODES", "1") or 1),
        )
        real_policy_eval_available = bool(policy_eval_results) and all(item.get("real_policy_eval_available") for item in policy_eval_results)
        for item in policy_eval_results:
            warnings.extend(item.get("warnings", []))
        num_policy_eval_success = sum(
            1
            for item in policy_eval_results
            if item.get("current_policy_eval", {}).get("real_policy_eval_available")
            and item.get("best_policy_eval", {}).get("real_policy_eval_available")
            and item.get("current_policy_eval", {}).get("episode_results")
            and item.get("best_policy_eval", {}).get("episode_results")
        )
    elif not valid_candidates:
        warnings.append("Policy evaluation skipped because no schema+constraint valid candidates were available.")
    else:
        warnings.append("Policy evaluation skipped because smoke checkpoints were unavailable.")

    _write_json(
        output_dir / "falcon_miniloop_policy_eval.json",
        {
            "schema_version": "falcon.miniloop_policy_eval.v1",
            "current_checkpoint_path": current_checkpoint,
            "best_checkpoint_path": best_checkpoint,
            "current_and_best_same_checkpoint": current_and_best_same,
            "policy_eval_results": policy_eval_results,
            "warnings": sorted(set(warnings)),
        },
    )

    current_policy_evals = [item.get("current_policy_eval", {}) for item in policy_eval_results]
    best_policy_evals = [item.get("best_policy_eval", {}) for item in policy_eval_results]
    difficulty_results = []
    if policy_eval_results:
        difficulty_results = DifficultyEvaluator().evaluate_batch(
            valid_candidates,
            current_policy_evals,
            best_policy_evals,
            {"scenario_vectors": [failure_summary["scenario_vector"]]},
            failure_summary,
            valid_constraint_results,
        )
    else:
        warnings.append("Difficulty evaluation skipped because policy evaluation produced no results.")

    _write_json(
        output_dir / "falcon_miniloop_difficulty_eval.json",
        {
            "schema_version": "falcon.miniloop_difficulty_eval.v1",
            "difficulty_results": difficulty_results,
            "warnings": sorted(set(warnings)),
        },
    )

    pool = CurriculumPool()
    pool.add_batch(valid_candidates, difficulty_results, source=_candidate_source())
    pool_path = output_dir / "falcon_curriculum_pool.json"
    pool.save(pool_path)

    scheduler = CurriculumScheduler()
    sampling_plan = scheduler.build_sampling_plan(
        pool,
        base_scenarios=[
            {
                "scenario_id": "base_2v2_NoWeapon_Selfplay",
                "source": "original",
                "scenario_yaml_path": str(base_config_path),
                "target_failure_modes": [],
                "priority_level": "base",
            }
        ],
        random_scenarios=None,
        num_samples=int(os.environ.get("FALCON_MINILOOP_NUM_SAMPLES", "6") or 6),
    )
    if scheduler_warnings := sampling_plan.get("warnings"):
        warnings.extend(scheduler_warnings)
    sampling_plan_path = output_dir / "falcon_sampling_plan.json"
    scheduler.save_sampling_plan(sampling_plan, sampling_plan_path)

    accepted_items = pool.get_accepted()
    summary = {
        "schema_version": "falcon.miniloop_summary.v1",
        "qwen_generation_success": bool(candidates),
        "num_candidates_generated": len(candidates),
        "num_schema_valid": _schema_valid_count(schema_validations),
        "num_constraint_valid": sum(1 for result in constraint_results if result.get("is_valid")),
        "num_yaml_generated": len(generated_yaml_paths),
        "real_policy_eval_available": real_policy_eval_available,
        "num_policy_eval_success": num_policy_eval_success,
        "num_difficulty_evaluated": len(difficulty_results),
        "num_accepted_into_pool": len(accepted_items),
        "sampling_plan_generated": bool(sampling_plan.get("sampled_scenarios")),
        "current_checkpoint_path": current_checkpoint,
        "best_checkpoint_path": best_checkpoint,
        "current_and_best_same_checkpoint": current_and_best_same,
        "curriculum_pool_path": str(pool_path),
        "sampling_plan_path": str(sampling_plan_path),
        "failure_stage": None,
        "warnings": sorted(set(warnings)),
    }
    if not accepted_items and difficulty_results:
        rejected = [result.get("rejection_reasons", []) for result in difficulty_results]
        summary["warnings"].append(f"No LLM scenarios entered the accepted pool; difficulty rejection reasons: {rejected}")
    summary["failure_stage"] = _failure_stage(summary)
    _write_json(output_dir / "falcon_miniloop_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
