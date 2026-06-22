#!/usr/bin/env python
"""Smoke test for FALCON Ollama + qwen3:8b scenario generation.

Expected local setup:
  ollama serve
  ollama pull qwen3:8b

The LLM is used only as an offline candidate scenario generator.
"""

import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.failure_analyzer import FailureAnalyzer  # noqa: E402
from falcon.llm_scenario_generator import QwenScenarioGenerator  # noqa: E402
from falcon.scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config, save_scenario_yaml  # noqa: E402
from falcon.trajectory_recorder import extract_scenario_vector, load_trajectory, summarize_episode  # noqa: E402


SCENARIO_VECTOR_BOUNDS = {
    "team_center_distance": [6000.0, 18000.0],
    "own_formation_spread": [1000.0, 8000.0],
    "opponent_formation_spread": [1000.0, 8000.0],
    "altitude_difference": [-3000.0, 3000.0],
    "velocity_difference": [-80.0, 80.0],
    "heading_difference": [0.0, 2.0 * math.pi],
    "approximate_aspect_angle": [0.0, 2.0 * math.pi],
}


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _raw_response_text(raw_responses):
    if not raw_responses:
        return "No raw response was captured.\n"
    chunks = []
    for item in raw_responses:
        chunks.append(f"--- attempt {item.get('attempt')} ---")
        if item.get("error"):
            chunks.append(f"ERROR: {item.get('error')}")
        else:
            chunks.append(str(item.get("content", "")))
            if item.get("raw_response") is not None:
                chunks.append("--- provider raw response json ---")
                chunks.append(json.dumps(item.get("raw_response"), indent=2, sort_keys=True, ensure_ascii=False))
    return "\n".join(chunks) + "\n"


def _first_raw_content(raw_responses):
    for item in raw_responses:
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _quality_warnings(raw_text, parsed_candidates, attempts):
    warnings = []
    lower = raw_text.lower()
    disallowed_patterns = {
        "direct_control_action": ["aileron", "elevator", "rudder", "throttle", "control command", "动作指令", "控制指令", "油门", "副翼", "升降舵", "方向舵"],
        "flight_strategy_instruction": ["fly toward", "turn left", "turn right", "climb to", "dive to", "飞行策略", "机动策略", "爬升到", "俯冲到"],
        "reward_function_modification": ["reward function", "modify reward", "奖励函数", "修改奖励"],
        "dynamics_model_modification": ["dynamics model", "modify dynamics", "动力学模型", "修改动力学"],
        "mappo_modification": ["mappo", "ppo update", "policy gradient", "修改mappo", "修改 mappo"],
        "markdown_or_codeblock": ["```"],
    }
    for label, patterns in disallowed_patterns.items():
        if any(pattern in lower for pattern in patterns):
            warnings.append(f"LLM raw response may contain disallowed content: {label}.")
    if "<think" in lower or "</think>" in lower:
        warnings.append("LLM raw response contained <think> tags.")
    for idx, candidate in enumerate(parsed_candidates):
        if not candidate.get("target_failure_modes"):
            warnings.append(f"Parsed candidate {idx} is missing target_failure_modes.")
        changed = candidate.get("changed_factors")
        if not changed:
            warnings.append(f"Parsed candidate {idx} is missing changed_factors.")
        elif isinstance(changed, list) and len(changed) > 3:
            warnings.append(f"Parsed candidate {idx} changes too many factors ({len(changed)} > 3).")
        vector = candidate.get("scenario_vector") or {}
        for key, bounds in SCENARIO_VECTOR_BOUNDS.items():
            value = vector.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                warnings.append(f"Parsed candidate {idx} scenario_vector.{key} is not numeric.")
                continue
            if not bounds[0] <= value <= bounds[1]:
                warnings.append(f"Parsed candidate {idx} scenario_vector.{key}={value} is outside {bounds}.")
    for attempt in attempts:
        for warning in attempt.get("warnings") or []:
            text = str(warning)
            if any(token in text for token in ("filled", "clipped", "missing", "Extracted JSON", "think")):
                warnings.append(f"Repair/retry evidence: {text}")
    return sorted(set(warnings))


def _failure_stage(summary):
    if not summary["server_reachable"]:
        return "server"
    if not summary["model_available"]:
        return "model"
    if not summary["raw_response_non_empty"]:
        return "generation"
    if not summary["json_parse_success"]:
        return "parse"
    if summary["num_schema_valid"] <= 0:
        return "schema"
    if summary["num_constraint_valid"] <= 0:
        return "constraint"
    if summary["num_yaml_generated"] <= 0:
        return "yaml"
    if summary["num_env_load_success"] <= 0:
        return "env_load"
    if not summary["difficulty_evaluator_consumed"]:
        return "difficulty_evaluator"
    return None


def main():
    base_config_path = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
    fixture_path = ROOT_DIR / "tests" / "fixtures" / "falcon_coordination_failure_v2.json"
    output_dir = ROOT_DIR / "tests" / "tmp_falcon_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = QwenScenarioGenerator(
        {
            "provider": "ollama",
            "provider_mode": "ollama_native",
            "base_url_native": "http://localhost:11434",
            "base_url_openai": "http://localhost:11434/v1",
            "model_name": "qwen3:8b",
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 4096,
            "timeout": 180.0,
            "stream": False,
            "think": False,
            "reasoning_effort": "none",
            "num_retries": 2,
        }
    )
    health = generator.check_llm_server()
    warnings = list(health.get("warnings", []))

    trajectory = load_trajectory(fixture_path)
    trajectory["_source_trajectory"] = str(fixture_path)
    failure_summary = FailureAnalyzer().analyze_trajectory(
        trajectory,
        success_stats={"mean_success_team_reward": 500.0},
    )
    failure_summary["scenario_vector"] = extract_scenario_vector(trajectory)
    failure_summary["episode_summary"] = summarize_episode(trajectory)
    base_config = load_base_scenario_config(base_config_path)

    candidates = []
    generation_result = {
        "schema_version": "falcon.qwen_generation_result.v1",
        "provider": "ollama",
        "provider_mode": "ollama_native",
        "model_name": "qwen3:8b",
        "raw_responses": [],
        "attempts": [],
        "warnings": warnings,
    }
    if health.get("server_reachable") and health.get("model_available"):
        candidates = generator.generate_from_failure_summary(failure_summary, base_config, num_scenarios=3)
        generation_result = generator.last_result
        warnings.extend(generation_result.get("warnings", []))

    raw_path = output_dir / "ollama_qwen8b_raw_response.txt"
    raw_path.write_text(_raw_response_text(generation_result.get("raw_responses", [])), encoding="utf-8")
    raw_content = _first_raw_content(generation_result.get("raw_responses", []))
    raw_parse = generator.parse_llm_response(raw_content) if raw_content else {"is_valid_json": False, "json_text": "", "thinking_detected": False, "warnings": []}

    parsed_candidates = []
    for attempt in generation_result.get("attempts", []):
        parsed_candidates.extend((attempt.get("parse_result") or {}).get("candidates") or [])
    if not parsed_candidates:
        parsed_candidates.extend(raw_parse.get("candidates") or [])

    schema_validations = [{"scenario_id": candidate.get("scenario_id"), **validate_candidate_schema(candidate)} for candidate in candidates]
    checker = ConstraintChecker()
    constraint_results = checker.validate_batch(candidates)
    valid_candidates = [
        candidate
        for candidate, constraint in zip(candidates, constraint_results)
        if constraint.get("is_valid")
    ]

    generated_yaml_paths = []
    env_load_results = []
    for idx, candidate in enumerate(valid_candidates):
        yaml_config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
        yaml_path = output_dir / f"ollama_qwen8b_generated_{idx:04d}.yaml"
        save_scenario_yaml(yaml_config, yaml_path)
        generated_yaml_paths.append(str(yaml_path))
        if not env_load_results:
            env_checker = ConstraintChecker({"enable_env_load_check": True})
            env_result = env_checker.validate_yaml_config(
                yaml_config,
                enable_env_load_check=True,
                temp_config_name=f"ollama_qwen8b_generated_{idx:04d}",
            )
            env_load_results.append(env_result)
            warnings.extend(env_result.get("warnings", []))

    if not generated_yaml_paths:
        warnings.append("No valid Ollama qwen3:8b candidate YAML was generated; env load check was skipped.")

    current_policy_evals = [{"win_rate": 0.35, "mean_return": -150.0, "num_eval_episodes": 8} for _ in valid_candidates]
    best_policy_evals = [{"win_rate": 0.78, "mean_return": 120.0, "num_eval_episodes": 8} for _ in valid_candidates]
    pool_stats = {"scenario_vectors": [failure_summary["scenario_vector"]]}
    difficulty_results = DifficultyEvaluator().evaluate_batch(
        valid_candidates,
        current_policy_evals,
        best_policy_evals,
        pool_stats,
        failure_summary,
        [result for result in constraint_results if result.get("is_valid")],
    )

    attempt_schema_validations = [
        item
        for attempt in generation_result.get("attempts", [])
        for item in attempt.get("schema_validations", [])
    ]
    attempt_constraint_results = [
        item
        for attempt in generation_result.get("attempts", [])
        for item in attempt.get("constraint_results", [])
    ]
    quality_warnings = _quality_warnings(raw_content, parsed_candidates, generation_result.get("attempts", []))
    warnings.extend(quality_warnings)
    json_parse_success = bool(raw_parse.get("is_valid_json") or any((attempt.get("parse_result") or {}).get("is_valid_json") for attempt in generation_result.get("attempts", [])))
    raw_response_json_only = bool(raw_content.strip()) and bool(raw_parse.get("is_valid_json")) and raw_content.strip() == str(raw_parse.get("json_text", "")).strip()
    repair_triggered = any(attempt.get("warnings") for attempt in generation_result.get("attempts", [])) or any("Repair/retry evidence" in warning for warning in quality_warnings)
    num_schema_valid = sum(1 for item in attempt_schema_validations if item.get("is_valid"))
    if not attempt_schema_validations:
        num_schema_valid = sum(1 for item in schema_validations if item.get("is_valid"))
    num_constraint_valid = sum(1 for item in attempt_constraint_results if item.get("is_valid"))
    if not attempt_constraint_results:
        num_constraint_valid = sum(1 for item in constraint_results if item.get("is_valid"))
    thinking_detected = bool(raw_parse.get("thinking_detected")) or bool(generation_result.get("thinking_detected")) or any(item.get("thinking_detected") for item in generation_result.get("raw_responses", []))

    summary_core = {
        "provider": "ollama",
        "provider_mode": "ollama_native",
        "base_url": "http://localhost:11434",
        "model_name": "qwen3:8b",
        "server_reachable": bool(health.get("server_reachable")),
        "model_available": bool(health.get("model_available")),
        "think_disabled_requested": True,
        "thinking_detected": thinking_detected,
        "raw_response_non_empty": bool(raw_content.strip()),
        "raw_response_json_only": raw_response_json_only,
        "json_parse_success": json_parse_success,
        "repair_triggered": bool(repair_triggered),
        "num_candidates_generated": len(parsed_candidates) if parsed_candidates else len(candidates),
        "num_schema_valid": num_schema_valid,
        "num_constraint_valid": num_constraint_valid,
        "num_yaml_generated": len(generated_yaml_paths),
        "num_env_load_success": sum(1 for item in env_load_results if item.get("is_valid")),
        "difficulty_evaluator_consumed": bool(valid_candidates and len(difficulty_results) == len(valid_candidates)),
        "failure_stage": None,
        "warnings": sorted(set(warnings)),
    }
    summary_core["failure_stage"] = _failure_stage(summary_core)

    candidate_json = {
        "schema_version": "falcon.ollama_qwen8b_candidate_scenarios.v1",
        "health": health,
        "generation_result": generation_result,
        "parsed_candidates": parsed_candidates,
        "valid_candidates": candidates,
    }
    validated_json = {
        "schema_version": "falcon.ollama_qwen8b_validated_scenarios.v1",
        "schema_validations": schema_validations,
        "constraint_results": constraint_results,
        "valid_candidates": valid_candidates,
        "generated_yaml_paths": generated_yaml_paths,
        "env_load_results": env_load_results,
        "warnings": sorted(set(warnings)),
    }
    difficulty_json = {
        "schema_version": "falcon.ollama_qwen8b_difficulty_evaluation.v1",
        "difficulty_results": difficulty_results,
        "warnings": sorted(set(warnings)),
    }

    _write_json(output_dir / "ollama_qwen8b_candidate_scenarios.json", candidate_json)
    _write_json(output_dir / "ollama_qwen8b_validated_scenarios.json", validated_json)
    _write_json(output_dir / "ollama_qwen8b_difficulty_evaluation.json", difficulty_json)
    _write_json(output_dir / "ollama_qwen8b_smoke_summary.json", {"schema_version": "falcon.ollama_qwen8b_smoke_summary.v1", **summary_core})
    print(json.dumps({"schema_version": "falcon.ollama_qwen8b_smoke.v1", **summary_core}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
