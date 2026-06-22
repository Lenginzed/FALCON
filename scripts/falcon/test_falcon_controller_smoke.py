#!/usr/bin/env python
"""Run one minimal FALCON controller smoke loop.

Round 0 creates a current checkpoint, builds failure-aware candidates, evaluates
them, updates the curriculum pool, and builds a sampling plan.
Round 1 consumes the sampling plan through the training-entry bridge and runs a
tiny MAPPO smoke train.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.falcon_controller import FalconController  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "tests" / "tmp_falcon_controller_smoke"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _failure_stage(summary: Mapping[str, Any]) -> str | None:
    if not summary.get("round0_initial_training_success"):
        return "round0_initial_training"
    if not summary.get("qwen_generation_success"):
        return "qwen_generation"
    if summary.get("num_schema_valid", 0) <= 0:
        return "schema"
    if summary.get("num_constraint_valid", 0) <= 0:
        return "constraint"
    if summary.get("num_yaml_generated", 0) <= 0:
        return "yaml"
    if summary.get("num_policy_eval_success", 0) <= 0:
        return "policy_eval"
    if summary.get("num_difficulty_evaluated", 0) <= 0:
        return "difficulty"
    if not summary.get("sampling_plan_generated"):
        return "sampling_plan"
    if not summary.get("round1_training_started"):
        return "round1_training_start"
    if not summary.get("round1_training_finished"):
        return "round1_training"
    if not summary.get("round1_checkpoint_saved"):
        return "round1_checkpoint"
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    controller = FalconController({"output_dir": str(OUTPUT_DIR)})
    round0 = controller.run_round(0)
    round1 = controller.run_round(1)
    controller.save_controller_state(OUTPUT_DIR / "falcon_controller_state_final.json")

    initial = round0.get("initial_training", {})
    generation = round0.get("candidate_generation", {})
    validation = round0.get("candidate_validation", {})
    policy_eval = round0.get("policy_eval", {})
    difficulty = round0.get("difficulty_results", [])
    sampling_path = round0.get("sampling_plan_path")
    train_summary = round1.get("train_summary", {})
    warnings = []
    warnings.extend(controller.state.get("warnings", []))
    warnings.extend(generation.get("warnings", []))
    warnings.extend(policy_eval.get("warnings", []))
    warnings.extend(round1.get("warnings", []))

    policy_eval_results = policy_eval.get("policy_eval_results", [])
    summary = {
        "schema_version": "falcon.controller_smoke_summary.v1",
        "round0_initial_training_success": bool(initial.get("checkpoint_saved")),
        "round0_checkpoint_path": initial.get("actor_checkpoint_path"),
        "failure_summary_generated": bool(round0.get("failure_summary")),
        "qwen_generation_success": bool(generation.get("candidates")),
        "num_candidates_generated": len(generation.get("candidates") or []),
        "num_schema_valid": sum(1 for item in validation.get("schema_validations", []) if item.get("is_valid")),
        "num_constraint_valid": sum(1 for item in validation.get("constraint_results", []) if item.get("is_valid")),
        "num_yaml_generated": len(validation.get("yaml_paths") or []),
        "num_policy_eval_success": sum(
            1
            for item in policy_eval_results
            if item.get("current_policy_eval", {}).get("real_policy_eval_available")
            and item.get("best_policy_eval", {}).get("real_policy_eval_available")
            and item.get("current_policy_eval", {}).get("episode_results")
            and item.get("best_policy_eval", {}).get("episode_results")
        ),
        "num_difficulty_evaluated": len(difficulty),
        "num_accepted_into_pool": sum(1 for item in difficulty if item.get("accepted_into_curriculum_pool")),
        "sampling_plan_generated": bool(sampling_path and Path(sampling_path).exists()),
        "round1_training_started": bool(train_summary.get("training_started")),
        "round1_training_finished": bool(train_summary.get("training_finished")),
        "round1_checkpoint_saved": bool(train_summary.get("checkpoint_saved")),
        "round1_checkpoint_path": train_summary.get("actor_checkpoint_path"),
        "used_sampling_plan": bool(round1.get("used_sampling_plan")),
        "fallback_used": bool(round1.get("fallback_used")),
        "fallback_reason": round1.get("fallback_reason"),
        "selected_fallback_scenario": round1.get("selected_fallback_scenario"),
        "failure_stage": None,
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }
    summary["failure_stage"] = _failure_stage(summary)
    _write_json(OUTPUT_DIR / "falcon_controller_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
