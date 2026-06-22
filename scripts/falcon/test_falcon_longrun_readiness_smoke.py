#!/usr/bin/env python
"""Long-run readiness smoke for the FALCON controller.

This is not a formal experiment. It verifies that the controller can safely run
two tiny outer-loop rounds, persist state/checkpoints/pool files, and consume
FALCON-generated scenario YAMLs through the training entry's
``--scenario-config-path`` path.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from config import get_config  # noqa: E402
from falcon.falcon_controller import FalconController  # noqa: E402
from scripts.train.train_jsbsim import make_train_env, parse_args  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "tests" / "tmp_falcon_longrun_readiness"
BASE_CONFIG_PATH = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def _build_env_args(scenario_config_path: str | None = None):
    args = [
        "--env-name",
        "MultipleCombat",
        "--algorithm-name",
        "mappo",
        "--scenario-name",
        "2v2/NoWeapon/Selfplay",
        "--experiment-name",
        "falcon_longrun_readiness_env_smoke",
        "--seed",
        "5",
        "--n-rollout-threads",
        "1",
        "--n-training-threads",
        "1",
        "--user-name",
        "falcon_smoke",
    ]
    if scenario_config_path:
        args.extend(["--scenario-config-path", scenario_config_path])
    return parse_args(args, get_config())


def _smoke_make_train_env(scenario_config_path: str | None = None) -> Dict[str, Any]:
    envs = None
    result = {
        "success": False,
        "obs_shape": None,
        "share_obs_shape": None,
        "failure_stage": None,
        "warnings": [],
    }
    try:
        all_args = _build_env_args(scenario_config_path)
        envs = make_train_env(all_args)
        obs, share_obs = envs.reset()
        result["obs_shape"] = list(getattr(obs, "shape", []))
        result["share_obs_shape"] = list(getattr(share_obs, "shape", []))
        result["success"] = True
    except Exception as exc:  # noqa: BLE001
        result["failure_stage"] = "make_train_env"
        result["warnings"].append(f"make_train_env smoke failed: {type(exc).__name__}: {exc}")
        result["warnings"].append(traceback.format_exc())
    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception as exc:  # noqa: BLE001
                result["warnings"].append(f"Failed to close envs: {type(exc).__name__}: {exc}")
    return result


def _failure_stage(summary: Mapping[str, Any]) -> str | None:
    if not summary.get("original_scenario_name_still_supported"):
        return "original_scenario_name"
    if not summary.get("scenario_config_path_supported"):
        return "scenario_config_path"
    if summary.get("completed_rounds", 0) < summary.get("max_rounds", 0):
        return "controller_rounds"
    if not summary.get("resume_state_saved"):
        return "resume_state"
    if summary.get("total_training_runs", 0) <= 0:
        return "training"
    if summary.get("total_checkpoints_saved", 0) <= 0:
        return "checkpoint"
    return None


def main() -> None:
    cli = argparse.ArgumentParser(description="Run FALCON long-run readiness smoke.")
    cli.add_argument("--output-dir", default=str(OUTPUT_DIR))
    cli.add_argument("--max-rounds", type=int, default=2)
    cli.add_argument("--train-steps-per-round", type=int, default=8)
    cli.add_argument("--qwen-candidates-per-round", type=int, default=2)
    cli.add_argument("--policy-eval-episodes-per-candidate", type=int, default=1)
    cli.add_argument("--eval-episodes-per-round", type=int, default=1)
    args = cli.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_env_smoke = _smoke_make_train_env()
    external_env_smoke = _smoke_make_train_env(str(BASE_CONFIG_PATH))
    warnings = []
    warnings.extend(original_env_smoke.get("warnings", []))
    warnings.extend(external_env_smoke.get("warnings", []))

    config = {
        "output_dir": str(output_dir),
        "base_config_path": str(BASE_CONFIG_PATH),
        "max_rounds": int(args.max_rounds),
        "train_steps_per_round": int(args.train_steps_per_round),
        "eval_episodes_per_round": int(args.eval_episodes_per_round),
        "qwen_candidates_per_round": int(args.qwen_candidates_per_round),
        "policy_eval_episodes_per_candidate": int(args.policy_eval_episodes_per_candidate),
        "sampling_num_samples": 6,
        "max_pool_size": 32,
        "use_real_failure_trajectory": True,
        "initial_training": {
            "num_env_steps": int(args.train_steps_per_round),
            "buffer_size": int(args.train_steps_per_round),
            "seed": 61,
            "scenario_name": "2v2/NoWeapon/Selfplay",
        },
        "round1_training": {
            "num_env_steps": int(args.train_steps_per_round),
            "buffer_size": int(args.train_steps_per_round),
            "seed": 71,
        },
        "qwen": {
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
        },
    }
    _write_json(output_dir / "falcon_longrun_config.json", config)

    controller = FalconController(config)
    run_result = controller.run(max_rounds=int(args.max_rounds))
    warnings.extend(run_result.get("warnings", []))
    warnings.extend(controller.state.get("warnings", []))

    registry = controller.state.get("checkpoint_registry") or {}
    pool_stats = controller.pool.get_stats()
    round_states = list((controller.state.get("rounds") or {}).values())
    total_candidates_generated = sum(len((item.get("candidate_generation") or {}).get("candidates") or []) for item in round_states)
    total_candidates_validated = sum(
        sum(1 for validation in (item.get("candidate_validation") or {}).get("schema_validations", []) if validation.get("is_valid"))
        for item in round_states
    )
    total_candidates_accepted = int(pool_stats.get("accepted_items", 0))
    training_results = [controller.state.get("initial_training")] + [
        (item.get("training_result") or {}).get("train_summary")
        for item in round_states
        if item.get("training_result")
    ]
    training_results = [item for item in training_results if isinstance(item, Mapping)]
    training_fallback_results = [
        item.get("training_result")
        for item in round_states
        if isinstance(item.get("training_result"), Mapping) and item.get("training_result", {}).get("fallback_used")
    ]
    for item in training_fallback_results:
        if item.get("fallback_reason"):
            warnings.append(str(item["fallback_reason"]))
    checkpoint_entries = [item for item in registry.get("checkpoints", []) if item.get("exists")]
    resume_load_supported = False
    resume_rounds = []
    resume_pool_items = 0
    try:
        resume_controller = FalconController(
            {
                "output_dir": str(output_dir),
                "resume_from_state": str(output_dir / "falcon_controller_state_final.json"),
            }
        )
        resume_rounds = sorted(str(key) for key in (resume_controller.state.get("rounds") or {}).keys())
        resume_pool_items = len(resume_controller.pool.get_all())
        resume_load_supported = bool(resume_rounds and resume_pool_items >= 0)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Resume load check failed: {type(exc).__name__}: {exc}")

    summary = {
        "schema_version": "falcon.longrun_readiness_summary.v1",
        "max_rounds": int(args.max_rounds),
        "completed_rounds": int(run_result.get("completed_rounds", 0)),
        "all_rounds_finished": int(run_result.get("completed_rounds", 0)) == int(args.max_rounds),
        "scenario_config_path_supported": bool(external_env_smoke.get("success")),
        "original_scenario_name_still_supported": bool(original_env_smoke.get("success")),
        "real_failure_trajectory_used_count": int(
            (controller.state.get("failure_collection_stats") or {}).get("real_failure_trajectory_used_count", 0)
        ),
        "fallback_failure_used_count": int(
            (controller.state.get("failure_collection_stats") or {}).get("fallback_failure_used_count", 0)
        ),
        "total_candidates_generated": total_candidates_generated,
        "total_candidates_validated": total_candidates_validated,
        "total_candidates_accepted": total_candidates_accepted,
        "total_training_runs": sum(1 for item in training_results if item.get("training_started")),
        "total_checkpoints_saved": sum(1 for item in training_results if item.get("checkpoint_saved")) or len(checkpoint_entries),
        "training_fallback_used_count": len(training_fallback_results),
        "total_fallback_used_count": len(training_fallback_results)
        + int((controller.state.get("failure_collection_stats") or {}).get("fallback_failure_used_count", 0)),
        "best_checkpoint_path": registry.get("best_checkpoint") or controller.state.get("best_checkpoint_path"),
        "latest_checkpoint_path": registry.get("latest_checkpoint") or controller.state.get("latest_checkpoint_path"),
        "resume_state_saved": (output_dir / "falcon_controller_state_final.json").exists(),
        "resume_load_supported": resume_load_supported,
        "resume_rounds": resume_rounds,
        "resume_pool_items": resume_pool_items,
        "checkpoint_registry_saved": (output_dir / "falcon_checkpoint_registry.json").exists(),
        "final_pool_saved": (output_dir / "falcon_curriculum_pool_final.json").exists(),
        "original_env_smoke": original_env_smoke,
        "external_yaml_env_smoke": external_env_smoke,
        "failure_stage": None,
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }
    summary["failure_stage"] = _failure_stage(summary)
    _write_json(output_dir / "falcon_longrun_readiness_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
