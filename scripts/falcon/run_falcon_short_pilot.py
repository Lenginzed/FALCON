#!/usr/bin/env python
"""Run a short FALCON long-run pilot.

This runner is for stability validation before overnight training. It reuses
the existing FalconController and does not implement FSN, baselines, multi-seed
experiments, or MAPPO algorithm changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.falcon_controller import FalconController  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT_DIR / "tests" / "tmp_falcon_short_pilot"
BASE_CONFIG_PATH = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short FALCON controller pilot.")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--train-steps-per-round", type=int, default=256)
    parser.add_argument("--eval-episodes-per-round", type=int, default=2)
    parser.add_argument("--qwen-candidates-per-round", type=int, default=2)
    parser.add_argument("--policy-eval-episodes-per-candidate", type=int, default=2)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume-from-state", default=None)
    parser.add_argument("--sampling-num-samples", type=int, default=8)
    parser.add_argument("--max-pool-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--qwen-timeout", type=float, default=180.0)
    parser.add_argument("--save-every-round", dest="save_every_round", action="store_true", default=True)
    parser.add_argument("--no-save-every-round", dest="save_every_round", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
    start_time = time.time()
    warnings: List[str] = []
    controller: Optional[FalconController] = None
    run_result: Dict[str, Any] = {
        "schema_version": "falcon.short_pilot_run_result.v1",
        "completed_rounds": 0,
        "warnings": [],
    }
    failure_stage: Optional[str] = None

    config = _build_controller_config(args, output_dir)
    _write_json(output_dir / "falcon_short_pilot_config.json", config)

    try:
        controller = FalconController(config)
        run_result = controller.run(max_rounds=int(args.max_rounds))
        warnings.extend(run_result.get("warnings", []))
        warnings.extend(controller.state.get("warnings", []))
    except Exception as exc:  # noqa: BLE001 - pilot should leave a diagnostic summary
        failure_stage = "controller_run"
        warnings.append(f"Short pilot controller run failed: {type(exc).__name__}: {exc}")
        warnings.append(traceback.format_exc())

    finished_at = _timestamp()
    runtime_seconds = round(time.time() - start_time, 3)

    if controller is None:
        summary = _empty_summary(args, started_at, finished_at, runtime_seconds, failure_stage, warnings)
        _write_json(output_dir / "falcon_short_pilot_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    round_rows = _build_round_rows(controller, args)
    _write_round_dashboard(output_dir / "falcon_short_pilot_rounds.csv", round_rows)
    summary = _build_summary(
        controller=controller,
        args=args,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=runtime_seconds,
        run_result=run_result,
        round_rows=round_rows,
        failure_stage=failure_stage,
        warnings=warnings,
    )
    _write_json(output_dir / "falcon_short_pilot_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_controller_config(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    train_steps = int(args.train_steps_per_round)
    return {
        "output_dir": str(output_dir),
        "base_config_path": str(BASE_CONFIG_PATH),
        "max_rounds": int(args.max_rounds),
        "train_steps_per_round": train_steps,
        "eval_episodes_per_round": int(args.eval_episodes_per_round),
        "qwen_candidates_per_round": int(args.qwen_candidates_per_round),
        "policy_eval_episodes_per_candidate": int(args.policy_eval_episodes_per_candidate),
        "resume_from_state": args.resume_from_state,
        "save_every_round": bool(args.save_every_round),
        "sampling_num_samples": int(args.sampling_num_samples),
        "max_pool_size": int(args.max_pool_size),
        "use_real_failure_trajectory": True,
        "initial_training": {
            "num_env_steps": train_steps,
            "buffer_size": train_steps,
            "seed": int(args.seed),
            "scenario_name": "2v2/NoWeapon/Selfplay",
        },
        "round1_training": {
            "num_env_steps": train_steps,
            "buffer_size": train_steps,
            "seed": int(args.seed) + 1000,
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
            "timeout": float(args.qwen_timeout),
            "num_retries": 2,
        },
    }


def _build_round_rows(controller: FalconController, args: argparse.Namespace) -> List[Dict[str, Any]]:
    state = controller.state
    rounds = state.get("rounds") or {}
    rows: List[Dict[str, Any]] = []
    for round_id in sorted((int(key) for key in rounds.keys() if str(key).isdigit())):
        round_state = rounds.get(str(round_id)) or {}
        generation = _mapping(round_state.get("candidate_generation"))
        validation = _mapping(round_state.get("candidate_validation"))
        policy_eval = _mapping(round_state.get("policy_eval"))
        difficulty = [item for item in round_state.get("difficulty_results", []) if isinstance(item, MappingABC)]
        pool_stats = _load_pool_stats(round_state.get("pool_path"))
        current_win_rate = _mean_win_rate(policy_eval.get("policy_eval_results"), "current_policy_eval")
        best_win_rate = _mean_win_rate(policy_eval.get("policy_eval_results"), "best_policy_eval")
        training_result = _mapping(round_state.get("training_result"))
        train_summary = _mapping(training_result.get("train_summary"))
        failure_source = _mapping(_mapping(round_state.get("failure_summary")).get("failure_source"))
        fallback_failure_used = int(str(failure_source.get("type", "")) != "real_failure_trajectory")
        current_checkpoint = policy_eval.get("current_checkpoint_path") or train_summary.get("actor_checkpoint_path")
        best_checkpoint = policy_eval.get("best_checkpoint_path") or (state.get("checkpoint_registry") or {}).get("best_checkpoint")
        rows.append(
            {
                "round_id": round_id,
                "train_steps": int(args.train_steps_per_round),
                "eval_episodes": int(args.eval_episodes_per_round),
                "qwen_candidates": len(generation.get("candidates") or []),
                "num_schema_valid": sum(1 for item in validation.get("schema_validations", []) if item.get("is_valid")),
                "num_constraint_valid": sum(1 for item in validation.get("constraint_results", []) if item.get("is_valid")),
                "num_policy_eval_success": _policy_eval_success_count(policy_eval.get("policy_eval_results")),
                "num_accepted": sum(1 for item in difficulty if item.get("accepted_into_curriculum_pool")),
                "fallback_failure_used": fallback_failure_used,
                "training_fallback_used": int(bool(training_result.get("fallback_used"))),
                "current_checkpoint": current_checkpoint,
                "best_checkpoint": best_checkpoint,
                "round_win_rate": current_win_rate,
                "best_win_rate": best_win_rate,
                "pool_size": int(pool_stats.get("total_items", len(controller.pool.get_all()))),
                "accepted_pool_size": int(pool_stats.get("accepted_items", len(controller.pool.get_accepted()))),
                "qwen_failed": int(_qwen_failed(generation)),
                "policy_eval_failures": _policy_eval_failure_count(policy_eval.get("policy_eval_results")),
                "difficulty_filter_empty": int(bool(difficulty) and not any(item.get("accepted_into_curriculum_pool") for item in difficulty)),
            }
        )
    return rows


def _build_summary(
    controller: FalconController,
    args: argparse.Namespace,
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    run_result: Mapping[str, Any],
    round_rows: Sequence[Mapping[str, Any]],
    failure_stage: Optional[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    state = controller.state
    registry = _mapping(state.get("checkpoint_registry"))
    pool_stats = controller.pool.get_stats()
    total_generated = sum(_round_generated_count(state, row) for row in round_rows)
    total_validated = sum(int(row.get("num_schema_valid", 0)) for row in round_rows)
    total_accepted = int(pool_stats.get("accepted_items", 0))
    training_runs = _training_runs(state)
    checkpoint_saved = sum(1 for item in training_runs if item.get("checkpoint_saved"))
    accepted_rate = round(total_accepted / max(total_generated, 1), 6)
    qwen_failure_count = sum(int(row.get("qwen_failed", 0)) for row in round_rows)
    policy_eval_failure_count = sum(int(row.get("policy_eval_failures", 0)) for row in round_rows)
    difficulty_filter_empty_count = sum(int(row.get("difficulty_filter_empty", 0)) for row in round_rows)
    training_fallback_used_count = sum(int(row.get("training_fallback_used", 0)) for row in round_rows)
    fallback_failure_used_count = int((_mapping(state.get("failure_collection_stats"))).get("fallback_failure_used_count", 0))

    if failure_stage is None:
        failure_stage = _failure_stage(
            completed_rounds=int(run_result.get("completed_rounds", 0)),
            max_rounds=int(args.max_rounds),
            checkpoint_saved=checkpoint_saved,
        )

    return {
        "schema_version": "falcon.short_pilot_summary.v1",
        "started_at": started_at,
        "finished_at": finished_at,
        "total_runtime_seconds": runtime_seconds,
        "max_rounds": int(args.max_rounds),
        "completed_rounds": int(run_result.get("completed_rounds", 0)),
        "all_rounds_finished": int(run_result.get("completed_rounds", 0)) >= int(args.max_rounds),
        "total_training_runs": len(training_runs),
        "total_checkpoints_saved": checkpoint_saved,
        "total_candidates_generated": total_generated,
        "total_candidates_validated": total_validated,
        "total_candidates_accepted": total_accepted,
        "accepted_rate": accepted_rate,
        "fallback_failure_used_count": fallback_failure_used_count,
        "training_fallback_used_count": training_fallback_used_count,
        "qwen_failure_count": qwen_failure_count,
        "policy_eval_failure_count": policy_eval_failure_count,
        "difficulty_filter_empty_count": difficulty_filter_empty_count,
        "latest_checkpoint_path": registry.get("latest_checkpoint") or state.get("latest_checkpoint_path"),
        "best_checkpoint_path": registry.get("best_checkpoint") or state.get("best_checkpoint_path"),
        "final_pool_size": int(pool_stats.get("total_items", 0)),
        "accepted_pool_size": int(pool_stats.get("accepted_items", 0)),
        "resume_state_path": str(Path(args.output_dir) / "falcon_controller_state_final.json"),
        "failure_stage": failure_stage,
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }


def _empty_summary(
    args: argparse.Namespace,
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    failure_stage: Optional[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.short_pilot_summary.v1",
        "started_at": started_at,
        "finished_at": finished_at,
        "total_runtime_seconds": runtime_seconds,
        "max_rounds": int(args.max_rounds),
        "completed_rounds": 0,
        "all_rounds_finished": False,
        "total_training_runs": 0,
        "total_checkpoints_saved": 0,
        "total_candidates_generated": 0,
        "total_candidates_validated": 0,
        "total_candidates_accepted": 0,
        "accepted_rate": 0.0,
        "fallback_failure_used_count": 0,
        "training_fallback_used_count": 0,
        "qwen_failure_count": 0,
        "policy_eval_failure_count": 0,
        "difficulty_filter_empty_count": 0,
        "latest_checkpoint_path": None,
        "best_checkpoint_path": None,
        "final_pool_size": 0,
        "accepted_pool_size": 0,
        "resume_state_path": str(Path(args.output_dir) / "falcon_controller_state_final.json"),
        "failure_stage": failure_stage or "controller_initialization",
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }


def _write_round_dashboard(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round_id",
        "train_steps",
        "eval_episodes",
        "qwen_candidates",
        "num_schema_valid",
        "num_constraint_valid",
        "num_policy_eval_success",
        "num_accepted",
        "fallback_failure_used",
        "training_fallback_used",
        "current_checkpoint",
        "best_checkpoint",
        "round_win_rate",
        "best_win_rate",
        "pool_size",
        "accepted_pool_size",
        "qwen_failed",
        "policy_eval_failures",
        "difficulty_filter_empty",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _training_runs(state: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    runs: List[Mapping[str, Any]] = []
    initial = state.get("initial_training")
    if isinstance(initial, MappingABC):
        runs.append(initial)
    for round_state in (_mapping(state.get("rounds"))).values():
        if not isinstance(round_state, MappingABC):
            continue
        train_summary = _mapping(_mapping(round_state.get("training_result")).get("train_summary"))
        if train_summary:
            runs.append(train_summary)
    return runs


def _round_generated_count(state: Mapping[str, Any], row: Mapping[str, Any]) -> int:
    round_id = str(row.get("round_id"))
    round_state = _mapping(_mapping(state.get("rounds")).get(round_id))
    validation = _mapping(round_state.get("candidate_validation"))
    generation = _mapping(round_state.get("candidate_generation"))
    return max(len(generation.get("candidates") or []), len(validation.get("schema_validations") or []))


def _policy_eval_success_count(policy_eval_results: Any) -> int:
    return sum(
        1
        for item in _list_of_mappings(policy_eval_results)
        if _mapping(item.get("current_policy_eval")).get("real_policy_eval_available")
        and _mapping(item.get("best_policy_eval")).get("real_policy_eval_available")
    )


def _policy_eval_failure_count(policy_eval_results: Any) -> int:
    return sum(
        1
        for item in _list_of_mappings(policy_eval_results)
        if not (
            _mapping(item.get("current_policy_eval")).get("real_policy_eval_available")
            and _mapping(item.get("best_policy_eval")).get("real_policy_eval_available")
        )
    )


def _mean_win_rate(policy_eval_results: Any, key: str) -> float:
    values = [
        _float(_mapping(item.get(key)).get("win_rate"))
        for item in _list_of_mappings(policy_eval_results)
        if _mapping(item.get(key)).get("real_policy_eval_available")
    ]
    return round(sum(values) / len(values), 6) if values else 0.0


def _qwen_failed(generation: Mapping[str, Any]) -> bool:
    health = _mapping(generation.get("health"))
    return not (
        health.get("server_reachable")
        and health.get("model_available")
        and len(generation.get("candidates") or []) > 0
    )


def _load_pool_stats(path_value: Any) -> Dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data.get("stats") or {})
    except Exception:
        return {}


def _failure_stage(completed_rounds: int, max_rounds: int, checkpoint_saved: int) -> Optional[str]:
    if completed_rounds < max_rounds:
        return "controller_rounds"
    if checkpoint_saved <= 0:
        return "checkpoint"
    return None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, MappingABC) else {}


def _list_of_mappings(value: Any) -> List[Mapping[str, Any]]:
    return [item for item in (value or []) if isinstance(item, MappingABC)]


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value != value or value in {float("inf"), float("-inf")}:
        return 0.0
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, MappingABC):
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
