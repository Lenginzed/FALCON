"""Run a 5-round repaired+hardness-v2 FSN replacement smoke.

This script is intentionally limited to a short engineering smoke. It does not
start a 20-round replacement pilot and does not make policy-performance claims.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402
from falcon.falcon_controller import FalconController  # noqa: E402
from scripts.falcon.test_fsn_replacement_training_smoke import (  # noqa: E402
    _existing_path,
    _round_metrics,
    _timestamp,
    _training_sources,
    _weighted_rate,
)


PROTOCOL_PATH = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "configs"
    / "experiment_protocol_fsn25_hardness_v2_5r_smoke.yaml"
)


def main() -> int:
    protocol = _load_yaml(PROTOCOL_PATH)
    output_dir = _resolve(protocol["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
    started = time.perf_counter()
    warnings: List[str] = []
    failure_stage: Optional[str] = None
    controller_result: Dict[str, Any] = {}
    hard_eval: Dict[str, Any] = {}

    fixed_opponent = _resolve(protocol["evaluation"]["opponent_checkpoint"])
    best_checkpoint = _formal_seed4_best_checkpoint()
    config = _controller_config(protocol, output_dir, fixed_opponent, best_checkpoint)
    _write_json(output_dir / "fsn_hardness_v2_5r_smoke_config.json", config)

    try:
        controller = FalconController(config)
        controller_result = controller.run(max_rounds=int(protocol["max_rounds"]))
        latest_checkpoint = _existing_path(
            (controller_result.get("checkpoint_registry") or {}).get(
                "latest_checkpoint"
            )
        )
        if latest_checkpoint is None:
            failure_stage = "training_checkpoint"
            warnings.append("Controller completed without a usable latest checkpoint.")
        else:
            evaluator = EvalSetEvaluator(
                _resolve(protocol["evaluation"]["hard_eval_manifest"]),
                {
                    "base_config_path": str(
                        ROOT_DIR
                        / "envs"
                        / "JSBSim"
                        / "configs"
                        / "2v2"
                        / "NoWeapon"
                        / "Selfplay.yaml"
                    )
                },
            )
            hard_eval = evaluator.evaluate_checkpoint(
                latest_checkpoint,
                episodes_per_scenario=int(
                    protocol["evaluation"]["hard_eval_episodes_per_scenario"]
                ),
                seed=int(protocol["seed"]),
                scenario_limit=int(protocol["evaluation"]["hard_eval_scenario_limit"]),
                group=str(protocol["group"]),
                checkpoint_role="latest",
                opponent_mode="fixed_checkpoint",
                opponent_checkpoint=fixed_opponent,
            )
            evaluator.save(
                hard_eval,
                output_dir / "fsn_hardness_v2_5r_small_hard_eval_summary.json",
            )
            if hard_eval.get("failure_stage"):
                failure_stage = "hard_eval"
                warnings.extend(hard_eval.get("warnings") or [])
    except Exception as exc:  # noqa: BLE001 - leave diagnostic files behind
        failure_stage = failure_stage or "controller_run"
        warnings.append(f"{type(exc).__name__}: {exc}")

    rows = _augment_round_metrics(_round_metrics(controller_result), controller_result)
    _write_csv(output_dir / "fsn_hardness_v2_5r_round_metrics.csv", rows)
    summary = _build_summary(
        protocol=protocol,
        output_dir=output_dir,
        controller_result=controller_result,
        rows=rows,
        hard_eval=hard_eval,
        fixed_opponent=fixed_opponent,
        best_checkpoint=best_checkpoint,
        started_at=started_at,
        runtime_seconds=round(time.perf_counter() - started, 3),
        failure_stage=failure_stage,
        warnings=warnings,
    )
    _write_json(output_dir / "fsn_hardness_v2_5r_smoke_summary.json", summary)
    (output_dir / "fsn_hardness_v2_5r_report.txt").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("smoke_passed") else 1


def _controller_config(
    protocol: Mapping[str, Any],
    output_dir: Path,
    fixed_opponent: Path,
    best_checkpoint: Path,
) -> Dict[str, Any]:
    steps = int(protocol["train_steps_per_round"])
    fsn_replacement = dict(protocol["fsn_replacement"])
    fsn_replacement["fsn_model_path"] = str(
        _resolve(fsn_replacement["fsn_model_path"])
    )
    fsn_replacement["fsn_surrogate_model_path"] = str(
        _resolve(fsn_replacement["fsn_surrogate_model_path"])
    )
    return {
        "output_dir": str(output_dir),
        "base_config_path": str(_resolve(protocol["base_scenario_config"])),
        "max_rounds": int(protocol["max_rounds"]),
        "train_steps_per_round": steps,
        "eval_episodes_per_round": int(protocol["eval_episodes_per_round"]),
        "policy_eval_episodes_per_candidate": int(
            protocol["policy_eval_episodes_per_candidate"]
        ),
        "qwen_candidates_per_round": int(fsn_replacement["qwen_quota"]),
        "num_candidates": int(fsn_replacement["total_candidates_per_round"]),
        "sampling_num_samples": 6,
        "save_every_round": True,
        "use_real_failure_trajectory": False,
        "candidate_env_load_check": True,
        "best_checkpoint_path": str(best_checkpoint),
        "initial_training": {
            "num_env_steps": steps,
            "buffer_size": steps,
            "seed": int(protocol["seed"]),
            "scenario_name": str(protocol["environment"]),
        },
        "round1_training": {
            "num_env_steps": steps,
            "buffer_size": steps,
            "seed": int(protocol["seed"]) + 100,
        },
        "policy_evaluation": {
            "opponent_mode": "fixed_checkpoint",
            "opponent_checkpoint": str(fixed_opponent),
        },
        "qwen": dict(protocol.get("qwen") or {}),
        "fsn_replacement": fsn_replacement,
    }


def _augment_round_metrics(
    rows: List[Dict[str, Any]], controller_result: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    states = {
        int(state.get("round_id")): state
        for state in controller_result.get("rounds") or []
        if state.get("round_id") is not None
    }
    for row in rows:
        state = states.get(int(row.get("round_id") or 0), {})
        generation_wrapper = dict(state.get("candidate_generation") or {})
        generation = dict(generation_wrapper.get("generation_result") or {})
        candidates = list(generation_wrapper.get("candidates") or [])
        source_by_id = {
            str(item.get("scenario_id")): str(item.get("generator_type") or "unknown")
            for item in candidates
        }
        fsn_rejections = Counter(
            reason
            for result in state.get("difficulty_results") or []
            if source_by_id.get(str(result.get("scenario_id"))) == "fsn"
            for reason in result.get("rejection_reasons") or []
        )
        hardness = dict(generation.get("fsn_hardness_v2_result") or {})
        row["fsn_hardness_v2_enabled"] = bool(
            generation.get("fsn_hardness_v2_enabled")
        )
        row["fsn_overgenerated_candidates"] = int(
            hardness.get("overgenerated_candidates") or 0
        )
        row["fsn_repaired_candidates"] = int(
            hardness.get("repair_success_count") or 0
        )
        row["fsn_post_yaml_valid_candidates_from_filter"] = int(
            hardness.get("post_yaml_valid_candidates") or 0
        )
        row["fsn_selected_candidates_from_filter"] = int(
            hardness.get("selected_candidates") or 0
        )
        row["fsn_post_yaml_constraint_valid_rate"] = row.get(
            "fsn_constraint_valid_rate", 0.0
        )
        row["fsn_rejection_reasons"] = json.dumps(
            dict(sorted(fsn_rejections.items())), sort_keys=True
        )
        row["checkpoint_saved"] = bool(row.get("checkpoint_saved"))
    return rows


def _build_summary(
    protocol: Mapping[str, Any],
    output_dir: Path,
    controller_result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    hard_eval: Mapping[str, Any],
    fixed_opponent: Path,
    best_checkpoint: Path,
    started_at: str,
    runtime_seconds: float,
    failure_stage: Optional[str],
    warnings: Iterable[str],
) -> Dict[str, Any]:
    max_rounds = int(protocol["max_rounds"])
    completed = int(controller_result.get("completed_rounds") or 0)
    registry = dict(controller_result.get("checkpoint_registry") or {})
    latest = _existing_path(registry.get("latest_checkpoint"))
    qwen_count = sum(int(row.get("num_qwen_candidates") or 0) for row in rows)
    fsn_count = sum(int(row.get("num_fsn_candidates") or 0) for row in rows)
    random_count = sum(
        int(row.get("random_fallback_candidate_count") or 0) for row in rows
    )
    total_candidates = qwen_count + fsn_count + random_count
    fsn_difficulty = sum(
        int(row.get("fsn_difficulty_evaluated") or 0) for row in rows
    )
    fsn_accepted = sum(int(row.get("fsn_accepted_count") or 0) for row in rows)
    qwen_accepted = sum(int(row.get("qwen_accepted_count") or 0) for row in rows)
    random_accepted = sum(
        int(row.get("random_accepted_count") or 0) for row in rows
    )
    training_sources = _training_sources(output_dir)
    fsn_trained = int(training_sources.get("fsn", 0))
    fsn_schema_rate = _weighted_rate(rows, "fsn_candidate_valid_rate", "num_fsn_candidates")
    fsn_constraint_rate = _weighted_rate(rows, "fsn_constraint_valid_rate", "num_fsn_candidates")
    fsn_env_rate = _weighted_rate(rows, "fsn_env_load_rate", "num_fsn_candidates")
    fsn_difficulty_rate = _rate(fsn_difficulty, fsn_count)
    actual_fsn_share = _rate(fsn_count, total_candidates)
    target_fsn_share = float(protocol["fsn_replacement"]["target_fsn_ratio"])
    share_error = round(abs(actual_fsn_share - target_fsn_share), 6)
    hard_aggregate = dict(hard_eval.get("aggregate_result") or {})
    small_eval_completed = bool(
        hard_eval
        and hard_eval.get("failure_stage") is None
        and hard_eval.get("num_scenarios_evaluated")
        == int(protocol["evaluation"]["hard_eval_scenario_limit"])
        and hard_eval.get("same_actor") is False
    )
    fsn_rejections = Counter()
    for row in rows:
        try:
            fsn_rejections.update(json.loads(row.get("fsn_rejection_reasons") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    summary_warnings = list(warnings)
    if fsn_accepted <= 0:
        summary_warnings.append("FSN accepted count was 0 in this 5-round smoke.")
    if fsn_trained <= 0:
        summary_warnings.append("No FSN scenario was selected for actual MAPPO smoke training.")
    if fsn_rejections.get("too_easy_for_current_policy", 0) > 0:
        summary_warnings.append("too_easy_for_current_policy remained an FSN rejection reason.")
    smoke_passed = bool(
        completed == max_rounds
        and failure_stage is None
        and fsn_constraint_rate >= 0.95
        and fsn_env_rate >= 0.95
        and fsn_difficulty_rate >= 0.95
        and fsn_accepted > 0
        and fsn_trained > 0
        and latest is not None
        and small_eval_completed
    )
    return {
        "schema_version": "falcon.fsn_hardness_v2_5r_smoke_summary.v1",
        "started_at": started_at,
        "finished_at": _timestamp(),
        "runtime_seconds": runtime_seconds,
        "group": protocol.get("group"),
        "seed": protocol.get("seed"),
        "max_rounds": max_rounds,
        "completed_rounds": completed,
        "all_rounds_finished": completed == max_rounds and failure_stage is None,
        "failure_stage": failure_stage,
        "train_steps_per_round": protocol.get("train_steps_per_round"),
        "target_fsn_share": target_fsn_share,
        "actual_fsn_share": actual_fsn_share,
        "actual_qwen_share": _rate(qwen_count, total_candidates),
        "actual_random_fallback_share": _rate(random_count, total_candidates),
        "share_error": share_error,
        "num_qwen_candidates": qwen_count,
        "num_fsn_candidates": fsn_count,
        "num_random_fallback_candidates": random_count,
        "fsn_overgenerated_candidates": sum(
            int(row.get("fsn_overgenerated_candidates") or 0) for row in rows
        ),
        "fsn_repaired_candidates": sum(
            int(row.get("fsn_repaired_candidates") or 0) for row in rows
        ),
        "fsn_post_yaml_constraint_valid_rate": fsn_constraint_rate,
        "fsn_schema_valid_rate": fsn_schema_rate,
        "fsn_env_load_rate": fsn_env_rate,
        "fsn_difficulty_evaluated_rate": fsn_difficulty_rate,
        "fsn_difficulty_evaluated": fsn_difficulty,
        "qwen_accepted_count": qwen_accepted,
        "fsn_accepted_count": fsn_accepted,
        "random_accepted_count": random_accepted,
        "fsn_rejection_reasons": dict(sorted(fsn_rejections.items())),
        "fsn_actual_trained_count": fsn_trained,
        "training_scenarios_by_source": training_sources,
        "random_fallback_count": sum(
            int(row.get("random_fallback_count") or 0) for row in rows
        ),
        "qwen_api_calls": sum(int(row.get("qwen_api_calls_attempted") or 0) for row in rows),
        "qwen_api_calls_successful": sum(int(row.get("qwen_api_calls_successful") or 0) for row in rows),
        "qwen_api_calls_failed": sum(int(row.get("qwen_api_calls_failed") or 0) for row in rows),
        "qwen_api_retries": sum(int(row.get("qwen_api_retries") or 0) for row in rows),
        "qwen_runtime_seconds": round(
            sum(float(row.get("qwen_runtime_seconds") or 0.0) for row in rows),
            6,
        ),
        "fsn_runtime_seconds": round(
            sum(float(row.get("fsn_runtime_seconds") or 0.0) for row in rows),
            6,
        ),
        "checkpoint_saved": latest is not None,
        "latest_checkpoint_path": str(latest) if latest else None,
        "best_checkpoint_path": str(best_checkpoint),
        "fixed_opponent_checkpoint": str(fixed_opponent),
        "small_eval_completed": small_eval_completed,
        "small_eval_num_scenarios": hard_eval.get("num_scenarios_evaluated", 0),
        "small_eval_win_rate": hard_aggregate.get("final_win_rate"),
        "small_eval_mean_return": hard_aggregate.get("final_mean_return"),
        "same_actor": hard_eval.get("same_actor"),
        "smoke_passed": smoke_passed,
        "recommend_10_round_smoke": bool(smoke_passed),
        "recommend_20_round_replacement": False,
        "recommend_opd": False,
        "performance_claim_allowed": False,
        "warnings": sorted(set(str(item) for item in summary_warnings if item)),
    }


def _report(summary: Mapping[str, Any]) -> str:
    too_easy = (summary.get("fsn_rejection_reasons") or {}).get(
        "too_easy_for_current_policy", 0
    )
    return "\n".join(
        [
            "FSN repaired+hardness-v2 5-round replacement smoke",
            "",
            f"- Completed rounds: {summary.get('completed_rounds')}/{summary.get('max_rounds')}",
            f"- Failure stage: {summary.get('failure_stage')}",
            f"- Target / actual FSN share: {summary.get('target_fsn_share')} / {summary.get('actual_fsn_share')}",
            f"- FSN schema / post-YAML constraint / env-load / difficulty rates: {summary.get('fsn_schema_valid_rate')} / {summary.get('fsn_post_yaml_constraint_valid_rate')} / {summary.get('fsn_env_load_rate')} / {summary.get('fsn_difficulty_evaluated_rate')}",
            f"- FSN accepted / actually trained: {summary.get('fsn_accepted_count')} / {summary.get('fsn_actual_trained_count')}",
            f"- Qwen / FSN / Random accepted: {summary.get('qwen_accepted_count')} / {summary.get('fsn_accepted_count')} / {summary.get('random_accepted_count')}",
            f"- Random fallback candidates: {summary.get('num_random_fallback_candidates')}",
            f"- Qwen API calls / runtime: {summary.get('qwen_api_calls')} / {summary.get('qwen_runtime_seconds')}s",
            f"- FSN runtime: {summary.get('fsn_runtime_seconds')}s",
            f"- Checkpoint saved: {summary.get('checkpoint_saved')}",
            f"- Small fixed-opponent hard eval completed: {summary.get('small_eval_completed')}",
            f"- Smoke passed: {summary.get('smoke_passed')}",
            "",
            "Answers",
            f"1. repaired+hardness-v2 fixed post-YAML invalid: {summary.get('fsn_post_yaml_constraint_valid_rate', 0) >= 0.95}.",
            f"2. FSN accepted=0 avoided: {int(summary.get('fsn_accepted_count') or 0) > 0}.",
            f"3. FSN scenarios actually entered training: {int(summary.get('fsn_actual_trained_count') or 0) > 0}.",
            f"4. too_easy remains a main rejection reason: {too_easy > 0}; count={too_easy}.",
            f"5. Recommend 10-round smoke: {summary.get('recommend_10_round_smoke')}.",
            "6. Recommend 20-round replacement: false.",
            "7. Recommend OPD: false.",
            "",
            "This smoke only validates engineering flow and candidate quality gates. It does not support any policy-performance claim.",
        ]
    ) + "\n"


def _formal_seed4_best_checkpoint() -> Path:
    summary_path = (
        ROOT_DIR
        / "experiments"
        / "falcon_2v2_noweapon"
        / "results"
        / "falcon_no_fsn"
        / "seed_4"
        / "pilot_run"
        / "baseline_experiment_summary.json"
    )
    data = _load_json(summary_path)
    path = _existing_path(data.get("best_checkpoint_path"))
    if path is None:
        raise FileNotFoundError("Formal seed-4 FALCON best checkpoint was not found.")
    return path


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0:
            return 0.0
        return round(float(numerator) / denominator_value, 6)
    except (TypeError, ValueError):
        return 0.0


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT_DIR / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True, default=str)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
