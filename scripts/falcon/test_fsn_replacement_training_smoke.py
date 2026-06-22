"""Run the controlled 25% FSN replacement training smoke."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402
from falcon.falcon_controller import FalconController  # noqa: E402


PROTOCOL_PATH = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "configs"
    / "experiment_protocol_fsn_replacement_smoke.yaml"
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
    config = {
        "output_dir": str(output_dir),
        "max_rounds": int(protocol["max_rounds"]),
        "train_steps_per_round": int(protocol["train_steps_per_round"]),
        "eval_episodes_per_round": int(protocol["eval_episodes_per_round"]),
        "policy_eval_episodes_per_candidate": int(
            protocol["policy_eval_episodes_per_candidate"]
        ),
        "qwen_candidates_per_round": int(
            protocol["fsn_replacement"]["qwen_candidates_per_round"]
        ),
        "num_candidates": int(
            protocol["fsn_replacement"]["total_candidates_per_round"]
        ),
        "sampling_num_samples": 6,
        "save_every_round": True,
        "use_real_failure_trajectory": False,
        "candidate_env_load_check": True,
        "best_checkpoint_path": str(best_checkpoint),
        "initial_training": {
            "num_env_steps": int(protocol["train_steps_per_round"]),
            "buffer_size": int(protocol["train_steps_per_round"]),
            "seed": int(protocol["seed"]),
            "scenario_name": "2v2/NoWeapon/Selfplay",
        },
        "round1_training": {
            "num_env_steps": int(protocol["train_steps_per_round"]),
            "buffer_size": int(protocol["train_steps_per_round"]),
            "seed": int(protocol["seed"]) + 100,
        },
        "policy_evaluation": {
            "opponent_mode": "fixed_checkpoint",
            "opponent_checkpoint": str(fixed_opponent),
        },
        "fsn_replacement": {
            **dict(protocol["fsn_replacement"]),
            "fsn_model_path": str(
                _resolve(protocol["fsn_replacement"]["fsn_model_path"])
            ),
        },
    }
    _write_json(output_dir / "fsn_replacement_smoke_config.json", config)

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
                scenario_limit=int(
                    protocol["evaluation"]["hard_eval_scenario_limit"]
                ),
                group="falcon_fsn_25",
                checkpoint_role="latest",
                opponent_mode="fixed_checkpoint",
                opponent_checkpoint=fixed_opponent,
            )
            evaluator.save(
                hard_eval,
                output_dir / "fsn_replacement_smoke_hard_eval_summary.json",
            )
            if hard_eval.get("failure_stage"):
                failure_stage = "hard_eval"
                warnings.extend(hard_eval.get("warnings") or [])
    except Exception as exc:  # noqa: BLE001 - smoke must leave a diagnostic summary
        failure_stage = failure_stage or "controller_run"
        warnings.append(f"{type(exc).__name__}: {exc}")

    round_metrics = _round_metrics(controller_result)
    _write_csv(output_dir / "fsn_replacement_smoke_round_metrics.csv", round_metrics)
    summary = _build_summary(
        protocol=protocol,
        output_dir=output_dir,
        controller_result=controller_result,
        round_metrics=round_metrics,
        hard_eval=hard_eval,
        fixed_opponent=fixed_opponent,
        best_checkpoint=best_checkpoint,
        started_at=started_at,
        runtime_seconds=round(time.perf_counter() - started, 3),
        failure_stage=failure_stage,
        warnings=warnings,
    )
    _write_json(output_dir / "fsn_replacement_smoke_summary.json", summary)
    (output_dir / "fsn_replacement_smoke_report.txt").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("training_finished") and summary.get("smoke_eval_completed") else 1


def _round_metrics(controller_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_state in controller_result.get("rounds") or []:
        generation_wrapper = dict(round_state.get("candidate_generation") or {})
        generation = dict(generation_wrapper.get("generation_result") or {})
        legacy_generation = "qwen_candidate_slots_saved" not in generation
        candidates = list(generation_wrapper.get("candidates") or [])
        validation = dict(round_state.get("candidate_validation") or {})
        valid_candidates = list(validation.get("valid_candidates") or [])
        valid_ids = {str(item.get("scenario_id")) for item in valid_candidates}
        schema_by_id = {
            str(item.get("scenario_id")): item
            for item in validation.get("schema_validations") or []
        }
        constraint_by_id = {
            str(item.get("scenario_id")): item
            for item in validation.get("constraint_results") or []
        }
        difficulty_by_id = {
            str(item.get("scenario_id")): item
            for item in round_state.get("difficulty_results") or []
        }
        source_by_id = {
            str(item.get("scenario_id")): str(item.get("generator_type") or "unknown")
            for item in candidates
        }
        source_rows: Dict[str, Dict[str, Any]] = {}
        for source in ("qwen", "fsn", "random"):
            ids = [scenario_id for scenario_id, value in source_by_id.items() if value == source]
            constraint_valid = [
                scenario_id
                for scenario_id in ids
                if (constraint_by_id.get(scenario_id) or {}).get("is_valid")
            ]
            env_valid = [
                scenario_id
                for scenario_id in ids
                if (
                    (constraint_by_id.get(scenario_id) or {})
                    .get("physical_constraint_check", {})
                    .get("scenario_loadable_env_check")
                    is True
                )
            ]
            accepted = [
                scenario_id
                for scenario_id in ids
                if (difficulty_by_id.get(scenario_id) or {}).get(
                    "accepted_into_curriculum_pool"
                )
            ]
            values = [
                _float((difficulty_by_id.get(scenario_id) or {}).get("final_value_score"))
                for scenario_id in ids
                if scenario_id in difficulty_by_id
            ]
            source_rows[source] = {
                "count": len(ids),
                "schema_valid": sum(
                    1 for scenario_id in ids if (schema_by_id.get(scenario_id) or {}).get("is_valid")
                ),
                "constraint_valid": len(constraint_valid),
                "env_valid": len(env_valid),
                "accepted": len(accepted),
                "mean_value": round(sum(values) / len(values), 6) if values else 0.0,
                "difficulty_evaluated": sum(1 for scenario_id in ids if scenario_id in difficulty_by_id),
            }
        training = dict(round_state.get("training_result") or {})
        train_summary = dict(training.get("train_summary") or {})
        rows.append(
            {
                "round_id": round_state.get("round_id"),
                "num_qwen_candidates": source_rows["qwen"]["count"],
                "num_fsn_candidates": source_rows["fsn"]["count"],
                "qwen_candidate_valid_rate": _rate(source_rows["qwen"]["schema_valid"], source_rows["qwen"]["count"]),
                "fsn_candidate_valid_rate": _rate(source_rows["fsn"]["schema_valid"], source_rows["fsn"]["count"]),
                "qwen_constraint_valid_rate": _rate(source_rows["qwen"]["constraint_valid"], source_rows["qwen"]["count"]),
                "fsn_constraint_valid_rate": _rate(source_rows["fsn"]["constraint_valid"], source_rows["fsn"]["count"]),
                "qwen_env_load_rate": _rate(source_rows["qwen"]["env_valid"], source_rows["qwen"]["count"]),
                "fsn_env_load_rate": _rate(source_rows["fsn"]["env_valid"], source_rows["fsn"]["count"]),
                "qwen_accepted_count": source_rows["qwen"]["accepted"],
                "fsn_accepted_count": source_rows["fsn"]["accepted"],
                "random_fallback_candidate_count": source_rows["random"]["count"],
                "random_accepted_count": source_rows["random"]["accepted"],
                "random_constraint_valid_rate": _rate(
                    source_rows["random"]["constraint_valid"],
                    source_rows["random"]["count"],
                ),
                "random_env_load_rate": _rate(
                    source_rows["random"]["env_valid"],
                    source_rows["random"]["count"],
                ),
                "qwen_difficulty_evaluated": source_rows["qwen"]["difficulty_evaluated"],
                "fsn_difficulty_evaluated": source_rows["fsn"]["difficulty_evaluated"],
                "qwen_mean_value": source_rows["qwen"]["mean_value"],
                "fsn_mean_value": source_rows["fsn"]["mean_value"],
                "qwen_runtime_seconds": generation.get("qwen_runtime_seconds", 0.0),
                "fsn_runtime_seconds": generation.get("fsn_runtime_seconds", 0.0),
                "qwen_calls_saved": (
                    0
                    if legacy_generation
                    else generation.get("qwen_calls_saved", 0)
                ),
                "qwen_api_call_count": generation.get(
                    "qwen_api_call_count",
                    len(
                        (
                            generation.get("qwen_generation_result") or {}
                        ).get("raw_responses")
                        or []
                    ),
                ),
                "qwen_api_calls_attempted": generation.get(
                    "qwen_api_calls_attempted",
                    generation.get("qwen_api_call_count", 0),
                ),
                "qwen_api_calls_successful": generation.get(
                    "qwen_api_calls_successful", 0
                ),
                "qwen_api_calls_failed": generation.get(
                    "qwen_api_calls_failed", 0
                ),
                "qwen_api_retries": generation.get("qwen_api_retries", 0),
                "qwen_candidates_requested": generation.get(
                    "qwen_candidates_requested", 0
                ),
                "qwen_candidates_raw_returned": generation.get(
                    "qwen_candidates_raw_returned", 0
                ),
                "qwen_candidates_valid": generation.get(
                    "qwen_candidates_valid", source_rows["qwen"]["count"]
                ),
                "fsn_candidates_requested": generation.get(
                    "fsn_candidates_requested", 0
                ),
                "fsn_candidates_valid": generation.get(
                    "fsn_candidates_valid", source_rows["fsn"]["count"]
                ),
                "random_fallback_count": generation.get(
                    "random_fallback_count", source_rows["random"]["count"]
                ),
                "qwen_shortfall_count": generation.get(
                    "qwen_shortfall_count", 0
                ),
                "actual_fsn_candidate_share": generation.get(
                    "actual_fsn_candidate_share",
                    _rate(source_rows["fsn"]["count"], len(candidates)),
                ),
                "actual_qwen_candidate_share": generation.get(
                    "actual_qwen_candidate_share",
                    _rate(source_rows["qwen"]["count"], len(candidates)),
                ),
                "actual_fsn_accepted_share": _rate(
                    source_rows["fsn"]["accepted"],
                    source_rows["fsn"]["accepted"]
                    + source_rows["qwen"]["accepted"]
                    + source_rows["random"]["accepted"],
                ),
                "quota_satisfied": bool(
                    (generation.get("quota") or {}).get("quota_satisfied")
                ),
                "qwen_source_quota_satisfied": bool(
                    (generation.get("quota") or {}).get(
                        "qwen_source_quota_satisfied"
                    )
                ),
                "fsn_source_quota_satisfied": bool(
                    (generation.get("quota") or {}).get(
                        "fsn_source_quota_satisfied"
                    )
                ),
                "qwen_calls_saved_estimated": generation.get(
                    "qwen_calls_saved_estimated", 0
                ),
                "true_api_call_reduction_estimate": generation.get(
                    "true_api_call_reduction_estimate", 0
                ),
                "candidate_slots_saved_estimated": generation.get(
                    "candidate_slots_saved_estimated",
                    generation.get("qwen_candidate_slots_saved", 0),
                ),
                "qwen_candidate_slots_saved": generation.get(
                    "qwen_candidate_slots_saved",
                    generation.get("qwen_calls_saved", 0),
                ),
                "fsn_fallback_count": generation.get("fsn_fallback_count", 0),
                "num_valid_candidates": len(valid_ids),
                "training_started": bool(train_summary.get("training_started")),
                "training_finished": bool(train_summary.get("training_finished")),
                "checkpoint_saved": bool(train_summary.get("checkpoint_saved")),
                "training_fallback_used": bool(training.get("fallback_used")),
            }
        )
    return rows


def _build_summary(
    protocol: Mapping[str, Any],
    output_dir: Path,
    controller_result: Mapping[str, Any],
    round_metrics: List[Mapping[str, Any]],
    hard_eval: Mapping[str, Any],
    fixed_opponent: Path,
    best_checkpoint: Path,
    started_at: str,
    runtime_seconds: float,
    failure_stage: Optional[str],
    warnings: Iterable[str],
) -> Dict[str, Any]:
    pool_path = output_dir / "falcon_curriculum_pool_final.json"
    pool = _load_json(pool_path) if pool_path.exists() else {}
    accepted_sources = Counter(
        str(item.get("source") or "unknown")
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    )
    registry = dict(controller_result.get("checkpoint_registry") or {})
    checkpoints = list(registry.get("checkpoints") or [])
    qwen_count = sum(int(row.get("num_qwen_candidates") or 0) for row in round_metrics)
    fsn_count = sum(int(row.get("num_fsn_candidates") or 0) for row in round_metrics)
    actual_fsn_share = _rate(fsn_count, qwen_count + fsn_count)
    requested_fsn_share = _float(protocol["fsn_replacement"]["replacement_ratio"])
    ratio_within_tolerance = abs(actual_fsn_share - requested_fsn_share) <= 0.10
    summary_warnings = list(warnings)
    if not ratio_within_tolerance:
        summary_warnings.append(
            "Actual FSN candidate share differed from the requested replacement ratio because Qwen under-filled some batched requests."
        )
    if sum(int(row.get("qwen_calls_saved") or 0) for row in round_metrics) == 0:
        summary_warnings.append(
            "The batched Qwen request path did not reduce actual API call count; it reduced requested Qwen candidate slots only."
        )
    training_sources = _training_sources(output_dir)
    latest = _existing_path(registry.get("latest_checkpoint"))
    initial_ok = _existing_path(
        ((controller_result.get("rounds") or [{}])[0].get("policy_eval") or {}).get(
            "current_checkpoint_path"
        )
        if controller_result.get("rounds")
        else None
    )
    total_training_runs = 1 + sum(
        1 for row in round_metrics if row.get("training_started")
    )
    training_finished = (
        controller_result.get("completed_rounds") == int(protocol["max_rounds"])
        and latest is not None
    )
    hard_aggregate = dict(hard_eval.get("aggregate_result") or {})
    return {
        "schema_version": "falcon.fsn_replacement_smoke_summary.v1",
        "started_at": started_at,
        "finished_at": _timestamp(),
        "runtime_seconds": runtime_seconds,
        "group": protocol.get("group"),
        "seed": protocol.get("seed"),
        "replacement_ratio": protocol["fsn_replacement"]["replacement_ratio"],
        "max_rounds": protocol["max_rounds"],
        "completed_rounds": controller_result.get("completed_rounds", 0),
        "num_qwen_candidates": qwen_count,
        "num_fsn_candidates": fsn_count,
        "actual_fsn_candidate_share": actual_fsn_share,
        "requested_fsn_candidate_share": requested_fsn_share,
        "actual_replacement_ratio_within_tolerance": ratio_within_tolerance,
        "qwen_candidate_valid_rate": _weighted_rate(round_metrics, "qwen_candidate_valid_rate", "num_qwen_candidates"),
        "fsn_candidate_valid_rate": _weighted_rate(round_metrics, "fsn_candidate_valid_rate", "num_fsn_candidates"),
        "qwen_constraint_valid_rate": _weighted_rate(round_metrics, "qwen_constraint_valid_rate", "num_qwen_candidates"),
        "fsn_constraint_valid_rate": _weighted_rate(round_metrics, "fsn_constraint_valid_rate", "num_fsn_candidates"),
        "qwen_env_load_rate": _weighted_rate(round_metrics, "qwen_env_load_rate", "num_qwen_candidates"),
        "fsn_env_load_rate": _weighted_rate(round_metrics, "fsn_env_load_rate", "num_fsn_candidates"),
        "qwen_accepted_count": sum(int(row.get("qwen_accepted_count") or 0) for row in round_metrics),
        "fsn_accepted_count": sum(int(row.get("fsn_accepted_count") or 0) for row in round_metrics),
        "qwen_difficulty_evaluated": sum(int(row.get("qwen_difficulty_evaluated") or 0) for row in round_metrics),
        "fsn_difficulty_evaluated": sum(int(row.get("fsn_difficulty_evaluated") or 0) for row in round_metrics),
        "qwen_mean_value": _mean(round_metrics, "qwen_mean_value", nonzero_weight_key="num_qwen_candidates"),
        "fsn_mean_value": _mean(round_metrics, "fsn_mean_value", nonzero_weight_key="num_fsn_candidates"),
        "qwen_runtime_seconds": round(sum(_float(row.get("qwen_runtime_seconds")) for row in round_metrics), 6),
        "fsn_runtime_seconds": round(sum(_float(row.get("fsn_runtime_seconds")) for row in round_metrics), 6),
        "qwen_calls_saved": sum(int(row.get("qwen_calls_saved") or 0) for row in round_metrics),
        "qwen_api_call_count": sum(
            int(row.get("qwen_api_call_count") or 0) for row in round_metrics
        ),
        "qwen_candidate_slots_saved": sum(
            int(row.get("qwen_candidate_slots_saved") or 0)
            for row in round_metrics
        ),
        "fsn_fallback_count": sum(int(row.get("fsn_fallback_count") or 0) for row in round_metrics),
        "curriculum_pool_accepted_by_source": dict(sorted(accepted_sources.items())),
        "training_scenarios_by_source": training_sources,
        "fsn_scenarios_actually_trained": int(training_sources.get("fsn", 0)),
        "fsn_training_used": int(training_sources.get("fsn", 0)) > 0,
        "training_started": bool(initial_ok or checkpoints),
        "training_finished": training_finished,
        "total_training_runs": total_training_runs,
        "checkpoint_saved": latest is not None,
        "latest_checkpoint_path": str(latest) if latest else None,
        "best_checkpoint_path": str(best_checkpoint),
        "fixed_opponent_checkpoint": str(fixed_opponent),
        "same_actor": hard_eval.get("same_actor"),
        "same_checkpoint": hard_eval.get("same_checkpoint"),
        "smoke_eval_completed": bool(
            hard_eval
            and hard_eval.get("failure_stage") is None
            and hard_eval.get("num_scenarios_evaluated") == int(
                protocol["evaluation"]["hard_eval_scenario_limit"]
            )
        ),
        "smoke_eval_num_scenarios": hard_eval.get("num_scenarios_evaluated", 0),
        "smoke_eval_win_rate": hard_aggregate.get("final_win_rate"),
        "smoke_eval_mean_return": hard_aggregate.get("final_mean_return"),
        "fsn_connected_to_outer_loop": fsn_count > 0,
        "fsn_entered_difficulty_evaluator": any(
            int(row.get("fsn_difficulty_evaluated") or 0) > 0 for row in round_metrics
        ),
        "qwen_api_call_reduction_observed": sum(
            int(row.get("qwen_calls_saved") or 0) for row in round_metrics
        )
        > 0,
        "qwen_candidate_request_reduction_observed": sum(
            int(row.get("qwen_candidate_slots_saved") or 0)
            for row in round_metrics
        )
        > 0,
        "recommend_20_round_25_percent_pilot": bool(
            training_finished
            and fsn_count > 0
            and _weighted_rate(round_metrics, "fsn_env_load_rate", "num_fsn_candidates") >= 0.8
            and any(int(row.get("fsn_difficulty_evaluated") or 0) > 0 for row in round_metrics)
            and hard_eval.get("failure_stage") is None
            and ratio_within_tolerance
        ),
        "performance_claim_allowed": False,
        "failure_stage": failure_stage,
        "warnings": sorted(set(str(item) for item in summary_warnings if item)),
    }


def _report(summary: Mapping[str, Any]) -> str:
    accepted = int(summary.get("fsn_accepted_count") or 0)
    recommendation = (
        "建议进入受控 20-round、25% replacement pilot。"
        if summary.get("recommend_20_round_25_percent_pilot")
        else "暂不建议进入 20-round replacement pilot，先处理 smoke 中的失败项。"
    )
    return "\n".join(
        [
            "FALCON FSN Stage 4 Controlled Replacement Smoke",
            "",
            f"- FSN 成功接入外循环: {summary.get('fsn_connected_to_outer_loop')}",
            f"- 完成 rounds: {summary.get('completed_rounds')}/{summary.get('max_rounds')}",
            f"- Qwen / FSN candidates: {summary.get('num_qwen_candidates')} / {summary.get('num_fsn_candidates')}",
            f"- FSN schema / constraint / env-load rate: {summary.get('fsn_candidate_valid_rate')} / {summary.get('fsn_constraint_valid_rate')} / {summary.get('fsn_env_load_rate')}",
            f"- FSN difficulty-evaluated / accepted: {summary.get('fsn_difficulty_evaluated')} / {accepted}",
            f"- Qwen 候选请求槽位节省: {summary.get('qwen_candidate_slots_saved')}",
            f"- Qwen 实际 API 调用 / 调用节省: {summary.get('qwen_api_call_count')} / {summary.get('qwen_calls_saved')}",
            f"- 实际 FSN candidate share: {summary.get('actual_fsn_candidate_share')}",
            f"- FSN 场景实际参与训练次数: {summary.get('fsn_scenarios_actually_trained')}",
            f"- 训练完成并保存 checkpoint: {summary.get('training_finished')} / {summary.get('checkpoint_saved')}",
            f"- 固定对手 hard-eval smoke 完成: {summary.get('smoke_eval_completed')}",
            f"- {recommendation}",
            "",
            "限制：本结果只验证低比例替代的工程链路与候选价值。Qwen 使用批量请求，因此本 smoke 只减少了 Qwen 候选请求槽位，没有减少实际 API 调用次数；也不支持 FSN 提升策略性能或等价替代 Qwen 的结论。",
        ]
    )


def _training_sources(output_dir: Path) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for path in sorted(output_dir.glob("falcon_controller_training_round*_manifest.json")):
        try:
            manifest = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        counts[str(manifest.get("source") or "unknown")] += 1
    return dict(sorted(counts.items()))


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


def _weighted_rate(rows: Iterable[Mapping[str, Any]], rate_key: str, count_key: str) -> float:
    numerator = 0.0
    denominator = 0
    for row in rows:
        count = int(row.get(count_key) or 0)
        numerator += _float(row.get(rate_key)) * count
        denominator += count
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean(rows: Iterable[Mapping[str, Any]], key: str, nonzero_weight_key: Optional[str] = None) -> float:
    values = [
        _float(row.get(key))
        for row in rows
        if nonzero_weight_key is None or int(row.get(nonzero_weight_key) or 0) > 0
    ]
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT_DIR / path


def _existing_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    path = _resolve(value)
    return path if path.exists() else None


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
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
