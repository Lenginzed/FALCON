"""Run strict-quota FSN replacement smoke with API-call accounting."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402
from falcon.falcon_controller import FalconController  # noqa: E402
from scripts.falcon.test_fsn_replacement_training_smoke import (  # noqa: E402
    _existing_path,
    _formal_seed4_best_checkpoint,
    _load_json,
    _load_yaml,
    _rate,
    _resolve,
    _round_metrics,
    _timestamp,
    _training_sources,
    _weighted_rate,
    _write_csv,
    _write_json,
)


PROTOCOL_PATH = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "configs"
    / "experiment_protocol_fsn_replacement_controlled_smoke.yaml"
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
            protocol["fsn_replacement"]["qwen_quota"]
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
            "seed": int(protocol["seed"]) + 200,
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
    _write_json(output_dir / "fsn_replacement_controlled_config.json", config)

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
            warnings.append("Controller did not save a usable latest checkpoint.")
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
                group="falcon_fsn_25_controlled",
                checkpoint_role="latest",
                opponent_mode="fixed_checkpoint",
                opponent_checkpoint=fixed_opponent,
            )
            evaluator.save(
                hard_eval,
                output_dir / "fsn_replacement_controlled_hard_eval_summary.json",
            )
            if hard_eval.get("failure_stage"):
                failure_stage = "hard_eval"
                warnings.extend(hard_eval.get("warnings") or [])
    except Exception as exc:  # noqa: BLE001 - leave diagnostic artifacts
        failure_stage = failure_stage or "controller_run"
        warnings.append(f"{type(exc).__name__}: {exc}")

    round_metrics = _round_metrics(controller_result)
    _write_csv(
        output_dir / "fsn_replacement_controlled_round_metrics.csv",
        round_metrics,
    )
    summary = _build_summary(
        protocol,
        output_dir,
        controller_result,
        round_metrics,
        hard_eval,
        fixed_opponent,
        best_checkpoint,
        started_at,
        round(time.perf_counter() - started, 3),
        failure_stage,
        warnings,
    )
    _write_json(
        output_dir / "fsn_replacement_controlled_smoke_summary.json", summary
    )
    (output_dir / "fsn_replacement_controlled_report.txt").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("controlled_replacement_passed") else 1


def _build_summary(
    protocol: Mapping[str, Any],
    output_dir: Path,
    controller_result: Mapping[str, Any],
    rows: List[Mapping[str, Any]],
    hard_eval: Mapping[str, Any],
    fixed_opponent: Path,
    best_checkpoint: Path,
    started_at: str,
    runtime_seconds: float,
    failure_stage: Optional[str],
    warnings: List[str],
) -> Dict[str, Any]:
    target = float(protocol["fsn_replacement"]["target_fsn_ratio"])
    total_qwen = sum(int(row.get("num_qwen_candidates") or 0) for row in rows)
    total_fsn = sum(int(row.get("num_fsn_candidates") or 0) for row in rows)
    total_random = sum(
        int(row.get("random_fallback_candidate_count") or 0) for row in rows
    )
    total_candidates = total_qwen + total_fsn + total_random
    actual_fsn_share = _rate(total_fsn, total_candidates)
    share_error = round(abs(actual_fsn_share - target), 6)
    total_accepted = sum(
        int(row.get("qwen_accepted_count") or 0)
        + int(row.get("fsn_accepted_count") or 0)
        + int(row.get("random_accepted_count") or 0)
        for row in rows
    )
    fsn_accepted = sum(int(row.get("fsn_accepted_count") or 0) for row in rows)
    actual_fsn_accepted_share = _rate(fsn_accepted, total_accepted)
    registry = dict(controller_result.get("checkpoint_registry") or {})
    latest = _existing_path(registry.get("latest_checkpoint"))
    completed_rounds = int(controller_result.get("completed_rounds") or 0)
    quota_rounds = sum(1 for row in rows if row.get("quota_satisfied"))
    qwen_quota_rounds = sum(
        1 for row in rows if row.get("qwen_source_quota_satisfied")
    )
    clean_mix_achieved = qwen_quota_rounds == len(rows) == completed_rounds
    fsn_difficulty = sum(
        int(row.get("fsn_difficulty_evaluated") or 0) for row in rows
    )
    fsn_valid_rate = _weighted_rate(
        rows, "fsn_candidate_valid_rate", "num_fsn_candidates"
    )
    fsn_constraint_rate = _weighted_rate(
        rows, "fsn_constraint_valid_rate", "num_fsn_candidates"
    )
    fsn_env_rate = _weighted_rate(rows, "fsn_env_load_rate", "num_fsn_candidates")
    small_eval_completed = bool(
        hard_eval
        and hard_eval.get("failure_stage") is None
        and hard_eval.get("num_scenarios_evaluated")
        == int(protocol["evaluation"]["hard_eval_scenario_limit"])
    )
    passed = bool(
        completed_rounds == int(protocol["max_rounds"])
        and failure_stage is None
        and actual_fsn_share
        <= float(protocol["fsn_replacement"]["max_actual_fsn_share"])
        and share_error <= 0.05
        and len(rows) == int(protocol["max_rounds"])
        and all("qwen_shortfall_count" in row for row in rows)
        and all("qwen_api_calls_attempted" in row for row in rows)
        and fsn_valid_rate == 1.0
        and fsn_constraint_rate == 1.0
        and fsn_env_rate == 1.0
        and fsn_difficulty > 0
        and latest is not None
        and small_eval_completed
    )
    summary_warnings = list(warnings)
    true_api_reduction = sum(
        int(row.get("true_api_call_reduction_estimate") or 0) for row in rows
    )
    if true_api_reduction <= 0:
        summary_warnings.append(
            "No API-call reduction was estimated; controlled replacement reduced candidate slots only."
        )
    if not clean_mix_achieved:
        summary_warnings.append(
            "The strict FSN share was maintained with Random fallback, but the intended 75% Qwen / 25% FSN source mix was not achieved."
        )
    pool_path = output_dir / "falcon_curriculum_pool_final.json"
    pool = _load_json(pool_path) if pool_path.exists() else {}
    accepted_sources = Counter(
        str(item.get("source") or "unknown")
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    )
    hard_aggregate = dict(hard_eval.get("aggregate_result") or {})
    return {
        "schema_version": "falcon.fsn_replacement_controlled_smoke_summary.v1",
        "started_at": started_at,
        "finished_at": _timestamp(),
        "runtime_seconds": runtime_seconds,
        "group": protocol.get("group"),
        "seed": protocol.get("seed"),
        "completed_rounds": completed_rounds,
        "max_rounds": protocol.get("max_rounds"),
        "failure_stage": failure_stage,
        "quota_satisfied_rounds": quota_rounds,
        "all_round_quotas_satisfied": quota_rounds == len(rows) == completed_rounds,
        "target_fsn_share": target,
        "actual_fsn_share": actual_fsn_share,
        "share_error": share_error,
        "max_actual_fsn_share": protocol["fsn_replacement"][
            "max_actual_fsn_share"
        ],
        "actual_fsn_accepted_share": actual_fsn_accepted_share,
        "actual_qwen_candidate_share": _rate(total_qwen, total_candidates),
        "random_fallback_share": _rate(total_random, total_candidates),
        "num_qwen_candidates": total_qwen,
        "num_fsn_candidates": total_fsn,
        "random_fallback_count": total_random,
        "qwen_shortfall_count": sum(
            int(row.get("qwen_shortfall_count") or 0) for row in rows
        ),
        "qwen_source_quota_satisfied_rounds": qwen_quota_rounds,
        "clean_qwen_fsn_mix_achieved": clean_mix_achieved,
        "total_qwen_api_calls": sum(
            int(row.get("qwen_api_calls_attempted") or 0) for row in rows
        ),
        "qwen_api_calls_successful": sum(
            int(row.get("qwen_api_calls_successful") or 0) for row in rows
        ),
        "qwen_api_calls_failed": sum(
            int(row.get("qwen_api_calls_failed") or 0) for row in rows
        ),
        "qwen_api_retries": sum(
            int(row.get("qwen_api_retries") or 0) for row in rows
        ),
        "qwen_candidates_requested": sum(
            int(row.get("qwen_candidates_requested") or 0) for row in rows
        ),
        "qwen_candidates_raw_returned": sum(
            int(row.get("qwen_candidates_raw_returned") or 0) for row in rows
        ),
        "qwen_candidates_valid": sum(
            int(row.get("qwen_candidates_valid") or 0) for row in rows
        ),
        "fsn_candidates_requested": sum(
            int(row.get("fsn_candidates_requested") or 0) for row in rows
        ),
        "fsn_candidates_valid": sum(
            int(row.get("fsn_candidates_valid") or 0) for row in rows
        ),
        "total_qwen_runtime_seconds": round(
            sum(float(row.get("qwen_runtime_seconds") or 0.0) for row in rows), 6
        ),
        "total_fsn_runtime_seconds": round(
            sum(float(row.get("fsn_runtime_seconds") or 0.0) for row in rows), 6
        ),
        "candidate_slots_saved_estimated": sum(
            int(row.get("candidate_slots_saved_estimated") or 0) for row in rows
        ),
        "true_api_call_reduction_estimate": true_api_reduction,
        "api_call_reduction_counterfactual_measured": False,
        "fsn_schema_valid_rate": fsn_valid_rate,
        "fsn_constraint_valid_rate": fsn_constraint_rate,
        "fsn_env_load_rate": fsn_env_rate,
        "fsn_difficulty_evaluated": fsn_difficulty,
        "fsn_accepted_count": fsn_accepted,
        "curriculum_pool_accepted_by_source": dict(sorted(accepted_sources.items())),
        "training_scenarios_by_source": _training_sources(output_dir),
        "checkpoint_saved": latest is not None,
        "latest_checkpoint_path": str(latest) if latest else None,
        "best_checkpoint_path": str(best_checkpoint),
        "fixed_opponent_checkpoint": str(fixed_opponent),
        "same_actor": hard_eval.get("same_actor"),
        "same_checkpoint": hard_eval.get("same_checkpoint"),
        "small_eval_completed": small_eval_completed,
        "small_eval_num_scenarios": hard_eval.get("num_scenarios_evaluated", 0),
        "small_eval_win_rate": hard_aggregate.get("final_win_rate"),
        "small_eval_mean_return": hard_aggregate.get("final_mean_return"),
        "controlled_replacement_passed": passed,
        "recommend_20_round_25_percent_pilot": passed,
        "recommend_20_round_engineering_pilot": passed,
        "recommend_clean_replacement_performance_comparison": bool(
            passed and clean_mix_achieved
        ),
        "performance_claim_allowed": False,
        "warnings": sorted(set(str(item) for item in summary_warnings if item)),
    }


def _report(summary: Mapping[str, Any]) -> str:
    api_statement = (
        f"估算减少 {summary.get('true_api_call_reduction_estimate')} 次 Qwen API 调用。"
        if int(summary.get("true_api_call_reduction_estimate") or 0) > 0
        else "没有观察到或估算出真实 API-call 减少，只减少了 Qwen 候选槽位。"
    )
    recommendation = (
        "满足进入受控 20-round、25% replacement 工程稳定性 pilot 的门槛；尚不满足干净的 replacement 性能对比门槛。"
        if summary.get("recommend_20_round_25_percent_pilot")
        else "暂不满足 20-round replacement pilot 门槛。"
    )
    return "\n".join(
        [
            "FALCON FSN Controlled Replacement Smoke",
            "",
            f"- 完成 rounds: {summary.get('completed_rounds')}/{summary.get('max_rounds')}",
            f"- 严格 quota 满足 rounds: {summary.get('quota_satisfied_rounds')}",
            f"- 目标 / 实际 FSN share: {summary.get('target_fsn_share')} / {summary.get('actual_fsn_share')}",
            f"- share error: {summary.get('share_error')}",
            f"- Qwen shortfall / Random fallback: {summary.get('qwen_shortfall_count')} / {summary.get('random_fallback_count')}",
            f"- Qwen source quota 满足 rounds: {summary.get('qwen_source_quota_satisfied_rounds')}",
            f"- 干净 75% Qwen / 25% FSN mix: {summary.get('clean_qwen_fsn_mix_achieved')}",
            f"- Qwen API attempted / successful / failed / retries: {summary.get('total_qwen_api_calls')} / {summary.get('qwen_api_calls_successful')} / {summary.get('qwen_api_calls_failed')} / {summary.get('qwen_api_retries')}",
            f"- Qwen / FSN runtime seconds: {summary.get('total_qwen_runtime_seconds')} / {summary.get('total_fsn_runtime_seconds')}",
            f"- FSN schema / constraint / env-load: {summary.get('fsn_schema_valid_rate')} / {summary.get('fsn_constraint_valid_rate')} / {summary.get('fsn_env_load_rate')}",
            f"- FSN difficulty evaluated / accepted: {summary.get('fsn_difficulty_evaluated')} / {summary.get('fsn_accepted_count')}",
            f"- {api_statement}",
            f"- 训练 checkpoint / small eval: {summary.get('checkpoint_saved')} / {summary.get('small_eval_completed')}",
            f"- {recommendation}",
            "",
            "限制：本 smoke 只验证严格替代比例、调用计量和工程链路，不支持 FSN 提升策略性能、等价替代 Qwen 或正式节省成本的结论。",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
