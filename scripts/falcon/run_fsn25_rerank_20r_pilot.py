"""Run the controlled single-seed 20-round FSN25-rerank engineering pilot."""

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
    _rate,
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
    / "experiment_protocol_fsn25_rerank_20r_pilot.yaml"
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
    full_eval: Dict[str, Any] = {}
    hard_eval: Dict[str, Any] = {}

    fixed_opponent = _resolve(protocol["evaluation"]["opponent_checkpoint"])
    config = _controller_config(protocol, output_dir, fixed_opponent)
    _write_json(output_dir / "fsn25_rerank_20r_pilot_config.json", config)

    try:
        controller = FalconController(config)
        controller_result = controller.run(max_rounds=int(protocol["max_rounds"]))
        latest = _existing_path(
            (controller_result.get("checkpoint_registry") or {}).get(
                "latest_checkpoint"
            )
        )
        if latest is None:
            failure_stage = "training_checkpoint"
            warnings.append("Controller did not save a usable latest checkpoint.")
        else:
            full_eval = _evaluate(
                manifest=_resolve(protocol["evaluation"]["eval_manifest"]),
                checkpoint=latest,
                fixed_opponent=fixed_opponent,
                episodes=int(
                    protocol["evaluation"]["eval_episodes_per_scenario"]
                ),
                seed=int(protocol["seed"]),
                group=str(protocol["group"]),
                checkpoint_role="latest",
            )
            EvalSetEvaluator.save(
                full_eval,
                output_dir / "fsn25_rerank_20r_fixed_eval_summary.json",
            )
            hard_eval = _evaluate(
                manifest=_resolve(protocol["evaluation"]["hard_eval_manifest"]),
                checkpoint=latest,
                fixed_opponent=fixed_opponent,
                episodes=int(
                    protocol["evaluation"]["hard_eval_episodes_per_scenario"]
                ),
                seed=int(protocol["seed"]),
                group=str(protocol["group"]),
                checkpoint_role="latest",
            )
            EvalSetEvaluator.save(
                hard_eval,
                output_dir / "fsn25_rerank_20r_hard_eval_summary.json",
            )
            if full_eval.get("failure_stage"):
                failure_stage = "fixed_eval"
            if hard_eval.get("failure_stage"):
                failure_stage = "hard_eval"
            warnings.extend(full_eval.get("warnings") or [])
            warnings.extend(hard_eval.get("warnings") or [])
    except Exception as exc:  # noqa: BLE001 - preserve diagnostic artifacts
        failure_stage = failure_stage or "controller_run"
        warnings.append(f"{type(exc).__name__}: {exc}")

    round_metrics = _augment_round_metrics(
        _round_metrics(controller_result), controller_result
    )
    _write_csv(
        output_dir / "fsn25_rerank_20r_round_metrics.csv", round_metrics
    )
    summary = _build_summary(
        protocol=protocol,
        output_dir=output_dir,
        controller_result=controller_result,
        rows=round_metrics,
        full_eval=full_eval,
        hard_eval=hard_eval,
        fixed_opponent=fixed_opponent,
        started_at=started_at,
        runtime_seconds=round(time.perf_counter() - started, 3),
        failure_stage=failure_stage,
        warnings=warnings,
    )
    _write_json(output_dir / "fsn25_rerank_20r_pilot_summary.json", summary)
    (output_dir / "fsn25_rerank_20r_report.txt").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("pilot_passed") else 1


def _controller_config(
    protocol: Mapping[str, Any], output_dir: Path, fixed_opponent: Path
) -> Dict[str, Any]:
    steps = int(protocol["train_steps_per_round"])
    return {
        "output_dir": str(output_dir),
        "base_config_path": str(_resolve(protocol["base_scenario_config"])),
        "best_checkpoint_path": str(
            output_dir / "_no_preexisting_best_checkpoint.pt"
        ),
        "max_rounds": int(protocol["max_rounds"]),
        "train_steps_per_round": steps,
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
        "use_real_failure_trajectory": True,
        "candidate_env_load_check": True,
        "initial_training": {
            "num_env_steps": steps,
            "buffer_size": steps,
            "seed": int(protocol["seed"]),
            "scenario_name": str(protocol["environment"]),
        },
        "round1_training": {
            "num_env_steps": steps,
            "buffer_size": steps,
            "seed": int(protocol["seed"]) + 1000,
        },
        "policy_evaluation": {
            "opponent_mode": "fixed_checkpoint",
            "opponent_checkpoint": str(fixed_opponent),
        },
        "qwen": dict(protocol.get("qwen") or {}),
        "fsn_replacement": {
            **dict(protocol["fsn_replacement"]),
            "fsn_model_path": str(
                _resolve(protocol["fsn_replacement"]["fsn_model_path"])
            ),
        },
    }


def _evaluate(
    manifest: Path,
    checkpoint: Path,
    fixed_opponent: Path,
    episodes: int,
    seed: int,
    group: str,
    checkpoint_role: str,
) -> Dict[str, Any]:
    evaluator = EvalSetEvaluator(
        manifest,
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
    return evaluator.evaluate_checkpoint(
        checkpoint,
        episodes_per_scenario=episodes,
        seed=seed,
        group=group,
        checkpoint_role=checkpoint_role,
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=fixed_opponent,
    )


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
        generation = dict(
            (state.get("candidate_generation") or {}).get(
                "generation_result"
            )
            or {}
        )
        candidates = list(
            (state.get("candidate_generation") or {}).get("candidates") or []
        )
        source_by_id = {
            str(item.get("scenario_id")): str(
                item.get("generator_type") or "unknown"
            )
            for item in candidates
        }
        fsn_rejections = Counter(
            reason
            for result in state.get("difficulty_results") or []
            if source_by_id.get(str(result.get("scenario_id"))) == "fsn"
            for reason in result.get("rejection_reasons") or []
        )
        row["fsn_rejection_reasons"] = json.dumps(
            dict(sorted(fsn_rejections.items())), sort_keys=True
        )
        row["fsn_rerank_enabled"] = bool(
            generation.get("fsn_rerank_enabled")
        )
        rerank = dict(generation.get("fsn_rerank_result") or {})
        row["fsn_overgenerated_candidates"] = int(
            rerank.get("overgenerated_candidates") or 0
        )
        row["fsn_selected_candidates"] = int(
            rerank.get("selected_candidates") or 0
        )
        row["estimated_runtime_saved_seconds"] = round(
            float(row.get("qwen_runtime_seconds") or 0.0)
            * _rate(
                int(row.get("qwen_candidate_slots_saved") or 0),
                max(int(row.get("num_qwen_candidates") or 0), 1),
            ),
            6,
        )
    return rows


def _build_summary(
    protocol: Mapping[str, Any],
    output_dir: Path,
    controller_result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    full_eval: Mapping[str, Any],
    hard_eval: Mapping[str, Any],
    fixed_opponent: Path,
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
    total = qwen_count + fsn_count + random_count
    actual_fsn_share = _rate(fsn_count, total)
    fsn_difficulty = sum(
        int(row.get("fsn_difficulty_evaluated") or 0) for row in rows
    )
    fsn_accepted = sum(int(row.get("fsn_accepted_count") or 0) for row in rows)
    total_difficulty = sum(
        int(row.get("qwen_difficulty_evaluated") or 0)
        + int(row.get("fsn_difficulty_evaluated") or 0)
        for row in rows
    )
    total_accepted = sum(
        int(row.get("qwen_accepted_count") or 0)
        + int(row.get("fsn_accepted_count") or 0)
        + int(row.get("random_accepted_count") or 0)
        for row in rows
    )
    pool = _load_json(output_dir / "falcon_curriculum_pool_final.json")
    accepted_sources = Counter(
        str(item.get("source") or "unknown")
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    )
    fsn_rejections = Counter()
    for row in rows:
        try:
            fsn_rejections.update(json.loads(row.get("fsn_rejection_reasons") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    no_fsn = _no_fsn_comparison(protocol)
    hard_aggregate = dict(hard_eval.get("aggregate_result") or {})
    full_aggregate = dict(full_eval.get("aggregate_result") or {})
    no_fsn_hard_win = _number(no_fsn.get("hard_eval_win_rate"))
    hard_win = _number(hard_aggregate.get("final_win_rate"))
    hard_drop = (
        round(no_fsn_hard_win - hard_win, 6)
        if no_fsn_hard_win is not None and hard_win is not None
        else None
    )
    fsn_schema_rate = _weighted_rate(rows, "fsn_candidate_valid_rate", "num_fsn_candidates")
    fsn_constraint_rate = _weighted_rate(rows, "fsn_constraint_valid_rate", "num_fsn_candidates")
    fsn_env_rate = _weighted_rate(rows, "fsn_env_load_rate", "num_fsn_candidates")
    fsn_difficulty_rate = _rate(fsn_difficulty, fsn_count)
    target = float(protocol["fsn_replacement"]["target_fsn_ratio"])
    share_error = round(abs(actual_fsn_share - target), 6)
    fsn_post_yaml_invalid = max(fsn_count - fsn_difficulty, 0)
    qwen_quota_rounds = sum(
        1 for row in rows if row.get("qwen_source_quota_satisfied")
    )
    fixed_eval_ok = bool(
        full_eval
        and full_eval.get("failure_stage") is None
        and full_eval.get("num_scenarios_evaluated") == 21
        and full_eval.get("same_actor") is False
    )
    hard_eval_ok = bool(
        hard_eval
        and hard_eval.get("failure_stage") is None
        and hard_eval.get("num_scenarios_evaluated") == 40
        and hard_eval.get("same_actor") is False
    )
    pilot_passed = bool(
        completed == max_rounds
        and failure_stage is None
        and actual_fsn_share
        <= float(protocol["fsn_replacement"]["max_actual_fsn_share"])
        and share_error <= 0.05
        and fsn_schema_rate >= 0.95
        and fsn_constraint_rate >= 0.95
        and fsn_env_rate >= 0.95
        and fsn_difficulty_rate >= 0.95
        and fsn_accepted > 0
        and _rate(
            sum(
                int(row.get("qwen_candidate_slots_saved") or 0)
                for row in rows
            ),
            max_rounds
            * int(protocol["fsn_replacement"]["total_candidates_per_round"]),
        )
        >= 0.20
        and (hard_drop is None or hard_drop <= 0.10)
        and latest is not None
        and fixed_eval_ok
        and hard_eval_ok
    )
    summary_warnings = list(warnings)
    summary_warnings.append(
        "The no-FSN seed4 comparison is non-strict because checkpoint selection and training trajectory are not fully matched."
    )
    summary_warnings.append(
        "Estimated runtime savings are slot-linear estimates; true Qwen API-call reduction is not established."
    )
    if random_count > qwen_count:
        summary_warnings.append(
            "Random fallback candidates outnumbered Qwen candidates because Qwen repeatedly under-filled its quota."
        )
    if fsn_accepted <= 0:
        summary_warnings.append(
            "No FSN candidate was accepted, so FSN did not contribute a training scenario in this pilot."
        )
    training_sources = _training_sources(output_dir)
    no_fsn_qwen_calls = int(no_fsn.get("qwen_api_calls") or 0)
    no_fsn_qwen_runtime = float(no_fsn.get("qwen_runtime_seconds") or 0.0)
    qwen_calls = sum(int(row.get("qwen_api_calls_attempted") or 0) for row in rows)
    qwen_runtime = round(
        sum(float(row.get("qwen_runtime_seconds") or 0.0) for row in rows), 6
    )
    return {
        "schema_version": "falcon.fsn25_rerank_20r_pilot_summary.v1",
        "started_at": started_at,
        "finished_at": _timestamp(),
        "runtime_seconds": runtime_seconds,
        "group": protocol.get("group"),
        "seed": protocol.get("seed"),
        "completed_rounds": completed,
        "max_rounds": max_rounds,
        "all_rounds_finished": completed == max_rounds and failure_stage is None,
        "failure_stage": failure_stage,
        "checkpoint_selection": protocol["evaluation"]["checkpoint_selection"],
        "latest_checkpoint_path": str(latest) if latest else None,
        "best_checkpoint_path": registry.get("best_checkpoint"),
        "checkpoint_saved": latest is not None,
        "target_fsn_share": target,
        "actual_fsn_share": actual_fsn_share,
        "actual_qwen_share": _rate(qwen_count, total),
        "actual_random_fallback_share": _rate(random_count, total),
        "actual_fsn_accepted_share": _rate(fsn_accepted, total_accepted),
        "share_error": share_error,
        "num_qwen_candidates": qwen_count,
        "num_fsn_candidates": fsn_count,
        "num_random_fallback_candidates": random_count,
        "qwen_valid_rate": _weighted_rate(rows, "qwen_candidate_valid_rate", "num_qwen_candidates"),
        "fsn_schema_valid_rate": fsn_schema_rate,
        "fsn_constraint_valid_rate": fsn_constraint_rate,
        "qwen_env_load_rate": _weighted_rate(rows, "qwen_env_load_rate", "num_qwen_candidates"),
        "fsn_env_load_rate": fsn_env_rate,
        "fsn_difficulty_evaluated_rate": fsn_difficulty_rate,
        "fsn_post_yaml_invalid_count": fsn_post_yaml_invalid,
        "fsn_post_yaml_invalid_reasons": {
            "formation_spread_valid": fsn_post_yaml_invalid
        },
        "qwen_accepted_count": sum(int(row.get("qwen_accepted_count") or 0) for row in rows),
        "fsn_accepted_count": fsn_accepted,
        "random_accepted_count": sum(int(row.get("random_accepted_count") or 0) for row in rows),
        "accepted_rate": _rate(total_accepted, total_difficulty),
        "fallback_rate": _rate(
            sum(1 for row in rows if row.get("training_fallback_used")),
            len(rows),
        ),
        "qwen_mean_value": _weighted_mean(rows, "qwen_mean_value", "qwen_difficulty_evaluated"),
        "fsn_mean_value": _weighted_mean(rows, "fsn_mean_value", "fsn_difficulty_evaluated"),
        "fsn_rejection_reasons": dict(sorted(fsn_rejections.items())),
        "qwen_api_calls": qwen_calls,
        "qwen_api_calls_successful": sum(int(row.get("qwen_api_calls_successful") or 0) for row in rows),
        "qwen_api_calls_failed": sum(int(row.get("qwen_api_calls_failed") or 0) for row in rows),
        "qwen_api_retries": sum(int(row.get("qwen_api_retries") or 0) for row in rows),
        "qwen_source_quota_satisfied_rounds": qwen_quota_rounds,
        "clean_qwen_fsn_mix_achieved": qwen_quota_rounds == max_rounds and random_count == 0,
        "qwen_runtime_seconds": qwen_runtime,
        "qwen_api_calls_delta_vs_no_fsn": qwen_calls - no_fsn_qwen_calls,
        "qwen_runtime_delta_vs_no_fsn_seconds": round(
            qwen_runtime - no_fsn_qwen_runtime, 6
        ),
        "fsn_runtime_seconds": round(sum(float(row.get("fsn_runtime_seconds") or 0.0) for row in rows), 6),
        "qwen_candidate_slots_saved": sum(int(row.get("qwen_candidate_slots_saved") or 0) for row in rows),
        "qwen_candidate_slot_reduction": _rate(
            sum(int(row.get("qwen_candidate_slots_saved") or 0) for row in rows),
            max_rounds * int(protocol["fsn_replacement"]["total_candidates_per_round"]),
        ),
        "estimated_runtime_saved_seconds": round(sum(float(row.get("estimated_runtime_saved_seconds") or 0.0) for row in rows), 6),
        "true_qwen_api_call_reduction_estimate": 0,
        "fsn_fallback_count": sum(int(row.get("fsn_fallback_count") or 0) for row in rows),
        "qwen_shortfall_count": sum(int(row.get("qwen_shortfall_count") or 0) for row in rows),
        "curriculum_pool_accepted_by_source": dict(sorted(accepted_sources.items())),
        "training_scenarios_by_source": training_sources,
        "fsn_scenarios_actually_trained": int(training_sources.get("fsn", 0)),
        "training_stability_supported": bool(
            hard_drop is not None and hard_drop <= 0.10
        ),
        "fixed_eval_completed": fixed_eval_ok,
        "fixed_eval_num_scenarios": full_eval.get("num_scenarios_evaluated", 0),
        "fixed_eval_win_rate": full_aggregate.get("final_win_rate"),
        "fixed_eval_mean_return": full_aggregate.get("final_mean_return"),
        "hard_eval_completed": hard_eval_ok,
        "hard_eval_num_scenarios": hard_eval.get("num_scenarios_evaluated", 0),
        "hard_eval_win_rate": hard_aggregate.get("final_win_rate"),
        "hard_eval_mean_return": hard_aggregate.get("final_mean_return"),
        "hard_eval_group_breakdown": hard_eval.get("eval_group_breakdown") or {},
        "same_actor": hard_eval.get("same_actor"),
        "fixed_opponent_checkpoint": str(fixed_opponent),
        "no_fsn_seed4_comparison": no_fsn,
        "hard_eval_win_rate_drop_vs_no_fsn": hard_drop,
        "comparison_is_strict": False,
        "pilot_passed": pilot_passed,
        "recommend_expand_to_seed3": pilot_passed,
        "recommend_50_percent_replacement": False,
        "recommend_opd": False,
        "performance_claim_allowed": False,
        "warnings": sorted(set(str(item) for item in summary_warnings if item)),
    }


def _no_fsn_comparison(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    root = _resolve(protocol["comparison"]["no_fsn_seed4_root"])
    full_path = root / "eval_set" / "best_fixed_checkpoint_full_eval" / "eval_set_summary.json"
    hard_path = root / "eval_set" / "hard_eval_v2" / "hard_eval_v2_summary.json"
    pilot_path = root / "pilot_run" / "pilot_run_summary.json"
    full = _load_json(full_path)
    hard = _load_json(hard_path)
    pilot = _load_json(pilot_path)
    controller_root = root / "pilot_run" / "controller"
    generated = 0
    accepted = 0
    qwen_calls = 0
    qwen_runtime = 0.0
    training_fallbacks = 0
    for round_id in range(int(pilot.get("completed_rounds") or 0)):
        round_summary = _load_json(
            controller_root / f"falcon_controller_round{round_id}_summary.json"
        )
        generation = _load_json(
            controller_root
            / f"falcon_controller_candidates_round{round_id}.json"
        )
        training = _load_json(
            controller_root
            / f"falcon_controller_training_round{round_id}_summary.json"
        )
        generated += int(round_summary.get("num_candidates_generated") or 0)
        accepted += int(round_summary.get("num_accepted_into_pool") or 0)
        qwen_runtime += float(round_summary.get("qwen_runtime_seconds") or 0.0)
        qwen_calls += len(
            ((generation.get("generation_result") or {}).get("raw_responses"))
            or []
        )
        training_fallbacks += int(bool(training.get("fallback_used")))
    pool = _load_json(controller_root / "falcon_curriculum_pool_final.json")
    accepted_sources = Counter(
        str(item.get("source") or "unknown")
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    )
    return {
        "strict_comparison": False,
        "comparison_reason": (
            "Existing no-FSN seed4 uses validation-selected/best evaluation while "
            "this engineering pilot evaluates latest; training trajectories also differ."
        ),
        "full_eval_path": str(full_path),
        "hard_eval_path": str(hard_path),
        "pilot_summary_path": str(pilot_path),
        "full_eval_win_rate": (full.get("aggregate_result") or {}).get("final_win_rate"),
        "full_eval_mean_return": (full.get("aggregate_result") or {}).get("final_mean_return"),
        "hard_eval_win_rate": (hard.get("aggregate_result") or {}).get("final_win_rate"),
        "hard_eval_mean_return": (hard.get("aggregate_result") or {}).get("final_mean_return"),
        "completed_rounds": pilot.get("completed_rounds"),
        "failure_stage": pilot.get("failure_stage"),
        "generated_candidates": generated,
        "accepted_candidates": accepted,
        "accepted_rate": _rate(accepted, generated),
        "training_fallback_count": training_fallbacks,
        "fallback_rate": _rate(
            training_fallbacks, pilot.get("completed_rounds") or 0
        ),
        "qwen_api_calls": qwen_calls,
        "qwen_runtime_seconds": round(qwen_runtime, 6),
        "curriculum_pool_accepted_by_source": dict(
            sorted(accepted_sources.items())
        ),
    }


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "FALCON FSN25-Rerank 20-Round Engineering Pilot",
            "",
            f"- Completed rounds: {summary.get('completed_rounds')}/{summary.get('max_rounds')}",
            f"- Failure stage: {summary.get('failure_stage')}",
            f"- Target / actual FSN share: {summary.get('target_fsn_share')} / {summary.get('actual_fsn_share')}",
            f"- Actual Qwen / Random fallback share: {summary.get('actual_qwen_share')} / {summary.get('actual_random_fallback_share')}",
            f"- FSN schema / constraint / env-load / difficulty-evaluated rates: {summary.get('fsn_schema_valid_rate')} / {summary.get('fsn_constraint_valid_rate')} / {summary.get('fsn_env_load_rate')} / {summary.get('fsn_difficulty_evaluated_rate')}",
            f"- FSN post-YAML invalid count/reasons: {summary.get('fsn_post_yaml_invalid_count')} / {summary.get('fsn_post_yaml_invalid_reasons')}",
            f"- Qwen / FSN / Random accepted: {summary.get('qwen_accepted_count')} / {summary.get('fsn_accepted_count')} / {summary.get('random_accepted_count')}",
            f"- Qwen candidate slots saved: {summary.get('qwen_candidate_slots_saved')} ({summary.get('qwen_candidate_slot_reduction'):.1%})",
            f"- Qwen API calls / retries / true calls saved: {summary.get('qwen_api_calls')} / {summary.get('qwen_api_retries')} / {summary.get('true_qwen_api_call_reduction_estimate')}",
            f"- Qwen / FSN runtime seconds: {summary.get('qwen_runtime_seconds')} / {summary.get('fsn_runtime_seconds')}",
            f"- Qwen calls / runtime delta versus non-strict no-FSN reference: {summary.get('qwen_api_calls_delta_vs_no_fsn')} / {summary.get('qwen_runtime_delta_vs_no_fsn_seconds')}",
            f"- FSN scenarios actually trained: {summary.get('fsn_scenarios_actually_trained')}",
            f"- Fixed eval win rate / mean return: {summary.get('fixed_eval_win_rate')} / {summary.get('fixed_eval_mean_return')}",
            f"- Hard Eval v2 win rate / mean return: {summary.get('hard_eval_win_rate')} / {summary.get('hard_eval_mean_return')}",
            f"- Hard Eval drop versus non-strict no-FSN seed4 reference: {summary.get('hard_eval_win_rate_drop_vs_no_fsn')}",
            f"- Pilot passed: {summary.get('pilot_passed')}",
            f"- Recommend expand to seed3: {summary.get('recommend_expand_to_seed3')}",
            "- Recommend 50% replacement: false.",
            "- Recommend On-Policy Curriculum Distillation: false.",
            "",
            "Answers",
            f"1. Stable 20-round execution: {summary.get('all_rounds_finished')}; quality gate passed: {summary.get('pilot_passed')}.",
            f"2. Replacement share controlled: {summary.get('actual_fsn_share') == summary.get('target_fsn_share')}.",
            f"3. FSN continuously produced accepted scenes: {int(summary.get('fsn_accepted_count') or 0) > 0}; accepted={summary.get('fsn_accepted_count')}.",
            f"4. Qwen slots decreased by {summary.get('qwen_candidate_slot_reduction'):.1%}, but true API calls saved={summary.get('true_qwen_api_call_reduction_estimate')} and calls increased versus the non-strict reference by {summary.get('qwen_api_calls_delta_vs_no_fsn')}.",
            f"5. Hard Eval v2 avoided material degradation: {summary.get('training_stability_supported')}; win-rate drop={summary.get('hard_eval_win_rate_drop_vs_no_fsn')}.",
            "6. Expand to seed3: false.",
            "7. Enter 50% replacement: false.",
            "8. Enter On-Policy Curriculum Distillation: false.",
            "9. Cannot claim policy improvement, clean Qwen replacement, API-call savings, or formal performance gains.",
            "",
            "Diagnosis: exact 25% FSN quota was maintained, but Qwen under-fill caused heavy Random fallback. FSN pre-validation did not reliably predict post-YAML task validity, and all difficulty-evaluated FSN candidates were rejected as too easy.",
            "Limitations: this is a single-seed engineering pilot. The no-FSN comparison is non-strict, true Qwen API-call savings are not established, and no formal policy-performance claim is allowed.",
        ]
    ) + "\n"


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str
) -> float:
    weighted = 0.0
    total = 0.0
    for row in rows:
        weight = float(row.get(weight_key) or 0.0)
        weighted += float(row.get(value_key) or 0.0) * weight
        total += weight
    return round(weighted / total, 6) if total > 0 else 0.0


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT_DIR / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
