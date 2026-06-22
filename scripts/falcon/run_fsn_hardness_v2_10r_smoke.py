"""Run a 10-round repaired+hardness-v2 FSN replacement smoke."""

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
from scripts.falcon.run_fsn_hardness_v2_5r_smoke import (  # noqa: E402
    _augment_round_metrics,
    _controller_config,
    _formal_seed4_best_checkpoint,
    _rate,
    _resolve,
)
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
    / "experiment_protocol_fsn25_hardness_v2_10r_smoke.yaml"
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
    fixed_eval: Dict[str, Any] = {}
    hard_eval: Dict[str, Any] = {}

    fixed_opponent = _resolve(protocol["evaluation"]["opponent_checkpoint"])
    best_checkpoint = _formal_seed4_best_checkpoint()
    config = _controller_config(protocol, output_dir, fixed_opponent, best_checkpoint)
    _write_json(output_dir / "fsn_hardness_v2_10r_smoke_config.json", config)

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
            fixed_eval = _evaluate(
                manifest=_resolve(protocol["evaluation"]["eval_manifest"]),
                checkpoint=latest_checkpoint,
                fixed_opponent=fixed_opponent,
                episodes=int(protocol["evaluation"]["eval_episodes_per_scenario"]),
                seed=int(protocol["seed"]),
                group=str(protocol["group"]),
                checkpoint_role="latest",
                scenario_limit=None,
            )
            hard_eval = _evaluate(
                manifest=_resolve(protocol["evaluation"]["hard_eval_manifest"]),
                checkpoint=latest_checkpoint,
                fixed_opponent=fixed_opponent,
                episodes=int(
                    protocol["evaluation"]["hard_eval_episodes_per_scenario"]
                ),
                seed=int(protocol["seed"]),
                group=str(protocol["group"]),
                checkpoint_role="latest",
                scenario_limit=int(protocol["evaluation"]["hard_eval_scenario_limit"]),
            )
            EvalSetEvaluator.save(
                {
                    "schema_version": "falcon.fsn_hardness_v2_10r_eval_summary.v1",
                    "fixed_21_eval": fixed_eval,
                    "hard_eval_smoke": hard_eval,
                },
                output_dir / "fsn_hardness_v2_10r_eval_summary.json",
            )
            if fixed_eval.get("failure_stage"):
                failure_stage = "fixed_eval"
                warnings.extend(fixed_eval.get("warnings") or [])
            if hard_eval.get("failure_stage"):
                failure_stage = "hard_eval"
                warnings.extend(hard_eval.get("warnings") or [])
    except Exception as exc:  # noqa: BLE001 - preserve diagnostics
        failure_stage = failure_stage or "controller_run"
        warnings.append(f"{type(exc).__name__}: {exc}")

    rows = _augment_round_metrics(_round_metrics(controller_result), controller_result)
    _write_csv(output_dir / "fsn_hardness_v2_10r_round_metrics.csv", rows)
    summary = _build_summary(
        protocol=protocol,
        output_dir=output_dir,
        controller_result=controller_result,
        rows=rows,
        fixed_eval=fixed_eval,
        hard_eval=hard_eval,
        fixed_opponent=fixed_opponent,
        best_checkpoint=best_checkpoint,
        started_at=started_at,
        runtime_seconds=round(time.perf_counter() - started, 3),
        failure_stage=failure_stage,
        warnings=warnings,
    )
    _write_json(output_dir / "fsn_hardness_v2_10r_smoke_summary.json", summary)
    (output_dir / "fsn_hardness_v2_10r_report.txt").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("smoke_passed") else 1


def _evaluate(
    manifest: Path,
    checkpoint: Path,
    fixed_opponent: Path,
    episodes: int,
    seed: int,
    group: str,
    checkpoint_role: str,
    scenario_limit: Optional[int],
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
        scenario_limit=scenario_limit,
        group=group,
        checkpoint_role=checkpoint_role,
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=fixed_opponent,
    )


def _build_summary(
    protocol: Mapping[str, Any],
    output_dir: Path,
    controller_result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    fixed_eval: Mapping[str, Any],
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
    pool_sources = _pool_accepted_by_source(output_dir)
    fsn_schema_rate = _weighted_rate(rows, "fsn_candidate_valid_rate", "num_fsn_candidates")
    fsn_constraint_rate = _weighted_rate(rows, "fsn_constraint_valid_rate", "num_fsn_candidates")
    fsn_env_rate = _weighted_rate(rows, "fsn_env_load_rate", "num_fsn_candidates")
    fsn_difficulty_rate = _rate(fsn_difficulty, fsn_count)
    actual_fsn_share = _rate(fsn_count, total_candidates)
    actual_random_share = _rate(random_count, total_candidates)
    target_fsn_share = float(protocol["fsn_replacement"]["target_fsn_ratio"])
    share_error = round(abs(actual_fsn_share - target_fsn_share), 6)
    fixed_aggregate = dict(fixed_eval.get("aggregate_result") or {})
    hard_aggregate = dict(hard_eval.get("aggregate_result") or {})
    fixed_eval_completed = bool(
        fixed_eval
        and fixed_eval.get("failure_stage") is None
        and fixed_eval.get("num_scenarios_evaluated") == 21
        and fixed_eval.get("same_actor") is False
    )
    hard_eval_completed = bool(
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
    too_easy_count = int(fsn_rejections.get("too_easy_for_current_policy", 0))
    too_easy_rate = _rate(too_easy_count, fsn_difficulty)
    summary_warnings = list(warnings)
    if too_easy_rate >= 0.80:
        summary_warnings.append("FSN too_easy rejection rate reached the warning threshold.")
    if actual_random_share >= 0.50:
        summary_warnings.append("Random fallback share reached the warning threshold.")
    if fsn_accepted <= 0:
        summary_warnings.append("FSN accepted count was 0 in this 10-round smoke.")
    if fsn_trained <= 0:
        summary_warnings.append("No FSN scenario was selected for actual MAPPO smoke training.")
    hard_win = _number(hard_aggregate.get("final_win_rate"))
    hard_eval_not_crashed = bool(hard_eval_completed and hard_win is not None)
    smoke_passed = bool(
        completed == max_rounds
        and failure_stage is None
        and actual_fsn_share <= float(protocol["fsn_replacement"]["max_actual_fsn_share"])
        and fsn_constraint_rate >= 0.95
        and fsn_env_rate >= 0.95
        and fsn_difficulty_rate >= 0.95
        and fsn_accepted > 0
        and fsn_trained > 0
        and latest is not None
        and hard_eval_not_crashed
    )
    return {
        "schema_version": "falcon.fsn_hardness_v2_10r_smoke_summary.v1",
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
        "actual_random_fallback_share": actual_random_share,
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
        "fsn_schema_valid_rate": fsn_schema_rate,
        "fsn_post_yaml_constraint_valid_rate": fsn_constraint_rate,
        "fsn_env_load_rate": fsn_env_rate,
        "fsn_difficulty_evaluated_rate": fsn_difficulty_rate,
        "fsn_difficulty_evaluated": fsn_difficulty,
        "qwen_accepted_count": qwen_accepted,
        "fsn_accepted_count": fsn_accepted,
        "random_accepted_count": random_accepted,
        "fsn_rejection_reasons": dict(sorted(fsn_rejections.items())),
        "fsn_too_easy_rejection_rate": too_easy_rate,
        "fsn_actual_trained_count": fsn_trained,
        "training_scenarios_by_source": training_sources,
        "curriculum_pool_accepted_by_source": pool_sources,
        "random_fallback_count": sum(
            int(row.get("random_fallback_count") or 0) for row in rows
        ),
        "qwen_shortfall_count": sum(
            int(row.get("qwen_shortfall_count") or 0) for row in rows
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
        "fixed_eval_completed": fixed_eval_completed,
        "fixed_eval_num_scenarios": fixed_eval.get("num_scenarios_evaluated", 0),
        "fixed_eval_win_rate": fixed_aggregate.get("final_win_rate"),
        "fixed_eval_mean_return": fixed_aggregate.get("final_mean_return"),
        "hard_eval_smoke_completed": hard_eval_completed,
        "hard_eval_smoke_num_scenarios": hard_eval.get("num_scenarios_evaluated", 0),
        "hard_eval_smoke_win_rate": hard_aggregate.get("final_win_rate"),
        "hard_eval_smoke_mean_return": hard_aggregate.get("final_mean_return"),
        "hard_eval_not_crashed": hard_eval_not_crashed,
        "same_actor": hard_eval.get("same_actor"),
        "smoke_passed": smoke_passed,
        "recommend_20_round_replacement": False,
        "recommend_opd": False,
        "performance_claim_allowed": False,
        "warnings": sorted(set(str(item) for item in summary_warnings if item)),
    }


def _pool_accepted_by_source(output_dir: Path) -> Dict[str, int]:
    pool_path = output_dir / "falcon_curriculum_pool_final.json"
    if not pool_path.exists():
        return {}
    pool = _load_json(pool_path)
    counts = Counter(
        str(item.get("source") or "unknown")
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    )
    return dict(sorted(counts.items()))


def _report(summary: Mapping[str, Any]) -> str:
    too_easy = (summary.get("fsn_rejection_reasons") or {}).get(
        "too_easy_for_current_policy", 0
    )
    return "\n".join(
        [
            "FSN repaired+hardness-v2 10-round replacement smoke",
            "",
            f"- Completed rounds: {summary.get('completed_rounds')}/{summary.get('max_rounds')}",
            f"- Failure stage: {summary.get('failure_stage')}",
            f"- Target / actual FSN share: {summary.get('target_fsn_share')} / {summary.get('actual_fsn_share')}",
            f"- Actual Qwen / Random share: {summary.get('actual_qwen_share')} / {summary.get('actual_random_fallback_share')}",
            f"- FSN schema / post-YAML constraint / env-load / difficulty rates: {summary.get('fsn_schema_valid_rate')} / {summary.get('fsn_post_yaml_constraint_valid_rate')} / {summary.get('fsn_env_load_rate')} / {summary.get('fsn_difficulty_evaluated_rate')}",
            f"- FSN accepted / actually trained: {summary.get('fsn_accepted_count')} / {summary.get('fsn_actual_trained_count')}",
            f"- Qwen / FSN / Random accepted: {summary.get('qwen_accepted_count')} / {summary.get('fsn_accepted_count')} / {summary.get('random_accepted_count')}",
            f"- FSN rejection reasons: {summary.get('fsn_rejection_reasons')}",
            f"- Qwen shortfall / Random fallback count: {summary.get('qwen_shortfall_count')} / {summary.get('random_fallback_count')}",
            f"- Qwen API calls / runtime: {summary.get('qwen_api_calls')} / {summary.get('qwen_runtime_seconds')}s",
            f"- FSN runtime: {summary.get('fsn_runtime_seconds')}s",
            f"- Checkpoint saved: {summary.get('checkpoint_saved')}",
            f"- Fixed 21-scenario eval completed: {summary.get('fixed_eval_completed')}",
            f"- Hard Eval v2 smoke completed/not crashed: {summary.get('hard_eval_smoke_completed')} / {summary.get('hard_eval_not_crashed')}",
            f"- Hard Eval v2 smoke win rate / mean return: {summary.get('hard_eval_smoke_win_rate')} / {summary.get('hard_eval_smoke_mean_return')}",
            f"- Smoke passed: {summary.get('smoke_passed')}",
            "",
            "Answers",
            f"1. Post-YAML legality held over 10 rounds: {summary.get('fsn_post_yaml_constraint_valid_rate', 0) >= 0.95}.",
            f"2. FSN continued to produce accepted scenes: {int(summary.get('fsn_accepted_count') or 0) > 0}.",
            f"3. FSN scenes entered training multiple times: {int(summary.get('fsn_actual_trained_count') or 0) > 1}.",
            f"4. too_easy remains a main rejection reason: {too_easy > 0}; count={too_easy}, rate={summary.get('fsn_too_easy_rejection_rate')}.",
            f"5. Random fallback affects interpretation: {float(summary.get('actual_random_fallback_share') or 0.0) >= 0.5}.",
            f"6. Hard Eval did not crash: {summary.get('hard_eval_not_crashed')}.",
            "7. Recommend 20-round repaired+hardness-v2 pilot: false.",
            "8. Recommend OPD: false.",
            "",
            "This is a 10-round engineering smoke. It does not support policy-performance claims.",
        ]
    ) + "\n"


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
