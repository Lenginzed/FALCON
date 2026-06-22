"""Run offline hardness-v2 FSN shadow difficulty validation without training."""

from __future__ import annotations

import copy
import csv
import json
import math
import re
import statistics
import sys
import time
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.difficulty_evaluator import DEFAULT_CONFIG as DIFFICULTY_CONFIG  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.fsn_dataset import load_jsonl  # noqa: E402
from falcon.fsn_generator import FSNScenarioGenerator  # noqa: E402
from falcon.fsn_hardness_surrogate import (  # noqa: E402
    collect_surrogate_samples,
    load_hardness_surrogate,
    train_surrogate,
)
from falcon.policy_evaluator import PolicyEvaluator  # noqa: E402
from falcon.random_scenario_generator import RandomScenarioGenerator  # noqa: E402
from falcon.scenario_adapter import (  # noqa: E402
    apply_initial_config_to_yaml,
    load_base_scenario_config,
    save_scenario_yaml,
)
from falcon.trajectory_recorder import SCENARIO_VECTOR_KEYS  # noqa: E402


BASE_CONFIG_PATH = (
    ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
)
OPPONENT_MANIFEST = (
    ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "eval_opponent.json"
)
FSN_MODEL = (
    ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "fsn" / "stage2" / "fsn_stage2_model.pt"
)
FSN_DATASET = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage2"
    / "failure_to_scenario_dataset_dedup.jsonl"
)
STAGE3_FAILURES = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage3"
    / "fsn_shadow_failure_summaries.json"
)
STAGE3_CANDIDATES = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage3"
    / "fsn_policy_evaluated_shadow_candidates.json"
)
STAGE5_CANDIDATES = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage5_repair"
    / "fsn_repair_shadow_candidates.json"
)
OUTPUT_DIR = (
    ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "fsn" / "stage6_hardness_v2"
)
REPORT_PATH = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "reports"
    / "fsn_hardness_v2_report.txt"
)
SURROGATE_PATH = OUTPUT_DIR / "fsn_hardness_surrogate_model.pt"

MODES = (
    "fsn_repaired",
    "fsn_repaired_hardness_v1",
    "fsn_repaired_hardness_v2",
    "historical_qwen",
    "random",
)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--failure-count", type=int, default=30)
    parser.add_argument("--candidates-per-mode", type=int, default=4)
    parser.add_argument("--v2-overgenerate", type=int, default=64)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base_config = load_base_scenario_config(BASE_CONFIG_PATH)
    dataset = load_jsonl(FSN_DATASET)
    surrogate_summary = _ensure_surrogate()
    surrogate = load_hardness_surrogate(SURROGATE_PATH)
    failures, failure_warnings = _collect_failure_records(target_count=args.failure_count)
    historical = _load_historical_candidates()
    opponent = _resolve(_load_json(OPPONENT_MANIFEST)["checkpoint_path"])
    checker = ConstraintChecker()
    difficulty_evaluator = DifficultyEvaluator()
    policy_evaluator = PolicyEvaluator(
        {
            "base_config_path": str(BASE_CONFIG_PATH),
            "opponent_mode": "fixed_checkpoint",
            "opponent_checkpoint": str(opponent),
            "deterministic": True,
            "device": "cpu",
        }
    )
    all_records: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    yaml_root = OUTPUT_DIR / "generated_yamls"

    for failure_index, failure_record in enumerate(failures):
        failure_summary = dict(failure_record.get("failure_summary") or {})
        pool_stats = _pool_stats(dataset, failure_record)
        generated = _generate_modes(
            failure_index,
            failure_summary,
            base_config,
            pool_stats,
            surrogate,
            historical,
            num_scenarios=args.candidates_per_mode,
            v2_overgenerate=args.v2_overgenerate,
        )
        for mode_index, mode in enumerate(MODES):
            payload = generated.get(mode) or {"candidates": [], "runtime_seconds": 0.0}
            mode_started = time.perf_counter()
            records = []
            for candidate_index, candidate in enumerate(payload["candidates"]):
                candidate = copy.deepcopy(candidate)
                candidate["scenario_id"] = (
                    f"hardness_v2_f{failure_index:02d}_{mode}_{candidate_index:02d}"
                )
                record = _materialize_candidate(
                    candidate,
                    failure_record,
                    mode,
                    failure_index,
                    candidate_index,
                    yaml_root,
                    base_config,
                    checker,
                )
                if record["post_yaml_constraint_valid"] and record["env_load_success"]:
                    _evaluate_difficulty(
                        record,
                        failure_record,
                        failure_summary,
                        pool_stats,
                        policy_evaluator,
                        difficulty_evaluator,
                        seed=990000 + failure_index * 1000 + mode_index * 100 + candidate_index,
                    )
                records.append(record)
                all_records.append(record)
            generation_rows.append(
                {
                    "failure_id": failure_record.get("failure_id"),
                    "seed": failure_record.get("seed"),
                    "round_id": failure_record.get("round_id"),
                    "mode": mode,
                    "candidate_count": len(records),
                    "generation_runtime_seconds": payload.get("runtime_seconds", 0.0),
                    "validation_and_eval_runtime_seconds": round(
                        time.perf_counter() - mode_started, 6
                    ),
                    "repair_success_count": payload.get("repair_success_count", 0),
                }
            )

    metrics = {mode: _mode_metrics(mode, all_records, generation_rows) for mode in MODES}
    judgement = _judgement(metrics)
    summary = {
        "schema_version": "falcon.fsn_hardness_v2_shadow_summary.v1",
        "num_failure_summaries": len(failures),
        "candidate_count_per_mode": {mode: metrics[mode]["candidate_count"] for mode in MODES},
        "episodes_per_policy_per_candidate": 1,
        "fixed_opponent_checkpoint": str(opponent),
        "same_actor": False,
        "mappo_training_started": False,
        "entered_training_loop": False,
        "surrogate_training_summary_path": str(
            (OUTPUT_DIR / "fsn_hardness_surrogate_training_summary.json").resolve()
        ),
        "surrogate_model_path": str(SURROGATE_PATH.resolve()),
        "surrogate_metrics": surrogate_summary.get("metrics", {}),
        "mode_metrics": metrics,
        "judgement": judgement,
        "failure_stage": None,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "warnings": failure_warnings
        + [
            "No MAPPO training, controller replacement, Qwen prompt changes, or hard-filter changes were run.",
            "Historical Qwen candidates are nearest-neighbor reused from Stage 3 records; no new Qwen calls are made.",
        ],
    }
    _write_json(OUTPUT_DIR / "fsn_hardness_v2_shadow_candidates.json", {
        "schema_version": "falcon.fsn_hardness_v2_shadow_candidates.v1",
        "candidate_records": all_records,
        "generation_rows": generation_rows,
    })
    _write_json(OUTPUT_DIR / "fsn_hardness_v2_shadow_summary.json", summary)
    _write_metrics_csv(OUTPUT_DIR / "fsn_hardness_v2_shadow_metrics.csv", metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _ensure_surrogate() -> Mapping[str, Any]:
    if SURROGATE_PATH.exists() and (
        OUTPUT_DIR / "fsn_hardness_surrogate_training_summary.json"
    ).exists():
        return _load_json(OUTPUT_DIR / "fsn_hardness_surrogate_training_summary.json")
    samples = collect_surrogate_samples(
        fsn_dataset_path=FSN_DATASET,
        candidate_record_paths=[STAGE3_CANDIDATES, STAGE5_CANDIDATES],
        failure_summary_paths=[STAGE3_FAILURES],
    )
    return train_surrogate(samples, OUTPUT_DIR)


def _collect_failure_records(target_count: int) -> tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    records: List[Dict[str, Any]] = []
    roots = sorted(
        (
            ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
        ).glob("results*/falcon_no_fsn/seed_*/pilot_run/controller")
    )
    for root in roots:
        registry_path = root / "falcon_checkpoint_registry.json"
        if not registry_path.exists():
            continue
        registry = _load_json(registry_path)
        best_checkpoint = registry.get("best_checkpoint")
        if not best_checkpoint or not Path(best_checkpoint).exists():
            continue
        for path in sorted(root.glob("falcon_controller_failure_summary_round*.json")):
            failure = _load_json(path)
            round_id = _round_id_from_path(path)
            seed = _seed_from_path(path)
            current = _current_checkpoint_for_failure(failure, registry, round_id)
            if not current or not Path(current).exists():
                continue
            records.append(
                {
                    "failure_id": f"seed{seed}_round{round_id}_{path.stem}",
                    "seed": seed,
                    "round_id": round_id,
                    "failure_summary": failure,
                    "current_checkpoint": current,
                    "best_checkpoint": best_checkpoint,
                    "source_path": str(path.resolve()),
                    "dominant_failure_mode": _dominant_failure_mode(failure),
                }
            )
    selected = _select_balanced_failures(records, target_count)
    if len(selected) < target_count:
        warnings.append(
            f"Only {len(selected)} evaluable failure summaries were selected; requested {target_count}."
        )
    mode_counts = Counter(item.get("dominant_failure_mode") for item in selected)
    if len(mode_counts) < 3:
        warnings.append(
            f"Failure-mode coverage is narrow in selected summaries: {dict(mode_counts)}."
        )
    _write_json(OUTPUT_DIR / "fsn_hardness_v2_failure_summaries.json", {
        "schema_version": "falcon.fsn_hardness_v2_failure_summaries.v1",
        "failure_summaries": selected,
        "dominant_failure_mode_counts": dict(mode_counts),
    })
    return selected, warnings


def _select_balanced_failures(
    records: Sequence[Mapping[str, Any]], target_count: int
) -> List[Dict[str, Any]]:
    by_seed: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[int(record.get("seed", -1))].append(record)
    selected: List[Dict[str, Any]] = []
    per_seed_target = max(math.ceil(target_count / max(len(by_seed), 1)), 1)
    for seed in sorted(by_seed):
        rows = sorted(by_seed[seed], key=lambda item: int(item.get("round_id", 0)))
        if not rows:
            continue
        if len(rows) <= per_seed_target:
            picks = rows
        else:
            step = (len(rows) - 1) / max(per_seed_target - 1, 1)
            indices = sorted({round(i * step) for i in range(per_seed_target)})
            picks = [rows[index] for index in indices]
        for item in picks:
            if len(selected) < target_count:
                selected.append(dict(item))
    if len(selected) < target_count:
        seen = {item["failure_id"] for item in selected}
        for item in records:
            if item["failure_id"] not in seen:
                selected.append(dict(item))
                if len(selected) >= target_count:
                    break
    return selected[:target_count]


def _generate_modes(
    failure_index: int,
    failure_summary: Mapping[str, Any],
    base_config: Mapping[str, Any],
    pool_stats: Mapping[str, Any],
    surrogate: Any,
    historical: Sequence[Mapping[str, Any]],
    num_scenarios: int = 4,
    v2_overgenerate: int = 64,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    generator = FSNScenarioGenerator(
        FSN_MODEL,
        {"seed": 16000 + failure_index, "noise_scale": 0.11},
    )
    started = time.perf_counter()
    repaired = generator.generate_repaired_from_failure_summary(
        failure_summary, base_config, num_scenarios=num_scenarios, overgenerate_count=max(16, num_scenarios * 4)
    )
    result["fsn_repaired"] = {
        "candidates": repaired,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "repair_success_count": generator.last_repair_result.get("repair_success_count", 0),
    }
    started = time.perf_counter()
    hardness_v1 = generator.generate_hardness_filtered_from_failure_summary(
        failure_summary,
        base_config,
        num_scenarios=num_scenarios,
        overgenerate_count=max(32, num_scenarios * 8),
        pool_stats=pool_stats,
    )
    result["fsn_repaired_hardness_v1"] = {
        "candidates": hardness_v1,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "repair_success_count": generator.last_hardness_result.get("repair_success_count", 0),
    }
    started = time.perf_counter()
    hardness_v2 = generator.generate_hardness_v2_from_failure_summary(
        failure_summary,
        base_config,
        surrogate=surrogate,
        num_scenarios=num_scenarios,
        overgenerate_count=max(int(v2_overgenerate), num_scenarios),
        pool_stats=pool_stats,
    )
    result["fsn_repaired_hardness_v2"] = {
        "candidates": hardness_v2,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "repair_success_count": generator.last_hardness_v2_result.get("repair_success_count", 0),
    }
    nearest = _nearest_historical(failure_summary, historical)
    for mode in ("historical_qwen", "random"):
        result[mode] = {
            "candidates": [copy.deepcopy(item) for item in nearest.get(mode, [])[:num_scenarios]],
            "runtime_seconds": 0.0,
        }
    if len(result["random"]["candidates"]) < num_scenarios:
        result["random"]["candidates"] = RandomScenarioGenerator(
            {"seed": 18000 + failure_index}
        ).generate_from_failure_summary(failure_summary, base_config, num_scenarios)
    return result


def _load_historical_candidates() -> List[Dict[str, Any]]:
    payload = _load_json(STAGE3_CANDIDATES)
    failures = {
        str(item.get("failure_id")): item
        for item in (_load_json(STAGE3_FAILURES).get("failure_summaries") or [])
    }
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in payload.get("candidate_records") or []:
        generator = str(record.get("generator_type") or "")
        if generator not in {"historical_qwen", "random"}:
            continue
        failure_id = str(record.get("failure_id"))
        entry = grouped.setdefault(
            failure_id,
            {
                "failure_id": failure_id,
                "failure_summary": (failures.get(failure_id) or {}).get("failure_summary") or {},
                "historical_qwen": [],
                "random": [],
            },
        )
        entry[generator].append(dict(record.get("candidate") or {}))
    return list(grouped.values())


def _nearest_historical(
    failure_summary: Mapping[str, Any],
    historical: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not historical:
        return {"historical_qwen": [], "random": []}
    target = failure_summary.get("failure_scores") or {}
    def distance(item: Mapping[str, Any]) -> float:
        other = (item.get("failure_summary") or {}).get("failure_scores") or {}
        return sum(
            (_number(target.get(key), 0.0) - _number(other.get(key), 0.0)) ** 2
            for key in ("coordination_failure", "target_assignment_confusion", "initial_disadvantage", "generalization_failure", "failure_severity")
        )
    return min(historical, key=distance)


def _materialize_candidate(
    candidate: Mapping[str, Any],
    failure_record: Mapping[str, Any],
    mode: str,
    failure_index: int,
    candidate_index: int,
    yaml_root: Path,
    base_config: Mapping[str, Any],
    checker: ConstraintChecker,
) -> Dict[str, Any]:
    started = time.perf_counter()
    schema = validate_candidate_schema(candidate)
    pre = checker.validate_candidate(candidate)
    yaml_config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
    yaml_config["scenario_id"] = candidate.get("scenario_id")
    yaml_path = yaml_root / mode / f"failure_{failure_index:02d}" / f"candidate_{candidate_index:02d}.yaml"
    save_scenario_yaml(yaml_config, yaml_path)
    post = checker.validate_yaml_config(
        yaml_config,
        enable_env_load_check=True,
        temp_config_name=f"hardness_v2_{mode}_{failure_index}_{candidate_index}",
    )
    env_ok = bool((post.get("physical_constraint_check") or {}).get("scenario_loadable_env_check"))
    metadata = candidate.get("metadata") or {}
    return {
        "schema_version": "falcon.fsn_hardness_v2_shadow_candidate.v1",
        "failure_id": failure_record.get("failure_id"),
        "seed": failure_record.get("seed"),
        "round_id": failure_record.get("round_id"),
        "mode": mode,
        "generator_type": candidate.get("generator_type"),
        "candidate": dict(candidate),
        "schema_valid": bool(schema.get("is_valid")),
        "pre_constraint_valid": bool(pre.get("is_valid")),
        "post_yaml_constraint_valid": bool(post.get("is_valid")),
        "post_yaml_rejection_reasons": list(post.get("rejection_reasons") or []),
        "formation_spread_valid": (post.get("task_constraint_check") or {}).get("formation_spread_valid"),
        "yaml_generated": True,
        "yaml_path": str(yaml_path.resolve()),
        "env_load_success": env_ok,
        "constraint_result": post,
        "repair_applied": bool(metadata.get("adapter_repair_applied")),
        "repair_actions": list(metadata.get("repair_actions") or []),
        "hardness_proxy": metadata.get("hardness_proxy"),
        "hardness_components": metadata.get("hardness_components") or {},
        "hardness_v2_score": metadata.get("hardness_v2_score"),
        "hardness_v2_components": metadata.get("hardness_v2_components") or {},
        "hardness_v2_surrogate_prediction": metadata.get("hardness_v2_surrogate_prediction") or {},
        "difficulty_result": {},
        "validation_runtime_seconds": round(time.perf_counter() - started, 6),
    }


def _evaluate_difficulty(
    record: Dict[str, Any],
    failure_record: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    pool_stats: Mapping[str, Any],
    policy_evaluator: PolicyEvaluator,
    difficulty_evaluator: DifficultyEvaluator,
    seed: int,
) -> None:
    current = policy_evaluator.evaluate_policy_on_scenario(
        failure_record["current_checkpoint"], record["yaml_path"], num_episodes=1, seed=seed
    )
    best = policy_evaluator.evaluate_policy_on_scenario(
        failure_record["best_checkpoint"], record["yaml_path"], num_episodes=1, seed=seed
    )
    record["current_policy_eval"] = current
    record["best_policy_eval"] = best
    if current.get("real_policy_eval_available") and best.get("real_policy_eval_available"):
        record["difficulty_result"] = difficulty_evaluator.evaluate_candidate(
            record["candidate"], current, best, pool_stats, failure_summary, record["constraint_result"]
        )


def _mode_metrics(
    mode: str,
    records: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    subset = [record for record in records if record.get("mode") == mode]
    evaluated = [record for record in subset if record.get("difficulty_result")]
    accepted = [
        record
        for record in evaluated
        if record["difficulty_result"].get("accepted_into_curriculum_pool")
    ]
    rejections = Counter(
        reason
        for record in evaluated
        for reason in record["difficulty_result"].get("rejection_reasons") or []
    )
    values = [
        _number(record["difficulty_result"].get("final_value_score"))
        for record in evaluated
        if record["difficulty_result"].get("final_value_score") is not None
    ]
    learning = [
        _number(record["difficulty_result"].get("learning_potential"))
        for record in evaluated
        if record["difficulty_result"].get("learning_potential") is not None
    ]
    actions = Counter(
        str(action.get("field") or action.get("action") or "unknown")
        for record in subset
        for action in record.get("repair_actions") or []
    )
    return {
        "mode": mode,
        "candidate_count": len(subset),
        "schema_valid_rate": _rate(sum(1 for record in subset if record.get("schema_valid")), len(subset)),
        "pre_constraint_valid_rate": _rate(sum(1 for record in subset if record.get("pre_constraint_valid")), len(subset)),
        "post_yaml_constraint_valid_rate": _rate(sum(1 for record in subset if record.get("post_yaml_constraint_valid")), len(subset)),
        "env_load_rate": _rate(sum(1 for record in subset if record.get("env_load_success")), len(subset)),
        "formation_spread_valid_failure_count": sum(1 for record in subset if record.get("formation_spread_valid") is False),
        "difficulty_evaluated_count": len(evaluated),
        "accepted_count": len(accepted),
        "accepted_rate": _rate(len(accepted), len(subset)),
        "difficulty_accepted_rate": _rate(len(accepted), len(evaluated)),
        "too_easy_rejection_rate": _rate(rejections["too_easy_for_current_policy"], len(evaluated)),
        "not_solvable_rejection_rate": _rate(rejections["not_solvable_by_historical_best_policy"], len(evaluated)),
        "rejection_reason_distribution": dict(sorted(rejections.items())),
        "mean_learning_potential": _mean(learning),
        "mean_final_value_score": _mean(values),
        "diversity": _candidate_diversity([record.get("candidate") or {} for record in subset]),
        "repair_success_rate": _rate(sum(1 for record in subset if record.get("repair_applied")), len(subset)),
        "repair_actions_distribution": dict(sorted(actions.items())),
        "runtime_seconds": round(
            sum(
                _number(row.get("generation_runtime_seconds"))
                + _number(row.get("validation_and_eval_runtime_seconds"))
                for row in generation_rows
                if row.get("mode") == mode
            ),
            6,
        ),
        "surrogate_predicted_learning_mean": _mean([
            _number((record.get("hardness_v2_surrogate_prediction") or {}).get("predicted_learning_potential"))
            for record in subset
            if (record.get("hardness_v2_surrogate_prediction") or {}).get("predicted_learning_potential") is not None
        ]),
        "surrogate_predicted_accepted_probability_mean": _mean([
            _number((record.get("hardness_v2_surrogate_prediction") or {}).get("predicted_accepted_probability"))
            for record in subset
            if (record.get("hardness_v2_surrogate_prediction") or {}).get("predicted_accepted_probability") is not None
        ]),
    }


def _judgement(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    v1 = metrics["fsn_repaired_hardness_v1"]
    v2 = metrics["fsn_repaired_hardness_v2"]
    not_solvable_delta = v2["not_solvable_rejection_rate"] - v1["not_solvable_rejection_rate"]
    diversity_delta = v2["diversity"] - v1["diversity"]
    offline_gate = bool(
        v2["post_yaml_constraint_valid_rate"] >= 0.95
        and v2["too_easy_rejection_rate"] < v1["too_easy_rejection_rate"]
        and v2["accepted_rate"] > v1["accepted_rate"]
        and (v2["mean_learning_potential"] or 0.0) > (v1["mean_learning_potential"] or 0.0)
        and diversity_delta >= -0.02
        and not_solvable_delta <= 0.10
    )
    return {
        "post_yaml_constraint_gate_passed": v2["post_yaml_constraint_valid_rate"] >= 0.95,
        "too_easy_reduction_vs_v1": round(v1["too_easy_rejection_rate"] - v2["too_easy_rejection_rate"], 6),
        "accepted_rate_delta_vs_v1": round(v2["accepted_rate"] - v1["accepted_rate"], 6),
        "learning_potential_delta_vs_v1": round((v2["mean_learning_potential"] or 0.0) - (v1["mean_learning_potential"] or 0.0), 6),
        "diversity_delta_vs_v1": round(diversity_delta, 6),
        "not_solvable_delta_vs_v1": round(not_solvable_delta, 6),
        "hardness_v2_passed_offline_gate": offline_gate,
        "recommend_5_round_replacement_smoke": offline_gate,
        "recommend_20_round_replacement": False,
        "recommend_opd": False,
        "can_use_as_opd_teacher_signal_basis": bool(
            v2["post_yaml_constraint_valid_rate"] >= 0.95
            and v2["difficulty_evaluated_count"] > 0
        ),
    }


def _pool_stats(dataset: Sequence[Mapping[str, Any]], failure_record: Mapping[str, Any]) -> Dict[str, Any]:
    seed = failure_record.get("seed")
    round_id = failure_record.get("round_id")
    return {
        "scenario_vectors": [
            item.get("candidate_scenario_vector")
            for item in dataset
            if item.get("label") == "accepted"
            and item.get("seed") == seed
            and item.get("round_id") is not None
            and round_id is not None
            and int(item["round_id"]) < int(round_id)
            and item.get("candidate_scenario_vector")
        ]
    }


def _current_checkpoint_for_failure(
    failure: Mapping[str, Any], registry: Mapping[str, Any], round_id: int
) -> Optional[str]:
    policy_eval = ((failure.get("failure_source") or {}).get("policy_eval") or {})
    for key in ("checkpoint_path", "agent_checkpoint"):
        value = policy_eval.get(key)
        if value:
            return str(value)
    checkpoints = registry.get("checkpoints") or []
    candidates = [
        item
        for item in checkpoints
        if int(item.get("round_id") or 0) == int(round_id)
        and item.get("checkpoint_path")
        and item.get("exists", True)
    ]
    if candidates:
        return str(candidates[-1]["checkpoint_path"])
    return registry.get("current_checkpoint") or registry.get("latest_checkpoint")


def _dominant_failure_mode(failure: Mapping[str, Any]) -> str:
    primary = list(failure.get("primary_failure_modes") or [])
    if primary:
        return str(primary[0])
    scores = failure.get("failure_scores") or {}
    if not scores:
        return "unknown"
    actionable = {
        key: value
        for key, value in scores.items()
        if key
        in {
            "coordination_failure",
            "target_assignment_confusion",
            "initial_disadvantage",
            "generalization_failure",
        }
    }
    if actionable:
        return max(actionable, key=lambda key: _number(actionable.get(key)))
    return max(scores, key=lambda key: _number(scores.get(key)))


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _round_id_from_path(path: Path) -> int:
    match = re.search(r"round(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _candidate_diversity(candidates: Sequence[Mapping[str, Any]]) -> float:
    if len(candidates) < 2:
        return 0.0
    scales = DIFFICULTY_CONFIG["scenario_vector_scales"]
    distances: List[float] = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left = candidates[left_index].get("scenario_vector") or {}
            right = candidates[right_index].get("scenario_vector") or {}
            parts = []
            for key in SCENARIO_VECTOR_KEYS:
                if left.get(key) is None or right.get(key) is None:
                    continue
                parts.append(((_number(left.get(key)) - _number(right.get(key))) / float(scales[key])) ** 2)
            if parts:
                distances.append(math.sqrt(sum(parts) / len(parts)))
    return round(statistics.fmean(distances), 6) if distances else 0.0


def _report(summary: Mapping[str, Any]) -> str:
    metrics = summary["mode_metrics"]
    judgement = summary["judgement"]
    lines = [
        "FALCON FSN Hardness Proxy v2 Shadow Validation",
        "",
        f"- Failure summaries: {summary['num_failure_summaries']}",
        "- No MAPPO training, replacement pilot, OPD, prompt change, or hard-filter change was run.",
        "",
        "Mode metrics:",
    ]
    for mode in MODES:
        item = metrics[mode]
        lines.append(
            f"- {mode}: post={item['post_yaml_constraint_valid_rate']}, env={item['env_load_rate']}, "
            f"accepted={item['accepted_count']}/{item['candidate_count']} ({item['accepted_rate']}), "
            f"too_easy={item['too_easy_rejection_rate']}, not_solvable={item['not_solvable_rejection_rate']}, "
            f"LP={item['mean_learning_potential']}, value={item['mean_final_value_score']}, diversity={item['diversity']}."
        )
    lines.extend(
        [
            "",
            "Answers:",
            f"1. Surrogate metrics: {summary.get('surrogate_metrics', {})}",
            f"2. Hardness-v2 accepted delta vs v1: {judgement['accepted_rate_delta_vs_v1']}; LP delta: {judgement['learning_potential_delta_vs_v1']}.",
            f"3. Too-easy reduction vs v1: {judgement['too_easy_reduction_vs_v1']}.",
            f"4. Hardness-v2 passed offline gate: {judgement['hardness_v2_passed_offline_gate']}.",
            f"5. Recommend 5-round replacement smoke: {judgement['recommend_5_round_replacement_smoke']}.",
            f"6. Recommend 20-round replacement: {judgement['recommend_20_round_replacement']}.",
            f"7. Can serve as an OPD teacher-signal basis only offline: {judgement['can_use_as_opd_teacher_signal_basis']}; OPD itself remains not recommended.",
            "",
            "Limitations: all policy checks are offline one-episode current/best rollouts with a fixed opponent. This validates candidate ranking quality, not training performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_metrics_csv(path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    rows = []
    for mode, item in metrics.items():
        row = {"mode": mode}
        row.update({key: value for key, value in item.items() if not isinstance(value, (dict, list))})
        rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _rate(numerator: Any, denominator: Any) -> float:
    denominator = _number(denominator)
    return round(_number(numerator) / denominator, 6) if denominator > 0 else 0.0


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 6) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
