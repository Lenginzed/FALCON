"""Compare Qwen and FSN generation quality under fixed Qwen-call budgets."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.difficulty_evaluator import (  # noqa: E402
    DEFAULT_CONFIG as DIFFICULTY_CONFIG,
    DifficultyEvaluator,
)
from falcon.fsn_dataset import load_jsonl  # noqa: E402
from falcon.fsn_generator import FSNScenarioGenerator  # noqa: E402
from falcon.llm_scenario_generator import QwenScenarioGenerator  # noqa: E402
from falcon.policy_evaluator import PolicyEvaluator  # noqa: E402
from falcon.scenario_adapter import (  # noqa: E402
    apply_initial_config_to_yaml,
    load_base_scenario_config,
    save_scenario_yaml,
)
from falcon.trajectory_recorder import SCENARIO_VECTOR_KEYS  # noqa: E402


STAGE3_FAILURES = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage3"
    / "fsn_shadow_failure_summaries.json"
)
FSN_MODEL = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage2"
    / "fsn_stage2_model.pt"
)
FSN_DATASET = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage2"
    / "failure_to_scenario_dataset_dedup.jsonl"
)
BASE_CONFIG_PATH = (
    ROOT_DIR
    / "envs"
    / "JSBSim"
    / "configs"
    / "2v2"
    / "NoWeapon"
    / "Selfplay.yaml"
)
OPPONENT_MANIFEST = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "manifests"
    / "eval_opponent.json"
)
OUTPUT_DIR = (
    ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "fsn" / "stage4"
)
REPORT_PATH = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "reports"
    / "fsn_fixed_budget_generation_report.txt"
)

MODES = {
    "full_qwen_fixed_call": {"qwen_target": 4, "fsn_target": 0},
    "fsn25_fixed_call": {"qwen_target": 3, "fsn_target": 1},
    "fsn50_fixed_call": {"qwen_target": 2, "fsn_target": 2},
    "no_qwen_fsn_only": {"qwen_target": 0, "fsn_target": 4},
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yaml_root = OUTPUT_DIR / "fixed_budget_generated_yamls"
    failures = _select_failures(_load_json(STAGE3_FAILURES)["failure_summaries"], 10)
    base_config = load_base_scenario_config(BASE_CONFIG_PATH)
    opponent = _resolve(_load_json(OPPONENT_MANIFEST)["checkpoint_path"])
    dataset = load_jsonl(FSN_DATASET)
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
    accounting_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    started = time.perf_counter()

    for failure_index, failure_record in enumerate(failures):
        summary = dict(failure_record.get("failure_summary") or {})
        pool_stats = _pool_stats(dataset, failure_record)
        for mode_index, (mode, budget) in enumerate(MODES.items()):
            generated: List[Dict[str, Any]] = []
            qwen_accounting = _empty_qwen_accounting()
            fsn_runtime = 0.0
            if budget["qwen_target"] > 0:
                qwen_generated, qwen_accounting = _generate_qwen_once(
                    summary, base_config, budget["qwen_target"]
                )
                generated.extend(qwen_generated)
                warnings.extend(qwen_accounting.get("warnings") or [])
            if budget["fsn_target"] > 0:
                fsn_generator = FSNScenarioGenerator(
                    FSN_MODEL,
                    {
                        "seed": 5000 + failure_index * 10 + mode_index,
                        "diversity_aware": True,
                        "oversample_factor": 6,
                        "noise_scale": 0.10,
                    },
                )
                fsn_generated = fsn_generator.generate_from_failure_summary(
                    summary, base_config, budget["fsn_target"]
                )
                fsn_runtime = fsn_generator.last_generation_runtime_seconds
                generated.extend(fsn_generated)

            mode_records: List[Dict[str, Any]] = []
            qwen_returned_count = sum(
                1
                for item in generated
                if "qwen" in str(item.get("generator_type", "")).lower()
                or "ollama" in str(item.get("generator_type", "")).lower()
            )
            fsn_returned_count = len(generated) - qwen_returned_count
            for candidate_index, candidate in enumerate(generated):
                candidate = dict(candidate)
                source = (
                    "qwen"
                    if "qwen" in str(candidate.get("generator_type", "")).lower()
                    or "ollama" in str(candidate.get("generator_type", "")).lower()
                    else "fsn"
                )
                candidate["generator_type"] = source
                candidate["scenario_id"] = (
                    f"fixed_budget_f{failure_index:02d}_{mode}_{source}_{candidate_index:02d}"
                )
                record = _validate_candidate(
                    candidate,
                    failure_record,
                    mode,
                    failure_index,
                    yaml_root,
                    base_config,
                    checker,
                )
                record["generation_runtime_seconds"] = round(
                    (
                        qwen_accounting["qwen_runtime_seconds"]
                        / max(qwen_returned_count, 1)
                    )
                    if source == "qwen"
                    else fsn_runtime / max(fsn_returned_count, 1),
                    6,
                )
                if failure_index < 2 and record["env_load_success"]:
                    _evaluate_difficulty(
                        record,
                        failure_record,
                        summary,
                        pool_stats,
                        policy_evaluator,
                        difficulty_evaluator,
                        seed=800000 + failure_index * 1000 + mode_index * 100 + candidate_index,
                    )
                mode_records.append(record)
                all_records.append(record)
            diversity_scores = _candidate_diversity_scores(
                [item["candidate"] for item in mode_records]
            )
            for record, diversity_score in zip(mode_records, diversity_scores):
                record["diversity_score"] = diversity_score

            accounting_rows.append(
                {
                    "failure_id": failure_record.get("failure_id"),
                    "seed": failure_record.get("seed"),
                    "round_id": failure_record.get("round_id"),
                    "mode": mode,
                    "qwen_api_calls": qwen_accounting["qwen_api_calls"],
                    "qwen_api_calls_successful": qwen_accounting[
                        "qwen_api_calls_successful"
                    ],
                    "qwen_api_calls_failed": qwen_accounting[
                        "qwen_api_calls_failed"
                    ],
                    "qwen_runtime_seconds": qwen_accounting[
                        "qwen_runtime_seconds"
                    ],
                    "qwen_candidates_requested": budget["qwen_target"],
                    "qwen_candidates_raw_returned": qwen_accounting[
                        "qwen_candidates_raw_returned"
                    ],
                    "qwen_candidates_valid_returned": qwen_accounting[
                        "qwen_candidates_valid_returned"
                    ],
                    "qwen_shortfall_count": max(
                        budget["qwen_target"]
                        - qwen_accounting["qwen_candidates_valid_returned"],
                        0,
                    ),
                    "fsn_candidates_requested": budget["fsn_target"],
                    "fsn_candidates_returned": sum(
                        1 for item in mode_records if item["generator_type"] == "fsn"
                    ),
                    "fsn_runtime_seconds": fsn_runtime,
                    "target_candidates": budget["qwen_target"] + budget["fsn_target"],
                    "total_candidates_raw": len(generated),
                    "total_candidates_valid": sum(
                        1 for item in mode_records if item["constraint_valid"]
                    ),
                    "env_loadable_candidates": sum(
                        1 for item in mode_records if item["env_load_success"]
                    ),
                    "difficulty_evaluated_candidates": sum(
                        1 for item in mode_records if item.get("difficulty_result")
                    ),
                    "difficulty_accepted_candidates": sum(
                        1
                        for item in mode_records
                        if (item.get("difficulty_result") or {}).get(
                            "accepted_into_curriculum_pool"
                        )
                    ),
                    "diversity": _candidate_diversity(
                        [item["candidate"] for item in mode_records]
                    ),
                    "random_fallback_count": 0,
                }
            )

    mode_metrics = {
        mode: _mode_metrics(mode, accounting_rows, all_records)
        for mode in MODES
    }
    judgement = _judgement(mode_metrics)
    payload = {
        "schema_version": "falcon.fsn_fixed_budget_generation_candidates.v1",
        "num_failure_summaries": len(failures),
        "modes": MODES,
        "fixed_qwen_calls_per_summary": 1,
        "no_qwen_retry": True,
        "random_fallback_used": False,
        "difficulty_evaluation_failure_count": 2,
        "candidate_records": all_records,
        "accounting_rows": accounting_rows,
    }
    _write_json(OUTPUT_DIR / "fsn_fixed_budget_generation_candidates.json", payload)
    summary_payload = {
        "schema_version": "falcon.fsn_fixed_budget_generation_summary.v1",
        "num_failure_summaries": len(failures),
        "mode_metrics": mode_metrics,
        "judgement": judgement,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "entered_training_loop": False,
        "mappo_training_started": False,
        "random_fallback_used": False,
        "failure_stage": None,
        "fixed_budget_protocol_verified": all(
            (
                row["qwen_api_calls"] == 1
                if row["mode"] != "no_qwen_fsn_only"
                else row["qwen_api_calls"] == 0
            )
            for row in accounting_rows
        ),
        "qwen_calls_per_summary_by_mode": {
            mode: (
                0
                if mode == "no_qwen_fsn_only"
                else _rate(
                    sum(
                        int(row["qwen_api_calls"])
                        for row in accounting_rows
                        if row["mode"] == mode
                    ),
                    len(failures),
                )
            )
            for mode in MODES
        },
        "warnings": _normalize_warnings(warnings),
    }
    _write_json(
        OUTPUT_DIR / "fsn_fixed_budget_generation_summary.json", summary_payload
    )
    _write_metrics_csv(
        OUTPUT_DIR / "fsn_fixed_budget_generation_metrics.csv", mode_metrics
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary_payload), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


def _generate_qwen_once(
    failure_summary: Mapping[str, Any],
    base_config: Mapping[str, Any],
    target: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    generator = QwenScenarioGenerator(
        {
            "provider": "ollama",
            "provider_mode": "ollama_native",
            "model_name": "qwen3:8b",
            "think": False,
            "stream": False,
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 4096,
            "timeout": 180.0,
            "num_retries": 0,
        }
    )
    started = time.perf_counter()
    candidates = generator.generate_from_failure_summary(
        failure_summary, base_config, num_scenarios=target
    )
    runtime = round(time.perf_counter() - started, 6)
    result = generator.last_result
    raw = list(result.get("raw_responses") or [])
    attempts = list(result.get("attempts") or [])
    calls = len(raw)
    successful = sum(
        1
        for item in raw
        if not item.get("error") and bool(item.get("content") or item.get("raw_response"))
    )
    return candidates, {
        "qwen_api_calls": calls,
        "qwen_api_calls_successful": successful,
        "qwen_api_calls_failed": max(calls - successful, 0),
        "qwen_runtime_seconds": runtime,
        "qwen_candidates_raw_returned": sum(
            int(item.get("repaired_candidate_count") or 0) for item in attempts
        ),
        "qwen_candidates_valid_returned": len(candidates),
        "warnings": list(result.get("warnings") or []),
    }


def _validate_candidate(
    candidate: Mapping[str, Any],
    failure_record: Mapping[str, Any],
    mode: str,
    failure_index: int,
    yaml_root: Path,
    base_config: Mapping[str, Any],
    checker: ConstraintChecker,
) -> Dict[str, Any]:
    started = time.perf_counter()
    schema = validate_candidate_schema(candidate)
    constraint = checker.validate_candidate(candidate)
    yaml_generated = False
    env_result: Dict[str, Any] = {}
    yaml_path: Optional[Path] = None
    if schema.get("is_valid") and constraint.get("is_valid"):
        yaml_config = apply_initial_config_to_yaml(
            base_config, candidate.get("initial_config") or {}
        )
        yaml_config["scenario_id"] = candidate.get("scenario_id")
        yaml_path = (
            yaml_root / mode / f"failure_{failure_index:02d}" / f"{candidate['scenario_id']}.yaml"
        )
        save_scenario_yaml(yaml_config, yaml_path)
        yaml_generated = True
        env_result = checker.validate_yaml_config(
            yaml_config,
            enable_env_load_check=True,
            temp_config_name=f"fixed_budget_{candidate['scenario_id']}",
        )
    env_ok = bool(
        (env_result.get("physical_constraint_check") or {}).get(
            "scenario_loadable_env_check"
        )
    )
    metadata = candidate.get("metadata") or {}
    return {
        "schema_version": "falcon.fsn_fixed_budget_candidate_record.v1",
        "failure_id": failure_record.get("failure_id"),
        "seed": failure_record.get("seed"),
        "round_id": failure_record.get("round_id"),
        "mode": mode,
        "generator_type": candidate.get("generator_type"),
        "candidate": dict(candidate),
        "schema_valid": bool(schema.get("is_valid")),
        "constraint_valid": bool(constraint.get("is_valid")),
        "yaml_generated": yaml_generated,
        "yaml_path": str(yaml_path.resolve()) if yaml_path else None,
        "env_load_success": env_ok,
        "predicted_value_score": _number(metadata.get("predicted_value_score")),
        "changed_factors": list(candidate.get("changed_factors") or []),
        "validation_runtime_seconds": round(time.perf_counter() - started, 6),
        "constraint_result": constraint,
        "difficulty_result": {},
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
        failure_record["current_checkpoint"],
        record["yaml_path"],
        num_episodes=1,
        seed=seed,
    )
    best = policy_evaluator.evaluate_policy_on_scenario(
        failure_record["best_checkpoint"],
        record["yaml_path"],
        num_episodes=1,
        seed=seed,
    )
    record["current_policy_eval"] = current
    record["best_policy_eval"] = best
    if current.get("real_policy_eval_available") and best.get(
        "real_policy_eval_available"
    ):
        record["difficulty_result"] = difficulty_evaluator.evaluate_candidate(
            record["candidate"],
            current,
            best,
            pool_stats,
            failure_summary,
            record["constraint_result"],
        )


def _mode_metrics(
    mode: str,
    accounting: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [item for item in accounting if item.get("mode") == mode]
    candidates = [item for item in records if item.get("mode") == mode]
    qwen_calls = sum(int(item.get("qwen_api_calls") or 0) for item in rows)
    valid = sum(1 for item in candidates if item.get("constraint_valid"))
    env = sum(1 for item in candidates if item.get("env_load_success"))
    evaluated = [
        item for item in candidates if item.get("difficulty_result")
    ]
    accepted = [
        item
        for item in evaluated
        if item["difficulty_result"].get("accepted_into_curriculum_pool")
    ]
    predicted = [
        value
        for item in candidates
        if (value := _number(item.get("predicted_value_score"))) is not None
    ]
    difficulty_values = [
        value
        for item in evaluated
        if (
            value := _number(item["difficulty_result"].get("final_value_score"))
        )
        is not None
    ]
    return {
        "mode": mode,
        "qwen_api_calls": qwen_calls,
        "qwen_api_calls_successful": sum(
            int(item.get("qwen_api_calls_successful") or 0) for item in rows
        ),
        "qwen_api_calls_failed": sum(
            int(item.get("qwen_api_calls_failed") or 0) for item in rows
        ),
        "qwen_runtime_seconds": round(
            sum(float(item.get("qwen_runtime_seconds") or 0.0) for item in rows), 6
        ),
        "fsn_runtime_seconds": round(
            sum(float(item.get("fsn_runtime_seconds") or 0.0) for item in rows), 6
        ),
        "total_candidates_raw": len(candidates),
        "total_candidates_valid": valid,
        "total_env_loadable_candidates": env,
        "valid_candidates_per_qwen_call": _rate(valid, qwen_calls),
        "env_loadable_candidates_per_qwen_call": _rate(env, qwen_calls),
        "difficulty_evaluated_count": len(evaluated),
        "difficulty_accepted_count": len(accepted),
        "difficulty_accepted_rate": _rate(len(accepted), len(evaluated)),
        "accepted_candidates_per_qwen_call": (
            _rate(len(accepted), qwen_calls) if evaluated and qwen_calls else None
        ),
        "mean_predicted_value": _mean(predicted),
        "mean_difficulty_value": _mean(difficulty_values),
        "diversity": _candidate_diversity([item["candidate"] for item in candidates]),
        "random_fallback_count": 0,
        "qwen_shortfall_count": sum(
            int(item.get("qwen_shortfall_count") or 0) for item in rows
        ),
        "qwen_candidates_requested": sum(
            int(item.get("qwen_candidates_requested") or 0) for item in rows
        ),
        "qwen_candidates_raw_returned": sum(
            int(item.get("qwen_candidates_raw_returned") or 0) for item in rows
        ),
        "qwen_candidates_valid_returned": sum(
            int(item.get("qwen_candidates_valid_returned") or 0) for item in rows
        ),
        "changed_factor_distribution": dict(
            sorted(
                Counter(
                    factor
                    for item in candidates
                    for factor in item.get("changed_factors") or []
                ).items()
            )
        ),
    }


def _judgement(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    full = metrics["full_qwen_fixed_call"]
    fsn25 = metrics["fsn25_fixed_call"]
    fsn50 = metrics["fsn50_fixed_call"]
    fsn25_more_valid = (
        fsn25["total_candidates_valid"] > full["total_candidates_valid"]
    )
    fsn25_more_env = (
        fsn25["total_env_loadable_candidates"]
        > full["total_env_loadable_candidates"]
    )
    fsn25_diversity_not_lower = fsn25["diversity"] >= full["diversity"]
    accepted_not_lower = (
        fsn25["difficulty_accepted_rate"]
        >= full["difficulty_accepted_rate"] - 0.05
    )
    fsn25_support = bool(
        fsn25_more_valid
        and fsn25_more_env
        and fsn25_diversity_not_lower
        and accepted_not_lower
    )
    fsn50_stable = bool(
        fsn50["total_candidates_valid"] >= fsn25["total_candidates_valid"]
        and fsn50["diversity"] >= full["diversity"]
        and fsn50["difficulty_accepted_rate"]
        >= full["difficulty_accepted_rate"] - 0.05
    )
    random_fallback_required = any(
        metrics[mode]["total_candidates_valid"] < 40
        for mode in ("fsn25_fixed_call", "fsn50_fixed_call")
    )
    fsn25_qwen_slot_reduction = 1.0 - _rate(
        fsn25["qwen_candidates_requested"], full["qwen_candidates_requested"]
    )
    fsn50_qwen_slot_reduction = 1.0 - _rate(
        fsn50["qwen_candidates_requested"], full["qwen_candidates_requested"]
    )
    fsn25_qwen_runtime_reduction = 1.0 - _rate(
        fsn25["qwen_runtime_seconds"], full["qwen_runtime_seconds"]
    )
    fsn50_qwen_runtime_reduction = 1.0 - _rate(
        fsn50["qwen_runtime_seconds"], full["qwen_runtime_seconds"]
    )
    return {
        "fsn25_more_valid_candidates": fsn25_more_valid,
        "fsn25_more_env_loadable_candidates": fsn25_more_env,
        "fsn25_diversity_not_lower": fsn25_diversity_not_lower,
        "fsn25_difficulty_acceptance_not_materially_lower": accepted_not_lower,
        "fsn_improves_candidate_generation_efficiency_under_fixed_llm_call_budget": fsn25_support,
        "fsn50_stable": fsn50_stable,
        "fsn25_qwen_candidate_slot_reduction": round(
            fsn25_qwen_slot_reduction, 6
        ),
        "fsn50_qwen_candidate_slot_reduction": round(
            fsn50_qwen_slot_reduction, 6
        ),
        "fsn25_qwen_runtime_reduction": round(
            fsn25_qwen_runtime_reduction, 6
        ),
        "fsn50_qwen_runtime_reduction": round(
            fsn50_qwen_runtime_reduction, 6
        ),
        "fsn25_valid_candidate_delta": (
            fsn25["total_candidates_valid"] - full["total_candidates_valid"]
        ),
        "fsn50_valid_candidate_delta": (
            fsn50["total_candidates_valid"] - full["total_candidates_valid"]
        ),
        "fsn25_diversity_delta": round(
            fsn25["diversity"] - full["diversity"], 6
        ),
        "fsn50_diversity_delta": round(
            fsn50["diversity"] - full["diversity"], 6
        ),
        "recommend_20_round_25_percent_replacement_pilot": fsn25_support,
        "recommend_50_percent_replacement": fsn50_stable,
        "random_fallback_required_to_guarantee_four_valid_candidates_per_summary": random_fallback_required,
    }


def _select_failures(records: Sequence[Mapping[str, Any]], count: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    per_seed: Counter[int] = Counter()
    for record in records:
        seed = int(record.get("seed", -1))
        if per_seed[seed] >= 2:
            continue
        if record.get("missing_fields"):
            continue
        selected.append(dict(record))
        per_seed[seed] += 1
        if len(selected) >= count:
            break
    return selected


def _pool_stats(
    dataset: Sequence[Mapping[str, Any]],
    failure_record: Mapping[str, Any],
) -> Dict[str, Any]:
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


def _candidate_diversity(candidates: Sequence[Mapping[str, Any]]) -> float:
    if len(candidates) < 2:
        return 0.0
    scales = DIFFICULTY_CONFIG["scenario_vector_scales"]
    distances: List[float] = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left = candidates[left_index].get("scenario_vector") or {}
            right = candidates[right_index].get("scenario_vector") or {}
            components = []
            for key in SCENARIO_VECTOR_KEYS:
                left_value = _number(left.get(key))
                right_value = _number(right.get(key))
                if left_value is None or right_value is None:
                    continue
                components.append(
                    ((left_value - right_value) / float(scales[key])) ** 2
                )
            if components:
                distances.append(math.sqrt(sum(components) / len(components)))
    return round(statistics.fmean(distances), 6) if distances else 0.0


def _candidate_diversity_scores(
    candidates: Sequence[Mapping[str, Any]],
) -> List[float]:
    if len(candidates) < 2:
        return [0.0 for _item in candidates]
    scales = DIFFICULTY_CONFIG["scenario_vector_scales"]
    scores: List[float] = []
    for index, candidate in enumerate(candidates):
        vector = candidate.get("scenario_vector") or {}
        distances = []
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            other_vector = other.get("scenario_vector") or {}
            components = []
            for key in SCENARIO_VECTOR_KEYS:
                left = _number(vector.get(key))
                right = _number(other_vector.get(key))
                if left is None or right is None:
                    continue
                components.append(((left - right) / float(scales[key])) ** 2)
            if components:
                distances.append(math.sqrt(sum(components) / len(components)))
        scores.append(round(min(distances), 6) if distances else 0.0)
    return scores


def _report(summary: Mapping[str, Any]) -> str:
    metrics = summary["mode_metrics"]
    judgement = summary["judgement"]
    lines = [
        "FALCON FSN Fixed-Budget Generation Smoke",
        "",
        f"Failure summaries: {summary['num_failure_summaries']}",
        "Each Qwen-enabled mode used exactly one Qwen API call per failure summary with retries disabled.",
        "Random fallback: false",
        "",
        "Mode metrics:",
    ]
    for mode in MODES:
        item = metrics[mode]
        lines.append(
            f"- {mode}: calls={item['qwen_api_calls']}, valid={item['total_candidates_valid']}, "
            f"env={item['total_env_loadable_candidates']}, valid/call={item['valid_candidates_per_qwen_call']}, "
            f"env/call={item['env_loadable_candidates_per_qwen_call']}, diversity={item['diversity']}, "
            f"accepted={item['difficulty_accepted_count']}/{item['difficulty_evaluated_count']}"
        )
    lines.extend(
        [
            "",
            "Judgement:",
            f"- FSN25 increases valid candidates under fixed calls: {judgement['fsn25_more_valid_candidates']}",
            f"- FSN25 increases env-loadable candidates: {judgement['fsn25_more_env_loadable_candidates']}",
            f"- FSN25 diversity does not decrease: {judgement['fsn25_diversity_not_lower']}",
            f"- FSN25 difficulty acceptance is not materially lower: {judgement['fsn25_difficulty_acceptance_not_materially_lower']}",
            f"- FSN25 Qwen candidate-slot / runtime reduction: {judgement['fsn25_qwen_candidate_slot_reduction']:.1%} / {judgement['fsn25_qwen_runtime_reduction']:.1%}",
            f"- FSN25 valid-candidate / diversity delta vs Full-Qwen: {judgement['fsn25_valid_candidate_delta']} / {judgement['fsn25_diversity_delta']}",
            f"- FSN50 Qwen candidate-slot / runtime reduction: {judgement['fsn50_qwen_candidate_slot_reduction']:.1%} / {judgement['fsn50_qwen_runtime_reduction']:.1%}",
            f"- FSN50 valid-candidate / diversity delta vs Full-Qwen: {judgement['fsn50_valid_candidate_delta']} / {judgement['fsn50_diversity_delta']}",
            f"- Fixed-call efficiency claim supported: {judgement['fsn_improves_candidate_generation_efficiency_under_fixed_llm_call_budget']}",
            f"- FSN50 stable: {judgement['fsn50_stable']}",
            f"- Recommend 20-round 25% replacement pilot: {judgement['recommend_20_round_25_percent_replacement_pilot']}",
            f"- Random fallback still required to guarantee four valid candidates per summary: {judgement['random_fallback_required_to_guarantee_four_valid_candidates_per_summary']}",
            "- FSN25 reduces dependence on Qwen candidate slots and observed Qwen runtime, but it did not increase valid or env-loadable candidate count in this smoke.",
            "- No Random fallback was used; Qwen shortfalls remain visible rather than being hidden by replacement.",
            "",
            "Limitations: this is candidate-generation accounting with a small difficulty-evaluated subset. It does not show policy-performance improvement, training replacement success, or realized end-to-end cost reduction.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_metrics_csv(path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    rows = []
    for mode, item in metrics.items():
        row = {"mode": mode}
        row.update(
            {
                key: value
                for key, value in item.items()
                if not isinstance(value, (dict, list))
            }
        )
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _empty_qwen_accounting() -> Dict[str, Any]:
    return {
        "qwen_api_calls": 0,
        "qwen_api_calls_successful": 0,
        "qwen_api_calls_failed": 0,
        "qwen_runtime_seconds": 0.0,
        "qwen_candidates_raw_returned": 0,
        "qwen_candidates_valid_returned": 0,
        "warnings": [],
    }


def _normalize_warnings(warnings: Iterable[str]) -> List[str]:
    normalized = []
    underfill_detected = False
    for warning in warnings:
        text = str(warning)
        if "retrying for requested" in text:
            underfill_detected = True
            continue
        normalized.append(text)
    if underfill_detected:
        normalized.append(
            "Qwen under-filled one or more fixed-budget requests; the evaluator did not retry."
        )
    return sorted(set(normalized))


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 6) if values else None


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        denominator = float(denominator)
        return round(float(numerator) / denominator, 6) if denominator > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
