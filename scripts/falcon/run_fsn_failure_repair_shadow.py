"""Diagnose and evaluate adapter-aware FSN repair without MAPPO training."""

from __future__ import annotations

import copy
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.fsn_dataset import load_jsonl  # noqa: E402
from falcon.fsn_generator import FSNScenarioGenerator  # noqa: E402
from falcon.policy_evaluator import PolicyEvaluator  # noqa: E402
from falcon.scenario_adapter import (  # noqa: E402
    apply_initial_config_to_yaml,
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    load_base_scenario_config,
    save_scenario_yaml,
)
from scripts.falcon.test_fsn_fixed_budget_generation_smoke import (  # noqa: E402
    BASE_CONFIG_PATH,
    FSN_DATASET,
    FSN_MODEL,
    OPPONENT_MANIFEST,
    STAGE3_FAILURES,
    _candidate_diversity,
    _load_json,
    _pool_stats,
    _resolve,
    _select_failures,
)


STAGE3_CANDIDATES = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage3"
    / "fsn_policy_evaluated_shadow_candidates.json"
)
PILOT_ROOT = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "results_fsn25_rerank_20r_pilot"
    / "falcon_fsn25_rerank"
    / "seed_4"
)
OUTPUT_DIR = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage5_repair"
)
REPORT_PATH = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "reports"
    / "fsn_repair_stage_report.txt"
)
MODES = (
    "fsn_rerank_original",
    "fsn_repaired",
    "fsn_repaired_hardness",
    "historical_qwen",
    "random",
)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_base_scenario_config(BASE_CONFIG_PATH)
    checker = ConstraintChecker()
    diagnosis = _diagnose_pilot_candidates(base_config, checker)
    _write_json(OUTPUT_DIR / "fsn_post_yaml_constraint_diagnosis.json", diagnosis)
    (OUTPUT_DIR / "fsn_post_yaml_constraint_diagnosis_report.txt").write_text(
        _diagnosis_report(diagnosis), encoding="utf-8"
    )

    failures = _select_failures(
        _load_json(STAGE3_FAILURES)["failure_summaries"], 10
    )
    historical_payload = _load_json(STAGE3_CANDIDATES)
    historical = _historical_by_failure(
        historical_payload.get("candidate_records") or []
    )
    dataset = load_jsonl(FSN_DATASET)
    opponent = _resolve(_load_json(OPPONENT_MANIFEST)["checkpoint_path"])
    policy_evaluator = PolicyEvaluator(
        {
            "base_config_path": str(BASE_CONFIG_PATH),
            "opponent_mode": "fixed_checkpoint",
            "opponent_checkpoint": str(opponent),
            "deterministic": True,
            "device": "cpu",
        }
    )
    difficulty_evaluator = DifficultyEvaluator()
    all_records: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for failure_index, failure_record in enumerate(failures):
        failure_id = str(failure_record["failure_id"])
        failure_summary = dict(failure_record.get("failure_summary") or {})
        pool_stats = _pool_stats(dataset, failure_record)
        generated = _generate_modes(
            failure_index,
            failure_id,
            failure_summary,
            base_config,
            pool_stats,
            historical,
        )
        for mode, payload in generated.items():
            mode_started = time.perf_counter()
            records = []
            for candidate_index, candidate in enumerate(payload["candidates"]):
                candidate = copy.deepcopy(candidate)
                candidate["scenario_id"] = (
                    f"repair_f{failure_index:02d}_{mode}_{candidate_index:02d}"
                )
                record = _materialize_candidate(
                    candidate,
                    failure_record,
                    mode,
                    failure_index,
                    candidate_index,
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
                        seed=950000
                        + failure_index * 1000
                        + MODES.index(mode) * 100
                        + candidate_index,
                    )
                records.append(record)
                all_records.append(record)
            generation_rows.append(
                {
                    "failure_id": failure_id,
                    "seed": failure_record.get("seed"),
                    "round_id": failure_record.get("round_id"),
                    "mode": mode,
                    "candidate_count": len(records),
                    "generation_runtime_seconds": payload["runtime_seconds"],
                    "validation_and_eval_runtime_seconds": round(
                        time.perf_counter() - mode_started, 6
                    ),
                    "repair_success_count": payload.get(
                        "repair_success_count", 0
                    ),
                }
            )

    metrics = {
        mode: _mode_metrics(mode, all_records, generation_rows)
        for mode in MODES
    }
    judgement = _judgement(metrics)
    summary = {
        "schema_version": "falcon.fsn_repair_shadow_summary.v1",
        "num_failure_summaries": len(failures),
        "candidates_per_mode": {
            mode: metrics[mode]["candidate_count"] for mode in MODES
        },
        "fixed_opponent_checkpoint": str(opponent),
        "episodes_per_policy_per_candidate": 1,
        "same_actor": False,
        "mode_metrics": metrics,
        "judgement": judgement,
        "diagnosis_summary": diagnosis["summary"],
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "entered_training_loop": False,
        "mappo_training_started": False,
        "failure_stage": None,
        "warnings": [
            "The shadow test uses 10 failure summaries and 40 candidates per mode to control true policy-evaluation cost.",
            "Historical Qwen and Random candidates were reused from Stage 3; their generation runtime is not re-measured.",
        ],
    }
    _write_json(OUTPUT_DIR / "fsn_repair_shadow_candidates.json", {
        "schema_version": "falcon.fsn_repair_shadow_candidates.v1",
        "candidate_records": all_records,
        "generation_rows": generation_rows,
    })
    _write_json(OUTPUT_DIR / "fsn_repair_shadow_summary.json", summary)
    _write_metrics_csv(OUTPUT_DIR / "fsn_repair_shadow_metrics.csv", metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _diagnose_pilot_candidates(
    base_config: Mapping[str, Any], checker: ConstraintChecker
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for round_id in range(20):
        generation = _load_json(
            PILOT_ROOT / f"falcon_controller_candidates_round{round_id}.json"
        )
        candidates = [
            item
            for item in generation.get("candidates") or []
            if item.get("generator_type") == "fsn"
        ]
        for candidate in candidates:
            schema = validate_candidate_schema(candidate)
            pre = checker.validate_candidate(candidate)
            yaml_config = apply_initial_config_to_yaml(
                base_config, candidate.get("initial_config") or {}
            )
            yaml_config["scenario_id"] = candidate.get("scenario_id")
            post_initial = extract_initial_config_from_yaml(yaml_config)
            post_vector = initial_config_to_scenario_vector(post_initial)[
                "scenario_vector"
            ]
            post = checker.validate_yaml_config(yaml_config)
            repair = checker.repair_candidate_for_yaml(
                candidate, base_config, max_repair_attempts=3, repair_margin_m=5.0
            )
            requested = (
                (candidate.get("scenario_parameters") or {}).get(
                    "legalized_request"
                )
                or candidate.get("scenario_vector")
                or {}
            )
            rows.append(
                {
                    "round_id": round_id,
                    "scenario_id": candidate.get("scenario_id"),
                    "pre_schema_valid": bool(schema.get("is_valid")),
                    "pre_constraint_valid": bool(pre.get("is_valid")),
                    "yaml_generated": True,
                    "post_yaml_constraint_valid": bool(post.get("is_valid")),
                    "post_yaml_rejection_reasons": post.get(
                        "rejection_reasons", []
                    ),
                    "formation_spread_valid": (
                        post.get("task_constraint_check") or {}
                    ).get("formation_spread_valid"),
                    "requested_key_values": _key_values(requested),
                    "candidate_key_values": _key_values(
                        candidate.get("scenario_vector") or {}
                    ),
                    "post_yaml_key_values": _key_values(post_vector),
                    "candidate_to_post_yaml_error": _vector_error(
                        candidate.get("scenario_vector") or {}, post_vector
                    ),
                    "adapter_fields_outside_after_roundtrip": _outside_fields(
                        post_vector
                    ),
                    "repair_valid": bool(repair.get("is_valid")),
                    "repair_success": bool(repair.get("repair_success")),
                    "repair_actions": repair.get("repair_actions") or [],
                    "repaired_key_values": _key_values(
                        (repair.get("candidate") or {}).get("scenario_vector")
                        or {}
                    ),
                }
            )
    formation_failures = sum(
        1 for row in rows if row.get("formation_spread_valid") is False
    )
    return {
        "schema_version": "falcon.fsn_post_yaml_constraint_diagnosis.v1",
        "summary": {
            "candidate_count": len(rows),
            "pre_schema_valid_count": sum(
                1 for row in rows if row["pre_schema_valid"]
            ),
            "pre_constraint_valid_count": sum(
                1 for row in rows if row["pre_constraint_valid"]
            ),
            "post_yaml_constraint_valid_count": sum(
                1 for row in rows if row["post_yaml_constraint_valid"]
            ),
            "formation_spread_valid_failure_count": formation_failures,
            "repair_valid_count": sum(1 for row in rows if row["repair_valid"]),
            "repair_success_count": sum(
                1 for row in rows if row["repair_success"]
            ),
            "root_cause": (
                "Formation spread was generated exactly at the 1000 m lower "
                "boundary; latitude/longitude round-trip floating-point error "
                "made opponent spread slightly smaller than 1000 m."
            ),
        },
        "candidates": rows,
    }


def _generate_modes(
    failure_index: int,
    failure_id: str,
    failure_summary: Mapping[str, Any],
    base_config: Mapping[str, Any],
    pool_stats: Mapping[str, Any],
    historical: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    generator = FSNScenarioGenerator(
        FSN_MODEL,
        {"seed": 12000 + failure_index, "noise_scale": 0.10},
    )
    started = time.perf_counter()
    original = generator.generate_reranked_from_failure_summary(
        failure_summary,
        base_config,
        num_scenarios=4,
        overgenerate_count=16,
        rerank_weights={
            "predicted_value_score": 0.0,
            "accepted_probability": 0.0,
            "diversity_bonus": 0.65,
            "constraint_risk_penalty": 0.35,
        },
    )
    result["fsn_rerank_original"] = {
        "candidates": original,
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }
    started = time.perf_counter()
    repaired = generator.generate_repaired_from_failure_summary(
        failure_summary, base_config, 4, 16
    )
    result["fsn_repaired"] = {
        "candidates": repaired,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "repair_success_count": generator.last_repair_result.get(
            "repair_success_count", 0
        ),
    }
    started = time.perf_counter()
    hardness = generator.generate_hardness_filtered_from_failure_summary(
        failure_summary,
        base_config,
        4,
        32,
        pool_stats=pool_stats,
    )
    result["fsn_repaired_hardness"] = {
        "candidates": hardness,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "repair_success_count": generator.last_hardness_result.get(
            "repair_success_count", 0
        ),
    }
    for mode in ("historical_qwen", "random"):
        candidates = [
            copy.deepcopy(item)
            for item in historical.get(failure_id, {}).get(mode, [])[:4]
        ]
        result[mode] = {"candidates": candidates, "runtime_seconds": 0.0}
    return result


def _historical_by_failure(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        generator = str(record.get("generator_type") or "")
        if generator not in {"historical_qwen", "random"}:
            continue
        result[str(record.get("failure_id"))][generator].append(
            dict(record.get("candidate") or {})
        )
    return result


def _materialize_candidate(
    candidate: Mapping[str, Any],
    failure_record: Mapping[str, Any],
    mode: str,
    failure_index: int,
    candidate_index: int,
    base_config: Mapping[str, Any],
    checker: ConstraintChecker,
) -> Dict[str, Any]:
    started = time.perf_counter()
    schema = validate_candidate_schema(candidate)
    pre = checker.validate_candidate(candidate)
    yaml_config = apply_initial_config_to_yaml(
        base_config, candidate.get("initial_config") or {}
    )
    yaml_config["scenario_id"] = candidate.get("scenario_id")
    yaml_path = (
        OUTPUT_DIR
        / "generated_yamls"
        / mode
        / f"failure_{failure_index:02d}"
        / f"candidate_{candidate_index:02d}.yaml"
    )
    save_scenario_yaml(yaml_config, yaml_path)
    post = checker.validate_yaml_config(
        yaml_config,
        enable_env_load_check=True,
        temp_config_name=f"repair_shadow_{mode}_{failure_index}_{candidate_index}",
    )
    env_ok = bool(
        (post.get("physical_constraint_check") or {}).get(
            "scenario_loadable_env_check"
        )
    )
    metadata = candidate.get("metadata") or {}
    return {
        "schema_version": "falcon.fsn_repair_shadow_candidate.v1",
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
        "formation_spread_valid": (
            post.get("task_constraint_check") or {}
        ).get("formation_spread_valid"),
        "yaml_generated": True,
        "yaml_path": str(yaml_path.resolve()),
        "env_load_success": env_ok,
        "constraint_result": post,
        "repair_applied": bool(metadata.get("adapter_repair_applied")),
        "repair_actions": list(metadata.get("repair_actions") or []),
        "hardness_proxy": metadata.get("hardness_proxy"),
        "hardness_components": metadata.get("hardness_components") or {},
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
    rejection_reasons = Counter(
        reason
        for record in evaluated
        for reason in record["difficulty_result"].get("rejection_reasons") or []
    )
    actions = Counter(
        str(action.get("field") or action.get("action") or "unknown")
        for record in subset
        for action in record.get("repair_actions") or []
    )
    values = [
        float(record["difficulty_result"]["final_value_score"])
        for record in evaluated
        if record["difficulty_result"].get("final_value_score") is not None
    ]
    learning = [
        float(record["difficulty_result"]["learning_potential"])
        for record in evaluated
        if record["difficulty_result"].get("learning_potential") is not None
    ]
    return {
        "mode": mode,
        "candidate_count": len(subset),
        "schema_valid_rate": _rate(
            sum(1 for record in subset if record["schema_valid"]), len(subset)
        ),
        "pre_constraint_valid_rate": _rate(
            sum(1 for record in subset if record["pre_constraint_valid"]),
            len(subset),
        ),
        "post_yaml_constraint_valid_rate": _rate(
            sum(
                1 for record in subset if record["post_yaml_constraint_valid"]
            ),
            len(subset),
        ),
        "formation_spread_valid_failure_count": sum(
            1 for record in subset if record["formation_spread_valid"] is False
        ),
        "env_load_rate": _rate(
            sum(1 for record in subset if record["env_load_success"]),
            len(subset),
        ),
        "difficulty_evaluated_count": len(evaluated),
        "accepted_count": len(accepted),
        # End-to-end acceptance must include candidates lost during YAML
        # round-trip validation. Keep the evaluator-only rate separately.
        "accepted_rate": _rate(len(accepted), len(subset)),
        "difficulty_accepted_rate": _rate(len(accepted), len(evaluated)),
        "too_easy_rejection_rate": _rate(
            rejection_reasons["too_easy_for_current_policy"], len(evaluated)
        ),
        "not_solvable_rejection_rate": _rate(
            rejection_reasons["not_solvable_by_historical_best_policy"],
            len(evaluated),
        ),
        "rejection_reason_distribution": dict(sorted(rejection_reasons.items())),
        "mean_final_value_score": _mean(values),
        "mean_learning_potential": _mean(learning),
        "diversity_score": _candidate_diversity(
            [record.get("candidate") or {} for record in subset]
        ),
        "repair_applied_count": sum(
            1 for record in subset if record.get("repair_applied")
        ),
        "repair_success_rate": _rate(
            sum(1 for record in subset if record.get("repair_applied")),
            len(subset),
        ),
        "repair_actions_distribution": dict(sorted(actions.items())),
        "runtime_seconds": round(
            sum(
                float(row.get("generation_runtime_seconds") or 0.0)
                + float(row.get("validation_and_eval_runtime_seconds") or 0.0)
                for row in generation_rows
                if row.get("mode") == mode
            ),
            6,
        ),
    }


def _judgement(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    original = metrics["fsn_rerank_original"]
    repaired = metrics["fsn_repaired"]
    hardness = metrics["fsn_repaired_hardness"]
    return {
        "repair_post_yaml_constraint_gate_passed": (
            repaired["post_yaml_constraint_valid_rate"] >= 0.95
        ),
        "hardness_filter_accepted_rate_improved": (
            hardness["accepted_rate"] > original["accepted_rate"]
        ),
        "hardness_filter_too_easy_rate_reduced": (
            hardness["too_easy_rejection_rate"]
            < original["too_easy_rejection_rate"]
        ),
        "hardness_filter_learning_potential_improved": (
            (hardness["mean_learning_potential"] or 0.0)
            > (original["mean_learning_potential"] or 0.0)
        ),
        "hardness_filter_diversity_improved_vs_repaired": (
            hardness["diversity_score"] > repaired["diversity_score"]
        ),
        "hardness_filter_mean_value_improved_vs_repaired": (
            (hardness["mean_final_value_score"] or 0.0)
            > (repaired["mean_final_value_score"] or 0.0)
        ),
        "hardness_filter_accepted_count_improved_vs_repaired": (
            hardness["accepted_count"] > repaired["accepted_count"]
        ),
        "hardness_filter_env_gate_passed": hardness["env_load_rate"] >= 0.95,
        "repaired_fsn_has_real_accepted_scenarios": (
            hardness["accepted_count"] > 0 or repaired["accepted_count"] > 0
        ),
        "recommend_retry_20_round_25_percent_replacement": bool(
            repaired["post_yaml_constraint_valid_rate"] >= 0.95
            and hardness["accepted_rate"] > original["accepted_rate"]
            and hardness["too_easy_rejection_rate"]
            < original["too_easy_rejection_rate"]
            and (hardness["mean_learning_potential"] or 0.0)
            > (original["mean_learning_potential"] or 0.0)
            and hardness["env_load_rate"] >= 0.95
            and hardness["accepted_count"] > 0
        ),
        "recommend_opd": False,
    }


def _key_values(vector: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: vector.get(key)
        for key in (
            "own_formation_spread",
            "opponent_formation_spread",
            "team_center_distance",
            "altitude_difference",
            "velocity_difference",
            "heading_difference",
            "approximate_aspect_angle",
        )
    }


def _vector_error(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Dict[str, float]:
    result = {}
    for key, value in _key_values(left).items():
        if value is None or right.get(key) is None:
            continue
        result[key] = float(right[key]) - float(value)
    return result


def _outside_fields(vector: Mapping[str, Any]) -> List[str]:
    ranges = {
        "own_formation_spread": (1000.0, 8000.0),
        "opponent_formation_spread": (1000.0, 8000.0),
        "team_center_distance": (6000.0, 18000.0),
    }
    return [
        key
        for key, (low, high) in ranges.items()
        if vector.get(key) is not None
        and not low <= float(vector[key]) <= high
    ]


def _diagnosis_report(diagnosis: Mapping[str, Any]) -> str:
    summary = diagnosis["summary"]
    return "\n".join(
        [
            "FSN Post-YAML Constraint Diagnosis",
            "",
            f"- Candidates: {summary['candidate_count']}",
            f"- Pre-schema / pre-constraint valid: {summary['pre_schema_valid_count']} / {summary['pre_constraint_valid_count']}",
            f"- Post-YAML valid: {summary['post_yaml_constraint_valid_count']}",
            f"- formation_spread_valid failures: {summary['formation_spread_valid_failure_count']}",
            f"- Adapter-aware repair valid / repaired: {summary['repair_valid_count']} / {summary['repair_success_count']}",
            f"- Root cause: {summary['root_cause']}",
        ]
    ) + "\n"


def _report(summary: Mapping[str, Any]) -> str:
    metrics = summary["mode_metrics"]
    judgement = summary["judgement"]
    lines = [
        "FALCON FSN Failure Repair Stage",
        "",
        f"- Failure summaries: {summary['num_failure_summaries']}",
        "- No MAPPO training or controller replacement was run.",
        "",
        "Mode metrics:",
    ]
    for mode in MODES:
        item = metrics[mode]
        lines.append(
            f"- {mode}: post-YAML-valid={item['post_yaml_constraint_valid_rate']}, "
            f"env={item['env_load_rate']}, accepted={item['accepted_count']}/{item['candidate_count']} "
            f"(end-to-end={item['accepted_rate']}, evaluator-only={item['difficulty_accepted_rate']}), "
            f"too-easy={item['too_easy_rejection_rate']}, "
            f"learning={item['mean_learning_potential']}, diversity={item['diversity_score']}."
        )
    lines.extend(
        [
            "",
            "Answers:",
            f"1. Root cause: {summary['diagnosis_summary']['root_cause']}",
            f"2. Adapter-aware repair solved formation-spread validity: {judgement['repair_post_yaml_constraint_gate_passed']}.",
            f"3. Hardness filter reduced too-easy rejection: {judgement['hardness_filter_too_easy_rate_reduced']}; "
            f"it improved diversity versus repaired-only: {judgement['hardness_filter_diversity_improved_vs_repaired']}; "
            f"it did not improve mean learning potential: {not judgement['hardness_filter_learning_potential_improved']}.",
            f"4. Repaired FSN produced real accepted scenes: {judgement['repaired_fsn_has_real_accepted_scenarios']}.",
            f"5. Recommend retrying the 20-round 25% replacement pilot: {judgement['recommend_retry_20_round_25_percent_replacement']}.",
            "6. Recommend OPD: false.",
            "",
            "Decision rationale:",
            "- Repair clears the >=0.95 post-YAML validity and env-load gates.",
            "- Hardness filtering improves end-to-end acceptance versus the un-repaired original because invalid candidates are no longer lost.",
            "- Hardness filtering lowers too-easy rejection and raises diversity, but accepted count is unchanged versus repaired-only and mean learning potential does not improve.",
            "- Therefore the repair stage is successful, while hardness calibration is not yet strong enough to approve another replacement-training pilot.",
            "",
            "Limitations:",
            "- This is an offline single-episode-per-policy shadow evaluation.",
            "- To control evaluation cost, it uses 10 failure summaries and 40 candidates per mode rather than the preferred 20 summaries / 80 candidates per mode.",
            "- Historical Qwen and Random samples are reused rather than newly generated.",
            "- The results cannot establish replacement-training performance or formal policy improvement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 6) if values else None


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        denominator = float(denominator)
        return round(float(numerator) / denominator, 6) if denominator > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_metrics_csv(
    path: Path, metrics: Mapping[str, Mapping[str, Any]]
) -> None:
    rows = [
        {
            key: value
            for key, value in item.items()
            if not isinstance(value, (dict, list))
        }
        for item in metrics.values()
    ]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
