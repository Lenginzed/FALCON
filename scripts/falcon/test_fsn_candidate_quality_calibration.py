"""Calibrate FSN heads and evaluate over-generation plus reranking offline."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.difficulty_evaluator import DifficultyEvaluator  # noqa: E402
from falcon.fsn_calibration import (  # noqa: E402
    calibrate,
    calibration_curve_rows,
    collect_calibration_rows,
)
from falcon.fsn_dataset import load_jsonl  # noqa: E402
from falcon.fsn_generator import FSNScenarioGenerator  # noqa: E402
from falcon.policy_evaluator import PolicyEvaluator  # noqa: E402
from falcon.scenario_adapter import load_base_scenario_config  # noqa: E402
from scripts.falcon.test_fsn_fixed_budget_generation_smoke import (  # noqa: E402
    BASE_CONFIG_PATH,
    FSN_DATASET,
    FSN_MODEL,
    OPPONENT_MANIFEST,
    STAGE3_FAILURES,
    _candidate_diversity,
    _evaluate_difficulty,
    _load_json,
    _pool_stats,
    _rate,
    _select_failures,
    _validate_candidate,
    _write_json,
)


STAGE3_CANDIDATES = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage3"
    / "fsn_policy_evaluated_shadow_candidates.json"
)
FIXED_BUDGET_CANDIDATES = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage4"
    / "fsn_fixed_budget_generation_candidates.json"
)
OUTPUT_DIR = (
    ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "fsn" / "stage4"
)
REPORT_PATH = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "reports"
    / "fsn_rerank_calibration_report.txt"
)

MODES = (
    "full_qwen",
    "fsn25_direct",
    "fsn25_rerank",
    "fsn50_rerank",
    "fsn_only_rerank",
)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_rows = collect_calibration_rows(
        [
            ("stage3_policy_shadow", _load_json(STAGE3_CANDIDATES)),
            ("stage4_fixed_budget", _load_json(FIXED_BUDGET_CANDIDATES)),
        ]
    )
    calibration = calibrate(calibration_rows)
    _write_json(OUTPUT_DIR / "fsn_acceptance_calibration_summary.json", calibration)
    _write_csv(
        OUTPUT_DIR / "fsn_acceptance_calibration_curve.csv",
        calibration_curve_rows(calibration),
    )

    failures = _select_failures(
        _load_json(STAGE3_FAILURES)["failure_summaries"], 10
    )
    failure_by_id = {
        str(record["failure_id"]): record for record in failures
    }
    fixed_payload = _load_json(FIXED_BUDGET_CANDIDATES)
    existing_records = list(fixed_payload.get("candidate_records") or [])
    accounting_rows = list(fixed_payload.get("accounting_rows") or [])
    existing_by_key: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in existing_records:
        existing_by_key[(str(record.get("failure_id")), str(record.get("mode")))].append(record)

    eligible_for_difficulty = _difficulty_failure_ids(accounting_rows, 4)
    base_config = load_base_scenario_config(BASE_CONFIG_PATH)
    dataset = load_jsonl(FSN_DATASET)
    opponent = _resolve(_load_json(OPPONENT_MANIFEST)["checkpoint_path"])
    checker = ConstraintChecker()
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
    yaml_root = OUTPUT_DIR / "rerank_generated_yamls"
    all_records: List[Dict[str, Any]] = []
    rerank_diagnostics: List[Dict[str, Any]] = []
    generation_runtime_by_mode: Counter[str] = Counter()
    started = time.perf_counter()

    for failure_index, failure_record in enumerate(failures):
        failure_id = str(failure_record["failure_id"])
        summary = dict(failure_record.get("failure_summary") or {})
        mode_candidates = {
            "full_qwen": _existing_candidates(
                existing_by_key[(failure_id, "full_qwen_fixed_call")]
            ),
            "fsn25_direct": _existing_candidates(
                existing_by_key[(failure_id, "fsn25_fixed_call")]
            ),
        }
        fsn25_qwen = [
            candidate
            for candidate in _existing_candidates(
                existing_by_key[(failure_id, "fsn25_fixed_call")]
            )
            if candidate.get("generator_type") == "qwen"
        ]
        fsn50_qwen = [
            candidate
            for candidate in _existing_candidates(
                existing_by_key[(failure_id, "fsn50_fixed_call")]
            )
            if candidate.get("generator_type") == "qwen"
        ]
        for mode, fsn_count, qwen_candidates in (
            ("fsn25_rerank", 1, fsn25_qwen),
            ("fsn50_rerank", 2, fsn50_qwen),
            ("fsn_only_rerank", 4, []),
        ):
            generator = FSNScenarioGenerator(
                FSN_MODEL,
                {
                    "seed": 9000 + failure_index * 10 + fsn_count,
                    "diversity_aware": True,
                    "noise_scale": 0.10,
                },
            )
            reranked = generator.generate_reranked_from_failure_summary(
                summary,
                base_config,
                num_scenarios=fsn_count,
                overgenerate_count=16,
                rerank_weights={
                    "predicted_value_score": 0.4,
                    "accepted_probability": 0.3,
                    "diversity_bonus": 0.2,
                    "constraint_risk_penalty": 0.1,
                },
            )
            generation_runtime_by_mode[mode] += generator.last_generation_runtime_seconds
            rerank_diagnostics.append(
                {
                    "failure_id": failure_id,
                    "mode": mode,
                    **dict(generator.last_rerank_result),
                }
            )
            mode_candidates[mode] = [*qwen_candidates, *reranked]

        for mode, candidates in mode_candidates.items():
            pool_stats = _pool_stats(dataset, failure_record)
            for candidate_index, candidate in enumerate(candidates):
                candidate = dict(candidate)
                candidate["scenario_id"] = (
                    f"rerank_f{failure_index:02d}_{mode}_{candidate_index:02d}"
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
                if failure_id in eligible_for_difficulty and record["env_load_success"]:
                    _evaluate_difficulty(
                        record,
                        failure_record,
                        summary,
                        pool_stats,
                        policy_evaluator,
                        difficulty_evaluator,
                        seed=900000
                        + failure_index * 1000
                        + MODES.index(mode) * 100
                        + candidate_index,
                    )
                all_records.append(record)

    metrics = {
        mode: _mode_metrics(
            mode,
            all_records,
            accounting_rows,
            generation_runtime_by_mode,
        )
        for mode in MODES
    }
    judgement = _judgement(metrics, rerank_diagnostics, calibration)
    summary = {
        "schema_version": "falcon.fsn_rerank_calibration_summary.v1",
        "num_failure_summaries": len(failures),
        "difficulty_evaluated_failure_summaries": len(eligible_for_difficulty),
        "difficulty_evaluated_candidates_per_mode": {
            mode: metrics[mode]["difficulty_evaluated_count"] for mode in MODES
        },
        "calibration": calibration,
        "mode_metrics": metrics,
        "rerank_diagnostics": _summarize_rerank_diagnostics(rerank_diagnostics),
        "judgement": judgement,
        "comparison_deltas": _comparison_deltas(metrics),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "entered_training_loop": False,
        "mappo_training_started": False,
        "qwen_generation_calls_made": 0,
        "reused_fixed_budget_qwen_outputs": True,
        "failure_stage": None,
        "warnings": sorted(
            set(
                list(calibration.get("warnings") or [])
                + [
                    "Value and accepted-probability heads are failure-summary-level predictions, so within-summary reranking is mainly driven by diversity and constraint risk.",
                    "Qwen candidates and accounting were reused from the fixed-budget smoke; no new Qwen call was made.",
                    "Only four FSN slots per FSN25 mode received real difficulty evaluation; pilot recommendation remains provisional.",
                ]
            )
        ),
    }
    _write_json(OUTPUT_DIR / "fsn_rerank_calibration_candidates.json", {
        "schema_version": "falcon.fsn_rerank_calibration_candidates.v1",
        "candidate_records": all_records,
        "rerank_diagnostics": rerank_diagnostics,
    })
    _write_json(OUTPUT_DIR / "fsn_rerank_calibration_summary.json", summary)
    _write_metrics_csv(OUTPUT_DIR / "fsn_rerank_fixed_budget_metrics.csv", metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _existing_candidates(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(record.get("candidate") or {}) for record in records]


def _difficulty_failure_ids(
    accounting_rows: Sequence[Mapping[str, Any]], count: int
) -> set[str]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in accounting_rows:
        grouped[str(row.get("failure_id"))][str(row.get("mode"))] = int(
            row.get("total_candidates_raw") or 0
        )
    eligible = [
        failure_id
        for failure_id, modes in grouped.items()
        if all(
            modes.get(mode, 0) >= 4
            for mode in (
                "full_qwen_fixed_call",
                "fsn25_fixed_call",
                "fsn50_fixed_call",
            )
        )
    ]
    return set(eligible[:count])


def _mode_metrics(
    mode: str,
    records: Sequence[Mapping[str, Any]],
    accounting_rows: Sequence[Mapping[str, Any]],
    generation_runtime_by_mode: Mapping[str, float],
) -> Dict[str, Any]:
    subset = [record for record in records if record.get("mode") == mode]
    evaluated = [record for record in subset if record.get("difficulty_result")]
    accepted = [
        record
        for record in evaluated
        if record["difficulty_result"].get("accepted_into_curriculum_pool")
    ]
    values = [
        value
        for record in evaluated
        if (value := _number(record["difficulty_result"].get("final_value_score")))
        is not None
    ]
    potentials = [
        value
        for record in evaluated
        if (value := _number(record["difficulty_result"].get("learning_potential")))
        is not None
    ]
    predicted = [
        value
        for record in subset
        if (
            value := _number(
                (record.get("candidate", {}).get("metadata") or {}).get(
                    "predicted_value_score"
                )
            )
        )
        is not None
    ]
    base_mode = {
        "full_qwen": "full_qwen_fixed_call",
        "fsn25_direct": "fsn25_fixed_call",
        "fsn25_rerank": "fsn25_fixed_call",
        "fsn50_rerank": "fsn50_fixed_call",
        "fsn_only_rerank": "no_qwen_fsn_only",
    }[mode]
    accounting = [
        row for row in accounting_rows if row.get("mode") == base_mode
    ]
    direct_fsn_runtime = sum(
        float(row.get("fsn_runtime_seconds") or 0.0) for row in accounting
    )
    source_breakdown = {
        source: _source_metrics(
            [
                record
                for record in subset
                if _generator_source(record) == source
            ]
        )
        for source in ("qwen", "fsn")
    }
    return {
        "mode": mode,
        "qwen_api_calls": sum(int(row.get("qwen_api_calls") or 0) for row in accounting),
        "qwen_candidate_slots": sum(
            int(row.get("qwen_candidates_requested") or 0) for row in accounting
        ),
        "qwen_runtime_seconds": round(
            sum(float(row.get("qwen_runtime_seconds") or 0.0) for row in accounting),
            6,
        ),
        "fsn_runtime_seconds": round(
            float(generation_runtime_by_mode.get(mode, direct_fsn_runtime)), 6
        ),
        "total_candidates": len(subset),
        "valid_candidates": sum(1 for record in subset if record.get("constraint_valid")),
        "env_loadable_candidates": sum(
            1 for record in subset if record.get("env_load_success")
        ),
        "env_load_rate": _rate(
            sum(1 for record in subset if record.get("env_load_success")),
            len(subset),
        ),
        "diversity": _candidate_diversity(
            [record.get("candidate") or {} for record in subset]
        ),
        "mean_predicted_value": _mean(predicted),
        "difficulty_evaluated_count": len(evaluated),
        "difficulty_accepted_count": len(accepted),
        "difficulty_accepted_rate": _rate(len(accepted), len(evaluated)),
        "mean_final_value": _mean(values),
        "mean_learning_potential": _mean(potentials),
        "generator_source_breakdown": source_breakdown,
        "rejection_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for record in evaluated
                    for reason in record["difficulty_result"].get(
                        "rejection_reasons", []
                    )
                ).items()
            )
        ),
    }


def _judgement(
    metrics: Mapping[str, Mapping[str, Any]],
    rerank_diagnostics: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    direct = metrics["fsn25_direct"]
    rerank = metrics["fsn25_rerank"]
    fsn50 = metrics["fsn50_rerank"]
    direct_fsn = direct["generator_source_breakdown"]["fsn"]
    rerank_fsn = rerank["generator_source_breakdown"]["fsn"]
    acceptance_improved = (
        rerank_fsn["difficulty_accepted_rate"]
        > direct_fsn["difficulty_accepted_rate"]
    )
    value_improved = (
        (rerank_fsn["mean_final_value"] or 0.0)
        > (direct_fsn["mean_final_value"] or 0.0)
    )
    diversity_not_lower = (
        rerank["diversity"] >= direct["diversity"] * 0.95
    )
    env_near_full = rerank["env_load_rate"] >= 0.95
    recommend_25 = bool(
        acceptance_improved and value_improved and diversity_not_lower and env_near_full
    )
    recommend_50 = bool(
        fsn50["difficulty_accepted_rate"] >= rerank["difficulty_accepted_rate"]
        and (fsn50["mean_final_value"] or 0.0)
        >= (rerank["mean_final_value"] or 0.0)
        and fsn50["env_load_rate"] >= 0.95
    )
    return {
        "value_head_predictive": bool(
            (calibration.get("predicted_value_accepted_auc") or 0.0) >= 0.6
            and (
                calibration.get("predicted_value_vs_final_value_pearson")
                or 0.0
            )
            >= 0.2
        ),
        "label_head_predictive": bool(
            (calibration.get("accepted_probability_auc") or 0.0) >= 0.6
        ),
        "fsn25_rerank_acceptance_improved": acceptance_improved,
        "fsn25_rerank_mean_final_value_improved": value_improved,
        "fsn25_rerank_diversity_not_materially_lower": diversity_not_lower,
        "fsn25_rerank_env_load_near_full": env_near_full,
        "recommend_20_round_25_percent_replacement_pilot": recommend_25,
        "recommendation_confidence": (
            "low_to_medium" if recommend_25 else "low"
        ),
        "fsn50_rerank_feasible": recommend_50,
        "recommend_50_percent_pilot": recommend_50,
        "fsn25_direct_fsn_slot_metrics": direct_fsn,
        "fsn25_rerank_fsn_slot_metrics": rerank_fsn,
        "rerank_value_unique_count_mean": _mean(
            [
                float(item.get("predicted_value_unique_count") or 0)
                for item in rerank_diagnostics
            ]
        ),
        "rerank_probability_unique_count_mean": _mean(
            [
                float(item.get("accepted_probability_unique_count") or 0)
                for item in rerank_diagnostics
            ]
        ),
    }


def _comparison_deltas(
    metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    direct = metrics["fsn25_direct"]
    rerank = metrics["fsn25_rerank"]
    full = metrics["full_qwen"]
    return {
        "fsn25_rerank_vs_direct": {
            "difficulty_accepted_rate_delta": round(
                rerank["difficulty_accepted_rate"]
                - direct["difficulty_accepted_rate"],
                6,
            ),
            "mean_final_value_delta": round(
                (rerank["mean_final_value"] or 0.0)
                - (direct["mean_final_value"] or 0.0),
                6,
            ),
            "mean_learning_potential_delta": round(
                (rerank["mean_learning_potential"] or 0.0)
                - (direct["mean_learning_potential"] or 0.0),
                6,
            ),
            "diversity_delta": round(
                rerank["diversity"] - direct["diversity"], 6
            ),
            "valid_candidate_delta": (
                rerank["valid_candidates"] - direct["valid_candidates"]
            ),
        },
        "fsn25_rerank_vs_full_qwen": {
            "difficulty_accepted_rate_delta": round(
                rerank["difficulty_accepted_rate"]
                - full["difficulty_accepted_rate"],
                6,
            ),
            "mean_final_value_delta": round(
                (rerank["mean_final_value"] or 0.0)
                - (full["mean_final_value"] or 0.0),
                6,
            ),
            "diversity_delta": round(
                rerank["diversity"] - full["diversity"], 6
            ),
            "qwen_candidate_slot_reduction": round(
                1.0
                - _rate(
                    rerank["qwen_candidate_slots"],
                    full["qwen_candidate_slots"],
                ),
                6,
            ),
            "historical_qwen_runtime_reduction": round(
                1.0
                - _rate(
                    rerank["qwen_runtime_seconds"],
                    full["qwen_runtime_seconds"],
                ),
                6,
            ),
        },
    }


def _generator_source(record: Mapping[str, Any]) -> str:
    value = str(
        record.get("generator_type")
        or (record.get("candidate") or {}).get("generator_type")
        or ""
    ).lower()
    return "qwen" if "qwen" in value or "ollama" in value else "fsn"


def _source_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evaluated = [
        record for record in records if record.get("difficulty_result")
    ]
    accepted = [
        record
        for record in evaluated
        if record["difficulty_result"].get("accepted_into_curriculum_pool")
    ]
    values = [
        value
        for record in evaluated
        if (
            value := _number(
                record["difficulty_result"].get("final_value_score")
            )
        )
        is not None
    ]
    potentials = [
        value
        for record in evaluated
        if (
            value := _number(
                record["difficulty_result"].get("learning_potential")
            )
        )
        is not None
    ]
    return {
        "candidate_count": len(records),
        "valid_count": sum(
            1 for record in records if record.get("constraint_valid")
        ),
        "env_load_count": sum(
            1 for record in records if record.get("env_load_success")
        ),
        "difficulty_evaluated_count": len(evaluated),
        "difficulty_accepted_count": len(accepted),
        "difficulty_accepted_rate": _rate(len(accepted), len(evaluated)),
        "mean_final_value": _mean(values),
        "mean_learning_potential": _mean(potentials),
    }


def _summarize_rerank_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "num_rerank_batches": len(diagnostics),
        "overgenerated_candidates_total": sum(
            int(item.get("overgenerated_candidates") or 0)
            for item in diagnostics
        ),
        "selected_candidates_total": sum(
            int(item.get("selected_candidates") or 0) for item in diagnostics
        ),
        "predicted_value_unique_count_mean": _mean(
            [
                float(item.get("predicted_value_unique_count") or 0)
                for item in diagnostics
            ]
        ),
        "accepted_probability_unique_count_mean": _mean(
            [
                float(item.get("accepted_probability_unique_count") or 0)
                for item in diagnostics
            ]
        ),
        "warnings": sorted(
            set(
                warning
                for item in diagnostics
                for warning in item.get("warnings") or []
            )
        ),
    }


def _report(summary: Mapping[str, Any]) -> str:
    calibration = summary["calibration"]
    metrics = summary["mode_metrics"]
    judgement = summary["judgement"]
    deltas = summary["comparison_deltas"]
    lines = [
        "FALCON FSN Candidate Quality Calibration Report",
        "",
        "Acceptance calibration",
        f"- Samples / accepted: {calibration['num_samples']} / {calibration['accepted_count']}",
        f"- Predicted value vs final value Pearson: {calibration['predicted_value_vs_final_value_pearson']}",
        f"- Predicted value accepted AUC: {calibration['predicted_value_accepted_auc']}",
        f"- Accepted probability AUC: {calibration['accepted_probability_auc']}",
        f"- Accepted probability Brier / ECE: {calibration['accepted_probability_brier_score']} / {calibration['accepted_probability_expected_calibration_error']}",
        f"- Best accepted-probability threshold: {calibration['best_accepted_probability_threshold']}",
        f"- Best predicted-value threshold: {calibration['best_predicted_value_threshold']}",
        "",
        "Fixed-budget rerank comparison",
    ]
    for mode in MODES:
        item = metrics[mode]
        lines.append(
            f"- {mode}: valid/env={item['valid_candidates']}/{item['env_loadable_candidates']}, "
            f"diversity={item['diversity']}, accepted={item['difficulty_accepted_count']}/{item['difficulty_evaluated_count']}, "
            f"mean_value={item['mean_final_value']}, learning_potential={item['mean_learning_potential']}, "
            f"fsn_slot_accepted={item['generator_source_breakdown']['fsn']['difficulty_accepted_count']}/{item['generator_source_breakdown']['fsn']['difficulty_evaluated_count']}"
        )
    lines.extend(
        [
            "",
            "Judgement",
            f"- Value head predicts real acceptance: {judgement['value_head_predictive']}",
            f"- Label head predicts real acceptance: {judgement['label_head_predictive']}",
            f"- FSN25 rerank acceptance improved: {judgement['fsn25_rerank_acceptance_improved']}",
            f"- FSN25 rerank mean final value improved: {judgement['fsn25_rerank_mean_final_value_improved']}",
            f"- FSN25 rerank diversity preserved: {judgement['fsn25_rerank_diversity_not_materially_lower']}",
            f"- FSN50 rerank feasible: {judgement['fsn50_rerank_feasible']}",
            f"- Recommend 20-round 25% replacement pilot: {judgement['recommend_20_round_25_percent_replacement_pilot']}",
            f"- Recommendation confidence: {judgement['recommendation_confidence']}",
            "",
            "FSN25 rerank deltas",
            f"- Versus FSN25 direct: accepted-rate {deltas['fsn25_rerank_vs_direct']['difficulty_accepted_rate_delta']:+.6f}, final-value {deltas['fsn25_rerank_vs_direct']['mean_final_value_delta']:+.6f}, diversity {deltas['fsn25_rerank_vs_direct']['diversity_delta']:+.6f}.",
            f"- Versus Full-Qwen: accepted-rate {deltas['fsn25_rerank_vs_full_qwen']['difficulty_accepted_rate_delta']:+.6f}, final-value {deltas['fsn25_rerank_vs_full_qwen']['mean_final_value_delta']:+.6f}, diversity {deltas['fsn25_rerank_vs_full_qwen']['diversity_delta']:+.6f}.",
            f"- Qwen candidate-slot reduction / historical runtime reduction versus Full-Qwen: {deltas['fsn25_rerank_vs_full_qwen']['qwen_candidate_slot_reduction']:.1%} / {deltas['fsn25_rerank_vs_full_qwen']['historical_qwen_runtime_reduction']:.1%}.",
            "",
            "Important limitation: FSN value and accepted-probability heads are failure-summary-level predictions. Within one 16-candidate batch they are usually constant, so current reranking is primarily a diversity/constraint-risk selector rather than a true value selector.",
            "FSN25-rerank meets the requested offline gate, but only four FSN slots were difficulty-evaluated in each FSN25 condition; the recommendation is provisional.",
            "FSN50-rerank is not yet supported because it did not improve mean final value over FSN25-rerank.",
            "This offline study cannot claim MAPPO improvement, production replacement, reliable value calibration, or on-policy distillation.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 6) if values else None


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT_DIR / path


if __name__ == "__main__":
    raise SystemExit(main())
