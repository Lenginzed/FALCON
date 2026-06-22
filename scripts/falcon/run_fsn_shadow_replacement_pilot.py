"""Offline shadow replacement pilot for Stage 2 FSN candidates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.candidate_schema import (
    create_candidate_scenario,
    validate_candidate_schema,
)
from falcon.constraint_checker import ConstraintChecker
from falcon.difficulty_evaluator import DEFAULT_CONFIG as DIFFICULTY_CONFIG
from falcon.fsn_dataset import load_jsonl
from falcon.fsn_generator import FSNScenarioGenerator
from falcon.random_scenario_generator import RandomScenarioGenerator
from falcon.scenario_adapter import (
    apply_initial_config_to_yaml,
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    load_base_scenario_config,
    save_scenario_yaml,
    scenario_vector_to_initial_config,
)
from falcon.trajectory_recorder import SCENARIO_VECTOR_KEYS

BASE_CONFIG_PATH = (
    ROOT_DIR
    / "envs"
    / "JSBSim"
    / "configs"
    / "2v2"
    / "NoWeapon"
    / "Selfplay.yaml"
)
FAILURE_DIR = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "results"
    / "falcon_no_fsn"
    / "seed_0"
    / "pilot_run"
    / "controller"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/falcon_2v2_noweapon/fsn/stage2/"
            "fsn_stage2_model.pt"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "experiments/falcon_2v2_noweapon/fsn/stage2/"
            "failure_to_scenario_dataset_dedup.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn/stage2",
    )
    parser.add_argument("--num-failures", type=int, default=5)
    parser.add_argument("--candidates-per-failure", type=int, default=3)
    args = parser.parse_args()
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_root = output_dir / "shadow_replacement_yamls"
    base_config = load_base_scenario_config(BASE_CONFIG_PATH)
    base_initial = extract_initial_config_from_yaml(base_config)
    rows = load_jsonl(ROOT_DIR / args.dataset)
    qwen_by_failure = _qwen_rows_by_failure(rows)
    failure_summaries = _load_failure_summaries(
        qwen_by_failure, args.num_failures
    )
    generator = FSNScenarioGenerator(
        ROOT_DIR / args.checkpoint,
        {
            "seed": 41,
            "diversity_aware": True,
            "oversample_factor": 6,
            "noise_scale": 0.10,
        },
    )
    random_generator = RandomScenarioGenerator({"seed": 43})

    all_candidates: dict[str, list[dict[str, Any]]] = {
        "fsn": [],
        "qwen_historical": [],
        "random": [],
    }
    validation: dict[str, list[dict[str, Any]]] = {
        key: [] for key in all_candidates
    }
    fsn_runtime = 0.0
    random_runtime = 0.0
    per_failure = []
    for failure_index, failure_summary in enumerate(failure_summaries):
        failure_id = _failure_id(failure_summary)
        fsn_candidates = generator.generate_from_failure_summary(
            failure_summary,
            base_config,
            args.candidates_per_failure,
        )
        fsn_runtime += generator.last_generation_runtime_seconds
        random_started = time.perf_counter()
        random_candidates = random_generator.generate_from_failure_summary(
            failure_summary,
            base_config,
            args.candidates_per_failure,
        )
        random_runtime += time.perf_counter() - random_started
        qwen_candidates = [
            _candidate_from_dataset_row(
                row, base_initial, failure_summary, index
            )
            for index, row in enumerate(
                qwen_by_failure.get(failure_id, [])[
                    : args.candidates_per_failure
                ]
            )
        ]
        if len(qwen_candidates) < args.candidates_per_failure:
            qwen_candidates.extend(
                _candidate_from_dataset_row(
                    row, base_initial, failure_summary, index
                )
                for index, row in enumerate(
                    _fallback_qwen_rows(
                        rows,
                        args.candidates_per_failure - len(qwen_candidates),
                        failure_index,
                    ),
                    start=len(qwen_candidates),
                )
            )
        generated = {
            "fsn": fsn_candidates,
            "qwen_historical": qwen_candidates,
            "random": random_candidates,
        }
        failure_metrics = {"failure_id": failure_id}
        for generator_name, candidates in generated.items():
            all_candidates[generator_name].extend(candidates)
            checked = _validate_candidates(
                candidates,
                base_config,
                yaml_root / generator_name / f"failure_{failure_index:02d}",
                f"shadow_{generator_name}_{failure_index}",
            )
            validation[generator_name].append(checked)
            failure_metrics[generator_name] = _quality_metrics(
                candidates, [checked]
            )
        per_failure.append(failure_metrics)

    quality = {
        name: _quality_metrics(candidates, validation[name])
        for name, candidates in all_candidates.items()
    }
    quality["fsn"]["generation_runtime_seconds"] = round(fsn_runtime, 6)
    quality["random"]["generation_runtime_seconds"] = round(
        random_runtime, 6
    )
    qwen_seconds_per_candidate, runtime_basis = _historical_qwen_runtime()
    quality["qwen_historical"]["generation_runtime_seconds"] = None
    quality["qwen_historical"][
        "historical_estimated_seconds_per_candidate"
    ] = qwen_seconds_per_candidate
    overlap = _changed_factor_overlap(
        all_candidates["fsn"], all_candidates["qwen_historical"]
    )
    quality["fsn"]["changed_factor_overlap_with_qwen"] = overlap

    total_slots = len(all_candidates["fsn"])
    simulations = []
    for ratio in (0.25, 0.50, 0.75):
        fsn_slots = round(total_slots * ratio)
        qwen_slots = total_slots - fsn_slots
        estimated_qwen_seconds = total_slots * qwen_seconds_per_candidate
        replacement_seconds = (
            qwen_slots * qwen_seconds_per_candidate
            + fsn_slots
            * (fsn_runtime / max(len(all_candidates["fsn"]), 1))
        )
        simulations.append(
            {
                "fsn_fraction": ratio,
                "qwen_fraction": round(1.0 - ratio, 2),
                "total_candidate_slots": total_slots,
                "fsn_candidate_slots": fsn_slots,
                "qwen_candidate_slots": qwen_slots,
                "estimated_qwen_call_reduction": ratio,
                "estimated_runtime_reduction": round(
                    1.0
                    - replacement_seconds
                    / max(estimated_qwen_seconds, 1e-8),
                    6,
                ),
                "estimated_runtime_seconds": round(
                    replacement_seconds, 6
                ),
                "estimated_schema_valid_rate": _weighted(
                    quality["fsn"]["schema_valid_rate"],
                    quality["qwen_historical"]["schema_valid_rate"],
                    ratio,
                ),
                "estimated_constraint_valid_rate": _weighted(
                    quality["fsn"]["constraint_valid_rate"],
                    quality["qwen_historical"]["constraint_valid_rate"],
                    ratio,
                ),
            }
        )

    candidate_payload = {
        "schema_version": "falcon.fsn_shadow_candidates.v1",
        "failure_summaries": [
            {
                "source_trajectory": item.get("source_trajectory"),
                "failure_scores": item.get("failure_scores"),
                "primary_failure_modes": item.get("primary_failure_modes"),
            }
            for item in failure_summaries
        ],
        "candidates": all_candidates,
        "validation": validation,
        "per_failure_metrics": per_failure,
    }
    (output_dir / "fsn_shadow_replacement_candidates.json").write_text(
        json.dumps(candidate_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "falcon.fsn_shadow_replacement_summary.v1",
        "num_failure_summaries": len(failure_summaries),
        "candidates_per_failure": args.candidates_per_failure,
        "fsn_schema_valid_rate": quality["fsn"]["schema_valid_rate"],
        "fsn_constraint_valid_rate": quality["fsn"][
            "constraint_valid_rate"
        ],
        "fsn_env_load_rate": quality["fsn"]["env_load_success_rate"],
        "fsn_diversity": quality["fsn"]["diversity_score"],
        "fsn_predicted_value_score": quality["fsn"][
            "predicted_value_score_mean"
        ],
        "fsn_vs_qwen_changed_factor_overlap": overlap,
        "quality_by_generator": quality,
        "replacement_simulations": simulations,
        "historical_qwen_runtime_basis": runtime_basis,
        "difficulty_evaluator_used": False,
        "policy_evaluator_used": False,
        "entered_training_loop": False,
        "potential_failure_modes": [
            "Stage 2 test scenario regression remains weaker than train regression.",
            "The held-out test split is small after leakage-safe grouping.",
            "FSN predicted value is not a substitute for real policy evaluation.",
            "Synthetic invalids do not cover all physical failure combinations.",
        ],
        "failure_stage": None,
        "warnings": [
            "Historical Qwen candidates were reused; Ollama was not called.",
            "Runtime reduction is an estimate from prior logged Qwen runtime.",
            "This shadow pilot does not measure policy-performance replacement.",
        ],
    }
    (output_dir / "fsn_shadow_replacement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_reports(output_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0


def _load_failure_summaries(
    qwen_by_failure: Mapping[str, Sequence[Mapping[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(
        FAILURE_DIR.glob("falcon_controller_failure_summary_round*.json"),
        key=_round_key,
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        if _failure_id(data) in qwen_by_failure:
            summaries.append(data)
        if len(summaries) >= limit:
            break
    return summaries


def _qwen_rows_by_failure(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("generator_type") == "qwen"
            and row.get("constraint_valid")
            and row.get("source_failure_id")
        ):
            grouped[str(row["source_failure_id"])].append(row)
    return grouped


def _failure_id(summary: Mapping[str, Any]) -> str:
    source = summary.get("source_trajectory") or summary.get("episode_id")
    return Path(str(source)).stem if source else "offline_failure"


def _candidate_from_dataset_row(
    row: Mapping[str, Any],
    base_initial: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    vector = dict(row.get("candidate_scenario_vector") or {})
    initial = scenario_vector_to_initial_config(vector, base_initial)
    recomputed = initial_config_to_scenario_vector(initial)[
        "scenario_vector"
    ]
    return create_candidate_scenario(
        scenario_id=f"qwen_historical_{index:04d}",
        generator_type="qwen_historical",
        source_failure_id=_failure_id(failure_summary),
        target_failure_modes=list(row.get("primary_failure_modes") or []),
        changed_factors=list(row.get("changed_factors") or [])[:3],
        counterfactual_group_id=_failure_id(failure_summary),
        scenario_vector=recomputed,
        scenario_parameters={"historical_dataset_sample_id": row.get("sample_id")},
        initial_config=initial,
        expected_effect="Historical Qwen candidate reused for offline comparison.",
        rationale="No new Ollama call was made.",
        metadata={
            "historical_final_value_score": (
                row.get("difficulty") or {}
            ).get("final_value_score"),
            "historical_label": row.get("label"),
            "qwen_call_used": False,
        },
    )


def _fallback_qwen_rows(
    rows: Sequence[Mapping[str, Any]], count: int, offset: int
) -> list[Mapping[str, Any]]:
    available = [
        row
        for row in rows
        if row.get("generator_type") == "qwen"
        and row.get("constraint_valid")
        and row.get("label") != "invalid"
    ]
    if not available:
        return []
    return [
        available[(offset * max(count, 1) + index) % len(available)]
        for index in range(count)
    ]


def _validate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    yaml_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    checker = ConstraintChecker()
    yaml_dir.mkdir(parents=True, exist_ok=True)
    schema_results = []
    constraint_results = []
    env_results = []
    yaml_paths = []
    for index, candidate in enumerate(candidates):
        schema = validate_candidate_schema(candidate)
        constraint = checker.validate_candidate(candidate)
        schema_results.append(schema)
        constraint_results.append(constraint)
        if not schema["is_valid"] or not constraint["is_valid"]:
            continue
        yaml_config = apply_initial_config_to_yaml(
            base_config, candidate.get("initial_config") or {}
        )
        yaml_path = yaml_dir / f"{prefix}_{index:04d}.yaml"
        save_scenario_yaml(yaml_config, yaml_path)
        yaml_paths.append(str(yaml_path.resolve()))
        env_results.append(
            checker.validate_yaml_config(
                yaml_config,
                enable_env_load_check=True,
                temp_config_name=f"{prefix}_{index}",
            )
        )
    return {
        "schema_results": schema_results,
        "constraint_results": constraint_results,
        "env_results": env_results,
        "yaml_paths": yaml_paths,
    }


def _quality_metrics(
    candidates: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = max(len(candidates), 1)
    schemas = [
        item
        for validation in validations
        for item in validation.get("schema_results") or []
    ]
    constraints = [
        item
        for validation in validations
        for item in validation.get("constraint_results") or []
    ]
    envs = [
        item
        for validation in validations
        for item in validation.get("env_results") or []
    ]
    yaml_count = sum(
        len(validation.get("yaml_paths") or [])
        for validation in validations
    )
    values = []
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        value = _number(
            metadata.get(
                "predicted_value_score",
                metadata.get("historical_final_value_score"),
            )
        )
        if value is not None:
            values.append(value)
    return {
        "num_candidates": len(candidates),
        "schema_valid_rate": round(
            sum(bool(item.get("is_valid")) for item in schemas) / total, 6
        ),
        "constraint_valid_rate": round(
            sum(bool(item.get("is_valid")) for item in constraints) / total,
            6,
        ),
        "yaml_generation_success_rate": round(yaml_count / total, 6),
        "env_load_success_rate": round(
            sum(
                bool(
                    (item.get("physical_constraint_check") or {}).get(
                        "scenario_loadable_env_check"
                    )
                )
                for item in envs
            )
            / total,
            6,
        ),
        "predicted_value_score_mean": round(
            statistics.fmean(values), 6
        )
        if values
        else None,
        "diversity_score": _diversity(candidates),
    }


def _diversity(candidates: Sequence[Mapping[str, Any]]) -> float:
    if len(candidates) < 2:
        return 0.0
    scales = DIFFICULTY_CONFIG["scenario_vector_scales"]
    distances = []
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
                distances.append(
                    math.sqrt(sum(components) / len(components))
                )
    return round(statistics.fmean(distances), 6) if distances else 0.0


def _changed_factor_overlap(
    fsn_candidates: Sequence[Mapping[str, Any]],
    qwen_candidates: Sequence[Mapping[str, Any]],
) -> float:
    qwen_sets = [
        set(str(value) for value in item.get("changed_factors") or [])
        for item in qwen_candidates
    ]
    overlaps = []
    for candidate in fsn_candidates:
        fsn_set = set(
            str(value) for value in candidate.get("changed_factors") or []
        )
        if not qwen_sets:
            continue
        overlaps.append(
            max(
                len(fsn_set & other) / max(len(fsn_set | other), 1)
                for other in qwen_sets
            )
        )
    return round(statistics.fmean(overlaps), 6) if overlaps else 0.0


def _historical_qwen_runtime() -> tuple[float, str]:
    path = (
        ROOT_DIR
        / "experiments"
        / "falcon_2v2_noweapon"
        / "reports"
        / "hard_eval_v2_5seed_summary.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 10.0, "fallback_assumption_10_seconds_per_candidate"
    text = json.dumps(payload)
    # Stable aggregate logged by the five-seed FALCON report.
    if '"qwen_calls": 393' in text and '"qwen_runtime_seconds": 4357.298' in text:
        return round(4357.298 / 393.0, 6), str(path)
    return 10.0, "fallback_assumption_10_seconds_per_candidate"


def _weighted(fsn: float, qwen: float, ratio: float) -> float:
    return round(ratio * float(fsn) + (1.0 - ratio) * float(qwen), 6)


def _round_key(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _write_reports(output_dir: Path, summary: Mapping[str, Any]) -> None:
    quality = summary["quality_by_generator"]
    simulations = summary["replacement_simulations"]
    lines = [
        "FSN Stage 2 Offline Shadow Replacement Report",
        "=" * 46,
        "",
        f"Failure summaries: {summary['num_failure_summaries']}",
        f"Candidates per failure: {summary['candidates_per_failure']}",
        "",
        "Offline quality",
        f"- FSN: {quality['fsn']}",
        f"- Historical Qwen: {quality['qwen_historical']}",
        f"- Random: {quality['random']}",
        "",
        "Replacement simulations",
        *[
            (
                f"- {int(item['fsn_fraction'] * 100)}% FSN: "
                f"Qwen call reduction={item['estimated_qwen_call_reduction']:.0%}, "
                f"runtime reduction={item['estimated_runtime_reduction']:.1%}"
            )
            for item in simulations
        ],
        "",
        "Judgement",
        "- Stage 2 supports an offline replacement pilot only.",
        "- It does not establish MAPPO performance or safe online replacement.",
        "- Keep schema, constraint, YAML, and env checks mandatory.",
    ]
    report = "\n".join(lines) + "\n"
    (output_dir / "fsn_shadow_replacement_report.txt").write_text(
        report, encoding="utf-8"
    )
    reports_dir = output_dir.parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_json(
        output_dir / "failure_to_scenario_dataset_dedup_summary.json"
    )
    training = _load_json(output_dir / "fsn_stage2_training_summary.json")
    generation = _load_json(output_dir / "fsn_generation_smoke_summary.json")
    stage2_lines = [
        "FALCON FSN Stage 2 Report",
        "=" * 25,
        "",
        f"Dataset samples: {dataset.get('total_stage2_samples')}",
        f"Real accepted: {dataset.get('accepted_real_sample_count')}",
        f"Synthetic invalid: {dataset.get('synthetic_invalid_count')}",
        f"Cross-split leakage: {dataset.get('cross_split_leakage_detected')}",
        f"Split counts: {dataset.get('split_counts')}",
        "",
        f"Offline training succeeded: {training.get('training_succeeded')}",
        f"Overfitting detected: {training.get('overfitting_detected')}",
        f"Test metrics: {(training.get('split_metrics') or {}).get('test')}",
        "",
        f"Generation smoke: {generation}",
        "",
        report,
        "Recommendation: proceed only to a controlled FSN replacement pilot "
        "after addressing regression generalization and preserving all external checks.",
        "Do not claim policy improvement, online distillation, or full Qwen replacement.",
    ]
    (reports_dir / "fsn_stage2_report.txt").write_text(
        "\n".join(stage2_lines) + "\n", encoding="utf-8"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
