"""Validate offline FSN generation against schema, constraints, YAML, and env."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema
from falcon.constraint_checker import ConstraintChecker
from falcon.difficulty_evaluator import DEFAULT_CONFIG as DIFFICULTY_CONFIG
from falcon.fsn_generator import FSNScenarioGenerator
from falcon.random_scenario_generator import RandomScenarioGenerator
from falcon.scenario_adapter import (
    apply_initial_config_to_yaml,
    load_base_scenario_config,
    save_scenario_yaml,
)
from falcon.trajectory_recorder import SCENARIO_VECTOR_KEYS

BASE_CONFIG_PATH = (
    ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
)
DEFAULT_FAILURE = (
    ROOT_DIR
    / "tests"
    / "tmp_falcon_trajectories"
    / "falcon_coordination_failure_v2_analysis.json"
)
QWEN_CANDIDATES = (
    ROOT_DIR
    / "tests"
    / "tmp_falcon_trajectories"
    / "ollama_qwen8b_candidate_scenarios.json"
)
QWEN_DIFFICULTY = (
    ROOT_DIR
    / "tests"
    / "tmp_falcon_trajectories"
    / "ollama_qwen8b_difficulty_evaluation.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="experiments/falcon_2v2_noweapon/fsn/fsn_offline_smoke.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn",
    )
    parser.add_argument("--num-scenarios", type=int, default=6)
    args = parser.parse_args()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_dir / "fsn_generated_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    failure_summary = _load_json(DEFAULT_FAILURE)
    base_config = load_base_scenario_config(BASE_CONFIG_PATH)
    generator = FSNScenarioGenerator(ROOT_DIR / args.checkpoint)
    candidates = generator.generate_from_failure_summary(
        failure_summary, base_config, num_scenarios=args.num_scenarios
    )
    (output_dir / "fsn_generated_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": "falcon.fsn_generated_candidates.v1",
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fsn_validation = _validate_candidates(
        candidates, base_config, yaml_dir, "fsn", enable_env=True
    )
    (output_dir / "fsn_validated_candidates.json").write_text(
        json.dumps(fsn_validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    random_started = time.perf_counter()
    random_candidates = RandomScenarioGenerator({"seed": 23}).generate_from_failure_summary(
        failure_summary, base_config, args.num_scenarios
    )
    random_runtime = time.perf_counter() - random_started
    random_validation = _validate_candidates(
        random_candidates,
        base_config,
        output_dir / "offline_comparison_random_yamls",
        "random",
        enable_env=True,
    )

    qwen_payload = _load_json(QWEN_CANDIDATES)
    qwen_candidates = list(
        qwen_payload.get("valid_candidates")
        or qwen_payload.get("parsed_candidates")
        or []
    )
    qwen_validation = _validate_candidates(
        qwen_candidates,
        base_config,
        output_dir / "offline_comparison_qwen_yamls",
        "qwen",
        enable_env=True,
    )
    qwen_difficulty = _load_json(QWEN_DIFFICULTY).get("difficulty_results") or []

    fsn_metrics = _quality_metrics(
        candidates,
        fsn_validation,
        generation_runtime=generator.last_generation_runtime_seconds,
        predicted_values=[
            (item.get("metadata") or {}).get("predicted_value_score")
            for item in candidates
        ],
    )
    random_metrics = _quality_metrics(
        random_candidates,
        random_validation,
        generation_runtime=random_runtime,
        predicted_values=[],
    )
    qwen_metrics = _quality_metrics(
        qwen_candidates,
        qwen_validation,
        generation_runtime=None,
        predicted_values=[
            item.get("final_value_score") for item in qwen_difficulty
        ],
    )
    overlap = _changed_factor_overlap(candidates, qwen_candidates)
    fsn_metrics["changed_factor_overlap_with_qwen"] = overlap
    dataset_audit = _load_json(output_dir / "fsn_dataset_summary.json")
    shadow_ready = (
        fsn_metrics["schema_valid_rate"] == 1.0
        and fsn_metrics["constraint_valid_rate"] == 1.0
        and fsn_metrics["env_load_success_rate"] == 1.0
    )
    summary = {
        "schema_version": "falcon.fsn_generation_smoke_summary.v1",
        "checkpoint_path": str((ROOT_DIR / args.checkpoint).resolve()),
        "num_requested": args.num_scenarios,
        "fsn": fsn_metrics,
        "qwen_existing_offline_reference": qwen_metrics,
        "random": random_metrics,
        "qwen_call_reduction_potential": {
            "calls_avoided_per_fsn_batch": len(candidates),
            "fraction": 1.0 if candidates else 0.0,
            "qualification": (
                "Potential only: no policy-training replacement experiment was run."
            ),
        },
        "readiness": {
            "offline_shadow_replacement_pilot": shadow_ready,
            "policy_training_replacement_pilot": False,
            "blocking_reasons": [
                "cross_split_scenario_vector_leakage"
                if dataset_audit.get("cross_split_leakage_detected")
                else None,
                "invalid_class_underrepresented"
                if int(dataset_audit.get("invalid_count") or 0) < 20
                else None,
                "degenerate_failure_features"
                if dataset_audit.get("degenerate_failure_features")
                else None,
            ],
        },
        "failure_stage": None,
        "warnings": [
            "Qwen runtime is unavailable in the historical smoke artifact.",
            "Offline validity does not demonstrate policy-performance improvement.",
            "FSN remains outside the FALCON training loop.",
        ],
    }
    summary["readiness"]["blocking_reasons"] = [
        value for value in summary["readiness"]["blocking_reasons"] if value
    ]
    (output_dir / "fsn_generation_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_stage1_report(output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if fsn_metrics["env_load_success_rate"] > 0 else 1


def _validate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    yaml_dir: Path,
    prefix: str,
    enable_env: bool,
) -> Dict[str, Any]:
    yaml_dir.mkdir(parents=True, exist_ok=True)
    checker = ConstraintChecker()
    schema_results: List[Dict[str, Any]] = []
    constraint_results: List[Dict[str, Any]] = []
    env_results: List[Dict[str, Any]] = []
    yaml_paths: List[str] = []
    valid_candidates: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        schema = validate_candidate_schema(candidate)
        schema["scenario_id"] = candidate.get("scenario_id")
        schema_results.append(schema)
        constraint = checker.validate_candidate(candidate)
        constraint_results.append(constraint)
        if not schema["is_valid"] or not constraint["is_valid"]:
            continue
        valid_candidates.append(dict(candidate))
        yaml_config = apply_initial_config_to_yaml(
            base_config, candidate.get("initial_config") or {}
        )
        yaml_path = yaml_dir / f"{prefix}_{index:04d}.yaml"
        save_scenario_yaml(yaml_config, yaml_path)
        yaml_paths.append(str(yaml_path.resolve()))
        env_results.append(
            checker.validate_yaml_config(
                yaml_config,
                enable_env_load_check=enable_env,
                temp_config_name=f"{prefix}_fsn_smoke_{index}",
            )
        )
    return {
        "schema_version": "falcon.fsn_candidate_validation.v1",
        "schema_results": schema_results,
        "constraint_results": constraint_results,
        "valid_candidates": valid_candidates,
        "yaml_paths": yaml_paths,
        "env_results": env_results,
    }


def _quality_metrics(
    candidates: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    generation_runtime: float | None,
    predicted_values: Sequence[Any],
) -> Dict[str, Any]:
    total = max(len(candidates), 1)
    schema_results = validation.get("schema_results") or []
    constraint_results = validation.get("constraint_results") or []
    env_results = validation.get("env_results") or []
    clean_values = [_float(value) for value in predicted_values if _float(value) is not None]
    return {
        "num_candidates": len(candidates),
        "schema_valid_rate": round(
            sum(bool(item.get("is_valid")) for item in schema_results) / total, 6
        ),
        "constraint_valid_rate": round(
            sum(bool(item.get("is_valid")) for item in constraint_results) / total, 6
        ),
        "yaml_generation_success_rate": round(
            len(validation.get("yaml_paths") or []) / total, 6
        ),
        "env_load_success_rate": round(
            sum(
                bool(
                    (item.get("physical_constraint_check") or {}).get(
                        "scenario_loadable_env_check"
                    )
                )
                for item in env_results
            )
            / total,
            6,
        ),
        "predicted_value_score_mean": round(statistics.mean(clean_values), 6)
        if clean_values
        else None,
        "diversity_score": _diversity(candidates),
        "generation_runtime_seconds": None
        if generation_runtime is None
        else round(generation_runtime, 6),
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
                left_value = _float(left.get(key))
                right_value = _float(right.get(key))
                if left_value is None or right_value is None:
                    continue
                components.append(
                    ((left_value - right_value) / max(float(scales[key]), 1e-8))
                    ** 2
                )
            if components:
                distances.append(math.sqrt(sum(components) / len(components)))
    return round(statistics.mean(distances), 6) if distances else 0.0


def _changed_factor_overlap(
    fsn_candidates: Sequence[Mapping[str, Any]],
    qwen_candidates: Sequence[Mapping[str, Any]],
) -> float:
    qwen_sets = [
        set(str(value) for value in item.get("changed_factors") or [])
        for item in qwen_candidates
    ]
    scores = []
    for candidate in fsn_candidates:
        fsn_set = set(str(value) for value in candidate.get("changed_factors") or [])
        candidate_scores = []
        for qwen_set in qwen_sets:
            union = fsn_set | qwen_set
            candidate_scores.append(len(fsn_set & qwen_set) / max(len(union), 1))
        if candidate_scores:
            scores.append(max(candidate_scores))
    return round(statistics.mean(scores), 6) if scores else 0.0


def _write_stage1_report(output_dir: Path, generation: Mapping[str, Any]) -> None:
    dataset = _load_json(output_dir / "fsn_dataset_summary.json")
    training = _load_json(output_dir / "fsn_offline_training_summary.json")
    fsn = generation.get("fsn") or {}
    qwen = generation.get("qwen_existing_offline_reference") or {}
    random = generation.get("random") or {}
    labels = dataset.get("label_counts") or {}
    shadow_recommend = (
        bool(training.get("training_succeeded"))
        and fsn.get("schema_valid_rate") == 1.0
        and fsn.get("constraint_valid_rate") == 1.0
        and fsn.get("env_load_success_rate") == 1.0
    )
    lines = [
        "FALCON FSN Stage-1 Offline Smoke Report",
        "=" * 43,
        "",
        "Dataset",
        f"- Total samples: {dataset.get('total_samples')}",
        f"- Labels: {labels}",
        f"- Splits: {dataset.get('split_counts')}",
        f"- Cross-split leakage detected: {dataset.get('cross_split_leakage_detected')}",
        f"- Degenerate failure features: {dataset.get('degenerate_failure_features')}",
        "",
        "Offline training",
        f"- Training succeeded: {training.get('training_succeeded')}",
        f"- Runtime seconds: {training.get('runtime_seconds')}",
        f"- Test metrics: {(training.get('split_metrics') or {}).get('test')}",
        "",
        "Generation validity",
        f"- FSN: {fsn}",
        f"- Historical Qwen reference: {qwen}",
        f"- Random reference: {random}",
        "",
        "Judgement",
        f"- Recommend an offline shadow replacement pilot: {shadow_recommend}",
        "- Recommend a policy-training replacement pilot: False",
        f"- Blocking reasons: {(generation.get('readiness') or {}).get('blocking_reasons')}",
        "- No MAPPO performance improvement has been measured.",
        "- Invalid data are scarce and split leakage is substantial.",
        "- Keep candidate_schema, constraint_checker, and env load checks mandatory.",
    ]
    (output_dir.parent / "reports" / "fsn_stage1_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


if __name__ == "__main__":
    raise SystemExit(main())
