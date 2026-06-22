"""Run Stage 2 FSN diversity, legality, YAML, and env-load smoke checks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.candidate_schema import validate_candidate_schema
from falcon.constraint_checker import ConstraintChecker
from falcon.difficulty_evaluator import DEFAULT_CONFIG as DIFFICULTY_CONFIG
from falcon.fsn_generator import FSNScenarioGenerator
from falcon.scenario_adapter import (
    apply_initial_config_to_yaml,
    load_base_scenario_config,
    save_scenario_yaml,
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
DEFAULT_FAILURE = (
    ROOT_DIR
    / "tests"
    / "tmp_falcon_trajectories"
    / "falcon_coordination_failure_v2_analysis.json"
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
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn/stage2",
    )
    parser.add_argument("--num-scenarios", type=int, default=8)
    args = parser.parse_args()
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_dir / "fsn_generated_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    failure_summary = json.loads(DEFAULT_FAILURE.read_text(encoding="utf-8"))
    base_config = load_base_scenario_config(BASE_CONFIG_PATH)
    diversity_generator = FSNScenarioGenerator(
        ROOT_DIR / args.checkpoint,
        {
            "seed": 29,
            "diversity_aware": True,
            "oversample_factor": 6,
            "noise_scale": 0.10,
        },
    )
    plain_generator = FSNScenarioGenerator(
        ROOT_DIR / args.checkpoint,
        {
            "seed": 29,
            "diversity_aware": False,
            "noise_scale": 0.02,
        },
    )
    diversity_candidates = diversity_generator.generate_from_failure_summary(
        failure_summary, base_config, args.num_scenarios
    )
    plain_candidates = plain_generator.generate_from_failure_summary(
        failure_summary, base_config, args.num_scenarios
    )
    generated_payload = {
        "schema_version": "falcon.fsn_stage2_generated_candidates.v1",
        "diversity_aware_candidates": diversity_candidates,
        "no_diversity_candidates": plain_candidates,
    }
    (output_dir / "fsn_generated_candidates.json").write_text(
        json.dumps(generated_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    validated = _validate(diversity_candidates, base_config, yaml_dir)
    (output_dir / "fsn_validated_candidates.json").write_text(
        json.dumps(validated, indent=2, sort_keys=True), encoding="utf-8"
    )
    total = max(len(diversity_candidates), 1)
    schema_valid = sum(
        bool(item.get("is_valid")) for item in validated["schema_results"]
    )
    constraint_valid = sum(
        bool(item.get("is_valid"))
        for item in validated["constraint_results"]
    )
    env_valid = sum(
        bool(
            (item.get("physical_constraint_check") or {}).get(
                "scenario_loadable_env_check"
            )
        )
        for item in validated["env_results"]
    )
    diversity_score = _diversity(diversity_candidates)
    plain_diversity = _diversity(plain_candidates)
    summary = {
        "schema_version": "falcon.fsn_stage2_generation_smoke.v1",
        "checkpoint_path": str((ROOT_DIR / args.checkpoint).resolve()),
        "num_candidates": len(diversity_candidates),
        "schema_valid_rate": round(schema_valid / total, 6),
        "constraint_valid_rate": round(constraint_valid / total, 6),
        "yaml_generation_success_rate": round(
            len(validated["yaml_paths"]) / total, 6
        ),
        "env_load_success_rate": round(env_valid / total, 6),
        "no_diversity_score": plain_diversity,
        "diversity_aware_score": diversity_score,
        "diversity_improvement": round(diversity_score - plain_diversity, 6),
        "generation_runtime_seconds": (
            diversity_generator.last_generation_runtime_seconds
        ),
        "predicted_value_score_mean": _mean(
            [
                (item.get("metadata") or {}).get("predicted_value_score")
                for item in diversity_candidates
            ]
        ),
        "failure_stage": None if env_valid == len(diversity_candidates) else "env_load",
        "warnings": [
            "This smoke validates offline generation only; no MAPPO training was run.",
            "constraint_checker remains mandatory for every FSN proposal.",
        ],
    }
    (output_dir / "fsn_generation_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failure_stage"] is None else 1


def _validate(
    candidates: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    yaml_dir: Path,
) -> dict[str, Any]:
    checker = ConstraintChecker()
    schema_results = []
    constraint_results = []
    env_results = []
    yaml_paths = []
    valid_candidates = []
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
        yaml_path = yaml_dir / f"fsn_stage2_{index:04d}.yaml"
        save_scenario_yaml(yaml_config, yaml_path)
        yaml_paths.append(str(yaml_path.resolve()))
        valid_candidates.append(dict(candidate))
        env_results.append(
            checker.validate_yaml_config(
                yaml_config,
                enable_env_load_check=True,
                temp_config_name=f"fsn_stage2_generation_{index}",
            )
        )
    return {
        "schema_version": "falcon.fsn_stage2_validated_candidates.v1",
        "valid_candidates": valid_candidates,
        "schema_results": schema_results,
        "constraint_results": constraint_results,
        "env_results": env_results,
        "yaml_paths": yaml_paths,
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


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[Any]) -> float | None:
    clean = [number for value in values if (number := _number(value)) is not None]
    return round(statistics.fmean(clean), 6) if clean else None


if __name__ == "__main__":
    raise SystemExit(main())
