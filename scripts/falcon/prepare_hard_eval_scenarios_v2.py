#!/usr/bin/env python
"""Prepare a harder held-out evaluation set without touching training artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.candidate_schema import create_candidate_scenario, validate_candidate_schema  # noqa: E402
from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.scenario_adapter import (  # noqa: E402
    apply_initial_config_to_yaml,
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    load_base_scenario_config,
    save_scenario_yaml,
    scenario_vector_to_initial_config,
)
from falcon.trajectory_recorder import extract_scenario_vector, load_trajectory  # noqa: E402

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"
DEFAULT_OUTPUT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "hard_eval_scenarios_v2.json"
SCENARIO_VECTOR_KEYS = (
    "team_center_distance",
    "own_formation_spread",
    "opponent_formation_spread",
    "altitude_difference",
    "velocity_difference",
    "heading_difference",
    "approximate_aspect_angle",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Hard Held-out Eval Set v2.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = _resolve(args.output)
    scenario_dir = output_path.with_suffix("")
    if output_path.exists() and not args.force:
        print(json.dumps(_load_json(output_path), indent=2, sort_keys=True))
        return
    protocol = _load_yaml_like(args.protocol)
    base_config_path = _resolve(protocol["base_scenario_config"])
    base_config = load_base_scenario_config(base_config_path)
    base_initial = extract_initial_config_from_yaml(base_config)
    base_vector = initial_config_to_scenario_vector(base_initial)["scenario_vector"]
    checker = ConstraintChecker({"enable_env_load_check": not args.skip_env_check})
    rng = random.Random(args.seed)
    failure_sources = _load_failure_sources(
        ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "results" / "falcon_no_fsn",
        limit=10,
    )

    specs = []
    specs.extend(_initial_disadvantage_specs())
    specs.extend(_coordination_stress_specs())
    specs.extend(_target_assignment_stress_specs())
    specs.extend(_replay_variant_specs(failure_sources, rng))

    scenarios: List[Dict[str, Any]] = []
    invalid_attempts: List[Dict[str, Any]] = []
    for spec in specs:
        candidate, yaml_config, yaml_path, constraint = _build_candidate(
            spec,
            base_config,
            base_initial,
            base_vector,
            scenario_dir,
            checker,
        )
        schema_result = validate_candidate_schema(candidate)
        env_ok = bool((constraint.get("physical_constraint_check") or {}).get("scenario_loadable_env_check"))
        valid = bool(schema_result.get("is_valid") and constraint.get("is_valid") and (args.skip_env_check or env_ok))
        if valid:
            save_scenario_yaml(yaml_config, yaml_path)
            scenarios.append(
                {
                    "scenario_id": candidate["scenario_id"],
                    "scenario_group": spec["scenario_group"],
                    "difficulty_intent": spec["difficulty_intent"],
                    "scenario_yaml_path": str(_relative_to_root(yaml_path)),
                    "scenario_vector": candidate["scenario_vector"],
                    "source": "hard_eval_v2_generator",
                    "is_replay": bool(spec.get("is_replay")),
                    "source_failure_id": spec.get("source_failure_id"),
                    "source_failure_path": spec.get("source_failure_path"),
                    "perturbation": spec.get("perturbation"),
                    "target_failure_modes": candidate.get("target_failure_modes"),
                    "changed_factors": candidate.get("changed_factors"),
                    "candidate_schema_validation": schema_result,
                    "constraint_result": _compact_constraint(constraint),
                    "yaml_structure_loadable": bool((constraint.get("physical_constraint_check") or {}).get("scenario_loadable")),
                    "env_load_reset_success": env_ok if not args.skip_env_check else None,
                    "not_in_training_pool": True,
                    "metadata": candidate.get("metadata") or {},
                }
            )
        else:
            invalid_attempts.append(
                {
                    "scenario_id": candidate["scenario_id"],
                    "schema_result": schema_result,
                    "constraint_result": _compact_constraint(constraint),
                    "env_load_reset_success": env_ok,
                }
            )
    manifest = {
        "schema_version": "falcon.hard_eval_scenario_manifest.v2",
        "created_at": _timestamp(),
        "scenario_count": len(scenarios),
        "scenario_group_counts": dict(Counter(item["scenario_group"] for item in scenarios)),
        "environment": "2v2/NoWeapon/Selfplay",
        "base_config_path": str(_relative_to_root(base_config_path)),
        "shared_across_groups": True,
        "held_out_from_training": True,
        "fixed_opponent_required": True,
        "scenarios": scenarios,
        "invalid_attempts": invalid_attempts,
        "warnings": _manifest_warnings(scenarios, invalid_attempts, args.skip_env_check),
    }
    _write_json(output_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _build_candidate(
    spec: Mapping[str, Any],
    base_config: Mapping[str, Any],
    base_initial: Mapping[str, Any],
    base_vector: Mapping[str, Any],
    scenario_dir: Path,
    checker: ConstraintChecker,
) -> tuple[Dict[str, Any], Dict[str, Any], Path, Dict[str, Any]]:
    vector = dict(base_vector)
    vector.update(spec["scenario_vector"])
    vector["own_center_x"] = base_vector.get("own_center_x")
    vector["own_center_y"] = base_vector.get("own_center_y")
    vector["own_center_z"] = spec["scenario_vector"].get("own_center_z", base_vector.get("own_center_z"))
    vector["opponent_center_x"] = None
    vector["opponent_center_y"] = None
    vector["opponent_center_z"] = None
    initial_config = scenario_vector_to_initial_config(vector, base_initial)
    recomputed = initial_config_to_scenario_vector(initial_config)["scenario_vector"]
    candidate = create_candidate_scenario(
        scenario_id=spec["scenario_id"],
        generator_type="hard_eval_v2_generator",
        source_failure_id=spec.get("source_failure_id"),
        target_failure_modes=spec.get("target_failure_modes"),
        changed_factors=spec.get("changed_factors") or list(SCENARIO_VECTOR_KEYS),
        counterfactual_group_id=spec.get("scenario_group"),
        scenario_vector=recomputed,
        scenario_parameters={"requested_scenario_vector": vector},
        initial_config=initial_config,
        expected_effect=spec["difficulty_intent"],
        rationale=spec["rationale"],
        metadata={
            "scenario_group": spec["scenario_group"],
            "difficulty_intent": spec["difficulty_intent"],
            "is_replay": bool(spec.get("is_replay")),
            "source_failure_id": spec.get("source_failure_id"),
            "perturbation": spec.get("perturbation"),
            "not_for_training_pool": True,
        },
    )
    yaml_config = apply_initial_config_to_yaml(base_config, initial_config)
    yaml_config["scenario_id"] = spec["scenario_id"]
    yaml_path = scenario_dir / f"{spec['scenario_id']}.yaml"
    constraint = checker.validate_yaml_config(yaml_config, enable_env_load_check=checker.config.get("enable_env_load_check"))
    return candidate, yaml_config, yaml_path, constraint


def _initial_disadvantage_specs() -> List[Dict[str, Any]]:
    specs = []
    for idx in range(10):
        specs.append(
            {
                "scenario_id": f"hard_initial_disadvantage_{idx:03d}",
                "scenario_group": "hard_random_initial_disadvantage",
                "difficulty_intent": "own aircraft start from a tactically disadvantaged geometry",
                "rationale": "Opponent begins closer to a rear/side aspect, higher, and faster while remaining constraint-valid.",
                "target_failure_modes": ["initial_disadvantage", "generalization_failure"],
                "changed_factors": [
                    "team_center_distance",
                    "altitude_difference",
                    "velocity_difference",
                    "heading_difference",
                    "approximate_aspect_angle",
                ],
                "scenario_vector": {
                    "team_center_distance": 6200.0 + idx * 420.0,
                    "own_formation_spread": 1800.0 + (idx % 4) * 350.0,
                    "opponent_formation_spread": 1200.0 + (idx % 3) * 250.0,
                    "altitude_difference": 1300.0 + (idx % 4) * 180.0,
                    "velocity_difference": 36.0 + (idx % 5) * 4.0,
                    "heading_difference": 0.25 + (idx % 5) * 0.12,
                    "approximate_aspect_angle": 2.35 + (idx % 5) * 0.13,
                },
            }
        )
    return specs


def _coordination_stress_specs() -> List[Dict[str, Any]]:
    specs = []
    for idx in range(10):
        specs.append(
            {
                "scenario_id": f"hard_coordination_stress_{idx:03d}",
                "scenario_group": "hard_coordination_stress",
                "difficulty_intent": "own formation starts wide while opponent formation is compact",
                "rationale": "Wide own spacing stresses coordination and mutual support recovery.",
                "target_failure_modes": ["coordination_failure"],
                "changed_factors": [
                    "own_formation_spread",
                    "opponent_formation_spread",
                    "team_center_distance",
                    "approximate_aspect_angle",
                ],
                "scenario_vector": {
                    "team_center_distance": 7800.0 + idx * 620.0,
                    "own_formation_spread": 6500.0 + (idx % 4) * 320.0,
                    "opponent_formation_spread": 1050.0 + (idx % 4) * 180.0,
                    "altitude_difference": 450.0 + (idx % 5) * 220.0,
                    "velocity_difference": 18.0 + (idx % 4) * 7.0,
                    "heading_difference": 0.55 + (idx % 5) * 0.18,
                    "approximate_aspect_angle": 1.55 + (idx % 5) * 0.25,
                },
            }
        )
    return specs


def _target_assignment_stress_specs() -> List[Dict[str, Any]]:
    specs = []
    for idx in range(10):
        specs.append(
            {
                "scenario_id": f"hard_target_assignment_stress_{idx:03d}",
                "scenario_group": "hard_target_assignment_stress",
                "difficulty_intent": "enemy geometry is compact and ambiguous for target assignment",
                "rationale": "Two similarly threatening opponents can induce duplicated coverage or switching.",
                "target_failure_modes": ["target_assignment_confusion"],
                "changed_factors": [
                    "opponent_formation_spread",
                    "own_formation_spread",
                    "team_center_distance",
                    "heading_difference",
                    "approximate_aspect_angle",
                ],
                "scenario_vector": {
                    "team_center_distance": 6800.0 + idx * 360.0,
                    "own_formation_spread": 3600.0 + (idx % 5) * 520.0,
                    "opponent_formation_spread": 1200.0 + (idx % 4) * 220.0,
                    "altitude_difference": -300.0 + (idx % 7) * 120.0,
                    "velocity_difference": -8.0 + (idx % 5) * 4.0,
                    "heading_difference": 2.35 + (idx % 5) * 0.16,
                    "approximate_aspect_angle": 0.35 + (idx % 5) * 0.18,
                },
            }
        )
    return specs


def _replay_variant_specs(failure_sources: Sequence[Mapping[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    specs = []
    if not failure_sources:
        failure_sources = [{"source_failure_id": f"synthetic_replay_source_{idx:03d}", "scenario_vector": {}} for idx in range(10)]
    for idx in range(10):
        source = failure_sources[idx % len(failure_sources)]
        source_vector = dict(source.get("scenario_vector") or {})
        perturbation = {
            "team_center_distance_delta": rng.choice([-900.0, -600.0, 600.0, 900.0]),
            "own_formation_spread_delta": rng.choice([700.0, 1000.0, 1300.0]),
            "opponent_formation_spread_delta": rng.choice([-350.0, 300.0, 550.0]),
            "altitude_difference_delta": rng.choice([-500.0, 500.0, 800.0]),
            "velocity_difference_delta": rng.choice([18.0, 26.0, 34.0]),
            "aspect_angle_delta": rng.choice([0.35, 0.55, -0.45]),
        }
        vector = {
            "team_center_distance": _clip(_num(source_vector.get("team_center_distance"), 9200.0) + perturbation["team_center_distance_delta"], 6400.0, 14200.0),
            "own_formation_spread": _clip(max(_num(source_vector.get("own_formation_spread"), 2400.0), 1400.0) + perturbation["own_formation_spread_delta"], 1500.0, 7600.0),
            "opponent_formation_spread": _clip(max(_num(source_vector.get("opponent_formation_spread"), 1600.0), 1400.0) + perturbation["opponent_formation_spread_delta"], 1200.0, 4200.0),
            "altitude_difference": _clip(_num(source_vector.get("altitude_difference"), 0.0) + perturbation["altitude_difference_delta"], -1600.0, 1800.0),
            "velocity_difference": _clip(_num(source_vector.get("velocity_difference"), 0.0) + perturbation["velocity_difference_delta"], -20.0, 55.0),
            "heading_difference": _clip(_num(source_vector.get("heading_difference"), 1.2), 0.25, 2.9),
            "approximate_aspect_angle": _clip(_num(source_vector.get("approximate_aspect_angle"), 1.4) + perturbation["aspect_angle_delta"], 0.25, 2.95),
        }
        specs.append(
            {
                "scenario_id": f"replay_failure_variant_{idx:03d}",
                "scenario_group": "replay_failure_variants",
                "difficulty_intent": "small perturbation around a real FALCON failure trajectory",
                "rationale": "Replay-derived held-out probe uses failure context without copying the training scenario exactly.",
                "target_failure_modes": ["coordination_failure", "target_assignment_confusion"],
                "changed_factors": list(perturbation.keys()),
                "scenario_vector": vector,
                "is_replay": True,
                "source_failure_id": source.get("source_failure_id"),
                "source_failure_path": source.get("source_failure_path"),
                "perturbation": perturbation,
            }
        )
    return specs


def _load_failure_sources(root: Path, limit: int) -> List[Dict[str, Any]]:
    paths = sorted(root.rglob("*_failure.json"))
    sources = []
    seen_vectors = set()
    for path in paths:
        try:
            trajectory = load_trajectory(path)
            vector = extract_scenario_vector(trajectory)
        except Exception:
            continue
        key = tuple(round(_num(vector.get(item), 0.0), 3) for item in SCENARIO_VECTOR_KEYS)
        if key in seen_vectors:
            continue
        seen_vectors.add(key)
        sources.append(
            {
                "source_failure_id": trajectory.get("episode_id") or path.stem,
                "source_failure_path": str(_relative_to_root(path)),
                "scenario_vector": vector,
            }
        )
        if len(sources) >= limit:
            break
    return sources


def _manifest_warnings(scenarios: Sequence[Mapping[str, Any]], invalid: Sequence[Mapping[str, Any]], skip_env: bool) -> List[str]:
    warnings = []
    if len(scenarios) != 40:
        warnings.append(f"Expected 40 valid hard eval scenarios, got {len(scenarios)}.")
    if invalid:
        warnings.append(f"{len(invalid)} generated attempts failed schema/constraint/env checks.")
    if skip_env:
        warnings.append("Env load/reset check was skipped by command-line flag.")
    counts = Counter(item.get("scenario_group") for item in scenarios)
    for group in (
        "hard_random_initial_disadvantage",
        "hard_coordination_stress",
        "hard_target_assignment_stress",
        "replay_failure_variants",
    ):
        if counts.get(group, 0) != 10:
            warnings.append(f"Expected 10 scenarios for {group}, got {counts.get(group, 0)}.")
    return warnings


def _compact_constraint(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": result.get("schema_version"),
        "scenario_id": result.get("scenario_id"),
        "is_valid": result.get("is_valid"),
        "validity_score": result.get("validity_score"),
        "rejection_reasons": result.get("rejection_reasons"),
        "physical_constraint_check": result.get("physical_constraint_check"),
        "task_constraint_check": result.get("task_constraint_check"),
        "missing_fields": result.get("missing_fields"),
        "warnings": result.get("warnings"),
    }


def _load_yaml_like(path: str | Path) -> Dict[str, Any]:
    import yaml

    with _resolve(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _relative_to_root(path: str | Path) -> Path:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve())
    except ValueError:
        return path


def _num(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _clip(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    main()
