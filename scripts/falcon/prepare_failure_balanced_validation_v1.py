#!/usr/bin/env python
"""Prepare an independent failure-balanced checkpoint validation proxy."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.scenario_adapter import (  # noqa: E402
    apply_initial_config_to_yaml,
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    load_base_scenario_config,
    save_scenario_yaml,
    scenario_vector_to_initial_config,
)

OUTPUT = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "manifests"
    / "failure_balanced_validation_v1.json"
)
HARD_MANIFEST = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "manifests"
    / "hard_eval_scenarios_v2.json"
)
BASE_CONFIG = (
    ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
)
VECTOR_KEYS = (
    "team_center_distance",
    "own_formation_spread",
    "opponent_formation_spread",
    "altitude_difference",
    "velocity_difference",
    "heading_difference",
    "approximate_aspect_angle",
)
RANGES = {
    "team_center_distance": (6000.0, 18000.0),
    "own_formation_spread": (1000.0, 8000.0),
    "opponent_formation_spread": (1000.0, 8000.0),
    "altitude_difference": (-3000.0, 3000.0),
    "velocity_difference": (-80.0, 80.0),
    "heading_difference": (0.0, math.pi),
    "approximate_aspect_angle": (0.0, math.pi),
}


def main() -> None:
    if OUTPUT.exists():
        manifest = _load_json(OUTPUT)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    base_config = load_base_scenario_config(BASE_CONFIG)
    base_initial = extract_initial_config_from_yaml(base_config)
    base_vector = initial_config_to_scenario_vector(base_initial)["scenario_vector"]
    checker = ConstraintChecker({"enable_env_load_check": True})
    hard_manifest = _load_json(HARD_MANIFEST)
    hard_vectors = [
        dict(item.get("scenario_vector") or {})
        for item in hard_manifest.get("scenarios") or []
    ]
    hard_geometry_hashes = {
        _vector_hash(item) for item in hard_vectors if _vector_hash(item)
    }
    scenario_dir = OUTPUT.with_suffix("")
    scenarios = []
    invalid = []
    for spec in _specs():
        requested_vector = dict(base_vector)
        requested_vector.update(spec["scenario_vector"])
        requested_vector["own_center_x"] = base_vector.get("own_center_x")
        requested_vector["own_center_y"] = base_vector.get("own_center_y")
        requested_vector["own_center_z"] = base_vector.get("own_center_z")
        requested_vector["opponent_center_x"] = None
        requested_vector["opponent_center_y"] = None
        requested_vector["opponent_center_z"] = None
        initial = scenario_vector_to_initial_config(requested_vector, base_initial)
        vector = initial_config_to_scenario_vector(initial)["scenario_vector"]
        yaml_config = apply_initial_config_to_yaml(base_config, initial)
        yaml_config["scenario_id"] = spec["scenario_id"]
        yaml_path = scenario_dir / f"{spec['scenario_id']}.yaml"
        constraint = checker.validate_yaml_config(
            yaml_config,
            enable_env_load_check=True,
            temp_config_name=spec["scenario_id"],
        )
        env_ok = bool(
            (constraint.get("physical_constraint_check") or {}).get(
                "scenario_loadable_env_check"
            )
        )
        geometry_hash = _vector_hash(vector)
        nearest_hard_distance = _nearest_distance(vector, hard_vectors)
        exact_hard_overlap = geometry_hash in hard_geometry_hashes
        if constraint.get("is_valid") and env_ok and not exact_hard_overlap:
            save_scenario_yaml(yaml_config, yaml_path)
            scenarios.append(
                {
                    "scenario_id": spec["scenario_id"],
                    "scenario_group": spec["scenario_group"],
                    "difficulty_intent": spec["difficulty_intent"],
                    "scenario_yaml_path": str(yaml_path.relative_to(ROOT_DIR)),
                    "scenario_vector": vector,
                    "source": "failure_balanced_validation_generator_v1",
                    "held_out_from_training": True,
                    "not_in_training_pool": True,
                    "selection_only": True,
                    "constraint_valid": True,
                    "env_load_reset_success": True,
                    "geometry_sha256": geometry_hash,
                    "nearest_hard_eval_v2_distance": nearest_hard_distance,
                    "exact_hard_eval_v2_overlap": False,
                    "constraint_result": _compact_constraint(constraint),
                }
            )
        else:
            invalid.append(
                {
                    "scenario_id": spec["scenario_id"],
                    "constraint_valid": constraint.get("is_valid"),
                    "env_load_reset_success": env_ok,
                    "exact_hard_eval_v2_overlap": exact_hard_overlap,
                    "rejection_reasons": constraint.get("rejection_reasons") or [],
                    "warnings": constraint.get("warnings") or [],
                }
            )
    group_counts = dict(Counter(item["scenario_group"] for item in scenarios))
    overlap_count = sum(
        bool(item.get("exact_hard_eval_v2_overlap")) for item in scenarios
    )
    manifest = {
        "schema_version": "falcon.failure_balanced_validation_manifest.v1",
        "manifest_id": "failure_balanced_validation_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenario_count": len(scenarios),
        "scenario_group_counts": group_counts,
        "environment": "2v2/NoWeapon/Selfplay",
        "base_config_path": str(BASE_CONFIG.relative_to(ROOT_DIR)),
        "held_out_from_training": True,
        "not_in_training_pool": True,
        "selection_only": True,
        "fixed_opponent_required": True,
        "same_actor_allowed": False,
        "hard_eval_v2_source": str(HARD_MANIFEST.relative_to(ROOT_DIR)),
        "exact_hard_eval_v2_overlap_count": overlap_count,
        "minimum_normalized_distance_to_hard_eval_v2": min(
            item["nearest_hard_eval_v2_distance"] for item in scenarios
        )
        if scenarios
        else None,
        "scenarios": scenarios,
        "invalid_attempts": invalid,
        "failure_stage": None
        if len(scenarios) >= 12 and overlap_count == 0 and not invalid
        else "proxy_generation_validation",
        "warnings": _warnings(scenarios, invalid, overlap_count),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _specs() -> list[Dict[str, Any]]:
    groups = {
        "initial_disadvantage_validation": [
            (6550, 2150, 1450, 1120, 31, 0.42, 2.12),
            (7450, 2850, 1850, 1480, 43, 0.83, 2.62),
            (8650, 3450, 1250, 1780, 51, 1.17, 2.88),
            (10150, 1950, 2350, 960, 37, 0.68, 2.34),
        ],
        "coordination_stress_validation": [
            (7350, 5900, 1450, 300, 12, 0.72, 1.25),
            (8950, 7100, 1950, 750, 25, 1.14, 1.78),
            (11050, 6250, 2550, 1100, 38, 1.58, 2.15),
            (13250, 7550, 1650, -400, 8, 1.95, 1.48),
        ],
        "target_assignment_validation": [
            (7150, 3200, 1350, -180, -5, 2.18, 0.28),
            (8250, 4100, 1750, 260, 6, 2.48, 0.62),
            (9550, 5250, 2200, 620, 14, 2.72, 0.94),
            (11250, 6000, 1550, -540, 22, 2.92, 1.18),
        ],
        "replay_like_validation": [
            (7850, 2750, 2100, 680, 28, 1.36, 2.05),
            (9250, 4300, 1700, -720, 34, 2.06, 0.88),
            (11850, 5600, 2850, 1250, 18, 0.96, 2.46),
            (14650, 3650, 3300, -1050, -12, 2.66, 1.42),
        ],
    }
    output = []
    for group, vectors in groups.items():
        for index, values in enumerate(vectors):
            output.append(
                {
                    "scenario_id": f"fbv1_{group}_{index:02d}",
                    "scenario_group": group,
                    "difficulty_intent": group.replace("_", " "),
                    "scenario_vector": dict(zip(VECTOR_KEYS, values)),
                }
            )
    return output


def _compact_constraint(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "is_valid": result.get("is_valid"),
        "validity_score": result.get("validity_score"),
        "rejection_reasons": result.get("rejection_reasons") or [],
        "physical_constraint_check": result.get("physical_constraint_check") or {},
        "task_constraint_check": result.get("task_constraint_check") or {},
        "warnings": result.get("warnings") or [],
    }


def _vector_hash(vector: Mapping[str, Any]) -> str:
    values = []
    for key in VECTOR_KEYS:
        value = _number(vector.get(key))
        if value is None:
            return ""
        values.append(round(value, 6))
    return hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()


def _nearest_distance(
    vector: Mapping[str, Any], others: Sequence[Mapping[str, Any]]
) -> float | None:
    distances = [_distance(vector, other) for other in others]
    distances = [item for item in distances if item is not None]
    return round(min(distances), 6) if distances else None


def _distance(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float | None:
    total = 0.0
    count = 0
    for key in VECTOR_KEYS:
        lv = _number(left.get(key))
        rv = _number(right.get(key))
        if lv is None or rv is None:
            continue
        low, high = RANGES[key]
        total += ((lv - rv) / max(high - low, 1e-8)) ** 2
        count += 1
    return math.sqrt(total / count) if count else None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _warnings(
    scenarios: Sequence[Mapping[str, Any]],
    invalid: Sequence[Mapping[str, Any]],
    overlap_count: int,
) -> list[str]:
    warnings = []
    if len(scenarios) < 12:
        warnings.append(f"Only {len(scenarios)} valid proxy scenarios were generated.")
    if invalid:
        warnings.append(f"{len(invalid)} proxy scenarios failed validation.")
    if overlap_count:
        warnings.append(f"{overlap_count} exact Hard Eval v2 geometry overlaps detected.")
    return warnings


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
