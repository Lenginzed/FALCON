#!/usr/bin/env python
"""Freeze the shared baseline evaluation scenario set without calling Qwen."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.random_scenario_generator import RandomScenarioGenerator  # noqa: E402
from falcon.scenario_adapter import (  # noqa: E402
    apply_initial_config_to_yaml,
    load_base_scenario_config,
    save_scenario_yaml,
    trajectory_to_replay_scenario,
)

EXPERIMENT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
SCENARIO_DIR = EXPERIMENT_DIR / "manifests" / "eval_scenarios"
MANIFEST_PATH = EXPERIMENT_DIR / "manifests" / "eval_scenarios.json"
BASE_PATH = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
PILOT_DIR = ROOT_DIR / "tests" / "tmp_falcon_pilot_20r"


def main() -> None:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    records.append(_copy_scenario(BASE_PATH, "base_000", "base", "fixed_base"))
    records.extend(_generate_random_group("random", 5, seed=410, harder=False))
    records.extend(_generate_random_group("hard_random", 5, seed=420, harder=True))
    records.extend(_copy_qwen_pilot_scenarios(5))
    records.extend(_generate_replay_scenarios(5))
    counts = Counter(item["scenario_group"] for item in records)
    constraint_invalid = [item["scenario_id"] for item in records if not item.get("constraint_valid")]
    manifest = {
        "schema_version": "falcon.eval_scenario_manifest.v1",
        "environment": "2v2/NoWeapon/Selfplay",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenario_count": len(records),
        "scenario_group_counts": dict(sorted(counts.items())),
        "shared_across_groups": True,
        "constraint_validity_is_annotation_only": True,
        "all_yaml_structures_loadable": all(item.get("yaml_structure_loadable") for item in records),
        "scenarios": records,
        "warnings": [
            "Base and replay scenarios are retained in the fixed evaluation set even when they fall outside candidate-generation constraints."
        ]
        if constraint_invalid
        else [],
    }
    _write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"manifest_path": str(MANIFEST_PATH), "scenario_count": len(records), "scenario_group_counts": counts}, indent=2, default=dict))


def _generate_random_group(group: str, count: int, seed: int, harder: bool) -> List[Dict[str, Any]]:
    config: Dict[str, Any] = {"seed": seed}
    if harder:
        config.update(
            {
                "team_center_distance_range": [15000.0, 17500.0],
                "own_formation_spread_range": [5000.0, 7500.0],
                "opponent_formation_spread_range": [5000.0, 7500.0],
                "altitude_difference_range": [-1800.0, 1800.0],
                "velocity_difference_range": [-38.0, 38.0],
                "heading_difference_range": [1.6, 4.7],
                "approximate_aspect_angle_range": [0.5, 1.4],
            }
        )
    base = load_base_scenario_config(BASE_PATH)
    candidates = RandomScenarioGenerator(config).generate_from_base(base, count)
    records = []
    for idx, candidate in enumerate(candidates):
        scenario_id = f"{group}_{idx:03d}"
        yaml_config = apply_initial_config_to_yaml(base, candidate["initial_config"])
        yaml_config["scenario_id"] = scenario_id
        output = SCENARIO_DIR / f"{scenario_id}.yaml"
        save_scenario_yaml(yaml_config, output)
        records.append(_record(scenario_id, group, "deterministic_random_generator", output, candidate.get("scenario_vector")))
    return records


def _copy_qwen_pilot_scenarios(count: int) -> List[Dict[str, Any]]:
    paths = []
    for path in sorted(PILOT_DIR.glob("falcon_controller_candidate_round*_*.yaml")):
        config = load_base_scenario_config(path)
        if ConstraintChecker().validate_yaml_config(config).get("is_valid"):
            paths.append(path)
        if len(paths) >= count:
            break
    if len(paths) < count:
        raise FileNotFoundError(f"Need {count} constraint-valid pilot Qwen YAML files under {PILOT_DIR}, found {len(paths)}.")
    return [
        _copy_scenario(path, f"qwen_pilot_{idx:03d}", "qwen_pilot", "qwen3:8b_pilot_snapshot")
        for idx, path in enumerate(paths)
    ]


def _generate_replay_scenarios(count: int) -> List[Dict[str, Any]]:
    paths = sorted(PILOT_DIR.glob("round*_failure_trajectories/*.json"))[:count]
    if len(paths) < count:
        raise FileNotFoundError(f"Need {count} failure trajectories under {PILOT_DIR}, found {len(paths)}.")
    records = []
    for idx, trajectory in enumerate(paths):
        scenario_id = f"replay_failure_{idx:03d}"
        output = SCENARIO_DIR / f"{scenario_id}.yaml"
        trajectory_to_replay_scenario(trajectory, output, base_config_path=BASE_PATH)
        records.append(_record(scenario_id, "replay_failure", "pilot_failure_trajectory", output, source_path=trajectory))
    return records


def _copy_scenario(source: Path, scenario_id: str, group: str, source_type: str) -> Dict[str, Any]:
    output = SCENARIO_DIR / f"{scenario_id}.yaml"
    shutil.copy2(source, output)
    return _record(scenario_id, group, source_type, output, source_path=source)


def _record(
    scenario_id: str,
    group: str,
    source_type: str,
    output: Path,
    scenario_vector: Mapping[str, Any] | None = None,
    source_path: Path | None = None,
) -> Dict[str, Any]:
    config = load_base_scenario_config(output)
    validation = ConstraintChecker().validate_yaml_config(config)
    return {
        "scenario_id": scenario_id,
        "scenario_group": group,
        "source": source_type,
        "scenario_yaml_path": _relative(output),
        "source_path": _relative(source_path) if source_path else None,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "constraint_valid": validation.get("is_valid"),
        "yaml_structure_loadable": validation.get("physical_constraint_check", {}).get("scenario_loadable"),
        "constraint_warnings": validation.get("warnings") or [],
        "scenario_vector": dict(scenario_vector or {}),
    }


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
