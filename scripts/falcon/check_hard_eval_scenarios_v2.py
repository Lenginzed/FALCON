#!/usr/bin/env python
"""Sanity checks for Hard Held-out Eval Set v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.constraint_checker import ConstraintChecker  # noqa: E402
from falcon.scenario_adapter import load_base_scenario_config  # noqa: E402

DEFAULT_MANIFEST = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "hard_eval_scenarios_v2.json"
DEFAULT_OUTPUT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "hard_eval_scenarios_v2_check.json"
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
    parser = argparse.ArgumentParser(description="Check Hard Eval v2 scenarios.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-env-check", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    manifest = _load_json(manifest_path)
    checker = ConstraintChecker({"enable_env_load_check": not args.skip_env_check})
    pool_vectors = _load_training_pool_vectors(ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "results" / "falcon_no_fsn")
    rows = []
    warnings: List[str] = []
    for scenario in manifest.get("scenarios") or []:
        yaml_path = _resolve(scenario.get("scenario_yaml_path"))
        yaml_exists = yaml_path.exists()
        constraint = {}
        if yaml_exists:
            try:
                yaml_config = load_base_scenario_config(yaml_path)
                constraint = checker.validate_yaml_config(yaml_config, enable_env_load_check=not args.skip_env_check)
            except Exception as exc:  # noqa: BLE001
                constraint = {
                    "is_valid": False,
                    "rejection_reasons": ["yaml_check_exception"],
                    "warnings": [str(exc)],
                    "physical_constraint_check": {"scenario_loadable_env_check": False},
                }
        else:
            constraint = {
                "is_valid": False,
                "rejection_reasons": ["missing_yaml"],
                "warnings": ["YAML file does not exist."],
                "physical_constraint_check": {"scenario_loadable_env_check": False},
            }
        nearest = _nearest_pool_distance(scenario.get("scenario_vector") or {}, pool_vectors)
        rows.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "scenario_group": scenario.get("scenario_group"),
                "yaml_exists": yaml_exists,
                "constraint_valid": bool(constraint.get("is_valid")),
                "env_load_reset_success": (constraint.get("physical_constraint_check") or {}).get("scenario_loadable_env_check"),
                "nearest_training_pool_distance": nearest,
                "is_replay": bool(scenario.get("is_replay")),
                "rejection_reasons": ";".join(str(item) for item in constraint.get("rejection_reasons") or []),
                "warnings": ";".join(str(item) for item in constraint.get("warnings") or []),
            }
        )
    vector_stats = _scenario_vector_stats(manifest.get("scenarios") or [])
    yaml_ok = sum(1 for row in rows if row["yaml_exists"])
    constraint_ok = sum(1 for row in rows if row["constraint_valid"])
    env_ok = sum(1 for row in rows if row["env_load_reset_success"] is True)
    near_005 = sum(1 for row in rows if row["nearest_training_pool_distance"] is not None and row["nearest_training_pool_distance"] < 0.05)
    near_010 = sum(1 for row in rows if row["nearest_training_pool_distance"] is not None and row["nearest_training_pool_distance"] < 0.10)
    if len(rows) != 40:
        warnings.append(f"Expected 40 scenarios, got {len(rows)}.")
    if yaml_ok != len(rows):
        warnings.append(f"YAML exists for {yaml_ok}/{len(rows)} scenarios.")
    if constraint_ok != len(rows):
        warnings.append(f"Constraint valid for {constraint_ok}/{len(rows)} scenarios.")
    if not args.skip_env_check and env_ok != len(rows):
        warnings.append(f"Env load/reset succeeded for {env_ok}/{len(rows)} scenarios.")
    if rows and near_005 / len(rows) > 0.25:
        warnings.append("More than 25% of hard eval scenarios are very close to the training curriculum pool.")
    output = {
        "schema_version": "falcon.hard_eval_v2_check.v1",
        "created_at": _timestamp(),
        "manifest_path": str(manifest_path),
        "scenario_count": len(rows),
        "yaml_exists_count": yaml_ok,
        "constraint_valid_count": constraint_ok,
        "env_load_reset_success_count": env_ok,
        "scenario_group_counts": dict(Counter(row["scenario_group"] for row in rows)),
        "training_pool_vector_count": len(pool_vectors),
        "nearest_training_pool_distance": {
            "mean": _mean(row["nearest_training_pool_distance"] for row in rows),
            "min": _min(row["nearest_training_pool_distance"] for row in rows),
            "max": _max(row["nearest_training_pool_distance"] for row in rows),
            "count_below_0_05": near_005,
            "count_below_0_10": near_010,
        },
        "scenario_vector_stats": vector_stats,
        "rows": rows,
        "failure_stage": None if not warnings else "sanity_warning",
        "warnings": warnings,
    }
    output_path = _resolve(args.output)
    _write_json(output_path, output)
    _write_csv(output_path.with_suffix(".csv"), rows)
    print(json.dumps(output, indent=2, sort_keys=True))


def _load_training_pool_vectors(root: Path) -> List[Dict[str, Any]]:
    vectors = []
    for pool_path in sorted(root.rglob("falcon_curriculum_pool_final.json")):
        try:
            data = _load_json(pool_path)
        except Exception:
            continue
        for item in data.get("items") or []:
            vector = item.get("scenario_vector")
            if not isinstance(vector, MappingABC):
                candidate = item.get("candidate_scenario") if isinstance(item.get("candidate_scenario"), MappingABC) else {}
                vector = candidate.get("scenario_vector")
            if isinstance(vector, MappingABC):
                vectors.append(dict(vector))
    return vectors


def _nearest_pool_distance(vector: Mapping[str, Any], pool_vectors: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not pool_vectors:
        return None
    distances = [_normalized_distance(vector, item) for item in pool_vectors]
    distances = [item for item in distances if item is not None]
    return round(min(distances), 6) if distances else None


def _normalized_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> Optional[float]:
    total = 0.0
    count = 0
    for key in VECTOR_KEYS:
        lv = _num(left.get(key))
        rv = _num(right.get(key))
        if lv is None or rv is None:
            continue
        low, high = RANGES[key]
        denom = max(high - low, 1e-8)
        total += ((lv - rv) / denom) ** 2
        count += 1
    if count == 0:
        return None
    return math.sqrt(total / count)


def _scenario_vector_stats(scenarios: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stats = {}
    for key in VECTOR_KEYS:
        values = [_num((item.get("scenario_vector") or {}).get(key)) for item in scenarios]
        values = [item for item in values if item is not None]
        stats[key] = {
            "min": round(min(values), 6) if values else None,
            "mean": round(sum(values) / len(values), 6) if values else None,
            "max": round(max(values), 6) if values else None,
            "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0 if values else None,
        }
    return stats


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def _min(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None]
    return round(min(clean), 6) if clean else None


def _max(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None]
    return round(max(clean), 6) if clean else None


def _num(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    main()
