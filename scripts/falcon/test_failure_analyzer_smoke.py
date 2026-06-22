#!/usr/bin/env python
"""Smoke test for FALCON failure analysis fixtures."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.failure_analyzer import FailureAnalyzer, FAILURE_MODE_KEYS  # noqa: E402
from falcon.trajectory_recorder import load_trajectory  # noqa: E402


def _assert_scores(result):
    scores = result["failure_scores"]
    for key, value in scores.items():
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"{key}={value} outside [0, 1]")
    expected_primary = [mode for mode in FAILURE_MODE_KEYS if scores[mode] >= 0.7]
    expected_secondary = [mode for mode in FAILURE_MODE_KEYS if 0.4 <= scores[mode] < 0.7]
    if result["primary_failure_modes"] != expected_primary:
        raise AssertionError((result["primary_failure_modes"], expected_primary))
    if result["secondary_failure_modes"] != expected_secondary:
        raise AssertionError((result["secondary_failure_modes"], expected_secondary))


def main():
    analyzer = FailureAnalyzer()
    fixture_dir = ROOT_DIR / "tests" / "fixtures"
    output_dir = ROOT_DIR / "tests" / "tmp_falcon_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    success_stats = {"mean_success_team_reward": 500.0}
    results = {}
    for name in ("falcon_coordination_failure_v2.json", "falcon_target_switch_failure_v2.json"):
        data = load_trajectory(fixture_dir / name)
        data["_source_trajectory"] = str(fixture_dir / name)
        result = analyzer.analyze_trajectory(data, success_stats=success_stats)
        _assert_scores(result)
        out_path = output_dir / f"{Path(name).stem}_analysis.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        results[name] = result

    minimal = {"schema_version": "bad", "frames": []}
    missing_result = analyzer.analyze_trajectory(minimal)
    if not missing_result["missing_fields"]:
        raise AssertionError("Missing-field fixture did not report missing_fields.")
    results["missing_fields_case"] = missing_result
    print(json.dumps({"schema_version": "falcon.failure_analyzer_smoke.v1", "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
