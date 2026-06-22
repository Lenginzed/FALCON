#!/usr/bin/env python
"""Minimal CLI for running FALCON failure analysis on one trajectory JSON."""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.failure_analyzer import FailureAnalyzer  # noqa: E402
from falcon.trajectory_recorder import load_trajectory  # noqa: E402


def _load_optional_json(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a FALCON episode trajectory.")
    parser.add_argument("trajectory", help="Path to a saved episode trajectory JSON file.")
    parser.add_argument("--pool-stats", default=None, help="Optional training scenario pool statistics JSON.")
    parser.add_argument("--success-stats", default=None, help="Optional successful episode statistics JSON.")
    parser.add_argument("--historical-eval", default=None, help="Optional historical policy evaluation JSON.")
    parser.add_argument("--output", default=None, help="Optional path to save the analysis JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = FailureAnalyzer()
    trajectory = load_trajectory(args.trajectory)
    trajectory["_source_trajectory"] = args.trajectory
    result = analyzer.analyze_trajectory(
        trajectory_data=trajectory,
        pool_stats=_load_optional_json(args.pool_stats),
        success_stats=_load_optional_json(args.success_stats),
        policy_eval_stats=_load_optional_json(args.historical_eval),
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
