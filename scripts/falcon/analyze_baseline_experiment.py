#!/usr/bin/env python
"""Aggregate baseline preparation dry-run and smoke-run outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import BaselineExperimentAnalyzer  # noqa: E402

DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FALCON baseline preparation outputs.")
    parser.add_argument("--results-root", default=str(DEFAULT_EXPERIMENT_DIR / "results"))
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPERIMENT_DIR / "reports"))
    args = parser.parse_args()
    report = BaselineExperimentAnalyzer(args.results_root).export(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

