#!/usr/bin/env python
"""Analyze an existing FALCON short pilot output directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.postrun_analyzer import FalconPostRunAnalyzer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FALCON short pilot outputs without running training.")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "tests" / "tmp_falcon_short_pilot"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    analyzer = FalconPostRunAnalyzer(output_dir)
    report = analyzer.export_report(output_dir / "falcon_postrun_report.json")
    concise = {
        "schema_version": report.get("schema_version"),
        "stable": report.get("diagnostics", {}).get("stable"),
        "recommend_medium_pilot": report.get("diagnostics", {}).get("recommend_medium_pilot"),
        "accepted_rate": report.get("curriculum_pool", {}).get("accepted_rate"),
        "fallback_rate": report.get("fallbacks", {}).get("fallback_rate"),
        "main_risks": report.get("diagnostics", {}).get("main_risks"),
        "outputs": [
            str(output_dir / "falcon_postrun_report.json"),
            str(output_dir / "falcon_postrun_report.txt"),
            str(output_dir / "falcon_postrun_round_metrics.csv"),
        ],
    }
    print(json.dumps(concise, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
