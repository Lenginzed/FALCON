"""Build the offline Failure-to-Scenario Network dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.fsn_dataset import (
    FSNDatasetBuilder,
    FSNStage2DatasetBuilder,
    load_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn",
    )
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument(
        "--stage1-dataset",
        default="experiments/falcon_2v2_noweapon/fsn/failure_to_scenario_dataset.jsonl",
    )
    args = parser.parse_args()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    if args.stage2:
        stage1_samples = load_jsonl(ROOT_DIR / args.stage1_dataset)
        builder = FSNStage2DatasetBuilder()
        samples, synthetic, summary, audit = builder.build(stage1_samples)
        builder.write(samples, synthetic, summary, audit, output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if samples and not summary["cross_split_leakage_detected"] else 1
    builder = FSNDatasetBuilder(ROOT_DIR)
    samples, summary = builder.build()
    builder.write(
        samples,
        summary,
        output_dir / "failure_to_scenario_dataset.jsonl",
        output_dir / "failure_to_scenario_dataset_summary.json",
        output_dir / "failure_to_scenario_feature_stats.csv",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
