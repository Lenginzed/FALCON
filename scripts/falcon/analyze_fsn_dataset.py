"""Audit FSN dataset labels, features, duplicates, and split leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.fsn_dataset import analyze_samples, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="experiments/falcon_2v2_noweapon/fsn/failure_to_scenario_dataset.jsonl",
    )
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/falcon_2v2_noweapon/fsn/fsn_dataset_summary.json",
    )
    args = parser.parse_args()
    samples = load_jsonl(ROOT_DIR / args.dataset)
    audit = analyze_samples(samples)
    if args.stage2:
        audit["schema_version"] = "falcon.fsn_stage2_dataset_audit.v1"
    output = ROOT_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
