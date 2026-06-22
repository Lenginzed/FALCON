"""Train the de-leaked Stage 2 FSN with boundary-negative supervision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.fsn_trainer import FSNTrainer, FSNTrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "experiments/falcon_2v2_noweapon/fsn/stage2/"
            "failure_to_scenario_dataset_dedup.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn/stage2",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    trainer = FSNTrainer(
        FSNTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_constraint_head=True,
            lambda_constraint=0.40,
        )
    )
    summary = trainer.train(
        ROOT_DIR / args.dataset,
        output_dir,
        checkpoint_name="fsn_stage2_model.pt",
        summary_name="fsn_stage2_training_summary.json",
        stage_name="stage2",
    )
    metrics_path = output_dir / "fsn_stage2_metrics.csv"
    metric_keys = sorted(
        {
            key
            for metrics in summary["split_metrics"].values()
            for key in metrics
        }
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", *metric_keys]
        )
        writer.writeheader()
        for split, metrics in summary["split_metrics"].items():
            writer.writerow({"split": split, **metrics})
    print(
        json.dumps(
            {
                "training_succeeded": summary["training_succeeded"],
                "checkpoint_path": summary["checkpoint_path"],
                "split_counts": summary["split_counts"],
                "split_metrics": summary["split_metrics"],
                "overfitting_detected": summary["overfitting_detected"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
