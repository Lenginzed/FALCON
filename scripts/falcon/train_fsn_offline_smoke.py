"""Train the lightweight FSN on the existing offline dataset."""

from __future__ import annotations

import argparse
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
        default="experiments/falcon_2v2_noweapon/fsn/failure_to_scenario_dataset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    trainer = FSNTrainer(
        FSNTrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
    )
    summary = trainer.train(ROOT_DIR / args.dataset, ROOT_DIR / args.output_dir)
    print(
        json.dumps(
            {
                "training_succeeded": summary["training_succeeded"],
                "checkpoint_path": summary["checkpoint_path"],
                "split_counts": summary["split_counts"],
                "split_metrics": summary["split_metrics"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
