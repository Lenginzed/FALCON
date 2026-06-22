"""Train the FSN hardness-v2 dual-boundary surrogate from offline artifacts."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.fsn_hardness_surrogate import (  # noqa: E402
    SurrogateTrainingConfig,
    collect_surrogate_samples,
    train_surrogate,
)


DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage6_hardness_v2"
)
DEFAULT_FSN_DATASET = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage2"
    / "failure_to_scenario_dataset_dedup.jsonl"
)
DEFAULT_CANDIDATE_RECORDS = [
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage3"
    / "fsn_policy_evaluated_shadow_candidates.json",
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage5_repair"
    / "fsn_repair_shadow_candidates.json",
]
DEFAULT_FAILURE_SUMMARIES = [
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "fsn"
    / "stage3"
    / "fsn_shadow_failure_summaries.json"
]


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples = collect_surrogate_samples(
        fsn_dataset_path=DEFAULT_FSN_DATASET,
        candidate_record_paths=DEFAULT_CANDIDATE_RECORDS,
        failure_summary_paths=DEFAULT_FAILURE_SUMMARIES,
    )
    summary = train_surrogate(
        samples,
        output_dir,
        SurrogateTrainingConfig(
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
