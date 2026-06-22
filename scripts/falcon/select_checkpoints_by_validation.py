#!/usr/bin/env python
"""Select baseline checkpoints using the frozen validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import SUPPORTED_GROUPS, load_yaml  # noqa: E402
from falcon.checkpoint_selector import (  # noqa: E402
    CheckpointSelector,
    DEFAULT_CANDIDATE_ROUNDS,
    create_eval_split,
)

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"
DEFAULT_SPLIT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "eval_split_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select checkpoints with validation scenarios.")
    parser.add_argument("--groups", nargs="+", choices=SUPPORTED_GROUPS, default=list(SUPPORTED_GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--candidate-rounds", nargs="+", type=int, default=list(DEFAULT_CANDIDATE_ROUNDS))
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    protocol = load_yaml(args.protocol)
    evaluation = dict(protocol.get("evaluation") or {})
    results_root = _resolve(protocol["output_root"])
    split = create_eval_split(
        evaluation.get("eval_scenarios") or protocol["evaluation_scenarios"],
        args.split,
        force=args.force_split,
    )
    selector = CheckpointSelector(
        results_root=results_root,
        base_config_path=protocol["base_scenario_config"],
        validation_manifest_path=split["validation"]["manifest_path"],
        opponent_checkpoint=evaluation["opponent_checkpoint"],
        opponent_mode=evaluation.get("opponent_mode", "fixed_checkpoint"),
    )
    job_summaries = []
    for group in args.groups:
        for seed in args.seeds:
            output_dir = (
                results_root
                / group
                / f"seed_{int(seed)}"
                / "eval_set"
                / "validation_checkpoint_selection"
            )
            summary = selector.select_for_group_seed(
                group=group,
                seed=seed,
                output_dir=output_dir,
                candidate_rounds=args.candidate_rounds,
                episodes_per_scenario=args.episodes_per_scenario,
                include_latest=True,
                force=args.force_eval,
            )
            job_summaries.append(summary)
    result = {
        "schema_version": "falcon.validation_checkpoint_selection.batch.v1",
        "split_path": str(_resolve(args.split)),
        "validation_manifest_path": split["validation"]["manifest_path"],
        "groups": args.groups,
        "seeds": args.seeds,
        "candidate_rounds": args.candidate_rounds,
        "episodes_per_scenario": args.episodes_per_scenario,
        "jobs": job_summaries,
        "num_jobs": len(job_summaries),
        "num_failed_jobs": sum(1 for item in job_summaries if item.get("failure_stage")),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


if __name__ == "__main__":
    main()
