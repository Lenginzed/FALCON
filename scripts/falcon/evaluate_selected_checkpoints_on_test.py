#!/usr/bin/env python
"""Evaluate validation-selected checkpoints on held-out test scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import SUPPORTED_GROUPS, load_yaml  # noqa: E402
from falcon.checkpoint_selector import create_eval_split, write_per_scenario_csv  # noqa: E402
from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"
DEFAULT_SPLIT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "eval_split_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out test eval for validation-selected checkpoints.")
    parser.add_argument("--groups", nargs="+", choices=SUPPORTED_GROUPS, default=list(SUPPORTED_GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--episodes-per-scenario", type=int, default=3)
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    protocol = load_yaml(args.protocol)
    evaluation = dict(protocol.get("evaluation") or {})
    results_root = _resolve(protocol["output_root"])
    split = create_eval_split(
        evaluation.get("eval_scenarios") or protocol["evaluation_scenarios"],
        args.split,
        force=False,
    )
    evaluator = EvalSetEvaluator(
        split["test"]["manifest_path"],
        {"base_config_path": str(_resolve(protocol["base_scenario_config"]))},
    )
    jobs = []
    for group in args.groups:
        for seed in args.seeds:
            seed_dir = results_root / group / f"seed_{int(seed)}"
            selection_path = seed_dir / "eval_set" / "validation_checkpoint_selection" / "validation_selected_checkpoint.json"
            output_dir = seed_dir / "eval_set" / "validation_selected_test_eval"
            output_path = output_dir / "heldout_test_summary.json"
            per_scenario_path = output_dir / "heldout_test_per_scenario.csv"
            if not selection_path.exists():
                jobs.append(
                    {
                        "group": group,
                        "seed": int(seed),
                        "failure_stage": "missing_validation_selection",
                        "selection_path": str(selection_path),
                        "warnings": ["Run select_checkpoints_by_validation.py first."],
                    }
                )
                continue
            selection = _load_json(selection_path)
            checkpoint_path = selection.get("selected_checkpoint")
            if not checkpoint_path or not Path(str(checkpoint_path)).exists():
                jobs.append(
                    {
                        "group": group,
                        "seed": int(seed),
                        "failure_stage": "missing_selected_checkpoint",
                        "selection_path": str(selection_path),
                        "checkpoint_path": checkpoint_path,
                        "warnings": ["Selected checkpoint path is missing."],
                    }
                )
                continue
            if output_path.exists() and not args.force_eval:
                summary = _load_json(output_path)
            else:
                summary = evaluator.evaluate_checkpoint(
                    checkpoint_path,
                    episodes_per_scenario=int(args.episodes_per_scenario),
                    seed=int(seed),
                    group=group,
                    checkpoint_role="validation_selected",
                    opponent_mode=evaluation.get("opponent_mode", "fixed_checkpoint"),
                    opponent_checkpoint=_resolve(evaluation.get("opponent_checkpoint")),
                )
                summary["validation_selection_path"] = str(selection_path)
                summary["validation_selected_round_id"] = selection.get("selected_round_id")
                summary["validation_selected_candidate_label"] = selection.get("selected_candidate_label")
                summary["validation_win_rate"] = selection.get("validation_win_rate")
                summary["validation_mean_return"] = selection.get("validation_mean_return")
                EvalSetEvaluator.save(summary, output_path)
            write_per_scenario_csv(summary, per_scenario_path)
            jobs.append(
                {
                    "group": group,
                    "seed": int(seed),
                    "selected_checkpoint": checkpoint_path,
                    "selected_round_id": selection.get("selected_round_id"),
                    "heldout_test_summary_path": str(output_path),
                    "heldout_test_per_scenario_path": str(per_scenario_path),
                    "num_scenarios_evaluated": summary.get("num_scenarios_evaluated"),
                    "failure_stage": summary.get("failure_stage"),
                    "same_actor": summary.get("same_actor"),
                    "same_checkpoint": summary.get("same_checkpoint"),
                    "opponent_mode": summary.get("opponent_mode"),
                    "aggregate_result": summary.get("aggregate_result"),
                    "warnings": summary.get("warnings") or [],
                }
            )
    result = {
        "schema_version": "falcon.validation_selected_test_eval.batch.v1",
        "split_path": str(_resolve(args.split)),
        "test_manifest_path": split["test"]["manifest_path"],
        "episodes_per_scenario": args.episodes_per_scenario,
        "jobs": jobs,
        "num_jobs": len(jobs),
        "num_failed_jobs": sum(1 for item in jobs if item.get("failure_stage")),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
