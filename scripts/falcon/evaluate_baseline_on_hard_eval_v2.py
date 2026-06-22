#!/usr/bin/env python
"""Evaluate existing baseline checkpoints on Hard Held-out Eval Set v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import SUPPORTED_GROUPS, load_yaml  # noqa: E402
from falcon.checkpoint_selector import write_per_scenario_csv  # noqa: E402
from falcon.eval_set_evaluator import EvalSetEvaluator, resolve_group_checkpoint  # noqa: E402

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"
DEFAULT_MANIFEST = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "hard_eval_scenarios_v2.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baselines on Hard Eval v2.")
    parser.add_argument("--groups", nargs="+", choices=SUPPORTED_GROUPS, default=list(SUPPORTED_GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--episodes-per-scenario", type=int, default=3)
    parser.add_argument("--checkpoint-source", choices=("validation_selected", "formal_best", "latest"), default="validation_selected")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    protocol = load_yaml(args.protocol)
    evaluation = dict(protocol.get("evaluation") or {})
    results_root = _resolve(protocol["output_root"])
    manifest_path = _resolve(args.manifest)
    evaluator = EvalSetEvaluator(
        manifest_path,
        {"base_config_path": str(_resolve(protocol["base_scenario_config"]))},
    )
    jobs = []
    for group in args.groups:
        for seed in args.seeds:
            checkpoint_path, checkpoint_source_detail = _resolve_checkpoint(
                results_root,
                group,
                seed,
                args.checkpoint_source,
            )
            output_dir = results_root / group / f"seed_{int(seed)}" / "eval_set" / "hard_eval_v2"
            output_path = output_dir / "hard_eval_v2_summary.json"
            per_scenario_path = output_dir / "hard_eval_v2_per_scenario.csv"
            if checkpoint_path is None:
                jobs.append(
                    {
                        "group": group,
                        "seed": int(seed),
                        "checkpoint_source": args.checkpoint_source,
                        "checkpoint_source_detail": checkpoint_source_detail,
                        "failure_stage": "checkpoint_resolution",
                        "warnings": ["Could not resolve checkpoint for Hard Eval v2."],
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
                    checkpoint_role=f"hard_eval_v2_{args.checkpoint_source}",
                    opponent_mode=evaluation.get("opponent_mode", "fixed_checkpoint"),
                    opponent_checkpoint=_resolve(evaluation.get("opponent_checkpoint")),
                )
                summary["hard_eval_v2_manifest_path"] = str(manifest_path)
                summary["checkpoint_source"] = args.checkpoint_source
                summary["checkpoint_source_detail"] = checkpoint_source_detail
                EvalSetEvaluator.save(summary, output_path)
            write_per_scenario_csv(summary, per_scenario_path)
            jobs.append(
                {
                    "group": group,
                    "seed": int(seed),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_source": args.checkpoint_source,
                    "checkpoint_source_detail": checkpoint_source_detail,
                    "summary_path": str(output_path),
                    "per_scenario_csv": str(per_scenario_path),
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
        "schema_version": "falcon.hard_eval_v2.batch.v1",
        "manifest_path": str(manifest_path),
        "episodes_per_scenario": args.episodes_per_scenario,
        "checkpoint_source": args.checkpoint_source,
        "jobs": jobs,
        "num_jobs": len(jobs),
        "num_failed_jobs": sum(1 for item in jobs if item.get("failure_stage")),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def _resolve_checkpoint(
    results_root: Path,
    group: str,
    seed: int,
    checkpoint_source: str,
) -> tuple[Optional[Path], Dict[str, Any]]:
    seed_dir = results_root / group / f"seed_{int(seed)}"
    if checkpoint_source == "validation_selected":
        selection_path = seed_dir / "eval_set" / "validation_checkpoint_selection" / "validation_selected_checkpoint.json"
        if selection_path.exists():
            selection = _load_json(selection_path)
            checkpoint = selection.get("selected_checkpoint")
            path = Path(str(checkpoint)) if checkpoint else None
            if path is not None and path.exists():
                return path, {
                    "selection_path": str(selection_path),
                    "selected_round_id": selection.get("selected_round_id"),
                    "validation_win_rate": selection.get("validation_win_rate"),
                    "validation_mean_return": selection.get("validation_mean_return"),
                }
        return None, {"selection_path": str(selection_path), "warning": "missing validation-selected checkpoint"}
    role = "best" if checkpoint_source == "formal_best" else "latest"
    path = resolve_group_checkpoint(results_root, group, seed, role)
    return path, {"checkpoint_role": role}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
