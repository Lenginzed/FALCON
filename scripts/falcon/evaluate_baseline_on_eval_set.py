#!/usr/bin/env python
"""Evaluate one baseline checkpoint on the shared frozen eval scenario set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import SUPPORTED_GROUPS, load_yaml  # noqa: E402
from falcon.eval_set_evaluator import EvalSetEvaluator, resolve_group_checkpoint  # noqa: E402
from falcon.policy_evaluator import SUPPORTED_OPPONENT_MODES  # noqa: E402

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline checkpoint on the frozen eval set.")
    parser.add_argument("--group", required=True, choices=SUPPORTED_GROUPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--checkpoint", choices=("latest", "best"), default=None)
    parser.add_argument("--agent-checkpoint", default=None)
    parser.add_argument("--opponent-checkpoint", default=None)
    parser.add_argument("--opponent-mode", choices=SUPPORTED_OPPONENT_MODES, default=None)
    parser.add_argument("--episodes-per-scenario", type=int, default=None)
    parser.add_argument("--smoke-eval", action="store_true")
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    protocol = load_yaml(args.protocol)
    evaluation_protocol = dict(protocol.get("evaluation") or {})
    results_root = _resolve(protocol["output_root"])
    manifest_path = _resolve(evaluation_protocol.get("eval_scenarios") or protocol["evaluation_scenarios"])
    checkpoint_role = args.checkpoint or evaluation_protocol.get("checkpoint_selection") or "best"
    checkpoint_path = (
        _resolve(args.agent_checkpoint)
        if args.agent_checkpoint
        else resolve_group_checkpoint(results_root, args.group, args.seed, checkpoint_role)
    )
    opponent_mode = args.opponent_mode or evaluation_protocol.get("opponent_mode") or "fixed_checkpoint"
    opponent_checkpoint_value = (
        args.opponent_checkpoint or evaluation_protocol.get("opponent_checkpoint")
        if opponent_mode == "fixed_checkpoint"
        else args.opponent_checkpoint
    )
    opponent_checkpoint = _resolve(opponent_checkpoint_value) if opponent_checkpoint_value else None
    episodes_per_scenario = int(args.episodes_per_scenario or evaluation_protocol.get("episodes_per_scenario") or 1)
    if checkpoint_path is None:
        summary = {
            "schema_version": "falcon.eval_set_summary.v1",
            "group": args.group,
            "checkpoint_role": checkpoint_role,
            "checkpoint_path": None,
            "agent_checkpoint": None,
            "opponent_mode": opponent_mode,
            "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint is not None else None,
            "same_actor": opponent_mode == "same_actor",
            "same_actor_eval": opponent_mode == "same_actor",
            "eval_protocol_frozen": False,
            "num_scenarios_evaluated": 0,
            "failure_stage": "checkpoint_resolution",
            "warnings": ["Could not resolve a usable baseline checkpoint."],
        }
    else:
        limit = args.scenario_limit
        if limit is None and args.smoke_eval:
            limit = int((protocol.get("pilot") or {}).get("evaluation_scenario_limit", 3))
        evaluator = EvalSetEvaluator(
            manifest_path,
            {"base_config_path": str(_resolve(protocol["base_scenario_config"]))},
        )
        summary = evaluator.evaluate_checkpoint(
            checkpoint_path,
            episodes_per_scenario=episodes_per_scenario,
            seed=args.seed,
            scenario_limit=limit,
            group=args.group,
            checkpoint_role=checkpoint_role,
            opponent_mode=opponent_mode,
            opponent_checkpoint=opponent_checkpoint,
        )
    summary["evaluation_protocol_version"] = evaluation_protocol.get("protocol_version")
    summary["agent_team"] = evaluation_protocol.get("agent_team", "A")
    summary["opponent_team"] = evaluation_protocol.get("opponent_team", "B")
    summary["same_actor_allowed"] = bool(evaluation_protocol.get("same_actor_allowed", False))

    mode = "smoke_eval" if args.smoke_eval else "full_eval"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_root / args.group / f"seed_{args.seed}" / "eval_set" / f"{checkpoint_role}_{opponent_mode}_{mode}"
    )
    output_path = output_dir / "eval_set_summary.json"
    EvalSetEvaluator.save(summary, output_path)
    print(json.dumps({"output_path": str(output_path), **summary}, indent=2, sort_keys=True))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


if __name__ == "__main__":
    main()
