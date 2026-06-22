#!/usr/bin/env python
"""Screen an independent MAPPO candidate before freezing it as eval opponent."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import load_yaml  # noqa: E402
from falcon.eval_set_evaluator import EvalSetEvaluator, resolve_group_checkpoint  # noqa: E402

DEFAULT_ROOT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "opponents" / "fixed_baseline_opponent"
DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"
DEFAULT_FORMAL_MANIFEST = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "eval_opponent.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed baseline opponent strength on the frozen eval set.")
    parser.add_argument("--candidate-manifest", default=None)
    parser.add_argument("--opponent-checkpoint", default=None)
    parser.add_argument("--opponent-seed", type=int, default=999)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--agent-checkpoint", default=None)
    parser.add_argument("--agent-group", default="mappo_base")
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--agent-checkpoint-role", choices=("latest", "best"), default="latest")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "eval_opponent_strength.json"))
    parser.add_argument("--formal-manifest", default=str(DEFAULT_FORMAL_MANIFEST))
    parser.add_argument("--update-formal-on-accept", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    protocol_path = _resolve(args.protocol)
    protocol = load_yaml(protocol_path)
    evaluation = dict(protocol.get("evaluation") or {})
    candidate = _load_candidate(args.candidate_manifest)
    opponent_path = _resolve(args.opponent_checkpoint or candidate.get("checkpoint_path") or "")
    training_steps = int(args.training_steps if args.training_steps is not None else candidate.get("training_steps") or 0)
    opponent_seed = int(candidate.get("seed", args.opponent_seed))
    agent_path = (
        _resolve(args.agent_checkpoint)
        if args.agent_checkpoint
        else resolve_group_checkpoint(_resolve(protocol["output_root"]), args.agent_group, args.agent_seed, args.agent_checkpoint_role)
    )
    output_path = _resolve(args.output)
    warnings = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    if agent_path is None or not agent_path.exists():
        result = _failure_result(opponent_path, agent_path, opponent_seed, training_steps, "agent_checkpoint_resolution")
        _write_json(output_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not opponent_path.exists():
        result = _failure_result(opponent_path, agent_path, opponent_seed, training_steps, "opponent_checkpoint_resolution")
        _write_json(output_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    manifest_path = _resolve(evaluation.get("eval_scenarios") or protocol["evaluation_scenarios"])
    evaluator = EvalSetEvaluator(
        manifest_path,
        {"base_config_path": str(_resolve(protocol["base_scenario_config"]))},
    )
    eval_result = evaluator.evaluate_checkpoint(
        agent_path,
        episodes_per_scenario=args.episodes_per_scenario,
        seed=args.agent_seed,
        scenario_limit=args.scenario_limit,
        group="fixed_opponent_strength_screen",
        checkpoint_role=args.agent_checkpoint_role,
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=opponent_path,
    )
    requested = int(eval_result.get("num_scenarios_requested", 0))
    evaluated = int(eval_result.get("num_scenarios_evaluated", 0))
    failed = int(eval_result.get("num_scenarios_failed", 0))
    agent_win_rate = float((eval_result.get("aggregate_result") or {}).get("final_win_rate", 0.0))
    opponent_win_rate = _mean_policy_metric(eval_result.get("per_scenario_results") or [], "loss_rate")
    full_manifest = int(eval_result.get("manifest_scenario_count", 0))
    full_eval_completed = bool(requested == full_manifest and evaluated == full_manifest and failed == 0)
    checkpoint_loadable = bool(evaluated > 0 and eval_result.get("failure_stage") is None)
    non_extreme_strength = bool(0.05 < agent_win_rate < 0.95)
    accepted = bool(checkpoint_loadable and full_eval_completed and non_extreme_strength)

    if not full_eval_completed:
        warnings.append("Candidate was not evaluated successfully on the complete frozen eval scenario set.")
    if agent_win_rate <= 0.05:
        warnings.append("Agent win rate is <= 0.05; candidate opponent may be too strong.")
    if agent_win_rate >= 0.95:
        warnings.append("Agent win rate is >= 0.95; candidate opponent may be too weak.")
    if training_steps < 2048:
        warnings.append("Candidate has fewer than 2048 training steps; use caution even if the strength screen passes.")
    if args.episodes_per_scenario > 1 and _all_scenario_returns_deterministic(eval_result.get("per_scenario_results") or []):
        warnings.append(
            "Repeated episodes were deterministic within each scenario; strength confidence comes from scenario diversity, "
            "not stochastic repeat variance."
        )
    warnings.extend(eval_result.get("warnings") or [])

    strength_summary = {
        "schema_version": "falcon.fixed_opponent_strength_eval.v1",
        "opponent_checkpoint_path": str(opponent_path),
        "opponent_checkpoint_sha256": _sha256(opponent_path),
        "opponent_seed": opponent_seed,
        "training_steps": training_steps,
        "eval_scenarios_count": requested,
        "eval_manifest_scenarios_count": full_manifest,
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "agent_checkpoint_path": str(agent_path),
        "agent_checkpoint_role": args.agent_checkpoint_role,
        "agent_win_rate_against_opponent": agent_win_rate,
        "opponent_win_rate": opponent_win_rate,
        "mean_return": (eval_result.get("aggregate_result") or {}).get("final_mean_return"),
        "scenario_group_breakdown": eval_result.get("eval_group_breakdown") or {},
        "checkpoint_loadable": checkpoint_loadable,
        "full_eval_completed": full_eval_completed,
        "non_extreme_strength": non_extreme_strength,
        "accepted_for_formal_eval": accepted,
        "acceptance_criteria": {
            "checkpoint_loadable": True,
            "complete_eval_set_required": True,
            "minimum_agent_win_rate_exclusive": 0.05,
            "maximum_agent_win_rate_exclusive": 0.95,
            "zero_eval_failures_required": True,
        },
        "num_eval_failures": failed,
        "evaluation_result": eval_result,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "failure_stage": None if accepted else "candidate_not_accepted",
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }
    _write_json(output_path, strength_summary)

    if accepted and args.update_formal_on_accept:
        formal_manifest = {
            "schema_version": "falcon.eval_opponent_manifest.v1",
            "opponent_id": candidate.get("opponent_id") or f"independent_mappo_seed{opponent_seed}_steps{training_steps}",
            "opponent_mode": "fixed_checkpoint",
            "checkpoint_path": _portable_path(opponent_path),
            "checkpoint_sha256": _sha256(opponent_path),
            "sha256": _sha256(opponent_path),
            "seed": opponent_seed,
            "training_steps": training_steps,
            "source": "independent_mappo_baseline",
            "environment": protocol.get("environment", "2v2/NoWeapon/Selfplay"),
            "accepted_for_formal_eval": True,
            "strength_eval_summary": _portable_path(output_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "warnings": strength_summary["warnings"],
        }
        _write_json(_resolve(args.formal_manifest), formal_manifest)
        if args.candidate_manifest:
            candidate["accepted_for_formal_eval"] = True
            candidate["strength_eval_summary"] = _portable_path(output_path)
            candidate["checkpoint_sha256"] = _sha256(opponent_path)
            candidate["sha256"] = _sha256(opponent_path)
            candidate["warnings"] = strength_summary["warnings"]
            _write_json(_resolve(args.candidate_manifest), candidate)
        evaluation["opponent_checkpoint"] = _portable_path(opponent_path)
        evaluation["opponent_manifest"] = _portable_path(_resolve(args.formal_manifest))
        protocol["evaluation"] = evaluation
        protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=False), encoding="utf-8")
        strength_summary["formal_manifest_updated"] = True
        strength_summary["protocol_updated"] = True
        _write_json(output_path, strength_summary)
    else:
        strength_summary["formal_manifest_updated"] = False
        strength_summary["protocol_updated"] = False
        _write_json(output_path, strength_summary)

    print(json.dumps({"output_path": str(output_path), **strength_summary}, indent=2, sort_keys=True))


def _load_candidate(value: str | None) -> dict:
    if value:
        return json.loads(_resolve(value).read_text(encoding="utf-8"))
    candidates = sorted(DEFAULT_ROOT.glob("candidate_seed*_steps*/fixed_opponent_candidate_manifest.json"))
    return json.loads(candidates[-1].read_text(encoding="utf-8")) if candidates else {}


def _failure_result(opponent_path: Path, agent_path: Path | None, seed: int, steps: int, stage: str) -> dict:
    return {
        "schema_version": "falcon.fixed_opponent_strength_eval.v1",
        "opponent_checkpoint_path": str(opponent_path),
        "opponent_seed": seed,
        "training_steps": steps,
        "agent_checkpoint_path": str(agent_path) if agent_path else None,
        "eval_scenarios_count": 0,
        "episodes_per_scenario": 0,
        "accepted_for_formal_eval": False,
        "formal_manifest_updated": False,
        "failure_stage": stage,
        "warnings": [f"Could not resolve required checkpoint during {stage}."],
    }


def _mean_policy_metric(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float((row.get("policy_eval") or {}).get(key)))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 6) if values else 0.0


def _all_scenario_returns_deterministic(rows: list[dict]) -> bool:
    values = []
    for row in rows:
        try:
            values.append(float(row.get("std_return")))
        except (TypeError, ValueError):
            continue
    return bool(values) and all(value == 0.0 for value in values)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


if __name__ == "__main__":
    main()
