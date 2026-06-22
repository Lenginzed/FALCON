"""Evaluate baseline checkpoints on the shared frozen scenario set."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .policy_evaluator import PolicyEvaluator

ROOT_DIR = Path(__file__).resolve().parents[1]


class EvalSetEvaluator:
    """Run deterministic policy evaluation across a frozen scenario manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        policy_evaluator_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.policy_evaluator_config = dict(policy_evaluator_config or {})

    def load_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, MappingABC):
            raise ValueError(f"Eval scenario manifest is not an object: {self.manifest_path}")
        return dict(data)

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        episodes_per_scenario: int = 1,
        seed: int = 0,
        scenario_limit: Optional[int] = None,
        group: Optional[str] = None,
        checkpoint_role: str = "latest",
        opponent_mode: str = "fixed_checkpoint",
        opponent_checkpoint: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        eval_started_at = _timestamp()
        eval_start = time.perf_counter()
        manifest = self.load_manifest()
        scenarios = list(manifest.get("scenarios") or [])
        if scenario_limit is not None:
            scenarios = scenarios[: max(int(scenario_limit), 0)]
        evaluator_config = dict(self.policy_evaluator_config)
        evaluator_config.update(
            {
                "opponent_mode": opponent_mode,
                "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint is not None else None,
            }
        )
        evaluator = PolicyEvaluator(evaluator_config)
        per_scenario: List[Dict[str, Any]] = []
        warnings: List[str] = []
        if opponent_mode == "fixed_checkpoint" and opponent_checkpoint is None:
            warnings.append(
                "opponent_mode=fixed_checkpoint requires opponent_checkpoint; "
                "same-actor fallback is disabled."
            )
        for idx, scenario in enumerate(scenarios):
            scenario_started_at = _timestamp()
            scenario_start = time.perf_counter()
            yaml_path = _resolve_path(scenario.get("scenario_yaml_path"))
            result = evaluator.evaluate_policy_on_scenario(
                checkpoint_path,
                yaml_path,
                num_episodes=int(episodes_per_scenario),
                seed=int(seed) + idx * 100,
            )
            scenario_runtime = round(time.perf_counter() - scenario_start, 3)
            row = {
                "schema_version": "falcon.eval_set_scenario_result.v1",
                "scenario_id": scenario.get("scenario_id"),
                "scenario_group": scenario.get("scenario_group"),
                "scenario_yaml_path": str(yaml_path),
                "started_at": scenario_started_at,
                "finished_at": _timestamp(),
                "runtime_seconds": scenario_runtime,
                "runtime_human_readable": _human_duration(scenario_runtime),
                "real_policy_eval_available": bool(result.get("real_policy_eval_available")),
                "win_rate": result.get("win_rate"),
                "mean_return": result.get("mean_return"),
                "std_return": result.get("std_return"),
                "mean_episode_length": result.get("mean_episode_length"),
                "failure_stage": result.get("failure_stage"),
                "policy_eval": result,
                "warnings": list(result.get("warnings") or []),
            }
            warnings.extend(row["warnings"])
            per_scenario.append(row)
        successful = [row for row in per_scenario if row.get("real_policy_eval_available")]
        eval_runtime = round(time.perf_counter() - eval_start, 3)
        return {
            "schema_version": "falcon.eval_set_summary.v1",
            "started_at": eval_started_at,
            "finished_at": _timestamp(),
            "eval_set_runtime_seconds": eval_runtime,
            "eval_set_runtime_human_readable": _human_duration(eval_runtime),
            "group": group,
            "checkpoint_role": checkpoint_role,
            "checkpoint_path": str(checkpoint_path),
            "agent_checkpoint": str(checkpoint_path),
            "opponent_mode": opponent_mode,
            "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint is not None else None,
            "same_actor": opponent_mode == "same_actor",
            "same_actor_eval": opponent_mode == "same_actor",
            "same_checkpoint": bool(
                opponent_checkpoint is not None
                and _same_path(Path(checkpoint_path), Path(opponent_checkpoint))
            ),
            "eval_protocol_frozen": bool(
                opponent_mode == "fixed_checkpoint"
                and opponent_checkpoint is not None
                and Path(opponent_checkpoint).exists()
                and not _same_path(Path(checkpoint_path), Path(opponent_checkpoint))
            ),
            "manifest_path": str(self.manifest_path),
            "manifest_scenario_count": int(manifest.get("scenario_count", len(manifest.get("scenarios") or []))),
            "episodes_per_scenario": int(episodes_per_scenario),
            "scenario_limit": scenario_limit,
            "num_scenarios_requested": len(scenarios),
            "num_scenarios_evaluated": len(successful),
            "num_scenarios_failed": len(per_scenario) - len(successful),
            "per_scenario_results": per_scenario,
            "aggregate_result": _aggregate(successful),
            "eval_group_breakdown": _group_breakdown(successful),
            "failure_stage": None if len(successful) == len(scenarios) else "scenario_evaluation",
            "warnings": sorted(set(str(item) for item in warnings if item)),
        }

    @staticmethod
    def save(result: Mapping[str, Any], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(dict(result), f, indent=2, sort_keys=True)


def resolve_group_checkpoint(
    results_root: str | Path,
    group: str,
    seed: int,
    checkpoint: str = "latest",
) -> Optional[Path]:
    direct = Path(str(checkpoint))
    if checkpoint not in {"latest", "best"} and direct.exists():
        return direct
    seed_dir = Path(results_root) / group / f"seed_{int(seed)}"
    pilot_summary = seed_dir / "pilot_run" / "pilot_run_summary.json"
    if pilot_summary.exists():
        data = _load_json(pilot_summary)
        value = data.get(f"{checkpoint}_checkpoint_path")
        if value and Path(str(value)).exists():
            return Path(str(value))
    smoke_summary = seed_dir / "smoke_run" / "baseline_experiment_summary.json"
    if smoke_summary.exists():
        data = _load_json(smoke_summary)
        value = data.get("best_checkpoint_path") if checkpoint == "best" else data.get("checkpoint_path")
        if not value:
            value = data.get("checkpoint_path")
        if value and Path(str(value)).exists():
            return Path(str(value))
    registry_candidates = [
        seed_dir / "pilot_run" / "checkpoint_registry.json",
        seed_dir / "pilot_run" / "controller" / "falcon_checkpoint_registry.json",
    ]
    for registry_path in registry_candidates:
        if not registry_path.exists():
            continue
        data = _load_json(registry_path)
        key = "best_checkpoint" if checkpoint == "best" else "latest_checkpoint"
        value = data.get(key)
        if not value and checkpoint == "latest":
            value = data.get("current_checkpoint")
        if value and Path(str(value)).exists():
            return Path(str(value))
    return None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.eval_set_aggregate.v1",
        "final_win_rate": _mean(rows, "win_rate"),
        "final_mean_return": _mean(rows, "mean_return"),
        "mean_std_return": _mean(rows, "std_return"),
        "mean_episode_length": _mean(rows, "mean_episode_length"),
        "num_scenarios": len(rows),
    }


def _group_breakdown(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scenario_group") or "unknown")].append(row)
    return {
        key: {
            "num_scenarios": len(items),
            "win_rate": _mean(items, "win_rate"),
            "mean_return": _mean(items, "mean_return"),
            "mean_episode_length": _mean(items, "mean_episode_length"),
        }
        for key, items in sorted(grouped.items())
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key)))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 6) if values else 0.0


def _resolve_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _human_duration(seconds: Any) -> str:
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "unknown"
    hours, remainder = divmod(max(total, 0), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
