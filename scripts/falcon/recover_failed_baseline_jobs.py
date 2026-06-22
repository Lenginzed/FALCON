#!/usr/bin/env python
"""Recover baseline jobs that finished training but failed during summary/eval handoff."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import (  # noqa: E402
    GROUP_DEFINITIONS,
    SUPPORTED_GROUPS,
    _human_duration,
    _safe_float,
    _sum_safe_float,
)
from falcon.eval_set_evaluator import EvalSetEvaluator, resolve_group_checkpoint  # noqa: E402

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover failed formal baseline jobs from existing checkpoints.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--groups", nargs="+", default=list(SUPPORTED_GROUPS), choices=SUPPORTED_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--use-existing-checkpoints", action="store_true")
    parser.add_argument("--run-missing-eval", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = _load_yaml(protocol_path)
    output_root = _resolve(protocol.get("output_root"))
    reports_dir = protocol_path.parents[1] / "reports"
    status_path = reports_dir / "formal_baseline_queue_status.json"
    status = _load_json(status_path) if status_path.exists() else _new_status(protocol_path, output_root)

    recovered: List[Dict[str, Any]] = []
    for group in args.groups:
        for seed in args.seeds:
            item = recover_job(
                protocol=protocol,
                protocol_path=protocol_path,
                output_root=output_root,
                group=group,
                seed=int(seed),
                status=status,
                use_existing_checkpoints=bool(args.use_existing_checkpoints),
                run_missing_eval=bool(args.run_missing_eval),
                force_eval=bool(args.force_eval),
                dry_run=bool(args.dry_run),
            )
            recovered.append(item)

    if not args.dry_run:
        _save_status(status, reports_dir)
    result = {
        "schema_version": "falcon.baseline_recovery_result.v1",
        "created_at": _timestamp(),
        "protocol_path": str(protocol_path),
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
        "recovered_jobs": recovered,
        "num_jobs_requested": len(recovered),
        "num_recovered": sum(1 for item in recovered if item.get("recovered")),
        "num_eval_completed": sum(1 for item in recovered if item.get("eval_21_scenarios_complete")),
    }
    if not args.dry_run:
        out = reports_dir / "formal_baseline_recovery_summary.json"
        _write_json(out, result)
        result["recovery_summary_path"] = str(out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def recover_job(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    output_root: Path,
    group: str,
    seed: int,
    status: Dict[str, Any],
    use_existing_checkpoints: bool,
    run_missing_eval: bool,
    force_eval: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    seed_dir = output_root / group / f"seed_{seed}"
    pilot_dir = seed_dir / "pilot_run"
    train_summary_path = pilot_dir / "baseline_experiment_summary.json"
    eval_dir_name = f"{_checkpoint_selection(protocol)}_{_opponent_mode(protocol)}_full_eval"
    eval_summary_path = seed_dir / "eval_set" / eval_dir_name / "eval_set_summary.json"
    job = _find_or_create_job(status, group, seed)
    warnings: List[str] = []

    registry_path = _registry_path(pilot_dir, group)
    registry = _load_json(registry_path) if registry_path.exists() else {}
    best_checkpoint = registry.get("best_checkpoint") or registry.get("best_checkpoint_path")
    latest_checkpoint = registry.get("latest_checkpoint") or registry.get("current_checkpoint")
    checkpoint_count = _count_existing_checkpoints(registry)
    best_exists = bool(best_checkpoint and Path(str(best_checkpoint)).exists())
    latest_exists = bool(latest_checkpoint and Path(str(latest_checkpoint)).exists())

    if use_existing_checkpoints and (not registry_path.exists() or not best_exists or not latest_exists):
        reason = "checkpoint_artifacts_missing"
        _record_job_failure(job, reason, warnings)
        return _result_row(
            group,
            seed,
            False,
            reason,
            train_summary_path,
            eval_summary_path,
            registry_path,
            best_checkpoint,
            latest_checkpoint,
            checkpoint_count,
            warnings,
        )

    if train_summary_path.exists():
        train_summary = _load_json(train_summary_path)
        train_rebuilt = False
    else:
        train_summary, train_rebuilt = _rebuild_train_summary(
            protocol=protocol,
            protocol_path=protocol_path,
            output_root=output_root,
            group=group,
            seed=seed,
            pilot_dir=pilot_dir,
            registry_path=registry_path,
            registry=registry,
        )
        warnings.extend(train_summary.get("warnings") or [])
        if not dry_run:
            _write_json(pilot_dir / "pilot_run_summary.json", train_summary["execution"])
            _write_json(train_summary_path, train_summary)

    eval_summary: Dict[str, Any] = {}
    eval_ran = False
    if eval_summary_path.exists() and not force_eval:
        eval_summary = _load_json(eval_summary_path)
    elif run_missing_eval:
        checkpoint = resolve_group_checkpoint(output_root, group, seed, _checkpoint_selection(protocol))
        if checkpoint is None and best_checkpoint and Path(str(best_checkpoint)).exists():
            checkpoint = Path(str(best_checkpoint))
        if checkpoint is None:
            warnings.append("Could not resolve best checkpoint for eval recovery.")
            eval_summary = {"failure_stage": "checkpoint_resolution", "warnings": warnings}
        else:
            eval_summary = _run_fixed_opponent_eval(protocol, group, seed, checkpoint)
            eval_ran = True
            if not dry_run:
                EvalSetEvaluator.save(eval_summary, eval_summary_path)
    else:
        warnings.append("Eval summary missing and --run-missing-eval was not set.")

    eval_complete = _eval_complete(eval_summary)
    same_actor_false = eval_summary.get("same_actor") is False and eval_summary.get("same_actor_eval") is False
    opponent_fixed = eval_summary.get("opponent_mode") == "fixed_checkpoint"
    recovered = bool(
        train_summary_path.exists() or train_rebuilt
    ) and bool(best_exists and latest_exists and eval_complete and same_actor_false and opponent_fixed)
    if not dry_run:
        job.update(
            {
                "status": "completed_recovered" if recovered else "recovery_failed",
                "train_summary_path": str(train_summary_path) if train_summary_path.exists() or train_rebuilt else None,
                "eval_summary_path": str(eval_summary_path) if eval_summary_path.exists() or eval_ran else None,
                "failure_stage": None if recovered else (eval_summary.get("failure_stage") or "recovery_incomplete"),
                "finished_at": _timestamp(),
                "recovered_at": _timestamp(),
                "warnings": _merge_warnings(job.get("warnings"), warnings, eval_summary.get("warnings")),
            }
        )
    return _result_row(
        group,
        seed,
        recovered,
        None if recovered else (eval_summary.get("failure_stage") or "recovery_incomplete"),
        train_summary_path,
        eval_summary_path,
        registry_path,
        best_checkpoint,
        latest_checkpoint,
        checkpoint_count,
        _merge_warnings(warnings, eval_summary.get("warnings")),
        train_rebuilt=train_rebuilt,
        eval_ran=eval_ran,
        eval_21_scenarios_complete=eval_complete,
        same_actor_false=same_actor_false,
        opponent_mode=eval_summary.get("opponent_mode"),
    )


def _rebuild_train_summary(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    output_root: Path,
    group: str,
    seed: int,
    pilot_dir: Path,
    registry_path: Path,
    registry: Mapping[str, Any],
) -> tuple[Dict[str, Any], bool]:
    round_summaries = [_load_json(path) for path in sorted(pilot_dir.glob("round_*/round_summary.json"))]
    warnings: List[str] = []
    max_rounds = int(protocol.get("max_rounds") or 0)
    completed_rounds = len(round_summaries)
    failure_stages = [row.get("failure_stage") for row in round_summaries if row.get("failure_stage")]
    if completed_rounds < max_rounds:
        warnings.append(f"Only {completed_rounds}/{max_rounds} round summaries were found.")
    if failure_stages:
        warnings.append(f"Recovered round summaries contain failure stages: {sorted(set(map(str, failure_stages)))}")
    started_at = _first_value(round_summaries, "started_at")
    finished_at = _last_value(round_summaries, "finished_at")
    runtime = _sum_safe_float(round_summaries, "round_runtime_seconds")
    registry_best = registry.get("best_checkpoint") or registry.get("best_checkpoint_path")
    registry_latest = registry.get("latest_checkpoint") or registry.get("current_checkpoint")
    params = {
        "max_rounds": int(protocol.get("max_rounds") or max_rounds),
        "train_steps_per_round": int(protocol.get("train_steps_per_round") or 0),
        "eval_episodes_per_round": int(protocol.get("eval_episodes_per_round") or 0),
        "policy_eval_episodes_per_candidate": int(protocol.get("policy_eval_episodes_per_candidate") or 0),
        "qwen_candidates_per_round": int(protocol.get("qwen_candidates_per_round") or 0),
        "random_candidates_per_round": int(protocol.get("random_candidates_per_round") or 0),
    }
    execution = {
        "schema_version": "falcon.baseline_pilot_summary.v1",
        "group": group,
        "seed": seed,
        **GROUP_DEFINITIONS[group],
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": runtime,
        "runtime_human_readable": _human_duration(runtime),
        "training_runtime_seconds": _sum_safe_float(round_summaries, "training_runtime_seconds"),
        "policy_eval_runtime_seconds": _sum_safe_float(round_summaries, "policy_eval_runtime_seconds"),
        "qwen_runtime_seconds": _sum_safe_float(round_summaries, "qwen_runtime_seconds"),
        "protocol_parameters": params,
        "completed_rounds": completed_rounds,
        "max_rounds": max_rounds,
        "all_rounds_finished": completed_rounds == max_rounds and not failure_stages,
        "round_summaries": round_summaries,
        "checkpoint_registry_path": str(registry_path),
        "latest_checkpoint_path": registry_latest,
        "best_checkpoint_path": registry_best,
        "failure_stage": None if completed_rounds == max_rounds and not failure_stages else "recovered_rounds_incomplete",
        "warnings": sorted(set(warnings + ["Recovered from existing checkpoint registry and round summaries."])),
    }
    group_config_path = _resolve(protocol.get("group_configs", {}).get(group))
    group_config = _load_yaml(group_config_path) if group_config_path.exists() else {}
    summary = {
        "schema_version": "falcon.baseline_experiment_run.v1",
        "protocol_path": str(protocol_path),
        "protocol_status": protocol.get("status"),
        "group": group,
        "group_config_path": str(group_config_path),
        "group_config": group_config,
        "seed": seed,
        "mode": "pilot_run",
        "output_dir": str(pilot_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": runtime,
        "runtime_human_readable": _human_duration(runtime),
        "ready": execution["failure_stage"] is None,
        "dry_run": False,
        "smoke_run": False,
        "pilot_run": True,
        "input_checks": {},
        "execution": execution,
        "checkpoint_path": registry_latest,
        "best_checkpoint_path": registry_best,
        "failure_stage": execution["failure_stage"],
        "warnings": execution["warnings"],
    }
    return summary, True


def _run_fixed_opponent_eval(
    protocol: Mapping[str, Any],
    group: str,
    seed: int,
    checkpoint: Path,
) -> Dict[str, Any]:
    evaluation = dict(protocol.get("evaluation") or {})
    manifest_path = _resolve(evaluation.get("eval_scenarios") or protocol.get("evaluation_scenarios"))
    opponent_mode = str(evaluation.get("opponent_mode") or "fixed_checkpoint")
    opponent_checkpoint = _resolve(evaluation.get("opponent_checkpoint")) if evaluation.get("opponent_checkpoint") else None
    evaluator = EvalSetEvaluator(manifest_path, {"base_config_path": str(_resolve(protocol["base_scenario_config"]))})
    summary = evaluator.evaluate_checkpoint(
        checkpoint,
        episodes_per_scenario=int(evaluation.get("episodes_per_scenario") or 1),
        seed=seed,
        group=group,
        checkpoint_role=_checkpoint_selection(protocol),
        opponent_mode=opponent_mode,
        opponent_checkpoint=opponent_checkpoint,
    )
    summary["evaluation_protocol_version"] = evaluation.get("protocol_version")
    summary["agent_team"] = evaluation.get("agent_team", "A")
    summary["opponent_team"] = evaluation.get("opponent_team", "B")
    summary["same_actor_allowed"] = bool(evaluation.get("same_actor_allowed", False))
    return summary


def _result_row(
    group: str,
    seed: int,
    recovered: bool,
    failure_stage: Optional[str],
    train_summary_path: Path,
    eval_summary_path: Path,
    registry_path: Path,
    best_checkpoint: Any,
    latest_checkpoint: Any,
    checkpoint_count: int,
    warnings: Sequence[str],
    train_rebuilt: bool = False,
    eval_ran: bool = False,
    eval_21_scenarios_complete: bool = False,
    same_actor_false: bool = False,
    opponent_mode: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.baseline_recovery_job.v1",
        "group": group,
        "seed": seed,
        "recovered": recovered,
        "failure_stage": failure_stage,
        "train_summary_path": str(train_summary_path),
        "train_summary_exists": train_summary_path.exists() or train_rebuilt,
        "train_rebuilt": train_rebuilt,
        "eval_summary_path": str(eval_summary_path),
        "eval_summary_exists": eval_summary_path.exists() or eval_ran,
        "eval_ran": eval_ran,
        "eval_21_scenarios_complete": eval_21_scenarios_complete,
        "same_actor_false": same_actor_false,
        "opponent_mode": opponent_mode,
        "checkpoint_registry_path": str(registry_path),
        "checkpoint_registry_exists": registry_path.exists(),
        "best_checkpoint_path": str(best_checkpoint) if best_checkpoint else None,
        "latest_checkpoint_path": str(latest_checkpoint) if latest_checkpoint else None,
        "best_checkpoint_exists": bool(best_checkpoint and Path(str(best_checkpoint)).exists()),
        "latest_checkpoint_exists": bool(latest_checkpoint and Path(str(latest_checkpoint)).exists()),
        "checkpoint_count": checkpoint_count,
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }


def _registry_path(pilot_dir: Path, group: str) -> Path:
    if group == "falcon_no_fsn":
        candidate = pilot_dir / "controller" / "falcon_checkpoint_registry.json"
        if candidate.exists():
            return candidate
    return pilot_dir / "checkpoint_registry.json"


def _checkpoint_selection(protocol: Mapping[str, Any]) -> str:
    return str((protocol.get("evaluation") or {}).get("checkpoint_selection") or "best")


def _opponent_mode(protocol: Mapping[str, Any]) -> str:
    return str((protocol.get("evaluation") or {}).get("opponent_mode") or "fixed_checkpoint")


def _eval_complete(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary
        and summary.get("failure_stage") is None
        and int(summary.get("num_scenarios_evaluated") or 0) == 21
        and int(summary.get("num_scenarios_failed") or 0) == 0
    )


def _count_existing_checkpoints(registry: Mapping[str, Any]) -> int:
    count = 0
    for item in registry.get("checkpoints") or []:
        checkpoint = item.get("checkpoint_path")
        if checkpoint and Path(str(checkpoint)).exists():
            count += 1
    return count


def _find_or_create_job(status: Dict[str, Any], group: str, seed: int) -> Dict[str, Any]:
    jobs = status.setdefault("jobs", [])
    for job in jobs:
        if job.get("group") == group and int(job.get("seed", -1)) == int(seed):
            return job
    job = {
        "schema_version": "falcon.formal_baseline_queue_job.v1",
        "group": group,
        "seed": int(seed),
        "status": "pending_recovery",
        "warnings": [],
    }
    jobs.append(job)
    return job


def _record_job_failure(job: Dict[str, Any], failure_stage: str, warnings: Sequence[str]) -> None:
    job["status"] = "recovery_failed"
    job["failure_stage"] = failure_stage
    job["finished_at"] = _timestamp()
    job["warnings"] = _merge_warnings(job.get("warnings"), warnings)


def _save_status(status: Dict[str, Any], reports_dir: Path) -> None:
    status["updated_at"] = _timestamp()
    status["summary"] = _status_counts(status.get("jobs") or [])
    status_path = reports_dir / "formal_baseline_queue_status.json"
    csv_path = reports_dir / "formal_baseline_queue_status.csv"
    _write_json(status_path, status)
    fields = [
        "group",
        "seed",
        "status",
        "started_at",
        "finished_at",
        "runtime_seconds",
        "train_summary_path",
        "eval_summary_path",
        "failure_stage",
        "warnings",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in status.get("jobs") or []:
            row = {key: job.get(key) for key in fields}
            row["warnings"] = " | ".join(str(item) for item in _as_list(row.get("warnings")))
            writer.writerow(row)


def _status_counts(jobs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for job in jobs:
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    counts["total"] = len(jobs)
    return counts


def _new_status(protocol_path: Path, output_root: Path) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.formal_baseline_queue_status.v1",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "mode": "recovery",
        "protocol_path": str(protocol_path),
        "output_root": str(output_root),
        "jobs": [],
        "summary": {},
    }


def _first_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    for row in rows:
        if row.get(key) is not None:
            return row.get(key)
    return None


def _last_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        if row.get(key) is not None:
            return row.get(key)
    return None


def _merge_warnings(*parts: Any) -> List[str]:
    warnings: List[str] = []
    for part in parts:
        warnings.extend(str(item) for item in _as_list(part) if item)
    return sorted(set(warnings))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(data)), f, indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MappingABC):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    number = _safe_float(value, None)
    if isinstance(value, float) and number is None:
        return None
    return value


def _resolve(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT_DIR / path


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    main()
