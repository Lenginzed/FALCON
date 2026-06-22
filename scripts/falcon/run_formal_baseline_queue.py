#!/usr/bin/env python
"""Queue formal seed/group baseline pilots and fixed-opponent evals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import BaselineExperimentRunner, SUPPORTED_GROUPS, load_yaml  # noqa: E402
from falcon.eval_set_evaluator import EvalSetEvaluator, resolve_group_checkpoint  # noqa: E402
from falcon.experiment_time_estimator import ExperimentTimeEstimator  # noqa: E402

DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
DEFAULT_PROTOCOL = DEFAULT_EXPERIMENT_DIR / "configs" / "experiment_protocol.yaml"
DEFAULT_REPORTS_DIR = DEFAULT_EXPERIMENT_DIR / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or estimate the formal FALCON baseline queue.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--groups", nargs="+", choices=SUPPORTED_GROUPS, default=list(SUPPORTED_GROUPS))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-failure", choices=("true", "false"), default="false")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = load_yaml(protocol_path)
    seeds = args.seeds or [int(seed) for seed in protocol.get("planned_formal_seeds") or protocol.get("seeds") or [0]]
    output_root = _resolve(args.output_root or protocol["output_root"])
    reports_dir = DEFAULT_REPORTS_DIR
    estimator = ExperimentTimeEstimator(protocol_path, results_root=output_root, reports_dir=reports_dir)
    overrides = _formal_overrides(protocol)
    jobs = _build_jobs(args.groups, seeds, output_root, estimator, overrides, skip_existing=args.skip_existing)
    status = _new_status(protocol_path, output_root, jobs, mode=_mode(args))
    _save_status(status, reports_dir)

    if args.estimate_only:
        estimate = estimator.estimate_queue(seeds, args.groups, overrides=overrides, skip_existing=args.skip_existing)
        paths = estimator.export_estimate(estimate, reports_dir)
        status["estimate_paths"] = paths
        status["estimate_summary"] = {
            "estimated_total_time_seconds": estimate.get("estimated_total_time_seconds"),
            "estimated_total_time_human_readable": estimate.get("estimated_total_time_human_readable"),
            "slowest_group": estimate.get("slowest_group"),
        }
        _save_status(status, reports_dir)
        print(json.dumps({"status_path": str(_status_json_path(reports_dir)), **_public_status(status), "estimate": estimate}, indent=2, sort_keys=True))
        return

    stop_on_failure = args.stop_on_failure == "true"
    for index, job in enumerate(status["jobs"]):
        if args.skip_existing and _job_complete(output_root, job["group"], int(job["seed"])):
            _mark_job(job, "skipped", warnings=["Existing completed pilot/eval outputs found."])
            _save_status(status, reports_dir)
            continue
        if args.dry_run:
            _run_single_dry_run(job, protocol_path, output_root)
            _save_status(status, reports_dir)
            continue
        _run_single_formal_job(job, protocol, protocol_path, output_root, overrides, resume=bool(args.resume))
        _save_status(status, reports_dir)
        if job.get("status") == "failed" and stop_on_failure:
            for remaining in status["jobs"][index + 1 :]:
                if remaining.get("status") == "pending":
                    remaining["status"] = "skipped"
                    remaining["warnings"] = list(remaining.get("warnings") or []) + ["Skipped because stop_on_failure=true after a prior job failed."]
            _save_status(status, reports_dir)
            break
    status["finished_at"] = _timestamp()
    status["runtime_seconds"] = round(time.time() - float(status["_started_time_seconds"]), 3)
    status["runtime_human_readable"] = _human_duration(status["runtime_seconds"])
    status.pop("_started_time_seconds", None)
    _save_status(status, reports_dir)
    print(json.dumps({"status_path": str(_status_json_path(reports_dir)), **_public_status(status)}, indent=2, sort_keys=True))


def _build_jobs(
    groups: Sequence[str],
    seeds: Sequence[int],
    output_root: Path,
    estimator: ExperimentTimeEstimator,
    overrides: Mapping[str, Any],
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    jobs = []
    for group in groups:
        for seed in seeds:
            estimate = estimator.estimate_job(group, int(seed), overrides=overrides)
            jobs.append(
                {
                    "schema_version": "falcon.formal_baseline_queue_job.v1",
                    "group": group,
                    "seed": int(seed),
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "runtime_seconds": None,
                    "runtime_human_readable": "unknown",
                    "train_summary_path": None,
                    "eval_summary_path": None,
                    "failure_stage": None,
                    "warnings": [],
                    "estimated_total_time_seconds": estimate.get("estimated_total_time_seconds"),
                    "estimated_total_time_human_readable": estimate.get("estimated_total_time_human_readable"),
                    "would_skip_existing": bool(skip_existing and _job_complete(output_root, group, int(seed))),
                }
            )
    return jobs


def _run_single_dry_run(job: Dict[str, Any], protocol_path: Path, output_root: Path) -> None:
    _mark_job(job, "running")
    start = time.time()
    try:
        output_dir = output_root / job["group"] / f"seed_{int(job['seed'])}" / "dry_run"
        runner = BaselineExperimentRunner(protocol_path, job["group"], int(job["seed"]), output_dir=output_dir)
        result = runner.dry_run()
        job["train_summary_path"] = str(output_dir / "baseline_experiment_summary.json")
        job["failure_stage"] = result.get("failure_stage")
        status = "completed" if result.get("ready") and result.get("failure_stage") is None else "failed"
        _mark_job(job, status, start=start, warnings=result.get("warnings") or [])
    except Exception as exc:  # noqa: BLE001
        _mark_job(job, "failed", start=start, failure_stage="dry_run_exception", warnings=[f"{type(exc).__name__}: {exc}", traceback.format_exc()])


def _run_single_formal_job(
    job: Dict[str, Any],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    output_root: Path,
    overrides: Mapping[str, Any],
    resume: bool,
) -> None:
    _mark_job(job, "running")
    start = time.time()
    try:
        group = job["group"]
        seed = int(job["seed"])
        pilot_dir = output_root / group / f"seed_{seed}" / "pilot_run"
        job_overrides = dict(overrides)
        if resume and group == "falcon_no_fsn":
            resume_path = _latest_controller_state(pilot_dir / "controller")
            if resume_path:
                job_overrides["resume_from_state"] = str(resume_path)
                job.setdefault("warnings", []).append(f"Resuming FALCON controller from {resume_path}.")
            else:
                job.setdefault("warnings", []).append("Resume requested but no FALCON controller state was found; starting from scratch.")
        runner = BaselineExperimentRunner(protocol_path, group, seed, output_dir=pilot_dir)
        result = runner.pilot_run(job_overrides)
        train_summary_path = pilot_dir / "baseline_experiment_summary.json"
        job["train_summary_path"] = str(train_summary_path)
        job["failure_stage"] = result.get("failure_stage")
        job["warnings"] = sorted(set(list(job.get("warnings") or []) + list(result.get("warnings") or [])))
        if result.get("failure_stage"):
            _mark_job(job, "failed", start=start, failure_stage=result.get("failure_stage"))
            return
        eval_path = _run_eval_for_job(protocol, output_root, group, seed)
        job["eval_summary_path"] = str(eval_path)
        eval_summary = _load_json(eval_path)
        if eval_summary.get("failure_stage"):
            _mark_job(job, "failed", start=start, failure_stage=eval_summary.get("failure_stage"), warnings=eval_summary.get("warnings") or [])
            return
        _mark_job(job, "completed", start=start, warnings=eval_summary.get("warnings") or [])
    except Exception as exc:  # noqa: BLE001
        _mark_job(job, "failed", start=start, failure_stage="formal_job_exception", warnings=[f"{type(exc).__name__}: {exc}", traceback.format_exc()])


def _run_eval_for_job(protocol: Mapping[str, Any], output_root: Path, group: str, seed: int) -> Path:
    evaluation = dict(protocol.get("evaluation") or {})
    checkpoint_role = str(evaluation.get("checkpoint_selection") or "best")
    checkpoint = resolve_group_checkpoint(output_root, group, int(seed), checkpoint_role)
    if checkpoint is None:
        output_path = output_root / group / f"seed_{int(seed)}" / "eval_set" / f"{checkpoint_role}_fixed_checkpoint_full_eval" / "eval_set_summary.json"
        summary = {
            "schema_version": "falcon.eval_set_summary.v1",
            "group": group,
            "checkpoint_role": checkpoint_role,
            "checkpoint_path": None,
            "num_scenarios_evaluated": 0,
            "failure_stage": "checkpoint_resolution",
            "warnings": ["Could not resolve checkpoint for formal eval."],
        }
        EvalSetEvaluator.save(summary, output_path)
        return output_path
    manifest_path = _resolve(evaluation.get("eval_scenarios") or protocol["evaluation_scenarios"])
    opponent_mode = str(evaluation.get("opponent_mode") or "fixed_checkpoint")
    opponent_checkpoint = _resolve(evaluation.get("opponent_checkpoint")) if evaluation.get("opponent_checkpoint") else None
    episodes = int(evaluation.get("episodes_per_scenario") or 1)
    evaluator = EvalSetEvaluator(manifest_path, {"base_config_path": str(_resolve(protocol["base_scenario_config"]))})
    summary = evaluator.evaluate_checkpoint(
        checkpoint,
        episodes_per_scenario=episodes,
        seed=int(seed),
        group=group,
        checkpoint_role=checkpoint_role,
        opponent_mode=opponent_mode,
        opponent_checkpoint=opponent_checkpoint,
    )
    summary["evaluation_protocol_version"] = evaluation.get("protocol_version")
    summary["agent_team"] = evaluation.get("agent_team", "A")
    summary["opponent_team"] = evaluation.get("opponent_team", "B")
    summary["same_actor_allowed"] = bool(evaluation.get("same_actor_allowed", False))
    output_path = output_root / group / f"seed_{int(seed)}" / "eval_set" / f"{checkpoint_role}_{opponent_mode}_full_eval" / "eval_set_summary.json"
    EvalSetEvaluator.save(summary, output_path)
    return output_path


def _new_status(protocol_path: Path, output_root: Path, jobs: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.formal_baseline_queue_status.v1",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "_started_time_seconds": time.time(),
        "mode": mode,
        "protocol_path": str(protocol_path),
        "output_root": str(output_root),
        "jobs": [dict(job) for job in jobs],
        "summary": _status_counts(jobs),
    }


def _save_status(status: Dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = _timestamp()
    status["summary"] = _status_counts(status.get("jobs") or [])
    serializable = _public_status(status)
    json_path = _status_json_path(reports_dir)
    csv_path = reports_dir / "formal_baseline_queue_status.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
    _write_status_csv(csv_path, status.get("jobs") or [])


def _public_status(status: Mapping[str, Any]) -> Dict[str, Any]:
    output = {key: value for key, value in dict(status).items() if not str(key).startswith("_")}
    output["jobs"] = [
        {key: value for key, value in dict(job).items() if not str(key).startswith("_")}
        for job in status.get("jobs") or []
    ]
    return output


def _write_status_csv(path: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            row = {key: job.get(key) for key in fields}
            row["warnings"] = " | ".join(str(item) for item in job.get("warnings") or [])
            writer.writerow(row)


def _mark_job(
    job: Dict[str, Any],
    status: str,
    start: Optional[float] = None,
    failure_stage: Optional[str] = None,
    warnings: Optional[Sequence[str]] = None,
) -> None:
    if status == "running":
        job["started_at"] = _timestamp()
        job["_started_time_seconds"] = time.time()
    else:
        job["finished_at"] = _timestamp()
        started = start if start is not None else job.get("_started_time_seconds")
        if started is not None:
            runtime = round(max(0.0, time.time() - float(started)), 3)
            job["runtime_seconds"] = runtime
            job["runtime_human_readable"] = _human_duration(runtime)
        job.pop("_started_time_seconds", None)
    job["status"] = status
    if failure_stage:
        job["failure_stage"] = failure_stage
    if warnings:
        job["warnings"] = sorted(set(list(job.get("warnings") or []) + [str(item) for item in warnings if item]))


def _status_counts(jobs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0, "skipped": 0}
    for job in jobs:
        status = str(job.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    counts["total"] = len(jobs)
    return counts


def _formal_overrides(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "max_rounds": protocol.get("max_rounds"),
        "train_steps_per_round": protocol.get("train_steps_per_round"),
        "eval_episodes_per_round": protocol.get("eval_episodes_per_round"),
        "policy_eval_episodes_per_candidate": protocol.get("policy_eval_episodes_per_candidate"),
        "qwen_candidates_per_round": protocol.get("qwen_candidates_per_round"),
        "random_candidates_per_round": protocol.get("random_candidates_per_round"),
    }


def _job_complete(output_root: Path, group: str, seed: int) -> bool:
    pilot = output_root / group / f"seed_{int(seed)}" / "pilot_run" / "pilot_run_summary.json"
    eval_summary = output_root / group / f"seed_{int(seed)}" / "eval_set" / "best_fixed_checkpoint_full_eval" / "eval_set_summary.json"
    if not pilot.exists() or not eval_summary.exists():
        return False
    try:
        pilot_data = _load_json(pilot)
        eval_data = _load_json(eval_summary)
        return bool(
            pilot_data.get("all_rounds_finished")
            and eval_data.get("failure_stage") is None
            and int(eval_data.get("num_scenarios_evaluated", 0)) >= 21
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _latest_controller_state(controller_dir: Path) -> Optional[Path]:
    if not controller_dir.exists():
        return None
    final = controller_dir / "falcon_controller_state_final.json"
    if final.exists():
        return final
    states = sorted(controller_dir.glob("controller_state_round*.json"), key=lambda path: path.stat().st_mtime)
    return states[-1] if states else None


def _mode(args: argparse.Namespace) -> str:
    if args.estimate_only:
        return "estimate_only"
    if args.dry_run:
        return "dry_run"
    return "run"


def _status_json_path(reports_dir: Path) -> Path:
    return reports_dir / "formal_baseline_queue_status.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _resolve(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT_DIR / path


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


if __name__ == "__main__":
    main()
