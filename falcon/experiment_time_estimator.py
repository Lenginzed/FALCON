"""Estimate wall-clock cost for queued FALCON baseline experiments."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .baseline_experiment import GROUP_DEFINITIONS, ROOT_DIR, SUPPORTED_GROUPS, load_yaml, write_json

DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
DEFAULT_PROTOCOL = DEFAULT_EXPERIMENT_DIR / "configs" / "experiment_protocol.yaml"

FALLBACK_EVAL_SECONDS_PER_SCENARIO_EPISODE = {
    "mappo_base": 1.7,
    "mappo_random_curriculum": 1.6,
    "mappo_qwen_only": 1.6,
    "falcon_no_fsn": 1.9,
}


class ExperimentTimeEstimator:
    """Use previous seed-0 pilot records to estimate formal queue runtime."""

    def __init__(
        self,
        protocol_path: str | Path = DEFAULT_PROTOCOL,
        results_root: Optional[str | Path] = None,
        reports_dir: Optional[str | Path] = None,
    ) -> None:
        self.protocol_path = Path(protocol_path)
        self.protocol = load_yaml(self.protocol_path)
        self.results_root = Path(results_root) if results_root else _resolve(self.protocol["output_root"])
        self.reports_dir = Path(reports_dir) if reports_dir else DEFAULT_EXPERIMENT_DIR / "reports"

    def estimate_job(
        self,
        group: str,
        seed: int,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if group not in SUPPORTED_GROUPS:
            raise ValueError(f"Unsupported group: {group}")
        config = self._target_config(overrides)
        history = self._load_group_history(group)
        train_estimate = self._estimate_training(group, config, history)
        eval_estimate = self._estimate_eval(group, config, history)
        total = _none_safe_sum(train_estimate.get("seconds"), eval_estimate.get("seconds"))
        warnings = list(train_estimate.get("warnings") or []) + list(eval_estimate.get("warnings") or [])
        confidence = _merge_confidence(train_estimate.get("confidence"), eval_estimate.get("confidence"))
        return {
            "schema_version": "falcon.formal_baseline_time_estimate_job.v1",
            "group": group,
            "seed": int(seed),
            **GROUP_DEFINITIONS[group],
            "target_config": config,
            "estimated_train_time_seconds": train_estimate.get("seconds"),
            "estimated_eval_time_seconds": eval_estimate.get("seconds"),
            "estimated_total_time_seconds": total,
            "estimated_total_time_human_readable": _human_duration(total),
            "confidence": confidence,
            "basis": {
                "training": train_estimate.get("basis"),
                "evaluation": eval_estimate.get("basis"),
            },
            "warnings": sorted(set(str(item) for item in warnings if item)),
        }

    def estimate_queue(
        self,
        seeds: Sequence[int],
        groups: Sequence[str] = SUPPORTED_GROUPS,
        overrides: Optional[Mapping[str, Any]] = None,
        skip_existing: bool = False,
    ) -> Dict[str, Any]:
        jobs = []
        for group in groups:
            for seed in seeds:
                job = self.estimate_job(group, int(seed), overrides=overrides)
                job["would_skip_existing"] = bool(skip_existing and _job_complete(self.results_root, group, int(seed)))
                jobs.append(job)
        active_jobs = [job for job in jobs if not job.get("would_skip_existing")]
        group_totals: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            group_jobs = [job for job in active_jobs if job.get("group") == group]
            seconds = _sum_known(job.get("estimated_total_time_seconds") for job in group_jobs)
            group_totals[group] = {
                "num_jobs": len(group_jobs),
                "estimated_total_time_seconds": seconds,
                "estimated_total_time_human_readable": _human_duration(seconds),
            }
        total_seconds = _sum_known(job.get("estimated_total_time_seconds") for job in active_jobs)
        slowest_group = max(group_totals.items(), key=lambda item: item[1].get("estimated_total_time_seconds") or 0.0)[0] if group_totals else None
        return {
            "schema_version": "falcon.formal_baseline_time_estimate.v1",
            "created_at": _timestamp(),
            "protocol_path": str(self.protocol_path),
            "results_root": str(self.results_root),
            "seeds": [int(seed) for seed in seeds],
            "groups": list(groups),
            "skip_existing": bool(skip_existing),
            "jobs": jobs,
            "group_totals": group_totals,
            "estimated_total_time_seconds": total_seconds,
            "estimated_total_time_human_readable": _human_duration(total_seconds),
            "slowest_group": slowest_group,
            "suggested_run_order": list(groups),
            "suggested_batches": _suggest_batches(group_totals),
            "recommend_split_batches": bool(total_seconds and total_seconds > 3 * 3600),
            "warnings": _estimate_warnings(jobs),
        }

    def export_estimate(
        self,
        estimate: Mapping[str, Any],
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir) if output_dir else self.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "formal_baseline_time_estimate.json"
        text_path = output_dir / "formal_baseline_time_estimate.txt"
        write_json(json_path, dict(estimate))
        text_path.write_text(_format_estimate_text(estimate), encoding="utf-8")
        return {"json_path": str(json_path), "text_path": str(text_path)}

    def _target_config(self, overrides: Optional[Mapping[str, Any]]) -> Dict[str, int]:
        values = {
            "max_rounds": self.protocol.get("max_rounds"),
            "train_steps_per_round": self.protocol.get("train_steps_per_round"),
            "eval_episodes_per_round": self.protocol.get("eval_episodes_per_round"),
            "policy_eval_episodes_per_candidate": self.protocol.get("policy_eval_episodes_per_candidate"),
            "qwen_candidates_per_round": self.protocol.get("qwen_candidates_per_round"),
            "random_candidates_per_round": self.protocol.get("random_candidates_per_round"),
            "eval_set_size": _eval_set_size(self.protocol),
            "eval_episodes_per_scenario": (self.protocol.get("evaluation") or {}).get("episodes_per_scenario"),
        }
        for key, value in dict(overrides or {}).items():
            if value is not None and key in values:
                values[key] = value
        return {key: int(value) for key, value in values.items() if value is not None}

    def _load_group_history(self, group: str) -> Dict[str, Any]:
        seed0_dir = self.results_root / group / "seed_0"
        files = {
            "baseline_summary": seed0_dir / "pilot_run" / "baseline_experiment_summary.json",
            "pilot_summary": seed0_dir / "pilot_run" / "pilot_run_summary.json",
            "eval_summary": seed0_dir / "eval_set" / "best_fixed_checkpoint_full_eval" / "eval_set_summary.json",
            "global_baseline_summary": DEFAULT_EXPERIMENT_DIR / "reports" / "baseline_summary.json",
        }
        history = {"files": {key: str(path) for key, path in files.items() if path.exists()}}
        for key, path in files.items():
            if path.exists():
                history[key] = _load_json(path)
        history["short_pilots"] = _load_short_pilot_summaries()
        return history

    def _estimate_training(self, group: str, config: Mapping[str, int], history: Mapping[str, Any]) -> Dict[str, Any]:
        summary = history.get("baseline_summary") if isinstance(history.get("baseline_summary"), MappingABC) else {}
        pilot = history.get("pilot_summary") if isinstance(history.get("pilot_summary"), MappingABC) else {}
        seconds = _duration_seconds(summary) or _duration_seconds(pilot)
        basis = []
        warnings = []
        confidence = "low"
        if seconds is not None:
            hist_params = _historical_params(summary, pilot)
            scale = _scale_factor(group, config, hist_params)
            estimate = round(seconds * scale, 3)
            basis.append(
                {
                    "source": (history.get("files") or {}).get("baseline_summary") or (history.get("files") or {}).get("pilot_summary"),
                    "historical_seconds": seconds,
                    "historical_params": hist_params,
                    "scale_factor": round(scale, 6),
                }
            )
            confidence = "high" if abs(scale - 1.0) < 0.05 and summary.get("runtime_seconds") is not None else "medium"
        else:
            estimate = _fallback_training_seconds(group, config)
            basis.append({"source": "fallback_group_multiplier", "group": group})
            warnings.append("Training runtime was not found in historical summaries; used a coarse group fallback.")
        return {"seconds": estimate, "confidence": confidence, "basis": basis, "warnings": warnings}

    def _estimate_eval(self, group: str, config: Mapping[str, int], history: Mapping[str, Any]) -> Dict[str, Any]:
        eval_summary = history.get("eval_summary") if isinstance(history.get("eval_summary"), MappingABC) else {}
        seconds = _duration_seconds(eval_summary)
        basis = []
        warnings = []
        if seconds is not None:
            hist_units = max(
                1,
                int(eval_summary.get("num_scenarios_evaluated", 0))
                * int(eval_summary.get("episodes_per_scenario", 1)),
            )
            target_units = max(1, int(config.get("eval_set_size", 0)) * int(config.get("eval_episodes_per_scenario", 1)))
            scale = target_units / hist_units
            return {
                "seconds": round(seconds * scale, 3),
                "confidence": "high" if abs(scale - 1.0) < 0.05 else "medium",
                "basis": [
                    {
                        "source": (history.get("files") or {}).get("eval_summary"),
                        "historical_seconds": seconds,
                        "historical_eval_units": hist_units,
                        "target_eval_units": target_units,
                        "scale_factor": round(scale, 6),
                    }
                ],
                "warnings": [],
            }
        per_unit = FALLBACK_EVAL_SECONDS_PER_SCENARIO_EPISODE.get(group, 1.8)
        target_units = max(1, int(config.get("eval_set_size", 0)) * int(config.get("eval_episodes_per_scenario", 1)))
        warnings.append("Eval summary did not contain runtime fields; used conservative per-scenario-episode fallback.")
        basis.append(
            {
                "source": "fallback_eval_seconds_per_scenario_episode",
                "seconds_per_scenario_episode": per_unit,
                "target_eval_units": target_units,
            }
        )
        return {
            "seconds": round(per_unit * target_units, 3),
            "confidence": "low",
            "basis": basis,
            "warnings": warnings,
        }


def _historical_params(summary: Mapping[str, Any], pilot: Mapping[str, Any]) -> Dict[str, int]:
    execution = summary.get("execution") if isinstance(summary.get("execution"), MappingABC) else {}
    values = (
        summary.get("protocol_parameters")
        if isinstance(summary.get("protocol_parameters"), MappingABC)
        else execution.get("protocol_parameters")
    )
    if not isinstance(values, MappingABC):
        values = pilot.get("protocol_parameters") if isinstance(pilot.get("protocol_parameters"), MappingABC) else {}
    output = {}
    for key, value in values.items():
        try:
            output[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    completed = execution.get("completed_rounds") or pilot.get("completed_rounds")
    if completed is not None:
        output.setdefault("max_rounds", int(completed))
    return output


def _scale_factor(group: str, target: Mapping[str, int], history: Mapping[str, int]) -> float:
    def ratio(key: str) -> float:
        historical = float(history.get(key) or target.get(key) or 1)
        return float(target.get(key) or historical) / max(historical, 1.0)

    round_factor = ratio("max_rounds")
    step_factor = ratio("train_steps_per_round")
    qwen_factor = ratio("qwen_candidates_per_round")
    random_factor = ratio("random_candidates_per_round")
    policy_eval_factor = ratio("policy_eval_episodes_per_candidate")
    if group == "mappo_base":
        return round_factor * step_factor
    if group == "mappo_random_curriculum":
        return round_factor * (0.8 * step_factor + 0.2 * random_factor)
    if group == "mappo_qwen_only":
        return round_factor * (0.45 * step_factor + 0.55 * qwen_factor)
    if group == "falcon_no_fsn":
        return round_factor * (0.35 * step_factor + 0.3 * qwen_factor + 0.35 * qwen_factor * policy_eval_factor)
    return round_factor * step_factor


def _fallback_training_seconds(group: str, config: Mapping[str, int]) -> float:
    rounds = max(1, int(config.get("max_rounds", 20)))
    steps = max(1, int(config.get("train_steps_per_round", 512)))
    base_per_round = {
        "mappo_base": 9.5,
        "mappo_random_curriculum": 9.5,
        "mappo_qwen_only": 38.0,
        "falcon_no_fsn": 92.0,
    }.get(group, 20.0)
    return round(base_per_round * rounds * (steps / 512.0), 3)


def _eval_set_size(protocol: Mapping[str, Any]) -> int:
    evaluation = protocol.get("evaluation") if isinstance(protocol.get("evaluation"), MappingABC) else {}
    path = _resolve(evaluation.get("eval_scenarios") or protocol.get("evaluation_scenarios"))
    if not path.exists():
        return 0
    try:
        data = _load_json(path)
        scenarios = data.get("scenarios") if isinstance(data.get("scenarios"), list) else []
        return int(data.get("scenario_count") or len(scenarios))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _load_short_pilot_summaries() -> List[Dict[str, Any]]:
    output = []
    for path in sorted((ROOT_DIR / "tests").glob("**/falcon_short_pilot_summary.json")):
        try:
            item = _load_json(path)
            item["_summary_path"] = str(path)
            output.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    return output


def _duration_seconds(data: Mapping[str, Any]) -> Optional[float]:
    for key in ("runtime_seconds", "total_runtime_seconds", "eval_set_runtime_seconds"):
        try:
            value = data.get(key)
            if value is not None:
                return round(float(value), 3)
        except (TypeError, ValueError):
            pass
    start = _parse_time(data.get("started_at"))
    finish = _parse_time(data.get("finished_at"))
    if start is not None and finish is not None:
        return round(max(0.0, finish - start), 3)
    return None


def _parse_time(value: Any) -> Optional[float]:
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _job_complete(results_root: Path, group: str, seed: int) -> bool:
    base = results_root / group / f"seed_{int(seed)}"
    pilot = base / "pilot_run" / "pilot_run_summary.json"
    eval_summary = base / "eval_set" / "best_fixed_checkpoint_full_eval" / "eval_set_summary.json"
    if not pilot.exists() or not eval_summary.exists():
        return False
    try:
        pilot_data = _load_json(pilot)
        eval_data = _load_json(eval_summary)
        return bool(
            pilot_data.get("all_rounds_finished")
            and eval_data.get("failure_stage") is None
            and int(eval_data.get("num_scenarios_evaluated", 0)) >= 1
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _suggest_batches(group_totals: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    batches = [
        {"batch_id": 1, "groups": ["mappo_base", "mappo_random_curriculum"], "reason": "Fast non-Qwen baselines."},
        {"batch_id": 2, "groups": ["mappo_qwen_only"], "reason": "Uses Ollama qwen3:8b but no dual-boundary policy filtering."},
        {"batch_id": 3, "groups": ["falcon_no_fsn"], "reason": "Slowest group; includes Qwen, policy eval, and dual-boundary filtering."},
    ]
    for batch in batches:
        seconds = _sum_known((group_totals.get(group) or {}).get("estimated_total_time_seconds") for group in batch["groups"])
        batch["estimated_total_time_seconds"] = seconds
        batch["estimated_total_time_human_readable"] = _human_duration(seconds)
    return batches


def _format_estimate_text(estimate: Mapping[str, Any]) -> str:
    lines = [
        "FALCON formal baseline time estimate",
        "",
        f"Seeds: {estimate.get('seeds')}",
        f"Groups: {estimate.get('groups')}",
        f"Total estimate: {estimate.get('estimated_total_time_human_readable')}",
        f"Slowest group: {estimate.get('slowest_group')}",
        f"Recommend split batches: {estimate.get('recommend_split_batches')}",
        "",
        "Per group:",
    ]
    for group, data in (estimate.get("group_totals") or {}).items():
        lines.append(f"- {group}: {data.get('estimated_total_time_human_readable')} for {data.get('num_jobs')} job(s)")
    lines.append("")
    lines.append("Per job:")
    for job in estimate.get("jobs") or []:
        lines.append(
            f"- {job.get('group')} seed={job.get('seed')}: total={job.get('estimated_total_time_human_readable')}, "
            f"train={_human_duration(job.get('estimated_train_time_seconds'))}, "
            f"eval={_human_duration(job.get('estimated_eval_time_seconds'))}, confidence={job.get('confidence')}"
        )
    warnings = estimate.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _estimate_warnings(jobs: Sequence[Mapping[str, Any]]) -> List[str]:
    warnings = []
    for job in jobs:
        for warning in job.get("warnings") or []:
            warnings.append(f"{job.get('group')} seed={job.get('seed')}: {warning}")
    return sorted(set(warnings))


def _none_safe_sum(*values: Any) -> Optional[float]:
    known = [float(value) for value in values if value is not None]
    return round(sum(known), 3) if known else None


def _sum_known(values: Sequence[Any]) -> Optional[float]:
    known = [float(value) for value in values if value is not None]
    return round(sum(known), 3) if known else None


def _merge_confidence(left: Any, right: Any) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return min((str(left or "low"), str(right or "low")), key=lambda value: order.get(value, 0))


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
    if seconds is None:
        return "unknown"
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
