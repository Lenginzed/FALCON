"""Estimate coverage-aware longer-budget seeds 3/4 from existing pilot timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
OLD_ROOT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "results_longer_budget"
DEFAULT_OUTPUT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "reports"
SMOKE_SUMMARY = (
    ROOT_DIR
    / "tests"
    / "tmp_falcon_coverage_aware_scheduler"
    / "coverage_aware_training_summary.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate coverage-aware longer-budget pilot runtime.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    smoke = _load_json(SMOKE_SUMMARY)
    smoke_runtime = float(smoke.get("runtime_seconds") or 0.0)
    smoke_scenarios = max(int(smoke.get("scenarios_actually_trained") or 0), 1)
    per_launch_overhead = smoke_runtime / smoke_scenarios
    jobs = []
    for seed in args.seeds:
        source = (
            OLD_ROOT
            / "falcon_no_fsn"
            / f"seed_{seed}"
            / "pilot_run"
            / "pilot_run_summary.json"
        )
        old = _load_json(source)
        old_runtime = float(old.get("runtime_seconds") or 0.0)
        rounds = max(int(old.get("completed_rounds") or 40), 1)
        round_summaries = old.get("round_summaries") or []
        old_training = sum(float(row.get("training_runtime_seconds") or 0.0) for row in round_summaries)
        non_training = max(old_runtime - old_training, 0.0)
        sequential_launches = max(rounds - 1, 0) * 8
        old_launches = max(rounds - 1, 0)
        extra_launches = max(sequential_launches - old_launches, 0)
        launch_overhead = extra_launches * per_launch_overhead
        estimated_training = old_training + launch_overhead
        validation_seconds = 6 * 6 * 1.9
        hard_eval_seconds = 40 * 3 * 1.9
        estimated_total = non_training + estimated_training + validation_seconds + hard_eval_seconds
        jobs.append(
            {
                "seed": seed,
                "old_scheduler_runtime_seconds": round(old_runtime, 3),
                "old_training_runtime_seconds": round(old_training, 3),
                "estimated_non_training_seconds": round(non_training, 3),
                "estimated_coverage_training_seconds": round(estimated_training, 3),
                "estimated_validation_seconds": round(validation_seconds, 3),
                "estimated_hard_eval_seconds": round(hard_eval_seconds, 3),
                "estimated_total_time_seconds": round(estimated_total, 3),
                "estimated_total_time_human_readable": _human(estimated_total),
                "basis": {
                    "old_pilot_summary": str(source),
                    "coverage_smoke_summary": str(SMOKE_SUMMARY),
                    "per_additional_training_launch_seconds": round(per_launch_overhead, 6),
                    "sequential_launches": sequential_launches,
                    "old_launches": old_launches,
                },
                "confidence": "medium",
                "warnings": [
                    "Qwen and policy-evaluation runtimes are assumed unchanged from the old-scheduler run.",
                    "Sequential environment startup overhead is extrapolated from the 6-scenario smoke.",
                ],
            }
        )
    total = sum(job["estimated_total_time_seconds"] for job in jobs)
    estimate = {
        "schema_version": "falcon.coverage_aware_longer_budget_time_estimate.v1",
        "seeds": args.seeds,
        "jobs": jobs,
        "mean_job_time_seconds": round(mean(job["estimated_total_time_seconds"] for job in jobs), 3),
        "estimated_total_time_seconds": round(total, 3),
        "estimated_total_time_human_readable": _human(total),
        "recommended_execution": "sequential",
        "confidence": "medium",
        "warnings": [
            "Estimate includes 40-round training, validation selection, and Hard Eval v2.",
            "Actual Ollama latency remains the largest uncertainty.",
        ],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "coverage_aware_longer_budget_time_estimate.json"
    txt_path = output_dir / "coverage_aware_longer_budget_time_estimate.txt"
    json_path.write_text(json.dumps(estimate, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "Coverage-Aware Longer-Budget Time Estimate",
        "==========================================",
        "",
    ]
    for job in jobs:
        lines.append(
            f"Seed {job['seed']}: {job['estimated_total_time_human_readable']} "
            f"(old scheduler { _human(job['old_scheduler_runtime_seconds']) })"
        )
    lines.extend(
        [
            "",
            f"Total: {estimate['estimated_total_time_human_readable']}",
            "Confidence: medium",
            "Recommended execution: sequential, seeds 3 then 4.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(estimate, indent=2))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _human(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


if __name__ == "__main__":
    main()
