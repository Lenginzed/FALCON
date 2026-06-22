#!/usr/bin/env python
"""Analyze Hard Held-out Eval Set v2 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import SUPPORTED_GROUPS, load_yaml  # noqa: E402

DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
DEFAULT_PROTOCOL = DEFAULT_EXPERIMENT_DIR / "configs" / "experiment_protocol.yaml"
DEFAULT_REPORT_DIR = DEFAULT_EXPERIMENT_DIR / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hard Eval v2 baseline results.")
    parser.add_argument("--groups", nargs="+", choices=SUPPORTED_GROUPS, default=list(SUPPORTED_GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    protocol = load_yaml(args.protocol)
    results_root = _resolve(protocol["output_root"])
    jobs = _load_jobs(results_root, args.groups, args.seeds)
    group_summary = _summarize_groups(jobs, args.groups)
    comparisons = _comparisons(jobs)
    saturation = _saturation_summary(group_summary)
    support = _support_summary(group_summary, comparisons)
    training_context = _load_training_context(DEFAULT_REPORT_DIR / "formal_baseline_multiseed_summary.json")
    report = {
        "schema_version": "falcon.hard_eval_v2_summary.v1",
        "hard_eval_v2_complete": all(row.get("failure_stage") is None for row in jobs) and len(jobs) == len(args.groups) * len(args.seeds),
        "groups": args.groups,
        "seeds": args.seeds,
        "jobs": jobs,
        "group_summary": group_summary,
        "comparisons": comparisons,
        "saturation": saturation,
        "support_summary": support,
        "training_context": training_context,
        "recommendation": _recommendation(saturation, support, group_summary),
    }
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hard_eval_v2_summary.json"
    csv_path = output_dir / "hard_eval_v2_summary.csv"
    txt_path = output_dir / "hard_eval_v2_report.txt"
    _write_json(json_path, report)
    _write_csv(csv_path, _csv_rows(group_summary, jobs))
    txt_path.write_text(_render_text(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "txt": str(txt_path), **report}, indent=2, sort_keys=True))


def _load_jobs(results_root: Path, groups: Sequence[str], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for group in groups:
        for seed in seeds:
            summary_path = results_root / group / f"seed_{int(seed)}" / "eval_set" / "hard_eval_v2" / "hard_eval_v2_summary.json"
            summary = _load_json(summary_path) if summary_path.exists() else {}
            aggregate = dict(summary.get("aggregate_result") or {})
            rows.append(
                {
                    "group": group,
                    "seed": int(seed),
                    "summary_path": str(summary_path),
                    "summary_exists": summary_path.exists(),
                    "checkpoint_source": summary.get("checkpoint_source"),
                    "checkpoint_source_detail": summary.get("checkpoint_source_detail"),
                    "checkpoint_path": summary.get("checkpoint_path"),
                    "hard_eval_win_rate": _safe_float(aggregate.get("final_win_rate")),
                    "hard_eval_mean_return": _safe_float(aggregate.get("final_mean_return")),
                    "hard_eval_num_scenarios": aggregate.get("num_scenarios"),
                    "num_scenarios_evaluated": summary.get("num_scenarios_evaluated"),
                    "failure_stage": summary.get("failure_stage") if summary else "missing_summary",
                    "same_actor": summary.get("same_actor"),
                    "same_checkpoint": summary.get("same_checkpoint"),
                    "opponent_mode": summary.get("opponent_mode"),
                    "opponent_checkpoint": summary.get("opponent_checkpoint"),
                    "eval_group_breakdown": summary.get("eval_group_breakdown") or {},
                    "warnings": list(summary.get("warnings") or []),
                }
            )
    return rows


def _summarize_groups(rows: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> Dict[str, Any]:
    summary = {}
    for group in groups:
        group_rows = [row for row in rows if row.get("group") == group]
        wins = [_safe_float(row.get("hard_eval_win_rate")) for row in group_rows]
        returns = [_safe_float(row.get("hard_eval_mean_return")) for row in group_rows]
        summary[group] = {
            "num_seeds": len(group_rows),
            "win_rate_per_seed": {str(row.get("seed")): row.get("hard_eval_win_rate") for row in group_rows},
            "win_rate_mean": _mean(wins),
            "win_rate_std": _std(wins),
            "mean_return_per_seed": {str(row.get("seed")): row.get("hard_eval_mean_return") for row in group_rows},
            "mean_return_mean": _mean(returns),
            "mean_return_std": _std(returns),
            "scenario_group_breakdown": _group_breakdown(group_rows),
            "num_failed": sum(1 for row in group_rows if row.get("failure_stage")),
        }
    return summary


def _group_breakdown(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"win_rate": [], "mean_return": []})
    for row in rows:
        for group, metrics in (row.get("eval_group_breakdown") or {}).items():
            grouped[str(group)]["win_rate"].append(_safe_float(metrics.get("win_rate"), 0.0))
            grouped[str(group)]["mean_return"].append(_safe_float(metrics.get("mean_return"), 0.0))
    return {
        group: {
            "win_rate_mean": _mean(values["win_rate"]),
            "win_rate_std": _std(values["win_rate"]),
            "mean_return_mean": _mean(values["mean_return"]),
            "mean_return_std": _std(values["mean_return"]),
        }
        for group, values in sorted(grouped.items())
    }


def _comparisons(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key = {(row.get("group"), row.get("seed")): row for row in rows}
    seeds = sorted({int(row.get("seed")) for row in rows if row.get("seed") is not None})
    pairs = [
        ("falcon_no_fsn", "mappo_base"),
        ("falcon_no_fsn", "mappo_random_curriculum"),
        ("falcon_no_fsn", "mappo_qwen_only"),
        ("mappo_qwen_only", "mappo_random_curriculum"),
        ("mappo_random_curriculum", "mappo_base"),
    ]
    result = {}
    for left, right in pairs:
        diffs = {}
        wins = 0
        for seed in seeds:
            left_row = by_key.get((left, seed))
            right_row = by_key.get((right, seed))
            if not left_row or not right_row:
                continue
            diff = _safe_float(left_row.get("hard_eval_win_rate"), 0.0) - _safe_float(
                right_row.get("hard_eval_win_rate"), 0.0
            )
            diffs[str(seed)] = round(diff, 6)
            if diff > 0:
                wins += 1
        result[f"{left}_vs_{right}"] = {
            "win_rate_diff_per_seed": diffs,
            "mean_win_rate_diff": _mean(diffs.values()),
            "left_better_seed_count": wins,
            "num_seed_comparisons": len(diffs),
            "left_better_at_least_2_of_3": wins >= 2,
        }
    return result


def _saturation_summary(group_summary: Mapping[str, Any]) -> Dict[str, Any]:
    win_rates = {
        group: _safe_float(summary.get("win_rate_mean"), 0.0)
        for group, summary in group_summary.items()
    }
    return {
        "all_groups_win_rate_above_0_90": all(value > 0.9 for value in win_rates.values()),
        "all_groups_win_rate_above_0_95": all(value > 0.95 for value in win_rates.values()),
        "win_rate_means": win_rates,
        "range_between_group_means": round(max(win_rates.values()) - min(win_rates.values()), 6) if win_rates else None,
    }


def _support_summary(group_summary: Mapping[str, Any], comparisons: Mapping[str, Any]) -> Dict[str, Any]:
    falcon = group_summary.get("falcon_no_fsn") or {}
    qwen = group_summary.get("mappo_qwen_only") or {}
    random_group = group_summary.get("mappo_random_curriculum") or {}
    base = group_summary.get("mappo_base") or {}
    return {
        "falcon_mean_win_rate_above_all_baselines": _safe_float(falcon.get("win_rate_mean"), 0.0)
        > max(
            _safe_float(qwen.get("win_rate_mean"), 0.0),
            _safe_float(random_group.get("win_rate_mean"), 0.0),
            _safe_float(base.get("win_rate_mean"), 0.0),
        ),
        "falcon_better_than_qwen_2_of_3": (comparisons.get("falcon_no_fsn_vs_mappo_qwen_only") or {}).get(
            "left_better_at_least_2_of_3"
        ),
        "falcon_better_than_random_2_of_3": (comparisons.get("falcon_no_fsn_vs_mappo_random_curriculum") or {}).get(
            "left_better_at_least_2_of_3"
        ),
        "falcon_better_than_base_2_of_3": (comparisons.get("falcon_no_fsn_vs_mappo_base") or {}).get(
            "left_better_at_least_2_of_3"
        ),
        "failure_replay_group_win_rate": {
            "falcon_no_fsn": _group_win(falcon, "replay_failure_variants"),
            "mappo_qwen_only": _group_win(qwen, "replay_failure_variants"),
            "mappo_random_curriculum": _group_win(random_group, "replay_failure_variants"),
            "mappo_base": _group_win(base, "replay_failure_variants"),
        },
        "coordination_stress_group_win_rate": {
            "falcon_no_fsn": _group_win(falcon, "hard_coordination_stress"),
            "mappo_qwen_only": _group_win(qwen, "hard_coordination_stress"),
        },
        "target_assignment_stress_group_win_rate": {
            "falcon_no_fsn": _group_win(falcon, "hard_target_assignment_stress"),
            "mappo_qwen_only": _group_win(qwen, "hard_target_assignment_stress"),
        },
    }


def _group_win(summary: Mapping[str, Any], scenario_group: str) -> Optional[float]:
    return ((summary.get("scenario_group_breakdown") or {}).get(scenario_group) or {}).get("win_rate_mean")


def _load_training_context(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False}
    data = _load_json(path)
    groups = data.get("groups") or data.get("group_summary") or {}
    context = {"available": True, "groups": {}}
    for group, summary in groups.items():
        if isinstance(summary, dict):
            context["groups"][group] = {
                "accepted_rate_mean": summary.get("accepted_rate_mean"),
                "fallback_rate_mean": summary.get("fallback_rate_mean"),
                "qwen_calls": summary.get("qwen_calls"),
                "qwen_calls_per_accepted": summary.get("qwen_calls_per_accepted"),
            }
    return context


def _recommendation(saturation: Mapping[str, Any], support: Mapping[str, Any], group_summary: Mapping[str, Any]) -> Dict[str, Any]:
    notes = []
    if saturation.get("all_groups_win_rate_above_0_90"):
        notes.append("Hard Eval v2 is still saturated above 0.90 win_rate for all groups; opponent/scenario difficulty may still be insufficient.")
    if support.get("falcon_better_than_qwen_2_of_3"):
        notes.append("FALCON beats qwen-only on at least 2/3 seeds under Hard Eval v2.")
    else:
        notes.append("FALCON does not beat qwen-only on at least 2/3 seeds under Hard Eval v2.")
    falcon = group_summary.get("falcon_no_fsn") or {}
    qwen = group_summary.get("mappo_qwen_only") or {}
    falcon_return = _safe_float(falcon.get("mean_return_mean"), 0.0)
    qwen_return = _safe_float(qwen.get("mean_return_mean"), 0.0)
    if falcon_return > qwen_return:
        notes.append("FALCON mean_return is above qwen-only; use this as a secondary signal if win_rate is saturated.")
    return {
        "hard_eval_v2_resolved_saturation": not saturation.get("all_groups_win_rate_above_0_90"),
        "longer_budget_recommended": bool(saturation.get("all_groups_win_rate_above_0_90")),
        "add_seed_3_4_recommended": True,
        "enter_fsn_stage_recommended": bool(
            not saturation.get("all_groups_win_rate_above_0_90")
            and support.get("falcon_mean_win_rate_above_all_baselines")
            and support.get("falcon_better_than_qwen_2_of_3")
        ),
        "notes": notes,
    }


def _csv_rows(group_summary: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for group, summary in group_summary.items():
        rows.append(
            {
                "row_type": "group_summary",
                "group": group,
                "seed": "",
                "win_rate": summary.get("win_rate_mean"),
                "win_rate_std": summary.get("win_rate_std"),
                "mean_return": summary.get("mean_return_mean"),
                "mean_return_std": summary.get("mean_return_std"),
                "scenario_group_breakdown": json.dumps(summary.get("scenario_group_breakdown"), sort_keys=True),
            }
        )
    for row in jobs:
        rows.append(
            {
                "row_type": "seed_result",
                "group": row.get("group"),
                "seed": row.get("seed"),
                "win_rate": row.get("hard_eval_win_rate"),
                "win_rate_std": "",
                "mean_return": row.get("hard_eval_mean_return"),
                "mean_return_std": "",
                "scenario_group_breakdown": json.dumps(row.get("eval_group_breakdown"), sort_keys=True),
            }
        )
    return rows


def _render_text(report: Mapping[str, Any]) -> str:
    lines = [
        "Hard Held-out Eval v2 Report",
        "",
        f"complete: {report.get('hard_eval_v2_complete')}",
        f"saturation_all_groups_gt_0.90: {(report.get('saturation') or {}).get('all_groups_win_rate_above_0_90')}",
        f"group_mean_range: {(report.get('saturation') or {}).get('range_between_group_means')}",
        "",
        "Group results:",
    ]
    for group, summary in (report.get("group_summary") or {}).items():
        lines.append(
            f"- {group}: win_rate {summary.get('win_rate_mean')} +/- {summary.get('win_rate_std')}, "
            f"mean_return {summary.get('mean_return_mean')} +/- {summary.get('mean_return_std')}"
        )
    lines.extend(["", "Key comparisons:"])
    for name, comp in (report.get("comparisons") or {}).items():
        lines.append(f"- {name}: mean_diff={comp.get('mean_win_rate_diff')}, per_seed={comp.get('win_rate_diff_per_seed')}")
    lines.extend(["", "Scenario-group signals:"])
    lines.append(json.dumps(report.get("support_summary"), indent=2, sort_keys=True))
    lines.extend(["", "Recommendation:"])
    for note in (report.get("recommendation") or {}).get("notes") or []:
        lines.append(f"- {note}")
    lines.append(f"enter_fsn_stage_recommended: {(report.get('recommendation') or {}).get('enter_fsn_stage_recommended')}")
    return "\n".join(lines) + "\n"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return None
    if len(clean) < 2:
        return 0.0
    return round(statistics.stdev(clean), 6)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
