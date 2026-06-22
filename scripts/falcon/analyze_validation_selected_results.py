#!/usr/bin/env python
"""Analyze validation-selected held-out test results for baseline groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import SUPPORTED_GROUPS, load_yaml  # noqa: E402

DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
DEFAULT_PROTOCOL = DEFAULT_EXPERIMENT_DIR / "configs" / "experiment_protocol.yaml"
DEFAULT_SPLIT = DEFAULT_EXPERIMENT_DIR / "manifests" / "eval_split_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation-selected test results.")
    parser.add_argument("--groups", nargs="+", choices=SUPPORTED_GROUPS, default=list(SUPPORTED_GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPERIMENT_DIR / "reports"))
    args = parser.parse_args()

    protocol = load_yaml(args.protocol)
    results_root = _resolve(protocol["output_root"])
    split = _load_json(_resolve(args.split)) if _resolve(args.split).exists() else {}
    job_rows = _load_jobs(results_root, args.groups, args.seeds)
    group_summary = _summarize_groups(job_rows, args.groups)
    comparisons = _compare_groups(job_rows)
    qwen_variance = _qwen_variance_summary(job_rows)
    falcon_vs_qwen = _falcon_vs_qwen_seed_wins(job_rows)
    complete = all(
        row.get("heldout_test_failure_stage") is None
        and row.get("same_actor") is False
        and row.get("opponent_mode") == "fixed_checkpoint"
        for row in job_rows
    ) and len(job_rows) == len(args.groups) * len(args.seeds)
    recommendation = _make_recommendation(group_summary, falcon_vs_qwen, qwen_variance, complete)
    report = {
        "schema_version": "falcon.validation_selected_multiseed_summary.v1",
        "formal_validation_selected_complete": complete,
        "split_path": str(_resolve(args.split)),
        "validation_scenario_count": (split.get("validation") or {}).get("scenario_count"),
        "test_scenario_count": (split.get("test") or {}).get("scenario_count"),
        "groups": args.groups,
        "seeds": args.seeds,
        "jobs": job_rows,
        "group_summary": group_summary,
        "comparisons": comparisons,
        "qwen_only_variance": qwen_variance,
        "falcon_vs_qwen": falcon_vs_qwen,
        "recommendation": recommendation,
    }
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_selected_multiseed_summary.json"
    csv_path = output_dir / "validation_selected_multiseed_summary.csv"
    txt_path = output_dir / "validation_selected_report.txt"
    _write_json(json_path, report)
    _write_csv(csv_path, _summary_csv_rows(group_summary, job_rows))
    txt_path.write_text(_render_text_report(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "txt": str(txt_path), **report}, indent=2, sort_keys=True))


def _load_jobs(results_root: Path, groups: Sequence[str], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for group in groups:
        for seed in seeds:
            seed_dir = results_root / group / f"seed_{int(seed)}"
            selection_path = seed_dir / "eval_set" / "validation_checkpoint_selection" / "validation_selected_checkpoint.json"
            test_path = seed_dir / "eval_set" / "validation_selected_test_eval" / "heldout_test_summary.json"
            selection = _load_json(selection_path) if selection_path.exists() else {}
            test_summary = _load_json(test_path) if test_path.exists() else {}
            aggregate = dict(test_summary.get("aggregate_result") or {})
            row = {
                "group": group,
                "seed": int(seed),
                "selection_summary_path": str(selection_path),
                "heldout_test_summary_path": str(test_path),
                "selection_exists": selection_path.exists(),
                "heldout_test_exists": test_path.exists(),
                "selected_checkpoint": selection.get("selected_checkpoint"),
                "selected_round_id": selection.get("selected_round_id"),
                "selected_candidate_label": selection.get("selected_candidate_label"),
                "validation_win_rate": _safe_float(selection.get("validation_win_rate")),
                "validation_mean_return": _safe_float(selection.get("validation_mean_return")),
                "heldout_test_win_rate": _safe_float(aggregate.get("final_win_rate")),
                "heldout_test_mean_return": _safe_float(aggregate.get("final_mean_return")),
                "heldout_test_num_scenarios": aggregate.get("num_scenarios"),
                "num_scenarios_evaluated": test_summary.get("num_scenarios_evaluated"),
                "heldout_test_failure_stage": test_summary.get("failure_stage") if test_summary else "missing_test_summary",
                "same_actor": test_summary.get("same_actor"),
                "same_checkpoint": test_summary.get("same_checkpoint"),
                "opponent_mode": test_summary.get("opponent_mode"),
                "opponent_checkpoint": test_summary.get("opponent_checkpoint"),
                "eval_group_breakdown": test_summary.get("eval_group_breakdown") or {},
                "warnings": list(selection.get("warnings") or []) + list(test_summary.get("warnings") or []),
            }
            rows.append(row)
    return rows


def _summarize_groups(rows: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for group in groups:
        group_rows = [row for row in rows if row.get("group") == group]
        win_values = [_safe_float(row.get("heldout_test_win_rate")) for row in group_rows]
        return_values = [_safe_float(row.get("heldout_test_mean_return")) for row in group_rows]
        selected_rounds = [_safe_int(row.get("selected_round_id")) for row in group_rows]
        summary[group] = {
            "num_seeds": len(group_rows),
            "heldout_test_win_rate_per_seed": {
                str(row.get("seed")): row.get("heldout_test_win_rate") for row in group_rows
            },
            "heldout_test_win_rate_mean": _mean(win_values),
            "heldout_test_win_rate_std": _std(win_values),
            "heldout_test_mean_return_per_seed": {
                str(row.get("seed")): row.get("heldout_test_mean_return") for row in group_rows
            },
            "heldout_test_mean_return_mean": _mean(return_values),
            "heldout_test_mean_return_std": _std(return_values),
            "selected_round_per_seed": {
                str(row.get("seed")): row.get("selected_round_id") for row in group_rows
            },
            "selected_round_counts": dict(Counter(str(item) for item in selected_rounds if item is not None)),
            "scenario_group_breakdown": _aggregate_breakdowns(group_rows),
            "num_failed": sum(1 for row in group_rows if row.get("heldout_test_failure_stage")),
        }
    return summary


def _aggregate_breakdowns(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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


def _compare_groups(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_group_seed = {(row.get("group"), row.get("seed")): row for row in rows}
    pairs = [
        ("falcon_no_fsn", "mappo_base"),
        ("falcon_no_fsn", "mappo_random_curriculum"),
        ("falcon_no_fsn", "mappo_qwen_only"),
        ("mappo_qwen_only", "mappo_random_curriculum"),
        ("mappo_random_curriculum", "mappo_base"),
    ]
    result = {}
    seeds = sorted({int(row.get("seed")) for row in rows if row.get("seed") is not None})
    for left, right in pairs:
        diffs = []
        wins = 0
        for seed in seeds:
            left_row = by_group_seed.get((left, seed))
            right_row = by_group_seed.get((right, seed))
            if not left_row or not right_row:
                continue
            diff = _safe_float(left_row.get("heldout_test_win_rate"), 0.0) - _safe_float(
                right_row.get("heldout_test_win_rate"), 0.0
            )
            diffs.append(diff)
            if diff > 0:
                wins += 1
        result[f"{left}_vs_{right}"] = {
            "win_rate_diff_per_seed": {str(seed): round(diffs[idx], 6) for idx, seed in enumerate(seeds[: len(diffs)])},
            "mean_win_rate_diff": _mean(diffs),
            "left_better_seed_count": wins,
            "num_seed_comparisons": len(diffs),
            "left_better_at_least_2_of_3": wins >= 2,
        }
    return result


def _qwen_variance_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    qwen_rows = [row for row in rows if row.get("group") == "mappo_qwen_only"]
    wins = [_safe_float(row.get("heldout_test_win_rate")) for row in qwen_rows]
    selected_rounds = [_safe_int(row.get("selected_round_id")) for row in qwen_rows]
    return {
        "win_rate_per_seed": {str(row.get("seed")): row.get("heldout_test_win_rate") for row in qwen_rows},
        "win_rate_std": _std(wins),
        "win_rate_range": round((max(wins) - min(wins)), 6) if wins else None,
        "selected_rounds": {str(row.get("seed")): row.get("selected_round_id") for row in qwen_rows},
        "early_checkpoint_selected_count": sum(1 for item in selected_rounds if item is not None and item <= 1),
        "high_variance": bool(wins and (max(wins) - min(wins) >= 0.3 or _std(wins) >= 0.2)),
    }


def _falcon_vs_qwen_seed_wins(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_group_seed = {(row.get("group"), row.get("seed")): row for row in rows}
    seeds = sorted({int(row.get("seed")) for row in rows if row.get("seed") is not None})
    per_seed = {}
    wins = 0
    for seed in seeds:
        falcon = by_group_seed.get(("falcon_no_fsn", seed))
        qwen = by_group_seed.get(("mappo_qwen_only", seed))
        if not falcon or not qwen:
            continue
        diff = _safe_float(falcon.get("heldout_test_win_rate"), 0.0) - _safe_float(
            qwen.get("heldout_test_win_rate"), 0.0
        )
        per_seed[str(seed)] = round(diff, 6)
        if diff > 0:
            wins += 1
    return {
        "falcon_minus_qwen_win_rate_per_seed": per_seed,
        "falcon_better_seed_count": wins,
        "num_seed_comparisons": len(per_seed),
        "falcon_better_at_least_2_of_3": wins >= 2,
    }


def _make_recommendation(
    group_summary: Mapping[str, Any],
    falcon_vs_qwen: Mapping[str, Any],
    qwen_variance: Mapping[str, Any],
    complete: bool,
) -> Dict[str, Any]:
    falcon_win = _safe_float((group_summary.get("falcon_no_fsn") or {}).get("heldout_test_win_rate_mean"), 0.0)
    qwen_win = _safe_float((group_summary.get("mappo_qwen_only") or {}).get("heldout_test_win_rate_mean"), 0.0)
    base_win = _safe_float((group_summary.get("mappo_base") or {}).get("heldout_test_win_rate_mean"), 0.0)
    random_win = _safe_float((group_summary.get("mappo_random_curriculum") or {}).get("heldout_test_win_rate_mean"), 0.0)
    notes = []
    if not complete:
        notes.append("Validation-selected protocol has incomplete jobs; fix missing eval before drawing conclusions.")
    if qwen_variance.get("high_variance"):
        notes.append("Qwen-only remains high variance under validation-selected evaluation.")
    if not falcon_vs_qwen.get("falcon_better_at_least_2_of_3"):
        notes.append("FALCON does not beat qwen-only on at least 2/3 seeds under the current split.")
    if falcon_win <= max(qwen_win, base_win, random_win):
        notes.append("FALCON mean held-out win rate is not strictly above all baselines.")
    return {
        "enter_fsn_stage_recommended": bool(
            complete
            and falcon_win > max(qwen_win, base_win, random_win)
            and falcon_vs_qwen.get("falcon_better_at_least_2_of_3")
        ),
        "need_seed_3_4": bool(qwen_variance.get("high_variance") or not falcon_vs_qwen.get("falcon_better_at_least_2_of_3")),
        "need_more_eval_episodes": False,
        "notes": notes,
    }


def _summary_csv_rows(group_summary: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for group, summary in group_summary.items():
        rows.append(
            {
                "row_type": "group_summary",
                "group": group,
                "seed": "",
                "heldout_test_win_rate": summary.get("heldout_test_win_rate_mean"),
                "heldout_test_win_rate_std": summary.get("heldout_test_win_rate_std"),
                "heldout_test_mean_return": summary.get("heldout_test_mean_return_mean"),
                "heldout_test_mean_return_std": summary.get("heldout_test_mean_return_std"),
                "selected_rounds": json.dumps(summary.get("selected_round_per_seed"), sort_keys=True),
            }
        )
    for row in jobs:
        rows.append(
            {
                "row_type": "seed_result",
                "group": row.get("group"),
                "seed": row.get("seed"),
                "heldout_test_win_rate": row.get("heldout_test_win_rate"),
                "heldout_test_win_rate_std": "",
                "heldout_test_mean_return": row.get("heldout_test_mean_return"),
                "heldout_test_mean_return_std": "",
                "selected_rounds": row.get("selected_round_id"),
            }
        )
    return rows


def _render_text_report(report: Mapping[str, Any]) -> str:
    lines = [
        "FALCON Validation-Selected Checkpoint Report",
        "",
        f"complete: {report.get('formal_validation_selected_complete')}",
        f"validation scenarios: {report.get('validation_scenario_count')}",
        f"held-out test scenarios: {report.get('test_scenario_count')}",
        "",
        "Group held-out test results:",
    ]
    for group, summary in (report.get("group_summary") or {}).items():
        lines.append(
            f"- {group}: win_rate {summary.get('heldout_test_win_rate_mean')} +/- "
            f"{summary.get('heldout_test_win_rate_std')}, mean_return "
            f"{summary.get('heldout_test_mean_return_mean')} +/- {summary.get('heldout_test_mean_return_std')}, "
            f"selected_rounds={summary.get('selected_round_per_seed')}"
        )
    lines.extend(["", "FALCON vs qwen-only:"])
    lines.append(json.dumps(report.get("falcon_vs_qwen"), indent=2, sort_keys=True))
    lines.extend(["", "Recommendation:"])
    for note in (report.get("recommendation") or {}).get("notes") or []:
        lines.append(f"- {note}")
    lines.append(f"enter_fsn_stage_recommended: {(report.get('recommendation') or {}).get('enter_fsn_stage_recommended')}")
    lines.append(f"need_seed_3_4: {(report.get('recommendation') or {}).get('need_seed_3_4')}")
    return "\n".join(lines) + "\n"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None]
    if len(clean) < 2:
        return 0.0 if clean else None
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


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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
