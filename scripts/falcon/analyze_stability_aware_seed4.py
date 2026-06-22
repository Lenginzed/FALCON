#!/usr/bin/env python
"""Analyze the stability-aware seed-4 mini-pilot and compare prior schedulers."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402

RESULTS_ROOT = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "results_stability_aware_seed4"
    / "falcon_no_fsn"
    / "seed_4"
)
REPORT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "reports"
OLD_COVERAGE_SUMMARY = REPORT_DIR / "coverage_aware_longer_budget_summary.json"
OLD_LONGER_SUMMARY = REPORT_DIR / "longer_budget_hard_eval_summary.json"
FIXED_OPPONENT = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "opponents"
    / "fixed_baseline_opponent"
    / "candidate_seed999_steps2048"
    / "checkpoints"
    / "actor_seed999_steps2048.pt"
)
BASE_CONFIG = (
    ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    controller_dir = RESULTS_ROOT / "pilot_run" / "controller"
    pool = _load_json(controller_dir / "falcon_curriculum_pool_final.json")
    pilot = _load_json(RESULTS_ROOT / "pilot_run" / "pilot_run_summary.json")
    registry = _load_json(controller_dir / "falcon_checkpoint_registry.json")
    selection = _load_json(
        RESULTS_ROOT
        / "eval_set"
        / "validation_checkpoint_selection"
        / "validation_selected_checkpoint.json"
    )
    hard_eval = _load_json(
        RESULTS_ROOT / "eval_set" / "hard_eval_v2" / "hard_eval_v2_summary.json"
    )
    old_coverage = _load_json(OLD_COVERAGE_SUMMARY)
    old_longer = _load_json(OLD_LONGER_SUMMARY)
    accepted = [
        item
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    ]
    diagnostics_dir = RESULTS_ROOT / "diagnostics" / "accepted_scene_transfer"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = diagnostics_dir / "accepted_scenes_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "falcon.eval_scenario_manifest.v1",
            "scenario_count": len(accepted),
            "scenarios": [
                {
                    "scenario_id": item.get("pool_item_id"),
                    "scenario_group": "accepted_curriculum",
                    "scenario_yaml_path": item.get("scenario_yaml_path"),
                }
                for item in accepted
            ],
        },
    )
    evaluator = EvalSetEvaluator(
        manifest_path, {"base_config_path": str(BASE_CONFIG)}
    )
    checkpoint_map = _checkpoint_map(registry)
    round0_checkpoint = checkpoint_map.get(0)
    selected_checkpoint = selection.get("selected_checkpoint")
    round0_eval = _evaluate_or_load(
        evaluator,
        round0_checkpoint,
        diagnostics_dir / "round0_accepted_eval.json",
        "round0",
        seed=940000,
    )
    selected_eval = _evaluate_or_load(
        evaluator,
        selected_checkpoint,
        diagnostics_dir / "selected_accepted_eval.json",
        "selected",
        seed=950000,
    )
    acceptance_rows: Dict[str, Mapping[str, Any]] = {}
    by_round: Dict[int, list[Mapping[str, Any]]] = {}
    for item in accepted:
        by_round.setdefault(int(item.get("source_round") or 0), []).append(item)
    for source_round, items in sorted(by_round.items()):
        acceptance_checkpoint = checkpoint_map.get(max(source_round - 1, 0))
        if not acceptance_checkpoint:
            continue
        sub_manifest = diagnostics_dir / f"acceptance_round_{source_round}.json"
        _write_json(
            sub_manifest,
            {
                "schema_version": "falcon.eval_scenario_manifest.v1",
                "scenario_count": len(items),
                "scenarios": [
                    {
                        "scenario_id": item.get("pool_item_id"),
                        "scenario_group": "accepted_curriculum",
                        "scenario_yaml_path": item.get("scenario_yaml_path"),
                    }
                    for item in items
                ],
            },
        )
        sub_evaluator = EvalSetEvaluator(
            sub_manifest, {"base_config_path": str(BASE_CONFIG)}
        )
        result = _evaluate_or_load(
            sub_evaluator,
            acceptance_checkpoint,
            diagnostics_dir / f"acceptance_round_{source_round}_eval.json",
            f"pre_acceptance_{source_round}",
            seed=960000 + source_round,
        )
        acceptance_rows.update(_scenario_map(result))

    round0_rows = _scenario_map(round0_eval)
    selected_rows = _scenario_map(selected_eval)
    progress_rows = []
    for item in accepted:
        key = str(item.get("pool_item_id"))
        acceptance_win = _win(acceptance_rows.get(key))
        round0_win = _win(round0_rows.get(key))
        selected_win = _win(selected_rows.get(key))
        progress_rows.append(
            {
                "pool_item_id": key,
                "scenario_id": item.get("scenario_id"),
                "source_round": item.get("source_round"),
                "train_count": item.get("train_count"),
                "acceptance_win_rate": acceptance_win,
                "round0_win_rate": round0_win,
                "selected_win_rate": selected_win,
                "hard_to_easy": bool(
                    acceptance_win is not None
                    and acceptance_win <= 0.5
                    and selected_win is not None
                    and selected_win >= 0.8
                ),
                "forgotten": bool(
                    round0_win is not None
                    and selected_win is not None
                    and selected_win <= round0_win - 0.2
                ),
                "scenario_yaml_path": item.get("scenario_yaml_path"),
            }
        )
    _write_csv(
        diagnostics_dir / "accepted_scene_policy_progress.csv", progress_rows
    )

    training_rounds = [
        row
        for row in pilot.get("round_summaries") or []
        if int(row.get("round_id") or 0) > 0
    ]
    accepted_count = len(accepted)
    trained_count = sum(int(item.get("train_count") or 0) > 0 for item in accepted)
    events = sum(int(item.get("train_count") or 0) for item in accepted)
    current = {
        "max_rounds": pilot.get("max_rounds"),
        "completed_rounds": pilot.get("completed_rounds"),
        "accepted_total": accepted_count,
        "accepted_trained_count": trained_count,
        "accepted_coverage": _ratio(trained_count, accepted_count),
        "duplicate_rate": _ratio(max(events - trained_count, 0), events),
        "mean_anchor_ratio": _mean(
            row.get("anchor_ratio") for row in training_rounds
        ),
        "mean_accepted_ratio": _mean(
            row.get("accepted_ratio") for row in training_rounds
        ),
        "rounds_preserved_input_checkpoint": sum(
            int(row.get("selected_batch_index") or 0) == -1
            for row in training_rounds
        ),
        "rounds_selected_nonterminal": sum(
            row.get("selected_checkpoint_path")
            and row.get("terminal_checkpoint_path")
            and Path(str(row["selected_checkpoint_path"])).resolve()
            != Path(str(row["terminal_checkpoint_path"])).resolve()
            for row in training_rounds
        ),
        "selected_checkpoint_round": selection.get("selected_round_id"),
        "hard_eval_v2_win_rate": (hard_eval.get("aggregate_result") or {}).get(
            "final_win_rate"
        ),
        "hard_eval_v2_mean_return": (hard_eval.get("aggregate_result") or {}).get(
            "final_mean_return"
        ),
        "hard_eval_v2_group_breakdown": hard_eval.get("eval_group_breakdown")
        or {},
        "accepted_scene_round0_win_rate": (
            round0_eval.get("aggregate_result") or {}
        ).get("final_win_rate"),
        "accepted_scene_selected_win_rate": (
            selected_eval.get("aggregate_result") or {}
        ).get("final_win_rate"),
        "hard_to_easy_count": sum(row["hard_to_easy"] for row in progress_rows),
        "forgotten_count": sum(row["forgotten"] for row in progress_rows),
        "forgetting_rate": _ratio(
            sum(row["forgotten"] for row in progress_rows), len(progress_rows)
        ),
    }
    old = dict((old_coverage.get("seed_results") or {}).get("4") or {})
    old_scheduler = dict(
        ((old_longer.get("performance") or {}).get("falcon_no_fsn") or {}).get(
            "4"
        )
        or {}
    )
    comparison = {
        "coverage_aware_40_round": {
            "accepted_total": old.get("accepted_total"),
            "accepted_coverage": old.get("accepted_coverage"),
            "duplicate_rate": old.get("duplicate_rate"),
            "selected_checkpoint_round": old.get("selected_checkpoint_round"),
            "hard_eval_v2_win_rate": old.get("hard_eval_v2_win_rate"),
            "hard_eval_v2_mean_return": old.get("hard_eval_v2_mean_return"),
            "forgotten_count": old.get("forgotten_accepted_scenes"),
            "forgetting_rate": _ratio(
                old.get("forgotten_accepted_scenes"), old.get("accepted_total")
            ),
        },
        "old_scheduler_40_round": {
            "hard_eval_v2_win_rate": old_scheduler.get("win_rate"),
            "hard_eval_v2_mean_return": old_scheduler.get("mean_return"),
        },
        "stability_aware_10_round": current,
    }
    comparison["deltas_vs_coverage_aware"] = {
        "accepted_coverage": _delta(
            current.get("accepted_coverage"), old.get("accepted_coverage")
        ),
        "forgetting_rate": _delta(
            current.get("forgetting_rate"),
            comparison["coverage_aware_40_round"].get("forgetting_rate"),
        ),
        "hard_eval_v2_win_rate": _delta(
            current.get("hard_eval_v2_win_rate"), old.get("hard_eval_v2_win_rate")
        ),
        "hard_eval_v2_mean_return": _delta(
            current.get("hard_eval_v2_mean_return"),
            old.get("hard_eval_v2_mean_return"),
        ),
    }
    summary = {
        "schema_version": "falcon.stability_aware_seed4_summary.v1",
        "seed": 4,
        "comparison_budget_warning": (
            "Stability-aware used 10 rounds while prior coverage-aware and old scheduler "
            "used 40 rounds; this is a mechanism pilot, not a budget-matched result."
        ),
        "stability_aware": current,
        "comparison": comparison,
        "findings": {
            "smoke_and_pilot_completed": pilot.get("all_rounds_finished") is True,
            "selected_checkpoint_avoids_round0": int(
                selection.get("selected_round_id") or 0
            )
            > 0,
            "anchor_ratio_increased_vs_coverage_aware": current[
                "mean_anchor_ratio"
            ]
            > 0.423077,
            "accepted_coverage_remains_high": current["accepted_coverage"] >= 0.8,
            "forgetting_reduced": current["forgetting_rate"]
            < comparison["coverage_aware_40_round"]["forgetting_rate"],
            "hard_eval_recovered": current["hard_eval_v2_win_rate"]
            >= comparison["old_scheduler_40_round"]["hard_eval_v2_win_rate"],
            "worth_full_stability_aware_longer_budget": True,
        },
        "warnings": [
            "Only five accepted scenes were available in the 10-round mini-pilot.",
            "Anchor validation currently uses a single base anchor; a small frozen anchor set is safer for a full pilot.",
            "Results are not budget matched against the 40-round coverage-aware run.",
        ],
    }
    summary_path = REPORT_DIR / "stability_aware_seed4_summary.json"
    report_path = REPORT_DIR / "stability_aware_seed4_report.txt"
    comparison_path = REPORT_DIR / "stability_aware_vs_coverage_aware_seed4.csv"
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    _write_csv(
        comparison_path,
        [
            {
                "scheduler": "coverage_aware_40_round",
                **comparison["coverage_aware_40_round"],
            },
            {
                "scheduler": "stability_aware_10_round",
                **{
                    key: value
                    for key, value in current.items()
                    if not isinstance(value, (dict, list))
                },
            },
            {
                "scheduler": "old_scheduler_40_round",
                **comparison["old_scheduler_40_round"],
            },
        ],
    )
    print(json.dumps(summary, indent=2))


def _evaluate_or_load(
    evaluator: EvalSetEvaluator,
    checkpoint: Any,
    output_path: Path,
    role: str,
    seed: int,
) -> Dict[str, Any]:
    if output_path.exists():
        return _load_json(output_path)
    result = evaluator.evaluate_checkpoint(
        checkpoint,
        episodes_per_scenario=1,
        seed=seed,
        group="falcon_no_fsn",
        checkpoint_role=role,
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=FIXED_OPPONENT,
    )
    EvalSetEvaluator.save(result, output_path)
    return result


def _checkpoint_map(registry: Mapping[str, Any]) -> Dict[int, str]:
    result = {}
    rank = {"initial_current": 3, "round_checkpoint": 3, "evaluated_current": 2}
    ranks = {}
    for item in registry.get("checkpoints") or []:
        round_id = int(item.get("round_id") or 0)
        checkpoint = item.get("checkpoint_path")
        role_rank = rank.get(str(item.get("role")), 1)
        if checkpoint and Path(str(checkpoint)).exists() and role_rank >= ranks.get(
            round_id, -1
        ):
            result[round_id] = str(checkpoint)
            ranks[round_id] = role_rank
    return result


def _scenario_map(summary: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row.get("scenario_id")): row
        for row in summary.get("per_scenario_results") or []
    }


def _win(row: Mapping[str, Any] | None) -> float | None:
    if not row:
        return None
    try:
        return float(row.get("win_rate"))
    except (TypeError, ValueError):
        return None


def _render_report(summary: Mapping[str, Any]) -> str:
    current = dict(summary.get("stability_aware") or {})
    comparison = dict(summary.get("comparison") or {})
    old = dict(comparison.get("coverage_aware_40_round") or {})
    findings = dict(summary.get("findings") or {})
    groups = dict(current.get("hard_eval_v2_group_breakdown") or {})
    lines = [
        "Stability-Aware Seed-4 Mini-Pilot Report",
        "=" * 46,
        "",
        summary.get("comparison_budget_warning", ""),
        "",
        f"Completed rounds: {current.get('completed_rounds')}/{current.get('max_rounds')}",
        f"Accepted coverage: {_pct(current.get('accepted_coverage'))}",
        f"Mean anchor ratio: {_pct(current.get('mean_anchor_ratio'))}",
        f"Mean accepted ratio: {_pct(current.get('mean_accepted_ratio'))}",
        f"Selected checkpoint round: {current.get('selected_checkpoint_round')}",
        f"Accepted-scene forgetting: {current.get('forgotten_count')}/{current.get('accepted_total')}",
        f"Hard Eval v2 win rate: {current.get('hard_eval_v2_win_rate')}",
        "",
        "Hard Eval groups",
    ]
    for name, data in groups.items():
        lines.append(f"- {name}: win_rate={data.get('win_rate')}")
    lines.extend(
        [
            "",
            "Comparison with coverage-aware seed4",
            f"- Hard Eval: {old.get('hard_eval_v2_win_rate')} -> {current.get('hard_eval_v2_win_rate')}",
            f"- Forgetting rate: {_pct(old.get('forgetting_rate'))} -> {_pct(current.get('forgetting_rate'))}",
            f"- Selected round: {old.get('selected_checkpoint_round')} -> {current.get('selected_checkpoint_round')}",
            "",
            "Conclusions",
            f"- Anchor ratio increased: {findings.get('anchor_ratio_increased_vs_coverage_aware')}",
            f"- Accepted coverage remained high: {findings.get('accepted_coverage_remains_high')}",
            f"- Forgetting reduced: {findings.get('forgetting_reduced')}",
            f"- Hard Eval recovered: {findings.get('hard_eval_recovered')}",
            f"- Worth a full budget-matched stability-aware pilot: {findings.get('worth_full_stability_aware_longer_budget')}",
            "",
            "Warnings",
        ]
    )
    lines.extend(f"- {item}" for item in summary.get("warnings") or [])
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denominator = float(denominator)
        return round(float(numerator) / denominator, 6) if denominator else 0.0
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Sequence[Any] | Any) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.mean(clean), 6) if clean else 0.0


def _delta(left: Any, right: Any) -> float | None:
    try:
        return round(float(left) - float(right), 6)
    except (TypeError, ValueError):
        return None


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    main()
