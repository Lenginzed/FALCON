#!/usr/bin/env python
"""Analyze the budget-matched stability-aware FALCON seed-4 pilot."""

from __future__ import annotations

import csv
import json
import re
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
    / "results_stability_aware_longer_budget"
    / "falcon_no_fsn"
    / "seed_4"
)
REPORT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "reports"
OLD_COVERAGE_SUMMARY = REPORT_DIR / "coverage_aware_longer_budget_summary.json"
HARD_MANIFEST = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "manifests"
    / "hard_eval_scenarios_v2.json"
)
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
    pilot_dir = RESULTS_ROOT / "pilot_run"
    controller_dir = pilot_dir / "controller"
    pilot = _load_json(pilot_dir / "pilot_run_summary.json")
    pool = _load_json(controller_dir / "falcon_curriculum_pool_final.json")
    registry = _load_json(controller_dir / "falcon_checkpoint_registry.json")
    selection = _load_json(
        RESULTS_ROOT
        / "eval_set"
        / "validation_checkpoint_selection"
        / "validation_selected_checkpoint.json"
    )
    selected_hard_eval = _load_json(
        RESULTS_ROOT / "eval_set" / "hard_eval_v2" / "hard_eval_v2_summary.json"
    )
    old_summary = _load_json(OLD_COVERAGE_SUMMARY)
    old_coverage = dict((old_summary.get("seed_results") or {}).get("4") or {})
    old_scheduler = dict(
        (old_summary.get("old_scheduler_seed_results") or {}).get("4") or {}
    )

    accepted = [
        item
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    ]
    diagnostics_dir = RESULTS_ROOT / "diagnostics" / "stability_aware_40r"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    accepted_manifest = diagnostics_dir / "accepted_scenes_manifest.json"
    _write_json(
        accepted_manifest,
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

    round_summaries = list(pilot.get("round_summaries") or [])
    trained_rounds = [
        item for item in round_summaries if int(item.get("round_id") or 0) > 0
    ]
    last_round = trained_rounds[-1]
    checkpoint_map = _checkpoint_map(registry)
    checkpoints = {
        "round0": checkpoint_map.get(0),
        "validation_selected": selection.get("selected_checkpoint"),
        "batch_best": last_round.get("selected_checkpoint_path"),
        "terminal": last_round.get("terminal_checkpoint_path"),
    }

    accepted_evaluator = EvalSetEvaluator(
        accepted_manifest, {"base_config_path": str(BASE_CONFIG)}
    )
    accepted_evals = {}
    for index, (role, checkpoint) in enumerate(checkpoints.items()):
        accepted_evals[role] = _evaluate_or_load(
            accepted_evaluator,
            checkpoint,
            diagnostics_dir / f"{role}_accepted_eval.json",
            role,
            episodes=1,
            seed=970000 + index * 1000,
        )

    acceptance_rows: Dict[str, Mapping[str, Any]] = {}
    accepted_by_round: Dict[int, list[Mapping[str, Any]]] = {}
    for item in accepted:
        accepted_by_round.setdefault(int(item.get("source_round") or 0), []).append(
            item
        )
    for source_round, items in sorted(accepted_by_round.items()):
        pre_acceptance_checkpoint = checkpoint_map.get(max(source_round - 1, 0))
        if not pre_acceptance_checkpoint:
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
        result = _evaluate_or_load(
            EvalSetEvaluator(
                sub_manifest, {"base_config_path": str(BASE_CONFIG)}
            ),
            pre_acceptance_checkpoint,
            diagnostics_dir / f"acceptance_round_{source_round}_eval.json",
            f"pre_acceptance_{source_round}",
            episodes=1,
            seed=980000 + source_round,
        )
        acceptance_rows.update(_scenario_map(result))

    accepted_maps = {
        role: _scenario_map(result) for role, result in accepted_evals.items()
    }
    progress_rows = []
    for item in accepted:
        key = str(item.get("pool_item_id"))
        acceptance_win = _win(acceptance_rows.get(key))
        round0_win = _win(accepted_maps["round0"].get(key))
        selected_win = _win(accepted_maps["validation_selected"].get(key))
        batch_best_win = _win(accepted_maps["batch_best"].get(key))
        terminal_win = _win(accepted_maps["terminal"].get(key))
        progress_rows.append(
            {
                "pool_item_id": key,
                "scenario_id": item.get("scenario_id"),
                "source_round": item.get("source_round"),
                "train_count": item.get("train_count"),
                "acceptance_win_rate": acceptance_win,
                "round0_win_rate": round0_win,
                "validation_selected_win_rate": selected_win,
                "batch_best_win_rate": batch_best_win,
                "terminal_win_rate": terminal_win,
                "hard_to_easy": _hard_to_easy(acceptance_win, selected_win),
                "forgotten_vs_round0": _forgotten(round0_win, selected_win),
                "terminal_forgotten_vs_batch_best": _forgotten(
                    batch_best_win, terminal_win
                ),
                "scenario_yaml_path": item.get("scenario_yaml_path"),
            }
        )
    _write_csv(
        diagnostics_dir / "accepted_scene_policy_progress.csv", progress_rows
    )

    hard_evaluator = EvalSetEvaluator(
        HARD_MANIFEST, {"base_config_path": str(BASE_CONFIG)}
    )
    hard_evals = {
        "validation_selected": selected_hard_eval,
        "batch_best": _evaluate_or_load(
            hard_evaluator,
            checkpoints["batch_best"],
            diagnostics_dir / "batch_best_hard_eval_v2.json",
            "batch_best",
            episodes=3,
            seed=990000,
        ),
        "terminal": _evaluate_or_load(
            hard_evaluator,
            checkpoints["terminal"],
            diagnostics_dir / "terminal_hard_eval_v2.json",
            "terminal",
            episodes=3,
            seed=991000,
        ),
    }

    train_counts = [int(item.get("train_count") or 0) for item in accepted]
    accepted_total = len(accepted)
    accepted_trained = sum(count > 0 for count in train_counts)
    training_events = sum(train_counts)
    current = {
        "max_rounds": pilot.get("max_rounds"),
        "completed_rounds": pilot.get("completed_rounds"),
        "all_rounds_finished": pilot.get("all_rounds_finished"),
        "failure_stage": pilot.get("failure_stage"),
        "runtime_seconds": pilot.get("runtime_seconds"),
        "accepted_total": accepted_total,
        "accepted_trained_count": accepted_trained,
        "accepted_coverage": _ratio(accepted_trained, accepted_total),
        "accepted_training_events": training_events,
        "duplicate_rate": _ratio(
            max(training_events - accepted_trained, 0), training_events
        ),
        "training_hhi": _hhi(train_counts),
        "mean_anchor_ratio": _mean(
            row.get("anchor_ratio") for row in trained_rounds
        ),
        "mean_accepted_ratio": _mean(
            row.get("accepted_ratio") for row in trained_rounds
        ),
        "difficulty_empty_rounds": sum(
            int(row.get("num_accepted_into_pool") or 0) == 0
            for row in round_summaries
        ),
        "rounds_selected_nonterminal": sum(
            _different_paths(
                row.get("selected_checkpoint_path"),
                row.get("terminal_checkpoint_path"),
            )
            for row in trained_rounds
        ),
        "rounds_preserved_input_checkpoint": sum(
            row.get("selected_batch_index") == -1 for row in trained_rounds
        ),
        "anchor_validation_scenario_count": _anchor_scenario_count(trained_rounds),
        "validation_selected_round": selection.get("selected_round_id"),
        "validation_selected_checkpoint": checkpoints["validation_selected"],
        "batch_best_checkpoint": checkpoints["batch_best"],
        "batch_best_source_round": _checkpoint_source_round(
            checkpoints["batch_best"]
        ),
        "terminal_checkpoint": checkpoints["terminal"],
        "hard_to_easy_count": sum(row["hard_to_easy"] for row in progress_rows),
        "forgotten_count": sum(
            row["forgotten_vs_round0"] for row in progress_rows
        ),
        "terminal_forgotten_vs_batch_best_count": sum(
            row["terminal_forgotten_vs_batch_best"] for row in progress_rows
        ),
        "accepted_scene_evaluation": {
            role: _aggregate(result) for role, result in accepted_evals.items()
        },
        "hard_eval_v2": {
            role: {
                **_aggregate(result),
                "group_breakdown": result.get("eval_group_breakdown") or {},
            }
            for role, result in hard_evals.items()
        },
    }

    comparison_rows = [
        _comparison_row("old_scheduler_40_round", old_scheduler),
        _comparison_row("coverage_aware_40_round", old_coverage),
        _comparison_row("stability_aware_40_round", current),
    ]
    selected_hard = current["hard_eval_v2"]["validation_selected"]
    batch_hard = current["hard_eval_v2"]["batch_best"]
    terminal_hard = current["hard_eval_v2"]["terminal"]
    summary = {
        "schema_version": "falcon.stability_aware_40r_seed4_summary.v1",
        "seed": 4,
        "protocol_budget": {
            "max_rounds": 40,
            "total_train_steps_per_round": 512,
            "scenario_batch_size": 8,
            "per_scenario_train_steps": 64,
            "budget_matched": True,
        },
        "stability_aware_40_round": current,
        "prior_results": {
            "old_scheduler_40_round": old_scheduler,
            "coverage_aware_40_round": old_coverage,
        },
        "findings": {
            "accepted_coverage_high": current["accepted_coverage"] >= 0.9,
            "anchor_ratio_requirement_met": current["mean_anchor_ratio"] >= 0.5,
            "accepted_ratio_requirement_met": current["mean_accepted_ratio"] <= 0.5,
            "terminal_checkpoint_drift_detected": (
                terminal_hard.get("win_rate") != batch_hard.get("win_rate")
                or terminal_hard.get("mean_return") != batch_hard.get("mean_return")
            ),
            "terminal_regression_avoided_by_selection": (
                selected_hard.get("win_rate", 0.0)
                >= terminal_hard.get("win_rate", 0.0)
            ),
            "accepted_scene_forgetting_lower_than_coverage_aware": (
                current["forgotten_count"]
                < int(old_coverage.get("forgotten_accepted_scenes") or 0)
            ),
            "hard_eval_at_least_old_scheduler": (
                selected_hard.get("win_rate", 0.0)
                >= float(old_scheduler.get("hard_eval_v2_win_rate") or 0.0)
            ),
            "hard_eval_better_than_coverage_aware": (
                selected_hard.get("win_rate", 0.0)
                > float(old_coverage.get("hard_eval_v2_win_rate") or 0.0)
            ),
            "selected_checkpoint_avoids_round0": int(
                current.get("validation_selected_round") or 0
            )
            > 0,
        },
        "recommendation": {
            "expand_to_seed3": True,
            "expand_to_five_seeds": False,
            "reason": (
                "The seed-4 budget-matched mechanism is successful, but one seed is "
                "insufficient for a five-seed claim. Run seed3 next before wider expansion."
            ),
        },
        "warnings": [
            "Independent validation win rate saturated at 1.0 for all sampled checkpoints; mean return determined selection.",
            "This is a single-seed mechanism result and does not establish multi-seed significance.",
        ],
    }
    summary_path = REPORT_DIR / "stability_aware_40r_seed4_summary.json"
    report_path = REPORT_DIR / "stability_aware_40r_seed4_report.txt"
    comparison_path = REPORT_DIR / "stability_aware_40r_seed4_vs_old.csv"
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    _write_csv(comparison_path, comparison_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _evaluate_or_load(
    evaluator: EvalSetEvaluator,
    checkpoint: Any,
    output_path: Path,
    role: str,
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    if output_path.exists():
        return _load_json(output_path)
    result = evaluator.evaluate_checkpoint(
        checkpoint,
        episodes_per_scenario=episodes,
        seed=seed,
        group="falcon_no_fsn",
        checkpoint_role=role,
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=FIXED_OPPONENT,
    )
    EvalSetEvaluator.save(result, output_path)
    return result


def _checkpoint_map(registry: Mapping[str, Any]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    ranks: Dict[int, int] = {}
    role_rank = {"initial_current": 3, "round_checkpoint": 3, "evaluated_current": 2}
    for item in registry.get("checkpoints") or []:
        round_id = int(item.get("round_id") or 0)
        checkpoint = item.get("checkpoint_path")
        rank = role_rank.get(str(item.get("role")), 1)
        if (
            checkpoint
            and Path(str(checkpoint)).exists()
            and rank >= ranks.get(round_id, -1)
        ):
            result[round_id] = str(checkpoint)
            ranks[round_id] = rank
    return result


def _aggregate(result: Mapping[str, Any]) -> Dict[str, Any]:
    aggregate = dict(result.get("aggregate_result") or {})
    return {
        "win_rate": aggregate.get("final_win_rate"),
        "mean_return": aggregate.get("final_mean_return"),
        "num_scenarios": aggregate.get("num_scenarios"),
        "failure_stage": result.get("failure_stage"),
        "same_actor": result.get("same_actor"),
        "opponent_mode": result.get("opponent_mode"),
    }


def _comparison_row(name: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    hard = dict(data.get("hard_eval_v2") or {})
    selected_hard = dict(hard.get("validation_selected") or {})
    return {
        "scheduler": name,
        "accepted_total": data.get("accepted_total"),
        "accepted_trained_count": data.get("accepted_trained_count"),
        "accepted_coverage": data.get("accepted_coverage"),
        "duplicate_rate": data.get("duplicate_rate"),
        "training_hhi": data.get("training_hhi"),
        "anchor_ratio": data.get("mean_anchor_ratio")
        or (data.get("scenario_training_distribution") or {}).get("anchor_ratio"),
        "accepted_ratio": data.get("mean_accepted_ratio"),
        "forgotten_accepted_scenes": data.get("forgotten_count")
        if data.get("forgotten_count") is not None
        else data.get("forgotten_accepted_scenes"),
        "hard_to_easy_converted": data.get("hard_to_easy_count")
        if data.get("hard_to_easy_count") is not None
        else data.get("hard_to_easy_converted_accepted_scenes"),
        "selected_checkpoint_round": data.get("validation_selected_round")
        if data.get("validation_selected_round") is not None
        else data.get("selected_checkpoint_round"),
        "hard_eval_v2_win_rate": selected_hard.get("win_rate")
        if selected_hard
        else data.get("hard_eval_v2_win_rate"),
        "hard_eval_v2_mean_return": selected_hard.get("mean_return")
        if selected_hard
        else data.get("hard_eval_v2_mean_return"),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    current = dict(summary.get("stability_aware_40_round") or {})
    hard = dict(current.get("hard_eval_v2") or {})
    selected = dict(hard.get("validation_selected") or {})
    batch_best = dict(hard.get("batch_best") or {})
    terminal = dict(hard.get("terminal") or {})
    findings = dict(summary.get("findings") or {})
    groups = dict(selected.get("group_breakdown") or {})
    lines = [
        "Stability-Aware FALCON 40-Round Seed-4 Report",
        "=" * 50,
        "",
        "Budget and completion",
        f"- Completed rounds: {current.get('completed_rounds')}/{current.get('max_rounds')}",
        f"- Failure stage: {current.get('failure_stage')}",
        f"- Runtime seconds: {current.get('runtime_seconds')}",
        "",
        "Curriculum transfer",
        f"- Accepted coverage: {_pct(current.get('accepted_coverage'))} ({current.get('accepted_trained_count')}/{current.get('accepted_total')})",
        f"- Duplicate rate: {_pct(current.get('duplicate_rate'))}",
        f"- Training HHI: {current.get('training_hhi')}",
        f"- Mean anchor ratio: {_pct(current.get('mean_anchor_ratio'))}",
        f"- Mean accepted ratio: {_pct(current.get('mean_accepted_ratio'))}",
        f"- Hard-to-easy accepted scenes: {current.get('hard_to_easy_count')}",
        f"- Forgotten accepted scenes: {current.get('forgotten_count')}",
        "",
        "Checkpoint selection",
        f"- Validation-selected round: {current.get('validation_selected_round')}",
        f"- Batch-best source round: {current.get('batch_best_source_round')}",
        f"- Rounds selecting a nonterminal checkpoint: {current.get('rounds_selected_nonterminal')}",
        f"- Rounds preserving the input checkpoint: {current.get('rounds_preserved_input_checkpoint')}",
        "",
        "Hard Eval v2",
        f"- Validation-selected: win_rate={selected.get('win_rate')}, mean_return={selected.get('mean_return')}",
        f"- Batch-best: win_rate={batch_best.get('win_rate')}, mean_return={batch_best.get('mean_return')}",
        f"- Terminal: win_rate={terminal.get('win_rate')}, mean_return={terminal.get('mean_return')}",
    ]
    for group, values in groups.items():
        lines.append(
            f"- {group}: win_rate={values.get('win_rate')}, mean_return={values.get('mean_return')}"
        )
    lines.extend(
        [
            "",
            "Conclusions",
            f"- High accepted coverage retained: {findings.get('accepted_coverage_high')}",
            f"- Anchor minimum met: {findings.get('anchor_ratio_requirement_met')}",
            f"- Terminal drift detected: {findings.get('terminal_checkpoint_drift_detected')}",
            f"- Selection avoided terminal regression: {findings.get('terminal_regression_avoided_by_selection')}",
            f"- Forgetting lower than coverage-aware: {findings.get('accepted_scene_forgetting_lower_than_coverage_aware')}",
            f"- Hard Eval at least old scheduler 0.925: {findings.get('hard_eval_at_least_old_scheduler')}",
            f"- Better than coverage-aware 0.775: {findings.get('hard_eval_better_than_coverage_aware')}",
            f"- Selected checkpoint is not round0: {findings.get('selected_checkpoint_avoids_round0')}",
            "",
            "Recommendation",
            f"- Expand to seed3: {summary.get('recommendation', {}).get('expand_to_seed3')}",
            f"- Expand directly to five seeds: {summary.get('recommendation', {}).get('expand_to_five_seeds')}",
            f"- Reason: {summary.get('recommendation', {}).get('reason')}",
            "",
            "Warnings",
        ]
    )
    lines.extend(f"- {item}" for item in summary.get("warnings") or [])
    return "\n".join(lines) + "\n"


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


def _hard_to_easy(before: float | None, after: float | None) -> bool:
    return before is not None and before <= 0.5 and after is not None and after >= 0.8


def _forgotten(before: float | None, after: float | None) -> bool:
    return before is not None and after is not None and after <= before - 0.2


def _different_paths(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return Path(str(left)).resolve() != Path(str(right)).resolve()


def _checkpoint_source_round(path: Any) -> int | None:
    match = re.search(r"round(\d+)_multi_scenario_training", str(path or ""))
    return int(match.group(1)) if match else None


def _anchor_scenario_count(rows: Sequence[Mapping[str, Any]]) -> int:
    for row in reversed(rows):
        validation = dict(row.get("selected_checkpoint_validation") or {})
        if validation.get("num_scenarios") is not None:
            return int(validation["num_scenarios"])
    return 0


def _hhi(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return round(sum((count / total) ** 2 for count in counts if count > 0), 6)


def _mean(values: Sequence[Any] | Any) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.mean(clean), 6) if clean else 0.0


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denominator_value = float(denominator)
        return (
            round(float(numerator) / denominator_value, 6)
            if denominator_value
            else 0.0
        )
    except (TypeError, ValueError):
        return 0.0


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
