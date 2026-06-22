#!/usr/bin/env python
"""Combine seed-3/4 budget-matched stability-aware FALCON diagnostics."""

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

EXPERIMENT_ROOT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
RESULTS_ROOT = EXPERIMENT_ROOT / "results_stability_aware_longer_budget"
REPORT_DIR = EXPERIMENT_ROOT / "reports"
HARD_MANIFEST = EXPERIMENT_ROOT / "manifests" / "hard_eval_scenarios_v2.json"
FIXED_OPPONENT = (
    EXPERIMENT_ROOT
    / "opponents"
    / "fixed_baseline_opponent"
    / "candidate_seed999_steps2048"
    / "checkpoints"
    / "actor_seed999_steps2048.pt"
)
BASE_CONFIG = (
    ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
)
PRIOR_SUMMARY = REPORT_DIR / "coverage_aware_longer_budget_summary.json"
SEED4_SUMMARY = REPORT_DIR / "stability_aware_40r_seed4_summary.json"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prior = _load_json(PRIOR_SUMMARY)
    seed4_report = _load_json(SEED4_SUMMARY)
    seed_results = {
        "3": _analyze_seed(
            3,
            dict((prior.get("seed_results") or {}).get("3") or {}),
            dict((prior.get("old_scheduler_seed_results") or {}).get("3") or {}),
        ),
        "4": dict(seed4_report.get("stability_aware_40_round") or {}),
    }
    for seed in (3, 4):
        current = seed_results[str(seed)]
        current["seed"] = seed
        current["coverage_aware_reference"] = dict(
            (prior.get("seed_results") or {}).get(str(seed)) or {}
        )
        current["old_scheduler_reference"] = dict(
            (prior.get("old_scheduler_seed_results") or {}).get(str(seed)) or {}
        )

    stability_wins = [
        float(seed_results[str(seed)]["hard_eval_v2"]["validation_selected"]["win_rate"])
        for seed in (3, 4)
    ]
    old_wins = [
        float(seed_results[str(seed)]["old_scheduler_reference"]["hard_eval_v2_win_rate"])
        for seed in (3, 4)
    ]
    coverage_wins = [
        float(seed_results[str(seed)]["coverage_aware_reference"]["hard_eval_v2_win_rate"])
        for seed in (3, 4)
    ]
    summary = {
        "schema_version": "falcon.stability_aware_40r_seed34_summary.v1",
        "seeds": [3, 4],
        "budget": {
            "max_rounds": 40,
            "total_train_steps_per_round": 512,
            "scenario_batch_size": 8,
            "per_scenario_train_steps": 64,
            "fixed_opponent": str(FIXED_OPPONENT),
            "hard_eval_scenarios": 40,
            "hard_eval_episodes_per_scenario": 3,
        },
        "seed_results": seed_results,
        "aggregate": {
            "stability_aware_hard_eval_win_rate_mean": _mean(stability_wins),
            "stability_aware_hard_eval_win_rate_std": _std(stability_wins),
            "old_scheduler_hard_eval_win_rate_mean": _mean(old_wins),
            "coverage_aware_hard_eval_win_rate_mean": _mean(coverage_wins),
            "accepted_coverage_mean": _mean(
                seed_results[str(seed)].get("accepted_coverage") for seed in (3, 4)
            ),
            "forgotten_accepted_scenes_total": sum(
                int(seed_results[str(seed)].get("forgotten_count") or 0)
                for seed in (3, 4)
            ),
            "selected_rounds": [
                seed_results[str(seed)].get("validation_selected_round")
                for seed in (3, 4)
            ],
            "hard_eval_improved_vs_old_scheduler_seeds": sum(
                stability_wins[index] > old_wins[index] for index in range(2)
            ),
            "hard_eval_maintained_or_improved_vs_old_scheduler_seeds": sum(
                stability_wins[index] >= old_wins[index] for index in range(2)
            ),
            "hard_eval_improved_vs_coverage_aware_seeds": sum(
                stability_wins[index] > coverage_wins[index] for index in range(2)
            ),
        },
        "findings": {
            "high_coverage_reproduced": all(
                float(seed_results[str(seed)].get("accepted_coverage") or 0.0) >= 0.9
                for seed in (3, 4)
            ),
            "round0_selection_avoided_both_seeds": all(
                int(seed_results[str(seed)].get("validation_selected_round") or 0) > 0
                for seed in (3, 4)
            ),
            "forgetting_reduced_vs_old_scheduler_both_seeds": all(
                int(seed_results[str(seed)].get("forgotten_count") or 0)
                < int(
                    seed_results[str(seed)]["old_scheduler_reference"].get(
                        "forgotten_accepted_scenes"
                    )
                    or 0
                )
                for seed in (3, 4)
            ),
            "forgetting_not_worse_than_coverage_aware_both_seeds": all(
                int(seed_results[str(seed)].get("forgotten_count") or 0)
                <= int(
                    seed_results[str(seed)]["coverage_aware_reference"].get(
                        "forgotten_accepted_scenes"
                    )
                    or 0
                )
                for seed in (3, 4)
            ),
            "hard_eval_gain_reproduced_both_seeds": all(
                stability_wins[index] >= old_wins[index] for index in range(2)
            ),
            "seed3_hard_generalization_regression": stability_wins[0] < old_wins[0],
            "seed3_terminal_outperforms_validation_selected": (
                float(
                    seed_results["3"]["hard_eval_v2"]["terminal"].get("win_rate")
                    or 0.0
                )
                > float(
                    seed_results["3"]["hard_eval_v2"]["validation_selected"].get(
                        "win_rate"
                    )
                    or 0.0
                )
            ),
            "checkpoint_selection_metric_mismatch_detected": (
                float(
                    seed_results["3"]["hard_eval_v2"]["terminal"].get("win_rate")
                    or 0.0
                )
                > float(
                    seed_results["3"]["hard_eval_v2"]["validation_selected"].get(
                        "win_rate"
                    )
                    or 0.0
                )
            ),
            "seed4_hard_generalization_gain": stability_wins[1] > old_wins[1],
            "scheduler_mechanism_reproducible": True,
            "performance_gain_reproducible": False,
            "csa_scheduler_ready_as_core_module": False,
        },
        "recommendation": {
            "expand_directly_to_five_seeds": False,
            "next_priority": "diagnose seed3 initial-disadvantage generalization and selection metric mismatch",
            "csa_status": (
                "Promising scheduler module: coverage and retention effects reproduce, "
                "but hard-eval performance gains do not yet reproduce."
            ),
        },
        "warnings": [
            "Seed3 stability-aware Hard Eval v2 is below both old scheduler and coverage-aware references.",
            "Seed3 terminal checkpoint reaches 0.975 Hard Eval win rate, above the validation-selected checkpoint at 0.85; checkpoint-selection metrics are not fully aligned with Hard Eval generalization.",
            "Seed4 reaches saturated Hard Eval win rate, while seed3 remains weak on initial-disadvantage scenarios.",
            "Two seeds support a mechanism claim about coverage and forgetting, not a stable performance-superiority claim.",
        ],
    }
    json_path = REPORT_DIR / "stability_aware_40r_seed34_summary.json"
    report_path = REPORT_DIR / "stability_aware_40r_seed34_report.txt"
    csv_path = REPORT_DIR / "stability_aware_40r_seed34_vs_old.csv"
    _write_json(json_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    _write_csv(csv_path, _comparison_rows(seed_results))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _analyze_seed(
    seed: int,
    coverage_reference: Mapping[str, Any],
    old_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    seed_root = RESULTS_ROOT / "falcon_no_fsn" / f"seed_{seed}"
    pilot_dir = seed_root / "pilot_run"
    controller_dir = pilot_dir / "controller"
    diagnostics_dir = seed_root / "diagnostics" / "stability_aware_40r"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pilot = _load_json(pilot_dir / "pilot_run_summary.json")
    pool = _load_json(controller_dir / "falcon_curriculum_pool_final.json")
    registry = _load_json(controller_dir / "falcon_checkpoint_registry.json")
    selection = _load_json(
        seed_root
        / "eval_set"
        / "validation_checkpoint_selection"
        / "validation_selected_checkpoint.json"
    )
    selected_hard = _load_json(
        seed_root / "eval_set" / "hard_eval_v2" / "hard_eval_v2_summary.json"
    )
    accepted = [
        item
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool")
    ]
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
    trained_rounds = [
        row
        for row in pilot.get("round_summaries") or []
        if int(row.get("round_id") or 0) > 0
    ]
    last_round = trained_rounds[-1]
    checkpoint_map = _checkpoint_map(registry)
    checkpoints = {
        "round0": checkpoint_map.get(0),
        "validation_selected": selection.get("selected_checkpoint"),
        "batch_best": last_round.get("selected_checkpoint_path"),
        "terminal": last_round.get("terminal_checkpoint_path"),
    }
    evaluator = EvalSetEvaluator(
        accepted_manifest, {"base_config_path": str(BASE_CONFIG)}
    )
    accepted_evals: Dict[str, Dict[str, Any]] = {}
    seen_checkpoints: Dict[str, Dict[str, Any]] = {}
    for index, (role, checkpoint) in enumerate(checkpoints.items()):
        checkpoint_key = str(Path(str(checkpoint)).resolve()) if checkpoint else role
        if checkpoint_key in seen_checkpoints:
            accepted_evals[role] = seen_checkpoints[checkpoint_key]
            continue
        result = _evaluate_or_load(
            evaluator,
            checkpoint,
            diagnostics_dir / f"{role}_accepted_eval.json",
            role,
            episodes=1,
            seed=1000000 + seed * 10000 + index * 1000,
        )
        accepted_evals[role] = result
        seen_checkpoints[checkpoint_key] = result

    acceptance_rows: Dict[str, Mapping[str, Any]] = {}
    by_round: Dict[int, list[Mapping[str, Any]]] = {}
    for item in accepted:
        by_round.setdefault(int(item.get("source_round") or 0), []).append(item)
    for source_round, items in sorted(by_round.items()):
        checkpoint = checkpoint_map.get(max(source_round - 1, 0))
        if not checkpoint:
            continue
        manifest_path = diagnostics_dir / f"acceptance_round_{source_round}.json"
        _write_json(
            manifest_path,
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
                manifest_path, {"base_config_path": str(BASE_CONFIG)}
            ),
            checkpoint,
            diagnostics_dir / f"acceptance_round_{source_round}_eval.json",
            f"pre_acceptance_{source_round}",
            episodes=1,
            seed=1100000 + seed * 10000 + source_round,
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
        batch_win = _win(accepted_maps["batch_best"].get(key))
        terminal_win = _win(accepted_maps["terminal"].get(key))
        progress_rows.append(
            {
                "pool_item_id": key,
                "source_round": item.get("source_round"),
                "train_count": item.get("train_count"),
                "acceptance_win_rate": acceptance_win,
                "round0_win_rate": round0_win,
                "selected_win_rate": selected_win,
                "batch_best_win_rate": batch_win,
                "terminal_win_rate": terminal_win,
                "hard_to_easy": _hard_to_easy(acceptance_win, selected_win),
                "forgotten_vs_round0": _forgotten(round0_win, selected_win),
                "terminal_forgotten_vs_batch_best": _forgotten(
                    batch_win, terminal_win
                ),
            }
        )
    _write_csv(diagnostics_dir / "accepted_scene_policy_progress.csv", progress_rows)

    hard_evaluator = EvalSetEvaluator(
        HARD_MANIFEST, {"base_config_path": str(BASE_CONFIG)}
    )
    hard_evals = {"validation_selected": selected_hard}
    hard_seen = {
        str(Path(str(checkpoints["validation_selected"])).resolve()): selected_hard
    }
    for index, role in enumerate(("batch_best", "terminal")):
        checkpoint = checkpoints[role]
        key = str(Path(str(checkpoint)).resolve())
        if key in hard_seen:
            hard_evals[role] = hard_seen[key]
            continue
        result = _evaluate_or_load(
            hard_evaluator,
            checkpoint,
            diagnostics_dir / f"{role}_hard_eval_v2.json",
            role,
            episodes=3,
            seed=1200000 + seed * 10000 + index * 1000,
        )
        hard_evals[role] = result
        hard_seen[key] = result

    train_counts = [int(item.get("train_count") or 0) for item in accepted]
    trained_count = sum(count > 0 for count in train_counts)
    events = sum(train_counts)
    return {
        "max_rounds": pilot.get("max_rounds"),
        "completed_rounds": pilot.get("completed_rounds"),
        "all_rounds_finished": pilot.get("all_rounds_finished"),
        "failure_stage": pilot.get("failure_stage"),
        "runtime_seconds": pilot.get("runtime_seconds"),
        "accepted_total": len(accepted),
        "accepted_trained_count": trained_count,
        "accepted_coverage": _ratio(trained_count, len(accepted)),
        "accepted_training_events": events,
        "duplicate_rate": _ratio(max(events - trained_count, 0), events),
        "training_hhi": _hhi(train_counts),
        "mean_anchor_ratio": _mean(
            row.get("anchor_ratio") for row in trained_rounds
        ),
        "mean_accepted_ratio": _mean(
            row.get("accepted_ratio") for row in trained_rounds
        ),
        "difficulty_empty_rounds": sum(
            int(row.get("num_accepted_into_pool") or 0) == 0
            for row in pilot.get("round_summaries") or []
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
        "validation_selected_round": selection.get("selected_round_id"),
        "validation_selected_checkpoint": checkpoints["validation_selected"],
        "batch_best_checkpoint": checkpoints["batch_best"],
        "batch_best_source_round": _source_round(checkpoints["batch_best"]),
        "terminal_checkpoint": checkpoints["terminal"],
        "hard_to_easy_count": sum(row["hard_to_easy"] for row in progress_rows),
        "forgotten_count": sum(row["forgotten_vs_round0"] for row in progress_rows),
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
        "coverage_aware_reference": dict(coverage_reference),
        "old_scheduler_reference": dict(old_reference),
    }


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


def _comparison_rows(
    seed_results: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows = []
    for seed in (3, 4):
        current = seed_results[str(seed)]
        rows.extend(
            [
                _row(seed, "old_scheduler", current["old_scheduler_reference"]),
                _row(
                    seed,
                    "coverage_aware",
                    current["coverage_aware_reference"],
                ),
                _row(seed, "stability_aware", current),
            ]
        )
    return rows


def _row(seed: int, scheduler: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    selected = dict((data.get("hard_eval_v2") or {}).get("validation_selected") or {})
    return {
        "seed": seed,
        "scheduler": scheduler,
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
        "hard_eval_v2_win_rate": selected.get("win_rate")
        if selected
        else data.get("hard_eval_v2_win_rate"),
        "hard_eval_v2_mean_return": selected.get("mean_return")
        if selected
        else data.get("hard_eval_v2_mean_return"),
        "difficulty_empty_rounds": data.get("difficulty_empty_rounds"),
        "hard_coordination_stress_win_rate": (
            selected.get("group_breakdown") or data.get("hard_eval_v2_group_breakdown") or {}
        ).get("hard_coordination_stress", {}).get("win_rate"),
        "hard_random_initial_disadvantage_win_rate": (
            selected.get("group_breakdown") or data.get("hard_eval_v2_group_breakdown") or {}
        ).get("hard_random_initial_disadvantage", {}).get("win_rate"),
        "hard_target_assignment_stress_win_rate": (
            selected.get("group_breakdown") or data.get("hard_eval_v2_group_breakdown") or {}
        ).get("hard_target_assignment_stress", {}).get("win_rate"),
        "replay_failure_variants_win_rate": (
            selected.get("group_breakdown") or data.get("hard_eval_v2_group_breakdown") or {}
        ).get("replay_failure_variants", {}).get("win_rate"),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    seeds = dict(summary.get("seed_results") or {})
    aggregate = dict(summary.get("aggregate") or {})
    findings = dict(summary.get("findings") or {})
    lines = [
        "Stability-Aware FALCON 40-Round Seed3/4 Replication Report",
        "=" * 62,
        "",
    ]
    for seed in (3, 4):
        current = seeds[str(seed)]
        hard = current["hard_eval_v2"]["validation_selected"]
        terminal = current["hard_eval_v2"]["terminal"]
        old = current["old_scheduler_reference"]
        coverage = current["coverage_aware_reference"]
        lines.extend(
            [
                f"Seed {seed}",
                f"- Accepted coverage: {_pct(current.get('accepted_coverage'))}",
                f"- Forgotten accepted scenes: {current.get('forgotten_count')}/{current.get('accepted_total')}",
                f"- Hard-to-easy converted scenes: {current.get('hard_to_easy_count')}",
                f"- Selected checkpoint round: {current.get('validation_selected_round')}",
                f"- Hard Eval v2: win_rate={hard.get('win_rate')}, mean_return={hard.get('mean_return')}",
                f"- Terminal Hard Eval: win_rate={terminal.get('win_rate')}, mean_return={terminal.get('mean_return')}",
                f"- Old scheduler Hard Eval: {old.get('hard_eval_v2_win_rate')}",
                f"- Coverage-aware Hard Eval: {coverage.get('hard_eval_v2_win_rate')}",
                f"- Mean anchor/accepted ratio: {_pct(current.get('mean_anchor_ratio'))}/{_pct(current.get('mean_accepted_ratio'))}",
                f"- Duplicate rate: {_pct(current.get('duplicate_rate'))}",
                "",
            ]
        )
    lines.extend(
        [
            "Aggregate",
            f"- Stability-aware Hard Eval mean/std: {aggregate.get('stability_aware_hard_eval_win_rate_mean')} / {aggregate.get('stability_aware_hard_eval_win_rate_std')}",
            f"- Old scheduler Hard Eval mean: {aggregate.get('old_scheduler_hard_eval_win_rate_mean')}",
            f"- Coverage-aware Hard Eval mean: {aggregate.get('coverage_aware_hard_eval_win_rate_mean')}",
            f"- Accepted coverage mean: {_pct(aggregate.get('accepted_coverage_mean'))}",
            f"- Total forgotten accepted scenes: {aggregate.get('forgotten_accepted_scenes_total')}",
            "",
            "Judgement",
            f"- High coverage reproduced: {findings.get('high_coverage_reproduced')}",
            f"- Round0 selection avoided in both seeds: {findings.get('round0_selection_avoided_both_seeds')}",
            f"- Forgetting reduced vs old scheduler in both seeds: {findings.get('forgetting_reduced_vs_old_scheduler_both_seeds')}",
            f"- Forgetting not worse than coverage-aware in both seeds: {findings.get('forgetting_not_worse_than_coverage_aware_both_seeds')}",
            f"- Hard Eval gain reproduced in both seeds: {findings.get('hard_eval_gain_reproduced_both_seeds')}",
            f"- Checkpoint-selection metric mismatch detected: {findings.get('checkpoint_selection_metric_mismatch_detected')}",
            f"- Scheduler mechanism reproducible: {findings.get('scheduler_mechanism_reproducible')}",
            f"- Performance gain reproducible: {findings.get('performance_gain_reproducible')}",
            f"- Ready to freeze CSA as a core FALCON module: {findings.get('csa_scheduler_ready_as_core_module')}",
            "",
            "Recommendation",
            f"- {summary.get('recommendation', {}).get('csa_status')}",
            f"- Next priority: {summary.get('recommendation', {}).get('next_priority')}",
            f"- Expand directly to five seeds: {summary.get('recommendation', {}).get('expand_directly_to_five_seeds')}",
            "",
            "Warnings",
        ]
    )
    lines.extend(f"- {item}" for item in summary.get("warnings") or [])
    return "\n".join(lines) + "\n"


def _checkpoint_map(registry: Mapping[str, Any]) -> Dict[int, str]:
    output: Dict[int, str] = {}
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
            output[round_id] = str(checkpoint)
            ranks[round_id] = rank
    return output


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


def _source_round(path: Any) -> int | None:
    match = re.search(r"round(\d+)_multi_scenario_training", str(path or ""))
    return int(match.group(1)) if match else None


def _different_paths(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return Path(str(left)).resolve() != Path(str(right)).resolve()


def _hhi(counts: Sequence[int]) -> float:
    total = sum(counts)
    return (
        round(sum((count / total) ** 2 for count in counts if count > 0), 6)
        if total
        else 0.0
    )


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


def _mean(values: Sequence[Any] | Any) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.mean(clean), 6) if clean else 0.0


def _std(values: Sequence[Any]) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0


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
