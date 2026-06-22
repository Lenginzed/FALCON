#!/usr/bin/env python
"""Analyze coverage-aware FALCON longer-budget runs against the old scheduler."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import load_yaml  # noqa: E402
from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402

DEFAULT_PROTOCOL = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "configs"
    / "experiment_protocol_coverage_aware_longer_budget.yaml"
)
DEFAULT_OLD_SUMMARY = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "reports"
    / "curriculum_transfer_diagnosis_summary.json"
)
DEFAULT_OLD_HARD_EVAL = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "reports"
    / "longer_budget_hard_eval_summary.json"
)
DEFAULT_OLD_RESULTS_ROOT = (
    ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "results_longer_budget"
)
DEFAULT_REPORT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "reports"

HARD_TO_EASY_ACCEPTANCE_MAX = 0.5
HARD_TO_EASY_LATEST_MIN = 0.8
FORGETTING_DELTA = 0.2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--old-summary", default=str(DEFAULT_OLD_SUMMARY))
    parser.add_argument("--old-hard-eval", default=str(DEFAULT_OLD_HARD_EVAL))
    parser.add_argument("--old-results-root", default=str(DEFAULT_OLD_RESULTS_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--accepted-eval-episodes", type=int, default=1)
    parser.add_argument("--evaluate-accepted-scenes", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    protocol_path = _resolve(args.protocol)
    protocol = load_yaml(protocol_path)
    results_root = _resolve(protocol["output_root"])
    report_dir = _resolve(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    old_summary = _load_json(_resolve(args.old_summary))
    old_hard_eval = _load_json(_resolve(args.old_hard_eval))
    evaluation = dict(protocol.get("evaluation") or {})
    opponent_checkpoint = _resolve(evaluation.get("opponent_checkpoint"))
    base_config_path = _resolve(protocol["base_scenario_config"])

    seed_results: Dict[str, Dict[str, Any]] = {}
    progress_rows = []
    for seed in args.seeds:
        seed_dir = results_root / "falcon_no_fsn" / f"seed_{seed}"
        pilot_dir = seed_dir / "pilot_run"
        controller_dir = pilot_dir / "controller"
        pool_path = controller_dir / "falcon_curriculum_pool_final.json"
        pilot_summary_path = pilot_dir / "pilot_run_summary.json"
        registry_path = controller_dir / "falcon_checkpoint_registry.json"
        selection_path = (
            seed_dir
            / "eval_set"
            / "validation_checkpoint_selection"
            / "validation_selected_checkpoint.json"
        )
        hard_eval_path = seed_dir / "eval_set" / "hard_eval_v2" / "hard_eval_v2_summary.json"
        pool = _load_json(pool_path)
        pilot_summary = _load_json(pilot_summary_path)
        registry = _load_json(registry_path)
        selection = _load_json(selection_path)
        hard_eval = _load_json(hard_eval_path)
        accepted = [
            item
            for item in pool.get("items") or []
            if bool(item.get("accepted_into_curriculum_pool"))
        ]
        coverage = _coverage_metrics(accepted)
        round_metrics = _round_metrics(pilot_summary, controller_dir)
        training_distribution = _training_distribution(controller_dir)
        checkpoint_map = _checkpoint_map(registry)

        accepted_progress = []
        if args.evaluate_accepted_scenes:
            accepted_progress = _evaluate_accepted_scenes(
                seed=seed,
                seed_dir=seed_dir,
                accepted=accepted,
                checkpoint_map=checkpoint_map,
                selected_checkpoint=selection.get("selected_checkpoint"),
                opponent_checkpoint=opponent_checkpoint,
                base_config_path=base_config_path,
                episodes=args.accepted_eval_episodes,
                force_eval=args.force_eval,
            )
            progress_rows.extend(accepted_progress)
        transfer = _transfer_metrics(accepted_progress)
        hard_aggregate = dict(hard_eval.get("aggregate_result") or {})
        seed_results[str(seed)] = {
            "seed": seed,
            "completed_rounds": pilot_summary.get("completed_rounds"),
            "training_failure_stage": pilot_summary.get("failure_stage"),
            "accepted_total": coverage["accepted_total"],
            "accepted_trained_count": coverage["accepted_trained_count"],
            "accepted_unseen_count": coverage["accepted_unseen_count"],
            "accepted_coverage": coverage["accepted_coverage"],
            "accepted_training_events": coverage["accepted_training_events"],
            "duplicate_rate": coverage["duplicate_rate"],
            "training_hhi": coverage["training_hhi"],
            "effective_trained_scene_count": coverage["effective_trained_scene_count"],
            "mean_train_count": coverage["mean_train_count"],
            "max_train_count": coverage["max_train_count"],
            "fallback_count": round_metrics["fallback_count"],
            "fallback_rate": round_metrics["fallback_rate"],
            "difficulty_empty_rounds": round_metrics["difficulty_empty_rounds"],
            "empty_round_rate": round_metrics["empty_round_rate"],
            "scenario_training_distribution": training_distribution,
            "selected_checkpoint_round": selection.get("selected_round_id"),
            "selected_validation_win_rate": selection.get("validation_win_rate"),
            "selected_validation_mean_return": selection.get("validation_mean_return"),
            "latest_checkpoint_round": max(checkpoint_map) if checkpoint_map else None,
            "hard_eval_v2_win_rate": hard_aggregate.get("final_win_rate"),
            "hard_eval_v2_mean_return": hard_aggregate.get("final_mean_return"),
            "hard_eval_v2_group_breakdown": hard_eval.get("eval_group_breakdown") or {},
            "hard_to_easy_converted_accepted_scenes": transfer["hard_to_easy_count"],
            "forgotten_accepted_scenes": transfer["forgotten_count"],
            "accepted_scene_transfer": transfer,
            "warnings": _seed_warnings(
                coverage=coverage,
                selection=selection,
                training_distribution=training_distribution,
            ),
        }

    old_by_seed = _old_metrics_by_seed(
        old_summary,
        old_hard_eval,
        _resolve(args.old_results_root),
        args.seeds,
    )
    comparisons = []
    for seed in args.seeds:
        old = old_by_seed.get(str(seed), {})
        new = seed_results.get(str(seed), {})
        comparisons.append(_comparison_row(seed, old, new))

    summary = _build_summary(
        protocol_path=protocol_path,
        results_root=results_root,
        seeds=args.seeds,
        seed_results=seed_results,
        old_by_seed=old_by_seed,
        comparisons=comparisons,
        accepted_eval_episodes=args.accepted_eval_episodes,
        accepted_scenes_evaluated=args.evaluate_accepted_scenes,
    )
    summary_path = report_dir / "coverage_aware_longer_budget_summary.json"
    summary_csv_path = report_dir / "coverage_aware_longer_budget_summary.csv"
    report_path = report_dir / "coverage_aware_longer_budget_report.txt"
    comparison_path = report_dir / "coverage_aware_vs_old_scheduler_comparison.csv"
    progress_path = report_dir / "coverage_aware_accepted_scene_policy_progress.csv"
    _write_json(summary_path, summary)
    _write_csv(summary_csv_path, [_flatten_seed_row(item) for item in seed_results.values()])
    _write_csv(comparison_path, comparisons)
    if progress_rows:
        _write_csv(progress_path, progress_rows)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _coverage_metrics(accepted: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train_counts = [max(_int(item.get("train_count")), 0) for item in accepted]
    trained_count = sum(count > 0 for count in train_counts)
    events = sum(train_counts)
    training_hhi = (
        sum((count / events) ** 2 for count in train_counts if count > 0)
        if events > 0
        else 0.0
    )
    return {
        "accepted_total": len(accepted),
        "accepted_trained_count": trained_count,
        "accepted_unseen_count": len(accepted) - trained_count,
        "accepted_coverage": _ratio(trained_count, len(accepted)),
        "accepted_training_events": events,
        "duplicate_rate": _ratio(max(events - trained_count, 0), events),
        "training_hhi": round(training_hhi, 6),
        "effective_trained_scene_count": round(1.0 / training_hhi, 6) if training_hhi else 0.0,
        "mean_train_count": _mean(train_counts),
        "max_train_count": max(train_counts, default=0),
    }


def _round_metrics(pilot_summary: Mapping[str, Any], controller_dir: Path) -> Dict[str, Any]:
    rounds = list(pilot_summary.get("round_summaries") or [])
    fallback_count = 0
    for row in rounds:
        round_id = _int(row.get("round_id"))
        path = controller_dir / f"falcon_controller_training_round{round_id}_summary.json"
        if path.exists() and bool(_load_json(path).get("fallback_used")):
            fallback_count += 1
    empty = sum(_int(row.get("num_accepted_into_pool")) == 0 for row in rounds)
    return {
        "fallback_count": fallback_count,
        "fallback_rate": _ratio(fallback_count, len(rounds)),
        "difficulty_empty_rounds": empty,
        "empty_round_rate": _ratio(empty, len(rounds)),
    }


def _training_distribution(controller_dir: Path) -> Dict[str, Any]:
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    anchor_count = 0
    accepted_count = 0
    replay_count = 0
    random_count = 0
    total = 0
    for path in sorted(controller_dir.glob("falcon_controller_training_round*_summary.json")):
        data = _load_json(path)
        results = list((data.get("multi_scenario_training_summary") or {}).get("training_results") or [])
        for item in results:
            if not item.get("training_succeeded"):
                continue
            total += 1
            category = str(item.get("sampling_category") or "unknown")
            source = str(item.get("source") or "unknown")
            category_counts[category] += 1
            source_counts[source] += 1
            if item.get("anchor_role"):
                anchor_count += 1
            if item.get("pool_item_id"):
                accepted_count += 1
            if category == "replay_failure" or source == "replay":
                replay_count += 1
            if category == "random_explore" or source == "random":
                random_count += 1
    return {
        "training_events": total,
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "accepted_event_ratio": _ratio(accepted_count, total),
        "anchor_ratio": _ratio(anchor_count, total),
        "replay_ratio": _ratio(replay_count, total),
        "random_ratio": _ratio(random_count, total),
    }


def _checkpoint_map(registry: Mapping[str, Any]) -> Dict[int, str]:
    by_round: Dict[int, str] = {}
    rank = {"initial_current": 3, "round_checkpoint": 3, "evaluated_current": 2}
    chosen_rank: Dict[int, int] = {}
    for item in registry.get("checkpoints") or []:
        round_id = _int(item.get("round_id"), -1)
        path = str(item.get("checkpoint_path") or "")
        role = str(item.get("role") or "")
        if round_id < 0 or not path or not Path(path).exists():
            continue
        item_rank = rank.get(role, 1)
        if item_rank >= chosen_rank.get(round_id, -1):
            by_round[round_id] = path
            chosen_rank[round_id] = item_rank
    return by_round


def _evaluate_accepted_scenes(
    *,
    seed: int,
    seed_dir: Path,
    accepted: Sequence[Mapping[str, Any]],
    checkpoint_map: Mapping[int, str],
    selected_checkpoint: Any,
    opponent_checkpoint: Path,
    base_config_path: Path,
    episodes: int,
    force_eval: bool,
) -> list[Dict[str, Any]]:
    diagnostics_dir = seed_dir / "diagnostics" / "coverage_aware_transfer"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [_manifest_scenario(item) for item in accepted if _scenario_yaml(item).exists()]
    manifest_path = diagnostics_dir / "accepted_scenes_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "falcon.eval_scenario_manifest.v1",
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
        },
    )
    evaluator = EvalSetEvaluator(manifest_path, {"base_config_path": str(base_config_path)})
    role_paths = {
        "round0": checkpoint_map.get(0),
        "selected": str(selected_checkpoint or ""),
        "latest": checkpoint_map.get(max(checkpoint_map)) if checkpoint_map else None,
    }
    role_results: Dict[str, Dict[str, Any]] = {}
    for role, checkpoint in role_paths.items():
        if not checkpoint or not Path(checkpoint).exists():
            continue
        output_path = diagnostics_dir / f"{role}_accepted_eval.json"
        role_results[role] = _evaluate_or_load(
            evaluator=evaluator,
            checkpoint=checkpoint,
            output_path=output_path,
            episodes=episodes,
            seed=seed,
            opponent_checkpoint=opponent_checkpoint,
            checkpoint_role=role,
            force_eval=force_eval,
        )

    acceptance_results: Dict[str, Mapping[str, Any]] = {}
    by_source_round: Dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in accepted:
        if _scenario_yaml(item).exists():
            by_source_round[_int(item.get("source_round"))].append(item)
    for source_round, items in sorted(by_source_round.items()):
        acceptance_round = max(source_round - 1, 0)
        checkpoint = checkpoint_map.get(acceptance_round)
        if not checkpoint:
            continue
        sub_manifest = diagnostics_dir / f"acceptance_round_{source_round:03d}_manifest.json"
        _write_json(
            sub_manifest,
            {
                "schema_version": "falcon.eval_scenario_manifest.v1",
                "scenario_count": len(items),
                "scenarios": [_manifest_scenario(item) for item in items],
            },
        )
        sub_evaluator = EvalSetEvaluator(sub_manifest, {"base_config_path": str(base_config_path)})
        output_path = diagnostics_dir / f"acceptance_round_{source_round:03d}_eval.json"
        result = _evaluate_or_load(
            evaluator=sub_evaluator,
            checkpoint=checkpoint,
            output_path=output_path,
            episodes=episodes,
            seed=seed + source_round * 1000,
            opponent_checkpoint=opponent_checkpoint,
            checkpoint_role=f"pre_acceptance_round_{source_round}",
            force_eval=force_eval,
        )
        acceptance_results.update(_scenario_result_map(result))

    role_maps = {key: _scenario_result_map(value) for key, value in role_results.items()}
    rows = []
    for item in accepted:
        pool_item_id = str(item.get("pool_item_id") or "")
        acceptance = acceptance_results.get(pool_item_id, {})
        round0 = role_maps.get("round0", {}).get(pool_item_id, {})
        selected = role_maps.get("selected", {}).get(pool_item_id, {})
        latest = role_maps.get("latest", {}).get(pool_item_id, {})
        acceptance_win = _float(acceptance.get("win_rate"))
        round0_win = _float(round0.get("win_rate"))
        selected_win = _float(selected.get("win_rate"))
        latest_win = _float(latest.get("win_rate"))
        rows.append(
            {
                "seed": seed,
                "pool_item_id": pool_item_id,
                "scenario_id": item.get("scenario_id"),
                "source_round": item.get("source_round"),
                "scenario_yaml_path": str(_scenario_yaml(item)),
                "train_count": _int(item.get("train_count")),
                "first_trained_round": item.get("first_trained_round"),
                "last_trained_round": item.get("last_trained_round"),
                "coverage_status": item.get("coverage_status"),
                "acceptance_fixed_win_rate": acceptance_win,
                "round0_win_rate": round0_win,
                "selected_win_rate": selected_win,
                "latest_win_rate": latest_win,
                "latest_minus_acceptance": _difference(latest_win, acceptance_win),
                "latest_minus_round0": _difference(latest_win, round0_win),
                "hard_to_easy_by_latest": bool(
                    acceptance_win is not None
                    and latest_win is not None
                    and acceptance_win <= HARD_TO_EASY_ACCEPTANCE_MAX
                    and latest_win >= HARD_TO_EASY_LATEST_MIN
                ),
                "forgotten_by_latest": bool(
                    round0_win is not None
                    and latest_win is not None
                    and latest_win <= round0_win - FORGETTING_DELTA
                ),
                "target_failure_modes": "|".join(item.get("target_failure_modes") or []),
            }
        )
    return rows


def _evaluate_or_load(
    *,
    evaluator: EvalSetEvaluator,
    checkpoint: str,
    output_path: Path,
    episodes: int,
    seed: int,
    opponent_checkpoint: Path,
    checkpoint_role: str,
    force_eval: bool,
) -> Dict[str, Any]:
    if output_path.exists() and not force_eval:
        return _load_json(output_path)
    result = evaluator.evaluate_checkpoint(
        checkpoint,
        episodes_per_scenario=episodes,
        seed=seed,
        group="falcon_no_fsn",
        checkpoint_role=checkpoint_role,
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=opponent_checkpoint,
    )
    EvalSetEvaluator.save(result, output_path)
    return result


def _manifest_scenario(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_id": str(item.get("pool_item_id") or item.get("scenario_id")),
        "scenario_group": "accepted_curriculum",
        "scenario_yaml_path": str(_scenario_yaml(item)),
        "source_round": item.get("source_round"),
        "original_scenario_id": item.get("scenario_id"),
    }


def _scenario_yaml(item: Mapping[str, Any]) -> Path:
    return Path(str(item.get("scenario_yaml_path") or ""))


def _scenario_result_map(result: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row.get("scenario_id")): row
        for row in result.get("per_scenario_results") or []
        if row.get("real_policy_eval_available")
    }


def _transfer_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "hard_to_easy_count": None,
            "forgotten_count": None,
            "acceptance_win_rate_mean": None,
            "round0_win_rate_mean": None,
            "selected_win_rate_mean": None,
            "latest_win_rate_mean": None,
        }
    return {
        "count": len(rows),
        "hard_to_easy_count": sum(bool(row.get("hard_to_easy_by_latest")) for row in rows),
        "forgotten_count": sum(bool(row.get("forgotten_by_latest")) for row in rows),
        "acceptance_win_rate_mean": _mean_optional(row.get("acceptance_fixed_win_rate") for row in rows),
        "round0_win_rate_mean": _mean_optional(row.get("round0_win_rate") for row in rows),
        "selected_win_rate_mean": _mean_optional(row.get("selected_win_rate") for row in rows),
        "latest_win_rate_mean": _mean_optional(row.get("latest_win_rate") for row in rows),
        "trained": _transfer_subset(rows, trained=True),
        "unseen": _transfer_subset(rows, trained=False),
    }


def _transfer_subset(rows: Sequence[Mapping[str, Any]], *, trained: bool) -> Dict[str, Any]:
    subset = [row for row in rows if (_int(row.get("train_count")) > 0) is trained]
    return {
        "count": len(subset),
        "hard_to_easy_count": sum(bool(row.get("hard_to_easy_by_latest")) for row in subset),
        "forgotten_count": sum(bool(row.get("forgotten_by_latest")) for row in subset),
        "latest_win_rate_mean": _mean_optional(row.get("latest_win_rate") for row in subset),
    }


def _old_metrics_by_seed(
    old_summary: Mapping[str, Any],
    old_hard_eval: Mapping[str, Any],
    old_results_root: Path,
    seeds: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    coverage = dict((old_summary.get("sampling_coverage") or {}).get("seeds") or {})
    transfer = dict(old_summary.get("accepted_scene_transfer") or {})
    old_performance = dict((old_hard_eval.get("performance") or {}).get("falcon_no_fsn") or {})
    old_selection = dict((old_hard_eval.get("validation_selected") or {}).get("falcon_no_fsn") or {})
    result = {}
    for seed in seeds:
        key = str(seed)
        cover = dict(coverage.get(key) or {})
        accepted_per_round = list(cover.get("accepted_per_round") or [])
        transfer_all = dict((transfer.get(key) or {}).get("all") or {})
        performance = dict(old_performance.get(key) or {})
        selection = dict(old_selection.get(key) or {})
        seed_hard_eval = _load_json(
            old_results_root
            / "falcon_no_fsn"
            / f"seed_{seed}"
            / "eval_set"
            / "hard_eval_v2"
            / "hard_eval_v2_summary.json"
        )
        result[key] = {
            "accepted_total": cover.get("accepted_total"),
            "accepted_trained_count": cover.get("accepted_ever_trained"),
            "accepted_coverage": cover.get("accepted_actual_training_coverage_rate"),
            "duplicate_rate": cover.get("accepted_training_repeat_rate"),
            "training_hhi": cover.get("accepted_training_hhi"),
            "fallback_rate": cover.get("fallback_actual_training_ratio"),
            "difficulty_empty_rounds": sum(_int(value) == 0 for value in accepted_per_round),
            "empty_round_rate": _ratio(
                sum(_int(value) == 0 for value in accepted_per_round),
                len(accepted_per_round),
            ),
            "selected_checkpoint_round": selection.get("round_id"),
            "hard_eval_v2_win_rate": performance.get("win_rate"),
            "hard_eval_v2_mean_return": performance.get("mean_return"),
            "hard_eval_v2_group_breakdown": seed_hard_eval.get("eval_group_breakdown") or {},
            "hard_to_easy_converted_accepted_scenes": transfer_all.get("hard_to_easy_count"),
            "forgotten_accepted_scenes": transfer_all.get("forgotten_by_latest_count"),
        }
    return result


def _comparison_row(seed: int, old: Mapping[str, Any], new: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "seed": seed,
        "old_accepted_total": old.get("accepted_total"),
        "coverage_aware_accepted_total": new.get("accepted_total"),
        "old_accepted_trained_count": old.get("accepted_trained_count"),
        "coverage_aware_accepted_trained_count": new.get("accepted_trained_count"),
        "old_accepted_coverage": old.get("accepted_coverage"),
        "coverage_aware_accepted_coverage": new.get("accepted_coverage"),
        "accepted_coverage_delta": _difference(
            _float(new.get("accepted_coverage")), _float(old.get("accepted_coverage"))
        ),
        "old_duplicate_rate": old.get("duplicate_rate"),
        "coverage_aware_duplicate_rate": new.get("duplicate_rate"),
        "duplicate_rate_delta": _difference(
            _float(new.get("duplicate_rate")), _float(old.get("duplicate_rate"))
        ),
        "old_training_hhi": old.get("training_hhi"),
        "coverage_aware_training_hhi": new.get("training_hhi"),
        "training_hhi_delta": _difference(
            _float(new.get("training_hhi")), _float(old.get("training_hhi"))
        ),
        "old_fallback_rate": old.get("fallback_rate"),
        "coverage_aware_fallback_rate": new.get("fallback_rate"),
        "old_difficulty_empty_rounds": old.get("difficulty_empty_rounds"),
        "coverage_aware_difficulty_empty_rounds": new.get("difficulty_empty_rounds"),
        "old_empty_round_rate": old.get("empty_round_rate"),
        "coverage_aware_empty_round_rate": new.get("empty_round_rate"),
        "old_hard_to_easy_count": old.get("hard_to_easy_converted_accepted_scenes"),
        "coverage_aware_hard_to_easy_count": new.get("hard_to_easy_converted_accepted_scenes"),
        "old_forgotten_count": old.get("forgotten_accepted_scenes"),
        "coverage_aware_forgotten_count": new.get("forgotten_accepted_scenes"),
        "old_selected_round": old.get("selected_checkpoint_round"),
        "coverage_aware_selected_round": new.get("selected_checkpoint_round"),
        "old_hard_eval_v2_win_rate": old.get("hard_eval_v2_win_rate"),
        "coverage_aware_hard_eval_v2_win_rate": new.get("hard_eval_v2_win_rate"),
        "hard_eval_v2_win_rate_delta": _difference(
            _float(new.get("hard_eval_v2_win_rate")), _float(old.get("hard_eval_v2_win_rate"))
        ),
        "old_hard_eval_v2_mean_return": old.get("hard_eval_v2_mean_return"),
        "coverage_aware_hard_eval_v2_mean_return": new.get("hard_eval_v2_mean_return"),
        "hard_eval_v2_mean_return_delta": _difference(
            _float(new.get("hard_eval_v2_mean_return")), _float(old.get("hard_eval_v2_mean_return"))
        ),
    }
    old_groups = dict(old.get("hard_eval_v2_group_breakdown") or {})
    new_groups = dict(new.get("hard_eval_v2_group_breakdown") or {})
    for group_name in sorted(set(old_groups) | set(new_groups)):
        old_group = dict(old_groups.get(group_name) or {})
        new_group = dict(new_groups.get(group_name) or {})
        prefix = group_name.replace("hard_", "")
        row[f"old_{prefix}_win_rate"] = old_group.get("win_rate")
        row[f"coverage_aware_{prefix}_win_rate"] = new_group.get("win_rate")
        row[f"{prefix}_win_rate_delta"] = _difference(
            _float(new_group.get("win_rate")), _float(old_group.get("win_rate"))
        )
    return row


def _build_summary(
    *,
    protocol_path: Path,
    results_root: Path,
    seeds: Sequence[int],
    seed_results: Mapping[str, Mapping[str, Any]],
    old_by_seed: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    accepted_eval_episodes: int,
    accepted_scenes_evaluated: bool,
) -> Dict[str, Any]:
    coverages = [_float(item.get("accepted_coverage")) for item in seed_results.values()]
    old_coverages = [_float(item.get("accepted_coverage")) for item in old_by_seed.values()]
    duplicate_rates = [_float(item.get("duplicate_rate")) for item in seed_results.values()]
    old_duplicates = [_float(item.get("duplicate_rate")) for item in old_by_seed.values()]
    hard_rates = [_float(item.get("hard_eval_v2_win_rate")) for item in seed_results.values()]
    old_hard_rates = [_float(item.get("hard_eval_v2_win_rate")) for item in old_by_seed.values()]
    selected_rounds = [item.get("selected_checkpoint_round") for item in seed_results.values()]
    old_selected_rounds = [item.get("selected_checkpoint_round") for item in old_by_seed.values()]
    coverage_improved = all(
        _float(row.get("accepted_coverage_delta"), 0.0) > 0.0 for row in comparisons
    )
    duplicate_reduced = all(
        _float(row.get("duplicate_rate_delta"), 0.0) < 0.0 for row in comparisons
    )
    selected_improved = any(
        _int(new_round) > _int(old_round)
        for new_round, old_round in zip(selected_rounds, old_selected_rounds)
    )
    hard_eval_improved = _mean_optional(hard_rates) > _mean_optional(old_hard_rates)
    forgetting_old = sum(
        _int(item.get("forgotten_accepted_scenes")) for item in old_by_seed.values()
    )
    forgetting_new = sum(
        _int(item.get("forgotten_accepted_scenes"))
        for item in seed_results.values()
        if item.get("forgotten_accepted_scenes") is not None
    )
    forgetting_reduced = accepted_scenes_evaluated and forgetting_new < forgetting_old
    warnings = []
    if 0 in [_int(value) for value in selected_rounds]:
        warnings.append("At least one coverage-aware seed still selected round0.")
    if any(
        _float((item.get("scenario_training_distribution") or {}).get("replay_ratio"), 0.0) == 0.0
        for item in seed_results.values()
    ):
        warnings.append(
            "Replay quota was redistributed because no replay candidates were supplied to the scheduler."
        )
    if any(
        _float((item.get("scenario_training_distribution") or {}).get("random_ratio"), 0.0) == 0.0
        for item in seed_results.values()
    ):
        warnings.append(
            "Random-explore quota was redistributed because no random candidates were supplied."
        )
    return {
        "schema_version": "falcon.coverage_aware_longer_budget_summary.v1",
        "protocol_path": str(protocol_path),
        "results_root": str(results_root),
        "seeds": list(seeds),
        "accepted_scene_eval_episodes": accepted_eval_episodes,
        "accepted_scene_evaluation_completed": accepted_scenes_evaluated,
        "seed_results": dict(seed_results),
        "old_scheduler_seed_results": dict(old_by_seed),
        "comparison": {
            "per_seed": list(comparisons),
            "old_accepted_coverage_mean": _mean_optional(old_coverages),
            "coverage_aware_accepted_coverage_mean": _mean_optional(coverages),
            "accepted_coverage_mean_delta": _difference(
                _mean_optional(coverages), _mean_optional(old_coverages)
            ),
            "old_duplicate_rate_mean": _mean_optional(old_duplicates),
            "coverage_aware_duplicate_rate_mean": _mean_optional(duplicate_rates),
            "duplicate_rate_mean_delta": _difference(
                _mean_optional(duplicate_rates), _mean_optional(old_duplicates)
            ),
            "old_hard_eval_v2_win_rate_mean": _mean_optional(old_hard_rates),
            "coverage_aware_hard_eval_v2_win_rate_mean": _mean_optional(hard_rates),
            "hard_eval_v2_win_rate_mean_delta": _difference(
                _mean_optional(hard_rates), _mean_optional(old_hard_rates)
            ),
            "old_selected_rounds": old_selected_rounds,
            "coverage_aware_selected_rounds": selected_rounds,
            "old_forgotten_count_total": forgetting_old,
            "coverage_aware_forgotten_count_total": forgetting_new,
        },
        "findings": {
            "accepted_coverage_significantly_improved": coverage_improved
            and _mean_optional(coverages) - _mean_optional(old_coverages) >= 0.3,
            "duplicate_rate_reduced": duplicate_reduced,
            "forgetting_reduced": forgetting_reduced,
            "selected_checkpoint_improved_for_at_least_one_seed": selected_improved,
            "all_seeds_avoid_round0": all(_int(value) > 0 for value in selected_rounds),
            "hard_eval_v2_improved": hard_eval_improved,
            "worth_expanding_to_five_seeds": bool(
                coverage_improved and hard_eval_improved and not all(_int(value) == 0 for value in selected_rounds)
            ),
            "ready_for_fsn_data_organization": True,
        },
        "warnings": warnings,
    }


def _seed_warnings(
    *,
    coverage: Mapping[str, Any],
    selection: Mapping[str, Any],
    training_distribution: Mapping[str, Any],
) -> list[str]:
    warnings = []
    if _float(coverage.get("accepted_coverage"), 0.0) < 0.8:
        warnings.append("Accepted-scene training coverage remained below 80%.")
    if _int(selection.get("selected_round_id")) == 0:
        warnings.append("Validation selector still chose round0.")
    if _float(training_distribution.get("replay_ratio"), 0.0) == 0.0:
        warnings.append("No replay/failure scenario was actually trained.")
    if _float(training_distribution.get("random_ratio"), 0.0) == 0.0:
        warnings.append("No random-explore scenario was actually trained.")
    return warnings


def _flatten_seed_row(item: Mapping[str, Any]) -> Dict[str, Any]:
    distribution = dict(item.get("scenario_training_distribution") or {})
    return {
        "seed": item.get("seed"),
        "completed_rounds": item.get("completed_rounds"),
        "accepted_total": item.get("accepted_total"),
        "accepted_trained_count": item.get("accepted_trained_count"),
        "accepted_unseen_count": item.get("accepted_unseen_count"),
        "accepted_coverage": item.get("accepted_coverage"),
        "accepted_training_events": item.get("accepted_training_events"),
        "duplicate_rate": item.get("duplicate_rate"),
        "training_hhi": item.get("training_hhi"),
        "effective_trained_scene_count": item.get("effective_trained_scene_count"),
        "fallback_rate": item.get("fallback_rate"),
        "difficulty_empty_rounds": item.get("difficulty_empty_rounds"),
        "empty_round_rate": item.get("empty_round_rate"),
        "accepted_event_ratio": distribution.get("accepted_event_ratio"),
        "anchor_ratio": distribution.get("anchor_ratio"),
        "replay_ratio": distribution.get("replay_ratio"),
        "random_ratio": distribution.get("random_ratio"),
        "hard_to_easy_converted_accepted_scenes": item.get(
            "hard_to_easy_converted_accepted_scenes"
        ),
        "forgotten_accepted_scenes": item.get("forgotten_accepted_scenes"),
        "selected_checkpoint_round": item.get("selected_checkpoint_round"),
        "hard_eval_v2_win_rate": item.get("hard_eval_v2_win_rate"),
        "hard_eval_v2_mean_return": item.get("hard_eval_v2_mean_return"),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    comparison = dict(summary.get("comparison") or {})
    findings = dict(summary.get("findings") or {})
    lines = [
        "FALCON Coverage-Aware Longer-Budget Pilot Report",
        "=" * 52,
        "",
        "Scope",
        f"- Seeds: {summary.get('seeds')}",
        "- Scheduler comparison: old single-YAML scheduler vs coverage-aware multi-scenario scheduler",
        "- Training budget: 40 rounds x 512 steps, held constant",
        "- Hard Eval v2: 40 scenarios x 3 episodes, frozen fixed opponent",
        "",
        "Per-seed results",
    ]
    for seed, item in (summary.get("seed_results") or {}).items():
        lines.extend(
            [
                f"- Seed {seed}: accepted coverage={_pct(item.get('accepted_coverage'))}, "
                f"duplicate rate={_pct(item.get('duplicate_rate'))}, "
                f"selected round={item.get('selected_checkpoint_round')}, "
                f"Hard Eval win rate={_fmt(item.get('hard_eval_v2_win_rate'))}.",
                f"  hard-to-easy={item.get('hard_to_easy_converted_accepted_scenes')}, "
                f"forgotten={item.get('forgotten_accepted_scenes')}, "
                f"fallback rate={_pct(item.get('fallback_rate'))}, "
                f"empty rounds={item.get('difficulty_empty_rounds')}.",
            ]
        )
    lines.extend(
        [
            "",
            "Old vs coverage-aware",
            f"- Accepted coverage: {_pct(comparison.get('old_accepted_coverage_mean'))} -> "
            f"{_pct(comparison.get('coverage_aware_accepted_coverage_mean'))}.",
            f"- Duplicate rate: {_pct(comparison.get('old_duplicate_rate_mean'))} -> "
            f"{_pct(comparison.get('coverage_aware_duplicate_rate_mean'))}.",
            f"- Selected rounds: {comparison.get('old_selected_rounds')} -> "
            f"{comparison.get('coverage_aware_selected_rounds')}.",
            f"- Hard Eval v2 win rate: {_fmt(comparison.get('old_hard_eval_v2_win_rate_mean'))} -> "
            f"{_fmt(comparison.get('coverage_aware_hard_eval_v2_win_rate_mean'))}.",
            f"- Forgotten accepted scenes: {comparison.get('old_forgotten_count_total')} -> "
            f"{comparison.get('coverage_aware_forgotten_count_total')}.",
            "",
            "Conclusions",
            f"- Coverage significantly improved: {findings.get('accepted_coverage_significantly_improved')}.",
            f"- Duplicate rate reduced: {findings.get('duplicate_rate_reduced')}.",
            f"- Forgetting reduced: {findings.get('forgetting_reduced')}.",
            f"- All seeds avoid round0: {findings.get('all_seeds_avoid_round0')}.",
            f"- Hard Eval v2 improved: {findings.get('hard_eval_v2_improved')}.",
            f"- Worth expanding to five seeds: {findings.get('worth_expanding_to_five_seeds')}.",
            f"- Ready for FSN data organization: {findings.get('ready_for_fsn_data_organization')}.",
            "",
            "Risks",
        ]
    )
    for warning in summary.get("warnings") or []:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _resolve(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT_DIR / path


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ratio(numerator: Any, denominator: Any) -> float:
    denominator_value = _float(denominator, 0.0) or 0.0
    if denominator_value <= 0:
        return 0.0
    return round((_float(numerator, 0.0) or 0.0) / denominator_value, 6)


def _mean(values: Iterable[Any]) -> float:
    numbers = [_float(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return round(statistics.mean(clean), 6) if clean else 0.0


def _mean_optional(values: Iterable[Any]) -> Optional[float]:
    numbers = [_float(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return round(statistics.mean(clean), 6) if clean else None


def _difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _fmt(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.4f}"


def _pct(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{100.0 * number:.2f}%"


if __name__ == "__main__":
    main()
