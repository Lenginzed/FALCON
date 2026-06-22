#!/usr/bin/env python
"""Select existing checkpoints with the independent failure-balanced proxy."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from falcon.checkpoint_selector import (  # noqa: E402
    FailureBalancedCheckpointSelector,
    write_per_scenario_csv,
)
from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402

EXPERIMENT_ROOT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
RESULTS_ROOT = EXPERIMENT_ROOT / "results_stability_aware_longer_budget"
REPORT_DIR = EXPERIMENT_ROOT / "reports"
PROXY_MANIFEST = EXPERIMENT_ROOT / "manifests" / "failure_balanced_validation_v1.json"
HARD_MANIFEST = EXPERIMENT_ROOT / "manifests" / "hard_eval_scenarios_v2.json"
ABLATION_SUMMARY = REPORT_DIR / "checkpoint_selector_ablation_summary.json"
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


def main() -> None:
    manifest = _load_json(PROXY_MANIFEST)
    if manifest.get("failure_stage") is not None:
        raise RuntimeError(
            f"Failure-balanced proxy manifest is not valid: {manifest.get('failure_stage')}"
        )
    if int(manifest.get("exact_hard_eval_v2_overlap_count") or 0) != 0:
        raise RuntimeError("Failure-balanced proxy overlaps Hard Eval v2.")
    ablation = _load_json(ABLATION_SUMMARY)
    selector = FailureBalancedCheckpointSelector()
    selections = {}
    all_rows = []
    for seed in (3, 4):
        rows = _evaluate_proxy_candidates(seed, ablation)
        weighted = selector.select(rows, mode="weighted")
        worst = selector.select(rows, mode="worst_group")
        current_anchor = _existing_selector(ablation, "current_anchor_selector", seed)
        terminal = _existing_selector(ablation, "terminal_selector", seed)
        selection = {
            "schema_version": "falcon.failure_balanced_proxy_selection.v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seed": seed,
            "proxy_manifest": str(PROXY_MANIFEST),
            "proxy_scenario_count": manifest.get("scenario_count"),
            "proxy_group_counts": manifest.get("scenario_group_counts"),
            "hard_eval_overlap_count": manifest.get(
                "exact_hard_eval_v2_overlap_count"
            ),
            "selection_weights": selector.weights,
            "selection_inputs": [
                "proxy initial_disadvantage_win",
                "proxy coordination_win",
                "proxy target_assignment_win",
                "proxy overall_validation_win",
                "accepted_scene_win_rate",
                "proxy mean_return",
                "source_round",
            ],
            "hard_eval_used_for_selection": False,
            "candidate_results": rows,
            "current_anchor_selector": current_anchor,
            "terminal_selector": terminal,
            "failure_balanced_proxy_selector": _selection_view(weighted),
            "worst_group_proxy_selector": _selection_view(worst),
            "selected_checkpoint": weighted.get("checkpoint_path") if weighted else None,
            "selected_candidate_id": weighted.get("candidate_id") if weighted else None,
            "selected_source_round": weighted.get("source_round") if weighted else None,
            "failure_stage": None if weighted else "no_proxy_selection",
            "warnings": [],
        }
        output_dir = (
            RESULTS_ROOT
            / "falcon_no_fsn"
            / f"seed_{seed}"
            / "eval_set"
            / "failure_balanced_proxy_selection"
        )
        output_path = output_dir / f"failure_balanced_proxy_selection_seed{seed}.json"
        _write_json(output_path, selection)
        selections[str(seed)] = selection
        for row in rows:
            all_rows.append(
                {
                    "seed": seed,
                    **row,
                    "selected_failure_balanced": bool(
                        weighted
                        and row.get("candidate_id") == weighted.get("candidate_id")
                    ),
                    "selected_worst_group": bool(
                        worst
                        and row.get("candidate_id") == worst.get("candidate_id")
                    ),
                }
            )
    selection_csv = REPORT_DIR / "failure_balanced_proxy_selection_results.csv"
    _write_csv(selection_csv, all_rows)

    # Hard Eval is deliberately executed only after proxy selection artifacts exist.
    hard_results = {
        seed: _evaluate_selected_on_hard_eval(int(seed), selection)
        for seed, selection in selections.items()
    }
    summary = _build_summary(manifest, selections, hard_results, ablation)
    _write_json(REPORT_DIR / "failure_balanced_proxy_selector_summary.json", summary)
    _write_csv(
        REPORT_DIR / "failure_balanced_proxy_selector.csv",
        _selector_comparison_rows(summary),
    )
    (
        REPORT_DIR / "failure_balanced_proxy_selector_report.txt"
    ).write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary["diagnosis"], indent=2, sort_keys=True))


def _evaluate_proxy_candidates(
    seed: int, ablation: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    candidate_info = dict(
        ((ablation.get("candidate_checkpoints") or {}).get(str(seed)) or {})
    )
    accepted_metrics = dict(
        ((ablation.get("candidate_metrics") or {}).get(str(seed)) or {})
    )
    evaluator = EvalSetEvaluator(
        PROXY_MANIFEST, {"base_config_path": str(BASE_CONFIG)}
    )
    seed_root = RESULTS_ROOT / "falcon_no_fsn" / f"seed_{seed}"
    rows = []
    for candidate_id, info in candidate_info.items():
        checkpoint = str(info.get("checkpoint_path"))
        output_path = (
            seed_root
            / "eval_set"
            / "failure_balanced_proxy_selection"
            / "candidates"
            / candidate_id
            / "proxy_eval_summary.json"
        )
        if output_path.exists():
            eval_summary = _load_json(output_path)
        else:
            eval_summary = evaluator.evaluate_checkpoint(
                checkpoint,
                episodes_per_scenario=1,
                seed=1400000 + seed * 10000 + len(rows) * 1000,
                group="falcon_no_fsn",
                checkpoint_role=f"failure_balanced_proxy_{candidate_id}",
                opponent_mode="fixed_checkpoint",
                opponent_checkpoint=FIXED_OPPONENT,
            )
            EvalSetEvaluator.save(eval_summary, output_path)
        aggregate = dict(eval_summary.get("aggregate_result") or {})
        groups = dict(eval_summary.get("eval_group_breakdown") or {})
        accepted = dict(accepted_metrics.get(candidate_id) or {})
        row = {
            "candidate_id": candidate_id,
            "checkpoint_path": checkpoint,
            "aliases": "|".join(info.get("aliases") or []),
            "source_round": info.get("source_round"),
            "initial_disadvantage_win": _group_win(
                groups, "initial_disadvantage_validation"
            ),
            "coordination_win": _group_win(
                groups, "coordination_stress_validation"
            ),
            "target_assignment_win": _group_win(
                groups, "target_assignment_validation"
            ),
            "replay_like_win": _group_win(groups, "replay_like_validation"),
            "overall_validation_win": aggregate.get("final_win_rate"),
            "proxy_mean_return": aggregate.get("final_mean_return"),
            "accepted_scene_win_rate": accepted.get("accepted_scene_win_rate"),
            "accepted_scene_mean_return": accepted.get("accepted_scene_mean_return"),
            "forgetting_rate": accepted.get("forgetting_rate"),
            "num_proxy_scenarios": eval_summary.get("num_scenarios_evaluated"),
            "same_actor": eval_summary.get("same_actor"),
            "same_checkpoint": eval_summary.get("same_checkpoint"),
            "opponent_mode": eval_summary.get("opponent_mode"),
            "failure_stage": eval_summary.get("failure_stage"),
            "proxy_eval_summary_path": str(output_path),
        }
        row.update(FailureBalancedCheckpointSelector().score(row))
        rows.append(row)
    return rows


def _evaluate_selected_on_hard_eval(
    seed: int, selection: Mapping[str, Any]
) -> Dict[str, Any]:
    output_dir = (
        RESULTS_ROOT
        / "falcon_no_fsn"
        / f"seed_{seed}"
        / "eval_set"
        / "failure_balanced_proxy_selected_hard_eval_v2"
    )
    output_path = output_dir / "hard_eval_v2_summary.json"
    per_scenario_path = output_dir / "hard_eval_v2_per_scenario.csv"
    if output_path.exists():
        summary = _load_json(output_path)
    else:
        evaluator = EvalSetEvaluator(
            HARD_MANIFEST, {"base_config_path": str(BASE_CONFIG)}
        )
        summary = evaluator.evaluate_checkpoint(
            selection["selected_checkpoint"],
            episodes_per_scenario=3,
            seed=1500000 + seed * 10000,
            group="falcon_no_fsn",
            checkpoint_role="failure_balanced_proxy_selected",
            opponent_mode="fixed_checkpoint",
            opponent_checkpoint=FIXED_OPPONENT,
        )
        summary["checkpoint_selection_source"] = (
            "failure_balanced_validation_v1"
        )
        summary["selection_artifact"] = str(
            RESULTS_ROOT
            / "falcon_no_fsn"
            / f"seed_{seed}"
            / "eval_set"
            / "failure_balanced_proxy_selection"
            / f"failure_balanced_proxy_selection_seed{seed}.json"
        )
        EvalSetEvaluator.save(summary, output_path)
    write_per_scenario_csv(summary, per_scenario_path)
    return summary


def _build_summary(
    manifest: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
    hard_results: Mapping[str, Mapping[str, Any]],
    ablation: Mapping[str, Any],
) -> Dict[str, Any]:
    proxy_wins = []
    current_wins = []
    oracle_wins = []
    seed_results = {}
    for seed in ("3", "4"):
        selection = selections[seed]
        hard = hard_results[seed]
        aggregate = dict(hard.get("aggregate_result") or {})
        current = dict(
            (((ablation.get("selector_results") or {}).get(
                "current_anchor_selector"
            ) or {}).get(seed) or {})
        )
        oracle = dict(
            (((ablation.get("selector_results") or {}).get(
                "failure_balanced_selector"
            ) or {}).get(seed) or {})
        )
        proxy_win = aggregate.get("final_win_rate")
        proxy_wins.append(proxy_win)
        current_wins.append(current.get("hard_eval_win_rate"))
        oracle_wins.append(oracle.get("hard_eval_win_rate"))
        seed_results[seed] = {
            "proxy_selected_candidate_id": selection.get("selected_candidate_id"),
            "proxy_selected_round": selection.get("selected_source_round"),
            "proxy_selected_checkpoint": selection.get("selected_checkpoint"),
            "proxy_selection": selection.get(
                "failure_balanced_proxy_selector"
            ),
            "worst_group_proxy_selection": selection.get(
                "worst_group_proxy_selector"
            ),
            "current_anchor_selection": current,
            "terminal_selection": selection.get("terminal_selector"),
            "oracle_selection": oracle,
            "hard_eval_v2": {
                "win_rate": proxy_win,
                "mean_return": aggregate.get("final_mean_return"),
                "group_breakdown": hard.get("eval_group_breakdown") or {},
                "same_actor": hard.get("same_actor"),
                "same_checkpoint": hard.get("same_checkpoint"),
                "opponent_mode": hard.get("opponent_mode"),
                "failure_stage": hard.get("failure_stage"),
            },
        }
    proxy_mean = _mean(proxy_wins)
    current_mean = _mean(current_wins)
    oracle_mean = _mean(oracle_wins)
    seed3_candidates = {
        str(item.get("candidate_id")): item
        for item in selections["3"].get("candidate_results") or []
    }
    seed4_candidates = {
        str(item.get("candidate_id")): item
        for item in selections["4"].get("candidate_results") or []
    }
    seed3_round17 = seed3_candidates.get("round17", {})
    seed3_terminal = seed3_candidates.get("terminal", {})
    seed3_current = seed_results["3"]["current_anchor_selection"]
    seed3_terminal_hard = seed_results["3"]["terminal_selection"]
    seed3_initial_rank_inversion = (
        float(seed3_round17.get("initial_disadvantage_win") or 0.0)
        > float(seed3_terminal.get("initial_disadvantage_win") or 0.0)
        and float(seed3_current.get("initial_disadvantage_win") or 0.0)
        < float(seed3_terminal_hard.get("initial_disadvantage_win") or 0.0)
    )
    seed4_full_score_count = sum(
        float(item.get("failure_balanced_score") or 0.0) >= 0.999999
        for item in seed4_candidates.values()
    )
    seed4_proxy_saturated = seed4_full_score_count >= max(
        2, len(seed4_candidates) - 1
    )
    construct_validity_supported = (
        not seed3_initial_rank_inversion
        and not seed4_proxy_saturated
        and proxy_mean >= current_mean
    )
    return {
        "schema_version": "falcon.failure_balanced_proxy_selector_summary.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "proxy_manifest": str(PROXY_MANIFEST),
        "proxy_manifest_audit": {
            "scenario_count": manifest.get("scenario_count"),
            "scenario_group_counts": manifest.get("scenario_group_counts"),
            "exact_hard_eval_v2_overlap_count": manifest.get(
                "exact_hard_eval_v2_overlap_count"
            ),
            "minimum_normalized_distance_to_hard_eval_v2": manifest.get(
                "minimum_normalized_distance_to_hard_eval_v2"
            ),
            "all_constraint_valid": all(
                item.get("constraint_valid")
                for item in manifest.get("scenarios") or []
            ),
            "all_env_load_reset_success": all(
                item.get("env_load_reset_success")
                for item in manifest.get("scenarios") or []
            ),
            "held_out_from_training": manifest.get("held_out_from_training"),
            "not_in_training_pool": manifest.get("not_in_training_pool"),
        },
        "selection_protocol": {
            "weights": FailureBalancedCheckpointSelector().weights,
            "tie_breakers": [
                "accepted_scene_win_rate",
                "proxy_mean_return",
                "later_round",
            ],
            "hard_eval_used_for_selection": False,
            "worst_group_score_reported": True,
        },
        "seed_results": seed_results,
        "aggregate": {
            "proxy_selected_hard_eval_win_rate_mean": proxy_mean,
            "proxy_selected_hard_eval_win_rate_std": _std(proxy_wins),
            "current_anchor_hard_eval_win_rate_mean": current_mean,
            "oracle_hard_eval_win_rate_mean": oracle_mean,
            "proxy_vs_current_delta": round(proxy_mean - current_mean, 6),
            "proxy_vs_oracle_delta": round(proxy_mean - oracle_mean, 6),
        },
        "construct_validity_audit": {
            "seed3_initial_disadvantage_proxy_rank_inversion": (
                seed3_initial_rank_inversion
            ),
            "seed3_proxy_initial_disadvantage": {
                "round17": seed3_round17.get("initial_disadvantage_win"),
                "terminal": seed3_terminal.get("initial_disadvantage_win"),
            },
            "seed3_hard_eval_initial_disadvantage": {
                "round17": seed3_current.get("initial_disadvantage_win"),
                "terminal": seed3_terminal_hard.get(
                    "initial_disadvantage_win"
                ),
            },
            "seed4_full_score_candidate_count": seed4_full_score_count,
            "seed4_candidate_count": len(seed4_candidates),
            "seed4_proxy_score_saturated": seed4_proxy_saturated,
            "proxy_construct_validity_supported": construct_validity_supported,
            "interpretation": (
                "The v1 proxy is independent from Hard Eval v2, but it does "
                "not preserve the checkpoint ranking needed for selection."
            ),
        },
        "diagnosis": {
            "seed3_selects_terminal_or_round39": (
                selections["3"].get("selected_source_round") == 39
                or "terminal"
                in (
                    selections["3"]
                    .get("failure_balanced_proxy_selector", {})
                    .get("aliases")
                    or ""
                )
            ),
            "seed4_selects_non_degraded_checkpoint": (
                seed_results["4"]["hard_eval_v2"]["win_rate"] >= 1.0
            ),
            "proxy_close_to_oracle_0_9875": abs(proxy_mean - 0.9875) <= 0.025,
            "proxy_better_than_current_anchor_0_925": proxy_mean > 0.925,
            "data_leakage_detected": False,
            "can_replace_current_protocol": (
                proxy_mean > current_mean
                and int(
                    manifest.get("exact_hard_eval_v2_overlap_count") or 0
                )
                == 0
                and construct_validity_supported
            ),
            "worth_integrating_into_csa_scheduler": (
                proxy_mean > current_mean and construct_validity_supported
            ),
            "formal_replacement_recommended": False,
            "need_new_training": False,
            "recommended_next_step": (
                "Redesign a frozen proxy v2 with more diverse independent "
                "initial-disadvantage geometries and less saturated scenarios, "
                "then re-evaluate the same existing checkpoints."
            ),
        },
        "warnings": [
            "This proxy was designed after observing selector weakness, so confirmation on additional existing seeds or a future frozen run is still desirable.",
            "Hard Eval v2 was used only after selection to measure generalization.",
            "Seed3 shows a proxy-to-test rank inversion on initial-disadvantage scenarios.",
            "Seed4 proxy scores saturate across nearly all non-round0 checkpoints, so the tie-breaker selects an early checkpoint.",
        ],
    }


def _existing_selector(
    ablation: Mapping[str, Any], selector: str, seed: int
) -> Dict[str, Any]:
    return dict(
        (((ablation.get("selector_results") or {}).get(selector) or {}).get(
            str(seed)
        ) or {})
    )


def _selection_view(selected: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not selected:
        return {}
    keys = (
        "candidate_id",
        "checkpoint_path",
        "aliases",
        "source_round",
        "failure_balanced_score",
        "worst_group_score",
        "initial_disadvantage_win",
        "coordination_win",
        "target_assignment_win",
        "replay_like_win",
        "overall_validation_win",
        "proxy_mean_return",
        "accepted_scene_win_rate",
        "forgetting_rate",
    )
    return {key: selected.get(key) for key in keys}


def _selector_comparison_rows(
    summary: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    rows = []
    for seed, result in (summary.get("seed_results") or {}).items():
        hard = dict(result.get("hard_eval_v2") or {})
        for selector, selected in (
            ("failure_balanced_proxy", result.get("proxy_selection") or {}),
            ("worst_group_proxy", result.get("worst_group_proxy_selection") or {}),
            ("current_anchor", result.get("current_anchor_selection") or {}),
            ("terminal", result.get("terminal_selection") or {}),
            ("failure_balanced_oracle", result.get("oracle_selection") or {}),
        ):
            rows.append(
                {
                    "seed": seed,
                    "selector": selector,
                    "selected_candidate_id": selected.get("candidate_id")
                    or selected.get("selected_candidate_id"),
                    "selected_round": selected.get("source_round")
                    or selected.get("selected_source_round"),
                    "failure_balanced_score": selected.get(
                        "failure_balanced_score"
                    ),
                    "worst_group_score": selected.get("worst_group_score"),
                    "hard_eval_win_rate": hard.get("win_rate")
                    if selector == "failure_balanced_proxy"
                    else selected.get("hard_eval_win_rate"),
                    "hard_eval_mean_return": hard.get("mean_return")
                    if selector == "failure_balanced_proxy"
                    else selected.get("hard_eval_mean_return"),
                    "test_informed": selector == "failure_balanced_oracle",
                }
            )
    return rows


def _render_report(summary: Mapping[str, Any]) -> str:
    aggregate = dict(summary.get("aggregate") or {})
    diagnosis = dict(summary.get("diagnosis") or {})
    audit = dict(summary.get("proxy_manifest_audit") or {})
    construct = dict(summary.get("construct_validity_audit") or {})
    lines = [
        "Failure-Balanced Validation Proxy Selector Report",
        "=" * 51,
        "",
        "Proxy manifest audit",
        f"- Scenarios: {audit.get('scenario_count')}",
        f"- Groups: {audit.get('scenario_group_counts')}",
        f"- Exact Hard Eval overlap: {audit.get('exact_hard_eval_v2_overlap_count')}",
        f"- Minimum normalized Hard Eval distance: {audit.get('minimum_normalized_distance_to_hard_eval_v2')}",
        f"- All constraint valid: {audit.get('all_constraint_valid')}",
        f"- All env load/reset successful: {audit.get('all_env_load_reset_success')}",
        "",
        "Per-seed selection",
    ]
    for seed, result in (summary.get("seed_results") or {}).items():
        proxy = result.get("proxy_selection") or {}
        hard = result.get("hard_eval_v2") or {}
        lines.extend(
            [
                f"- Seed {seed}: candidate={proxy.get('candidate_id')}, round={proxy.get('source_round')}, "
                f"proxy_score={proxy.get('failure_balanced_score')}, Hard Eval={hard.get('win_rate')}, "
                f"mean_return={hard.get('mean_return')}"
            ]
        )
    lines.extend(
        [
            "",
            "Aggregate",
            f"- Proxy-selected Hard Eval mean/std: {aggregate.get('proxy_selected_hard_eval_win_rate_mean')} / {aggregate.get('proxy_selected_hard_eval_win_rate_std')}",
            f"- Current anchor mean: {aggregate.get('current_anchor_hard_eval_win_rate_mean')}",
            f"- Oracle mean: {aggregate.get('oracle_hard_eval_win_rate_mean')}",
            f"- Proxy vs current delta: {aggregate.get('proxy_vs_current_delta')}",
            f"- Proxy vs oracle delta: {aggregate.get('proxy_vs_oracle_delta')}",
            "",
            "Construct-validity audit",
            f"- Seed3 initial-disadvantage rank inversion: {construct.get('seed3_initial_disadvantage_proxy_rank_inversion')}",
            f"- Seed3 proxy initial wins (round17/terminal): {construct.get('seed3_proxy_initial_disadvantage')}",
            f"- Seed3 Hard Eval initial wins (round17/terminal): {construct.get('seed3_hard_eval_initial_disadvantage')}",
            f"- Seed4 full-score candidates: {construct.get('seed4_full_score_candidate_count')} / {construct.get('seed4_candidate_count')}",
            f"- Seed4 proxy saturated: {construct.get('seed4_proxy_score_saturated')}",
            f"- Proxy construct validity supported: {construct.get('proxy_construct_validity_supported')}",
            "",
            "Judgement",
            f"- Seed3 selected terminal/round39: {diagnosis.get('seed3_selects_terminal_or_round39')}",
            f"- Seed4 selected non-degraded checkpoint: {diagnosis.get('seed4_selects_non_degraded_checkpoint')}",
            f"- Close to oracle 0.9875: {diagnosis.get('proxy_close_to_oracle_0_9875')}",
            f"- Better than current anchor 0.925: {diagnosis.get('proxy_better_than_current_anchor_0_925')}",
            f"- Data leakage detected: {diagnosis.get('data_leakage_detected')}",
            f"- Can replace current selection protocol: {diagnosis.get('can_replace_current_protocol')}",
            f"- Formal replacement recommended: {diagnosis.get('formal_replacement_recommended')}",
            f"- Worth integrating into CSA Scheduler: {diagnosis.get('worth_integrating_into_csa_scheduler')}",
            f"- Need new training: {diagnosis.get('need_new_training')}",
            f"- Recommended next step: {diagnosis.get('recommended_next_step')}",
            "",
            "Warnings",
        ]
    )
    lines.extend(f"- {item}" for item in summary.get("warnings") or [])
    return "\n".join(lines) + "\n"


def _group_win(groups: Mapping[str, Any], group: str) -> float:
    try:
        return float((groups.get(group) or {}).get("win_rate"))
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Sequence[Any]) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.mean(clean), 6) if clean else 0.0


def _std(values: Sequence[Any]) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0


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
