#!/usr/bin/env python
"""Evaluate and ablate checkpoint selectors on existing stability-aware runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402

EXPERIMENT_ROOT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon"
RESULTS_ROOT = EXPERIMENT_ROOT / "results_stability_aware_longer_budget"
REPORT_DIR = EXPERIMENT_ROOT / "reports"
ANCHOR_MANIFEST = EXPERIMENT_ROOT / "manifests" / "anchor_validation_scenarios_v1.json"
VALIDATION_MANIFEST = EXPERIMENT_ROOT / "manifests" / "eval_split_v1_validation.json"
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
REQUESTED_ROUNDS = (0, 10, 17, 20, 30, 35, 39)
SELECTOR_NAMES = (
    "current_anchor_selector",
    "terminal_selector",
    "batch_best_selector",
    "overall_validation_selector",
    "failure_balanced_selector",
    "worst_group_selector",
    "accepted_transfer_selector",
)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    seed_payloads = {str(seed): _evaluate_seed(seed) for seed in (3, 4)}
    selector_results = {
        selector: {
            str(seed): _select(selector, seed_payloads[str(seed)])
            for seed in (3, 4)
        }
        for selector in SELECTOR_NAMES
    }
    selector_aggregate = {
        selector: _aggregate_selector(results)
        for selector, results in selector_results.items()
    }
    best_deployable = _best_selector(
        selector_aggregate,
        allowed={
            "current_anchor_selector",
            "terminal_selector",
            "batch_best_selector",
            "overall_validation_selector",
        },
    )
    best_diagnostic = _best_selector(selector_aggregate, allowed=set(SELECTOR_NAMES))
    summary = {
        "schema_version": "falcon.checkpoint_selector_ablation.v1",
        "seeds": [3, 4],
        "evaluation_protocol": {
            "opponent_mode": "fixed_checkpoint",
            "opponent_checkpoint": str(FIXED_OPPONENT),
            "same_actor": False,
            "anchor_manifest": str(ANCHOR_MANIFEST),
            "anchor_episodes_per_scenario": 1,
            "validation_manifest": str(VALIDATION_MANIFEST),
            "validation_episodes_per_scenario": 1,
            "hard_eval_manifest": str(HARD_MANIFEST),
            "hard_eval_episodes_per_scenario": 3,
            "accepted_scene_episodes_per_scenario": 1,
        },
        "candidate_checkpoints": {
            seed: payload["candidate_checkpoints"]
            for seed, payload in seed_payloads.items()
        },
        "candidate_metrics": {
            seed: payload["candidate_metrics"] for seed, payload in seed_payloads.items()
        },
        "selector_results": selector_results,
        "selector_aggregate": selector_aggregate,
        "diagnosis": _diagnose(
            seed_payloads,
            selector_results,
            selector_aggregate,
            best_deployable,
            best_diagnostic,
        ),
        "warnings": [
            "failure_balanced_selector and worst_group_selector use Hard Eval v2 labels and are diagnostic oracles, not valid formal selectors.",
            "accepted_transfer_selector uses Hard Eval only as a final tie-breaker and therefore is also test-informed in this ablation.",
            "Only seeds 3 and 4 are included; selector stability estimates remain low-sample.",
        ],
    }
    json_path = REPORT_DIR / "checkpoint_selector_ablation_summary.json"
    csv_path = REPORT_DIR / "checkpoint_selector_ablation.csv"
    report_path = REPORT_DIR / "checkpoint_selector_ablation_report.txt"
    _write_json(json_path, summary)
    _write_csv(csv_path, _csv_rows(seed_payloads, selector_results))
    report_path.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary["diagnosis"], indent=2, sort_keys=True))


def _evaluate_seed(seed: int) -> Dict[str, Any]:
    seed_root = RESULTS_ROOT / "falcon_no_fsn" / f"seed_{seed}"
    pilot = _load_json(seed_root / "pilot_run" / "pilot_run_summary.json")
    registry = _load_json(
        seed_root / "pilot_run" / "controller" / "falcon_checkpoint_registry.json"
    )
    selection = _load_json(
        seed_root
        / "eval_set"
        / "validation_checkpoint_selection"
        / "validation_selected_checkpoint.json"
    )
    pool = _load_json(
        seed_root / "pilot_run" / "controller" / "falcon_curriculum_pool_final.json"
    )
    candidates = _collect_candidates(seed, pilot, registry, selection)
    diagnostics_dir = seed_root / "diagnostics" / "checkpoint_selector_ablation"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    accepted_manifest = diagnostics_dir / "accepted_curriculum_scenes.json"
    accepted = [
        item
        for item in pool.get("items") or []
        if item.get("accepted_into_curriculum_pool") and item.get("scenario_yaml_path")
    ]
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
    evaluators = {
        "anchor": EvalSetEvaluator(
            ANCHOR_MANIFEST, {"base_config_path": str(BASE_CONFIG)}
        ),
        "validation": EvalSetEvaluator(
            VALIDATION_MANIFEST, {"base_config_path": str(BASE_CONFIG)}
        ),
        "hard_eval": EvalSetEvaluator(
            HARD_MANIFEST, {"base_config_path": str(BASE_CONFIG)}
        ),
        "accepted": EvalSetEvaluator(
            accepted_manifest, {"base_config_path": str(BASE_CONFIG)}
        ),
    }
    episodes = {"anchor": 1, "validation": 1, "hard_eval": 3, "accepted": 1}
    candidate_metrics: Dict[str, Dict[str, Any]] = {}
    round0_accepted_map: Dict[str, Mapping[str, Any]] = {}
    for candidate_id, candidate in candidates.items():
        checkpoint = candidate["checkpoint_path"]
        eval_results = {}
        for set_name, evaluator in evaluators.items():
            output_path = diagnostics_dir / candidate_id / f"{set_name}_summary.json"
            eval_results[set_name] = _evaluate_or_load(
                evaluator,
                checkpoint,
                output_path,
                role=f"{candidate_id}_{set_name}",
                episodes=episodes[set_name],
                seed=1300000
                + seed * 10000
                + _stable_int(candidate_id) % 1000
                + {"anchor": 0, "validation": 1000, "hard_eval": 2000, "accepted": 3000}[set_name],
            )
        accepted_map = _scenario_map(eval_results["accepted"])
        if "round0" in candidate["aliases"]:
            round0_accepted_map = accepted_map
        candidate_metrics[candidate_id] = _metrics(candidate, eval_results)
        candidate_metrics[candidate_id]["_accepted_map"] = accepted_map
    for candidate_id, metrics in candidate_metrics.items():
        accepted_map = metrics.pop("_accepted_map")
        forgetting = _forgetting_metrics(round0_accepted_map, accepted_map)
        metrics.update(forgetting)
    forgetting_ids = [
        scenario_id
        for scenario_id, row in round0_accepted_map.items()
        if _float(row.get("win_rate"), 0.0) >= 0.8
    ]
    for metrics in candidate_metrics.values():
        metrics["accepted_forgetting_set_size"] = len(forgetting_ids)
    return {
        "seed": seed,
        "candidate_checkpoints": candidates,
        "candidate_metrics": candidate_metrics,
        "accepted_scene_count": len(accepted),
        "accepted_forgetting_set_size": len(forgetting_ids),
    }


def _collect_candidates(
    seed: int,
    pilot: Mapping[str, Any],
    registry: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    round_paths = _checkpoint_map(registry)
    last_round = list(pilot.get("round_summaries") or [])[-1]
    requested: list[tuple[str, Any]] = []
    for round_id in REQUESTED_ROUNDS:
        path = round_paths.get(round_id)
        if path:
            requested.append((f"round{round_id}", path))
    requested.extend(
        [
            ("batch_best", last_round.get("selected_checkpoint_path")),
            ("current_anchor_selected", last_round.get("selected_checkpoint_path")),
            ("validation_selected", selection.get("selected_checkpoint")),
            ("terminal", last_round.get("terminal_checkpoint_path")),
        ]
    )
    by_path: Dict[str, Dict[str, Any]] = {}
    for alias, raw_path in requested:
        if not raw_path:
            continue
        path = Path(str(raw_path)).resolve()
        if not path.exists():
            continue
        key = str(path)
        item = by_path.setdefault(
            key,
            {
                "checkpoint_path": key,
                "aliases": [],
                "source_round": _source_round(path),
                "sha256": _sha256(path),
            },
        )
        if alias not in item["aliases"]:
            item["aliases"].append(alias)
    output: Dict[str, Dict[str, Any]] = {}
    ordered = sorted(
        by_path.values(),
        key=lambda item: (
            -1 if "round0" in item["aliases"] else int(item.get("source_round") or 9999),
            item["checkpoint_path"],
        ),
    )
    for index, item in enumerate(ordered):
        preferred = next(
            (
                alias
                for alias in (
                    "round0",
                    "round10",
                    "round17",
                    "round20",
                    "round30",
                    "round35",
                    "round39",
                    "batch_best",
                    "validation_selected",
                    "terminal",
                )
                if alias in item["aliases"]
            ),
            f"checkpoint_{index:02d}",
        )
        candidate_id = preferred
        suffix = 1
        while candidate_id in output:
            suffix += 1
            candidate_id = f"{preferred}_{suffix}"
        item["candidate_id"] = candidate_id
        item["seed"] = seed
        output[candidate_id] = item
    return output


def _metrics(
    candidate: Mapping[str, Any], eval_results: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    anchor = _aggregate(eval_results["anchor"])
    validation = _aggregate(eval_results["validation"])
    hard = _aggregate(eval_results["hard_eval"])
    accepted = _aggregate(eval_results["accepted"])
    hard_groups = dict(eval_results["hard_eval"].get("eval_group_breakdown") or {})
    return {
        "candidate_id": candidate.get("candidate_id"),
        "checkpoint_path": candidate.get("checkpoint_path"),
        "aliases": list(candidate.get("aliases") or []),
        "source_round": candidate.get("source_round"),
        "sha256": candidate.get("sha256"),
        "anchor_win_rate": anchor["win_rate"],
        "anchor_mean_return": anchor["mean_return"],
        "validation_win_rate": validation["win_rate"],
        "validation_mean_return": validation["mean_return"],
        "hard_eval_win_rate": hard["win_rate"],
        "hard_eval_mean_return": hard["mean_return"],
        "initial_disadvantage_win": _group_win(
            hard_groups, "hard_random_initial_disadvantage"
        ),
        "coordination_win": _group_win(hard_groups, "hard_coordination_stress"),
        "target_assignment_win": _group_win(
            hard_groups, "hard_target_assignment_stress"
        ),
        "replay_failure_win": _group_win(hard_groups, "replay_failure_variants"),
        "accepted_scene_win_rate": accepted["win_rate"],
        "accepted_scene_mean_return": accepted["mean_return"],
        "same_actor": eval_results["hard_eval"].get("same_actor"),
        "opponent_mode": eval_results["hard_eval"].get("opponent_mode"),
        "failure_stage": _first_failure(eval_results.values()),
    }


def _forgetting_metrics(
    round0_map: Mapping[str, Mapping[str, Any]],
    candidate_map: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    eligible = [
        scenario_id
        for scenario_id, row in round0_map.items()
        if _float(row.get("win_rate"), 0.0) >= 0.8
    ]
    forgotten = 0
    wins = []
    for scenario_id in eligible:
        before = _float(round0_map[scenario_id].get("win_rate"))
        after = _float((candidate_map.get(scenario_id) or {}).get("win_rate"))
        if after is not None:
            wins.append(after)
        if before is not None and after is not None and after <= before - 0.2:
            forgotten += 1
    return {
        "forgotten_scene_count": forgotten,
        "forgetting_rate": _ratio(forgotten, len(eligible)),
        "forgetting_set_win_rate": _mean(wins),
        "accepted_transfer_score": round(
            _float(
                _aggregate_from_rows(candidate_map.values()).get("win_rate"), 0.0
            )
            - _ratio(forgotten, len(eligible)),
            6,
        ),
    }


def _select(selector: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = list((payload.get("candidate_metrics") or {}).values())
    if selector == "terminal_selector":
        selected = _by_alias(candidates, "terminal")
        score = None
        reason = "direct terminal/latest checkpoint"
    elif selector == "batch_best_selector":
        selected = _by_alias(candidates, "batch_best")
        score = None
        reason = "checkpoint preserved by the within-batch anchor selector"
    elif selector == "current_anchor_selector":
        selected = max(
            candidates,
            key=lambda item: (
                _score(item.get("anchor_win_rate")),
                _score(item.get("anchor_mean_return")),
                -int(item.get("source_round") or 0),
            ),
        )
        score = selected.get("anchor_win_rate")
        reason = "highest frozen anchor win rate, then mean return"
    elif selector == "overall_validation_selector":
        selected = max(
            candidates,
            key=lambda item: (
                _score(item.get("validation_win_rate")),
                _score(item.get("validation_mean_return")),
                -int(item.get("source_round") or 0),
            ),
        )
        score = selected.get("validation_win_rate")
        reason = "highest independent validation win rate, then mean return"
    elif selector == "failure_balanced_selector":
        for item in candidates:
            item["_failure_balanced_score"] = round(
                0.25 * _score(item.get("initial_disadvantage_win"))
                + 0.25 * _score(item.get("coordination_win"))
                + 0.25 * _score(item.get("target_assignment_win"))
                + 0.25 * _score(item.get("hard_eval_win_rate")),
                6,
            )
        selected = max(
            candidates,
            key=lambda item: (
                item["_failure_balanced_score"],
                _score(item.get("hard_eval_mean_return")),
            ),
        )
        score = selected["_failure_balanced_score"]
        reason = "diagnostic oracle using Hard Eval failure-group balance"
    elif selector == "worst_group_selector":
        for item in candidates:
            item["_worst_group_score"] = min(
                _score(item.get("initial_disadvantage_win")),
                _score(item.get("coordination_win")),
                _score(item.get("target_assignment_win")),
            )
        selected = max(
            candidates,
            key=lambda item: (
                item["_worst_group_score"],
                _score(item.get("hard_eval_win_rate")),
                _score(item.get("hard_eval_mean_return")),
            ),
        )
        score = selected["_worst_group_score"]
        reason = "diagnostic oracle maximizing the weakest Hard Eval failure group"
    elif selector == "accepted_transfer_selector":
        selected = max(
            candidates,
            key=lambda item: (
                _score(item.get("accepted_transfer_score")),
                _score(item.get("hard_eval_win_rate")),
                _score(item.get("hard_eval_mean_return")),
            ),
        )
        score = selected.get("accepted_transfer_score")
        reason = "accepted-scene win rate minus forgetting; Hard Eval used only for tie-break"
    else:
        raise ValueError(f"Unsupported selector: {selector}")
    return {
        "selector": selector,
        "seed": payload.get("seed"),
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_checkpoint": selected.get("checkpoint_path"),
        "selected_source_round": selected.get("source_round"),
        "selected_aliases": selected.get("aliases"),
        "selector_score": score,
        "selection_reason": reason,
        "hard_eval_win_rate": selected.get("hard_eval_win_rate"),
        "hard_eval_mean_return": selected.get("hard_eval_mean_return"),
        "initial_disadvantage_win": selected.get("initial_disadvantage_win"),
        "coordination_win": selected.get("coordination_win"),
        "target_assignment_win": selected.get("target_assignment_win"),
        "replay_failure_win": selected.get("replay_failure_win"),
        "anchor_win_rate": selected.get("anchor_win_rate"),
        "validation_win_rate": selected.get("validation_win_rate"),
        "accepted_scene_win_rate": selected.get("accepted_scene_win_rate"),
        "forgetting_rate": selected.get("forgetting_rate"),
        "test_informed": selector
        in {
            "failure_balanced_selector",
            "worst_group_selector",
            "accepted_transfer_selector",
        },
    }


def _aggregate_selector(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    wins = [item.get("hard_eval_win_rate") for item in results.values()]
    returns = [item.get("hard_eval_mean_return") for item in results.values()]
    return {
        "hard_eval_win_rate_mean": _mean(wins),
        "hard_eval_win_rate_std": _std(wins),
        "hard_eval_win_rate_min": min(_score(value) for value in wins),
        "hard_eval_mean_return_mean": _mean(returns),
        "selected_rounds": [
            item.get("selected_source_round") for item in results.values()
        ],
        "seed_results": dict(results),
    }


def _diagnose(
    payloads: Mapping[str, Mapping[str, Any]],
    selector_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    aggregate: Mapping[str, Mapping[str, Any]],
    best_deployable: str,
    best_diagnostic: str,
) -> Dict[str, Any]:
    seed3_terminal = selector_results["terminal_selector"]["3"]
    seed3_validation = selector_results["overall_validation_selector"]["3"]
    seed3_anchor = selector_results["current_anchor_selector"]["3"]
    replacement_mean = aggregate[best_deployable]["hard_eval_win_rate_mean"]
    return {
        "seed3_current_selector_failure": {
            "anchor_selected_round": seed3_anchor.get("selected_source_round"),
            "anchor_selected_hard_eval": seed3_anchor.get("hard_eval_win_rate"),
            "overall_validation_selected_round": seed3_validation.get(
                "selected_source_round"
            ),
            "overall_validation_selected_hard_eval": seed3_validation.get(
                "hard_eval_win_rate"
            ),
            "terminal_round": seed3_terminal.get("selected_source_round"),
            "terminal_hard_eval": seed3_terminal.get("hard_eval_win_rate"),
            "explanation": (
                "The frozen anchor and independent validation sets reward the round17 "
                "checkpoint, but that checkpoint is weak on the broader initial-disadvantage "
                "distribution. The terminal checkpoint generalizes better to Hard Eval even "
                "though it is not preferred by those small selection sets."
            ),
        },
        "terminal_assessment": {
            "seed3_terminal_is_better_on_hard_eval": (
                _score(seed3_terminal.get("hard_eval_win_rate"))
                > _score(seed3_validation.get("hard_eval_win_rate"))
            ),
            "seed3_terminal_anchor_win_rate": seed3_terminal.get("anchor_win_rate"),
            "seed3_terminal_validation_win_rate": seed3_terminal.get(
                "validation_win_rate"
            ),
            "seed3_terminal_accepted_scene_win_rate": seed3_terminal.get(
                "accepted_scene_win_rate"
            ),
            "conclusion": (
                "Terminal is not universally superior: its advantage is strongest on the "
                "broader Hard Eval distribution. It remains strong on accepted scenes, but "
                "the selector mismatch rather than simple terminal superiority is the key issue."
            ),
        },
        "best_deployable_selector": best_deployable,
        "best_diagnostic_selector": best_diagnostic,
        "failure_balanced_selects_seed3_high_performance_checkpoint": (
            selector_results["failure_balanced_selector"]["3"].get(
                "hard_eval_win_rate"
            )
            >= 0.95
        ),
        "recommended_protocol_replacement": (
            "Build a larger held-out failure-balanced validation proxy. Do not adopt "
            "terminal-only or Hard-Eval-informed selection as the formal protocol."
        ),
        "recomputed_seed34_mean_hard_eval": replacement_mean,
        "need_new_training": False,
        "need_selector_revision": True,
        "retrospective_best_non_test_selector": best_deployable,
        "formal_protocol_warning": (
            "Hard Eval based selectors are oracles and cannot be adopted directly. "
            "Use their selected-group pattern to construct a held-out failure-balanced "
            "validation set, then rerun selection on existing checkpoints."
        ),
    }


def _best_selector(
    aggregate: Mapping[str, Mapping[str, Any]], allowed: set[str]
) -> str:
    return max(
        allowed,
        key=lambda name: (
            _score(aggregate[name].get("hard_eval_win_rate_mean")),
            _score(aggregate[name].get("hard_eval_win_rate_min")),
            _score(aggregate[name].get("hard_eval_mean_return_mean")),
        ),
    )


def _csv_rows(
    payloads: Mapping[str, Mapping[str, Any]],
    selector_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[Dict[str, Any]]:
    rows = []
    for seed, payload in payloads.items():
        selected_by = {}
        for selector, per_seed in selector_results.items():
            candidate_id = per_seed[seed].get("selected_candidate_id")
            selected_by.setdefault(candidate_id, []).append(selector)
        for candidate_id, metrics in payload["candidate_metrics"].items():
            rows.append(
                {
                    "seed": seed,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if not isinstance(value, (dict, list))
                    },
                    "aliases": "|".join(metrics.get("aliases") or []),
                    "selected_by": "|".join(selected_by.get(candidate_id, [])),
                }
            )
    return rows


def _render_report(summary: Mapping[str, Any]) -> str:
    diagnosis = dict(summary.get("diagnosis") or {})
    aggregates = dict(summary.get("selector_aggregate") or {})
    lines = [
        "Checkpoint Selection Diagnosis & Selector Ablation",
        "=" * 52,
        "",
        "Selector outcomes",
    ]
    for selector in SELECTOR_NAMES:
        item = aggregates[selector]
        lines.append(
            f"- {selector}: mean Hard Eval={item.get('hard_eval_win_rate_mean')}, "
            f"std={item.get('hard_eval_win_rate_std')}, min={item.get('hard_eval_win_rate_min')}, "
            f"rounds={item.get('selected_rounds')}"
        )
    seed3 = diagnosis["seed3_current_selector_failure"]
    terminal = diagnosis["terminal_assessment"]
    lines.extend(
        [
            "",
            "Seed3 diagnosis",
            f"- Anchor selector: round={seed3.get('anchor_selected_round')}, Hard Eval={seed3.get('anchor_selected_hard_eval')}",
            f"- Overall validation selector: round={seed3.get('overall_validation_selected_round')}, Hard Eval={seed3.get('overall_validation_selected_hard_eval')}",
            f"- Terminal: round={seed3.get('terminal_round')}, Hard Eval={seed3.get('terminal_hard_eval')}",
            f"- Cause: {seed3.get('explanation')}",
            "",
            "Terminal assessment",
            f"- Better on seed3 Hard Eval: {terminal.get('seed3_terminal_is_better_on_hard_eval')}",
            f"- Anchor win rate: {terminal.get('seed3_terminal_anchor_win_rate')}",
            f"- Validation win rate: {terminal.get('seed3_terminal_validation_win_rate')}",
            f"- Accepted-scene win rate: {terminal.get('seed3_terminal_accepted_scene_win_rate')}",
            f"- Conclusion: {terminal.get('conclusion')}",
            "",
            "Recommendation",
            f"- Retrospective best non-test selector: {diagnosis.get('best_deployable_selector')}",
            f"- Best diagnostic oracle: {diagnosis.get('best_diagnostic_selector')}",
            f"- Failure-balanced finds seed3 high-performance checkpoint: {diagnosis.get('failure_balanced_selects_seed3_high_performance_checkpoint')}",
            f"- Recomputed seed3/4 mean Hard Eval: {diagnosis.get('recomputed_seed34_mean_hard_eval')}",
            f"- Recommended protocol replacement: {diagnosis.get('recommended_protocol_replacement')}",
            f"- Need new training: {diagnosis.get('need_new_training')}",
            f"- Need selector revision: {diagnosis.get('need_selector_revision')}",
            f"- Protocol note: {diagnosis.get('formal_protocol_warning')}",
            "",
            "Warnings",
        ]
    )
    lines.extend(f"- {warning}" for warning in summary.get("warnings") or [])
    return "\n".join(lines) + "\n"


def _evaluate_or_load(
    evaluator: EvalSetEvaluator,
    checkpoint: str,
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


def _aggregate(summary: Mapping[str, Any]) -> Dict[str, Any]:
    aggregate = dict(summary.get("aggregate_result") or {})
    return {
        "win_rate": aggregate.get("final_win_rate"),
        "mean_return": aggregate.get("final_mean_return"),
    }


def _aggregate_from_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [_float(row.get("win_rate")) for row in rows]
    return {"win_rate": _mean(value for value in values if value is not None)}


def _scenario_map(summary: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row.get("scenario_id")): row
        for row in summary.get("per_scenario_results") or []
    }


def _group_win(groups: Mapping[str, Any], group: str) -> float:
    return _float((groups.get(group) or {}).get("win_rate"), 0.0)


def _first_failure(results: Iterable[Mapping[str, Any]]) -> Any:
    return next(
        (result.get("failure_stage") for result in results if result.get("failure_stage")),
        None,
    )


def _by_alias(
    candidates: Sequence[Mapping[str, Any]], alias: str
) -> Mapping[str, Any]:
    return next(item for item in candidates if alias in (item.get("aliases") or []))


def _source_round(path: Path) -> int:
    match = re.search(r"round(\d+)_", str(path))
    return int(match.group(1)) if match else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _score(value: Any) -> float:
    parsed = _float(value)
    return parsed if parsed is not None else float("-inf")


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ratio(numerator: Any, denominator: Any) -> float:
    denominator_value = _float(denominator, 0.0) or 0.0
    return (
        round((_float(numerator, 0.0) or 0.0) / denominator_value, 6)
        if denominator_value
        else 0.0
    )


def _mean(values: Iterable[Any]) -> float:
    clean = [_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    return round(statistics.mean(clean), 6) if clean else 0.0


def _std(values: Iterable[Any]) -> float:
    clean = [_float(value) for value in values]
    clean = [value for value in clean if value is not None]
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
