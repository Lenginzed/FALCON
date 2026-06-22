#!/usr/bin/env python
"""Seed-4 stability-aware scheduling and checkpoint-preservation smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from falcon.curriculum_pool import CurriculumPool  # noqa: E402
from falcon.curriculum_scheduler import CurriculumScheduler  # noqa: E402
from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402
from falcon.falcon_controller import _train_mappo_smoke  # noqa: E402
from falcon.policy_evaluator import PolicyEvaluator  # noqa: E402
from falcon.training_plan_adapter import (  # noqa: E402
    MultiScenarioTrainingBridge,
    TrainingPlanAdapter,
)

BASE_CONFIG = (
    ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
)
COVERAGE_SEED4 = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "results_coverage_aware_longer_budget"
    / "falcon_no_fsn"
    / "seed_4"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scenario-batch-size", type=int, default=8)
    parser.add_argument("--total-train-steps", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "tests" / "tmp_falcon_stability_aware_seed4"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = CurriculumPool({"trained_count_threshold": 3}).load(
        COVERAGE_SEED4
        / "pilot_run"
        / "controller"
        / "falcon_curriculum_pool_final.json"
    )
    selection = _load_json(
        COVERAGE_SEED4
        / "eval_set"
        / "validation_checkpoint_selection"
        / "validation_selected_checkpoint.json"
    )
    current_checkpoint = Path(str(selection.get("selected_checkpoint") or ""))
    if not current_checkpoint.exists():
        raise FileNotFoundError(
            f"Coverage-aware seed4 selected checkpoint was not found: {current_checkpoint}"
        )

    evaluator = PolicyEvaluator(
        {
            "base_config_path": str(BASE_CONFIG),
            "opponent_mode": "fixed_checkpoint",
            "opponent_checkpoint": str(FIXED_OPPONENT),
        }
    )
    accepted_sample = sorted(
        pool.get_accepted(),
        key=lambda item: (
            int(item.get("train_count") or 0),
            float(item.get("final_value_score") or 0.0),
        ),
        reverse=True,
    )[:8]
    sample_manifest = {
        "schema_version": "falcon.eval_scenario_manifest.v1",
        "scenario_count": len(accepted_sample),
        "scenarios": [
            {
                "scenario_id": item.get("pool_item_id"),
                "scenario_group": "accepted_curriculum",
                "scenario_yaml_path": item.get("scenario_yaml_path"),
            }
            for item in accepted_sample
        ],
    }
    sample_manifest_path = output_dir / "accepted_sample_manifest.json"
    _write_json(sample_manifest_path, sample_manifest)
    sample_evaluator = EvalSetEvaluator(
        sample_manifest_path, {"base_config_path": str(BASE_CONFIG)}
    )
    before_pool = pool.get_stats()
    before_duplicate = _duplicate_rate(pool)
    before_anchor = _evaluate_base(evaluator, current_checkpoint, seed=810000)
    before_sample = sample_evaluator.evaluate_checkpoint(
        current_checkpoint,
        episodes_per_scenario=1,
        seed=820000,
        group="falcon_no_fsn",
        checkpoint_role="stability_smoke_input",
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=FIXED_OPPONENT,
    )
    EvalSetEvaluator.save(before_sample, output_dir / "accepted_sample_before.json")

    round_results = []
    for offset in range(max(args.rounds, 0)):
        round_id = 40 + offset
        scheduler = CurriculumScheduler(
            {
                "seed": 4000 + round_id,
                "coverage_aware_enabled": True,
                "stability_aware_enabled": True,
                "scenario_batch_size": args.scenario_batch_size,
                "total_train_steps_per_round": args.total_train_steps,
                "category_quota": {
                    "accepted_llm": 4,
                    "base_anchor": 2,
                    "replay_failure": 1,
                    "random_explore": 1,
                },
                "anchor_ratio_min": 0.5,
                "accepted_ratio_max": 0.5,
                "fallback_reallocate_to_anchor": True,
                "interleave_anchors": True,
                "unseen_bonus": 2.0,
                "trained_count_threshold": 3,
            }
        )
        plan = scheduler.build_sampling_plan(
            pool,
            base_scenarios=[_base_scenario()],
            current_round=round_id,
            coverage_aware=True,
            scenario_batch_size=args.scenario_batch_size,
            total_train_steps_per_round=args.total_train_steps,
        )
        scheduler.save_sampling_plan(
            plan, output_dir / f"stability_aware_sampling_plan_round{round_id}.json"
        )
        bridge = MultiScenarioTrainingBridge(
            TrainingPlanAdapter(
                {
                    "seed": 4,
                    "lag_config_root": str(ROOT_DIR / "envs" / "JSBSim" / "configs"),
                }
            ),
            {
                "seed": 4,
                "default_per_scenario_train_steps": max(
                    args.total_train_steps // max(args.scenario_batch_size, 1), 1
                ),
                "preserve_best_within_batch": True,
                "round_checkpoint_selection": "anchor_validation",
            },
        )
        validation_counter = {"value": 0}

        def validate_checkpoint(
            checkpoint_path: str, context: Mapping[str, Any]
        ) -> Dict[str, Any]:
            validation_counter["value"] += 1
            result = _evaluate_base(
                evaluator,
                Path(checkpoint_path),
                seed=830000 + round_id * 100 + validation_counter["value"],
            )
            return {
                "schema_version": "falcon.anchor_checkpoint_validation.v1",
                "win_rate": result.get("win_rate"),
                "mean_return": result.get("mean_return"),
                "failure_stage": result.get("failure_stage"),
                "warnings": result.get("warnings") or [],
            }

        training = bridge.run_batch(
            plan,
            train_fn=_train_mappo_smoke,
            output_dir=output_dir / f"round{round_id}_multi_scenario_training",
            base_config_path=BASE_CONFIG,
            initial_checkpoint_path=current_checkpoint,
            round_id=round_id,
            curriculum_pool=pool,
            checkpoint_validation_fn=validate_checkpoint,
        )
        _write_json(
            output_dir / f"stability_aware_training_summary_round{round_id}.json",
            training,
        )
        selected = Path(str(training.get("selected_checkpoint_path") or ""))
        terminal = Path(str(training.get("terminal_checkpoint_path") or ""))
        if selected.exists():
            current_checkpoint = selected
        round_results.append(
            {
                "round_id": round_id,
                "scenario_order": [
                    item.get("sampling_category")
                    for item in plan.get("scenario_batch") or []
                ],
                "anchor_ratio": training.get("anchor_ratio"),
                "accepted_ratio": training.get("accepted_ratio"),
                "scenarios_actually_trained": training.get(
                    "scenarios_actually_trained"
                ),
                "selected_batch_index": training.get("selected_batch_index"),
                "selected_checkpoint_path": str(selected) if selected.exists() else None,
                "terminal_checkpoint_path": str(terminal) if terminal.exists() else None,
                "selected_validation": training.get(
                    "selected_checkpoint_validation"
                ),
                "terminal_validation": _candidate_validation(
                    training, training.get("scenario_batch_size", 0) - 1
                ),
                "selected_differs_from_terminal": (
                    selected.exists()
                    and terminal.exists()
                    and selected.resolve() != terminal.resolve()
                ),
                "checkpoint_continuity_complete": training.get(
                    "checkpoint_continuity_complete"
                ),
            }
        )

    pool.save(output_dir / "stability_aware_curriculum_pool_after_smoke.json")
    after_pool = pool.get_stats()
    after_duplicate = _duplicate_rate(pool)
    after_anchor = _evaluate_base(evaluator, current_checkpoint, seed=840000)
    after_sample = sample_evaluator.evaluate_checkpoint(
        current_checkpoint,
        episodes_per_scenario=1,
        seed=850000,
        group="falcon_no_fsn",
        checkpoint_role="stability_smoke_selected",
        opponent_mode="fixed_checkpoint",
        opponent_checkpoint=FIXED_OPPONENT,
    )
    EvalSetEvaluator.save(after_sample, output_dir / "accepted_sample_after.json")
    before_rows = _scenario_map(before_sample)
    after_rows = _scenario_map(after_sample)
    forgotten = sum(
        float(after_rows.get(key, {}).get("win_rate") or 0.0)
        <= float(value.get("win_rate") or 0.0) - 0.2
        for key, value in before_rows.items()
        if key in after_rows
    )
    summary = {
        "schema_version": "falcon.stability_aware_smoke_summary.v1",
        "seed": 4,
        "rounds": args.rounds,
        "scenario_batch_size": args.scenario_batch_size,
        "total_train_steps_per_round": args.total_train_steps,
        "initial_checkpoint_path": str(selection.get("selected_checkpoint")),
        "final_selected_checkpoint_path": str(current_checkpoint),
        "accepted_total": before_pool.get("accepted_items"),
        "accepted_trained_before": before_pool.get("accepted_trained_items"),
        "accepted_trained_after": after_pool.get("accepted_trained_items"),
        "accepted_coverage_before": before_pool.get("accepted_training_coverage"),
        "accepted_coverage_after": after_pool.get("accepted_training_coverage"),
        "duplicate_rate_before": before_duplicate,
        "duplicate_rate_after": after_duplicate,
        "anchor_win_rate_before": before_anchor.get("win_rate"),
        "anchor_win_rate_after": after_anchor.get("win_rate"),
        "anchor_mean_return_before": before_anchor.get("mean_return"),
        "anchor_mean_return_after": after_anchor.get("mean_return"),
        "accepted_sample_win_rate_before": (
            before_sample.get("aggregate_result") or {}
        ).get("final_win_rate"),
        "accepted_sample_win_rate_after": (
            after_sample.get("aggregate_result") or {}
        ).get("final_win_rate"),
        "accepted_sample_forgotten_count": forgotten,
        "mean_anchor_ratio": _mean(
            item.get("anchor_ratio") for item in round_results
        ),
        "mean_accepted_ratio": _mean(
            item.get("accepted_ratio") for item in round_results
        ),
        "rounds_preserved_nonterminal_checkpoint": sum(
            bool(item.get("selected_differs_from_terminal")) for item in round_results
        ),
        "checkpoint_continuity_complete": all(
            bool(item.get("checkpoint_continuity_complete"))
            for item in round_results
        ),
        "checkpoint_saved": current_checkpoint.exists(),
        "round_results": round_results,
        "failure_stage": None if current_checkpoint.exists() else "checkpoint_selection",
        "warnings": [],
    }
    _write_json(output_dir / "stability_aware_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _evaluate_base(
    evaluator: PolicyEvaluator, checkpoint: Path, seed: int
) -> Dict[str, Any]:
    return evaluator.evaluate_policy_on_scenario(
        checkpoint, BASE_CONFIG, num_episodes=1, seed=seed
    )


def _candidate_validation(
    training: Mapping[str, Any], batch_index: int
) -> Mapping[str, Any] | None:
    for candidate in training.get("checkpoint_candidates") or []:
        if candidate.get("batch_index") == batch_index:
            return candidate.get("validation")
    return None


def _scenario_map(summary: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(item.get("scenario_id")): item
        for item in summary.get("per_scenario_results") or []
    }


def _base_scenario() -> Dict[str, Any]:
    return {
        "scenario_id": "base_2v2_NoWeapon_Selfplay",
        "source": "original",
        "scenario_yaml_path": str(BASE_CONFIG),
        "priority_level": "base",
        "target_failure_modes": [],
    }


def _duplicate_rate(pool: CurriculumPool) -> float:
    counts = [
        max(int(item.get("train_count") or 0), 0) for item in pool.get_accepted()
    ]
    events = sum(counts)
    trained = sum(count > 0 for count in counts)
    return round(1.0 - trained / events, 6) if events else 0.0


def _mean(values: Sequence[Any] | Any) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 6) if clean else 0.0


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
