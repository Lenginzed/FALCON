"""Coverage-aware curriculum scheduling and sequential-training smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.curriculum_pool import CurriculumPool  # noqa: E402
from falcon.curriculum_scheduler import CurriculumScheduler  # noqa: E402
from falcon.falcon_controller import _train_mappo_smoke  # noqa: E402
from falcon.training_plan_adapter import MultiScenarioTrainingBridge, TrainingPlanAdapter  # noqa: E402


BASE_CONFIG = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
RESULTS_ROOT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "results_longer_budget"
COVERAGE_HISTORY = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "reports"
    / "accepted_scene_training_coverage.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke coverage-aware scheduling and multi-scenario MAPPO continuation.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--scenario-batch-size", type=int, default=6)
    parser.add_argument("--total-train-steps", type=int, default=24)
    parser.add_argument("--simulation-rounds", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "tests" / "tmp_falcon_coverage_aware_scheduler"),
    )
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    pool = _load_hydrated_pool(args.seed, warnings)
    checkpoint = _latest_checkpoint(args.seed)
    scheduler = _coverage_scheduler(
        seed=args.seed,
        scenario_batch_size=args.scenario_batch_size,
        total_train_steps=args.total_train_steps,
    )
    plan = scheduler.build_sampling_plan(
        pool,
        base_scenarios=[_base_scenario()],
        num_samples=args.scenario_batch_size,
        current_round=40,
        coverage_aware=True,
        scenario_batch_size=args.scenario_batch_size,
        total_train_steps_per_round=args.total_train_steps,
    )
    scheduler.save_sampling_plan(plan, output_dir / "coverage_aware_sampling_plan.json")

    before = pool.get_stats()
    duplicate_before = _duplicate_rate(pool)
    if args.skip_training:
        training_summary = {
            "schema_version": "falcon.multi_scenario_training_summary.v1",
            "scenario_batch_size": len(plan.get("scenario_batch") or []),
            "scenarios_actually_trained": 0,
            "checkpoint_saved": False,
            "latest_checkpoint_path": str(checkpoint) if checkpoint else None,
            "failure_stage": "training_skipped",
            "warnings": ["Training was skipped by --skip-training."],
        }
    elif checkpoint is None:
        training_summary = {
            "schema_version": "falcon.multi_scenario_training_summary.v1",
            "scenario_batch_size": len(plan.get("scenario_batch") or []),
            "scenarios_actually_trained": 0,
            "checkpoint_saved": False,
            "latest_checkpoint_path": None,
            "failure_stage": "checkpoint_missing",
            "warnings": [f"Latest seed-{args.seed} checkpoint was not found."],
        }
    else:
        bridge = MultiScenarioTrainingBridge(
            TrainingPlanAdapter(
                {
                    "seed": args.seed,
                    "lag_config_root": str(ROOT_DIR / "envs" / "JSBSim" / "configs"),
                }
            ),
            {
                "seed": args.seed,
                "default_per_scenario_train_steps": max(
                    args.total_train_steps // max(args.scenario_batch_size, 1),
                    1,
                ),
            },
        )
        training_summary = bridge.run_batch(
            plan,
            train_fn=_train_mappo_smoke,
            output_dir=output_dir / "multi_scenario_training",
            base_config_path=BASE_CONFIG,
            initial_checkpoint_path=checkpoint,
            round_id=40,
            curriculum_pool=pool,
        )
    _write_json(output_dir / "coverage_aware_training_summary.json", training_summary)
    pool.save(output_dir / "coverage_aware_curriculum_pool_after_smoke.json")

    after = pool.get_stats()
    duplicate_after = _duplicate_rate(pool)
    simulation = _simulate_schedulers(args.simulation_rounds)
    _write_json(output_dir / "scheduler_coverage_simulation.json", simulation)
    _write_simulation_csv(output_dir / "scheduler_coverage_simulation.csv", simulation)

    warnings.extend(plan.get("warnings") or [])
    warnings.extend(training_summary.get("warnings") or [])
    summary = {
        "schema_version": "falcon.coverage_aware_smoke_summary.v1",
        "accepted_total": before.get("accepted_items", 0),
        "accepted_trained_before": before.get("accepted_trained_items", 0),
        "accepted_trained_after": after.get("accepted_trained_items", 0),
        "accepted_coverage_before": before.get("accepted_training_coverage", 0.0),
        "accepted_coverage_after": after.get("accepted_training_coverage", 0.0),
        "duplicate_rate_before": duplicate_before,
        "duplicate_rate_after": duplicate_after,
        "scenario_batch_size": len(plan.get("scenario_batch") or []),
        "scenarios_actually_trained": training_summary.get("scenarios_actually_trained", 0),
        "checkpoint_saved": bool(training_summary.get("checkpoint_saved")),
        "checkpoint_continuity_complete": bool(
            training_summary.get("checkpoint_continuity_complete")
        ),
        "initial_checkpoint_path": str(checkpoint) if checkpoint else None,
        "latest_checkpoint_path": training_summary.get("latest_checkpoint_path"),
        "anchor_scenarios_used": training_summary.get("anchor_scenarios_used", []),
        "anchor_ratio": training_summary.get("anchor_ratio", 0.0),
        "accepted_ratio": training_summary.get("accepted_ratio", 0.0),
        "replay_ratio": training_summary.get("replay_ratio", 0.0),
        "failure_stage": (
            None
            if training_summary.get("checkpoint_saved") or args.skip_training
            else training_summary.get("failure_stage")
        ),
        "warnings": sorted(set(warnings)),
    }
    _write_json(output_dir / "coverage_aware_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _load_hydrated_pool(seed: int, warnings: list[str]) -> CurriculumPool:
    pool_path = (
        RESULTS_ROOT
        / "falcon_no_fsn"
        / f"seed_{seed}"
        / "pilot_run"
        / "controller"
        / "falcon_curriculum_pool_final.json"
    )
    pool = CurriculumPool({"trained_count_threshold": 3}).load(pool_path)
    history = _coverage_history(seed)
    if not history:
        warnings.append(f"No historical coverage rows were found for seed {seed}; pool starts unseen.")
        return pool
    by_yaml = {_path_key(row.get("yaml_path")): row for row in history}
    for item in pool.items:
        row = by_yaml.get(_path_key(item.get("scenario_yaml_path")))
        if row is None:
            continue
        train_count = _int(row.get("actual_training_count"))
        rounds = [_int(value) for value in str(row.get("actual_training_rounds") or "").split(",") if value.strip()]
        item["train_count"] = train_count
        item["first_trained_round"] = min(rounds) if rounds else None
        item["last_trained_round"] = max(rounds) if rounds else None
        item["cumulative_train_steps"] = train_count * 512
        item["coverage_status"] = (
            "unseen" if train_count == 0 else "trained" if train_count >= 3 else "undertrained"
        )
    return pool


def _coverage_history(seed: int) -> list[Dict[str, str]]:
    if not COVERAGE_HISTORY.exists():
        return []
    with COVERAGE_HISTORY.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if _int(row.get("seed")) == seed]


def _latest_checkpoint(seed: int) -> Path | None:
    registry_path = (
        RESULTS_ROOT
        / "falcon_no_fsn"
        / f"seed_{seed}"
        / "pilot_run"
        / "controller"
        / "falcon_checkpoint_registry.json"
    )
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    value = registry.get("latest_checkpoint") or registry.get("current_checkpoint")
    path = Path(str(value)) if value else None
    return path if path and path.exists() else None


def _coverage_scheduler(seed: int, scenario_batch_size: int, total_train_steps: int) -> CurriculumScheduler:
    base_quota = 2 if scenario_batch_size >= 6 else 1
    return CurriculumScheduler(
        {
            "seed": seed,
            "coverage_aware_enabled": True,
            "scenario_batch_size": scenario_batch_size,
            "total_train_steps_per_round": total_train_steps,
            "category_quota": {
                "accepted_llm": max(scenario_batch_size - base_quota, 1),
                "base_anchor": base_quota,
                "replay_failure": 0,
                "random_explore": 0,
            },
            "unseen_bonus": 2.0,
            "trained_count_threshold": 3,
        }
    )


def _simulate_schedulers(rounds: int) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for seed in (3, 4):
        for scheduler_name in ("legacy_single_yaml", "coverage_aware_batch"):
            warnings: list[str] = []
            pool = _load_hydrated_pool(seed, warnings)
            initial = pool.get_stats()
            future_events = 0
            future_unique: set[str] = set()
            event_sequence: list[str] = []
            within_batch_duplicate_events = 0
            within_batch_total_events = 0
            category_counts: Counter[str] = Counter()
            for offset in range(rounds):
                current_round = 40 + offset
                if scheduler_name == "legacy_single_yaml":
                    scheduler = CurriculumScheduler({"seed": seed + current_round})
                    plan = scheduler.build_sampling_plan(
                        pool,
                        base_scenarios=[_base_scenario()],
                        num_samples=6,
                    )
                    selected = next(
                        (
                            item
                            for item in plan.get("sampled_scenarios") or []
                            if item.get("pool_item_id")
                        ),
                        None,
                    )
                    batch = [selected] if selected else []
                else:
                    scheduler = _coverage_scheduler(seed, 6, 512)
                    plan = scheduler.build_sampling_plan(
                        pool,
                        base_scenarios=[_base_scenario()],
                        num_samples=6,
                        current_round=current_round,
                        coverage_aware=True,
                        scenario_batch_size=6,
                        total_train_steps_per_round=512,
                    )
                    batch = list(plan.get("scenario_batch") or [])
                batch_accepted_keys = [
                    _path_key(item.get("scenario_yaml_path"))
                    for item in batch
                    if item.get("pool_item_id")
                ]
                within_batch_total_events += len(batch_accepted_keys)
                within_batch_duplicate_events += max(
                    len(batch_accepted_keys) - len(set(batch_accepted_keys)),
                    0,
                )
                for item in batch:
                    category_counts[str(item.get("sampling_category") or "accepted_llm")] += 1
                    if not item.get("pool_item_id"):
                        continue
                    future_events += 1
                    scenario_key = _path_key(item.get("scenario_yaml_path"))
                    future_unique.add(scenario_key)
                    event_sequence.append(scenario_key)
                    pool.record_training(item, round_id=current_round, train_steps=int(item.get("assigned_train_steps", 512)))
            final = pool.get_stats()
            rows.append(
                {
                    "seed": seed,
                    "scheduler": scheduler_name,
                    "simulation_rounds": rounds,
                    "accepted_total": final.get("accepted_items", 0),
                    "accepted_trained_before": initial.get("accepted_trained_items", 0),
                    "accepted_trained_after": final.get("accepted_trained_items", 0),
                    "expected_accepted_coverage_after": final.get("accepted_training_coverage", 0.0),
                    "expected_unseen_accepted_after": final.get("accepted_unseen_items", 0),
                    "future_accepted_training_events": future_events,
                    "future_unique_accepted_scenes": len(future_unique),
                    "expected_future_duplicate_rate": round(
                        1.0 - len(future_unique) / future_events,
                        6,
                    )
                    if future_events
                    else 0.0,
                    "within_batch_duplicate_rate": round(
                        within_batch_duplicate_events / within_batch_total_events,
                        6,
                    )
                    if within_batch_total_events
                    else 0.0,
                    "equal_20_event_duplicate_rate": _prefix_duplicate_rate(event_sequence, 20),
                    "category_distribution": dict(sorted(category_counts.items())),
                    "warnings": warnings,
                }
            )
    return {
        "schema_version": "falcon.scheduler_coverage_simulation.v1",
        "rounds": rounds,
        "rows": rows,
    }


def _base_scenario() -> Dict[str, Any]:
    return {
        "scenario_id": "base_2v2_NoWeapon_Selfplay",
        "source": "original",
        "scenario_yaml_path": str(BASE_CONFIG),
        "priority_level": "base",
        "target_failure_modes": [],
        "sampling_weight": 1.0,
    }


def _duplicate_rate(pool: CurriculumPool) -> float:
    counts = [
        max(_int(item.get("train_count")), 0)
        for item in pool.get_accepted()
    ]
    events = sum(counts)
    unique = sum(1 for count in counts if count > 0)
    return round(1.0 - unique / events, 6) if events else 0.0


def _prefix_duplicate_rate(sequence: Iterable[str], limit: int) -> float:
    values = list(sequence)[: max(int(limit), 0)]
    return round(1.0 - len(set(values)) / len(values), 6) if values else 0.0


def _write_simulation_csv(path: Path, simulation: Mapping[str, Any]) -> None:
    rows = list(simulation.get("rows") or [])
    fieldnames = [
        "seed",
        "scheduler",
        "simulation_rounds",
        "accepted_total",
        "accepted_trained_before",
        "accepted_trained_after",
        "expected_accepted_coverage_after",
        "expected_unseen_accepted_after",
        "future_accepted_training_events",
        "future_unique_accepted_scenes",
        "expected_future_duplicate_rate",
        "within_batch_duplicate_rate",
        "equal_20_event_duplicate_rate",
        "category_distribution",
        "warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["category_distribution"] = json.dumps(output.get("category_distribution") or {}, sort_keys=True)
            output["warnings"] = " | ".join(output.get("warnings") or [])
            writer.writerow(output)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")


def _path_key(value: Any) -> str:
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve()).lower()
    except OSError:
        return str(value).lower()


def _int(value: Any) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0
    return int(parsed) if math.isfinite(parsed) else 0


if __name__ == "__main__":
    main()
