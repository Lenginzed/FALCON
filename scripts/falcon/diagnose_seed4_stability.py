#!/usr/bin/env python
"""Diagnose within-batch forgetting in the coverage-aware seed-4 run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from falcon.eval_set_evaluator import EvalSetEvaluator  # noqa: E402

DEFAULT_RUN_DIR = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "results_coverage_aware_longer_budget"
    / "falcon_no_fsn"
    / "seed_4"
    / "pilot_run"
)
DEFAULT_VALIDATION_MANIFEST = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "manifests"
    / "eval_split_v1_validation.json"
)
DEFAULT_OPPONENT = (
    ROOT_DIR
    / "experiments"
    / "falcon_2v2_noweapon"
    / "opponents"
    / "fixed_baseline_opponent"
    / "candidate_seed999_steps2048"
    / "checkpoints"
    / "actor_seed999_steps2048.pt"
)
DEFAULT_REPORT_DIR = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--rounds", nargs="+", type=int, default=[20, 39])
    parser.add_argument("--validation-manifest", default=str(DEFAULT_VALIDATION_MANIFEST))
    parser.add_argument("--opponent-checkpoint", default=str(DEFAULT_OPPONENT))
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    controller_dir = run_dir / "controller"
    diagnostics_dir = run_dir.parent / "diagnostics" / "seed4_forgetting"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = []
    for path in sorted(
        controller_dir.glob("falcon_controller_training_round*_summary.json"),
        key=_round_from_path,
    ):
        data = _load_json(path)
        multi = dict(data.get("multi_scenario_training_summary") or {})
        results = list(multi.get("training_results") or [])
        if not results:
            continue
        categories = [str(item.get("sampling_category") or "unknown") for item in results]
        sources = [str(item.get("source") or "unknown") for item in results]
        all_rounds.append(
            {
                "round_id": _round_from_path(path),
                "scenario_batch_size": len(results),
                "scenario_order": [
                    {
                        "batch_index": item.get("batch_index"),
                        "scenario_id": item.get("scenario_id"),
                        "scenario_yaml_path": item.get("scenario_yaml_path"),
                        "sampling_category": item.get("sampling_category"),
                        "anchor_role": item.get("anchor_role"),
                        "pool_item_id": item.get("pool_item_id"),
                        "train_steps": item.get("train_steps"),
                        "checkpoint": item.get("output_checkpoint_path"),
                    }
                    for item in results
                ],
                "category_order": categories,
                "source_order": sources,
                "base_anchor_positions": [
                    idx for idx, value in enumerate(categories) if value == "base_anchor"
                ],
                "accepted_positions": [
                    idx for idx, value in enumerate(categories) if value == "accepted_llm"
                ],
                "last_two_accepted_count": sum(
                    value == "accepted_llm" for value in categories[-2:]
                ),
                "anchors_all_before_accepted": _anchors_before_accepted(categories),
                "anchor_ratio": multi.get("anchor_ratio"),
                "accepted_ratio": multi.get("accepted_ratio"),
                "checkpoint_continuity_complete": multi.get(
                    "checkpoint_continuity_complete"
                ),
            }
        )

    plan_stats = _plan_redistribution_stats(controller_dir)
    validation_curve = _load_validation_curve(run_dir.parent)
    evaluator = EvalSetEvaluator(
        args.validation_manifest,
        {
            "base_config_path": str(
                ROOT_DIR
                / "envs"
                / "JSBSim"
                / "configs"
                / "2v2"
                / "NoWeapon"
                / "Selfplay.yaml"
            )
        },
    )
    intermediate_evaluations = {}
    for round_id in args.rounds:
        training_path = (
            controller_dir / f"falcon_controller_training_round{round_id}_summary.json"
        )
        training = _load_json(training_path)
        results = list(
            (training.get("multi_scenario_training_summary") or {}).get(
                "training_results"
            )
            or []
        )
        rows = []
        for item in results:
            checkpoint = item.get("output_checkpoint_path")
            if not checkpoint or not Path(str(checkpoint)).exists():
                continue
            output_path = (
                diagnostics_dir
                / f"round{round_id:02d}_batch{int(item.get('batch_index') or 0):02d}_validation.json"
            )
            if output_path.exists() and not args.force_eval:
                result = _load_json(output_path)
            else:
                result = evaluator.evaluate_checkpoint(
                    checkpoint,
                    episodes_per_scenario=args.episodes_per_scenario,
                    seed=4000 + round_id * 100 + int(item.get("batch_index") or 0),
                    group="falcon_no_fsn",
                    checkpoint_role=f"seed4_round{round_id}_batch{item.get('batch_index')}",
                    opponent_mode="fixed_checkpoint",
                    opponent_checkpoint=args.opponent_checkpoint,
                )
                EvalSetEvaluator.save(result, output_path)
            aggregate = dict(result.get("aggregate_result") or {})
            rows.append(
                {
                    "round_id": round_id,
                    "batch_index": item.get("batch_index"),
                    "scenario_id": item.get("scenario_id"),
                    "scenario_yaml_path": item.get("scenario_yaml_path"),
                    "sampling_category": item.get("sampling_category"),
                    "anchor_role": item.get("anchor_role"),
                    "validation_win_rate": aggregate.get("final_win_rate"),
                    "validation_mean_return": aggregate.get("final_mean_return"),
                    "checkpoint": checkpoint,
                    "failure_stage": result.get("failure_stage"),
                }
            )
        intermediate_evaluations[str(round_id)] = {
            "rows": rows,
            "best_batch_index": _best_batch_index(rows),
            "terminal_minus_best_win_rate": _terminal_minus_best(rows, "validation_win_rate"),
            "terminal_minus_best_mean_return": _terminal_minus_best(
                rows, "validation_mean_return"
            ),
            "accepted_step_mean_delta": _mean_step_delta(
                rows, category="accepted_llm"
            ),
            "anchor_step_mean_delta": _mean_step_delta(rows, category="base_anchor"),
        }

    summary = {
        "schema_version": "falcon.seed4_forgetting_diagnosis.v1",
        "run_dir": str(run_dir),
        "round_count": len(all_rounds),
        "batch_order": {
            "rounds_with_anchors_all_before_accepted": sum(
                bool(row["anchors_all_before_accepted"]) for row in all_rounds
            ),
            "rounds_with_last_two_both_accepted": sum(
                row["last_two_accepted_count"] == 2 for row in all_rounds
            ),
            "mean_anchor_ratio": _mean(row.get("anchor_ratio") for row in all_rounds),
            "mean_accepted_ratio": _mean(
                row.get("accepted_ratio") for row in all_rounds
            ),
            "base_anchor_mean_position": _mean(
                pos for row in all_rounds for pos in row["base_anchor_positions"]
            ),
            "accepted_mean_position": _mean(
                pos for row in all_rounds for pos in row["accepted_positions"]
            ),
        },
        "quota_redistribution": plan_stats,
        "sparse_validation_curve": validation_curve,
        "intermediate_checkpoint_validation": intermediate_evaluations,
        "per_round_batches": all_rounds,
    }
    findings = _findings(summary)
    summary["findings"] = findings
    summary_path = diagnostics_dir / "seed4_forgetting_diagnosis.json"
    report_path = diagnostics_dir / "seed4_forgetting_diagnosis_report.txt"
    _write_json(summary_path, summary)
    rendered_report = _render_report(summary)
    report_path.write_text(rendered_report, encoding="utf-8")
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(DEFAULT_REPORT_DIR / "seed4_forgetting_diagnosis.json", summary)
    (DEFAULT_REPORT_DIR / "seed4_forgetting_diagnosis_report.txt").write_text(
        rendered_report, encoding="utf-8"
    )
    _write_batch_csv(diagnostics_dir / "seed4_batch_order.csv", all_rounds)
    print(json.dumps(summary, indent=2))


def _plan_redistribution_stats(controller_dir: Path) -> Dict[str, Any]:
    requested: Counter[str] = Counter()
    resolved: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    plans = 0
    for path in controller_dir.glob("falcon_controller_sampling_plan_round*.json"):
        plan = _load_json(path)
        if not plan.get("coverage_aware"):
            continue
        plans += 1
        requested.update(
            {key: int(value or 0) for key, value in (plan.get("category_quota") or {}).items()}
        )
        resolved.update(
            {
                key: int(value or 0)
                for key, value in (plan.get("resolved_category_quota") or {}).items()
            }
        )
        for warning in plan.get("warnings") or []:
            if "was empty" in warning:
                warnings[str(warning)] += 1
    return {
        "plans": plans,
        "requested_total": dict(requested),
        "resolved_total": dict(resolved),
        "empty_category_warning_counts": dict(warnings),
        "accepted_reallocation_delta": resolved.get("accepted_llm", 0)
        - requested.get("accepted_llm", 0),
        "anchor_reallocation_delta": resolved.get("base_anchor", 0)
        - requested.get("base_anchor", 0),
        "replay_actual_total": resolved.get("replay_failure", 0),
        "random_actual_total": resolved.get("random_explore", 0),
    }


def _load_validation_curve(seed_dir: Path) -> list[Dict[str, Any]]:
    path = (
        seed_dir
        / "eval_set"
        / "validation_checkpoint_selection"
        / "checkpoint_validation_results.csv"
    )
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "round_id": _int(row.get("round_id")),
            "validation_win_rate": _float(row.get("validation_win_rate")),
            "validation_mean_return": _float(row.get("validation_mean_return")),
            "selected": str(row.get("selected")).lower() == "true",
        }
        for row in rows
    ]


def _best_batch_index(rows: Sequence[Mapping[str, Any]]) -> Any:
    if not rows:
        return None
    best = max(
        rows,
        key=lambda row: (
            _float(row.get("validation_win_rate"), -1.0),
            _float(row.get("validation_mean_return"), float("-inf")),
            -_int(row.get("batch_index")),
        ),
    )
    return best.get("batch_index")


def _terminal_minus_best(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    if not rows:
        return None
    values = [_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(clean[-1] - max(clean), 6)


def _mean_step_delta(rows: Sequence[Mapping[str, Any]], category: str) -> Any:
    deltas = []
    for index, row in enumerate(rows):
        if index == 0 or row.get("sampling_category") != category:
            continue
        current = _float(row.get("validation_win_rate"))
        previous = _float(rows[index - 1].get("validation_win_rate"))
        if current is not None and previous is not None:
            deltas.append(current - previous)
    return round(statistics.mean(deltas), 6) if deltas else None


def _anchors_before_accepted(categories: Sequence[str]) -> bool:
    anchor_positions = [i for i, value in enumerate(categories) if value == "base_anchor"]
    accepted_positions = [
        i for i, value in enumerate(categories) if value == "accepted_llm"
    ]
    return bool(anchor_positions and accepted_positions) and max(anchor_positions) < min(
        accepted_positions
    )


def _findings(summary: Mapping[str, Any]) -> Dict[str, Any]:
    order = dict(summary.get("batch_order") or {})
    redistribution = dict(summary.get("quota_redistribution") or {})
    intermediate = dict(summary.get("intermediate_checkpoint_validation") or {})
    terminal_regressions = [
        _float(item.get("terminal_minus_best_win_rate"), 0.0)
        for item in intermediate.values()
    ]
    accepted_deltas = [
        _float(item.get("accepted_step_mean_delta"), 0.0)
        for item in intermediate.values()
    ]
    anchor_deltas = [
        _float(item.get("anchor_step_mean_delta"), 0.0)
        for item in intermediate.values()
    ]
    round_count = max(_int(summary.get("round_count")), 1)
    return {
        "anchors_front_loaded": _int(
            order.get("rounds_with_anchors_all_before_accepted")
        )
        >= round_count * 0.8,
        "rounds_usually_end_with_accepted": _int(
            order.get("rounds_with_last_two_both_accepted")
        )
        >= round_count * 0.5,
        "missing_quota_overallocated_to_accepted": _int(
            redistribution.get("accepted_reallocation_delta")
        )
        > _int(redistribution.get("anchor_reallocation_delta")),
        "base_anchor_was_actually_trained": _float(
            order.get("mean_anchor_ratio"), 0.0
        )
        > 0.0,
        "within_batch_terminal_regression_detected": any(
            value is not None and value < 0.0 for value in terminal_regressions
        ),
        "accepted_steps_more_destabilizing_than_anchors": (
            _mean(accepted_deltas) < _mean(anchor_deltas)
        ),
        "catastrophic_forgetting_risk_supported": bool(
            any(value is not None and value < 0.0 for value in terminal_regressions)
            or _mean(accepted_deltas) < 0.0
        ),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    order = dict(summary.get("batch_order") or {})
    quota = dict(summary.get("quota_redistribution") or {})
    findings = dict(summary.get("findings") or {})
    lines = [
        "Seed 4 Coverage-Aware Forgetting Diagnosis",
        "=" * 48,
        "",
        f"Analyzed rounds: {summary.get('round_count')}",
        f"Mean anchor ratio: {_pct(order.get('mean_anchor_ratio'))}",
        f"Mean accepted ratio: {_pct(order.get('mean_accepted_ratio'))}",
        f"Rounds with all anchors before accepted: {order.get('rounds_with_anchors_all_before_accepted')}",
        f"Rounds ending with two accepted scenes: {order.get('rounds_with_last_two_both_accepted')}",
        "",
        "Quota redistribution",
        f"- Requested: {quota.get('requested_total')}",
        f"- Resolved: {quota.get('resolved_total')}",
        f"- Accepted reallocation delta: {quota.get('accepted_reallocation_delta')}",
        f"- Anchor reallocation delta: {quota.get('anchor_reallocation_delta')}",
        f"- Replay/random actual totals: {quota.get('replay_actual_total')}/{quota.get('random_actual_total')}",
        "",
        "Intermediate checkpoint validation",
    ]
    for round_id, item in (summary.get("intermediate_checkpoint_validation") or {}).items():
        lines.append(
            f"- Round {round_id}: best batch index={item.get('best_batch_index')}, "
            f"terminal-best win-rate delta={item.get('terminal_minus_best_win_rate')}, "
            f"accepted-step mean delta={item.get('accepted_step_mean_delta')}, "
            f"anchor-step mean delta={item.get('anchor_step_mean_delta')}."
        )
    lines.extend(
        [
            "",
            "Findings",
            f"- Anchors are front-loaded: {findings.get('anchors_front_loaded')}.",
            f"- Rounds usually end with accepted scenes: {findings.get('rounds_usually_end_with_accepted')}.",
            f"- Missing quota is overallocated to accepted scenes: {findings.get('missing_quota_overallocated_to_accepted')}.",
            f"- Base/historical anchors are actually trained: {findings.get('base_anchor_was_actually_trained')}.",
            f"- Within-batch terminal regression detected: {findings.get('within_batch_terminal_regression_detected')}.",
            f"- Catastrophic-forgetting risk is supported: {findings.get('catastrophic_forgetting_risk_supported')}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_batch_csv(path: Path, rounds: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for round_data in rounds:
        for item in round_data.get("scenario_order") or []:
            rows.append({"round_id": round_data.get("round_id"), **item})
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _round_from_path(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits or 0)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")


def _float(value: Any, default: Any = None) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[Any] | Any) -> float:
    clean = []
    for value in values:
        number = _float(value)
        if number is not None:
            clean.append(number)
    return round(statistics.mean(clean), 6) if clean else 0.0


def _pct(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number * 100.0:.2f}%"


if __name__ == "__main__":
    main()
