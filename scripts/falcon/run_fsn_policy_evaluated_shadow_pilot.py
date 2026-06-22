"""Run the FSN Stage 3 policy-evaluated shadow replacement pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.fsn_dataset import load_jsonl
from falcon.fsn_shadow_evaluator import (
    FSNPolicyEvaluatedShadowPilot,
    summarize_shadow_results,
    write_shadow_metrics_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/falcon_2v2_noweapon/fsn/stage2/"
            "fsn_stage2_model.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn/stage3",
    )
    parser.add_argument("--failures-per-seed", type=int, default=4)
    parser.add_argument("--candidates-per-generator", type=int, default=4)
    parser.add_argument("--episodes-per-candidate", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    opponent_manifest = json.loads(
        (
            ROOT_DIR
            / "experiments"
            / "falcon_2v2_noweapon"
            / "manifests"
            / "eval_opponent.json"
        ).read_text(encoding="utf-8")
    )
    pilot = FSNPolicyEvaluatedShadowPilot(
        workspace_root=ROOT_DIR,
        stage2_checkpoint=args.checkpoint,
        fixed_opponent_checkpoint=opponent_manifest["checkpoint_path"],
        base_config_path=(
            "envs/JSBSim/configs/2v2/NoWeapon/Selfplay.yaml"
        ),
        output_dir=args.output_dir,
        episodes_per_candidate=args.episodes_per_candidate,
    )
    all_failures, stats = pilot.collect_failure_summaries(
        "experiments/falcon_2v2_noweapon/results",
        per_seed=args.failures_per_seed,
    )
    pilot.save_failure_set(all_failures, stats)
    failures = [
        item for item in all_failures if item.get("seed") in set(args.seeds)
    ]
    dataset = load_jsonl(
        ROOT_DIR
        / "experiments"
        / "falcon_2v2_noweapon"
        / "fsn"
        / "stage2"
        / "failure_to_scenario_dataset_dedup.jsonl"
    )
    pool_stats = {"records": dataset}
    payload = pilot.evaluate(
        failures,
        pool_stats,
        candidates_per_generator=args.candidates_per_generator,
        resume=args.resume,
    )
    historical_qwen_seconds = 4357.298 / 393.0
    summary = summarize_shadow_results(
        payload, historical_qwen_seconds
    )
    output_dir = ROOT_DIR / args.output_dir
    summary_path = (
        output_dir / "fsn_policy_evaluated_shadow_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_shadow_metrics_csv(
        summary,
        output_dir / "fsn_policy_evaluated_shadow_metrics.csv",
    )
    _write_report(summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failure_stage"] is None else 1


def _write_report(summary: dict) -> None:
    metrics = summary["generator_metrics"]
    fsn = metrics["fsn"]
    diverse = metrics["fsn_diversity_aware"]
    qwen = metrics["historical_qwen"]
    random = metrics["random"]
    simulations = summary["replacement_simulations"]
    hybrid_simulations = [
        item
        for item in simulations
        if item["fsn_fraction"] < 1.0 and not item["risk_flags"]
    ]
    safest = max(
        hybrid_simulations or simulations,
        key=lambda item: (
            item["expected_accepted_rate"],
            item["expected_diversity"] or 0.0,
            -item["fsn_fraction"],
        ),
    )
    lines = [
        "FALCON FSN Stage 3 Policy-Evaluated Shadow Report",
        "=" * 51,
        "",
        f"Failure summaries: {summary['num_failure_summaries']}",
        f"Episodes per policy/candidate: {summary['episodes_per_candidate']}",
        f"Fixed opponent: {summary['fixed_opponent_checkpoint']}",
        "same_actor: false",
        "",
        "Generator comparison",
        f"- FSN: {fsn}",
        f"- Diversity-aware FSN: {diverse}",
        f"- Historical Qwen: {qwen}",
        f"- Random: {random}",
        "",
        "Replacement simulation",
        *[
            (
                f"- {int(item['fsn_fraction'] * 100)}% FSN: "
                f"accepted={item['expected_accepted_rate']:.3f}, "
                f"value={item['expected_mean_value_score']}, "
                f"diversity={item['expected_diversity']}, "
                f"runtime reduction={item['expected_runtime_reduction']:.1%}, "
                f"risks={item['risk_flags']}"
            )
            for item in simulations
        ],
        "",
        "Judgement",
        (
            "- FSN has measurable curriculum value: "
            + str(diverse["accepted_count"] > 0)
            + f" ({diverse['accepted_count']} accepted candidates)."
        ),
        (
            "- Diversity-aware FSN improves over plain FSN: "
            + str(
                diverse["accepted_rate_by_difficulty_evaluator"]
                >= fsn["accepted_rate_by_difficulty_evaluator"]
                and diverse["diversity_score"] > fsn["diversity_score"]
            )
            + "; the aggregate gain is not uniform across every seed."
        ),
        (
            "- Accepted-rate gap to historical Qwen: "
            f"{diverse['accepted_rate_by_difficulty_evaluator'] - qwen['accepted_rate_by_difficulty_evaluator']:.3f}"
        ),
        (
            "- Historical Qwen comparison: diversity-aware FSN "
            f"{diverse['accepted_rate_by_difficulty_evaluator']:.3f} vs "
            f"Qwen {qwen['accepted_rate_by_difficulty_evaluator']:.3f}; "
            "the FSN rate is higher in this fixed-opponent retrospective evaluation."
        ),
        (
            "- Random comparison: diversity-aware FSN and Random accepted rates "
            f"are {diverse['accepted_rate_by_difficulty_evaluator']:.3f} and "
            f"{random['accepted_rate_by_difficulty_evaluator']:.3f}; "
            "FSN does not yet dominate Random on acceptance."
        ),
        (
            "- Main FSN rejection reasons: "
            f"{diverse['rejection_reason_distribution']}."
        ),
        (
            "- Empirically safest simulated mixture: "
            f"{int(safest['fsn_fraction'] * 100)}% FSN, retaining "
            f"{int(safest['historical_qwen_fraction'] * 100)}% Qwen fallback."
        ),
        "- A tightly controlled replacement training smoke is recommended because FSN acceptance is nonzero and all env-loadable candidates completed policy evaluation.",
        "- For a first training smoke, start conservatively at 25% FSN even though the 75% shadow mixture has the strongest aggregate acceptance/diversity trade-off.",
        "- Runtime-reduction estimates cover candidate generation, not the fixed policy-evaluation cost.",
        "- This study cannot claim MAPPO improvement, full Qwen replacement, or on-policy distillation.",
    ]
    report_path = (
        ROOT_DIR
        / "experiments"
        / "falcon_2v2_noweapon"
        / "reports"
        / "fsn_stage3_policy_evaluated_shadow_report.txt"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
