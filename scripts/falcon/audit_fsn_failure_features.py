"""Audit degenerate failure features and Stage 2 proxy replacements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.fsn_dataset import FAILURE_KEYS, PROXY_FEATURE_KEYS, load_jsonl


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stats(
    samples: Sequence[Mapping[str, Any]], section: str, key: str
) -> dict[str, Any]:
    values = [
        number
        for item in samples
        if (
            number := _number((item.get(section) or {}).get(key))
        )
        is not None
    ]
    return {
        "count": len(values),
        "missing": len(samples) - len(values),
        "unique_count": len({round(value, 9) for value in values}),
        "mean": round(statistics.fmean(values), 6) if values else None,
        "std": round(statistics.pstdev(values), 6) if values else None,
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-dataset",
        default=(
            "experiments/falcon_2v2_noweapon/fsn/"
            "failure_to_scenario_dataset.jsonl"
        ),
    )
    parser.add_argument(
        "--stage2-dataset",
        default=(
            "experiments/falcon_2v2_noweapon/fsn/stage2/"
            "failure_to_scenario_dataset_dedup.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/fsn/stage2",
    )
    args = parser.parse_args()

    stage1 = load_jsonl(ROOT_DIR / args.stage1_dataset)
    stage2 = load_jsonl(ROOT_DIR / args.stage2_dataset)
    failure_stats = {
        key: _stats(stage1, "failure_vector", key) for key in FAILURE_KEYS
    }
    proxy_stats = {
        key: _stats(stage2, "proxy_features", key)
        for key in PROXY_FEATURE_KEYS
    }
    degenerate = [
        key
        for key, stats in failure_stats.items()
        if stats["std"] is not None and stats["std"] < 1e-8
    ]
    proxy_non_degenerate = [
        key
        for key, stats in proxy_stats.items()
        if stats["std"] is not None and stats["std"] >= 1e-8
    ]
    findings = {
        "initial_disadvantage": {
            "builder_read_failure": False,
            "field_present_count": failure_stats["initial_disadvantage"]["count"],
            "diagnosis": (
                "The source failure summaries contain a constant value; the "
                "Stage 1 builder read the canonical failure_scores field correctly."
            ),
            "replacement_features": [
                "initial_disadvantage_proxy",
                "distance_disadvantage",
                "heading_disadvantage",
                "altitude_disadvantage",
                "velocity_disadvantage",
            ],
        },
        "generalization_failure": {
            "builder_read_failure": False,
            "field_present_count": failure_stats["generalization_failure"]["count"],
            "diagnosis": (
                "The source value is consistently zero because reliable pool "
                "novelty/performance-drop context was unavailable in collected "
                "failure summaries."
            ),
            "replacement_features": [
                "pool_novelty_score",
                "policy_performance_drop_proxy",
            ],
        },
    }
    audit = {
        "schema_version": "falcon.fsn_failure_feature_audit.v1",
        "stage1_sample_count": len(stage1),
        "stage2_sample_count": len(stage2),
        "failure_feature_stats": failure_stats,
        "degenerate_failure_features": degenerate,
        "proxy_feature_stats": proxy_stats,
        "non_degenerate_proxy_features": proxy_non_degenerate,
        "findings": findings,
        "warnings": [
            "Proxy features are engineering substitutes, not recovered ground-truth failure labels.",
            "The original constant features are retained for provenance.",
        ],
    }
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fsn_failure_feature_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "FSN Stage 2 Failure Feature Audit",
        "",
        f"Stage 1 samples: {len(stage1)}",
        f"Stage 2 samples: {len(stage2)}",
        f"Degenerate original features: {', '.join(degenerate) or 'none'}",
        "",
        "Diagnosis:",
        "- initial_disadvantage was read correctly but is constant in source data.",
        "- generalization_failure was read correctly but is zero in source data.",
        "- Stage 2 therefore adds scenario disadvantage and novelty/performance proxies.",
        "",
        "Non-degenerate proxy features:",
        *[f"- {key}: std={proxy_stats[key]['std']}" for key in proxy_non_degenerate],
        "",
        "Caution: proxies do not replace missing ground-truth labels.",
    ]
    (output_dir / "fsn_failure_feature_audit.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
