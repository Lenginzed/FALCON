#!/usr/bin/env python
"""Smoke-test safe runtime aggregation used by baseline recovery."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import _safe_float, _sum_safe_float  # noqa: E402


def main() -> None:
    rows = [
        {"training_runtime_seconds": "1.25", "policy_eval_runtime_seconds": 2},
        {"training_runtime_seconds": None, "policy_eval_runtime_seconds": "nan"},
        {"training_runtime_seconds": "bad", "policy_eval_runtime_seconds": "3.5"},
        {"qwen_runtime_seconds": "4.0"},
    ]
    assert _safe_float("2.5") == 2.5
    assert _safe_float(None, 7.0) == 7.0
    assert _safe_float("nan", 0.0) == 0.0
    assert _safe_float("bad", 0.0) == 0.0
    assert _sum_safe_float(rows, "training_runtime_seconds") == 1.25
    assert _sum_safe_float(rows, "policy_eval_runtime_seconds") == 5.5
    assert _sum_safe_float(rows, "qwen_runtime_seconds") == 4.0
    print(
        {
            "schema_version": "falcon.baseline_summary_recovery_smoke.v1",
            "safe_float_ok": True,
            "runtime_aggregation_ok": True,
        }
    )


if __name__ == "__main__":
    main()
