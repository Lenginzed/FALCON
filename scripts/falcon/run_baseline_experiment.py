#!/usr/bin/env python
"""Run a frozen baseline group dry-run or bounded smoke-run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import BaselineExperimentRunner, SUPPORTED_GROUPS  # noqa: E402

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"


def main() -> None:
    launched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    launch_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run a FALCON baseline preparation dry-run or smoke-run.")
    parser.add_argument("--group", required=True, choices=SUPPORTED_GROUPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--train-steps-per-round", type=int, default=None)
    parser.add_argument("--eval-episodes-per-round", type=int, default=None)
    parser.add_argument("--policy-eval-episodes-per-candidate", type=int, default=None)
    parser.add_argument("--qwen-candidates-per-round", type=int, default=None)
    parser.add_argument("--random-candidates-per-round", type=int, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke-run", action="store_true")
    mode.add_argument("--pilot-run", action="store_true")
    args = parser.parse_args()

    runner = BaselineExperimentRunner(args.protocol, args.group, args.seed, output_dir=args.output_dir)
    if args.dry_run:
        result = runner.dry_run()
    elif args.smoke_run:
        result = runner.smoke_run()
    else:
        result = runner.pilot_run(
            {
                "max_rounds": args.max_rounds,
                "train_steps_per_round": args.train_steps_per_round,
                "eval_episodes_per_round": args.eval_episodes_per_round,
                "policy_eval_episodes_per_candidate": args.policy_eval_episodes_per_candidate,
                "qwen_candidates_per_round": args.qwen_candidates_per_round,
                "random_candidates_per_round": args.random_candidates_per_round,
            }
        )
    launcher_runtime = round(time.perf_counter() - launch_start, 3)
    result["launcher_started_at"] = launched_at
    result["launcher_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["launcher_runtime_seconds"] = launcher_runtime
    result["launcher_runtime_human_readable"] = _human_duration(launcher_runtime)
    print(json.dumps(result, indent=2, sort_keys=True))


def _human_duration(seconds: float) -> str:
    total = int(round(max(float(seconds), 0.0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
