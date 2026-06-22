#!/usr/bin/env python
"""Minimal MAPPO checkpoint smoke for 2v2 NoWeapon.

This script reuses the existing MAPPO runner and does not modify the training
algorithm. It performs a tiny rollout/update only to verify that a loadable
``actor_latest.pt`` and ``critic_latest.pt`` can be produced.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

if not hasattr(np, "product"):
    np.product = np.prod  # NumPy 2.x compatibility for the existing flattener.

from config import get_config  # noqa: E402
from runner.share_jsbsim_runner import ShareJSBSimRunner  # noqa: E402
from scripts.train.train_jsbsim import make_train_env, parse_args  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _build_train_args(num_env_steps: int, buffer_size: int, seed: int) -> argparse.Namespace:
    smoke_args = [
        "--env-name",
        "MultipleCombat",
        "--algorithm-name",
        "mappo",
        "--scenario-name",
        "2v2/NoWeapon/Selfplay",
        "--experiment-name",
        "falcon_minimal_mappo_2v2_smoke",
        "--seed",
        str(seed),
        "--n-training-threads",
        "1",
        "--n-rollout-threads",
        "1",
        "--num-env-steps",
        str(num_env_steps),
        "--buffer-size",
        str(buffer_size),
        "--num-mini-batch",
        "1",
        "--ppo-epoch",
        "1",
        "--data-chunk-length",
        str(max(1, min(4, buffer_size))),
        "--log-interval",
        "1",
        "--save-interval",
        "1",
        "--lr",
        "3e-4",
        "--gamma",
        "0.99",
        "--clip-param",
        "0.2",
        "--max-grad-norm",
        "2",
        "--entropy-coef",
        "1e-3",
        "--user-name",
        "falcon_smoke",
    ]
    parser = get_config()
    return parse_args(smoke_args, parser)


def main() -> None:
    cli = argparse.ArgumentParser(description="Train a minimal MAPPO 2v2 smoke checkpoint.")
    cli.add_argument("--output-dir", default=str(ROOT_DIR / "tests" / "tmp_falcon_policy_smoke"))
    cli.add_argument("--num-env-steps", type=int, default=8)
    cli.add_argument("--buffer-size", type=int, default=8)
    cli.add_argument("--seed", type=int, default=0)
    args = cli.parse_args()

    output_dir = Path(args.output_dir)
    save_dir = output_dir / "mappo_2v2_smoke"
    summary_path = output_dir / "minimal_mappo_train_summary.json"
    actor_path = save_dir / "actor_latest.pt"
    critic_path = save_dir / "critic_latest.pt"

    warnings = []
    summary = {
        "schema_version": "falcon.minimal_mappo_train_smoke.v1",
        "train_started": False,
        "train_finished": False,
        "checkpoint_found": False,
        "actor_checkpoint_path": str(actor_path),
        "critic_checkpoint_path": str(critic_path),
        "num_env_steps": int(args.num_env_steps),
        "buffer_size": int(args.buffer_size),
        "save_dir": str(save_dir),
        "failure_stage": None,
        "warnings": warnings,
    }

    if args.num_env_steps < args.buffer_size:
        summary["failure_stage"] = "invalid_smoke_args"
        warnings.append("num_env_steps must be >= buffer_size so ShareJSBSimRunner.run() executes at least one update.")
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    envs = None
    start_time = time.time()
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        all_args = _build_train_args(args.num_env_steps, args.buffer_size, args.seed)

        np.random.seed(all_args.seed)
        random.seed(all_args.seed)
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

        summary["train_started"] = True
        envs = make_train_env(all_args)
        runner = ShareJSBSimRunner(
            {
                "all_args": all_args,
                "envs": envs,
                "eval_envs": None,
                "device": device,
                "run_dir": save_dir,
                "render_mode": "txt",
            }
        )
        runner.run()
        summary["train_finished"] = True
    except Exception as exc:  # noqa: BLE001 - smoke should report structured failure
        summary["failure_stage"] = "training_failed"
        warnings.append(f"Minimal MAPPO training failed: {type(exc).__name__}: {exc}")
        warnings.append(traceback.format_exc())
    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to close training envs cleanly: {type(exc).__name__}: {exc}")

    actor_exists = actor_path.exists()
    critic_exists = critic_path.exists()
    summary["checkpoint_found"] = bool(actor_exists and critic_exists)
    summary["actor_checkpoint_path"] = str(actor_path) if actor_exists else None
    summary["critic_checkpoint_path"] = str(critic_path) if critic_exists else None
    if summary["failure_stage"] is None and not summary["checkpoint_found"]:
        summary["failure_stage"] = "checkpoint_not_saved"
        warnings.append("Training finished but actor_latest.pt and critic_latest.pt were not both found.")
    if actor_exists and actor_path.stat().st_mtime < start_time:
        warnings.append("actor_latest.pt exists but was not modified during this smoke run.")
    if critic_exists and critic_path.stat().st_mtime < start_time:
        warnings.append("critic_latest.pt exists but was not modified during this smoke run.")

    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
