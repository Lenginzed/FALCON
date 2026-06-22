#!/usr/bin/env python
"""Train a tiny 2v2 MAPPO run and save early/later actor checkpoints.

This is a smoke test only. It reuses the existing ShareJSBSimRunner and does
not modify MAPPO. The early checkpoint is saved immediately after runner
initialization; the later checkpoint is saved after a small number of PPO
updates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

if not hasattr(np, "product"):
    np.product = np.prod  # NumPy 2.x compatibility for the existing flattener.

from config import get_config  # noqa: E402
from runner.share_jsbsim_runner import ShareJSBSimRunner  # noqa: E402
from scripts.train.train_jsbsim import make_train_env, parse_args  # noqa: E402


def _write_json(path: Path, data: Dict[str, Any]) -> None:
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
        "falcon_multicheckpoint_mappo_2v2_smoke",
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


def _actor_delta_l2(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in a.keys() & b.keys():
        diff = a[key].detach().cpu().float() - b[key].detach().cpu().float()
        total += float(torch.sum(diff * diff).item())
    return float(total ** 0.5)


def main() -> None:
    cli = argparse.ArgumentParser(description="Create early/later MAPPO checkpoints for FALCON smoke.")
    cli.add_argument("--output-dir", default=str(ROOT_DIR / "tests" / "tmp_falcon_multicheckpoint"))
    cli.add_argument("--num-env-steps", type=int, default=64)
    cli.add_argument("--buffer-size", type=int, default=16)
    cli.add_argument("--seed", type=int, default=3)
    args = cli.parse_args()

    output_dir = Path(args.output_dir)
    save_dir = output_dir / "mappo_2v2_multicheckpoint"
    summary_path = output_dir / "multicheckpoint_train_summary.json"
    early_path = save_dir / "actor_early.pt"
    later_path = save_dir / "actor_later.pt"
    actor_latest_path = save_dir / "actor_latest.pt"
    critic_latest_path = save_dir / "critic_latest.pt"
    warnings = []
    summary = {
        "schema_version": "falcon.multicheckpoint_train_smoke.v1",
        "train_started": False,
        "train_finished": False,
        "early_checkpoint_path": str(early_path),
        "later_checkpoint_path": str(later_path),
        "actor_latest_path": str(actor_latest_path),
        "critic_latest_path": str(critic_latest_path),
        "num_env_steps": int(args.num_env_steps),
        "buffer_size": int(args.buffer_size),
        "num_updates_requested": int(args.num_env_steps // max(args.buffer_size, 1)),
        "num_checkpoints_saved": 0,
        "early_later_parameter_l2": None,
        "warnings": warnings,
        "failure_stage": None,
    }

    if args.num_env_steps < args.buffer_size * 2:
        summary["warnings"].append("num_env_steps is very small; early/later checkpoints may show little separation.")
    if args.num_env_steps < args.buffer_size:
        summary["failure_stage"] = "invalid_smoke_args"
        warnings.append("num_env_steps must be >= buffer_size so ShareJSBSimRunner.run() executes at least one update.")
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    envs = None
    start_time = time.time()
    early_state = None
    later_state = None
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
        early_state = {key: value.detach().cpu().clone() for key, value in runner.policy.actor.state_dict().items()}
        torch.save(early_state, early_path)
        runner.run()
        later_state = {key: value.detach().cpu().clone() for key, value in runner.policy.actor.state_dict().items()}
        torch.save(later_state, later_path)
        summary["train_finished"] = True
    except Exception as exc:  # noqa: BLE001 - smoke should report structured failure
        summary["failure_stage"] = "training_failed"
        warnings.append(f"Multi-checkpoint MAPPO training failed: {type(exc).__name__}: {exc}")
        warnings.append(traceback.format_exc())
    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to close training envs cleanly: {type(exc).__name__}: {exc}")

    checkpoint_paths = [early_path, later_path, actor_latest_path, critic_latest_path]
    existing = [path for path in checkpoint_paths if path.exists()]
    summary["num_checkpoints_saved"] = len(existing)
    summary["early_checkpoint_path"] = str(early_path) if early_path.exists() else None
    summary["later_checkpoint_path"] = str(later_path) if later_path.exists() else None
    summary["actor_latest_path"] = str(actor_latest_path) if actor_latest_path.exists() else None
    summary["critic_latest_path"] = str(critic_latest_path) if critic_latest_path.exists() else None
    if early_state is not None and later_state is not None:
        summary["early_later_parameter_l2"] = round(_actor_delta_l2(early_state, later_state), 8)
        if summary["early_later_parameter_l2"] == 0.0:
            warnings.append("Early and later actor parameters are identical; longer smoke training may be required.")
    if summary["failure_stage"] is None and (not early_path.exists() or not later_path.exists()):
        summary["failure_stage"] = "checkpoint_not_saved"
        warnings.append("Training did not produce both early and later actor checkpoints.")
    if actor_latest_path.exists() and actor_latest_path.stat().st_mtime < start_time:
        warnings.append("actor_latest.pt exists but was not modified during this smoke run.")
    if critic_latest_path.exists() and critic_latest_path.stat().st_mtime < start_time:
        warnings.append("critic_latest.pt exists but was not modified during this smoke run.")

    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
