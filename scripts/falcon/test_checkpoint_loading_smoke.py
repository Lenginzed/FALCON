#!/usr/bin/env python
"""Checkpoint loading smoke for a minimal MAPPO actor.

The test verifies actor loading, 2v2 env reset, observation shape, and one
deterministic action. It does not train or change MAPPO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

if not hasattr(np, "product"):
    np.product = np.prod  # NumPy 2.x compatibility for the existing flattener.

from config import get_config  # noqa: E402
from algorithms.mappo.ppo_policy import PPOPolicy  # noqa: E402
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _resolve_checkpoint(explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_path = os.environ.get("FALCON_POLICY_SMOKE_CHECKPOINT")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(ROOT_DIR / "tests" / "tmp_falcon_policy_smoke" / "mappo_2v2_smoke" / "actor_latest.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _policy_args():
    args = get_config().parse_known_args([])[0]
    args.env_name = "MultipleCombat"
    args.algorithm_name = "mappo"
    args.scenario_name = "2v2/NoWeapon/Selfplay"
    args.n_rollout_threads = 1
    args.n_eval_rollout_threads = 1
    args.use_selfplay = False
    return args


def main() -> None:
    cli = argparse.ArgumentParser(description="Smoke load a MAPPO actor and generate one action.")
    cli.add_argument("--checkpoint", default=None)
    cli.add_argument("--output-dir", default=str(ROOT_DIR / "tests" / "tmp_falcon_policy_smoke"))
    args = cli.parse_args()

    output_path = Path(args.output_dir) / "checkpoint_loading_smoke.json"
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    warnings = []
    summary = {
        "schema_version": "falcon.checkpoint_loading_smoke.v1",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "actor_loaded": False,
        "env_reset_success": False,
        "obs_shape": None,
        "share_obs_shape": None,
        "action_generated": False,
        "action_shape": None,
        "failure_stage": None,
        "warnings": warnings,
    }

    if checkpoint_path is None or not checkpoint_path.exists():
        summary["failure_stage"] = "checkpoint_not_found"
        warnings.append(f"Actor checkpoint not found: {checkpoint_path}")
        _write_json(output_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    env = None
    try:
        env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
        env.seed(0)
        policy = PPOPolicy(_policy_args(), env.observation_space, env.share_observation_space, env.action_space, device=torch.device("cpu"))
        state_dict = torch.load(str(checkpoint_path), map_location=torch.device("cpu"), weights_only=True)
        policy.actor.load_state_dict(state_dict)
        policy.prep_rollout()
        summary["actor_loaded"] = True

        obs, share_obs = env.reset()
        summary["env_reset_success"] = True
        summary["obs_shape"] = list(np.asarray(obs).shape)
        summary["share_obs_shape"] = list(np.asarray(share_obs).shape)

        rnn_states = np.zeros((env.num_agents, policy.args.recurrent_hidden_layers, policy.args.recurrent_hidden_size), dtype=np.float32)
        masks = np.ones((env.num_agents, 1), dtype=np.float32)
        with torch.no_grad():
            actions, _ = policy.act(obs, rnn_states, masks, deterministic=True)
        actions_np = actions.detach().cpu().numpy()
        summary["action_generated"] = True
        summary["action_shape"] = list(actions_np.shape)
    except Exception as exc:  # noqa: BLE001 - smoke should report structured failure
        if not summary["actor_loaded"]:
            summary["failure_stage"] = "actor_loading"
        elif not summary["env_reset_success"]:
            summary["failure_stage"] = "env_reset"
        elif not summary["action_generated"]:
            summary["failure_stage"] = "action_generation"
        else:
            summary["failure_stage"] = "unknown"
        warnings.append(f"Checkpoint loading smoke failed: {type(exc).__name__}: {exc}")
        warnings.append(traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to close env cleanly: {type(exc).__name__}: {exc}")

    _write_json(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
