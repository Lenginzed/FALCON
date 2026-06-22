#!/usr/bin/env python
"""Smoke test for standardized FALCON trajectory recording."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from envs.JSBSim.envs import MultipleCombatEnv  # noqa: E402
from falcon.trajectory_recorder import (  # noqa: E402
    extract_scenario_vector,
    load_trajectory,
    summarize_episode,
    validate_trajectory,
)


def main():
    output_dir = ROOT_DIR / "tests" / "tmp_falcon_trajectories"
    env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
    env.enable_trajectory_recording(
        str(output_dir),
        save_success=True,
        prefix="trajectory_smoke",
        metadata={"policy_id": "smoke_policy", "training_steps": 0},
        save_reason="evaluation",
    )
    for condition in env.task.termination_conditions:
        if hasattr(condition, "max_steps"):
            condition.max_steps = 1
    env.reset()
    action = np.array([[20, 20, 20, 0], [20, 20, 20, 0], [20, 20, 20, 0], [20, 20, 20, 0]])
    env.step(action)
    path = env._trajectory_recorder.last_saved_path
    env.close()

    if path is None:
        raise RuntimeError("Trajectory recorder did not save an episode.")
    data = load_trajectory(path)
    validation = validate_trajectory(data)
    scenario_vector = extract_scenario_vector(data)
    episode_summary = summarize_episode(data)
    result = {
        "schema_version": "falcon.trajectory_recorder_smoke.v1",
        "trajectory_path": str(path),
        "validation": validation,
        "scenario_vector": scenario_vector,
        "episode_summary": episode_summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not validation["is_valid"]:
        raise AssertionError(validation)


if __name__ == "__main__":
    main()
