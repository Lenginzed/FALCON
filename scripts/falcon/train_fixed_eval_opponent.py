#!/usr/bin/env python
"""Train an independent fixed-scenario MAPPO candidate for formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.falcon_controller import _train_mappo_smoke  # noqa: E402

DEFAULT_OUTPUT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "opponents" / "fixed_baseline_opponent"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an independent fixed MAPPO baseline opponent candidate.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-steps", type=int, default=2048)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    run_dir = output_dir / f"candidate_seed{args.seed}_steps{args.train_steps}"
    summary_path = run_dir / "fixed_opponent_training_summary.json"
    manifest_path = run_dir / "fixed_opponent_candidate_manifest.json"
    warnings = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    start_time = time.time()

    if args.train_steps < args.buffer_size:
        result = {
            "schema_version": "falcon.fixed_opponent_training.v1",
            "train_started": False,
            "train_finished": False,
            "checkpoint_saved": False,
            "seed": args.seed,
            "training_steps": args.train_steps,
            "buffer_size": args.buffer_size,
            "failure_stage": "invalid_training_config",
            "warnings": ["train_steps must be greater than or equal to buffer_size."],
        }
        _write_json(summary_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    training = _train_mappo_smoke(
        scenario_name="2v2/NoWeapon/Selfplay",
        output_dir=run_dir,
        num_env_steps=args.train_steps,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )
    actor_path = _path_or_none(training.get("actor_checkpoint_path"))
    critic_path = _path_or_none(training.get("critic_checkpoint_path"))
    snapshot_dir = run_dir / "checkpoints"
    actor_snapshot = snapshot_dir / f"actor_seed{args.seed}_steps{args.train_steps}.pt"
    critic_snapshot = snapshot_dir / f"critic_seed{args.seed}_steps{args.train_steps}.pt"
    if actor_path and actor_path.exists():
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actor_path, actor_snapshot)
    if critic_path and critic_path.exists():
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(critic_path, critic_snapshot)

    checkpoint_saved = bool(actor_snapshot.exists() and critic_snapshot.exists())
    if args.train_steps < 2048:
        warnings.append(
            "Candidate uses fewer than 2048 training steps and is intended for pipeline/strength screening only."
        )
    warnings.extend(training.get("warnings") or [])
    failure_stage = training.get("failure_stage")
    if failure_stage is None and not checkpoint_saved:
        failure_stage = "checkpoint_snapshot_failed"
        warnings.append("Training completed but independent actor/critic snapshots were not both saved.")

    summary = {
        "schema_version": "falcon.fixed_opponent_training.v1",
        "opponent_candidate_id": f"independent_mappo_seed{args.seed}_steps{args.train_steps}",
        "train_started": bool(training.get("training_started")),
        "train_finished": bool(training.get("training_finished")),
        "checkpoint_saved": checkpoint_saved,
        "actor_checkpoint_path": str(actor_snapshot) if actor_snapshot.exists() else None,
        "critic_checkpoint_path": str(critic_snapshot) if critic_snapshot.exists() else None,
        "runner_actor_latest_path": str(actor_path) if actor_path else None,
        "runner_critic_latest_path": str(critic_path) if critic_path else None,
        "seed": int(args.seed),
        "training_steps": int(args.train_steps),
        "buffer_size": int(args.buffer_size),
        "environment": "2v2/NoWeapon/Selfplay",
        "algorithm": "MAPPO",
        "fixed_base_scenario_only": True,
        "falcon_used": False,
        "qwen_used": False,
        "random_curriculum_used": False,
        "difficulty_evaluator_used": False,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(time.time() - start_time, 3),
        "failure_stage": failure_stage,
        "warnings": sorted(set(str(item) for item in warnings if item)),
    }
    candidate_manifest = {
        "schema_version": "falcon.fixed_opponent_candidate_manifest.v1",
        "opponent_id": summary["opponent_candidate_id"],
        "opponent_mode": "fixed_checkpoint",
        "checkpoint_path": _portable_path(actor_snapshot) if actor_snapshot.exists() else None,
        "critic_checkpoint_path": _portable_path(critic_snapshot) if critic_snapshot.exists() else None,
        "checkpoint_sha256": _sha256(actor_snapshot) if actor_snapshot.exists() else None,
        "sha256": _sha256(actor_snapshot) if actor_snapshot.exists() else None,
        "source": "independent_mappo_baseline",
        "seed": int(args.seed),
        "training_steps": int(args.train_steps),
        "environment": "2v2/NoWeapon/Selfplay",
        "accepted_for_formal_eval": False,
        "strength_eval_summary": None,
        "created_at": summary["finished_at"],
        "warnings": summary["warnings"],
    }
    _write_json(summary_path, summary)
    _write_json(manifest_path, candidate_manifest)
    print(json.dumps({"summary_path": str(summary_path), "manifest_path": str(manifest_path), **summary}, indent=2, sort_keys=True))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _path_or_none(value: object) -> Path | None:
    return Path(str(value)) if value else None


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


if __name__ == "__main__":
    main()
