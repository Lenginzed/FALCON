#!/usr/bin/env python
"""Register a fixed MAPPO checkpoint as the frozen baseline eval opponent."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from falcon.baseline_experiment import BaselineExperimentRunner, load_yaml  # noqa: E402
from falcon.eval_set_evaluator import resolve_group_checkpoint  # noqa: E402

DEFAULT_PROTOCOL = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "configs" / "experiment_protocol.yaml"
DEFAULT_OUTPUT = ROOT_DIR / "experiments" / "falcon_2v2_noweapon" / "manifests" / "eval_opponent.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the frozen fixed-checkpoint evaluation opponent.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--checkpoint", default=None, help="Explicit actor checkpoint path.")
    parser.add_argument("--source-group", default="mappo_base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-role", choices=("latest", "best"), default="latest")
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    protocol_path = _resolve(args.protocol)
    protocol = load_yaml(protocol_path)
    results_root = _resolve(protocol["output_root"])
    warnings = []

    checkpoint = _resolve(args.checkpoint) if args.checkpoint else resolve_group_checkpoint(
        results_root,
        args.source_group,
        args.seed,
        args.checkpoint_role,
    )
    source = f"existing_{args.source_group}_{args.checkpoint_role}_checkpoint"
    if checkpoint is None and args.train_if_missing:
        runner = BaselineExperimentRunner(protocol_path, args.source_group, args.seed)
        smoke_result = runner.smoke_run()
        value = smoke_result.get("checkpoint_path")
        checkpoint = Path(str(value)) if value else None
        source = f"new_short_{args.source_group}_smoke_checkpoint"
        warnings.append("No existing checkpoint was found; prepared opponent with a short smoke training run.")

    if checkpoint is None or not checkpoint.exists():
        raise SystemExit(
            "No usable evaluation opponent checkpoint found. Provide --checkpoint or use --train-if-missing."
        )

    training_steps = _infer_training_steps(checkpoint)
    if training_steps is None or training_steps < 512:
        warnings.append(
            "Registered opponent is smoke-grade and weak; freeze a stronger independent baseline checkpoint "
            "before interpreting formal performance results."
        )

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "falcon.eval_opponent_manifest.v1",
        "opponent_id": f"{args.source_group}_seed{args.seed}_{args.checkpoint_role}_fixed",
        "opponent_mode": "fixed_checkpoint",
        "checkpoint_path": _portable_path(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source": source,
        "source_group": args.source_group,
        "source_seed": args.seed,
        "checkpoint_role": args.checkpoint_role,
        "training_steps": training_steps,
        "environment": protocol.get("environment", "2v2/NoWeapon/Selfplay"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "warnings": warnings,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_path": str(output), **manifest}, indent=2, sort_keys=True))


def _infer_training_steps(checkpoint: Path) -> int | None:
    for parent in checkpoint.parents:
        summary_path = parent / "pilot_run_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            rounds = data.get("round_summaries") or []
            values = [
                int((item.get("training_summary") or {}).get("num_env_steps", 0))
                for item in rounds
            ]
            return sum(values)
    return None


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
