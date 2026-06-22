#!/usr/bin/env python
"""Smoke test for sampling_plan -> MAPPO training-entry bridge.

This does not implement a curriculum trainer. It only verifies that one
scenario YAML selected from a FALCON sampling source can be staged into the LAG
config tree and used by the existing MAPPO runner for a tiny training smoke.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

if not hasattr(np, "product"):
    np.product = np.prod  # NumPy 2.x compatibility for the existing flattener.

from config import get_config  # noqa: E402
from falcon.random_scenario_generator import RandomScenarioGenerator  # noqa: E402
from falcon.scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config, save_scenario_yaml  # noqa: E402
from falcon.training_plan_adapter import TrainingPlanAdapter  # noqa: E402
from runner.share_jsbsim_runner import ShareJSBSimRunner  # noqa: E402
from scripts.train.train_jsbsim import make_train_env, parse_args  # noqa: E402


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _selected_is_falcon_generated(selected: Optional[Mapping[str, Any]], base_config_path: Path) -> bool:
    if not selected:
        return False
    source = str(selected.get("source") or "").lower()
    yaml_path = selected.get("scenario_yaml_path") or selected.get("yaml_path")
    if not yaml_path:
        return False
    try:
        if Path(str(yaml_path)).resolve() == base_config_path.resolve():
            return False
    except OSError:
        pass
    return source not in {"original", "base"} and bool(Path(str(yaml_path)).exists())


def _plan_from_accepted_pool(pool_path: Path) -> Optional[Dict[str, Any]]:
    data = _load_json(pool_path)
    if not data:
        return None
    sampled = []
    for item in data.get("items", []):
        if not isinstance(item, dict) or not item.get("accepted_into_curriculum_pool"):
            continue
        sampled.append(
            {
                "scenario_id": item.get("scenario_id"),
                "source": item.get("source") or (item.get("candidate_scenario") or {}).get("generator_type"),
                "scenario_yaml_path": item.get("scenario_yaml_path"),
                "sampling_weight": item.get("sampling_weight"),
                "final_value_score": item.get("final_value_score"),
                "target_failure_modes": item.get("target_failure_modes") or [],
                "priority_level": item.get("priority_level"),
            }
        )
    if not sampled:
        return None
    return {
        "schema_version": "falcon.training_bridge_pool_plan.v1",
        "num_samples": len(sampled),
        "sampled_scenarios": sampled,
        "warnings": [f"Built bridge sampling plan from accepted items in {pool_path}."],
    }


def _random_fallback_plan(output_dir: Path, base_config_path: Path) -> Dict[str, Any]:
    base_config = load_base_scenario_config(base_config_path)
    candidate = RandomScenarioGenerator({"seed": 41}).generate_from_base(base_config, num_scenarios=1)[0]
    yaml_config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
    yaml_path = output_dir / "training_bridge_random_fallback.yaml"
    save_scenario_yaml(yaml_config, yaml_path)
    return {
        "schema_version": "falcon.training_bridge_random_fallback_plan.v1",
        "num_samples": 1,
        "sampled_scenarios": [
            {
                "scenario_id": candidate.get("scenario_id"),
                "source": "random_fallback",
                "scenario_yaml_path": str(yaml_path),
                "sampling_weight": 1.0,
                "final_value_score": 0.0,
                "target_failure_modes": candidate.get("target_failure_modes") or [],
                "priority_level": "smoke",
            }
        ],
        "warnings": ["Generated a legal random fallback scenario because no accepted FALCON scenario was available."],
    }


def _load_or_build_plan(output_dir: Path, base_config_path: Path, warnings: list[str]) -> tuple[Dict[str, Any], bool, bool]:
    sampling_plan_path = ROOT_DIR / "tests" / "tmp_falcon_miniloop" / "falcon_sampling_plan.json"
    pool_path = ROOT_DIR / "tests" / "tmp_falcon_multicheckpoint" / "multicheckpoint_curriculum_pool.json"
    adapter = TrainingPlanAdapter()
    sampling_plan_loaded = False
    fallback_used = False
    plan = None
    if sampling_plan_path.exists():
        plan = adapter.load_sampling_plan(sampling_plan_path)
        sampling_plan_loaded = True
        selected = adapter.select_scenario(plan, strategy="weighted").get("selected_scenario")
        if not _selected_is_falcon_generated(selected, base_config_path):
            warnings.append("Loaded sampling plan did not contain an accepted/generated training scenario; trying accepted curriculum pool fallback.")
            plan = None
            fallback_used = True
    if plan is None:
        plan = _plan_from_accepted_pool(pool_path)
        if plan is not None:
            fallback_used = True
            warnings.extend(plan.get("warnings", []))
    if plan is None:
        plan = _random_fallback_plan(output_dir, base_config_path)
        fallback_used = True
        warnings.extend(plan.get("warnings", []))
    return plan, sampling_plan_loaded, fallback_used


def _build_train_args(scenario_name: str, num_env_steps: int, buffer_size: int, seed: int) -> argparse.Namespace:
    smoke_args = [
        "--env-name",
        "MultipleCombat",
        "--algorithm-name",
        "mappo",
        "--scenario-name",
        scenario_name,
        "--experiment-name",
        "falcon_training_plan_bridge_smoke",
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


def _train_smoke(
    config_name: str,
    output_dir: Path,
    num_env_steps: int,
    buffer_size: int,
    seed: int,
    training_config_path: Optional[str] = None,
    requires_parse_config_patch: bool = False,
) -> Dict[str, Any]:
    save_dir = output_dir / "mappo_training_plan_bridge"
    actor_path = save_dir / "actor_latest.pt"
    critic_path = save_dir / "critic_latest.pt"
    warnings: list[str] = []
    summary = {
        "schema_version": "falcon.training_plan_bridge_train_summary.v1",
        "training_started": False,
        "training_finished": False,
        "checkpoint_saved": False,
        "actor_checkpoint_path": str(actor_path),
        "critic_checkpoint_path": str(critic_path),
        "config_name_or_path": config_name,
        "training_config_path": training_config_path,
        "requires_parse_config_patch": bool(requires_parse_config_patch),
        "num_env_steps": int(num_env_steps),
        "buffer_size": int(buffer_size),
        "failure_stage": None,
        "warnings": warnings,
    }
    envs = None
    original_parse_config = None
    start_time = time.time()
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        if requires_parse_config_patch:
            if not training_config_path or not Path(training_config_path).exists():
                raise FileNotFoundError(f"training_config_path for parse_config patch does not exist: {training_config_path}")
            import yaml
            from envs.JSBSim.envs import env_base

            with Path(training_config_path).open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            original_parse_config = env_base.parse_config

            def _parse_config_override(filename):
                if filename == config_name:
                    return type("EnvConfig", (object,), config_data)
                return original_parse_config(filename)

            env_base.parse_config = _parse_config_override
            warnings.append("Applied temporary parse_config mapping for FALCON staged YAML.")
        all_args = _build_train_args(config_name, num_env_steps, buffer_size, seed)
        np.random.seed(all_args.seed)
        random.seed(all_args.seed)
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
        summary["training_started"] = True
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
        summary["training_finished"] = True
    except Exception as exc:  # noqa: BLE001 - smoke should report structured failure
        summary["failure_stage"] = "training_failed"
        warnings.append(f"Training plan bridge MAPPO smoke failed: {type(exc).__name__}: {exc}")
        warnings.append(traceback.format_exc())
    finally:
        if original_parse_config is not None:
            try:
                from envs.JSBSim.envs import env_base

                env_base.parse_config = original_parse_config
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to restore parse_config cleanly: {type(exc).__name__}: {exc}")
        if envs is not None:
            try:
                envs.close()
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to close training envs cleanly: {type(exc).__name__}: {exc}")

    actor_exists = actor_path.exists()
    critic_exists = critic_path.exists()
    summary["checkpoint_saved"] = bool(actor_exists and critic_exists)
    summary["actor_checkpoint_path"] = str(actor_path) if actor_exists else None
    summary["critic_checkpoint_path"] = str(critic_path) if critic_exists else None
    if summary["failure_stage"] is None and not summary["checkpoint_saved"]:
        summary["failure_stage"] = "checkpoint_not_saved"
        warnings.append("Training finished but actor_latest.pt and critic_latest.pt were not both found.")
    if actor_exists and actor_path.stat().st_mtime < start_time:
        warnings.append("actor_latest.pt exists but was not modified during this smoke run.")
    return summary


def _failure_stage(summary: Mapping[str, Any]) -> Optional[str]:
    if not summary.get("sampling_plan_loaded") and summary.get("fallback_used") is False:
        return "sampling_plan"
    if not summary.get("selected_scenario_yaml_exists"):
        return "scenario_yaml"
    if not summary.get("training_config_prepared"):
        return "training_config"
    if not summary.get("training_started"):
        return "training_start"
    if not summary.get("training_finished"):
        return "training"
    if not summary.get("checkpoint_saved"):
        return "checkpoint"
    return None


def main() -> None:
    cli = argparse.ArgumentParser(description="Smoke bridge FALCON sampling plan to MAPPO training entry.")
    cli.add_argument("--output-dir", default=str(ROOT_DIR / "tests" / "tmp_falcon_training_bridge"))
    cli.add_argument("--num-env-steps", type=int, default=8)
    cli.add_argument("--buffer-size", type=int, default=8)
    cli.add_argument("--seed", type=int, default=9)
    args = cli.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"
    warnings: list[str] = []
    adapter = TrainingPlanAdapter({"lag_config_root": str(ROOT_DIR / "envs" / "JSBSim" / "configs")})
    plan, sampling_plan_loaded, fallback_used = _load_or_build_plan(output_dir, base_config_path, warnings)
    selection = adapter.select_scenario(plan, strategy="weighted")
    warnings.extend(selection.get("warnings", []))
    selected = selection.get("selected_scenario") or {}
    selected_yaml = selected.get("scenario_yaml_path") or selected.get("yaml_path")
    selected_yaml_exists = bool(selected_yaml and Path(str(selected_yaml)).exists())
    manifest = adapter.prepare_training_config(selected, base_config_path=base_config_path, output_dir=output_dir)
    warnings.extend(manifest.get("warnings", []))
    manifest_path = output_dir / "training_plan_bridge_manifest.json"
    adapter.export_training_config_manifest(manifest, manifest_path)

    train_summary = {
        "training_started": False,
        "training_finished": False,
        "checkpoint_saved": False,
        "actor_checkpoint_path": None,
        "warnings": [],
    }
    if manifest.get("config_name_or_path"):
        train_summary = _train_smoke(
            str(manifest["config_name_or_path"]),
            output_dir,
            int(args.num_env_steps),
            int(args.buffer_size),
            int(args.seed),
            training_config_path=manifest.get("training_config_path"),
            requires_parse_config_patch=bool(manifest.get("requires_parse_config_patch")),
        )
    else:
        train_summary["warnings"].append("Training skipped because TrainingPlanAdapter did not produce config_name_or_path.")
    _write_json(output_dir / "training_plan_bridge_train_summary.json", train_summary)
    warnings.extend(train_summary.get("warnings", []))

    used_falcon_generated = _selected_is_falcon_generated(selected, base_config_path)
    summary = {
        "schema_version": "falcon.training_plan_bridge_smoke_summary.v1",
        "sampling_plan_loaded": sampling_plan_loaded,
        "selected_scenario_id": selected.get("scenario_id"),
        "selected_scenario_source": selected.get("source"),
        "selected_scenario_yaml_exists": selected_yaml_exists,
        "training_config_prepared": bool(manifest.get("config_name_or_path") and manifest.get("training_config_path")),
        "training_started": bool(train_summary.get("training_started")),
        "training_finished": bool(train_summary.get("training_finished")),
        "checkpoint_saved": bool(train_summary.get("checkpoint_saved")),
        "actor_checkpoint_path": train_summary.get("actor_checkpoint_path"),
        "used_falcon_generated_scenario": used_falcon_generated,
        "fallback_used": fallback_used,
        "manifest_path": str(manifest_path),
        "failure_stage": None,
        "warnings": sorted(set(warnings)),
    }
    summary["failure_stage"] = _failure_stage(summary)
    _write_json(output_dir / "training_plan_bridge_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
