"""Adapt FALCON sampling plans into LAG training-entry configs."""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import time
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

TRAINING_PLAN_ADAPTER_SCHEMA_VERSION = "falcon.training_plan_adapter.v1"

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "lag_config_root": "envs/JSBSim/configs",
    "staging_subdir": "falcon_training_smoke",
    "prefer_external_config_path": True,
}


class TrainingPlanAdapter:
    """Convert FALCON sampling-plan entries to ``scenario_name`` values."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))
        self.rng = random.Random(int(self.config.get("seed", 0)))

    def load_sampling_plan(self, path: Union[str, Path]) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, MappingABC):
            raise ValueError(f"Sampling plan JSON is not an object: {path}")
        return dict(data)

    def select_scenario(self, plan: Mapping[str, Any], strategy: str = "weighted") -> Dict[str, Any]:
        warnings = []
        scenarios = [dict(item) for item in plan.get("sampled_scenarios", []) if isinstance(item, MappingABC)]
        if not scenarios:
            return {
                "schema_version": "falcon.training_plan_selection.v1",
                "selected_scenario": None,
                "warnings": ["Sampling plan had no sampled_scenarios."],
            }
        if strategy == "first":
            selected = scenarios[0]
        else:
            weights = [_scenario_weight(item) for item in scenarios]
            if sum(weights) <= 0.0:
                warnings.append("All scenario weights were zero; selected uniformly.")
                weights = [1.0 for _ in scenarios]
            selected = _weighted_choice(self.rng, scenarios, weights)
        return {
            "schema_version": "falcon.training_plan_selection.v1",
            "selected_scenario": selected,
            "warnings": warnings,
        }

    def select_scenario_batch(self, plan: Mapping[str, Any]) -> Dict[str, Any]:
        scenarios = plan.get("scenario_batch")
        if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
            scenarios = plan.get("sampled_scenarios")
        batch = [dict(item) for item in (scenarios or []) if isinstance(item, MappingABC)]
        warnings: List[str] = []
        if not batch:
            warnings.append("Sampling plan had no scenario_batch or sampled_scenarios.")
        return {
            "schema_version": "falcon.training_plan_batch_selection.v1",
            "scenario_batch": batch,
            "scenario_batch_size": len(batch),
            "warnings": warnings,
        }

    def prepare_training_batch(
        self,
        plan: Mapping[str, Any],
        base_config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        selection = self.select_scenario_batch(plan)
        manifests: List[Dict[str, Any]] = []
        warnings = list(selection.get("warnings") or [])
        root = Path(output_dir) if output_dir is not None else None
        for index, scenario in enumerate(selection["scenario_batch"]):
            manifest = self.prepare_training_config(
                scenario,
                base_config_path=base_config_path,
                output_dir=(root / f"scenario_{index:03d}") if root is not None else None,
            )
            manifest["batch_index"] = index
            manifest["pool_item_id"] = scenario.get("pool_item_id")
            manifest["sampling_category"] = scenario.get("sampling_category")
            manifest["assigned_train_steps"] = int(scenario.get("assigned_train_steps", 0) or 0)
            manifest["anchor_role"] = scenario.get("anchor_role")
            warnings.extend(manifest.get("warnings") or [])
            manifests.append(manifest)
        return {
            "schema_version": "falcon.training_config_batch_manifest.v1",
            "scenario_batch_size": len(manifests),
            "total_train_steps": sum(int(item.get("assigned_train_steps", 0)) for item in manifests),
            "manifests": manifests,
            "warnings": sorted(set(warnings)),
        }

    def prepare_training_config(
        self,
        selected_scenario: Mapping[str, Any],
        base_config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        warnings = []
        scenario_id = str(selected_scenario.get("scenario_id") or "falcon_selected_scenario")
        scenario_yaml_path = selected_scenario.get("scenario_yaml_path") or selected_scenario.get("yaml_path")
        if not scenario_yaml_path and base_config_path:
            scenario_yaml_path = str(base_config_path)
            warnings.append("Selected scenario did not include scenario_yaml_path; fell back to base_config_path.")
        source_path = Path(str(scenario_yaml_path)) if scenario_yaml_path else None
        if source_path is None or not source_path.exists():
            return {
                "schema_version": "falcon.training_config_manifest.v1",
                "selected_scenario_id": scenario_id,
                "source": selected_scenario.get("source"),
                "scenario_yaml_path": str(source_path) if source_path else None,
                "training_config_path": None,
                "config_name_or_path": None,
                "scenario_config_path": None,
                "requires_parse_config_patch": False,
                "sampling_weight": _float(selected_scenario.get("sampling_weight")),
                "final_value_score": _float(selected_scenario.get("final_value_score")),
                "warnings": warnings + [f"Selected scenario YAML does not exist: {source_path}"],
            }

        lag_root = Path(self.config.get("lag_config_root", DEFAULT_CONFIG["lag_config_root"]))
        if not lag_root.is_absolute():
            lag_root = Path.cwd() / lag_root
        lag_root = lag_root.resolve()
        source_resolved = source_path.resolve()

        config_name = _relative_config_name(source_resolved, lag_root)
        training_config_path = source_resolved
        requires_patch = False
        scenario_config_path = str(source_resolved)
        if bool(self.config.get("prefer_external_config_path", True)):
            config_name = config_name or str(source_resolved)
        elif config_name is None:
            staging_root = lag_root / str(self.config.get("staging_subdir", DEFAULT_CONFIG["staging_subdir"]))
            safe_name = _safe_name(scenario_id)
            try:
                staging_root.mkdir(parents=True, exist_ok=True)
                training_config_path = (staging_root / f"{safe_name}.yaml").resolve()
                if source_resolved != training_config_path:
                    shutil.copyfile(source_resolved, training_config_path)
                config_name = _relative_config_name(training_config_path, lag_root)
                warnings.append(
                    "LAG training entry does not read arbitrary external YAML paths; copied selected YAML into envs/JSBSim/configs."
                )
            except OSError as exc:
                if output_dir is None:
                    raise
                fallback_root = Path(output_dir) / str(self.config.get("staging_subdir", DEFAULT_CONFIG["staging_subdir"]))
                fallback_root.mkdir(parents=True, exist_ok=True)
                training_config_path = (fallback_root / f"{safe_name}.yaml").resolve()
                if source_resolved != training_config_path:
                    shutil.copyfile(source_resolved, training_config_path)
                config_name = f"falcon_memory_training/{safe_name}"
                requires_patch = True
                warnings.append(
                    "Could not copy selected YAML into envs/JSBSim/configs "
                    f"({type(exc).__name__}: {exc}); prepared output-dir YAML and requires a temporary parse_config mapping."
                )
                scenario_config_path = str(training_config_path)
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {
            "schema_version": "falcon.training_config_manifest.v1",
            "selected_scenario_id": scenario_id,
            "source": selected_scenario.get("source"),
            "scenario_yaml_path": str(source_resolved),
            "training_config_path": str(training_config_path),
            "config_name_or_path": config_name,
            "scenario_config_path": scenario_config_path,
            "requires_parse_config_patch": requires_patch,
            "sampling_weight": _float(selected_scenario.get("sampling_weight")),
            "final_value_score": _float(selected_scenario.get("final_value_score")),
            "warnings": warnings,
        }

    def export_training_config_manifest(self, manifest: Mapping[str, Any], output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(dict(manifest), f, indent=2, sort_keys=True)


class MultiScenarioTrainingBridge:
    """Sequentially continue one MAPPO checkpoint across a scenario batch."""

    def __init__(
        self,
        adapter: Optional[TrainingPlanAdapter] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = _deep_merge(
            {
                "seed": 0,
                "default_per_scenario_train_steps": 8,
                "minimum_train_steps": 1,
                "preserve_best_within_batch": False,
                "round_checkpoint_selection": "terminal",
            },
            dict(config or {}),
        )
        self.adapter = adapter or TrainingPlanAdapter({"seed": self.config.get("seed", 0)})

    def run_batch(
        self,
        plan: Mapping[str, Any],
        train_fn: Callable[..., Mapping[str, Any]],
        output_dir: Union[str, Path],
        base_config_path: Optional[Union[str, Path]] = None,
        initial_checkpoint_path: Optional[Union[str, Path]] = None,
        round_id: int = 0,
        curriculum_pool: Any = None,
        checkpoint_validation_fn: Optional[
            Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        batch_manifest = self.adapter.prepare_training_batch(
            plan,
            base_config_path=base_config_path,
            output_dir=output_root / "prepared_configs",
        )
        self.adapter.export_training_config_manifest(
            batch_manifest,
            output_root / "scenario_batch_manifest.json",
        )
        warnings = list(batch_manifest.get("warnings") or [])
        current_checkpoint = str(initial_checkpoint_path) if initial_checkpoint_path else None
        results: List[Dict[str, Any]] = []
        successful_records: List[Dict[str, Any]] = []
        checkpoint_candidates: List[Dict[str, Any]] = []
        preserve_best = bool(self.config.get("preserve_best_within_batch", False))

        if preserve_best and checkpoint_validation_fn is not None and current_checkpoint:
            initial_validation = _run_checkpoint_validation(
                checkpoint_validation_fn,
                current_checkpoint,
                {
                    "batch_index": -1,
                    "round_id": int(round_id),
                    "scenario_id": "round_input_checkpoint",
                    "sampling_category": "round_input",
                    "anchor_role": "preserved_input",
                },
            )
            checkpoint_candidates.append(
                {
                    "batch_index": -1,
                    "checkpoint_path": current_checkpoint,
                    "scenario_id": "round_input_checkpoint",
                    "sampling_category": "round_input",
                    "anchor_role": "preserved_input",
                    "validation": initial_validation,
                }
            )

        for index, manifest in enumerate(batch_manifest.get("manifests") or []):
            steps = int(
                manifest.get("assigned_train_steps")
                or self.config.get("default_per_scenario_train_steps", 8)
            )
            steps = max(steps, int(self.config.get("minimum_train_steps", 1)))
            scenario_result: Dict[str, Any] = {
                "schema_version": "falcon.multi_scenario_training_item.v1",
                "batch_index": index,
                "round_id": int(round_id),
                "scenario_id": manifest.get("selected_scenario_id"),
                "pool_item_id": manifest.get("pool_item_id"),
                "scenario_yaml_path": manifest.get("scenario_yaml_path"),
                "source": manifest.get("source"),
                "sampling_category": manifest.get("sampling_category"),
                "anchor_role": manifest.get("anchor_role"),
                "train_steps": steps,
                "input_checkpoint_path": current_checkpoint,
                "training_succeeded": False,
                "checkpoint_saved": False,
                "output_checkpoint_path": None,
                "failure_stage": None,
                "warnings": [],
            }
            if not manifest.get("training_config_path"):
                scenario_result["failure_stage"] = "training_config"
                scenario_result["warnings"].append("Training config was not prepared; skipped scenario.")
                results.append(scenario_result)
                warnings.extend(scenario_result["warnings"])
                continue
            try:
                train_summary = dict(
                    train_fn(
                        scenario_name=str(manifest.get("config_name_or_path")),
                        output_dir=output_root / f"scenario_{index:03d}_training",
                        num_env_steps=steps,
                        buffer_size=steps,
                        seed=int(self.config.get("seed", 0)) + int(round_id) * 100 + index,
                        training_config_path=manifest.get("training_config_path"),
                        scenario_config_path=manifest.get("scenario_config_path"),
                        requires_parse_config_patch=bool(manifest.get("requires_parse_config_patch")),
                        model_dir=_checkpoint_model_dir(current_checkpoint),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                train_summary = {
                    "failure_stage": "training_exception",
                    "checkpoint_saved": False,
                    "warnings": [f"Sequential training callback failed: {type(exc).__name__}: {exc}"],
                }
            output_checkpoint = train_summary.get("actor_checkpoint_path")
            checkpoint_saved = bool(
                train_summary.get("checkpoint_saved")
                and output_checkpoint
                and Path(str(output_checkpoint)).exists()
            )
            scenario_result.update(
                {
                    "training_succeeded": bool(
                        train_summary.get("training_finished") and checkpoint_saved
                    ),
                    "checkpoint_saved": checkpoint_saved,
                    "output_checkpoint_path": output_checkpoint if checkpoint_saved else None,
                    "failure_stage": train_summary.get("failure_stage"),
                    "train_summary": train_summary,
                    "warnings": list(train_summary.get("warnings") or []),
                }
            )
            if scenario_result["training_succeeded"]:
                current_checkpoint = str(output_checkpoint)
                successful_records.append(scenario_result)
                if checkpoint_validation_fn is not None:
                    checkpoint_validation = _run_checkpoint_validation(
                        checkpoint_validation_fn,
                        current_checkpoint,
                        scenario_result,
                    )
                    scenario_result["checkpoint_validation"] = checkpoint_validation
                    checkpoint_candidates.append(
                        {
                            "batch_index": index,
                            "checkpoint_path": current_checkpoint,
                            "scenario_id": scenario_result.get("scenario_id"),
                            "sampling_category": scenario_result.get(
                                "sampling_category"
                            ),
                            "anchor_role": scenario_result.get("anchor_role"),
                            "validation": checkpoint_validation,
                        }
                    )
                if curriculum_pool is not None and hasattr(curriculum_pool, "record_training"):
                    updated = curriculum_pool.record_training(
                        scenario_result,
                        round_id=int(round_id),
                        train_steps=steps,
                    )
                    scenario_result["pool_coverage_updated"] = updated is not None
                    if updated is not None:
                        scenario_result["coverage_status_after"] = updated.get("coverage_status")
                        scenario_result["train_count_after"] = updated.get("train_count")
            results.append(scenario_result)
            warnings.extend(scenario_result["warnings"])

        category_counts: Dict[str, int] = {}
        for item in successful_records:
            category = str(item.get("sampling_category") or "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        successful_count = len(successful_records)
        terminal_checkpoint = current_checkpoint
        selected_candidate = _select_checkpoint_candidate(
            checkpoint_candidates,
            terminal_checkpoint=terminal_checkpoint,
            preserve_best=preserve_best,
        )
        selected_checkpoint = (
            selected_candidate.get("checkpoint_path")
            if selected_candidate
            else terminal_checkpoint
        )
        return {
            "schema_version": "falcon.multi_scenario_training_summary.v1",
            "round_id": int(round_id),
            "scenario_batch_size": len(batch_manifest.get("manifests") or []),
            "scenarios_actually_trained": successful_count,
            "scenarios_failed": len(results) - successful_count,
            "training_results": results,
            "initial_checkpoint_path": str(initial_checkpoint_path) if initial_checkpoint_path else None,
            "latest_checkpoint_path": selected_checkpoint,
            "terminal_checkpoint_path": terminal_checkpoint,
            "selected_checkpoint_path": selected_checkpoint,
            "selected_batch_index": (
                selected_candidate.get("batch_index") if selected_candidate else None
            ),
            "selected_checkpoint_validation": (
                selected_candidate.get("validation") if selected_candidate else None
            ),
            "checkpoint_candidates": checkpoint_candidates,
            "preserve_best_within_batch": preserve_best,
            "round_checkpoint_selection": self.config.get(
                "round_checkpoint_selection", "terminal"
            ),
            "checkpoint_saved": bool(successful_count and selected_checkpoint),
            "checkpoint_continuity_complete": all(
                not item.get("training_succeeded")
                or index == 0
                or item.get("input_checkpoint_path")
                == _previous_successful_checkpoint(results, index)
                for index, item in enumerate(results)
            ),
            "anchor_scenarios_used": [
                item.get("scenario_id")
                for item in successful_records
                if item.get("sampling_category") == "base_anchor"
            ],
            "anchor_ratio": round(category_counts.get("base_anchor", 0) / successful_count, 6)
            if successful_count
            else 0.0,
            "accepted_ratio": round(category_counts.get("accepted_llm", 0) / successful_count, 6)
            if successful_count
            else 0.0,
            "replay_ratio": round(category_counts.get("replay_failure", 0) / successful_count, 6)
            if successful_count
            else 0.0,
            "total_train_steps_completed": sum(
                int(item.get("train_steps", 0)) for item in successful_records
            ),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "failure_stage": None if successful_count else "all_scenarios_failed",
            "warnings": sorted(set(warnings)),
        }


def _relative_config_name(path: Path, lag_config_root: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(lag_config_root.resolve())
    except ValueError:
        return None
    if rel.suffix.lower() != ".yaml":
        return None
    return rel.with_suffix("").as_posix()


def _checkpoint_model_dir(checkpoint_path: Optional[Union[str, Path]]) -> Optional[str]:
    if not checkpoint_path:
        return None
    path = Path(str(checkpoint_path))
    if path.is_dir():
        candidate_dir = path
    else:
        candidate_dir = path.parent
    actor = candidate_dir / "actor_latest.pt"
    critic = candidate_dir / "critic_latest.pt"
    return str(candidate_dir) if actor.exists() and critic.exists() else None


def _previous_successful_checkpoint(results: Sequence[Mapping[str, Any]], index: int) -> Optional[str]:
    for item in reversed(results[:index]):
        if item.get("training_succeeded") and item.get("output_checkpoint_path"):
            return str(item["output_checkpoint_path"])
    current = results[index].get("input_checkpoint_path") if index < len(results) else None
    return str(current) if current else None


def _run_checkpoint_validation(
    validation_fn: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    checkpoint_path: str,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        result = dict(validation_fn(checkpoint_path, context))
        result.setdefault("failure_stage", None)
        result.setdefault("warnings", [])
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "win_rate": None,
            "mean_return": None,
            "failure_stage": "checkpoint_validation_exception",
            "warnings": [
                f"Checkpoint validation failed: {type(exc).__name__}: {exc}"
            ],
        }


def _select_checkpoint_candidate(
    candidates: Sequence[Mapping[str, Any]],
    terminal_checkpoint: Optional[str],
    preserve_best: bool,
) -> Optional[Dict[str, Any]]:
    valid = [
        dict(item)
        for item in candidates
        if item.get("checkpoint_path")
        and (item.get("validation") or {}).get("failure_stage") is None
        and _optional_float((item.get("validation") or {}).get("win_rate")) is not None
    ]
    if preserve_best and valid:
        return max(
            valid,
            key=lambda item: (
                _score_value(
                    (item.get("validation") or {}).get("win_rate"), 0.0
                ),
                _score_value(
                    (item.get("validation") or {}).get("mean_return"),
                    float("-inf"),
                ),
                -int(item.get("batch_index", -1)),
            ),
        )
    for item in reversed(candidates):
        if str(item.get("checkpoint_path")) == str(terminal_checkpoint):
            return dict(item)
    return None


def _optional_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _score_value(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _scenario_weight(item: Mapping[str, Any]) -> float:
    for key in ("sampling_weight", "final_value_score"):
        value = _float(item.get(key))
        if value > 0.0:
            return value
    return 1.0


def _weighted_choice(rng: random.Random, items: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> Dict[str, Any]:
    total = sum(max(_float(weight), 0.0) for weight in weights)
    if total <= 0.0:
        return dict(items[rng.randrange(len(items))])
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += max(_float(weight), 0.0)
        if cumulative >= threshold:
            return dict(item)
    return dict(items[-1])


def _safe_name(value: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "scenario")).strip("._")
    return value or "scenario"


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
