"""Preparation-stage baseline experiment runner utilities."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from .candidate_schema import validate_candidate_schema
from .constraint_checker import ConstraintChecker
from .curriculum_pool import CurriculumPool
from .curriculum_scheduler import CurriculumScheduler
from .falcon_controller import FalconController, _train_mappo_smoke
from .llm_scenario_generator import QwenScenarioGenerator
from .policy_evaluator import PolicyEvaluator
from .random_scenario_generator import RandomScenarioGenerator
from .scenario_adapter import (
    apply_initial_config_to_yaml,
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    load_base_scenario_config,
    save_scenario_yaml,
)
from .training_plan_adapter import TrainingPlanAdapter

ROOT_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_GROUPS = (
    "mappo_base",
    "mappo_random_curriculum",
    "mappo_qwen_only",
    "falcon_no_fsn",
)
GROUP_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "mappo_base": {
        "qwen_used": False,
        "random_used": False,
        "failure_analyzer_used": False,
        "constraint_checker_used": False,
        "policy_evaluator_used_for_filtering": False,
        "difficulty_evaluator_used": False,
        "curriculum_pool_used": False,
        "curriculum_scheduler_used": False,
        "falcon_full_pipeline_used": False,
    },
    "mappo_random_curriculum": {
        "qwen_used": False,
        "random_used": True,
        "failure_analyzer_used": False,
        "constraint_checker_used": True,
        "policy_evaluator_used_for_filtering": False,
        "difficulty_evaluator_used": False,
        "curriculum_pool_used": True,
        "curriculum_scheduler_used": True,
        "falcon_full_pipeline_used": False,
    },
    "mappo_qwen_only": {
        "qwen_used": True,
        "random_used": False,
        "failure_analyzer_used": False,
        "constraint_checker_used": True,
        "policy_evaluator_used_for_filtering": False,
        "difficulty_evaluator_used": False,
        "curriculum_pool_used": True,
        "curriculum_scheduler_used": True,
        "falcon_full_pipeline_used": False,
    },
    "falcon_no_fsn": {
        "qwen_used": True,
        "random_used": False,
        "failure_analyzer_used": True,
        "constraint_checker_used": True,
        "policy_evaluator_used_for_filtering": True,
        "difficulty_evaluator_used": True,
        "curriculum_pool_used": True,
        "curriculum_scheduler_used": True,
        "falcon_full_pipeline_used": True,
    },
}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, MappingABC):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return dict(data)


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


class BaselineExperimentRunner:
    """Resolve and smoke-test one frozen baseline group."""

    def __init__(
        self,
        protocol_path: str | Path,
        group: str,
        seed: int,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        if group not in SUPPORTED_GROUPS:
            raise ValueError(f"Unsupported group {group!r}; choose from {SUPPORTED_GROUPS}.")
        self.protocol_path = Path(protocol_path)
        self.protocol = load_yaml(self.protocol_path)
        self.group = group
        self.seed = int(seed)
        group_path = _resolve_path(self.protocol["group_configs"][group])
        self.group_config_path = group_path
        self.group_config = load_yaml(group_path)
        root = _resolve_path(self.protocol["output_root"])
        self._explicit_output_dir = Path(output_dir) if output_dir else None
        self._seed_output_dir = self._explicit_output_dir or root / group / f"seed_{self.seed}"
        self.output_dir = self._seed_output_dir
        self.warnings: List[str] = []

    def dry_run(self) -> Dict[str, Any]:
        self._select_mode_output("dry_run")
        result = self._base_result("dry_run")
        checks = self._validate_inputs()
        result.update(
            {
                "dry_run": True,
                "smoke_run": False,
                "input_checks": checks,
                "ready": all(checks.values()),
                "failure_stage": None if all(checks.values()) else "input_validation",
            }
        )
        self._save_outputs(result)
        return result

    def smoke_run(self) -> Dict[str, Any]:
        self._select_mode_output("smoke_run")
        result = self._base_result("smoke_run")
        checks = self._validate_inputs()
        result["input_checks"] = checks
        if not all(checks.values()):
            result.update({"ready": False, "failure_stage": "input_validation"})
            self._save_outputs(result)
            return result

        if self.group == "mappo_base":
            execution = self._run_mappo_base_smoke()
        elif self.group == "mappo_random_curriculum":
            execution = self._run_validity_curriculum_smoke(generator_type="random")
        elif self.group == "mappo_qwen_only":
            execution = self._run_validity_curriculum_smoke(generator_type="qwen")
        elif self.group == "falcon_no_fsn":
            execution = self._run_falcon_smoke()
        else:
            execution = self._prepare_curriculum_group_smoke()
        result.update(
            {
                "dry_run": False,
                "smoke_run": True,
                "ready": execution.get("failure_stage") is None,
                "execution": execution,
                "checkpoint_path": execution.get("checkpoint_path"),
                "failure_stage": execution.get("failure_stage"),
                "warnings": sorted(set(self.warnings + list(execution.get("warnings") or []))),
            }
        )
        self._save_outputs(result)
        return result

    def pilot_run(self, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        self._select_mode_output("pilot_run")
        result = self._base_result("pilot_run")
        checks = self._validate_inputs()
        result["input_checks"] = checks
        if not all(checks.values()):
            result.update({"ready": False, "pilot_run": True, "failure_stage": "input_validation"})
            self._save_outputs(result)
            return result
        params = self._pilot_params(overrides)
        if self.group == "falcon_no_fsn":
            execution = self._run_falcon_pilot(params)
        else:
            execution = self._run_simple_group_pilot(params)
        result.update(
            {
                "dry_run": False,
                "smoke_run": False,
                "pilot_run": True,
                "ready": execution.get("failure_stage") is None,
                "protocol_parameters": params,
                "execution": execution,
                "checkpoint_path": execution.get("latest_checkpoint_path"),
                "best_checkpoint_path": execution.get("best_checkpoint_path"),
                "failure_stage": execution.get("failure_stage"),
                "warnings": sorted(set(self.warnings + list(execution.get("warnings") or []))),
            }
        )
        self._save_outputs(result)
        return result

    def _run_mappo_base_smoke(self) -> Dict[str, Any]:
        smoke = self.protocol.get("smoke") or {}
        steps = int(smoke.get("train_steps_per_round", 8))
        summary = _train_mappo_smoke(
            scenario_name=str(self.protocol["environment"]),
            output_dir=self.output_dir / "smoke_training",
            num_env_steps=steps,
            buffer_size=steps,
            seed=self.seed,
        )
        checkpoint = summary.get("actor_checkpoint_path")
        evaluation = self._evaluate_checkpoint(checkpoint)
        return {
            "schema_version": "falcon.baseline_group_smoke.v1",
            "group": self.group,
            **GROUP_DEFINITIONS[self.group],
            "training_summary": summary,
            "evaluation": evaluation,
            "checkpoint_path": checkpoint,
            "failure_stage": summary.get("failure_stage"),
            "warnings": list(summary.get("warnings") or []) + list(evaluation.get("warnings") or []),
        }

    def _run_falcon_smoke(self) -> Dict[str, Any]:
        smoke = self.protocol.get("smoke") or {}
        steps = int(smoke.get("train_steps_per_round", 8))
        controller_dir = self.output_dir / "smoke_controller"
        config = {
            "output_dir": str(controller_dir),
            "base_config_path": str(_resolve_path(self.protocol["base_scenario_config"])),
            "max_rounds": int(smoke.get("max_rounds", 1)),
            "train_steps_per_round": steps,
            "eval_episodes_per_round": int(smoke.get("eval_episodes_per_round", 1)),
            "qwen_candidates_per_round": int(smoke.get("qwen_candidates_per_round", 1)),
            "policy_eval_episodes_per_candidate": int(smoke.get("policy_eval_episodes_per_candidate", 1)),
            "save_every_round": True,
            "use_real_failure_trajectory": True,
            "initial_training": {
                "num_env_steps": steps,
                "buffer_size": steps,
                "seed": self.seed,
                "scenario_name": str(self.protocol["environment"]),
            },
            "round1_training": {
                "num_env_steps": steps,
                "buffer_size": steps,
                "seed": self.seed + 1000,
            },
            "coverage_aware_training": dict(
                self.protocol.get("coverage_aware_training") or {}
            ),
            "stability_aware_training": dict(
                self.protocol.get("stability_aware_training") or {}
            ),
            "qwen": dict(self.protocol.get("llm") or {}),
        }
        controller = FalconController(config)
        controller_result = controller.run(max_rounds=config["max_rounds"])
        checkpoint = controller.state.get("latest_checkpoint_path")
        evaluation = self._evaluate_checkpoint(checkpoint)
        return {
            "schema_version": "falcon.baseline_group_smoke.v1",
            "group": self.group,
            **GROUP_DEFINITIONS[self.group],
            "controller_result": controller_result,
            "evaluation": evaluation,
            "checkpoint_path": checkpoint,
            "failure_stage": None if controller_result.get("completed_rounds") == config["max_rounds"] else "controller_rounds",
            "warnings": list(controller_result.get("warnings") or []) + list(evaluation.get("warnings") or []),
        }

    def _run_falcon_pilot(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        pilot_started_at = _timestamp()
        pilot_start = time.perf_counter()
        steps = int(params["train_steps_per_round"])
        controller_dir = self.output_dir / "controller"
        config = {
            "output_dir": str(controller_dir),
            "base_config_path": str(_resolve_path(self.protocol["base_scenario_config"])),
            "best_checkpoint_path": str(self.output_dir / "_no_preexisting_best_checkpoint.pt"),
            "max_rounds": int(params["max_rounds"]),
            "train_steps_per_round": steps,
            "eval_episodes_per_round": int(params["eval_episodes_per_round"]),
            "qwen_candidates_per_round": int(params["qwen_candidates_per_round"]),
            "policy_eval_episodes_per_candidate": int(params["policy_eval_episodes_per_candidate"]),
            "save_every_round": True,
            "use_real_failure_trajectory": True,
            "initial_training": {
                "num_env_steps": steps,
                "buffer_size": steps,
                "seed": self.seed,
                "scenario_name": str(self.protocol["environment"]),
            },
            "round1_training": {
                "num_env_steps": steps,
                "buffer_size": steps,
                "seed": self.seed + 1000,
            },
            "coverage_aware_training": dict(
                self.protocol.get("coverage_aware_training") or {}
            ),
            "stability_aware_training": dict(
                self.protocol.get("stability_aware_training") or {}
            ),
            "qwen": dict(self.protocol.get("llm") or {}),
        }
        if params.get("resume_from_state"):
            config["resume_from_state"] = params.get("resume_from_state")
        controller = FalconController(config)
        controller_result = controller.run(max_rounds=int(params["max_rounds"]))
        pilot_runtime = round(time.perf_counter() - pilot_start, 3)
        latest = controller.state.get("latest_checkpoint_path")
        best = controller.state.get("best_checkpoint_path") or latest
        registry_path = controller_dir / "falcon_checkpoint_registry.json"
        round_summaries = []
        for round_id in range(int(params["max_rounds"])):
            path = controller_dir / f"falcon_controller_round{round_id}_summary.json"
            if path.exists():
                round_summary = _load_json(path)
                if round_summary.get("round_runtime_seconds") is None:
                    round_summary.update(_infer_round_runtime_from_files(controller_dir, round_id))
                round_summaries.append(round_summary)
        summary = {
            "schema_version": "falcon.baseline_pilot_summary.v1",
            "group": self.group,
            "seed": self.seed,
            **GROUP_DEFINITIONS[self.group],
            "started_at": pilot_started_at,
            "finished_at": _timestamp(),
            "runtime_seconds": pilot_runtime,
            "runtime_human_readable": _human_duration(pilot_runtime),
            "protocol_parameters": dict(params),
            "completed_rounds": int(controller_result.get("completed_rounds", 0)),
            "max_rounds": int(params["max_rounds"]),
            "all_rounds_finished": int(controller_result.get("completed_rounds", 0)) >= int(params["max_rounds"]),
            "round_summaries": round_summaries,
            "checkpoint_registry_path": str(registry_path),
            "latest_checkpoint_path": latest,
            "best_checkpoint_path": best,
            "failure_stage": None
            if int(controller_result.get("completed_rounds", 0)) >= int(params["max_rounds"])
            else "controller_rounds",
            "warnings": list(controller_result.get("warnings") or []),
        }
        write_json(self.output_dir / "pilot_run_summary.json", summary)
        return summary

    def _run_simple_group_pilot(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        pilot_started_at = _timestamp()
        pilot_start = time.perf_counter()
        base_path = _resolve_path(self.protocol["base_scenario_config"])
        base_config = load_base_scenario_config(base_path)
        pool = CurriculumPool()
        available_random: List[Dict[str, Any]] = []
        previous_checkpoint: Optional[str] = None
        registry: Dict[str, Any] = {
            "schema_version": "falcon.baseline_checkpoint_registry.v1",
            "group": self.group,
            "seed": self.seed,
            "checkpoints": [],
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "best_win_rate": None,
        }
        round_summaries: List[Dict[str, Any]] = []
        warnings: List[str] = []
        failure_stage: Optional[str] = None

        for round_id in range(int(params["max_rounds"])):
            round_started_at = _timestamp()
            round_start = time.perf_counter()
            generation_runtime = 0.0
            qwen_runtime = 0.0
            validation_runtime = 0.0
            sampling_runtime = 0.0
            training_runtime = 0.0
            policy_eval_runtime = 0.0
            round_dir = self.output_dir / f"round_{round_id:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            selected = {
                "scenario_id": "base_2v2_NoWeapon_Selfplay",
                "source": "original",
                "scenario_yaml_path": str(base_path),
                "sampling_weight": 1.0,
                "final_value_score": 0.0,
                "target_failure_modes": [],
                "priority_level": "base",
            }
            generation_result: Dict[str, Any] = {"candidates": [], "warnings": []}
            validation: Dict[str, Any] = {"valid_candidates": [], "warnings": []}
            plan: Dict[str, Any] = {"sampled_scenarios": [selected], "warnings": []}

            if self.group in {"mappo_random_curriculum", "mappo_qwen_only"}:
                generator_type = "random" if self.group == "mappo_random_curriculum" else "qwen"
                requested = int(
                    params["random_candidates_per_round"]
                    if generator_type == "random"
                    else params["qwen_candidates_per_round"]
                )
                generation_start = time.perf_counter()
                candidates, generation_result = self._generate_validity_candidates(
                    generator_type,
                    requested,
                    base_config,
                    round_id,
                )
                generation_runtime = round(time.perf_counter() - generation_start, 3)
                if generator_type == "qwen":
                    qwen_runtime = generation_runtime
                validation_start = time.perf_counter()
                valid, validation = self._validate_and_materialize_candidates(candidates, output_dir=round_dir)
                validation_runtime = round(time.perf_counter() - validation_start, 3)
                write_json(round_dir / "candidate_generation.json", generation_result)
                write_json(round_dir / "candidate_validation.json", validation)
                if valid:
                    acceptance_results = [_validity_only_acceptance(candidate) for candidate in valid]
                    source = "random" if generator_type == "random" else "llm_qwen8b"
                    pool.add_batch(valid, acceptance_results, source=source, source_round=round_id)
                    if generator_type == "random":
                        available_random.extend(valid)
                    scheduler = CurriculumScheduler(
                        {
                            "seed": self.seed + round_id,
                            "category_ratios": {
                                "base": 0.0,
                                "random": 1.0 if generator_type == "random" else 0.0,
                                "llm_qwen8b": 1.0 if generator_type == "qwen" else 0.0,
                                "replay": 0.0,
                            },
                        }
                    )
                    sampling_start = time.perf_counter()
                    plan = scheduler.build_sampling_plan(
                        pool,
                        random_scenarios=available_random if generator_type == "random" else None,
                        num_samples=max(1, len(valid)),
                    )
                    selection = TrainingPlanAdapter({"seed": self.seed + round_id}).select_scenario(plan)
                    sampling_runtime = round(time.perf_counter() - sampling_start, 3)
                    selected = selection.get("selected_scenario") or selected
                else:
                    warnings.append(f"Round {round_id} produced no valid {generator_type} candidates; used base scenario.")
                pool.save(self.output_dir / "curriculum_pool.json")
                CurriculumScheduler().save_sampling_plan(plan, round_dir / "sampling_plan.json")

            adapter = TrainingPlanAdapter({"seed": self.seed + round_id})
            manifest = adapter.prepare_training_config(selected, base_config_path=base_path, output_dir=round_dir / "training_config")
            adapter.export_training_config_manifest(manifest, round_dir / "training_manifest.json")
            training_start = time.perf_counter()
            train_summary = _train_mappo_smoke(
                scenario_name=str(manifest.get("config_name_or_path") or self.protocol["environment"]),
                output_dir=round_dir / "training",
                num_env_steps=int(params["train_steps_per_round"]),
                buffer_size=int(params["train_steps_per_round"]),
                seed=self.seed + round_id,
                training_config_path=manifest.get("training_config_path"),
                scenario_config_path=manifest.get("scenario_config_path"),
                requires_parse_config_patch=bool(manifest.get("requires_parse_config_patch")),
                model_dir=_checkpoint_model_dir(previous_checkpoint),
            )
            training_runtime = round(time.perf_counter() - training_start, 3)
            checkpoint = train_summary.get("actor_checkpoint_path")
            policy_eval_start = time.perf_counter()
            round_eval = self._evaluate_checkpoint(
                checkpoint,
                scenario_limit=1,
                episodes_per_scenario=int(params["eval_episodes_per_round"]),
            )
            policy_eval_runtime = round(time.perf_counter() - policy_eval_start, 3)
            round_win_rate = _mean_eval_metric(round_eval.get("results") or [], "win_rate")
            registry["checkpoints"].append(
                {
                    "round_id": round_id,
                    "checkpoint_path": checkpoint,
                    "exists": bool(checkpoint and Path(checkpoint).exists()),
                    "continued_from_checkpoint": bool(previous_checkpoint),
                    "eval_win_rate": round_win_rate,
                }
            )
            registry["latest_checkpoint"] = checkpoint
            if registry["best_win_rate"] is None or round_win_rate > float(registry["best_win_rate"]):
                registry["best_win_rate"] = round_win_rate
                registry["best_checkpoint"] = checkpoint
            round_summary = {
                "schema_version": "falcon.baseline_pilot_round.v1",
                "group": self.group,
                "round_id": round_id,
                "started_at": round_started_at,
                "finished_at": _timestamp(),
                "round_runtime_seconds": round(time.perf_counter() - round_start, 3),
                "round_runtime_human_readable": _human_duration(time.perf_counter() - round_start),
                "candidate_generation_runtime_seconds": generation_runtime,
                "qwen_runtime_seconds": qwen_runtime,
                "candidate_validation_runtime_seconds": validation_runtime,
                "sampling_runtime_seconds": sampling_runtime,
                "training_runtime_seconds": training_runtime,
                "policy_eval_runtime_seconds": policy_eval_runtime,
                "selected_scenario_id": selected.get("scenario_id"),
                "selected_scenario_source": selected.get("source"),
                "num_candidates_generated": len(generation_result.get("candidates") or []),
                "num_candidates_valid": len(validation.get("valid_candidates") or []),
                "sampling_plan_generated": bool(plan.get("sampled_scenarios")),
                "continued_from_checkpoint": bool(previous_checkpoint),
                "training_summary": train_summary,
                "round_evaluation": round_eval,
                "round_win_rate": round_win_rate,
                "checkpoint_path": checkpoint,
                "failure_stage": train_summary.get("failure_stage"),
                "warnings": sorted(
                    set(
                        list(generation_result.get("warnings") or [])
                        + list(validation.get("warnings") or [])
                        + list(plan.get("warnings") or [])
                        + list(train_summary.get("warnings") or [])
                    )
                ),
            }
            write_json(round_dir / "round_summary.json", round_summary)
            round_summaries.append(round_summary)
            warnings.extend(round_summary["warnings"])
            if train_summary.get("failure_stage") or not checkpoint:
                failure_stage = f"round_{round_id}_training"
                break
            previous_checkpoint = checkpoint

        write_json(self.output_dir / "checkpoint_registry.json", registry)
        summary = {
            "schema_version": "falcon.baseline_pilot_summary.v1",
            "group": self.group,
            "seed": self.seed,
            **GROUP_DEFINITIONS[self.group],
            "started_at": pilot_started_at,
            "finished_at": _timestamp(),
            "runtime_seconds": round(time.perf_counter() - pilot_start, 3),
            "runtime_human_readable": _human_duration(time.perf_counter() - pilot_start),
            "training_runtime_seconds": _sum_safe_float(round_summaries, "training_runtime_seconds"),
            "policy_eval_runtime_seconds": _sum_safe_float(round_summaries, "policy_eval_runtime_seconds"),
            "qwen_runtime_seconds": _sum_safe_float(round_summaries, "qwen_runtime_seconds"),
            "protocol_parameters": dict(params),
            "completed_rounds": len(round_summaries),
            "max_rounds": int(params["max_rounds"]),
            "all_rounds_finished": len(round_summaries) == int(params["max_rounds"]) and failure_stage is None,
            "round_summaries": round_summaries,
            "checkpoint_registry_path": str(self.output_dir / "checkpoint_registry.json"),
            "latest_checkpoint_path": registry.get("latest_checkpoint"),
            "best_checkpoint_path": registry.get("best_checkpoint"),
            "failure_stage": failure_stage,
            "warnings": sorted(set(warnings)),
        }
        write_json(self.output_dir / "pilot_run_summary.json", summary)
        return summary

    def _generate_validity_candidates(
        self,
        generator_type: str,
        requested: int,
        base_config: Mapping[str, Any],
        round_id: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if generator_type == "random":
            candidates = _scope_candidate_ids(
                RandomScenarioGenerator({"seed": self.seed + round_id}).generate_from_base(base_config, requested),
                self.group,
                round_id,
            )
            return candidates, {
                "schema_version": "falcon.random_baseline_generation.v1",
                "requested_num_scenarios": requested,
                "candidates": candidates,
                "warnings": [],
            }
        generator = QwenScenarioGenerator(dict(self.protocol.get("llm") or {}))
        health = generator.check_llm_server()
        candidates: List[Dict[str, Any]] = []
        warnings = list(health.get("warnings") or [])
        if health.get("server_reachable") and health.get("model_available"):
            candidates = _scope_candidate_ids(
                generator.generate_from_failure_summary(
                    _generic_qwen_task_context(base_config),
                    base_config,
                    num_scenarios=requested,
                ),
                self.group,
                round_id,
            )
        else:
            warnings.append("Qwen-only generation skipped because Ollama qwen3:8b health check failed.")
        return candidates, {
            "schema_version": "falcon.qwen_only_baseline_generation.v1",
            "generic_task_context_used": True,
            "failure_analyzer_used": False,
            "health": health,
            "generator_result": generator.last_result,
            "candidates": candidates,
            "warnings": sorted(set(warnings + list(generator.last_result.get("warnings") or []))),
        }

    def _run_validity_curriculum_smoke(self, generator_type: str) -> Dict[str, Any]:
        smoke = self.protocol.get("smoke") or {}
        steps = int(smoke.get("train_steps_per_round", 8))
        base_path = _resolve_path(self.protocol["base_scenario_config"])
        base_config = load_base_scenario_config(base_path)
        warnings: List[str] = []
        generation_result: Dict[str, Any]
        if generator_type == "random":
            requested = int(smoke.get("random_candidates_per_round", 1))
            candidates = RandomScenarioGenerator({"seed": self.seed}).generate_from_base(base_config, requested)
            generation_result = {
                "schema_version": "falcon.random_baseline_generation.v1",
                "requested_num_scenarios": requested,
                "candidates": candidates,
                "warnings": [],
            }
        else:
            requested = int(smoke.get("qwen_candidates_per_round", 1))
            generator = QwenScenarioGenerator(dict(self.protocol.get("llm") or {}))
            health = generator.check_llm_server()
            warnings.extend(health.get("warnings") or [])
            candidates = []
            if health.get("server_reachable") and health.get("model_available"):
                candidates = generator.generate_from_failure_summary(
                    _generic_qwen_task_context(base_config),
                    base_config,
                    num_scenarios=requested,
                )
            else:
                warnings.append("Qwen-only generation skipped because Ollama qwen3:8b health check failed.")
            generation_result = {
                "schema_version": "falcon.qwen_only_baseline_generation.v1",
                "generic_task_context_used": True,
                "failure_analyzer_used": False,
                "health": health,
                "generator_result": generator.last_result,
                "candidates": candidates,
                "warnings": sorted(set(warnings + list(generator.last_result.get("warnings") or []))),
            }

        valid_candidates, validation = self._validate_and_materialize_candidates(candidates)
        generation_path = self.output_dir / "candidate_generation.json"
        validation_path = self.output_dir / "candidate_validation.json"
        write_json(generation_path, generation_result)
        write_json(validation_path, validation)
        if not valid_candidates:
            return self._validity_curriculum_failure(
                generator_type,
                generation_result,
                validation,
                "candidate_validation",
                warnings + ["No valid candidate scenarios were available for curriculum smoke training."],
            )

        pool = CurriculumPool()
        acceptance_results = [_validity_only_acceptance(candidate) for candidate in valid_candidates]
        pool.add_batch(valid_candidates, acceptance_results, source="random" if generator_type == "random" else "llm_qwen8b")
        pool_path = self.output_dir / "curriculum_pool.json"
        pool.save(pool_path)
        scheduler_config = {
            "seed": self.seed,
            "category_ratios": {
                "base": 0.0,
                "random": 1.0 if generator_type == "random" else 0.0,
                "llm_qwen8b": 1.0 if generator_type == "qwen" else 0.0,
                "replay": 0.0,
            },
        }
        scheduler = CurriculumScheduler(scheduler_config)
        random_scenarios = valid_candidates if generator_type == "random" else None
        plan = scheduler.build_sampling_plan(
            pool,
            base_scenarios=None,
            random_scenarios=random_scenarios,
            num_samples=max(1, len(valid_candidates)),
        )
        plan_path = self.output_dir / "sampling_plan.json"
        scheduler.save_sampling_plan(plan, plan_path)
        selection = TrainingPlanAdapter({"seed": self.seed}).select_scenario(plan, strategy="weighted")
        selected = selection.get("selected_scenario")
        if not selected:
            return self._validity_curriculum_failure(
                generator_type,
                generation_result,
                validation,
                "sampling_plan",
                warnings + list(plan.get("warnings") or []) + ["Sampling plan did not select a curriculum scenario."],
                pool_path=pool_path,
                plan_path=plan_path,
            )
        adapter = TrainingPlanAdapter({"seed": self.seed})
        manifest = adapter.prepare_training_config(selected, base_config_path=base_path, output_dir=self.output_dir / "training_config")
        manifest_path = self.output_dir / "training_manifest.json"
        adapter.export_training_config_manifest(manifest, manifest_path)
        train_summary = _train_mappo_smoke(
            scenario_name=str(manifest.get("config_name_or_path") or self.protocol["environment"]),
            output_dir=self.output_dir / "smoke_training",
            num_env_steps=steps,
            buffer_size=steps,
            seed=self.seed,
            training_config_path=manifest.get("training_config_path"),
            scenario_config_path=manifest.get("scenario_config_path"),
            requires_parse_config_patch=bool(manifest.get("requires_parse_config_patch")),
        )
        checkpoint = train_summary.get("actor_checkpoint_path")
        evaluation = self._evaluate_checkpoint(checkpoint)
        warnings.extend(generation_result.get("warnings") or [])
        warnings.extend(validation.get("warnings") or [])
        warnings.extend(plan.get("warnings") or [])
        warnings.extend(manifest.get("warnings") or [])
        warnings.extend(train_summary.get("warnings") or [])
        warnings.extend(evaluation.get("warnings") or [])
        candidate_prefix = "random" if generator_type == "random" else "qwen"
        summary = {
            "schema_version": "falcon.validity_curriculum_baseline_smoke.v1",
            "group": self.group,
            "seed": self.seed,
            "smoke_run": True,
            **GROUP_DEFINITIONS[self.group],
            "qwen_model": self.protocol.get("llm", {}).get("model_name") if generator_type == "qwen" else None,
            f"{candidate_prefix}_candidates_generated": len(candidates),
            f"{candidate_prefix}_candidates_valid": len(valid_candidates),
            "sampling_plan_generated": bool(plan.get("sampled_scenarios")),
            "selected_scenario_id": selected.get("scenario_id"),
            "training_started": bool(train_summary.get("training_started")),
            "training_finished": bool(train_summary.get("training_finished")),
            "checkpoint_saved": bool(train_summary.get("checkpoint_saved")),
            "checkpoint_path": checkpoint,
            "generation_path": str(generation_path),
            "validation_path": str(validation_path),
            "pool_path": str(pool_path),
            "sampling_plan_path": str(plan_path),
            "training_manifest_path": str(manifest_path),
            "training_summary": train_summary,
            "evaluation": evaluation,
            "failure_stage": train_summary.get("failure_stage"),
            "warnings": sorted(set(str(item) for item in warnings if item)),
        }
        write_json(self.output_dir / "group_summary.json", summary)
        return summary

    def _validate_and_materialize_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        output_dir: Optional[Path] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        checker = ConstraintChecker()
        base_config = load_base_scenario_config(_resolve_path(self.protocol["base_scenario_config"]))
        schema_results: List[Dict[str, Any]] = []
        constraint_results: List[Dict[str, Any]] = []
        valid_candidates: List[Dict[str, Any]] = []
        yaml_paths: List[str] = []
        warnings: List[str] = []
        scenario_dir = (output_dir or self.output_dir) / "generated_scenarios"
        for idx, candidate_value in enumerate(candidates):
            candidate = dict(candidate_value)
            schema_result = {"scenario_id": candidate.get("scenario_id"), **validate_candidate_schema(candidate)}
            schema_results.append(schema_result)
            if not schema_result.get("is_valid"):
                warnings.append(f"Candidate {candidate.get('scenario_id', idx)} failed schema validation.")
                continue
            constraint_result = checker.validate_candidate(candidate)
            constraint_results.append(constraint_result)
            if not constraint_result.get("is_valid"):
                warnings.append(f"Candidate {candidate.get('scenario_id', idx)} failed constraint validation.")
                continue
            config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
            config["scenario_id"] = candidate.get("scenario_id")
            path = scenario_dir / f"{idx:04d}_{candidate.get('scenario_id', 'candidate')}.yaml"
            save_scenario_yaml(config, path)
            enriched = dict(candidate)
            enriched["scenario_yaml_path"] = str(path)
            enriched["yaml_path"] = str(path)
            valid_candidates.append(enriched)
            yaml_paths.append(str(path))
        return valid_candidates, {
            "schema_version": "falcon.validity_curriculum_validation.v1",
            "schema_validations": schema_results,
            "constraint_results": constraint_results,
            "valid_candidates": valid_candidates,
            "yaml_paths": yaml_paths,
            "warnings": warnings,
        }

    def _validity_curriculum_failure(
        self,
        generator_type: str,
        generation_result: Mapping[str, Any],
        validation: Mapping[str, Any],
        failure_stage: str,
        warnings: Sequence[str],
        pool_path: Optional[Path] = None,
        plan_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        candidate_prefix = "random" if generator_type == "random" else "qwen"
        candidates = generation_result.get("candidates") or []
        valid = validation.get("valid_candidates") or []
        summary = {
            "schema_version": "falcon.validity_curriculum_baseline_smoke.v1",
            "group": self.group,
            "seed": self.seed,
            "smoke_run": True,
            **GROUP_DEFINITIONS[self.group],
            "qwen_model": self.protocol.get("llm", {}).get("model_name") if generator_type == "qwen" else None,
            f"{candidate_prefix}_candidates_generated": len(candidates),
            f"{candidate_prefix}_candidates_valid": len(valid),
            "sampling_plan_generated": bool(plan_path),
            "training_started": False,
            "training_finished": False,
            "checkpoint_saved": False,
            "checkpoint_path": None,
            "pool_path": str(pool_path) if pool_path else None,
            "sampling_plan_path": str(plan_path) if plan_path else None,
            "failure_stage": failure_stage,
            "warnings": sorted(set(str(item) for item in warnings if item)),
        }
        write_json(self.output_dir / "group_summary.json", summary)
        return summary

    def _prepare_curriculum_group_smoke(self) -> Dict[str, Any]:
        warning = (
            f"{self.group} smoke is preparation-only: group wiring and frozen inputs were validated, "
            "but its dedicated training loop is intentionally not implemented in this preparation round."
        )
        self.warnings.append(warning)
        return {
            "schema_version": "falcon.baseline_group_smoke.v1",
            "group": self.group,
            "prepared_only": True,
            "smoke_execution_available": False,
            "checkpoint_path": None,
            "failure_stage": "group_training_loop_not_implemented",
            "warnings": [warning],
        }

    def _evaluate_checkpoint(
        self,
        checkpoint: Optional[str],
        scenario_limit: Optional[int] = None,
        episodes_per_scenario: int = 1,
    ) -> Dict[str, Any]:
        eval_started_at = _timestamp()
        eval_start = time.perf_counter()
        manifest = self._load_eval_manifest()
        limit = int(
            scenario_limit
            if scenario_limit is not None
            else (self.protocol.get("smoke") or {}).get("evaluation_scenario_limit", 1)
        )
        scenarios = list(manifest.get("scenarios") or [])[:limit]
        if not checkpoint or not Path(checkpoint).exists():
            runtime = round(time.perf_counter() - eval_start, 3)
            return {
                "schema_version": "falcon.baseline_smoke_eval.v1",
                "started_at": eval_started_at,
                "finished_at": _timestamp(),
                "policy_eval_runtime_seconds": runtime,
                "policy_eval_runtime_human_readable": _human_duration(runtime),
                "results": [],
                "warnings": ["Checkpoint unavailable for smoke evaluation."],
            }
        evaluator = PolicyEvaluator({"base_config_path": str(_resolve_path(self.protocol["base_scenario_config"]))})
        results = []
        warnings: List[str] = []
        for scenario in scenarios:
            yaml_path = _resolve_path(scenario.get("scenario_yaml_path"))
            try:
                item = evaluator.evaluate_policy_on_scenario(
                    checkpoint,
                    yaml_path,
                    num_episodes=int(episodes_per_scenario),
                    seed=self.seed,
                )
                results.append(item)
                warnings.extend(item.get("warnings") or [])
            except Exception as exc:  # noqa: BLE001 - smoke result must stay structured
                warnings.append(f"Evaluation failed for {scenario.get('scenario_id')}: {type(exc).__name__}: {exc}")
        runtime = round(time.perf_counter() - eval_start, 3)
        return {
            "schema_version": "falcon.baseline_smoke_eval.v1",
            "started_at": eval_started_at,
            "finished_at": _timestamp(),
            "policy_eval_runtime_seconds": runtime,
            "policy_eval_runtime_human_readable": _human_duration(runtime),
            "num_scenarios_requested": len(scenarios),
            "num_scenarios_evaluated": len(results),
            "results": results,
            "warnings": warnings,
        }

    def _validate_inputs(self) -> Dict[str, bool]:
        manifest = self._load_eval_manifest()
        scenario_paths = [_resolve_path(item.get("scenario_yaml_path")) for item in manifest.get("scenarios") or []]
        return {
            "protocol_exists": self.protocol_path.exists(),
            "group_config_exists": self.group_config_path.exists(),
            "base_scenario_exists": _resolve_path(self.protocol.get("base_scenario_config")).exists(),
            "eval_manifest_exists": _resolve_path(self.protocol.get("evaluation_scenarios")).exists(),
            "eval_manifest_non_empty": len(scenario_paths) > 0,
            "all_eval_scenarios_exist": bool(scenario_paths) and all(path.exists() for path in scenario_paths),
            "seed_allowed": self.seed in _allowed_seeds(self.protocol),
            "fsn_disabled": self.group_config.get("fsn_enabled") is False,
        }

    def _load_eval_manifest(self) -> Dict[str, Any]:
        path = _resolve_path(self.protocol["evaluation_scenarios"])
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _base_result(self, mode: str) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "schema_version": "falcon.baseline_experiment_run.v1",
            "_runtime_start_seconds": time.time(),
            "protocol_path": str(self.protocol_path),
            "protocol_status": self.protocol.get("status"),
            "group": self.group,
            "group_config_path": str(self.group_config_path),
            "group_config": self.group_config,
            "seed": self.seed,
            "mode": mode,
            "output_dir": str(self.output_dir),
            "started_at": _timestamp(),
            "warnings": [],
        }

    def _pilot_params(self, overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        values = dict(self.protocol.get("pilot") or {})
        for key, value in dict(overrides or {}).items():
            if value is not None:
                values[key] = value
        required = (
            "max_rounds",
            "train_steps_per_round",
            "eval_episodes_per_round",
            "policy_eval_episodes_per_candidate",
            "qwen_candidates_per_round",
            "random_candidates_per_round",
        )
        for key in required:
            if key not in values:
                values[key] = self.protocol.get(key)
            values[key] = int(values[key])
        if (overrides or {}).get("resume_from_state"):
            values["resume_from_state"] = str((overrides or {}).get("resume_from_state"))
        return values

    def _save_outputs(self, result: Mapping[str, Any]) -> None:
        output = dict(result)
        start_seconds = output.pop("_runtime_start_seconds", None)
        output["finished_at"] = _timestamp()
        if start_seconds is not None:
            runtime = round(max(0.0, time.time() - float(start_seconds)), 3)
            output["runtime_seconds"] = runtime
            output["runtime_human_readable"] = _human_duration(runtime)
        write_json(self.output_dir / "experiment_manifest.json", self._experiment_manifest())
        write_json(self.output_dir / "baseline_experiment_summary.json", output)
        if isinstance(result, dict):
            result.clear()
            result.update(output)

    def _select_mode_output(self, mode: str) -> None:
        self.output_dir = self._explicit_output_dir or self._seed_output_dir / mode

    def _experiment_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": "falcon.baseline_experiment_manifest.v1",
            "protocol_path": str(self.protocol_path),
            "protocol": self.protocol,
            "group": self.group,
            "group_config_path": str(self.group_config_path),
            "group_config": self.group_config,
            "group_definition": GROUP_DEFINITIONS[self.group],
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "evaluation_scenarios": str(_resolve_path(self.protocol["evaluation_scenarios"])),
            "created_at": _timestamp(),
        }


class BaselineExperimentAnalyzer:
    """Aggregate baseline preparation smoke outputs without making claims."""

    def __init__(self, results_root: str | Path) -> None:
        self.results_root = Path(results_root)

    def analyze(self) -> Dict[str, Any]:
        summaries = []
        for path in sorted(self.results_root.glob("**/baseline_experiment_summary.json")):
            with path.open("r", encoding="utf-8") as f:
                item = json.load(f)
            summaries.append(_summarize_run(item, path))
        eval_summaries = []
        for path in sorted(self.results_root.glob("**/eval_set_summary.json")):
            data = _load_json(path)
            data["_summary_path"] = str(path)
            eval_summaries.append(data)
        groups = {}
        for group in SUPPORTED_GROUPS:
            rows = [item for item in summaries if item.get("group") == group]
            smoke_rows = [row for row in rows if row.get("smoke_run")]
            pilot_rows = [row for row in rows if row.get("pilot_run")]
            group_evals = [item for item in eval_summaries if item.get("group") == group]
            selected_eval = max(
                group_evals,
                key=lambda item: (
                    bool(item.get("eval_protocol_frozen")),
                    item.get("opponent_mode") == "fixed_checkpoint",
                    int(item.get("num_scenarios_evaluated", 0)),
                ),
                default={},
            )
            aggregate = selected_eval.get("aggregate_result") if isinstance(selected_eval.get("aggregate_result"), MappingABC) else {}
            groups[group] = {
                "group_available": any(row.get("ready") for row in smoke_rows),
                "num_runs": len(rows),
                "num_ready": sum(1 for row in rows if row.get("ready")),
                "dry_run_passed": any(row.get("dry_run") and row.get("ready") for row in rows),
                "smoke_run_passed": any(row.get("smoke_run") and row.get("ready") for row in rows),
                "pilot_run_passed": any(row.get("pilot_run") and row.get("ready") for row in rows),
                "checkpoint_saved": any(row.get("checkpoint_saved") for row in smoke_rows),
                "pilot_checkpoint_saved": any(row.get("checkpoint_saved") for row in pilot_rows),
                "eval_set_available": bool(group_evals),
                "eval_set_num_scenarios": int(selected_eval.get("num_scenarios_evaluated", 0)),
                "final_eval_win_rate": aggregate.get("final_win_rate"),
                "final_eval_mean_return": aggregate.get("final_mean_return"),
                "eval_group_breakdown": selected_eval.get("eval_group_breakdown") or {},
                "opponent_mode": _eval_opponent_mode(selected_eval),
                "opponent_checkpoint": selected_eval.get("opponent_checkpoint"),
                "same_actor_eval": _same_actor_eval(selected_eval),
                "eval_protocol_frozen": bool(selected_eval.get("eval_protocol_frozen")),
                **GROUP_DEFINITIONS[group],
                "checkpoints": [row.get("checkpoint_path") for row in rows if row.get("checkpoint_path")],
                "failure_stages": [row.get("failure_stage") for row in rows if row.get("failure_stage")],
            }
        fixed_evals = [item for item in eval_summaries if item.get("eval_protocol_frozen")]
        return {
            "schema_version": "falcon.baseline_summary.v1",
            "results_root": str(self.results_root),
            "num_runs": len(summaries),
            "groups": groups,
            "eval_set_summaries": eval_summaries,
            "formal_comparison_available": False,
            "opponent_mode": fixed_evals[0].get("opponent_mode") if fixed_evals else None,
            "opponent_checkpoint": fixed_evals[0].get("opponent_checkpoint") if fixed_evals else None,
            "same_actor_eval": any(bool(item.get("same_actor_eval")) for item in fixed_evals),
            "eval_protocol_frozen": bool(fixed_evals),
            "warnings": ["Preparation smoke outputs are not formal baseline comparison results."],
            "runs": summaries,
        }

    def export(self, output_dir: str | Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.analyze()
        write_json(output_dir / "baseline_summary.json", report)
        self._write_csv(output_dir / "baseline_summary.csv", report.get("runs") or [])
        self._write_text(output_dir / "baseline_report.txt", report)
        return report

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        fields = [
            "group",
            "seed",
            "mode",
            "ready",
            "dry_run",
            "smoke_run",
            "pilot_run",
            "checkpoint_saved",
            "qwen_used",
            "random_used",
            "difficulty_evaluator_used",
            "falcon_full_pipeline_used",
            "checkpoint_path",
            "opponent_mode",
            "opponent_checkpoint",
            "same_actor_eval",
            "eval_protocol_frozen",
            "failure_stage",
            "output_dir",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    @staticmethod
    def _write_text(path: Path, report: Mapping[str, Any]) -> None:
        lines = [
            "FALCON baseline experiment preparation report",
            "",
            "This report summarizes dry-run and bounded smoke readiness only.",
            "It is not a formal baseline comparison.",
            "",
        ]
        for group, stats in (report.get("groups") or {}).items():
            lines.append(
                f"{group}: runs={stats.get('num_runs')}, dry_run_passed={stats.get('dry_run_passed')}, "
                f"smoke_run_passed={stats.get('smoke_run_passed')}, pilot_run_passed={stats.get('pilot_run_passed')}, "
                f"eval_set_available={stats.get('eval_set_available')}, "
                f"opponent_mode={stats.get('opponent_mode')}, eval_protocol_frozen={stats.get('eval_protocol_frozen')}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT_DIR / path


def _allowed_seeds(protocol: Mapping[str, Any]) -> List[int]:
    values = list(protocol.get("seeds") or []) + list(protocol.get("planned_formal_seeds") or [])
    output = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(output))


def _human_duration(seconds: Any) -> str:
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "unknown"
    hours, remainder = divmod(max(total, 0), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _sum_safe_float(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return round(sum(_safe_float(row.get(key), 0.0) or 0.0 for row in rows), 3)


def _infer_round_runtime_from_files(controller_dir: Path, round_id: int) -> Dict[str, Any]:
    candidates = [
        controller_dir / f"falcon_controller_failure_summary_round{round_id}.json",
        controller_dir / f"falcon_controller_candidates_round{round_id}.json",
        controller_dir / f"falcon_controller_validated_candidates_round{round_id}.json",
        controller_dir / f"falcon_controller_policy_eval_round{round_id}.json",
        controller_dir / f"falcon_controller_difficulty_round{round_id}.json",
        controller_dir / f"falcon_controller_sampling_plan_round{round_id}.json",
        controller_dir / f"falcon_controller_training_round{round_id}_summary.json",
        controller_dir / f"falcon_controller_round{round_id}_summary.json",
    ]
    times = [path.stat().st_mtime for path in candidates if path.exists()]
    if len(times) < 2:
        return {
            "round_runtime_seconds": None,
            "round_runtime_human_readable": "unknown",
            "runtime_inference": "insufficient_round_file_timestamps",
        }
    runtime = round(max(times) - min(times), 3)
    return {
        "round_runtime_seconds": runtime,
        "round_runtime_human_readable": _human_duration(runtime),
        "runtime_inference": "estimated_from_round_file_timestamps",
    }


def _checkpoint_model_dir(checkpoint_path: Any) -> Optional[str]:
    if not checkpoint_path:
        return None
    path = Path(str(checkpoint_path))
    return str(path.parent) if path.exists() else None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, MappingABC) else {}


def _mean_eval_metric(results: Sequence[Mapping[str, Any]], key: str) -> float:
    values = []
    for item in results:
        try:
            values.append(float(item.get(key)))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 6) if values else 0.0


def _same_actor_eval(summary: Mapping[str, Any]) -> Optional[bool]:
    if summary.get("same_actor_eval") is not None:
        return bool(summary.get("same_actor_eval"))
    warnings = " ".join(str(item).lower() for item in summary.get("warnings") or [])
    if "used same actor for both teams" in warnings:
        return True
    return None


def _eval_opponent_mode(summary: Mapping[str, Any]) -> Optional[str]:
    if summary.get("opponent_mode"):
        return str(summary.get("opponent_mode"))
    return "same_actor" if _same_actor_eval(summary) else None


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _summarize_run(item: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    execution = item.get("execution") if isinstance(item.get("execution"), MappingABC) else {}
    controller = execution.get("controller_result") if isinstance(execution.get("controller_result"), MappingABC) else {}
    training = execution.get("training_summary") if isinstance(execution.get("training_summary"), MappingABC) else {}
    evaluation = execution.get("evaluation") if isinstance(execution.get("evaluation"), MappingABC) else {}
    pool_stats = controller.get("pool_stats") if isinstance(controller.get("pool_stats"), MappingABC) else {}
    return {
        "schema_version": "falcon.baseline_run_summary.v1",
        "group": item.get("group"),
        "seed": item.get("seed"),
        "mode": item.get("mode"),
        "ready": bool(item.get("ready")),
        "dry_run": bool(item.get("dry_run")),
        "smoke_run": bool(item.get("smoke_run")),
        "pilot_run": bool(item.get("pilot_run")),
        "checkpoint_path": item.get("checkpoint_path"),
        "checkpoint_saved": bool(training.get("checkpoint_saved") or item.get("checkpoint_path")),
        "completed_rounds": controller.get("completed_rounds"),
        "accepted_items": pool_stats.get("accepted_items"),
        "num_eval_scenarios_evaluated": evaluation.get("num_scenarios_evaluated"),
        **GROUP_DEFINITIONS.get(str(item.get("group")), {}),
        "failure_stage": item.get("failure_stage"),
        "output_dir": item.get("output_dir"),
        "summary_path": str(path),
        "warnings": list(item.get("warnings") or []),
    }


def _generic_qwen_task_context(base_config: Mapping[str, Any]) -> Dict[str, Any]:
    initial_config = extract_initial_config_from_yaml(base_config)
    scenario_vector = initial_config_to_scenario_vector(initial_config)["scenario_vector"]
    return {
        "schema_version": "falcon.qwen_only_task_context.v1",
        "source_trajectory": None,
        "primary_failure_modes": ["scenario_diversification"],
        "secondary_failure_modes": [],
        "failure_scores": {},
        "failure_severity": None,
        "submetrics": {},
        "evidence": {
            "task_context": [
                "Generate legal 2v2 NoWeapon training scenario variations for a Qwen-only baseline.",
                "This context was not produced by FailureAnalyzer and must not be interpreted as a failure-aware FALCON signal.",
            ]
        },
        "scenario_vector": scenario_vector,
        "episode_summary": {},
        "metadata": {
            "baseline_group": "mappo_qwen_only",
            "failure_analyzer_used": False,
            "policy_evaluator_used_for_filtering": False,
            "difficulty_evaluator_used": False,
        },
    }


def _validity_only_acceptance(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.validity_only_acceptance.v1",
        "scenario_id": candidate.get("scenario_id"),
        "accepted_into_curriculum_pool": True,
        "hard_filter_passed": True,
        "final_value_score": 1.0,
        "sampling_weight": 1.0,
        "priority_level": "medium",
        "rejection_reasons": [],
        "warnings": ["Accepted by schema and constraint validity only; DifficultyEvaluator was not used."],
    }


def _scope_candidate_ids(
    candidates: Sequence[Mapping[str, Any]],
    group: str,
    round_id: int,
) -> List[Dict[str, Any]]:
    output = []
    for idx, candidate_value in enumerate(candidates):
        candidate = dict(candidate_value)
        original_id = str(candidate.get("scenario_id") or f"candidate_{idx:04d}")
        candidate["scenario_id"] = f"{group}_round{int(round_id):03d}_{original_id}"
        metadata = dict(candidate.get("metadata") or {})
        metadata["original_scenario_id"] = original_id
        metadata["baseline_round_id"] = int(round_id)
        candidate["metadata"] = metadata
        output.append(candidate)
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
