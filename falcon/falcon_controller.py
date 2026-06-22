"""Minimal FALCON outer-loop controller for smoke testing."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
import traceback
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .candidate_schema import validate_candidate_schema
from .constraint_checker import ConstraintChecker
from .curriculum_pool import CurriculumPool
from .curriculum_scheduler import CurriculumScheduler
from .difficulty_evaluator import DifficultyEvaluator
from .failure_analyzer import FailureAnalyzer
from .fsn_replacement_generator import FSNReplacementGenerator
from .llm_scenario_generator import QwenScenarioGenerator
from .policy_evaluator import PolicyEvaluator
from .random_scenario_generator import RandomScenarioGenerator
from .scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config, save_scenario_yaml
from .training_plan_adapter import MultiScenarioTrainingBridge, TrainingPlanAdapter
from .trajectory_recorder import extract_scenario_vector, load_trajectory, summarize_episode

if not hasattr(np, "product"):
    np.product = np.prod  # NumPy 2.x compatibility for the existing flattener.

ROOT_DIR = Path(__file__).resolve().parents[1]
FALCON_CONTROLLER_SCHEMA_VERSION = "falcon.controller.v1"

DEFAULT_CONFIG: Dict[str, Any] = {
    "output_dir": str(ROOT_DIR / "tests" / "tmp_falcon_controller_smoke"),
    "base_config_path": str(ROOT_DIR / "envs" / "JSBSim" / "configs" / "2v2" / "NoWeapon" / "Selfplay.yaml"),
    "failure_fixture_path": str(ROOT_DIR / "tests" / "fixtures" / "falcon_coordination_failure_v2.json"),
    "max_rounds": 2,
    "train_steps_per_round": 8,
    "eval_episodes_per_round": 1,
    "qwen_candidates_per_round": 3,
    "candidate_env_load_check": False,
    "policy_evaluation": {},
    "policy_eval_episodes_per_candidate": 1,
    "resume_from_state": None,
    "save_every_round": True,
    "use_real_failure_trajectory": True,
    "max_pool_size": 100,
    "initial_training": {
        "num_env_steps": 8,
        "buffer_size": 8,
        "seed": 21,
        "scenario_name": "2v2/NoWeapon/Selfplay",
    },
    "round1_training": {
        "num_env_steps": 8,
        "buffer_size": 8,
        "seed": 22,
    },
    "num_candidates": 3,
    "num_eval_episodes": 1,
    "sampling_num_samples": 6,
    "coverage_aware_training": {
        "enabled": False,
        "scenarios_per_round": 8,
        "category_quota": {
            "accepted_llm": 4,
            "base_anchor": 2,
            "replay_failure": 1,
            "random_explore": 1,
        },
        "unseen_bonus": 2.0,
        "trained_count_threshold": 3,
    },
    "stability_aware_training": {
        "enabled": False,
        "interleave_anchors": True,
        "anchor_ratio_min": 0.5,
        "accepted_ratio_max": 0.5,
        "fallback_reallocate_to_anchor": True,
        "save_intermediate_checkpoints": True,
        "round_checkpoint_selection": "anchor_validation",
        "preserve_best_within_batch": True,
        "anchor_validation_episodes": 1,
        "anchor_validation_max_scenarios": 1,
        "anchor_validation_manifest": None,
        "opponent_mode": "fixed_checkpoint",
        "opponent_checkpoint": None,
    },
    "best_checkpoint_path": str(
        ROOT_DIR
        / "tests"
        / "tmp_falcon_multicheckpoint"
        / "mappo_2v2_multicheckpoint"
        / "actor_later.pt"
    ),
    "qwen": {
        "provider": "ollama",
        "provider_mode": "ollama_native",
        "model_name": "qwen3:8b",
        "think": False,
        "stream": False,
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 4096,
        "timeout": 180.0,
        "num_retries": 2,
    },
    "fsn_replacement": {
        "enabled": False,
        "fsn_model_path": str(
            ROOT_DIR
            / "experiments"
            / "falcon_2v2_noweapon"
            / "fsn"
            / "stage2"
            / "fsn_stage2_model.pt"
        ),
        "replacement_ratio": 0.25,
        "target_fsn_ratio": 0.25,
        "use_diversity_aware_fsn": True,
        "qwen_ratio": 0.75,
        "fsn_candidates_per_round": 1,
        "qwen_candidates_per_round": 3,
        "qwen_quota": 3,
        "fsn_quota": 1,
        "total_candidates_per_round": 4,
        "enforce_quota": True,
        "max_actual_fsn_share": 0.30,
        "qwen_retry_to_fill_quota": True,
        "qwen_max_retries_per_round": 2,
        "fallback_when_qwen_shortfall": "random",
        "do_not_backfill_qwen_shortfall_with_fsn": True,
        "fallback_to_qwen_if_fsn_invalid": True,
        "record_generator_source": True,
    },
}


class FalconController:
    """Coordinate a minimal FALCON outer-loop smoke round."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = {
            "schema_version": FALCON_CONTROLLER_SCHEMA_VERSION,
            "created_at": _timestamp(),
            "rounds": {},
            "current_checkpoint_path": None,
            "best_checkpoint_path": None,
            "latest_checkpoint_path": None,
            "checkpoint_registry": {
                "schema_version": "falcon.checkpoint_registry.v1",
                "checkpoints": [],
                "current_checkpoint": None,
                "best_checkpoint": None,
                "latest_checkpoint": None,
                "best_win_rate": None,
                "warnings": [],
            },
            "failure_collection_stats": {
                "real_failure_trajectory_used_count": 0,
                "fallback_failure_used_count": 0,
            },
            "warnings": [],
        }
        self.pool = CurriculumPool({"max_pool_size": self.config.get("max_pool_size")})
        resume_path = self.config.get("resume_from_state")
        if resume_path:
            self.load_controller_state(resume_path)

    def run_round(self, round_id: int) -> Dict[str, Any]:
        if int(round_id) == 0:
            return self._run_round0()
        if int(round_id) == 1:
            return self._run_round1()
        raise ValueError("FalconController smoke supports only round_id 0 or 1.")

    def run(self, max_rounds: Optional[int] = None) -> Dict[str, Any]:
        run_started_at = _timestamp()
        run_start = time.perf_counter()
        max_rounds = int(max_rounds if max_rounds is not None else self.config.get("max_rounds", 2))
        completed_rounds = []
        existing_round_ids = sorted(
            int(key) for key in self.state.get("rounds", {}).keys() if str(key).isdigit()
        )
        start_round = (existing_round_ids[-1] + 1) if existing_round_ids else 0
        if not self.state.get("current_checkpoint_path"):
            initial_training = self.run_initial_training()
            self.state["initial_training"] = initial_training
            self._register_checkpoint(
                round_id=0,
                role="initial_current",
                checkpoint_path=initial_training.get("actor_checkpoint_path"),
                training_summary=initial_training,
            )
        for round_id in range(start_round, max_rounds):
            training_result = None
            if round_id > 0:
                training_result = self.run_training_with_sampling_plan(round_id=round_id)
                self._register_checkpoint(
                    round_id=round_id,
                    role="round_checkpoint",
                    checkpoint_path=(training_result.get("train_summary") or {}).get("actor_checkpoint_path"),
                    training_summary=training_result.get("train_summary"),
                )
            round_state = self._run_course_update_round(round_id, training_result=training_result)
            completed_rounds.append(round_state)
        self.state["completed_rounds"] = len(self.state.get("rounds", {}))
        self.state["latest_checkpoint_path"] = self.state.get("current_checkpoint_path")
        self._save_checkpoint_registry()
        self.pool.save(self.output_dir / "falcon_curriculum_pool_final.json")
        self.save_controller_state(self.output_dir / "falcon_controller_state_final.json")
        runtime = round(time.perf_counter() - run_start, 3)
        return {
            "schema_version": "falcon.controller_run.v1",
            "started_at": run_started_at,
            "finished_at": _timestamp(),
            "runtime_seconds": runtime,
            "runtime_human_readable": _human_duration(runtime),
            "max_rounds": max_rounds,
            "start_round": start_round,
            "completed_rounds": len(self.state.get("rounds", {})),
            "new_rounds_completed": len(completed_rounds),
            "rounds": completed_rounds,
            "checkpoint_registry": self.state.get("checkpoint_registry"),
            "pool_stats": self.pool.get_stats(),
            "warnings": sorted(set(self.state.get("warnings", []))),
        }

    def run_initial_training(self) -> Dict[str, Any]:
        cfg = self.config["initial_training"]
        train_dir = self.output_dir / "round0_initial_training"
        summary = _train_mappo_smoke(
            scenario_name=str(cfg.get("scenario_name", "2v2/NoWeapon/Selfplay")),
            output_dir=train_dir,
            num_env_steps=int(cfg.get("num_env_steps", self.config.get("train_steps_per_round", 8))),
            buffer_size=int(cfg.get("buffer_size", self.config.get("train_steps_per_round", 8))),
            seed=int(cfg.get("seed", 21)),
        )
        checkpoint = summary.get("actor_checkpoint_path") if summary.get("checkpoint_saved") else None
        self.state["current_checkpoint_path"] = checkpoint
        self.state["latest_checkpoint_path"] = checkpoint
        configured_best = Path(str(self.config.get("best_checkpoint_path") or ""))
        if configured_best.exists():
            self.state["best_checkpoint_path"] = str(configured_best)
            self.state["checkpoint_registry"]["best_checkpoint"] = str(configured_best)
        else:
            self.state["best_checkpoint_path"] = checkpoint
            self.state["checkpoint_registry"]["best_checkpoint"] = checkpoint
            self.state["warnings"].append("No separate best checkpoint was found; using current checkpoint as best for smoke evaluation.")
        return summary

    def collect_or_load_failure_summary(self, round_id: Optional[int] = None) -> Dict[str, Any]:
        if self.config.get("use_real_failure_trajectory", True):
            real_summary = self._collect_real_failure_summary(round_id=round_id)
            if real_summary is not None:
                return real_summary
        trajectory = load_trajectory(self.config["failure_fixture_path"])
        trajectory["_source_trajectory"] = str(self.config["failure_fixture_path"])
        failure_summary = FailureAnalyzer().analyze_trajectory(
            trajectory,
            success_stats={"mean_success_team_reward": 500.0},
        )
        failure_summary["scenario_vector"] = extract_scenario_vector(trajectory)
        failure_summary["episode_summary"] = summarize_episode(trajectory)
        failure_summary["failure_source"] = {
            "type": "fixture_fallback",
            "trajectory_path": str(self.config["failure_fixture_path"]),
            "round_id": round_id,
        }
        self.state.setdefault("failure_collection_stats", {}).setdefault("fallback_failure_used_count", 0)
        self.state["failure_collection_stats"]["fallback_failure_used_count"] += 1
        self.state["warnings"].append("Used failure fixture because no real rollout failure trajectory was available.")
        return failure_summary

    def generate_candidate_scenarios(self, failure_summary: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        base_config = load_base_scenario_config(self.config["base_config_path"])
        replacement_config = dict(self.config.get("fsn_replacement") or {})
        if replacement_config.get("enabled", False):
            replacement_config["qwen"] = _deep_merge(
                dict(self.config.get("qwen") or {}),
                dict(replacement_config.get("qwen") or {}),
            )
            generator = FSNReplacementGenerator(replacement_config)
            candidates, mixed_result = generator.generate_from_failure_summary(
                failure_summary,
                base_config,
                pool_stats=self.pool.get_stats(),
            )
            return candidates, {
                "schema_version": "falcon.controller_candidate_generation.v1",
                "health": mixed_result.get("health"),
                "generation_mode": "fsn_replacement",
                "generation_result": mixed_result,
                "candidates": candidates,
                "warnings": list(mixed_result.get("warnings") or []),
            }
        generator = QwenScenarioGenerator(self.config.get("qwen"))
        health = generator.check_llm_server()
        candidates: List[Dict[str, Any]] = []
        warnings = list(health.get("warnings", []))
        if health.get("server_reachable") and health.get("model_available"):
            candidates = generator.generate_from_failure_summary(
                failure_summary,
                base_config,
                num_scenarios=int(self.config.get("qwen_candidates_per_round", self.config.get("num_candidates", 3))),
            )
            warnings.extend(generator.last_result.get("warnings", []))
        else:
            warnings.append("Skipped qwen3:8b generation because Ollama health check did not pass.")
        result = {
            "schema_version": "falcon.controller_candidate_generation.v1",
            "health": health,
            "generation_result": generator.last_result,
            "candidates": candidates,
            "warnings": sorted(set(warnings)),
        }
        return candidates, result

    def validate_candidates(self, candidates: Sequence[Mapping[str, Any]], round_id: int = 0) -> Dict[str, Any]:
        base_config = load_base_scenario_config(self.config["base_config_path"])
        checker = ConstraintChecker()
        enable_env_load_check = bool(self.config.get("candidate_env_load_check", False))
        schema_validations: List[Dict[str, Any]] = []
        constraint_results: List[Dict[str, Any]] = []
        valid_candidates: List[Dict[str, Any]] = []
        yaml_paths: List[str] = []
        for idx, candidate in enumerate(candidates):
            validation = {"scenario_id": candidate.get("scenario_id"), **validate_candidate_schema(candidate)}
            schema_validations.append(validation)
            if not validation.get("is_valid"):
                constraint_results.append(_invalid_constraint(candidate, validation, idx))
                continue
            constraint = checker.validate_candidate(candidate)
            if not constraint.get("is_valid"):
                constraint_results.append(constraint)
                continue
            yaml_config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
            yaml_config["scenario_id"] = candidate.get("scenario_id")
            if enable_env_load_check:
                constraint = checker.validate_yaml_config(
                    yaml_config,
                    enable_env_load_check=True,
                    temp_config_name=f"falcon_controller_candidate_round{round_id}_{idx:04d}",
                )
            constraint_results.append(constraint)
            if not constraint.get("is_valid"):
                continue
            yaml_path = self.output_dir / f"falcon_controller_candidate_round{round_id}_{idx:04d}.yaml"
            save_scenario_yaml(yaml_config, yaml_path)
            enriched = dict(candidate)
            enriched["yaml_path"] = str(yaml_path)
            enriched["scenario_yaml_path"] = str(yaml_path)
            valid_candidates.append(enriched)
            yaml_paths.append(str(yaml_path))
        return {
            "schema_version": "falcon.controller_candidate_validation.v1",
            "schema_validations": schema_validations,
            "constraint_results": constraint_results,
            "valid_candidates": valid_candidates,
            "yaml_paths": yaml_paths,
            "env_load_check_enabled": enable_env_load_check,
        }

    def evaluate_candidates(self, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        current = self.state.get("current_checkpoint_path")
        best = self.state.get("best_checkpoint_path") or current
        warnings: List[str] = []
        if current and best and Path(current).resolve() == Path(best).resolve():
            warnings.append("Current and best checkpoints are identical; learning_potential may not be meaningful.")
        results: List[Dict[str, Any]] = []
        if current and best and candidates:
            evaluator_config = _deep_merge(
                {"base_config_path": str(self.config["base_config_path"])},
                dict(self.config.get("policy_evaluation") or {}),
            )
            evaluator = PolicyEvaluator(evaluator_config)
            results = evaluator.evaluate_current_and_best(
                current,
                best,
                candidates,
                num_episodes=int(self.config.get("policy_eval_episodes_per_candidate", self.config.get("num_eval_episodes", 1))),
            )
            for item in results:
                warnings.extend(item.get("warnings", []))
        elif not candidates:
            warnings.append("Policy evaluation skipped because there were no valid candidates.")
        else:
            warnings.append("Policy evaluation skipped because current/best checkpoints were unavailable.")
        return {
            "schema_version": "falcon.controller_policy_eval.v1",
            "current_checkpoint_path": current,
            "best_checkpoint_path": best,
            "policy_eval_results": results,
            "warnings": sorted(set(warnings)),
        }

    def update_curriculum_pool(
        self,
        candidates: Sequence[Mapping[str, Any]],
        policy_eval: Mapping[str, Any],
        failure_summary: Mapping[str, Any],
        validation: Mapping[str, Any],
        round_id: int = 0,
    ) -> Dict[str, Any]:
        policy_eval_results = list(policy_eval.get("policy_eval_results") or [])
        valid_constraints = [item for item in validation.get("constraint_results", []) if item.get("is_valid")]
        difficulty_results: List[Dict[str, Any]] = [_policy_eval_failed_difficulty(candidate, idx) for idx, candidate in enumerate(candidates)]
        eval_rows: List[Tuple[int, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
        for idx, candidate in enumerate(candidates):
            eval_result = policy_eval_results[idx] if idx < len(policy_eval_results) else {}
            current_eval = eval_result.get("current_policy_eval", {}) if isinstance(eval_result, MappingABC) else {}
            best_eval = eval_result.get("best_policy_eval", {}) if isinstance(eval_result, MappingABC) else {}
            if current_eval.get("real_policy_eval_available") and best_eval.get("real_policy_eval_available"):
                constraint = valid_constraints[idx] if idx < len(valid_constraints) else {}
                eval_rows.append((idx, candidate, current_eval, best_eval, constraint))
        if eval_rows:
            evaluated = DifficultyEvaluator().evaluate_batch(
                [row[1] for row in eval_rows],
                [row[2] for row in eval_rows],
                [row[3] for row in eval_rows],
                {"scenario_vectors": [failure_summary.get("scenario_vector", {})]},
                failure_summary,
                [row[4] for row in eval_rows],
            )
            for row, difficulty in zip(eval_rows, evaluated):
                difficulty_results[row[0]] = difficulty
        for idx, candidate in enumerate(candidates):
            self.pool.add_candidate(
                candidate,
                difficulty_results[idx] if idx < len(difficulty_results) else {},
                source=_candidate_pool_source(candidate),
                source_round=round_id,
                failure_summary=failure_summary,
                constraint_result=valid_constraints[idx] if idx < len(valid_constraints) else {},
                policy_eval_result=policy_eval_results[idx] if idx < len(policy_eval_results) else {},
            )
        pool_path = self.output_dir / f"falcon_controller_pool_round{round_id}.json"
        self.pool.save(pool_path)
        result = {
            "schema_version": "falcon.controller_difficulty_pool.v1",
            "difficulty_results": difficulty_results,
            "pool_path": str(pool_path),
            "pool_stats": self.pool.get_stats(),
        }
        self.state["pool_path"] = str(pool_path)
        self.state[f"difficulty_results_round{round_id}"] = difficulty_results
        self.state[f"valid_candidates_round{round_id}"] = [dict(item) for item in candidates]
        return result

    def build_sampling_plan(self, pool_result: Mapping[str, Any], round_id: int = 0) -> Dict[str, Any]:
        pool = self.pool if self.pool.get_all() else CurriculumPool().load(pool_result["pool_path"])
        base_config_path = Path(self.config["base_config_path"])
        coverage_config = dict(self.config.get("coverage_aware_training") or {})
        stability_config = dict(self.config.get("stability_aware_training") or {})
        scheduler = CurriculumScheduler(
            {
                "seed": int(self.config.get("initial_training", {}).get("seed", 21)) + int(round_id),
                "coverage_aware_enabled": bool(coverage_config.get("enabled", False)),
                "scenario_batch_size": int(coverage_config.get("scenarios_per_round", 8)),
                "total_train_steps_per_round": int(self.config.get("train_steps_per_round", 8)),
                "category_quota": dict(coverage_config.get("category_quota") or {}),
                "unseen_bonus": coverage_config.get("unseen_bonus", 2.0),
                "trained_count_threshold": coverage_config.get("trained_count_threshold", 3),
                "stability_aware_enabled": bool(
                    stability_config.get("enabled", False)
                ),
                "interleave_anchors": bool(
                    stability_config.get("interleave_anchors", True)
                ),
                "anchor_ratio_min": stability_config.get("anchor_ratio_min", 0.5),
                "accepted_ratio_max": stability_config.get(
                    "accepted_ratio_max", 0.5
                ),
                "fallback_reallocate_to_anchor": bool(
                    stability_config.get("fallback_reallocate_to_anchor", True)
                ),
            }
        )
        plan = scheduler.build_sampling_plan(
            pool,
            base_scenarios=[
                {
                    "scenario_id": "base_2v2_NoWeapon_Selfplay",
                    "source": "original",
                    "scenario_yaml_path": str(base_config_path),
                    "priority_level": "base",
                    "target_failure_modes": [],
                }
            ],
            random_scenarios=None,
            num_samples=int(self.config.get("sampling_num_samples", 6)),
            current_round=round_id,
            coverage_aware=bool(coverage_config.get("enabled", False)),
            scenario_batch_size=int(coverage_config.get("scenarios_per_round", 8)),
            total_train_steps_per_round=int(self.config.get("train_steps_per_round", 8)),
        )
        plan_path = self.output_dir / f"falcon_controller_sampling_plan_round{round_id}.json"
        scheduler.save_sampling_plan(plan, plan_path)
        self.state["sampling_plan_path"] = str(plan_path)
        self.state[f"sampling_plan_round{round_id}"] = plan
        return {"schema_version": "falcon.controller_sampling_plan.v1", "path": str(plan_path), "plan": plan}

    def run_training_with_sampling_plan(self, round_id: int = 1) -> Dict[str, Any]:
        started_at = _timestamp()
        step_start = time.perf_counter()
        sampling_plan_path = self.state.get("sampling_plan_path")
        adapter = TrainingPlanAdapter({"lag_config_root": str(ROOT_DIR / "envs" / "JSBSim" / "configs")})
        warnings: List[str] = []
        fallback_used = False
        fallback_reason = None
        selected_fallback_scenario = None
        plan = adapter.load_sampling_plan(sampling_plan_path) if sampling_plan_path else {"sampled_scenarios": []}
        coverage_enabled = bool(
            (self.config.get("coverage_aware_training") or {}).get("enabled", False)
        )
        stability_config = dict(self.config.get("stability_aware_training") or {})
        stability_enabled = bool(stability_config.get("enabled", False))
        if coverage_enabled and plan.get("scenario_batch"):
            bridge = MultiScenarioTrainingBridge(
                adapter=adapter,
                config={
                    "seed": int(self.config.get("round1_training", {}).get("seed", 22)),
                    "default_per_scenario_train_steps": max(
                        int(self.config.get("train_steps_per_round", 8))
                        // max(len(plan.get("scenario_batch") or []), 1),
                        1,
                    ),
                    "preserve_best_within_batch": bool(
                        stability_config.get("preserve_best_within_batch", False)
                    ),
                    "round_checkpoint_selection": stability_config.get(
                        "round_checkpoint_selection", "terminal"
                    ),
                },
            )
            checkpoint_validation_fn = (
                self._build_stability_checkpoint_validator(plan, round_id)
                if stability_enabled
                and stability_config.get("round_checkpoint_selection")
                == "anchor_validation"
                else None
            )
            training_start = time.perf_counter()
            batch_summary = bridge.run_batch(
                plan,
                train_fn=_train_mappo_smoke,
                output_dir=self.output_dir / f"round{round_id}_multi_scenario_training",
                base_config_path=self.config["base_config_path"],
                initial_checkpoint_path=self.state.get("current_checkpoint_path"),
                round_id=round_id,
                curriculum_pool=self.pool,
                checkpoint_validation_fn=checkpoint_validation_fn,
            )
            training_runtime = round(time.perf_counter() - training_start, 3)
            warnings.extend(batch_summary.get("warnings", []))
            successful = [
                item
                for item in batch_summary.get("training_results", [])
                if item.get("training_succeeded")
            ]
            selected_batch_index = batch_summary.get("selected_batch_index")
            selected_record = next(
                (
                    item
                    for item in successful
                    if item.get("batch_index") == selected_batch_index
                ),
                None,
            )
            selected_checkpoint = batch_summary.get("selected_checkpoint_path")
            last_train_summary = (
                dict(selected_record.get("train_summary") or {})
                if selected_record
                else {
                    "schema_version": "falcon.stability_selected_checkpoint.v1",
                    "training_started": True,
                    "training_finished": bool(successful),
                    "checkpoint_saved": bool(selected_checkpoint),
                    "actor_checkpoint_path": selected_checkpoint,
                    "failure_stage": None if selected_checkpoint else batch_summary.get("failure_stage"),
                    "selected_from_round_input": selected_batch_index == -1,
                    "warnings": [],
                }
                if successful
                else {
                    "failure_stage": batch_summary.get("failure_stage"),
                    "checkpoint_saved": False,
                    "actor_checkpoint_path": None,
                    "warnings": list(batch_summary.get("warnings") or []),
                }
            )
            pool_path = self.output_dir / f"falcon_controller_pool_after_training_round{round_id}.json"
            self.pool.save(pool_path)
            self.state["pool_path"] = str(pool_path)
            total_runtime = round(time.perf_counter() - step_start, 3)
            result = {
                "schema_version": "falcon.controller_multi_scenario_training.v1",
                "round_id": round_id,
                "started_at": started_at,
                "finished_at": _timestamp(),
                "runtime_seconds": total_runtime,
                "runtime_human_readable": _human_duration(total_runtime),
                "training_runtime_seconds": training_runtime,
                "training_runtime_human_readable": _human_duration(training_runtime),
                "manifest_path": str(
                    self.output_dir
                    / f"round{round_id}_multi_scenario_training"
                    / "scenario_batch_manifest.json"
                ),
                "manifest": {
                    "scenario_batch_size": batch_summary.get("scenario_batch_size"),
                    "scenario_batch": plan.get("scenario_batch"),
                },
                "train_summary": last_train_summary,
                "multi_scenario_training_summary": batch_summary,
                "used_sampling_plan": bool(sampling_plan_path),
                "fallback_used": False,
                "fallback_reason": None,
                "selected_fallback_scenario": None,
                "anchor_scenarios_used": batch_summary.get("anchor_scenarios_used", []),
                "anchor_ratio": batch_summary.get("anchor_ratio", 0.0),
                "accepted_ratio": batch_summary.get("accepted_ratio", 0.0),
                "replay_ratio": batch_summary.get("replay_ratio", 0.0),
                "stability_aware": stability_enabled,
                "selected_batch_index": selected_batch_index,
                "selected_checkpoint_path": selected_checkpoint,
                "terminal_checkpoint_path": batch_summary.get(
                    "terminal_checkpoint_path"
                ),
                "selected_checkpoint_validation": batch_summary.get(
                    "selected_checkpoint_validation"
                ),
                "pool_after_training_path": str(pool_path),
                "warnings": sorted(set(warnings)),
            }
            self.state[f"round{round_id}_training"] = result
            latest_checkpoint = batch_summary.get("selected_checkpoint_path") or batch_summary.get("latest_checkpoint_path")
            if latest_checkpoint:
                self.state["current_checkpoint_path"] = latest_checkpoint
                self.state["latest_checkpoint_path"] = latest_checkpoint
            return result

        selection = adapter.select_scenario(plan, strategy="weighted")
        warnings.extend(selection.get("warnings", []))
        selected = selection.get("selected_scenario") or {}
        if not _is_falcon_generated_training_scenario(selected, self.config["base_config_path"]):
            plan_selected = _first_falcon_generated_from_plan(plan, self.config["base_config_path"])
            if plan_selected:
                selected = plan_selected
                warnings.append("Weighted sampling selected a base/non-generated scenario; used a generated scenario already present in the sampling plan for readiness smoke.")
            else:
                fallback_used = True
                selected, fallback_reason = self._fallback_training_scenario()
                selected_fallback_scenario = selected.get("scenario_id") if selected else None
        manifest = adapter.prepare_training_config(
            selected,
            base_config_path=self.config["base_config_path"],
            output_dir=self.output_dir / f"round{round_id}_training_config",
        )
        warnings.extend(manifest.get("warnings", []))
        manifest_path = self.output_dir / f"falcon_controller_training_round{round_id}_manifest.json"
        adapter.export_training_config_manifest(manifest, manifest_path)
        cfg = self.config.get("round1_training", {})
        training_start = time.perf_counter()
        train_summary = _train_mappo_smoke(
            scenario_name=str(manifest.get("config_name_or_path")),
            output_dir=self.output_dir / f"round{round_id}_training",
            num_env_steps=int(cfg.get("num_env_steps", self.config.get("train_steps_per_round", 8))),
            buffer_size=int(cfg.get("buffer_size", self.config.get("train_steps_per_round", 8))),
            seed=int(cfg.get("seed", 22)) + max(round_id - 1, 0),
            training_config_path=manifest.get("training_config_path"),
            scenario_config_path=manifest.get("scenario_config_path"),
            requires_parse_config_patch=bool(manifest.get("requires_parse_config_patch")),
            model_dir=_checkpoint_model_dir(self.state.get("current_checkpoint_path")),
        )
        training_runtime = round(time.perf_counter() - training_start, 3)
        warnings.extend(train_summary.get("warnings", []))
        total_runtime = round(time.perf_counter() - step_start, 3)
        result = {
            "schema_version": "falcon.controller_round1_training.v1",
            "round_id": round_id,
            "started_at": started_at,
            "finished_at": _timestamp(),
            "runtime_seconds": total_runtime,
            "runtime_human_readable": _human_duration(total_runtime),
            "training_runtime_seconds": training_runtime,
            "training_runtime_human_readable": _human_duration(training_runtime),
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "train_summary": train_summary,
            "used_sampling_plan": bool(sampling_plan_path),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "selected_fallback_scenario": selected_fallback_scenario,
            "warnings": sorted(set(warnings)),
        }
        self.state[f"round{round_id}_training"] = result
        if train_summary.get("actor_checkpoint_path"):
            self.state["current_checkpoint_path"] = train_summary.get("actor_checkpoint_path")
            self.state["latest_checkpoint_path"] = train_summary.get("actor_checkpoint_path")
        return result

    def _build_stability_checkpoint_validator(
        self,
        plan: Mapping[str, Any],
        round_id: int,
    ):
        stability_config = dict(self.config.get("stability_aware_training") or {})
        max_scenarios = max(
            int(stability_config.get("anchor_validation_max_scenarios", 1)), 1
        )
        validation_scenarios = []
        validation_manifest = stability_config.get("anchor_validation_manifest")
        validation_manifest_path = None
        if validation_manifest:
            validation_manifest_path = Path(str(validation_manifest))
            if not validation_manifest_path.is_absolute():
                validation_manifest_path = ROOT_DIR / validation_manifest_path
            if validation_manifest_path.exists():
                try:
                    manifest_data = json.loads(
                        validation_manifest_path.read_text(encoding="utf-8")
                    )
                    for item in manifest_data.get("scenarios") or []:
                        scenario = dict(item)
                        scenario_path = scenario.get("scenario_yaml_path")
                        if not scenario_path:
                            continue
                        resolved_path = Path(str(scenario_path))
                        if not resolved_path.is_absolute():
                            resolved_path = ROOT_DIR / resolved_path
                        if not resolved_path.exists():
                            continue
                        scenario["scenario_yaml_path"] = str(resolved_path)
                        validation_scenarios.append(scenario)
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    validation_scenarios = []
        validation_scenarios = validation_scenarios[:max_scenarios]
        if not validation_scenarios:
            anchor_candidates = [
                dict(item)
                for item in plan.get("scenario_batch") or []
                if item.get("sampling_category") == "base_anchor"
                and item.get("scenario_yaml_path")
            ]
            anchor_candidates.sort(
                key=lambda item: (
                    0 if item.get("anchor_role") == "base_scenario" else 1,
                    str(item.get("scenario_yaml_path") or ""),
                )
            )
            validation_scenarios = anchor_candidates[:max_scenarios]
        if not validation_scenarios:
            validation_scenarios = [
                {
                    "scenario_id": "base_2v2_NoWeapon_Selfplay",
                    "scenario_yaml_path": self.config["base_config_path"],
                    "anchor_role": "base_scenario",
                }
            ]
        opponent_checkpoint = stability_config.get("opponent_checkpoint")
        if opponent_checkpoint:
            opponent_path = Path(str(opponent_checkpoint))
            if not opponent_path.is_absolute():
                opponent_path = ROOT_DIR / opponent_path
            opponent_checkpoint = str(opponent_path)
        evaluator = PolicyEvaluator(
            {
                "base_config_path": str(self.config["base_config_path"]),
                "opponent_mode": stability_config.get(
                    "opponent_mode", "fixed_checkpoint"
                ),
                "opponent_checkpoint": opponent_checkpoint,
            }
        )
        episodes = max(
            int(stability_config.get("anchor_validation_episodes", 1)), 1
        )

        def validate_checkpoint(
            checkpoint_path: str,
            context: Mapping[str, Any],
        ) -> Dict[str, Any]:
            results = []
            warnings = []
            for index, scenario in enumerate(validation_scenarios):
                result = evaluator.evaluate_policy_on_scenario(
                    checkpoint_path,
                    scenario["scenario_yaml_path"],
                    num_episodes=episodes,
                    seed=700000
                    + int(round_id) * 1000
                    + int(context.get("batch_index", -1) + 1) * 10
                    + index,
                )
                results.append(
                    {
                        "scenario_id": scenario.get("scenario_id"),
                        "scenario_yaml_path": scenario.get("scenario_yaml_path"),
                        "win_rate": result.get("win_rate"),
                        "mean_return": result.get("mean_return"),
                        "failure_stage": result.get("failure_stage"),
                    }
                )
                warnings.extend(result.get("warnings") or [])
            successful = [
                item for item in results if item.get("failure_stage") is None
            ]
            if not successful:
                return {
                    "schema_version": "falcon.anchor_checkpoint_validation.v1",
                    "win_rate": None,
                    "mean_return": None,
                    "num_scenarios": len(results),
                    "failure_stage": "anchor_validation_failed",
                    "validation_manifest": (
                        str(validation_manifest_path)
                        if validation_manifest_path
                        else None
                    ),
                    "scenario_results": results,
                    "warnings": sorted(set(warnings)),
                }
            return {
                "schema_version": "falcon.anchor_checkpoint_validation.v1",
                "win_rate": round(
                    sum(float(item.get("win_rate") or 0.0) for item in successful)
                    / len(successful),
                    6,
                ),
                "mean_return": round(
                    sum(
                        float(item.get("mean_return") or 0.0)
                        for item in successful
                    )
                    / len(successful),
                    6,
                ),
                "num_scenarios": len(successful),
                "failure_stage": None,
                "validation_manifest": (
                    str(validation_manifest_path)
                    if validation_manifest_path
                    else None
                ),
                "scenario_results": results,
                "warnings": sorted(set(warnings)),
            }

        return validate_checkpoint

    def save_controller_state(self, path: str | Path) -> None:
        _write_json(Path(path), self.state)

    def load_controller_state(self, path: str | Path) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            self.state = json.load(f)
        pool_path = self.state.get("pool_path")
        if pool_path and Path(str(pool_path)).exists():
            self.pool = CurriculumPool({"max_pool_size": self.config.get("max_pool_size")}).load(pool_path)
        return self.state

    def _run_round0(self) -> Dict[str, Any]:
        round_started_at = _timestamp()
        round_start = time.perf_counter()
        training_start = time.perf_counter()
        initial_training = self.run_initial_training()
        training_runtime = round(time.perf_counter() - training_start, 3)
        failure_start = time.perf_counter()
        failure_summary = self.collect_or_load_failure_summary()
        failure_runtime = round(time.perf_counter() - failure_start, 3)
        generation_start = time.perf_counter()
        candidates, generation = self.generate_candidate_scenarios(failure_summary)
        generation_runtime = round(time.perf_counter() - generation_start, 3)
        if not candidates:
            base_config = load_base_scenario_config(self.config["base_config_path"])
            candidates = RandomScenarioGenerator({"seed": 31}).generate_from_failure_summary(failure_summary, base_config, self.config["num_candidates"])
            generation.setdefault("warnings", []).append("Generated random fallback candidates because qwen3:8b returned none.")
        validation_start = time.perf_counter()
        validation = self.validate_candidates(candidates)
        validation_runtime = round(time.perf_counter() - validation_start, 3)
        policy_eval_start = time.perf_counter()
        policy_eval = self.evaluate_candidates(validation["valid_candidates"])
        policy_eval_runtime = round(time.perf_counter() - policy_eval_start, 3)
        pool_start = time.perf_counter()
        pool_result = self.update_curriculum_pool(validation["valid_candidates"], policy_eval, failure_summary, validation)
        pool_runtime = round(time.perf_counter() - pool_start, 3)
        sampling_start = time.perf_counter()
        sampling = self.build_sampling_plan(pool_result)
        sampling_runtime = round(time.perf_counter() - sampling_start, 3)
        round_runtime = round(time.perf_counter() - round_start, 3)
        round_state = {
            "schema_version": "falcon.controller_round0.v1",
            "round_id": 0,
            "started_at": round_started_at,
            "finished_at": _timestamp(),
            "round_runtime_seconds": round_runtime,
            "round_runtime_human_readable": _human_duration(round_runtime),
            "training_runtime_seconds": training_runtime,
            "failure_runtime_seconds": failure_runtime,
            "qwen_runtime_seconds": generation_runtime,
            "candidate_generation_runtime_seconds": generation_runtime,
            "candidate_validation_runtime_seconds": validation_runtime,
            "policy_eval_runtime_seconds": policy_eval_runtime,
            "difficulty_pool_runtime_seconds": pool_runtime,
            "sampling_runtime_seconds": sampling_runtime,
            "initial_training": initial_training,
            "failure_summary": failure_summary,
            "candidate_generation": generation,
            "candidate_validation": validation,
            "policy_eval": policy_eval,
            "difficulty_results": pool_result.get("difficulty_results", []),
            "pool_path": pool_result.get("pool_path"),
            "sampling_plan_path": sampling.get("path"),
        }
        self.state["rounds"]["0"] = round_state
        self._save_round0_files(round_state)
        self.save_controller_state(self.output_dir / "falcon_controller_state_round0.json")
        return round_state

    def _run_round1(self) -> Dict[str, Any]:
        result = self.run_training_with_sampling_plan()
        self.state["rounds"]["1"] = result
        _write_json(self.output_dir / "falcon_controller_training_round1_summary.json", result)
        self.save_controller_state(self.output_dir / "falcon_controller_state_round1.json")
        return result

    def _run_course_update_round(self, round_id: int, training_result: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        round_started_at = _timestamp()
        round_start = time.perf_counter()
        failure_start = time.perf_counter()
        failure_summary = self.collect_or_load_failure_summary(round_id=round_id)
        failure_runtime = round(time.perf_counter() - failure_start, 3)
        generation_start = time.perf_counter()
        candidates, generation = self.generate_candidate_scenarios(failure_summary)
        generation_runtime = round(time.perf_counter() - generation_start, 3)
        if not candidates:
            base_config = load_base_scenario_config(self.config["base_config_path"])
            candidates = RandomScenarioGenerator({"seed": 31 + round_id}).generate_from_failure_summary(
                failure_summary,
                base_config,
                int(self.config.get("qwen_candidates_per_round", self.config.get("num_candidates", 3))),
            )
            generation.setdefault("warnings", []).append("Generated random fallback candidates because qwen3:8b returned none.")
        validation_start = time.perf_counter()
        validation = self.validate_candidates(candidates, round_id=round_id)
        validation_runtime = round(time.perf_counter() - validation_start, 3)
        policy_eval_start = time.perf_counter()
        policy_eval = self.evaluate_candidates(validation["valid_candidates"])
        policy_eval_runtime = round(time.perf_counter() - policy_eval_start, 3)
        pool_start = time.perf_counter()
        pool_result = self.update_curriculum_pool(
            validation["valid_candidates"],
            policy_eval,
            failure_summary,
            validation,
            round_id=round_id,
        )
        pool_runtime = round(time.perf_counter() - pool_start, 3)
        sampling_start = time.perf_counter()
        sampling = self.build_sampling_plan(pool_result, round_id=round_id)
        sampling_runtime = round(time.perf_counter() - sampling_start, 3)
        self._maybe_update_best_checkpoint(round_id, policy_eval)
        round_runtime = round(time.perf_counter() - round_start, 3)
        round_state = {
            "schema_version": "falcon.controller_course_update_round.v1",
            "round_id": round_id,
            "started_at": round_started_at,
            "finished_at": _timestamp(),
            "round_runtime_seconds": round_runtime,
            "round_runtime_human_readable": _human_duration(round_runtime),
            "failure_runtime_seconds": failure_runtime,
            "qwen_runtime_seconds": generation_runtime,
            "candidate_generation_runtime_seconds": generation_runtime,
            "candidate_validation_runtime_seconds": validation_runtime,
            "policy_eval_runtime_seconds": policy_eval_runtime,
            "difficulty_pool_runtime_seconds": pool_runtime,
            "sampling_runtime_seconds": sampling_runtime,
            "training_result": dict(training_result or {}),
            "failure_summary": failure_summary,
            "candidate_generation": generation,
            "candidate_validation": validation,
            "policy_eval": policy_eval,
            "difficulty_results": pool_result.get("difficulty_results", []),
            "pool_path": pool_result.get("pool_path"),
            "sampling_plan_path": sampling.get("path"),
        }
        self.state["rounds"][str(round_id)] = round_state
        self._save_round_files(round_id, round_state)
        if self.config.get("save_every_round", True):
            self.save_controller_state(self.output_dir / f"controller_state_round{round_id}.json")
        return round_state

    def _collect_real_failure_summary(self, round_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        current = self.state.get("current_checkpoint_path")
        if not current or not Path(str(current)).exists():
            self.state["warnings"].append("Real failure trajectory collection skipped because current checkpoint was unavailable.")
            return None
        round_label = f"round{round_id}" if round_id is not None else "round_unknown"
        trajectory_dir = self.output_dir / f"{round_label}_failure_trajectories"
        base_config = load_base_scenario_config(self.config["base_config_path"])
        eval_config = copy.deepcopy(base_config)
        eval_config["scenario_id"] = f"{round_label}_failure_collection"
        eval_config["trajectory_recording"] = {
            "enabled": True,
            "output_dir": str(trajectory_dir),
            "save_success": True,
            "prefix": f"{round_label}_failure_eval",
            "metadata": {
                "round_id": round_id,
                "policy_id": Path(str(current)).stem,
                "save_reason": "failure_collection",
            },
        }
        try:
            evaluator_config = _deep_merge(
                {"base_config_path": str(self.config["base_config_path"])},
                dict(self.config.get("policy_evaluation") or {}),
            )
            eval_result = PolicyEvaluator(evaluator_config).evaluate_policy_on_scenario(
                current,
                eval_config,
                num_episodes=int(self.config.get("eval_episodes_per_round", 1)),
                seed=1000 + int(round_id or 0),
            )
        except Exception as exc:  # noqa: BLE001
            self.state["warnings"].append(f"Real failure trajectory rollout failed: {type(exc).__name__}: {exc}")
            return None
        trajectory_files = sorted(trajectory_dir.glob("*.json")) if trajectory_dir.exists() else []
        if not trajectory_files:
            self.state["warnings"].append("Real failure trajectory rollout produced no saved trajectory files.")
            return None
        selected_path, source_type = _select_failure_trajectory(trajectory_files)
        try:
            trajectory = load_trajectory(selected_path)
        except Exception as exc:  # noqa: BLE001
            self.state["warnings"].append(f"Failed to load real rollout trajectory {selected_path}: {type(exc).__name__}: {exc}")
            return None
        trajectory["_source_trajectory"] = str(selected_path)
        failure_summary = FailureAnalyzer().analyze_trajectory(
            trajectory,
            success_stats={"mean_success_team_reward": 500.0},
        )
        failure_summary["scenario_vector"] = extract_scenario_vector(trajectory)
        failure_summary["episode_summary"] = summarize_episode(trajectory)
        failure_summary["failure_source"] = {
            "type": source_type,
            "trajectory_path": str(selected_path),
            "round_id": round_id,
            "policy_eval": eval_result,
        }
        stats = self.state.setdefault(
            "failure_collection_stats",
            {"real_failure_trajectory_used_count": 0, "fallback_failure_used_count": 0},
        )
        if source_type == "real_failure_trajectory":
            stats["real_failure_trajectory_used_count"] = int(stats.get("real_failure_trajectory_used_count", 0)) + 1
        else:
            stats["fallback_failure_used_count"] = int(stats.get("fallback_failure_used_count", 0)) + 1
            self.state["warnings"].append("No failed episode was found; used hardest low-return real trajectory for failure analysis.")
        return failure_summary

    def _register_checkpoint(
        self,
        round_id: int,
        role: str,
        checkpoint_path: Optional[str],
        training_summary: Optional[Mapping[str, Any]] = None,
        eval_win_rate: Optional[float] = None,
    ) -> None:
        if not checkpoint_path:
            return
        registry = self.state.setdefault("checkpoint_registry", {})
        registry.setdefault("schema_version", "falcon.checkpoint_registry.v1")
        registry.setdefault("checkpoints", [])
        entry = {
            "round_id": round_id,
            "role": role,
            "checkpoint_path": checkpoint_path,
            "exists": Path(str(checkpoint_path)).exists(),
            "eval_win_rate": eval_win_rate,
            "training_summary": dict(training_summary or {}),
            "created_at": _timestamp(),
        }
        registry["checkpoints"].append(entry)
        registry["latest_checkpoint"] = checkpoint_path
        registry["current_checkpoint"] = self.state.get("current_checkpoint_path") or checkpoint_path
        self.state["latest_checkpoint_path"] = checkpoint_path
        if registry.get("best_checkpoint") is None:
            registry["best_checkpoint"] = self.state.get("best_checkpoint_path") or checkpoint_path
            self.state["best_checkpoint_path"] = registry["best_checkpoint"]

    def _maybe_update_best_checkpoint(self, round_id: int, policy_eval: Mapping[str, Any]) -> None:
        current = self.state.get("current_checkpoint_path")
        if not current:
            return
        eval_results = list(policy_eval.get("policy_eval_results") or [])
        win_rates = [
            _float(item.get("current_policy_eval", {}).get("win_rate"))
            for item in eval_results
            if item.get("current_policy_eval", {}).get("real_policy_eval_available")
        ]
        if not win_rates:
            return
        mean_win_rate = sum(win_rates) / len(win_rates)
        registry = self.state.setdefault("checkpoint_registry", {})
        best_win_rate = registry.get("best_win_rate")
        if len(win_rates) < 3:
            warning = "Best checkpoint update is based on very few eval episodes; smoke readiness only."
            registry.setdefault("warnings", [])
            if warning not in registry["warnings"]:
                registry["warnings"].append(warning)
        if best_win_rate is None or mean_win_rate > _float(best_win_rate):
            registry["best_win_rate"] = mean_win_rate
            registry["best_checkpoint"] = current
            self.state["best_checkpoint_path"] = current
        self._register_checkpoint(round_id, "evaluated_current", current, eval_win_rate=mean_win_rate)

    def _save_checkpoint_registry(self) -> None:
        registry = self.state.get("checkpoint_registry") or {}
        registry["current_checkpoint"] = self.state.get("current_checkpoint_path")
        registry["best_checkpoint"] = self.state.get("best_checkpoint_path") or registry.get("best_checkpoint")
        registry["latest_checkpoint"] = self.state.get("latest_checkpoint_path") or registry.get("latest_checkpoint")
        _write_json(self.output_dir / "falcon_checkpoint_registry.json", registry)

    def _fallback_training_scenario(self) -> Tuple[Dict[str, Any], str]:
        round_ids = sorted((int(key) for key in self.state.get("rounds", {}).keys() if str(key).isdigit()), reverse=True)
        valid_candidates: List[Mapping[str, Any]] = []
        difficulty_results: List[Mapping[str, Any]] = []
        for round_id in round_ids + [0]:
            valid_candidates = list(self.state.get(f"valid_candidates_round{round_id}") or [])
            difficulty_results = list(self.state.get(f"difficulty_results_round{round_id}") or [])
            if valid_candidates and difficulty_results:
                break
        if valid_candidates and difficulty_results:
            best_idx = max(range(len(valid_candidates)), key=lambda idx: _float(difficulty_results[idx].get("final_value_score")) if idx < len(difficulty_results) else 0.0)
            candidate = dict(valid_candidates[best_idx])
            result = difficulty_results[best_idx]
            return (
                {
                    "scenario_id": candidate.get("scenario_id"),
                    "source": "fallback_best_validated_candidate",
                    "scenario_yaml_path": candidate.get("scenario_yaml_path") or candidate.get("yaml_path"),
                    "sampling_weight": 1.0,
                    "final_value_score": result.get("final_value_score"),
                    "target_failure_modes": candidate.get("target_failure_modes") or [],
                    "priority_level": result.get("priority_level"),
                },
                "accepted pool was empty or sampling plan selected base; using best validated candidate by final_value_score.",
            )
        return (
            {
                "scenario_id": "base_2v2_NoWeapon_Selfplay",
                "source": "base",
                "scenario_yaml_path": str(self.config["base_config_path"]),
                "sampling_weight": 1.0,
                "final_value_score": 0.0,
                "target_failure_modes": [],
                "priority_level": "base",
            },
            "no validated candidate was available; using base scenario.",
        )

    def _save_round_files(self, round_id: int, round_state: Mapping[str, Any]) -> None:
        _write_json(self.output_dir / "falcon_longrun_config.json", self.config)
        _write_json(self.output_dir / f"falcon_controller_failure_summary_round{round_id}.json", round_state["failure_summary"])
        _write_json(self.output_dir / f"falcon_controller_candidates_round{round_id}.json", round_state["candidate_generation"])
        _write_json(self.output_dir / f"falcon_controller_validated_candidates_round{round_id}.json", round_state["candidate_validation"])
        _write_json(self.output_dir / f"falcon_controller_policy_eval_round{round_id}.json", round_state["policy_eval"])
        _write_json(
            self.output_dir / f"falcon_controller_difficulty_round{round_id}.json",
            {
                "schema_version": "falcon.controller_difficulty_round.v1",
                "round_id": round_id,
                "difficulty_results": round_state.get("difficulty_results", []),
            },
        )
        if round_state.get("training_result"):
            _write_json(self.output_dir / f"falcon_controller_training_round{round_id}_summary.json", round_state["training_result"])
        training_result = round_state.get("training_result") or {}
        multi_training = training_result.get("multi_scenario_training_summary") or {}
        round_summary = {
            "schema_version": "falcon.controller_round_summary.v1",
            "round_id": round_id,
            "started_at": round_state.get("started_at"),
            "finished_at": round_state.get("finished_at"),
            "round_runtime_seconds": round_state.get("round_runtime_seconds"),
            "round_runtime_human_readable": round_state.get("round_runtime_human_readable"),
            "training_runtime_seconds": ((round_state.get("training_result") or {}).get("training_runtime_seconds")),
            "failure_runtime_seconds": round_state.get("failure_runtime_seconds"),
            "qwen_runtime_seconds": round_state.get("qwen_runtime_seconds"),
            "candidate_generation_runtime_seconds": round_state.get("candidate_generation_runtime_seconds"),
            "candidate_validation_runtime_seconds": round_state.get("candidate_validation_runtime_seconds"),
            "policy_eval_runtime_seconds": round_state.get("policy_eval_runtime_seconds"),
            "difficulty_pool_runtime_seconds": round_state.get("difficulty_pool_runtime_seconds"),
            "sampling_runtime_seconds": round_state.get("sampling_runtime_seconds"),
            "checkpoint_path": self.state.get("current_checkpoint_path"),
            "scenario_batch_size": multi_training.get("scenario_batch_size", 1 if training_result else 0),
            "scenarios_actually_trained": multi_training.get(
                "scenarios_actually_trained",
                1 if (training_result.get("train_summary") or {}).get("checkpoint_saved") else 0,
            ),
            "anchor_scenarios_used": training_result.get("anchor_scenarios_used", []),
            "anchor_ratio": training_result.get("anchor_ratio", 0.0),
            "accepted_ratio": training_result.get("accepted_ratio", 0.0),
            "replay_ratio": training_result.get("replay_ratio", 0.0),
            "checkpoint_continuity_complete": multi_training.get("checkpoint_continuity_complete"),
            "stability_aware": training_result.get("stability_aware", False),
            "selected_batch_index": training_result.get("selected_batch_index"),
            "selected_checkpoint_path": training_result.get(
                "selected_checkpoint_path"
            ),
            "terminal_checkpoint_path": training_result.get(
                "terminal_checkpoint_path"
            ),
            "selected_checkpoint_validation": training_result.get(
                "selected_checkpoint_validation"
            ),
            "num_candidates_generated": len((round_state.get("candidate_generation") or {}).get("candidates") or []),
            "num_schema_valid": sum(1 for item in (round_state.get("candidate_validation") or {}).get("schema_validations", []) if item.get("is_valid")),
            "num_constraint_valid": sum(1 for item in (round_state.get("candidate_validation") or {}).get("constraint_results", []) if item.get("is_valid")),
            "num_policy_eval_success": sum(
                1
                for item in (round_state.get("policy_eval") or {}).get("policy_eval_results", [])
                if item.get("current_policy_eval", {}).get("real_policy_eval_available")
                and item.get("best_policy_eval", {}).get("real_policy_eval_available")
            ),
            "num_difficulty_evaluated": len(round_state.get("difficulty_results", [])),
            "num_accepted_into_pool": sum(1 for item in round_state.get("difficulty_results", []) if item.get("accepted_into_curriculum_pool")),
            "sampling_plan_path": round_state.get("sampling_plan_path"),
            "warnings": sorted(
                set(
                    (round_state.get("candidate_generation") or {}).get("warnings", [])
                    + (round_state.get("policy_eval") or {}).get("warnings", [])
                )
            ),
        }
        _write_json(self.output_dir / f"falcon_controller_round{round_id}_summary.json", round_summary)

    def _save_round0_files(self, round_state: Mapping[str, Any]) -> None:
        _write_json(self.output_dir / "falcon_controller_config.json", self.config)
        _write_json(self.output_dir / "falcon_controller_failure_summary_round0.json", round_state["failure_summary"])
        _write_json(self.output_dir / "falcon_controller_candidates_round0.json", round_state["candidate_generation"])
        _write_json(self.output_dir / "falcon_controller_policy_eval_round0.json", round_state["policy_eval"])
        _write_json(
            self.output_dir / "falcon_controller_difficulty_round0.json",
            {
                "schema_version": "falcon.controller_difficulty_round0.v1",
                "difficulty_results": round_state.get("difficulty_results", []),
            },
        )


def _build_train_args(
    scenario_name: str,
    num_env_steps: int,
    buffer_size: int,
    seed: int,
    scenario_config_path: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> argparse.Namespace:
    from config import get_config
    from scripts.train.train_jsbsim import parse_args

    smoke_args = [
        "--env-name",
        "MultipleCombat",
        "--algorithm-name",
        "mappo",
        "--scenario-name",
        scenario_name,
        "--experiment-name",
        "falcon_controller_smoke",
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
    if scenario_config_path:
        smoke_args.extend(["--scenario-config-path", str(scenario_config_path)])
    if model_dir:
        smoke_args.extend(["--model-dir", str(model_dir)])
    return parse_args(smoke_args, get_config())


def _train_mappo_smoke(
    scenario_name: str,
    output_dir: Path,
    num_env_steps: int,
    buffer_size: int,
    seed: int,
    training_config_path: Optional[str] = None,
    scenario_config_path: Optional[str] = None,
    requires_parse_config_patch: bool = False,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    from runner.share_jsbsim_runner import ShareJSBSimRunner
    from scripts.train.train_jsbsim import make_train_env

    save_dir = output_dir / "mappo"
    actor_path = save_dir / "actor_latest.pt"
    critic_path = save_dir / "critic_latest.pt"
    warnings: List[str] = []
    summary = {
        "schema_version": "falcon.controller_mappo_train_smoke.v1",
        "started_at": _timestamp(),
        "training_started": False,
        "training_finished": False,
        "checkpoint_saved": False,
        "actor_checkpoint_path": str(actor_path),
        "critic_checkpoint_path": str(critic_path),
        "scenario_name": scenario_name,
        "training_config_path": training_config_path,
        "scenario_config_path": scenario_config_path,
        "requires_parse_config_patch": bool(requires_parse_config_patch),
        "model_dir": model_dir,
        "continued_from_checkpoint": bool(model_dir),
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
                raise FileNotFoundError(f"training_config_path does not exist: {training_config_path}")
            import yaml
            from envs.JSBSim.envs import env_base

            with Path(training_config_path).open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            original_parse_config = env_base.parse_config

            def _parse_config_override(filename):
                if filename == scenario_name:
                    return type("EnvConfig", (object,), config_data)
                return original_parse_config(filename)

            env_base.parse_config = _parse_config_override
            warnings.append("Applied temporary parse_config mapping for FALCON staged YAML.")
        all_args = _build_train_args(
            scenario_name,
            num_env_steps,
            buffer_size,
            seed,
            scenario_config_path=scenario_config_path,
            model_dir=model_dir,
        )
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
    except Exception as exc:  # noqa: BLE001
        summary["failure_stage"] = "training_failed"
        warnings.append(f"MAPPO smoke training failed: {type(exc).__name__}: {exc}")
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
    runtime = round(max(0.0, time.time() - start_time), 3)
    summary["finished_at"] = _timestamp()
    summary["training_runtime_seconds"] = runtime
    summary["training_runtime_human_readable"] = _human_duration(runtime)
    summary["checkpoint_saved"] = bool(actor_exists and critic_exists)
    summary["actor_checkpoint_path"] = str(actor_path) if actor_exists else None
    summary["critic_checkpoint_path"] = str(critic_path) if critic_exists else None
    if summary["failure_stage"] is None and not summary["checkpoint_saved"]:
        summary["failure_stage"] = "checkpoint_not_saved"
        warnings.append("Training finished but actor_latest.pt and critic_latest.pt were not both found.")
    if actor_exists and actor_path.stat().st_mtime < start_time:
        warnings.append("actor_latest.pt exists but was not modified during this smoke run.")
    return summary


def _invalid_constraint(candidate: Mapping[str, Any], validation: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scenario_id": candidate.get("scenario_id", f"candidate_{idx:04d}"),
        "is_valid": False,
        "validity_score": 0.0,
        "rejection_reasons": ["candidate_schema_invalid"],
        "physical_constraint_check": {},
        "task_constraint_check": {},
        "missing_fields": validation.get("missing_fields", []),
        "warnings": validation.get("warnings", []),
    }


def _policy_eval_failed_difficulty(candidate: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "schema_version": "falcon.difficulty_evaluation.v1",
        "scenario_id": candidate.get("scenario_id", f"candidate_{idx:04d}"),
        "current_policy_weakness": 0.0,
        "historical_solvability": 0.0,
        "learning_potential": 0.0,
        "scenario_diversity": 0.0,
        "failure_mode_match": 0.0,
        "constraint_validity": 0.0,
        "hard_filter_passed": False,
        "rejection_reasons": ["policy_eval_failed_or_missing"],
        "final_value_score": 0.0,
        "accepted_into_curriculum_pool": False,
        "sampling_weight": 0.0,
        "priority_level": "low",
        "warnings": ["Policy evaluation did not produce real current and best rollout results; skipped difficulty scoring."],
    }


def _candidate_batch_source(candidates: Sequence[Mapping[str, Any]]) -> str:
    generator_types = " ".join(str(candidate.get("generator_type", "")) for candidate in candidates).lower()
    if "qwen3_8b" in generator_types or "qwen3:8b" in generator_types or "ollama" in generator_types:
        return "llm_qwen8b"
    if "random" in generator_types:
        return "random"
    return "controller_round0"


def _candidate_pool_source(candidate: Mapping[str, Any]) -> str:
    generator_type = str(candidate.get("generator_type") or "").lower()
    if generator_type == "fsn" or "fsn" in generator_type:
        return "fsn"
    if generator_type == "qwen" or "qwen" in generator_type or "ollama" in generator_type:
        return "llm_qwen8b"
    if "random" in generator_type:
        return "random"
    if "replay" in generator_type or "failure" in generator_type:
        return "replay"
    return _candidate_batch_source([candidate])


def _checkpoint_model_dir(checkpoint_path: Any) -> Optional[str]:
    if not checkpoint_path:
        return None
    path = Path(str(checkpoint_path))
    return str(path.parent) if path.exists() else None


def _is_falcon_generated_training_scenario(selected: Mapping[str, Any], base_config_path: str | Path) -> bool:
    if not selected:
        return False
    yaml_path = selected.get("scenario_yaml_path") or selected.get("yaml_path")
    if not yaml_path:
        return False
    try:
        if Path(str(yaml_path)).resolve() == Path(base_config_path).resolve():
            return False
    except OSError:
        pass
    return Path(str(yaml_path)).exists()


def _first_falcon_generated_from_plan(plan: Mapping[str, Any], base_config_path: str | Path) -> Optional[Dict[str, Any]]:
    for item in plan.get("sampled_scenarios", []) or []:
        if isinstance(item, MappingABC) and _is_falcon_generated_training_scenario(item, base_config_path):
            return dict(item)
    return None


def _select_failure_trajectory(paths: Sequence[Path]) -> Tuple[Path, str]:
    loaded: List[Tuple[Path, Dict[str, Any], float]] = []
    for path in paths:
        try:
            data = load_trajectory(path)
            summary = data.get("episode_summary") if isinstance(data.get("episode_summary"), MappingABC) else summarize_episode(data)
            reward = _team_reward_own(summary.get("total_team_reward") or data.get("total_team_reward"))
            loaded.append((path, data, reward))
        except Exception:
            loaded.append((path, {}, 0.0))
    for path, data, _reward in loaded:
        result = str(data.get("episode_result") or (data.get("episode_summary") or {}).get("final_outcome") or "").lower()
        if result and result not in {"win", "success"}:
            return path, "real_failure_trajectory"
    if loaded:
        return min(loaded, key=lambda item: item[2])[0], "hardest_low_return_trajectory"
    raise ValueError("No trajectory paths were provided.")


def _team_reward_own(value: Any) -> float:
    if isinstance(value, MappingABC):
        return _float(value.get("own", value.get("team", 0.0)))
    return _float(value)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
