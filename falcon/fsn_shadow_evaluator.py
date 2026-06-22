"""Policy-evaluated offline shadow replacement utilities for FSN Stage 3."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .candidate_schema import validate_candidate_schema
from .constraint_checker import ConstraintChecker
from .difficulty_evaluator import DifficultyEvaluator
from .fsn_generator import FSNScenarioGenerator
from .policy_evaluator import PolicyEvaluator
from .random_scenario_generator import RandomScenarioGenerator
from .scenario_adapter import (
    apply_initial_config_to_yaml,
    load_base_scenario_config,
    save_scenario_yaml,
)


class FSNPolicyEvaluatedShadowPilot:
    """Evaluate FSN/Qwen/random candidates without adding them to training."""

    def __init__(
        self,
        workspace_root: str | Path,
        stage2_checkpoint: str | Path,
        fixed_opponent_checkpoint: str | Path,
        base_config_path: str | Path,
        output_dir: str | Path,
        episodes_per_candidate: int = 1,
    ) -> None:
        self.root = Path(workspace_root).resolve()
        self.checkpoint = self._resolve(stage2_checkpoint)
        self.opponent_checkpoint = self._resolve(fixed_opponent_checkpoint)
        self.base_config_path = self._resolve(base_config_path)
        self.output_dir = self._resolve(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = max(int(episodes_per_candidate), 1)
        self.base_config = load_base_scenario_config(self.base_config_path)
        self.constraint_checker = ConstraintChecker()
        self.difficulty_evaluator = DifficultyEvaluator()
        self.policy_evaluator = PolicyEvaluator(
            {
                "base_config_path": str(self.base_config_path),
                "opponent_mode": "fixed_checkpoint",
                "opponent_checkpoint": str(self.opponent_checkpoint),
                "deterministic": True,
                "device": "cpu",
            }
        )

    def collect_failure_summaries(
        self,
        results_root: str | Path,
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        per_seed: int = 4,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results_root = self._resolve(results_root)
        records: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for seed in seeds:
            run_dir = (
                results_root
                / "falcon_no_fsn"
                / f"seed_{seed}"
                / "pilot_run"
            )
            controller_dir = run_dir / "controller"
            failure_paths = sorted(
                controller_dir.glob(
                    "falcon_controller_failure_summary_round*.json"
                ),
                key=_round_from_path,
            )
            selected_paths = _evenly_select(failure_paths, per_seed)
            baseline_summary = _load_json(
                run_dir / "baseline_experiment_summary.json"
            )
            best_checkpoint = baseline_summary.get(
                "best_checkpoint_path"
            ) or (baseline_summary.get("execution") or {}).get(
                "best_checkpoint_path"
            )
            registry = _load_json(
                controller_dir / "falcon_checkpoint_registry.json"
            )
            for path in selected_paths:
                round_id = _round_from_path(path)
                failure_summary = _load_json(path)
                current_checkpoint = _checkpoint_for_round(
                    registry, round_id
                )
                qwen_path = (
                    controller_dir
                    / f"falcon_controller_candidates_round{round_id}.json"
                )
                missing = []
                if not current_checkpoint or not Path(current_checkpoint).exists():
                    missing.append("current_checkpoint")
                if not best_checkpoint or not Path(best_checkpoint).exists():
                    missing.append("best_checkpoint")
                if not qwen_path.exists():
                    missing.append("historical_qwen_candidates")
                if missing:
                    warnings.append(
                        f"seed {seed} round {round_id} missing: "
                        + ", ".join(missing)
                    )
                raw_failure_id = _failure_id(failure_summary)
                records.append(
                    {
                        "schema_version": (
                            "falcon.fsn_stage3_failure_summary.v1"
                        ),
                        "failure_id": f"seed{seed}_{raw_failure_id}",
                        "raw_failure_id": raw_failure_id,
                        "seed": seed,
                        "round_id": round_id,
                        "failure_summary_path": str(path.resolve()),
                        "historical_qwen_candidates_path": str(
                            qwen_path.resolve()
                        ),
                        "current_checkpoint": current_checkpoint,
                        "best_checkpoint": best_checkpoint,
                        "failure_summary": failure_summary,
                        "missing_fields": missing,
                    }
                )

        score_values: Dict[str, List[float]] = {}
        mode_counts: Counter[str] = Counter()
        for record in records:
            summary = record["failure_summary"]
            mode_counts.update(
                set(summary.get("primary_failure_modes") or [])
                | set(summary.get("secondary_failure_modes") or [])
            )
            for key, value in (summary.get("failure_scores") or {}).items():
                number = _number(value)
                if number is not None:
                    score_values.setdefault(str(key), []).append(number)
        mode_names = (
            "coordination_failure",
            "target_assignment_confusion",
            "initial_disadvantage",
            "generalization_failure",
            "failure_severity",
        )
        sparse_modes = [
            key
            for key in mode_names
            if mode_counts.get(key, 0) == 0
            and not any(
                abs(value) > 1e-8 for value in score_values.get(key, [])
            )
        ]
        if sparse_modes:
            warnings.append(
                "Failure modes without nonzero/label coverage: "
                + ", ".join(sparse_modes)
            )
        stats = {
            "schema_version": "falcon.fsn_stage3_failure_stats.v1",
            "total_failure_summaries": len(records),
            "seed_counts": dict(
                sorted(Counter(str(item["seed"]) for item in records).items())
            ),
            "round_ids_by_seed": {
                str(seed): [
                    item["round_id"]
                    for item in records
                    if item["seed"] == seed
                ]
                for seed in seeds
            },
            "failure_mode_counts": dict(sorted(mode_counts.items())),
            "failure_score_stats": {
                key: _distribution(values)
                for key, values in sorted(score_values.items())
            },
            "sparse_failure_modes": sparse_modes,
            "all_current_checkpoints_available": all(
                Path(str(item["current_checkpoint"])).exists()
                for item in records
                if item.get("current_checkpoint")
            ),
            "all_best_checkpoints_available": all(
                Path(str(item["best_checkpoint"])).exists()
                for item in records
                if item.get("best_checkpoint")
            ),
            "warnings": sorted(set(warnings)),
        }
        return records, stats

    def save_failure_set(
        self,
        records: Sequence[Mapping[str, Any]],
        stats: Mapping[str, Any],
    ) -> None:
        _write_json(
            self.output_dir / "fsn_shadow_failure_summaries.json",
            {
                "schema_version": "falcon.fsn_stage3_failure_set.v1",
                "failure_summaries": list(records),
            },
        )
        _write_json(
            self.output_dir / "fsn_shadow_failure_summary_stats.json",
            stats,
        )

    def evaluate(
        self,
        failure_records: Sequence[Mapping[str, Any]],
        pool_stats: Mapping[str, Any],
        candidates_per_generator: int = 4,
        resume: bool = True,
    ) -> Dict[str, Any]:
        output_path = (
            self.output_dir
            / "fsn_policy_evaluated_shadow_candidates.json"
        )
        payload = (
            _load_json(output_path)
            if resume and output_path.exists()
            else {}
        )
        completed_ids = set(payload.get("completed_failure_ids") or [])
        evaluated_records = list(payload.get("candidate_records") or [])
        started = time.perf_counter()

        for failure_index, failure_record in enumerate(failure_records):
            failure_id = str(failure_record.get("failure_id"))
            if failure_id in completed_ids:
                continue
            if failure_record.get("missing_fields"):
                evaluated_records.append(
                    {
                        "failure_id": failure_id,
                        "seed": failure_record.get("seed"),
                        "round_id": failure_record.get("round_id"),
                        "skipped": True,
                        "warnings": [
                            "Failure record skipped because required artifacts are missing."
                        ],
                    }
                )
                completed_ids.add(failure_id)
                continue
            failure_summary = dict(
                failure_record.get("failure_summary") or {}
            )
            generated = self._generate_candidates(
                failure_record,
                candidates_per_generator,
                failure_index,
            )
            for generator_type, generator_payload in generated.items():
                for candidate_index, candidate in enumerate(
                    generator_payload["candidates"]
                ):
                    unique_id = (
                        f"s{failure_record['seed']}_r"
                        f"{failure_record['round_id']}_{generator_type}_"
                        f"{candidate_index:02d}"
                    )
                    candidate = dict(candidate)
                    candidate["scenario_id"] = unique_id
                    evaluated_records.append(
                        self._evaluate_candidate(
                            candidate=candidate,
                            generator_type=generator_type,
                            generation_runtime_seconds=(
                                generator_payload[
                                    "runtime_seconds_per_candidate"
                                ]
                            ),
                            failure_record=failure_record,
                            failure_summary=failure_summary,
                            pool_stats=self._pool_stats_for_failure(
                                pool_stats, failure_record
                            ),
                            seed=100000
                            + failure_index * 100
                            + candidate_index,
                        )
                    )
            completed_ids.add(failure_id)
            payload = {
                "schema_version": (
                    "falcon.fsn_stage3_policy_evaluated_candidates.v1"
                ),
                "episodes_per_candidate": self.episodes,
                "fixed_opponent_checkpoint": str(
                    self.opponent_checkpoint
                ),
                "same_actor": False,
                "completed_failure_ids": sorted(completed_ids),
                "candidate_records": evaluated_records,
                "runtime_seconds": round(
                    time.perf_counter() - started, 6
                ),
                "warnings": [
                    "No candidate was added to a training pool.",
                    "No Qwen generation call was made.",
                ],
            }
            _write_json(output_path, payload)
        return payload

    @staticmethod
    def _pool_stats_for_failure(
        pool_stats: Mapping[str, Any],
        failure_record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        records = pool_stats.get("records")
        if not isinstance(records, list):
            return dict(pool_stats)
        seed = failure_record.get("seed")
        round_id = failure_record.get("round_id")
        vectors = [
            item.get("candidate_scenario_vector")
            for item in records
            if item.get("label") == "accepted"
            and not item.get("synthetic")
            and item.get("seed") == seed
            and (
                item.get("round_id") is not None
                and round_id is not None
                and int(item["round_id"]) < int(round_id)
            )
            and item.get("candidate_scenario_vector")
        ]
        return {"scenario_vectors": vectors}

    def _generate_candidates(
        self,
        failure_record: Mapping[str, Any],
        count: int,
        failure_index: int,
    ) -> Dict[str, Dict[str, Any]]:
        summary = failure_record.get("failure_summary") or {}
        plain = FSNScenarioGenerator(
            self.checkpoint,
            {
                "seed": 1000 + failure_index,
                "diversity_aware": False,
                "noise_scale": 0.02,
            },
        )
        diverse = FSNScenarioGenerator(
            self.checkpoint,
            {
                "seed": 2000 + failure_index,
                "diversity_aware": True,
                "oversample_factor": 6,
                "noise_scale": 0.10,
            },
        )
        random_generator = RandomScenarioGenerator(
            {"seed": 3000 + failure_index}
        )
        started = time.perf_counter()
        plain_candidates = plain.generate_from_failure_summary(
            summary, self.base_config, count
        )
        plain_runtime = time.perf_counter() - started
        started = time.perf_counter()
        diverse_candidates = diverse.generate_from_failure_summary(
            summary, self.base_config, count
        )
        diverse_runtime = time.perf_counter() - started
        started = time.perf_counter()
        random_candidates = random_generator.generate_from_failure_summary(
            summary, self.base_config, count
        )
        random_runtime = time.perf_counter() - started
        qwen_payload = _load_json(
            Path(
                str(
                    failure_record.get(
                        "historical_qwen_candidates_path"
                    )
                )
            )
        )
        historical = [
            dict(candidate)
            for candidate in qwen_payload.get("candidates") or []
        ][:count]
        return {
            "fsn": {
                "candidates": plain_candidates,
                "runtime_seconds_per_candidate": plain_runtime
                / max(len(plain_candidates), 1),
            },
            "fsn_diversity_aware": {
                "candidates": diverse_candidates,
                "runtime_seconds_per_candidate": diverse_runtime
                / max(len(diverse_candidates), 1),
            },
            "random": {
                "candidates": random_candidates,
                "runtime_seconds_per_candidate": random_runtime
                / max(len(random_candidates), 1),
            },
            "historical_qwen": {
                "candidates": historical,
                "runtime_seconds_per_candidate": None,
            },
        }

    def _evaluate_candidate(
        self,
        candidate: Mapping[str, Any],
        generator_type: str,
        generation_runtime_seconds: Optional[float],
        failure_record: Mapping[str, Any],
        failure_summary: Mapping[str, Any],
        pool_stats: Mapping[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        schema_result = validate_candidate_schema(candidate)
        constraint_result = self.constraint_checker.validate_candidate(
            candidate
        )
        yaml_path = (
            self.output_dir
            / "generated_yamls"
            / generator_type
            / f"{candidate['scenario_id']}.yaml"
        )
        env_result: Dict[str, Any] = {}
        yaml_generated = False
        current_eval: Dict[str, Any] = {}
        best_eval: Dict[str, Any] = {}
        difficulty_result: Dict[str, Any] = {}
        if schema_result["is_valid"] and constraint_result["is_valid"]:
            yaml_config = apply_initial_config_to_yaml(
                self.base_config, candidate.get("initial_config") or {}
            )
            yaml_config["scenario_id"] = candidate["scenario_id"]
            save_scenario_yaml(yaml_config, yaml_path)
            yaml_generated = True
            env_result = self.constraint_checker.validate_yaml_config(
                yaml_config,
                enable_env_load_check=True,
                temp_config_name=f"fsn_stage3_{candidate['scenario_id']}",
            )
            env_ok = bool(
                (env_result.get("physical_constraint_check") or {}).get(
                    "scenario_loadable_env_check"
                )
            )
            if env_ok:
                current_eval = (
                    self.policy_evaluator.evaluate_policy_on_scenario(
                        failure_record["current_checkpoint"],
                        yaml_path,
                        num_episodes=self.episodes,
                        seed=seed,
                    )
                )
                if _same_path(
                    failure_record["current_checkpoint"],
                    failure_record["best_checkpoint"],
                ):
                    best_eval = dict(current_eval)
                    best_eval.setdefault("warnings", []).append(
                        "Current and best checkpoints are identical for this failure."
                    )
                else:
                    best_eval = (
                        self.policy_evaluator.evaluate_policy_on_scenario(
                            failure_record["best_checkpoint"],
                            yaml_path,
                            num_episodes=self.episodes,
                            seed=seed,
                        )
                    )
                if current_eval.get(
                    "real_policy_eval_available"
                ) and best_eval.get("real_policy_eval_available"):
                    difficulty_result = (
                        self.difficulty_evaluator.evaluate_candidate(
                            candidate,
                            current_eval,
                            best_eval,
                            pool_stats,
                            failure_summary,
                            constraint_result,
                        )
                    )
        return {
            "schema_version": "falcon.fsn_stage3_candidate_record.v1",
            "failure_id": failure_record.get("failure_id"),
            "seed": failure_record.get("seed"),
            "round_id": failure_record.get("round_id"),
            "generator_type": generator_type,
            "candidate": dict(candidate),
            "schema_result": schema_result,
            "constraint_result": constraint_result,
            "yaml_generated": yaml_generated,
            "yaml_path": str(yaml_path.resolve())
            if yaml_generated
            else None,
            "env_result": env_result,
            "current_checkpoint": failure_record.get(
                "current_checkpoint"
            ),
            "best_checkpoint": failure_record.get("best_checkpoint"),
            "fixed_opponent_checkpoint": str(self.opponent_checkpoint),
            "same_actor": False,
            "current_policy_eval": current_eval,
            "best_policy_eval": best_eval,
            "difficulty_result": difficulty_result,
            "generation_runtime_seconds": generation_runtime_seconds,
            "policy_and_validation_runtime_seconds": round(
                time.perf_counter() - started, 6
            ),
            "entered_training_pool": False,
        }

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path)
        return (
            value.resolve()
            if value.is_absolute()
            else (self.root / value).resolve()
        )


def summarize_shadow_results(
    payload: Mapping[str, Any],
    historical_qwen_seconds_per_candidate: float,
) -> Dict[str, Any]:
    records = [
        item
        for item in payload.get("candidate_records") or []
        if not item.get("skipped")
    ]
    generator_names = (
        "fsn",
        "fsn_diversity_aware",
        "random",
        "historical_qwen",
    )
    metrics: Dict[str, Dict[str, Any]] = {}
    for generator in generator_names:
        subset = [
            item
            for item in records
            if item.get("generator_type") == generator
        ]
        total = max(len(subset), 1)
        accepted = [
            item
            for item in subset
            if (item.get("difficulty_result") or {}).get(
                "accepted_into_curriculum_pool"
            )
        ]
        values = [
            value
            for item in subset
            if (
                value := _number(
                    (item.get("difficulty_result") or {}).get(
                        "final_value_score"
                    )
                )
            )
            is not None
        ]
        potentials = [
            value
            for item in subset
            if (
                value := _number(
                    (item.get("difficulty_result") or {}).get(
                        "learning_potential"
                    )
                )
            )
            is not None
        ]
        generator_runtime = [
            value
            for item in subset
            if (
                value := _number(item.get("generation_runtime_seconds"))
            )
            is not None
        ]
        rejection_counts: Counter[str] = Counter()
        factor_counts: Counter[str] = Counter()
        for item in subset:
            difficulty_reasons = (
                (item.get("difficulty_result") or {}).get(
                    "rejection_reasons"
                )
                or []
            )
            constraint_reasons = (
                (item.get("constraint_result") or {}).get(
                    "rejection_reasons"
                )
                or []
            )
            rejection_counts.update(
                difficulty_reasons or constraint_reasons
            )
            factor_counts.update(
                (item.get("candidate") or {}).get("changed_factors")
                or []
            )
        env_load_count = sum(
            bool(
                (
                    (item.get("env_result") or {}).get(
                        "physical_constraint_check"
                    )
                    or {}
                ).get("scenario_loadable_env_check")
            )
            for item in subset
        )
        policy_success_count = sum(
            bool(
                (item.get("current_policy_eval") or {}).get(
                    "real_policy_eval_available"
                )
                and (item.get("best_policy_eval") or {}).get(
                    "real_policy_eval_available"
                )
            )
            for item in subset
        )
        metrics[generator] = {
            "num_candidates": len(subset),
            "schema_valid_rate": round(
                sum(
                    bool((item.get("schema_result") or {}).get("is_valid"))
                    for item in subset
                )
                / total,
                6,
            ),
            "constraint_valid_rate": round(
                sum(
                    bool(
                        (item.get("constraint_result") or {}).get(
                            "is_valid"
                        )
                    )
                    for item in subset
                )
                / total,
                6,
            ),
            "env_load_rate": round(
                sum(
                    bool(
                        (
                            (item.get("env_result") or {}).get(
                                "physical_constraint_check"
                            )
                            or {}
                        ).get("scenario_loadable_env_check")
                    )
                    for item in subset
                )
                / total,
                6,
            ),
            "policy_eval_success_rate": round(
                policy_success_count / max(env_load_count, 1),
                6,
            ),
            "policy_eval_coverage_rate": round(
                policy_success_count / total, 6
            ),
            "difficulty_evaluated_count": sum(
                bool(item.get("difficulty_result")) for item in subset
            ),
            "difficulty_evaluation_coverage_rate": round(
                sum(bool(item.get("difficulty_result")) for item in subset)
                / total,
                6,
            ),
            "accepted_count": len(accepted),
            "accepted_rate_by_difficulty_evaluator": round(
                len(accepted) / total, 6
            ),
            "mean_final_value_score": _mean(values),
            "mean_learning_potential": _mean(potentials),
            "diversity_score": _candidate_diversity(subset),
            "rejection_reason_distribution": dict(
                sorted(rejection_counts.items())
            ),
            "changed_factor_distribution": dict(
                sorted(factor_counts.items())
            ),
            "runtime_seconds_per_candidate": (
                historical_qwen_seconds_per_candidate
                if generator == "historical_qwen"
                else _mean(generator_runtime)
            ),
            "policy_and_validation_runtime_seconds_per_candidate": _mean(
                [
                    value
                    for item in subset
                    if (
                        value := _number(
                            item.get(
                                "policy_and_validation_runtime_seconds"
                            )
                        )
                    )
                    is not None
                ]
            ),
        }
    per_seed_metrics: Dict[str, Dict[str, Any]] = {}
    seeds = sorted(
        {
            int(item["seed"])
            for item in records
            if item.get("seed") is not None
        }
    )
    for seed in seeds:
        per_seed_metrics[str(seed)] = {}
        for generator in generator_names:
            subset = [
                item
                for item in records
                if item.get("seed") == seed
                and item.get("generator_type") == generator
            ]
            total = max(len(subset), 1)
            evaluated = [
                item
                for item in subset
                if item.get("difficulty_result")
            ]
            accepted = sum(
                bool(
                    (item.get("difficulty_result") or {}).get(
                        "accepted_into_curriculum_pool"
                    )
                )
                for item in subset
            )
            per_seed_metrics[str(seed)][generator] = {
                "num_candidates": len(subset),
                "difficulty_evaluated_count": len(evaluated),
                "accepted_count": accepted,
                "accepted_rate": round(accepted / total, 6),
                "mean_final_value_score": _mean(
                    [
                        value
                        for item in evaluated
                        if (
                            value := _number(
                                (item.get("difficulty_result") or {}).get(
                                    "final_value_score"
                                )
                            )
                        )
                        is not None
                    ]
                ),
            }
    simulations = []
    fsn_metrics = metrics["fsn_diversity_aware"]
    qwen_metrics = metrics["historical_qwen"]
    for ratio in (0.25, 0.50, 0.75, 1.0):
        accepted_rate = _blend(
            fsn_metrics["accepted_rate_by_difficulty_evaluator"],
            qwen_metrics["accepted_rate_by_difficulty_evaluator"],
            ratio,
        )
        mean_value = _blend_optional(
            fsn_metrics["mean_final_value_score"],
            qwen_metrics["mean_final_value_score"],
            ratio,
        )
        diversity = _blend_optional(
            fsn_metrics["diversity_score"],
            qwen_metrics["diversity_score"],
            ratio,
        )
        fsn_runtime = float(
            fsn_metrics["runtime_seconds_per_candidate"] or 0.0
        )
        replacement_runtime = (
            ratio * fsn_runtime
            + (1.0 - ratio) * historical_qwen_seconds_per_candidate
        )
        risk_flags = []
        if accepted_rate < qwen_metrics[
            "accepted_rate_by_difficulty_evaluator"
        ]:
            risk_flags.append("expected_acceptance_below_historical_qwen")
        if diversity is not None and diversity < 0.20:
            risk_flags.append("low_expected_diversity")
        if ratio == 1.0:
            risk_flags.append("no_qwen_teacher_fallback")
        simulations.append(
            {
                "fsn_fraction": ratio,
                "historical_qwen_fraction": round(1.0 - ratio, 2),
                "expected_accepted_rate": accepted_rate,
                "expected_accepted_candidates_per_100": round(
                    accepted_rate * 100.0, 3
                ),
                "expected_mean_value_score": mean_value,
                "expected_diversity": diversity,
                "expected_qwen_call_reduction": ratio,
                "expected_runtime_reduction": round(
                    1.0
                    - replacement_runtime
                    / max(historical_qwen_seconds_per_candidate, 1e-8),
                    6,
                ),
                "risk_flags": risk_flags,
            }
        )
    return {
        "schema_version": "falcon.fsn_stage3_shadow_summary.v1",
        "num_failure_summaries": len(
            payload.get("completed_failure_ids") or []
        ),
        "episodes_per_candidate": payload.get("episodes_per_candidate"),
        "fixed_opponent_checkpoint": payload.get(
            "fixed_opponent_checkpoint"
        ),
        "same_actor": False,
        "generator_metrics": metrics,
        "per_seed_generator_metrics": per_seed_metrics,
        "replacement_simulations": simulations,
        "failure_stage": None
        if all(
            item["policy_eval_success_rate"] == 1.0
            for item in metrics.values()
            if item["num_candidates"] > 0
        )
        else "policy_evaluation_partial_failure",
        "warnings": [
            "One deterministic episode per policy/candidate gives coarse 0/1 win rates.",
            "Historical Qwen runtime is estimated from prior logs.",
            "No candidate entered MAPPO training or the curriculum pool.",
        ],
    }


def write_shadow_metrics_csv(
    summary: Mapping[str, Any], output_path: str | Path
) -> None:
    rows = []
    for generator, metrics in (
        summary.get("generator_metrics") or {}
    ).items():
        row = {"row_type": "generator", "name": generator}
        row.update(
            {
                key: value
                for key, value in metrics.items()
                if not isinstance(value, (dict, list))
            }
        )
        rows.append(row)
    for simulation in summary.get("replacement_simulations") or []:
        row = {
            "row_type": "replacement_simulation",
            "name": f"fsn_{int(simulation['fsn_fraction'] * 100)}",
        }
        row.update(
            {
                key: value
                for key, value in simulation.items()
                if not isinstance(value, (dict, list))
            }
        )
        row["risk_flags"] = "|".join(
            simulation.get("risk_flags") or []
        )
        rows.append(row)
    for seed, generators in (
        summary.get("per_seed_generator_metrics") or {}
    ).items():
        for generator, metrics in generators.items():
            rows.append(
                {
                    "row_type": "generator_seed",
                    "name": generator,
                    "seed": seed,
                    **metrics,
                }
            )
    fields = sorted({key for row in rows for key in row})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_for_round(
    registry: Mapping[str, Any], round_id: int
) -> Optional[str]:
    candidates = [
        item
        for item in registry.get("checkpoints") or []
        if item.get("round_id") == round_id
        and item.get("exists", True)
        and item.get("checkpoint_path")
    ]
    role_order = {
        "evaluated_current": 0,
        "round_checkpoint": 1,
        "initial_current": 2,
    }
    candidates.sort(key=lambda item: role_order.get(item.get("role"), 9))
    return str(candidates[0]["checkpoint_path"]) if candidates else None


def _evenly_select(paths: Sequence[Path], count: int) -> List[Path]:
    if len(paths) <= count:
        return list(paths)
    if count <= 1:
        return [paths[0]]
    indices = [
        round(index * (len(paths) - 1) / (count - 1))
        for index in range(count)
    ]
    return [paths[index] for index in sorted(set(indices))]


def _round_from_path(path: Path) -> int:
    match = re.search(r"round(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _failure_id(summary: Mapping[str, Any]) -> str:
    source = summary.get("source_trajectory") or summary.get("episode_id")
    return Path(str(source)).stem if source else "unknown_failure"


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "std": round(statistics.pstdev(values), 6)
        if len(values) > 1
        else 0.0,
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
        "nonzero_count": sum(abs(value) > 1e-8 for value in values),
    }


def _candidate_diversity(records: Sequence[Mapping[str, Any]]) -> float:
    vectors = [
        (item.get("candidate") or {}).get("scenario_vector") or {}
        for item in records
    ]
    if len(vectors) < 2:
        return 0.0
    from .difficulty_evaluator import DEFAULT_CONFIG
    from .trajectory_recorder import SCENARIO_VECTOR_KEYS

    scales = DEFAULT_CONFIG["scenario_vector_scales"]
    distances = []
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            components = []
            for key in SCENARIO_VECTOR_KEYS:
                left = _number(vectors[left_index].get(key))
                right = _number(vectors[right_index].get(key))
                if left is None or right is None:
                    continue
                components.append(
                    ((left - right) / float(scales[key])) ** 2
                )
            if components:
                distances.append(
                    math.sqrt(sum(components) / len(components))
                )
    return round(statistics.fmean(distances), 6) if distances else 0.0


def _blend(fsn: float, qwen: float, ratio: float) -> float:
    return round(ratio * float(fsn) + (1.0 - ratio) * float(qwen), 6)


def _blend_optional(
    fsn: Optional[float], qwen: Optional[float], ratio: float
) -> Optional[float]:
    if fsn is None or qwen is None:
        return None
    return _blend(fsn, qwen, ratio)


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 6) if values else None


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _same_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return str(left) == str(right)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
