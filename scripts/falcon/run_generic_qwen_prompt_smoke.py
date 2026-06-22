"""Run an isolated failure-aware versus generic Qwen prompt smoke.

This script does not train policies or alter FALCON's frozen prompt, hard
filter, MAPPO implementation, or formal experiment outputs. It compares the
frozen failure-aware generator with a generic task-only prompt, then applies
the same offline schema, constraint, YAML, environment-load, policy-evaluation,
and difficulty-evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from falcon.candidate_schema import CANDIDATE_SCHEMA_VERSION, validate_candidate_schema
from falcon.constraint_checker import ConstraintChecker
from falcon.difficulty_evaluator import DifficultyEvaluator
from falcon.llm_scenario_generator import QwenScenarioGenerator
from falcon.policy_evaluator import PolicyEvaluator
from falcon.scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config, save_scenario_yaml
from falcon.trajectory_recorder import SCENARIO_VECTOR_KEYS


SCHEMA_VERSION = "falcon.generic_qwen_prompt_smoke.v1"
PROMPT_TYPES = ("failure_aware", "generic")
CORE_FAILURE_MODES = (
    "coordination_failure",
    "target_assignment_confusion",
    "initial_disadvantage",
    "generalization_failure",
    "failure_severity",
)
FAILURE_FACTOR_MAP = {
    "coordination_failure": {
        "own_formation_spread",
        "opponent_formation_spread",
        "team_center_distance",
    },
    "target_assignment_confusion": {
        "team_center_distance",
        "opponent_formation_spread",
        "heading_difference",
        "approximate_aspect_angle",
    },
    "initial_disadvantage": {
        "team_center_distance",
        "own_formation_spread",
        "opponent_formation_spread",
        "altitude_difference",
        "velocity_difference",
        "heading_difference",
        "approximate_aspect_angle",
    },
    "generalization_failure": {
        "team_center_distance",
        "own_formation_spread",
        "opponent_formation_spread",
        "altitude_difference",
        "velocity_difference",
        "heading_difference",
        "approximate_aspect_angle",
    },
    "failure_severity": {
        "team_center_distance",
        "own_formation_spread",
        "opponent_formation_spread",
        "altitude_difference",
        "velocity_difference",
        "heading_difference",
        "approximate_aspect_angle",
    },
}


class GenericQwenScenarioGenerator(QwenScenarioGenerator):
    """Use the frozen Qwen interface with a task-only generic prompt."""

    def _build_messages(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int,
        pool_stats: Optional[Mapping[str, Any]] = None,
        retry_feedback: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        del failure_summary, base_config, pool_stats
        prompt = self.generic_prompt(int(num_scenarios))
        if retry_feedback:
            prompt += (
                "\n\nYour previous response failed validation. Return repaired JSON only. "
                "Validation feedback:\n" + retry_feedback
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are an offline training scenario generator. "
                    "You are not a controller. Do not output thinking. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def generic_prompt(self, num_scenarios: int) -> str:
        schema = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "required_fields": [
                "schema_version",
                "scenario_id",
                "generator_type",
                "target_failure_modes",
                "changed_factors",
                "scenario_vector",
                "metadata",
            ],
            "scenario_vector_keys": list(SCENARIO_VECTOR_KEYS),
        }
        return (
            "Generate challenging but valid 2v2 NoWeapon MultiCombat curriculum scenarios "
            "for improving multi-UAV policy robustness.\n"
            "You are not a flight controller. Do not output UAV actions, control commands, "
            "flight strategies, reward changes, dynamics changes, task-rule changes, or "
            "training-algorithm changes.\n"
            "Return JSON only. Do not return markdown, prose, code blocks, thinking, or "
            "reasoning traces.\n"
            f"Generate exactly {num_scenarios} CandidateScenario JSON objects.\n"
            "Use generator_type \"ollama_qwen3_8b_generic_smoke\".\n"
            "Each candidate must include target_failure_modes and changed_factors, and may "
            "change only 1 to 3 key factors. Infer broad task-relevant target labels without "
            "access to any episode-specific diagnosis. Set source_failure_id to null.\n"
            "Prefer medium-high difficulty scenarios that are challenging but not clearly "
            "unsolvable. Every numeric scenario_vector value must stay inside the allowed "
            "parameter space. All units are SI; headings and aspect angles are radians.\n\n"
            "CandidateScenario schema:\n"
            + json.dumps(schema, indent=2, sort_keys=True)
            + "\n\nAllowed parameter space:\n"
            + json.dumps(self.allowed_parameter_space, indent=2, sort_keys=True)
            + "\n\nReturn exactly this outer object shape:\n"
            + json.dumps(
                {
                    "schema_version": "falcon.qwen_scenario_response.v1",
                    "candidates": [
                        {
                            "schema_version": CANDIDATE_SCHEMA_VERSION,
                            "scenario_id": "generic_qwen_0000",
                            "generator_type": "ollama_qwen3_8b_generic_smoke",
                            "source_failure_id": None,
                            "target_failure_modes": [],
                            "changed_factors": [],
                            "counterfactual_group_id": "generic_task_context",
                            "scenario_vector": {key: None for key in SCENARIO_VECTOR_KEYS},
                            "scenario_parameters": {},
                            "initial_config": None,
                            "expected_effect": "Short expected curriculum effect.",
                            "rationale": "Short task-level rationale.",
                            "metadata": {},
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--failure-set",
        default="experiments/falcon_2v2_noweapon/fsn/stage3/fsn_shadow_failure_summaries.json",
    )
    parser.add_argument(
        "--base-config",
        default="envs/JSBSim/configs/2v2/NoWeapon/Selfplay.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falcon_2v2_noweapon/generic_qwen_smoke",
    )
    parser.add_argument("--candidates-per-summary", type=int, default=4)
    parser.add_argument("--policy-eval-per-group", type=int, default=40)
    parser.add_argument("--episodes-per-candidate", type=int, default=3)
    parser.add_argument("--sampling-seed", type=int, default=20260608)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    yamls_dir = output_dir / "generated_yamls"
    yamls_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "generic_qwen_smoke_candidates.json"
    progress_log = output_dir / "generic_qwen_smoke_progress.log"

    failures_payload = _load_json(ROOT_DIR / args.failure_set)
    failures = list(failures_payload.get("failure_summaries") or [])[:20]
    base_config = load_base_scenario_config(ROOT_DIR / args.base_config)
    _validate_failure_set(failures)

    common_qwen_config = {
        "provider": "ollama",
        "provider_mode": "ollama_native",
        "model_name": "qwen3:8b",
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 4096,
        "timeout": 180.0,
        "stream": False,
        "think": False,
        "num_retries": 2,
    }
    aware_generator = QwenScenarioGenerator(
        {**common_qwen_config, "generator_type": "ollama_qwen3_8b_failure_aware_smoke"}
    )
    generic_generator = GenericQwenScenarioGenerator(
        {**common_qwen_config, "generator_type": "ollama_qwen3_8b_generic_smoke"}
    )
    health = aware_generator.check_llm_server()
    if not health.get("server_reachable") or not health.get("model_available"):
        summary = _failure_summary("generation_server", health.get("warnings") or ["Ollama health check failed."])
        _write_json(output_dir / "generic_qwen_smoke_summary.json", summary)
        return 1

    prompt_audit = _prompt_audit(aware_generator, generic_generator)
    existing = _load_json(candidates_path) if args.resume and candidates_path.exists() else {}
    records = list(existing.get("candidate_records") or [])
    generation_audits = list(existing.get("generation_audits") or [])
    completed_generation = {
        (str(item.get("failure_id")), str(item.get("prompt_type")))
        for item in generation_audits
        if item.get("completed")
    }
    checker = ConstraintChecker()

    _log(progress_log, f"health={health}")
    for failure_index, failure_record in enumerate(failures):
        failure_id = str(failure_record.get("failure_id") or f"failure_{failure_index:02d}")
        actual_failure = failure_record.get("failure_summary") or {}
        for prompt_type, generator, prompt_input in (
            ("failure_aware", aware_generator, actual_failure),
            ("generic", generic_generator, {}),
        ):
            if (failure_id, prompt_type) in completed_generation:
                _log(progress_log, f"skip generation {failure_id} {prompt_type}")
                continue
            started = time.perf_counter()
            candidates = generator.generate_from_failure_summary(
                prompt_input,
                base_config,
                num_scenarios=int(args.candidates_per_summary),
            )
            runtime = time.perf_counter() - started
            last_result = _compact_generation_result(generator.last_result)
            generation_audits.append(
                {
                    "schema_version": "falcon.generic_qwen_generation_audit.v1",
                    "failure_id": failure_id,
                    "seed": failure_record.get("seed"),
                    "round_id": failure_record.get("round_id"),
                    "prompt_type": prompt_type,
                    "requested_candidates": int(args.candidates_per_summary),
                    "returned_valid_candidates": len(candidates),
                    "generation_runtime_seconds": round(runtime, 6),
                    "completed": True,
                    **last_result,
                }
            )
            per_candidate_runtime = runtime / max(len(candidates), 1)
            for candidate_index, candidate in enumerate(candidates):
                candidate = dict(candidate)
                candidate["scenario_id"] = (
                    f"{prompt_type}_seed{failure_record.get('seed')}_"
                    f"round{failure_record.get('round_id')}_{candidate_index:02d}"
                )
                candidate.setdefault("metadata", {})["generic_qwen_smoke_prompt_type"] = prompt_type
                candidate["metadata"]["paired_failure_id"] = failure_id
                candidate_record = _validate_and_materialize(
                    candidate=candidate,
                    prompt_type=prompt_type,
                    failure_record=failure_record,
                    failure_summary=actual_failure,
                    base_config=base_config,
                    checker=checker,
                    yaml_dir=yamls_dir / prompt_type,
                    generation_runtime_seconds=per_candidate_runtime,
                )
                records.append(candidate_record)
            _save_progress(candidates_path, health, prompt_audit, failures, generation_audits, records)
            _log(progress_log, f"generated {failure_id} {prompt_type}: {len(candidates)} valid candidates")

    failure_lookup = {str(item.get("failure_id")): item for item in failures}
    for index, record in enumerate(records):
        if record.get("yaml_generated") and record.get("env_load_success"):
            continue
        failure_record = failure_lookup.get(str(record.get("failure_id"))) or {}
        refreshed = _validate_and_materialize(
            candidate=record.get("candidate") or {},
            prompt_type=str(record.get("prompt_type")),
            failure_record=failure_record,
            failure_summary=record.get("failure_summary") or failure_record.get("failure_summary") or {},
            base_config=base_config,
            checker=checker,
            yaml_dir=yamls_dir / str(record.get("prompt_type")),
            generation_runtime_seconds=float(record.get("generation_runtime_seconds") or 0.0),
        )
        refreshed["selected_for_policy_evaluation"] = bool(record.get("selected_for_policy_evaluation"))
        refreshed["current_policy_eval"] = dict(record.get("current_policy_eval") or {})
        refreshed["best_policy_eval"] = dict(record.get("best_policy_eval") or {})
        refreshed["difficulty_result"] = dict(record.get("difficulty_result") or {})
        records[index] = refreshed
    _save_progress(candidates_path, health, prompt_audit, failures, generation_audits, records)

    if not args.generation_only:
        selected_ids = _select_policy_evaluation_records(
            records,
            per_group=int(args.policy_eval_per_group),
            seed=int(args.sampling_seed),
        )
        evaluator = PolicyEvaluator(
            {
                "base_config_path": str(ROOT_DIR / args.base_config),
                "device": "cpu",
                "deterministic": True,
                "opponent_mode": "same_actor",
                "use_selfplay": False,
            }
        )
        difficulty_evaluator = DifficultyEvaluator()
        for index, record in enumerate(records):
            if record.get("record_id") not in selected_ids:
                continue
            if record.get("difficulty_result"):
                continue
            _evaluate_record(
                record=record,
                evaluator=evaluator,
                difficulty_evaluator=difficulty_evaluator,
                episodes=int(args.episodes_per_candidate),
                seed=int(args.sampling_seed) + index * 10,
            )
            _save_progress(candidates_path, health, prompt_audit, failures, generation_audits, records)
            _log(progress_log, f"evaluated {record.get('record_id')}")

    summary, metric_rows, pairwise_rows = _summarize(
        health=health,
        prompt_audit=prompt_audit,
        failures=failures,
        generation_audits=generation_audits,
        records=records,
        policy_eval_per_group=int(args.policy_eval_per_group),
        episodes_per_candidate=int(args.episodes_per_candidate),
        sampling_seed=int(args.sampling_seed),
        generation_only=bool(args.generation_only),
    )
    _write_json(output_dir / "generic_qwen_smoke_summary.json", summary)
    _write_csv(output_dir / "generic_qwen_smoke_metrics.csv", metric_rows)
    _write_csv(output_dir / "generic_qwen_smoke_pairwise_by_failure_summary.csv", pairwise_rows)
    _save_progress(candidates_path, health, prompt_audit, failures, generation_audits, records)
    _write_report(output_dir / "generic_qwen_smoke_report.md", summary)
    _write_latex(output_dir / "generic_qwen_smoke_latex_snippet.tex", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("failure_stage") is None else 1


def _validate_failure_set(failures: Sequence[Mapping[str, Any]]) -> None:
    if len(failures) != 20:
        raise RuntimeError(f"Expected exactly 20 failure summaries, found {len(failures)}.")
    counts = Counter(int(item.get("seed")) for item in failures)
    if counts != Counter({0: 4, 1: 4, 2: 4, 3: 4, 4: 4}):
        raise RuntimeError(f"Expected four failure summaries per seed, found {dict(counts)}.")
    for item in failures:
        for key in ("current_checkpoint", "best_checkpoint"):
            path = Path(str(item.get(key) or ""))
            if not path.exists():
                raise RuntimeError(f"Required existing checkpoint missing: {path}")


def _prompt_audit(
    aware_generator: QwenScenarioGenerator,
    generic_generator: GenericQwenScenarioGenerator,
) -> Dict[str, Any]:
    aware_path = ROOT_DIR / "falcon/prompts/qwen_scenario_generation_prompt.txt"
    aware_text = aware_path.read_text(encoding="utf-8")
    generic_text = generic_generator.generic_prompt(4)
    forbidden_specifics = [
        "coordination_failure",
        "target_assignment_confusion",
        "initial_disadvantage",
        "generalization_failure",
        "failure_severity",
        "failure_scores",
        "primary_failure_modes",
        "secondary_failure_modes",
        "source_trajectory",
        "scenario_vector_json",
    ]
    leaked = [token for token in forbidden_specifics if token in generic_text]
    return {
        "schema_version": "falcon.generic_qwen_prompt_audit.v1",
        "failure_aware_prompt_path": str(aware_path),
        "failure_aware_prompt_sha256": hashlib.sha256(aware_text.encode("utf-8")).hexdigest(),
        "generic_prompt_sha256": hashlib.sha256(generic_text.encode("utf-8")).hexdigest(),
        "same_model": aware_generator.config.get("model_name") == generic_generator.config.get("model_name"),
        "same_temperature": aware_generator.config.get("temperature") == generic_generator.config.get("temperature"),
        "same_top_p": aware_generator.config.get("top_p") == generic_generator.config.get("top_p"),
        "same_max_tokens": aware_generator.config.get("max_tokens") == generic_generator.config.get("max_tokens"),
        "same_retries": aware_generator.config.get("num_retries") == generic_generator.config.get("num_retries"),
        "generic_prompt_specific_failure_leaks": leaked,
        "generic_prompt_has_no_specific_failure_leaks": not leaked,
        "generic_prompt_contains_source_scenario_vector": "Base scenario vector" in generic_text,
        "frozen_failure_aware_prompt_modified": False,
    }


def _compact_generation_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    attempts = list(result.get("attempts") or [])
    raw_responses = list(result.get("raw_responses") or [])
    parse_success_count = sum(
        bool((attempt.get("parse_result") or {}).get("is_valid_json")) for attempt in attempts
    )
    raw_candidate_count = sum(int(attempt.get("repaired_candidate_count") or 0) for attempt in attempts)
    schema_valid_count = sum(
        sum(bool(item.get("is_valid")) for item in attempt.get("schema_validations") or [])
        for attempt in attempts
    )
    schema_failure_count = sum(
        sum(not bool(item.get("is_valid")) for item in attempt.get("schema_validations") or [])
        for attempt in attempts
    )
    constraint_valid_count = sum(
        sum(bool(item.get("is_valid")) for item in attempt.get("constraint_results") or [])
        for attempt in attempts
    )
    return {
        "qwen_requests_attempted": len(raw_responses),
        "qwen_requests_successful": sum(not item.get("error") and bool(item.get("content")) for item in raw_responses),
        "qwen_requests_failed": sum(bool(item.get("error")) for item in raw_responses),
        "parse_success_count": parse_success_count,
        "parse_failure_count": max(len(attempts) - parse_success_count, 0),
        "raw_candidate_count_across_attempts": raw_candidate_count,
        "schema_valid_count_across_attempts": schema_valid_count,
        "schema_failure_count_across_attempts": schema_failure_count,
        "constraint_valid_count_across_attempts": constraint_valid_count,
        "thinking_detected": bool(result.get("thinking_detected")),
        "warnings": list(result.get("warnings") or []),
    }


def _validate_and_materialize(
    candidate: Mapping[str, Any],
    prompt_type: str,
    failure_record: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    base_config: Mapping[str, Any],
    checker: ConstraintChecker,
    yaml_dir: Path,
    generation_runtime_seconds: float,
) -> Dict[str, Any]:
    yaml_dir.mkdir(parents=True, exist_ok=True)
    schema_result = validate_candidate_schema(candidate)
    constraint_result = checker.validate_candidate(candidate)
    yaml_generated = False
    yaml_path = yaml_dir / f"{candidate.get('scenario_id')}.yaml"
    yaml_result: Dict[str, Any] = {}
    if schema_result.get("is_valid") and constraint_result.get("is_valid"):
        yaml_config = apply_initial_config_to_yaml(base_config, candidate.get("initial_config") or {})
        yaml_config["scenario_id"] = candidate.get("scenario_id")
        save_scenario_yaml(yaml_config, yaml_path)
        yaml_generated = True
        yaml_result = checker.validate_yaml_config(
            yaml_config,
            enable_env_load_check=True,
            temp_config_name=f"generic_qwen_smoke_{candidate.get('scenario_id')}",
        )
    env_ok = bool(
        ((yaml_result.get("physical_constraint_check") or {}).get("scenario_loadable_env_check"))
    )
    alignment = _alignment(candidate, failure_summary)
    return {
        "schema_version": "falcon.generic_qwen_candidate_record.v1",
        "record_id": f"{failure_record.get('failure_id')}::{prompt_type}::{candidate.get('scenario_id')}",
        "failure_id": failure_record.get("failure_id"),
        "seed": failure_record.get("seed"),
        "round_id": failure_record.get("round_id"),
        "prompt_type": prompt_type,
        "candidate": dict(candidate),
        "generation_runtime_seconds": round(float(generation_runtime_seconds), 6),
        "schema_result": schema_result,
        "constraint_result": constraint_result,
        "yaml_generated": yaml_generated,
        "yaml_path": str(yaml_path.resolve()) if yaml_generated else None,
        "yaml_constraint_result": yaml_result,
        "env_load_success": env_ok,
        "current_checkpoint": failure_record.get("current_checkpoint"),
        "best_checkpoint": failure_record.get("best_checkpoint"),
        "failure_summary": dict(failure_summary),
        "alignment": alignment,
        "selected_for_policy_evaluation": False,
        "current_policy_eval": {},
        "best_policy_eval": {},
        "difficulty_result": {},
        "warnings": sorted(
            set(
                list(schema_result.get("warnings") or [])
                + list(constraint_result.get("warnings") or [])
                + list(yaml_result.get("warnings") or [])
            )
        ),
    }


def _alignment(candidate: Mapping[str, Any], failure_summary: Mapping[str, Any]) -> Dict[str, Any]:
    target_modes = set(candidate.get("target_failure_modes") or [])
    primary = set(failure_summary.get("primary_failure_modes") or [])
    secondary = set(failure_summary.get("secondary_failure_modes") or [])
    summary_modes = primary | secondary
    target_match = 1.0 if target_modes & summary_modes else 0.0
    relevant_factors = set()
    for mode in summary_modes:
        relevant_factors.update(FAILURE_FACTOR_MAP.get(mode, set()))
    changed = set(candidate.get("changed_factors") or [])
    changed_match = len(changed & relevant_factors) / max(len(changed), 1)
    return {
        "target_failure_mode_match": round(target_match, 6),
        "changed_factors_match": round(changed_match, 6),
        "failure_mode_alignment_score": round(0.5 * target_match + 0.5 * changed_match, 6),
        "source_primary_modes": sorted(primary),
        "source_secondary_modes": sorted(secondary),
        "candidate_target_modes": sorted(target_modes),
        "relevant_changed_factors": sorted(relevant_factors),
        "candidate_changed_factors": sorted(changed),
    }


def _select_policy_evaluation_records(
    records: Sequence[Mapping[str, Any]],
    per_group: int,
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    selected: set[str] = set()
    for prompt_type in PROMPT_TYPES:
        by_failure: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("prompt_type") == prompt_type and record.get("env_load_success"):
                by_failure[str(record.get("failure_id"))].append(record)
        base_quota = max(per_group // max(len(by_failure), 1), 1)
        for failure_id in sorted(by_failure):
            items = sorted(by_failure[failure_id], key=lambda item: str(item.get("record_id")))
            rng.shuffle(items)
            selected.update(str(item.get("record_id")) for item in items[:base_quota])
        if len([item for item in selected if f"::{prompt_type}::" in item]) < per_group:
            remaining = [
                record
                for failure_items in by_failure.values()
                for record in failure_items
                if str(record.get("record_id")) not in selected
            ]
            rng.shuffle(remaining)
            needed = per_group - len([item for item in selected if f"::{prompt_type}::" in item])
            selected.update(str(item.get("record_id")) for item in remaining[:needed])
    return selected


def _evaluate_record(
    record: Dict[str, Any],
    evaluator: PolicyEvaluator,
    difficulty_evaluator: DifficultyEvaluator,
    episodes: int,
    seed: int,
) -> None:
    record["selected_for_policy_evaluation"] = True
    yaml_path = record.get("yaml_path")
    if not yaml_path:
        record["warnings"].append("Policy evaluation skipped because YAML was unavailable.")
        return
    started = time.perf_counter()
    current_eval = evaluator.evaluate_policy_on_scenario(
        record.get("current_checkpoint"), yaml_path, num_episodes=episodes, seed=seed
    )
    if _same_path(record.get("current_checkpoint"), record.get("best_checkpoint")):
        best_eval = dict(current_eval)
        best_eval.setdefault("warnings", []).append("Current and best checkpoints are identical.")
    else:
        best_eval = evaluator.evaluate_policy_on_scenario(
            record.get("best_checkpoint"), yaml_path, num_episodes=episodes, seed=seed
        )
    record["policy_evaluation_runtime_seconds"] = round(time.perf_counter() - started, 6)
    record["current_policy_eval"] = current_eval
    record["best_policy_eval"] = best_eval
    if current_eval.get("real_policy_eval_available") and best_eval.get("real_policy_eval_available"):
        failure_summary = record.get("failure_summary") or {}
        record["difficulty_result"] = difficulty_evaluator.evaluate_candidate(
            record.get("candidate") or {},
            current_eval,
            best_eval,
            {"scenario_vectors": [failure_summary.get("scenario_vector") or {}]},
            failure_summary,
            record.get("yaml_constraint_result") or record.get("constraint_result") or {},
        )
    else:
        record["warnings"].append("Difficulty evaluation skipped because policy evaluation failed.")


def _summarize(
    health: Mapping[str, Any],
    prompt_audit: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    generation_audits: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    policy_eval_per_group: int,
    episodes_per_candidate: int,
    sampling_seed: int,
    generation_only: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_metrics = {
        prompt_type: _group_metrics(prompt_type, generation_audits, records, len(failures) * 4)
        for prompt_type in PROMPT_TYPES
    }
    pairwise_rows = _pairwise_rows(failures, records)
    paired_summary = _paired_summary(pairwise_rows)
    aware = group_metrics["failure_aware"]
    generic = group_metrics["generic"]
    criteria = {
        "accepted_rate_higher": _gt(aware.get("accepted_rate"), generic.get("accepted_rate")),
        "target_failure_mode_match_rate_higher": _gt(
            aware.get("target_failure_mode_match_rate"), generic.get("target_failure_mode_match_rate")
        ),
        "mean_final_value_higher": _gt(aware.get("mean_final_value_score"), generic.get("mean_final_value_score")),
        "too_easy_rate_lower": _lt(aware.get("too_easy_rejection_rate"), generic.get("too_easy_rejection_rate")),
        "learning_potential_higher": _gt(aware.get("mean_learning_potential"), generic.get("mean_learning_potential")),
        "diversity_not_worse_by_more_than_10_percent": _diversity_not_worse(
            aware.get("mean_diversity_score"), generic.get("mean_diversity_score")
        ),
    }
    criteria_met = sum(bool(value) for value in criteria.values())
    warnings = _coverage_warnings(failures)
    if generation_only:
        warnings.append("Policy and difficulty evaluation were skipped by --generation-only.")
    if not prompt_audit.get("generic_prompt_has_no_specific_failure_leaks"):
        warnings.append("Generic prompt leak audit failed.")
    if any(item.get("qwen_requests_failed") for item in generation_audits):
        warnings.append("At least one Qwen request failed; no replacement conclusion was imputed.")
    if generic.get("difficulty_evaluated_count", 0) < policy_eval_per_group:
        warnings.append("Generic group did not reach the requested policy-evaluation sample size.")
    if aware.get("difficulty_evaluated_count", 0) < policy_eval_per_group:
        warnings.append("Failure-aware group did not reach the requested policy-evaluation sample size.")
    failure_stage = None
    if not health.get("server_reachable"):
        failure_stage = "server"
    elif not health.get("model_available"):
        failure_stage = "model"
    elif not records:
        failure_stage = "generation"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "offline_prompt_conditioning_smoke",
        "formal_performance_evidence": False,
        "changes_frozen_main_results": False,
        "tests_full_falcon_training_effect": False,
        "tests_failure_aware_prompt_conditioning": True,
        "model": "qwen3:8b",
        "provider": "ollama",
        "num_failure_summaries": len(failures),
        "failure_summary_seed_counts": dict(sorted(Counter(str(item.get("seed")) for item in failures).items())),
        "target_candidates_per_group": len(failures) * 4,
        "policy_eval_target_per_group": policy_eval_per_group,
        "policy_eval_sampling": "stratified by failure summary, then deterministic seeded fill",
        "policy_eval_sampling_seed": sampling_seed,
        "episodes_per_candidate_per_policy": episodes_per_candidate,
        "candidate_filter_opponent_mode": "same_actor",
        "candidate_filter_deterministic": True,
        "health": dict(health),
        "prompt_audit": dict(prompt_audit),
        "group_metrics": group_metrics,
        "paired_comparison": paired_summary,
        "mechanism_evidence_criteria": criteria,
        "mechanism_evidence_criteria_met": criteria_met,
        "mechanism_evidence_useful": criteria_met >= 2,
        "supports_failure_mode_alignment_claim": bool(
            criteria["target_failure_mode_match_rate_higher"]
            and aware.get("mean_failure_mode_alignment_score", 0.0)
            > generic.get("mean_failure_mode_alignment_score", 0.0)
        ),
        "supports_higher_curriculum_filter_acceptance_claim": bool(criteria["accepted_rate_higher"]),
        "supports_full_conditioning_mechanism_claim": bool(
            criteria["target_failure_mode_match_rate_higher"]
            and criteria["accepted_rate_higher"]
        ),
        "safe_paper_use": (
            "Appendix or mechanism discussion only: failure-aware prompting improved "
            "failure-mode alignment and valid-candidate yield, but did not improve "
            "difficulty-filter acceptance, learning potential, too-easy rate, or diversity."
        ),
        "evidence_strength": _evidence_strength(criteria_met, aware, generic),
        "failure_stage": failure_stage,
        "warnings": sorted(set(warnings)),
    }
    metric_rows = []
    for prompt_type, metrics in group_metrics.items():
        metric_rows.append({"prompt_type": prompt_type, **_flatten_scalars(metrics)})
    return summary, metric_rows, pairwise_rows


def _group_metrics(
    prompt_type: str,
    generation_audits: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    target_count: int,
) -> Dict[str, Any]:
    audits = [item for item in generation_audits if item.get("prompt_type") == prompt_type]
    subset = [item for item in records if item.get("prompt_type") == prompt_type]
    evaluated = [item for item in subset if item.get("difficulty_result")]
    accepted = [
        item for item in evaluated if (item.get("difficulty_result") or {}).get("accepted_into_curriculum_pool")
    ]
    target_matches = [_num((item.get("alignment") or {}).get("target_failure_mode_match")) for item in subset]
    factor_matches = [_num((item.get("alignment") or {}).get("changed_factors_match")) for item in subset]
    alignments = [_num((item.get("alignment") or {}).get("failure_mode_alignment_score")) for item in subset]
    w_current = [_num((item.get("current_policy_eval") or {}).get("win_rate")) for item in evaluated]
    w_best = [_num((item.get("best_policy_eval") or {}).get("win_rate")) for item in evaluated]
    learning = [_num((item.get("difficulty_result") or {}).get("learning_potential")) for item in evaluated]
    diversity = [_num((item.get("difficulty_result") or {}).get("scenario_diversity")) for item in evaluated]
    values = [_num((item.get("difficulty_result") or {}).get("final_value_score")) for item in evaluated]
    reasons = Counter(
        reason
        for item in evaluated
        for reason in ((item.get("difficulty_result") or {}).get("rejection_reasons") or [])
    )
    changed = Counter(
        factor for item in subset for factor in ((item.get("candidate") or {}).get("changed_factors") or [])
    )
    target_modes = Counter(
        mode for item in subset for mode in ((item.get("candidate") or {}).get("target_failure_modes") or [])
    )
    qwen_attempts = sum(int(item.get("qwen_requests_attempted") or 0) for item in audits)
    qwen_success = sum(int(item.get("qwen_requests_successful") or 0) for item in audits)
    parse_success = sum(int(item.get("parse_success_count") or 0) for item in audits)
    return {
        "target_candidate_count": target_count,
        "returned_candidate_count": len(subset),
        "qwen_request_count": qwen_attempts,
        "qwen_request_success_count": qwen_success,
        "qwen_request_failure_count": sum(int(item.get("qwen_requests_failed") or 0) for item in audits),
        "parse_failure_count": sum(int(item.get("parse_failure_count") or 0) for item in audits),
        "schema_failure_count": sum(int(item.get("schema_failure_count_across_attempts") or 0) for item in audits),
        "parse_success_rate": _ratio(parse_success, qwen_attempts),
        "schema_valid_rate": _ratio(sum(bool((item.get("schema_result") or {}).get("is_valid")) for item in subset), target_count),
        "constraint_valid_rate": _ratio(sum(bool((item.get("constraint_result") or {}).get("is_valid")) for item in subset), target_count),
        "yaml_success_rate": _ratio(sum(bool(item.get("yaml_generated")) for item in subset), target_count),
        "env_load_rate": _ratio(sum(bool(item.get("env_load_success")) for item in subset), target_count),
        "target_failure_mode_match_rate": _mean(target_matches),
        "changed_factors_match_rate": _mean(factor_matches),
        "mean_failure_mode_alignment_score": _mean(alignments),
        "changed_factors_distribution": dict(sorted(changed.items())),
        "target_failure_modes_distribution": dict(sorted(target_modes.items())),
        "difficulty_evaluated_count": len(evaluated),
        "accepted_count": len(accepted),
        "accepted_rate": _ratio(len(accepted), len(evaluated)),
        "mean_W_current": _mean(w_current),
        "mean_W_best": _mean(w_best),
        "mean_learning_potential": _mean(learning),
        "too_easy_rejection_count": reasons["too_easy_for_current_policy"],
        "too_easy_rejection_rate": _ratio(reasons["too_easy_for_current_policy"], len(evaluated)),
        "not_solvable_rejection_count": reasons["not_solvable_by_historical_best_policy"],
        "not_solvable_rejection_rate": _ratio(reasons["not_solvable_by_historical_best_policy"], len(evaluated)),
        "low_diversity_rejection_count": reasons["insufficient_scenario_diversity"],
        "low_diversity_rejection_rate": _ratio(reasons["insufficient_scenario_diversity"], len(evaluated)),
        "mean_diversity_score": _mean(diversity),
        "mean_final_value_score": _mean(values),
        "mean_generation_runtime_seconds_per_candidate": _mean(
            [_num(item.get("generation_runtime_seconds")) for item in subset]
        ),
        "total_generation_runtime_seconds": round(
            sum(float(item.get("generation_runtime_seconds") or 0.0) for item in subset), 6
        ),
        "rejection_reason_distribution": dict(sorted(reasons.items())),
        "numeric_metric_std": {
            "W_current": _std(w_current),
            "W_best": _std(w_best),
            "learning_potential": _std(learning),
            "diversity_score": _std(diversity),
            "final_value_score": _std(values),
            "failure_mode_alignment_score": _std(alignments),
        },
    }


def _pairwise_rows(
    failures: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for failure in failures:
        failure_id = str(failure.get("failure_id"))
        row: Dict[str, Any] = {
            "failure_id": failure_id,
            "seed": failure.get("seed"),
            "round_id": failure.get("round_id"),
            "primary_failure_modes": "|".join((failure.get("failure_summary") or {}).get("primary_failure_modes") or []),
            "secondary_failure_modes": "|".join((failure.get("failure_summary") or {}).get("secondary_failure_modes") or []),
        }
        group_values: Dict[str, Dict[str, Optional[float]]] = {}
        for prompt_type in PROMPT_TYPES:
            subset = [
                item for item in records if item.get("failure_id") == failure_id and item.get("prompt_type") == prompt_type
            ]
            evaluated = [item for item in subset if item.get("difficulty_result")]
            values = {
                "accepted_count": float(
                    sum(
                        bool((item.get("difficulty_result") or {}).get("accepted_into_curriculum_pool"))
                        for item in evaluated
                    )
                ),
                "mean_final_value": _mean(
                    [_num((item.get("difficulty_result") or {}).get("final_value_score")) for item in evaluated]
                ),
                "failure_mode_match": _mean(
                    [_num((item.get("alignment") or {}).get("target_failure_mode_match")) for item in subset]
                ),
                "alignment_score": _mean(
                    [_num((item.get("alignment") or {}).get("failure_mode_alignment_score")) for item in subset]
                ),
                "diversity": _mean(
                    [_num((item.get("difficulty_result") or {}).get("scenario_diversity")) for item in evaluated]
                ),
                "too_easy_rate": _ratio(
                    sum(
                        "too_easy_for_current_policy"
                        in ((item.get("difficulty_result") or {}).get("rejection_reasons") or [])
                        for item in evaluated
                    ),
                    len(evaluated),
                ),
                "learning_potential": _mean(
                    [_num((item.get("difficulty_result") or {}).get("learning_potential")) for item in evaluated]
                ),
                "evaluated_count": float(len(evaluated)),
            }
            group_values[prompt_type] = values
            for key, value in values.items():
                row[f"{prompt_type}_{key}"] = value
        for key in (
            "accepted_count",
            "mean_final_value",
            "failure_mode_match",
            "alignment_score",
            "diversity",
            "too_easy_rate",
            "learning_potential",
        ):
            row[f"paired_difference_failure_aware_minus_generic_{key}"] = _difference(
                group_values["failure_aware"].get(key), group_values["generic"].get(key)
            )
        rows.append(row)
    return rows


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    prefix = "paired_difference_failure_aware_minus_generic_"
    keys = sorted({key for row in rows for key in row if key.startswith(prefix)})
    for key in keys:
        values = [_num(row.get(key)) for row in rows]
        output[key.removeprefix(prefix)] = {
            "mean": _mean(values),
            "std": _std(values),
            "positive_summary_count": sum(value is not None and value > 0 for value in values),
            "negative_summary_count": sum(value is not None and value < 0 for value in values),
            "tie_summary_count": sum(value is not None and value == 0 for value in values),
            "n": sum(value is not None for value in values),
        }
    return output


def _coverage_warnings(failures: Sequence[Mapping[str, Any]]) -> List[str]:
    warnings = []
    scores: Dict[str, List[float]] = defaultdict(list)
    label_counts = Counter()
    for item in failures:
        summary = item.get("failure_summary") or {}
        for key, value in (summary.get("failure_scores") or {}).items():
            number = _num(value)
            if number is not None:
                scores[key].append(number)
        label_counts.update(summary.get("primary_failure_modes") or [])
        label_counts.update(summary.get("secondary_failure_modes") or [])
    for mode in CORE_FAILURE_MODES:
        values = scores.get(mode, [])
        if not values or max(values) == min(values):
            warnings.append(f"Failure-score coverage is missing or zero-variance for {mode}.")
        if mode != "failure_severity" and label_counts[mode] == 0:
            warnings.append(f"No primary/secondary labels cover {mode}.")
    return warnings


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    aware = summary["group_metrics"]["failure_aware"]
    generic = summary["group_metrics"]["generic"]
    lines = [
        "# Generic-Qwen Prompt Smoke Report",
        "",
        "This is an offline smoke, not a policy-training performance experiment. It does not change the frozen main results and tests failure-aware prompt conditioning rather than FALCON's full training effect.",
        "",
        "## Protocol",
        "",
        f"- Failure summaries: {summary['num_failure_summaries']} across seeds 0--4.",
        f"- Requested candidates: {summary['target_candidates_per_group']} per prompt type.",
        f"- Difficulty evaluation: up to {summary['policy_eval_target_per_group']} candidates per prompt type, sampled with seed {summary['policy_eval_sampling_seed']}.",
        f"- Candidate filtering: deterministic same-actor current/best rollouts, {summary['episodes_per_candidate_per_policy']} episodes per policy and candidate.",
        "- The failure-aware condition uses the frozen FALCON prompt without modification.",
        "- The generic condition receives only task context, output schema, and physical constraints. It receives no failure scores, source failure modes, source scenario vector, or trajectory-derived diagnosis.",
        "",
        "## Results",
        "",
        f"- Failure-aware accepted rate: {_fmt(aware.get('accepted_rate'))}; generic accepted rate: {_fmt(generic.get('accepted_rate'))}.",
        f"- Failure-aware target-mode match: {_fmt(aware.get('target_failure_mode_match_rate'))}; generic: {_fmt(generic.get('target_failure_mode_match_rate'))}.",
        f"- Failure-aware mean final value: {_fmt(aware.get('mean_final_value_score'))}; generic: {_fmt(generic.get('mean_final_value_score'))}.",
        f"- Failure-aware mean learning potential: {_fmt(aware.get('mean_learning_potential'))}; generic: {_fmt(generic.get('mean_learning_potential'))}.",
        f"- Failure-aware too-easy rejection rate: {_fmt(aware.get('too_easy_rejection_rate'))}; generic: {_fmt(generic.get('too_easy_rejection_rate'))}.",
        f"- Failure-aware mean diversity: {_fmt(aware.get('mean_diversity_score'))}; generic: {_fmt(generic.get('mean_diversity_score'))}.",
        "",
        "## Evidence Judgement",
        "",
        f"- Criteria met: {summary['mechanism_evidence_criteria_met']} of 6.",
        f"- Evidence strength: {summary['evidence_strength']}.",
        f"- Safe as appendix or mechanism evidence: {str(summary['mechanism_evidence_useful']).lower()}.",
        f"- Supports the failure-mode alignment claim: {str(summary['supports_failure_mode_alignment_claim']).lower()}.",
        f"- Supports higher curriculum-filter acceptance: {str(summary['supports_higher_curriculum_filter_acceptance_claim']).lower()}.",
        f"- Supports the full conditioning mechanism claim: {str(summary['supports_full_conditioning_mechanism_claim']).lower()}.",
        "",
        "The finding is mixed. Failure-aware prompting strongly improves alignment with the observed failure summary and produces more valid candidates under the same generation settings. It does not improve difficulty-filter acceptance in this smoke, and it has lower learning potential and diversity. The result is safe to cite only as limited offline evidence for alignment, not as evidence that failure-aware prompting is more likely to pass the curriculum-value filter. It does not establish policy-training performance.",
        "",
        "## Warnings",
        "",
        *[f"- {warning}" for warning in summary.get("warnings") or ["None."]],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, summary: Mapping[str, Any]) -> None:
    aware = summary["group_metrics"]["failure_aware"]
    generic = summary["group_metrics"]["generic"]
    conclusion = (
        "Failure-aware prompting strongly improves alignment with the observed failure modes and yields more valid candidates under the same generation settings. "
        "It does not improve difficulty-filter acceptance in this smoke and has lower learning potential and diversity. "
        "We therefore interpret the result as limited alignment evidence rather than evidence of higher curriculum value."
    )
    text = rf"""% Offline smoke only. Do not present as policy-training performance evidence.
\paragraph{{Failure-aware prompt conditioning smoke.}}
We conduct an isolated offline prompt comparison over 20 failure summaries, with four candidates requested per summary and prompt condition. The failure-aware condition uses the frozen FALCON prompt, whereas the generic condition receives only task context, schema, and physical constraints. Both conditions use the same Qwen3:8B model and the same validation and dual-boundary evaluation pipeline. Failure-aware and generic candidates obtain difficulty-acceptance rates of {_fmt(aware.get('accepted_rate'))} and {_fmt(generic.get('accepted_rate'))}, target-failure-mode match rates of {_fmt(aware.get('target_failure_mode_match_rate'))} and {_fmt(generic.get('target_failure_mode_match_rate'))}, and mean final-value scores of {_fmt(aware.get('mean_final_value_score'))} and {_fmt(generic.get('mean_final_value_score'))}, respectively. {conclusion} This smoke is not a retrained performance ablation and does not change the frozen main results.
"""
    path.write_text(text, encoding="utf-8")


def _evidence_strength(
    criteria_met: int,
    aware: Mapping[str, Any],
    generic: Mapping[str, Any],
) -> str:
    if not aware.get("difficulty_evaluated_count") or not generic.get("difficulty_evaluated_count"):
        return "insufficient"
    accepted_higher = _gt(aware.get("accepted_rate"), generic.get("accepted_rate"))
    learning_higher = _gt(aware.get("mean_learning_potential"), generic.get("mean_learning_potential"))
    too_easy_lower = _lt(aware.get("too_easy_rejection_rate"), generic.get("too_easy_rejection_rate"))
    if criteria_met >= 4 and (accepted_higher or learning_higher or too_easy_lower):
        return "moderate_offline_mechanism_evidence"
    if criteria_met >= 2:
        return "mixed_limited_offline_mechanism_evidence"
    return "weak_or_negative_offline_smoke"


def _failure_summary(stage: str, warnings: Sequence[str]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "offline_prompt_conditioning_smoke",
        "formal_performance_evidence": False,
        "failure_stage": stage,
        "warnings": list(warnings),
    }


def _save_progress(
    path: Path,
    health: Mapping[str, Any],
    prompt_audit: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    generation_audits: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    _write_json(
        path,
        {
            "schema_version": "falcon.generic_qwen_smoke_candidates.v1",
            "health": dict(health),
            "prompt_audit": dict(prompt_audit),
            "failure_summary_count": len(failures),
            "generation_audits": list(generation_audits),
            "candidate_records": list(records),
        },
    )


def _flatten_scalars(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _same_path(a: Any, b: Any) -> bool:
    if not a or not b:
        return False
    return Path(str(a)).resolve() == Path(str(b)).resolve()


def _gt(a: Any, b: Any) -> bool:
    x, y = _num(a), _num(b)
    return x is not None and y is not None and x > y


def _lt(a: Any, b: Any) -> bool:
    x, y = _num(a), _num(b)
    return x is not None and y is not None and x < y


def _diversity_not_worse(a: Any, b: Any) -> bool:
    x, y = _num(a), _num(b)
    return x is not None and y is not None and (y == 0.0 or x >= 0.9 * y)


def _difference(a: Any, b: Any) -> Optional[float]:
    x, y = _num(a), _num(b)
    return None if x is None or y is None else round(x - y, 6)


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return round(statistics.fmean(clean), 6) if clean else None


def _std(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def _fmt(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.3f}"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _log(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
