"""Optional mixed Qwen/FSN generator for controlled replacement smokes."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .candidate_schema import validate_candidate_schema
from .constraint_checker import ConstraintChecker
from .fsn_generator import FSNScenarioGenerator
from .llm_scenario_generator import QwenScenarioGenerator
from .random_scenario_generator import RandomScenarioGenerator


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "fsn_model_path": "experiments/falcon_2v2_noweapon/fsn/stage2/fsn_stage2_model.pt",
    "replacement_ratio": 0.25,
    "use_diversity_aware_fsn": True,
    "qwen_ratio": 0.75,
    "fsn_candidates_per_round": 1,
    "qwen_candidates_per_round": 3,
    "target_fsn_ratio": 0.25,
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
    "use_fsn_rerank": False,
    "use_hardness_v2": False,
    "use_fsn_repair": False,
    "adapter_aware_projection": False,
    "post_yaml_constraint_check": False,
    "fsn_surrogate_model_path": "experiments/falcon_2v2_noweapon/fsn/stage6_hardness_v2/fsn_hardness_surrogate_model.pt",
    "fsn_overgenerate_n": 16,
    "fsn_select_top_k": 1,
    "fsn_repair_max_attempts": 3,
    "fsn_repair_margin_m": 5.0,
    "rerank_score": {
        "predicted_value_weight": 0.0,
        "accepted_probability_weight": 0.0,
        "diversity_weight": 0.6,
        "constraint_risk_weight": 0.4,
    },
    "hardness_v2_score": {
        "predicted_learning_potential": 0.30,
        "predicted_accepted_probability": 0.25,
        "diversity_bonus": 0.20,
        "pool_novelty": 0.10,
        "predicted_too_easy_probability": 0.10,
        "predicted_not_solvable_probability": 0.10,
        "constraint_risk": 0.05,
    },
    "fsn": {},
    "qwen": {},
}


class FSNReplacementGenerator:
    """Generate a controlled mixture while keeping Qwen as the fallback path."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))
        self.last_result: Dict[str, Any] = {
            "schema_version": "falcon.fsn_replacement_generation.v1",
            "candidates": [],
            "warnings": [],
        }

    def generate_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        pool_stats: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        started = time.perf_counter()
        warnings: List[str] = []
        fallback_reasons: List[str] = []
        enforce_quota = bool(self.config.get("enforce_quota", True))
        total_requested = max(
            int(self.config.get("total_candidates_per_round", 4)),
            0,
        )
        fsn_quota = max(
            int(
                self.config.get(
                    "fsn_quota",
                    round(
                        total_requested
                        * float(self.config.get("target_fsn_ratio", 0.25))
                    ),
                )
            ),
            0,
        )
        qwen_quota = max(
            int(self.config.get("qwen_quota", total_requested - fsn_quota)),
            0,
        )
        fsn_requested = (
            fsn_quota
            if enforce_quota
            else max(int(self.config.get("fsn_candidates_per_round", fsn_quota)), 0)
        )
        qwen_requested = (
            qwen_quota
            if enforce_quota
            else max(int(self.config.get("qwen_candidates_per_round", qwen_quota)), 0)
        )
        checker = ConstraintChecker()

        fsn_start = time.perf_counter()
        fsn_raw: List[Dict[str, Any]] = []
        fsn_candidates: List[Dict[str, Any]] = []
        fsn_validation: List[Dict[str, Any]] = []
        fsn_rerank_result: Dict[str, Any] = {}
        fsn_hardness_v2_result: Dict[str, Any] = {}
        try:
            model_path = _resolve_path(self.config.get("fsn_model_path"))
            fsn_config = dict(self.config.get("fsn") or {})
            fsn_config["diversity_aware"] = bool(
                self.config.get("use_diversity_aware_fsn", True)
            )
            fsn_generator = FSNScenarioGenerator(model_path, fsn_config)
            if self.config.get("use_hardness_v2", False):
                select_top_k = max(
                    int(self.config.get("fsn_select_top_k", fsn_requested)),
                    fsn_requested,
                )
                fsn_raw = fsn_generator.generate_hardness_v2_from_failure_summary(
                    failure_summary,
                    base_config,
                    surrogate_path=_resolve_path(
                        self.config.get("fsn_surrogate_model_path")
                    ),
                    num_scenarios=select_top_k,
                    overgenerate_count=max(
                        int(self.config.get("fsn_overgenerate_n", 64)),
                        select_top_k,
                    ),
                    pool_stats=pool_stats,
                    max_repair_attempts=int(
                        self.config.get("fsn_repair_max_attempts", 3)
                    ),
                    repair_margin_m=float(
                        self.config.get("fsn_repair_margin_m", 5.0)
                    ),
                    score_weights=dict(
                        self.config.get("hardness_v2_score") or {}
                    ),
                )
                fsn_hardness_v2_result = copy.deepcopy(
                    fsn_generator.last_hardness_v2_result
                )
            elif self.config.get("use_fsn_rerank", False):
                score_config = dict(self.config.get("rerank_score") or {})
                select_top_k = max(
                    int(self.config.get("fsn_select_top_k", fsn_requested)),
                    fsn_requested,
                )
                fsn_raw = fsn_generator.generate_reranked_from_failure_summary(
                    failure_summary,
                    base_config,
                    num_scenarios=select_top_k,
                    overgenerate_count=max(
                        int(self.config.get("fsn_overgenerate_n", 16)),
                        select_top_k,
                    ),
                    rerank_weights={
                        "predicted_value_score": float(
                            score_config.get("predicted_value_weight", 0.0)
                        ),
                        "accepted_probability": float(
                            score_config.get(
                                "accepted_probability_weight", 0.0
                            )
                        ),
                        "diversity_bonus": float(
                            score_config.get("diversity_weight", 0.6)
                        ),
                        "constraint_risk_penalty": float(
                            score_config.get("constraint_risk_weight", 0.4)
                        ),
                    },
                )
                fsn_rerank_result = copy.deepcopy(
                    fsn_generator.last_rerank_result
                )
            else:
                fsn_raw = fsn_generator.generate_from_failure_summary(
                    failure_summary,
                    base_config,
                    num_scenarios=fsn_requested,
                )
            for index, candidate in enumerate(fsn_raw):
                normalized = _normalize_source(candidate, "fsn", index)
                schema = validate_candidate_schema(normalized)
                constraint = checker.validate_candidate(normalized)
                valid = bool(schema.get("is_valid") and constraint.get("is_valid"))
                fsn_validation.append(
                    {
                        "scenario_id": normalized.get("scenario_id"),
                        "schema_validation": schema,
                        "constraint_result": constraint,
                        "is_valid": valid,
                    }
                )
                if valid:
                    fsn_candidates.append(normalized)
                else:
                    fallback_reasons.append(
                        f"FSN candidate {normalized.get('scenario_id')} failed pre-validation."
                    )
        except Exception as exc:  # noqa: BLE001 - replacement must fail back to Qwen
            fallback_reasons.append(f"FSN generation failed: {type(exc).__name__}: {exc}")
        fsn_runtime = round(time.perf_counter() - fsn_start, 6)
        if enforce_quota:
            fsn_candidates = fsn_candidates[:fsn_quota]

        missing_fsn = max(fsn_requested - len(fsn_candidates), 0)
        fsn_fallback_to_qwen_count = (
            missing_fsn
            if self.config.get("fallback_to_qwen_if_fsn_invalid", True)
            else 0
        )
        qwen_target = qwen_requested + fsn_fallback_to_qwen_count
        qwen_start = time.perf_counter()
        qwen_config = dict(self.config.get("qwen") or {})
        if self.config.get("qwen_retry_to_fill_quota", True):
            qwen_config["num_retries"] = 0
        qwen_generator = QwenScenarioGenerator(qwen_config)
        health = qwen_generator.check_llm_server()
        qwen_candidates: List[Dict[str, Any]] = []
        qwen_call_records: List[Dict[str, Any]] = []
        qwen_seen = set()
        qwen_candidates_requested = 0
        qwen_candidates_raw_returned = 0
        qwen_candidates_valid_returned = 0
        qwen_api_calls_attempted = 0
        qwen_api_calls_successful = 0
        qwen_api_calls_failed = 0
        max_qwen_invocations = 1
        if self.config.get("qwen_retry_to_fill_quota", True):
            max_qwen_invocations += max(
                int(self.config.get("qwen_max_retries_per_round", 2)), 0
            )
        if qwen_target > 0 and health.get("server_reachable") and health.get("model_available"):
            for invocation in range(max_qwen_invocations):
                shortfall = max(qwen_target - len(qwen_candidates), 0)
                if shortfall <= 0:
                    break
                qwen_candidates_requested += shortfall
                returned = qwen_generator.generate_from_failure_summary(
                    failure_summary,
                    base_config,
                    num_scenarios=shortfall,
                    pool_stats=pool_stats,
                )
                result = copy.deepcopy(qwen_generator.last_result)
                raw_responses = list(result.get("raw_responses") or [])
                attempts = list(result.get("attempts") or [])
                attempted = len(raw_responses)
                successful = sum(
                    1
                    for item in raw_responses
                    if not item.get("error")
                    and bool(item.get("content") or item.get("raw_response"))
                )
                failed = max(attempted - successful, 0)
                raw_returned = sum(
                    int(item.get("repaired_candidate_count") or 0)
                    for item in attempts
                )
                qwen_api_calls_attempted += attempted
                qwen_api_calls_successful += successful
                qwen_api_calls_failed += failed
                qwen_candidates_raw_returned += raw_returned
                qwen_candidates_valid_returned += len(returned)
                added = 0
                for candidate in returned:
                    normalized = _normalize_source(
                        candidate, "qwen", len(qwen_candidates)
                    )
                    identity = _candidate_identity(normalized)
                    if identity in qwen_seen:
                        continue
                    qwen_seen.add(identity)
                    qwen_candidates.append(normalized)
                    added += 1
                    if len(qwen_candidates) >= qwen_target:
                        break
                qwen_call_records.append(
                    {
                        "invocation": invocation,
                        "requested_candidates": shortfall,
                        "raw_candidates_returned": raw_returned,
                        "valid_candidates_returned": len(returned),
                        "unique_candidates_added": added,
                        "api_calls_attempted": attempted,
                        "api_calls_successful": successful,
                        "api_calls_failed": failed,
                        "runtime_recorded_by_generator": None,
                        "warnings": list(result.get("warnings") or []),
                    }
                )
                warnings.extend(result.get("warnings") or [])
        elif qwen_target > 0:
            warnings.extend(health.get("warnings") or [])
            warnings.append("Qwen generation was unavailable during FSN replacement generation.")
        qwen_runtime = round(time.perf_counter() - qwen_start, 6)
        if enforce_quota:
            qwen_candidates = qwen_candidates[:qwen_target]

        qwen_shortfall = max(qwen_quota - len(qwen_candidates), 0)
        total_shortfall = max(
            total_requested - len(fsn_candidates) - len(qwen_candidates), 0
        )
        random_candidates: List[Dict[str, Any]] = []
        random_validation: List[Dict[str, Any]] = []
        if (
            total_shortfall > 0
            and str(self.config.get("fallback_when_qwen_shortfall", "random")).lower()
            == "random"
        ):
            random_generator = RandomScenarioGenerator(
                {"seed": int(time.time_ns() % 2_147_483_647)}
            )
            random_raw = random_generator.generate_from_failure_summary(
                failure_summary,
                base_config,
                num_scenarios=max(total_shortfall * 2, total_shortfall),
            )
            for candidate in random_raw:
                normalized = _normalize_source(
                    candidate, "random", len(random_candidates)
                )
                schema = validate_candidate_schema(normalized)
                constraint = checker.validate_candidate(normalized)
                valid = bool(schema.get("is_valid") and constraint.get("is_valid"))
                random_validation.append(
                    {
                        "scenario_id": normalized.get("scenario_id"),
                        "schema_validation": schema,
                        "constraint_result": constraint,
                        "is_valid": valid,
                    }
                )
                if valid:
                    random_candidates.append(normalized)
                if len(random_candidates) >= total_shortfall:
                    break

        candidates = (
            qwen_candidates[:qwen_quota]
            + fsn_candidates[:fsn_quota]
            + qwen_candidates[qwen_quota:]
            + random_candidates
        )[:total_requested]
        source_counts = {
            "qwen": sum(1 for item in candidates if item.get("generator_type") == "qwen"),
            "fsn": sum(1 for item in candidates if item.get("generator_type") == "fsn"),
            "random": sum(1 for item in candidates if item.get("generator_type") == "random"),
        }
        actual_fsn_share = _rate(source_counts["fsn"], len(candidates))
        actual_qwen_share = _rate(source_counts["qwen"], len(candidates))
        target_fsn_share = float(
            self.config.get("target_fsn_ratio", self.config.get("replacement_ratio", 0.25))
        )
        max_actual_fsn_share = float(self.config.get("max_actual_fsn_share", 0.30))
        quota_satisfied = bool(
            len(candidates) == total_requested
            and source_counts["fsn"] <= fsn_quota
            and actual_fsn_share <= max_actual_fsn_share
            and (
                not enforce_quota
                or abs(actual_fsn_share - target_fsn_share) <= 1e-9
            )
        )
        # A true API-call reduction needs a paired all-Qwen counterfactual.
        # Slot replacement alone is not evidence that an HTTP call was saved.
        qwen_calls_saved_estimated = 0
        warnings.extend(fallback_reasons)
        self.last_result = {
            "schema_version": "falcon.fsn_replacement_generation.v2",
            "enabled": True,
            "requested": {
                "total": total_requested,
                "qwen": qwen_requested,
                "fsn": fsn_requested,
            },
            "replacement_ratio": target_fsn_share,
            "target_fsn_ratio": target_fsn_share,
            "qwen_ratio": self.config.get("qwen_ratio"),
            "quota": {
                "enforce_quota": enforce_quota,
                "total_candidates_per_round": total_requested,
                "qwen_quota": qwen_quota,
                "fsn_quota": fsn_quota,
                "max_actual_fsn_share": max_actual_fsn_share,
                "quota_satisfied": quota_satisfied,
                "qwen_source_quota_satisfied": source_counts["qwen"] >= qwen_quota,
                "fsn_source_quota_satisfied": source_counts["fsn"] == fsn_quota,
                "do_not_backfill_qwen_shortfall_with_fsn": bool(
                    self.config.get("do_not_backfill_qwen_shortfall_with_fsn", True)
                ),
            },
            "health": health,
            "fsn_candidates_requested": fsn_requested,
            "fsn_raw_count": len(fsn_raw),
            "fsn_prevalidated_count": len(fsn_candidates),
            "fsn_candidates_valid": len(fsn_candidates),
            "fsn_validation": fsn_validation,
            "fsn_runtime_seconds": fsn_runtime,
            "fsn_rerank_enabled": bool(
                self.config.get("use_fsn_rerank", False)
            ),
            "fsn_rerank_result": fsn_rerank_result,
            "fsn_repair_enabled": bool(self.config.get("use_fsn_repair", False)),
            "fsn_hardness_v2_enabled": bool(
                self.config.get("use_hardness_v2", False)
            ),
            "adapter_aware_projection": bool(
                self.config.get("adapter_aware_projection", False)
            ),
            "post_yaml_constraint_check_requested": bool(
                self.config.get("post_yaml_constraint_check", False)
            ),
            "fsn_hardness_v2_result": fsn_hardness_v2_result,
            "qwen_requested_with_fallback": qwen_target,
            "qwen_candidates_requested": qwen_candidates_requested,
            "qwen_candidates_raw_returned": qwen_candidates_raw_returned,
            "qwen_candidates_valid_returned": qwen_candidates_valid_returned,
            "qwen_candidates_valid": len(qwen_candidates),
            "qwen_generated_count": len(qwen_candidates),
            "qwen_runtime_seconds": qwen_runtime,
            "qwen_api_call_count": qwen_api_calls_attempted,
            "qwen_api_calls_attempted": qwen_api_calls_attempted,
            "qwen_api_calls_successful": qwen_api_calls_successful,
            "qwen_api_calls_failed": qwen_api_calls_failed,
            "qwen_api_retries": max(len(qwen_call_records) - 1, 0),
            "qwen_call_records": qwen_call_records,
            "qwen_calls_saved": qwen_calls_saved_estimated,
            "qwen_calls_saved_estimated": qwen_calls_saved_estimated,
            "estimated_all_qwen_api_calls": None,
            "api_call_reduction_counterfactual_measured": False,
            "true_api_call_reduction_estimate": qwen_calls_saved_estimated,
            "qwen_candidate_slots_saved": len(fsn_candidates),
            "candidate_slots_saved_estimated": len(fsn_candidates),
            "qwen_shortfall_count": qwen_shortfall,
            "random_fallback_count": len(random_candidates),
            "random_validation": random_validation,
            "actual_fsn_candidate_share": actual_fsn_share,
            "actual_qwen_candidate_share": actual_qwen_share,
            "fsn_fallback_count": fsn_fallback_to_qwen_count,
            "fsn_fallback_reasons": fallback_reasons,
            "source_counts": source_counts,
            "qwen_generation_result": copy.deepcopy(qwen_generator.last_result),
            "candidates": candidates,
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "warnings": sorted(set(str(item) for item in warnings if item)),
        }
        return candidates, copy.deepcopy(self.last_result)


def _normalize_source(
    candidate: Mapping[str, Any],
    source: str,
    index: int,
) -> Dict[str, Any]:
    result = copy.deepcopy(dict(candidate))
    original_type = result.get("generator_type")
    result["generator_type"] = source
    result["scenario_id"] = f"{source}_{index:04d}_{result.get('scenario_id', 'candidate')}"
    metadata = result.setdefault("metadata", {})
    metadata["replacement_generator_source"] = source
    metadata["original_generator_type"] = original_type
    metadata["fsn_replacement_smoke"] = True
    return result


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    vector = candidate.get("scenario_vector") or {}
    normalized = {}
    if isinstance(vector, MappingABC):
        for key, value in sorted(vector.items()):
            try:
                normalized[str(key)] = round(float(value), 6)
            except (TypeError, ValueError):
                normalized[str(key)] = value
    return json.dumps(normalized, sort_keys=True, default=str)


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0:
            return 0.0
        return round(float(numerator) / denominator_value, 6)
    except (TypeError, ValueError):
        return 0.0


def _resolve_path(value: Any) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
