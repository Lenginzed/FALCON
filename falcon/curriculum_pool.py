"""Curriculum scenario pool for FALCON outer-loop smoke tests."""

from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .trajectory_recorder import SCENARIO_VECTOR_KEYS

CURRICULUM_POOL_SCHEMA_VERSION = "falcon.curriculum_pool.v1"

DEFAULT_CONFIG: Dict[str, Any] = {
    "pool_item_prefix": "pool_item",
    "timestamp_format": "%Y-%m-%dT%H:%M:%S%z",
    "max_pool_size": None,
    "trained_count_threshold": 3,
}


class CurriculumPool:
    """Store evaluated curriculum candidates and expose reusable pool stats."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))
        self.items: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "schema_version": CURRICULUM_POOL_SCHEMA_VERSION,
            "created_at": _timestamp(self.config),
        }

    def add_candidate(
        self,
        candidate: Mapping[str, Any],
        difficulty_result: Mapping[str, Any],
        source: str = "llm",
        source_round: Optional[int] = None,
        failure_summary: Optional[Mapping[str, Any]] = None,
        constraint_result: Optional[Mapping[str, Any]] = None,
        policy_eval_result: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        scenario_id = str(candidate.get("scenario_id") or difficulty_result.get("scenario_id") or f"scenario_{len(self.items):04d}")
        accepted = bool(difficulty_result.get("accepted_into_curriculum_pool"))
        final_value_score = _float(difficulty_result.get("final_value_score"))
        item = {
            "schema_version": "falcon.curriculum_pool_item.v1",
            "pool_item_id": f"{self.config['pool_item_prefix']}_{len(self.items):04d}",
            "scenario_id": scenario_id,
            "source": source,
            "source_round": source_round,
            "candidate_scenario": copy.deepcopy(dict(candidate)),
            "scenario_yaml_path": _scenario_yaml_path(candidate, difficulty_result),
            "scenario_vector": copy.deepcopy(dict(candidate.get("scenario_vector") or {})),
            "target_failure_modes": list(candidate.get("target_failure_modes") or []),
            "difficulty_result": copy.deepcopy(dict(difficulty_result)),
            "constraint_result": copy.deepcopy(dict(constraint_result or {})),
            "policy_eval_result": copy.deepcopy(dict(policy_eval_result or {})),
            "failure_vector": _failure_vector(failure_summary),
            "final_value_score": final_value_score,
            "sampling_weight": _float(difficulty_result.get("sampling_weight")),
            "priority_level": str(difficulty_result.get("priority_level") or "low"),
            "accepted_into_curriculum_pool": accepted,
            "train_count": 0,
            "first_trained_round": None,
            "last_trained_round": None,
            "cumulative_train_steps": 0,
            "coverage_status": "unseen",
            "created_at": _timestamp(self.config),
            "source_failure_id": candidate.get("source_failure_id"),
            "metadata": {
                "candidate_metadata": copy.deepcopy(dict(candidate.get("metadata") or {})),
                "difficulty_warnings": list(difficulty_result.get("warnings") or []),
                "rejection_reasons": list(difficulty_result.get("rejection_reasons") or []),
                "fsn_training_sample": {
                    "failure_vector": _failure_vector(failure_summary),
                    "candidate_scenario": copy.deepcopy(dict(candidate)),
                    "constraint_result": copy.deepcopy(dict(constraint_result or {})),
                    "difficulty_result": copy.deepcopy(dict(difficulty_result)),
                    "policy_eval_result": copy.deepcopy(dict(policy_eval_result or {})),
                    "accepted_label": accepted,
                    "changed_factors": list(candidate.get("changed_factors") or []),
                    "final_value_score": final_value_score,
                },
            },
        }
        self.items.append(item)
        self._enforce_max_pool_size()
        return item

    def add_batch(
        self,
        candidates: Sequence[Mapping[str, Any]],
        difficulty_results: Sequence[Mapping[str, Any]],
        source: str = "llm",
        source_round: Optional[int] = None,
        failure_summary: Optional[Mapping[str, Any]] = None,
        constraint_results: Optional[Sequence[Mapping[str, Any]]] = None,
        policy_eval_results: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        added = []
        for idx, candidate in enumerate(candidates):
            difficulty = difficulty_results[idx] if idx < len(difficulty_results) else {}
            constraint = constraint_results[idx] if constraint_results is not None and idx < len(constraint_results) else {}
            policy_eval = policy_eval_results[idx] if policy_eval_results is not None and idx < len(policy_eval_results) else {}
            added.append(
                self.add_candidate(
                    candidate,
                    difficulty,
                    source=source,
                    source_round=source_round,
                    failure_summary=failure_summary,
                    constraint_result=constraint,
                    policy_eval_result=policy_eval,
                )
            )
        return added

    def load(self, path: Union[str, Path]) -> "CurriculumPool":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = [
            _normalize_coverage_fields(dict(item), self.config)
            for item in data.get("items", [])
            if isinstance(item, MappingABC)
        ]
        self.metadata = dict(data.get("metadata") or {})
        return self

    def save(self, path: Union[str, Path]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CURRICULUM_POOL_SCHEMA_VERSION,
            "metadata": self.metadata,
            "items": self.items,
            "stats": self.get_stats(),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def get_all(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(item) for item in self.items]

    def get_accepted(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(item) for item in self.items if item.get("accepted_into_curriculum_pool")]

    def get_by_priority(self, priority_level: str) -> List[Dict[str, Any]]:
        return [copy.deepcopy(item) for item in self.items if item.get("priority_level") == priority_level]

    def record_training(
        self,
        scenario: Union[str, Mapping[str, Any]],
        round_id: int,
        train_steps: int,
    ) -> Optional[Dict[str, Any]]:
        """Record one successful training use without changing pool acceptance."""

        keys = _scenario_identity_keys(scenario)
        if not keys:
            return None
        for item in self.items:
            if not keys.intersection(_scenario_identity_keys(item)):
                continue
            item = _normalize_coverage_fields(item, self.config)
            item["train_count"] = int(item.get("train_count", 0)) + 1
            if item.get("first_trained_round") is None:
                item["first_trained_round"] = int(round_id)
            item["last_trained_round"] = int(round_id)
            item["cumulative_train_steps"] = int(item.get("cumulative_train_steps", 0)) + max(int(train_steps), 0)
            item["coverage_status"] = _coverage_status(item["train_count"], self.config)
            metadata = item.setdefault("metadata", {})
            history = metadata.setdefault("training_history", [])
            history.append(
                {
                    "round_id": int(round_id),
                    "train_steps": max(int(train_steps), 0),
                    "recorded_at": _timestamp(self.config),
                }
            )
            return copy.deepcopy(item)
        return None

    def record_training_batch(
        self,
        training_records: Sequence[Mapping[str, Any]],
        round_id: int,
    ) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        for record in training_records:
            if not record.get("training_succeeded", record.get("checkpoint_saved", False)):
                continue
            item = self.record_training(
                record,
                round_id=round_id,
                train_steps=int(record.get("train_steps", record.get("num_env_steps", 0)) or 0),
            )
            if item is not None:
                updated.append(item)
        return updated

    def get_stats(self) -> Dict[str, Any]:
        total_items = len(self.items)
        accepted_items = sum(1 for item in self.items if item.get("accepted_into_curriculum_pool"))
        values = [_float(item.get("final_value_score")) for item in self.items]
        source_counts = Counter(str(item.get("source", "unknown")) for item in self.items)
        priority_counts = Counter(str(item.get("priority_level", "unknown")) for item in self.items)
        mode_counts: Counter[str] = Counter()
        for item in self.items:
            mode_counts.update(str(mode) for mode in item.get("target_failure_modes") or [])
        vectors = [item.get("scenario_vector") for item in self.items if isinstance(item.get("scenario_vector"), MappingABC)]
        coverage_counts = Counter(
            str(_normalize_coverage_fields(item, self.config).get("coverage_status", "unseen"))
            for item in self.items
            if item.get("accepted_into_curriculum_pool")
        )
        accepted_train_counts = [
            int(_normalize_coverage_fields(item, self.config).get("train_count", 0))
            for item in self.items
            if item.get("accepted_into_curriculum_pool")
        ]
        return {
            "schema_version": "falcon.curriculum_pool_stats.v1",
            "total_items": total_items,
            "accepted_items": accepted_items,
            "rejected_items": max(total_items - accepted_items, 0),
            "source_counts": dict(sorted(source_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
            "mean_value_score": round(sum(values) / len(values), 6) if values else 0.0,
            "target_failure_mode_counts": dict(sorted(mode_counts.items())),
            "scenario_vector_mean": _vector_stat(vectors, "mean"),
            "scenario_vector_std": _vector_stat(vectors, "std"),
            "scenario_vectors": [dict(vector) for vector in vectors],
            "coverage_counts": {
                status: int(coverage_counts.get(status, 0))
                for status in ("unseen", "undertrained", "trained")
            },
            "accepted_trained_items": sum(1 for count in accepted_train_counts if count > 0),
            "accepted_unseen_items": sum(1 for count in accepted_train_counts if count == 0),
            "accepted_training_coverage": round(
                sum(1 for count in accepted_train_counts if count > 0) / len(accepted_train_counts),
                6,
            )
            if accepted_train_counts
            else 0.0,
            "accepted_mean_train_count": round(sum(accepted_train_counts) / len(accepted_train_counts), 6)
            if accepted_train_counts
            else 0.0,
        }

    def export_training_manifest(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        accepted = self.get_accepted()
        manifest = {
            "schema_version": "falcon.curriculum_training_manifest.v1",
            "num_scenarios": len(accepted),
            "scenarios": [
                {
                    "pool_item_id": item.get("pool_item_id"),
                    "scenario_id": item.get("scenario_id"),
                    "source": item.get("source"),
                    "scenario_yaml_path": item.get("scenario_yaml_path"),
                    "sampling_weight": item.get("sampling_weight"),
                    "final_value_score": item.get("final_value_score"),
                    "priority_level": item.get("priority_level"),
                    "target_failure_modes": item.get("target_failure_modes"),
                    "train_count": item.get("train_count", 0),
                    "first_trained_round": item.get("first_trained_round"),
                    "last_trained_round": item.get("last_trained_round"),
                    "cumulative_train_steps": item.get("cumulative_train_steps", 0),
                    "coverage_status": item.get("coverage_status", "unseen"),
                }
                for item in accepted
            ],
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return manifest

    def _enforce_max_pool_size(self) -> None:
        max_pool_size = self.config.get("max_pool_size")
        try:
            max_pool_size = int(max_pool_size)
        except (TypeError, ValueError):
            return
        if max_pool_size <= 0 or len(self.items) <= max_pool_size:
            return
        indexed = list(enumerate(self.items))
        indexed.sort(key=lambda pair: _retention_key(pair[0], pair[1]), reverse=True)
        keep_indices = {idx for idx, _item in indexed[:max_pool_size]}
        self.items = [item for idx, item in enumerate(self.items) if idx in keep_indices]
        self.metadata["trimmed_to_max_pool_size"] = max_pool_size


def _scenario_yaml_path(candidate: Mapping[str, Any], difficulty_result: Mapping[str, Any]) -> Optional[str]:
    for key in ("scenario_yaml_path", "yaml_path"):
        if candidate.get(key):
            return str(candidate[key])
    metadata = difficulty_result.get("metadata") if isinstance(difficulty_result.get("metadata"), MappingABC) else {}
    for eval_key in ("current_policy_eval", "best_policy_eval"):
        policy_eval = metadata.get(eval_key) if isinstance(metadata.get(eval_key), MappingABC) else {}
        value = policy_eval.get("scenario_yaml")
        if value and value != "memory":
            return str(value)
    return None


def _failure_vector(failure_summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(failure_summary, MappingABC):
        return {}
    return {
        "failure_scores": copy.deepcopy(dict(failure_summary.get("failure_scores") or {})),
        "primary_failure_modes": list(failure_summary.get("primary_failure_modes") or []),
        "secondary_failure_modes": list(failure_summary.get("secondary_failure_modes") or []),
        "scenario_vector": copy.deepcopy(dict(failure_summary.get("scenario_vector") or {})),
        "failure_severity": (failure_summary.get("failure_scores") or {}).get("failure_severity"),
        "source_trajectory": failure_summary.get("source_trajectory"),
    }


def _retention_key(index: int, item: Mapping[str, Any]) -> tuple:
    priority_rank = {"high": 3, "medium": 2, "low": 1}.get(str(item.get("priority_level", "low")), 0)
    accepted_rank = 1 if item.get("accepted_into_curriculum_pool") else 0
    return (
        accepted_rank,
        priority_rank,
        _float(item.get("final_value_score")),
        _float(item.get("sampling_weight")),
        index,
    )


def _coverage_status(train_count: int, config: Mapping[str, Any]) -> str:
    count = max(int(train_count), 0)
    if count == 0:
        return "unseen"
    try:
        threshold = max(int(config.get("trained_count_threshold", 3)), 1)
    except (TypeError, ValueError):
        threshold = 3
    return "trained" if count >= threshold else "undertrained"


def _normalize_coverage_fields(item: Dict[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        train_count = max(int(item.get("train_count", 0) or 0), 0)
    except (TypeError, ValueError):
        train_count = 0
    try:
        cumulative_steps = max(int(item.get("cumulative_train_steps", 0) or 0), 0)
    except (TypeError, ValueError):
        cumulative_steps = 0
    item["train_count"] = train_count
    item["first_trained_round"] = _int_or_none(item.get("first_trained_round"))
    item["last_trained_round"] = _int_or_none(item.get("last_trained_round"))
    item["cumulative_train_steps"] = cumulative_steps
    item["coverage_status"] = _coverage_status(train_count, config)
    return item


def _scenario_identity_keys(scenario: Union[str, Mapping[str, Any]]) -> set[str]:
    if isinstance(scenario, str):
        return {scenario} if scenario else set()
    if not isinstance(scenario, MappingABC):
        return set()
    keys: set[str] = set()
    for key in ("scenario_yaml_path", "yaml_path"):
        value = scenario.get(key)
        if value:
            try:
                keys.add(f"yaml:{Path(str(value)).resolve()}")
            except OSError:
                keys.add(f"yaml:{value}")
    if keys:
        return keys
    if scenario.get("scenario_id"):
        keys.add(f"scenario_id:{scenario['scenario_id']}")
        return keys
    if scenario.get("pool_item_id"):
        keys.add(f"pool_item_id:{scenario['pool_item_id']}")
    return keys


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _vector_stat(vectors: Sequence[Mapping[str, Any]], kind: str) -> Dict[str, Optional[float]]:
    output: Dict[str, Optional[float]] = {}
    for key in SCENARIO_VECTOR_KEYS:
        values = [_float_or_none(vector.get(key)) for vector in vectors if isinstance(vector, MappingABC)]
        values = [value for value in values if value is not None]
        if not values:
            output[key] = None
        elif kind == "mean":
            output[key] = round(sum(values) / len(values), 6)
        elif len(values) == 1:
            output[key] = 0.0
        else:
            mean = sum(values) / len(values)
            output[key] = round(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 6)
    return output


def _timestamp(config: Mapping[str, Any]) -> str:
    return time.strftime(str(config.get("timestamp_format", DEFAULT_CONFIG["timestamp_format"])))


def _float(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
