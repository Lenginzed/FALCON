"""Sampling-plan scheduler for FALCON curriculum pools."""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

CURRICULUM_SCHEDULER_SCHEMA_VERSION = "falcon.curriculum_scheduler.v1"

DEFAULT_CONFIG: Dict[str, Any] = {
    "category_ratios": {
        "base": 0.30,
        "random": 0.20,
        "llm_qwen8b": 0.40,
        "replay": 0.10,
    },
    "seed": 0,
    "coverage_aware_enabled": False,
    "scenario_batch_size": 8,
    "total_train_steps_per_round": 512,
    "category_quota": {
        "accepted_llm": 4,
        "base_anchor": 2,
        "replay_failure": 1,
        "random_explore": 1,
    },
    "source_weights": {
        "accepted_llm": 1.0,
        "base_anchor": 0.8,
        "replay_failure": 1.1,
        "random_explore": 0.7,
    },
    "minimum_value_score": 0.05,
    "unseen_bonus": 2.0,
    "recency_round_interval": 5,
    "recency_bonus_per_interval": 0.25,
    "max_recency_bonus": 1.0,
    "trained_count_threshold": 3,
    "recent_anchor_rounds": 5,
    "stability_aware_enabled": False,
    "interleave_anchors": True,
    "anchor_ratio_min": 0.5,
    "accepted_ratio_max": 0.5,
    "fallback_reallocate_to_anchor": True,
}


class CurriculumScheduler:
    """Build a light-weight next-round sampling plan from a curriculum pool."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))

    def build_sampling_plan(
        self,
        curriculum_pool: Any,
        base_scenarios: Optional[Sequence[Mapping[str, Any]]] = None,
        random_scenarios: Optional[Sequence[Mapping[str, Any]]] = None,
        num_samples: int = 10,
        current_round: int = 0,
        coverage_aware: Optional[bool] = None,
        scenario_batch_size: Optional[int] = None,
        total_train_steps_per_round: Optional[int] = None,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        pool_items = _pool_items(curriculum_pool)
        categories = {
            "base": [_normalize_external_scenario(item, source="original") for item in (base_scenarios or [])],
            "random": [_normalize_external_scenario(item, source="random") for item in (random_scenarios or [])],
            "llm_qwen8b": [
                _normalize_pool_item(item)
                for item in pool_items
                if item.get("accepted_into_curriculum_pool") and _source_category(item.get("source")) == "llm_qwen8b"
            ],
            "replay": [
                _normalize_pool_item(item)
                for item in pool_items
                if item.get("accepted_into_curriculum_pool") and _source_category(item.get("source")) == "replay"
            ],
        }
        ratios = dict(self.config.get("category_ratios") or {})
        active_ratios = {
            category: _float(ratios.get(category))
            for category, scenarios in categories.items()
            if scenarios and _float(ratios.get(category)) > 0.0
        }
        total_ratio = sum(active_ratios.values())
        if total_ratio <= 0.0:
            warnings.append("No non-empty scenario categories were available; sampling plan is empty.")
            normalized_ratios = {category: 0.0 for category in ratios}
            sampled: List[Dict[str, Any]] = []
        else:
            missing_categories = [category for category, scenarios in categories.items() if not scenarios and _float(ratios.get(category)) > 0.0]
            for category in missing_categories:
                warnings.append(f"Category {category} is empty; its sampling ratio was redistributed.")
            normalized_ratios = {
                category: round(active_ratios.get(category, 0.0) / total_ratio, 6)
                for category in ratios
            }
            sampled = self._sample_by_category(categories, normalized_ratios, int(num_samples))

        use_coverage_aware = (
            bool(self.config.get("coverage_aware_enabled"))
            if coverage_aware is None
            else bool(coverage_aware)
        )
        result = {
            "schema_version": "falcon.curriculum_sampling_plan.v1",
            "num_samples": int(num_samples),
            "category_ratios": ratios,
            "normalized_category_ratios": normalized_ratios,
            "sampled_scenarios": sampled,
            "warnings": sorted(set(warnings)),
        }
        if use_coverage_aware:
            batch = self.build_scenario_batch(
                curriculum_pool,
                base_scenarios=base_scenarios,
                random_scenarios=random_scenarios,
                current_round=current_round,
                scenario_batch_size=scenario_batch_size,
                total_train_steps_per_round=total_train_steps_per_round,
            )
            result.update(batch)
            result["schema_version"] = "falcon.curriculum_sampling_plan.v2"
            result["sampled_scenarios"] = list(batch["scenario_batch"])
            result["warnings"] = sorted(set(result["warnings"] + batch.get("warnings", [])))
        return result

    def compute_sampling_distribution(
        self,
        pool_items: Sequence[Mapping[str, Any]],
        current_round: int = 0,
        coverage_aware: Optional[bool] = None,
        sampling_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        normalized = [_normalize_pool_item(item) for item in pool_items]
        use_coverage_aware = (
            bool(self.config.get("coverage_aware_enabled"))
            if coverage_aware is None
            else bool(coverage_aware)
        )
        if use_coverage_aware:
            weights = [
                self._coverage_weight(item, current_round=current_round, sampling_category=sampling_category)
                for item in normalized
            ]
        else:
            weights = [_scenario_weight(item) for item in normalized]
        total = sum(weights)
        if total <= 0.0 and normalized:
            weights = [1.0 for _ in normalized]
            total = float(len(normalized))
        output = []
        for item, weight in zip(normalized, weights):
            item = dict(item)
            item["sampling_weight"] = round(weight / total, 6) if total > 0.0 else 0.0
            output.append(item)
        return output

    def build_scenario_batch(
        self,
        curriculum_pool: Any,
        base_scenarios: Optional[Sequence[Mapping[str, Any]]] = None,
        random_scenarios: Optional[Sequence[Mapping[str, Any]]] = None,
        current_round: int = 0,
        scenario_batch_size: Optional[int] = None,
        total_train_steps_per_round: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a no-replacement, coverage-aware batch for sequential training."""

        warnings: List[str] = []
        pool_items = _pool_items(curriculum_pool)
        accepted = [item for item in pool_items if item.get("accepted_into_curriculum_pool")]
        accepted_llm = [
            _normalize_pool_item(item)
            for item in accepted
            if _source_category(item.get("source")) == "llm_qwen8b"
        ]
        replay = [
            _normalize_pool_item(item)
            for item in accepted
            if _source_category(item.get("source")) == "replay"
        ]
        base = [_normalize_external_scenario(item, source="original") for item in (base_scenarios or [])]
        random_items = [_normalize_external_scenario(item, source="random") for item in (random_scenarios or [])]
        anchors = self._anchor_candidates(accepted_llm, base, current_round=current_round)
        categories: Dict[str, List[Dict[str, Any]]] = {
            "accepted_llm": accepted_llm,
            "base_anchor": anchors,
            "replay_failure": replay,
            "random_explore": random_items,
        }
        batch_size = max(
            int(scenario_batch_size or self.config.get("scenario_batch_size", 8)),
            0,
        )
        total_steps = max(
            int(total_train_steps_per_round or self.config.get("total_train_steps_per_round", 512)),
            0,
        )
        requested_quota = {
            key: max(int(value or 0), 0)
            for key, value in dict(self.config.get("category_quota") or {}).items()
        }
        quota = _fit_quota_to_batch(requested_quota, batch_size)
        stability_enabled = bool(self.config.get("stability_aware_enabled", False))
        if stability_enabled:
            actual_quota, redistribution_warnings = _resolve_stability_quota(
                quota,
                categories,
                batch_size,
                anchor_ratio_min=_float(self.config.get("anchor_ratio_min", 0.5)),
                accepted_ratio_max=_float(self.config.get("accepted_ratio_max", 0.5)),
                fallback_reallocate_to_anchor=bool(
                    self.config.get("fallback_reallocate_to_anchor", True)
                ),
            )
        else:
            actual_quota, redistribution_warnings = _redistribute_empty_quota(
                quota,
                categories,
                batch_size,
            )
        warnings.extend(redistribution_warnings)

        rng = random.Random(int(self.config.get("seed", 0)) + int(current_round))
        selected: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        category_order = ("base_anchor", "accepted_llm", "replay_failure", "random_explore")
        for category in category_order:
            count = actual_quota.get(category, 0)
            if count <= 0:
                continue
            candidates = [item for item in categories.get(category, []) if _scenario_identity(item) not in selected_ids]
            if category == "base_anchor":
                chosen = self._sample_anchor_candidates(rng, candidates, count, current_round)
            else:
                chosen = self._weighted_sample_without_replacement(
                    rng,
                    candidates,
                    count,
                    current_round=current_round,
                    sampling_category=category,
                )
            for item in chosen:
                normalized = dict(item)
                normalized["sampling_category"] = category
                normalized["coverage_weight"] = round(
                    self._coverage_weight(normalized, current_round, category),
                    8,
                )
                selected.append(normalized)
                selected_ids.add(_scenario_identity(normalized))

        if len(selected) < batch_size:
            leftovers: List[Dict[str, Any]] = []
            fill_order = (
                ("base_anchor", "replay_failure", "random_explore", "accepted_llm")
                if stability_enabled
                else category_order
            )
            accepted_cap = max(
                int(math.floor(batch_size * _float(self.config.get("accepted_ratio_max", 0.5)))),
                0,
            )
            for category in fill_order:
                for item in categories.get(category, []):
                    if _scenario_identity(item) in selected_ids:
                        continue
                    if (
                        stability_enabled
                        and category == "accepted_llm"
                        and sum(
                            existing.get("sampling_category") == "accepted_llm"
                            for existing in selected
                        )
                        >= accepted_cap
                    ):
                        continue
                    candidate = dict(item)
                    candidate["sampling_category"] = category
                    leftovers.append(candidate)
            fill = self._weighted_sample_without_replacement(
                rng,
                leftovers,
                batch_size - len(selected),
                current_round=current_round,
                sampling_category=None,
            )
            for item in fill:
                item = dict(item)
                item["coverage_weight"] = round(
                    self._coverage_weight(item, current_round, item.get("sampling_category")),
                    8,
                )
                selected.append(item)
                selected_ids.add(_scenario_identity(item))
        if stability_enabled and bool(self.config.get("interleave_anchors", True)):
            selected = _interleave_stability_batch(selected)
        if len(selected) < batch_size:
            warnings.append(
                f"Scenario batch requested {batch_size} unique scenarios but only {len(selected)} were available."
            )

        step_allocations = _allocate_steps(total_steps, len(selected))
        for item, steps in zip(selected, step_allocations):
            item["assigned_train_steps"] = steps
        before = _coverage_snapshot(accepted, self.config)
        projected_trained_ids = {
            _scenario_identity(item)
            for item in selected
            if item.get("pool_item_id")
        }
        after = _projected_coverage_snapshot(accepted, projected_trained_ids, self.config)
        category_counts = {
            category: sum(1 for item in selected if item.get("sampling_category") == category)
            for category in requested_quota
        }
        return {
            "coverage_aware": True,
            "stability_aware": stability_enabled,
            "batch_order_strategy": (
                "interleaved_anchor_accepted"
                if stability_enabled and bool(self.config.get("interleave_anchors", True))
                else "category_grouped"
            ),
            "scenario_batch": selected,
            "scenario_batch_size": len(selected),
            "requested_scenario_batch_size": batch_size,
            "per_scenario_train_steps": min(step_allocations) if step_allocations else 0,
            "total_train_steps_per_round": sum(step_allocations),
            "category_quota": requested_quota,
            "resolved_category_quota": category_counts,
            "coverage_before": before,
            "coverage_after": after,
            "anchor_scenarios_used": [
                item.get("scenario_id")
                for item in selected
                if item.get("sampling_category") == "base_anchor"
            ],
            "anchor_ratio": round(category_counts.get("base_anchor", 0) / len(selected), 6)
            if selected
            else 0.0,
            "accepted_ratio": round(category_counts.get("accepted_llm", 0) / len(selected), 6)
            if selected
            else 0.0,
            "replay_ratio": round(category_counts.get("replay_failure", 0) / len(selected), 6)
            if selected
            else 0.0,
            "warnings": sorted(set(warnings)),
        }

    def save_sampling_plan(self, plan: Mapping[str, Any], output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(dict(plan), f, indent=2, sort_keys=True)

    def _sample_by_category(
        self,
        categories: Mapping[str, Sequence[Mapping[str, Any]]],
        normalized_ratios: Mapping[str, float],
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        if num_samples <= 0:
            return []
        rng = random.Random(int(self.config.get("seed", 0)))
        counts = _allocate_counts(normalized_ratios, num_samples)
        sampled: List[Dict[str, Any]] = []
        for category, count in counts.items():
            scenarios = list(categories.get(category) or [])
            if not scenarios or count <= 0:
                continue
            distribution = self.compute_sampling_distribution(scenarios)
            weights = [item.get("sampling_weight", 0.0) for item in distribution]
            for _ in range(count):
                chosen = _weighted_choice(rng, distribution, weights)
                chosen = dict(chosen)
                chosen["sampling_category"] = category
                sampled.append(chosen)
        return sampled[:num_samples]

    def _coverage_weight(
        self,
        item: Mapping[str, Any],
        current_round: int,
        sampling_category: Optional[str],
    ) -> float:
        value_score = max(
            _float(item.get("final_value_score")),
            _float(self.config.get("minimum_value_score", 0.05)),
        )
        category = sampling_category or _batch_category(item)
        source_weight = max(
            _float((self.config.get("source_weights") or {}).get(category, 1.0)),
            0.0,
        )
        train_count = max(int(_float(item.get("train_count"))), 0)
        undertrained_bonus = 1.0 / math.sqrt(1.0 + train_count)
        if train_count == 0 and item.get("pool_item_id"):
            undertrained_bonus *= max(_float(self.config.get("unseen_bonus", 2.0)), 1.0)
        last_round = _int_or_none(item.get("last_trained_round"))
        recency_bonus = 1.0
        if last_round is not None:
            interval = max(int(self.config.get("recency_round_interval", 5)), 1)
            elapsed = max(int(current_round) - last_round, 0)
            extra = (elapsed / interval) * _float(self.config.get("recency_bonus_per_interval", 0.25))
            recency_bonus += min(extra, _float(self.config.get("max_recency_bonus", 1.0)))
        return max(value_score * source_weight * undertrained_bonus * recency_bonus, 0.0)

    def _weighted_sample_without_replacement(
        self,
        rng: random.Random,
        candidates: Sequence[Mapping[str, Any]],
        count: int,
        current_round: int,
        sampling_category: Optional[str],
    ) -> List[Dict[str, Any]]:
        remaining = [dict(item) for item in candidates]
        output: List[Dict[str, Any]] = []
        while remaining and len(output) < count:
            weights = [
                self._coverage_weight(item, current_round=current_round, sampling_category=sampling_category)
                for item in remaining
            ]
            selected = _weighted_choice(rng, remaining, weights)
            selected_id = _scenario_identity(selected)
            output.append(dict(selected))
            remaining = [item for item in remaining if _scenario_identity(item) != selected_id]
        return output

    def _anchor_candidates(
        self,
        accepted_llm: Sequence[Mapping[str, Any]],
        base: Sequence[Mapping[str, Any]],
        current_round: int,
    ) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        for item in base:
            candidate = dict(item)
            candidate["anchor_role"] = "base_scenario"
            anchors.append(candidate)
        trained = [dict(item) for item in accepted_llm if int(_float(item.get("train_count"))) > 0]
        trained.sort(
            key=lambda item: (
                _float(item.get("final_value_score")),
                -int(_float(item.get("train_count"))),
            ),
            reverse=True,
        )
        recent_window = max(int(self.config.get("recent_anchor_rounds", 5)), 0)
        high_value_count = max(int(math.ceil(len(trained) * 0.25)), 1) if trained else 0
        high_value_ids = {_scenario_identity(item) for item in trained[:high_value_count]}
        for item in trained:
            candidate = dict(item)
            last_round = _int_or_none(candidate.get("last_trained_round"))
            if last_round is not None and int(current_round) - last_round <= recent_window:
                candidate["anchor_role"] = "recently_solved_or_trained"
            elif _scenario_identity(candidate) in high_value_ids:
                candidate["anchor_role"] = "historical_high_value"
            else:
                candidate["anchor_role"] = "historical_accepted"
            anchors.append(candidate)
        return anchors

    def _sample_anchor_candidates(
        self,
        rng: random.Random,
        candidates: Sequence[Mapping[str, Any]],
        count: int,
        current_round: int,
    ) -> List[Dict[str, Any]]:
        if count <= 0:
            return []
        base = [dict(item) for item in candidates if item.get("anchor_role") == "base_scenario"]
        selected = base[:1]
        selected_ids = {_scenario_identity(item) for item in selected}
        remaining = [item for item in candidates if _scenario_identity(item) not in selected_ids]
        selected.extend(
            self._weighted_sample_without_replacement(
                rng,
                remaining,
                max(count - len(selected), 0),
                current_round=current_round,
                sampling_category="base_anchor",
            )
        )
        return selected[:count]


def _pool_items(curriculum_pool: Any) -> List[Mapping[str, Any]]:
    if hasattr(curriculum_pool, "get_all"):
        return curriculum_pool.get_all()
    if isinstance(curriculum_pool, MappingABC):
        items = curriculum_pool.get("items") or []
        return [item for item in items if isinstance(item, MappingABC)]
    if isinstance(curriculum_pool, Sequence) and not isinstance(curriculum_pool, (str, bytes)):
        return [item for item in curriculum_pool if isinstance(item, MappingABC)]
    return []


def _normalize_pool_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_id": item.get("scenario_id"),
        "source": item.get("source"),
        "scenario_yaml_path": item.get("scenario_yaml_path"),
        "sampling_weight": _float(item.get("sampling_weight")),
        "final_value_score": _float(item.get("final_value_score")),
        "target_failure_modes": list(item.get("target_failure_modes") or []),
        "priority_level": item.get("priority_level") or "low",
        "pool_item_id": item.get("pool_item_id"),
        "train_count": max(int(_float(item.get("train_count"))), 0),
        "first_trained_round": _int_or_none(item.get("first_trained_round")),
        "last_trained_round": _int_or_none(item.get("last_trained_round")),
        "cumulative_train_steps": max(int(_float(item.get("cumulative_train_steps"))), 0),
        "coverage_status": item.get("coverage_status") or (
            "unseen" if _float(item.get("train_count")) <= 0.0 else "undertrained"
        ),
        "anchor_role": item.get("anchor_role"),
    }


def _normalize_external_scenario(item: Mapping[str, Any], source: str) -> Dict[str, Any]:
    return {
        "scenario_id": item.get("scenario_id") or item.get("name") or Path(str(item.get("scenario_yaml_path") or item.get("yaml_path") or source)).stem,
        "source": item.get("source") or source,
        "scenario_yaml_path": item.get("scenario_yaml_path") or item.get("yaml_path"),
        "sampling_weight": _float(item.get("sampling_weight", item.get("weight", 0.0))),
        "final_value_score": _float(item.get("final_value_score", 0.0)),
        "target_failure_modes": list(item.get("target_failure_modes") or []),
        "priority_level": item.get("priority_level") or "base",
        "pool_item_id": item.get("pool_item_id"),
        "train_count": max(int(_float(item.get("train_count"))), 0),
        "first_trained_round": _int_or_none(item.get("first_trained_round")),
        "last_trained_round": _int_or_none(item.get("last_trained_round")),
        "cumulative_train_steps": max(int(_float(item.get("cumulative_train_steps"))), 0),
        "coverage_status": item.get("coverage_status") or "external",
        "anchor_role": item.get("anchor_role"),
    }


def _source_category(source: Any) -> str:
    value = str(source or "").lower()
    if "fsn" in value:
        return "llm_qwen8b"
    if "qwen8b" in value or "qwen3:8b" in value or "llm_qwen8b" in value:
        return "llm_qwen8b"
    if "replay" in value or "failure" in value:
        return "replay"
    if "random" in value:
        return "random"
    if "base" in value or "original" in value:
        return "base"
    if "llm" in value or "qwen" in value:
        return "llm_qwen8b"
    return value or "unknown"


def _scenario_weight(item: Mapping[str, Any]) -> float:
    for key in ("sampling_weight", "final_value_score"):
        value = _float(item.get(key))
        if value > 0.0:
            return value
    return 1.0


def _batch_category(item: Mapping[str, Any]) -> str:
    if item.get("anchor_role"):
        return "base_anchor"
    category = item.get("sampling_category")
    if category:
        return str(category)
    source = _source_category(item.get("source"))
    return {
        "llm_qwen8b": "accepted_llm",
        "replay": "replay_failure",
        "random": "random_explore",
        "base": "base_anchor",
    }.get(source, "accepted_llm")


def _scenario_identity(item: Mapping[str, Any]) -> str:
    for key in ("scenario_yaml_path", "scenario_id", "pool_item_id"):
        if item.get(key):
            return f"{key}:{item[key]}"
    return json.dumps(dict(item), sort_keys=True, default=str)


def _fit_quota_to_batch(quota: Mapping[str, int], batch_size: int) -> Dict[str, int]:
    if batch_size <= 0:
        return {key: 0 for key in quota}
    total = sum(max(int(value), 0) for value in quota.values())
    if total <= 0:
        return {key: 0 for key in quota}
    ratios = {key: max(int(value), 0) / total for key, value in quota.items()}
    return _allocate_counts(ratios, batch_size)


def _redistribute_empty_quota(
    quota: Mapping[str, int],
    categories: Mapping[str, Sequence[Mapping[str, Any]]],
    batch_size: int,
) -> tuple[Dict[str, int], List[str]]:
    warnings: List[str] = []
    resolved = {key: 0 for key in quota}
    available_capacity = {key: len(categories.get(key) or []) for key in quota}
    remaining = batch_size
    for key, requested in quota.items():
        allocated = min(max(int(requested), 0), available_capacity.get(key, 0))
        resolved[key] = allocated
        remaining -= allocated
        if requested > 0 and available_capacity.get(key, 0) == 0:
            warnings.append(f"Scenario batch category {key} was empty; its quota was redistributed.")
    priority = ("accepted_llm", "base_anchor", "replay_failure", "random_explore")
    while remaining > 0:
        progressed = False
        for key in priority:
            if key not in resolved:
                continue
            if resolved[key] >= available_capacity.get(key, 0):
                continue
            resolved[key] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return resolved, warnings


def _resolve_stability_quota(
    quota: Mapping[str, int],
    categories: Mapping[str, Sequence[Mapping[str, Any]]],
    batch_size: int,
    anchor_ratio_min: float,
    accepted_ratio_max: float,
    fallback_reallocate_to_anchor: bool,
) -> tuple[Dict[str, int], List[str]]:
    """Resolve missing categories while protecting an anchor-heavy batch."""

    warnings: List[str] = []
    keys = tuple(quota)
    resolved = {key: 0 for key in keys}
    capacity = {key: len(categories.get(key) or []) for key in keys}
    anchor_min = min(
        max(int(math.ceil(batch_size * max(anchor_ratio_min, 0.0))), 0),
        capacity.get("base_anchor", 0),
        batch_size,
    )
    accepted_cap = min(
        max(int(math.floor(batch_size * max(accepted_ratio_max, 0.0))), 0),
        capacity.get("accepted_llm", 0),
        batch_size,
    )
    resolved["base_anchor"] = anchor_min
    remaining = batch_size - anchor_min

    for key in ("replay_failure", "random_explore"):
        requested = max(int(quota.get(key, 0)), 0)
        allocated = min(requested, capacity.get(key, 0), remaining)
        resolved[key] = allocated
        remaining -= allocated
        if requested > 0 and capacity.get(key, 0) == 0:
            warnings.append(
                f"Scenario batch category {key} was empty; its quota was reallocated to anchors first."
            )

    requested_accepted = max(int(quota.get("accepted_llm", 0)), 0)
    accepted = min(requested_accepted, accepted_cap, remaining)
    resolved["accepted_llm"] = accepted
    remaining -= accepted

    if remaining > 0 and fallback_reallocate_to_anchor:
        extra_anchor = min(
            remaining,
            max(capacity.get("base_anchor", 0) - resolved.get("base_anchor", 0), 0),
        )
        resolved["base_anchor"] += extra_anchor
        remaining -= extra_anchor

    fill_order = ("replay_failure", "random_explore", "accepted_llm", "base_anchor")
    while remaining > 0:
        progressed = False
        for key in fill_order:
            if key not in resolved:
                continue
            category_cap = capacity.get(key, 0)
            if key == "accepted_llm":
                category_cap = min(category_cap, accepted_cap)
            if resolved[key] >= category_cap:
                continue
            resolved[key] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    if resolved.get("base_anchor", 0) < int(math.ceil(batch_size * anchor_ratio_min)):
        warnings.append(
            "Stability-aware anchor minimum could not be met because too few unique anchors were available."
        )
    if resolved.get("accepted_llm", 0) > accepted_cap:
        resolved["accepted_llm"] = accepted_cap
    return resolved, warnings


def _interleave_stability_batch(
    selected: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    anchors = [
        dict(item) for item in selected if item.get("sampling_category") == "base_anchor"
    ]
    nonanchors = [
        dict(item) for item in selected if item.get("sampling_category") != "base_anchor"
    ]
    output: List[Dict[str, Any]] = []
    while anchors or nonanchors:
        if anchors:
            output.append(anchors.pop(0))
        if nonanchors:
            output.append(nonanchors.pop(0))
    return output


def _allocate_steps(total_steps: int, count: int) -> List[int]:
    if count <= 0:
        return []
    base = max(int(total_steps), 0) // count
    remainder = max(int(total_steps), 0) % count
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _coverage_snapshot(items: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    counts = [max(int(_float(item.get("train_count"))), 0) for item in items]
    threshold = max(int(config.get("trained_count_threshold", 3)), 1)
    return {
        "accepted_total": len(counts),
        "accepted_trained": sum(1 for count in counts if count > 0),
        "accepted_unseen": sum(1 for count in counts if count == 0),
        "accepted_undertrained": sum(1 for count in counts if 0 < count < threshold),
        "accepted_fully_trained": sum(1 for count in counts if count >= threshold),
        "accepted_coverage": round(sum(1 for count in counts if count > 0) / len(counts), 6)
        if counts
        else 0.0,
    }


def _projected_coverage_snapshot(
    items: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    projected: List[Dict[str, Any]] = []
    for item in items:
        candidate = dict(item)
        if _scenario_identity(item) in selected_ids:
            candidate["train_count"] = max(int(_float(item.get("train_count"))), 0) + 1
        projected.append(candidate)
    return _coverage_snapshot(projected, config)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _allocate_counts(ratios: Mapping[str, float], total: int) -> Dict[str, int]:
    raw = {category: max(_float(ratio), 0.0) * total for category, ratio in ratios.items()}
    counts = {category: int(value) for category, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], raw[key]), reverse=True)
    for category in order[: max(remaining, 0)]:
        counts[category] += 1
    return counts


def _weighted_choice(rng: random.Random, items: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> Dict[str, Any]:
    total = sum(max(_float(weight), 0.0) for weight in weights)
    if total <= 0.0:
        return copy.deepcopy(dict(items[rng.randrange(len(items))]))
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += max(_float(weight), 0.0)
        if cumulative >= threshold:
            return copy.deepcopy(dict(item))
    return copy.deepcopy(dict(items[-1]))


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if value == value and value not in (float("inf"), float("-inf")) else 0.0


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
