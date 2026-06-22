"""Offline CandidateScenario generation from a trained FSN checkpoint."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .candidate_schema import create_candidate_scenario
from .constraint_checker import ConstraintChecker
from .fsn_trainer import load_fsn_checkpoint
from .fsn_hardness_surrogate import (
    DualBoundarySurrogate,
    load_hardness_surrogate,
    score_candidate_with_surrogate,
)
from .scenario_adapter import (
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    scenario_vector_to_initial_config,
)

GENERATABLE_FACTORS = (
    "team_center_distance",
    "own_formation_spread",
    "opponent_formation_spread",
    "altitude_difference",
    "velocity_difference",
    "heading_difference",
    "approximate_aspect_angle",
)

FACTOR_RANGES = {
    "team_center_distance": (6000.0, 18000.0),
    "own_formation_spread": (1000.0, 8000.0),
    "opponent_formation_spread": (1000.0, 8000.0),
    "altitude_difference": (-2500.0, 2500.0),
    "velocity_difference": (-60.0, 60.0),
    "heading_difference": (0.0, 2.0 * math.pi),
    "approximate_aspect_angle": (0.0, 2.0 * math.pi),
}


class FSNScenarioGenerator:
    """Generate legalizable scenario proposals without invoking an LLM."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.model, self.codec, self.payload = load_fsn_checkpoint(
            self.checkpoint_path
        )
        self.config = {
            "seed": 11,
            "factor_threshold": 0.45,
            "max_changed_factors": 3,
            "noise_scale": 0.08,
            "diversity_aware": True,
            "oversample_factor": 4,
            "min_diversity_distance": 0.08,
            **dict(config or {}),
        }
        self.rng = random.Random(self.config["seed"])
        self.last_generation_runtime_seconds = 0.0
        self.last_rerank_result: Dict[str, Any] = {}
        self.last_repair_result: Dict[str, Any] = {}
        self.last_hardness_result: Dict[str, Any] = {}
        self.last_hardness_v2_result: Dict[str, Any] = {}

    def generate_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int = 5,
        policy_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        base_initial = extract_initial_config_from_yaml(base_config)
        base_vector = initial_config_to_scenario_vector(base_initial)[
            "scenario_vector"
        ]
        input_sample = {
            "failure_vector": dict(failure_summary.get("failure_scores") or {}),
            "policy_eval": dict(policy_context or {}),
            "proxy_features": _proxy_features(
                base_vector, policy_context or {}
            ),
            "candidate_scenario_vector": {},
            "changed_factors": [],
            "difficulty": {},
            "label": "accepted",
            "sample_weight": 1.0,
        }
        encoded = self.codec.encode(input_sample)
        features = torch.tensor(
            [encoded["features"]], dtype=torch.float32
        )
        with torch.no_grad():
            outputs = self.model(features)
        base_prediction = (
            outputs["scenario_vector"][0].detach().cpu().tolist()
        )
        predicted_vector = self.codec.decode_scenario(base_prediction)
        factor_probs = torch.sigmoid(
            outputs["changed_factor_logits"][0]
        ).detach().cpu().tolist()
        factor_scores = {
            factor: factor_probs[index]
            for index, factor in enumerate(self.codec.factor_vocab)
            if factor in GENERATABLE_FACTORS
        }
        ranked_factors = sorted(
            factor_scores,
            key=lambda factor: factor_scores[factor],
            reverse=True,
        )
        selected_factors = [
            factor
            for factor in ranked_factors
            if factor_scores[factor] >= float(self.config["factor_threshold"])
        ][: int(self.config["max_changed_factors"])]
        if not selected_factors:
            selected_factors = ranked_factors[:1] or ["team_center_distance"]
        predicted_value = float(outputs["value"][0].detach().cpu())
        label_probs = torch.softmax(outputs["label_logits"][0], dim=-1)
        predicted_label_index = int(label_probs.argmax().item())
        predicted_label = self.codec.label_vocab[predicted_label_index]
        accepted_index = self.codec.label_vocab.index("accepted")
        accepted_probability = float(label_probs[accepted_index].item())
        constraint_probability = (
            float(outputs["constraint_valid_probability"][0].item())
            if "constraint_valid_probability" in outputs
            else None
        )

        target_modes = list(
            failure_summary.get("primary_failure_modes")
            or failure_summary.get("secondary_failure_modes")
            or []
        )
        source_failure_id = (
            failure_summary.get("source_trajectory")
            or failure_summary.get("episode_id")
            or "offline_failure"
        )
        requested = max(int(num_scenarios), 0)
        proposal_count = requested
        if self.config.get("diversity_aware") and requested > 1:
            proposal_count = max(
                requested,
                requested * max(int(self.config["oversample_factor"]), 1),
            )
        proposals: List[Dict[str, Any]] = []
        for index in range(proposal_count):
            vector = self._legalized_vector(
                predicted_vector, base_vector, selected_factors, index
            )
            initial_config = scenario_vector_to_initial_config(
                vector, base_initial
            )
            recomputed = initial_config_to_scenario_vector(initial_config)[
                "scenario_vector"
            ]
            changed = self._candidate_factors(selected_factors, index)
            proposals.append(
                create_candidate_scenario(
                    scenario_id=f"fsn_{index:04d}",
                    generator_type="fsn",
                    source_failure_id=str(source_failure_id),
                    target_failure_modes=target_modes,
                    changed_factors=changed,
                    counterfactual_group_id=f"fsn_{source_failure_id}",
                    scenario_vector=recomputed,
                    scenario_parameters={
                        "raw_fsn_prediction": predicted_vector,
                        "legalized_request": vector,
                    },
                    initial_config=initial_config,
                    expected_effect=(
                        "Offline FSN proposal targeting "
                        + ", ".join(target_modes or ["observed failure"])
                    ),
                    rationale=(
                        "Lightweight distilled proposal; external schema and "
                        "constraint validation remain mandatory."
                    ),
                    metadata={
                        "checkpoint_path": str(self.checkpoint_path),
                        "predicted_value_score": round(predicted_value, 6),
                        "predicted_accepted_probability": round(
                            accepted_probability, 6
                        ),
                        "predicted_label": predicted_label,
                        "predicted_constraint_valid_probability": (
                            None
                            if constraint_probability is None
                            else round(constraint_probability, 6)
                        ),
                        "changed_factor_probabilities": {
                            key: round(value, 6)
                            for key, value in factor_scores.items()
                        },
                        "offline_generation_only": True,
                        "qwen_call_used": False,
                        "diversity_aware_generation": bool(
                            self.config.get("diversity_aware")
                        ),
                    },
                )
            )
        candidates = (
            self._select_diverse(proposals, requested)
            if self.config.get("diversity_aware")
            else proposals[:requested]
        )
        self.last_generation_runtime_seconds = round(
            time.perf_counter() - started, 6
        )
        return candidates

    def generate_reranked_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int = 4,
        overgenerate_count: int = 16,
        policy_context: Optional[Mapping[str, Any]] = None,
        rerank_weights: Optional[Mapping[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Over-generate, score, and return the highest-ranked FSN candidates."""

        started = time.perf_counter()
        weights = {
            "predicted_value_score": 0.4,
            "accepted_probability": 0.3,
            "diversity_bonus": 0.2,
            "constraint_risk_penalty": 0.1,
            **dict(rerank_weights or {}),
        }
        requested = max(int(num_scenarios), 0)
        overgenerate_count = max(int(overgenerate_count), requested)
        original_diversity = self.config.get("diversity_aware", True)
        try:
            self.config["diversity_aware"] = False
            proposals = self.generate_from_failure_summary(
                failure_summary,
                base_config,
                num_scenarios=overgenerate_count,
                policy_context=policy_context,
            )
        finally:
            self.config["diversity_aware"] = original_diversity
        checker = ConstraintChecker()
        diversity = _proposal_diversity_scores(proposals)
        scored: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for index, proposal in enumerate(proposals):
            candidate = dict(proposal)
            metadata = candidate.setdefault("metadata", {})
            value = _optional_clip01(metadata.get("predicted_value_score"))
            accepted_probability = _optional_clip01(
                metadata.get("predicted_accepted_probability")
            )
            constraint = checker.validate_candidate(candidate)
            validity = _optional_clip01(constraint.get("validity_score"))
            risk = None if validity is None else 1.0 - validity
            components = {
                "predicted_value_score": value,
                "accepted_probability": accepted_probability,
                "diversity_bonus": diversity[index],
                "constraint_risk_penalty": risk,
            }
            available_weight = sum(
                abs(float(weights[key]))
                for key, component in components.items()
                if component is not None
            )
            if available_weight <= 0:
                score = diversity[index]
                warnings.append(
                    "All configured rerank fields were unavailable; used diversity only."
                )
            else:
                score = (
                    float(weights["predicted_value_score"]) * (value or 0.0)
                    + float(weights["accepted_probability"])
                    * (accepted_probability or 0.0)
                    + float(weights["diversity_bonus"]) * diversity[index]
                    - float(weights["constraint_risk_penalty"]) * (risk or 0.0)
                ) / available_weight
            metadata["rerank_score"] = round(score, 6)
            metadata["rerank_components"] = {
                key: None if component is None else round(component, 6)
                for key, component in components.items()
            }
            metadata["rerank_constraint_valid"] = bool(constraint.get("is_valid"))
            scored.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "constraint_valid": bool(constraint.get("is_valid")),
                    "constraint_result": constraint,
                }
            )
        scored.sort(
            key=lambda item: (
                item["constraint_valid"],
                item["score"],
                item["candidate"]["metadata"]["rerank_components"][
                    "diversity_bonus"
                ],
            ),
            reverse=True,
        )
        selected = [dict(item["candidate"]) for item in scored[:requested]]
        for index, candidate in enumerate(selected):
            candidate["scenario_id"] = f"fsn_rerank_{index:04d}"
            candidate.setdefault("metadata", {})["rerank_selected_rank"] = index + 1
            candidate["metadata"]["overgenerated_candidate_count"] = overgenerate_count
        self.last_generation_runtime_seconds = round(
            time.perf_counter() - started, 6
        )
        self.last_rerank_result = {
            "schema_version": "falcon.fsn_rerank_result.v1",
            "requested_candidates": requested,
            "overgenerated_candidates": len(proposals),
            "selected_candidates": len(selected),
            "weights": weights,
            "selected_scores": [
                candidate.get("metadata", {}).get("rerank_score")
                for candidate in selected
            ],
            "predicted_value_unique_count": len(
                {
                    candidate.get("metadata", {}).get("predicted_value_score")
                    for candidate in proposals
                }
            ),
            "accepted_probability_unique_count": len(
                {
                    candidate.get("metadata", {}).get(
                        "predicted_accepted_probability"
                    )
                    for candidate in proposals
                }
            ),
            "runtime_seconds": self.last_generation_runtime_seconds,
            "warnings": sorted(set(warnings)),
        }
        return selected

    def generate_repaired_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int = 4,
        overgenerate_count: int = 16,
        max_repair_attempts: int = 3,
        repair_margin_m: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Generate reranked proposals and repair their YAML round-trip."""
        started = time.perf_counter()
        proposals = self.generate_reranked_from_failure_summary(
            failure_summary,
            base_config,
            num_scenarios=num_scenarios,
            overgenerate_count=overgenerate_count,
            rerank_weights={
                "predicted_value_score": 0.0,
                "accepted_probability": 0.0,
                "diversity_bonus": 0.65,
                "constraint_risk_penalty": 0.35,
            },
        )
        repaired, details = self._repair_candidates(
            proposals,
            base_config,
            max_repair_attempts=max_repair_attempts,
            repair_margin_m=repair_margin_m,
        )
        for index, candidate in enumerate(repaired):
            candidate["scenario_id"] = f"fsn_repaired_{index:04d}"
            candidate["generator_type"] = "fsn_repaired"
        self.last_generation_runtime_seconds = round(
            time.perf_counter() - started, 6
        )
        self.last_repair_result = {
            "schema_version": "falcon.fsn_repair_result.v1",
            "requested_candidates": num_scenarios,
            "input_candidates": len(proposals),
            "valid_candidates": len(repaired),
            "repair_success_count": sum(
                1 for item in details if item.get("repair_success")
            ),
            "details": details,
            "runtime_seconds": self.last_generation_runtime_seconds,
        }
        return repaired

    def generate_hardness_filtered_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int = 4,
        overgenerate_count: int = 32,
        pool_stats: Optional[Mapping[str, Any]] = None,
        max_repair_attempts: int = 3,
        repair_margin_m: float = 5.0,
        hardness_weights: Optional[Mapping[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Over-generate, YAML-repair, then select hard and diverse proposals."""
        started = time.perf_counter()
        weights = {
            "initial_disadvantage_proxy": 0.25,
            "formation_stress": 0.20,
            "heading_aspect_stress": 0.15,
            "altitude_velocity_stress": 0.15,
            "pool_novelty": 0.15,
            "target_assignment_ambiguity_proxy": 0.10,
            **dict(hardness_weights or {}),
        }
        original_diversity = self.config.get("diversity_aware", True)
        try:
            self.config["diversity_aware"] = False
            proposals = self.generate_from_failure_summary(
                failure_summary,
                base_config,
                num_scenarios=max(int(overgenerate_count), int(num_scenarios)),
            )
        finally:
            self.config["diversity_aware"] = original_diversity
        repaired, repair_details = self._repair_candidates(
            proposals,
            base_config,
            max_repair_attempts=max_repair_attempts,
            repair_margin_m=repair_margin_m,
        )
        diversity_scores = _proposal_diversity_scores(repaired)
        scored = []
        for candidate, diversity in zip(repaired, diversity_scores):
            components = _hardness_components(candidate, pool_stats or {})
            hardness = sum(
                float(weights[key]) * float(components[key])
                for key in weights
            ) / max(sum(abs(float(value)) for value in weights.values()), 1e-8)
            score = 0.75 * hardness + 0.25 * diversity
            metadata = candidate.setdefault("metadata", {})
            metadata["hardness_proxy"] = round(hardness, 6)
            metadata["hardness_components"] = {
                key: round(value, 6) for key, value in components.items()
            }
            metadata["hardness_diversity_score"] = round(diversity, 6)
            metadata["hardness_selection_score"] = round(score, 6)
            scored.append((score, diversity, candidate))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [item[2] for item in scored[: max(int(num_scenarios), 0)]]
        for index, candidate in enumerate(selected):
            candidate["scenario_id"] = f"fsn_hardness_{index:04d}"
            candidate["generator_type"] = "fsn_repaired_hardness"
            candidate.setdefault("metadata", {})[
                "hardness_selected_rank"
            ] = index + 1
        self.last_generation_runtime_seconds = round(
            time.perf_counter() - started, 6
        )
        self.last_hardness_result = {
            "schema_version": "falcon.fsn_hardness_filter_result.v1",
            "overgenerated_candidates": len(proposals),
            "post_yaml_valid_candidates": len(repaired),
            "selected_candidates": len(selected),
            "repair_success_count": sum(
                1 for item in repair_details if item.get("repair_success")
            ),
            "weights": weights,
            "selected_scores": [
                candidate.get("metadata", {}).get("hardness_selection_score")
                for candidate in selected
            ],
            "selected_hardness": [
                candidate.get("metadata", {}).get("hardness_proxy")
                for candidate in selected
            ],
            "repair_details": repair_details,
            "runtime_seconds": self.last_generation_runtime_seconds,
        }
        return selected

    def generate_hardness_v2_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        surrogate: Optional[DualBoundarySurrogate] = None,
        surrogate_path: Optional[str | Path] = None,
        num_scenarios: int = 4,
        overgenerate_count: int = 64,
        pool_stats: Optional[Mapping[str, Any]] = None,
        max_repair_attempts: int = 3,
        repair_margin_m: float = 5.0,
        score_weights: Optional[Mapping[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Select repaired FSN proposals with a dual-boundary surrogate proxy."""
        started = time.perf_counter()
        if surrogate is None:
            if surrogate_path is None:
                raise ValueError("surrogate or surrogate_path is required for hardness v2")
            surrogate = load_hardness_surrogate(surrogate_path)
        weights = {
            "predicted_learning_potential": 0.30,
            "predicted_accepted_probability": 0.25,
            "diversity_bonus": 0.20,
            "pool_novelty": 0.10,
            "predicted_too_easy_probability": 0.10,
            "predicted_not_solvable_probability": 0.10,
            "constraint_risk": 0.05,
            **dict(score_weights or {}),
        }
        original_diversity = self.config.get("diversity_aware", True)
        try:
            self.config["diversity_aware"] = False
            proposals = self.generate_from_failure_summary(
                failure_summary,
                base_config,
                num_scenarios=max(int(overgenerate_count), int(num_scenarios)),
            )
        finally:
            self.config["diversity_aware"] = original_diversity
        repaired, repair_details = self._repair_candidates(
            proposals,
            base_config,
            max_repair_attempts=max_repair_attempts,
            repair_margin_m=repair_margin_m,
        )
        diversity_scores = _proposal_diversity_scores(repaired)
        checker = ConstraintChecker()
        scored = []
        for candidate, diversity in zip(repaired, diversity_scores):
            prediction = score_candidate_with_surrogate(
                candidate,
                surrogate,
                failure_summary=failure_summary,
                pool_stats=pool_stats or {},
            )
            constraint = checker.validate_candidate(candidate)
            constraint_risk = 1.0 - _clip01(
                float(constraint.get("validity_score") or 0.0)
            )
            pool_novelty = (
                prediction.get("proxy_features") or {}
            ).get("runtime_pool_novelty", 0.5)
            components = {
                "predicted_learning_potential": _clip01(
                    prediction.get("predicted_learning_potential", 0.0)
                ),
                "predicted_accepted_probability": _clip01(
                    prediction.get("predicted_accepted_probability", 0.0)
                ),
                "diversity_bonus": _clip01(diversity),
                "pool_novelty": _clip01(pool_novelty),
                "predicted_too_easy_probability": _clip01(
                    prediction.get("predicted_too_easy_probability", 0.0)
                ),
                "predicted_not_solvable_probability": _clip01(
                    prediction.get("predicted_not_solvable_probability", 0.0)
                ),
                "constraint_risk": _clip01(constraint_risk),
            }
            positive = (
                float(weights["predicted_learning_potential"])
                * components["predicted_learning_potential"]
                + float(weights["predicted_accepted_probability"])
                * components["predicted_accepted_probability"]
                + float(weights["diversity_bonus"])
                * components["diversity_bonus"]
                + float(weights["pool_novelty"]) * components["pool_novelty"]
            )
            negative = (
                float(weights["predicted_too_easy_probability"])
                * components["predicted_too_easy_probability"]
                + float(weights["predicted_not_solvable_probability"])
                * components["predicted_not_solvable_probability"]
                + float(weights["constraint_risk"])
                * components["constraint_risk"]
            )
            normalizer = max(sum(abs(float(value)) for value in weights.values()), 1e-8)
            score = (positive - negative) / normalizer
            metadata = candidate.setdefault("metadata", {})
            metadata["hardness_v2_score"] = round(score, 6)
            metadata["hardness_v2_components"] = {
                key: round(value, 6) for key, value in components.items()
            }
            metadata["hardness_v2_surrogate_prediction"] = prediction
            metadata["hardness_v2_constraint_valid"] = bool(constraint.get("is_valid"))
            scored.append((score, diversity, candidate))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [item[2] for item in scored[: max(int(num_scenarios), 0)]]
        for index, candidate in enumerate(selected):
            candidate["scenario_id"] = f"fsn_hardness_v2_{index:04d}"
            candidate["generator_type"] = "fsn_repaired_hardness_v2"
            candidate.setdefault("metadata", {})["hardness_v2_selected_rank"] = index + 1
        self.last_generation_runtime_seconds = round(
            time.perf_counter() - started, 6
        )
        self.last_hardness_v2_result = {
            "schema_version": "falcon.fsn_hardness_v2_filter_result.v1",
            "overgenerated_candidates": len(proposals),
            "post_yaml_valid_candidates": len(repaired),
            "selected_candidates": len(selected),
            "repair_success_count": sum(
                1 for item in repair_details if item.get("repair_success")
            ),
            "weights": weights,
            "selected_scores": [
                candidate.get("metadata", {}).get("hardness_v2_score")
                for candidate in selected
            ],
            "repair_details": repair_details,
            "runtime_seconds": self.last_generation_runtime_seconds,
        }
        return selected

    def _repair_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        base_config: Mapping[str, Any],
        max_repair_attempts: int,
        repair_margin_m: float,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        checker = ConstraintChecker()
        repaired: List[Dict[str, Any]] = []
        details: List[Dict[str, Any]] = []
        for candidate in candidates:
            result = checker.repair_candidate_for_yaml(
                candidate,
                base_config,
                max_repair_attempts=max_repair_attempts,
                repair_margin_m=repair_margin_m,
                enable_env_load_check=False,
            )
            details.append(
                {
                    "scenario_id": candidate.get("scenario_id"),
                    "is_valid": result.get("is_valid"),
                    "repair_success": result.get("repair_success"),
                    "repair_actions": result.get("repair_actions") or [],
                    "history": result.get("history") or [],
                    "rejection_reasons": (
                        result.get("constraint_result") or {}
                    ).get("rejection_reasons", []),
                }
            )
            if result.get("is_valid"):
                repaired.append(dict(result["candidate"]))
        return repaired, details

    def _select_diverse(
        self,
        proposals: Sequence[Mapping[str, Any]],
        requested: int,
    ) -> List[Dict[str, Any]]:
        if requested <= 0 or not proposals:
            return []
        selected = [dict(proposals[0])]
        remaining = [dict(item) for item in proposals[1:]]
        threshold = float(self.config["min_diversity_distance"])
        while remaining and len(selected) < requested:
            scored = [
                (
                    min(
                        _normalized_vector_distance(
                            item.get("scenario_vector") or {},
                            chosen.get("scenario_vector") or {},
                        )
                        for chosen in selected
                    ),
                    index,
                )
                for index, item in enumerate(remaining)
            ]
            distance, best_index = max(scored)
            candidate = remaining.pop(best_index)
            candidate.setdefault("metadata", {})[
                "nearest_selected_distance_at_selection"
            ] = round(distance, 6)
            if distance < threshold and len(remaining) >= requested - len(selected):
                continue
            selected.append(candidate)
        if len(selected) < requested:
            selected.extend(remaining[: requested - len(selected)])
        for index, candidate in enumerate(selected):
            candidate["scenario_id"] = f"fsn_{index:04d}"
        return selected

    def _legalized_vector(
        self,
        prediction: Mapping[str, Any],
        base_vector: Mapping[str, Any],
        selected_factors: Sequence[str],
        index: int,
    ) -> Dict[str, Any]:
        vector = dict(base_vector)
        for factor in GENERATABLE_FACTORS:
            low, high = FACTOR_RANGES[factor]
            value = _finite_float(prediction.get(factor), base_vector.get(factor))
            span = high - low
            noise = (
                self.rng.uniform(-1.0, 1.0)
                * float(self.config["noise_scale"])
                * span
                * (1.0 + 0.15 * index)
            )
            if factor not in selected_factors:
                noise *= 0.25
            vector[factor] = min(max(value + noise, low), high)
        vector["own_center_x"] = base_vector.get("own_center_x")
        vector["own_center_y"] = base_vector.get("own_center_y")
        vector["own_center_z"] = base_vector.get("own_center_z")
        vector["opponent_center_x"] = None
        vector["opponent_center_y"] = None
        vector["opponent_center_z"] = None
        return vector

    def _candidate_factors(
        self, selected_factors: Sequence[str], index: int
    ) -> List[str]:
        count = min(
            max(1 + index % max(len(selected_factors), 1), 1),
            int(self.config["max_changed_factors"]),
            len(selected_factors),
        )
        return list(selected_factors[:count])


def _finite_float(value: Any, fallback: Any) -> float:
    for candidate in (value, fallback, 0.0):
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return 0.0


def _proxy_features(
    scenario_vector: Mapping[str, Any],
    policy_context: Mapping[str, Any],
) -> Dict[str, float]:
    distance = _finite_float(
        scenario_vector.get("team_center_distance"), 12000.0
    )
    own_spread = _finite_float(
        scenario_vector.get("own_formation_spread"), 3000.0
    )
    opponent_spread = _finite_float(
        scenario_vector.get("opponent_formation_spread"), 3000.0
    )
    altitude = abs(
        _finite_float(scenario_vector.get("altitude_difference"), 0.0)
    )
    velocity = abs(
        _finite_float(scenario_vector.get("velocity_difference"), 0.0)
    )
    heading = abs(
        _finite_float(scenario_vector.get("heading_difference"), math.pi)
    )
    aspect = abs(
        _finite_float(
            scenario_vector.get("approximate_aspect_angle"), math.pi
        )
    )
    distance_penalty = _clip01(abs(distance - 12000.0) / 6000.0)
    formation_penalty = _clip01(
        abs(own_spread - 3000.0) / 3000.0
        + max(opponent_spread - own_spread, 0.0) / 5000.0
    )
    altitude_penalty = _clip01(altitude / 2500.0)
    velocity_penalty = _clip01(velocity / 80.0)
    heading_penalty = _clip01(
        min(abs(heading - math.pi), abs(aspect - math.pi)) / math.pi
    )
    current = _finite_float(policy_context.get("W_current"), 0.0)
    best = _finite_float(policy_context.get("W_best"), 0.0)
    return {
        "initial_disadvantage_proxy": _clip01(
            0.25 * distance_penalty
            + 0.20 * formation_penalty
            + 0.20 * altitude_penalty
            + 0.15 * velocity_penalty
            + 0.20 * heading_penalty
        ),
        "distance_disadvantage": distance_penalty,
        "heading_disadvantage": heading_penalty,
        "altitude_disadvantage": altitude_penalty,
        "velocity_disadvantage": velocity_penalty,
        "pool_novelty_score": _clip01(
            _finite_float(policy_context.get("pool_novelty_score"), 0.0)
        ),
        "policy_performance_drop_proxy": _clip01(best - current),
    }


def _normalized_vector_distance(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    squared = []
    for factor, (low, high) in FACTOR_RANGES.items():
        span = max(high - low, 1e-8)
        left_value = _finite_float(left.get(factor), low)
        right_value = _finite_float(right.get(factor), low)
        squared.append(((left_value - right_value) / span) ** 2)
    return math.sqrt(sum(squared) / max(len(squared), 1))


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _optional_clip01(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return _clip01(value)


def _proposal_diversity_scores(
    candidates: Sequence[Mapping[str, Any]],
) -> List[float]:
    if len(candidates) < 2:
        return [0.0 for _candidate in candidates]
    raw = []
    for index, candidate in enumerate(candidates):
        distances = [
            _normalized_vector_distance(
                candidate.get("scenario_vector") or {},
                other.get("scenario_vector") or {},
            )
            for other_index, other in enumerate(candidates)
            if other_index != index
        ]
        raw.append(sum(distances) / len(distances) if distances else 0.0)
    minimum = min(raw)
    maximum = max(raw)
    span = maximum - minimum
    if span <= 1e-12:
        return [0.0 for _value in raw]
    return [(value - minimum) / span for value in raw]


def _hardness_components(
    candidate: Mapping[str, Any], pool_stats: Mapping[str, Any]
) -> Dict[str, float]:
    vector = candidate.get("scenario_vector") or {}
    proxy = _proxy_features(vector, {})
    own_spread = _finite_float(vector.get("own_formation_spread"), 1000.0)
    opponent_spread = _finite_float(
        vector.get("opponent_formation_spread"), 1000.0
    )
    formation_span = FACTOR_RANGES["own_formation_spread"][1] - FACTOR_RANGES[
        "own_formation_spread"
    ][0]
    formation_stress = _clip01(
        0.6 * (own_spread - 1000.0) / formation_span
        + 0.4 * (8000.0 - opponent_spread) / formation_span
    )
    heading = _finite_float(vector.get("heading_difference"), math.pi)
    aspect = _finite_float(vector.get("approximate_aspect_angle"), math.pi)
    heading_aspect_stress = _clip01(
        0.5 * abs(heading - math.pi) / math.pi
        + 0.5 * abs(aspect - math.pi) / math.pi
    )
    altitude_velocity_stress = _clip01(
        0.5
        * abs(_finite_float(vector.get("altitude_difference"), 0.0))
        / 2500.0
        + 0.5
        * abs(_finite_float(vector.get("velocity_difference"), 0.0))
        / 60.0
    )
    pool_novelty = _pool_novelty(vector, pool_stats)
    ambiguity = _target_assignment_ambiguity(
        candidate.get("initial_config") or {}
    )
    return {
        "initial_disadvantage_proxy": proxy["initial_disadvantage_proxy"],
        "formation_stress": formation_stress,
        "heading_aspect_stress": heading_aspect_stress,
        "altitude_velocity_stress": altitude_velocity_stress,
        "pool_novelty": pool_novelty,
        "target_assignment_ambiguity_proxy": ambiguity,
    }


def _pool_novelty(
    vector: Mapping[str, Any], pool_stats: Mapping[str, Any]
) -> float:
    vectors = list(
        pool_stats.get("scenario_vectors")
        or pool_stats.get("vectors")
        or []
    )
    distances = [
        _normalized_vector_distance(vector, other)
        for other in vectors
        if isinstance(other, Mapping)
    ]
    return _clip01(min(distances)) if distances else 0.5


def _target_assignment_ambiguity(initial_config: Mapping[str, Any]) -> float:
    agents = {
        str(agent.get("agent_id")): agent
        for agent in initial_config.get("agents") or []
    }
    own_ids = list(initial_config.get("own_ids") or [])
    opponent_ids = list(initial_config.get("opponent_ids") or [])
    if len(own_ids) < 2 or len(opponent_ids) < 2:
        return 0.0
    scores = []
    for own_id in own_ids:
        own = agents.get(str(own_id), {})
        own_position = own.get("position_neu")
        if not isinstance(own_position, Sequence):
            continue
        distances = []
        for opponent_id in opponent_ids:
            opponent = agents.get(str(opponent_id), {})
            position = opponent.get("position_neu")
            if not isinstance(position, Sequence):
                continue
            distances.append(
                math.sqrt(
                    sum(
                        (
                            _finite_float(own_position[index], 0.0)
                            - _finite_float(position[index], 0.0)
                        )
                        ** 2
                        for index in range(min(len(own_position), len(position)))
                    )
                )
            )
        if len(distances) >= 2:
            scores.append(
                1.0
                - abs(distances[0] - distances[1])
                / max(distances[0] + distances[1], 1e-8)
            )
    return _clip01(sum(scores) / len(scores)) if scores else 0.0
