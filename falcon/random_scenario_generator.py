"""Random baseline scenario generator for testing the FALCON pipeline."""

from __future__ import annotations

import random
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .candidate_schema import create_candidate_scenario
from .scenario_adapter import (
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    scenario_vector_to_initial_config,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 7,
    "team_center_distance_range": [7000.0, 17000.0],
    "own_formation_spread_range": [1200.0, 6000.0],
    "opponent_formation_spread_range": [1200.0, 6000.0],
    "altitude_difference_range": [-1200.0, 1200.0],
    "velocity_difference_range": [-30.0, 30.0],
    "heading_difference_range": [2.2, 4.1],
    "approximate_aspect_angle_range": [0.0, 1.2],
}


class RandomScenarioGenerator:
    """Generate legal-ish random candidates without Qwen/FSN dependencies."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))
        self.rng = random.Random(self.config.get("seed"))

    def generate_from_base(self, base_config: Mapping[str, Any], num_scenarios: int) -> List[Dict[str, Any]]:
        initial_config = extract_initial_config_from_yaml(base_config)
        base_vector = initial_config_to_scenario_vector(initial_config)["scenario_vector"]
        return [
            self._generate_candidate(
                base_initial_config=initial_config,
                base_vector=base_vector,
                index=idx,
                target_failure_modes=[],
                source_failure_id=None,
            )
            for idx in range(num_scenarios)
        ]

    def generate_from_failure_summary(
        self,
        failure_summary: Mapping[str, Any],
        base_config: Mapping[str, Any],
        num_scenarios: int,
    ) -> List[Dict[str, Any]]:
        initial_config = extract_initial_config_from_yaml(base_config)
        base_vector = initial_config_to_scenario_vector(initial_config)["scenario_vector"]
        target_modes = list(failure_summary.get("primary_failure_modes") or failure_summary.get("secondary_failure_modes") or [])
        source_failure_id = failure_summary.get("source_trajectory") or failure_summary.get("episode_id")
        return [
            self._generate_candidate(
                base_initial_config=initial_config,
                base_vector=base_vector,
                index=idx,
                target_failure_modes=target_modes,
                source_failure_id=source_failure_id,
            )
            for idx in range(num_scenarios)
        ]

    def _generate_candidate(
        self,
        base_initial_config: Mapping[str, Any],
        base_vector: Mapping[str, Any],
        index: int,
        target_failure_modes: Sequence[str],
        source_failure_id: Optional[str],
    ) -> Dict[str, Any]:
        vector = dict(base_vector)
        changed_factors = [
            "team_center_distance",
            "own_formation_spread",
            "opponent_formation_spread",
            "altitude_difference",
            "velocity_difference",
            "heading_difference",
            "approximate_aspect_angle",
        ]
        for factor in changed_factors:
            low, high = self.config[f"{factor}_range"]
            vector[factor] = self.rng.uniform(float(low), float(high))
        vector["own_center_x"] = base_vector.get("own_center_x")
        vector["own_center_y"] = base_vector.get("own_center_y")
        vector["own_center_z"] = base_vector.get("own_center_z")
        vector["opponent_center_x"] = None
        vector["opponent_center_y"] = None
        vector["opponent_center_z"] = None
        initial_config = scenario_vector_to_initial_config(vector, base_initial_config)
        recomputed = initial_config_to_scenario_vector(initial_config)["scenario_vector"]
        scenario_id = f"random_{index:04d}"
        return create_candidate_scenario(
            scenario_id=scenario_id,
            generator_type="random",
            source_failure_id=source_failure_id,
            target_failure_modes=target_failure_modes,
            changed_factors=changed_factors,
            counterfactual_group_id=source_failure_id,
            scenario_vector=recomputed,
            scenario_parameters={"requested_scenario_vector": vector},
            initial_config=initial_config,
            expected_effect="randomized curriculum probe",
            rationale="Random baseline perturbation of 2v2 initial geometry.",
            metadata={"generator_config": dict(self.config)},
        )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
