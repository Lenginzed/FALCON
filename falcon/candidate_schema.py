"""Canonical CandidateScenario helpers for FALCON scenario generators."""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .trajectory_recorder import SCENARIO_VECTOR_KEYS

CANDIDATE_SCHEMA_VERSION = "falcon.candidate_scenario.v1"
REQUIRED_FIELDS = (
    "schema_version",
    "scenario_id",
    "generator_type",
    "target_failure_modes",
    "changed_factors",
    "scenario_vector",
    "metadata",
)


def create_candidate_scenario(
    scenario_id: str,
    generator_type: str,
    scenario_vector: Mapping[str, Any],
    target_failure_modes: Optional[Sequence[str]] = None,
    source_failure_id: Optional[str] = None,
    changed_factors: Optional[Sequence[str]] = None,
    counterfactual_group_id: Optional[str] = None,
    scenario_parameters: Optional[Mapping[str, Any]] = None,
    initial_config: Optional[Mapping[str, Any]] = None,
    expected_effect: Optional[str] = None,
    rationale: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a normalized CandidateScenario dictionary."""
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "scenario_id": str(scenario_id),
        "generator_type": str(generator_type),
        "source_failure_id": source_failure_id,
        "target_failure_modes": list(target_failure_modes or []),
        "changed_factors": list(changed_factors or []),
        "counterfactual_group_id": counterfactual_group_id,
        "scenario_vector": {key: scenario_vector.get(key) for key in SCENARIO_VECTOR_KEYS},
        "scenario_parameters": dict(scenario_parameters or {}),
        "initial_config": dict(initial_config or {}) if initial_config is not None else None,
        "expected_effect": expected_effect,
        "rationale": rationale,
        "metadata": dict(metadata or {}),
    }


def validate_candidate_schema(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate CandidateScenario shape without raising on missing fields."""
    missing_fields: List[str] = []
    warnings: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in candidate:
            missing_fields.append(field)
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        warnings.append(f"Expected schema_version {CANDIDATE_SCHEMA_VERSION}, got {candidate.get('schema_version')!r}.")
    scenario_vector = candidate.get("scenario_vector")
    if not isinstance(scenario_vector, MappingABC):
        missing_fields.append("scenario_vector")
    else:
        for key in SCENARIO_VECTOR_KEYS:
            if key not in scenario_vector:
                missing_fields.append(f"scenario_vector.{key}")
    if not isinstance(candidate.get("target_failure_modes", []), list):
        warnings.append("target_failure_modes should be a list.")
    if not isinstance(candidate.get("changed_factors", []), list):
        warnings.append("changed_factors should be a list.")
    return {
        "schema_version": "falcon.candidate_validation.v1",
        "is_valid": len(missing_fields) == 0,
        "missing_fields": sorted(set(missing_fields)),
        "warnings": warnings,
    }


def candidate_to_json(candidate: Mapping[str, Any], output_path: Optional[Union[str, Path]] = None) -> str:
    """Serialize a candidate scenario to JSON, optionally saving it."""
    text = json.dumps(dict(candidate), indent=2, sort_keys=True)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(text)
    return text


def load_candidate_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a CandidateScenario JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data
