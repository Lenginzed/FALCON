"""Constraint checks for generated FALCON candidate scenarios."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .scenario_adapter import (
    apply_initial_config_to_yaml,
    extract_initial_config_from_yaml,
    initial_config_to_scenario_vector,
    project_scenario_vector_to_legal_range,
    scenario_vector_to_initial_config,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "altitude_min_m": 3000.0,
    "altitude_max_m": 9000.0,
    "velocity_min_mps": 180.0,
    "velocity_max_mps": 300.0,
    "max_local_distance_m": 30000.0,
    "team_center_distance_min_m": 6000.0,
    "team_center_distance_max_m": 18000.0,
    "formation_spread_min_m": 1000.0,
    "formation_spread_max_m": 8000.0,
    "minimum_separation_m": 500.0,
    "enable_env_load_check": False,
}


class ConstraintChecker:
    """Validate generated scenarios before policy evaluation or curriculum use."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))

    def validate_candidate(
        self,
        candidate_scenario: Mapping[str, Any],
        enable_env_load_check: Optional[bool] = None,
    ) -> Dict[str, Any]:
        scenario_id = str(candidate_scenario.get("scenario_id", "unknown"))
        initial_config = candidate_scenario.get("initial_config")
        if not isinstance(initial_config, MappingABC):
            initial_config = None
        scenario_vector = candidate_scenario.get("scenario_vector") if isinstance(candidate_scenario.get("scenario_vector"), MappingABC) else {}
        result = self._validate(scenario_id, initial_config, scenario_vector)
        if self._env_check_enabled(enable_env_load_check):
            result["physical_constraint_check"]["scenario_loadable_env_check"] = False
            result["warnings"].append("scenario_loadable_env_check requires a YAML config; use validate_yaml_config for env smoke checks.")
            self._refresh_validity(result)
        return result

    def validate_yaml_config(
        self,
        yaml_config: Mapping[str, Any],
        enable_env_load_check: Optional[bool] = None,
        temp_config_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        scenario_id = str(yaml_config.get("scenario_id", "yaml_config"))
        initial_config = extract_initial_config_from_yaml(yaml_config)
        scenario_vector = initial_config_to_scenario_vector(initial_config)["scenario_vector"]
        result = self._validate(scenario_id, initial_config, scenario_vector)
        result["physical_constraint_check"]["scenario_loadable"] = self._yaml_structure_loadable(yaml_config)
        if self._env_check_enabled(enable_env_load_check):
            env_ok, env_warnings = self._yaml_env_loadable(yaml_config, temp_config_name=temp_config_name)
            result["physical_constraint_check"]["scenario_loadable_env_check"] = env_ok
            result["warnings"].extend(env_warnings)
        self._refresh_validity(result)
        return result

    def validate_batch(self, candidate_scenarios: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [self.validate_candidate(candidate) for candidate in candidate_scenarios]

    def repair_candidate_for_yaml(
        self,
        candidate_scenario: Mapping[str, Any],
        base_yaml_config: Mapping[str, Any],
        max_repair_attempts: int = 3,
        repair_margin_m: float = 5.0,
        enable_env_load_check: bool = False,
    ) -> Dict[str, Any]:
        """Materialize, validate, and project a candidate until YAML-valid."""
        working = copy.deepcopy(dict(candidate_scenario))
        base_initial = extract_initial_config_from_yaml(base_yaml_config)
        history: List[Dict[str, Any]] = []
        repair_actions: List[Dict[str, Any]] = []
        final_yaml: Dict[str, Any] = {}
        final_result: Dict[str, Any] = {}
        attempts = max(int(max_repair_attempts), 0)
        for attempt in range(attempts + 1):
            final_yaml = apply_initial_config_to_yaml(
                base_yaml_config, working.get("initial_config") or {}
            )
            final_yaml["scenario_id"] = working.get("scenario_id")
            post_initial = extract_initial_config_from_yaml(final_yaml)
            post_vector = initial_config_to_scenario_vector(post_initial)[
                "scenario_vector"
            ]
            final_result = self.validate_yaml_config(
                final_yaml,
                enable_env_load_check=enable_env_load_check,
                temp_config_name=f"fsn_repair_{working.get('scenario_id')}_{attempt}",
            )
            history.append(
                {
                    "attempt": attempt,
                    "is_valid": bool(final_result.get("is_valid")),
                    "rejection_reasons": list(
                        final_result.get("rejection_reasons") or []
                    ),
                    "scenario_vector": post_vector,
                    "roundtrip_error": _vector_error(
                        working.get("scenario_vector") or {}, post_vector
                    ),
                }
            )
            working["initial_config"] = post_initial
            working["scenario_vector"] = post_vector
            if final_result.get("is_valid"):
                break
            if attempt >= attempts:
                continue
            source_vector = (
                (working.get("scenario_parameters") or {}).get(
                    "legalized_request"
                )
                or working.get("scenario_vector")
                or {}
            )
            projected = project_scenario_vector_to_legal_range(
                source_vector,
                self.config,
                margin_m=float(repair_margin_m) * (attempt + 1),
            )
            actions = list(projected.pop("repair_actions", []))
            repair_actions.extend(
                [{"attempt": attempt + 1, **action} for action in actions]
            )
            working["initial_config"] = scenario_vector_to_initial_config(
                projected, base_initial
            )
            working["scenario_vector"] = initial_config_to_scenario_vector(
                working["initial_config"]
            )["scenario_vector"]
            working.setdefault("scenario_parameters", {})[
                "adapter_repaired_request"
            ] = projected
        metadata = working.setdefault("metadata", {})
        metadata["adapter_repair_applied"] = bool(repair_actions)
        metadata["repair_actions"] = repair_actions
        metadata["repair_attempts"] = max(len(history) - 1, 0)
        metadata["post_yaml_constraint_valid"] = bool(
            final_result.get("is_valid")
        )
        return {
            "schema_version": "falcon.candidate_yaml_repair.v1",
            "candidate": working,
            "yaml_config": final_yaml,
            "constraint_result": final_result,
            "is_valid": bool(final_result.get("is_valid")),
            "repair_success": bool(
                final_result.get("is_valid") and repair_actions
            ),
            "repair_actions": repair_actions,
            "history": history,
            "warnings": list(final_result.get("warnings") or []),
        }

    def _validate(
        self,
        scenario_id: str,
        initial_config: Optional[Mapping[str, Any]],
        scenario_vector: Mapping[str, Any],
    ) -> Dict[str, Any]:
        missing_fields: List[str] = []
        warnings: List[str] = []
        agents = initial_config.get("agents") if isinstance(initial_config, MappingABC) else []
        if not agents:
            missing_fields.append("initial_config.agents")
            warnings.append("Agent-level physical checks are limited because initial_config is missing.")
        vector = dict(scenario_vector or {})

        physical = {
            "altitude_valid": self._altitude_valid(agents, missing_fields),
            "velocity_valid": self._velocity_valid(agents, missing_fields),
            "heading_valid": self._heading_valid(agents, missing_fields),
            "position_valid": self._position_valid(agents, missing_fields),
            "minimum_separation_valid": self._minimum_separation_valid(agents, missing_fields),
            "scenario_loadable": self._initial_config_structure_loadable(initial_config),
            "scenario_loadable_env_check": None,
        }
        task = {
            "team_center_distance_valid": self._range_check(
                vector.get("team_center_distance"),
                self.config["team_center_distance_min_m"],
                self.config["team_center_distance_max_m"],
                "scenario_vector.team_center_distance",
                missing_fields,
            ),
            "formation_spread_valid": self._formation_spread_valid(vector, missing_fields),
        }
        result = {
            "schema_version": "1.0",
            "scenario_id": scenario_id,
            "is_valid": True,
            "validity_score": 0.0,
            "rejection_reasons": [],
            "physical_constraint_check": physical,
            "task_constraint_check": task,
            "missing_fields": sorted(set(missing_fields)),
            "warnings": warnings,
        }
        self._refresh_validity(result)
        return result

    def _refresh_validity(self, result: Dict[str, Any]) -> None:
        checks = {}
        checks.update(result.get("physical_constraint_check") or {})
        checks.update(result.get("task_constraint_check") or {})
        bool_checks = {key: value for key, value in checks.items() if isinstance(value, bool)}
        rejection_reasons = [key for key, value in bool_checks.items() if value is False]
        result["rejection_reasons"] = sorted(set(rejection_reasons))
        result["validity_score"] = round(sum(1 for value in bool_checks.values() if value is True) / max(len(bool_checks), 1), 6)
        result["is_valid"] = len(rejection_reasons) == 0 and not result.get("missing_fields")

    def _altitude_valid(self, agents: Sequence[Mapping[str, Any]], missing_fields: List[str]) -> bool:
        if not agents:
            return False
        valid = True
        for agent in agents:
            altitude = _value(agent, "altitude", "initial_altitude")
            if altitude is None:
                missing_fields.append(f"initial_config.agents.{agent.get('agent_id', '?')}.altitude")
                valid = False
            elif not self.config["altitude_min_m"] <= altitude <= self.config["altitude_max_m"]:
                valid = False
        return valid

    def _velocity_valid(self, agents: Sequence[Mapping[str, Any]], missing_fields: List[str]) -> bool:
        if not agents:
            return False
        valid = True
        for agent in agents:
            velocity = _value(agent, "velocity")
            if velocity is None:
                velocity_vec = _vector(agent.get("velocity_neu") or agent.get("initial_velocity"))
                velocity = _norm(velocity_vec) if velocity_vec is not None else None
            if velocity is None:
                missing_fields.append(f"initial_config.agents.{agent.get('agent_id', '?')}.velocity")
                valid = False
            elif not self.config["velocity_min_mps"] <= velocity <= self.config["velocity_max_mps"]:
                valid = False
        return valid

    def _heading_valid(self, agents: Sequence[Mapping[str, Any]], missing_fields: List[str]) -> bool:
        if not agents:
            return False
        valid = True
        for agent in agents:
            heading = _value(agent, "heading", "initial_heading")
            heading_deg = _value(agent, "heading_deg")
            if heading is None and heading_deg is None:
                missing_fields.append(f"initial_config.agents.{agent.get('agent_id', '?')}.heading")
                valid = False
            elif heading_deg is not None:
                valid = valid and 0.0 <= heading_deg < 360.0
            else:
                valid = valid and 0.0 <= heading < 2.0 * math.pi
        return valid

    def _position_valid(self, agents: Sequence[Mapping[str, Any]], missing_fields: List[str]) -> bool:
        if not agents:
            return False
        valid = True
        for agent in agents:
            pos = _vector(agent.get("position_neu") or agent.get("initial_position_neu"))
            lat = _value(agent, "latitude", "initial_latitude")
            lon = _value(agent, "longitude", "initial_longitude")
            if pos is None and (lat is None or lon is None):
                missing_fields.append(f"initial_config.agents.{agent.get('agent_id', '?')}.position")
                valid = False
                continue
            if pos is not None:
                finite = all(_is_finite(value) for value in pos)
                in_range = _norm(pos[:2]) <= self.config["max_local_distance_m"]
                valid = valid and finite and in_range
            if lat is not None and lon is not None:
                valid = valid and _is_finite(lat) and _is_finite(lon)
        return valid

    def _minimum_separation_valid(self, agents: Sequence[Mapping[str, Any]], missing_fields: List[str]) -> bool:
        positions = []
        for agent in agents:
            pos = _vector(agent.get("position_neu") or agent.get("initial_position_neu"))
            if pos is not None:
                positions.append((agent.get("agent_id"), pos))
        if len(positions) < 2:
            missing_fields.append("initial_config.agents.position_neu")
            return False
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if _norm(_sub(positions[i][1], positions[j][1])) < self.config["minimum_separation_m"]:
                    return False
        return True

    def _formation_spread_valid(self, scenario_vector: Mapping[str, Any], missing_fields: List[str]) -> bool:
        own = scenario_vector.get("own_formation_spread")
        opponent = scenario_vector.get("opponent_formation_spread")
        valid = True
        for key, value in (("own_formation_spread", own), ("opponent_formation_spread", opponent)):
            if value is None:
                missing_fields.append(f"scenario_vector.{key}")
                valid = False
            elif not self.config["formation_spread_min_m"] <= _float(value) <= self.config["formation_spread_max_m"]:
                valid = False
        return valid

    def _range_check(self, value: Any, min_value: float, max_value: float, field: str, missing_fields: List[str]) -> bool:
        if value is None:
            missing_fields.append(field)
            return False
        return min_value <= _float(value) <= max_value

    def _initial_config_structure_loadable(self, initial_config: Optional[Mapping[str, Any]]) -> bool:
        agents = initial_config.get("agents") if isinstance(initial_config, MappingABC) else None
        return isinstance(agents, list) and len(agents) >= 4

    def _yaml_structure_loadable(self, yaml_config: Mapping[str, Any]) -> bool:
        aircraft_configs = yaml_config.get("aircraft_configs")
        if not isinstance(aircraft_configs, MappingABC) or len(aircraft_configs) < 4:
            return False
        required = {"ic_long_gc_deg", "ic_lat_geod_deg", "ic_h_sl_ft", "ic_psi_true_deg", "ic_u_fps"}
        for aircraft in aircraft_configs.values():
            init_state = (aircraft or {}).get("init_state") if isinstance(aircraft, MappingABC) else None
            if not isinstance(init_state, MappingABC) or not required <= set(init_state.keys()):
                return False
        return True

    def _env_check_enabled(self, override: Optional[bool]) -> bool:
        return bool(self.config.get("enable_env_load_check", False) if override is None else override)

    def _yaml_env_loadable(self, yaml_config: Mapping[str, Any], temp_config_name: Optional[str] = None) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        env = None
        original_parse_config = None
        env_name = temp_config_name or "falcon_memory_env_check"
        try:
            from envs.JSBSim.envs import env_base
            from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

            original_parse_config = env_base.parse_config
            config_copy = dict(yaml_config)

            def _parse_config_override(_filename):
                return type("EnvConfig", (object,), config_copy)

            env_base.parse_config = _parse_config_override
            env = MultipleCombatEnv(env_name)
            env.reset()
            return True, warnings
        except Exception as exc:  # noqa: BLE001 - optional smoke check must not crash callers
            warnings.append(f"scenario_loadable_env_check failed for {env_name}: {exc}")
            return False, warnings
        finally:
            if original_parse_config is not None:
                try:
                    from envs.JSBSim.envs import env_base

                    env_base.parse_config = original_parse_config
                except Exception:
                    pass
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass


def _value(data: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if data.get(key) is not None:
            return _float(data.get(key))
    return None


def _vector(value: Any) -> Optional[List[float]]:
    if isinstance(value, MappingABC):
        return [_float(value.get("north", value.get("x"))), _float(value.get("east", value.get("y"))), _float(value.get("up", value.get("z")))]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return [_float(value[0]), _float(value[1]), _float(value[2])]
    return None


def _sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_float(a[i]) - _float(b[i]) for i in range(min(len(a), len(b)))]


def _norm(value: Optional[Sequence[float]]) -> float:
    if value is None:
        return 0.0
    return math.sqrt(sum(_float(item) ** 2 for item in value))


def _is_finite(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(value) or math.isinf(value))


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _vector_error(
    requested: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in (
        "team_center_distance",
        "own_formation_spread",
        "opponent_formation_spread",
        "altitude_difference",
        "velocity_difference",
        "heading_difference",
        "approximate_aspect_angle",
    ):
        if requested.get(key) is None or recomputed.get(key) is None:
            continue
        result[key] = _float(recomputed.get(key)) - _float(requested.get(key))
    return result
