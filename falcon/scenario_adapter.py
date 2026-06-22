"""Adapters between FALCON scenario objects and LAG 2v2 YAML configs."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .trajectory_recorder import SCENARIO_VECTOR_KEYS, compute_scenario_vector, load_trajectory

SCENARIO_ADAPTER_SCHEMA_VERSION = "falcon.scenario_adapter.v1"
EARTH_RADIUS_M = 6371000.0
FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
FPS_TO_MPS = 0.3048
MPS_TO_FPS = 1.0 / FPS_TO_MPS


def load_base_scenario_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Read a LAG 2v2 YAML scenario config."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, MappingABC):
        raise ValueError(f"Scenario YAML did not contain a mapping: {config_path}")
    return dict(config)


def extract_initial_config_from_yaml(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract standardized initial state for the four LAG aircraft."""
    aircraft_configs = config.get("aircraft_configs") or {}
    origin = _origin(config)
    agents: List[Dict[str, Any]] = []
    for agent_id, aircraft in aircraft_configs.items():
        init_state = dict((aircraft or {}).get("init_state") or {})
        lon = _float(init_state.get("ic_long_gc_deg"), origin[0])
        lat = _float(init_state.get("ic_lat_geod_deg"), origin[1])
        altitude_m = _float(init_state.get("ic_h_sl_ft"), origin[2] * M_TO_FT) * FT_TO_M
        heading_deg = _float(init_state.get("ic_psi_true_deg"), 0.0) % 360.0
        velocity_mps = _float(init_state.get("ic_u_fps"), 0.0) * FPS_TO_MPS
        position_neu = latlon_to_local_neu(lat, lon, altitude_m, origin_lat=origin[1], origin_lon=origin[0], origin_alt=origin[2])
        color = str((aircraft or {}).get("color") or "").lower()
        team = color or ("red" if str(agent_id).startswith("A") else "blue")
        agents.append(
            {
                "agent_id": str(agent_id),
                "team": team,
                "team_id": str(agent_id)[0] if agent_id else None,
                "latitude": lat,
                "longitude": lon,
                "altitude": altitude_m,
                "position_neu": position_neu,
                "heading": math.radians(heading_deg),
                "heading_deg": heading_deg,
                "velocity": velocity_mps,
                "velocity_neu": _heading_speed_to_neu(heading_deg, velocity_mps),
                "roll": _maybe_deg_or_rad(init_state.get("ic_phi_deg"), init_state.get("ic_phi_rad")),
                "pitch": _maybe_deg_or_rad(init_state.get("ic_theta_deg"), init_state.get("ic_theta_rad")),
                "alive": True,
                "blood": 100.0,
                "missile_count": (aircraft or {}).get("missile", 0),
                "raw_init_state": init_state,
            }
        )
    own_ids, opponent_ids = _split_agent_ids([agent["agent_id"] for agent in agents])
    return {
        "schema_version": "falcon.initial_config.v1",
        "origin": {"longitude": origin[0], "latitude": origin[1], "altitude": origin[2]},
        "agents": agents,
        "own_ids": own_ids,
        "opponent_ids": opponent_ids,
        "warnings": [],
    }


def initial_config_to_scenario_vector(initial_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert initial_config into FALCON scenario_vector plus warnings."""
    agents = initial_config.get("agents") or []
    vector, missing_fields = compute_scenario_vector(
        agents,
        own_ids=initial_config.get("own_ids"),
        opponent_ids=initial_config.get("opponent_ids"),
    )
    warnings = []
    if missing_fields:
        warnings.append("Some scenario_vector fields could not be computed from initial_config.")
    return {
        "schema_version": "falcon.scenario_vector_result.v1",
        "scenario_vector": vector,
        "missing_fields": missing_fields,
        "warnings": warnings,
    }


def scenario_vector_to_initial_config(
    scenario_vector: Mapping[str, Any],
    base_initial_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Generate a four-aircraft initial_config from a scenario_vector.

    The geometry is a first-version local approximation, not an exact inverse
    of every possible 2v2 configuration. It preserves the key curriculum
    factors while keeping a valid LAG-compatible layout.
    """
    base_agents = [dict(agent) for agent in base_initial_config.get("agents", [])]
    if len(base_agents) < 4:
        raise ValueError("base_initial_config must contain at least four agents")
    own_ids = list(base_initial_config.get("own_ids") or [agent["agent_id"] for agent in base_agents[:2]])
    opponent_ids = list(base_initial_config.get("opponent_ids") or [agent["agent_id"] for agent in base_agents[2:4]])
    by_id = {agent["agent_id"]: agent for agent in base_agents}
    origin = _origin_from_initial_config(base_initial_config)
    base_vector = initial_config_to_scenario_vector(base_initial_config)["scenario_vector"]

    own_center = _vector3_from_keys(scenario_vector, "own_center") or _vector3_from_keys(base_vector, "own_center") or _team_center(by_id, own_ids)
    center_distance = _value(scenario_vector, "team_center_distance", _value(base_vector, "team_center_distance", 10000.0))
    altitude_difference = _value(scenario_vector, "altitude_difference", _value(base_vector, "altitude_difference", 0.0))
    aspect_angle = _value(scenario_vector, "approximate_aspect_angle", 0.0)
    opponent_center = _vector3_from_keys(scenario_vector, "opponent_center")
    if opponent_center is None:
        opponent_center = [
            own_center[0] + center_distance * math.cos(aspect_angle),
            own_center[1] + center_distance * math.sin(aspect_angle),
            own_center[2] + altitude_difference,
        ]
    own_center[2] = _value(scenario_vector, "own_center_z", own_center[2])
    opponent_center[2] = _value(scenario_vector, "opponent_center_z", own_center[2] + altitude_difference)

    own_spread = _value(scenario_vector, "own_formation_spread", _value(base_vector, "own_formation_spread", 1000.0))
    opponent_spread = _value(scenario_vector, "opponent_formation_spread", _value(base_vector, "opponent_formation_spread", 1000.0))
    velocity_difference = _value(scenario_vector, "velocity_difference", _value(base_vector, "velocity_difference", 0.0))
    heading_difference = _value(scenario_vector, "heading_difference", _value(base_vector, "heading_difference", math.pi))

    own_speed = _mean([_float(by_id[agent_id].get("velocity"), 240.0) for agent_id in own_ids])
    opponent_speed = max(0.0, own_speed + velocity_difference)
    own_heading = _mean_angle([_float(by_id[agent_id].get("heading"), 0.0) for agent_id in own_ids])
    opponent_heading = (own_heading + heading_difference) % (2.0 * math.pi)

    placement = {
        own_ids[0]: _offset_position(own_center, -own_spread / 2.0),
        own_ids[1]: _offset_position(own_center, own_spread / 2.0),
        opponent_ids[0]: _offset_position(opponent_center, -opponent_spread / 2.0),
        opponent_ids[1]: _offset_position(opponent_center, opponent_spread / 2.0),
    }
    generated_agents = []
    for agent in base_agents:
        agent_id = agent["agent_id"]
        pos = placement.get(agent_id, _position(agent))
        lat, lon, alt = local_neu_to_latlon(pos[0], pos[1], pos[2], origin_lat=origin[1], origin_lon=origin[0], origin_alt=origin[2])
        is_own = agent_id in own_ids
        heading = own_heading if is_own else opponent_heading
        speed = own_speed if is_own else opponent_speed
        raw = dict(agent.get("raw_init_state") or {})
        raw.update(
            {
                "ic_long_gc_deg": lon,
                "ic_lat_geod_deg": lat,
                "ic_h_sl_ft": alt * M_TO_FT,
                "ic_psi_true_deg": math.degrees(heading) % 360.0,
                "ic_u_fps": speed * MPS_TO_FPS,
            }
        )
        generated = dict(agent)
        generated.update(
            {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "position_neu": pos,
                "heading": heading,
                "heading_deg": math.degrees(heading) % 360.0,
                "velocity": speed,
                "velocity_neu": _heading_speed_to_neu(math.degrees(heading), speed),
                "raw_init_state": raw,
            }
        )
        generated_agents.append(generated)
    return {
        "schema_version": "falcon.initial_config.v1",
        "origin": {"longitude": origin[0], "latitude": origin[1], "altitude": origin[2]},
        "agents": generated_agents,
        "own_ids": own_ids,
        "opponent_ids": opponent_ids,
        "source_scenario_vector": {key: scenario_vector.get(key) for key in SCENARIO_VECTOR_KEYS},
        "warnings": ["Generated with local planar approximation; use for small-area 2v2 initial scenarios."],
    }


def project_scenario_vector_to_legal_range(
    scenario_vector: Mapping[str, Any],
    constraint_config: Optional[Mapping[str, Any]] = None,
    margin_m: float = 5.0,
) -> Dict[str, Any]:
    """Project a scenario vector into the interior of the legal YAML range.

    The interior margin avoids floating-point boundary failures after the
    local-plane -> latitude/longitude -> local-plane round trip.
    """
    config = {
        "altitude_min_m": 3000.0,
        "altitude_max_m": 9000.0,
        "velocity_min_mps": 180.0,
        "velocity_max_mps": 300.0,
        "team_center_distance_min_m": 6000.0,
        "team_center_distance_max_m": 18000.0,
        "formation_spread_min_m": 1000.0,
        "formation_spread_max_m": 8000.0,
        "minimum_separation_m": 500.0,
        **dict(constraint_config or {}),
    }
    projected = dict(scenario_vector)
    actions: List[Dict[str, Any]] = []
    margin = max(_float(margin_m), 0.0)

    def _clip_field(key: str, low: float, high: float, field_margin: float = 0.0) -> None:
        if projected.get(key) is None:
            return
        inner_low = low + field_margin
        inner_high = high - field_margin
        if inner_low > inner_high:
            inner_low, inner_high = low, high
        original = _float(projected.get(key))
        repaired = min(max(original, inner_low), inner_high)
        projected[key] = repaired
        if abs(repaired - original) > 1e-9:
            actions.append(
                {
                    "action": "clip",
                    "field": key,
                    "before": original,
                    "after": repaired,
                    "legal_range": [low, high],
                    "interior_margin": field_margin,
                }
            )

    _clip_field(
        "team_center_distance",
        _float(config["team_center_distance_min_m"]),
        _float(config["team_center_distance_max_m"]),
        margin,
    )
    for key in ("own_formation_spread", "opponent_formation_spread"):
        _clip_field(
            key,
            max(
                _float(config["formation_spread_min_m"]),
                _float(config["minimum_separation_m"]),
            ),
            _float(config["formation_spread_max_m"]),
            margin,
        )
    _clip_field("altitude_difference", -2500.0, 2500.0)
    _clip_field("velocity_difference", -60.0, 60.0)
    _clip_field("heading_difference", 0.0, 2.0 * math.pi)
    _clip_field("approximate_aspect_angle", 0.0, 2.0 * math.pi)

    own_z = projected.get("own_center_z")
    altitude_difference = projected.get("altitude_difference")
    if own_z is not None:
        _clip_field(
            "own_center_z",
            _float(config["altitude_min_m"]) + margin,
            _float(config["altitude_max_m"]) - margin,
        )
        own_z = _float(projected.get("own_center_z"))
        if altitude_difference is not None:
            low = _float(config["altitude_min_m"]) + margin - own_z
            high = _float(config["altitude_max_m"]) - margin - own_z
            _clip_field("altitude_difference", low, high)
    projected["repair_actions"] = actions
    return projected


def apply_initial_config_to_yaml(base_config: Mapping[str, Any], initial_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new YAML config mapping with initial_config written into it."""
    config = copy.deepcopy(dict(base_config))
    aircraft_configs = config.setdefault("aircraft_configs", {})
    agents = initial_config.get("agents") or []
    for agent in agents:
        agent_id = agent.get("agent_id")
        if agent_id not in aircraft_configs:
            aircraft_configs[agent_id] = {"color": _team_color(agent.get("team")), "model": "f16", "init_state": {}}
        aircraft = aircraft_configs[agent_id]
        aircraft.setdefault("init_state", {})
        raw = dict(agent.get("raw_init_state") or {})
        raw.update(
            {
                "ic_long_gc_deg": _float(agent.get("longitude", agent.get("initial_longitude"))),
                "ic_lat_geod_deg": _float(agent.get("latitude", agent.get("initial_latitude"))),
                "ic_h_sl_ft": _float(agent.get("altitude", agent.get("initial_altitude"))) * M_TO_FT,
                "ic_psi_true_deg": math.degrees(_float(agent.get("heading", agent.get("initial_heading")))) % 360.0,
                "ic_u_fps": _float(agent.get("velocity", _norm(agent.get("initial_velocity")) if agent.get("initial_velocity") is not None else 0.0)) * MPS_TO_FPS,
            }
        )
        aircraft["init_state"] = raw
        if agent.get("missile_count") is not None and _float(agent.get("missile_count")) > 0:
            aircraft["missile"] = int(_float(agent.get("missile_count")))
    return config


def save_scenario_yaml(config: Mapping[str, Any], output_path: Union[str, Path]) -> None:
    """Save a generated scenario YAML without modifying the source file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False, allow_unicode=False)


def trajectory_to_replay_scenario(
    trajectory_path: Union[str, Path],
    output_yaml_path: Union[str, Path],
    base_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Create a replay YAML from a saved trajectory's initial_config."""
    trajectory = load_trajectory(trajectory_path)
    config_path = base_config_path or trajectory.get("config_path")
    if config_path is None:
        raw_config = ((trajectory.get("initial_config") or {}).get("raw_config") or {})
        base_config = raw_config if raw_config else None
    else:
        base_config = load_base_scenario_config(config_path)
    if base_config is None:
        raise ValueError("Could not determine base YAML config for replay scenario.")
    initial_config = trajectory.get("initial_config") or {}
    if "agents" not in initial_config and trajectory.get("frames"):
        from .trajectory_recorder import _initial_agents_from_frame  # local private fallback for old records

        initial_config = {
            "agents": _initial_agents_from_frame(trajectory["frames"][0], trajectory.get("env") or {}),
            "own_ids": (trajectory.get("env") or {}).get("ego_ids"),
            "opponent_ids": (trajectory.get("env") or {}).get("enm_ids"),
            "origin": _origin(base_config),
        }
    replay_config = apply_initial_config_to_yaml(base_config, initial_config)
    save_scenario_yaml(replay_config, output_yaml_path)
    return {
        "schema_version": "falcon.replay_scenario.v1",
        "source_trajectory": str(trajectory_path),
        "output_yaml_path": str(output_yaml_path),
        "config": replay_config,
    }


def latlon_to_local_neu(
    lat: float,
    lon: float,
    alt: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
) -> List[float]:
    """Approximate geodetic coordinates as local NEU meters.

    This uses a local equirectangular approximation and is intended only for
    small-area simulation scenario generation, not precise navigation.
    """
    lat_rad = math.radians(lat)
    origin_lat_rad = math.radians(origin_lat)
    north = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    east = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos((lat_rad + origin_lat_rad) / 2.0)
    up = alt - origin_alt
    return [north, east, up]


def local_neu_to_latlon(
    north: float,
    east: float,
    up: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
) -> Tuple[float, float, float]:
    """Inverse local equirectangular approximation for small LAG scenarios."""
    lat = origin_lat + math.degrees(north / EARTH_RADIUS_M)
    avg_lat_rad = math.radians((lat + origin_lat) / 2.0)
    lon = origin_lon + math.degrees(east / (EARTH_RADIUS_M * max(math.cos(avg_lat_rad), 1e-8)))
    alt = origin_alt + up
    return lat, lon, alt


def _origin(config: Mapping[str, Any]) -> List[float]:
    value = config.get("battle_field_center") or [120.0, 60.0, 0.0]
    return [_float(value[0], 120.0), _float(value[1], 60.0), _float(value[2], 0.0)]


def _origin_from_initial_config(initial_config: Mapping[str, Any]) -> List[float]:
    origin = initial_config.get("origin")
    if isinstance(origin, MappingABC):
        return [_float(origin.get("longitude"), 120.0), _float(origin.get("latitude"), 60.0), _float(origin.get("altitude"), 0.0)]
    if isinstance(origin, Sequence) and not isinstance(origin, (str, bytes)) and len(origin) >= 3:
        return [_float(origin[0], 120.0), _float(origin[1], 60.0), _float(origin[2], 0.0)]
    return [120.0, 60.0, 0.0]


def _split_agent_ids(agent_ids: Sequence[str]) -> Tuple[List[str], List[str]]:
    if not agent_ids:
        return [], []
    prefix = str(agent_ids[0])[0]
    own_ids = [agent_id for agent_id in agent_ids if str(agent_id).startswith(prefix)]
    opponent_ids = [agent_id for agent_id in agent_ids if agent_id not in own_ids]
    return own_ids, opponent_ids


def _team_center(by_id: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> List[float]:
    positions = [_position(by_id[agent_id]) for agent_id in ids if agent_id in by_id]
    return [sum(pos[i] for pos in positions) / len(positions) for i in range(3)]


def _position(agent: Mapping[str, Any]) -> List[float]:
    value = agent.get("position_neu") or agent.get("initial_position_neu") or [0.0, 0.0, _float(agent.get("altitude"), 6000.0)]
    return [_float(value[0]), _float(value[1]), _float(value[2])] if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [0.0, 0.0, 6000.0]


def _vector3_from_keys(data: Mapping[str, Any], prefix: str) -> Optional[List[float]]:
    keys = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    if all(data.get(key) is not None for key in keys):
        return [_float(data[key]) for key in keys]
    return None


def _offset_position(center: Sequence[float], east_offset: float) -> List[float]:
    return [_float(center[0]), _float(center[1]) + east_offset, _float(center[2])]


def _norm(value: Any) -> float:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return 0.0
    return math.sqrt(sum(_float(item) ** 2 for item in value))


def _heading_speed_to_neu(heading_deg: float, speed_mps: float) -> List[float]:
    heading = math.radians(heading_deg)
    return [speed_mps * math.cos(heading), speed_mps * math.sin(heading), 0.0]


def _maybe_deg_or_rad(deg_value: Any, rad_value: Any) -> Optional[float]:
    if rad_value is not None:
        return _float(rad_value)
    if deg_value is not None:
        return math.radians(_float(deg_value))
    return None


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_angle(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.atan2(_mean([math.sin(v) for v in values]), _mean([math.cos(v) for v in values]))


def _value(data: Mapping[str, Any], key: str, default: float) -> float:
    return default if data.get(key) is None else _float(data.get(key), default)


def _team_color(team: Any) -> str:
    value = str(team or "").lower()
    return "Blue" if value in {"blue", "b"} else "Red"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def dump_json(data: Mapping[str, Any], output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
