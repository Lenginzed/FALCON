"""Versioned episode trajectory recording helpers for FALCON."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA_VERSION = "falcon.multi_combat_trajectory.v2"

SCENARIO_VECTOR_KEYS = (
    "team_center_distance",
    "own_formation_spread",
    "opponent_formation_spread",
    "altitude_difference",
    "velocity_difference",
    "heading_difference",
    "approximate_aspect_angle",
    "own_center_x",
    "own_center_y",
    "own_center_z",
    "opponent_center_x",
    "opponent_center_y",
    "opponent_center_z",
)

REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "episode_id",
    "env_name",
    "scenario_type",
    "task_name",
    "config_path",
    "timestamp",
    "save_reason",
    "episode_result",
    "total_team_reward",
    "episode_length",
    "initial_config",
    "frames",
    "episode_summary",
)

REQUIRED_FRAME_FIELDS = (
    "timestep",
    "agents",
)

REQUIRED_AGENT_FIELDS = (
    "agent_id",
    "team",
    "position_neu",
    "latitude",
    "longitude",
    "altitude",
    "velocity_neu",
    "speed",
    "heading",
    "roll",
    "pitch",
    "action",
    "reward",
    "done",
    "info",
    "alive",
    "crash",
    "shotdown",
    "blood",
    "missile_count",
    "low_altitude",
    "overload",
    "extreme_state",
    "horizontal_distance_from_center_m",
)


def to_jsonable(value: Any) -> Any:
    """Convert numpy-heavy rollout objects into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, MappingABC):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_trajectory(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a trajectory JSON file without requiring a perfect schema."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "frames" not in data and "steps" in data:
        data = dict(data)
        data["frames"] = data.get("steps") or []
    return data


def validate_trajectory(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate required trajectory fields, returning warnings instead of raising."""
    missing_fields: List[str] = []
    warnings: List[str] = []

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            missing_fields.append(field)

    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        warnings.append(f"Expected schema_version {SCHEMA_VERSION}, got {schema_version!r}.")

    frames = _frames(data)
    if not isinstance(frames, list) or len(frames) == 0:
        missing_fields.append("frames[0]")
    else:
        for field in REQUIRED_FRAME_FIELDS:
            if field not in frames[0]:
                missing_fields.append(f"frames[0].{field}")
        agents = frames[0].get("agents") if isinstance(frames[0], MappingABC) else None
        if not isinstance(agents, MappingABC) or len(agents) == 0:
            missing_fields.append("frames[0].agents")
        else:
            first_agent_id = next(iter(agents.keys()))
            first_agent = agents[first_agent_id] or {}
            for field in REQUIRED_AGENT_FIELDS:
                if field not in first_agent:
                    missing_fields.append(f"frames[0].agents.{first_agent_id}.{field}")

    initial_config = data.get("initial_config")
    if isinstance(initial_config, MappingABC):
        if "agents" not in initial_config:
            missing_fields.append("initial_config.agents")
        if "scenario_vector" not in initial_config:
            missing_fields.append("initial_config.scenario_vector")
    elif "initial_config" in data:
        warnings.append("initial_config exists but is not a mapping.")

    return {
        "schema_version": "falcon.trajectory_validation.v1",
        "is_valid": len(missing_fields) == 0,
        "missing_fields": sorted(set(missing_fields)),
        "warnings": warnings,
    }


def extract_scenario_vector(data: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Return the scenario vector, deriving it from initial state if needed."""
    initial_config = data.get("initial_config") or {}
    if isinstance(initial_config, MappingABC):
        vector = initial_config.get("scenario_vector")
        if isinstance(vector, MappingABC):
            return {key: _nullable_float(vector.get(key)) for key in SCENARIO_VECTOR_KEYS}
        agents = initial_config.get("agents")
        env_meta = _env_meta(data)
        if isinstance(agents, list):
            computed, _missing = compute_scenario_vector(
                agents,
                own_ids=env_meta.get("ego_ids") or env_meta.get("own_ids"),
                opponent_ids=env_meta.get("enm_ids") or env_meta.get("opponent_ids"),
            )
            return computed

    frames = _frames(data)
    if frames:
        agents = _initial_agents_from_frame(frames[0], _env_meta(data))
        computed, _missing = compute_scenario_vector(
            agents,
            own_ids=(_env_meta(data).get("ego_ids") or _env_meta(data).get("own_ids")),
            opponent_ids=(_env_meta(data).get("enm_ids") or _env_meta(data).get("opponent_ids")),
        )
        return computed
    return {key: None for key in SCENARIO_VECTOR_KEYS}


def summarize_episode(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute a standard episode summary from trajectory frames."""
    frames = _frames(data)
    env_meta = _env_meta(data)
    own_ids, opponent_ids = _team_ids_from_data(data)
    agent_ids = _agent_order(data)

    total_agent_reward = {agent_id: 0.0 for agent_id in agent_ids}
    reward_samples: Dict[str, List[float]] = {agent_id: [] for agent_id in agent_ids}
    own_distances: List[float] = []
    center_distances: List[float] = []
    own_blood: List[float] = []
    alive_samples = 0
    alive_total = 0
    first_failure_timestep = None
    crashed_agents = set()
    shotdown_agents = set()

    for frame in frames:
        agents = frame.get("agents") if isinstance(frame, MappingABC) else {}
        if not isinstance(agents, MappingABC):
            continue
        timestep = frame.get("timestep")
        for agent_id, agent in agents.items():
            reward = _nullable_float(_agent_reward(frame, agent_id, agent))
            if reward is not None:
                total_agent_reward.setdefault(agent_id, 0.0)
                total_agent_reward[agent_id] += reward
                reward_samples.setdefault(agent_id, []).append(reward)
            alive = bool(agent.get("alive", True))
            alive_total += 1
            alive_samples += int(alive)
            if agent.get("crash"):
                crashed_agents.add(agent_id)
            if agent.get("shotdown"):
                shotdown_agents.add(agent_id)
            if agent_id in own_ids:
                blood = _nullable_float(agent.get("blood", agent.get("bloods")))
                if blood is not None:
                    own_blood.append(blood)
                if first_failure_timestep is None and _agent_has_failure(agent):
                    first_failure_timestep = timestep

        own_positions = [_position(agents.get(agent_id, {})) for agent_id in own_ids]
        own_positions = [pos for pos in own_positions if pos is not None]
        opponent_positions = [_position(agents.get(agent_id, {})) for agent_id in opponent_ids]
        opponent_positions = [pos for pos in opponent_positions if pos is not None]
        if len(own_positions) >= 2:
            own_distances.append(_norm(_sub(own_positions[0], own_positions[1])))
        if own_positions and opponent_positions:
            own_center = _center(own_positions)
            opponent_center = _center(opponent_positions)
            center_distances.append(_norm(_sub(opponent_center, own_center)))

    total_team_reward = _team_rewards(total_agent_reward, own_ids, opponent_ids)
    episode_length = _episode_length(frames)
    episode_result = _episode_result(data)
    summary = {
        "schema_version": "falcon.episode_summary.v1",
        "episode_length": episode_length,
        "total_agent_reward": total_agent_reward,
        "total_team_reward": total_team_reward,
        "mean_agent_reward": {agent_id: _mean(values) for agent_id, values in reward_samples.items()},
        "mean_team_reward": {
            "own": _mean([total_agent_reward.get(agent_id, 0.0) for agent_id in own_ids]),
            "opponent": _mean([total_agent_reward.get(agent_id, 0.0) for agent_id in opponent_ids]),
        },
        "own_teammate_distance": _min_mean_max(own_distances),
        "team_center_distance": _min_mean_max(center_distances),
        "own_blood": _min_mean_max(own_blood),
        "alive_ratio_over_episode": _safe_div(alive_samples, alive_total),
        "crash_count": len(crashed_agents),
        "shotdown_count": len(shotdown_agents),
        "first_failure_timestep": first_failure_timestep,
        "final_outcome": episode_result,
        "own_ids": own_ids,
        "opponent_ids": opponent_ids,
        "env": {
            "env_name": data.get("env_name") or env_meta.get("env_name"),
            "task_name": data.get("task_name") or env_meta.get("task_name") or env_meta.get("task"),
        },
    }
    return to_jsonable(summary)


def compute_scenario_vector(
    agents: Sequence[Mapping[str, Any]],
    own_ids: Optional[Sequence[str]] = None,
    opponent_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Compute first-version FALCON scenario vector from initial agents."""
    agent_by_id = {str(agent.get("agent_id") or agent.get("uid")): agent for agent in agents if isinstance(agent, MappingABC)}
    if own_ids is None or opponent_ids is None:
        own_ids, opponent_ids = _split_agent_ids(list(agent_by_id.keys()))

    own_positions = [_position(agent_by_id.get(agent_id, {})) for agent_id in own_ids]
    opponent_positions = [_position(agent_by_id.get(agent_id, {})) for agent_id in opponent_ids]
    own_positions = [pos for pos in own_positions if pos is not None]
    opponent_positions = [pos for pos in opponent_positions if pos is not None]
    own_velocities = [_velocity(agent_by_id.get(agent_id, {})) for agent_id in own_ids]
    opponent_velocities = [_velocity(agent_by_id.get(agent_id, {})) for agent_id in opponent_ids]
    own_velocities = [vel for vel in own_velocities if vel is not None]
    opponent_velocities = [vel for vel in opponent_velocities if vel is not None]
    own_headings = [_nullable_float(agent_by_id.get(agent_id, {}).get("initial_heading", agent_by_id.get(agent_id, {}).get("heading"))) for agent_id in own_ids]
    opponent_headings = [_nullable_float(agent_by_id.get(agent_id, {}).get("initial_heading", agent_by_id.get(agent_id, {}).get("heading"))) for agent_id in opponent_ids]
    own_headings = [heading for heading in own_headings if heading is not None]
    opponent_headings = [heading for heading in opponent_headings if heading is not None]

    missing_fields: List[str] = []
    own_center = _center(own_positions) if own_positions else None
    opponent_center = _center(opponent_positions) if opponent_positions else None
    own_velocity_center = _center(own_velocities) if own_velocities else None
    opponent_velocity_center = _center(opponent_velocities) if opponent_velocities else None

    vector: Dict[str, Optional[float]] = {
        "team_center_distance": _norm(_sub(opponent_center, own_center)) if own_center and opponent_center else None,
        "own_formation_spread": _norm(_sub(own_positions[0], own_positions[1])) if len(own_positions) >= 2 else None,
        "opponent_formation_spread": _norm(_sub(opponent_positions[0], opponent_positions[1])) if len(opponent_positions) >= 2 else None,
        "altitude_difference": (opponent_center[2] - own_center[2]) if own_center and opponent_center else None,
        "velocity_difference": (_norm(opponent_velocity_center) - _norm(own_velocity_center)) if own_velocity_center and opponent_velocity_center else None,
        "heading_difference": _mean_heading_difference(own_headings, opponent_headings) if own_headings and opponent_headings else None,
        "approximate_aspect_angle": _angle_between(own_velocity_center, _sub(opponent_center, own_center)) if own_velocity_center and own_center and opponent_center else None,
        "own_center_x": own_center[0] if own_center else None,
        "own_center_y": own_center[1] if own_center else None,
        "own_center_z": own_center[2] if own_center else None,
        "opponent_center_x": opponent_center[0] if opponent_center else None,
        "opponent_center_y": opponent_center[1] if opponent_center else None,
        "opponent_center_z": opponent_center[2] if opponent_center else None,
    }

    for key, value in vector.items():
        if value is None:
            missing_fields.append(f"initial_config.scenario_vector.{key}")
    return {key: _nullable_float(vector.get(key)) for key in SCENARIO_VECTOR_KEYS}, missing_fields


class EpisodeTrajectoryRecorder:
    """Collect and save one standardized MultiCombat episode at a time."""

    def __init__(
        self,
        output_dir: str,
        save_success: bool = False,
        prefix: str = "episode",
        metadata: Optional[Mapping[str, Any]] = None,
        policy_id: Optional[str] = None,
        training_steps: Optional[int] = None,
        save_reason: str = "auto",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_success = save_success
        self.prefix = prefix
        self.metadata = dict(metadata or {})
        self.policy_id = policy_id if policy_id is not None else self.metadata.get("policy_id")
        self.training_steps = training_steps if training_steps is not None else self.metadata.get("training_steps")
        self.save_reason = save_reason or self.metadata.get("save_reason", "auto")
        self.episode_index = 0
        self.current: Optional[Dict[str, Any]] = None
        self.last_saved_path: Optional[Path] = None

    def start_episode(
        self,
        env_metadata: Mapping[str, Any],
        initial_scenario_config: Mapping[str, Any],
        initial_frame: Mapping[str, Any],
    ) -> None:
        self.episode_index += 1
        env_metadata = to_jsonable(env_metadata)
        initial_frame = self._standardize_frame(initial_frame)
        initial_agents = _initial_agents_from_frame(initial_frame, env_metadata)
        scenario_vector, vector_missing = compute_scenario_vector(
            initial_agents,
            own_ids=env_metadata.get("ego_ids"),
            opponent_ids=env_metadata.get("enm_ids"),
        )
        episode_id = self.metadata.get("episode_id") or f"{self.prefix}_{self.episode_index:06d}"
        task_name = env_metadata.get("task_name") or _task_name(env_metadata.get("task"), env_metadata.get("config_name"))
        config_path = env_metadata.get("config_path") or _config_path(env_metadata.get("config_name"))
        self.current = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_index": self.episode_index,
            "env_name": env_metadata.get("env_name", "MultipleCombat"),
            "scenario_type": env_metadata.get("scenario_type", "2v2"),
            "task_name": task_name,
            "config_path": config_path,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "random_seed": env_metadata.get("seed"),
            "policy_id": self.policy_id,
            "training_steps": self.training_steps,
            "save_reason": self.save_reason if self.save_reason != "auto" else "unknown",
            "episode_result": "unknown",
            "done_reason": "unknown",
            "total_team_reward": {"own": 0.0, "opponent": 0.0},
            "episode_length": 0,
            "metadata": to_jsonable(self.metadata),
            "env": env_metadata,
            "initial_config": {
                "schema_version": "falcon.initial_config.v1",
                "raw_config": to_jsonable(initial_scenario_config),
                "agents": initial_agents,
                "scenario_vector": scenario_vector,
                "missing_fields": vector_missing,
            },
            "frames": [initial_frame],
            "episode_summary": None,
            "missing_fields": list(vector_missing),
            "warnings": [],
            "outcome": None,
        }

    def record_step(self, frame: Mapping[str, Any]) -> None:
        if self.current is None:
            return
        self.current["frames"].append(self._standardize_frame(frame))

    def finalize(self, outcome: Mapping[str, Any]) -> Optional[Path]:
        if self.current is None:
            return None
        outcome = to_jsonable(outcome)
        self.current["outcome"] = outcome
        episode_result = _episode_result({**self.current, "outcome": outcome})
        save_reason = self.save_reason if self.save_reason != "auto" else ("failure" if episode_result in {"loss", "crash", "shotdown"} else "success")
        done_reason = outcome.get("done_reason") or _infer_done_reason(self.current.get("frames", []), outcome, episode_result)
        summary = summarize_episode({**self.current, "episode_result": episode_result, "done_reason": done_reason})
        validation = validate_trajectory({**self.current, "episode_summary": summary})

        self.current["episode_result"] = episode_result
        self.current["done_reason"] = done_reason
        self.current["save_reason"] = save_reason
        self.current["episode_summary"] = summary
        self.current["episode_length"] = summary["episode_length"]
        self.current["total_team_reward"] = summary["total_team_reward"]
        self.current["missing_fields"] = sorted(set(self.current.get("missing_fields", []) + validation["missing_fields"]))
        self.current["warnings"] = sorted(set(self.current.get("warnings", []) + validation["warnings"]))

        should_save = self.save_success or episode_result in {"loss", "crash", "shotdown"}
        saved_path = None
        if should_save:
            saved_path = self._save_current()
        self.current = None
        self.last_saved_path = saved_path
        return saved_path

    def abort(self) -> None:
        self.current = None

    def _standardize_frame(self, frame: Mapping[str, Any]) -> Dict[str, Any]:
        frame = to_jsonable(frame)
        agents = frame.get("agents") or {}
        actions = frame.get("actions") or {}
        rewards = frame.get("rewards") or {}
        dones = frame.get("dones") or {}
        frame_info = frame.get("info") or {}
        standardized_agents = {}
        if isinstance(agents, MappingABC):
            for agent_id, agent in agents.items():
                standardized_agents[agent_id] = _standardize_agent_frame(
                    agent_id,
                    agent or {},
                    action=_mapping_get(actions, agent_id),
                    reward=_mapping_get(rewards, agent_id),
                    done=_mapping_get(dones, agent_id),
                    frame_info=frame_info,
                )
        return {
            "timestep": frame.get("timestep"),
            "sim_time_sec": frame.get("sim_time_sec"),
            "agents": standardized_agents,
            "missiles": frame.get("missiles") or {},
            "info": frame_info,
        }

    def _save_current(self) -> Path:
        assert self.current is not None
        label = self.current.get("save_reason") or self.current.get("episode_result") or "episode"
        episode_index = int(self.current.get("episode_index", 0))
        path = self.output_dir / f"{self.prefix}_{episode_index:06d}_{label}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(self.current), f, indent=2, sort_keys=True)
        return path


def _standardize_agent_frame(
    agent_id: str,
    agent: Mapping[str, Any],
    action: Any,
    reward: Any,
    done: Any,
    frame_info: Mapping[str, Any],
) -> Dict[str, Any]:
    geodetic = agent.get("geodetic") if isinstance(agent.get("geodetic"), MappingABC) else {}
    attitude = agent.get("attitude_rad") if isinstance(agent.get("attitude_rad"), MappingABC) else {}
    flags = agent.get("constraint_flags") if isinstance(agent.get("constraint_flags"), MappingABC) else {}
    position = _position(agent)
    velocity = _velocity(agent)
    speed = _nullable_float(agent.get("speed_mps", agent.get("speed")))
    if speed is None and velocity is not None:
        speed = _norm(velocity)
    blood = _nullable_float(agent.get("blood", agent.get("bloods")))
    missile_count = agent.get("missile_count", agent.get("num_left_missiles", agent.get("num_missiles")))
    team = str(agent.get("team") or agent.get("color") or (agent_id[0] if agent_id else "")).lower()
    if team in {"a", "red"}:
        team = "red"
    elif team in {"b", "blue"}:
        team = "blue"
    return {
        **dict(agent),
        "agent_id": agent_id,
        "team": team,
        "team_id": agent.get("team_id") or (agent_id[0] if agent_id else None),
        "position_neu": _vector_or_none(position),
        "position_neu_m": agent.get("position_neu_m", _vector_to_neu_dict(position)),
        "latitude": _nullable_float(agent.get("latitude", geodetic.get("latitude_deg"))),
        "longitude": _nullable_float(agent.get("longitude", geodetic.get("longitude_deg"))),
        "altitude": _nullable_float(agent.get("altitude", agent.get("altitude_m", geodetic.get("altitude_m")))),
        "velocity_neu": _vector_or_none(velocity),
        "velocity_neu_mps": agent.get("velocity_neu_mps", _vector_to_neu_dict(velocity)),
        "speed": speed,
        "airspeed": _nullable_float(agent.get("airspeed", agent.get("airspeed_mps", speed))),
        "heading": _nullable_float(agent.get("heading", agent.get("heading_rad", attitude.get("heading")))),
        "roll": _nullable_float(agent.get("roll", attitude.get("roll"))),
        "pitch": _nullable_float(agent.get("pitch", attitude.get("pitch"))),
        "action": to_jsonable(action),
        "reward": _nullable_float(reward),
        "done": bool(done) if done is not None else None,
        "info": _agent_info(frame_info, agent_id),
        "alive": bool(agent.get("alive", True)),
        "crash": bool(agent.get("crash", flags.get("crash", False))),
        "shotdown": bool(agent.get("shotdown", flags.get("shotdown", False))),
        "blood": blood,
        "missile_count": _nullable_float(missile_count),
        "low_altitude": bool(agent.get("low_altitude", flags.get("low_altitude", False))),
        "overload": bool(agent.get("overload", flags.get("overload", False))),
        "extreme_state": bool(agent.get("extreme_state", flags.get("extreme_state", False))),
        "horizontal_distance_from_center_m": _nullable_float(
            agent.get("horizontal_distance_from_center_m", flags.get("horizontal_distance_from_center_m"))
        ),
    }


def _initial_agents_from_frame(frame: Mapping[str, Any], env_meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    agents = frame.get("agents") if isinstance(frame, MappingABC) else {}
    result = []
    for agent_id, agent in (agents or {}).items():
        if not isinstance(agent, MappingABC):
            continue
        result.append(
            {
                "agent_id": agent_id,
                "team": agent.get("team"),
                "initial_latitude": agent.get("latitude"),
                "initial_longitude": agent.get("longitude"),
                "initial_altitude": agent.get("altitude"),
                "initial_position_neu": agent.get("position_neu"),
                "initial_velocity": agent.get("velocity_neu"),
                "initial_heading": agent.get("heading"),
                "initial_roll": agent.get("roll"),
                "initial_pitch": agent.get("pitch"),
                "alive": agent.get("alive"),
                "blood": agent.get("blood"),
                "missile_count": agent.get("missile_count"),
                "color": agent.get("color"),
                "team_id": agent.get("team_id") or (agent_id[0] if agent_id else None),
            }
        )
    return result


def _frames(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    frames = data.get("frames") or data.get("steps") or []
    return [frame for frame in frames if isinstance(frame, MappingABC)]


def _env_meta(data: Mapping[str, Any]) -> Mapping[str, Any]:
    env_meta = data.get("env") or {}
    return env_meta if isinstance(env_meta, MappingABC) else {}


def _agent_order(data: Mapping[str, Any]) -> List[str]:
    env_meta = _env_meta(data)
    if env_meta.get("agent_order"):
        return list(env_meta.get("agent_order"))
    frames = _frames(data)
    if frames and isinstance(frames[0].get("agents"), MappingABC):
        return list(frames[0]["agents"].keys())
    initial_agents = ((data.get("initial_config") or {}).get("agents") if isinstance(data.get("initial_config"), MappingABC) else None) or []
    return [str(agent.get("agent_id")) for agent in initial_agents if isinstance(agent, MappingABC) and agent.get("agent_id")]


def _team_ids_from_data(data: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    env_meta = _env_meta(data)
    own_ids = list(env_meta.get("ego_ids") or env_meta.get("own_ids") or [])
    opponent_ids = list(env_meta.get("enm_ids") or env_meta.get("opponent_ids") or [])
    if own_ids and opponent_ids:
        return own_ids, opponent_ids
    return _split_agent_ids(_agent_order(data))


def _split_agent_ids(agent_ids: Sequence[str]) -> Tuple[List[str], List[str]]:
    if not agent_ids:
        return [], []
    own_prefix = str(agent_ids[0])[0]
    own_ids = [agent_id for agent_id in agent_ids if str(agent_id).startswith(own_prefix)]
    opponent_ids = [agent_id for agent_id in agent_ids if agent_id not in own_ids]
    return own_ids, opponent_ids


def _position(agent: Mapping[str, Any]) -> Optional[List[float]]:
    value = agent.get("position_neu") or agent.get("initial_position_neu") or agent.get("position_neu_m") or agent.get("position")
    return _vector_from_value(value)


def _velocity(agent: Mapping[str, Any]) -> Optional[List[float]]:
    value = agent.get("velocity_neu") or agent.get("initial_velocity") or agent.get("velocity_neu_mps") or agent.get("velocity")
    return _vector_from_value(value)


def _vector_from_value(value: Any) -> Optional[List[float]]:
    if isinstance(value, MappingABC):
        return [
            _safe_float(value.get("north", value.get("x", 0.0))),
            _safe_float(value.get("east", value.get("y", 0.0))),
            _safe_float(value.get("up", value.get("z", 0.0))),
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return [_safe_float(value[0]), _safe_float(value[1]), _safe_float(value[2])]
    return None


def _vector_or_none(value: Optional[Sequence[float]]) -> Optional[List[float]]:
    if value is None:
        return None
    return [_safe_float(item) for item in value[:3]]


def _vector_to_neu_dict(value: Optional[Sequence[float]]) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    return {"north": _safe_float(value[0]), "east": _safe_float(value[1]), "up": _safe_float(value[2])}


def _center(vectors: Sequence[Sequence[float]]) -> List[float]:
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(3)]


def _sub(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[List[float]]:
    if a is None or b is None:
        return None
    return [_safe_float(a[i]) - _safe_float(b[i]) for i in range(min(len(a), len(b)))]


def _norm(value: Optional[Sequence[float]]) -> float:
    if value is None:
        return 0.0
    return math.sqrt(sum(_safe_float(item) ** 2 for item in value))


def _angle_between(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    na = _norm(a)
    nb = _norm(b)
    if na <= 1e-8 or nb <= 1e-8:
        return None
    dot = sum(_safe_float(x) * _safe_float(y) for x, y in zip(a, b))
    return math.acos(max(-1.0, min(1.0, dot / (na * nb))))


def _mean_heading_difference(own_headings: Sequence[float], opponent_headings: Sequence[float]) -> Optional[float]:
    if not own_headings or not opponent_headings:
        return None
    own = _circular_mean(own_headings)
    opponent = _circular_mean(opponent_headings)
    return abs((opponent - own + math.pi) % (2 * math.pi) - math.pi)


def _circular_mean(values: Sequence[float]) -> float:
    return math.atan2(_mean(math.sin(v) for v in values), _mean(math.cos(v) for v in values))


def _team_rewards(total_agent_reward: Mapping[str, float], own_ids: Sequence[str], opponent_ids: Sequence[str]) -> Dict[str, float]:
    return {
        "own": sum(_safe_float(total_agent_reward.get(agent_id)) for agent_id in own_ids),
        "opponent": sum(_safe_float(total_agent_reward.get(agent_id)) for agent_id in opponent_ids),
    }


def _episode_length(frames: Sequence[Mapping[str, Any]]) -> int:
    if not frames:
        return 0
    timesteps = [_nullable_float(frame.get("timestep")) for frame in frames]
    timesteps = [step for step in timesteps if step is not None]
    if timesteps:
        return int(max(timesteps))
    return max(0, len(frames) - 1)


def _episode_result(data: Mapping[str, Any]) -> str:
    explicit = data.get("episode_result")
    if explicit and explicit != "unknown":
        return str(explicit)
    outcome = data.get("outcome") if isinstance(data.get("outcome"), MappingABC) else {}
    if outcome.get("timeout"):
        return "timeout"
    frames = _frames(data)
    own_ids, _opponent_ids = _team_ids_from_data(data)
    own_crash = any(_agent_has_flag(frame, own_ids, "crash") for frame in frames)
    own_shotdown = any(_agent_has_flag(frame, own_ids, "shotdown") for frame in frames)
    if own_crash:
        return "crash"
    if own_shotdown:
        return "shotdown"
    if outcome.get("ego_failed"):
        return "loss"
    winner = outcome.get("winner")
    ego_team = outcome.get("ego_team")
    if winner and ego_team and winner == ego_team:
        return "win"
    if winner and winner != "draw" and ego_team and winner != ego_team:
        return "loss"
    return "unknown"


def _infer_done_reason(frames: Sequence[Mapping[str, Any]], outcome: Mapping[str, Any], episode_result: str) -> str:
    if outcome.get("done_reason"):
        return str(outcome.get("done_reason"))
    if outcome.get("timeout") or episode_result == "timeout":
        return "timeout"
    if episode_result in {"crash", "shotdown", "loss", "win"}:
        return episode_result
    for frame in frames:
        info = frame.get("info") if isinstance(frame, MappingABC) else {}
        if isinstance(info, MappingABC) and info.get("done_reason"):
            return str(info.get("done_reason"))
    return "unknown"


def _agent_has_flag(frame: Mapping[str, Any], agent_ids: Sequence[str], flag: str) -> bool:
    agents = frame.get("agents") if isinstance(frame, MappingABC) else {}
    if not isinstance(agents, MappingABC):
        return False
    return any(bool((agents.get(agent_id) or {}).get(flag)) for agent_id in agent_ids)


def _agent_has_failure(agent: Mapping[str, Any]) -> bool:
    return bool(
        agent.get("crash")
        or agent.get("shotdown")
        or agent.get("low_altitude")
        or agent.get("overload")
        or agent.get("extreme_state")
        or agent.get("alive") is False
    )


def _agent_reward(frame: Mapping[str, Any], agent_id: str, agent: Mapping[str, Any]) -> Any:
    if "reward" in agent:
        return agent.get("reward")
    rewards = frame.get("rewards") if isinstance(frame.get("rewards"), MappingABC) else {}
    return rewards.get(agent_id)


def _mapping_get(value: Any, key: str) -> Any:
    if not isinstance(value, MappingABC):
        return None
    item = value.get(key)
    if isinstance(item, list) and len(item) == 1:
        return item[0]
    return item


def _agent_info(frame_info: Mapping[str, Any], agent_id: str) -> Dict[str, Any]:
    if not isinstance(frame_info, MappingABC):
        return {}
    info = dict(frame_info)
    agent_info = info.get(agent_id)
    if isinstance(agent_info, MappingABC):
        info.update(agent_info)
    return to_jsonable(info)


def _config_path(config_name: Optional[str]) -> Optional[str]:
    if not config_name:
        return None
    return f"envs/JSBSim/configs/{config_name}.yaml"


def _task_name(task: Optional[str], config_name: Optional[str]) -> str:
    task_label = task or "multiplecombat"
    if config_name and "NoWeapon" in config_name:
        mode = "NoWeapon"
    elif config_name and "ShootMissile" in config_name:
        mode = "ShootMissile"
    else:
        mode = "Unknown"
    return f"MultiCombat {mode}" if "multiplecombat" in str(task_label).lower() else str(task_label)


def _min_mean_max(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [_safe_float(value) for value in values if _nullable_float(value) is not None]
    if not vals:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(vals), "mean": _mean(vals), "max": max(vals)}


def _mean(values: Iterable[float]) -> float:
    vals = [_safe_float(value) for value in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _nullable_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _safe_float(value: Any) -> float:
    result = _nullable_float(value)
    return 0.0 if result is None else result
