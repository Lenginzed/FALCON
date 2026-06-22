import copy
import numpy as np
from collections.abc import Mapping as MappingABC
from typing import Tuple, Dict, Any, Mapping, Optional
from .env_base import BaseEnv
from falcon.trajectory_recorder import EpisodeTrajectoryRecorder, to_jsonable
from ..core.catalog import Catalog as c
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False
        self._trajectory_recorder = None  # type: Optional[EpisodeTrajectoryRecorder]
        trajectory_recording = getattr(self.config, 'trajectory_recording', None)
        if isinstance(trajectory_recording, MappingABC) and trajectory_recording.get('enabled', False):
            self.enable_trajectory_recording(
                output_dir=trajectory_recording.get('output_dir', './trajectory_records'),
                save_success=trajectory_recording.get('save_success', False),
                prefix=trajectory_recording.get('prefix', 'episode'),
                metadata=trajectory_recording.get('metadata', None),
            )

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def enable_trajectory_recording(
        self,
        output_dir: str,
        save_success: bool = False,
        prefix: str = "episode",
        metadata: Optional[Mapping[str, Any]] = None,
        policy_id: Optional[str] = None,
        training_steps: Optional[int] = None,
        save_reason: str = "auto",
    ):
        """Enable JSON trajectory recording for completed episodes.

        By default only ego-team failures are saved. Set ``save_success=True``
        to save every completed episode.
        """
        self._trajectory_recorder = EpisodeTrajectoryRecorder(
            output_dir=output_dir,
            save_success=save_success,
            prefix=prefix,
            metadata=metadata,
            policy_id=policy_id,
            training_steps=training_steps,
            save_reason=save_reason,
        )

    def disable_trajectory_recording(self):
        if self._trajectory_recorder is not None:
            self._trajectory_recorder.abort()
        self._trajectory_recorder = None

    def get_initial_scenario_config(self) -> Dict[str, Any]:
        return {
            "config_name": getattr(self, "config_name", None),
            "task": getattr(self.config, "task", None),
            "sim_freq": self.sim_freq,
            "agent_interaction_steps": self.agent_interaction_steps,
            "max_steps": self.max_steps,
            "battle_field_center": [self.center_lon, self.center_lat, self.center_alt],
            "aircraft_configs": copy.deepcopy(getattr(self.config, "aircraft_configs", {})),
            "termination": {
                "altitude_limit": getattr(self.config, "altitude_limit", None),
                "acceleration_limit_x": getattr(self.config, "acceleration_limit_x", None),
                "acceleration_limit_y": getattr(self.config, "acceleration_limit_y", None),
                "acceleration_limit_z": getattr(self.config, "acceleration_limit_z", None),
            },
        }

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        self._start_trajectory_episode(obs, share_obs)
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        action = self._unpack(action)
        raw_action = copy.deepcopy(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        self._record_trajectory_step(raw_action, obs, share_obs, rewards, dones, info)
        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info

    def _start_trajectory_episode(self, obs: Dict[str, np.ndarray], share_obs: Dict[str, np.ndarray]):
        if self._trajectory_recorder is None:
            return
        frame = self._capture_trajectory_frame(obs=obs, share_obs=share_obs)
        self._trajectory_recorder.start_episode(
            env_metadata=self._trajectory_env_metadata(),
            initial_scenario_config=self.get_initial_scenario_config(),
            initial_frame=frame,
        )

    def _record_trajectory_step(
        self,
        action: Dict[str, Any],
        obs: Dict[str, np.ndarray],
        share_obs: Dict[str, np.ndarray],
        rewards: Dict[str, Any],
        dones: Dict[str, Any],
        info: Dict[str, Any],
    ):
        if self._trajectory_recorder is None:
            return
        frame = self._capture_trajectory_frame(
            action=action,
            obs=obs,
            share_obs=share_obs,
            rewards=rewards,
            dones=dones,
            info=info,
        )
        self._trajectory_recorder.record_step(frame)
        episode_done = all(self._as_bool(value) for value in dones.values())
        if episode_done:
            self._trajectory_recorder.finalize(self._episode_outcome())

    def _trajectory_env_metadata(self) -> Dict[str, Any]:
        return {
            "env_name": "MultipleCombat",
            "scenario_type": "2v2",
            "config_name": getattr(self, "config_name", None),
            "config_path": f"envs/JSBSim/configs/{getattr(self, 'config_name', None)}.yaml",
            "task": getattr(self.config, "task", None),
            "task_name": "MultiCombat ShootMissile" if "ShootMissile" in getattr(self, "config_name", "") else "MultiCombat NoWeapon",
            "num_agents": self.num_agents,
            "agent_order": list((self.ego_ids + self.enm_ids)[:self.num_agents]),
            "ego_ids": list(self.ego_ids),
            "enm_ids": list(self.enm_ids),
            "max_steps": self.max_steps,
            "sim_freq": self.sim_freq,
            "agent_interaction_steps": self.agent_interaction_steps,
            "time_interval": self.time_interval,
            "seed": getattr(self, "seed_value", None),
            "state_variables": [prop.name_jsbsim for prop in self.task.state_var],
            "action_variables": [prop.name_jsbsim for prop in self.task.action_var],
        }

    def _capture_trajectory_frame(
        self,
        action: Optional[Dict[str, Any]] = None,
        obs: Optional[Dict[str, np.ndarray]] = None,
        share_obs: Optional[Dict[str, np.ndarray]] = None,
        rewards: Optional[Dict[str, Any]] = None,
        dones: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "timestep": int(self.current_step),
            "sim_time_sec": float(self.current_step * self.time_interval),
            "agents": {uid: self._agent_snapshot(uid, sim) for uid, sim in self.agents.items()},
            "missiles": {uid: self._temp_sim_snapshot(uid, sim) for uid, sim in self._tempsims.items()},
            "actions": to_jsonable(action) if action is not None else None,
            "obs": to_jsonable(obs) if obs is not None else None,
            "share_obs": to_jsonable(share_obs) if share_obs is not None else None,
            "rewards": self._scalar_dict(rewards),
            "dones": self._bool_dict(dones),
            "info": to_jsonable(info or {}),
        }

    def _agent_snapshot(self, uid: str, sim) -> Dict[str, Any]:
        geodetic = sim.get_geodetic()
        position = sim.get_position()
        rpy = sim.get_rpy()
        velocity = sim.get_velocity()
        state_values = sim.get_property_values(self.task.state_var)
        action_values = sim.get_property_values(self.task.action_var)
        return {
            "uid": uid,
            "team": sim.color.lower(),
            "team_id": uid[0],
            "color": sim.color,
            "model": sim.model,
            "alive": bool(sim.is_alive),
            "crash": bool(sim.is_crash),
            "shotdown": bool(sim.is_shotdown),
            "bloods": float(sim.bloods),
            "num_missiles": int(getattr(sim, "num_missiles", 0)),
            "num_left_missiles": int(getattr(sim, "num_left_missiles", 0)),
            "geodetic": {
                "longitude_deg": float(geodetic[0]),
                "latitude_deg": float(geodetic[1]),
                "altitude_m": float(geodetic[2]),
            },
            "position_neu_m": {
                "north": float(position[0]),
                "east": float(position[1]),
                "up": float(position[2]),
            },
            "attitude_rad": {
                "roll": float(rpy[0]),
                "pitch": float(rpy[1]),
                "heading": float(rpy[2]),
            },
            "heading_rad": float(rpy[2]),
            "altitude_m": float(geodetic[2]),
            "velocity_neu_mps": {
                "north": float(velocity[0]),
                "east": float(velocity[1]),
                "up": float(velocity[2]),
            },
            "speed_mps": float(np.linalg.norm(velocity)),
            "state": {prop.name_jsbsim: float(value) for prop, value in zip(self.task.state_var, state_values)},
            "controls": {prop.name_jsbsim: float(value) for prop, value in zip(self.task.action_var, action_values)},
            "constraint_flags": self._constraint_flags(sim),
        }

    def _temp_sim_snapshot(self, uid: str, sim) -> Dict[str, Any]:
        geodetic = sim.get_geodetic()
        position = sim.get_position()
        velocity = sim.get_velocity()
        rpy = sim.get_rpy()
        return {
            "uid": uid,
            "color": sim.color,
            "model": sim.model,
            "alive": bool(getattr(sim, "is_alive", False)),
            "success": bool(getattr(sim, "is_success", False)),
            "done": bool(getattr(sim, "is_done", False)),
            "parent": getattr(getattr(sim, "parent_aircraft", None), "uid", None),
            "target": getattr(getattr(sim, "target_aircraft", None), "uid", None),
            "geodetic": {
                "longitude_deg": float(geodetic[0]),
                "latitude_deg": float(geodetic[1]),
                "altitude_m": float(geodetic[2]),
            },
            "position_neu_m": {
                "north": float(position[0]),
                "east": float(position[1]),
                "up": float(position[2]),
            },
            "velocity_neu_mps": {
                "north": float(velocity[0]),
                "east": float(velocity[1]),
                "up": float(velocity[2]),
            },
            "attitude_rad": {
                "roll": float(rpy[0]),
                "pitch": float(rpy[1]),
                "heading": float(rpy[2]),
            },
        }

    def _constraint_flags(self, sim) -> Dict[str, Any]:
        altitude_m = float(sim.get_property_value(c.position_h_sl_m))
        sim_time = float(sim.get_property_value(c.simulation_sim_time_sec))
        accel_x = float(sim.get_property_value(c.accelerations_n_pilot_x_norm))
        accel_y = float(sim.get_property_value(c.accelerations_n_pilot_y_norm))
        accel_z = float(sim.get_property_value(c.accelerations_n_pilot_z_norm))
        overload = sim_time > 10.0 and (
            abs(accel_x) > getattr(self.config, "acceleration_limit_x", 10.0)
            or abs(accel_y) > getattr(self.config, "acceleration_limit_y", 10.0)
            or abs(accel_z + 1.0) > getattr(self.config, "acceleration_limit_z", 10.0)
        )
        position = sim.get_position()
        return {
            "low_altitude": altitude_m <= getattr(self.config, "altitude_limit", 2500),
            "overload": bool(overload),
            "extreme_state": bool(sim.get_property_value(c.detect_extreme_state)),
            "crash": bool(sim.is_crash),
            "shotdown": bool(sim.is_shotdown),
            "out_of_bounds": False,
            "horizontal_distance_from_center_m": float(np.linalg.norm(position[:2])),
        }

    def _episode_outcome(self) -> Dict[str, Any]:
        teams = sorted(set(uid[0] for uid in self.agents.keys()))
        alive_by_team = {
            team: [uid for uid, sim in self.agents.items() if uid[0] == team and sim.is_alive]
            for team in teams
        }
        dead_by_team = {
            team: [uid for uid, sim in self.agents.items() if uid[0] == team and not sim.is_alive]
            for team in teams
        }
        ego_team = self.ego_ids[0][0] if self.ego_ids else None
        enm_team = self.enm_ids[0][0] if self.enm_ids else None
        winner = "draw"
        if ego_team is not None and enm_team is not None:
            ego_alive = len(alive_by_team.get(ego_team, []))
            enm_alive = len(alive_by_team.get(enm_team, []))
            if ego_alive > 0 and enm_alive == 0:
                winner = ego_team
            elif enm_alive > 0 and ego_alive == 0:
                winner = enm_team
        return {
            "winner": winner,
            "ego_team": ego_team,
            "enemy_team": enm_team,
            "ego_failed": winner != ego_team,
            "alive_by_team": alive_by_team,
            "dead_by_team": dead_by_team,
            "final_step": int(self.current_step),
            "timeout": self.current_step >= self.max_steps,
        }

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, (list, tuple, np.ndarray)):
            return bool(np.asarray(value).item())
        return bool(value)

    @staticmethod
    def _as_scalar(value: Any):
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.size == 0:
                return None
            return arr.reshape(-1)[0].item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _scalar_dict(self, values: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if values is None:
            return None
        return {key: self._as_scalar(value) for key, value in values.items()}

    def _bool_dict(self, values: Optional[Dict[str, Any]]) -> Optional[Dict[str, bool]]:
        if values is None:
            return None
        return {key: self._as_bool(value) for key, value in values.items()}
