"""Offline policy evaluation bridge for FALCON candidate scenarios."""

from __future__ import annotations

import copy
import json
import math
from argparse import Namespace
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from .candidate_schema import validate_candidate_schema
from .scenario_adapter import apply_initial_config_to_yaml, load_base_scenario_config

if not hasattr(np, "product"):
    np.product = np.prod  # NumPy 2.x compatibility for the existing flattener.

POLICY_EVAL_SCHEMA_VERSION = "1.0"

DEFAULT_CONFIG: Dict[str, Any] = {
    "env_name": "MultipleCombat",
    "algorithm_name": "mappo",
    "scenario_name": "2v2/NoWeapon/Selfplay",
    "base_config_path": "envs/JSBSim/configs/2v2/NoWeapon/Selfplay.yaml",
    "device": "cpu",
    "deterministic": True,
    "use_selfplay": False,
    "opponent_mode": "same_actor",
    "opponent_checkpoint": None,
    "args_overrides": {},
}

SUPPORTED_OPPONENT_MODES = ("same_actor", "fixed_checkpoint", "env_default")


class OpponentConfigurationError(ValueError):
    """Raised when an explicit evaluation opponent cannot be prepared."""


class PolicyEvaluator:
    """Evaluate saved MAPPO actors on generated LAG 2v2 scenarios.

    The evaluator does not train or modify MAPPO. It instantiates
    ``MultipleCombatEnv`` from an in-memory YAML config, loads an existing actor
    checkpoint into the repository's ``PPOPolicy``, and performs deterministic
    rollout evaluation.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))

    def evaluate_policy_on_scenario(
        self,
        policy_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        scenario_yaml: Union[str, Path, Mapping[str, Any]],
        num_episodes: int = 5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        scenario_id, scenario_config, scenario_path, warnings = self._load_scenario(scenario_yaml)
        checkpoint_path = _resolve_checkpoint_path(policy_checkpoint)
        policy_id = _policy_id(policy_checkpoint, checkpoint_path)
        output = _empty_eval_result(
            policy_id=policy_id,
            checkpoint_path=str(policy_checkpoint or ""),
            scenario_id=scenario_id,
            scenario_yaml=scenario_path,
            num_episodes=num_episodes,
            warnings=warnings,
        )
        output["real_policy_eval_available"] = False
        output["actor_loaded"] = False
        output.update(self._opponent_metadata(checkpoint_path))

        if checkpoint_path is None:
            output["failure_stage"] = "checkpoint_loading_not_available"
            output["warnings"].append("No usable actor checkpoint was provided or discovered.")
            return output
        if not checkpoint_path.exists():
            output["failure_stage"] = "checkpoint_loading_not_available"
            output["warnings"].append(f"Actor checkpoint does not exist: {checkpoint_path}")
            return output

        opponent_error = self._validate_opponent_config()
        if opponent_error:
            output["failure_stage"] = "opponent_checkpoint_loading_not_available"
            output["warnings"].append(opponent_error)
            return output

        try:
            result = self._run_real_eval(
                checkpoint_path=checkpoint_path,
                scenario_config=scenario_config,
                scenario_id=scenario_id,
                scenario_path=scenario_path,
                num_episodes=int(num_episodes),
                seed=seed,
                policy_id=policy_id,
            )
            result["warnings"] = sorted(set(output["warnings"] + result.get("warnings", [])))
            result["actor_loaded"] = True
            return result
        except OpponentConfigurationError as exc:
            output["failure_stage"] = "opponent_checkpoint_loading_not_available"
            output["warnings"].append(str(exc))
            return output
        except Exception as exc:  # noqa: BLE001 - bridge must report structured failure
            output["failure_stage"] = "policy_rollout_failed"
            output["warnings"].append(f"Real policy evaluation failed: {type(exc).__name__}: {exc}")
            return output

    def evaluate_policy_on_candidates(
        self,
        policy_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        candidate_scenarios: Sequence[Mapping[str, Any]],
        num_episodes: int = 5,
    ) -> List[Dict[str, Any]]:
        results = []
        for idx, candidate in enumerate(candidate_scenarios):
            scenario_yaml = self._candidate_to_yaml_config(candidate)
            scenario_id = candidate.get("scenario_id") or f"candidate_{idx:04d}"
            if isinstance(scenario_yaml, MappingABC):
                scenario_yaml = dict(scenario_yaml)
                scenario_yaml["scenario_id"] = scenario_id
            results.append(self.evaluate_policy_on_scenario(policy_checkpoint, scenario_yaml, num_episodes=num_episodes))
        return results

    def evaluate_current_and_best(
        self,
        current_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        best_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        candidate_scenarios: Sequence[Mapping[str, Any]],
        num_episodes: int = 5,
    ) -> List[Dict[str, Any]]:
        combined = []
        for idx, candidate in enumerate(candidate_scenarios):
            scenario_yaml = self._candidate_to_yaml_config(candidate)
            scenario_id = str(candidate.get("scenario_id") or f"candidate_{idx:04d}")
            if isinstance(scenario_yaml, MappingABC):
                scenario_yaml = dict(scenario_yaml)
                scenario_yaml["scenario_id"] = scenario_id
            current_eval = self.evaluate_policy_on_scenario(current_checkpoint, scenario_yaml, num_episodes=num_episodes)
            best_eval = self.evaluate_policy_on_scenario(best_checkpoint, scenario_yaml, num_episodes=num_episodes)
            combined.append(
                {
                    "schema_version": POLICY_EVAL_SCHEMA_VERSION,
                    "scenario_id": scenario_id,
                    "current_policy_eval": current_eval,
                    "best_policy_eval": best_eval,
                    "real_policy_eval_available": bool(
                        current_eval.get("real_policy_eval_available") and best_eval.get("real_policy_eval_available")
                    ),
                    "warnings": sorted(set(current_eval.get("warnings", []) + best_eval.get("warnings", []))),
                }
            )
        return combined

    def save_policy_eval_results(self, results: Any, output_path: Union[str, Path]) -> None:
        save_policy_eval_results(results, output_path)

    def _run_real_eval(
        self,
        checkpoint_path: Path,
        scenario_config: Mapping[str, Any],
        scenario_id: str,
        scenario_path: str,
        num_episodes: int,
        seed: Optional[int],
        policy_id: str,
    ) -> Dict[str, Any]:
        import torch

        env = None
        try:
            env = _make_env_from_yaml_config(scenario_config, scenario_id=scenario_id)
            if seed is not None:
                env.seed(seed)
            args = _build_policy_args(self.config)
            device = torch.device(str(self.config.get("device", "cpu")))
            from algorithms.mappo.ppo_policy import PPOPolicy

            policy = PPOPolicy(args, env.observation_space, env.share_observation_space, env.action_space, device=device)
            state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
            policy.actor.load_state_dict(state_dict)
            policy.prep_rollout()
            opponent_policy = None
            opponent_mode = str(self.config.get("opponent_mode", "same_actor"))
            opponent_checkpoint = (
                _resolve_checkpoint_path(self.config.get("opponent_checkpoint"))
                if opponent_mode == "fixed_checkpoint"
                else None
            )
            if opponent_mode == "fixed_checkpoint":
                if opponent_checkpoint is None or not opponent_checkpoint.exists():
                    raise OpponentConfigurationError(
                        "opponent_mode=fixed_checkpoint requires an existing opponent_checkpoint; "
                        "same-actor fallback is disabled."
                    )
                opponent_policy = PPOPolicy(args, env.observation_space, env.share_observation_space, env.action_space, device=device)
                opponent_policy.actor.load_state_dict(torch.load(str(opponent_checkpoint), map_location=device, weights_only=True))
                opponent_policy.prep_rollout()
            elif opponent_mode == "same_actor":
                opponent_policy = policy
            elif opponent_mode != "env_default":
                raise OpponentConfigurationError(
                    f"Unsupported opponent_mode={opponent_mode!r}; expected one of {SUPPORTED_OPPONENT_MODES}."
                )

            episodes = []
            for episode_idx in range(max(int(num_episodes), 0)):
                episodes.append(
                    _rollout_episode(
                        env=env,
                        policy=policy,
                        opponent_policy=opponent_policy,
                        opponent_mode=opponent_mode,
                        args=args,
                        deterministic=bool(self.config.get("deterministic", True)),
                    )
                )
                if seed is not None:
                    env.seed(seed + episode_idx + 1)
            return _summarize_eval(
                policy_id=policy_id,
                checkpoint_path=str(checkpoint_path),
                scenario_id=scenario_id,
                scenario_yaml=scenario_path,
                num_episodes=num_episodes,
                episodes=episodes,
                warnings=_opponent_warnings(opponent_mode, checkpoint_path, opponent_checkpoint),
                opponent_mode=opponent_mode,
                opponent_checkpoint=str(opponent_checkpoint) if opponent_checkpoint is not None else None,
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def _validate_opponent_config(self) -> Optional[str]:
        opponent_mode = str(self.config.get("opponent_mode", "same_actor"))
        if opponent_mode not in SUPPORTED_OPPONENT_MODES:
            return f"Unsupported opponent_mode={opponent_mode!r}; expected one of {SUPPORTED_OPPONENT_MODES}."
        if opponent_mode == "fixed_checkpoint":
            opponent_checkpoint = _resolve_checkpoint_path(self.config.get("opponent_checkpoint"))
            if opponent_checkpoint is None:
                return (
                    "opponent_mode=fixed_checkpoint requires opponent_checkpoint; "
                    "same-actor fallback is disabled."
                )
            if not opponent_checkpoint.exists():
                return f"Fixed opponent checkpoint does not exist: {opponent_checkpoint}"
        return None

    def _opponent_metadata(self, agent_checkpoint: Optional[Path]) -> Dict[str, Any]:
        opponent_mode = str(self.config.get("opponent_mode", "same_actor"))
        opponent_checkpoint = (
            _resolve_checkpoint_path(self.config.get("opponent_checkpoint"))
            if opponent_mode == "fixed_checkpoint"
            else None
        )
        return {
            "agent_checkpoint": str(agent_checkpoint) if agent_checkpoint is not None else None,
            "opponent_mode": opponent_mode,
            "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint is not None else None,
            "same_actor": opponent_mode == "same_actor",
            "same_checkpoint": bool(
                agent_checkpoint is not None
                and opponent_checkpoint is not None
                and _same_path(agent_checkpoint, opponent_checkpoint)
            ),
        }

    def _candidate_to_yaml_config(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        if candidate.get("scenario_yaml") is not None:
            return candidate["scenario_yaml"]
        if candidate.get("yaml_path") is not None:
            return str(candidate["yaml_path"])
        base_config = load_base_scenario_config(self.config["base_config_path"])
        if isinstance(candidate.get("initial_config"), MappingABC):
            yaml_config = apply_initial_config_to_yaml(base_config, candidate["initial_config"])
            yaml_config["scenario_id"] = candidate.get("scenario_id")
            return yaml_config
        validation = validate_candidate_schema(candidate)
        yaml_config = copy.deepcopy(base_config)
        yaml_config["scenario_id"] = candidate.get("scenario_id")
        yaml_config.setdefault("_falcon_warnings", []).append(
            f"Candidate did not include initial_config; schema validation: {validation}"
        )
        return yaml_config

    def _load_scenario(self, scenario_yaml: Union[str, Path, Mapping[str, Any]]) -> Tuple[str, Dict[str, Any], str, List[str]]:
        warnings: List[str] = []
        if isinstance(scenario_yaml, MappingABC):
            config = copy.deepcopy(dict(scenario_yaml))
            scenario_id = str(config.get("scenario_id", "memory_scenario"))
            return scenario_id, config, "memory", warnings
        path = Path(scenario_yaml)
        if not path.exists():
            base_config = load_base_scenario_config(self.config["base_config_path"])
            warnings.append(f"Scenario YAML path not found; fell back to base_config_path: {path}")
            return path.stem, base_config, str(path), warnings
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, MappingABC):
            raise ValueError(f"Scenario YAML is not a mapping: {path}")
        scenario_id = str(config.get("scenario_id", path.stem))
        return scenario_id, dict(config), str(path), warnings


class MockPolicyEvaluator:
    """Deterministic mock evaluator for smoke tests when real checkpoints are absent."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge({"current_base_win_rate": 0.35, "best_base_win_rate": 0.78}, dict(config or {}))

    def evaluate_policy_on_scenario(
        self,
        policy_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        scenario_yaml: Union[str, Path, Mapping[str, Any]],
        num_episodes: int = 5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        scenario_id = _scenario_id_from_any(scenario_yaml)
        role = _policy_id(policy_checkpoint, _resolve_checkpoint_path(policy_checkpoint))
        is_best = "best" in role.lower()
        win_rate = float(self.config["best_base_win_rate"] if is_best else self.config["current_base_win_rate"])
        spread = 0.05 * (abs(hash(scenario_id)) % 5)
        win_rate = _clamp01(win_rate - spread if not is_best else win_rate - spread / 2.0)
        mean_return = 400.0 * win_rate - 150.0
        episodes = [
            {
                "episode_index": idx,
                "result": "win" if idx < round(win_rate * max(num_episodes, 1)) else "loss",
                "team_return": mean_return,
                "episode_length": 1000,
                "winner": "A" if idx < round(win_rate * max(num_episodes, 1)) else "B",
            }
            for idx in range(max(int(num_episodes), 0))
        ]
        return _summarize_eval(
            policy_id=role,
            checkpoint_path=str(policy_checkpoint or ""),
            scenario_id=scenario_id,
            scenario_yaml=str(scenario_yaml if not isinstance(scenario_yaml, MappingABC) else "memory"),
            num_episodes=num_episodes,
            episodes=episodes,
            warnings=["MockPolicyEvaluator used because real MAPPO checkpoints were unavailable."],
            real_available=False,
            failure_stage=None,
        )

    def evaluate_policy_on_candidates(
        self,
        policy_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        candidate_scenarios: Sequence[Mapping[str, Any]],
        num_episodes: int = 5,
    ) -> List[Dict[str, Any]]:
        return [
            self.evaluate_policy_on_scenario(policy_checkpoint, candidate, num_episodes=num_episodes)
            for candidate in candidate_scenarios
        ]

    def evaluate_current_and_best(
        self,
        current_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        best_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]],
        candidate_scenarios: Sequence[Mapping[str, Any]],
        num_episodes: int = 5,
    ) -> List[Dict[str, Any]]:
        results = []
        for candidate in candidate_scenarios:
            scenario_id = str(candidate.get("scenario_id", _scenario_id_from_any(candidate)))
            current_eval = self.evaluate_policy_on_scenario(current_checkpoint or {"policy_id": "mock_current"}, candidate, num_episodes)
            best_eval = self.evaluate_policy_on_scenario(best_checkpoint or {"policy_id": "mock_best"}, candidate, num_episodes)
            results.append(
                {
                    "schema_version": POLICY_EVAL_SCHEMA_VERSION,
                    "scenario_id": scenario_id,
                    "current_policy_eval": current_eval,
                    "best_policy_eval": best_eval,
                    "real_policy_eval_available": False,
                    "warnings": sorted(set(current_eval.get("warnings", []) + best_eval.get("warnings", []))),
                }
            )
        return results

    def save_policy_eval_results(self, results: Any, output_path: Union[str, Path]) -> None:
        save_policy_eval_results(results, output_path)


def save_policy_eval_results(results: Any, output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"schema_version": "falcon.policy_eval_results.v1", "results": results}, f, indent=2, sort_keys=True)


def discover_policy_checkpoints(root: Union[str, Path] = "results") -> Dict[str, Any]:
    root_path = Path(root)
    actor_files = sorted(root_path.rglob("actor*.pt")) if root_path.exists() else []
    return {
        "schema_version": "falcon.policy_checkpoint_discovery.v1",
        "root": str(root_path),
        "actor_checkpoints": [str(path) for path in actor_files],
        "current_checkpoint": str(actor_files[-1]) if actor_files else None,
        "best_checkpoint": str(actor_files[-1]) if actor_files else None,
        "warnings": [] if actor_files else [f"No MAPPO actor checkpoints found under {root_path}."],
    }


def _make_env_from_yaml_config(yaml_config: Mapping[str, Any], scenario_id: str):
    from envs.JSBSim.envs import env_base
    from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

    original_parse_config = env_base.parse_config
    config_copy = copy.deepcopy(dict(yaml_config))

    def _parse_config_override(_filename):
        return type("EnvConfig", (object,), config_copy)

    env_base.parse_config = _parse_config_override
    try:
        env = MultipleCombatEnv(f"falcon_memory_eval/{scenario_id}")
    finally:
        env_base.parse_config = original_parse_config
    return env


def _rollout_episode(
    env,
    policy,
    opponent_policy,
    opponent_mode: str,
    args: Namespace,
    deterministic: bool,
) -> Dict[str, Any]:
    import torch

    def _t2n(x):
        return x.detach().cpu().numpy()

    obs, share_obs = env.reset()
    num_agents = int(env.num_agents)
    half = num_agents // 2
    rnn_states = np.zeros((num_agents, args.recurrent_hidden_layers, args.recurrent_hidden_size), dtype=np.float32)
    masks = np.ones((num_agents, 1), dtype=np.float32)
    team_returns = {"own": 0.0, "opponent": 0.0}
    step_count = 0
    done_all = False
    with torch.no_grad():
        while not done_all and step_count < int(getattr(env, "max_steps", 1000)) + 5:
            own_actions, own_rnn = policy.act(obs[:half], rnn_states[:half], masks[:half], deterministic=deterministic)
            if opponent_mode == "env_default":
                opponent_actions = np.zeros_like(_t2n(own_actions))
                opponent_rnn = rnn_states[half:]
            else:
                opponent_actions, opponent_rnn = opponent_policy.act(obs[half:], rnn_states[half:], masks[half:], deterministic=deterministic)
            own_actions = _t2n(own_actions)
            own_rnn = _t2n(own_rnn)
            opponent_actions = _t2n(opponent_actions) if hasattr(opponent_actions, "detach") else np.asarray(opponent_actions)
            opponent_rnn = _t2n(opponent_rnn) if hasattr(opponent_rnn, "detach") else np.asarray(opponent_rnn)
            actions = np.concatenate((own_actions, opponent_actions), axis=0)
            rnn_states = np.concatenate((own_rnn, opponent_rnn), axis=0)
            obs, share_obs, rewards, dones, info = env.step(actions)
            rewards = np.asarray(rewards).reshape(num_agents, -1)
            dones_arr = np.asarray(dones).reshape(num_agents, -1)
            team_returns["own"] += float(np.mean(rewards[:half]))
            team_returns["opponent"] += float(np.mean(rewards[half:]))
            done_all = bool(np.all(dones_arr))
            if done_all:
                masks[:] = 0.0
            step_count += 1
    outcome = env._episode_outcome() if hasattr(env, "_episode_outcome") else {}
    winner = outcome.get("winner", "unknown")
    ego_team = outcome.get("ego_team", "A")
    timeout = bool(outcome.get("timeout", step_count >= int(getattr(env, "max_steps", 1000))))
    if winner == ego_team:
        result = "win"
    elif timeout or winner == "draw":
        result = "timeout"
    else:
        result = "loss"
    return {
        "episode_index": None,
        "result": result,
        "team_return": team_returns["own"],
        "opponent_return": team_returns["opponent"],
        "episode_length": step_count,
        "winner": winner,
        "outcome": outcome,
    }


def _summarize_eval(
    policy_id: str,
    checkpoint_path: str,
    scenario_id: str,
    scenario_yaml: str,
    num_episodes: int,
    episodes: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    real_available: bool = True,
    failure_stage: Optional[str] = None,
    opponent_mode: Optional[str] = None,
    opponent_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    returns = [_float(item.get("team_return")) for item in episodes]
    results = [str(item.get("result", "unknown")) for item in episodes]
    wins = sum(1 for result in results if result == "win")
    losses = sum(1 for result in results if result == "loss")
    timeouts = sum(1 for result in results if result == "timeout")
    denom = max(len(episodes), 1)
    lengths = [_float(item.get("episode_length")) for item in episodes]
    return {
        "schema_version": POLICY_EVAL_SCHEMA_VERSION,
        "policy_id": policy_id,
        "checkpoint_path": checkpoint_path,
        "agent_checkpoint": checkpoint_path,
        "opponent_mode": opponent_mode,
        "opponent_checkpoint": opponent_checkpoint,
        "same_actor": opponent_mode == "same_actor" if opponent_mode is not None else None,
        "same_checkpoint": bool(
            checkpoint_path
            and opponent_checkpoint
            and _same_path(Path(checkpoint_path), Path(opponent_checkpoint))
        ),
        "scenario_id": scenario_id,
        "scenario_yaml": scenario_yaml,
        "num_eval_episodes": int(num_episodes),
        "win_rate": round(wins / denom, 6),
        "loss_rate": round(losses / denom, 6),
        "timeout_rate": round(timeouts / denom, 6),
        "mean_return": round(float(np.mean(returns)) if returns else 0.0, 6),
        "std_return": round(float(np.std(returns)) if returns else 0.0, 6),
        "mean_episode_length": round(float(np.mean(lengths)) if lengths else 0.0, 6),
        "failure_rate": round((losses + timeouts) / denom, 6),
        "episode_results": [dict(item) for item in episodes],
        "real_policy_eval_available": bool(real_available),
        "failure_stage": failure_stage,
        "warnings": sorted(set(str(warning) for warning in warnings)),
    }


def _empty_eval_result(
    policy_id: str,
    checkpoint_path: str,
    scenario_id: str,
    scenario_yaml: str,
    num_episodes: int,
    warnings: Sequence[str],
) -> Dict[str, Any]:
    return _summarize_eval(policy_id, checkpoint_path, scenario_id, scenario_yaml, num_episodes, [], warnings, real_available=False)


def _opponent_warnings(
    opponent_mode: str,
    agent_checkpoint: Path,
    opponent_checkpoint: Optional[Path],
) -> List[str]:
    warnings: List[str] = []
    if opponent_mode == "same_actor":
        warnings.append("opponent_mode=same_actor; both teams use the evaluated actor.")
    elif opponent_mode == "env_default":
        warnings.append("opponent_mode=env_default uses deterministic neutral actions; no rule-based opponent is available.")
    elif opponent_checkpoint is not None and _same_path(agent_checkpoint, opponent_checkpoint):
        warnings.append("Fixed opponent checkpoint resolves to the same file as the agent checkpoint.")
    return warnings


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _build_policy_args(config: Mapping[str, Any]) -> Namespace:
    from config import get_config

    parser = get_config()
    args = parser.parse_known_args([])[0]
    args.env_name = config.get("env_name", "MultipleCombat")
    args.algorithm_name = config.get("algorithm_name", "mappo")
    args.scenario_name = config.get("scenario_name", "2v2/NoWeapon/Selfplay")
    args.n_rollout_threads = 1
    args.n_eval_rollout_threads = 1
    args.use_selfplay = bool(config.get("use_selfplay", False))
    for key, value in dict(config.get("args_overrides") or {}).items():
        setattr(args, key, value)
    return args


def _resolve_checkpoint_path(policy_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]]) -> Optional[Path]:
    if policy_checkpoint is None:
        return None
    if isinstance(policy_checkpoint, MappingABC):
        for key in ("checkpoint_path", "actor_path", "path"):
            if policy_checkpoint.get(key):
                return _resolve_checkpoint_path(policy_checkpoint[key])
        return None
    path = Path(policy_checkpoint)
    if path.is_dir():
        for name in ("actor_latest.pt", "actor_best.pt"):
            candidate = path / name
            if candidate.exists():
                return candidate
        actor_files = sorted(path.glob("actor_*.pt"))
        if actor_files:
            return actor_files[-1]
    return path


def _policy_id(policy_checkpoint: Optional[Union[str, Path, Mapping[str, Any]]], checkpoint_path: Optional[Path]) -> str:
    if isinstance(policy_checkpoint, MappingABC) and policy_checkpoint.get("policy_id"):
        return str(policy_checkpoint["policy_id"])
    if checkpoint_path is not None:
        return checkpoint_path.stem
    return "unknown_policy"


def _scenario_id_from_any(value: Any) -> str:
    if isinstance(value, MappingABC):
        return str(value.get("scenario_id", "memory_scenario"))
    return Path(str(value)).stem


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _clamp01(value: Any) -> float:
    value = _float(value)
    return max(0.0, min(1.0, value))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
