"""Failure mode analysis for FALCON 2v2 MultiCombat trajectories."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .trajectory_recorder import (
    SCENARIO_VECTOR_KEYS,
    extract_scenario_vector,
    load_trajectory,
    summarize_episode,
    validate_trajectory,
)

ANALYSIS_SCHEMA_VERSION = "falcon.failure_analysis.v1"

FAILURE_MODE_KEYS = (
    "coordination_failure",
    "target_assignment_confusion",
    "initial_disadvantage",
    "generalization_failure",
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "eps": 1e-8,
    "label_thresholds": {
        "primary": 0.70,
        "secondary": 0.40,
        "severity_high": 0.70,
        "severity_medium": 0.40,
    },
    "confidence": {
        "missing_field_penalty": 0.03,
        "warning_penalty": 0.08,
        "min_confidence": 0.05,
    },
    "return_loss": {
        "success_reward_keys": (
            "mean_success_team_reward",
            "average_success_team_reward",
            "success_episode_mean_return",
            "success_mean_return",
        ),
        "fallback_pool_keys": (
            "mean_success_team_reward",
            "average_success_team_reward",
            "mean_team_reward",
        ),
    },
    "coordination_failure": {
        "weights": {
            "separation_failure": 0.40,
            "role_overlap": 0.30,
            "return_loss": 0.30,
        },
        "separation_threshold_m": 6000.0,
        "separation_normalization_m": 6000.0,
    },
    "target_assignment_confusion": {
        "weights": {
            "target_switch": 0.45,
            "same_target_rate": 0.35,
            "assignment_entropy": 0.20,
        },
        "target_switch_threshold": 4.0,
    },
    "initial_disadvantage": {
        "weights": {
            "distance_penalty": 0.25,
            "formation_penalty": 0.20,
            "altitude_penalty": 0.20,
            "velocity_penalty": 0.15,
            "angle_penalty": 0.20,
        },
        "ideal_team_center_distance": 10000.0,
        "team_center_distance_tolerance": 10000.0,
        "ideal_own_formation_spread": 1000.0,
        "formation_tolerance": 3000.0,
        "altitude_reference": 0.0,
        "altitude_tolerance": 2000.0,
        "velocity_reference": 0.0,
        "velocity_tolerance": 150.0,
        "ideal_aspect_angle": 0.0,
        "aspect_angle_tolerance": math.pi,
        "historical_lambda": 0.70,
    },
    "generalization_failure": {
        "scenario_vector_scales": {
            "team_center_distance": 10000.0,
            "own_formation_spread": 3000.0,
            "opponent_formation_spread": 3000.0,
            "altitude_difference": 2000.0,
            "velocity_difference": 150.0,
            "heading_difference": math.pi,
            "approximate_aspect_angle": math.pi,
            "own_center_x": 10000.0,
            "own_center_y": 10000.0,
            "own_center_z": 8000.0,
            "opponent_center_x": 10000.0,
            "opponent_center_y": 10000.0,
            "opponent_center_z": 8000.0,
        },
        "novelty_distance_normalization": 1.0,
        "in_distribution_win_rate_keys": ("W_in", "in_distribution_win_rate", "train_distribution_win_rate"),
        "scenario_win_rate_keys": ("W_s", "scenario_win_rate", "current_scenario_win_rate"),
    },
    "failure_severity": {
        "weights": {
            "outcome_failure": 0.40,
            "early_failure": 0.30,
            "return_loss": 0.30,
        },
        "low_reward_return_loss_threshold": 0.50,
    },
}


class FailureAnalyzer:
    """Compute first-stage FALCON failure scores from trajectory data."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, dict(config or {}))

    def analyze_trajectory(
        self,
        trajectory_data: Mapping[str, Any],
        pool_stats: Optional[Mapping[str, Any]] = None,
        success_stats: Optional[Mapping[str, Any]] = None,
        policy_eval_stats: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = dict(trajectory_data)
        validation = validate_trajectory(data)
        missing_fields = list(validation.get("missing_fields", []))
        warnings = list(validation.get("warnings", []))
        submetrics: Dict[str, Any] = {}

        summary = data.get("episode_summary") if isinstance(data.get("episode_summary"), MappingABC) else summarize_episode(data)
        scenario_vector = extract_scenario_vector(data)
        return_loss, return_loss_report = self._return_loss(summary, success_stats, pool_stats)
        submetrics["return_loss"] = return_loss_report
        warnings.extend(return_loss_report.get("warnings", []))

        assignment_metrics = self._assignment_metrics(data)
        submetrics["target_assignment"] = assignment_metrics
        warnings.extend(assignment_metrics.get("warnings", []))

        coordination_sub = self._coordination_submetrics(data, assignment_metrics, return_loss)
        submetrics["coordination_failure"] = coordination_sub
        coordination_failure = _weighted_sum(
            coordination_sub,
            self.config["coordination_failure"]["weights"],
        )

        target_sub = {
            "target_switch": assignment_metrics["target_switch"],
            "same_target_rate": assignment_metrics["same_target_rate"],
            "assignment_entropy": assignment_metrics["assignment_entropy"],
        }
        submetrics["target_assignment_confusion"] = target_sub
        target_assignment_confusion = _weighted_sum(
            target_sub,
            self.config["target_assignment_confusion"]["weights"],
        )

        initial_sub = self._initial_disadvantage_submetrics(scenario_vector, policy_eval_stats)
        submetrics["initial_disadvantage"] = initial_sub
        missing_fields.extend(initial_sub.get("missing_fields", []))
        warnings.extend(initial_sub.get("warnings", []))
        initial_disadvantage = initial_sub["score"]

        generalization_sub = self._generalization_submetrics(scenario_vector, pool_stats, policy_eval_stats)
        submetrics["generalization_failure"] = generalization_sub
        missing_fields.extend(generalization_sub.get("missing_fields", []))
        warnings.extend(generalization_sub.get("warnings", []))
        generalization_failure = generalization_sub["score"]

        severity_sub = self._failure_severity_submetrics(data, summary, return_loss)
        submetrics["failure_severity"] = severity_sub
        failure_severity = _weighted_sum(
            severity_sub,
            self.config["failure_severity"]["weights"],
        )

        failure_scores = {
            "coordination_failure": _round01(coordination_failure),
            "target_assignment_confusion": _round01(target_assignment_confusion),
            "initial_disadvantage": _round01(initial_disadvantage),
            "generalization_failure": _round01(generalization_failure),
            "failure_severity": _round01(failure_severity),
        }
        primary, secondary = self._failure_mode_labels(failure_scores)
        severity_level = self._severity_level(failure_scores["failure_severity"])
        missing_fields = sorted(set(missing_fields))
        warnings = sorted(set(warnings))
        evidence = self._evidence(failure_scores, submetrics, warnings)

        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "source_trajectory": data.get("_source_trajectory") or data.get("episode_id"),
            "failure_scores": failure_scores,
            "submetrics": _jsonable_submetrics(submetrics),
            "evidence": evidence,
            "primary_failure_modes": primary,
            "secondary_failure_modes": secondary,
            "severity_level": severity_level,
            "confidence": self._confidence(missing_fields, warnings),
            "missing_fields": missing_fields,
            "warnings": warnings,
        }

    def analyze_file(self, path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        data = load_trajectory(path)
        data["_source_trajectory"] = str(path)
        result = self.analyze_trajectory(data)
        if output_path is not None:
            self._save_json(result, output_path)
        return result

    def batch_analyze(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for path in sorted(input_dir.glob("*.json")):
            output_path = output_dir / f"{path.stem}_failure_analysis.json"
            results.append(self.analyze_file(path, output_path=output_path))
        return results

    def analyze(
        self,
        episode_trajectory: Union[str, Path, Mapping[str, Any]],
        initial_scenario_config: Optional[Mapping[str, Any]] = None,
        training_scenario_pool_statistics: Optional[Mapping[str, Any]] = None,
        historical_policy_evaluation_results: Optional[Mapping[str, Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for the initial prototype API."""
        if isinstance(episode_trajectory, MappingABC):
            data = dict(episode_trajectory)
        else:
            data = load_trajectory(episode_trajectory)
            data["_source_trajectory"] = str(episode_trajectory)
        if initial_scenario_config and "initial_config" not in data:
            data["initial_config"] = {"raw_config": dict(initial_scenario_config)}
        result = self.analyze_trajectory(
            data,
            pool_stats=training_scenario_pool_statistics,
            policy_eval_stats=historical_policy_evaluation_results,
        )
        if output_path is not None:
            self._save_json(result, output_path)
        return result

    def _coordination_submetrics(
        self,
        data: Mapping[str, Any],
        assignment_metrics: Mapping[str, Any],
        return_loss: float,
    ) -> Dict[str, float]:
        own_ids, _opponent_ids = _team_ids(data)
        frames = _frames(data)
        distances = []
        for frame in frames:
            agents = _agents(frame)
            if len(own_ids) < 2:
                continue
            pos0 = _position(agents.get(own_ids[0], {}))
            pos1 = _position(agents.get(own_ids[1], {}))
            if pos0 is not None and pos1 is not None:
                distances.append(_norm(_sub(pos0, pos1)))
        threshold = _float(self.config["coordination_failure"]["separation_threshold_m"])
        normalizer = max(_float(self.config["coordination_failure"]["separation_normalization_m"]), self.config["eps"])
        penalties = [_clamp01((distance - threshold) / normalizer) for distance in distances]
        return {
            "separation_failure": _mean(penalties),
            "role_overlap": _float(assignment_metrics.get("same_target_rate")),
            "return_loss": return_loss,
            "mean_own_separation_m": _mean(distances),
            "max_own_separation_m": max(distances) if distances else 0.0,
        }

    def _assignment_metrics(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        own_ids, opponent_ids = _team_ids(data)
        frames = _frames(data)
        warnings: List[str] = []
        if len(own_ids) < 2 or len(opponent_ids) < 2:
            warnings.append("Cannot reliably compute 2v2 target assignment metrics: missing own or opponent IDs.")
            return {
                "target_switch": 0.0,
                "same_target_rate": 0.0,
                "assignment_entropy": 0.0,
                "target_switch_count": 0,
                "valid_assignment_steps": 0,
                "warnings": warnings,
            }

        previous_targets: Dict[str, str] = {}
        switches = 0
        same_target_steps = 0
        valid_steps = 0
        assignment_counter: Counter[str] = Counter()
        for frame in frames:
            agents = _agents(frame)
            if not agents:
                continue
            frame_targets: Dict[str, str] = {}
            for own_id in own_ids:
                target_id = _nearest_target(agents, own_id, opponent_ids)
                if target_id is None:
                    continue
                frame_targets[own_id] = target_id
                assignment_label = f"{own_id}->{target_id}"
                assignment_counter[assignment_label] += 1
                if own_id in previous_targets and previous_targets[own_id] != target_id:
                    switches += 1
                previous_targets[own_id] = target_id
            if len(frame_targets) == len(own_ids):
                valid_steps += 1
                if len(set(frame_targets.values())) < len(frame_targets):
                    same_target_steps += 1

        if valid_steps == 0:
            warnings.append("No valid target-assignment timesteps were available.")
            return {
                "target_switch": 0.0,
                "same_target_rate": 0.0,
                "assignment_entropy": 0.0,
                "target_switch_count": switches,
                "valid_assignment_steps": 0,
                "warnings": warnings,
            }

        switch_threshold = max(_float(self.config["target_assignment_confusion"]["target_switch_threshold"]), self.config["eps"])
        target_switch = _clamp01(switches / (switch_threshold * len(own_ids)))
        same_target_rate = _clamp01(same_target_steps / valid_steps)
        entropy = _normalized_entropy(assignment_counter.values(), max_bins=4)
        return {
            "target_switch": target_switch,
            "same_target_rate": same_target_rate,
            "assignment_entropy": entropy,
            "target_switch_count": switches,
            "valid_assignment_steps": valid_steps,
            "same_target_steps": same_target_steps,
            "assignment_distribution": dict(assignment_counter),
            "warnings": warnings,
        }

    def _initial_disadvantage_submetrics(
        self,
        scenario_vector: Mapping[str, Optional[float]],
        policy_eval_stats: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        cfg = self.config["initial_disadvantage"]
        missing_fields = [f"initial_config.scenario_vector.{key}" for key, value in scenario_vector.items() if value is None]
        warnings: List[str] = []

        distance_penalty = _abs_penalty(
            scenario_vector.get("team_center_distance"),
            cfg["ideal_team_center_distance"],
            cfg["team_center_distance_tolerance"],
        )
        formation_penalty = _abs_penalty(
            scenario_vector.get("own_formation_spread"),
            cfg["ideal_own_formation_spread"],
            cfg["formation_tolerance"],
        )
        altitude_penalty = _positive_penalty(
            scenario_vector.get("altitude_difference"),
            cfg["altitude_reference"],
            cfg["altitude_tolerance"],
        )
        velocity_penalty = _positive_penalty(
            scenario_vector.get("velocity_difference"),
            cfg["velocity_reference"],
            cfg["velocity_tolerance"],
        )
        angle_source = scenario_vector.get("approximate_aspect_angle")
        if angle_source is None:
            angle_source = scenario_vector.get("heading_difference")
            warnings.append("approximate_aspect_angle missing; falling back to heading_difference for angle_penalty.")
        angle_penalty = _abs_penalty(
            angle_source,
            cfg["ideal_aspect_angle"],
            cfg["aspect_angle_tolerance"],
        )
        rule_components = {
            "distance_penalty": distance_penalty,
            "formation_penalty": formation_penalty,
            "altitude_penalty": altitude_penalty,
            "velocity_penalty": velocity_penalty,
            "angle_penalty": angle_penalty,
        }
        rule_score = _weighted_sum(rule_components, cfg["weights"])

        historical_best_win_rate = _lookup(policy_eval_stats, ("historical_best_win_rate", "best_win_rate_on_similar_scenarios"))
        if historical_best_win_rate is not None:
            lam = _clamp01(cfg["historical_lambda"])
            score = lam * rule_score + (1.0 - lam) * (1.0 - _clamp01(historical_best_win_rate))
        else:
            score = rule_score
        return {
            **rule_components,
            "rule_score": _round01(rule_score),
            "historical_best_win_rate": historical_best_win_rate,
            "score": _round01(score),
            "missing_fields": missing_fields,
            "warnings": warnings,
        }

    def _generalization_submetrics(
        self,
        scenario_vector: Mapping[str, Optional[float]],
        pool_stats: Optional[Mapping[str, Any]],
        policy_eval_stats: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        missing_fields: List[str] = []
        novelty = 0.0
        performance_drop = 0.0
        reliable = True

        pool_vectors = _pool_vectors(pool_stats)
        if not pool_vectors:
            reliable = False
            warnings.append("generalization_failure cannot reliably estimate scenario_novelty without pool_stats scenario vectors.")
            missing_fields.append("pool_stats.scenario_vectors")
        else:
            novelty = self._scenario_novelty(scenario_vector, pool_vectors)

        w_in = _lookup(policy_eval_stats, self.config["generalization_failure"]["in_distribution_win_rate_keys"])
        w_s = _lookup(policy_eval_stats, self.config["generalization_failure"]["scenario_win_rate_keys"])
        if w_in is None or w_s is None:
            reliable = False
            warnings.append("generalization_failure cannot reliably estimate performance_drop without W_in and W_s.")
            if w_in is None:
                missing_fields.append("policy_eval_stats.W_in")
            if w_s is None:
                missing_fields.append("policy_eval_stats.W_s")
        else:
            performance_drop = _clamp01((w_in - w_s) / (w_in + self.config["eps"]))

        score = novelty * performance_drop if reliable else 0.0
        return {
            "scenario_novelty": _round01(novelty),
            "performance_drop": _round01(performance_drop),
            "score": _round01(score),
            "reliable": reliable,
            "missing_fields": missing_fields,
            "warnings": warnings,
        }

    def _failure_severity_submetrics(
        self,
        data: Mapping[str, Any],
        summary: Mapping[str, Any],
        return_loss: float,
    ) -> Dict[str, Any]:
        episode_result = str(data.get("episode_result") or summary.get("final_outcome") or "unknown")
        outcome_failure = self._outcome_failure(episode_result, return_loss)
        episode_length = max(_float(data.get("episode_length", summary.get("episode_length"))), 1.0)
        failure_timestep = summary.get("first_failure_timestep")
        if failure_timestep is None:
            failure_timestep = episode_length
        early_failure = _clamp01(1.0 - _float(failure_timestep) / episode_length)
        collapse_score = self._collapse_score(data, summary)
        return {
            "outcome_failure": outcome_failure,
            "early_failure": early_failure,
            "return_loss": return_loss,
            "collapse_score": collapse_score,
            "episode_result": episode_result,
            "failure_timestep": failure_timestep,
        }

    def _return_loss(
        self,
        summary: Mapping[str, Any],
        success_stats: Optional[Mapping[str, Any]],
        pool_stats: Optional[Mapping[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        warnings: List[str] = []
        current_return = _own_team_return(summary)
        success_return = _success_return(success_stats, self.config["return_loss"]["success_reward_keys"])
        source = "success_stats"
        if success_return is None:
            success_return = _success_return(pool_stats, self.config["return_loss"]["fallback_pool_keys"])
            source = "pool_stats"
        if success_return is None:
            warnings.append("return_loss uses 0.0 because no success_stats or batch fallback return was provided.")
            return 0.0, {
                "value": 0.0,
                "current_team_return": current_return,
                "success_reference_return": None,
                "source": None,
                "warnings": warnings,
            }
        value = _clamp01((success_return - current_return) / (abs(success_return) + self.config["eps"]))
        return value, {
            "value": _round01(value),
            "current_team_return": current_return,
            "success_reference_return": success_return,
            "source": source,
            "warnings": warnings,
        }

    def _scenario_novelty(self, scenario_vector: Mapping[str, Optional[float]], pool_vectors: Sequence[Mapping[str, Any]]) -> float:
        distances = [
            _normalized_vector_distance(
                scenario_vector,
                pool_vector,
                self.config["generalization_failure"]["scenario_vector_scales"],
            )
            for pool_vector in pool_vectors
        ]
        distances = [distance for distance in distances if distance is not None]
        if not distances:
            return 0.0
        normalizer = max(_float(self.config["generalization_failure"]["novelty_distance_normalization"]), self.config["eps"])
        return _clamp01(min(distances) / normalizer)

    def _outcome_failure(self, episode_result: str, return_loss: float) -> float:
        if episode_result in {"loss", "crash", "shotdown"}:
            return 1.0
        if episode_result == "timeout":
            return 0.5
        if episode_result == "win":
            return 0.0
        if return_loss >= self.config["failure_severity"]["low_reward_return_loss_threshold"]:
            return 0.5
        return 0.0

    def _collapse_score(self, data: Mapping[str, Any], summary: Mapping[str, Any]) -> float:
        own_ids, _opponent_ids = _team_ids(data)
        frames = _frames(data)
        if not frames or not own_ids:
            return 0.0
        final_agents = _agents(frames[-1])
        dead_ratio = _mean([0.0 if (final_agents.get(agent_id) or {}).get("alive", True) else 1.0 for agent_id in own_ids])
        blood_summary = summary.get("own_blood") if isinstance(summary.get("own_blood"), MappingABC) else {}
        mean_blood = blood_summary.get("mean")
        blood_loss = _clamp01((100.0 - _float(mean_blood)) / 100.0) if mean_blood is not None else 0.0
        crash_shot = _clamp01((_float(summary.get("crash_count")) + _float(summary.get("shotdown_count"))) / max(len(own_ids), 1))
        return _round01(0.50 * dead_ratio + 0.25 * blood_loss + 0.25 * crash_shot)

    def _failure_mode_labels(self, failure_scores: Mapping[str, float]) -> Tuple[List[str], List[str]]:
        primary_threshold = self.config["label_thresholds"]["primary"]
        secondary_threshold = self.config["label_thresholds"]["secondary"]
        primary = [mode for mode in FAILURE_MODE_KEYS if failure_scores.get(mode, 0.0) >= primary_threshold]
        secondary = [
            mode
            for mode in FAILURE_MODE_KEYS
            if secondary_threshold <= failure_scores.get(mode, 0.0) < primary_threshold
        ]
        return primary, secondary

    def _severity_level(self, severity: float) -> str:
        if severity >= self.config["label_thresholds"]["severity_high"]:
            return "high"
        if severity >= self.config["label_thresholds"]["severity_medium"]:
            return "medium"
        return "low"

    def _confidence(self, missing_fields: Sequence[str], warnings: Sequence[str]) -> float:
        cfg = self.config["confidence"]
        confidence = 1.0
        confidence -= min(0.30, len(missing_fields) * cfg["missing_field_penalty"])
        confidence -= min(0.40, len(warnings) * cfg["warning_penalty"])
        return _round01(max(cfg["min_confidence"], confidence))

    def _evidence(
        self,
        failure_scores: Mapping[str, float],
        submetrics: Mapping[str, Any],
        warnings: Sequence[str],
    ) -> Dict[str, List[str]]:
        evidence = {
            "coordination_failure": [],
            "target_assignment_confusion": [],
            "initial_disadvantage": [],
            "generalization_failure": [],
            "failure_severity": [],
        }
        coordination = submetrics.get("coordination_failure", {})
        sep_threshold = self.config["coordination_failure"]["separation_threshold_m"]
        if _float(coordination.get("mean_own_separation_m")) > sep_threshold:
            evidence["coordination_failure"].append(
                f"Mean teammate distance {coordination.get('mean_own_separation_m'):.1f} m exceeded threshold {sep_threshold:.1f} m."
            )
        if _float(coordination.get("role_overlap")) >= 0.4:
            evidence["coordination_failure"].append(
                f"Same-target overlap rate was high ({coordination.get('role_overlap'):.2f})."
            )
        if _float(coordination.get("return_loss")) >= 0.4:
            evidence["coordination_failure"].append(
                f"Team return loss relative to success baseline was {coordination.get('return_loss'):.2f}."
            )

        target = submetrics.get("target_assignment_confusion", {})
        assignment = submetrics.get("target_assignment", {})
        if _float(target.get("target_switch")) >= 0.4:
            evidence["target_assignment_confusion"].append(
                f"Target switch count was {assignment.get('target_switch_count')} over {assignment.get('valid_assignment_steps')} valid assignment steps."
            )
        if _float(target.get("same_target_rate")) >= 0.4:
            evidence["target_assignment_confusion"].append(
                f"Both own agents selected the same nearest target in {target.get('same_target_rate'):.2f} of valid steps."
            )
        if _float(target.get("assignment_entropy")) >= 0.7:
            evidence["target_assignment_confusion"].append(
                f"Assignment entropy was high ({target.get('assignment_entropy'):.2f}), indicating unstable target pairing."
            )

        initial = submetrics.get("initial_disadvantage", {})
        for key in ("distance_penalty", "formation_penalty", "altitude_penalty", "velocity_penalty", "angle_penalty"):
            if _float(initial.get(key)) >= 0.4 and len(evidence["initial_disadvantage"]) < 3:
                evidence["initial_disadvantage"].append(f"{key} contributed {initial.get(key):.2f} to the rule-based initial disadvantage score.")

        generalization = submetrics.get("generalization_failure", {})
        if not generalization.get("reliable", True):
            evidence["generalization_failure"].append("Generalization evidence is unreliable because required pool or policy evaluation statistics are missing.")
        else:
            evidence["generalization_failure"].append(
                f"Scenario novelty {generalization.get('scenario_novelty'):.2f} multiplied by performance drop {generalization.get('performance_drop'):.2f}."
            )

        severity = submetrics.get("failure_severity", {})
        evidence["failure_severity"].append(f"Episode result was {severity.get('episode_result', 'unknown')}.")
        if _float(severity.get("return_loss")) >= 0.4:
            evidence["failure_severity"].append(f"Return loss was {severity.get('return_loss'):.2f}.")
        if _float(severity.get("collapse_score")) > 0:
            evidence["failure_severity"].append(f"Collapse score from alive/crash/shotdown/blood signals was {severity.get('collapse_score'):.2f}.")

        for key, items in evidence.items():
            if not items:
                if key == "generalization_failure" and any("generalization_failure" in warning for warning in warnings):
                    evidence[key].append("Insufficient statistics for reliable generalization evidence.")
                else:
                    evidence[key].append("No strong evidence crossed the first-version reporting thresholds.")
            evidence[key] = items[:3]
        return evidence

    def _save_json(self, data: Mapping[str, Any], output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def analyze_failure(
    episode_trajectory: Union[str, Path, Mapping[str, Any]],
    initial_scenario_config: Optional[Mapping[str, Any]] = None,
    training_scenario_pool_statistics: Optional[Mapping[str, Any]] = None,
    historical_policy_evaluation_results: Optional[Mapping[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    return FailureAnalyzer().analyze(
        episode_trajectory=episode_trajectory,
        initial_scenario_config=initial_scenario_config,
        training_scenario_pool_statistics=training_scenario_pool_statistics,
        historical_policy_evaluation_results=historical_policy_evaluation_results,
        output_path=output_path,
    )


def _frames(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    frames = data.get("frames") or data.get("steps") or []
    return [frame for frame in frames if isinstance(frame, MappingABC)]


def _agents(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    agents = frame.get("agents") if isinstance(frame, MappingABC) else {}
    return agents if isinstance(agents, MappingABC) else {}


def _team_ids(data: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    env_meta = data.get("env") if isinstance(data.get("env"), MappingABC) else {}
    own_ids = list(env_meta.get("ego_ids") or env_meta.get("own_ids") or [])
    opponent_ids = list(env_meta.get("enm_ids") or env_meta.get("opponent_ids") or [])
    if own_ids and opponent_ids:
        return own_ids, opponent_ids
    agent_ids = []
    frames = _frames(data)
    if frames:
        agent_ids = list(_agents(frames[0]).keys())
    elif isinstance(data.get("initial_config"), MappingABC):
        agent_ids = [
            str(agent.get("agent_id"))
            for agent in data["initial_config"].get("agents", [])
            if isinstance(agent, MappingABC) and agent.get("agent_id")
        ]
    if not agent_ids:
        return [], []
    own_prefix = str(agent_ids[0])[0]
    own_ids = [agent_id for agent_id in agent_ids if str(agent_id).startswith(own_prefix)]
    opponent_ids = [agent_id for agent_id in agent_ids if agent_id not in own_ids]
    return own_ids, opponent_ids


def _position(agent: Mapping[str, Any]) -> Optional[List[float]]:
    value = agent.get("position_neu") or agent.get("position_neu_m") or agent.get("position")
    return _vector(value)


def _nearest_target(agents: Mapping[str, Any], source_id: str, opponent_ids: Sequence[str]) -> Optional[str]:
    source_pos = _position(agents.get(source_id, {}))
    if source_pos is None:
        return None
    best_id = None
    best_distance = float("inf")
    for opponent_id in opponent_ids:
        if not (agents.get(opponent_id, {}) or {}).get("alive", True):
            continue
        target_pos = _position(agents.get(opponent_id, {}))
        if target_pos is None:
            continue
        distance = _norm(_sub(target_pos, source_pos))
        if distance < best_distance:
            best_id = opponent_id
            best_distance = distance
    return best_id


def _vector(value: Any) -> Optional[List[float]]:
    if isinstance(value, MappingABC):
        return [
            _float(value.get("north", value.get("x", 0.0))),
            _float(value.get("east", value.get("y", 0.0))),
            _float(value.get("up", value.get("z", 0.0))),
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return [_float(value[0]), _float(value[1]), _float(value[2])]
    return None


def _sub(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[List[float]]:
    if a is None or b is None:
        return None
    return [_float(a[i]) - _float(b[i]) for i in range(min(len(a), len(b)))]


def _norm(value: Optional[Sequence[float]]) -> float:
    if value is None:
        return 0.0
    return math.sqrt(sum(_float(item) ** 2 for item in value))


def _weighted_sum(values: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return _clamp01(sum(_float(values.get(key)) * _float(weight) for key, weight in weights.items()))


def _abs_penalty(value: Any, ideal: float, tolerance: float) -> float:
    if value is None:
        return 0.0
    return _clamp01(abs(_float(value) - _float(ideal)) / max(_float(tolerance), 1e-8))


def _positive_penalty(value: Any, reference: float, tolerance: float) -> float:
    if value is None:
        return 0.0
    return _clamp01((_float(value) - _float(reference)) / max(_float(tolerance), 1e-8))


def _normalized_entropy(counts: Iterable[int], max_bins: int) -> float:
    values = [_float(count) for count in counts if _float(count) > 0]
    total = sum(values)
    if total <= 0 or max_bins <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return _clamp01(entropy / math.log(max_bins))


def _normalized_vector_distance(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    scales: Mapping[str, float],
) -> Optional[float]:
    parts = []
    for key in SCENARIO_VECTOR_KEYS:
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            continue
        scale = max(_float(scales.get(key, 1.0)), 1e-8)
        parts.append(((_float(va) - _float(vb)) / scale) ** 2)
    if not parts:
        return None
    return math.sqrt(sum(parts) / len(parts))


def _pool_vectors(pool_stats: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not isinstance(pool_stats, MappingABC):
        return []
    for key in ("scenario_vectors", "vector_pool", "training_scenario_vectors"):
        vectors = pool_stats.get(key)
        if isinstance(vectors, list):
            return [vector for vector in vectors if isinstance(vector, MappingABC)]
    return []


def _success_return(stats: Optional[Mapping[str, Any]], keys: Sequence[str]) -> Optional[float]:
    if not isinstance(stats, MappingABC):
        return None
    for key in keys:
        value = stats.get(key)
        if isinstance(value, MappingABC):
            for nested_key in ("own", "mean", "value"):
                if nested_key in value:
                    return _float(value[nested_key])
        if value is not None:
            return _float(value)
    value = stats.get("total_team_reward")
    if isinstance(value, MappingABC) and value.get("own") is not None:
        return _float(value["own"])
    return None


def _own_team_return(summary: Mapping[str, Any]) -> float:
    total_team_reward = summary.get("total_team_reward") if isinstance(summary.get("total_team_reward"), MappingABC) else {}
    if total_team_reward.get("own") is not None:
        return _float(total_team_reward.get("own"))
    mean_team_reward = summary.get("mean_team_reward") if isinstance(summary.get("mean_team_reward"), MappingABC) else {}
    return _float(mean_team_reward.get("own"))


def _lookup(stats: Optional[Mapping[str, Any]], keys: Sequence[str]) -> Optional[float]:
    if not isinstance(stats, MappingABC):
        return None
    for key in keys:
        if key in stats:
            return _clamp01(_float(stats[key]))
    return None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, MappingABC) and isinstance(result.get(key), MappingABC):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _jsonable_submetrics(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _jsonable_submetrics(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable_submetrics(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_submetrics(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return value


def _mean(values: Iterable[float]) -> float:
    vals = [_float(value) for value in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _clamp01(value: Any) -> float:
    value = _float(value)
    return max(0.0, min(1.0, value))


def _round01(value: Any) -> float:
    return round(_clamp01(value), 6)


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value
