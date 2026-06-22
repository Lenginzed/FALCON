"""Lightweight dual-boundary surrogate for offline FSN hardness selection."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


SCHEMA_VERSION = "falcon.fsn_hardness_surrogate.v1"

SCENARIO_FACTORS = (
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

FAILURE_KEYS = (
    "coordination_failure",
    "target_assignment_confusion",
    "initial_disadvantage",
    "generalization_failure",
    "failure_severity",
)

PROXY_KEYS = (
    "initial_disadvantage_proxy",
    "formation_stress",
    "heading_aspect_stress",
    "altitude_velocity_stress",
    "target_assignment_ambiguity_proxy",
    "runtime_pool_novelty",
    "distance_to_accepted_pool",
    "distance_to_too_easy_pool",
    "distance_to_not_solvable_pool",
)

TARGET_KEYS = (
    "W_current",
    "W_best",
    "learning_potential",
    "too_easy",
    "not_solvable",
    "accepted",
)


@dataclass
class SurrogateTrainingConfig:
    hidden_dim: int = 96
    epochs: int = 220
    batch_size: int = 128
    learning_rate: float = 0.002
    seed: int = 17


class DualBoundarySurrogateNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(TARGET_KEYS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


class SurrogateCodec:
    def __init__(self) -> None:
        self.factor_vocab = list(SCENARIO_FACTORS)
        self.failure_keys = list(FAILURE_KEYS)
        self.proxy_keys = list(PROXY_KEYS)
        self.feature_names: List[str] = []
        self.mean: List[float] = []
        self.std: List[float] = []
        self.accepted_pool: List[Dict[str, float]] = []
        self.too_easy_pool: List[Dict[str, float]] = []
        self.not_solvable_pool: List[Dict[str, float]] = []

    def fit(self, samples: Sequence[Mapping[str, Any]]) -> "SurrogateCodec":
        self.accepted_pool = [
            dict(sample.get("scenario_vector") or {})
            for sample in samples
            if sample.get("accepted") and sample.get("scenario_vector")
        ]
        self.too_easy_pool = [
            dict(sample.get("scenario_vector") or {})
            for sample in samples
            if sample.get("too_easy") and sample.get("scenario_vector")
        ]
        self.not_solvable_pool = [
            dict(sample.get("scenario_vector") or {})
            for sample in samples
            if sample.get("not_solvable") and sample.get("scenario_vector")
        ]
        raw = [self._raw_features(sample) for sample in samples]
        self.feature_names = list(raw[0].keys()) if raw else []
        values = [
            [float(row.get(name, 0.0)) for name in self.feature_names]
            for row in raw
        ]
        if not values:
            self.mean = []
            self.std = []
            return self
        tensor = torch.tensor(values, dtype=torch.float32)
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean = mean.tolist()
        self.std = std.tolist()
        return self

    def encode(self, sample: Mapping[str, Any]) -> List[float]:
        raw = self._raw_features(sample)
        if not self.feature_names:
            self.feature_names = list(raw.keys())
            self.mean = [0.0 for _ in self.feature_names]
            self.std = [1.0 for _ in self.feature_names]
        encoded = []
        for index, name in enumerate(self.feature_names):
            value = float(raw.get(name, 0.0))
            mean = self.mean[index] if index < len(self.mean) else 0.0
            std = self.std[index] if index < len(self.std) else 1.0
            encoded.append((value - mean) / max(std, 1e-6))
        return encoded

    def _raw_features(self, sample: Mapping[str, Any]) -> Dict[str, float]:
        vector = sample.get("scenario_vector") or {}
        failure = sample.get("failure_vector") or {}
        changed = set(sample.get("changed_factors") or [])
        proxy = _hardness_proxy_components(
            vector,
            sample.get("initial_config") or {},
            sample.get("runtime_pool_vectors") or [],
        )
        proxy["distance_to_accepted_pool"] = _distance_to_pool(
            vector, self.accepted_pool
        )
        proxy["distance_to_too_easy_pool"] = _distance_to_pool(
            vector, self.too_easy_pool
        )
        proxy["distance_to_not_solvable_pool"] = _distance_to_pool(
            vector, self.not_solvable_pool
        )
        features: Dict[str, float] = {}
        for factor in self.factor_vocab:
            features[f"scenario.{factor}"] = _normalize_factor(
                factor, vector.get(factor)
            )
        for key in self.failure_keys:
            features[f"failure.{key}"] = _clip01(_number(failure.get(key), 0.0))
        for factor in self.factor_vocab:
            features[f"changed.{factor}"] = 1.0 if factor in changed else 0.0
        for key in self.proxy_keys:
            features[f"proxy.{key}"] = _clip01(proxy.get(key, 0.0))
        return features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_vocab": self.factor_vocab,
            "failure_keys": self.failure_keys,
            "proxy_keys": self.proxy_keys,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "accepted_pool": self.accepted_pool,
            "too_easy_pool": self.too_easy_pool,
            "not_solvable_pool": self.not_solvable_pool,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SurrogateCodec":
        codec = cls()
        codec.factor_vocab = list(data.get("factor_vocab") or SCENARIO_FACTORS)
        codec.failure_keys = list(data.get("failure_keys") or FAILURE_KEYS)
        codec.proxy_keys = list(data.get("proxy_keys") or PROXY_KEYS)
        codec.feature_names = list(data.get("feature_names") or [])
        codec.mean = [float(value) for value in data.get("mean") or []]
        codec.std = [float(value) for value in data.get("std") or []]
        codec.accepted_pool = [
            dict(item) for item in data.get("accepted_pool") or []
        ]
        codec.too_easy_pool = [
            dict(item) for item in data.get("too_easy_pool") or []
        ]
        codec.not_solvable_pool = [
            dict(item) for item in data.get("not_solvable_pool") or []
        ]
        return codec


class DualBoundarySurrogate:
    def __init__(self, model: DualBoundarySurrogateNet, codec: SurrogateCodec) -> None:
        self.model = model
        self.codec = codec
        self.model.eval()

    def predict_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            features = torch.tensor([self.codec.encode(sample)], dtype=torch.float32)
            values = self.model(features)[0].detach().cpu().tolist()
        return {
            "predicted_W_current": round(float(values[0]), 6),
            "predicted_W_best": round(float(values[1]), 6),
            "predicted_learning_potential": round(float(values[2]), 6),
            "predicted_too_easy_probability": round(float(values[3]), 6),
            "predicted_not_solvable_probability": round(float(values[4]), 6),
            "predicted_accepted_probability": round(float(values[5]), 6),
            "proxy_features": _round_mapping(
                _hardness_proxy_components(
                    sample.get("scenario_vector") or {},
                    sample.get("initial_config") or {},
                    sample.get("runtime_pool_vectors") or [],
                )
            ),
        }

    def predict_candidate(
        self,
        candidate: Mapping[str, Any],
        failure_summary: Optional[Mapping[str, Any]] = None,
        pool_stats: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        sample = sample_from_candidate(
            candidate,
            failure_summary=failure_summary,
            pool_stats=pool_stats,
        )
        return self.predict_sample(sample)


def sample_from_candidate(
    candidate: Mapping[str, Any],
    failure_summary: Optional[Mapping[str, Any]] = None,
    pool_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "scenario_vector": dict(candidate.get("scenario_vector") or {}),
        "initial_config": dict(candidate.get("initial_config") or {}),
        "failure_vector": dict((failure_summary or {}).get("failure_scores") or {}),
        "changed_factors": list(candidate.get("changed_factors") or []),
        "runtime_pool_vectors": _pool_vectors(pool_stats or {}),
    }


def collect_surrogate_samples(
    fsn_dataset_path: Optional[Path] = None,
    candidate_record_paths: Optional[Sequence[Path]] = None,
    failure_summary_paths: Optional[Sequence[Path]] = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if fsn_dataset_path and fsn_dataset_path.exists():
        for item in _read_jsonl(fsn_dataset_path):
            sample = _sample_from_fsn_dataset_item(item)
            if sample:
                samples.append(sample)
    failure_by_id = _load_failure_by_id(failure_summary_paths or [])
    for path in candidate_record_paths or []:
        if not path.exists():
            continue
        payload = _read_json(path)
        for record in payload.get("candidate_records") or []:
            sample = _sample_from_candidate_record(record, failure_by_id)
            if sample:
                samples.append(sample)
    return samples


def train_surrogate(
    samples: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: Optional[SurrogateTrainingConfig] = None,
) -> Dict[str, Any]:
    cfg = config or SurrogateTrainingConfig()
    if not samples:
        raise ValueError("No surrogate training samples were provided.")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    splits = _split_samples(samples, cfg.seed)
    codec = SurrogateCodec().fit(splits["train"])
    train_x, train_y, train_w = _tensors(splits["train"], codec)
    model = DualBoundarySurrogateNet(
        input_dim=train_x.shape[1], hidden_dim=cfg.hidden_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    losses: List[float] = []
    for _epoch in range(max(int(cfg.epochs), 1)):
        permutation = torch.randperm(train_x.shape[0])
        epoch_losses = []
        for start in range(0, train_x.shape[0], max(int(cfg.batch_size), 1)):
            indices = permutation[start : start + int(cfg.batch_size)]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            weights = train_w[indices].unsqueeze(1)
            pred = model(batch_x)
            mse = ((pred[:, :3] - batch_y[:, :3]) ** 2) * weights
            bce = nn.functional.binary_cross_entropy(
                pred[:, 3:], batch_y[:, 3:], reduction="none"
            ) * weights
            loss = mse.mean() + bce.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / max(len(epoch_losses), 1))
    metrics = {
        name: _evaluate_split(model, codec, rows)
        for name, rows in splits.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "input_dim": train_x.shape[1],
        "hidden_dim": cfg.hidden_dim,
        "codec": codec.to_dict(),
        "target_keys": list(TARGET_KEYS),
        "training_config": cfg.__dict__,
        "metrics": metrics,
    }
    model_path = output_dir / "fsn_hardness_surrogate_model.pt"
    torch.save(payload, model_path)
    summary = {
        "schema_version": "falcon.fsn_hardness_surrogate_training_summary.v1",
        "model_path": str(model_path.resolve()),
        "num_samples": len(samples),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "label_counts": dict(Counter(str(item.get("label")) for item in samples)),
        "source_counts": dict(Counter(str(item.get("source")) for item in samples)),
        "feature_count": len(codec.feature_names),
        "feature_names": codec.feature_names,
        "final_training_loss": round(losses[-1], 6) if losses else None,
        "metrics": metrics,
        "warnings": _training_warnings(metrics, splits),
    }
    (output_dir / "fsn_hardness_surrogate_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_metrics_csv(output_dir / "fsn_hardness_surrogate_metrics.csv", metrics)
    return summary


def load_hardness_surrogate(path: str | Path) -> DualBoundarySurrogate:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    codec = SurrogateCodec.from_dict(payload["codec"])
    model = DualBoundarySurrogateNet(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 96)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return DualBoundarySurrogate(model, codec)


def score_candidate_with_surrogate(
    candidate: Mapping[str, Any],
    surrogate: DualBoundarySurrogate,
    failure_summary: Optional[Mapping[str, Any]] = None,
    pool_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return surrogate.predict_candidate(candidate, failure_summary, pool_stats)


def _sample_from_fsn_dataset_item(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    vector = item.get("candidate_scenario_vector") or {}
    policy = item.get("policy_eval") or {}
    difficulty = item.get("difficulty") or {}
    if not vector:
        return None
    reasons = set(difficulty.get("rejection_reasons") or [])
    label = str(item.get("label") or "")
    accepted = label == "accepted" or (
        bool(item.get("constraint_valid")) and not reasons and difficulty
    )
    w_current = _clip01(_number(policy.get("W_current"), 0.0))
    w_best = _clip01(_number(policy.get("W_best"), 0.0))
    return {
        "source": "fsn_dataset",
        "sample_id": item.get("sample_id"),
        "seed": item.get("seed"),
        "split": item.get("split"),
        "label": "accepted" if accepted else label,
        "scenario_vector": dict(vector),
        "failure_vector": dict(item.get("failure_vector") or {}),
        "changed_factors": list(item.get("changed_factors") or []),
        "W_current": w_current,
        "W_best": w_best,
        "learning_potential": _clip01(
            _number(difficulty.get("learning_potential"), max(w_best - w_current, 0.0))
        ),
        "too_easy": (
            "too_easy_for_current_policy" in reasons
            or label == "rejected_too_easy"
        ),
        "not_solvable": (
            "not_solvable_by_historical_best_policy" in reasons
            or label == "rejected_not_solvable"
        ),
        "accepted": accepted,
        "sample_weight": _number(item.get("sample_weight"), 1.0),
    }


def _sample_from_candidate_record(
    record: Mapping[str, Any],
    failure_by_id: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidate = record.get("candidate") or {}
    vector = candidate.get("scenario_vector") or {}
    difficulty = record.get("difficulty_result") or {}
    if not vector or not difficulty:
        return None
    current = record.get("current_policy_eval") or {}
    best = record.get("best_policy_eval") or {}
    reasons = set(difficulty.get("rejection_reasons") or [])
    accepted = bool(difficulty.get("accepted_into_curriculum_pool"))
    failure = failure_by_id.get(str(record.get("failure_id"))) or {}
    w_current = _clip01(_number(current.get("win_rate"), 0.0))
    w_best = _clip01(_number(best.get("win_rate"), 0.0))
    return {
        "source": str(record.get("mode") or record.get("generator_type") or "candidate_record"),
        "sample_id": record.get("scenario_id") or candidate.get("scenario_id"),
        "seed": record.get("seed"),
        "round_id": record.get("round_id"),
        "label": "accepted" if accepted else "rejected",
        "scenario_vector": dict(vector),
        "initial_config": dict(candidate.get("initial_config") or {}),
        "failure_vector": dict((failure.get("failure_summary") or failure).get("failure_scores") or {}),
        "changed_factors": list(candidate.get("changed_factors") or []),
        "W_current": w_current,
        "W_best": w_best,
        "learning_potential": _clip01(
            _number(difficulty.get("learning_potential"), max(w_best - w_current, 0.0))
        ),
        "too_easy": "too_easy_for_current_policy" in reasons,
        "not_solvable": "not_solvable_by_historical_best_policy" in reasons,
        "accepted": accepted,
        "sample_weight": 1.0 + _number(difficulty.get("final_value_score"), 0.0)
        if accepted
        else 0.5,
    }


def _split_samples(
    samples: Sequence[Mapping[str, Any]], seed: int
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped = {"train": [], "val": [], "test": []}
    for sample in samples:
        sample_seed = sample.get("seed")
        if sample_seed in (0, "0", 1, "1", 2, "2"):
            grouped["train"].append(sample)
        elif sample_seed in (3, "3"):
            grouped["val"].append(sample)
        elif sample_seed in (4, "4"):
            grouped["test"].append(sample)
    if min(len(grouped["train"]), len(grouped["val"]), len(grouped["test"])) > 0:
        return grouped
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_end = max(int(len(shuffled) * 0.70), 1)
    val_end = max(int(len(shuffled) * 0.85), train_end + 1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _tensors(
    samples: Sequence[Mapping[str, Any]], codec: SurrogateCodec
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([codec.encode(sample) for sample in samples], dtype=torch.float32)
    y = torch.tensor(
        [
            [
                _clip01(_number(sample.get("W_current"), 0.0)),
                _clip01(_number(sample.get("W_best"), 0.0)),
                _clip01(_number(sample.get("learning_potential"), 0.0)),
                1.0 if sample.get("too_easy") else 0.0,
                1.0 if sample.get("not_solvable") else 0.0,
                1.0 if sample.get("accepted") else 0.0,
            ]
            for sample in samples
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [max(_number(sample.get("sample_weight"), 1.0), 0.05) for sample in samples],
        dtype=torch.float32,
    )
    return x, y, weights


def _evaluate_split(
    model: DualBoundarySurrogateNet,
    codec: SurrogateCodec,
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not samples:
        return {"num_samples": 0}
    x, y, _weights = _tensors(samples, codec)
    with torch.no_grad():
        pred = model(x).detach().cpu()
    result: Dict[str, Any] = {"num_samples": len(samples)}
    for index, key in enumerate(TARGET_KEYS[:3]):
        result[f"{key}_mae"] = round(
            torch.mean(torch.abs(pred[:, index] - y[:, index])).item(), 6
        )
    for offset, key in enumerate(TARGET_KEYS[3:], start=3):
        metrics = _binary_metrics(y[:, offset].tolist(), pred[:, offset].tolist())
        for metric_key, value in metrics.items():
            result[f"{key}_{metric_key}"] = value
    return result


def _binary_metrics(targets: Sequence[float], scores: Sequence[float]) -> Dict[str, Any]:
    labels = [1 if value >= 0.5 else 0 for value in targets]
    preds = [1 if value >= 0.5 else 0 for value in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "auc": _auc(labels, scores),
        "positive_rate": round(sum(labels) / max(len(labels), 1), 6),
    }


def _auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = [(score, label) for score, label in zip(scores, labels) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 6)


def _hardness_proxy_components(
    scenario_vector: Mapping[str, Any],
    initial_config: Mapping[str, Any],
    runtime_pool_vectors: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    own_spread = _number(scenario_vector.get("own_formation_spread"), 3000.0)
    opponent_spread = _number(scenario_vector.get("opponent_formation_spread"), 3000.0)
    formation_span = FACTOR_RANGES["own_formation_spread"][1] - FACTOR_RANGES["own_formation_spread"][0]
    formation_stress = _clip01(
        0.6 * (own_spread - 1000.0) / formation_span
        + 0.4 * (8000.0 - opponent_spread) / formation_span
    )
    heading = _number(scenario_vector.get("heading_difference"), math.pi)
    aspect = _number(scenario_vector.get("approximate_aspect_angle"), math.pi)
    heading_aspect_stress = _clip01(
        0.5 * abs(heading - math.pi) / math.pi
        + 0.5 * abs(aspect - math.pi) / math.pi
    )
    altitude_velocity_stress = _clip01(
        0.5 * abs(_number(scenario_vector.get("altitude_difference"), 0.0)) / 2500.0
        + 0.5 * abs(_number(scenario_vector.get("velocity_difference"), 0.0)) / 60.0
    )
    distance = _number(scenario_vector.get("team_center_distance"), 12000.0)
    distance_disadvantage = _clip01(abs(distance - 12000.0) / 6000.0)
    initial_disadvantage_proxy = _clip01(
        0.25 * distance_disadvantage
        + 0.20 * formation_stress
        + 0.20 * altitude_velocity_stress
        + 0.20 * heading_aspect_stress
    )
    return {
        "initial_disadvantage_proxy": initial_disadvantage_proxy,
        "formation_stress": formation_stress,
        "heading_aspect_stress": heading_aspect_stress,
        "altitude_velocity_stress": altitude_velocity_stress,
        "target_assignment_ambiguity_proxy": _target_assignment_ambiguity(initial_config),
        "runtime_pool_novelty": _distance_to_pool(scenario_vector, runtime_pool_vectors, default=0.5),
    }


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
                        (_number(own_position[index], 0.0) - _number(position[index], 0.0)) ** 2
                        for index in range(min(len(own_position), len(position)))
                    )
                )
            )
        if len(distances) >= 2:
            scores.append(1.0 - abs(distances[0] - distances[1]) / max(distances[0] + distances[1], 1e-8))
    return _clip01(sum(scores) / len(scores)) if scores else 0.0


def _distance_to_pool(
    vector: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    default: float = 1.0,
) -> float:
    distances = [
        _normalized_vector_distance(vector, item)
        for item in pool
        if isinstance(item, Mapping)
    ]
    return _clip01(min(distances)) if distances else default


def _normalized_vector_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    parts = []
    for key in SCENARIO_FACTORS:
        low, high = FACTOR_RANGES[key]
        span = max(high - low, 1e-8)
        parts.append(((_number(left.get(key), low) - _number(right.get(key), low)) / span) ** 2)
    return math.sqrt(sum(parts) / max(len(parts), 1))


def _normalize_factor(key: str, value: Any) -> float:
    low, high = FACTOR_RANGES[key]
    return _clip01((_number(value, low) - low) / max(high - low, 1e-8))


def _pool_vectors(pool_stats: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("scenario_vectors", "vectors", "training_scenario_vectors", "vector_pool"):
        value = pool_stats.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _load_failure_by_id(paths: Sequence[Path]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = _read_json(path)
        for item in payload.get("failure_summaries") or []:
            result[str(item.get("failure_id"))] = item
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _clip01(value: Any) -> float:
    return min(max(_number(value, 0.0), 0.0), 1.0)


def _round_mapping(values: Mapping[str, Any]) -> Dict[str, float]:
    return {key: round(_number(value), 6) for key, value in values.items()}


def _write_metrics_csv(path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    rows = []
    for split, values in metrics.items():
        row = {"split": split}
        row.update(values)
        rows.append(row)
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _training_warnings(
    metrics: Mapping[str, Any], splits: Mapping[str, Sequence[Mapping[str, Any]]]
) -> List[str]:
    warnings = []
    if min(len(value) for value in splits.values()) == 0:
        warnings.append("At least one split is empty; surrogate metrics are incomplete.")
    accepted_auc = ((metrics.get("test") or {}).get("accepted_auc"))
    if accepted_auc is not None and accepted_auc < 0.6:
        warnings.append("Accepted-probability AUC is weak; use surrogate as a ranking proxy only.")
    if ((metrics.get("test") or {}).get("learning_potential_mae") or 1.0) > 0.3:
        warnings.append("Learning-potential MAE is high; policy evaluation remains mandatory.")
    return warnings
