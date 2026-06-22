"""Offline trainer for the lightweight Failure-to-Scenario Network."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .fsn_dataset import (
    FAILURE_KEYS,
    LABELS,
    POLICY_KEYS,
    PROXY_FEATURE_KEYS,
    load_jsonl,
)
from .fsn_model import FSNModelConfig, FailureToScenarioNetwork
from .trajectory_recorder import SCENARIO_VECTOR_KEYS


@dataclass
class FSNTrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 96
    dropout: float = 0.05
    seed: int = 7
    lambda_scenario: float = 1.0
    lambda_value: float = 0.35
    lambda_factor: float = 0.25
    lambda_label: float = 0.35
    lambda_constraint: float = 0.35
    lambda_rank: float = 0.0
    use_constraint_head: bool = False


class FSNFeatureCodec:
    """Fit numeric normalization and categorical vocabularies on train data."""

    def __init__(self) -> None:
        self.input_keys = [
            *(f"failure_vector.{key}" for key in FAILURE_KEYS),
            *(f"policy_eval.{key}" for key in POLICY_KEYS),
            *(f"proxy_features.{key}" for key in PROXY_FEATURE_KEYS),
        ]
        self.scenario_keys = list(SCENARIO_VECTOR_KEYS)
        self.factor_vocab: List[str] = []
        self.label_vocab = list(LABELS)
        self.input_mean: List[float] = []
        self.input_std: List[float] = []
        self.scenario_mean: List[float] = []
        self.scenario_std: List[float] = []

    def fit(self, samples: Sequence[Mapping[str, Any]]) -> "FSNFeatureCodec":
        self.factor_vocab = sorted(
            {
                str(factor)
                for item in samples
                for factor in item.get("changed_factors") or []
            }
        )
        input_columns = [
            [_feature_value(item, key) for item in samples] for key in self.input_keys
        ]
        scenario_columns = [
            [
                _nullable_float(
                    (item.get("candidate_scenario_vector") or {}).get(key)
                )
                for item in samples
            ]
            for key in self.scenario_keys
        ]
        self.input_mean, self.input_std = _fit_columns(input_columns)
        self.scenario_mean, self.scenario_std = _fit_columns(scenario_columns)
        return self

    def encode(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        inputs = []
        for index, key in enumerate(self.input_keys):
            value = _feature_value(sample, key)
            value = self.input_mean[index] if value is None else value
            inputs.append((value - self.input_mean[index]) / self.input_std[index])

        scenario = []
        scenario_mask = []
        vector = dict(sample.get("candidate_scenario_vector") or {})
        for index, key in enumerate(self.scenario_keys):
            value = _nullable_float(vector.get(key))
            scenario_mask.append(0.0 if value is None else 1.0)
            value = self.scenario_mean[index] if value is None else value
            scenario.append(
                (value - self.scenario_mean[index]) / self.scenario_std[index]
            )

        factors = set(str(value) for value in sample.get("changed_factors") or [])
        factor_target = [1.0 if key in factors else 0.0 for key in self.factor_vocab]
        label = str(sample.get("label") or "invalid")
        label_index = (
            self.label_vocab.index(label) if label in self.label_vocab else 0
        )
        value_score = _nullable_float(
            (sample.get("difficulty") or {}).get("final_value_score")
        )
        synthetic_invalid = bool(sample.get("synthetic")) and label == "invalid"
        sample_weight = float(sample.get("sample_weight") or 1.0)
        return {
            "features": inputs,
            "scenario": scenario,
            "scenario_mask": scenario_mask,
            "value": 0.0 if value_score is None else value_score,
            "value_mask": 0.0
            if value_score is None or synthetic_invalid
            else 1.0,
            "factors": factor_target,
            "label": label_index,
            "constraint_valid": 1.0
            if bool(sample.get("constraint_valid"))
            else 0.0,
            "sample_weight": sample_weight,
            "regression_weight": 0.0
            if label == "invalid"
            else sample_weight,
            "factor_weight": 0.0 if synthetic_invalid else sample_weight,
            "sample_id": sample.get("sample_id"),
        }

    def decode_scenario(self, normalized: Sequence[float]) -> Dict[str, float]:
        return {
            key: float(normalized[index]) * self.scenario_std[index]
            + self.scenario_mean[index]
            for index, key in enumerate(self.scenario_keys)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_keys": self.input_keys,
            "scenario_keys": self.scenario_keys,
            "factor_vocab": self.factor_vocab,
            "label_vocab": self.label_vocab,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "scenario_mean": self.scenario_mean,
            "scenario_std": self.scenario_std,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FSNFeatureCodec":
        codec = cls()
        for key in (
            "input_keys",
            "scenario_keys",
            "factor_vocab",
            "label_vocab",
            "input_mean",
            "input_std",
            "scenario_mean",
            "scenario_std",
        ):
            setattr(codec, key, list(data[key]))
        return codec


class _FSNDataset(Dataset):
    def __init__(
        self, samples: Sequence[Mapping[str, Any]], codec: FSNFeatureCodec
    ) -> None:
        self.rows = [codec.encode(item) for item in samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            "scenario": torch.tensor(row["scenario"], dtype=torch.float32),
            "scenario_mask": torch.tensor(
                row["scenario_mask"], dtype=torch.float32
            ),
            "value": torch.tensor(row["value"], dtype=torch.float32),
            "value_mask": torch.tensor(row["value_mask"], dtype=torch.float32),
            "factors": torch.tensor(row["factors"], dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "constraint_valid": torch.tensor(
                row["constraint_valid"], dtype=torch.float32
            ),
            "sample_weight": torch.tensor(
                row["sample_weight"], dtype=torch.float32
            ),
            "regression_weight": torch.tensor(
                row["regression_weight"], dtype=torch.float32
            ),
            "factor_weight": torch.tensor(
                row["factor_weight"], dtype=torch.float32
            ),
        }


class FSNTrainer:
    def __init__(
        self,
        config: Optional[FSNTrainingConfig | Mapping[str, Any]] = None,
    ) -> None:
        if config is None:
            config = FSNTrainingConfig()
        elif not isinstance(config, FSNTrainingConfig):
            config = FSNTrainingConfig(**dict(config))
        self.config = config
        self.codec = FSNFeatureCodec()
        self.model: Optional[FailureToScenarioNetwork] = None

    def train(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        checkpoint_name: str = "fsn_offline_smoke.pt",
        summary_name: str = "fsn_offline_training_summary.json",
        stage_name: str = "stage1",
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        _set_seed(self.config.seed)
        samples = load_jsonl(dataset_path)
        split_samples = {
            split: [item for item in samples if item.get("split") == split]
            for split in ("train", "val", "test")
        }
        train_samples = split_samples["train"]
        if not train_samples:
            raise ValueError("FSN dataset has no training samples")
        self.codec.fit(train_samples)
        model_config = FSNModelConfig(
            input_dim=len(self.codec.input_keys),
            scenario_dim=len(self.codec.scenario_keys),
            factor_dim=len(self.codec.factor_vocab),
            label_dim=len(self.codec.label_vocab),
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            constraint_head=self.config.use_constraint_head,
        )
        self.model = FailureToScenarioNetwork(model_config)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = DataLoader(
            _FSNDataset(train_samples, self.codec),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        history: List[Dict[str, Any]] = []
        best_val = float("inf")
        best_selection_score = float("inf")
        best_epoch = None
        best_state = None
        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(split_samples["val"])
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
            val_loss = val_metrics.get("total_loss")
            if val_loss is not None:
                best_val = min(best_val, float(val_loss))
            selection_score = _checkpoint_selection_score(val_metrics)
            if (
                selection_score is not None
                and selection_score < best_selection_score
            ):
                best_selection_score = selection_score
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
        if best_state is not None:
            self.model.load_state_dict(best_state)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / checkpoint_name
        torch.save(
            {
                "schema_version": "falcon.fsn_checkpoint.v1",
                "model_config": model_config.to_dict(),
                "model_state_dict": self.model.state_dict(),
                "codec": self.codec.to_dict(),
                "training_config": asdict(self.config),
                "factor_vocab": self.codec.factor_vocab,
                "label_vocab": self.codec.label_vocab,
            },
            checkpoint_path,
        )
        split_metrics = {
            split: self.evaluate(rows)
            for split, rows in split_samples.items()
        }
        train_loss = split_metrics["train"].get("total_loss")
        val_loss = split_metrics["val"].get("total_loss")
        overfit_gap = (
            None
            if train_loss is None or val_loss is None
            else float(val_loss) - float(train_loss)
        )
        overfitting_detected = bool(
            overfit_gap is not None
            and overfit_gap > max(0.15, abs(float(train_loss)) * 0.5)
        )
        warnings: List[str] = []
        if stage_name == "stage1":
            warnings.extend(
                [
                    "Stage 1 may contain scenario-vector leakage; metrics are smoke-only.",
                    "Invalid samples are scarce, so constraint_checker remains mandatory.",
                ]
            )
        else:
            warnings.extend(
                [
                    "Stage 2 is an offline distillation smoke and does not establish policy improvement.",
                    "constraint_checker remains mandatory after FSN inference.",
                ]
            )
        if overfitting_detected:
            warnings.append(
                "Validation loss is materially above training loss; possible overfitting was detected."
            )
        summary = {
            "schema_version": f"falcon.fsn_{stage_name}_training_summary.v1",
            "stage": stage_name,
            "training_succeeded": True,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "device": "cpu",
            "model_config": model_config.to_dict(),
            "training_config": asdict(self.config),
            "split_counts": {
                split: len(rows) for split, rows in split_samples.items()
            },
            "split_metrics": split_metrics,
            "factor_vocab": self.codec.factor_vocab,
            "label_vocab": self.codec.label_vocab,
            "best_val_loss": None if best_val == float("inf") else round(best_val, 6),
            "selected_epoch": best_epoch,
            "checkpoint_selection_score": None
            if best_selection_score == float("inf")
            else round(best_selection_score, 6),
            "checkpoint_selection_rule": (
                "validation_total_loss minus balanced label/factor/constraint F1 credits"
            ),
            "overfitting_detected": overfitting_detected,
            "train_validation_loss_gap": None
            if overfit_gap is None
            else round(overfit_gap, 6),
            "history": history,
            "ranking_loss_implemented": False,
            "ranking_loss_interface_available": True,
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "warnings": warnings
            + [
                "Pairwise ranking loss is reserved by lambda_rank but disabled."
            ],
        }
        (output_dir / summary_name).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def evaluate(self, samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not samples or self.model is None:
            return {
                "total_loss": None,
                "label_accuracy": None,
                "value_mae": None,
                "scenario_normalized_mae": None,
                "label_macro_f1": None,
                "changed_factor_micro_f1": None,
                "changed_factor_macro_f1": None,
                "constraint_accuracy": None,
                "constraint_f1": None,
                "sample_count": len(samples),
            }
        loader = DataLoader(
            _FSNDataset(samples, self.codec),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        return self._run_epoch(loader, optimizer=None)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, Any]:
        assert self.model is not None
        training = optimizer is not None
        self.model.train(training)
        totals: List[float] = []
        value_errors: List[float] = []
        scenario_errors: List[float] = []
        scenario_errors_by_feature: List[List[float]] = [
            [] for _ in self.codec.scenario_keys
        ]
        correct = 0
        count = 0
        all_labels: List[int] = []
        all_label_predictions: List[int] = []
        all_factor_targets: List[List[int]] = []
        all_factor_predictions: List[List[int]] = []
        all_constraint_targets: List[int] = []
        all_constraint_predictions: List[int] = []
        component_totals = CounterFloat()
        for batch in loader:
            features = batch["features"].float()
            scenario = batch["scenario"].float()
            scenario_mask = batch["scenario_mask"].float()
            value = batch["value"].float()
            value_mask = batch["value_mask"].float()
            factors = batch["factors"].float()
            labels = batch["label"].long()
            constraint_valid = batch["constraint_valid"].float()
            sample_weight = batch["sample_weight"].float()
            regression_weight = batch["regression_weight"].float()
            factor_weight = batch["factor_weight"].float()
            if training:
                optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                outputs = self.model(features)
                losses = self._loss(
                    outputs,
                    scenario,
                    scenario_mask,
                    value,
                    value_mask,
                    factors,
                    labels,
                    constraint_valid,
                    sample_weight,
                    regression_weight,
                    factor_weight,
                )
                if training:
                    losses["total"].backward()
                    optimizer.step()
            totals.append(float(losses["total"].detach()))
            for key in (
                "scenario",
                "value",
                "factor",
                "label",
                "constraint",
                "rank",
            ):
                component_totals.add(key, float(losses[key].detach()))
            predictions = outputs["label_logits"].argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            count += int(labels.numel())
            all_labels.extend(labels.detach().cpu().tolist())
            all_label_predictions.extend(predictions.detach().cpu().tolist())
            factor_predictions = (
                torch.sigmoid(outputs["changed_factor_logits"]) >= 0.5
            ).int()
            all_factor_targets.extend(factors.int().detach().cpu().tolist())
            all_factor_predictions.extend(
                factor_predictions.detach().cpu().tolist()
            )
            if "constraint_valid_probability" in outputs:
                constraint_predictions = (
                    outputs["constraint_valid_probability"] >= 0.5
                ).int()
                all_constraint_targets.extend(
                    constraint_valid.int().detach().cpu().tolist()
                )
                all_constraint_predictions.extend(
                    constraint_predictions.detach().cpu().tolist()
                )
            if value_mask.sum() > 0:
                value_errors.extend(
                    torch.abs(outputs["value"] - value)[value_mask > 0]
                    .detach()
                    .cpu()
                    .tolist()
                )
            mask = (scenario_mask > 0) & (regression_weight[:, None] > 0)
            if mask.any():
                scenario_errors.extend(
                    torch.abs(outputs["scenario_vector"] - scenario)[mask]
                    .detach()
                    .cpu()
                    .tolist()
                )
                absolute_normalized = torch.abs(
                    outputs["scenario_vector"] - scenario
                )
                for index, _key in enumerate(self.codec.scenario_keys):
                    feature_mask = mask[:, index]
                    if feature_mask.any():
                        raw_errors = (
                            absolute_normalized[:, index][feature_mask]
                            * float(self.codec.scenario_std[index])
                        )
                        scenario_errors_by_feature[index].extend(
                            raw_errors.detach().cpu().tolist()
                        )
        denominator = max(len(totals), 1)
        return {
            "total_loss": round(sum(totals) / denominator, 6) if totals else None,
            "scenario_loss": component_totals.mean("scenario", denominator),
            "value_loss": component_totals.mean("value", denominator),
            "factor_loss": component_totals.mean("factor", denominator),
            "label_loss": component_totals.mean("label", denominator),
            "constraint_loss": component_totals.mean(
                "constraint", denominator
            ),
            "rank_loss": component_totals.mean("rank", denominator),
            "label_accuracy": round(correct / max(count, 1), 6),
            "label_macro_f1": _macro_f1(
                all_labels,
                all_label_predictions,
                num_classes=len(self.codec.label_vocab),
            ),
            "changed_factor_micro_f1": _multilabel_micro_f1(
                all_factor_targets, all_factor_predictions
            ),
            "changed_factor_macro_f1": _multilabel_macro_f1(
                all_factor_targets, all_factor_predictions
            ),
            "constraint_accuracy": _accuracy(
                all_constraint_targets, all_constraint_predictions
            ),
            "constraint_f1": _binary_f1(
                all_constraint_targets, all_constraint_predictions
            ),
            "value_mae": round(float(np.mean(value_errors)), 6)
            if value_errors
            else None,
            "scenario_normalized_mae": round(
                float(np.mean(scenario_errors)), 6
            )
            if scenario_errors
            else None,
            "scenario_mae_by_feature": {
                key: round(float(np.mean(scenario_errors_by_feature[index])), 6)
                if scenario_errors_by_feature[index]
                else None
                for index, key in enumerate(self.codec.scenario_keys)
            },
            "sample_count": count,
        }

    def _loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        scenario: torch.Tensor,
        scenario_mask: torch.Tensor,
        value: torch.Tensor,
        value_mask: torch.Tensor,
        factors: torch.Tensor,
        labels: torch.Tensor,
        constraint_valid: torch.Tensor,
        sample_weight: torch.Tensor,
        regression_weight: torch.Tensor,
        factor_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scenario_squared = (
            (outputs["scenario_vector"] - scenario) ** 2 * scenario_mask
        ).sum(dim=-1) / scenario_mask.sum(dim=-1).clamp_min(1.0)
        scenario_loss = (
            scenario_squared * regression_weight
        ).sum() / regression_weight.sum().clamp_min(1.0)
        value_squared = (outputs["value"] - value) ** 2
        value_weights = sample_weight * value_mask
        value_loss = (
            value_squared * value_weights
        ).sum() / value_weights.sum().clamp_min(1.0)
        factor_per_sample = nn.functional.binary_cross_entropy_with_logits(
            outputs["changed_factor_logits"], factors, reduction="none"
        ).mean(dim=-1)
        factor_loss = (
            factor_per_sample * factor_weight
        ).sum() / factor_weight.sum().clamp_min(1.0)
        label_per_sample = nn.functional.cross_entropy(
            outputs["label_logits"], labels, reduction="none"
        )
        label_loss = (
            label_per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(1.0)
        if "constraint_valid_probability" in outputs:
            constraint_per_sample = nn.functional.binary_cross_entropy(
                outputs["constraint_valid_probability"],
                constraint_valid,
                reduction="none",
            )
            constraint_loss = (
                constraint_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(1.0)
        else:
            constraint_loss = label_loss * 0.0
        rank_loss = self.pairwise_ranking_loss(outputs["value"], value)
        total = (
            self.config.lambda_scenario * scenario_loss
            + self.config.lambda_value * value_loss
            + self.config.lambda_factor * factor_loss
            + self.config.lambda_label * label_loss
            + self.config.lambda_constraint * constraint_loss
            + self.config.lambda_rank * rank_loss
        )
        return {
            "total": total,
            "scenario": scenario_loss,
            "value": value_loss,
            "factor": factor_loss,
            "label": label_loss,
            "constraint": constraint_loss,
            "rank": rank_loss,
        }

    @staticmethod
    def pairwise_ranking_loss(
        predicted_value: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        """Stage-1 placeholder; lambda_rank remains zero."""

        return predicted_value.sum() * 0.0 + target_value.sum() * 0.0


class CounterFloat:
    def __init__(self) -> None:
        self.values: Dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        self.values[key] = self.values.get(key, 0.0) + value

    def mean(self, key: str, denominator: int) -> float:
        return round(self.values.get(key, 0.0) / max(denominator, 1), 6)


def load_fsn_checkpoint(
    checkpoint_path: str | Path,
) -> Tuple[FailureToScenarioNetwork, FSNFeatureCodec, Dict[str, Any]]:
    model, payload = FailureToScenarioNetwork.from_checkpoint(str(checkpoint_path))
    codec = FSNFeatureCodec.from_dict(payload["codec"])
    return model, codec, payload


def _feature_value(sample: Mapping[str, Any], dotted_key: str) -> Optional[float]:
    section, key = dotted_key.split(".", 1)
    return _nullable_float((sample.get(section) or {}).get(key))


def _fit_columns(
    columns: Sequence[Sequence[Optional[float]]],
) -> Tuple[List[float], List[float]]:
    means: List[float] = []
    stds: List[float] = []
    for column in columns:
        clean = [float(value) for value in column if value is not None]
        mean = float(np.mean(clean)) if clean else 0.0
        std = float(np.std(clean)) if clean else 1.0
        means.append(mean)
        stds.append(std if std > 1e-8 else 1.0)
    return means, stds


def _nullable_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _accuracy(targets: Sequence[int], predictions: Sequence[int]) -> Optional[float]:
    if not targets or len(targets) != len(predictions):
        return None
    return round(
        sum(int(a == b) for a, b in zip(targets, predictions)) / len(targets),
        6,
    )


def _binary_f1(
    targets: Sequence[int], predictions: Sequence[int]
) -> Optional[float]:
    if not targets or len(targets) != len(predictions):
        return None
    true_positive = sum(
        int(target == 1 and prediction == 1)
        for target, prediction in zip(targets, predictions)
    )
    false_positive = sum(
        int(target == 0 and prediction == 1)
        for target, prediction in zip(targets, predictions)
    )
    false_negative = sum(
        int(target == 1 and prediction == 0)
        for target, prediction in zip(targets, predictions)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return round(2 * true_positive / denominator, 6) if denominator else 0.0


def _macro_f1(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> Optional[float]:
    if not targets or len(targets) != len(predictions):
        return None
    scores = []
    for label in range(num_classes):
        binary_targets = [int(value == label) for value in targets]
        binary_predictions = [int(value == label) for value in predictions]
        score = _binary_f1(binary_targets, binary_predictions)
        scores.append(0.0 if score is None else score)
    return round(float(np.mean(scores)), 6) if scores else None


def _multilabel_micro_f1(
    targets: Sequence[Sequence[int]],
    predictions: Sequence[Sequence[int]],
) -> Optional[float]:
    if not targets or len(targets) != len(predictions):
        return None
    flat_targets = [value for row in targets for value in row]
    flat_predictions = [value for row in predictions for value in row]
    return _binary_f1(flat_targets, flat_predictions)


def _multilabel_macro_f1(
    targets: Sequence[Sequence[int]],
    predictions: Sequence[Sequence[int]],
) -> Optional[float]:
    if not targets or len(targets) != len(predictions) or not targets[0]:
        return None
    scores = []
    for index in range(len(targets[0])):
        score = _binary_f1(
            [row[index] for row in targets],
            [row[index] for row in predictions],
        )
        scores.append(0.0 if score is None else score)
    return round(float(np.mean(scores)), 6) if scores else None


def _checkpoint_selection_score(metrics: Mapping[str, Any]) -> Optional[float]:
    total_loss = _nullable_float(metrics.get("total_loss"))
    if total_loss is None:
        return None
    label_f1 = _nullable_float(metrics.get("label_macro_f1")) or 0.0
    factor_f1 = (
        _nullable_float(metrics.get("changed_factor_micro_f1")) or 0.0
    )
    constraint_f1 = _nullable_float(metrics.get("constraint_f1")) or 0.0
    return (
        total_loss
        - 0.50 * label_f1
        - 0.15 * factor_f1
        - 0.10 * constraint_f1
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
