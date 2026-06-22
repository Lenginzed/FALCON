"""Lightweight multi-head Failure-to-Scenario Network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


@dataclass
class FSNModelConfig:
    input_dim: int
    scenario_dim: int
    factor_dim: int
    label_dim: int
    hidden_dim: int = 96
    dropout: float = 0.05
    constraint_head: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureToScenarioNetwork(nn.Module):
    """Shared MLP encoder with scenario, value, factor, and label heads."""

    def __init__(self, config: FSNModelConfig | Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(config, FSNModelConfig):
            config = FSNModelConfig(**dict(config))
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.scenario_vector_head = nn.Linear(
            config.hidden_dim, config.scenario_dim
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.changed_factor_head = nn.Linear(
            config.hidden_dim, config.factor_dim
        )
        self.label_head = nn.Linear(config.hidden_dim, config.label_dim)
        self.constraint_head = (
            nn.Sequential(nn.Linear(config.hidden_dim, 1), nn.Sigmoid())
            if config.constraint_head
            else None
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(features)
        outputs = {
            "scenario_vector": self.scenario_vector_head(encoded),
            "value": self.value_head(encoded).squeeze(-1),
            "changed_factor_logits": self.changed_factor_head(encoded),
            "label_logits": self.label_head(encoded),
        }
        if self.constraint_head is not None:
            outputs["constraint_valid_probability"] = self.constraint_head(
                encoded
            ).squeeze(-1)
        return outputs

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> tuple["FailureToScenarioNetwork", Dict[str, Any]]:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        model = cls(payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, payload
