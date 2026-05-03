"""Neural parametrizations of predictable hedging strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

try:  # Allows both ``import model`` from notebooks and ``import src.model``.
    from .simulation import MarketConfig, make_features
except ImportError:  # pragma: no cover - exercised in notebook-style imports.
    from simulation import MarketConfig, make_features


ArchitectureName = Literal["mlp_simple", "mlp_deep", "lstm"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration of the neural hedging policy."""

    architecture: ArchitectureName = "mlp_simple"
    input_dim: int = 4
    hidden_dim: int = 32
    n_layers: int = 2
    output_bound: float = 2.0
    dropout: float = 0.0


class MLPHedger(nn.Module):
    """Feed-forward policy shared across hedging dates.

    Inputs are current normalized market features and the previous position.
    The output is the next stock position. The hyperbolic tangent bound avoids
    pathological leverage during early training while preserving differentiability.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_layers < 1:
            raise ValueError("n_layers must be positive")
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.n_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.output_bound = config.output_bound

    def forward_step(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_bound * torch.tanh(self.net(features)).squeeze(-1)

    def forward(self, paths: torch.Tensor, market: MarketConfig) -> torch.Tensor:
        """Generate a predictable stock position for every hedging date."""

        positions = []
        previous = torch.zeros(paths.shape[0], dtype=paths.dtype, device=paths.device)
        for step in range(market.n_steps):
            features = make_features(paths, market, step)
            policy_input = torch.cat([features, previous.unsqueeze(1)], dim=1)
            previous = self.forward_step(policy_input)
            positions.append(previous)
        return torch.stack(positions, dim=1)


class LSTMHedger(nn.Module):
    """Recurrent policy for the full path of features.

    At date k the LSTM has only consumed features up to k, so the generated
    strategy remains predictable. The previous position is included as a
    feature outside the recurrent state, matching the semi-recurrent structure
    discussed by Buehler et al.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_layers < 1:
            raise ValueError("n_layers must be positive")
        # The recurrent module sees market features only; previous position is
        # added to the policy head at each time step.
        self.market_input_dim = config.input_dim - 1
        self.lstm = nn.LSTM(
            input_size=self.market_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.output_bound = config.output_bound

    def forward_sequence(self, market_features: torch.Tensor) -> torch.Tensor:
        """Return positions for all dates from market features.

        Parameters
        ----------
        market_features:
            Tensor of shape ``(batch, n_steps, market_input_dim)``.
        """

        recurrent, _ = self.lstm(market_features)
        positions = []
        previous = torch.zeros(
            market_features.shape[0],
            1,
            dtype=market_features.dtype,
            device=market_features.device,
        )
        for step in range(market_features.shape[1]):
            head_input = torch.cat([recurrent[:, step, :], previous], dim=1)
            position = self.output_bound * torch.tanh(self.head(head_input))
            positions.append(position.squeeze(-1))
            previous = position
        return torch.stack(positions, dim=1)

    def forward(self, paths: torch.Tensor, market: MarketConfig) -> torch.Tensor:
        """Generate positions using only information available up to each date."""

        features = [make_features(paths, market, step) for step in range(market.n_steps)]
        market_features = torch.stack(features, dim=1)
        return self.forward_sequence(market_features)


def build_hedger(config: ModelConfig) -> nn.Module:
    """Factory for supported hedging architectures."""

    if config.architecture in {"mlp_simple", "mlp_deep"}:
        return MLPHedger(config)
    if config.architecture == "lstm":
        return LSTMHedger(config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")


def default_model_configs() -> dict[str, ModelConfig]:
    """Architecture grid used in the notebook and report."""

    return {
        "MLP simple": ModelConfig(
            architecture="mlp_simple",
            hidden_dim=32,
            n_layers=1,
            output_bound=2.0,
        ),
        "MLP profond": ModelConfig(
            architecture="mlp_deep",
            hidden_dim=64,
            n_layers=3,
            output_bound=2.0,
            dropout=0.02,
        ),
        "LSTM": ModelConfig(
            architecture="lstm",
            hidden_dim=32,
            n_layers=1,
            output_bound=2.0,
        ),
    }
