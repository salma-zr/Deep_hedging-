"""Training, evaluation and benchmark routines for deep hedging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from simulation import MarketConfig, black_scholes_put_delta, make_features, put_payoff, simulate_gbm_paths


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters for stochastic gradient training."""

    n_epochs: int = 25
    n_train_paths: int = 8192
    n_test_paths: int = 16384
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    grad_clip: float = 1.0
    seed: int = 1234
    device: str = "cpu"


def _transaction_cost(
    positions: torch.Tensor,
    paths: torch.Tensor,
    proportional_cost: float,
) -> torch.Tensor:
    """Proportional cost sum_k lambda S_k |delta_k - delta_{k-1}|."""

    if proportional_cost <= 0:
        return torch.zeros(paths.shape[0], dtype=paths.dtype, device=paths.device)
    zero = torch.zeros_like(positions[:, :1])
    positions_with_liquidation = torch.cat([zero, positions, zero], dim=1)
    trades = positions_with_liquidation[:, 1:] - positions_with_liquidation[:, :-1]
    trade_spots = paths[:, : trades.shape[1]]
    return proportional_cost * torch.sum(trade_spots * torch.abs(trades), dim=1)


def hedging_pnl(
    positions: torch.Tensor,
    paths: torch.Tensor,
    payoff: torch.Tensor,
    premium: float,
    proportional_cost: float = 0.0,
) -> torch.Tensor:
    """Terminal P&L of a self-financing short-put hedge."""

    gains = torch.sum(positions * (paths[:, 1:] - paths[:, :-1]), dim=1)
    costs = _transaction_cost(positions, paths, proportional_cost)
    return -payoff + premium + gains - costs


def quadratic_loss(pnl: torch.Tensor) -> torch.Tensor:
    """Mean squared terminal hedging error."""

    return torch.mean(pnl.pow(2))


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Differentiable empirical CVaR of losses -P&L at level alpha."""

    losses = -pnl
    var = torch.quantile(losses.detach(), alpha)
    return var + torch.relu(losses - var).mean() / (1.0 - alpha)


def train_deep_hedger(
    model: torch.nn.Module,
    market: MarketConfig,
    config: TrainConfig,
    *,
    premium: float | None = None,
    proportional_cost: float = 0.0,
    loss_name: str = "quadratic",
    cvar_alpha: float = 0.95,
    progress: bool = True,
) -> dict:
    """Train a hedging network by backpropagation through simulated paths."""

    device = torch.device(config.device)
    model.to(device)
    model.train()
    premium_value = market.option_premium if premium is None else premium

    paths = simulate_gbm_paths(
        market,
        config.n_train_paths,
        seed=config.seed,
        antithetic=True,
        device=device,
    )
    dataset = TensorDataset(paths)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: list[dict[str, float]] = []
    epochs: Iterable[int] = range(config.n_epochs)
    if progress:
        epochs = tqdm(epochs, desc=f"training {model.__class__.__name__}")

    for epoch in epochs:
        epoch_losses: list[float] = []
        for (batch_paths,) in loader:
            batch_paths = batch_paths.to(device)
            payoff = put_payoff(batch_paths, market.strike)
            positions = model(batch_paths, market)
            pnl = hedging_pnl(
                positions,
                batch_paths,
                payoff,
                premium_value,
                proportional_cost=proportional_cost,
            )
            if loss_name == "quadratic":
                loss = quadratic_loss(pnl)
            elif loss_name == "cvar":
                loss = cvar_loss(pnl, cvar_alpha)
            else:
                raise ValueError(f"Unknown loss_name: {loss_name}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(epoch_losses))})

    return {
        "model": model,
        "history": history,
        "market": asdict(market),
        "train_config": asdict(config),
        "premium": premium_value,
        "proportional_cost": proportional_cost,
        "loss_name": loss_name,
    }


def hedge_positions(
    model: torch.nn.Module,
    paths: torch.Tensor,
    market: MarketConfig,
) -> torch.Tensor:
    """Generate predictable hedge positions for every trading date."""

    if hasattr(model, "forward_sequence"):
        market_features = torch.stack(
            [make_features(paths, market, step) for step in range(market.n_steps)],
            dim=1,
        )
        return model.forward_sequence(market_features)

    positions = []
    previous = torch.zeros(paths.shape[0], dtype=paths.dtype, device=paths.device)
    for step in range(market.n_steps):
        features = make_features(paths, market, step)
        model_input = torch.cat([features, previous.unsqueeze(1)], dim=1)
        position = model(model_input)
        positions.append(position)
        previous = position
    return torch.stack(positions, dim=1)


@torch.no_grad()
def evaluate_strategy(
    model: torch.nn.Module,
    market: MarketConfig,
    n_paths: int,
    *,
    seed: int = 4321,
    premium: float | None = None,
    proportional_cost: float = 0.0,
    device: str = "cpu",
) -> dict:
    """Evaluate a trained hedger out-of-sample."""

    model.eval()
    model.to(device)
    paths = simulate_gbm_paths(market, n_paths, seed=seed, antithetic=True, device=device)
    payoff = put_payoff(paths, market.strike)
    positions = model(paths, market)
    premium_value = market.option_premium if premium is None else premium
    pnl = hedging_pnl(
        positions,
        paths,
        payoff,
        premium_value,
        proportional_cost=proportional_cost,
    )
    return summarize_pnl(pnl, name=model.__class__.__name__)


@torch.no_grad()
def evaluate_delta_hedge(
    market: MarketConfig,
    n_paths: int,
    *,
    seed: int = 4321,
    proportional_cost: float = 0.0,
    device: str = "cpu",
) -> dict:
    """Out-of-sample Black--Scholes delta-hedging benchmark."""

    paths = simulate_gbm_paths(market, n_paths, seed=seed, antithetic=True, device=device)
    n_paths_local = paths.shape[0]
    positions = []
    for step in range(market.n_steps):
        tau = torch.full(
            (n_paths_local,),
            market.maturity - step * market.dt,
            dtype=paths.dtype,
            device=paths.device,
        )
        positions.append(
            black_scholes_put_delta(
                paths[:, step], market.strike, tau, market.rate, market.volatility
            )
        )
    positions_t = torch.stack(positions, dim=1)
    payoff = put_payoff(paths, market.strike)
    pnl = hedging_pnl(
        positions_t,
        paths,
        payoff,
        market.option_premium,
        proportional_cost=proportional_cost,
    )
    return summarize_pnl(pnl, name="Black-Scholes delta")


def summarize_pnl(pnl: torch.Tensor, name: str = "strategy") -> dict:
    """Compute point estimates and a normal confidence interval for MSE."""

    x = pnl.detach().cpu().numpy()
    sq = x**2
    mse = float(np.mean(sq))
    se_mse = float(np.std(sq, ddof=1) / np.sqrt(len(sq))) if len(sq) > 1 else 0.0
    return {
        "strategy": name,
        "mean_pnl": float(np.mean(x)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "variance": float(np.var(x, ddof=1)),
        "std": float(np.std(x, ddof=1)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "mse_ci_low": mse - 1.96 * se_mse,
        "mse_ci_high": mse + 1.96 * se_mse,
    }

