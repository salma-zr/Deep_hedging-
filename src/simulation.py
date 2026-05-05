"""Monte Carlo simulation and Black--Scholes reference quantities.

Training and evaluation paths are generated under the real-world probability
measure P. Risk-neutral formulas are used only for the exogenous option
premium and for the classical delta-hedging benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, log, sqrt

import numpy as np
import torch


@dataclass(frozen=True)
class MarketConfig:
    """Parameters of the discretized Black--Scholes market."""

    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.0
    drift: float = 0.03
    volatility: float = 0.2
    n_steps: int = 30
    premium: float | None = None
    transaction_cost: float = 0.0

    @property
    def dt(self) -> float:
        return self.maturity / self.n_steps

    @property
    def option_premium(self) -> float:
        if self.premium is not None:
            return self.premium
        return black_scholes_put_price(
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
        )


def set_seed(seed: int) -> None:
    """Set NumPy and PyTorch seeds."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """Risk-neutral Black--Scholes price of a European put."""

    if maturity <= 0:
        return max(strike - spot, 0.0)
    if volatility <= 0:
        forward_payoff = max(strike - spot * np.exp(rate * maturity), 0.0)
        return float(np.exp(-rate * maturity) * forward_payoff)

    vol_sqrt_t = volatility * sqrt(maturity)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return strike * np.exp(-rate * maturity) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def black_scholes_put_delta(
    spot: torch.Tensor,
    strike: float,
    tau: torch.Tensor | float,
    rate: float,
    volatility: float,
) -> torch.Tensor:
    """Black--Scholes delta of a European put."""

    tau_t = torch.as_tensor(tau, dtype=spot.dtype, device=spot.device)
    tau_t = torch.clamp(tau_t, min=1.0e-8)
    vol_sqrt_t = volatility * torch.sqrt(tau_t)
    d1 = (torch.log(torch.clamp(spot, min=1.0e-8) / strike) + (rate + 0.5 * volatility**2) * tau_t) / vol_sqrt_t
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=spot.dtype, device=spot.device),
        torch.tensor(1.0, dtype=spot.dtype, device=spot.device),
    )
    return normal.cdf(d1) - 1.0


def simulate_gbm_paths(
    config: MarketConfig,
    n_paths: int,
    seed: int = 1234,
    antithetic: bool = True,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Simulate GBM paths under P with an exact log-Euler scheme."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n_base = (n_paths + 1) // 2 if antithetic else n_paths
    shocks = torch.randn(n_base, config.n_steps, generator=generator, dtype=dtype)
    if antithetic:
        shocks = torch.cat([shocks, -shocks], dim=0)[:n_paths]

    increments = (
        (config.drift - 0.5 * config.volatility**2) * config.dt
        + config.volatility * sqrt(config.dt) * shocks
    )
    log_paths = torch.cat(
        [torch.zeros(n_paths, 1, dtype=dtype), torch.cumsum(increments, dim=1)],
        dim=1,
    )
    return (config.spot * torch.exp(log_paths)).to(device)


def put_payoff(paths: torch.Tensor, strike: float) -> torch.Tensor:
    """Terminal payoff of a European put."""

    return torch.clamp(strike - paths[:, -1], min=0.0)


def make_features(paths: torch.Tensor, config: MarketConfig, step: int) -> torch.Tensor:
    """Observed state at a hedging date.

    Features are normalized spot, log-moneyness and relative time-to-maturity.
    """

    spot = paths[:, step]
    tau = config.maturity - step * config.dt
    spot_scaled = spot / config.spot - 1.0
    log_moneyness = torch.log(torch.clamp(spot, min=1.0e-8) / config.strike)
    tau_feature = torch.full_like(spot, tau / config.maturity)
    return torch.stack([spot_scaled, log_moneyness, tau_feature], dim=1)


def terminal_pnl(
    paths: torch.Tensor,
    positions: torch.Tensor,
    config: MarketConfig,
    premium: float | None = None,
    transaction_cost: float | None = None,
) -> torch.Tensor:
    """Compute seller terminal P&L: -Z + p0 + sum delta dS - costs."""

    p0 = config.option_premium if premium is None else premium
    tc = config.transaction_cost if transaction_cost is None else transaction_cost
    increments = paths[:, 1:] - paths[:, :-1]
    gains = torch.sum(positions * increments, dim=1)
    payoff = put_payoff(paths, config.strike)
    if tc > 0.0:
        initial_trade = torch.abs(positions[:, :1])
        rebalances = torch.abs(positions[:, 1:] - positions[:, :-1])
        liquidation = torch.abs(positions[:, -1:])
        traded = torch.cat([initial_trade, rebalances, liquidation], dim=1)
        cost_spots = torch.cat([paths[:, :-1], paths[:, -1:]], dim=1)
        costs = tc * torch.sum(cost_spots * traded, dim=1)
    else:
        costs = torch.zeros_like(gains)
    return -payoff + p0 + gains - costs


def delta_hedge_positions(paths: torch.Tensor, config: MarketConfig) -> torch.Tensor:
    """Classical Black--Scholes put delta at each trading date."""

    deltas = []
    for step in range(config.n_steps):
        tau = config.maturity - step * config.dt
        deltas.append(
            black_scholes_put_delta(
                paths[:, step],
                config.strike,
                tau,
                config.rate,
                config.volatility,
            )
        )
    return torch.stack(deltas, dim=1)
