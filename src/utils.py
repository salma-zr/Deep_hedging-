"""Utility functions for tables and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metrics_to_frame(results: Iterable[dict]) -> pd.DataFrame:
    """Convert strategy metrics into a display-ready DataFrame."""

    frame = pd.DataFrame(results)
    ordered = [
        "strategy",
        "mean_pnl",
        "mse",
        "rmse",
        "variance",
        "std",
        "q05",
        "q50",
        "q95",
        "mse_ci_low",
        "mse_ci_high",
    ]
    return frame[[column for column in ordered if column in frame.columns]]


def plot_training_history(histories: dict[str, list[dict]], output_path: str | Path | None = None) -> None:
    """Plot training losses by architecture."""

    plt.figure(figsize=(7, 4))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        losses = [row["loss"] for row in history]
        plt.plot(epochs, losses, label=name, linewidth=1.6)
    plt.xlabel("Epoch")
    plt.ylabel("Training objective")
    plt.title("Convergence of the deep hedging objective")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=160)


def plot_pnl_distribution(
    pnl_by_strategy: dict[str, np.ndarray],
    output_path: str | Path | None = None,
    bins: int = 70,
) -> None:
    """Plot out-of-sample terminal P&L distributions."""

    plt.figure(figsize=(7, 4))
    for name, values in pnl_by_strategy.items():
        plt.hist(values, bins=bins, density=True, alpha=0.42, label=name)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Terminal P&L")
    plt.ylabel("Density")
    plt.title("Out-of-sample hedging error distribution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=160)


def plot_metric_bar(
    table: pd.DataFrame,
    metric: str,
    output_path: str | Path | None = None,
) -> None:
    """Plot a bar chart for a metric indexed by strategy."""

    plt.figure(figsize=(7, 4))
    plt.bar(table["strategy"], table[metric])
    plt.ylabel(metric)
    plt.title(f"Comparison by {metric}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=160)
