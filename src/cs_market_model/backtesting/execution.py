"""Execution assumptions for portfolio backtests.

The Day 7 labels already apply venue fees and slippage when computing
``net_return_at_label``. Backtest execution therefore uses that realized net
return by default and only applies optional expected failed-execution dilution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Execution assumptions applied to model-selected trades."""

    failed_execution_probability: float = 0.0
    return_col: str = "net_return_at_label"

    def __post_init__(self) -> None:
        if not 0.0 <= self.failed_execution_probability <= 1.0:
            raise ValueError("failed_execution_probability must be between 0 and 1")

    @property
    def execution_probability(self) -> float:
        return 1.0 - self.failed_execution_probability


def realized_trade_return(
    row: pd.Series,
    assumptions: ExecutionAssumptions | None = None,
) -> float:
    """Return the expected realized net return for one selected prediction row."""
    resolved = assumptions or ExecutionAssumptions()
    raw_return = pd.to_numeric(pd.Series([row.get(resolved.return_col)]), errors="coerce").iloc[0]
    if not np.isfinite(raw_return):
        return np.nan
    return float(raw_return) * resolved.execution_probability
