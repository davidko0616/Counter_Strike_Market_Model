"""Shared trade rejection policy primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RejectionPolicy:
    """Configurable trade rejection thresholds.

    Each gate is applied in priority order. The first failing gate becomes the
    rejection reason. Rows with no rejection reason are accepted.
    """

    min_score_threshold: float = 0.0
    min_liquidity_quality: float = 0.0
    max_staleness_days: float = float("inf")
    max_price_jump_share: float = 1.0
    exclude_event_regime: bool = True
    min_row_coverage: float = 0.0
    reject_bear_without_alpha: bool = False
    min_entry_price: float = 0.0

    def __post_init__(self) -> None:
        if self.min_score_threshold < 0:
            raise ValueError("min_score_threshold must be non-negative")
        if self.min_liquidity_quality < 0:
            raise ValueError("min_liquidity_quality must be non-negative")
        if self.max_staleness_days <= 0:
            raise ValueError("max_staleness_days must be positive")
        if self.max_price_jump_share < 0:
            raise ValueError("max_price_jump_share must be non-negative")
        if self.min_row_coverage < 0:
            raise ValueError("min_row_coverage must be non-negative")
        if self.min_entry_price < 0:
            raise ValueError("min_entry_price must be non-negative")


def rejection_reason_for_row(
    row: pd.Series,
    policy: RejectionPolicy,
    *,
    score_col: str = "strong_buy_score",
) -> str | None:
    """Return the first rejection reason for one candidate row."""
    if policy.exclude_event_regime and _string_value(row, "label_market_regime") == "known_event":
        return "event_regime"

    if policy.min_score_threshold > 0:
        score = _numeric_value(row, score_col, default=0.0)
        if score < policy.min_score_threshold:
            return "low_score"

    if policy.min_liquidity_quality > 0 and "liquidity_quality_score_30d" in row.index:
        liquidity = _numeric_value(row, "liquidity_quality_score_30d", default=0.0)
        if liquidity < policy.min_liquidity_quality:
            return "low_liquidity"

    if policy.min_entry_price > 0 and "entry_close" in row.index:
        entry_price = _numeric_value(row, "entry_close", default=0.0)
        if entry_price < policy.min_entry_price:
            return "low_entry_price"

    if np.isfinite(policy.max_staleness_days) and "effective_staleness_days" in row.index:
        staleness = _numeric_value(row, "effective_staleness_days", default=0.0)
        if staleness > policy.max_staleness_days:
            return "stale_price"

    if policy.max_price_jump_share < 1.0 and "price_jump_50pct_share_30d" in row.index:
        jump_share = _numeric_value(row, "price_jump_50pct_share_30d", default=0.0)
        if jump_share > policy.max_price_jump_share:
            return "price_jump"

    if policy.min_row_coverage > 0 and "row_coverage_30d" in row.index:
        row_coverage = _numeric_value(row, "row_coverage_30d", default=0.0)
        if row_coverage < policy.min_row_coverage:
            return "low_coverage"

    if policy.reject_bear_without_alpha:
        if "cs2_index_regime_bear" in row.index and "item_excess_log_return_7d" in row.index:
            is_bear = bool(_numeric_value(row, "cs2_index_regime_bear", default=0.0))
            excess_return = _numeric_value(row, "item_excess_log_return_7d", default=0.0)
            if is_bear and excess_return <= 0:
                return "bear_no_alpha"

    return None


def _numeric_value(row: pd.Series, column: str, *, default: float) -> float:
    if column not in row.index:
        return default
    raw = row.get(column)
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _string_value(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    value = row.get(column)
    if pd.isna(value):
        return ""
    return str(value)
