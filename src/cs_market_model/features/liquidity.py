"""Liquidity and data-availability proxy features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_liquidity_features(
    feature_frame: pd.DataFrame,
    observed_row_windows: Sequence[int] = (7, 30),
    quality_windows: Sequence[int] = (7, 30),
) -> pd.DataFrame:
    """Add point-in-time data coverage and staleness proxy features."""
    required = {"market_hash_name", "timestamp", "close"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ValueError(f"Missing required liquidity columns: {sorted(missing)}")

    features = feature_frame.sort_values(["market_hash_name", "timestamp"]).copy()
    group = features.groupby("market_hash_name", group_keys=False)

    timestamp_diff = group["timestamp"].diff()
    features["days_since_previous_bar"] = (
        timestamp_diff.dt.total_seconds().div(86_400).fillna(0).astype(float)
    )
    features["missing_calendar_days_since_previous_bar"] = (
        features["days_since_previous_bar"].sub(1).clip(lower=0)
    )

    if "staleness_days" in features.columns:
        normalized_staleness = pd.to_numeric(features["staleness_days"], errors="coerce")
    else:
        normalized_staleness = 0
    features["effective_staleness_days"] = pd.concat(
        [
            features["missing_calendar_days_since_previous_bar"],
            pd.Series(normalized_staleness, index=features.index).fillna(0),
        ],
        axis=1,
    ).max(axis=1)

    features["has_volume"] = (
        features["volume"].notna() if "volume" in features.columns else False
    )
    features["has_trade_count"] = (
        features["trade_count"].notna() if "trade_count" in features.columns else False
    )

    for window in observed_row_windows:
        observed_rows = group["close"].transform(
            lambda series, window=window: series.rolling(window, min_periods=1).count()
        )
        features[f"observed_rows_{window}d"] = observed_rows
        features[f"row_coverage_{window}d"] = observed_rows / window

    log_return = _log_return_1d(features)
    abs_return = log_return.abs()
    jump_50 = abs_return.gt(np.log(1.5)).where(abs_return.notna(), False)
    jump_100 = abs_return.gt(np.log(2.0)).where(abs_return.notna(), False)
    for window in quality_windows:
        features[f"abs_return_median_{window}d"] = abs_return.groupby(
            features["market_hash_name"],
            group_keys=False,
        ).transform(lambda series, window=window: series.rolling(window, min_periods=1).median())
        features[f"abs_return_max_{window}d"] = abs_return.groupby(
            features["market_hash_name"],
            group_keys=False,
        ).transform(lambda series, window=window: series.rolling(window, min_periods=1).max())
        features[f"price_jump_50pct_count_{window}d"] = jump_50.astype(int).groupby(
            features["market_hash_name"],
            group_keys=False,
        ).transform(lambda series, window=window: series.rolling(window, min_periods=1).sum())
        features[f"price_jump_100pct_count_{window}d"] = jump_100.astype(int).groupby(
            features["market_hash_name"],
            group_keys=False,
        ).transform(lambda series, window=window: series.rolling(window, min_periods=1).sum())
        features[f"price_jump_50pct_share_{window}d"] = (
            features[f"price_jump_50pct_count_{window}d"] / window
        )
        features[f"price_jump_100pct_share_{window}d"] = (
            features[f"price_jump_100pct_count_{window}d"] / window
        )

    features["liquidity_quality_score_30d"] = (
        pd.to_numeric(features.get("row_coverage_30d", 1.0), errors="coerce").fillna(0.0)
        * (1.0 - pd.to_numeric(features["price_jump_50pct_share_30d"], errors="coerce").fillna(1.0))
        * (1.0 / (1.0 + pd.to_numeric(features["effective_staleness_days"], errors="coerce").fillna(0.0)))
    )

    return features


def _log_return_1d(features: pd.DataFrame) -> pd.Series:
    if "log_return_1d" in features.columns:
        return pd.to_numeric(features["log_return_1d"], errors="coerce")
    close = pd.to_numeric(features["close"], errors="coerce")
    log_close = np.log(close)
    return log_close.groupby(features["market_hash_name"], group_keys=False).diff()
