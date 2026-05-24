"""Liquidity and data-availability proxy features."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_liquidity_features(
    feature_frame: pd.DataFrame,
    observed_row_windows: Sequence[int] = (7, 30),
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

    return features
