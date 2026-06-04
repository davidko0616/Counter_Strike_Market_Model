"""Backward-looking volatility and drawdown features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_volatility_features(
    feature_frame: pd.DataFrame,
    volatility_windows: Sequence[int] = (7, 14, 30),
    ewm_spans: Sequence[int] = (14, 30),
    drawdown_windows: Sequence[int] = (30, 90),
) -> pd.DataFrame:
    """Add rolling volatility, EWMA volatility, and drawdown features."""
    required = {"market_hash_name", "timestamp", "close"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ValueError(f"Missing required volatility columns: {sorted(missing)}")

    features = feature_frame.sort_values(["market_hash_name", "timestamp"]).copy()
    group = features.groupby("market_hash_name", group_keys=False)

    if "log_close" not in features.columns:
        features["log_close"] = np.log(features["close"].astype(float))
    if "log_return_1d" not in features.columns:
        features["log_return_1d"] = group["log_close"].diff()

    for window in volatility_windows:
        rolling_std = group["log_return_1d"].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).std()
        )
        features[f"return_volatility_{window}d"] = rolling_std
        features[f"realized_volatility_{window}d"] = rolling_std * np.sqrt(window)

    for span in ewm_spans:
        features[f"ewm_volatility_{span}d"] = group["log_return_1d"].transform(
            lambda series, span=span: series.ewm(span=span, adjust=False, min_periods=span).std()
        )

    for window in drawdown_windows:
        rolling_high = group["close"].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).max()
        )
        features[f"drawdown_from_{window}d_high"] = (features["close"] / rolling_high) - 1.0

    return features
