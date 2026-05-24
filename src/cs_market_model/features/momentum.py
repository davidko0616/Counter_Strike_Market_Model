"""Price momentum and position features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_momentum_features(
    price_bars: pd.DataFrame,
    return_windows: Sequence[int] = (1, 7, 14, 30),
    moving_average_windows: Sequence[int] = (7, 14, 30),
    position_windows: Sequence[int] = (30,),
) -> pd.DataFrame:
    """Add backward-looking momentum features to normalized price bars."""
    required = {"market_hash_name", "timestamp", "close", "high", "low"}
    missing = required - set(price_bars.columns)
    if missing:
        raise ValueError(f"Missing required momentum columns: {sorted(missing)}")

    features = price_bars.sort_values(["market_hash_name", "timestamp"]).copy()
    group = features.groupby("market_hash_name", group_keys=False)
    features["log_close"] = np.log(features["close"].astype(float))

    for window in return_windows:
        features[f"log_return_{window}d"] = group["log_close"].diff(window)

    for window in moving_average_windows:
        rolling_mean = group["close"].transform(
            lambda series, window=window: series.rolling(
                window, min_periods=window
            ).mean()
        )
        features[f"close_to_ma_{window}d"] = (features["close"] / rolling_mean) - 1.0

    for window in position_windows:
        rolling_high = group["high"].transform(
            lambda series, window=window: series.rolling(
                window, min_periods=window
            ).max()
        )
        rolling_low = group["low"].transform(
            lambda series, window=window: series.rolling(
                window, min_periods=window
            ).min()
        )
        price_range = rolling_high - rolling_low
        position = (features["close"] - rolling_low) / price_range
        features[f"range_position_{window}d"] = position.where(price_range != 0)

    return features
