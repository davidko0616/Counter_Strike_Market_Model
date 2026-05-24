"""Internal CS2 market index and market-relative features."""

from __future__ import annotations

import numpy as np
import pandas as pd


INDEX_RETURN_WINDOWS = (1, 7, 14, 30, 90)
EXCESS_RETURN_WINDOWS = (1, 7, 14, 30)
BETA_WINDOWS = (30, 90)


def add_market_index_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Add an equal-weight internal CS2 skin index and relative-return features."""
    required = {"market_hash_name", "timestamp", "close"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ValueError(f"Missing required market-index columns: {sorted(missing)}")

    features = feature_frame.sort_values(["market_hash_name", "timestamp"]).copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    group = features.groupby("market_hash_name", group_keys=False)
    close = pd.to_numeric(features["close"], errors="coerce")
    first_close = group["close"].transform("first").astype(float)
    features["cs2_item_index_normalized_close"] = close / first_close * 100.0

    index = build_equal_weight_index(features)
    features = features.merge(index, on="timestamp", how="left", validate="many_to_one")

    for window in EXCESS_RETURN_WINDOWS:
        item_return_col = f"log_return_{window}d"
        index_return_col = f"cs2_index_log_return_{window}d"
        if item_return_col in features.columns and index_return_col in features.columns:
            features[f"item_excess_log_return_{window}d"] = (
                pd.to_numeric(features[item_return_col], errors="coerce")
                - pd.to_numeric(features[index_return_col], errors="coerce")
            )

    features = _add_item_beta_features(features)
    return features


def build_equal_weight_index(features: pd.DataFrame) -> pd.DataFrame:
    """Build an equal-weight index from normalized item closes."""
    index = (
        features.groupby("timestamp", as_index=False)["cs2_item_index_normalized_close"]
        .mean()
        .rename(columns={"cs2_item_index_normalized_close": "cs2_mvp_equal_weight_index"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    index["cs2_index_log_level"] = np.log(index["cs2_mvp_equal_weight_index"])
    for window in INDEX_RETURN_WINDOWS:
        index[f"cs2_index_return_{window}d"] = (
            index["cs2_mvp_equal_weight_index"].pct_change(window)
        )
        index[f"cs2_index_log_return_{window}d"] = index["cs2_index_log_level"].diff(window)

    index["cs2_index_volatility_14d"] = index["cs2_index_log_return_1d"].rolling(
        14,
        min_periods=14,
    ).std()
    index["cs2_index_volatility_30d"] = index["cs2_index_log_return_1d"].rolling(
        30,
        min_periods=30,
    ).std()
    for window in (30, 90):
        rolling_high = index["cs2_mvp_equal_weight_index"].rolling(
            window,
            min_periods=window,
        ).max()
        index[f"cs2_index_drawdown_{window}d"] = (
            index["cs2_mvp_equal_weight_index"] / rolling_high
        ) - 1.0

    ma_30 = index["cs2_mvp_equal_weight_index"].rolling(30, min_periods=30).mean()
    index["cs2_index_ma_distance_30d"] = (
        index["cs2_mvp_equal_weight_index"] / ma_30
    ) - 1.0
    index["cs2_index_trend_slope_30d"] = (
        index["cs2_mvp_equal_weight_index"].diff(30) / ma_30 / 30.0
    )
    index = _classify_market_regime(index)
    return index


def _classify_market_regime(index: pd.DataFrame) -> pd.DataFrame:
    frame = index.copy()
    frame["cs2_index_regime_bull"] = False
    frame["cs2_index_regime_neutral"] = True
    frame["cs2_index_regime_bear"] = False

    bull = (
        frame["cs2_index_ma_distance_30d"].gt(0)
        & frame["cs2_index_trend_slope_30d"].gt(0)
        & frame["cs2_index_drawdown_30d"].gt(-0.05)
    )
    bear = (
        frame["cs2_index_ma_distance_30d"].lt(0)
        & frame["cs2_index_trend_slope_30d"].lt(0)
    ) | frame["cs2_index_drawdown_30d"].le(-0.10)
    frame.loc[bull, "cs2_index_regime_bull"] = True
    frame.loc[bear, "cs2_index_regime_bear"] = True
    frame.loc[bull | bear, "cs2_index_regime_neutral"] = False
    return frame


def _add_item_beta_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.sort_values(["market_hash_name", "timestamp"]).copy()
    item_return = pd.to_numeric(frame.get("log_return_1d"), errors="coerce")
    index_return = pd.to_numeric(frame["cs2_index_log_return_1d"], errors="coerce")
    frame["_item_return_for_beta"] = item_return
    frame["_index_return_for_beta"] = index_return

    pieces = []
    for _, item_rows in frame.groupby("market_hash_name", sort=False):
        item_rows = item_rows.copy()
        for window in BETA_WINDOWS:
            covariance = item_rows["_item_return_for_beta"].rolling(
                window,
                min_periods=window,
            ).cov(item_rows["_index_return_for_beta"])
            variance = item_rows["_index_return_for_beta"].rolling(
                window,
                min_periods=window,
            ).var()
            item_rows[f"item_beta_to_cs2_index_{window}d"] = covariance / variance.replace(
                0,
                np.nan,
            )
            item_rows[f"item_correlation_to_cs2_index_{window}d"] = (
                item_rows["_item_return_for_beta"]
                .rolling(window, min_periods=window)
                .corr(item_rows["_index_return_for_beta"])
            )
            residual = (
                item_rows["_item_return_for_beta"]
                - item_rows[f"item_beta_to_cs2_index_{window}d"]
                * item_rows["_index_return_for_beta"]
            )
            item_rows[f"item_residual_volatility_to_cs2_index_{window}d"] = residual.rolling(
                window,
                min_periods=window,
            ).std()
        pieces.append(item_rows)

    return pd.concat(pieces, ignore_index=True).drop(
        columns=["_item_return_for_beta", "_index_return_for_beta"]
    )
