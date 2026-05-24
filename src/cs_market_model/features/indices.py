"""Cross-sectional category and weapon index features."""

from __future__ import annotations

import pandas as pd


def add_index_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Add same-timestamp relative features by category and weapon group."""
    required = {"market_hash_name", "timestamp", "close", "log_return_1d"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ValueError(f"Missing required index columns: {sorted(missing)}")

    features = feature_frame.sort_values(["market_hash_name", "timestamp"]).copy()
    item_group = features.groupby("market_hash_name", group_keys=False)
    features["normalized_close_since_start"] = features["close"] / item_group["close"].transform(
        "first"
    )

    universe_group = features.groupby("timestamp", dropna=False)
    features["universe_median_return_1d"] = universe_group["log_return_1d"].transform(
        "median"
    )
    features["item_minus_universe_return_1d"] = (
        features["log_return_1d"] - features["universe_median_return_1d"]
    )
    features["universe_median_normalized_close"] = universe_group[
        "normalized_close_since_start"
    ].transform("median")
    features["item_to_universe_index_ratio"] = (
        features["normalized_close_since_start"]
        / features["universe_median_normalized_close"]
    ) - 1.0

    for window in (7, 14, 30):
        return_column = f"log_return_{window}d"
        if return_column in features.columns:
            universe_return_column = f"universe_median_return_{window}d"
            features[universe_return_column] = universe_group[return_column].transform(
                "median"
            )
            features[f"item_minus_universe_return_{window}d"] = (
                features[return_column] - features[universe_return_column]
            )

    if "close_to_ma_30d" in features.columns:
        close_to_ma_30d = pd.to_numeric(features["close_to_ma_30d"], errors="coerce")
        above_ma_30d = close_to_ma_30d.gt(0).where(close_to_ma_30d.notna())
        features["universe_above_ma_30d_share"] = above_ma_30d.groupby(
            features["timestamp"],
            dropna=False,
        ).transform("mean")

    if "range_position_30d" in features.columns:
        range_position_30d = pd.to_numeric(features["range_position_30d"], errors="coerce")
        near_high_30d = range_position_30d.gt(0.8).where(range_position_30d.notna())
        features["universe_near_30d_high_share"] = near_high_30d.groupby(
            features["timestamp"],
            dropna=False,
        ).transform("mean")

    if "drawdown_from_30d_high" in features.columns:
        features["universe_median_drawdown_from_30d_high"] = universe_group[
            "drawdown_from_30d_high"
        ].transform("median")

    if "category" in features.columns:
        category_group = features.groupby(["timestamp", "category"], dropna=False)
        features["category_median_return_1d"] = category_group["log_return_1d"].transform(
            "median"
        )
        features["item_minus_category_return_1d"] = (
            features["log_return_1d"] - features["category_median_return_1d"]
        )
        features["category_median_normalized_close"] = category_group[
            "normalized_close_since_start"
        ].transform("median")
        features["item_to_category_index_ratio"] = (
            features["normalized_close_since_start"]
            / features["category_median_normalized_close"]
        ) - 1.0
        for window in (7, 14, 30):
            return_column = f"log_return_{window}d"
            if return_column in features.columns:
                category_return_column = f"category_median_return_{window}d"
                features[category_return_column] = category_group[return_column].transform(
                    "median"
                )
                features[f"item_minus_category_return_{window}d"] = (
                    features[return_column] - features[category_return_column]
                )
        if "close_to_ma_30d" in features.columns:
            features["category_above_ma_30d_share"] = above_ma_30d.groupby(
                [features["timestamp"], features["category"]],
                dropna=False,
            ).transform("mean")
        features["item_to_category_index_zscore"] = _group_zscore(
            features,
            group_columns=["timestamp", "category"],
            value_column="item_to_category_index_ratio",
        )

    if "weapon_type" in features.columns:
        weapon_group = features.groupby(["timestamp", "weapon_type"], dropna=False)
        features["weapon_median_return_1d"] = weapon_group["log_return_1d"].transform(
            "median"
        )
        features["item_minus_weapon_return_1d"] = (
            features["log_return_1d"] - features["weapon_median_return_1d"]
        )
        for window in (7, 14, 30):
            return_column = f"log_return_{window}d"
            if return_column in features.columns:
                weapon_return_column = f"weapon_median_return_{window}d"
                features[weapon_return_column] = weapon_group[return_column].transform(
                    "median"
                )
                features[f"item_minus_weapon_return_{window}d"] = (
                    features[return_column] - features[weapon_return_column]
                )
        features["weapon_median_normalized_close"] = weapon_group[
            "normalized_close_since_start"
        ].transform("median")
        features["item_to_weapon_index_ratio"] = (
            features["normalized_close_since_start"]
            / features["weapon_median_normalized_close"]
        ) - 1.0
        features["item_to_weapon_index_zscore"] = _group_zscore(
            features,
            group_columns=["timestamp", "weapon_type"],
            value_column="item_to_weapon_index_ratio",
        )

    if "rarity" in features.columns:
        rarity_group = features.groupby(["timestamp", "rarity"], dropna=False)
        features["rarity_median_normalized_close"] = rarity_group[
            "normalized_close_since_start"
        ].transform("median")
        features["item_to_rarity_index_ratio"] = (
            features["normalized_close_since_start"]
            / features["rarity_median_normalized_close"]
        ) - 1.0
        features["item_to_rarity_index_zscore"] = _group_zscore(
            features,
            group_columns=["timestamp", "rarity"],
            value_column="item_to_rarity_index_ratio",
        )

    if "collections" in features.columns:
        collection_group = features.groupby(["timestamp", "collections"], dropna=False)
        features["collection_median_normalized_close"] = collection_group[
            "normalized_close_since_start"
        ].transform("median")
        features["item_to_collection_index_ratio"] = (
            features["normalized_close_since_start"]
            / features["collection_median_normalized_close"]
        ) - 1.0
        features["item_to_collection_index_zscore"] = _group_zscore(
            features,
            group_columns=["timestamp", "collections"],
            value_column="item_to_collection_index_ratio",
        )

    return features


def _group_zscore(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str,
) -> pd.Series:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    group_keys = [frame[column] for column in group_columns]
    group_mean = values.groupby(group_keys, dropna=False).transform("mean")
    group_std = values.groupby(group_keys, dropna=False).transform("std")
    return (values - group_mean) / group_std.replace(0, pd.NA)
