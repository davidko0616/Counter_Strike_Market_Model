"""Cross-sectional rank features for relative opportunity scoring."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_RANK_COLUMNS = (
    "log_return_7d",
    "log_return_14d",
    "log_return_30d",
    "close_to_ma_30d",
    "range_position_30d",
    "return_volatility_30d",
    "drawdown_from_30d_high",
    "item_to_category_index_ratio",
    "item_to_weapon_index_ratio",
    "item_to_rarity_index_ratio",
    "item_to_collection_index_ratio",
    "item_to_category_index_zscore",
    "liquidity_quality_score_30d",
)


def add_rank_features(
    feature_frame: pd.DataFrame,
    rank_columns: Sequence[str] = DEFAULT_RANK_COLUMNS,
) -> pd.DataFrame:
    """Add same-date percentile ranks across the full universe and local groups."""
    required = {"timestamp"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ValueError(f"Missing required rank columns: {sorted(missing)}")

    features = feature_frame.sort_values(["timestamp", "market_hash_name"]).copy()
    available_columns = [column for column in rank_columns if column in features.columns]

    for column in available_columns:
        features[f"{column}_universe_rank"] = _percentile_rank(
            features,
            group_columns=["timestamp"],
            value_column=column,
        )
        if "category" in features.columns:
            features[f"{column}_category_rank"] = _percentile_rank(
                features,
                group_columns=["timestamp", "category"],
                value_column=column,
            )
        if "weapon_type" in features.columns:
            features[f"{column}_weapon_rank"] = _percentile_rank(
                features,
                group_columns=["timestamp", "weapon_type"],
                value_column=column,
            )

    return features


def _percentile_rank(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str,
) -> pd.Series:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    return values.groupby([frame[column] for column in group_columns], dropna=False).rank(
        pct=True,
        method="average",
    )
