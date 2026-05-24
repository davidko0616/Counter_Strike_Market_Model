from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from cs_market_model.features.build_features import (
    audit_feature_table,
    build_feature_table,
    feature_columns,
)


def synthetic_price_bars(days: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=days, freq="D", tz="UTC")
    rows = []
    for item_index, market_hash_name in enumerate(["AK-47 | Test A", "AWP | Test B"]):
        for day_index, timestamp in enumerate(timestamps):
            close = 10.0 + item_index + day_index
            rows.append(
                {
                    "item_id": f"item_{item_index}",
                    "market_hash_name": market_hash_name,
                    "venue": "steam",
                    "timestamp": timestamp,
                    "timestamp_exchange": timestamp,
                    "open": close - 0.25,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "median_price": close,
                    "volume": pd.NA,
                    "trade_count": pd.NA,
                    "currency": "CNY",
                    "data_resolution": "1d",
                    "is_forward_filled": False,
                    "staleness_days": 0,
                    "source": "steamdt",
                }
            )
    return pd.DataFrame(rows)


def synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_hash_name": "AK-47 | Test A",
                "weapon_type": "AK-47",
                "category": "rifle",
                "rarity": "Restricted",
                "collections": "Test Collection",
                "cases": "Test Case",
            },
            {
                "market_hash_name": "AWP | Test B",
                "weapon_type": "AWP",
                "category": "sniper",
                "rarity": "Classified",
                "collections": "Test Collection",
                "cases": "Test Case",
            },
        ]
    )


def test_feature_factory_adds_expected_feature_columns() -> None:
    features = build_feature_table(synthetic_price_bars(), synthetic_metadata())

    expected = {
        "log_return_1d",
        "log_return_7d",
        "close_to_ma_7d",
        "range_position_30d",
        "return_volatility_7d",
        "ewm_volatility_14d",
        "drawdown_from_30d_high",
        "days_since_previous_bar",
        "row_coverage_30d",
        "category_median_return_1d",
        "item_minus_category_return_1d",
        "weapon_median_return_1d",
        "universe_median_return_14d",
        "item_minus_universe_return_14d",
        "universe_above_ma_30d_share",
        "log_return_14d_universe_rank",
        "close_to_ma_30d_weapon_rank",
        "item_to_rarity_index_zscore",
        "item_to_weapon_index_zscore",
        "price_jump_50pct_count_30d",
        "liquidity_quality_score_30d",
        "is_post_known_market_event_30d",
        "cs2_mvp_equal_weight_index",
        "cs2_index_return_30d",
        "item_excess_log_return_30d",
        "item_beta_to_cs2_index_30d",
        "cs2_index_regime_bear",
    }

    assert expected <= set(features.columns)
    assert features["market_hash_name"].nunique() == 2
    assert len(feature_columns(features)) >= len(expected)


def test_feature_factory_uses_only_prefix_history() -> None:
    price_bars = synthetic_price_bars(days=40)
    metadata = synthetic_metadata()
    cutoff = pd.Timestamp("2026-01-25", tz="UTC")

    full_features = build_feature_table(price_bars, metadata)
    prefix_features = build_feature_table(price_bars[price_bars["timestamp"] <= cutoff], metadata)

    compare_columns = [
        "market_hash_name",
        "timestamp",
        "log_return_1d",
        "log_return_7d",
        "close_to_ma_7d",
        "return_volatility_7d",
        "drawdown_from_30d_high",
        "row_coverage_30d",
        "item_minus_category_return_1d",
        "item_minus_weapon_return_1d",
        "universe_median_return_14d",
        "item_minus_universe_return_14d",
        "universe_above_ma_30d_share",
        "log_return_14d_universe_rank",
        "close_to_ma_30d_weapon_rank",
        "item_to_rarity_index_zscore",
        "price_jump_50pct_count_30d",
        "liquidity_quality_score_30d",
        "cs2_mvp_equal_weight_index",
        "item_excess_log_return_30d",
    ]
    full_prefix = full_features[full_features["timestamp"] <= cutoff][compare_columns]
    prefix_subset = prefix_features[compare_columns]

    assert_frame_equal(
        full_prefix.reset_index(drop=True),
        prefix_subset.reset_index(drop=True),
        check_dtype=False,
    )


def test_feature_audit_reports_missing_and_finite_counts() -> None:
    features = build_feature_table(synthetic_price_bars(days=10), synthetic_metadata())
    audit = audit_feature_table(features)

    log_return_7d = audit[audit["feature"] == "log_return_7d"].iloc[0]

    assert log_return_7d["missing_count"] > 0
    assert log_return_7d["finite_count"] > 0
    assert np.isfinite(log_return_7d["max"])
