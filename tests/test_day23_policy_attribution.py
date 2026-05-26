from __future__ import annotations

from pathlib import Path

import pandas as pd

from cs_market_model.research.day23_policy_attribution import (
    build_bucket_attribution,
    build_policy_summary,
    load_policy_trades,
)


def test_policy_summary_reports_concentration() -> None:
    trades = _trades()

    summary = build_policy_summary(trades)

    row = summary.iloc[0]
    assert row["policy_variant"] == "policy_a"
    assert row["trade_count"] == 3
    assert row["positive_item_count"] == 2
    assert row["top_3_item_pnl_share"] == 1.0
    assert row["top_2_period_pnl_share"] == 1.0
    assert row["total_original_pnl"] == 4.0
    assert row["average_capacity_multiplier"] == 0.5


def test_bucket_attribution_includes_execution_price_bucket() -> None:
    buckets = build_bucket_attribution(_trades())

    price = buckets[buckets["bucket_type"].eq("execution_price_bucket")]

    assert {"<=1", "1-5"} <= set(price["bucket_value"])


def test_load_policy_trades_tags_variant(tmp_path: Path) -> None:
    path = tmp_path / "accepted.csv"
    _trades().drop(columns=["policy_variant"]).to_csv(path, index=False)

    loaded = load_policy_trades({"policy_x": path})

    assert set(loaded["policy_variant"]) == {"policy_x"}
    assert "entry_month" in loaded.columns


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_variant": ["policy_a", "policy_a", "policy_a"],
            "model_name": ["lightgbm_rank_blend"] * 3,
            "market_hash_name": ["A", "B", "A"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-01"],
                utc=True,
            ),
            "entry_month": ["2026-01", "2026-01", "2026-02"],
            "realized_net_return": [0.1, 0.2, -0.1],
            "pnl": [1.0, 2.0, -0.5],
            "original_pnl": [2.0, 4.0, -2.0],
            "entry_close": [0.5, 2.0, 0.5],
            "notional": [0.5, 0.5, 0.5],
            "original_notional": [1.0, 1.0, 1.0],
            "capacity_notional_multiplier": [0.5, 0.5, 0.5],
            "execution_price_bucket": ["<=1", "1-5", "<=1"],
            "price_bucket": ["<=1", "1-5", "<=1"],
            "wear": ["Field-Tested", "Field-Tested", "Field-Tested"],
            "rarity": ["Consumer Grade", "Mil-Spec Grade", "Consumer Grade"],
            "category": ["rifle", "pistol", "rifle"],
            "label_market_regime": ["normal", "normal", "normal"],
        }
    )
