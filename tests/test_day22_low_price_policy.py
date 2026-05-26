from __future__ import annotations

import pytest
import pandas as pd

from cs_market_model.research.day22_low_price_policy import (
    CapacitySizingConfig,
    add_liquidity_depth,
    apply_capacity_adjusted_sizing,
    apply_price_bucket_execution_costs,
)


def test_price_bucket_execution_costs_subtract_expected_penalties() -> None:
    trades = pd.DataFrame(
        {
            "entry_close": [0.5, 2.0, 10.0, 25.0],
            "realized_net_return": [0.10, 0.10, 0.10, 0.10],
            "notional": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = apply_price_bucket_execution_costs(trades)

    assert result["execution_price_bucket"].tolist() == ["<=1", "1-5", "5-20", ">20"]
    assert result["extra_execution_cost_bps"].tolist() == [500.0, 250.0, 150.0, 100.0]
    assert result["realized_net_return"].tolist() == pytest.approx([0.05, 0.075, 0.085, 0.09])
    assert result["pnl"].tolist() == pytest.approx([0.05, 0.075, 0.085, 0.09])


def test_capacity_adjusted_sizing_caps_item_bucket_and_liquidity() -> None:
    trades = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"] * 3,
            "market_hash_name": ["A", "A", "B"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-01"],
                utc=True,
            ),
            "strong_buy_score": [0.9, 0.8, 0.7],
            "entry_close": [10.0, 10.0, 10.0],
            "execution_price_bucket": ["5-20", "5-20", "5-20"],
            "steam_sell_count": [10.0, 10.0, 10.0],
            "realized_net_return": [0.10, 0.10, 0.10],
            "notional": [0.05, 0.05, 0.05],
            "pnl": [0.005, 0.005, 0.005],
        }
    )

    result = apply_capacity_adjusted_sizing(
        trades,
        CapacitySizingConfig(
            portfolio_capital_usd=1_000.0,
            max_notional_per_trade=0.05,
            max_item_daily_notional=0.006,
            max_bucket_daily_notional=0.009,
            max_sell_count_participation=0.05,
            minimum_executable_notional=0.001,
        ),
    )

    assert result["capacity_adjusted_notional"].tolist() == pytest.approx(
        [0.005, 0.001, 0.003]
    )
    assert result["pnl"].tolist() == pytest.approx([0.0005, 0.0001, 0.0003])


def test_add_liquidity_depth_merges_steam_sell_count() -> None:
    trades = pd.DataFrame({"market_hash_name": ["A", "B"]})
    liquidity = pd.DataFrame(
        {
            "market_hash_name": ["A"],
            "steam_sell_count": [12],
            "max_sell_count": [20],
            "steam_sell_price": [1.23],
        }
    )

    result = add_liquidity_depth(trades, liquidity)

    assert result.loc[0, "steam_sell_count"] == 12
    assert pd.isna(result.loc[1, "steam_sell_count"])
