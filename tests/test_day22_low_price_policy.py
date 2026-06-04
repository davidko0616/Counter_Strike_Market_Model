from __future__ import annotations

import pandas as pd
import pytest

from cs_market_model.research.day22_low_price_policy import (
    CapacitySizingConfig,
    _active_open_positions,
    add_liquidity_depth,
    apply_capacity_adjusted_sizing,
    apply_price_bucket_execution_costs,
    cost_return_with_entry_exit_multipliers,
)


def test_price_bucket_execution_costs_apply_entry_exit_price_multipliers() -> None:
    trades = pd.DataFrame(
        {
            "entry_close": [0.5, 2.0, 10.0, 25.0],
            "realized_net_return": [0.10, 0.10, 0.10, 0.10],
            "notional": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = apply_price_bucket_execution_costs(trades)

    assert result["execution_price_bucket"].tolist() == ["<1", "1-5", "5-20", ">20"]
    assert result["extra_execution_cost_bps"].tolist() == [500.0, 250.0, 150.0, 100.0]
    assert result["execution_cost_method"].unique().tolist() == ["entry_exit_price_multiplier"]
    assert result["realized_net_return"].tolist() == pytest.approx(
        [
            (1.10 * 0.95 / 1.05) - 1.0,
            (1.10 * 0.975 / 1.025) - 1.0,
            (1.10 * 0.985 / 1.015) - 1.0,
            (1.10 * 0.99 / 1.01) - 1.0,
        ]
    )
    assert result["pnl"].tolist() == pytest.approx(result["realized_net_return"].tolist())


def test_high_return_price_bucket_cost_is_not_flat_subtraction() -> None:
    corrected = cost_return_with_entry_exit_multipliers(1.0, 0.05)

    assert corrected == pytest.approx((2.0 * 0.95 / 1.05) - 1.0)
    assert corrected < 0.95


def test_negative_return_price_bucket_cost_amplifies_loss() -> None:
    trades = pd.DataFrame(
        {
            "entry_close": [10.0],
            "realized_net_return": [-0.10],
            "notional": [1.0],
        }
    )

    result = apply_price_bucket_execution_costs(trades)

    assert result.loc[0, "realized_net_return"] == pytest.approx((0.90 * 0.985 / 1.015) - 1.0)
    assert result.loc[0, "realized_net_return"] < -0.10


def test_price_bucket_boundary_at_exactly_one_dollar() -> None:
    trades = pd.DataFrame(
        {
            "entry_close": [0.99, 1.00, 1.01],
            "realized_net_return": [0.10, 0.10, 0.10],
            "notional": [1.0, 1.0, 1.0],
        }
    )

    result = apply_price_bucket_execution_costs(trades)

    assert result["execution_price_bucket"].tolist() == ["<1", "1-5", "1-5"]


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
            max_notional_per_trade_fraction=0.05,
            max_item_daily_notional_fraction=0.006,
            max_bucket_daily_notional_fraction=0.009,
            max_sell_count_participation=0.05,
            minimum_executable_notional_fraction=0.001,
        ),
    )

    assert result["capacity_adjusted_notional"].tolist() == pytest.approx([0.005, 0.001, 0.003])
    assert result["pnl"].tolist() == pytest.approx([0.0005, 0.0001, 0.0003])


def test_capacity_adjusted_sizing_rejects_dollar_notional_trades() -> None:
    trades = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"],
            "market_hash_name": ["A"],
            "entry_timestamp": pd.to_datetime(["2026-01-01"], utc=True),
            "exit_timestamp": pd.to_datetime(["2026-01-15"], utc=True),
            "strong_buy_score": [0.9],
            "entry_close": [10.0],
            "execution_price_bucket": ["5-20"],
            "steam_sell_count": [50.0],
            "realized_net_return": [0.10],
            "notional": [50.0],
            "pnl": [5.0],
        }
    )

    with pytest.raises(ValueError, match="dollar terms"):
        apply_capacity_adjusted_sizing(trades)


def test_capacity_config_rejects_fraction_values_above_one() -> None:
    with pytest.raises(ValueError, match="at most 1.0"):
        CapacitySizingConfig(max_notional_per_trade_fraction=1.5)


def test_capacity_config_rejects_non_positive_capital() -> None:
    with pytest.raises(ValueError, match="portfolio_capital_usd"):
        CapacitySizingConfig(portfolio_capital_usd=0.0)


def test_capacity_adjusted_sizing_uses_market_hash_name_tiebreaker() -> None:
    trades = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"] * 3,
            "market_hash_name": ["C", "A", "B"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-01"],
                utc=True,
            ),
            "strong_buy_score": [0.9, 0.9, 0.9],
            "entry_close": [10.0, 10.0, 10.0],
            "execution_price_bucket": ["5-20", "5-20", "5-20"],
            "steam_sell_count": [100.0, 100.0, 100.0],
            "realized_net_return": [0.10, 0.10, 0.10],
            "notional": [0.05, 0.05, 0.05],
            "pnl": [0.005, 0.005, 0.005],
        }
    )

    result = apply_capacity_adjusted_sizing(
        trades,
        CapacitySizingConfig(
            portfolio_capital_usd=1_000.0,
            max_notional_per_trade_fraction=0.05,
            max_item_daily_notional_fraction=0.05,
            max_bucket_daily_notional_fraction=0.075,
            max_sell_count_participation=1.0,
            minimum_executable_notional_fraction=0.001,
        ),
    )

    assert result["market_hash_name"].tolist() == ["A", "B", "C"]
    assert result["capacity_adjusted_notional"].tolist() == pytest.approx([0.05, 0.025, 0.0])
    assert result["capacity_filled"].tolist() == [True, True, False]


def test_capacity_adjusted_sizing_keeps_model_budgets_separate() -> None:
    trades = pd.DataFrame(
        {
            "model_name": ["model_a", "model_b"],
            "market_hash_name": ["A", "B"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01"],
                utc=True,
            ),
            "strong_buy_score": [0.9, 0.9],
            "entry_close": [10.0, 10.0],
            "execution_price_bucket": ["5-20", "5-20"],
            "steam_sell_count": [100.0, 100.0],
            "realized_net_return": [0.10, 0.10],
            "notional": [0.05, 0.05],
            "pnl": [0.005, 0.005],
        }
    )

    result = apply_capacity_adjusted_sizing(
        trades,
        CapacitySizingConfig(
            portfolio_capital_usd=1_000.0,
            max_notional_per_trade_fraction=0.05,
            max_item_daily_notional_fraction=0.05,
            max_bucket_daily_notional_fraction=0.05,
            max_sell_count_participation=1.0,
            minimum_executable_notional_fraction=0.001,
        ),
    )

    assert result["capacity_filled"].tolist() == [True, True]
    assert result["capacity_adjusted_notional"].tolist() == pytest.approx([0.05, 0.05])


def test_capacity_adjusted_sizing_enforces_open_position_limits() -> None:
    trades = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"] * 3,
            "market_hash_name": ["A", "B", "C"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"],
                utc=True,
            ),
            "exit_timestamp": pd.to_datetime(
                ["2026-01-10", "2026-01-11", "2026-01-12"],
                utc=True,
            ),
            "strong_buy_score": [0.9, 0.8, 0.7],
            "entry_close": [10.0, 10.0, 10.0],
            "execution_price_bucket": ["5-20", "5-20", "5-20"],
            "steam_sell_count": [100.0, 100.0, 100.0],
            "realized_net_return": [0.10, 0.10, 0.10],
            "notional": [0.05, 0.05, 0.05],
            "pnl": [0.005, 0.005, 0.005],
        }
    )

    result = apply_capacity_adjusted_sizing(
        trades,
        CapacitySizingConfig(
            portfolio_capital_usd=1_000.0,
            max_notional_per_trade_fraction=0.05,
            max_item_daily_notional_fraction=0.05,
            max_bucket_daily_notional_fraction=0.15,
            max_sell_count_participation=1.0,
            minimum_executable_notional_fraction=0.001,
            max_open_positions=2,
            max_open_positions_per_bucket=1,
        ),
    )

    assert result["capacity_filled"].tolist() == [True, False, False]
    assert result["capacity_rejection_reason"].tolist() == [
        "",
        "open_position_limit",
        "open_position_limit",
    ]
    assert result["capacity_bucket_open_positions_before"].tolist() == [0, 1, 1]


def test_active_open_positions_treats_nat_exit_as_open() -> None:
    positions = [
        {"exit_timestamp": pd.NaT, "bucket": "5-20"},
        {"exit_timestamp": pd.Timestamp("2026-01-10", tz="UTC"), "bucket": "5-20"},
        {"exit_timestamp": pd.Timestamp("2026-01-01", tz="UTC"), "bucket": "5-20"},
    ]

    active = _active_open_positions(
        positions,
        pd.Timestamp("2026-01-05", tz="UTC"),
    )

    assert len(active) == 2


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
