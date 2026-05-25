from __future__ import annotations

import numpy as np
import pandas as pd

from cs_market_model.backtesting.portfolio import (
    PortfolioBacktestConfig,
    simulate_ranked_portfolios,
    summarize_backtest,
)
from cs_market_model.backtesting.rejection_policy import RejectionPolicy


def test_backtest_selects_daily_top_k_and_applies_same_item_cooldown() -> None:
    timestamps = pd.to_datetime(
        ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-15"],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "model_name": ["forest_rank_blend"] * 4,
            "split_id": ["wf_00"] * 4,
            "event_id": ["a", "b", "c", "d"],
            "market_hash_name": [
                "AK-47 | Test",
                "AWP | Test",
                "AK-47 | Test",
                "AK-47 | Test",
            ],
            "timestamp": timestamps,
            "exit_timestamp": timestamps + pd.Timedelta(days=7),
            "holding_period_days": [7, 7, 7, 7],
            "label_code": [2, 0, 2, 2],
            "label_class": ["Strong Buy", "Bad Time", "Strong Buy", "Strong Buy"],
            "strong_buy_score": [0.9, 0.8, 0.95, 0.7],
            "net_return_at_label": [0.10, -0.05, 0.20, 0.05],
        }
    )
    config = PortfolioBacktestConfig(
        top_k=1,
        max_position_fraction=0.5,
        same_item_cooldown_days=14,
    )

    ledger, daily_equity = simulate_ranked_portfolios(predictions, config)

    assert ledger["event_id"].tolist() == ["a", "d"]
    assert len(daily_equity) >= 2


def test_backtest_summary_reports_return_drawdown_and_precision() -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    predictions = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"] * 3,
            "event_id": ["a", "b", "c"],
            "market_hash_name": ["A", "B", "C"],
            "timestamp": timestamps,
            "exit_timestamp": timestamps + pd.Timedelta(days=1),
            "holding_period_days": [1, 1, 1],
            "label_code": [2, 0, 2],
            "label_class": ["Strong Buy", "Bad Time", "Strong Buy"],
            "strong_buy_score": [0.9, 0.8, 0.7],
            "net_return_at_label": [0.10, -0.20, 0.05],
        }
    )
    config = PortfolioBacktestConfig(top_k=1, max_position_fraction=0.5)

    ledger, daily_equity = simulate_ranked_portfolios(predictions, config)
    summary = summarize_backtest(ledger, daily_equity, config)

    assert summary.loc[0, "model_name"] == "lightgbm_rank_blend"
    assert summary.loc[0, "trade_count"] == 3
    assert summary.loc[0, "precision_at_selected"] == 2 / 3
    assert summary.loc[0, "max_drawdown"] <= 0
    assert np.isfinite(summary.loc[0, "total_return"])


def test_backtest_applies_rejection_policy_before_daily_top_k() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    predictions = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"] * 3,
            "event_id": ["event", "accepted", "low"],
            "market_hash_name": ["A", "B", "C"],
            "timestamp": [timestamp] * 3,
            "exit_timestamp": [timestamp + pd.Timedelta(days=1)] * 3,
            "holding_period_days": [1, 1, 1],
            "label_code": [2, 2, 2],
            "label_class": ["Strong Buy"] * 3,
            "strong_buy_score": [0.95, 0.90, 0.80],
            "net_return_at_label": [0.10, 0.10, 0.10],
            "label_market_regime": ["known_event", "normal", "normal"],
        }
    )
    config = PortfolioBacktestConfig(
        top_k=1,
        max_position_fraction=0.5,
        rejection_policy=RejectionPolicy(
            min_score_threshold=0.85,
            exclude_event_regime=True,
        ),
        rejection_policy_name="test_policy",
    )

    ledger, _ = simulate_ranked_portfolios(predictions, config)

    assert ledger["event_id"].tolist() == ["accepted"]
    assert ledger.loc[0, "rejection_policy_name"] == "test_policy"
    assert pd.isna(ledger.loc[0, "rejection_reason"])
