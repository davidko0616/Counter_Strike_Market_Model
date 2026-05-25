from __future__ import annotations

import pandas as pd

from cs_market_model.backtesting.walk_forward_rejection import (
    summarize_walk_forward_selection,
    walk_forward_score_threshold_selection,
)


def _ledger() -> pd.DataFrame:
    rows = []
    for month, returns in [
        ("2026-01", [0.10, 0.08, -0.06, -0.05]),
        ("2026-02", [0.09, 0.07, -0.04, -0.03]),
        ("2026-03", [0.06, 0.05, -0.02, -0.01]),
    ]:
        for idx, trade_return in enumerate(returns):
            score = [0.9, 0.8, 0.2, 0.1][idx]
            notional = 0.05
            timestamp = pd.Timestamp(f"{month}-{idx + 1:02d}", tz="UTC")
            rows.append(
                {
                    "model_name": "lightgbm_rank_blend",
                    "event_id": f"{month}-{idx}",
                    "market_hash_name": f"Item {idx}",
                    "entry_timestamp": timestamp,
                    "exit_timestamp": timestamp + pd.Timedelta(days=7),
                    "strong_buy_score": score,
                    "label_market_regime": "normal",
                    "realized_net_return": trade_return,
                    "pnl": notional * trade_return,
                    "notional": notional,
                    "is_actual_strong_buy": trade_return > 0,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_threshold_uses_prior_periods_only() -> None:
    selection, accepted = walk_forward_score_threshold_selection(
        _ledger(),
        thresholds=[0.0, 0.5, 0.85],
        min_training_trades=4,
        min_accepted_trades=1,
    )

    february = selection[selection["test_period"].eq("2026-02")].iloc[0]
    assert february["training_trade_count"] == 4
    assert february["selected_score_threshold"] >= 0.5
    assert not accepted.empty
    assert accepted["test_period"].min() == "2026-01"


def test_walk_forward_summary_aggregates_accepted_test_trades() -> None:
    selection, accepted = walk_forward_score_threshold_selection(
        _ledger(),
        thresholds=[0.0, 0.5],
        min_training_trades=4,
        min_accepted_trades=1,
    )
    summary = summarize_walk_forward_selection(selection, accepted)

    assert summary.loc[0, "model_name"] == "lightgbm_rank_blend"
    assert summary.loc[0, "walk_forward_periods"] == 3
    assert summary.loc[0, "accepted_count"] > 0
    assert summary.loc[0, "accepted_total_pnl"] > 0
