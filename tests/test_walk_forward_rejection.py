from __future__ import annotations

import pandas as pd

from cs_market_model.backtesting.rejection_policy import RejectionPolicy
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
                    "entry_close": [0.5, 2.0, 3.0, 4.0][idx],
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


def test_walk_forward_empty_ledger_returns_empty_outputs() -> None:
    selection, accepted = walk_forward_score_threshold_selection(pd.DataFrame())

    assert selection.empty
    assert accepted.empty


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


def test_walk_forward_applies_base_policy_price_gate() -> None:
    _, accepted = walk_forward_score_threshold_selection(
        _ledger(),
        thresholds=[0.0],
        min_training_trades=4,
        min_accepted_trades=1,
        base_policy=RejectionPolicy(exclude_event_regime=False, min_entry_price=1.0),
    )

    assert not accepted.empty
    assert accepted["entry_close"].ge(1.0).all()


def test_walk_forward_threshold_training_uses_same_base_price_gate() -> None:
    rows = []
    for month in ["2026-01", "2026-02"]:
        for name, score, entry_close, trade_return in [
            ("Low Price High Score", 0.95, 0.5, 0.20),
            ("High Price Low Score", 0.40, 10.0, 0.10),
        ]:
            timestamp = pd.Timestamp(f"{month}-01", tz="UTC")
            rows.append(
                {
                    "model_name": "lightgbm_rank_blend",
                    "event_id": f"{month}-{name}",
                    "market_hash_name": name,
                    "entry_timestamp": timestamp,
                    "exit_timestamp": timestamp + pd.Timedelta(days=7),
                    "strong_buy_score": score,
                    "label_market_regime": "normal",
                    "realized_net_return": trade_return,
                    "pnl": 0.05 * trade_return,
                    "notional": 0.05,
                    "is_actual_strong_buy": trade_return > 0,
                    "entry_close": entry_close,
                }
            )
    ledger = pd.DataFrame(rows)

    selection, accepted = walk_forward_score_threshold_selection(
        ledger,
        thresholds=[0.0, 0.5, 0.9],
        min_training_trades=1,
        min_accepted_trades=1,
        base_policy=RejectionPolicy(exclude_event_regime=False, min_entry_price=5.0),
    )

    february = selection[selection["test_period"].eq("2026-02")].iloc[0]
    assert february["raw_training_trade_count"] == 2
    assert february["training_trade_count"] == 1
    assert february["selected_score_threshold"] == 0.0
    assert accepted["entry_close"].ge(5.0).all()
