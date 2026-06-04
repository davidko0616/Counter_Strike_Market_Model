from __future__ import annotations

import pandas as pd

from cs_market_model.research.paper_trade_ab_test import (
    append_paper_trade_ledger,
    bootstrap_profit_factor_ci,
    evaluate_policy_trades,
    normalize_paper_trade_ledger,
)


def test_bootstrap_profit_factor_ci_returns_positive_bounds() -> None:
    low, high = bootstrap_profit_factor_ci(
        [0.02, 0.01, -0.005, 0.03, -0.002],
        n_bootstrap=100,
    )

    assert low >= 0
    assert high >= low


def test_evaluate_policy_trades_passes_only_after_conservative_review() -> None:
    dates = pd.date_range("2026-01-01", periods=60, freq="3D", tz="UTC")
    trades = pd.DataFrame(
        {
            "entry_timestamp": dates,
            "market_hash_name": [f"Item {index % 10}" for index in range(60)],
            "pnl": [0.01] * 50 + [-0.002] * 10,
        }
    )

    row = evaluate_policy_trades("paper_trade_5_plus", "strict", trades)

    assert row["accepted_count"] == 60
    assert row["review_status"] == "passes_paper_trade_review"
    assert row["final_recommendation"] == "paper_trade_only"


def test_evaluate_policy_trades_marks_long_low_sample_as_too_rare() -> None:
    trades = pd.DataFrame(
        {
            "entry_timestamp": pd.date_range("2026-01-01", periods=10, freq="7D", tz="UTC"),
            "market_hash_name": [f"Item {index}" for index in range(10)],
            "pnl": [0.01] * 10,
        }
    )

    row = evaluate_policy_trades("paper_trade_5_plus", "strict", trades)

    assert row["review_status"] == "signal_too_rare_for_review"


def test_normalize_and_append_paper_trade_ledger(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "market_hash_name": ["A"],
            "policy": ["paper_trade_5_plus"],
            "model_name": ["lightgbm_rank_blend"],
            "signal_score": [0.9],
            "entry_price": [10.0],
            "current_price": [10.5],
            "pnl": [0.05],
        }
    )

    normalized = normalize_paper_trade_ledger(rows)
    path = append_paper_trade_ledger(normalized, ledger_dir=tmp_path)

    assert normalized.loc[0, "status"] == "open"
    assert path.name == "trades_2026-01.parquet"
    assert len(pd.read_parquet(path)) == 1
