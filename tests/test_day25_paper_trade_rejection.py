from __future__ import annotations

import pandas as pd
import pytest

from cs_market_model.backtesting.rejection_policy import RejectionPolicy
from cs_market_model.research.day22_low_price_policy import CapacitySizingConfig
from cs_market_model.research.day25_paper_trade_rejection import (
    PaperTradeRejectionScenario,
    build_paper_trade_summary_rows,
    paper_trade_5_plus_policy,
)


def test_paper_trade_summary_separates_rejection_sources() -> None:
    selection = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"],
            "raw_test_trade_count": [10],
            "test_trade_count": [6],
            "test_rejected_count": [2],
            "selected_score_threshold": [0.55],
        }
    )
    filled = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"] * 3,
            "market_hash_name": ["A", "B", "C"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-01"],
                utc=True,
            ),
            "test_period": ["2026-01", "2026-01", "2026-02"],
            "realized_net_return": [0.10, -0.02, 0.05],
            "pnl": [0.010, -0.002, 0.005],
            "notional": [0.1, 0.1, 0.1],
            "capacity_filled": [True, True, True],
            "capacity_notional_multiplier": [1.0, 0.5, 1.0],
        }
    )
    sized = pd.concat(
        [
            filled,
            pd.DataFrame(
                {
                    "model_name": ["lightgbm_rank_blend"],
                    "market_hash_name": ["D"],
                    "entry_timestamp": pd.to_datetime(["2026-02-02"], utc=True),
                    "test_period": ["2026-02"],
                    "realized_net_return": [0.01],
                    "pnl": [0.0],
                    "notional": [0.0],
                    "capacity_filled": [False],
                    "capacity_notional_multiplier": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )

    rows = build_paper_trade_summary_rows(
        selection,
        sized,
        filled,
        PaperTradeRejectionScenario(
            name="test",
            policy_name="strict",
            description="test scenario",
            policy=RejectionPolicy(min_entry_price=5.0, max_abs_return_30d=0.35),
            capacity_config=CapacitySizingConfig(max_open_positions=10),
        ),
    )

    assert rows[0]["base_gate_rejected_count"] == 4
    assert rows[0]["score_rejected_count"] == 2
    assert rows[0]["capacity_rejected_count"] == 1
    assert rows[0]["total_rejected_count"] == 7
    assert rows[0]["total_candidate_count"] == 10
    assert rows[0]["rejection_rate"] == pytest.approx(0.7)
    assert rows[0]["accepted_total_pnl"] == pytest.approx(0.013)
    assert rows[0]["mean_selected_score_threshold"] == pytest.approx(0.55)
    assert rows[0]["survives_capacity_stress"] is False


def test_paper_trade_policy_loads_configured_stricter_gates() -> None:
    policy = paper_trade_5_plus_policy()

    assert policy.min_entry_price == 5.0
    assert policy.min_liquidity_quality == 0.7
    assert policy.min_row_coverage == 0.8
    assert policy.reject_bear_without_alpha is True
    assert policy.min_item_excess_log_return_7d == 0.0
    assert policy.max_abs_return_30d == 0.35
