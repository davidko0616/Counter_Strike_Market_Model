from __future__ import annotations

import pandas as pd

from cs_market_model.dashboard.app import (
    build_signal_table,
    candidate_rejection_reason,
    capacity_decision,
    selected_policy_threshold,
)


def test_dashboard_signal_table_keeps_accepted_and_rejected_candidates() -> None:
    artifacts = {
        "predictions": pd.DataFrame(
            {
                "model_name": ["lightgbm_rank_blend", "lightgbm_rank_blend"],
                "market_hash_name": ["A", "B"],
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
                "strong_buy_score": [0.90, 0.20],
                "liquidity_quality_score_30d": [0.8, 0.8],
            }
        ),
        "features": pd.DataFrame(
            {
                "market_hash_name": ["A", "B"],
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
                "category": ["rifle", "pistol"],
                "weapon_type": ["AK-47", "Glock-18"],
                "wear": ["Field-Tested", "Field-Tested"],
                "rarity": ["Restricted", "Classified"],
                "close": [10.0, 2.0],
                "row_coverage_30d": [1.0, 1.0],
                "effective_staleness_days": [0.0, 0.0],
            }
        ),
        "item_attribution": pd.DataFrame(
            {
                "policy_variant": ["min_price_5_costed"],
                "model_name": ["lightgbm_rank_blend"],
                "market_hash_name": ["A"],
                "accepted_trade_count": [3],
                "accepted_avg_return": [0.05],
                "accepted_win_rate": [0.67],
                "accepted_pnl": [0.01],
            }
        ),
    }

    table = build_signal_table(
        artifacts,
        model_name="lightgbm_rank_blend",
        min_score=0.5,
        min_liquidity=0.0,
        min_entry_price=5.0,
        categories=[],
        wears=[],
    )

    assert set(table["candidate_status"]) == {"accepted", "rejected"}
    rejected = table[table["market_hash_name"].eq("B")].iloc[0]
    assert rejected["rejection_reason"] == "below_min_price|low_score"


def test_dashboard_policy_threshold_uses_min_price_5_summary() -> None:
    summary = pd.DataFrame(
        {
            "policy_variant": ["raw_v2_costed", "min_price_5_costed"],
            "model_name": ["lightgbm_rank_blend", "lightgbm_rank_blend"],
            "selected_threshold_mean": [0.1, 0.7],
        }
    )

    assert selected_policy_threshold(summary, "lightgbm_rank_blend") == 0.7


def test_dashboard_capacity_decision_requires_all_scenarios_to_survive() -> None:
    survives = pd.DataFrame({"survives_capacity_stress": [True, True]})
    fails = pd.DataFrame({"survives_capacity_stress": [True, False]})

    assert capacity_decision(survives) == "paper trade"
    assert capacity_decision(fails) == "tighten gates"


def test_dashboard_candidate_rejection_reason_lists_failed_gates() -> None:
    row = pd.Series(
        {
            "close": 3.0,
            "strong_buy_score": 0.2,
            "liquidity_quality_score_30d": 0.1,
        }
    )

    assert candidate_rejection_reason(row, 5.0, 0.5, 0.2) == (
        "below_min_price|low_score|low_liquidity"
    )
