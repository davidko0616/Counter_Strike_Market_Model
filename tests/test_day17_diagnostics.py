from __future__ import annotations

import pandas as pd

from cs_market_model.research.day17_diagnostics import (
    build_accepted_rejected_profile,
    build_item_attribution,
    build_threshold_stability,
    tag_walk_forward_acceptance,
)


def _ledger() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "model_name": ["model"] * 4,
            "market_hash_name": ["A", "B", "A", "B"],
            "entry_timestamp": timestamps,
            "label_market_regime": ["normal"] * 4,
            "strong_buy_score": [0.9, 0.1, 0.8, 0.2],
            "realized_net_return": [0.10, -0.05, 0.08, -0.02],
            "pnl": [0.01, -0.005, 0.008, -0.002],
            "is_actual_strong_buy": [True, False, True, False],
            "liquidity_quality_score_30d": [0.8, 0.4, 0.7, 0.3],
            "item_excess_log_return_7d": [0.05, -0.01, 0.03, -0.02],
        }
    )


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["model", "model"],
            "test_period": ["2026-01", "2026-02"],
            "selected_score_threshold": [0.5, 0.5],
            "test_accepted_total_pnl": [0.01, 0.008],
        }
    )


def test_tag_walk_forward_acceptance_applies_period_thresholds() -> None:
    tagged = tag_walk_forward_acceptance(_ledger(), _selection())

    assert tagged["is_accepted"].tolist() == [True, False, True, False]
    assert set(tagged["selected_threshold"]) == {0.5}


def test_item_attribution_splits_base_and_accepted_pnl() -> None:
    tagged = tag_walk_forward_acceptance(_ledger(), _selection())
    attribution = build_item_attribution(tagged)
    item_a = attribution[attribution["market_hash_name"].eq("A")].iloc[0]

    assert item_a["base_trade_count"] == 2
    assert item_a["accepted_trade_count"] == 2
    assert item_a["accepted_pnl"] > 0


def test_accepted_rejected_profile_contains_both_buckets() -> None:
    tagged = tag_walk_forward_acceptance(_ledger(), _selection())
    profile = build_accepted_rejected_profile(tagged)

    assert set(profile["selection_bucket"]) == {"accepted", "rejected"}


def test_threshold_stability_counts_changes() -> None:
    stability = build_threshold_stability(_selection())

    assert stability.loc[0, "period_count"] == 2
    assert stability.loc[0, "threshold_changes"] == 0
