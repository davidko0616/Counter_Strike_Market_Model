from __future__ import annotations

import pandas as pd

from cs_market_model.backtesting.audit_outliers import (
    build_event_outcomes,
    build_outlier_audit,
    summarize_index_regime_performance,
    summarize_outlier_impact,
    summarize_regime_performance,
    summarize_return_sensitivity,
)


def test_outlier_audit_flags_extreme_selected_events() -> None:
    predictions = pd.DataFrame(
        {
            "event_id": ["a", "b", "a"],
            "market_hash_name": ["AK-47 | Test", "AWP | Test", "AK-47 | Test"],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-01"],
                utc=True,
            ),
            "exit_timestamp": pd.to_datetime(
                ["2026-01-02", "2026-01-03", "2026-01-02"],
                utc=True,
            ),
            "label_class": ["Strong Buy", "Bad Time", "Strong Buy"],
            "label_code": [2, 0, 2],
            "label_reason": ["upper_barrier", "lower_barrier", "upper_barrier"],
            "net_return_at_label": [1.5, -0.2, 1.5],
            "vertical_net_return": [1.4, -0.1, 1.4],
            "holding_period_days": [1, 1, 1],
        }
    )
    ledger = pd.DataFrame(
        {
            "event_id": ["a", "a", "b"],
            "model_name": ["forest_rank_blend", "lightgbm_rank_blend", "forest_rank_blend"],
            "realized_net_return": [1.5, 1.5, -0.2],
            "pnl": [0.15, 0.15, -0.02],
            "notional": [0.1, 0.1, 0.1],
            "daily_rank": [1, 1, 2],
            "strong_buy_score": [0.9, 0.8, 0.7],
            "label_market_regime": ["known_event", "known_event", "normal"],
            "cs2_index_regime_bull": [False, False, False],
            "cs2_index_regime_neutral": [False, False, False],
            "cs2_index_regime_bear": [True, True, True],
        }
    )

    outcomes = build_event_outcomes(predictions)
    audit = build_outlier_audit(ledger, outcomes, min_abs_return=1.0)
    impact = summarize_outlier_impact(ledger, audit)
    sensitivity = summarize_return_sensitivity(ledger, audit)
    regime_summary = summarize_regime_performance(ledger)
    index_regime_summary = summarize_index_regime_performance(ledger)

    assert audit["event_id"].tolist() == ["a"]
    assert "selected_by_multiple_models" in audit.loc[0, "audit_flags"]
    assert set(impact["model_name"]) == {"forest_rank_blend", "lightgbm_rank_blend"}
    assert "exclude_audited_extremes" in set(sensitivity["scenario"])
    assert {"known_event", "normal"} <= set(regime_summary["label_market_regime"])
    assert "bear" in set(index_regime_summary["cs2_index_regime"])
