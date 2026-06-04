from __future__ import annotations

import pandas as pd

from cs_market_model.research.model_health_monitor import (
    feature_coverage_rows,
    feature_drift_rows,
    paper_trade_health_rows,
    population_stability_index,
    should_keep_previous_model,
    summarize_health_status,
)


def test_feature_coverage_marks_low_important_feature_critical() -> None:
    features = pd.DataFrame({"feature_a": [1.0, None, None], "feature_b": [1.0, 2.0, 3.0]})
    importance = pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b"],
            "importance_gain": [10.0, 1.0],
        }
    )

    rows = feature_coverage_rows(features, importance)

    assert rows[0]["target"] == "feature_a"
    assert rows[0]["status"] == "critical"


def test_paper_trade_health_failed_review_is_critical() -> None:
    summary = pd.DataFrame(
        {
            "scenario": ["paper_trade_5_plus"],
            "accepted_count": [60],
            "review_status": ["failed_profit_factor"],
        }
    )

    rows = paper_trade_health_rows(summary)

    assert rows[0]["status"] == "critical"


def test_population_stability_index_detects_large_shift() -> None:
    stable = population_stability_index(pd.Series(range(100)), pd.Series(range(100)))
    shifted = population_stability_index(pd.Series(range(100)), pd.Series(range(100, 200)))

    assert stable < shifted
    assert shifted > 0.25


def test_feature_drift_rows_warn_on_top_feature_shift() -> None:
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, tz="UTC"),
            "feature_a": [0.0] * 20 + [10.0] * 20,
        }
    )
    importance = pd.DataFrame({"feature": ["feature_a"], "importance_gain": [1.0]})

    rows = feature_drift_rows(features, importance)

    assert rows[0]["status"] == "warning"


def test_should_keep_previous_model_uses_conservative_rollback_rules() -> None:
    assert should_keep_previous_model(
        previous_profit_factor=1.4,
        new_profit_factor=1.1,
        previous_drawdown=-0.05,
        new_drawdown=-0.06,
    )
    assert should_keep_previous_model(
        previous_profit_factor=1.4,
        new_profit_factor=1.35,
        previous_drawdown=-0.05,
        new_drawdown=-0.12,
    )


def test_summarize_health_status_rolls_up_warnings() -> None:
    monitor = pd.DataFrame(
        {
            "check": ["feature_coverage"],
            "target": ["feature_a"],
            "status": ["warning"],
            "detail": ["low"],
        }
    )

    status = summarize_health_status(monitor)

    assert status["status"] == "warning"
    assert status["warnings"] == ["feature_coverage:feature_a:low"]
