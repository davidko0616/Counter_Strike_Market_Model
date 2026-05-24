from __future__ import annotations

from math import log

import pandas as pd
from pandas.testing import assert_frame_equal

from cs_market_model.labeling.build_labels import audit_label_table, build_label_table
from cs_market_model.labeling.cusum import CusumConfig, sample_cusum_events
from cs_market_model.labeling.sensitivity import (
    LabelSensitivityScenario,
    run_label_sensitivity,
)
from cs_market_model.labeling.triple_barrier import (
    FeeModel,
    TripleBarrierConfig,
    apply_triple_barrier_labels,
)


def synthetic_label_features(prices_by_item: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for item_index, (market_hash_name, closes) in enumerate(prices_by_item.items()):
        timestamps = pd.date_range("2026-01-01", periods=len(closes), freq="D", tz="UTC")
        previous_close = None
        for timestamp, close in zip(timestamps, closes, strict=True):
            if previous_close is None:
                log_return = pd.NA
            else:
                log_return = log(close / previous_close)
            rows.append(
                {
                    "item_id": f"item_{item_index}",
                    "market_hash_name": market_hash_name,
                    "venue": "steam",
                    "timestamp": timestamp,
                    "close": close,
                    "log_return_1d": log_return,
                    "return_volatility_30d": 0.03,
                    "row_coverage_30d": 1.0,
                    "effective_staleness_days": 0.0,
                }
            )
            previous_close = close
    return pd.DataFrame(rows)


def test_cusum_events_do_not_change_when_future_rows_are_added() -> None:
    prices = {"AK-47 | Test": [100, 103, 106, 101, 98, 97, 108, 109]}
    features = synthetic_label_features(prices)
    cutoff = pd.Timestamp("2026-01-05", tz="UTC")
    config = CusumConfig(threshold_multiplier=1.0, min_threshold=0.02)

    full_events = sample_cusum_events(features, config)
    prefix_events = sample_cusum_events(features[features["timestamp"] <= cutoff], config)

    assert_frame_equal(
        full_events[full_events["timestamp"] <= cutoff].reset_index(drop=True),
        prefix_events.reset_index(drop=True),
        check_dtype=False,
    )


def test_triple_barrier_assigns_strong_good_and_bad_labels() -> None:
    features = synthetic_label_features(
        {
            "AK-47 | Strong": [100, 112, 113, 114],
            "AWP | Good": [100, 101, 102, 103],
            "M4A4 | Bad": [100, 94, 93, 92],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "event_000000",
                "market_hash_name": "AK-47 | Strong",
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
            },
            {
                "event_id": "event_000001",
                "market_hash_name": "AWP | Good",
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
            },
            {
                "event_id": "event_000002",
                "market_hash_name": "M4A4 | Bad",
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
            },
        ]
    )

    labels = apply_triple_barrier_labels(
        features,
        events,
        fee_model=FeeModel(buy_fee_bps=0, sell_fee_bps=0, slippage_bps=0),
        config=TripleBarrierConfig(
            horizon_days=3,
            required_profit_bps=500,
            stop_loss_bps=500,
            volatility_stop_multiplier=0,
        ),
    )

    label_map = dict(zip(labels["market_hash_name"], labels["label_class"], strict=True))
    assert label_map["AK-47 | Strong"] == "Strong Buy"
    assert label_map["AWP | Good"] == "Good Time"
    assert label_map["M4A4 | Bad"] == "Bad Time"


def test_build_label_table_and_audit_report_counts() -> None:
    features = synthetic_label_features({"AK-47 | Test": [100, 104, 99, 107, 106, 110]})

    labels = build_label_table(
        features,
        cusum_config=CusumConfig(threshold_multiplier=1.0, min_threshold=0.02),
        triple_barrier_config=TripleBarrierConfig(
            horizon_days=2,
            required_profit_bps=300,
            stop_loss_bps=500,
            volatility_stop_multiplier=0,
        ),
        fee_model=FeeModel(buy_fee_bps=0, sell_fee_bps=0, slippage_bps=0),
    )
    audit = audit_label_table(labels)

    assert not labels.empty
    assert {"event_count", "complete_label_count"} <= set(audit["metric"])


def test_label_sensitivity_reports_each_scenario() -> None:
    features = synthetic_label_features({"AK-47 | Test": [100, 104, 99, 107, 106, 110]})
    scenarios = [
        LabelSensitivityScenario(
            "zero_fee_short",
            FeeModel(buy_fee_bps=0, sell_fee_bps=0, slippage_bps=0),
            horizon_days=2,
        ),
        LabelSensitivityScenario(
            "zero_fee_long",
            FeeModel(buy_fee_bps=0, sell_fee_bps=0, slippage_bps=0),
            horizon_days=3,
        ),
    ]

    sensitivity = run_label_sensitivity(
        features,
        scenarios=scenarios,
        cusum_config=CusumConfig(threshold_multiplier=1.0, min_threshold=0.02),
    )

    assert sensitivity["scenario_name"].tolist() == ["zero_fee_short", "zero_fee_long"]
    assert {"strong_buy_pct", "median_net_return_at_label"} <= set(sensitivity.columns)
