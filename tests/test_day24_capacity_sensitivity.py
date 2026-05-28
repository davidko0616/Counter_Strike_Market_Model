from __future__ import annotations

import pytest
import pandas as pd

from cs_market_model.research.day22_low_price_policy import CapacitySizingConfig
from cs_market_model.research.day24_capacity_sensitivity import (
    CapacitySensitivityScenario,
    build_capacity_summary_rows,
    build_period_attribution,
    survives_capacity_stress,
)


def test_capacity_sensitivity_summary_reports_period_concentration_and_survival() -> None:
    selection = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend"],
            "test_rejected_count": [2],
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

    rows = build_capacity_summary_rows(
        selection,
        sized,
        filled,
        CapacitySensitivityScenario(
            name="test",
            description="test scenario",
            config=CapacitySizingConfig(max_open_positions=10),
        ),
    )

    assert rows[0]["accepted_count"] == 3
    assert rows[0]["capacity_rejected_count"] == 1
    assert rows[0]["total_rejected_count"] == 3
    assert rows[0]["accepted_total_pnl"] == pytest.approx(0.013)
    assert rows[0]["profit_factor"] == 7.5
    assert rows[0]["top_1_period_pnl_share"] == pytest.approx(0.008 / 0.013)
    assert rows[0]["survives_capacity_stress"] is False


def test_capacity_sensitivity_period_attribution_groups_by_test_period() -> None:
    filled = pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend", "lightgbm_rank_blend"],
            "test_period": ["2026-01", "2026-02"],
            "realized_net_return": [0.10, -0.02],
            "pnl": [0.010, -0.002],
            "notional": [0.1, 0.1],
        }
    )

    period = build_period_attribution(filled, "test")

    assert period["test_period"].tolist() == ["2026-01", "2026-02"]
    assert period["accepted_total_pnl"].tolist() == [0.010, -0.002]


def test_survives_capacity_stress_rejects_concentrated_pnl() -> None:
    periods = [f"2026-{month:02d}" for month in range(1, 7)]
    pnl_by_period = [0.45, 0.45, 0.03, 0.03, 0.02, 0.02]
    rows = []
    for period, pnl in zip(periods, pnl_by_period, strict=True):
        for index in range(10):
            rows.append(
                {
                    "model_name": "lightgbm_rank_blend",
                    "market_hash_name": f"{period}-{index}",
                    "entry_timestamp": pd.Timestamp(f"{period}-01", tz="UTC"),
                    "test_period": period,
                    "realized_net_return": 0.10,
                    "pnl": pnl / 10,
                    "notional": 0.01,
                }
            )
    filled = pd.DataFrame(rows)
    period_pnl = filled.groupby("test_period")["pnl"].sum()

    assert survives_capacity_stress(filled, period_pnl, float(period_pnl.sum())) is False
