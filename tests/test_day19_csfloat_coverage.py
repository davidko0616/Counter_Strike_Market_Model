from __future__ import annotations

import pandas as pd

from cs_market_model.research.day19_csfloat_coverage import build_csfloat_coverage_table


def test_coverage_detects_missing_post_snapshot_bars() -> None:
    bars = pd.DataFrame(
        {
            "market_hash_name": ["A", "A"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        }
    )
    csfloat = pd.DataFrame(
        {
            "market_hash_name": ["A"],
            "timestamp_ingested": [pd.Timestamp("2026-01-03", tz="UTC")],
        }
    )

    coverage = build_csfloat_coverage_table(bars, csfloat)

    assert coverage.loc[0, "has_csfloat_snapshot"]
    assert not coverage.loc[0, "has_post_snapshot_bar"]
    assert coverage.loc[0, "days_until_coverage"] == 1.0


def test_coverage_detects_available_post_snapshot_bars() -> None:
    bars = pd.DataFrame(
        {
            "market_hash_name": ["A", "A"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-04"], utc=True),
        }
    )
    csfloat = pd.DataFrame(
        {
            "market_hash_name": ["A"],
            "timestamp_ingested": [pd.Timestamp("2026-01-03", tz="UTC")],
        }
    )

    coverage = build_csfloat_coverage_table(bars, csfloat)

    assert coverage.loc[0, "has_post_snapshot_bar"]
    assert coverage.loc[0, "days_until_coverage"] == 0.0
