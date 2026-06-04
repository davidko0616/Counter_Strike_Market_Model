from __future__ import annotations

import pandas as pd

from cs_market_model.research.day18_csfloat_ablation import (
    MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION,
    build_ablation_setup_table,
)


def test_ablation_setup_reports_not_ready_without_snapshot_coverage() -> None:
    base = pd.DataFrame(
        {
            "market_hash_name": ["A", "A"],
            "timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "close": [1.0, 2.0],
            "feature_a": [0.1, 0.2],
        }
    )
    enabled = base.copy()
    enabled["csfloat_snapshot_available"] = False
    enabled["csfloat_listing_count"] = pd.NA

    table = build_ablation_setup_table(base, enabled, None)
    csfloat = table[table["variant"].eq("steamdt_plus_csfloat")].iloc[0]

    assert csfloat["csfloat_covered_rows"] == 0
    assert not csfloat["ablation_ready"]
    assert csfloat["status"] == "no_csfloat_snapshot_coverage"


def test_ablation_setup_distinguishes_snapshots_without_price_bar_coverage() -> None:
    base = pd.DataFrame(
        {
            "market_hash_name": ["A", "A"],
            "timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "close": [1.0, 2.0],
            "feature_a": [0.1, 0.2],
        }
    )
    enabled = base.copy()
    enabled["csfloat_snapshot_available"] = False
    enabled["csfloat_listing_count"] = pd.NA
    snapshots = pd.DataFrame(
        {
            "market_hash_name": ["A"],
            "timestamp_ingested": [pd.Timestamp("2026-01-03", tz="UTC")],
            "csfloat_listing_count": [3],
        }
    )

    table = build_ablation_setup_table(base, enabled, snapshots)
    csfloat = table[table["variant"].eq("steamdt_plus_csfloat")].iloc[0]

    assert csfloat["csfloat_covered_rows"] == 0
    assert not csfloat["ablation_ready"]
    assert csfloat["status"] == "snapshots_collected_no_price_bar_coverage"


def test_ablation_setup_reports_insufficient_until_coverage_threshold() -> None:
    base = pd.DataFrame(
        {
            "market_hash_name": ["A", "A"],
            "timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "close": [1.0, 2.0],
            "feature_a": [0.1, 0.2],
        }
    )
    enabled = base.copy()
    enabled["csfloat_snapshot_available"] = [False, True]
    enabled["csfloat_listing_count"] = [pd.NA, 3]
    snapshots = pd.DataFrame(
        {
            "market_hash_name": ["A"],
            "timestamp_ingested": [pd.Timestamp("2026-01-02", tz="UTC")],
            "csfloat_listing_count": [3],
        }
    )

    table = build_ablation_setup_table(base, enabled, snapshots)
    csfloat = table[table["variant"].eq("steamdt_plus_csfloat")].iloc[0]

    assert csfloat["csfloat_covered_rows"] == 1
    assert not csfloat["ablation_ready"]
    assert csfloat["status"] == "csfloat_coverage_insufficient"


def test_ablation_setup_reports_ready_with_enough_snapshot_coverage() -> None:
    base = pd.DataFrame(
        {
            "market_hash_name": ["A"] * MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION,
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION,
                tz="UTC",
            ),
            "close": [1.0] * MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION,
            "feature_a": [0.1] * MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION,
        }
    )
    enabled = base.copy()
    enabled["csfloat_snapshot_available"] = True
    enabled["csfloat_listing_count"] = 3
    snapshots = pd.DataFrame(
        {
            "market_hash_name": ["A"],
            "timestamp_ingested": [pd.Timestamp("2026-01-02", tz="UTC")],
            "csfloat_listing_count": [3],
        }
    )

    table = build_ablation_setup_table(base, enabled, snapshots)
    csfloat = table[table["variant"].eq("steamdt_plus_csfloat")].iloc[0]

    assert csfloat["csfloat_covered_rows"] == MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION
    assert csfloat["ablation_ready"]
    assert csfloat["status"] == "ready_for_model_ablation"
