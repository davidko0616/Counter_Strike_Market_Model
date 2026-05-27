from __future__ import annotations

import pandas as pd

from cs_market_model.research.day19_universe_expansion import (
    build_candidate_metrics,
    build_wear_variants,
    build_wear_variants_with_audit,
)


def test_candidate_metrics_use_current_steam_sell_count_when_volume_missing() -> None:
    items = [
        {
            "market_hash_name": "AK-47 | Test (Field-Tested)",
            "selection_score": 1,
            "wear": "Field-Tested",
        }
    ]
    bars = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test (Field-Tested)"] * 3,
            "timestamp": pd.to_datetime(["2026-01-01", "2026-06-20", "2026-06-30"], utc=True),
            "volume": [None, None, None],
            "trade_count": [None, None, None],
        }
    )
    liquidity = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test (Field-Tested)"],
            "steam_sell_count": [12],
        }
    )

    metrics = build_candidate_metrics(items, bars, liquidity)

    assert metrics.loc[0, "liquidity_filter_source"] == "current_steam_sell_count"
    assert metrics.loc[0, "liquidity_score"] == 12
    assert metrics.loc[0, "passes_liquidity_filter"] == True
    assert metrics.loc[0, "passes_history_filter"] == True


def test_build_wear_variants_rewrites_name_and_float_ranges() -> None:
    parent = {
        "market_hash_name": "AK-47 | Test (Field-Tested)",
        "wear": "Field-Tested",
        "min_float": 0,
        "max_float": 1,
    }

    variants = build_wear_variants([parent])

    assert [row["market_hash_name"] for row in variants] == [
        "AK-47 | Test (Minimal Wear)",
        "AK-47 | Test (Well-Worn)",
    ]
    assert variants[0]["min_float"] == 0.07
    assert variants[0]["max_float"] == 0.15
    assert variants[1]["min_float"] == 0.38
    assert variants[1]["max_float"] == 0.45


def test_build_wear_variants_intersects_parent_float_range() -> None:
    parent = {
        "market_hash_name": "AK-47 | Test (Field-Tested)",
        "wear": "Field-Tested",
        "min_float": 0.10,
        "max_float": 0.40,
    }

    variants = build_wear_variants([parent])

    assert variants[0]["wear"] == "Minimal Wear"
    assert variants[0]["min_float"] == 0.10
    assert variants[0]["max_float"] == 0.15
    assert variants[1]["wear"] == "Well-Worn"
    assert variants[1]["min_float"] == 0.38
    assert variants[1]["max_float"] == 0.40


def test_build_wear_variants_excludes_impossible_wear_variants() -> None:
    parent = {
        "market_hash_name": "AK-47 | Test (Field-Tested)",
        "wear": "Field-Tested",
        "min_float": 0.16,
        "max_float": 0.30,
    }

    variants, excluded = build_wear_variants_with_audit([parent])

    assert variants == []
    assert excluded["attempted_wear"].tolist() == ["Minimal Wear", "Well-Worn"]
    assert set(excluded["exclusion_reason"]) == {"no_overlap_with_wear_float_range"}
