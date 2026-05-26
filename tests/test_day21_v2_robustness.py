from __future__ import annotations

import pytest
import pandas as pd

from cs_market_model.research.day21_v2_robustness import (
    build_bucket_attribution,
    build_scenario_robustness,
    build_top_trade_audit,
    enrich_trades_with_features,
)


def test_v2_robustness_scenarios_cap_and_remove_top_trades() -> None:
    trades = _trades()
    features = _features()
    enriched = enrich_trades_with_features(trades, features)

    scenarios = build_scenario_robustness(enriched)
    lightgbm = scenarios[scenarios["model_name"].eq("lightgbm_rank_blend")]

    raw = lightgbm[lightgbm["scenario"].eq("raw_all_accepted")].iloc[0]
    cap = lightgbm[lightgbm["scenario"].eq("cap_returns_50pct")].iloc[0]
    without_top = lightgbm[lightgbm["scenario"].eq("exclude_top_1_pnl_trades")].iloc[0]

    assert raw["total_pnl"] == 0.24
    assert cap["total_pnl"] == 0.04
    assert without_top["total_pnl"] == pytest.approx(-0.01)


def test_v2_robustness_top_trade_flags_low_price_extreme_return() -> None:
    enriched = enrich_trades_with_features(_trades(), _features())

    top = build_top_trade_audit(enriched, top_n=1)

    assert "entry_price_lt_1" in top.loc[0, "audit_flags"]
    assert "return_ge_100pct" in top.loc[0, "audit_flags"]


def test_v2_bucket_attribution_includes_wear_and_price_bucket() -> None:
    enriched = enrich_trades_with_features(_trades(), _features())

    buckets = build_bucket_attribution(enriched)

    assert {"wear", "price_bucket"} <= set(buckets["bucket_type"])
    assert "Minimal Wear" in set(buckets["bucket_value"])


def test_v2_enrichment_fills_blank_category_from_features() -> None:
    trades = _trades()
    trades["category"] = ""

    enriched = enrich_trades_with_features(trades, _features())

    assert enriched.loc[0, "category"] == "rifle"


def test_v2_enrichment_dedupes_duplicate_feature_rows_with_diagnostics() -> None:
    features = pd.concat([_features(), _features().iloc[[0]]], ignore_index=True)

    enriched = enrich_trades_with_features(_trades(), features)

    assert len(enriched) == len(_trades())
    duplicate = enriched[enriched["market_hash_name"].eq("A")].iloc[0]
    assert duplicate["entry_close"] == 0.5
    assert duplicate["feature_join_duplicate_count"] == 2
    assert duplicate["feature_join_duplicate_conflict"] == False
    assert duplicate["feature_join_conflicting_columns"] == ""


def test_v2_enrichment_reports_conflicting_duplicate_feature_rows() -> None:
    features = _features()
    conflicting = features.iloc[[0]].copy()
    conflicting["close"] = 0.7
    features = pd.concat([features, conflicting], ignore_index=True)

    enriched = enrich_trades_with_features(_trades(), features)
    duplicate = enriched[enriched["market_hash_name"].eq("A")].iloc[0]

    assert duplicate["entry_close"] == 0.5
    assert duplicate["feature_join_duplicate_count"] == 2
    assert duplicate["feature_join_duplicate_conflict"] == True
    assert duplicate["feature_join_conflicting_columns"] == "close"


def test_v2_enrichment_strict_mode_rejects_conflicting_duplicate_feature_rows() -> None:
    features = _features()
    conflicting = features.iloc[[0]].copy()
    conflicting["close"] = 0.7
    features = pd.concat([features, conflicting], ignore_index=True)

    with pytest.raises(ValueError, match="Conflicting duplicate feature rows"):
        enrich_trades_with_features(
            _trades(),
            features,
            strict_duplicate_conflicts=True,
        )


def test_v2_top_trade_flags_feature_join_duplicate_conflict() -> None:
    features = _features()
    conflicting = features.iloc[[0]].copy()
    conflicting["close"] = 0.7
    features = pd.concat([features, conflicting], ignore_index=True)
    enriched = enrich_trades_with_features(_trades(), features)

    top = build_top_trade_audit(enriched, top_n=1)

    assert "feature_join_duplicate_conflict" in top.loc[0, "audit_flags"]


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["lightgbm_rank_blend", "lightgbm_rank_blend"],
            "market_hash_name": ["A", "B"],
            "entry_timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "exit_timestamp": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
            "label_market_regime": ["normal", "normal"],
            "label_overlaps_known_market_event": [False, False],
            "realized_net_return": [2.5, -0.1],
            "notional": [0.1, 0.1],
            "pnl": [0.25, -0.01],
            "strong_buy_score": [0.9, 0.8],
            "liquidity_quality_score_30d": [1.0, 1.0],
            "category": ["rifle", "pistol"],
        }
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_hash_name": ["A", "B"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "close": [0.5, 10.0],
            "weapon_type": ["AK-47", "Glock-18"],
            "rarity": ["Consumer Grade", "Restricted"],
            "wear": ["Minimal Wear", "Field-Tested"],
            "source_type": ["collection", "case"],
            "category": ["rifle", "pistol"],
            "row_coverage_30d": [1.0, 1.0],
            "effective_staleness_days": [0.0, 0.0],
            "abs_return_max_7d": [0.1, 0.1],
            "abs_return_max_30d": [0.2, 0.2],
            "price_jump_50pct_count_7d": [0, 0],
            "price_jump_100pct_count_7d": [0, 0],
            "price_jump_50pct_count_30d": [0, 0],
            "price_jump_100pct_count_30d": [0, 0],
        }
    )
