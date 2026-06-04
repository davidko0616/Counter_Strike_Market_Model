from __future__ import annotations

import json

import pandas as pd

from cs_market_model.features.csfloat_listings import (
    add_csfloat_listing_features,
    load_raw_csfloat_listing_features,
    parse_csfloat_envelope,
)


def test_parse_csfloat_envelope_extracts_listing_features(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "item_hash_name": "AK-47 | Test (Field-Tested)",
                "timestamp_ingested": "2026-01-02T00:00:00+00:00",
                "raw_json": {
                    "data": [
                        {
                            "price": 100,
                            "item": {
                                "float_value": 0.22,
                                "stickers": [{"name": "A"}, {"name": "B"}],
                            },
                        },
                        {
                            "price": 120,
                            "item": {"float_value": 0.30, "stickers": []},
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    row = parse_csfloat_envelope(path)

    assert row is not None
    assert row["csfloat_listing_count"] == 2
    assert row["csfloat_min_price"] == 1.0
    assert row["csfloat_median_price"] == 1.1
    assert row["csfloat_depth_1pct"] == 1
    assert row["csfloat_depth_5pct"] == 1
    assert row["csfloat_median_float"] == 0.26
    assert row["csfloat_mean_sticker_count"] == 1.0


def test_asof_join_only_uses_past_snapshots() -> None:
    timestamps = pd.to_datetime(
        ["2026-01-01", "2026-01-03"],
        utc=True,
    )
    features = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test (Field-Tested)"] * 2,
            "timestamp": timestamps,
            "close": [10.0, 20.0],
        }
    )
    snapshots = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test (Field-Tested)"],
            "timestamp_ingested": [pd.Timestamp("2026-01-02", tz="UTC")],
            "csfloat_listing_count": [2],
            "csfloat_min_price": [1.0],
            "csfloat_median_price": [1.2],
            "csfloat_mean_price": [1.1],
            "csfloat_price_spread_proxy": [0.2],
            "csfloat_min_float": [0.2],
            "csfloat_median_float": [0.25],
            "csfloat_max_float": [0.3],
            "csfloat_mean_sticker_count": [1.0],
        }
    )

    joined = add_csfloat_listing_features(features, snapshots)

    assert not joined.loc[0, "csfloat_snapshot_available"]
    assert joined.loc[1, "csfloat_snapshot_available"]
    assert joined.loc[1, "csfloat_min_price_to_close"] == 0.05


def test_load_raw_csfloat_listing_features_missing_dir_returns_empty(tmp_path) -> None:
    loaded = load_raw_csfloat_listing_features(tmp_path / "missing")

    assert loaded.empty
    assert "market_hash_name" in loaded.columns
