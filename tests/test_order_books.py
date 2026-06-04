from __future__ import annotations

import pytest

from cs_market_model.normalization.order_books import summarize_listing_depth


def test_summarize_listing_depth_counts_price_levels_float_and_stickers() -> None:
    raw_json = {
        "data": [
            {
                "price": 1000,
                "item": {"float_value": 0.11, "stickers": [{"name": "A"}]},
            },
            {
                "price": 1020,
                "item": {"float_value": 0.19, "stickers": [{"name": "B"}, {"name": "C"}]},
            },
            {"price": 1080, "item": {"float_value": 0.25, "stickers": []}},
        ]
    }

    summary = summarize_listing_depth(raw_json)

    assert summary["csfloat_listing_count"] == 3
    assert summary["csfloat_min_price"] == 10.0
    assert summary["csfloat_median_price"] == 10.2
    assert summary["csfloat_depth_1pct"] == 1
    assert summary["csfloat_depth_3pct"] == 2
    assert summary["csfloat_depth_5pct"] == 2
    assert summary["csfloat_median_float"] == pytest.approx(0.19)
    assert summary["csfloat_mean_sticker_count"] == pytest.approx(1.0)


def test_summarize_listing_depth_handles_missing_listing_payload() -> None:
    summary = summarize_listing_depth({"unexpected": []})

    assert summary["csfloat_listing_count"] == 0
    assert summary["csfloat_depth_1pct"] == 0
    assert summary["csfloat_depth_3pct"] == 0
    assert summary["csfloat_depth_5pct"] == 0
