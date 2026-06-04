"""CSFloat listing-depth normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CENTS_KEYS = frozenset({"price", "price_cents", "lowest_price"})
PRICE_KEYS = ("price", "price_cents", "lowest_price")
FLOAT_KEYS = ("float_value", "wear_float", "float")


def extract_listing_rows(raw_json: Any) -> list[dict[str, Any]]:
    """Return listing dictionaries from common CSFloat response envelopes."""
    if isinstance(raw_json, list):
        return [row for row in raw_json if isinstance(row, dict)]
    if not isinstance(raw_json, dict):
        return []
    for key in ("data", "listings", "items", "results"):
        value = raw_json.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def summarize_listing_depth(raw_json: Any) -> dict[str, float | int]:
    """Build listing-count, ask-depth, float, and sticker-count summaries."""
    listings = extract_listing_rows(raw_json)
    prices = [_extract_price(listing) for listing in listings]
    prices = [price for price in prices if price is not None and np.isfinite(price)]
    floats = [_extract_float_value(listing) for listing in listings]
    floats = [value for value in floats if value is not None and np.isfinite(value)]
    sticker_counts = [_extract_sticker_count(listing) for listing in listings]

    min_price = float(np.min(prices)) if prices else np.nan
    median_price = float(np.median(prices)) if prices else np.nan
    mean_price = float(np.mean(prices)) if prices else np.nan
    spread_proxy = (
        float((np.max(prices) - min_price) / min_price) if prices and min_price > 0 else np.nan
    )

    return {
        "csfloat_listing_count": int(len(listings)),
        "csfloat_min_price": min_price,
        "csfloat_median_price": median_price,
        "csfloat_mean_price": mean_price,
        "csfloat_price_spread_proxy": spread_proxy,
        "csfloat_depth_1pct": _depth_within(prices, min_price, 0.01),
        "csfloat_depth_3pct": _depth_within(prices, min_price, 0.03),
        "csfloat_depth_5pct": _depth_within(prices, min_price, 0.05),
        "csfloat_min_float": float(np.min(floats)) if floats else np.nan,
        "csfloat_median_float": float(np.median(floats)) if floats else np.nan,
        "csfloat_max_float": float(np.max(floats)) if floats else np.nan,
        "csfloat_mean_sticker_count": float(np.mean(sticker_counts))
        if sticker_counts
        else np.nan,
        "csfloat_max_sticker_count": int(np.max(sticker_counts)) if sticker_counts else 0,
    }


def _depth_within(prices: list[float], reference_price: float, pct: float) -> int:
    if not prices or not np.isfinite(reference_price) or reference_price <= 0:
        return 0
    limit = reference_price * (1.0 + pct)
    return int(sum(price <= limit for price in prices))


def _extract_price(listing: dict[str, Any]) -> float | None:
    return _extract_numeric(listing, PRICE_KEYS, cents_keys=CENTS_KEYS)


def _extract_float_value(listing: dict[str, Any]) -> float | None:
    direct = _extract_numeric(listing, FLOAT_KEYS)
    if direct is not None:
        return direct
    item = listing.get("item")
    if isinstance(item, dict):
        return _extract_numeric(item, FLOAT_KEYS)
    return None


def _extract_numeric(
    listing: dict[str, Any],
    keys: tuple[str, ...],
    *,
    cents_keys: frozenset[str] = frozenset(),
) -> float | None:
    for key in keys:
        value = _coerce_numeric(listing.get(key)) if key in listing else np.nan
        if np.isfinite(value):
            return float(value / 100.0) if key in cents_keys else float(value)
    item = listing.get("item")
    if isinstance(item, dict):
        for key in keys:
            value = _coerce_numeric(item.get(key)) if key in item else np.nan
            if np.isfinite(value):
                return float(value / 100.0) if key in cents_keys else float(value)
    return None


def _coerce_numeric(value: Any) -> float:
    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(coerced) if pd.notna(coerced) else np.nan


def _extract_sticker_count(listing: dict[str, Any]) -> int:
    item = listing.get("item")
    stickers = listing.get("stickers")
    if stickers is None and isinstance(item, dict):
        stickers = item.get("stickers")
    return len(stickers) if isinstance(stickers, list) else 0
