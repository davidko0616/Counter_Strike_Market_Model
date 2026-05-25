"""CSFloat listing snapshot parsing and point-in-time feature joins."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path

DEFAULT_RAW_CSFLOAT_DIR = data_path("raw", "csfloat")
DEFAULT_CSFLOAT_FEATURES_OUTPUT = data_path("features", "csfloat_listing_features.parquet")

CSFLOAT_FEATURE_COLUMNS = [
    "csfloat_snapshot_available",
    "csfloat_snapshot_age_days",
    "csfloat_listing_count",
    "csfloat_min_price",
    "csfloat_median_price",
    "csfloat_mean_price",
    "csfloat_price_spread_proxy",
    "csfloat_min_float",
    "csfloat_median_float",
    "csfloat_max_float",
    "csfloat_mean_sticker_count",
    "csfloat_min_price_to_close",
    "csfloat_median_price_to_close",
]


@dataclass(frozen=True)
class CSFloatBuildResult:
    """Result metadata for parsed CSFloat listing snapshots."""

    output_path: Path
    snapshot_count: int
    item_count: int


def build_csfloat_listing_feature_file(
    raw_dir: Path = DEFAULT_RAW_CSFLOAT_DIR,
    output_path: Path = DEFAULT_CSFLOAT_FEATURES_OUTPUT,
) -> CSFloatBuildResult:
    """Parse raw CSFloat JSON envelopes into a compact feature parquet."""
    features = load_raw_csfloat_listing_features(raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return CSFloatBuildResult(
        output_path=output_path,
        snapshot_count=len(features),
        item_count=int(features["market_hash_name"].nunique()) if not features.empty else 0,
    )


def main() -> None:
    """CLI entrypoint for parsing raw CSFloat listing snapshots."""
    import argparse

    parser = argparse.ArgumentParser(description="Build CSFloat listing feature snapshots.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_CSFLOAT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSFLOAT_FEATURES_OUTPUT)
    args = parser.parse_args()

    result = build_csfloat_listing_feature_file(
        raw_dir=args.raw_dir,
        output_path=args.output,
    )
    print(f"CSFloat snapshots: {result.snapshot_count}")
    print(f"CSFloat items: {result.item_count}")
    print(f"Wrote CSFloat features: {_display_path(result.output_path)}")


def load_raw_csfloat_listing_features(raw_dir: Path = DEFAULT_RAW_CSFLOAT_DIR) -> pd.DataFrame:
    """Parse every raw CSFloat listing envelope under *raw_dir*."""
    if not raw_dir.exists():
        return _empty_listing_features()
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        parsed = parse_csfloat_envelope(path)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        return _empty_listing_features()
    frame = pd.DataFrame(rows)
    frame["timestamp_ingested"] = pd.to_datetime(frame["timestamp_ingested"], utc=True)
    return frame.sort_values(["market_hash_name", "timestamp_ingested"]).reset_index(drop=True)


def parse_csfloat_envelope(path: Path) -> dict[str, Any] | None:
    """Parse one saved CSFloat raw response envelope."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    market_hash_name = payload.get("item_hash_name") or payload.get("market_hash_name")
    timestamp_ingested = payload.get("timestamp_ingested")
    if not market_hash_name or not timestamp_ingested:
        return None
    listings = _extract_listing_rows(payload.get("raw_json"))
    prices = [_extract_numeric(listing, ["price", "price_cents", "lowest_price"]) for listing in listings]
    prices = [price for price in prices if price is not None and np.isfinite(price)]
    floats = [_extract_float_value(listing) for listing in listings]
    floats = [value for value in floats if value is not None and np.isfinite(value)]
    sticker_counts = [_extract_sticker_count(listing) for listing in listings]

    min_price = float(np.min(prices)) if prices else np.nan
    median_price = float(np.median(prices)) if prices else np.nan
    mean_price = float(np.mean(prices)) if prices else np.nan
    spread_proxy = float((np.max(prices) - min_price) / min_price) if prices and min_price > 0 else np.nan

    return {
        "market_hash_name": str(market_hash_name),
        "timestamp_ingested": timestamp_ingested,
        "csfloat_listing_count": int(len(listings)),
        "csfloat_min_price": min_price,
        "csfloat_median_price": median_price,
        "csfloat_mean_price": mean_price,
        "csfloat_price_spread_proxy": spread_proxy,
        "csfloat_min_float": float(np.min(floats)) if floats else np.nan,
        "csfloat_median_float": float(np.median(floats)) if floats else np.nan,
        "csfloat_max_float": float(np.max(floats)) if floats else np.nan,
        "csfloat_mean_sticker_count": float(np.mean(sticker_counts)) if sticker_counts else np.nan,
    }


def add_csfloat_listing_features(
    features: pd.DataFrame,
    csfloat_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join CSFloat listing features using an item-level backward as-of join."""
    frame = features.copy()
    frame["timestamp"] = _as_utc_ns(frame["timestamp"])
    if csfloat_features is None or csfloat_features.empty:
        return _add_missing_csfloat_columns(frame)

    snapshots = csfloat_features.copy()
    snapshots["timestamp_ingested"] = _as_utc_ns(snapshots["timestamp_ingested"])
    snapshots = snapshots.sort_values(["market_hash_name", "timestamp_ingested"])
    joined_parts: list[pd.DataFrame] = []
    for item, item_rows in frame.sort_values(["market_hash_name", "timestamp"]).groupby(
        "market_hash_name",
        sort=False,
    ):
        item_snapshots = snapshots[snapshots["market_hash_name"].eq(item)].copy()
        if item_snapshots.empty:
            joined_parts.append(_add_missing_csfloat_columns(item_rows))
            continue
        merged = pd.merge_asof(
            item_rows.sort_values("timestamp"),
            item_snapshots.drop(columns=["market_hash_name"]).sort_values("timestamp_ingested"),
            left_on="timestamp",
            right_on="timestamp_ingested",
            direction="backward",
        )
        joined_parts.append(merged)
    joined = pd.concat(joined_parts, ignore_index=True, sort=False)
    joined = _finalize_csfloat_columns(joined)
    return joined.sort_values(["market_hash_name", "timestamp"]).reset_index(drop=True)


def _finalize_csfloat_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["csfloat_snapshot_available"] = result.get("timestamp_ingested").notna()
    if "timestamp_ingested" in result.columns:
        result["csfloat_snapshot_age_days"] = (
            result["timestamp"] - _as_utc_ns(result["timestamp_ingested"])
        ).dt.total_seconds() / 86400.0
    else:
        result["csfloat_snapshot_age_days"] = np.nan
    if "csfloat_min_price" in result.columns and "close" in result.columns:
        close = pd.to_numeric(result["close"], errors="coerce")
        result["csfloat_min_price_to_close"] = result["csfloat_min_price"] / close
        result["csfloat_median_price_to_close"] = result["csfloat_median_price"] / close
    for column in CSFLOAT_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = False if column == "csfloat_snapshot_available" else np.nan
    return result.drop(columns=["timestamp_ingested"], errors="ignore")


def _add_missing_csfloat_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CSFLOAT_FEATURE_COLUMNS:
        result[column] = False if column == "csfloat_snapshot_available" else np.nan
    return result


def _extract_listing_rows(raw_json: Any) -> list[dict[str, Any]]:
    if isinstance(raw_json, list):
        return [row for row in raw_json if isinstance(row, dict)]
    if not isinstance(raw_json, dict):
        return []
    for key in ("data", "listings", "items", "results"):
        value = raw_json.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _as_utc_ns(values: Any) -> pd.Series:
    index = values.index if isinstance(values, pd.Series) else None
    return pd.Series(pd.to_datetime(values, utc=True), index=index).astype("datetime64[ns, UTC]")


def _extract_numeric(listing: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in listing:
            value = pd.to_numeric(pd.Series([listing.get(key)]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    item = listing.get("item")
    if isinstance(item, dict):
        for key in keys:
            if key in item:
                value = pd.to_numeric(pd.Series([item.get(key)]), errors="coerce").iloc[0]
                if np.isfinite(value):
                    return float(value)
    return None


def _extract_float_value(listing: dict[str, Any]) -> float | None:
    direct = _extract_numeric(listing, ["float_value", "wear_float", "float"])
    if direct is not None:
        return direct
    item = listing.get("item")
    if isinstance(item, dict):
        return _extract_numeric(item, ["float_value", "wear_float", "float"])
    return None


def _extract_sticker_count(listing: dict[str, Any]) -> int:
    item = listing.get("item")
    stickers = listing.get("stickers")
    if stickers is None and isinstance(item, dict):
        stickers = item.get("stickers")
    return len(stickers) if isinstance(stickers, list) else 0


def _empty_listing_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "market_hash_name",
            "timestamp_ingested",
            *[
                column
                for column in CSFLOAT_FEATURE_COLUMNS
                if column
                not in {
                    "csfloat_snapshot_available",
                    "csfloat_snapshot_age_days",
                    "csfloat_min_price_to_close",
                    "csfloat_median_price_to_close",
                }
            ],
        ]
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
