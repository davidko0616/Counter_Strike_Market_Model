"""Price bar normalization utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


STEAMDT_KLINE_COLUMNS = {
    0: {
        "name": "timestamp_epoch_seconds",
        "meaning": "Exchange timestamp in Unix epoch seconds.",
    },
    1: {
        "name": "open",
        "meaning": "Opening price for the K-line interval.",
    },
    2: {
        "name": "close",
        "meaning": "Closing price for the K-line interval.",
    },
    3: {
        "name": "high",
        "meaning": "Highest observed price for the K-line interval.",
    },
    4: {
        "name": "low",
        "meaning": "Lowest observed price for the K-line interval.",
    },
}


@dataclass(frozen=True)
class SteamDTKlineParseResult:
    """Parsed SteamDT K-line data and audit metadata."""

    bars: pd.DataFrame
    audit: dict[str, Any]
    column_docs: pd.DataFrame


def stable_item_id(market_hash_name: str) -> str:
    """Create a stable internal item id from the market hash name."""
    digest = hashlib.sha1(market_hash_name.encode("utf-8")).hexdigest()[:12]
    readable = "".join(
        char.lower() if char.isalnum() else "_" for char in market_hash_name
    )
    readable = "_".join(part for part in readable.split("_") if part)
    return f"{readable[:48]}_{digest}"


def load_steamdt_kline_json(path: Path) -> dict[str, Any]:
    """Load a raw SteamDT K-line JSON response."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def infer_market_hash_name(path: Path, fallback: str = "AK-47 | Redline (Field-Tested)") -> str:
    """Infer the sample item name from the known Day 2 sample filename."""
    if path.name.startswith("AK-47__Redline_Field-Tested"):
        return "AK-47 | Redline (Field-Tested)"
    return fallback


def parse_steamdt_kline_payload(
    payload: dict[str, Any],
    market_hash_name: str,
    venue: str = "steam",
    currency: str = "CNY",
) -> pd.DataFrame:
    """Convert a SteamDT K-line response into normalized daily price bars."""
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ValueError("SteamDT payload field 'data' must be a list")

    raw = pd.DataFrame(rows, columns=[spec["name"] for spec in STEAMDT_KLINE_COLUMNS.values()])
    if raw.empty:
        return _empty_price_bars()

    raw["timestamp_epoch_seconds"] = pd.to_numeric(
        raw["timestamp_epoch_seconds"], errors="raise"
    ).astype("int64")
    for column in ["open", "close", "high", "low"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    timestamp_exchange = pd.to_datetime(raw["timestamp_epoch_seconds"], unit="s", utc=True)
    bars = pd.DataFrame(
        {
            "item_id": stable_item_id(market_hash_name),
            "market_hash_name": market_hash_name,
            "venue": venue,
            "timestamp": timestamp_exchange.dt.floor("D"),
            "timestamp_exchange": timestamp_exchange,
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "median_price": (raw["high"] + raw["low"]) / 2,
            "volume": pd.NA,
            "trade_count": pd.NA,
            "currency": currency,
            "data_resolution": "1d",
            "is_forward_filled": False,
            "staleness_days": 0,
            "source": "steamdt",
        }
    )
    return bars.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def audit_steamdt_kline_payload(
    payload: dict[str, Any],
    bars: pd.DataFrame,
    source_file: Path,
    market_hash_name: str,
    currency: str = "CNY",
) -> dict[str, Any]:
    """Build a one-row audit summary for a raw SteamDT K-line sample."""
    rows = payload.get("data", [])
    row_lengths = sorted({len(row) for row in rows if isinstance(row, list)})
    duplicate_timestamps = int(bars["timestamp"].duplicated().sum()) if not bars.empty else 0
    price_columns = ["open", "high", "low", "close"]
    missing_values = int(bars[price_columns].isna().sum().sum()) if not bars.empty else 0
    non_positive_price_rows = (
        int((bars[price_columns].le(0).any(axis=1)).sum()) if not bars.empty else 0
    )
    high_low_inversions = int((bars["high"] < bars["low"]).sum()) if not bars.empty else 0
    open_outside_high_low = (
        int(((bars["open"] > bars["high"]) | (bars["open"] < bars["low"])).sum())
        if not bars.empty
        else 0
    )
    close_outside_high_low = (
        int(((bars["close"] > bars["high"]) | (bars["close"] < bars["low"])).sum())
        if not bars.empty
        else 0
    )

    return {
        "item_id": stable_item_id(market_hash_name),
        "market_hash_name": market_hash_name,
        "source": "steamdt",
        "venue": "steam",
        "source_file": str(source_file),
        "success": bool(payload.get("success")),
        "error_code": payload.get("errorCode"),
        "error_message": payload.get("errorMsg"),
        "row_count": int(len(rows)),
        "raw_row_length_values": "|".join(str(value) for value in row_lengths),
        "min_timestamp": bars["timestamp"].min().isoformat() if not bars.empty else "",
        "max_timestamp": bars["timestamp"].max().isoformat() if not bars.empty else "",
        "duplicate_timestamps": duplicate_timestamps,
        "missing_price_values": missing_values,
        "non_positive_price_rows": non_positive_price_rows,
        "high_low_inversions": high_low_inversions,
        "open_outside_high_low": open_outside_high_low,
        "close_outside_high_low": close_outside_high_low,
        "has_volume_field": False,
        "has_trade_count_field": False,
        "currency": currency,
        "notes": "SteamDT Day K sample has timestamp, open, close, high, low; no volume field.",
    }


def steamdt_column_docs() -> pd.DataFrame:
    """Return the documented SteamDT K-line column mapping."""
    return pd.DataFrame(
        [
            {
                "raw_index": index,
                "normalized_name": spec["name"],
                "meaning": spec["meaning"],
            }
            for index, spec in STEAMDT_KLINE_COLUMNS.items()
        ]
    )


def parse_and_audit_steamdt_kline(
    path: Path,
    market_hash_name: str | None = None,
    currency: str = "CNY",
) -> SteamDTKlineParseResult:
    """Load, normalize, and audit a SteamDT K-line JSON file."""
    payload = load_steamdt_kline_json(path)
    resolved_name = market_hash_name or infer_market_hash_name(path)
    bars = parse_steamdt_kline_payload(payload, resolved_name, currency=currency)
    audit = audit_steamdt_kline_payload(payload, bars, path, resolved_name, currency=currency)
    return SteamDTKlineParseResult(
        bars=bars,
        audit=audit,
        column_docs=steamdt_column_docs(),
    )


def _empty_price_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "item_id",
            "market_hash_name",
            "venue",
            "timestamp",
            "timestamp_exchange",
            "open",
            "high",
            "low",
            "close",
            "median_price",
            "volume",
            "trade_count",
            "currency",
            "data_resolution",
            "is_forward_filled",
            "staleness_days",
            "source",
        ]
    )

