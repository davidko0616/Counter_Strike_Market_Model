"""Audit CSFloat listing API coverage for the MVP feature plan."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cs_market_model.config import PROJECT_ROOT
from cs_market_model.security.env import require_env


BASE_URL = "https://csfloat.com"
LISTINGS_ENDPOINT = "/api/v1/listings"
DEFAULT_ITEM = "AK-47 | Redline (Field-Tested)"


def fetch_csfloat_listings(market_hash_name: str, limit: int) -> tuple[Any, dict[str, str]]:
    api_key = require_env("CSFLOAT_API_KEY")
    query = urlencode(
        {
            "market_hash_name": market_hash_name,
            "limit": str(limit),
            "sort_by": "lowest_price",
            "type": "buy_now",
        }
    )
    request = Request(
        f"{BASE_URL}{LISTINGS_ENDPOINT}?{query}",
        headers={
            "Authorization": api_key,
            "User-Agent": "cs-market-model research audit",
            "Accept": "application/json",
        },
    )
    timeout = float(os.getenv("CSMM_REQUEST_TIMEOUT_SECONDS", "15"))
    with urlopen(request, timeout=timeout) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        payload = json.loads(response.read().decode("utf-8"))
    return payload, headers


def extract_listings(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ["data", "listings", "items"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def flatten_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.add(child_prefix)
            paths.update(flatten_paths(child, child_prefix))
    elif isinstance(value, list):
        paths.add(f"{prefix}[]" if prefix else "[]")
        if value:
            paths.update(flatten_paths(value[0], f"{prefix}[]" if prefix else "[]"))
    return paths


def has_any(paths: set[str], candidates: list[str]) -> bool:
    return any(candidate in paths for candidate in candidates)


def coverage_rows(paths: set[str], listings: list[dict[str, Any]]) -> list[dict[str, str]]:
    has_listing_prices = has_any(paths, ["price"])
    has_reference_price = has_any(paths, ["item.scm.price", "reference.base_price"])
    has_predicted_price = has_any(paths, ["reference.predicted_price"])
    has_sticker_reference = any(
        path.endswith(".reference.price") and "stickers" in path for path in paths
    )
    has_seller_stats = any(path.startswith("seller.statistics.") for path in paths)

    return [
        {
            "required_data": "active listings",
            "status": "yes" if listings else "no",
            "evidence": "GET /api/v1/listings returns active listing objects.",
            "mvp_implication": "Usable for current sell-side market snapshots.",
        },
        {
            "required_data": "best ask",
            "status": "yes" if has_listing_prices else "no",
            "evidence": "`price` field is present on listing objects." if has_listing_prices else "`price` missing.",
            "mvp_implication": "Can estimate executable ask from lowest-price listing.",
        },
        {
            "required_data": "sell listing count",
            "status": "partial",
            "evidence": "Docs expose `limit` max 50 and `cursor`; no total-count field is documented.",
            "mvp_implication": "Can count fetched/paginated listings, but exact full depth requires pagination and rate-limit handling.",
        },
        {
            "required_data": "listing depth near market price",
            "status": "yes" if has_listing_prices else "partial",
            "evidence": "Listing prices allow depth bands within 1/3/5 percent on fetched listings.",
            "mvp_implication": "Good for ask-side depth if pagination is implemented.",
        },
        {
            "required_data": "float values",
            "status": "yes" if has_any(paths, ["item.float_value"]) else "no",
            "evidence": "`item.float_value` present." if "item.float_value" in paths else "`item.float_value` missing.",
            "mvp_implication": "Usable for float distribution and float percentile features.",
        },
        {
            "required_data": "paint seed / pattern",
            "status": "yes" if has_any(paths, ["item.paint_seed", "item.paint_index"]) else "no",
            "evidence": "`item.paint_seed` and/or `item.paint_index` present.",
            "mvp_implication": "Useful later for rare pattern filtering; not required for MVP model.",
        },
        {
            "required_data": "stickers",
            "status": "yes" if any(path.startswith("item.stickers") for path in paths) else "no",
            "evidence": "`item.stickers[]` present." if any(path.startswith("item.stickers") for path in paths) else "`item.stickers[]` missing.",
            "mvp_implication": "Can build sticker count and sticker metadata features.",
        },
        {
            "required_data": "sticker reference prices",
            "status": "yes" if has_sticker_reference else "partial",
            "evidence": "Sticker `reference.price` present when stickers include reference metadata.",
            "mvp_implication": "Can build a rough sticker overpay proxy, but only when populated.",
        },
        {
            "required_data": "reference price",
            "status": "yes" if has_reference_price else "partial",
            "evidence": "`reference.base_price` present." if has_reference_price else "No reference price found in sample.",
            "mvp_implication": "Usable as a reference price, but SteamDT remains primary historical source.",
        },
        {
            "required_data": "predicted/fair price",
            "status": "partial" if has_predicted_price else "no",
            "evidence": "`reference.predicted_price` present in live sample."
            if has_predicted_price
            else "Official listing docs do not document a stable predicted-price field.",
            "mvp_implication": "Useful as a snapshot feature, but do not use it as the model label or historical return source.",
        },
        {
            "required_data": "seller trade reliability",
            "status": "yes" if has_seller_stats else "partial",
            "evidence": "`seller.statistics.*` present." if has_seller_stats else "Seller statistics missing in sample.",
            "mvp_implication": "Can use trade count / failed trades / median trade time as execution realism features.",
        },
        {
            "required_data": "bid data / spread",
            "status": "no",
            "evidence": "Official listing docs expose sell listings; bid/order-book bid side is not documented.",
            "mvp_implication": "Cannot directly compute true bid-ask spread from CSFloat alone.",
        },
        {
            "required_data": "historical daily prices",
            "status": "no",
            "evidence": "Official docs do not document historical price bars or sales history endpoint.",
            "mvp_implication": "CSFloat should not replace SteamDT for labels, returns, volatility, or backtest price history.",
        },
        {
            "required_data": "historical volume / trade count",
            "status": "no",
            "evidence": "Listing docs include current listing data and SCM reference volume fields, not CSFloat historical trade volume.",
            "mvp_implication": "Use SteamDT or another verified history source for volume/time-series modeling.",
        },
    ]


def listing_audit_row(market_hash_name: str, payload: Any, listings: list[dict[str, Any]]) -> dict[str, Any]:
    prices = [listing.get("price") for listing in listings if isinstance(listing.get("price"), int | float)]
    item_paths = flatten_paths(listings[0]) if listings else set()
    return {
        "market_hash_name": market_hash_name,
        "timestamp_ingested": datetime.now(UTC).isoformat(),
        "endpoint": f"{BASE_URL}{LISTINGS_ENDPOINT}",
        "response_type": type(payload).__name__,
        "listing_count_returned": len(listings),
        "best_ask_minor_units": min(prices) if prices else "",
        "max_ask_minor_units": max(prices) if prices else "",
        "has_cursor": isinstance(payload, dict) and "cursor" in payload,
        "sample_field_count": len(item_paths),
        "sample_field_paths": "|".join(sorted(item_paths)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_raw_snapshot(payload: Any, market_hash_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = PROJECT_ROOT / "data" / "raw" / "csfloat" / f"csfloat_listings_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "source": "csfloat",
        "item_hash_name": market_hash_name,
        "timestamp_ingested": datetime.now(UTC).isoformat(),
        "request_status": "success",
        "raw_json": payload,
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_csfloat_audit(
    market_hash_name: str,
    limit: int,
    coverage_output: Path,
    listing_audit_output: Path,
    save_raw: bool,
) -> None:
    payload, _headers = fetch_csfloat_listings(market_hash_name, limit=limit)
    listings = extract_listings(payload)
    paths = flatten_paths(listings[0]) if listings else set()

    write_csv(coverage_output, coverage_rows(paths, listings))
    write_csv(listing_audit_output, [listing_audit_row(market_hash_name, payload, listings)])
    raw_path = save_raw_snapshot(payload, market_hash_name) if save_raw else None

    print(f"Wrote CSFloat field coverage: {coverage_output}")
    print(f"Wrote CSFloat listing audit: {listing_audit_output}")
    print(f"Listings returned: {len(listings)}")
    if raw_path:
        print(f"Saved raw CSFloat snapshot: {raw_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CSFloat listing fields for MVP coverage.")
    parser.add_argument("--market-hash-name", default=DEFAULT_ITEM)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=Path("reports/tables/csfloat_api_field_coverage.csv"),
    )
    parser.add_argument(
        "--listing-audit-output",
        type=Path,
        default=Path("reports/tables/csfloat_listing_sample_audit.csv"),
    )
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    build_csfloat_audit(
        market_hash_name=args.market_hash_name,
        limit=args.limit,
        coverage_output=args.coverage_output,
        listing_audit_output=args.listing_audit_output,
        save_raw=args.save_raw,
    )


if __name__ == "__main__":
    main()
