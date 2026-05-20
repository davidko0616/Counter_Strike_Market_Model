"""CSFloat listing collector probe.

This module keeps CSFloat in the MVP as a listing and microstructure source.
Secrets are read from CSFLOAT_API_KEY instead of being stored in source.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from cs_market_model.config import data_path
from cs_market_model.security.env import require_env


BASE_URL = "https://csfloat.com"
LISTINGS_ENDPOINT = "/api/v1/listings"


class CSFloatClient:
    """Small CSFloat API client for listing snapshots."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or require_env("CSFLOAT_API_KEY")
        if not self.api_key:
            raise RuntimeError("CSFLOAT_API_KEY is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("CSMM_REQUEST_TIMEOUT_SECONDS", "15")
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": self.api_key,
            "User-Agent": "cs-market-model research collector",
        }

    def listings(
        self,
        market_hash_name: str,
        limit: int = 50,
        sort_by: str = "lowest_price",
    ) -> dict[str, Any] | list[Any]:
        response = requests.get(
            f"{self.base_url}{LISTINGS_ENDPOINT}",
            headers=self.headers,
            params={
                "market_hash_name": market_hash_name,
                "limit": limit,
                "sort_by": sort_by,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


def safe_item_name(market_hash_name: str) -> str:
    """Create a filesystem-safe item name while preserving readability."""
    replacements = {
        " ": "_",
        "|": "",
        "(": "",
        ")": "",
        "/": "_",
        "\\": "_",
        ":": "_",
    }
    safe = market_hash_name
    for old, new in replacements.items():
        safe = safe.replace(old, new)
    return safe


def save_raw_response(
    payload: dict[str, Any] | list[Any],
    market_hash_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Save an append-only raw CSFloat listing snapshot under data/raw/csfloat."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target_dir = output_dir or data_path("raw", "csfloat")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{safe_item_name(market_hash_name)}_listings_{timestamp}.json"
    envelope = {
        "source": "csfloat",
        "venue": "csfloat",
        "item_hash_name": market_hash_name,
        "timestamp_ingested": datetime.now(UTC).isoformat(),
        "request_status": "success",
        "raw_json": payload,
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def probe_listings(market_hash_name: str, limit: int = 50) -> Path:
    """Fetch and save one CSFloat listings response."""
    client = CSFloatClient()
    payload = client.listings(market_hash_name, limit=limit)
    return save_raw_response(payload, market_hash_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe CSFloat listings for one item.")
    parser.add_argument("market_hash_name")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    output_path = probe_listings(args.market_hash_name, limit=args.limit)
    print(f"Saved CSFloat raw response to {output_path}")


if __name__ == "__main__":
    main()
