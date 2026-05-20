"""SteamDT collector probe.

This is a sanitized rewrite of the exploratory script from the sibling project.
Secrets are read from STEAMDT_API_KEY instead of being stored in source.
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


BASE_URL = "https://open.steamdt.com"
KLINE_ENDPOINT = "/open/cs2/item/v1/kline"
PRICE_SINGLE_ENDPOINT = "/open/cs2/v1/price/single"


class SteamDTClient:
    """Small SteamDT API client for raw data collection."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or require_env("STEAMDT_API_KEY")
        if not self.api_key:
            raise RuntimeError("STEAMDT_API_KEY is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("CSMM_REQUEST_TIMEOUT_SECONDS", "15")
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def price_single(self, market_hash_name: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{PRICE_SINGLE_ENDPOINT}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"marketHashName": market_hash_name},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def kline(
        self,
        market_hash_name: str,
        kline_type: int = 2,
        platform: str = "STEAM",
        special_style: str = "",
    ) -> dict[str, Any]:
        body = {
            "marketHashName": market_hash_name,
            "type": kline_type,
            "platform": platform,
            "specialStyle": special_style,
        }
        response = requests.post(
            f"{self.base_url}{KLINE_ENDPOINT}",
            json=body,
            headers=self.headers,
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
    payload: dict[str, Any],
    market_hash_name: str,
    kline_type: int,
    output_dir: Path | None = None,
) -> Path:
    """Save an append-only raw SteamDT response under data/raw/kline."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target_dir = output_dir or data_path("raw", "kline")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{safe_item_name(market_hash_name)}_type{kline_type}_{timestamp}.json"
    envelope = {
        "source": "steamdt",
        "venue": "steam",
        "item_hash_name": market_hash_name,
        "timestamp_ingested": datetime.now(UTC).isoformat(),
        "request_status": "success" if payload.get("success") else "api_failure",
        "raw_json": payload,
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def probe_kline(market_hash_name: str, kline_type: int = 2) -> Path:
    """Fetch and save one SteamDT K-line response."""
    client = SteamDTClient()
    payload = client.kline(market_hash_name, kline_type=kline_type)
    return save_raw_response(payload, market_hash_name, kline_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SteamDT K-line data for one item.")
    parser.add_argument("market_hash_name")
    parser.add_argument("--kline-type", type=int, default=2, help="1=time, 2=day, 3=week")
    args = parser.parse_args()

    output_path = probe_kline(args.market_hash_name, kline_type=args.kline_type)
    print(f"Saved SteamDT raw response to {output_path}")


if __name__ == "__main__":
    main()
