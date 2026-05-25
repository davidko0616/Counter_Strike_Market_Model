"""Batch SteamDT collection for the Day 3 MVP universe."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cs_market_model.collectors.steamdt import SteamDTClient, save_raw_response
from cs_market_model.config import PROJECT_ROOT, data_path, load_yaml_config, reports_path
from cs_market_model.normalization.prices import parse_steamdt_kline_payload

DEFAULT_CONFIG_NAME = "universe_day3_first_pull.yaml"
DEFAULT_BARS_OUTPUT = data_path("processed", "price_bars_day3_first_pull.parquet")
DEFAULT_METADATA_OUTPUT = data_path("processed", "item_metadata_day3_first_pull.parquet")
DEFAULT_LOG_OUTPUT = reports_path("tables", "day3_steamdt_ingestion_log.csv")


@dataclass(frozen=True)
class BatchCollectionResult:
    """File outputs and ingestion summary for a batch collection run."""

    bars_output: Path
    metadata_output: Path
    log_output: Path
    requested_items: int
    successful_items: int
    failed_items: int
    price_bar_rows: int


def load_universe_items(config_name: str = DEFAULT_CONFIG_NAME) -> list[dict[str, Any]]:
    """Load item dictionaries from a universe config."""
    config = load_yaml_config(config_name)
    items = config.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {config_name} to contain an 'items' list")
    if not all(isinstance(item, dict) and item.get("market_hash_name") for item in items):
        raise ValueError(f"Every item in {config_name} must define market_hash_name")
    return items


def build_item_metadata(items: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Build a flat item metadata table from universe config rows."""
    rows = []
    for item in items:
        rows.append(
            {
                "market_hash_name": item["market_hash_name"],
                "weapon_type": item.get("weapon_type"),
                "category": item.get("category"),
                "requested_rarity": item.get("requested_rarity"),
                "rarity": item.get("rarity"),
                "rarity_substituted": item.get("rarity_substituted", False),
                "wear": item.get("wear"),
                "source_type": item.get("source_type"),
                "collections": "; ".join(item.get("collections") or []),
                "cases": "; ".join(item.get("cases") or []),
                "min_float": item.get("min_float"),
                "max_float": item.get("max_float"),
                "stattrak_flag": False,
                "souvenir_flag": False,
            }
        )
    return pd.DataFrame(rows)


def collect_steamdt_universe(
    items: Sequence[dict[str, Any]],
    client: SteamDTClient | None = None,
    kline_type: int = 2,
    currency: str = "CNY",
    raw_dir: Path | None = None,
    bars_output: Path = DEFAULT_BARS_OUTPUT,
    metadata_output: Path = DEFAULT_METADATA_OUTPUT,
    log_output: Path = DEFAULT_LOG_OUTPUT,
    sleep_seconds: float = 0.0,
    stop_on_error: bool = False,
) -> BatchCollectionResult:
    """Collect raw SteamDT K-lines and normalized bars for a universe of items."""
    resolved_client = client or SteamDTClient()
    raw_output_dir = raw_dir or data_path("raw", "kline")
    bars_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    log_output.parent.mkdir(parents=True, exist_ok=True)

    all_bars: list[pd.DataFrame] = []
    log_rows: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        market_hash_name = str(item["market_hash_name"])
        started_at = datetime.now(UTC)
        raw_path = ""
        status = "success"
        error_type = ""
        error_message = ""
        row_count = 0

        try:
            payload = resolved_client.kline(market_hash_name, kline_type=kline_type)
            raw_path = str(save_raw_response(payload, market_hash_name, kline_type, raw_output_dir))
            if not payload.get("success"):
                status = "api_failure"
                error_type = "SteamDTAPIError"
                error_message = str(payload.get("errorMsg") or payload.get("errorCode") or "")
            else:
                bars = parse_steamdt_kline_payload(
                    payload,
                    market_hash_name,
                    currency=currency,
                )
                row_count = int(len(bars))
                all_bars.append(bars)
        except Exception as exc:  # noqa: BLE001 - batch logs failures and optionally continues.
            status = "exception"
            error_type = type(exc).__name__
            error_message = str(exc)
            if stop_on_error:
                raise
        finally:
            completed_at = datetime.now(UTC)
            log_rows.append(
                {
                    "item_index": index,
                    "market_hash_name": market_hash_name,
                    "status": status,
                    "row_count": row_count,
                    "raw_path": raw_path,
                    "started_at": started_at.isoformat(),
                    "completed_at": completed_at.isoformat(),
                    "duration_seconds": (completed_at - started_at).total_seconds(),
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
            if sleep_seconds > 0 and index < len(items):
                time.sleep(sleep_seconds)

    if all_bars:
        price_bars = pd.concat(all_bars, ignore_index=True)
        price_bars = price_bars.sort_values(["market_hash_name", "timestamp"]).reset_index(
            drop=True
        )
    else:
        price_bars = pd.DataFrame()

    price_bars.to_parquet(bars_output, index=False)
    build_item_metadata(items).to_parquet(metadata_output, index=False)
    pd.DataFrame(log_rows).to_csv(log_output, index=False)

    successful_items = sum(row["status"] == "success" for row in log_rows)
    failed_items = len(log_rows) - successful_items
    return BatchCollectionResult(
        bars_output=bars_output,
        metadata_output=metadata_output,
        log_output=log_output,
        requested_items=len(items),
        successful_items=successful_items,
        failed_items=failed_items,
        price_bar_rows=int(len(price_bars)),
    )


def _limited_items(items: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    materialized = list(items)
    if limit is None:
        return materialized
    if limit < 1:
        raise ValueError("--limit must be positive when provided")
    return materialized[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SteamDT K-lines for a universe config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--kline-type", type=int, default=2, help="1=time, 2=day, 3=week")
    parser.add_argument("--currency", default="CNY")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--raw-dir", type=Path, default=data_path("raw", "kline"))
    parser.add_argument("--bars-output", type=Path, default=DEFAULT_BARS_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--log-output", type=Path, default=DEFAULT_LOG_OUTPUT)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    items = _limited_items(load_universe_items(args.config), args.limit)
    result = collect_steamdt_universe(
        items=items,
        kline_type=args.kline_type,
        currency=args.currency,
        raw_dir=args.raw_dir,
        bars_output=args.bars_output,
        metadata_output=args.metadata_output,
        log_output=args.log_output,
        sleep_seconds=args.sleep_seconds,
        stop_on_error=args.stop_on_error,
    )

    print(f"Requested items: {result.requested_items}")
    print(f"Successful items: {result.successful_items}")
    print(f"Failed items: {result.failed_items}")
    print(f"Price bar rows: {result.price_bar_rows}")
    print(f"Wrote bars: {_display_path(result.bars_output)}")
    print(f"Wrote metadata: {_display_path(result.metadata_output)}")
    print(f"Wrote ingestion log: {_display_path(result.log_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
