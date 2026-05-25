"""Batch CSFloat listing snapshot collector for configured item universes."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cs_market_model.collectors.csfloat import CSFloatClient, save_raw_response
from cs_market_model.config import PROJECT_ROOT, load_yaml_config, reports_path

DEFAULT_UNIVERSE_CONFIG = "universe_day3_first_pull.yaml"
DEFAULT_LOG_OUTPUT = reports_path("tables", "day18_5_csfloat_snapshot_log.csv")


@dataclass(frozen=True)
class CSFloatBatchResult:
    """Counts and output path for a CSFloat batch snapshot run."""

    log_output: Path
    requested_count: int
    success_count: int
    failure_count: int


def load_universe_items(config_name: str = DEFAULT_UNIVERSE_CONFIG) -> list[str]:
    """Load market hash names from a universe config."""
    config = load_yaml_config(config_name)
    raw_items = config.get("items", []) or []
    items: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if item.get("include_in_day3_first_pull", True) is False:
            continue
        market_hash_name = item.get("market_hash_name")
        if market_hash_name:
            items.append(str(market_hash_name))
    return items


def collect_csfloat_snapshots(
    items: list[str],
    *,
    client: Any | None = None,
    output_dir: Path | None = None,
    log_output: Path = DEFAULT_LOG_OUTPUT,
    limit: int = 50,
    sleep_seconds: float = 0.25,
) -> CSFloatBatchResult:
    """Collect one CSFloat listing snapshot for each item."""
    resolved_client = client or CSFloatClient()
    rows: list[dict[str, Any]] = []
    for index, market_hash_name in enumerate(items, start=1):
        row: dict[str, Any] = {
            "market_hash_name": market_hash_name,
            "request_index": index,
            "request_limit": limit,
            "status": "pending",
            "raw_output_path": None,
            "error": None,
        }
        try:
            payload = resolved_client.listings(market_hash_name, limit=limit)
            raw_path = save_raw_response(
                payload,
                market_hash_name,
                output_dir=output_dir,
            )
            row["status"] = "success"
            row["raw_output_path"] = _display_path(raw_path)
        except Exception as exc:  # pragma: no cover - exercised with fake client in tests.
            row["status"] = "failure"
            row["error"] = str(exc)
        rows.append(row)
        if sleep_seconds > 0 and index < len(items):
            time.sleep(sleep_seconds)

    log = pd.DataFrame(rows)
    log_output.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_output, index=False)
    success_count = int((log["status"] == "success").sum()) if not log.empty else 0
    failure_count = int((log["status"] == "failure").sum()) if not log.empty else 0
    return CSFloatBatchResult(
        log_output=log_output,
        requested_count=len(items),
        success_count=success_count,
        failure_count=failure_count,
    )


def collect_configured_universe(
    config_name: str = DEFAULT_UNIVERSE_CONFIG,
    *,
    output_dir: Path | None = None,
    log_output: Path = DEFAULT_LOG_OUTPUT,
    limit: int = 50,
    sleep_seconds: float = 0.25,
) -> CSFloatBatchResult:
    """Collect CSFloat snapshots for the configured universe."""
    items = load_universe_items(config_name)
    return collect_csfloat_snapshots(
        items,
        output_dir=output_dir,
        log_output=log_output,
        limit=limit,
        sleep_seconds=sleep_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CSFloat snapshots for a universe.")
    parser.add_argument("--config", default=DEFAULT_UNIVERSE_CONFIG)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=DEFAULT_LOG_OUTPUT)
    args = parser.parse_args()

    result = collect_configured_universe(
        args.config,
        output_dir=args.output_dir,
        log_output=args.log_output,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )
    print(f"CSFloat requested: {result.requested_count}")
    print(f"CSFloat successes: {result.success_count}")
    print(f"CSFloat failures: {result.failure_count}")
    print(f"Wrote log: {_display_path(result.log_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
