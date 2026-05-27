"""Day 19 MVP universe expansion from candidate price history and liquidity."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cs_market_model.collectors.steamdt import SteamDTClient
from cs_market_model.config import CONFIG_DIR, PROJECT_ROOT, data_path, load_yaml_config, reports_path

DEFAULT_CANDIDATE_CONFIG = CONFIG_DIR / "universe_gun_skins_candidate.yaml"
DEFAULT_CANDIDATE_BARS = data_path("processed", "price_bars_day19_candidates.parquet")
DEFAULT_LIQUIDITY_OUTPUT = data_path("processed", "day19_price_liquidity.csv")
DEFAULT_UNIVERSE_OUTPUT = CONFIG_DIR / "universe_mvp_v2.yaml"
DEFAULT_AUDIT_OUTPUT = reports_path("tables", "day19_universe_expansion_candidates.csv")
DEFAULT_EXCLUDED_VARIANTS_OUTPUT = reports_path("tables", "day19_excluded_wear_variants.csv")

WEAR_RANGES = {
    "Minimal Wear": (0.07, 0.15),
    "Field-Tested": (0.15, 0.38),
    "Well-Worn": (0.38, 0.45),
}


@dataclass(frozen=True)
class Day19UniverseResult:
    """Output paths and counts for the expanded universe build."""

    universe_output: Path
    audit_output: Path
    liquidity_output: Path
    excluded_variants_output: Path
    base_item_count: int
    variant_item_count: int
    total_item_count: int
    eligible_item_count: int
    excluded_variant_count: int


def collect_price_liquidity(
    items: list[dict[str, Any]],
    output_path: Path = DEFAULT_LIQUIDITY_OUTPUT,
    client: SteamDTClient | None = None,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """Collect current SteamDT sell/bid depth for candidate items."""
    resolved_client = client or SteamDTClient()
    rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        market_hash_name = str(item["market_hash_name"])
        row: dict[str, Any] = {
            "market_hash_name": market_hash_name,
            "status": "success",
            "error_type": "",
            "error_message": "",
            "collected_at": datetime.now(UTC).isoformat(),
        }
        try:
            payload = resolved_client.price_single(market_hash_name)
            platforms = payload.get("data") if payload.get("success") else []
            if not isinstance(platforms, list):
                platforms = []
            _add_platform_depth(row, platforms, "STEAM", "steam")
            _add_platform_depth(row, platforms, "BUFF", "buff")
            sell_counts = [
                _numeric(platform.get("sellCount"))
                for platform in platforms
                if isinstance(platform, dict)
            ]
            bid_counts = [
                _numeric(platform.get("biddingCount"))
                for platform in platforms
                if isinstance(platform, dict)
            ]
            row["max_sell_count"] = float(np.nanmax(sell_counts)) if sell_counts else np.nan
            row["max_bid_count"] = float(np.nanmax(bid_counts)) if bid_counts else np.nan
        except Exception as exc:  # noqa: BLE001 - batch collection should report failures.
            row["status"] = "exception"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
        rows.append(row)
        if sleep_seconds > 0 and index < len(items):
            time.sleep(sleep_seconds)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def build_day19_universe(
    candidate_config: Path = DEFAULT_CANDIDATE_CONFIG,
    candidate_bars: Path = DEFAULT_CANDIDATE_BARS,
    liquidity_input: Path = DEFAULT_LIQUIDITY_OUTPUT,
    universe_output: Path = DEFAULT_UNIVERSE_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    excluded_variants_output: Path = DEFAULT_EXCLUDED_VARIANTS_OUTPUT,
    base_count: int = 110,
    variant_parent_count: int = 20,
) -> Day19UniverseResult:
    """Filter candidate items and write the expanded MVP v2 universe config."""
    config = load_yaml_config(candidate_config.name)
    items = config.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {candidate_config} to contain an items list")
    bars = pd.read_parquet(candidate_bars)
    liquidity = pd.read_csv(liquidity_input) if liquidity_input.exists() else pd.DataFrame()

    metrics = build_candidate_metrics(items, bars, liquidity)
    eligible = metrics[metrics["is_day19_eligible"]].copy()
    selected = (
        eligible.sort_values(
            ["liquidity_score", "row_coverage_30d", "history_days", "selection_score", "market_hash_name"],
            ascending=[False, False, False, False, True],
        )
        .head(base_count)
        .copy()
    )
    selected_names = set(selected["market_hash_name"])
    base_items = [
        _tag_item(item, "field_tested_base")
        for item in items
        if str(item["market_hash_name"]) in selected_names
    ]
    base_items = sorted(
        base_items,
        key=lambda item: selected.set_index("market_hash_name").loc[item["market_hash_name"], "liquidity_score"],
        reverse=True,
    )

    top_parents = [item for item in base_items[:variant_parent_count]]
    variants, excluded_variants = build_wear_variants_with_audit(top_parents)
    all_items = _dedupe_items([*base_items, *variants])

    universe_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    excluded_variants_output.parent.mkdir(parents=True, exist_ok=True)
    audit = metrics.merge(
        pd.DataFrame({"market_hash_name": [item["market_hash_name"] for item in all_items], "selected_v2": True}),
        on="market_hash_name",
        how="left",
    )
    audit["selected_v2"] = audit["selected_v2"].fillna(False).astype(bool)
    audit.to_csv(audit_output, index=False)
    excluded_variants.to_csv(excluded_variants_output, index=False)
    _write_universe(universe_output, all_items)

    return Day19UniverseResult(
        universe_output=universe_output,
        audit_output=audit_output,
        liquidity_output=liquidity_input,
        excluded_variants_output=excluded_variants_output,
        base_item_count=len(base_items),
        variant_item_count=len(all_items) - len(base_items),
        total_item_count=len(all_items),
        eligible_item_count=len(eligible),
        excluded_variant_count=len(excluded_variants),
    )


def build_candidate_metrics(
    items: list[dict[str, Any]],
    bars: pd.DataFrame,
    liquidity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Day 19 eligibility metrics for candidate Field-Tested items."""
    item_frame = pd.DataFrame(items)
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = frame["timestamp"].max()
    recent_start = latest_timestamp - pd.Timedelta(days=29)
    grouped = frame.groupby("market_hash_name", sort=False)
    metrics = grouped.agg(
        first_timestamp=("timestamp", "min"),
        latest_timestamp=("timestamp", "max"),
        observed_bar_count=("timestamp", "size"),
    ).reset_index()
    recent_counts = (
        frame[frame["timestamp"].ge(recent_start)]
        .groupby("market_hash_name")
        .size()
        .rename("recent_30d_bar_count")
        .reset_index()
    )
    metrics = metrics.merge(recent_counts, on="market_hash_name", how="left")
    metrics["recent_30d_bar_count"] = metrics["recent_30d_bar_count"].fillna(0).astype(int)
    metrics["history_days"] = (
        metrics["latest_timestamp"] - metrics["first_timestamp"]
    ).dt.days + 1
    metrics["row_coverage_30d"] = metrics["recent_30d_bar_count"] / 30.0

    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    trade_count = pd.to_numeric(frame.get("trade_count"), errors="coerce")
    if volume.notna().any():
        volume_metrics = (
            frame.assign(_volume=volume)
            .groupby("market_hash_name")["_volume"]
            .mean()
            .rename("avg_daily_volume")
            .reset_index()
        )
        liquidity_source = "avg_daily_volume"
    elif trade_count.notna().any():
        volume_metrics = (
            frame.assign(_trade_count=trade_count)
            .groupby("market_hash_name")["_trade_count"]
            .mean()
            .rename("avg_daily_volume")
            .reset_index()
        )
        liquidity_source = "avg_daily_trade_count"
    else:
        volume_metrics = pd.DataFrame({"market_hash_name": metrics["market_hash_name"], "avg_daily_volume": np.nan})
        liquidity_source = "current_steam_sell_count"
    metrics = metrics.merge(volume_metrics, on="market_hash_name", how="left")

    if liquidity is not None and not liquidity.empty:
        liquidity_columns = [
            column
            for column in [
                "market_hash_name",
                "steam_sell_count",
                "buff_sell_count",
                "max_sell_count",
                "max_bid_count",
                "status",
            ]
            if column in liquidity.columns
        ]
        metrics = metrics.merge(liquidity[liquidity_columns], on="market_hash_name", how="left")
    for column in ["steam_sell_count", "buff_sell_count", "max_sell_count", "max_bid_count"]:
        if column not in metrics.columns:
            metrics[column] = np.nan
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    metrics = item_frame.merge(metrics, on="market_hash_name", how="left")
    metrics["selection_score"] = pd.to_numeric(metrics.get("selection_score"), errors="coerce").fillna(0.0)
    metrics["liquidity_filter_source"] = liquidity_source
    metrics["liquidity_score"] = pd.to_numeric(metrics["avg_daily_volume"], errors="coerce")
    missing_liquidity = metrics["liquidity_score"].isna()
    metrics.loc[missing_liquidity, "liquidity_score"] = metrics.loc[
        missing_liquidity,
        "steam_sell_count",
    ]
    still_missing = metrics["liquidity_score"].isna()
    metrics.loc[still_missing, "liquidity_score"] = metrics.loc[still_missing, "max_sell_count"]
    still_missing = metrics["liquidity_score"].isna()
    metrics.loc[still_missing, "liquidity_score"] = metrics.loc[still_missing, "selection_score"]
    metrics["passes_history_filter"] = metrics["history_days"].fillna(0).ge(180)
    metrics["passes_coverage_filter"] = metrics["row_coverage_30d"].fillna(0).gt(0.5)
    metrics["passes_liquidity_filter"] = metrics["liquidity_score"].fillna(0).gt(5)
    metrics["is_day19_eligible"] = (
        metrics["passes_history_filter"]
        & metrics["passes_coverage_filter"]
        & metrics["passes_liquidity_filter"]
    )
    return metrics.sort_values("liquidity_score", ascending=False).reset_index(drop=True)


def build_wear_variants(parent_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create Minimal Wear and Well-Worn variants for selected Field-Tested parents."""
    variants, _excluded = build_wear_variants_with_audit(parent_items)
    return variants


def build_wear_variants_with_audit(
    parent_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Create valid wear variants and audit excluded impossible variants."""
    variants: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for parent in parent_items:
        for wear in ("Minimal Wear", "Well-Worn"):
            parent_min = _numeric(parent.get("min_float"))
            parent_max = _numeric(parent.get("max_float"))
            lower, upper = WEAR_RANGES[wear]
            variant_min = max(parent_min, lower) if np.isfinite(parent_min) else np.nan
            variant_max = min(parent_max, upper) if np.isfinite(parent_max) else np.nan
            variant_name = str(parent["market_hash_name"]).replace(
                "(Field-Tested)",
                f"({wear})",
            )
            if (
                not np.isfinite(parent_min)
                or not np.isfinite(parent_max)
                or variant_min > variant_max
            ):
                excluded_rows.append(
                    {
                        "parent_market_hash_name": parent["market_hash_name"],
                        "attempted_market_hash_name": variant_name,
                        "attempted_wear": wear,
                        "parent_min_float": parent_min,
                        "parent_max_float": parent_max,
                        "wear_min_float": lower,
                        "wear_max_float": upper,
                        "intersection_min_float": variant_min,
                        "intersection_max_float": variant_max,
                        "exclusion_reason": _variant_exclusion_reason(
                            parent_min,
                            parent_max,
                            variant_min,
                            variant_max,
                        ),
                    }
                )
                continue
            variant = dict(parent)
            variant["market_hash_name"] = variant_name
            variant["wear"] = wear
            variant["min_float"] = variant_min
            variant["max_float"] = variant_max
            variant["parent_field_tested_name"] = parent["market_hash_name"]
            variant["include_in_day19_expansion"] = True
            variant["day19_variant_source"] = "top_liquidity_wear_expansion"
            variants.append(variant)
    return variants, _excluded_variants_frame(excluded_rows)


def _excluded_variants_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "parent_market_hash_name",
        "attempted_market_hash_name",
        "attempted_wear",
        "parent_min_float",
        "parent_max_float",
        "wear_min_float",
        "wear_max_float",
        "intersection_min_float",
        "intersection_max_float",
        "exclusion_reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def _variant_exclusion_reason(
    parent_min: float,
    parent_max: float,
    variant_min: float,
    variant_max: float,
) -> str:
    if not np.isfinite(parent_min) or not np.isfinite(parent_max):
        return "missing_parent_float_range"
    if variant_min > variant_max:
        return "no_overlap_with_wear_float_range"
    return "unknown"


def _add_platform_depth(
    row: dict[str, Any],
    platforms: list[Any],
    platform_name: str,
    prefix: str,
) -> None:
    match = next(
        (
            platform
            for platform in platforms
            if isinstance(platform, dict) and platform.get("platform") == platform_name
        ),
        {},
    )
    row[f"{prefix}_sell_price"] = _numeric(match.get("sellPrice"))
    row[f"{prefix}_sell_count"] = _numeric(match.get("sellCount"))
    row[f"{prefix}_bidding_price"] = _numeric(match.get("biddingPrice"))
    row[f"{prefix}_bidding_count"] = _numeric(match.get("biddingCount"))


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _tag_item(item: dict[str, Any], source: str) -> dict[str, Any]:
    tagged = dict(item)
    tagged["include_in_day19_expansion"] = True
    tagged["day19_variant_source"] = source
    return tagged


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        name = str(item["market_hash_name"])
        if name in seen:
            continue
        seen.add(name)
        deduped.append(item)
    return deduped


def _write_universe(path: Path, items: list[dict[str, Any]]) -> None:
    payload = {
        "description": "Day 19 expanded MVP v2 universe: liquid Field-Tested base plus MW/WW variants for top liquidity parents.",
        "items": items,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Day 19 expanded MVP v2 universe.")
    parser.add_argument("--candidate-config", type=Path, default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--candidate-bars", type=Path, default=DEFAULT_CANDIDATE_BARS)
    parser.add_argument("--liquidity-input", type=Path, default=DEFAULT_LIQUIDITY_OUTPUT)
    parser.add_argument("--liquidity-output", type=Path, default=DEFAULT_LIQUIDITY_OUTPUT)
    parser.add_argument("--universe-output", type=Path, default=DEFAULT_UNIVERSE_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--excluded-variants-output", type=Path, default=DEFAULT_EXCLUDED_VARIANTS_OUTPUT)
    parser.add_argument("--base-count", type=int, default=110)
    parser.add_argument("--variant-parent-count", type=int, default=20)
    parser.add_argument("--collect-liquidity", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    config = load_yaml_config(args.candidate_config.name)
    items = config.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {args.candidate_config} to contain an items list")
    liquidity_input = args.liquidity_input
    if args.collect_liquidity:
        collect_price_liquidity(
            items,
            output_path=args.liquidity_output,
            sleep_seconds=args.sleep_seconds,
        )
        liquidity_input = args.liquidity_output

    result = build_day19_universe(
        candidate_config=args.candidate_config,
        candidate_bars=args.candidate_bars,
        liquidity_input=liquidity_input,
        universe_output=args.universe_output,
        audit_output=args.audit_output,
        excluded_variants_output=args.excluded_variants_output,
        base_count=args.base_count,
        variant_parent_count=args.variant_parent_count,
    )
    print(f"Eligible candidate items: {result.eligible_item_count}")
    print(f"Base Field-Tested items: {result.base_item_count}")
    print(f"MW/WW variant items: {result.variant_item_count}")
    print(f"Excluded MW/WW variant items: {result.excluded_variant_count}")
    print(f"Total v2 universe items: {result.total_item_count}")
    print(f"Wrote universe: {_display_path(result.universe_output)}")
    print(f"Wrote audit: {_display_path(result.audit_output)}")
    print(f"Wrote excluded variants: {_display_path(result.excluded_variants_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
