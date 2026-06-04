"""Day 19 coverage monitor for CSFloat snapshots versus price bars."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path
from cs_market_model.features.csfloat_listings import DEFAULT_CSFLOAT_FEATURES_OUTPUT

DEFAULT_PRICE_BARS_INPUT = data_path("processed", "price_bars_day3_first_pull.parquet")
DEFAULT_COVERAGE_OUTPUT = reports_path("tables", "day19_csfloat_coverage.csv")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "day19_csfloat_coverage_report.md")


@dataclass(frozen=True)
class Day19CoverageResult:
    """Output paths and counts for Day 19 coverage monitoring."""

    coverage_output: Path
    report_output: Path
    item_count: int
    items_with_snapshots: int
    items_with_post_snapshot_bars: int


def run_day19_csfloat_coverage(
    price_bars_input: Path = DEFAULT_PRICE_BARS_INPUT,
    csfloat_features_input: Path = DEFAULT_CSFLOAT_FEATURES_OUTPUT,
    coverage_output: Path = DEFAULT_COVERAGE_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> Day19CoverageResult:
    """Build a coverage monitor for CSFloat feature usability."""
    price_bars = pd.read_parquet(price_bars_input)
    csfloat = (
        pd.read_parquet(csfloat_features_input)
        if csfloat_features_input.exists()
        else pd.DataFrame()
    )
    coverage = build_csfloat_coverage_table(price_bars, csfloat)

    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_output, index=False)
    report_output.write_text(build_coverage_report(coverage), encoding="utf-8")

    return Day19CoverageResult(
        coverage_output=coverage_output,
        report_output=report_output,
        item_count=int(len(coverage)),
        items_with_snapshots=int(coverage["has_csfloat_snapshot"].sum())
        if not coverage.empty
        else 0,
        items_with_post_snapshot_bars=int(coverage["has_post_snapshot_bar"].sum())
        if not coverage.empty
        else 0,
    )


def build_csfloat_coverage_table(
    price_bars: pd.DataFrame,
    csfloat: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether each item has price bars after CSFloat snapshots."""
    bars = price_bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bar_summary = bars.groupby("market_hash_name", as_index=False).agg(
        first_price_bar_timestamp=("timestamp", "min"),
        latest_price_bar_timestamp=("timestamp", "max"),
        price_bar_rows=("timestamp", "size"),
    )

    if csfloat.empty:
        bar_summary["first_csfloat_snapshot_timestamp"] = pd.NaT
        bar_summary["latest_csfloat_snapshot_timestamp"] = pd.NaT
        bar_summary["csfloat_snapshot_count"] = 0
    else:
        snapshots = csfloat.copy()
        snapshots["timestamp_ingested"] = pd.to_datetime(
            snapshots["timestamp_ingested"],
            utc=True,
        )
        snapshot_summary = snapshots.groupby("market_hash_name", as_index=False).agg(
            first_csfloat_snapshot_timestamp=("timestamp_ingested", "min"),
            latest_csfloat_snapshot_timestamp=("timestamp_ingested", "max"),
            csfloat_snapshot_count=("timestamp_ingested", "size"),
        )
        bar_summary = bar_summary.merge(
            snapshot_summary,
            on="market_hash_name",
            how="left",
        )
        bar_summary["csfloat_snapshot_count"] = (
            bar_summary["csfloat_snapshot_count"].fillna(0).astype(int)
        )

    for column in (
        "first_price_bar_timestamp",
        "latest_price_bar_timestamp",
        "first_csfloat_snapshot_timestamp",
        "latest_csfloat_snapshot_timestamp",
    ):
        bar_summary[column] = pd.to_datetime(bar_summary[column], utc=True)

    bar_summary["has_csfloat_snapshot"] = bar_summary["csfloat_snapshot_count"] > 0
    bar_summary["has_post_snapshot_bar"] = bar_summary["has_csfloat_snapshot"] & (
        bar_summary["latest_price_bar_timestamp"] >= bar_summary["first_csfloat_snapshot_timestamp"]
    )
    bar_summary["days_until_coverage"] = (
        bar_summary["first_csfloat_snapshot_timestamp"] - bar_summary["latest_price_bar_timestamp"]
    ).dt.total_seconds() / 86400.0
    bar_summary.loc[bar_summary["has_post_snapshot_bar"], "days_until_coverage"] = 0.0
    return bar_summary.sort_values("market_hash_name").reset_index(drop=True)


def build_coverage_report(coverage: pd.DataFrame) -> str:
    """Create a short Markdown report for Day 19."""
    if coverage.empty:
        summary = "No price-bar rows were available."
    else:
        item_count = len(coverage)
        snapshot_items = int(coverage["has_csfloat_snapshot"].sum())
        covered_items = int(coverage["has_post_snapshot_bar"].sum())
        latest_bar = coverage["latest_price_bar_timestamp"].max()
        first_snapshot = coverage["first_csfloat_snapshot_timestamp"].min()
        summary = (
            f"CSFloat snapshots exist for {snapshot_items}/{item_count} items. "
            f"Post-snapshot price bars exist for {covered_items}/{item_count} items. "
            f"Latest price bar: {latest_bar}. First CSFloat snapshot: {first_snapshot}."
        )
    return "\n".join(
        [
            "# Day 19 CSFloat Coverage Monitor",
            "",
            "## Summary",
            "",
            summary,
            "",
            "## Interpretation",
            "",
            "CSFloat features are point-in-time safe only when a price-bar row is at or after",
            "the snapshot ingestion timestamp. If post-snapshot bars are missing, model",
            "validation must remain SteamDT-only for historical rows.",
            "",
            "## Next Plan",
            "",
            "Continue collecting daily CSFloat snapshots and refresh SteamDT bars until",
            "post-snapshot rows exist. Then rerun Day 18 ablation and Day 16 rejection",
            "validation on the CSFloat-enabled feature set.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 19 CSFloat coverage monitor.")
    parser.add_argument("--price-bars-input", type=Path, default=DEFAULT_PRICE_BARS_INPUT)
    parser.add_argument(
        "--csfloat-features-input", type=Path, default=DEFAULT_CSFLOAT_FEATURES_OUTPUT
    )
    parser.add_argument("--coverage-output", type=Path, default=DEFAULT_COVERAGE_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_day19_csfloat_coverage(
        price_bars_input=args.price_bars_input,
        csfloat_features_input=args.csfloat_features_input,
        coverage_output=args.coverage_output,
        report_output=args.report_output,
    )
    print(f"Items: {result.item_count}")
    print(f"Items with CSFloat snapshots: {result.items_with_snapshots}")
    print(f"Items with post-snapshot bars: {result.items_with_post_snapshot_bars}")
    print(f"Wrote coverage: {_display_path(result.coverage_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
