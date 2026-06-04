"""Day 18 CSFloat feature coverage and ablation setup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path
from cs_market_model.features.build_features import (
    DEFAULT_METADATA_INPUT,
    DEFAULT_PRICE_BARS_INPUT,
    build_feature_table,
    feature_columns,
    load_item_metadata,
    load_price_bars,
)
from cs_market_model.features.csfloat_listings import (
    CSFLOAT_FEATURE_COLUMNS,
    DEFAULT_CSFLOAT_FEATURES_OUTPUT,
)

DEFAULT_STEAMDT_ONLY_FEATURES_OUTPUT = data_path(
    "features",
    "features_day18_steamdt_only.parquet",
)
DEFAULT_CSFLOAT_ENABLED_FEATURES_OUTPUT = data_path(
    "features",
    "features_day18_csfloat_enabled.parquet",
)
DEFAULT_ABLATION_OUTPUT = reports_path("tables", "day18_csfloat_ablation_setup.csv")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "day18_csfloat_ablation_report.md")
MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION = 500


@dataclass(frozen=True)
class Day18Result:
    """Output paths and counts for Day 18 CSFloat ablation setup."""

    steamdt_features_output: Path
    csfloat_features_output: Path
    ablation_output: Path
    report_output: Path
    ablation_rows: int
    csfloat_covered_rows: int


def run_day18_csfloat_ablation_setup(
    price_bars_input: Path = DEFAULT_PRICE_BARS_INPUT,
    metadata_input: Path = DEFAULT_METADATA_INPUT,
    csfloat_features_input: Path = DEFAULT_CSFLOAT_FEATURES_OUTPUT,
    steamdt_features_output: Path = DEFAULT_STEAMDT_ONLY_FEATURES_OUTPUT,
    csfloat_enabled_features_output: Path = DEFAULT_CSFLOAT_ENABLED_FEATURES_OUTPUT,
    ablation_output: Path = DEFAULT_ABLATION_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> Day18Result:
    """Build SteamDT-only and CSFloat-enabled feature artifacts plus coverage report."""
    price_bars = load_price_bars(price_bars_input)
    metadata = load_item_metadata(metadata_input)
    csfloat_features = (
        pd.read_parquet(csfloat_features_input) if csfloat_features_input.exists() else None
    )

    steamdt_only = build_feature_table(price_bars, metadata, csfloat_features=None).drop(
        columns=CSFLOAT_FEATURE_COLUMNS,
        errors="ignore",
    )
    csfloat_enabled = build_feature_table(price_bars, metadata, csfloat_features=csfloat_features)

    for output in (
        steamdt_features_output,
        csfloat_enabled_features_output,
        ablation_output,
        report_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    steamdt_only.to_parquet(steamdt_features_output, index=False)
    csfloat_enabled.to_parquet(csfloat_enabled_features_output, index=False)

    ablation = build_ablation_setup_table(steamdt_only, csfloat_enabled, csfloat_features)
    ablation.to_csv(ablation_output, index=False)
    report_output.write_text(build_ablation_report(ablation), encoding="utf-8")

    covered = int(csfloat_enabled["csfloat_snapshot_available"].sum())
    return Day18Result(
        steamdt_features_output=steamdt_features_output,
        csfloat_features_output=csfloat_enabled_features_output,
        ablation_output=ablation_output,
        report_output=report_output,
        ablation_rows=len(ablation),
        csfloat_covered_rows=covered,
    )


def build_ablation_setup_table(
    steamdt_only: pd.DataFrame,
    csfloat_enabled: pd.DataFrame,
    csfloat_snapshots: pd.DataFrame | None,
) -> pd.DataFrame:
    """Summarize whether the local data can support a CSFloat ablation."""
    csfloat_columns = [column for column in CSFLOAT_FEATURE_COLUMNS if column in csfloat_enabled]
    covered_rows = (
        int(csfloat_enabled["csfloat_snapshot_available"].sum())
        if "csfloat_snapshot_available" in csfloat_enabled
        else 0
    )
    coverage_rate = covered_rows / len(csfloat_enabled) if len(csfloat_enabled) else np.nan
    snapshot_count = 0 if csfloat_snapshots is None else len(csfloat_snapshots)
    snapshot_items = (
        0
        if csfloat_snapshots is None or csfloat_snapshots.empty
        else int(csfloat_snapshots["market_hash_name"].nunique())
    )
    ablation_ready = covered_rows >= MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION
    csfloat_status = (
        "ready_for_model_ablation" if ablation_ready else "csfloat_coverage_insufficient"
    )
    if covered_rows == 0:
        csfloat_status = "no_csfloat_snapshot_coverage"
    if snapshot_count > 0 and covered_rows == 0:
        csfloat_status = "snapshots_collected_no_price_bar_coverage"

    rows = [
        {
            "variant": "steamdt_only",
            "row_count": len(steamdt_only),
            "item_count": int(steamdt_only["market_hash_name"].nunique()),
            "feature_count": len(feature_columns(steamdt_only)),
            "csfloat_feature_count": 0,
            "csfloat_snapshot_count": 0,
            "csfloat_snapshot_item_count": 0,
            "csfloat_covered_rows": 0,
            "csfloat_coverage_rate": 0.0,
            "ablation_ready": False,
            "status": "baseline_feature_artifact_ready",
        },
        {
            "variant": "steamdt_plus_csfloat",
            "row_count": len(csfloat_enabled),
            "item_count": int(csfloat_enabled["market_hash_name"].nunique()),
            "feature_count": len(feature_columns(csfloat_enabled)),
            "csfloat_feature_count": len(csfloat_columns),
            "csfloat_snapshot_count": snapshot_count,
            "csfloat_snapshot_item_count": snapshot_items,
            "csfloat_covered_rows": covered_rows,
            "csfloat_coverage_rate": float(coverage_rate),
            "min_covered_rows_for_ablation": MIN_CSFLOAT_COVERED_ROWS_FOR_ABLATION,
            "ablation_ready": bool(ablation_ready),
            "status": csfloat_status,
        },
    ]
    return pd.DataFrame(rows)


def build_ablation_report(ablation: pd.DataFrame) -> str:
    """Build a concise Day 18 Markdown report."""
    csfloat_row = ablation[ablation["variant"].eq("steamdt_plus_csfloat")].iloc[0]
    ready = bool(csfloat_row["ablation_ready"])
    snapshot_count = int(csfloat_row["csfloat_snapshot_count"])
    if ready:
        interpretation = (
            "CSFloat snapshots are present, so the next step is to train both "
            "feature variants and compare model/backtest metrics."
        )
        next_plan = (
            "Train the SteamDT-only and CSFloat-enabled variants, then compare "
            "model metrics, backtest PnL, and rejection diagnostics."
        )
    elif int(csfloat_row["csfloat_covered_rows"]) > 0:
        covered_rows = int(csfloat_row["csfloat_covered_rows"])
        minimum_rows = int(csfloat_row["min_covered_rows_for_ablation"])
        interpretation = (
            f"CSFloat coverage exists ({covered_rows} rows), but it is below the "
            f"conservative {minimum_rows}-row threshold for a true model ablation. "
            "Keep the production research model SteamDT-only and use CSFloat only "
            "for liquidity diagnostics until coverage improves."
        )
        next_plan = (
            "Continue collecting CSFloat snapshots and newer SteamDT bars, rebuild "
            "features, and rerun Day 18 once coverage reaches the minimum threshold."
        )
    elif snapshot_count > 0:
        interpretation = (
            "CSFloat snapshots exist, but none can be joined yet under the "
            "point-in-time as-of rule. The current daily price bars end before "
            "the snapshot timestamps, so a real model ablation should wait for "
            "newer SteamDT bars or additional future snapshots."
        )
        next_plan = (
            "Collect the next daily SteamDT bars, rebuild features, and rerun "
            "this ablation once CSFloat coverage is nonzero."
        )
    else:
        interpretation = (
            "No CSFloat snapshot coverage is available locally. The feature join "
            "is implemented, but a real model ablation should wait until raw "
            "CSFloat listing snapshots are collected."
        )
        next_plan = (
            "Collect CSFloat listing snapshots for the MVP universe, rebuild the "
            "CSFloat feature parquet, rebuild features, and rerun Day 18."
        )
    return "\n".join(
        [
            "# Day 18 CSFloat Ablation Setup",
            "",
            "## Summary",
            "",
            _markdown_table(ablation),
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "## Next Plan",
            "",
            next_plan,
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else ""
        )
    columns = [str(column) for column in display.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 18 CSFloat ablation setup.")
    parser.add_argument("--price-bars-input", type=Path, default=DEFAULT_PRICE_BARS_INPUT)
    parser.add_argument("--metadata-input", type=Path, default=DEFAULT_METADATA_INPUT)
    parser.add_argument(
        "--csfloat-features-input", type=Path, default=DEFAULT_CSFLOAT_FEATURES_OUTPUT
    )
    args = parser.parse_args()

    result = run_day18_csfloat_ablation_setup(
        price_bars_input=args.price_bars_input,
        metadata_input=args.metadata_input,
        csfloat_features_input=args.csfloat_features_input,
    )
    print(f"Ablation rows: {result.ablation_rows}")
    print(f"CSFloat covered rows: {result.csfloat_covered_rows}")
    print(f"Wrote SteamDT-only features: {_display_path(result.steamdt_features_output)}")
    print(f"Wrote CSFloat-enabled features: {_display_path(result.csfloat_features_output)}")
    print(f"Wrote ablation setup: {_display_path(result.ablation_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
