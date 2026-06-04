"""Variant-aware attribution across low-price execution policies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, reports_path

DEFAULT_POLICY_INPUTS = {
    "raw_v2_costed": reports_path("backtests", "day22_raw_v2_costed_accepted_trades.csv"),
    "min_price_1_costed": reports_path("backtests", "day22_min_price_1_costed_accepted_trades.csv"),
    "min_price_5_costed": reports_path("backtests", "day22_min_price_5_costed_accepted_trades.csv"),
}
DEFAULT_SUMMARY_OUTPUT = reports_path("tables", "day23_policy_attribution_summary.csv")
DEFAULT_ITEM_OUTPUT = reports_path("tables", "day23_policy_item_attribution.csv")
DEFAULT_PERIOD_OUTPUT = reports_path("tables", "day23_policy_period_attribution.csv")
DEFAULT_BUCKET_OUTPUT = reports_path("tables", "day23_policy_bucket_attribution.csv")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "day23_policy_attribution_report.md")

BUCKET_COLUMNS = [
    "execution_price_bucket",
    "price_bucket",
    "wear",
    "rarity",
    "category",
    "entry_month",
    "label_market_regime",
]


@dataclass(frozen=True)
class Day23Result:
    """Output metadata for variant-aware attribution."""

    summary_output: Path
    item_output: Path
    period_output: Path
    bucket_output: Path
    report_output: Path
    summary_rows: int
    item_rows: int
    period_rows: int
    bucket_rows: int


def run_day23_policy_attribution(
    policy_inputs: dict[str, Path] | None = None,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    item_output: Path = DEFAULT_ITEM_OUTPUT,
    period_output: Path = DEFAULT_PERIOD_OUTPUT,
    bucket_output: Path = DEFAULT_BUCKET_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> Day23Result:
    """Build variant-aware attribution tables and report."""
    trades = load_policy_trades(policy_inputs or DEFAULT_POLICY_INPUTS)
    summary = build_policy_summary(trades)
    item = build_group_attribution(trades, ["policy_variant", "model_name", "market_hash_name"])
    period = build_group_attribution(trades, ["policy_variant", "model_name", "entry_month"])
    buckets = build_bucket_attribution(trades)
    report = build_markdown_report(summary, item, period, buckets)

    for output in (summary_output, item_output, period_output, bucket_output, report_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    item.to_csv(item_output, index=False)
    period.to_csv(period_output, index=False)
    buckets.to_csv(bucket_output, index=False)
    report_output.write_text(report, encoding="utf-8")

    return Day23Result(
        summary_output=summary_output,
        item_output=item_output,
        period_output=period_output,
        bucket_output=bucket_output,
        report_output=report_output,
        summary_rows=len(summary),
        item_rows=len(item),
        period_rows=len(period),
        bucket_rows=len(buckets),
    )


def load_policy_trades(policy_inputs: dict[str, Path]) -> pd.DataFrame:
    """Load accepted-trade files and tag rows by policy variant."""
    frames: list[pd.DataFrame] = []
    for policy_variant, path in policy_inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Accepted trades not found for {policy_variant}: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "policy_variant", policy_variant)
        frames.append(frame)
    trades = pd.concat(frames, ignore_index=True, sort=False)
    trades["entry_timestamp"] = pd.to_datetime(trades["entry_timestamp"], utc=True)
    if "entry_month" not in trades.columns:
        trades["entry_month"] = (
            trades["entry_timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
        )
    for column in [
        "realized_net_return",
        "pnl",
        "notional",
        "entry_close",
        "original_pnl",
        "original_notional",
        "capacity_notional_multiplier",
        "capacity_liquidity_notional_cap",
    ]:
        if column in trades.columns:
            trades[column] = pd.to_numeric(trades[column], errors="coerce")
    return trades


def build_policy_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize concentration and robustness by policy/model."""
    rows: list[dict[str, Any]] = []
    for (policy_variant, model_name), group in trades.groupby(
        ["policy_variant", "model_name"], sort=False
    ):
        total_pnl = float(group["pnl"].sum())
        top_items = group.groupby("market_hash_name")["pnl"].sum().sort_values(ascending=False)
        top_periods = group.groupby("entry_month")["pnl"].sum().sort_values(ascending=False)
        rows.append(
            {
                "policy_variant": policy_variant,
                "model_name": model_name,
                "trade_count": int(len(group)),
                "item_count": int(group["market_hash_name"].nunique()),
                "positive_item_count": int((top_items > 0).sum()),
                "period_count": int(group["entry_month"].nunique()),
                "positive_period_count": int((top_periods > 0).sum()),
                "total_pnl": total_pnl,
                "total_original_pnl": _optional_sum(group, "original_pnl"),
                "total_notional": float(group["notional"].sum()),
                "total_original_notional": _optional_sum(group, "original_notional"),
                "average_capacity_multiplier": _optional_mean(
                    group,
                    "capacity_notional_multiplier",
                ),
                "average_trade_return": float(group["realized_net_return"].mean()),
                "median_trade_return": float(group["realized_net_return"].median()),
                "win_rate": float(group["realized_net_return"].gt(0).mean()),
                "profit_factor": _profit_factor(group["pnl"]),
                "top_3_item_pnl": float(top_items.head(3).sum()),
                "top_3_item_pnl_share": _share(top_items.head(3).sum(), total_pnl),
                "top_2_period_pnl": float(top_periods.head(2).sum()),
                "top_2_period_pnl_share": _share(top_periods.head(2).sum(), total_pnl),
                "median_entry_close": float(group["entry_close"].median())
                if "entry_close" in group.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "policy_variant"]).reset_index(drop=True)


def build_group_attribution(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Build reusable attribution rows for a policy/model/group combination."""
    rows: list[dict[str, Any]] = []
    total_lookup = trades.groupby(["policy_variant", "model_name"])["pnl"].sum().to_dict()
    for keys, group in trades.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        total_pnl = float(group["pnl"].sum())
        model_total = float(total_lookup.get((row["policy_variant"], row["model_name"]), np.nan))
        row.update(
            {
                "trade_count": int(len(group)),
                "total_pnl": total_pnl,
                "total_original_pnl": _optional_sum(group, "original_pnl"),
                "total_notional": float(group["notional"].sum()),
                "total_original_notional": _optional_sum(group, "original_notional"),
                "average_capacity_multiplier": _optional_mean(
                    group,
                    "capacity_notional_multiplier",
                ),
                "pnl_share_of_model": _share(total_pnl, model_total),
                "average_trade_return": float(group["realized_net_return"].mean()),
                "median_trade_return": float(group["realized_net_return"].median()),
                "win_rate": float(group["realized_net_return"].gt(0).mean()),
                "median_entry_close": float(group["entry_close"].median())
                if "entry_close" in group.columns
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["policy_variant", "model_name", "total_pnl"],
        ascending=[True, True, False],
    )


def build_bucket_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    """Build attribution for policy/model by interpretable buckets."""
    rows: list[pd.DataFrame] = []
    for column in BUCKET_COLUMNS:
        if column not in trades.columns:
            continue
        attribution = build_group_attribution(
            trades,
            ["policy_variant", "model_name", column],
        )
        attribution = attribution.rename(columns={column: "bucket_value"})
        attribution.insert(2, "bucket_type", column)
        rows.append(attribution)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_markdown_report(
    summary: pd.DataFrame,
    item: pd.DataFrame,
    period: pd.DataFrame,
    buckets: pd.DataFrame,
) -> str:
    """Build a concise Markdown report for policy attribution."""
    lightgbm = summary[summary["model_name"].eq("lightgbm_rank_blend")]
    lines = [
        "# Day 23 Policy Attribution",
        "",
        "## LightGBM Policy Summary",
        "",
        _markdown_table(lightgbm),
        "",
        "## Top LightGBM Items By Policy",
        "",
        _markdown_table(
            item[item["model_name"].eq("lightgbm_rank_blend")]
            .groupby("policy_variant", group_keys=False)
            .head(10)
        ),
        "",
        "## LightGBM Period Attribution",
        "",
        _markdown_table(period[period["model_name"].eq("lightgbm_rank_blend")]),
        "",
        "## LightGBM Price Bucket Attribution",
        "",
        _markdown_table(
            buckets[
                buckets["model_name"].eq("lightgbm_rank_blend")
                & buckets["bucket_type"].eq("execution_price_bucket")
            ]
        ),
        "",
        "## Interpretation",
        "",
        _interpret(lightgbm),
        "",
    ]
    return "\n".join(lines)


def _interpret(lightgbm: pd.DataFrame) -> str:
    if lightgbm.empty:
        return "No LightGBM rows were available."
    notes = []
    for _, row in lightgbm.sort_values("policy_variant").iterrows():
        notes.append(
            f"{row['policy_variant']}: PnL {row['total_pnl']:.4f}, "
            f"{int(row['trade_count'])} trades, "
            f"{int(row['positive_item_count'])} positive items, "
            f"top-3 item share {row['top_3_item_pnl_share']:.2%}, "
            f"top-2 period share {row['top_2_period_pnl_share']:.2%}."
        )
    return "\n\n".join(notes)


def _profit_factor(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    gross_profit = values[values > 0].sum()
    gross_loss = abs(values[values < 0].sum())
    if gross_loss > 0:
        return float(gross_profit / gross_loss)
    return float("inf") if gross_profit > 0 else 0.0


def _optional_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _optional_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _share(part: float, total: float) -> float:
    return float(part / total) if total != 0 else np.nan


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
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
        rows.append(
            "| " + " | ".join(_markdown_cell(row[column]) for column in display.columns) + " |"
        )
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 23 policy attribution.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--item-output", type=Path, default=DEFAULT_ITEM_OUTPUT)
    parser.add_argument("--period-output", type=Path, default=DEFAULT_PERIOD_OUTPUT)
    parser.add_argument("--bucket-output", type=Path, default=DEFAULT_BUCKET_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_day23_policy_attribution(
        summary_output=args.summary_output,
        item_output=args.item_output,
        period_output=args.period_output,
        bucket_output=args.bucket_output,
        report_output=args.report_output,
    )
    print(f"Summary rows: {result.summary_rows}")
    print(f"Item rows: {result.item_rows}")
    print(f"Period rows: {result.period_rows}")
    print(f"Bucket rows: {result.bucket_rows}")
    print(f"Wrote summary: {_display_path(result.summary_output)}")
    print(f"Wrote item attribution: {_display_path(result.item_output)}")
    print(f"Wrote period attribution: {_display_path(result.period_output)}")
    print(f"Wrote bucket attribution: {_display_path(result.bucket_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
