"""Day 22 low-price execution realism policy comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.backtesting.walk_forward_rejection import (
    summarize_walk_forward_selection,
    walk_forward_score_threshold_selection,
)
from cs_market_model.backtesting.rejection_policy import RejectionPolicy
from cs_market_model.config import PROJECT_ROOT, data_path, reports_path
from cs_market_model.research.day21_v2_robustness import enrich_trades_with_features

DEFAULT_LEDGER_INPUT = reports_path("backtests", "day12_13_trade_ledger.csv")
DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_LIQUIDITY_INPUT = data_path("processed", "day19_price_liquidity.csv")
DEFAULT_OUTPUT_DIR = reports_path("tables")
DEFAULT_BACKTEST_OUTPUT_DIR = reports_path("backtests")
DEFAULT_COMPARISON_OUTPUT = reports_path("tables", "day22_low_price_policy_comparison.csv")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "day22_low_price_policy_report.md")

PRICE_BUCKET_COST_BPS = [
    {"bucket": "<1", "min_price": 0.0, "max_price": 1.0, "extra_cost_bps": 500.0},
    {"bucket": "1-5", "min_price": 1.0, "max_price": 5.0, "extra_cost_bps": 250.0},
    {"bucket": "5-20", "min_price": 5.0, "max_price": 20.0, "extra_cost_bps": 150.0},
    {"bucket": ">20", "min_price": 20.0, "max_price": np.inf, "extra_cost_bps": 100.0},
]

POLICY_VARIANTS = [
    ("raw_v2_costed", 0.0),
    ("min_price_1_costed", 1.0),
    ("min_price_5_costed", 5.0),
]


@dataclass(frozen=True)
class Day22Result:
    """Output metadata for Day 22 low-price policy comparison."""

    comparison_output: Path
    report_output: Path
    comparison_rows: int


@dataclass(frozen=True)
class CapacitySizingConfig:
    """Capacity and liquidity assumptions for turning signals into sized trades."""

    portfolio_capital_usd: float = 1_000.0
    max_notional_per_trade_fraction: float = 0.03
    max_item_daily_notional_fraction: float = 0.03
    max_bucket_daily_notional_fraction: float = 0.12
    max_sell_count_participation: float = 0.05
    max_csfloat_listing_participation: float = 0.10
    fallback_liquidity_multiplier: float = 0.25
    minimum_executable_notional_fraction: float = 0.001
    max_open_positions: int | None = None
    max_open_positions_per_bucket: int | None = None

    def __post_init__(self) -> None:
        if self.portfolio_capital_usd <= 0:
            raise ValueError("portfolio_capital_usd must be positive")
        for field_name in (
            "max_notional_per_trade_fraction",
            "max_item_daily_notional_fraction",
            "max_bucket_daily_notional_fraction",
            "max_sell_count_participation",
            "max_csfloat_listing_participation",
            "fallback_liquidity_multiplier",
            "minimum_executable_notional_fraction",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
            if getattr(self, field_name) > 1:
                raise ValueError(f"{field_name} must be at most 1.0")
        for field_name in ("max_open_positions", "max_open_positions_per_bucket"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative")


def run_day22_low_price_policy(
    ledger_input: Path = DEFAULT_LEDGER_INPUT,
    features_input: Path = DEFAULT_FEATURES_INPUT,
    liquidity_input: Path = DEFAULT_LIQUIDITY_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    backtest_output_dir: Path = DEFAULT_BACKTEST_OUTPUT_DIR,
    comparison_output: Path = DEFAULT_COMPARISON_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    capacity_config: CapacitySizingConfig | None = None,
) -> Day22Result:
    """Run walk-forward policy variants with price-bucket execution costs."""
    ledger = pd.read_csv(ledger_input)
    features = pd.read_parquet(features_input)
    liquidity = load_liquidity_table(liquidity_input)
    enriched = enrich_trades_with_features(ledger, features)
    costed = apply_price_bucket_execution_costs(enriched)
    costed = add_liquidity_depth(costed, liquidity)
    resolved_capacity = capacity_config or CapacitySizingConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[pd.DataFrame] = []
    for variant_name, min_entry_price in POLICY_VARIANTS:
        selection, accepted = walk_forward_score_threshold_selection(
            costed,
            base_policy=RejectionPolicy(
                exclude_event_regime=False,
                min_entry_price=min_entry_price,
            ),
        )
        accepted_sized = apply_capacity_adjusted_sizing(accepted, resolved_capacity)
        accepted_filled = (
            accepted_sized[accepted_sized["capacity_filled"]].copy()
            if not accepted_sized.empty and "capacity_filled" in accepted_sized.columns
            else accepted_sized.copy()
        )
        summary = summarize_walk_forward_selection(selection, accepted_filled)
        summary.insert(0, "policy_variant", variant_name)
        summary.insert(1, "min_entry_price", min_entry_price)
        summary.insert(2, "uses_price_bucket_costs", True)
        summary.insert(3, "uses_capacity_sizing", True)
        comparison_rows.append(summary)
        selection.to_csv(
            output_dir / f"day22_{variant_name}_threshold_selection.csv",
            index=False,
        )
        summary.to_csv(
            output_dir / f"day22_{variant_name}_threshold_summary.csv",
            index=False,
        )
        accepted_filled.to_csv(
            backtest_output_dir / f"day22_{variant_name}_accepted_trades.csv",
            index=False,
        )

    comparison = pd.concat(comparison_rows, ignore_index=True, sort=False)
    comparison.to_csv(comparison_output, index=False)
    report_output.write_text(
        build_markdown_report(comparison, resolved_capacity),
        encoding="utf-8",
    )
    return Day22Result(
        comparison_output=comparison_output,
        report_output=report_output,
        comparison_rows=len(comparison),
    )


def apply_price_bucket_execution_costs(
    trades: pd.DataFrame,
    cost_buckets: list[dict[str, float | str]] | None = None,
) -> pd.DataFrame:
    """Apply extra spread/slippage cost through entry and exit price multipliers."""
    buckets = cost_buckets or PRICE_BUCKET_COST_BPS
    frame = trades.copy()
    frame["entry_close"] = pd.to_numeric(frame["entry_close"], errors="coerce")
    frame["realized_net_return_raw"] = pd.to_numeric(
        frame["realized_net_return"],
        errors="coerce",
    )
    frame["extra_execution_cost_bps"] = 0.0
    frame["execution_price_bucket"] = "unknown"
    for bucket in buckets:
        lower = float(bucket["min_price"])
        upper = float(bucket["max_price"])
        mask = frame["entry_close"].ge(lower)
        if np.isfinite(upper):
            mask &= frame["entry_close"].lt(upper)
        bucket_name = str(bucket["bucket"])
        frame.loc[mask, "execution_price_bucket"] = bucket_name
        frame.loc[mask, "extra_execution_cost_bps"] = float(bucket["extra_cost_bps"])
    frame["extra_execution_cost_return"] = frame["extra_execution_cost_bps"] / 10_000.0
    frame["extra_entry_cost_multiplier"] = 1.0 + frame["extra_execution_cost_return"]
    frame["extra_exit_cost_multiplier"] = 1.0 - frame["extra_execution_cost_return"]
    frame["execution_cost_method"] = "entry_exit_price_multiplier"
    frame["realized_net_return"] = cost_return_with_entry_exit_multipliers(
        frame["realized_net_return_raw"],
        frame["extra_execution_cost_return"],
    )
    frame["pnl"] = pd.to_numeric(frame["notional"], errors="coerce") * frame["realized_net_return"]
    return frame


def cost_return_with_entry_exit_multipliers(
    raw_return: pd.Series | float,
    cost_return: pd.Series | float,
) -> pd.Series | float:
    """Apply extra cost as worse entry and exit prices, not flat return subtraction."""
    gross_return = 1.0 + raw_return
    entry_multiplier = 1.0 + cost_return
    exit_multiplier = 1.0 - cost_return
    return (gross_return * exit_multiplier / entry_multiplier) - 1.0


def load_liquidity_table(path: Path = DEFAULT_LIQUIDITY_INPUT) -> pd.DataFrame:
    """Load current SteamDT depth proxies when available."""
    if not path.exists():
        return pd.DataFrame(columns=["market_hash_name"])
    return pd.read_csv(path)


def add_liquidity_depth(trades: pd.DataFrame, liquidity: pd.DataFrame) -> pd.DataFrame:
    """Attach current SteamDT sell-count depth proxies to trade candidates."""
    frame = trades.copy()
    if liquidity.empty:
        for column in ("steam_sell_count", "max_sell_count", "steam_sell_price"):
            frame[column] = np.nan
        return frame
    liquidity_columns = [
        column
        for column in [
            "market_hash_name",
            "steam_sell_count",
            "max_sell_count",
            "steam_sell_price",
        ]
        if column in liquidity.columns
    ]
    joined = frame.merge(
        liquidity[liquidity_columns],
        on="market_hash_name",
        how="left",
        validate="many_to_one",
    )
    for column in ("steam_sell_count", "max_sell_count", "steam_sell_price"):
        if column not in joined.columns:
            joined[column] = np.nan
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    return joined


def apply_capacity_adjusted_sizing(
    trades: pd.DataFrame,
    config: CapacitySizingConfig | None = None,
) -> pd.DataFrame:
    """Resize accepted trades by per-trade, per-day, and depth-based capacity caps.

    ``notional`` and ``pnl`` must be capital-normalized units, not raw dollar
    amounts. The upstream portfolio backtest compounds equity, so notional may
    exceed 1.0, but obviously dollar-scale inputs are refused to avoid mixing
    units.
    """
    resolved = config or CapacitySizingConfig()
    if trades.empty:
        return trades.copy()

    frame = trades.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    if "exit_timestamp" in frame.columns:
        frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True)
    else:
        frame["exit_timestamp"] = frame["entry_timestamp"]
    frame["entry_date"] = frame["entry_timestamp"].dt.strftime("%Y-%m-%d")
    if "execution_price_bucket" not in frame.columns:
        frame["execution_price_bucket"] = "unknown"
    frame["original_notional"] = pd.to_numeric(frame["notional"], errors="coerce").fillna(0.0)
    frame["original_pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    _validate_capital_normalized_units(frame, resolved)
    frame["realized_net_return"] = pd.to_numeric(frame["realized_net_return"], errors="coerce")
    frame["entry_close"] = pd.to_numeric(frame["entry_close"], errors="coerce")

    sized_rows: list[dict[str, Any]] = []
    sort_columns = [
        column
        for column in [
            "model_name",
            "entry_timestamp",
            "strong_buy_score",
            "market_hash_name",
            "event_id",
        ]
        if column in frame.columns
    ]
    ascending = [
        False if column == "strong_buy_score" else True
        for column in sort_columns
    ]
    ordered = (
        frame.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        if sort_columns
        else frame
    )
    usage: dict[tuple[str, str, str, str], float] = {}
    open_positions: dict[str, list[dict[str, Any]]] = {}

    for _, row in ordered.iterrows():
        model_name = str(row.get("model_name", "model"))
        entry_date = str(row.get("entry_date", ""))
        item = str(row.get("market_hash_name", ""))
        bucket = str(row.get("execution_price_bucket", "unknown"))
        entry_timestamp = pd.Timestamp(row.get("entry_timestamp"))
        model_open = _active_open_positions(
            open_positions.get(model_name, []),
            entry_timestamp,
        )
        open_positions[model_name] = model_open
        open_position_count = len(model_open)
        bucket_open_position_count = sum(
            1 for position in model_open if position["bucket"] == bucket
        )
        open_position_capacity_available = _open_position_capacity_available(
            open_position_count,
            bucket_open_position_count,
            resolved,
        )
        original_notional = _finite_float(row.get("original_notional"), 0.0)
        item_key = (model_name, entry_date, "item", item)
        bucket_key = (model_name, entry_date, "bucket", bucket)
        item_remaining = max(
            resolved.max_item_daily_notional_fraction - usage.get(item_key, 0.0),
            0.0,
        )
        bucket_remaining = max(
            resolved.max_bucket_daily_notional_fraction - usage.get(bucket_key, 0.0),
            0.0,
        )
        liquidity_cap = _liquidity_notional_cap(row, resolved, original_notional)
        capacity_notional = min(
            original_notional,
            resolved.max_notional_per_trade_fraction,
            item_remaining,
            bucket_remaining,
            liquidity_cap,
        )
        capacity_notional = max(capacity_notional, 0.0)
        if not open_position_capacity_available:
            capacity_notional = 0.0
        capacity_filled = capacity_notional >= resolved.minimum_executable_notional_fraction

        output = row.to_dict()
        output["capacity_liquidity_notional_cap"] = float(liquidity_cap)
        output["capacity_item_remaining_before"] = float(item_remaining)
        output["capacity_bucket_remaining_before"] = float(bucket_remaining)
        output["capacity_open_positions_before"] = int(open_position_count)
        output["capacity_bucket_open_positions_before"] = int(bucket_open_position_count)
        output["capacity_max_open_positions"] = resolved.max_open_positions
        output["capacity_max_open_positions_per_bucket"] = resolved.max_open_positions_per_bucket
        output["capacity_adjusted_notional"] = float(capacity_notional)
        output["capacity_notional_multiplier"] = (
            float(capacity_notional / original_notional) if original_notional > 0 else 0.0
        )
        output["capacity_filled"] = bool(capacity_filled)
        output["capacity_rejection_reason"] = _capacity_rejection_reason(
            capacity_filled,
            open_position_capacity_available,
        )
        output["notional"] = float(capacity_notional)
        output["pnl"] = float(capacity_notional * _finite_float(row.get("realized_net_return"), 0.0))
        sized_rows.append(output)

        if capacity_filled:
            usage[item_key] = usage.get(item_key, 0.0) + capacity_notional
            usage[bucket_key] = usage.get(bucket_key, 0.0) + capacity_notional
            model_open.append(
                {
                    "exit_timestamp": pd.Timestamp(row.get("exit_timestamp")),
                    "bucket": bucket,
                }
            )
            open_positions[model_name] = model_open

    return pd.DataFrame(sized_rows)


def build_markdown_report(
    comparison: pd.DataFrame,
    capacity_config: CapacitySizingConfig | None = None,
) -> str:
    """Build a concise Markdown report for low-price policy variants."""
    lightgbm = comparison[comparison["model_name"].eq("lightgbm_rank_blend")].copy()
    resolved_capacity = capacity_config or CapacitySizingConfig()
    return "\n".join(
        [
            "# Day 22 Low-Price Policy Comparison",
            "",
            "## LightGBM Summary",
            "",
            _markdown_table(lightgbm),
            "",
            "## Execution Cost Assumptions",
            "",
            _markdown_table(pd.DataFrame(PRICE_BUCKET_COST_BPS)),
            "",
            "## Capacity Assumptions",
            "",
            _markdown_table(pd.DataFrame([resolved_capacity.__dict__])),
            "",
            "## Interpretation",
            "",
            _interpret(lightgbm),
            "",
        ]
    )


def _interpret(lightgbm: pd.DataFrame) -> str:
    if lightgbm.empty:
        return "No LightGBM comparison rows were generated."
    rows = []
    for _, row in lightgbm.sort_values("min_entry_price").iterrows():
        rows.append(
            f"{row['policy_variant']}: PnL {float(row['accepted_total_pnl']):.4f}, "
            f"trades {int(row['accepted_count'])}, "
            f"win rate {float(row['accepted_win_rate']):.2%}, "
            f"profit factor {float(row['profit_factor']):.4f}."
        )
    return "\n\n".join(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [str(column) for column in display.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(rows)


def _liquidity_notional_cap(row: pd.Series, config: CapacitySizingConfig, original_notional: float) -> float:
    entry_close = _finite_float(row.get("entry_close"), np.nan)
    if not np.isfinite(entry_close) or entry_close <= 0:
        return original_notional * config.fallback_liquidity_multiplier

    caps: list[float] = []
    sell_count = _first_finite(row, ["steam_sell_count", "max_sell_count"])
    if sell_count is not None and sell_count > 0:
        caps.append(entry_close * sell_count * config.max_sell_count_participation / config.portfolio_capital_usd)

    csfloat_listing_count = _finite_float(row.get("csfloat_listing_count"), np.nan)
    if np.isfinite(csfloat_listing_count) and csfloat_listing_count > 0:
        caps.append(
            entry_close
            * csfloat_listing_count
            * config.max_csfloat_listing_participation
            / config.portfolio_capital_usd
        )

    if not caps:
        return original_notional * config.fallback_liquidity_multiplier
    return max(min(caps), 0.0)


def _first_finite(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        value = _finite_float(row.get(column), np.nan)
        if np.isfinite(value):
            return value
    return None


def _active_open_positions(
    positions: list[dict[str, Any]],
    entry_timestamp: pd.Timestamp,
) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for position in positions:
        exit_timestamp = pd.Timestamp(position["exit_timestamp"])
        if pd.isna(exit_timestamp) or exit_timestamp > entry_timestamp:
            active.append(position)
    return active


def _open_position_capacity_available(
    open_position_count: int,
    bucket_open_position_count: int,
    config: CapacitySizingConfig,
) -> bool:
    if config.max_open_positions is not None and open_position_count >= config.max_open_positions:
        return False
    if (
        config.max_open_positions_per_bucket is not None
        and bucket_open_position_count >= config.max_open_positions_per_bucket
    ):
        return False
    return True


def _capacity_rejection_reason(
    capacity_filled: bool,
    open_position_capacity_available: bool,
) -> str:
    if capacity_filled:
        return ""
    if not open_position_capacity_available:
        return "open_position_limit"
    return "insufficient_capacity"


def _validate_capital_normalized_units(
    frame: pd.DataFrame,
    config: CapacitySizingConfig,
) -> None:
    if "original_notional" not in frame.columns or frame["original_notional"].empty:
        return
    max_abs = pd.to_numeric(frame["original_notional"], errors="coerce").abs().max()
    dollar_like_limit = max(20.0, config.portfolio_capital_usd * config.max_notional_per_trade_fraction)
    if pd.notna(max_abs) and max_abs > dollar_like_limit:
        raise ValueError(
            f"original_notional appears to be in dollar terms (max_abs={max_abs:.2f}). "
            "Capacity sizing expects capital-normalized units, not raw dollars."
        )


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 22 low-price policy comparison.")
    parser.add_argument("--ledger-input", type=Path, default=DEFAULT_LEDGER_INPUT)
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--liquidity-input", type=Path, default=DEFAULT_LIQUIDITY_INPUT)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_day22_low_price_policy(
        ledger_input=args.ledger_input,
        features_input=args.features_input,
        liquidity_input=args.liquidity_input,
        comparison_output=args.comparison_output,
        report_output=args.report_output,
    )
    print(f"Comparison rows: {result.comparison_rows}")
    print(f"Wrote comparison: {_display_path(result.comparison_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
