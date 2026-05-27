"""Day 24 capacity sensitivity for the conservative $5+ policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.backtesting.rejection_policy import RejectionPolicy
from cs_market_model.backtesting.walk_forward_rejection import walk_forward_score_threshold_selection
from cs_market_model.config import PROJECT_ROOT, data_path, reports_path
from cs_market_model.research.day21_v2_robustness import enrich_trades_with_features
from cs_market_model.research.day22_low_price_policy import (
    DEFAULT_LEDGER_INPUT,
    DEFAULT_LIQUIDITY_INPUT,
    CapacitySizingConfig,
    add_liquidity_depth,
    apply_capacity_adjusted_sizing,
    apply_price_bucket_execution_costs,
    load_liquidity_table,
)

DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_SUMMARY_OUTPUT = reports_path(
    "tables",
    "day24_min_price_5_capacity_sensitivity.csv",
)
DEFAULT_PERIOD_OUTPUT = reports_path(
    "tables",
    "day24_min_price_5_capacity_period_attribution.csv",
)
DEFAULT_REPORT_OUTPUT = reports_path(
    "backtests",
    "day24_min_price_5_capacity_sensitivity_report.md",
)
DEFAULT_BACKTEST_OUTPUT_DIR = reports_path("backtests")


@dataclass(frozen=True)
class CapacitySensitivityScenario:
    """Named capacity assumptions for sensitivity testing."""

    name: str
    description: str
    config: CapacitySizingConfig


@dataclass(frozen=True)
class Day24Result:
    """Output metadata for Day 24 capacity sensitivity."""

    summary_output: Path
    period_output: Path
    report_output: Path
    summary_rows: int
    period_rows: int


DEFAULT_SCENARIOS = [
    CapacitySensitivityScenario(
        name="current_day22",
        description="Current Day 22 capacity assumptions.",
        config=CapacitySizingConfig(),
    ),
    CapacitySensitivityScenario(
        name="strict_daily_half",
        description="Roughly halves daily capacity and adds moderate open-position limits.",
        config=CapacitySizingConfig(
            max_notional_per_trade_fraction=0.02,
            max_item_daily_notional_fraction=0.015,
            max_bucket_daily_notional_fraction=0.06,
            max_sell_count_participation=0.025,
            max_csfloat_listing_participation=0.05,
            fallback_liquidity_multiplier=0.15,
            max_open_positions=20,
            max_open_positions_per_bucket=8,
        ),
    ),
    CapacitySensitivityScenario(
        name="strict_open_10",
        description="Tight trade, daily, liquidity, and open-position limits.",
        config=CapacitySizingConfig(
            max_notional_per_trade_fraction=0.015,
            max_item_daily_notional_fraction=0.01,
            max_bucket_daily_notional_fraction=0.04,
            max_sell_count_participation=0.02,
            max_csfloat_listing_participation=0.04,
            fallback_liquidity_multiplier=0.10,
            max_open_positions=10,
            max_open_positions_per_bucket=4,
        ),
    ),
    CapacitySensitivityScenario(
        name="severe_open_5",
        description="Severe small-size paper-trade capacity assumptions.",
        config=CapacitySizingConfig(
            max_notional_per_trade_fraction=0.01,
            max_item_daily_notional_fraction=0.0075,
            max_bucket_daily_notional_fraction=0.025,
            max_sell_count_participation=0.01,
            max_csfloat_listing_participation=0.02,
            fallback_liquidity_multiplier=0.05,
            max_open_positions=5,
            max_open_positions_per_bucket=2,
        ),
    ),
]


def run_day24_capacity_sensitivity(
    ledger_input: Path = DEFAULT_LEDGER_INPUT,
    features_input: Path = DEFAULT_FEATURES_INPUT,
    liquidity_input: Path = DEFAULT_LIQUIDITY_INPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    period_output: Path = DEFAULT_PERIOD_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    backtest_output_dir: Path = DEFAULT_BACKTEST_OUTPUT_DIR,
    scenarios: list[CapacitySensitivityScenario] | None = None,
) -> Day24Result:
    """Run $5+ capacity scenarios on the same corrected walk-forward candidates."""
    ledger = pd.read_csv(ledger_input)
    features = pd.read_parquet(features_input)
    liquidity = load_liquidity_table(liquidity_input)
    enriched = enrich_trades_with_features(ledger, features)
    costed = apply_price_bucket_execution_costs(enriched)
    costed = add_liquidity_depth(costed, liquidity)
    selection, accepted = walk_forward_score_threshold_selection(
        costed,
        base_policy=RejectionPolicy(exclude_event_regime=False, min_entry_price=5.0),
    )

    scenario_rows: list[dict[str, Any]] = []
    period_frames: list[pd.DataFrame] = []
    resolved_scenarios = scenarios or DEFAULT_SCENARIOS
    backtest_output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in resolved_scenarios:
        sized = apply_capacity_adjusted_sizing(accepted, scenario.config)
        filled = (
            sized[sized["capacity_filled"]].copy()
            if not sized.empty and "capacity_filled" in sized.columns
            else sized.copy()
        )
        filled.to_csv(
            backtest_output_dir
            / f"day24_min_price_5_{scenario.name}_accepted_trades.csv",
            index=False,
        )
        scenario_rows.extend(
            build_capacity_summary_rows(selection, sized, filled, scenario),
        )
        period = build_period_attribution(filled, scenario.name)
        if not period.empty:
            period_frames.append(period)

    summary = pd.DataFrame(scenario_rows).sort_values(["model_name", "scenario"])
    period_summary = (
        pd.concat(period_frames, ignore_index=True, sort=False)
        if period_frames
        else pd.DataFrame()
    )
    report = build_markdown_report(summary, period_summary, resolved_scenarios)

    for output in (summary_output, period_output, report_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    period_summary.to_csv(period_output, index=False)
    report_output.write_text(report, encoding="utf-8")
    return Day24Result(
        summary_output=summary_output,
        period_output=period_output,
        report_output=report_output,
        summary_rows=len(summary),
        period_rows=len(period_summary),
    )


def build_capacity_summary_rows(
    selection: pd.DataFrame,
    sized: pd.DataFrame,
    filled: pd.DataFrame,
    scenario: CapacitySensitivityScenario,
) -> list[dict[str, Any]]:
    """Build model-level capacity sensitivity metrics."""
    rows: list[dict[str, Any]] = []
    model_names = sorted(selection["model_name"].astype(str).unique())
    for model_name in model_names:
        model_sized = sized[sized["model_name"].astype(str).eq(model_name)].copy()
        model_filled = filled[filled["model_name"].astype(str).eq(model_name)].copy()
        model_selection = selection[selection["model_name"].astype(str).eq(model_name)]
        rejected_by_capacity = int(len(model_sized) - len(model_filled))
        base_rejected = int(
            pd.to_numeric(model_selection["test_rejected_count"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        total_candidates = int(len(model_sized) + base_rejected)
        period_pnl = _period_pnl(model_filled)
        total_pnl = _pnl_sum(model_filled)
        rows.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "model_name": model_name,
                "accepted_count": int(len(model_filled)),
                "capacity_rejected_count": rejected_by_capacity,
                "total_rejected_count": base_rejected + rejected_by_capacity,
                "total_candidate_count": total_candidates,
                "rejection_rate": float((base_rejected + rejected_by_capacity) / total_candidates)
                if total_candidates
                else np.nan,
                "accepted_total_pnl": total_pnl,
                "profit_factor": _profit_factor(model_filled),
                "accepted_max_drawdown": _max_drawdown(model_filled),
                "accepted_win_rate": _win_rate(model_filled),
                "accepted_avg_return": _mean_return(model_filled),
                "total_notional": _optional_sum(model_filled, "notional"),
                "average_capacity_multiplier": _optional_mean(
                    model_filled,
                    "capacity_notional_multiplier",
                ),
                "period_count": int(period_pnl.shape[0]),
                "positive_period_share": float(period_pnl.gt(0).mean())
                if not period_pnl.empty
                else np.nan,
                "top_1_period_pnl_share": _top_period_share(period_pnl, total_pnl, 1),
                "top_2_period_pnl_share": _top_period_share(period_pnl, total_pnl, 2),
                "survives_capacity_stress": bool(
                    total_pnl > 0
                    and _profit_factor(model_filled) > 1.0
                    and len(model_filled) >= 50
                    and _top_period_share(period_pnl, total_pnl, 2) <= 1.0
                ),
                **_config_fields(scenario.config),
            }
        )
    return rows


def build_period_attribution(filled: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """Build scenario/model/month PnL concentration rows."""
    if filled.empty:
        return pd.DataFrame()
    frame = filled.copy()
    if "test_period" not in frame.columns:
        frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
        frame["test_period"] = frame["entry_timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
    rows = []
    for (model_name, period), group in frame.groupby(["model_name", "test_period"], sort=False):
        rows.append(
            {
                "scenario": scenario_name,
                "model_name": model_name,
                "test_period": period,
                "accepted_count": int(len(group)),
                "accepted_total_pnl": _pnl_sum(group),
                "profit_factor": _profit_factor(group),
                "accepted_max_drawdown": _max_drawdown(group),
                "accepted_win_rate": _win_rate(group),
                "total_notional": _optional_sum(group, "notional"),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "model_name", "test_period"])


def build_markdown_report(
    summary: pd.DataFrame,
    period: pd.DataFrame,
    scenarios: list[CapacitySensitivityScenario],
) -> str:
    """Build a concise Markdown report for $5+ capacity sensitivity."""
    lightgbm = summary[summary["model_name"].eq("lightgbm_rank_blend")].copy()
    lines = [
        "# Day 24 $5+ Capacity Sensitivity",
        "",
        "## LightGBM Summary",
        "",
        _markdown_table(lightgbm),
        "",
        "## Scenario Assumptions",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"scenario": scenario.name, "description": scenario.description, **_config_fields(scenario.config)}
                    for scenario in scenarios
                ]
            )
        ),
        "",
        "## Weakest LightGBM Periods",
        "",
        _markdown_table(
            period[period["model_name"].eq("lightgbm_rank_blend")]
            .sort_values(["scenario", "accepted_total_pnl"])
            .groupby("scenario", group_keys=False)
            .head(5)
        ),
        "",
        "## Decision",
        "",
        _decision(lightgbm),
        "",
    ]
    return "\n".join(lines)


def _decision(lightgbm: pd.DataFrame) -> str:
    if lightgbm.empty:
        return "No LightGBM capacity rows were generated."
    survivors = lightgbm[lightgbm["survives_capacity_stress"]]
    if survivors.empty:
        return (
            "The $5+ policy does not survive the configured capacity stress suite. "
            "Use paper-trade-only status and strengthen rejection gates before treating it as a default."
        )
    weakest = lightgbm.sort_values("accepted_total_pnl").iloc[0]
    return (
        "The $5+ policy survives the configured capacity stress suite. "
        f"The weakest LightGBM scenario is {weakest['scenario']} with "
        f"{weakest['accepted_total_pnl']:.4f} PnL and profit factor {weakest['profit_factor']:.4f}."
    )


def _period_pnl(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    period_column = "test_period" if "test_period" in frame.columns else "entry_month"
    if period_column not in frame.columns:
        temp = frame.copy()
        temp["entry_timestamp"] = pd.to_datetime(temp["entry_timestamp"], utc=True)
        temp["_period"] = temp["entry_timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
        period_column = "_period"
        frame = temp
    return pd.to_numeric(frame["pnl"], errors="coerce").groupby(frame[period_column].astype(str)).sum()


def _top_period_share(period_pnl: pd.Series, total_pnl: float, top_n: int) -> float:
    if period_pnl.empty or total_pnl <= 0:
        return np.nan
    positive = period_pnl[period_pnl > 0].sort_values(ascending=False)
    return float(positive.head(top_n).sum() / total_pnl) if not positive.empty else np.nan


def _pnl_sum(frame: pd.DataFrame) -> float:
    if frame.empty or "pnl" not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0).sum())


def _profit_factor(frame: pd.DataFrame) -> float:
    if frame.empty or "pnl" not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    gross_profit = values[values > 0].sum()
    gross_loss = abs(values[values < 0].sum())
    if gross_loss > 0:
        return float(gross_profit / gross_loss)
    return float("inf") if gross_profit > 0 else 0.0


def _max_drawdown(frame: pd.DataFrame) -> float:
    if frame.empty or "pnl" not in frame.columns:
        return np.nan
    ordered = frame.copy()
    if "entry_timestamp" in ordered.columns:
        ordered["entry_timestamp"] = pd.to_datetime(ordered["entry_timestamp"], utc=True)
        ordered = ordered.sort_values(["entry_timestamp", "market_hash_name"])
    equity = 1.0 + pd.to_numeric(ordered["pnl"], errors="coerce").fillna(0.0).cumsum()
    running_peak = equity.cummax()
    return float(equity.div(running_peak).sub(1.0).min())


def _win_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "realized_net_return" not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame["realized_net_return"], errors="coerce").gt(0).mean())


def _mean_return(frame: pd.DataFrame) -> float:
    if frame.empty or "realized_net_return" not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame["realized_net_return"], errors="coerce").mean())


def _optional_sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _optional_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _config_fields(config: CapacitySizingConfig) -> dict[str, Any]:
    return {
        "max_notional_per_trade_fraction": config.max_notional_per_trade_fraction,
        "max_item_daily_notional_fraction": config.max_item_daily_notional_fraction,
        "max_bucket_daily_notional_fraction": config.max_bucket_daily_notional_fraction,
        "max_sell_count_participation": config.max_sell_count_participation,
        "max_csfloat_listing_participation": config.max_csfloat_listing_participation,
        "fallback_liquidity_multiplier": config.fallback_liquidity_multiplier,
        "minimum_executable_notional_fraction": config.minimum_executable_notional_fraction,
        "max_open_positions": config.max_open_positions,
        "max_open_positions_per_bucket": config.max_open_positions_per_bucket,
    }


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
        rows.append("| " + " | ".join(_markdown_cell(row[column]) for column in display.columns) + " |")
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 24 $5+ capacity sensitivity.")
    parser.add_argument("--ledger-input", type=Path, default=DEFAULT_LEDGER_INPUT)
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--liquidity-input", type=Path, default=DEFAULT_LIQUIDITY_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--period-output", type=Path, default=DEFAULT_PERIOD_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_day24_capacity_sensitivity(
        ledger_input=args.ledger_input,
        features_input=args.features_input,
        liquidity_input=args.liquidity_input,
        summary_output=args.summary_output,
        period_output=args.period_output,
        report_output=args.report_output,
    )
    print(f"Summary rows: {result.summary_rows}")
    print(f"Period rows: {result.period_rows}")
    print(f"Wrote summary: {_display_path(result.summary_output)}")
    print(f"Wrote periods: {_display_path(result.period_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
