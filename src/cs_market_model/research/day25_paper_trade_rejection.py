"""Day 25 stricter rejection gates for the $5+ paper-trade candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.backtesting.portfolio import rejection_policy_from_yaml
from cs_market_model.backtesting.rejection_policy import RejectionPolicy
from cs_market_model.backtesting.walk_forward_rejection import (
    walk_forward_score_threshold_selection,
)
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
from cs_market_model.research.day24_capacity_sensitivity import (
    DEFAULT_SCENARIOS as DAY24_CAPACITY_SCENARIOS,
)
from cs_market_model.research.day24_capacity_sensitivity import (
    MAX_ACCEPTED_DRAWDOWN,
    MAX_TOP_2_PERIOD_PNL_SHARE,
    MIN_ACCEPTED_TRADES,
    MIN_POSITIVE_PERIOD_SHARE,
    _config_fields,
    _markdown_table,
    _max_drawdown,
    _mean_return,
    _optional_mean,
    _optional_sum,
    _period_pnl,
    _pnl_sum,
    _profit_factor,
    _top_period_share,
    _win_rate,
    build_period_attribution,
    survives_capacity_stress,
)

DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_SUMMARY_OUTPUT = reports_path(
    "tables",
    "day25_paper_trade_rejection_summary.csv",
)
DEFAULT_PERIOD_OUTPUT = reports_path(
    "tables",
    "day25_paper_trade_rejection_period_attribution.csv",
)
DEFAULT_REPORT_OUTPUT = reports_path(
    "backtests",
    "day25_paper_trade_rejection_report.md",
)
DEFAULT_TABLE_OUTPUT_DIR = reports_path("tables")
DEFAULT_BACKTEST_OUTPUT_DIR = reports_path("backtests")
PAPER_TRADE_POLICY_NAME = "paper_trade_5_plus"


@dataclass(frozen=True)
class PaperTradeRejectionScenario:
    """Policy and capacity assumptions for one Day 25 comparison row set."""

    name: str
    policy_name: str
    description: str
    policy: RejectionPolicy
    capacity_config: CapacitySizingConfig


@dataclass(frozen=True)
class Day25Result:
    """Output metadata for Day 25 stricter paper-trade rejection analysis."""

    summary_output: Path
    period_output: Path
    report_output: Path
    summary_rows: int
    period_rows: int
    selection_rows: int


def paper_trade_5_plus_policy() -> RejectionPolicy:
    """Load the configured stricter $5+ paper-trade rejection policy."""
    return rejection_policy_from_yaml(PAPER_TRADE_POLICY_NAME)


def default_scenarios() -> list[PaperTradeRejectionScenario]:
    """Return baseline and stricter paper-trade scenarios."""
    strict_open_10 = _day24_capacity_config("strict_open_10")
    return [
        PaperTradeRejectionScenario(
            name="current_day24_gates",
            policy_name="min_price_5_only",
            description="Current Day 24 $5+ base gates with Day 22 capacity assumptions.",
            policy=RejectionPolicy(exclude_event_regime=False, min_entry_price=5.0),
            capacity_config=CapacitySizingConfig(),
        ),
        PaperTradeRejectionScenario(
            name="paper_trade_5_plus_gates",
            policy_name=PAPER_TRADE_POLICY_NAME,
            description="Stricter $5+ paper-trade gates with Day 22 capacity assumptions.",
            policy=paper_trade_5_plus_policy(),
            capacity_config=CapacitySizingConfig(),
        ),
        PaperTradeRejectionScenario(
            name="paper_trade_5_plus_strict_open_10",
            policy_name=PAPER_TRADE_POLICY_NAME,
            description="Stricter $5+ paper-trade gates with strict Day 24 open limits.",
            policy=paper_trade_5_plus_policy(),
            capacity_config=strict_open_10,
        ),
    ]


def run_day25_paper_trade_rejection(
    ledger_input: Path = DEFAULT_LEDGER_INPUT,
    features_input: Path = DEFAULT_FEATURES_INPUT,
    liquidity_input: Path = DEFAULT_LIQUIDITY_INPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    period_output: Path = DEFAULT_PERIOD_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    table_output_dir: Path = DEFAULT_TABLE_OUTPUT_DIR,
    backtest_output_dir: Path = DEFAULT_BACKTEST_OUTPUT_DIR,
    scenarios: list[PaperTradeRejectionScenario] | None = None,
) -> Day25Result:
    """Compare the current $5+ gates with stricter paper-trade gates."""
    ledger = pd.read_csv(ledger_input)
    features = pd.read_parquet(features_input)
    liquidity = load_liquidity_table(liquidity_input)
    enriched = enrich_trades_with_features(ledger, features)
    costed = apply_price_bucket_execution_costs(enriched)
    costed = add_liquidity_depth(costed, liquidity)

    table_output_dir.mkdir(parents=True, exist_ok=True)
    backtest_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_scenarios = scenarios or default_scenarios()
    summary_rows: list[dict[str, Any]] = []
    period_frames: list[pd.DataFrame] = []
    selection_row_count = 0

    for scenario in resolved_scenarios:
        selection, accepted = walk_forward_score_threshold_selection(
            costed,
            base_policy=scenario.policy,
        )
        selection_row_count += len(selection)
        sized = apply_capacity_adjusted_sizing(accepted, scenario.capacity_config)
        filled = (
            sized[sized["capacity_filled"]].copy()
            if not sized.empty and "capacity_filled" in sized.columns
            else sized.copy()
        )

        selection.to_csv(
            table_output_dir / f"day25_{scenario.name}_threshold_selection.csv",
            index=False,
        )
        filled.to_csv(
            backtest_output_dir / f"day25_{scenario.name}_accepted_trades.csv",
            index=False,
        )

        summary_rows.extend(
            build_paper_trade_summary_rows(selection, sized, filled, scenario),
        )
        period = build_period_attribution(filled, scenario.name)
        if not period.empty:
            period_frames.append(period)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["model_name", "scenario"])
    period_summary = (
        pd.concat(period_frames, ignore_index=True, sort=False) if period_frames else pd.DataFrame()
    )
    report = build_markdown_report(summary, period_summary, resolved_scenarios)

    for output in (summary_output, period_output, report_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    period_summary.to_csv(period_output, index=False)
    report_output.write_text(report, encoding="utf-8")
    return Day25Result(
        summary_output=summary_output,
        period_output=period_output,
        report_output=report_output,
        summary_rows=len(summary),
        period_rows=len(period_summary),
        selection_rows=selection_row_count,
    )


def build_paper_trade_summary_rows(
    selection: pd.DataFrame,
    sized: pd.DataFrame,
    filled: pd.DataFrame,
    scenario: PaperTradeRejectionScenario,
) -> list[dict[str, Any]]:
    """Build model-level summary rows with base-gate and capacity rejection counts."""
    if selection.empty:
        return []

    rows: list[dict[str, Any]] = []
    for model_name in sorted(selection["model_name"].astype(str).unique()):
        model_selection = selection[selection["model_name"].astype(str).eq(model_name)]
        model_sized = _model_slice(sized, model_name)
        model_filled = _model_slice(filled, model_name)
        raw_test_count = _selection_count(
            model_selection,
            "raw_test_trade_count",
            default=len(model_sized),
        )
        eligible_test_count = _selection_count(
            model_selection,
            "test_trade_count",
            default=len(model_sized),
        )
        score_rejected = _selection_count(model_selection, "test_rejected_count")
        base_gate_rejected = max(raw_test_count - eligible_test_count, 0)
        capacity_rejected = max(len(model_sized) - len(model_filled), 0)
        total_rejected = max(raw_test_count - len(model_filled), 0)
        period_pnl = _period_pnl(model_filled)
        total_pnl = _pnl_sum(model_filled)

        rows.append(
            {
                "scenario": scenario.name,
                "policy_name": scenario.policy_name,
                "description": scenario.description,
                "model_name": model_name,
                "accepted_count": int(len(model_filled)),
                "base_gate_rejected_count": int(base_gate_rejected),
                "score_rejected_count": int(score_rejected),
                "capacity_rejected_count": int(capacity_rejected),
                "total_rejected_count": int(total_rejected),
                "total_candidate_count": int(raw_test_count),
                "rejection_rate": float(total_rejected / raw_test_count)
                if raw_test_count
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
                "mean_selected_score_threshold": _selection_mean(
                    model_selection,
                    "selected_score_threshold",
                ),
                "period_count": int(period_pnl.shape[0]),
                "positive_period_share": float(period_pnl.gt(0).mean())
                if not period_pnl.empty
                else np.nan,
                "top_1_period_pnl_share": _top_period_share(period_pnl, total_pnl, 1),
                "top_2_period_pnl_share": _top_period_share(period_pnl, total_pnl, 2),
                "survives_capacity_stress": survives_capacity_stress(
                    model_filled,
                    period_pnl,
                    total_pnl,
                ),
                **_policy_fields(scenario.policy),
                **_config_fields(scenario.capacity_config),
            }
        )
    return rows


def build_markdown_report(
    summary: pd.DataFrame,
    period: pd.DataFrame,
    scenarios: list[PaperTradeRejectionScenario],
) -> str:
    """Build a concise Markdown report for stricter $5+ paper-trade gates."""
    lightgbm = summary[summary["model_name"].eq("lightgbm_rank_blend")].copy()
    lines = [
        "# Day 25 $5+ Paper-Trade Rejection Gates",
        "",
        "## LightGBM Summary",
        "",
        _markdown_table(lightgbm),
        "",
        "## Rejection Gate Scenarios",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "scenario": scenario.name,
                        "policy_name": scenario.policy_name,
                        "description": scenario.description,
                        **_policy_fields(scenario.policy),
                    }
                    for scenario in scenarios
                ]
            )
        ),
        "",
        "## Survival Criteria",
        "",
        (
            f"Survival requires positive PnL, profit factor above 1.0, at least "
            f"{MIN_ACCEPTED_TRADES} accepted trades, top-2 period PnL share at or "
            f"below {MAX_TOP_2_PERIOD_PNL_SHARE:.0%}, positive-period share at or "
            f"above {MIN_POSITIVE_PERIOD_SHARE:.0%}, and max drawdown above "
            f"{MAX_ACCEPTED_DRAWDOWN:.0%}."
        ),
        "",
        (
            "CSFloat listing-count gates are intentionally excluded until point-in-time "
            "coverage is nonzero, so Day 18 can remain a true ablation."
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
        return "No LightGBM rows were generated."
    target = lightgbm[lightgbm["scenario"].eq("paper_trade_5_plus_strict_open_10")].copy()
    if target.empty:
        target = lightgbm[lightgbm["scenario"].eq("paper_trade_5_plus_gates")].copy()
    if target.empty:
        return "No stricter paper-trade row was generated."
    row = target.iloc[0]
    if not bool(row.get("survives_capacity_stress", False)):
        return (
            "The stricter $5+ paper-trade candidate does not clear the survival "
            "criteria. Keep it in research mode and reject live-capital promotion."
        )
    return (
        "The stricter $5+ paper-trade candidate clears the configured survival "
        "criteria, but the correct recommendation remains paper-trade-only until "
        "CSFloat point-in-time coverage and live-paper monitoring are available."
    )


def _day24_capacity_config(name: str) -> CapacitySizingConfig:
    for scenario in DAY24_CAPACITY_SCENARIOS:
        if scenario.name == name:
            return scenario.config
    raise ValueError(f"Unknown Day 24 capacity scenario: {name}")


def _selection_count(selection: pd.DataFrame, column: str, default: int = 0) -> int:
    if column not in selection.columns:
        return int(default)
    return int(pd.to_numeric(selection[column], errors="coerce").fillna(0.0).sum())


def _selection_mean(selection: pd.DataFrame, column: str) -> float:
    if column not in selection.columns:
        return np.nan
    values = pd.to_numeric(selection[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _model_slice(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if frame.empty or "model_name" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["model_name"].astype(str).eq(model_name)].copy()


def _policy_fields(policy: RejectionPolicy) -> dict[str, Any]:
    return {
        "min_entry_price": policy.min_entry_price,
        "min_liquidity_quality": policy.min_liquidity_quality,
        "max_staleness_days": policy.max_staleness_days,
        "max_price_jump_share": policy.max_price_jump_share,
        "exclude_event_regime": policy.exclude_event_regime,
        "min_row_coverage": policy.min_row_coverage,
        "reject_bear_without_alpha": policy.reject_bear_without_alpha,
        "min_item_excess_log_return_7d": policy.min_item_excess_log_return_7d,
        "max_abs_return_30d": policy.max_abs_return_30d,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Day 25 stricter $5+ paper-trade rejection gates.",
    )
    parser.add_argument("--ledger-input", type=Path, default=DEFAULT_LEDGER_INPUT)
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--liquidity-input", type=Path, default=DEFAULT_LIQUIDITY_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--period-output", type=Path, default=DEFAULT_PERIOD_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_day25_paper_trade_rejection(
        ledger_input=args.ledger_input,
        features_input=args.features_input,
        liquidity_input=args.liquidity_input,
        summary_output=args.summary_output,
        period_output=args.period_output,
        report_output=args.report_output,
    )
    print(f"Summary rows: {result.summary_rows}")
    print(f"Period rows: {result.period_rows}")
    print(f"Selection rows: {result.selection_rows}")
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
