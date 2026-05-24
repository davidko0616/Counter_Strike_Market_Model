"""Day 15 rejection-aware normal-regime trade analysis.

Adds a post-hoc rejection layer to the Backtest V1 trade ledger so the model
can abstain from low-conviction or low-quality trades.  The module sweeps
rejection thresholds across multiple dimensions and reports accepted-trade
metrics to identify policies that improve normal-regime PnL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.backtesting.portfolio import (
    DEFAULT_TRADE_LEDGER_OUTPUT,
    PortfolioBacktestConfig,
    _max_drawdown,
)
from cs_market_model.config import PROJECT_ROOT, load_yaml_config, reports_path

DEFAULT_REJECTION_CURVE_OUTPUT = reports_path("tables", "day15_rejection_curve.csv")
DEFAULT_POLICY_SUMMARY_OUTPUT = reports_path("tables", "day15_threshold_policy_summary.csv")
DEFAULT_NORMAL_ACCEPTED_OUTPUT = reports_path(
    "tables", "day15_normal_accepted_trade_summary.csv"
)


# ---------------------------------------------------------------------------
# Rejection policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RejectionPolicy:
    """Configurable trade rejection thresholds.

    Each gate is applied in priority order.  The first failing gate sets
    ``rejection_reason``.  Trades that pass all gates are accepted.
    """

    min_score_threshold: float = 0.0
    min_liquidity_quality: float = 0.0
    max_staleness_days: float = float("inf")
    max_price_jump_share: float = 1.0
    exclude_event_regime: bool = True
    min_row_coverage: float = 0.0
    reject_bear_without_alpha: bool = False

    def __post_init__(self) -> None:
        if self.min_score_threshold < 0:
            raise ValueError("min_score_threshold must be non-negative")
        if self.min_liquidity_quality < 0:
            raise ValueError("min_liquidity_quality must be non-negative")
        if self.max_staleness_days <= 0:
            raise ValueError("max_staleness_days must be positive")
        if self.max_price_jump_share < 0:
            raise ValueError("max_price_jump_share must be non-negative")
        if self.min_row_coverage < 0:
            raise ValueError("min_row_coverage must be non-negative")


# ---------------------------------------------------------------------------
# Apply rejection
# ---------------------------------------------------------------------------


def apply_rejection_policy(
    ledger: pd.DataFrame,
    policy: RejectionPolicy | None = None,
) -> pd.DataFrame:
    """Apply rejection gates to a trade ledger and tag each trade.

    Returns a copy of *ledger* with two new columns:

    * ``rejection_reason`` — first failing gate name, or ``None`` if accepted.
    * ``is_accepted`` — boolean flag.
    """
    resolved_policy = policy or RejectionPolicy()
    frame = ledger.copy()

    if frame.empty:
        frame["rejection_reason"] = pd.Series(dtype="object")
        frame["is_accepted"] = pd.Series(dtype="bool")
        return frame

    reasons: list[str | None] = [None] * len(frame)

    # Gate 1: event-regime exclusion
    if resolved_policy.exclude_event_regime and "label_market_regime" in frame.columns:
        event_mask = frame["label_market_regime"].astype(str).eq("known_event")
        for idx in frame.index[event_mask]:
            pos = frame.index.get_loc(idx)
            if reasons[pos] is None:
                reasons[pos] = "event_regime"

    # Gate 2: score threshold
    if resolved_policy.min_score_threshold > 0 and "strong_buy_score" in frame.columns:
        score = pd.to_numeric(frame["strong_buy_score"], errors="coerce").fillna(0.0)
        low_score = score.lt(resolved_policy.min_score_threshold)
        for idx in frame.index[low_score]:
            pos = frame.index.get_loc(idx)
            if reasons[pos] is None:
                reasons[pos] = "low_score"

    # Gate 3: liquidity quality
    if resolved_policy.min_liquidity_quality > 0:
        liq_col = "liquidity_quality_score_30d"
        if liq_col in frame.columns:
            liq = pd.to_numeric(frame[liq_col], errors="coerce").fillna(0.0)
            low_liq = liq.lt(resolved_policy.min_liquidity_quality)
            for idx in frame.index[low_liq]:
                pos = frame.index.get_loc(idx)
                if reasons[pos] is None:
                    reasons[pos] = "low_liquidity"

    # Gate 4: staleness
    if np.isfinite(resolved_policy.max_staleness_days):
        stale_col = "effective_staleness_days"
        if stale_col in frame.columns:
            stale = pd.to_numeric(frame[stale_col], errors="coerce").fillna(0.0)
            too_stale = stale.gt(resolved_policy.max_staleness_days)
            for idx in frame.index[too_stale]:
                pos = frame.index.get_loc(idx)
                if reasons[pos] is None:
                    reasons[pos] = "stale_price"

    # Gate 5: price jump share
    if resolved_policy.max_price_jump_share < 1.0:
        jump_col = "price_jump_50pct_share_30d"
        if jump_col in frame.columns:
            jump = pd.to_numeric(frame[jump_col], errors="coerce").fillna(0.0)
            high_jump = jump.gt(resolved_policy.max_price_jump_share)
            for idx in frame.index[high_jump]:
                pos = frame.index.get_loc(idx)
                if reasons[pos] is None:
                    reasons[pos] = "price_jump"

    # Gate 6: row coverage
    if resolved_policy.min_row_coverage > 0:
        cov_col = "row_coverage_30d"
        if cov_col in frame.columns:
            cov = pd.to_numeric(frame[cov_col], errors="coerce").fillna(0.0)
            low_cov = cov.lt(resolved_policy.min_row_coverage)
            for idx in frame.index[low_cov]:
                pos = frame.index.get_loc(idx)
                if reasons[pos] is None:
                    reasons[pos] = "low_coverage"

    # Gate 7: bear market regime — reject if CS2 index is bearish AND
    # the item is not outperforming the market (no excess alpha)
    if resolved_policy.reject_bear_without_alpha:
        bear_col = "cs2_index_regime_bear"
        excess_col = "item_excess_log_return_7d"
        if bear_col in frame.columns and excess_col in frame.columns:
            is_bear = pd.to_numeric(
                frame[bear_col], errors="coerce"
            ).fillna(0).astype(bool)
            excess = pd.to_numeric(frame[excess_col], errors="coerce").fillna(0.0)
            bear_no_alpha = is_bear & excess.le(0)
            for idx in frame.index[bear_no_alpha]:
                pos = frame.index.get_loc(idx)
                if reasons[pos] is None:
                    reasons[pos] = "bear_no_alpha"

    frame["rejection_reason"] = reasons
    frame["is_accepted"] = frame["rejection_reason"].isna()
    return frame


# ---------------------------------------------------------------------------
# Accepted-trade metrics
# ---------------------------------------------------------------------------


def _accepted_trade_metrics(
    accepted: pd.DataFrame,
    total_count: int,
) -> dict[str, Any]:
    """Compute summary metrics for a set of accepted trades."""
    if accepted.empty:
        return {
            "accepted_count": 0,
            "rejected_count": total_count,
            "rejection_rate": 1.0 if total_count > 0 else np.nan,
            "accepted_total_pnl": 0.0,
            "accepted_avg_return": np.nan,
            "accepted_win_rate": np.nan,
            "accepted_precision": np.nan,
            "accepted_max_drawdown": np.nan,
            "profit_factor": np.nan,
            "accepted_sharpe": np.nan,
        }

    returns = pd.to_numeric(accepted["realized_net_return"], errors="coerce")
    pnl = pd.to_numeric(accepted["pnl"], errors="coerce").fillna(0.0)
    accepted_count = len(accepted)
    rejected_count = total_count - accepted_count

    gross_wins = pnl[pnl > 0].sum()
    gross_losses = abs(pnl[pnl < 0].sum())
    if gross_losses > 0:
        profit_factor = float(gross_wins / gross_losses)
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Approximate Sharpe from trade returns
    if returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(365))
    else:
        sharpe = np.nan

    # Drawdown from cumulative PnL
    cum_pnl = pnl.cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd = float(dd.min()) if not dd.empty else np.nan

    return {
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "rejection_rate": float(rejected_count / total_count) if total_count > 0 else np.nan,
        "accepted_total_pnl": float(pnl.sum()),
        "accepted_avg_return": float(returns.mean()),
        "accepted_win_rate": float((returns > 0).mean()),
        "accepted_precision": float(accepted["is_actual_strong_buy"].mean())
        if "is_actual_strong_buy" in accepted.columns
        else np.nan,
        "accepted_max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "accepted_sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Rejection curves
# ---------------------------------------------------------------------------


def build_rejection_curve(
    ledger: pd.DataFrame,
    dimension: str,
    thresholds: list[float],
    *,
    base_policy: RejectionPolicy | None = None,
) -> pd.DataFrame:
    """Sweep a single rejection dimension and compute accepted-trade metrics.

    Parameters
    ----------
    ledger:
        Trade ledger with rejection-relevant columns.
    dimension:
        Name of the ``RejectionPolicy`` field to sweep.
    thresholds:
        Ordered list of threshold values to test.
    base_policy:
        Starting policy; the swept dimension is overridden at each step.
    """
    resolved_base = base_policy or RejectionPolicy(exclude_event_regime=True)
    rows: list[dict[str, Any]] = []

    for model_name, model_trades in ledger.groupby("model_name", sort=True):
        total = len(model_trades)
        for threshold in thresholds:
            policy_kwargs = {
                "min_score_threshold": resolved_base.min_score_threshold,
                "min_liquidity_quality": resolved_base.min_liquidity_quality,
                "max_staleness_days": resolved_base.max_staleness_days,
                "max_price_jump_share": resolved_base.max_price_jump_share,
                "exclude_event_regime": resolved_base.exclude_event_regime,
                "min_row_coverage": resolved_base.min_row_coverage,
            }
            policy_kwargs[dimension] = threshold
            policy = RejectionPolicy(**policy_kwargs)
            tagged = apply_rejection_policy(model_trades, policy)
            accepted = tagged[tagged["is_accepted"]]
            metrics = _accepted_trade_metrics(accepted, total)
            metrics["model_name"] = model_name
            metrics["dimension"] = dimension
            metrics["threshold_value"] = threshold
            rows.append(metrics)

    return pd.DataFrame(rows)


def build_multi_dimension_rejection_curves(
    ledger: pd.DataFrame,
    *,
    base_policy: RejectionPolicy | None = None,
) -> pd.DataFrame:
    """Sweep all key rejection dimensions and return combined curves."""
    dimension_sweeps: dict[str, list[float]] = {
        "min_score_threshold": [round(v, 2) for v in np.arange(0.0, 0.95, 0.05)],
        "min_liquidity_quality": [round(v, 1) for v in np.arange(0.0, 0.85, 0.1)],
        "min_row_coverage": [round(v, 1) for v in np.arange(0.0, 0.95, 0.1)],
        "max_staleness_days": [float("inf"), 10.0, 7.0, 5.0, 3.0, 2.0, 1.0],
        "max_price_jump_share": [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.0],
    }

    curves: list[pd.DataFrame] = []
    for dimension, thresholds in dimension_sweeps.items():
        curve = build_rejection_curve(
            ledger,
            dimension=dimension,
            thresholds=thresholds,
            base_policy=base_policy,
        )
        curves.append(curve)

    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


# ---------------------------------------------------------------------------
# Threshold policy comparison
# ---------------------------------------------------------------------------


def policies_from_config() -> list[tuple[str, RejectionPolicy]]:
    """Load named rejection policies from ``configs/backtest.yaml``."""
    config = load_yaml_config("backtest.yaml")
    raw_policies = config.get("rejection_policies", {}) or {}
    policies: list[tuple[str, RejectionPolicy]] = []
    for name, params in raw_policies.items():
        if not isinstance(params, dict):
            continue
        policies.append((
            str(name),
            RejectionPolicy(
                min_score_threshold=float(params.get("min_score_threshold", 0.0)),
                min_liquidity_quality=float(params.get("min_liquidity_quality", 0.0)),
                max_staleness_days=float(params.get("max_staleness_days", float("inf"))),
                max_price_jump_share=float(params.get("max_price_jump_share", 1.0)),
                exclude_event_regime=bool(params.get("exclude_event_regime", True)),
                min_row_coverage=float(params.get("min_row_coverage", 0.0)),
                reject_bear_without_alpha=bool(params.get("reject_bear_without_alpha", False)),
            ),
        ))
    return policies


def summarize_threshold_policies(
    ledger: pd.DataFrame,
    policies: list[tuple[str, RejectionPolicy]] | None = None,
) -> pd.DataFrame:
    """Apply each named policy and report per-model summary metrics."""
    resolved_policies = policies or _default_policies()
    rows: list[dict[str, Any]] = []

    for model_name, model_trades in ledger.groupby("model_name", sort=True):
        total = len(model_trades)
        for policy_name, policy in resolved_policies:
            tagged = apply_rejection_policy(model_trades, policy)
            accepted = tagged[tagged["is_accepted"]]
            metrics = _accepted_trade_metrics(accepted, total)
            metrics["policy_name"] = policy_name
            metrics["model_name"] = model_name
            rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    column_order = ["policy_name", "model_name"] + [
        col for col in result.columns if col not in ("policy_name", "model_name")
    ]
    return result[column_order].sort_values(
        ["model_name", "accepted_total_pnl"],
        ascending=[True, False],
    )


def _default_policies() -> list[tuple[str, RejectionPolicy]]:
    """Built-in policy presets for comparison."""
    return [
        ("no_filter", RejectionPolicy(exclude_event_regime=False)),
        ("normal_only", RejectionPolicy(exclude_event_regime=True)),
        ("conservative", RejectionPolicy(
            min_score_threshold=0.6,
            min_liquidity_quality=0.5,
            max_staleness_days=3.0,
            max_price_jump_share=0.1,
            exclude_event_regime=True,
            min_row_coverage=0.5,
        )),
        ("aggressive", RejectionPolicy(
            min_score_threshold=0.3,
            min_liquidity_quality=0.3,
            exclude_event_regime=True,
        )),
        ("quality_gate", RejectionPolicy(
            min_row_coverage=0.5,
            max_staleness_days=2.0,
            max_price_jump_share=0.05,
            exclude_event_regime=True,
        )),
        ("bear_regime_aware", RejectionPolicy(
            min_score_threshold=0.6,
            min_liquidity_quality=0.5,
            max_staleness_days=3.0,
            max_price_jump_share=0.1,
            exclude_event_regime=True,
            min_row_coverage=0.5,
            reject_bear_without_alpha=True,
        )),
    ]


# ---------------------------------------------------------------------------
# Normal-regime accepted-trade summary
# ---------------------------------------------------------------------------


def summarize_normal_accepted_trades(
    ledger: pd.DataFrame,
    policies: list[tuple[str, RejectionPolicy]] | None = None,
) -> pd.DataFrame:
    """Summarize accepted trades restricted to normal-regime rows only.

    Event-regime trades are excluded from the summary regardless of policy
    settings so threshold tuning never uses event-driven results.
    """
    resolved_policies = policies or _default_policies()

    # Pre-filter to normal-regime only
    if "label_market_regime" in ledger.columns:
        normal_ledger = ledger[
            ledger["label_market_regime"].astype(str).ne("known_event")
        ].copy()
    else:
        normal_ledger = ledger.copy()

    rows: list[dict[str, Any]] = []
    for model_name, model_trades in normal_ledger.groupby("model_name", sort=True):
        total = len(model_trades)
        for policy_name, policy in resolved_policies:
            # Force exclude_event_regime off since we've already filtered
            normal_policy = RejectionPolicy(
                min_score_threshold=policy.min_score_threshold,
                min_liquidity_quality=policy.min_liquidity_quality,
                max_staleness_days=policy.max_staleness_days,
                max_price_jump_share=policy.max_price_jump_share,
                exclude_event_regime=False,
                min_row_coverage=policy.min_row_coverage,
                reject_bear_without_alpha=policy.reject_bear_without_alpha,
            )
            tagged = apply_rejection_policy(model_trades, normal_policy)
            accepted = tagged[tagged["is_accepted"]]
            metrics = _accepted_trade_metrics(accepted, total)
            metrics["policy_name"] = policy_name
            metrics["model_name"] = model_name
            rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    column_order = ["policy_name", "model_name"] + [
        col for col in result.columns if col not in ("policy_name", "model_name")
    ]
    return result[column_order].sort_values(
        ["model_name", "accepted_total_pnl"],
        ascending=[True, False],
    )


# ---------------------------------------------------------------------------
# Day 15 orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Day15Result:
    """Output paths and counts for the Day 15 rejection analysis."""

    rejection_curve_output: Path
    policy_summary_output: Path
    normal_accepted_output: Path
    curve_rows: int
    policy_rows: int
    normal_rows: int


def run_day15_rejection_analysis(
    trade_ledger_input: Path = DEFAULT_TRADE_LEDGER_OUTPUT,
    rejection_curve_output: Path = DEFAULT_REJECTION_CURVE_OUTPUT,
    policy_summary_output: Path = DEFAULT_POLICY_SUMMARY_OUTPUT,
    normal_accepted_output: Path = DEFAULT_NORMAL_ACCEPTED_OUTPUT,
) -> Day15Result:
    """Run the full Day 15 rejection analysis pipeline."""
    if not trade_ledger_input.exists():
        raise FileNotFoundError(f"Trade ledger not found: {trade_ledger_input}")

    ledger = pd.read_csv(trade_ledger_input)
    print(f"Loaded trade ledger: {len(ledger)} trades")

    # Load config-based policies + built-in defaults
    config_policies = policies_from_config()
    all_policies = _default_policies()
    config_names = {name for name, _ in all_policies}
    for name, policy in config_policies:
        if name not in config_names:
            all_policies.append((name, policy))

    # Build multi-dimension rejection curves
    print("Building rejection curves...")
    curves = build_multi_dimension_rejection_curves(ledger)
    print(f"  Rejection curve rows: {len(curves)}")

    # Summarize named threshold policies
    print("Evaluating threshold policies...")
    policy_summary = summarize_threshold_policies(ledger, all_policies)
    print(f"  Policy summary rows: {len(policy_summary)}")

    # Normal-regime accepted-trade summary
    print("Summarizing normal-regime accepted trades...")
    normal_summary = summarize_normal_accepted_trades(ledger, all_policies)
    print(f"  Normal accepted summary rows: {len(normal_summary)}")

    # Write outputs
    for output in (rejection_curve_output, policy_summary_output, normal_accepted_output):
        output.parent.mkdir(parents=True, exist_ok=True)

    curves.to_csv(rejection_curve_output, index=False)
    policy_summary.to_csv(policy_summary_output, index=False)
    normal_summary.to_csv(normal_accepted_output, index=False)

    return Day15Result(
        rejection_curve_output=rejection_curve_output,
        policy_summary_output=policy_summary_output,
        normal_accepted_output=normal_accepted_output,
        curve_rows=len(curves),
        policy_rows=len(policy_summary),
        normal_rows=len(normal_summary),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 15 rejection analysis.")
    parser.add_argument(
        "--trade-ledger-input",
        type=Path,
        default=DEFAULT_TRADE_LEDGER_OUTPUT,
    )
    parser.add_argument(
        "--rejection-curve-output",
        type=Path,
        default=DEFAULT_REJECTION_CURVE_OUTPUT,
    )
    parser.add_argument(
        "--policy-summary-output",
        type=Path,
        default=DEFAULT_POLICY_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--normal-accepted-output",
        type=Path,
        default=DEFAULT_NORMAL_ACCEPTED_OUTPUT,
    )
    args = parser.parse_args()

    result = run_day15_rejection_analysis(
        trade_ledger_input=args.trade_ledger_input,
        rejection_curve_output=args.rejection_curve_output,
        policy_summary_output=args.policy_summary_output,
        normal_accepted_output=args.normal_accepted_output,
    )
    print(f"\nWrote rejection curves: {_display_path(result.rejection_curve_output)}")
    print(f"Wrote policy summary: {_display_path(result.policy_summary_output)}")
    print(f"Wrote normal accepted: {_display_path(result.normal_accepted_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
