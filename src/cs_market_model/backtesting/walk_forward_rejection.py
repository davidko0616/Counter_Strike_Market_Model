"""Day 16 walk-forward validation for rejection thresholds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.backtesting.portfolio import DEFAULT_TRADE_LEDGER_OUTPUT
from cs_market_model.backtesting.rejection import (
    _accepted_trade_metrics,
    apply_rejection_policy,
)
from cs_market_model.backtesting.rejection_policy import RejectionPolicy
from cs_market_model.config import PROJECT_ROOT, reports_path

DEFAULT_SELECTION_OUTPUT = reports_path(
    "tables",
    "day16_walk_forward_threshold_selection.csv",
)
DEFAULT_SUMMARY_OUTPUT = reports_path(
    "tables",
    "day16_walk_forward_threshold_summary.csv",
)
DEFAULT_ACCEPTED_OUTPUT = reports_path(
    "backtests",
    "day16_walk_forward_accepted_trades.csv",
)


@dataclass(frozen=True)
class Day16Result:
    """Output paths and counts for Day 16 walk-forward validation."""

    selection_output: Path
    summary_output: Path
    accepted_output: Path
    selection_rows: int
    summary_rows: int
    accepted_rows: int


def run_day16_walk_forward_rejection(
    trade_ledger_input: Path = DEFAULT_TRADE_LEDGER_OUTPUT,
    selection_output: Path = DEFAULT_SELECTION_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    accepted_output: Path = DEFAULT_ACCEPTED_OUTPUT,
    min_training_trades: int = 100,
    min_accepted_trades: int = 20,
    base_policy: RejectionPolicy | None = None,
) -> Day16Result:
    """Run walk-forward threshold selection from the no-filter trade ledger."""
    if not trade_ledger_input.exists():
        raise FileNotFoundError(f"Trade ledger not found: {trade_ledger_input}")

    ledger = pd.read_csv(trade_ledger_input)
    selection, accepted = walk_forward_score_threshold_selection(
        ledger,
        min_training_trades=min_training_trades,
        min_accepted_trades=min_accepted_trades,
        base_policy=base_policy,
    )
    summary = summarize_walk_forward_selection(selection, accepted)

    for output in (selection_output, summary_output, accepted_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_output, index=False)
    summary.to_csv(summary_output, index=False)
    accepted.to_csv(accepted_output, index=False)

    return Day16Result(
        selection_output=selection_output,
        summary_output=summary_output,
        accepted_output=accepted_output,
        selection_rows=len(selection),
        summary_rows=len(summary),
        accepted_rows=len(accepted),
    )


def walk_forward_score_threshold_selection(
    ledger: pd.DataFrame,
    thresholds: list[float] | None = None,
    *,
    min_training_trades: int = 100,
    min_accepted_trades: int = 20,
    base_policy: RejectionPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select score thresholds on prior months and evaluate the next month.

    The input is the no-filter trade ledger. Known-event rows are excluded before
    both threshold selection and test evaluation so the process validates normal
    regime behavior only.
    """
    if ledger.empty:
        return pd.DataFrame(), pd.DataFrame()

    threshold_grid = thresholds or [round(v, 2) for v in np.arange(0.0, 0.95, 0.05)]
    frame = _normal_ledger_with_period(ledger)
    resolved_base_policy = base_policy or RejectionPolicy(exclude_event_regime=False)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    selection_rows: list[dict[str, Any]] = []
    accepted_frames: list[pd.DataFrame] = []

    for model_name, model_trades in frame.groupby("model_name", sort=True):
        periods = sorted(model_trades["_period"].unique())
        for period_index, test_period in enumerate(periods):
            train_periods = periods[:period_index]
            train = model_trades[model_trades["_period"].isin(train_periods)].copy()
            test = model_trades[model_trades["_period"].eq(test_period)].copy()
            if test.empty:
                continue

            selected_threshold, train_metrics = _select_threshold_from_training(
                train,
                threshold_grid,
                min_training_trades=min_training_trades,
                min_accepted_trades=min_accepted_trades,
                base_policy=resolved_base_policy,
            )
            policy = replace(resolved_base_policy, min_score_threshold=selected_threshold)
            tagged_test = apply_rejection_policy(test, policy)
            accepted_test = tagged_test[tagged_test["is_accepted"]].copy()
            test_metrics = _accepted_trade_metrics(accepted_test, len(test))

            accepted_test["selected_threshold"] = selected_threshold
            accepted_test["test_period"] = test_period
            accepted_frames.append(accepted_test)

            row = {
                "model_name": model_name,
                "test_period": test_period,
                "training_period_count": len(train_periods),
                "training_trade_count": len(train),
                "selected_score_threshold": selected_threshold,
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"test_{key}": value for key, value in test_metrics.items()})
            selection_rows.append(row)

    selection = pd.DataFrame(selection_rows)
    accepted = (
        pd.concat(accepted_frames, ignore_index=True, sort=False)
        if accepted_frames
        else pd.DataFrame()
    )
    return selection, accepted


def summarize_walk_forward_selection(
    selection: pd.DataFrame,
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate Day 16 walk-forward accepted trades by model."""
    if selection.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for model_name, model_periods in selection.groupby("model_name", sort=True):
        model_accepted = (
            accepted[accepted["model_name"].eq(model_name)].copy()
            if not accepted.empty and "model_name" in accepted.columns
            else pd.DataFrame()
        )
        total_test_count = int(model_periods["test_accepted_count"].sum() + model_periods["test_rejected_count"].sum())
        metrics = _accepted_trade_metrics(model_accepted, total_test_count)
        rows.append(
            {
                "model_name": model_name,
                "walk_forward_periods": int(len(model_periods)),
                "mean_selected_score_threshold": float(
                    model_periods["selected_score_threshold"].mean()
                ),
                "median_selected_score_threshold": float(
                    model_periods["selected_score_threshold"].median()
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("accepted_total_pnl", ascending=False)


def _normal_ledger_with_period(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = ledger.copy()
    if "label_market_regime" in frame.columns:
        frame = frame[frame["label_market_regime"].astype(str).ne("known_event")].copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["_period"] = frame["entry_timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
    return frame


def _select_threshold_from_training(
    train: pd.DataFrame,
    thresholds: list[float],
    *,
    min_training_trades: int,
    min_accepted_trades: int,
    base_policy: RejectionPolicy | None = None,
) -> tuple[float, dict[str, Any]]:
    if len(train) < min_training_trades:
        return 0.0, _accepted_trade_metrics(train, len(train))

    candidates: list[dict[str, Any]] = []
    resolved_base_policy = base_policy or RejectionPolicy(exclude_event_regime=False)
    for threshold in thresholds:
        policy = replace(resolved_base_policy, min_score_threshold=threshold)
        tagged = apply_rejection_policy(train, policy)
        accepted = tagged[tagged["is_accepted"]]
        metrics = _accepted_trade_metrics(accepted, len(train))
        if metrics["accepted_count"] < min_accepted_trades:
            continue
        candidates.append({"threshold": threshold, **metrics})

    if not candidates:
        return 0.0, _accepted_trade_metrics(train, len(train))

    ranked = pd.DataFrame(candidates).sort_values(
        ["profit_factor", "accepted_win_rate", "accepted_count"],
        ascending=[False, False, False],
    )
    best = ranked.iloc[0]
    threshold = float(best["threshold"])
    return threshold, {
        key: best[key]
        for key in best.index
        if key != "threshold"
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Day 16 walk-forward rejection threshold validation.",
    )
    parser.add_argument("--trade-ledger-input", type=Path, default=DEFAULT_TRADE_LEDGER_OUTPUT)
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--accepted-output", type=Path, default=DEFAULT_ACCEPTED_OUTPUT)
    parser.add_argument("--min-training-trades", type=int, default=100)
    parser.add_argument("--min-accepted-trades", type=int, default=20)
    parser.add_argument("--min-entry-price", type=float, default=0.0)
    args = parser.parse_args()

    result = run_day16_walk_forward_rejection(
        trade_ledger_input=args.trade_ledger_input,
        selection_output=args.selection_output,
        summary_output=args.summary_output,
        accepted_output=args.accepted_output,
        min_training_trades=args.min_training_trades,
        min_accepted_trades=args.min_accepted_trades,
        base_policy=RejectionPolicy(
            exclude_event_regime=False,
            min_entry_price=args.min_entry_price,
        ),
    )
    print(f"Walk-forward periods: {result.selection_rows}")
    print(f"Accepted trades: {result.accepted_rows}")
    print(f"Wrote selection: {_display_path(result.selection_output)}")
    print(f"Wrote summary: {_display_path(result.summary_output)}")
    print(f"Wrote accepted trades: {_display_path(result.accepted_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
