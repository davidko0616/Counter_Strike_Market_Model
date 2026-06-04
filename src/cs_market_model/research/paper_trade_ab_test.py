"""Conservative paper-trade A/B review for current and stricter gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path

DEFAULT_CURRENT_ACCEPTED_INPUT = reports_path(
    "backtests",
    "day25_current_day24_gates_accepted_trades.csv",
)
DEFAULT_STRICT_ACCEPTED_INPUT = reports_path(
    "backtests",
    "day25_paper_trade_5_plus_gates_accepted_trades.csv",
)
DEFAULT_SUMMARY_OUTPUT = reports_path("tables", "paper_trade_ab_test_summary.csv")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "paper_trade_ab_test_report.md")
DEFAULT_LEDGER_DIR = data_path("paper_trades")

MIN_REVIEW_DAYS = 30
EXTENDED_REVIEW_DAYS = 60
MIN_ACCEPTED_TRADES = 50
MIN_PROFIT_FACTOR = 1.20
MIN_PROFIT_FACTOR_CI_LOWER = 1.00
MAX_DRAWDOWN = -0.10
MAX_TOP_1_ITEM_PNL_SHARE = 0.25
MAX_TOP_3_ITEM_PNL_SHARE = 0.50
MAX_TOP_2_PERIOD_PNL_SHARE = 0.80
MIN_POSITIVE_PERIOD_SHARE = 0.50

PAPER_TRADE_LEDGER_COLUMNS = [
    "date",
    "market_hash_name",
    "policy",
    "model_name",
    "signal_score",
    "entry_price",
    "current_price",
    "exit_price",
    "exit_date",
    "pnl",
    "status",
]


@dataclass(frozen=True)
class PaperTradeABResult:
    """Output metadata for the paper-trade A/B review."""

    summary_output: Path
    report_output: Path
    summary_rows: int


def run_paper_trade_ab_test(
    current_accepted_input: Path = DEFAULT_CURRENT_ACCEPTED_INPUT,
    strict_accepted_input: Path = DEFAULT_STRICT_ACCEPTED_INPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> PaperTradeABResult:
    """Write the conservative paper-trade A/B summary and Markdown report."""
    scenarios = [
        ("no_trade_baseline", "no_trade", pd.DataFrame()),
        (
            "current_day24_5_plus",
            "current_day24_gates",
            _read_trade_file(current_accepted_input),
        ),
        (
            "paper_trade_5_plus",
            "paper_trade_5_plus_gates",
            _read_trade_file(strict_accepted_input),
        ),
    ]
    rows = [evaluate_policy_trades(name, policy, trades) for name, policy, trades in scenarios]
    summary = pd.DataFrame(rows)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    report_output.write_text(build_paper_trade_report(summary), encoding="utf-8")
    return PaperTradeABResult(
        summary_output=summary_output,
        report_output=report_output,
        summary_rows=len(summary),
    )


def evaluate_policy_trades(
    scenario: str,
    policy: str,
    trades: pd.DataFrame,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return conservative review metrics for one policy's paper-trade set."""
    if trades.empty:
        return {
            "scenario": scenario,
            "policy": policy,
            "accepted_count": 0,
            "review_days": 0,
            "profit_factor": 0.0,
            "profit_factor_ci_low": 0.0,
            "profit_factor_ci_high": 0.0,
            "max_drawdown": np.nan,
            "top_1_item_pnl_share": np.nan,
            "top_3_item_pnl_share": np.nan,
            "top_2_period_pnl_share": np.nan,
            "positive_period_share": np.nan,
            "review_status": "no_trades",
            "final_recommendation": "paper_trade_only",
        }

    frame = _normalized_trade_frame(trades)
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    review_days = _review_days(frame)
    profit_factor = _profit_factor(pnl)
    ci_low, ci_high = bootstrap_profit_factor_ci(pnl, random_state=random_state)
    max_drawdown = _max_drawdown(frame)
    item_shares = _top_item_pnl_shares(frame)
    period_pnl = _period_pnl(frame)
    top_2_period_share = _top_positive_share(period_pnl, 2, float(pnl.sum()))
    positive_period_share = float(period_pnl.gt(0).mean()) if not period_pnl.empty else np.nan
    status = _review_status(
        accepted_count=len(frame),
        review_days=review_days,
        profit_factor=profit_factor,
        ci_low=ci_low,
        max_drawdown=max_drawdown,
        top_1_item_share=item_shares["top_1_item_pnl_share"],
        top_3_item_share=item_shares["top_3_item_pnl_share"],
        top_2_period_share=top_2_period_share,
        positive_period_share=positive_period_share,
    )
    return {
        "scenario": scenario,
        "policy": policy,
        "accepted_count": int(len(frame)),
        "review_days": int(review_days),
        "profit_factor": profit_factor,
        "profit_factor_ci_low": ci_low,
        "profit_factor_ci_high": ci_high,
        "max_drawdown": max_drawdown,
        **item_shares,
        "top_2_period_pnl_share": top_2_period_share,
        "positive_period_share": positive_period_share,
        "review_status": status,
        "final_recommendation": "paper_trade_only",
    }


def bootstrap_profit_factor_ci(
    pnl: pd.Series | list[float] | np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return bootstrap 95% confidence interval for profit factor."""
    values = pd.to_numeric(pd.Series(pnl), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(random_state)
    bootstrapped = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrapped[index] = _profit_factor(pd.Series(sample))
    finite = bootstrapped[np.isfinite(bootstrapped)]
    if len(finite) == 0:
        return float("inf"), float("inf")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def normalize_paper_trade_ledger(rows: pd.DataFrame) -> pd.DataFrame:
    """Return a ledger frame with the required paper-trade schema."""
    frame = rows.copy()
    for column in PAPER_TRADE_LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], utc=True, errors="coerce")
    frame["status"] = frame["status"].fillna("open").astype(str)
    return frame[PAPER_TRADE_LEDGER_COLUMNS]


def append_paper_trade_ledger(
    rows: pd.DataFrame,
    *,
    ledger_dir: Path = DEFAULT_LEDGER_DIR,
) -> Path:
    """Append rows to the month-level paper-trade parquet ledger."""
    normalized = normalize_paper_trade_ledger(rows)
    if normalized.empty or normalized["date"].isna().all():
        raise ValueError("Paper-trade rows must include at least one valid date.")
    month = normalized["date"].min().strftime("%Y-%m")
    output = ledger_dir / f"trades_{month}.parquet"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    combined = pd.concat([existing, normalized], ignore_index=True, sort=False)
    combined.to_parquet(output, index=False)
    return output


def build_paper_trade_report(summary: pd.DataFrame) -> str:
    """Build a compact Markdown report for paper-trade A/B review."""
    lines = [
        "# Paper-Trade A/B Test",
        "",
        "## Decision",
        "",
        "This system remains paper-trade-only. No live capital path is included.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Conservative Review Criteria",
        "",
        (
            f"Review requires at least {MIN_REVIEW_DAYS} calendar days, "
            f"{MIN_ACCEPTED_TRADES} accepted trades, profit factor >= "
            f"{MIN_PROFIT_FACTOR:.2f}, bootstrap lower bound >= "
            f"{MIN_PROFIT_FACTOR_CI_LOWER:.2f}, max drawdown better than "
            f"{MAX_DRAWDOWN:.0%}, top-1 item PnL share <= "
            f"{MAX_TOP_1_ITEM_PNL_SHARE:.0%}, top-3 item PnL share <= "
            f"{MAX_TOP_3_ITEM_PNL_SHARE:.0%}, top-2 period PnL share <= "
            f"{MAX_TOP_2_PERIOD_PNL_SHARE:.0%}, and positive-period share >= "
            f"{MIN_POSITIVE_PERIOD_SHARE:.0%}."
        ),
        "",
    ]
    return "\n".join(lines)


def _review_status(
    *,
    accepted_count: int,
    review_days: int,
    profit_factor: float,
    ci_low: float,
    max_drawdown: float,
    top_1_item_share: float,
    top_3_item_share: float,
    top_2_period_share: float,
    positive_period_share: float,
) -> str:
    if review_days < MIN_REVIEW_DAYS:
        return "paper_trade_window_too_short"
    if accepted_count < MIN_ACCEPTED_TRADES and review_days < EXTENDED_REVIEW_DAYS:
        return "extend_to_60_days"
    if accepted_count < MIN_ACCEPTED_TRADES:
        return "signal_too_rare_for_review"
    checks = {
        "profit_factor": profit_factor >= MIN_PROFIT_FACTOR,
        "profit_factor_ci": ci_low >= MIN_PROFIT_FACTOR_CI_LOWER,
        "drawdown": pd.notna(max_drawdown) and max_drawdown >= MAX_DRAWDOWN,
        "top_1_item": pd.notna(top_1_item_share)
        and top_1_item_share <= MAX_TOP_1_ITEM_PNL_SHARE,
        "top_3_item": pd.notna(top_3_item_share)
        and top_3_item_share <= MAX_TOP_3_ITEM_PNL_SHARE,
        "top_2_period": pd.notna(top_2_period_share)
        and top_2_period_share <= MAX_TOP_2_PERIOD_PNL_SHARE,
        "positive_period": pd.notna(positive_period_share)
        and positive_period_share >= MIN_POSITIVE_PERIOD_SHARE,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return "failed_" + "_".join(failed)
    return "passes_paper_trade_review"


def _normalized_trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    if "entry_timestamp" in frame.columns:
        frame["date"] = pd.to_datetime(frame["entry_timestamp"], utc=True, errors="coerce")
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    else:
        frame["date"] = pd.NaT
    if "pnl" not in frame.columns:
        frame["pnl"] = 0.0
    if "market_hash_name" not in frame.columns:
        frame["market_hash_name"] = "unknown"
    return frame.dropna(subset=["date"]).reset_index(drop=True)


def _read_trade_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _review_days(frame: pd.DataFrame) -> int:
    if frame.empty or frame["date"].isna().all():
        return 0
    return int((frame["date"].max() - frame["date"].min()).days + 1)


def _profit_factor(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    gross_profit = values[values > 0].sum()
    gross_loss = abs(values[values < 0].sum())
    if gross_loss > 0:
        return float(gross_profit / gross_loss)
    return float("inf") if gross_profit > 0 else 0.0


def _max_drawdown(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    ordered = frame.sort_values(["date", "market_hash_name"]).copy()
    equity = 1.0 + pd.to_numeric(ordered["pnl"], errors="coerce").fillna(0.0).cumsum()
    running_peak = equity.cummax()
    return float(equity.div(running_peak).sub(1.0).min())


def _top_item_pnl_shares(frame: pd.DataFrame) -> dict[str, float]:
    total_pnl = float(pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0).sum())
    if total_pnl <= 0:
        return {"top_1_item_pnl_share": np.nan, "top_3_item_pnl_share": np.nan}
    item_pnl = (
        pd.to_numeric(frame["pnl"], errors="coerce")
        .fillna(0.0)
        .groupby(frame["market_hash_name"].astype(str))
        .sum()
    )
    positive = item_pnl[item_pnl > 0].sort_values(ascending=False)
    return {
        "top_1_item_pnl_share": _top_positive_share(positive, 1, total_pnl),
        "top_3_item_pnl_share": _top_positive_share(positive, 3, total_pnl),
    }


def _period_pnl(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if "test_period" in frame.columns:
        period = frame["test_period"].astype(str)
    else:
        period = frame["date"].dt.tz_convert(None).dt.to_period("M").astype(str)
    return pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0).groupby(period).sum()


def _top_positive_share(values: pd.Series, top_n: int, total_pnl: float) -> float:
    if values.empty or total_pnl <= 0:
        return np.nan
    positive = values[values > 0].sort_values(ascending=False)
    if positive.empty:
        return np.nan
    return float(positive.head(top_n).sum() / total_pnl)


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
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conservative paper-trade A/B review.")
    parser.add_argument(
        "--current-accepted-input",
        type=Path,
        default=DEFAULT_CURRENT_ACCEPTED_INPUT,
    )
    parser.add_argument("--strict-accepted-input", type=Path, default=DEFAULT_STRICT_ACCEPTED_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_paper_trade_ab_test(
        current_accepted_input=args.current_accepted_input,
        strict_accepted_input=args.strict_accepted_input,
        summary_output=args.summary_output,
        report_output=args.report_output,
    )
    print(f"Summary rows: {result.summary_rows}")
    print(f"Wrote summary: {_display_path(result.summary_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
