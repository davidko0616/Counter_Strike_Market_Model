"""Audit extreme trades from Backtest V1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cs_market_model.backtesting.portfolio import (
    DEFAULT_BASELINE_PREDICTIONS_INPUT,
    DEFAULT_LIGHTGBM_PREDICTIONS_INPUT,
    DEFAULT_TRADE_LEDGER_OUTPUT,
    load_predictions,
)
from cs_market_model.config import CONFIG_DIR, PROJECT_ROOT, data_path, reports_path

DEFAULT_PRICE_BARS_INPUT = data_path("processed", "price_bars_day3_first_pull.parquet")
DEFAULT_MARKET_EVENTS_INPUT = CONFIG_DIR / "market_events.yaml"
DEFAULT_AUDIT_OUTPUT = reports_path("tables", "day14_outlier_audit.csv")
DEFAULT_IMPACT_OUTPUT = reports_path("tables", "day14_outlier_model_impact.csv")
DEFAULT_SENSITIVITY_OUTPUT = reports_path("tables", "day14_backtest_robustness.csv")
DEFAULT_REGIME_OUTPUT = reports_path("tables", "day14_backtest_regime_summary.csv")
DEFAULT_INDEX_REGIME_OUTPUT = reports_path("tables", "day14_backtest_index_regime_summary.csv")


@dataclass(frozen=True)
class OutlierAuditResult:
    """Output paths and row counts for an outlier audit run."""

    audit_output: Path
    impact_output: Path
    sensitivity_output: Path
    regime_output: Path
    index_regime_output: Path
    outlier_count: int
    model_count: int


def run_day14_outlier_audit(
    trade_ledger_input: Path = DEFAULT_TRADE_LEDGER_OUTPUT,
    prediction_inputs: list[Path] | None = None,
    price_bars_input: Path = DEFAULT_PRICE_BARS_INPUT,
    market_events_input: Path = DEFAULT_MARKET_EVENTS_INPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    impact_output: Path = DEFAULT_IMPACT_OUTPUT,
    sensitivity_output: Path = DEFAULT_SENSITIVITY_OUTPUT,
    regime_output: Path = DEFAULT_REGIME_OUTPUT,
    index_regime_output: Path = DEFAULT_INDEX_REGIME_OUTPUT,
    min_abs_return: float = 1.0,
    top_n: int = 50,
) -> OutlierAuditResult:
    """Create event-level and model-impact reports for extreme selected trades."""
    ledger = load_trade_ledger(trade_ledger_input)
    predictions = load_predictions(
        prediction_inputs
        or [DEFAULT_BASELINE_PREDICTIONS_INPUT, DEFAULT_LIGHTGBM_PREDICTIONS_INPUT]
    )
    event_outcomes = build_event_outcomes(predictions)
    audit = build_outlier_audit(
        ledger,
        event_outcomes,
        price_bars_input=price_bars_input,
        market_events_input=market_events_input,
        min_abs_return=min_abs_return,
        top_n=top_n,
    )
    impact = summarize_outlier_impact(ledger, audit)
    sensitivity = summarize_return_sensitivity(
        ledger,
        audit,
        market_events_input=market_events_input,
    )
    regime_summary = summarize_regime_performance(ledger)
    index_regime_summary = summarize_index_regime_performance(ledger)

    audit_output.parent.mkdir(parents=True, exist_ok=True)
    impact_output.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_output.parent.mkdir(parents=True, exist_ok=True)
    regime_output.parent.mkdir(parents=True, exist_ok=True)
    index_regime_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_output, index=False)
    impact.to_csv(impact_output, index=False)
    sensitivity.to_csv(sensitivity_output, index=False)
    regime_summary.to_csv(regime_output, index=False)
    index_regime_summary.to_csv(index_regime_output, index=False)

    return OutlierAuditResult(
        audit_output=audit_output,
        impact_output=impact_output,
        sensitivity_output=sensitivity_output,
        regime_output=regime_output,
        index_regime_output=index_regime_output,
        outlier_count=len(audit),
        model_count=int(impact["model_name"].nunique()) if not impact.empty else 0,
    )


def load_trade_ledger(path: Path = DEFAULT_TRADE_LEDGER_OUTPUT) -> pd.DataFrame:
    """Load the Backtest V1 trade ledger."""
    if not path.exists():
        raise FileNotFoundError(f"Trade ledger not found: {path}")
    return pd.read_csv(path)


def build_event_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicated model predictions into one realized outcome per event."""
    required = {
        "event_id",
        "market_hash_name",
        "timestamp",
        "exit_timestamp",
        "label_class",
        "label_code",
        "label_reason",
        "net_return_at_label",
        "vertical_net_return",
        "holding_period_days",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    outcomes = (
        predictions[list(required)]
        .drop_duplicates("event_id")
        .rename(columns={"timestamp": "entry_timestamp"})
        .copy()
    )
    outcomes["net_return_at_label"] = pd.to_numeric(
        outcomes["net_return_at_label"],
        errors="coerce",
    )
    outcomes["absolute_net_return"] = outcomes["net_return_at_label"].abs()
    finite_returns = outcomes["net_return_at_label"].replace([np.inf, -np.inf], np.nan)
    outcomes["event_return_percentile"] = finite_returns.rank(pct=True)
    outcomes["abs_event_return_percentile"] = outcomes["absolute_net_return"].rank(pct=True)
    return outcomes


def build_outlier_audit(
    ledger: pd.DataFrame,
    event_outcomes: pd.DataFrame,
    *,
    price_bars_input: Path = DEFAULT_PRICE_BARS_INPUT,
    market_events_input: Path = DEFAULT_MARKET_EVENTS_INPUT,
    min_abs_return: float = 1.0,
    top_n: int = 50,
) -> pd.DataFrame:
    """Build an event-level audit table for extreme selected trades."""
    if ledger.empty:
        return pd.DataFrame()

    selected_events = _selected_event_summary(ledger)
    audit = selected_events.merge(event_outcomes, on="event_id", how="left")
    audit = audit[
        audit["max_abs_selected_return"].ge(min_abs_return)
        | audit["absolute_net_return"].ge(min_abs_return)
    ].copy()
    audit = audit.sort_values("max_abs_selected_return", ascending=False).head(top_n)
    audit = _attach_price_bar_context(audit, price_bars_input)
    audit = _attach_known_event_context(audit, market_events_input)
    audit["audit_flags"] = audit.apply(_audit_flags, axis=1)
    return audit.reset_index(drop=True)


def summarize_outlier_impact(
    ledger: pd.DataFrame,
    outlier_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify how much selected outlier events affected each model."""
    if ledger.empty:
        return pd.DataFrame()

    outlier_events = set(outlier_audit["event_id"]) if not outlier_audit.empty else set()
    rows = []
    for model_name, model_trades in ledger.groupby("model_name", sort=False):
        pnl = pd.to_numeric(model_trades["pnl"], errors="coerce").fillna(0.0)
        outlier_mask = model_trades["event_id"].isin(outlier_events)
        outlier_pnl = pnl[outlier_mask].sum()
        total_pnl = pnl.sum()
        rows.append(
            {
                "model_name": model_name,
                "trade_count": int(len(model_trades)),
                "outlier_trade_count": int(outlier_mask.sum()),
                "total_pnl": float(total_pnl),
                "outlier_pnl": float(outlier_pnl),
                "pnl_without_outliers": float(total_pnl - outlier_pnl),
                "outlier_pnl_share": float(outlier_pnl / total_pnl) if total_pnl != 0 else np.nan,
                "max_selected_return": float(
                    pd.to_numeric(
                        model_trades["realized_net_return"],
                        errors="coerce",
                    ).max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_pnl", ascending=False)


def summarize_return_sensitivity(
    ledger: pd.DataFrame,
    outlier_audit: pd.DataFrame,
    *,
    market_events_input: Path = DEFAULT_MARKET_EVENTS_INPUT,
) -> pd.DataFrame:
    """Summarize model PnL under simple outlier-robust return scenarios."""
    if ledger.empty:
        return pd.DataFrame()

    outlier_events = set(outlier_audit["event_id"]) if not outlier_audit.empty else set()
    known_event_ids = (
        set(outlier_audit.loc[outlier_audit["known_event_id"].notna(), "event_id"])
        if "known_event_id" in outlier_audit.columns and not outlier_audit.empty
        else set()
    )
    if market_events_input.exists() and {"entry_timestamp", "market_hash_name"} <= set(
        ledger.columns
    ):
        known_event_ledger = _attach_known_event_context(ledger, market_events_input)
        known_event_ids |= set(
            known_event_ledger.loc[known_event_ledger["known_event_id"].notna(), "event_id"]
        )
    scenarios = [
        ("raw_all_regimes", None, set()),
        ("exclude_audited_extremes", None, outlier_events),
        ("normal_regime_excluding_known_events", None, known_event_ids),
        ("cap_all_returns_at_100pct", 1.0, set()),
        ("cap_all_returns_at_50pct", 0.5, set()),
    ]
    rows = []
    for scenario_name, cap, excluded_events in scenarios:
        scenario_ledger = ledger.copy()
        if excluded_events:
            scenario_ledger = scenario_ledger[
                ~scenario_ledger["event_id"].isin(excluded_events)
            ].copy()
        scenario_ledger["scenario_return"] = pd.to_numeric(
            scenario_ledger["realized_net_return"],
            errors="coerce",
        )
        if cap is not None:
            scenario_ledger["scenario_return"] = scenario_ledger["scenario_return"].clip(
                lower=-cap,
                upper=cap,
            )
        scenario_ledger["scenario_pnl"] = (
            pd.to_numeric(scenario_ledger["notional"], errors="coerce")
            * scenario_ledger["scenario_return"]
        )
        for model_name, model_trades in scenario_ledger.groupby("model_name", sort=False):
            rows.append(
                {
                    "scenario": scenario_name,
                    "model_name": model_name,
                    "trade_count": int(len(model_trades)),
                    "total_pnl": float(model_trades["scenario_pnl"].sum()),
                    "average_trade_return": float(model_trades["scenario_return"].mean()),
                    "win_rate": float((model_trades["scenario_return"] > 0).mean()),
                    "precision_at_selected": float(model_trades["is_actual_strong_buy"].mean())
                    if "is_actual_strong_buy" in model_trades.columns
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "total_pnl"], ascending=False)


def summarize_regime_performance(ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-trade performance by label regime."""
    if ledger.empty:
        return pd.DataFrame()

    frame = ledger.copy()
    if "label_market_regime" not in frame.columns:
        frame["label_market_regime"] = "unknown"
    frame["realized_net_return"] = pd.to_numeric(frame["realized_net_return"], errors="coerce")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce")
    rows = []
    for (model_name, regime), group in frame.groupby(
        ["model_name", "label_market_regime"],
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "model_name": model_name,
                "label_market_regime": regime,
                "trade_count": int(len(group)),
                "total_pnl": float(group["pnl"].sum()),
                "average_trade_return": float(group["realized_net_return"].mean()),
                "win_rate": float((group["realized_net_return"] > 0).mean()),
                "precision_at_selected": float(group["is_actual_strong_buy"].mean())
                if "is_actual_strong_buy" in group.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_name", "label_market_regime"],
        ascending=True,
    )


def summarize_index_regime_performance(ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-trade performance by CS2 index regime."""
    if ledger.empty:
        return pd.DataFrame()

    frame = ledger.copy()
    frame["cs2_index_regime"] = frame.apply(_index_regime_from_row, axis=1)
    frame["realized_net_return"] = pd.to_numeric(frame["realized_net_return"], errors="coerce")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce")
    if "label_market_regime" not in frame.columns:
        frame["label_market_regime"] = "unknown"

    rows = []
    for (model_name, label_regime, index_regime), group in frame.groupby(
        ["model_name", "label_market_regime", "cs2_index_regime"],
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "model_name": model_name,
                "label_market_regime": label_regime,
                "cs2_index_regime": index_regime,
                "trade_count": int(len(group)),
                "total_pnl": float(group["pnl"].sum()),
                "average_trade_return": float(group["realized_net_return"].mean()),
                "win_rate": float((group["realized_net_return"] > 0).mean()),
                "precision_at_selected": float(group["is_actual_strong_buy"].mean())
                if "is_actual_strong_buy" in group.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_name", "label_market_regime", "cs2_index_regime"],
        ascending=True,
    )


def _index_regime_from_row(row: pd.Series) -> str:
    if bool(row.get("cs2_index_regime_bull", False)):
        return "bull"
    if bool(row.get("cs2_index_regime_bear", False)):
        return "bear"
    if bool(row.get("cs2_index_regime_neutral", False)):
        return "neutral"
    return "unknown"


def _selected_event_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = ledger.copy()
    frame["realized_net_return"] = pd.to_numeric(frame["realized_net_return"], errors="coerce")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce")
    grouped = frame.groupby("event_id", sort=False)
    return grouped.agg(
        selected_by_models=("model_name", lambda values: ", ".join(sorted(set(values)))),
        selected_model_count=("model_name", "nunique"),
        selected_trade_count=("model_name", "size"),
        max_abs_selected_return=("realized_net_return", lambda values: values.abs().max()),
        max_selected_return=("realized_net_return", "max"),
        min_selected_return=("realized_net_return", "min"),
        selected_total_notional=("notional", "sum"),
        selected_total_pnl=("pnl", "sum"),
        best_selected_rank=("daily_rank", "min"),
        max_selected_score=("strong_buy_score", "max"),
    ).reset_index()


def _attach_price_bar_context(audit: pd.DataFrame, price_bars_input: Path) -> pd.DataFrame:
    if audit.empty or not price_bars_input.exists():
        audit = audit.copy()
        audit["price_bar_context_available"] = False
        audit["entry_close"] = np.nan
        audit["exit_close"] = np.nan
        audit["gross_close_return"] = np.nan
        return audit

    bars = pd.read_parquet(price_bars_input)
    required = {"market_hash_name", "timestamp", "close"}
    if required - set(bars.columns):
        return audit

    bars = bars[["market_hash_name", "timestamp", "close"]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    frame = audit.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True)
    frame = frame.merge(
        bars.rename(columns={"timestamp": "entry_timestamp", "close": "entry_close"}),
        on=["market_hash_name", "entry_timestamp"],
        how="left",
    )
    frame = frame.merge(
        bars.rename(columns={"timestamp": "exit_timestamp", "close": "exit_close"}),
        on=["market_hash_name", "exit_timestamp"],
        how="left",
    )
    frame["gross_close_return"] = frame["exit_close"] / frame["entry_close"] - 1.0
    frame["price_bar_context_available"] = (
        frame["entry_close"].notna() & frame["exit_close"].notna()
    )
    return frame


def _attach_known_event_context(audit: pd.DataFrame, market_events_input: Path) -> pd.DataFrame:
    frame = audit.copy()
    frame["known_event_id"] = pd.NA
    frame["known_event_name"] = pd.NA
    frame["known_event_action"] = pd.NA
    if frame.empty or not market_events_input.exists():
        return frame

    events = _load_market_events(market_events_input)
    if not events:
        return frame

    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    for event in events:
        start = pd.Timestamp(event["start_date"], tz="UTC")
        end = pd.Timestamp(event["end_date"], tz="UTC") + pd.Timedelta(days=1)
        if "exit_timestamp" in frame.columns:
            frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True)
            date_mask = frame["entry_timestamp"].lt(end) & frame["exit_timestamp"].ge(start)
        else:
            date_mask = frame["entry_timestamp"].ge(start) & frame["entry_timestamp"].lt(end)
        item_mask = frame["market_hash_name"].map(
            lambda item, event=event: _matches_event_query(str(item), event)
        )
        mask = date_mask & item_mask
        frame.loc[mask, "known_event_id"] = event["event_id"]
        frame.loc[mask, "known_event_name"] = event["name"]
        frame.loc[mask, "known_event_action"] = event["modeling_action"]
    return frame


def _load_market_events(path: Path) -> list[dict[str, object]]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    events = data.get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _matches_event_query(item_name: str, event: dict[str, object]) -> bool:
    query = event.get("affected_item_query", {})
    if not isinstance(query, dict):
        return True
    include_terms = [str(term) for term in query.get("include_terms", [])]
    exclude_terms = [str(term) for term in query.get("exclude_terms", [])]
    if include_terms and not any(term in item_name for term in include_terms):
        return False
    if exclude_terms and any(term in item_name for term in exclude_terms):
        return False
    return True


def _audit_flags(row: pd.Series) -> str:
    flags = []
    if row.get("max_abs_selected_return", 0) >= 1.0:
        flags.append("selected_return_ge_100pct")
    if row.get("abs_event_return_percentile", 0) >= 0.99:
        flags.append("top_1pct_abs_label_return")
    if row.get("selected_model_count", 0) > 1:
        flags.append("selected_by_multiple_models")
    if row.get("holding_period_days", np.inf) <= 1 and row.get("absolute_net_return", 0) >= 0.5:
        flags.append("large_one_day_label")
    if pd.notna(row.get("known_event_id")):
        flags.append("known_market_event")
    if not bool(row.get("price_bar_context_available", False)):
        flags.append("price_bars_missing_locally")
    return "|".join(flags)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Backtest V1 outlier trades.")
    parser.add_argument("--trade-ledger-input", type=Path, default=DEFAULT_TRADE_LEDGER_OUTPUT)
    parser.add_argument(
        "--prediction-input",
        type=Path,
        action="append",
        dest="prediction_inputs",
        help="Prediction parquet input. May be provided multiple times.",
    )
    parser.add_argument("--price-bars-input", type=Path, default=DEFAULT_PRICE_BARS_INPUT)
    parser.add_argument("--market-events-input", type=Path, default=DEFAULT_MARKET_EVENTS_INPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--impact-output", type=Path, default=DEFAULT_IMPACT_OUTPUT)
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        default=DEFAULT_SENSITIVITY_OUTPUT,
    )
    parser.add_argument("--regime-output", type=Path, default=DEFAULT_REGIME_OUTPUT)
    parser.add_argument(
        "--index-regime-output",
        type=Path,
        default=DEFAULT_INDEX_REGIME_OUTPUT,
    )
    parser.add_argument("--min-abs-return", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    result = run_day14_outlier_audit(
        trade_ledger_input=args.trade_ledger_input,
        prediction_inputs=args.prediction_inputs,
        price_bars_input=args.price_bars_input,
        market_events_input=args.market_events_input,
        audit_output=args.audit_output,
        impact_output=args.impact_output,
        sensitivity_output=args.sensitivity_output,
        regime_output=args.regime_output,
        index_regime_output=args.index_regime_output,
        min_abs_return=args.min_abs_return,
        top_n=args.top_n,
    )
    print(f"Outlier events audited: {result.outlier_count}")
    print(f"Models summarized: {result.model_count}")
    print(f"Wrote outlier audit: {_display_path(result.audit_output)}")
    print(f"Wrote model impact: {_display_path(result.impact_output)}")
    print(f"Wrote robustness sensitivity: {_display_path(result.sensitivity_output)}")
    print(f"Wrote regime summary: {_display_path(result.regime_output)}")
    print(f"Wrote index regime summary: {_display_path(result.index_regime_output)}")


if __name__ == "__main__":
    main()
