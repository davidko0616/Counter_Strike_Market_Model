"""Portfolio simulation for ranked buy-signal predictions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.backtesting.execution import (
    ExecutionAssumptions,
    realized_trade_return,
)
from cs_market_model.config import PROJECT_ROOT, data_path, load_yaml_config, reports_path

DEFAULT_BASELINE_PREDICTIONS_INPUT = data_path(
    "processed",
    "day8_9_baseline_predictions.parquet",
)
DEFAULT_LIGHTGBM_PREDICTIONS_INPUT = data_path(
    "processed",
    "day10_11_lightgbm_predictions.parquet",
)
DEFAULT_TRADE_LEDGER_OUTPUT = reports_path("backtests", "day12_13_trade_ledger.csv")
DEFAULT_SUMMARY_OUTPUT = reports_path("tables", "day12_13_backtest_summary.csv")
DEFAULT_DAILY_EQUITY_OUTPUT = reports_path("backtests", "day12_13_daily_equity.csv")
DEFAULT_MODEL_NAMES = (
    "forest_rank_blend",
    "lightgbm_rank_blend",
    "momentum_rule_14d",
)
OPTIONAL_TRADE_COLUMNS = [
    "label_market_regime",
    "label_overlaps_known_market_event",
    "label_overlaps_covert_tradeup_event",
    "label_overlaps_souvenir_tradeup_event",
    "known_event_rows_in_label_window",
    "days_to_first_known_event_in_label_window",
    "cs2_index_regime_bull",
    "cs2_index_regime_neutral",
    "cs2_index_regime_bear",
    "cs2_index_return_7d",
    "cs2_index_return_30d",
    "cs2_index_drawdown_30d",
    "cs2_index_ma_distance_30d",
    "item_excess_log_return_7d",
    "item_excess_log_return_30d",
    "liquidity_quality_score_30d",
    "effective_staleness_days",
    "row_coverage_30d",
    "price_jump_50pct_share_30d",
    "price_jump_100pct_share_30d",
]


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """Config for converting model scores into simulated trades."""

    initial_capital: float = 1.0
    top_k: int = 10
    max_position_fraction: float = 0.05
    max_category_fraction: float = 0.25
    min_liquidity_score: float = 0.3
    same_item_cooldown_days: int = 14
    score_col: str = "strong_buy_score"
    return_col: str = "net_return_at_label"
    timestamp_col: str = "timestamp"
    exit_timestamp_col: str = "exit_timestamp"
    item_col: str = "market_hash_name"
    category_col: str = "category"
    liquidity_col: str = "row_coverage_30d"
    failed_execution_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be between 0 and 1")
        if not 0 < self.max_category_fraction <= 1:
            raise ValueError("max_category_fraction must be between 0 and 1")
        if self.same_item_cooldown_days < 0:
            raise ValueError("same_item_cooldown_days cannot be negative")


@dataclass(frozen=True)
class BacktestResult:
    """Output paths and row counts for Backtest V1."""

    trade_ledger_output: Path
    summary_output: Path
    daily_equity_output: Path
    trade_count: int
    model_count: int


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    """Load and combine prediction parquet files."""
    frames = []
    missing = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(path)
    if not frames:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"No prediction files found. Missing: {missing_text}")
    return pd.concat(frames, ignore_index=True, sort=False)


def config_from_yaml() -> PortfolioBacktestConfig:
    """Build portfolio config from ``configs/backtest.yaml``."""
    config = load_yaml_config("backtest.yaml")
    transaction_costs = config.get("transaction_costs", {}) or {}
    return PortfolioBacktestConfig(
        initial_capital=float(config.get("initial_capital", 1.0)),
        top_k=int(config.get("top_k", 10)),
        max_position_fraction=float(config.get("max_position_fraction", 0.05)),
        max_category_fraction=float(config.get("max_category_fraction", 0.25)),
        min_liquidity_score=float(config.get("min_liquidity_score", 0.3)),
        same_item_cooldown_days=int(
            config.get("same_item_cooldown_days", config.get("prediction_horizon_days", 14))
        ),
        failed_execution_probability=float(
            transaction_costs.get("failed_execution_probability", 0.0)
        ),
    )


def run_day12_13_backtest(
    prediction_inputs: list[Path] | None = None,
    trade_ledger_output: Path = DEFAULT_TRADE_LEDGER_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    daily_equity_output: Path = DEFAULT_DAILY_EQUITY_OUTPUT,
    model_names: list[str] | None = None,
    config: PortfolioBacktestConfig | None = None,
) -> BacktestResult:
    """Run Backtest V1 from model prediction artifacts."""
    predictions = load_predictions(
        prediction_inputs
        or [DEFAULT_BASELINE_PREDICTIONS_INPUT, DEFAULT_LIGHTGBM_PREDICTIONS_INPUT]
    )
    resolved_config = config or config_from_yaml()
    selected_model_names = model_names or _available_default_models(predictions)
    if selected_model_names:
        predictions = predictions[predictions["model_name"].isin(selected_model_names)].copy()

    ledger, daily_equity = simulate_ranked_portfolios(predictions, resolved_config)
    summary = summarize_backtest(ledger, daily_equity, resolved_config)

    trade_ledger_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    daily_equity_output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(trade_ledger_output, index=False)
    summary.to_csv(summary_output, index=False)
    daily_equity.to_csv(daily_equity_output, index=False)

    return BacktestResult(
        trade_ledger_output=trade_ledger_output,
        summary_output=summary_output,
        daily_equity_output=daily_equity_output,
        trade_count=len(ledger),
        model_count=int(ledger["model_name"].nunique()) if not ledger.empty else 0,
    )


def simulate_ranked_portfolios(
    predictions: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one independent portfolio per model."""
    resolved_config = config or PortfolioBacktestConfig()
    if predictions.empty:
        return _empty_ledger(), _empty_daily_equity()

    _validate_prediction_columns(predictions, resolved_config)
    frame = predictions.copy()
    frame[resolved_config.timestamp_col] = pd.to_datetime(
        frame[resolved_config.timestamp_col],
        utc=True,
    )
    frame[resolved_config.exit_timestamp_col] = pd.to_datetime(
        frame[resolved_config.exit_timestamp_col],
        utc=True,
    )
    frame = frame[
        frame[resolved_config.return_col].notna()
        & frame[resolved_config.exit_timestamp_col].notna()
    ].copy()

    ledgers = []
    daily_equity_frames = []
    for model_name, model_rows in frame.groupby("model_name", sort=False):
        ledger, equity = _simulate_one_model(str(model_name), model_rows, resolved_config)
        ledgers.append(ledger)
        daily_equity_frames.append(equity)

    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else _empty_ledger()
    daily_equity = (
        pd.concat(daily_equity_frames, ignore_index=True)
        if daily_equity_frames
        else _empty_daily_equity()
    )
    return ledger, daily_equity


def summarize_backtest(
    ledger: pd.DataFrame,
    daily_equity: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
) -> pd.DataFrame:
    """Summarize trade and equity metrics by model."""
    resolved_config = config or PortfolioBacktestConfig()
    if ledger.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "trade_count",
                "total_return",
                "average_trade_return",
                "win_rate",
                "max_drawdown",
                "sharpe_ratio",
                "sortino_ratio",
                "average_holding_period_days",
                "turnover",
                "precision_at_selected",
                "average_strong_buy_score",
                "peak_open_positions",
                "peak_open_exposure",
            ]
        )

    rows = []
    for model_name, model_trades in ledger.groupby("model_name", sort=False):
        model_equity = daily_equity[daily_equity["model_name"] == model_name].copy()
        daily_returns = (
            model_equity.sort_values("date")["equity"].pct_change().dropna()
            if not model_equity.empty
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "model_name": model_name,
                "trade_count": int(len(model_trades)),
                "total_return": _total_return(model_equity, resolved_config.initial_capital),
                "average_trade_return": float(model_trades["realized_net_return"].mean()),
                "win_rate": float((model_trades["realized_net_return"] > 0).mean()),
                "max_drawdown": _max_drawdown(model_equity),
                "sharpe_ratio": _annualized_sharpe(daily_returns),
                "sortino_ratio": _annualized_sortino(daily_returns),
                "average_holding_period_days": float(
                    model_trades["holding_period_days"].mean()
                ),
                "turnover": float(
                    model_trades["notional"].sum() / resolved_config.initial_capital
                ),
                "precision_at_selected": float(
                    model_trades["is_actual_strong_buy"].mean()
                ),
                "average_strong_buy_score": float(
                    model_trades["strong_buy_score"].mean()
                ),
                "peak_open_positions": int(model_equity["open_positions"].max())
                if not model_equity.empty
                else 0,
                "peak_open_exposure": float(model_equity["open_exposure"].max())
                if not model_equity.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["total_return", "precision_at_selected"],
        ascending=False,
    )


def _simulate_one_model(
    model_name: str,
    model_rows: pd.DataFrame,
    config: PortfolioBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = float(config.initial_capital)
    open_positions: list[dict[str, Any]] = []
    closed_trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_entry_by_item: dict[str, pd.Timestamp] = {}
    execution = ExecutionAssumptions(
        failed_execution_probability=config.failed_execution_probability,
        return_col=config.return_col,
    )

    model_rows = model_rows.sort_values(
        [config.timestamp_col, config.score_col],
        ascending=[True, False],
    ).copy()

    for entry_date, day_rows in model_rows.groupby(
        pd.Grouper(key=config.timestamp_col, freq="D"),
        sort=True,
    ):
        cash = _close_due_positions(
            open_positions,
            closed_trades,
            cash,
            pd.Timestamp(entry_date),
        )
        equity_before_entries = cash + sum(position["notional"] for position in open_positions)
        day_selected = 0
        for _, row in day_rows.sort_values(config.score_col, ascending=False).iterrows():
            if day_selected >= config.top_k:
                break
            if not _passes_optional_liquidity(row, config):
                continue
            item = str(row[config.item_col])
            if not _passes_item_cooldown(
                item,
                pd.Timestamp(row[config.timestamp_col]),
                last_entry_by_item,
                config.same_item_cooldown_days,
            ):
                continue
            trade_return = realized_trade_return(row, execution)
            if not np.isfinite(trade_return):
                continue

            current_equity = cash + sum(position["notional"] for position in open_positions)
            target_notional = current_equity * config.max_position_fraction
            if config.category_col in row.index:
                target_notional = min(
                    target_notional,
                    _remaining_category_capacity(
                        row,
                        open_positions,
                        current_equity,
                        config,
                    ),
                )
            notional = min(target_notional, cash)
            if notional <= 0:
                break

            cash -= notional
            entry_timestamp = pd.Timestamp(row[config.timestamp_col])
            exit_timestamp = pd.Timestamp(row[config.exit_timestamp_col])
            position = {
                "model_name": model_name,
                "split_id": row.get("split_id"),
                "event_id": row.get("event_id"),
                "market_hash_name": item,
                "category": row.get(config.category_col)
                if config.category_col in row.index
                else None,
                "entry_timestamp": entry_timestamp,
                "exit_timestamp": exit_timestamp,
                "holding_period_days": float(
                    row.get(
                        "holding_period_days",
                        max((exit_timestamp - entry_timestamp).days, 0),
                    )
                ),
                "strong_buy_score": float(row[config.score_col]),
                "daily_rank": day_selected + 1,
                "label_code": int(row["label_code"])
                if pd.notna(row.get("label_code"))
                else np.nan,
                "label_class": row.get("label_class"),
                "is_actual_strong_buy": bool(row.get("label_code") == 2),
                "realized_net_return": trade_return,
                "notional": float(notional),
                "pnl": float(notional * trade_return),
                "cash_after_entry": float(cash),
                "_proceeds": float(notional * (1.0 + trade_return)),
            }
            for column in OPTIONAL_TRADE_COLUMNS:
                if column in row.index:
                    position[column] = row.get(column)
            open_positions.append(position)
            last_entry_by_item[item] = entry_timestamp
            day_selected += 1

        equity_rows.append(
            _equity_row(
                model_name,
                pd.Timestamp(entry_date),
                cash,
                open_positions,
                equity_before_entries,
            )
        )

    cash = _close_due_positions(
        open_positions,
        closed_trades,
        cash,
        pd.Timestamp.max.tz_localize("UTC"),
    )
    if model_rows.empty:
        final_date = pd.Timestamp.utcnow().floor("D")
    else:
        final_date = max(
            model_rows[config.timestamp_col].max(),
            pd.Series([trade["exit_timestamp"] for trade in closed_trades]).max()
            if closed_trades
            else model_rows[config.timestamp_col].max(),
        )
    equity_rows.append(_equity_row(model_name, final_date, cash, open_positions, cash))

    ledger = pd.DataFrame(closed_trades)
    if not ledger.empty:
        ledger = ledger.drop(columns=["_proceeds"], errors="ignore")
    return ledger, pd.DataFrame(equity_rows)


def _close_due_positions(
    open_positions: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
    cash: float,
    current_date: pd.Timestamp,
) -> float:
    still_open = []
    for position in open_positions:
        if pd.Timestamp(position["exit_timestamp"]) <= current_date:
            proceeds = float(position["_proceeds"])
            position["cash_after_exit"] = float(cash + proceeds)
            closed_trades.append(position)
            cash += proceeds
        else:
            still_open.append(position)
    open_positions[:] = still_open
    return cash


def _equity_row(
    model_name: str,
    date: pd.Timestamp,
    cash: float,
    open_positions: list[dict[str, Any]],
    equity_before_entries: float,
) -> dict[str, Any]:
    open_exposure = float(sum(position["notional"] for position in open_positions))
    equity = float(cash + open_exposure)
    return {
        "model_name": model_name,
        "date": date,
        "cash": float(cash),
        "open_exposure": open_exposure,
        "open_positions": int(len(open_positions)),
        "equity": equity,
        "equity_before_entries": float(equity_before_entries),
    }


def _passes_optional_liquidity(row: pd.Series, config: PortfolioBacktestConfig) -> bool:
    if config.liquidity_col not in row.index:
        return True
    liquidity = pd.to_numeric(pd.Series([row.get(config.liquidity_col)]), errors="coerce").iloc[0]
    if not np.isfinite(liquidity):
        return False
    return float(liquidity) >= config.min_liquidity_score


def _passes_item_cooldown(
    item: str,
    timestamp: pd.Timestamp,
    last_entry_by_item: dict[str, pd.Timestamp],
    cooldown_days: int,
) -> bool:
    previous_entry = last_entry_by_item.get(item)
    if previous_entry is None:
        return True
    return (timestamp - previous_entry).days >= cooldown_days


def _remaining_category_capacity(
    row: pd.Series,
    open_positions: list[dict[str, Any]],
    current_equity: float,
    config: PortfolioBacktestConfig,
) -> float:
    category = row.get(config.category_col)
    if pd.isna(category):
        return current_equity * config.max_category_fraction
    current_category_exposure = sum(
        position["notional"]
        for position in open_positions
        if position.get("category") == category
    )
    return max(
        current_equity * config.max_category_fraction - current_category_exposure,
        0.0,
    )


def _validate_prediction_columns(
    predictions: pd.DataFrame,
    config: PortfolioBacktestConfig,
) -> None:
    required = {
        "model_name",
        config.item_col,
        config.timestamp_col,
        config.exit_timestamp_col,
        config.score_col,
        config.return_col,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required backtest columns: {sorted(missing)}")


def _available_default_models(predictions: pd.DataFrame) -> list[str]:
    available = set(predictions["model_name"].dropna().astype(str))
    selected = [model_name for model_name in DEFAULT_MODEL_NAMES if model_name in available]
    return selected or sorted(available)


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_name",
            "split_id",
            "event_id",
            "market_hash_name",
            "category",
            "entry_timestamp",
            "exit_timestamp",
            "holding_period_days",
            "strong_buy_score",
            "daily_rank",
            "label_code",
            "label_class",
            "is_actual_strong_buy",
            "realized_net_return",
            "notional",
            "pnl",
            "cash_after_entry",
            "cash_after_exit",
            *OPTIONAL_TRADE_COLUMNS,
        ]
    )


def _empty_daily_equity() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_name",
            "date",
            "cash",
            "open_exposure",
            "open_positions",
            "equity",
            "equity_before_entries",
        ]
    )


def _total_return(model_equity: pd.DataFrame, initial_capital: float) -> float:
    if model_equity.empty:
        return np.nan
    final_equity = float(model_equity.sort_values("date")["equity"].iloc[-1])
    return (final_equity / initial_capital) - 1.0


def _max_drawdown(model_equity: pd.DataFrame) -> float:
    if model_equity.empty:
        return np.nan
    equity = model_equity.sort_values("date")["equity"].astype(float)
    running_peak = equity.cummax()
    drawdown = equity.div(running_peak).sub(1.0)
    return float(drawdown.min())


def _annualized_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.empty or daily_returns.std(ddof=0) == 0:
        return np.nan
    return float(daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(365))


def _annualized_sortino(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return np.nan
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=0)
    if downside.empty or downside_std == 0:
        return np.nan
    return float(daily_returns.mean() / downside_std * np.sqrt(365))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 12-13 Backtest V1.")
    parser.add_argument(
        "--prediction-input",
        type=Path,
        action="append",
        dest="prediction_inputs",
        help="Prediction parquet input. May be provided multiple times.",
    )
    parser.add_argument("--trade-ledger-output", type=Path, default=DEFAULT_TRADE_LEDGER_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--daily-equity-output", type=Path, default=DEFAULT_DAILY_EQUITY_OUTPUT)
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        help="Model to backtest. May be provided multiple times.",
    )
    args = parser.parse_args()

    result = run_day12_13_backtest(
        prediction_inputs=args.prediction_inputs,
        trade_ledger_output=args.trade_ledger_output,
        summary_output=args.summary_output,
        daily_equity_output=args.daily_equity_output,
        model_names=args.model_names,
    )
    print(f"Backtest trades: {result.trade_count}")
    print(f"Models backtested: {result.model_count}")
    print(f"Wrote trade ledger: {_display_path(result.trade_ledger_output)}")
    print(f"Wrote summary: {_display_path(result.summary_output)}")
    print(f"Wrote daily equity: {_display_path(result.daily_equity_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

