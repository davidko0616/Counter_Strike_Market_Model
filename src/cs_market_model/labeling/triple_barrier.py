"""Net-return-aware Triple Barrier labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LABEL_TO_CODE = {
    "Bad Time": 0,
    "Good Time": 1,
    "Strong Buy": 2,
}


@dataclass(frozen=True)
class FeeModel:
    """Round-trip execution assumptions in basis points."""

    buy_fee_bps: float = 0.0
    sell_fee_bps: float = 250.0
    slippage_bps: float = 100.0

    @property
    def entry_multiplier(self) -> float:
        return 1.0 + ((self.buy_fee_bps + self.slippage_bps) / 10_000.0)

    @property
    def exit_multiplier(self) -> float:
        return 1.0 - ((self.sell_fee_bps + self.slippage_bps) / 10_000.0)

    @property
    def flat_round_trip_cost(self) -> float:
        return 1.0 - (self.exit_multiplier / self.entry_multiplier)


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for simplified daily net-return labels."""

    horizon_days: int = 14
    required_profit_bps: float = 300.0
    stop_loss_bps: float = 500.0
    volatility_stop_multiplier: float = 1.0
    min_row_coverage: float = 0.3
    max_staleness_days: float = 3.0
    volatility_col: str = "return_volatility_30d"

    @property
    def upper_barrier_net_return(self) -> float:
        return self.required_profit_bps / 10_000.0

    @property
    def min_stop_loss(self) -> float:
        return self.stop_loss_bps / 10_000.0


def apply_triple_barrier_labels(
    feature_table: pd.DataFrame,
    events: pd.DataFrame,
    fee_model: FeeModel | None = None,
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Apply buy-signal labels to sampled events using forward net returns."""
    resolved_fee_model = fee_model or FeeModel()
    resolved_config = config or TripleBarrierConfig()
    _validate_fee_model(resolved_fee_model)

    if events.empty:
        return _empty_label_table()

    required_features = {"market_hash_name", "timestamp", "close"}
    missing_features = required_features - set(feature_table.columns)
    if missing_features:
        raise ValueError(f"Missing required label feature columns: {sorted(missing_features)}")

    required_events = {"event_id", "market_hash_name", "timestamp"}
    missing_events = required_events - set(events.columns)
    if missing_events:
        raise ValueError(f"Missing required event columns: {sorted(missing_events)}")

    features = feature_table.sort_values(["market_hash_name", "timestamp"]).copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    event_frame = events.sort_values(["market_hash_name", "timestamp"]).copy()
    event_frame["timestamp"] = pd.to_datetime(event_frame["timestamp"], utc=True)

    item_groups = {
        market_hash_name: rows.reset_index(drop=True)
        for market_hash_name, rows in features.groupby("market_hash_name", sort=False)
    }

    labels = [
        _label_one_event(event, item_groups, resolved_fee_model, resolved_config)
        for event in event_frame.to_dict("records")
    ]
    return (
        pd.DataFrame(labels).sort_values(["market_hash_name", "timestamp"]).reset_index(drop=True)
    )


def _label_one_event(
    event: dict[str, object],
    item_groups: dict[str, pd.DataFrame],
    fee_model: FeeModel,
    config: TripleBarrierConfig,
) -> dict[str, object]:
    market_hash_name = str(event["market_hash_name"])
    event_timestamp = pd.Timestamp(event["timestamp"])
    item_rows = item_groups.get(market_hash_name)
    if item_rows is None:
        return _incomplete_label_row(
            event,
            event_timestamp,
            reason="missing_item_history",
            fee_model=fee_model,
            config=config,
        )

    item_timestamps = item_rows["timestamp"]
    event_position = int(item_timestamps.searchsorted(event_timestamp))
    if event_position >= len(item_rows) or item_timestamps.iloc[event_position] != event_timestamp:
        return _incomplete_label_row(
            event,
            event_timestamp,
            reason="missing_event_row",
            fee_model=fee_model,
            config=config,
        )

    event_row = item_rows.iloc[event_position]
    vertical_timestamp = event_timestamp + pd.Timedelta(days=config.horizon_days)
    future_stop_position = int(item_timestamps.searchsorted(vertical_timestamp, side="right"))
    future_rows = item_rows.iloc[event_position + 1 : future_stop_position]
    has_complete_horizon = item_timestamps.iloc[-1] >= vertical_timestamp

    base_row = _base_label_row(
        event=event,
        event_row=event_row,
        vertical_timestamp=vertical_timestamp,
        fee_model=fee_model,
        config=config,
    )

    if future_rows.empty or not has_complete_horizon:
        base_row.update(
            {
                "label_class": None,
                "label_code": np.nan,
                "label_reason": "incomplete_horizon",
                "is_label_complete": False,
            }
        )
        return base_row

    entry_price = float(event_row["close"])
    entry_cost = entry_price * fee_model.entry_multiplier
    future_net_returns = (
        future_rows["close"].to_numpy(dtype=float) * fee_model.exit_multiplier / entry_cost
    ) - 1.0
    vertical_row = future_rows.iloc[-1]
    vertical_net_return = float(future_net_returns[-1])
    volatility = _finite_or_default(
        event_row.get(config.volatility_col, event.get("event_volatility")),
        default=0.0,
    )
    lower_barrier = -(
        fee_model.flat_round_trip_cost
        + max(config.min_stop_loss, volatility * config.volatility_stop_multiplier)
    )
    upper_barrier = config.upper_barrier_net_return

    touch = _first_barrier_touch(future_net_returns, upper_barrier, lower_barrier)
    liquidity_ok = _passes_liquidity_filter(event_row, config)

    if not liquidity_ok:
        label_class = "Bad Time"
        label_reason = "liquidity_filter"
        label_timestamp = vertical_row["timestamp"]
        net_return_at_label = vertical_net_return
        first_touch = "vertical"
    elif touch is not None:
        touched_row = future_rows.iloc[int(touch["position"])]
        label_timestamp = touched_row["timestamp"]
        net_return_at_label = float(future_net_returns[int(touch["position"])])
        first_touch = str(touch["barrier"])
        if first_touch == "upper":
            label_class = "Strong Buy"
            label_reason = "upper_barrier"
        else:
            label_class = "Bad Time"
            label_reason = "lower_barrier"
    else:
        label_timestamp = vertical_row["timestamp"]
        net_return_at_label = vertical_net_return
        first_touch = "vertical"
        if net_return_at_label > 0:
            label_class = "Good Time"
            label_reason = "positive_vertical_return"
        else:
            label_class = "Bad Time"
            label_reason = "negative_vertical_return"

    base_row.update(
        {
            "label_class": label_class,
            "label_code": LABEL_TO_CODE[label_class],
            "label_reason": label_reason,
            "first_touch": first_touch,
            "touch_timestamp": label_timestamp,
            "exit_timestamp": label_timestamp,
            "holding_period_days": (pd.Timestamp(label_timestamp) - event_timestamp).days,
            "net_return_at_label": net_return_at_label,
            "vertical_net_return": vertical_net_return,
            "max_net_return": float(future_net_returns.max()),
            "min_net_return": float(future_net_returns.min()),
            "upper_barrier_net_return": upper_barrier,
            "lower_barrier_net_return": lower_barrier,
            "is_label_complete": True,
            "passes_liquidity_filter": liquidity_ok,
        }
    )
    return base_row


def _base_label_row(
    event: dict[str, object],
    event_row: pd.Series,
    vertical_timestamp: pd.Timestamp,
    fee_model: FeeModel,
    config: TripleBarrierConfig,
) -> dict[str, object]:
    row_coverage = _finite_or_default(event_row.get("row_coverage_30d"), default=np.nan)
    staleness_days = _finite_or_default(event_row.get("effective_staleness_days"), default=np.nan)
    return {
        "event_id": event["event_id"],
        "market_hash_name": event["market_hash_name"],
        "item_id": event_row.get("item_id"),
        "venue": event_row.get("venue"),
        "timestamp": event["timestamp"],
        "event_side": event.get("event_side"),
        "event_return": event.get("event_return"),
        "event_threshold": event.get("event_threshold"),
        "event_volatility": event.get("event_volatility"),
        "entry_price": float(event_row["close"]),
        "vertical_barrier_timestamp": vertical_timestamp,
        "touch_timestamp": pd.NaT,
        "exit_timestamp": pd.NaT,
        "holding_period_days": np.nan,
        "first_touch": None,
        "net_return_at_label": np.nan,
        "vertical_net_return": np.nan,
        "max_net_return": np.nan,
        "min_net_return": np.nan,
        "upper_barrier_net_return": config.upper_barrier_net_return,
        "lower_barrier_net_return": np.nan,
        "flat_round_trip_cost": fee_model.flat_round_trip_cost,
        "buy_fee_bps": fee_model.buy_fee_bps,
        "sell_fee_bps": fee_model.sell_fee_bps,
        "slippage_bps": fee_model.slippage_bps,
        "row_coverage_30d": row_coverage,
        "effective_staleness_days": staleness_days,
        "passes_liquidity_filter": _passes_liquidity_filter(event_row, config),
        "label_class": None,
        "label_code": np.nan,
        "label_reason": None,
        "is_label_complete": False,
    }


def _incomplete_label_row(
    event: dict[str, object],
    event_timestamp: pd.Timestamp,
    reason: str,
    fee_model: FeeModel,
    config: TripleBarrierConfig,
) -> dict[str, object]:
    vertical_timestamp = event_timestamp + pd.Timedelta(days=config.horizon_days)
    return {
        "event_id": event.get("event_id"),
        "market_hash_name": event.get("market_hash_name"),
        "item_id": None,
        "venue": None,
        "timestamp": event_timestamp,
        "event_side": event.get("event_side"),
        "event_return": event.get("event_return"),
        "event_threshold": event.get("event_threshold"),
        "event_volatility": event.get("event_volatility"),
        "entry_price": np.nan,
        "vertical_barrier_timestamp": vertical_timestamp,
        "touch_timestamp": pd.NaT,
        "exit_timestamp": pd.NaT,
        "holding_period_days": np.nan,
        "first_touch": None,
        "net_return_at_label": np.nan,
        "vertical_net_return": np.nan,
        "max_net_return": np.nan,
        "min_net_return": np.nan,
        "upper_barrier_net_return": config.upper_barrier_net_return,
        "lower_barrier_net_return": np.nan,
        "flat_round_trip_cost": fee_model.flat_round_trip_cost,
        "buy_fee_bps": fee_model.buy_fee_bps,
        "sell_fee_bps": fee_model.sell_fee_bps,
        "slippage_bps": fee_model.slippage_bps,
        "row_coverage_30d": np.nan,
        "effective_staleness_days": np.nan,
        "passes_liquidity_filter": False,
        "label_class": None,
        "label_code": np.nan,
        "label_reason": reason,
        "is_label_complete": False,
    }


def _first_barrier_touch(
    future_net_returns: np.ndarray,
    upper_barrier: float,
    lower_barrier: float,
) -> dict[str, object] | None:
    upper_hits = np.flatnonzero(future_net_returns >= upper_barrier)
    lower_hits = np.flatnonzero(future_net_returns <= lower_barrier)
    candidates: list[dict[str, object]] = []
    if len(upper_hits):
        candidates.append({"position": int(upper_hits[0]), "barrier": "upper"})
    if len(lower_hits):
        candidates.append({"position": int(lower_hits[0]), "barrier": "lower"})
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: int(candidate["position"]))


def _passes_liquidity_filter(event_row: pd.Series, config: TripleBarrierConfig) -> bool:
    row_coverage = _finite_or_default(event_row.get("row_coverage_30d"), default=1.0)
    staleness_days = _finite_or_default(event_row.get("effective_staleness_days"), default=0.0)
    return row_coverage >= config.min_row_coverage and staleness_days <= config.max_staleness_days


def _finite_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _validate_fee_model(fee_model: FeeModel) -> None:
    if fee_model.entry_multiplier <= 0:
        raise ValueError("Entry multiplier must be positive")
    if fee_model.exit_multiplier <= 0:
        raise ValueError("Exit multiplier must be positive")


def _empty_label_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "market_hash_name",
            "item_id",
            "venue",
            "timestamp",
            "event_side",
            "event_return",
            "event_threshold",
            "event_volatility",
            "entry_price",
            "vertical_barrier_timestamp",
            "touch_timestamp",
            "exit_timestamp",
            "holding_period_days",
            "first_touch",
            "net_return_at_label",
            "vertical_net_return",
            "max_net_return",
            "min_net_return",
            "upper_barrier_net_return",
            "lower_barrier_net_return",
            "flat_round_trip_cost",
            "buy_fee_bps",
            "sell_fee_bps",
            "slippage_bps",
            "row_coverage_30d",
            "effective_staleness_days",
            "passes_liquidity_filter",
            "label_class",
            "label_code",
            "label_reason",
            "is_label_complete",
        ]
    )
