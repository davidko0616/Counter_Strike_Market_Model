"""CUSUM event sampling for daily price bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CusumConfig:
    """Configuration for volatility-scaled CUSUM event sampling."""

    threshold_multiplier: float = 0.75
    min_threshold: float = 0.02
    min_gap_days: int = 1
    volatility_lookback_days: int = 30
    return_col: str = "log_return_1d"
    volatility_col: str = "return_volatility_30d"


def sample_cusum_events(
    feature_table: pd.DataFrame,
    config: CusumConfig | None = None,
) -> pd.DataFrame:
    """Sample item-date events when cumulative log returns breach a threshold.

    Thresholds are computed from prior volatility so future rows cannot change whether
    an earlier row becomes an event.
    """
    resolved_config = config or CusumConfig()
    required = {"market_hash_name", "timestamp", "close"}
    missing = required - set(feature_table.columns)
    if missing:
        raise ValueError(f"Missing required CUSUM columns: {sorted(missing)}")

    frame = feature_table.sort_values(["market_hash_name", "timestamp"]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    group = frame.groupby("market_hash_name", group_keys=False)

    if resolved_config.return_col in frame.columns:
        frame["event_sample_return"] = pd.to_numeric(
            frame[resolved_config.return_col], errors="coerce"
        )
    else:
        frame["event_sample_return"] = group["close"].transform(
            lambda series: np.log(series.astype(float)).diff()
        )

    if resolved_config.volatility_col in frame.columns:
        frame["event_sample_volatility"] = group[resolved_config.volatility_col].shift(1)
        frame["event_sample_volatility"] = pd.to_numeric(
            frame["event_sample_volatility"], errors="coerce"
        )
    else:
        frame["event_sample_volatility"] = group["event_sample_return"].transform(
            lambda series: (
                series.rolling(
                    resolved_config.volatility_lookback_days,
                    min_periods=resolved_config.volatility_lookback_days,
                )
                .std()
                .shift(1)
            )
        )

    thresholds = (
        frame["event_sample_volatility"].abs() * resolved_config.threshold_multiplier
    ).clip(lower=resolved_config.min_threshold)
    frame["event_sample_threshold"] = thresholds.fillna(resolved_config.min_threshold)

    events: list[dict[str, object]] = []
    for market_hash_name, item_rows in frame.groupby("market_hash_name", sort=False):
        positive_sum = 0.0
        negative_sum = 0.0
        last_event_timestamp: pd.Timestamp | None = None

        for row in item_rows.itertuples(index=False):
            log_return = row.event_sample_return
            if not np.isfinite(log_return):
                continue

            timestamp = row.timestamp
            threshold = float(row.event_sample_threshold)
            positive_sum = max(0.0, positive_sum + float(log_return))
            negative_sum = min(0.0, negative_sum + float(log_return))

            if positive_sum > threshold:
                event_side = 1
                event_return = positive_sum
            elif negative_sum < -threshold:
                event_side = -1
                event_return = negative_sum
            else:
                continue

            if (
                last_event_timestamp is not None
                and (timestamp - last_event_timestamp).days < resolved_config.min_gap_days
            ):
                continue

            events.append(
                {
                    "market_hash_name": market_hash_name,
                    "timestamp": timestamp,
                    "event_side": event_side,
                    "event_return": event_return,
                    "event_threshold": threshold,
                    "event_volatility": row.event_sample_volatility,
                }
            )
            positive_sum = 0.0
            negative_sum = 0.0
            last_event_timestamp = timestamp

    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "market_hash_name",
                "timestamp",
                "event_side",
                "event_return",
                "event_threshold",
                "event_volatility",
            ]
        )

    event_frame = event_frame.sort_values(["market_hash_name", "timestamp"]).reset_index(drop=True)
    event_frame.insert(0, "event_id", [f"event_{index:06d}" for index in event_frame.index])
    return event_frame
