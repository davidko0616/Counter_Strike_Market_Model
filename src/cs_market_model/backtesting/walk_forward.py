"""Walk-forward split helpers with purging and embargo."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """One expanding-window train/test split."""

    split_id: str
    train_indices: list[int]
    test_indices: list[int]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_monthly_walk_forward_splits(
    labeled_events: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    label_end_col: str = "vertical_barrier_timestamp",
    min_train_days: int = 60,
    embargo_days: int = 7,
    min_train_rows: int = 100,
    min_test_rows: int = 20,
) -> list[WalkForwardSplit]:
    """Create month-by-month expanding splits for event labels.

    Training rows are limited to events before the test month. Rows whose label windows
    overlap the test period are purged, and an additional embargo gap is applied before
    the test start.
    """
    required = {timestamp_col, label_end_col}
    missing = required - set(labeled_events.columns)
    if missing:
        raise ValueError(f"Missing required walk-forward columns: {sorted(missing)}")

    if labeled_events.empty:
        return []

    frame = labeled_events.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    frame[label_end_col] = pd.to_datetime(frame[label_end_col], utc=True)
    frame = frame.sort_values([timestamp_col, "market_hash_name"]).reset_index()

    first_timestamp = frame[timestamp_col].min()
    last_timestamp = frame[timestamp_col].max()
    earliest_test_start = first_timestamp + pd.Timedelta(days=min_train_days)
    first_month_start = (
        earliest_test_start.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    )
    last_month_start = (
        last_timestamp.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    )
    month_starts = pd.date_range(
        first_month_start,
        last_month_start,
        freq="MS",
    )

    splits: list[WalkForwardSplit] = []
    for month_start in month_starts:
        test_start = max(month_start, earliest_test_start)
        next_month = month_start + pd.offsets.MonthBegin(1)
        test_end = min(next_month - pd.Timedelta(microseconds=1), last_timestamp)
        if test_start > test_end:
            continue

        embargo_start = test_start - pd.Timedelta(days=embargo_days)
        train_mask = (frame[timestamp_col] <= embargo_start) & (frame[label_end_col] < test_start)
        test_mask = (frame[timestamp_col] >= test_start) & (frame[timestamp_col] <= test_end)

        train_rows = frame[train_mask]
        test_rows = frame[test_mask]
        if len(train_rows) < min_train_rows or len(test_rows) < min_test_rows:
            continue

        splits.append(
            WalkForwardSplit(
                split_id=f"wf_{len(splits):02d}_{test_start:%Y_%m}",
                train_indices=train_rows["index"].astype(int).tolist(),
                test_indices=test_rows["index"].astype(int).tolist(),
                train_start=train_rows[timestamp_col].min(),
                train_end=train_rows[timestamp_col].max(),
                test_start=test_start,
                test_end=test_rows[timestamp_col].max(),
            )
        )

    return splits
