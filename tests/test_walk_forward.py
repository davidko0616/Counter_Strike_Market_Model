from __future__ import annotations

import pandas as pd

from cs_market_model.backtesting.walk_forward import make_monthly_walk_forward_splits


def test_walk_forward_splits_purge_overlaps_and_apply_embargo() -> None:
    timestamps = pd.date_range("2026-01-01", periods=90, freq="D", tz="UTC")
    labeled_events = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test"] * len(timestamps),
            "timestamp": timestamps,
            "vertical_barrier_timestamp": timestamps + pd.Timedelta(days=14),
        }
    )

    splits = make_monthly_walk_forward_splits(
        labeled_events,
        min_train_days=45,
        embargo_days=7,
        min_train_rows=10,
        min_test_rows=5,
    )

    assert splits
    first_split = splits[0]
    train_rows = labeled_events.loc[first_split.train_indices]
    test_start = first_split.test_start

    assert train_rows["timestamp"].max() <= test_start - pd.Timedelta(days=7)
    assert train_rows["vertical_barrier_timestamp"].max() < test_start
    assert labeled_events.loc[first_split.test_indices, "timestamp"].min() >= test_start
