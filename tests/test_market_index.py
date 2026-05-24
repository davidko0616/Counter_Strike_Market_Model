from __future__ import annotations

import pandas as pd
import numpy as np

from cs_market_model.features.market_index import add_market_index_features


def test_market_index_features_add_equal_weight_index_and_excess_returns() -> None:
    timestamps = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    rows = []
    for item, start, step in [("A", 100.0, 1.0), ("B", 200.0, -1.0)]:
        for day, timestamp in enumerate(timestamps):
            close = start + day * step
            rows.append(
                {
                    "market_hash_name": item,
                    "timestamp": timestamp,
                    "close": close,
                    "log_return_1d": pd.NA,
                    "log_return_30d": pd.NA,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["market_hash_name", "timestamp"])
    frame["log_return_1d"] = frame.groupby("market_hash_name")["close"].transform(
        lambda series: np.log(series / series.shift(1))
    )
    frame["log_return_30d"] = frame.groupby("market_hash_name")["close"].transform(
        lambda series: np.log(series / series.shift(30))
    )

    features = add_market_index_features(frame)

    assert "cs2_mvp_equal_weight_index" in features.columns
    assert "cs2_index_return_30d" in features.columns
    assert "item_excess_log_return_30d" in features.columns
    assert "item_beta_to_cs2_index_30d" in features.columns
    assert features["cs2_mvp_equal_weight_index"].notna().all()
