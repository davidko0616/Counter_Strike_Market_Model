from __future__ import annotations

import numpy as np
import pandas as pd

from cs_market_model.models.calibrate import (
    PlattCalibrator,
    binary_calibration_metrics,
    expected_calibration_error,
    time_ordered_calibration_split,
)


def test_expected_calibration_error_is_zero_for_perfect_bins() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(labels, scores, n_bins=2) < 1e-5


def test_platt_calibrator_keeps_probabilities_in_unit_interval() -> None:
    calibrator = PlattCalibrator().fit(
        np.array([0.05, 0.2, 0.8, 0.95]),
        np.array([0, 0, 1, 1]),
    )

    calibrated = calibrator.predict(np.array([0.1, 0.5, 0.9]))

    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert calibrated[0] < calibrated[-1]


def test_binary_calibration_metrics_returns_brier_and_ece() -> None:
    metrics = binary_calibration_metrics(
        np.array([0, 1, 1]),
        np.array([0.1, 0.8, 0.9]),
    )

    assert {"brier_score", "expected_calibration_error"} <= set(metrics)
    assert metrics["brier_score"] >= 0.0


def test_time_ordered_calibration_split_uses_later_rows_for_calibration() -> None:
    rows = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test"] * 10,
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
        }
    )

    fit_rows, calibration_rows = time_ordered_calibration_split(
        rows,
        calibration_fraction=0.3,
        min_calibration_rows=2,
    )

    assert len(fit_rows) == 7
    assert len(calibration_rows) == 3
    assert fit_rows["timestamp"].max() < calibration_rows["timestamp"].min()
