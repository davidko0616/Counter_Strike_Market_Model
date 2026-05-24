"""Probability calibration helpers for Strong Buy ranking models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class PlattCalibrator:
    """One-dimensional Platt calibrator with an identity fallback."""

    model: LogisticRegression | None = None

    def fit(
        self,
        scores: pd.Series | np.ndarray,
        labels: pd.Series | np.ndarray,
    ) -> PlattCalibrator:
        score_array = _clean_score_array(scores)
        label_array = np.asarray(labels, dtype=int)
        if len(score_array) == 0 or len(np.unique(label_array)) < 2:
            self.model = None
            return self

        self.model = LogisticRegression(max_iter=1_000)
        self.model.fit(_logit_transform(score_array).reshape(-1, 1), label_array)
        return self

    def predict(self, scores: pd.Series | np.ndarray) -> np.ndarray:
        score_array = _clean_score_array(scores)
        if self.model is None:
            return score_array
        return self.model.predict_proba(_logit_transform(score_array).reshape(-1, 1))[:, 1]


def expected_calibration_error(
    labels: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return binary expected calibration error for equal-width probability bins."""
    label_array = np.asarray(labels, dtype=int)
    score_array = _clean_score_array(scores)
    if len(label_array) == 0:
        return np.nan

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(score_array, bins, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        bin_weight = mask.mean()
        bin_accuracy = label_array[mask].mean()
        bin_confidence = score_array[mask].mean()
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(ece)


def binary_calibration_metrics(
    labels: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Return Brier score and ECE for Strong Buy probability scores."""
    label_array = np.asarray(labels, dtype=int)
    score_array = _clean_score_array(scores)
    if len(label_array) == 0:
        return {"brier_score": np.nan, "expected_calibration_error": np.nan}
    return {
        "brier_score": float(brier_score_loss(label_array, score_array)),
        "expected_calibration_error": expected_calibration_error(label_array, score_array),
    }


def time_ordered_calibration_split(
    train_rows: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    calibration_fraction: float = 0.2,
    min_calibration_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training rows into earlier fit rows and later calibration rows."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")

    rows = train_rows.sort_values([timestamp_col, "market_hash_name"]).reset_index(drop=True)
    calibration_size = max(min_calibration_rows, int(round(len(rows) * calibration_fraction)))
    if calibration_size >= len(rows):
        calibration_size = max(1, len(rows) // 5)
    split_at = len(rows) - calibration_size
    return rows.iloc[:split_at].copy(), rows.iloc[split_at:].copy()


def _clean_score_array(scores: pd.Series | np.ndarray) -> np.ndarray:
    score_array = pd.to_numeric(pd.Series(scores), errors="coerce").fillna(0.0).to_numpy(
        dtype=float
    )
    return np.clip(score_array, 1e-6, 1.0 - 1e-6)


def _logit_transform(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))
