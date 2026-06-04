from __future__ import annotations

import numpy as np
import pandas as pd

from cs_market_model.models.explain import (
    build_signal_explanations,
    lightgbm_contribution_explanations,
    summarize_global_importance,
)


def test_signal_explanations_use_latest_scores_and_top_importance() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "market_hash_name": ["A", "B"],
            "model_name": ["lightgbm_rank_blend", "lightgbm_rank_blend"],
            "strong_buy_score": [0.9, 0.7],
        }
    )
    features = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "market_hash_name": ["A", "B"],
            "momentum_7d": [0.1, -0.2],
            "liquidity_quality_score_30d": [0.8, 0.5],
        }
    )
    importance = pd.DataFrame(
        {
            "model_name": ["lightgbm_binary", "lightgbm_binary"],
            "feature": ["momentum_7d", "liquidity_quality_score_30d"],
            "importance_gain": [10.0, 5.0],
            "importance_split": [3, 2],
        }
    )

    explanations = build_signal_explanations(
        predictions,
        features,
        importance,
        top_n=2,
    )

    assert len(explanations) == 4
    first = explanations.iloc[0]
    assert first["market_hash_name"] == "A"
    assert first["feature"] == "momentum_7d"
    assert first["feature_value"] == 0.1


def test_summarize_global_importance_falls_back_to_component_models() -> None:
    importance = pd.DataFrame(
        {
            "model_name": ["lightgbm_binary", "baseline"],
            "feature": ["a", "b"],
            "importance_gain": [2.0, 100.0],
            "importance_split": [1, 10],
        }
    )

    summary = summarize_global_importance(
        importance,
        model_name="lightgbm_rank_blend",
        top_n=1,
    )

    assert summary.iloc[0]["feature"] == "a"


class FakeContributionModel:
    def predict(self, features: pd.DataFrame, *, pred_contrib: bool) -> np.ndarray:
        assert pred_contrib is True
        assert list(features.columns) == ["a", "b"]
        return np.array([[0.1, -0.4, 0.0]])


def test_lightgbm_contribution_explanations_returns_top_native_contributions() -> None:
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})

    explanations = lightgbm_contribution_explanations(
        FakeContributionModel(),
        frame,
        ["a", "b"],
        top_n=1,
    )

    assert explanations.iloc[0]["feature"] == "b"
    assert explanations.iloc[0]["feature_contribution"] == -0.4
