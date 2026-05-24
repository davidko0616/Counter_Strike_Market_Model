from __future__ import annotations

import pandas as pd

from cs_market_model.backtesting.metrics import evaluate_classification_predictions
from cs_market_model.models.baselines import QuantileRuleBaseline, RuleBaselineConfig
from cs_market_model.models.train import (
    _append_rank_ensembles,
    _binary_predictions,
    build_modeling_dataset,
)


def test_build_modeling_dataset_filters_incomplete_labels_and_keeps_numeric_features() -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    features = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test"] * 3,
            "timestamp": timestamps,
            "log_return_14d": [0.01, 0.02, 0.03],
            "close_to_ma_30d": [-0.05, 0.01, 0.03],
            "has_volume": [False, False, False],
        }
    )
    labels = pd.DataFrame(
        {
            "event_id": ["event_000000", "event_000001", "event_000002"],
            "market_hash_name": ["AK-47 | Test"] * 3,
            "item_id": ["item_0"] * 3,
            "venue": ["steam"] * 3,
            "timestamp": timestamps,
            "vertical_barrier_timestamp": timestamps + pd.Timedelta(days=14),
            "exit_timestamp": timestamps + pd.Timedelta(days=7),
            "label_class": ["Bad Time", "Strong Buy", None],
            "label_code": [0, 2, pd.NA],
            "label_reason": ["lower_barrier", "upper_barrier", "incomplete_horizon"],
            "net_return_at_label": [-0.2, 0.1, pd.NA],
            "vertical_net_return": [-0.1, 0.05, pd.NA],
            "holding_period_days": [3, 5, pd.NA],
            "is_label_complete": [True, True, False],
        }
    )

    dataset, feature_columns = build_modeling_dataset(features, labels)

    assert len(dataset) == 2
    assert {"log_return_14d", "close_to_ma_30d", "has_volume"} <= set(feature_columns)


def test_quantile_rule_baseline_predicts_three_classes() -> None:
    train = pd.DataFrame({"score": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    test = pd.DataFrame({"score": [-2.0, 0.0, 2.0]})
    model = QuantileRuleBaseline(
        RuleBaselineConfig(model_name="test_rule", score_column="score")
    ).fit(train)

    assert model.predict(test).tolist() == [0, 1, 2]


def test_evaluate_predictions_reports_top_k_precision_and_return() -> None:
    predictions = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Test", "AWP | Test", "M4A4 | Test"],
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
            "label_code": [2, 0, 2],
            "predicted_label_code": [2, 2, 0],
            "strong_buy_score": [0.9, 0.8, 0.1],
            "net_return_at_label": [0.2, -0.1, 0.3],
        }
    )

    metrics = evaluate_classification_predictions(predictions, top_k=2)

    assert metrics["precision_at_k"] == 0.5
    assert metrics["daily_top1_precision"] == 2 / 3
    assert metrics["cooldown_precision_at_k"] == 0.5
    assert metrics["predicted_strong_buy_count"] == 2
    assert metrics["actual_strong_buy_count"] == 2


def test_binary_predictions_rank_strong_buy_probability() -> None:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    rows = pd.DataFrame(
        {
            "event_id": [f"event_{index:06d}" for index in range(8)],
            "market_hash_name": ["AK-47 | Test"] * 8,
            "timestamp": timestamps,
            "vertical_barrier_timestamp": timestamps + pd.Timedelta(days=14),
            "exit_timestamp": timestamps + pd.Timedelta(days=7),
            "label_class": ["Bad Time", "Bad Time", "Strong Buy", "Strong Buy"] * 2,
            "label_code": [0, 0, 2, 2] * 2,
            "label_reason": ["lower_barrier", "lower_barrier", "upper_barrier", "upper_barrier"]
            * 2,
            "net_return_at_label": [-0.2, -0.1, 0.1, 0.2] * 2,
            "vertical_net_return": [-0.1, -0.05, 0.05, 0.1] * 2,
            "holding_period_days": [3, 4, 5, 6] * 2,
            "log_return_14d": [-1.0, -0.8, 0.8, 1.0] * 2,
        }
    )

    predictions = _binary_predictions(
        rows.iloc[:6],
        rows.iloc[6:],
        ["log_return_14d"],
        random_state=7,
    )

    assert {model_name for model_name, _ in predictions} == {
        "binary_logistic_regression",
        "binary_random_forest",
    }
    for _, prediction_frame in predictions:
        assert prediction_frame["strong_buy_score"].between(0, 1).all()


def test_rank_blend_ensemble_is_appended_from_forest_scores() -> None:
    predictions = pd.DataFrame(
        {
            "split_id": ["wf_00"] * 6,
            "event_id": ["a", "b", "c"] * 2,
            "model_name": ["binary_random_forest"] * 3 + ["random_forest"] * 3,
            "label_code": [2, 0, 2] * 2,
            "predicted_label_code": [2, 0, 2] * 2,
            "strong_buy_score": [0.9, 0.2, 0.4, 0.8, 0.3, 0.5],
        }
    )

    with_ensemble = _append_rank_ensembles(predictions)
    ensemble = with_ensemble[with_ensemble["model_name"] == "forest_rank_blend"]

    assert len(ensemble) == 3
    assert ensemble["strong_buy_score"].between(0, 1).all()
