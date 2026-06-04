"""Train Day 10-11 LightGBM models with walk-forward validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from cs_market_model.backtesting.metrics import (
    evaluate_classification_predictions,
    summarize_metrics,
)
from cs_market_model.backtesting.walk_forward import make_monthly_walk_forward_splits
from cs_market_model.config import PROJECT_ROOT, data_path, load_yaml_config, reports_path
from cs_market_model.models.baselines import LABEL_STRONG_BUY
from cs_market_model.models.calibrate import (
    PlattCalibrator,
    binary_calibration_metrics,
    time_ordered_calibration_split,
)
from cs_market_model.models.train import (
    DEFAULT_FEATURES_INPUT,
    DEFAULT_LABELS_INPUT,
    _attach_split_metadata,
    _base_prediction_frame,
    build_modeling_dataset,
    load_feature_table,
    load_label_table,
)

DEFAULT_PREDICTIONS_OUTPUT = data_path("processed", "day10_11_lightgbm_predictions.parquet")
DEFAULT_METRICS_OUTPUT = reports_path("tables", "day10_11_lightgbm_metrics.csv")
DEFAULT_COMPARISON_OUTPUT = reports_path("tables", "day10_11_model_comparison.csv")
DEFAULT_IMPORTANCE_OUTPUT = reports_path("tables", "day10_11_feature_importance.csv")
DEFAULT_CALIBRATION_OUTPUT = reports_path("tables", "day10_11_calibration.csv")
DEFAULT_BASELINE_METRICS_INPUT = reports_path("tables", "day8_9_baseline_metrics.csv")


@dataclass(frozen=True)
class LightGBMTrainResult:
    """Paths and counts for a completed LightGBM training run."""

    predictions_output: Path
    metrics_output: Path
    comparison_output: Path
    importance_output: Path
    calibration_output: Path
    row_count: int
    split_count: int
    model_count: int
    feature_count: int


def train_day10_11_lightgbm(
    features_input: Path = DEFAULT_FEATURES_INPUT,
    labels_input: Path = DEFAULT_LABELS_INPUT,
    predictions_output: Path = DEFAULT_PREDICTIONS_OUTPUT,
    metrics_output: Path = DEFAULT_METRICS_OUTPUT,
    comparison_output: Path = DEFAULT_COMPARISON_OUTPUT,
    importance_output: Path = DEFAULT_IMPORTANCE_OUTPUT,
    calibration_output: Path = DEFAULT_CALIBRATION_OUTPUT,
    baseline_metrics_input: Path = DEFAULT_BASELINE_METRICS_INPUT,
    top_k: int | None = None,
    min_train_days: int = 60,
    embargo_days: int | None = None,
    min_train_rows: int = 100,
    min_test_rows: int = 20,
    random_state: int = 42,
) -> LightGBMTrainResult:
    """Train LightGBM models and compare them against existing baselines."""
    backtest_config = load_yaml_config("backtest.yaml")
    resolved_top_k = int(top_k if top_k is not None else backtest_config.get("top_k", 10))
    resolved_embargo_days = int(
        embargo_days if embargo_days is not None else backtest_config.get("embargo_days", 7)
    )

    features = load_feature_table(features_input)
    labels = load_label_table(labels_input)
    dataset, model_features = build_modeling_dataset(features, labels, complete_only=True)
    splits = make_monthly_walk_forward_splits(
        dataset,
        min_train_days=min_train_days,
        embargo_days=resolved_embargo_days,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )

    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(splits):
        train_rows = dataset.loc[split.train_indices].copy()
        test_rows = dataset.loc[split.test_indices].copy()
        active_features = _active_features(train_rows, model_features)

        for model_name, prediction_frame, importances in _lightgbm_predictions(
            train_rows,
            test_rows,
            active_features,
            random_state=random_state + split_index,
        ):
            attached = _attach_split_metadata(prediction_frame, split_index, split)
            attached["model_name"] = model_name
            prediction_frames.append(attached)
            importance_rows.extend(
                {
                    "split_index": split_index,
                    "split_id": split.split_id,
                    "model_name": model_name,
                    "feature": feature,
                    **importance,
                }
                for feature, importance in importances.items()
            )

        for (
            model_name,
            prediction_frame,
            importances,
            calibration_metric,
        ) in _calibrated_lightgbm_predictions(
            train_rows,
            test_rows,
            active_features,
            random_state=random_state + split_index,
        ):
            attached = _attach_split_metadata(prediction_frame, split_index, split)
            attached["model_name"] = model_name
            prediction_frames.append(attached)
            calibration_rows.append(
                {
                    "split_index": split_index,
                    "split_id": split.split_id,
                    "model_name": model_name,
                    **calibration_metric,
                }
            )
            importance_rows.extend(
                {
                    "split_index": split_index,
                    "split_id": split.split_id,
                    "model_name": model_name,
                    "feature": feature,
                    **importance,
                }
                for feature, importance in importances.items()
            )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    predictions = _append_lightgbm_rank_blend(predictions)
    metrics = _evaluate_predictions(predictions, splits, top_k=resolved_top_k)
    calibration = _build_calibration_table(predictions, calibration_rows)
    importance = pd.DataFrame(importance_rows)
    comparison = _build_model_comparison(metrics, baseline_metrics_input)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    importance_output.parent.mkdir(parents=True, exist_ok=True)
    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False)
    metrics.to_csv(metrics_output, index=False)
    comparison.to_csv(comparison_output, index=False)
    importance.to_csv(importance_output, index=False)
    calibration.to_csv(calibration_output, index=False)

    return LightGBMTrainResult(
        predictions_output=predictions_output,
        metrics_output=metrics_output,
        comparison_output=comparison_output,
        importance_output=importance_output,
        calibration_output=calibration_output,
        row_count=len(dataset),
        split_count=len(splits),
        model_count=int(metrics["model_name"].nunique()) if not metrics.empty else 0,
        feature_count=len(model_features),
    )


def _lightgbm_predictions(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    active_features: list[str],
    *,
    random_state: int,
) -> list[tuple[str, pd.DataFrame, dict[str, dict[str, float]]]]:
    predictions: list[tuple[str, pd.DataFrame, dict[str, dict[str, float]]]] = []
    if not active_features:
        return predictions

    x_train = train_rows[active_features]
    x_test = test_rows[active_features]

    multiclass_model = _make_multiclass_lightgbm(random_state=random_state)
    multiclass_model.fit(x_train, train_rows["label_code"].astype(int))
    multiclass_frame = _base_prediction_frame(test_rows)
    multiclass_frame["predicted_label_code"] = multiclass_model.predict(x_test).astype(int)
    multiclass_frame["strong_buy_score"] = _class_probability(
        multiclass_model,
        x_test,
        positive_class=LABEL_STRONG_BUY,
    )
    predictions.append(
        (
            "lightgbm_multiclass",
            multiclass_frame,
            _feature_importances(multiclass_model, active_features),
        )
    )

    binary_model = _make_binary_lightgbm(random_state=random_state)
    binary_target = (train_rows["label_code"].astype(int) == LABEL_STRONG_BUY).astype(int)
    if binary_target.nunique() >= 2:
        binary_model.fit(x_train, binary_target)
        binary_scores = _class_probability(binary_model, x_test, positive_class=1)
        binary_frame = _base_prediction_frame(test_rows)
        binary_frame["predicted_label_code"] = np.where(binary_scores >= 0.5, LABEL_STRONG_BUY, 0)
        binary_frame["strong_buy_score"] = binary_scores
        predictions.append(
            (
                "lightgbm_binary",
                binary_frame,
                _feature_importances(binary_model, active_features),
            )
        )

    return predictions


def _calibrated_lightgbm_predictions(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    active_features: list[str],
    *,
    random_state: int,
) -> list[tuple[str, pd.DataFrame, dict[str, dict[str, float]], dict[str, float]]]:
    if not active_features:
        return []

    fit_rows, calibration_rows = time_ordered_calibration_split(train_rows)
    fit_target = (fit_rows["label_code"].astype(int) == LABEL_STRONG_BUY).astype(int)
    calibration_target = (calibration_rows["label_code"].astype(int) == LABEL_STRONG_BUY).astype(
        int
    )
    if fit_target.nunique() < 2:
        return []

    model = _make_binary_lightgbm(random_state=random_state)
    model.fit(fit_rows[active_features], fit_target)
    raw_calibration_scores = _class_probability(
        model,
        calibration_rows[active_features],
        positive_class=1,
    )
    calibrator = PlattCalibrator().fit(raw_calibration_scores, calibration_target)
    raw_test_scores = _class_probability(model, test_rows[active_features], positive_class=1)
    calibrated_scores = calibrator.predict(raw_test_scores)

    prediction_frame = _base_prediction_frame(test_rows)
    prediction_frame["predicted_label_code"] = np.where(
        calibrated_scores >= 0.5,
        LABEL_STRONG_BUY,
        0,
    )
    prediction_frame["strong_buy_score"] = calibrated_scores

    calibration_metric = {
        "calibration_rows": len(calibration_rows),
        "calibration_positive_rate": float(calibration_target.mean()),
        **{
            f"calibration_{key}": value
            for key, value in binary_calibration_metrics(
                calibration_target,
                calibrator.predict(raw_calibration_scores),
            ).items()
        },
    }
    return [
        (
            "lightgbm_binary_platt",
            prediction_frame,
            _feature_importances(model, active_features),
            calibration_metric,
        )
    ]


def _evaluate_predictions(
    predictions: pd.DataFrame,
    splits: list[Any],
    *,
    top_k: int,
) -> pd.DataFrame:
    metric_rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()

    for (split_index, split_id, model_name), group in predictions.groupby(
        ["split_index", "split_id", "model_name"],
        sort=False,
    ):
        split = splits[int(split_index)]
        metric_rows.append(
            {
                "split_index": split_index,
                "split_id": split_id,
                "model_name": model_name,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "train_rows": len(split.train_indices),
                "test_rows": len(split.test_indices),
                **evaluate_classification_predictions(group, top_k=top_k),
            }
        )
    return pd.DataFrame(metric_rows)


def _build_calibration_table(
    predictions: pd.DataFrame,
    calibration_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = calibration_rows.copy()
    if not predictions.empty:
        for (split_index, split_id, model_name), group in predictions.groupby(
            ["split_index", "split_id", "model_name"],
            sort=False,
        ):
            actual_strong = group["label_code"].astype(int) == LABEL_STRONG_BUY
            rows.append(
                {
                    "split_index": split_index,
                    "split_id": split_id,
                    "model_name": model_name,
                    "test_rows": len(group),
                    "test_positive_rate": float(actual_strong.mean()),
                    **{
                        f"test_{key}": value
                        for key, value in binary_calibration_metrics(
                            actual_strong.astype(int),
                            group["strong_buy_score"],
                        ).items()
                    },
                }
            )
    return pd.DataFrame(rows)


def _build_model_comparison(
    lightgbm_metrics: pd.DataFrame,
    baseline_metrics_input: Path,
) -> pd.DataFrame:
    metric_frames = []
    if baseline_metrics_input.exists():
        baseline_metrics = pd.read_csv(baseline_metrics_input)
        baseline_metrics["model_group"] = "baseline"
        metric_frames.append(baseline_metrics)
    if not lightgbm_metrics.empty:
        lightgbm_copy = lightgbm_metrics.copy()
        lightgbm_copy["model_group"] = "lightgbm"
        metric_frames.append(lightgbm_copy)
    if not metric_frames:
        return pd.DataFrame()

    combined = pd.concat(metric_frames, ignore_index=True, sort=False)
    summary = summarize_metrics(combined)
    group_lookup = combined.groupby("model_name")["model_group"].first().to_dict()
    summary.insert(1, "model_group", summary["model_name"].map(group_lookup))
    return summary


def _append_lightgbm_rank_blend(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions

    blends: list[pd.DataFrame] = []
    for _, split_predictions in predictions.groupby("split_id", sort=False):
        blend = _rank_blend_for_split(
            split_predictions,
            model_names=["lightgbm_binary", "lightgbm_multiclass"],
            ensemble_name="lightgbm_rank_blend",
        )
        if blend is not None:
            blends.append(blend)
    if not blends:
        return predictions
    return pd.concat([predictions, *blends], ignore_index=True)


def _rank_blend_for_split(
    split_predictions: pd.DataFrame,
    *,
    model_names: list[str],
    ensemble_name: str,
) -> pd.DataFrame | None:
    model_frames = {
        model_name: rows.set_index("event_id")
        for model_name, rows in split_predictions.groupby("model_name", sort=False)
        if model_name in model_names
    }
    if set(model_names) - set(model_frames):
        return None

    score_frame = pd.concat(
        [
            model_frames[model_name]["strong_buy_score"].rename(model_name)
            for model_name in model_names
        ],
        axis=1,
        join="inner",
    )
    if score_frame.empty:
        return None

    blended_score = score_frame.rank(pct=True).mean(axis=1)
    base = model_frames[model_names[0]].loc[blended_score.index].copy()
    base["model_name"] = ensemble_name
    base["strong_buy_score"] = blended_score.to_numpy(dtype=float)
    threshold = float(blended_score.quantile(0.75))
    base["predicted_label_code"] = np.where(
        blended_score >= threshold,
        LABEL_STRONG_BUY,
        0,
    )
    return base.reset_index()


def _active_features(train_rows: pd.DataFrame, model_features: list[str]) -> list[str]:
    return [
        column
        for column in model_features
        if pd.to_numeric(train_rows[column], errors="coerce").notna().any()
    ]


def _make_multiclass_lightgbm(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )


def _make_binary_lightgbm(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )


def _class_probability(
    model: LGBMClassifier,
    features: pd.DataFrame,
    *,
    positive_class: int,
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.classes_
    matches = np.flatnonzero(classes == positive_class)
    if len(matches) == 0:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, int(matches[0])]


def _feature_importances(
    model: LGBMClassifier,
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    booster = model.booster_
    split_importance = booster.feature_importance(importance_type="split")
    gain_importance = booster.feature_importance(importance_type="gain")
    return {
        feature: {
            "importance_split": float(split),
            "importance_gain": float(gain),
        }
        for feature, split, gain in zip(
            feature_names,
            split_importance,
            gain_importance,
            strict=True,
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Day 10-11 LightGBM models.")
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--labels-input", type=Path, default=DEFAULT_LABELS_INPUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--importance-output", type=Path, default=DEFAULT_IMPORTANCE_OUTPUT)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION_OUTPUT)
    parser.add_argument(
        "--baseline-metrics-input",
        type=Path,
        default=DEFAULT_BASELINE_METRICS_INPUT,
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--embargo-days", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-test-rows", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    result = train_day10_11_lightgbm(
        features_input=args.features_input,
        labels_input=args.labels_input,
        predictions_output=args.predictions_output,
        metrics_output=args.metrics_output,
        comparison_output=args.comparison_output,
        importance_output=args.importance_output,
        calibration_output=args.calibration_output,
        baseline_metrics_input=args.baseline_metrics_input,
        top_k=args.top_k,
        min_train_days=args.min_train_days,
        embargo_days=args.embargo_days,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        random_state=args.random_state,
    )
    print(f"Modeling rows: {result.row_count}")
    print(f"Walk-forward splits: {result.split_count}")
    print(f"LightGBM models evaluated: {result.model_count}")
    print(f"Feature columns: {result.feature_count}")
    print(f"Wrote predictions: {_display_path(result.predictions_output)}")
    print(f"Wrote metrics: {_display_path(result.metrics_output)}")
    print(f"Wrote comparison: {_display_path(result.comparison_output)}")
    print(f"Wrote feature importance: {_display_path(result.importance_output)}")
    print(f"Wrote calibration: {_display_path(result.calibration_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
