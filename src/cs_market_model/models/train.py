"""Train Day 8-9 baseline models with walk-forward validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from cs_market_model.backtesting.metrics import (
    evaluate_classification_predictions,
    summarize_metrics,
)
from cs_market_model.backtesting.walk_forward import make_monthly_walk_forward_splits
from cs_market_model.config import PROJECT_ROOT, data_path, load_yaml_config, reports_path
from cs_market_model.features.build_features import feature_columns
from cs_market_model.models.baselines import (
    LABEL_STRONG_BUY,
    QuantileRuleBaseline,
    class_probability,
    default_rule_configs,
    make_logistic_regression,
    make_random_forest,
    strong_buy_probability,
)

DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_LABELS_INPUT = data_path("labels", "labels_day7_v1.parquet")
DEFAULT_PREDICTIONS_OUTPUT = data_path("processed", "day8_9_baseline_predictions.parquet")
DEFAULT_METRICS_OUTPUT = reports_path("tables", "day8_9_baseline_metrics.csv")
DEFAULT_SUMMARY_OUTPUT = reports_path("tables", "day8_9_baseline_summary.csv")

LABEL_COLUMNS = [
    "event_id",
    "market_hash_name",
    "item_id",
    "venue",
    "timestamp",
    "vertical_barrier_timestamp",
    "exit_timestamp",
    "label_class",
    "label_code",
    "label_reason",
    "net_return_at_label",
    "vertical_net_return",
    "holding_period_days",
    "is_label_complete",
]
OPTIONAL_LABEL_COLUMNS = [
    "label_market_regime",
    "label_overlaps_known_market_event",
    "label_overlaps_covert_tradeup_event",
    "label_overlaps_souvenir_tradeup_event",
    "known_event_rows_in_label_window",
    "days_to_first_known_event_in_label_window",
]
OPTIONAL_DIAGNOSTIC_FEATURE_COLUMNS = [
    "cs2_index_regime_bull",
    "cs2_index_regime_neutral",
    "cs2_index_regime_bear",
    "cs2_index_return_7d",
    "cs2_index_return_30d",
    "cs2_index_drawdown_30d",
    "cs2_index_ma_distance_30d",
    "item_excess_log_return_7d",
    "item_excess_log_return_30d",
    "liquidity_quality_score_30d",
]


@dataclass(frozen=True)
class BaselineTrainResult:
    """Paths and counts for a completed baseline training run."""

    predictions_output: Path
    metrics_output: Path
    summary_output: Path
    row_count: int
    split_count: int
    model_count: int
    feature_count: int


def load_feature_table(path: Path = DEFAULT_FEATURES_INPUT) -> pd.DataFrame:
    """Load Day 5 features."""
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    return pd.read_parquet(path)


def load_label_table(path: Path = DEFAULT_LABELS_INPUT) -> pd.DataFrame:
    """Load Day 7 labels."""
    if not path.exists():
        raise FileNotFoundError(f"Label table not found: {path}")
    return pd.read_parquet(path)


def build_modeling_dataset(
    feature_table: pd.DataFrame,
    label_table: pd.DataFrame,
    *,
    complete_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Join complete labels to point-in-time feature rows."""
    labels = label_table.copy()
    if complete_only:
        labels = labels[labels["is_label_complete"]].copy()
    labels = labels[pd.to_numeric(labels["label_code"], errors="coerce").notna()].copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)

    features = feature_table.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    candidate_features = _numeric_feature_columns(features)

    label_columns = [*LABEL_COLUMNS, *[column for column in OPTIONAL_LABEL_COLUMNS if column in labels]]
    dataset = labels[label_columns].merge(
        features[["market_hash_name", "timestamp", *candidate_features]],
        on=["market_hash_name", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    dataset["label_code"] = dataset["label_code"].astype(int)

    retained_features = [
        column
        for column in candidate_features
        if column in dataset.columns
        and pd.to_numeric(dataset[column], errors="coerce").notna().any()
    ]
    dataset[retained_features] = dataset[retained_features].replace(
        [np.inf, -np.inf], np.nan
    )
    return dataset.sort_values(["timestamp", "market_hash_name"]).reset_index(
        drop=True
    ), retained_features


def train_day8_9_baselines(
    features_input: Path = DEFAULT_FEATURES_INPUT,
    labels_input: Path = DEFAULT_LABELS_INPUT,
    predictions_output: Path = DEFAULT_PREDICTIONS_OUTPUT,
    metrics_output: Path = DEFAULT_METRICS_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    top_k: int | None = None,
    min_train_days: int = 60,
    embargo_days: int | None = None,
    min_train_rows: int = 100,
    min_test_rows: int = 20,
    random_state: int = 42,
    include_binary_models: bool = True,
) -> BaselineTrainResult:
    """Train and evaluate Day 8-9 baselines."""
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
    metric_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(splits):
        train_rows = dataset.loc[split.train_indices].copy()
        test_rows = dataset.loc[split.test_indices].copy()

        for prediction_frame in _rule_predictions(
            train_rows,
            test_rows,
            model_features,
        ):
            prediction_frames.append(
                _attach_split_metadata(prediction_frame, split_index, split)
            )

        if train_rows["label_code"].nunique() >= 2:
            for model_name, prediction_frame in _sklearn_predictions(
                train_rows,
                test_rows,
                model_features,
                random_state=random_state,
            ):
                prediction_frame["model_name"] = model_name
                prediction_frames.append(
                    _attach_split_metadata(prediction_frame, split_index, split)
                )

        if include_binary_models and (train_rows["label_code"] == LABEL_STRONG_BUY).nunique() >= 2:
            for model_name, prediction_frame in _binary_predictions(
                train_rows,
                test_rows,
                model_features,
                random_state=random_state,
            ):
                prediction_frame["model_name"] = model_name
                prediction_frames.append(
                    _attach_split_metadata(prediction_frame, split_index, split)
                )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    predictions = _append_rank_ensembles(predictions)
    if not predictions.empty:
        for (split_index, split_id, model_name), group in predictions.groupby(
            ["split_index", "split_id", "model_name"], sort=False
        ):
            split = splits[int(split_index)]
            metrics = evaluate_classification_predictions(group, top_k=resolved_top_k)
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
                    "feature_count": len(model_features),
                    **metrics,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False)
    metrics.to_csv(metrics_output, index=False)
    summary.to_csv(summary_output, index=False)

    return BaselineTrainResult(
        predictions_output=predictions_output,
        metrics_output=metrics_output,
        summary_output=summary_output,
        row_count=len(dataset),
        split_count=len(splits),
        model_count=int(metrics["model_name"].nunique()) if not metrics.empty else 0,
        feature_count=len(model_features),
    )


def _numeric_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in feature_columns(feature_table):
        if is_numeric_dtype(feature_table[column]) or is_bool_dtype(feature_table[column]):
            columns.append(column)
    return columns


def _rule_predictions(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    model_features: list[str],
) -> list[pd.DataFrame]:
    predictions = []
    for config in default_rule_configs():
        if config.score_column not in model_features:
            continue
        model = QuantileRuleBaseline(config).fit(train_rows)
        predicted_labels = model.predict(test_rows)
        strong_buy_score = model.score(test_rows)
        prediction_frame = _base_prediction_frame(test_rows)
        prediction_frame["model_name"] = config.model_name
        prediction_frame["predicted_label_code"] = predicted_labels
        prediction_frame["strong_buy_score"] = strong_buy_score
        predictions.append(prediction_frame)
    return predictions


def _sklearn_predictions(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    model_features: list[str],
    *,
    random_state: int,
) -> list[tuple[str, pd.DataFrame]]:
    active_features = [
        column
        for column in model_features
        if pd.to_numeric(train_rows[column], errors="coerce").notna().any()
    ]
    x_train = train_rows[active_features]
    y_train = train_rows["label_code"].astype(int)
    x_test = test_rows[active_features]

    model_factories = {
        "logistic_regression": make_logistic_regression,
        "random_forest": make_random_forest,
    }
    predictions: list[tuple[str, pd.DataFrame]] = []
    for model_name, factory in model_factories.items():
        model = factory(random_state=random_state)
        model.fit(x_train, y_train)
        prediction_frame = _base_prediction_frame(test_rows)
        prediction_frame["predicted_label_code"] = model.predict(x_test).astype(int)
        prediction_frame["strong_buy_score"] = strong_buy_probability(model, x_test)
        predictions.append((model_name, prediction_frame))
    return predictions


def _binary_predictions(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    model_features: list[str],
    *,
    random_state: int,
) -> list[tuple[str, pd.DataFrame]]:
    active_features = [
        column
        for column in model_features
        if pd.to_numeric(train_rows[column], errors="coerce").notna().any()
    ]
    x_train = train_rows[active_features]
    y_train = (train_rows["label_code"].astype(int) == LABEL_STRONG_BUY).astype(int)
    x_test = test_rows[active_features]

    model_factories = {
        "binary_logistic_regression": make_logistic_regression,
        "binary_random_forest": make_random_forest,
    }
    predictions: list[tuple[str, pd.DataFrame]] = []
    for model_name, factory in model_factories.items():
        model = factory(random_state=random_state)
        model.fit(x_train, y_train)
        predicted_strong = model.predict(x_test).astype(int)
        prediction_frame = _base_prediction_frame(test_rows)
        prediction_frame["predicted_label_code"] = np.where(
            predicted_strong == 1,
            LABEL_STRONG_BUY,
            0,
        )
        prediction_frame["strong_buy_score"] = class_probability(
            model,
            x_test,
            positive_class=1,
        )
        predictions.append((model_name, prediction_frame))
    return predictions


def _base_prediction_frame(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id",
        "market_hash_name",
        "timestamp",
        "vertical_barrier_timestamp",
        "exit_timestamp",
        "label_class",
        "label_code",
        "label_reason",
        "net_return_at_label",
        "vertical_net_return",
        "holding_period_days",
    ]
    columns.extend(column for column in OPTIONAL_LABEL_COLUMNS if column in rows.columns)
    columns.extend(
        column for column in OPTIONAL_DIAGNOSTIC_FEATURE_COLUMNS if column in rows.columns
    )
    return rows[columns].copy()


def _attach_split_metadata(
    predictions: pd.DataFrame,
    split_index: int,
    split: Any,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame.insert(0, "split_index", split_index)
    frame.insert(1, "split_id", split.split_id)
    frame["train_start"] = split.train_start
    frame["train_end"] = split.train_end
    frame["test_start"] = split.test_start
    frame["test_end"] = split.test_end
    frame["is_actual_strong_buy"] = frame["label_code"].astype(int) == LABEL_STRONG_BUY
    return frame


def _append_rank_ensembles(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions

    ensembles: list[pd.DataFrame] = []
    for _, split_predictions in predictions.groupby("split_id", sort=False):
        blend = _rank_blend_for_split(
            split_predictions,
            model_names=["binary_random_forest", "random_forest"],
            ensemble_name="forest_rank_blend",
        )
        if blend is not None:
            ensembles.append(blend)

    if not ensembles:
        return predictions
    return pd.concat([predictions, *ensembles], ignore_index=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Day 8-9 baseline models.")
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--labels-input", type=Path, default=DEFAULT_LABELS_INPUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--embargo-days", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-test-rows", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-binary-models", action="store_true")
    args = parser.parse_args()

    result = train_day8_9_baselines(
        features_input=args.features_input,
        labels_input=args.labels_input,
        predictions_output=args.predictions_output,
        metrics_output=args.metrics_output,
        summary_output=args.summary_output,
        top_k=args.top_k,
        min_train_days=args.min_train_days,
        embargo_days=args.embargo_days,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        random_state=args.random_state,
        include_binary_models=not args.no_binary_models,
    )
    print(f"Modeling rows: {result.row_count}")
    print(f"Walk-forward splits: {result.split_count}")
    print(f"Models evaluated: {result.model_count}")
    print(f"Feature columns: {result.feature_count}")
    print(f"Wrote predictions: {_display_path(result.predictions_output)}")
    print(f"Wrote metrics: {_display_path(result.metrics_output)}")
    print(f"Wrote summary: {_display_path(result.summary_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
