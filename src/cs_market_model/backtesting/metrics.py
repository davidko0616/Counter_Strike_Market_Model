"""Baseline model evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

STRONG_BUY_CODE = 2


def evaluate_classification_predictions(
    predictions: pd.DataFrame,
    *,
    top_k: int = 10,
    cooldown_days: int = 14,
    label_col: str = "label_code",
    prediction_col: str = "predicted_label_code",
    score_col: str = "strong_buy_score",
    return_col: str = "net_return_at_label",
    timestamp_col: str = "timestamp",
    item_col: str = "market_hash_name",
) -> dict[str, Any]:
    """Evaluate multiclass labels and Strong Buy ranking quality."""
    if predictions.empty:
        return {
            "row_count": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "strong_buy_precision": np.nan,
            "strong_buy_recall": np.nan,
            "precision_at_k": np.nan,
            "top_k_mean_net_return": np.nan,
            "cooldown_precision_at_k": np.nan,
            "cooldown_top_k_mean_net_return": np.nan,
            "daily_top1_precision": np.nan,
            "daily_top1_mean_net_return": np.nan,
            "daily_top3_precision": np.nan,
            "daily_top3_mean_net_return": np.nan,
            "strong_buy_prevalence": np.nan,
            "strong_buy_average_precision": np.nan,
            "strong_buy_roc_auc": np.nan,
            "predicted_strong_buy_count": 0,
            "actual_strong_buy_count": 0,
        }

    y_true = predictions[label_col].astype(int)
    y_pred = predictions[prediction_col].astype(int)
    actual_strong = y_true == STRONG_BUY_CODE
    predicted_strong = y_pred == STRONG_BUY_CODE

    ranked = predictions.sort_values(score_col, ascending=False).head(top_k)
    top_k_true = ranked[label_col].astype(int) == STRONG_BUY_CODE
    top_k_returns = pd.to_numeric(ranked[return_col], errors="coerce")
    cooldown_ranked = _cooldown_top_k(
        predictions,
        top_k=top_k,
        cooldown_days=cooldown_days,
        timestamp_col=timestamp_col,
        item_col=item_col,
        score_col=score_col,
    )
    cooldown_true = cooldown_ranked[label_col].astype(int) == STRONG_BUY_CODE
    cooldown_returns = pd.to_numeric(cooldown_ranked[return_col], errors="coerce")
    daily_top1 = _daily_top_k_metrics(
        predictions,
        top_k=1,
        label_col=label_col,
        score_col=score_col,
        return_col=return_col,
        timestamp_col=timestamp_col,
    )
    daily_top3 = _daily_top_k_metrics(
        predictions,
        top_k=3,
        label_col=label_col,
        score_col=score_col,
        return_col=return_col,
        timestamp_col=timestamp_col,
    )

    return {
        "row_count": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _balanced_accuracy_without_warnings(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "strong_buy_precision": float(
            precision_score(
                actual_strong,
                predicted_strong,
                zero_division=0,
            )
        ),
        "strong_buy_recall": float(
            recall_score(actual_strong, predicted_strong, zero_division=0)
        ),
        "precision_at_k": float(top_k_true.mean()) if len(ranked) else np.nan,
        "top_k_mean_net_return": float(top_k_returns.mean()) if len(ranked) else np.nan,
        "cooldown_precision_at_k": float(cooldown_true.mean())
        if len(cooldown_ranked)
        else np.nan,
        "cooldown_top_k_mean_net_return": float(cooldown_returns.mean())
        if len(cooldown_ranked)
        else np.nan,
        "daily_top1_precision": daily_top1["precision"],
        "daily_top1_mean_net_return": daily_top1["mean_net_return"],
        "daily_top3_precision": daily_top3["precision"],
        "daily_top3_mean_net_return": daily_top3["mean_net_return"],
        "strong_buy_prevalence": float(actual_strong.mean()),
        "strong_buy_average_precision": _average_precision_without_warnings(
            actual_strong,
            predictions[score_col],
        ),
        "strong_buy_roc_auc": _roc_auc_without_warnings(
            actual_strong,
            predictions[score_col],
        ),
        "predicted_strong_buy_count": int(predicted_strong.sum()),
        "actual_strong_buy_count": int(actual_strong.sum()),
    }


def _balanced_accuracy_without_warnings(y_true: pd.Series, y_pred: pd.Series) -> float:
    recalls = []
    for class_code in sorted(y_true.unique()):
        class_mask = y_true == class_code
        recalls.append(float((y_pred[class_mask] == class_code).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def _average_precision_without_warnings(y_true: pd.Series, score: pd.Series) -> float:
    if y_true.nunique() < 2:
        return np.nan
    numeric_score = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return float(average_precision_score(y_true, numeric_score))


def _roc_auc_without_warnings(y_true: pd.Series, score: pd.Series) -> float:
    if y_true.nunique() < 2:
        return np.nan
    numeric_score = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return float(roc_auc_score(y_true, numeric_score))


def _daily_top_k_metrics(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    label_col: str,
    score_col: str,
    return_col: str,
    timestamp_col: str,
) -> dict[str, float]:
    if predictions.empty:
        return {"precision": np.nan, "mean_net_return": np.nan}

    frame = predictions.copy()
    frame["_date"] = pd.to_datetime(frame[timestamp_col], utc=True).dt.floor("D")
    precisions = []
    returns = []
    for _, day_rows in frame.groupby("_date", sort=False):
        top_rows = day_rows.sort_values(score_col, ascending=False).head(top_k)
        if top_rows.empty:
            continue
        precisions.append(float((top_rows[label_col].astype(int) == STRONG_BUY_CODE).mean()))
        returns.append(float(pd.to_numeric(top_rows[return_col], errors="coerce").mean()))

    return {
        "precision": float(np.mean(precisions)) if precisions else np.nan,
        "mean_net_return": float(np.mean(returns)) if returns else np.nan,
    }


def _cooldown_top_k(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    cooldown_days: int,
    timestamp_col: str,
    item_col: str,
    score_col: str,
) -> pd.DataFrame:
    selected_rows = []
    selected_timestamps_by_item: dict[str, pd.Timestamp] = {}
    ranked = predictions.sort_values(score_col, ascending=False)
    for _, row in ranked.iterrows():
        item = str(row[item_col])
        timestamp = pd.Timestamp(row[timestamp_col])
        previous_timestamp = selected_timestamps_by_item.get(item)
        if (
            previous_timestamp is not None
            and abs((timestamp - previous_timestamp).days) < cooldown_days
        ):
            continue
        selected_rows.append(row)
        selected_timestamps_by_item[item] = timestamp
        if len(selected_rows) >= top_k:
            break

    if not selected_rows:
        return predictions.iloc[0:0].copy()
    return pd.DataFrame(selected_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level metrics by model."""
    if metrics.empty:
        return pd.DataFrame()

    numeric_columns = [
        column
        for column in metrics.select_dtypes(include=["number", "bool"]).columns
        if column != "split_index"
    ]
    summary = (
        metrics.groupby("model_name")[numeric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary
