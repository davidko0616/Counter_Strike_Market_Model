"""Signal explanation utilities for LightGBM research artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path

DEFAULT_PREDICTIONS_INPUT = data_path("processed", "day10_11_lightgbm_predictions.parquet")
DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_IMPORTANCE_INPUT = reports_path("tables", "day10_11_feature_importance.csv")
DEFAULT_EXPLANATIONS_OUTPUT = reports_path("tables", "signal_explanations.csv")
DEFAULT_MODEL_NAME = "lightgbm_rank_blend"
LIGHTGBM_COMPONENT_MODELS = (
    "lightgbm_binary",
    "lightgbm_multiclass",
    "lightgbm_binary_platt",
)


@dataclass(frozen=True)
class ExplanationResult:
    """Output metadata for a signal-explanation build."""

    output_path: Path
    row_count: int
    signal_count: int


def build_signal_explanations(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    importance: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    top_n: int = 5,
    max_signals: int = 200,
) -> pd.DataFrame:
    """Build top-N global feature explanations for the latest scored signals."""
    if predictions.empty or features.empty or importance.empty:
        return _empty_explanations()

    latest_predictions = _latest_predictions(predictions, model_name, max_signals=max_signals)
    latest_features = _latest_feature_rows(features)
    importance_summary = summarize_global_importance(importance, model_name=model_name, top_n=top_n)
    if latest_predictions.empty or latest_features.empty or importance_summary.empty:
        return _empty_explanations()

    merged = latest_predictions.merge(
        latest_features,
        on="market_hash_name",
        how="left",
        suffixes=("", "_feature"),
    )
    rows: list[dict[str, Any]] = []
    for _, signal in merged.iterrows():
        for rank, feature_row in enumerate(importance_summary.itertuples(index=False), start=1):
            feature = str(feature_row.feature)
            rows.append(
                {
                    "timestamp": signal.get("timestamp"),
                    "market_hash_name": signal.get("market_hash_name"),
                    "model_name": model_name,
                    "strong_buy_score": signal.get("strong_buy_score"),
                    "explanation_rank": rank,
                    "feature": feature,
                    "feature_value": signal.get(feature, np.nan),
                    "importance_gain": feature_row.importance_gain,
                    "importance_split": feature_row.importance_split,
                    "explanation_method": "global_lightgbm_importance",
                }
            )
    return pd.DataFrame(rows)


def summarize_global_importance(
    importance: pd.DataFrame,
    *,
    model_name: str,
    top_n: int,
) -> pd.DataFrame:
    """Summarize LightGBM feature importance for one model or its component models."""
    if importance.empty or "feature" not in importance.columns:
        return pd.DataFrame(columns=["feature", "importance_gain", "importance_split"])
    frame = importance.copy()
    if "model_name" in frame.columns:
        direct = frame[frame["model_name"].astype(str).eq(model_name)].copy()
        if direct.empty and model_name == DEFAULT_MODEL_NAME:
            direct = frame[frame["model_name"].astype(str).isin(LIGHTGBM_COMPONENT_MODELS)].copy()
        frame = direct if not direct.empty else frame

    for column in ("importance_gain", "importance_split"):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    summary = frame.groupby("feature", as_index=False).agg(
        importance_gain=("importance_gain", "mean"),
        importance_split=("importance_split", "mean"),
    )
    summary = summary.sort_values(
        ["importance_gain", "importance_split", "feature"],
        ascending=[False, False, True],
    )
    return summary.head(top_n).reset_index(drop=True)


def lightgbm_contribution_explanations(
    model: Any,
    feature_frame: pd.DataFrame,
    feature_names: list[str],
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return per-row LightGBM native contribution explanations when a model exists."""
    if feature_frame.empty or not feature_names:
        return pd.DataFrame(
            columns=["row_index", "explanation_rank", "feature", "feature_contribution"]
        )
    missing = [feature for feature in feature_names if feature not in feature_frame.columns]
    if missing:
        raise ValueError(f"Feature frame is missing contribution columns: {missing}")

    contributions = model.predict(feature_frame[feature_names], pred_contrib=True)
    contributions = np.asarray(contributions)
    if contributions.ndim == 3:
        contributions = contributions[:, :, -1]
    if contributions.shape[1] == len(feature_names) + 1:
        contributions = contributions[:, :-1]

    rows: list[dict[str, Any]] = []
    for row_index, row_values in enumerate(contributions):
        order = np.argsort(np.abs(row_values))[::-1][:top_n]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "row_index": row_index,
                    "explanation_rank": rank,
                    "feature": feature_names[int(feature_index)],
                    "feature_contribution": float(row_values[int(feature_index)]),
                    "explanation_method": "lightgbm_pred_contrib",
                }
            )
    return pd.DataFrame(rows)


def write_signal_explanations(
    predictions_input: Path = DEFAULT_PREDICTIONS_INPUT,
    features_input: Path = DEFAULT_FEATURES_INPUT,
    importance_input: Path = DEFAULT_IMPORTANCE_INPUT,
    output_path: Path = DEFAULT_EXPLANATIONS_OUTPUT,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    top_n: int = 5,
    max_signals: int = 200,
) -> ExplanationResult:
    """Load local artifacts, build explanations, and write the explanation table."""
    predictions = _read_table(predictions_input)
    features = _read_table(features_input)
    importance = _read_table(importance_input)
    explanations = build_signal_explanations(
        predictions,
        features,
        importance,
        model_name=model_name,
        top_n=top_n,
        max_signals=max_signals,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explanations.to_csv(output_path, index=False)
    signal_count = (
        int(explanations["market_hash_name"].nunique()) if not explanations.empty else 0
    )
    return ExplanationResult(
        output_path=output_path,
        row_count=len(explanations),
        signal_count=signal_count,
    )


def _latest_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    *,
    max_signals: int,
) -> pd.DataFrame:
    frame = predictions.copy()
    if "model_name" in frame.columns:
        model_frame = frame[frame["model_name"].astype(str).eq(model_name)].copy()
        if model_frame.empty and model_name == DEFAULT_MODEL_NAME:
            component_mask = frame["model_name"].astype(str).isin(LIGHTGBM_COMPONENT_MODELS)
            model_frame = frame[component_mask].copy()
        frame = model_frame if not model_frame.empty else frame
    if "timestamp" not in frame.columns:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = frame["timestamp"].max()
    latest = frame[frame["timestamp"].eq(latest_timestamp)].copy()
    if "strong_buy_score" in latest.columns:
        latest["strong_buy_score"] = pd.to_numeric(
            latest["strong_buy_score"],
            errors="coerce",
        )
        latest = latest.sort_values("strong_buy_score", ascending=False)
    return latest.head(max_signals).reset_index(drop=True)


def _latest_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    if frame.empty or "timestamp" not in frame.columns:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return (
        frame.sort_values(["market_hash_name", "timestamp"])
        .groupby("market_hash_name", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _empty_explanations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "market_hash_name",
            "model_name",
            "strong_buy_score",
            "explanation_rank",
            "feature",
            "feature_value",
            "importance_gain",
            "importance_split",
            "explanation_method",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build latest signal explanation table.")
    parser.add_argument("--predictions-input", type=Path, default=DEFAULT_PREDICTIONS_INPUT)
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--importance-input", type=Path, default=DEFAULT_IMPORTANCE_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPLANATIONS_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-signals", type=int, default=200)
    args = parser.parse_args()

    result = write_signal_explanations(
        predictions_input=args.predictions_input,
        features_input=args.features_input,
        importance_input=args.importance_input,
        output_path=args.output,
        model_name=args.model_name,
        top_n=args.top_n,
        max_signals=args.max_signals,
    )
    print(f"Explanation rows: {result.row_count}")
    print(f"Signals explained: {result.signal_count}")
    print(f"Wrote explanations: {_display_path(result.output_path)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
