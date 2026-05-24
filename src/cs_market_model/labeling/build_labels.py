"""Build Day 7 event and label tables from the Day 5 feature table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import data_path, load_yaml_config, reports_path
from cs_market_model.labeling.cusum import CusumConfig, sample_cusum_events
from cs_market_model.labeling.triple_barrier import (
    FeeModel,
    TripleBarrierConfig,
    apply_triple_barrier_labels,
)

DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_LABELS_OUTPUT = data_path("labels", "labels_day7_v1.parquet")
DEFAULT_AUDIT_OUTPUT = reports_path("tables", "day7_label_audit.csv")
DEFAULT_EXAMPLES_OUTPUT = reports_path("tables", "day7_label_examples.csv")


@dataclass(frozen=True)
class LabelBuildResult:
    """Paths and summary counts for a completed label build."""

    labels_output: Path
    audit_output: Path
    examples_output: Path
    event_count: int
    complete_label_count: int
    incomplete_label_count: int


def load_feature_table(path: Path = DEFAULT_FEATURES_INPUT) -> pd.DataFrame:
    """Load the Day 5 feature table."""
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    return pd.read_parquet(path)


def fee_model_from_config(venue: str = "steam") -> FeeModel:
    """Load execution costs for a venue from configs/fees.yaml."""
    fees = load_yaml_config("fees.yaml")
    venues = fees.get("venues")
    if not isinstance(venues, dict) or venue not in venues:
        raise ValueError(f"Venue {venue!r} not found in configs/fees.yaml")
    venue_config = venues[venue]
    if not isinstance(venue_config, dict):
        raise ValueError(f"Expected mapping for venue {venue!r} in configs/fees.yaml")
    return FeeModel(
        buy_fee_bps=float(venue_config.get("buy_fee_bps", 0.0)),
        sell_fee_bps=float(venue_config.get("sell_fee_bps", 0.0)),
        slippage_bps=float(venue_config.get("slippage_bps", 0.0)),
    )


def barrier_config_from_config(
    *,
    horizon_days: int | None = None,
    required_profit_bps: float = 300.0,
    stop_loss_bps: float = 500.0,
    volatility_stop_multiplier: float = 1.0,
    min_row_coverage: float | None = None,
    max_staleness_days: float = 3.0,
) -> TripleBarrierConfig:
    """Load label settings from backtest config, with explicit CLI overrides."""
    backtest = load_yaml_config("backtest.yaml")
    resolved_horizon_days = int(
        horizon_days if horizon_days is not None else backtest.get("prediction_horizon_days", 14)
    )
    resolved_min_row_coverage = float(
        min_row_coverage
        if min_row_coverage is not None
        else backtest.get("min_liquidity_score", 0.3)
    )
    return TripleBarrierConfig(
        horizon_days=resolved_horizon_days,
        required_profit_bps=required_profit_bps,
        stop_loss_bps=stop_loss_bps,
        volatility_stop_multiplier=volatility_stop_multiplier,
        min_row_coverage=resolved_min_row_coverage,
        max_staleness_days=max_staleness_days,
    )


def build_label_table(
    feature_table: pd.DataFrame,
    cusum_config: CusumConfig | None = None,
    triple_barrier_config: TripleBarrierConfig | None = None,
    fee_model: FeeModel | None = None,
) -> pd.DataFrame:
    """Sample CUSUM events and apply net-return labels."""
    events = sample_cusum_events(feature_table, cusum_config)
    labels = apply_triple_barrier_labels(
        feature_table,
        events,
        fee_model=fee_model,
        config=triple_barrier_config,
    )
    return labels


def audit_label_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Build a compact metric table for label quality review."""
    metrics: list[dict[str, Any]] = [
        {"metric": "event_count", "value": len(labels)},
        {
            "metric": "complete_label_count",
            "value": int(labels["is_label_complete"].sum()) if not labels.empty else 0,
        },
        {
            "metric": "incomplete_label_count",
            "value": int((~labels["is_label_complete"]).sum()) if not labels.empty else 0,
        },
    ]

    if labels.empty:
        return pd.DataFrame(metrics)

    complete_labels = labels[labels["is_label_complete"]]
    for label_class, count in complete_labels["label_class"].value_counts().sort_index().items():
        metrics.append({"metric": f"label_count.{label_class}", "value": int(count)})

    for reason, count in labels["label_reason"].value_counts(dropna=False).sort_index().items():
        metrics.append({"metric": f"label_reason_count.{reason}", "value": int(count)})

    numeric_summaries = [
        "net_return_at_label",
        "vertical_net_return",
        "max_net_return",
        "min_net_return",
        "holding_period_days",
    ]
    for column in numeric_summaries:
        series = pd.to_numeric(complete_labels[column], errors="coerce")
        finite = series[np.isfinite(series)]
        if finite.empty:
            continue
        metrics.extend(
            [
                {"metric": f"{column}.mean", "value": float(finite.mean())},
                {"metric": f"{column}.median", "value": float(finite.median())},
                {"metric": f"{column}.min", "value": float(finite.min())},
                {"metric": f"{column}.max", "value": float(finite.max())},
            ]
        )

    return pd.DataFrame(metrics)


def label_examples(labels: pd.DataFrame, examples_per_class: int = 8) -> pd.DataFrame:
    """Return deterministic examples for manual label inspection."""
    if labels.empty:
        return labels.copy()
    complete = labels[labels["is_label_complete"]].copy()
    if complete.empty:
        return complete

    sort_columns = ["label_class", "timestamp", "market_hash_name"]
    examples = (
        complete.sort_values(sort_columns)
        .groupby("label_class", group_keys=False)
        .head(examples_per_class)
    )
    columns = [
        "event_id",
        "market_hash_name",
        "timestamp",
        "event_side",
        "entry_price",
        "label_class",
        "label_reason",
        "first_touch",
        "exit_timestamp",
        "holding_period_days",
        "net_return_at_label",
        "vertical_net_return",
        "max_net_return",
        "min_net_return",
        "upper_barrier_net_return",
        "lower_barrier_net_return",
        "flat_round_trip_cost",
        "row_coverage_30d",
        "effective_staleness_days",
    ]
    return examples[columns].reset_index(drop=True)


def build_day7_labels(
    features_input: Path = DEFAULT_FEATURES_INPUT,
    labels_output: Path = DEFAULT_LABELS_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    examples_output: Path = DEFAULT_EXAMPLES_OUTPUT,
    venue: str = "steam",
    cusum_config: CusumConfig | None = None,
    triple_barrier_config: TripleBarrierConfig | None = None,
    fee_model: FeeModel | None = None,
    examples_per_class: int = 8,
) -> LabelBuildResult:
    """Load features, build labels, and save Day 7 artifacts."""
    feature_table = load_feature_table(features_input)
    resolved_fee_model = fee_model or fee_model_from_config(venue)
    resolved_triple_barrier_config = triple_barrier_config or barrier_config_from_config()
    labels = build_label_table(
        feature_table,
        cusum_config=cusum_config,
        triple_barrier_config=resolved_triple_barrier_config,
        fee_model=resolved_fee_model,
    )
    audit = audit_label_table(labels)
    examples = label_examples(labels, examples_per_class=examples_per_class)

    labels_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    examples_output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_output, index=False)
    audit.to_csv(audit_output, index=False)
    examples.to_csv(examples_output, index=False)

    complete_count = int(labels["is_label_complete"].sum()) if not labels.empty else 0
    return LabelBuildResult(
        labels_output=labels_output,
        audit_output=audit_output,
        examples_output=examples_output,
        event_count=len(labels),
        complete_label_count=complete_count,
        incomplete_label_count=len(labels) - complete_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Day 7 event labels.")
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--examples-output", type=Path, default=DEFAULT_EXAMPLES_OUTPUT)
    parser.add_argument("--venue", default="steam")
    parser.add_argument("--cusum-threshold-multiplier", type=float, default=0.75)
    parser.add_argument("--cusum-min-threshold", type=float, default=0.02)
    parser.add_argument("--cusum-min-gap-days", type=int, default=1)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--required-profit-bps", type=float, default=300.0)
    parser.add_argument("--stop-loss-bps", type=float, default=500.0)
    parser.add_argument("--volatility-stop-multiplier", type=float, default=1.0)
    parser.add_argument("--min-row-coverage", type=float, default=None)
    parser.add_argument("--max-staleness-days", type=float, default=3.0)
    parser.add_argument("--examples-per-class", type=int, default=8)
    args = parser.parse_args()

    cusum_config = CusumConfig(
        threshold_multiplier=args.cusum_threshold_multiplier,
        min_threshold=args.cusum_min_threshold,
        min_gap_days=args.cusum_min_gap_days,
    )
    triple_barrier_config = barrier_config_from_config(
        horizon_days=args.horizon_days,
        required_profit_bps=args.required_profit_bps,
        stop_loss_bps=args.stop_loss_bps,
        volatility_stop_multiplier=args.volatility_stop_multiplier,
        min_row_coverage=args.min_row_coverage,
        max_staleness_days=args.max_staleness_days,
    )

    result = build_day7_labels(
        features_input=args.features_input,
        labels_output=args.labels_output,
        audit_output=args.audit_output,
        examples_output=args.examples_output,
        venue=args.venue,
        cusum_config=cusum_config,
        triple_barrier_config=triple_barrier_config,
        examples_per_class=args.examples_per_class,
    )
    print(f"Events: {result.event_count}")
    print(f"Complete labels: {result.complete_label_count}")
    print(f"Incomplete labels: {result.incomplete_label_count}")
    print(f"Wrote labels: {result.labels_output}")
    print(f"Wrote audit: {result.audit_output}")
    print(f"Wrote examples: {result.examples_output}")


if __name__ == "__main__":
    main()
