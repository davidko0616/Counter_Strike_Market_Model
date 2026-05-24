"""Label sensitivity checks for fee and horizon assumptions."""

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
DEFAULT_OUTPUT = reports_path("tables", "day9_5_label_sensitivity.csv")


@dataclass(frozen=True)
class LabelSensitivityScenario:
    """One label configuration to test."""

    scenario_name: str
    fee_model: FeeModel
    horizon_days: int
    required_profit_bps: float = 300.0
    stop_loss_bps: float = 500.0


def default_scenarios() -> list[LabelSensitivityScenario]:
    """Return a compact scenario grid for the current MVP."""
    baseline_fee = _fee_model_from_config("steam")
    youpin_fee = _fee_model_from_config("youpin")
    buff_fee = _fee_model_from_config("buff")
    csfloat_fee = _fee_model_from_config("csfloat")
    return [
        LabelSensitivityScenario("baseline_2_5pct_7d", baseline_fee, horizon_days=7),
        LabelSensitivityScenario("baseline_2_5pct_14d", baseline_fee, horizon_days=14),
        LabelSensitivityScenario("baseline_2_5pct_30d", baseline_fee, horizon_days=30),
        LabelSensitivityScenario("youpin_like_14d", youpin_fee, horizon_days=14),
        LabelSensitivityScenario("buff_like_14d", buff_fee, horizon_days=14),
        LabelSensitivityScenario("csfloat_like_7d", csfloat_fee, horizon_days=7),
        LabelSensitivityScenario("csfloat_like_14d", csfloat_fee, horizon_days=14),
        LabelSensitivityScenario("csfloat_like_30d", csfloat_fee, horizon_days=30),
    ]


def run_label_sensitivity(
    feature_table: pd.DataFrame,
    scenarios: list[LabelSensitivityScenario] | None = None,
    cusum_config: CusumConfig | None = None,
) -> pd.DataFrame:
    """Build a sensitivity summary over several label assumptions."""
    resolved_scenarios = scenarios or default_scenarios()
    events = sample_cusum_events(feature_table, cusum_config)
    rows: list[dict[str, Any]] = []

    for scenario in resolved_scenarios:
        labels = apply_triple_barrier_labels(
            feature_table,
            events,
            fee_model=scenario.fee_model,
            config=TripleBarrierConfig(
                horizon_days=scenario.horizon_days,
                required_profit_bps=scenario.required_profit_bps,
                stop_loss_bps=scenario.stop_loss_bps,
            ),
        )
        rows.append(_summarize_scenario(scenario, labels))

    return pd.DataFrame(rows)


def build_day9_5_label_sensitivity(
    features_input: Path = DEFAULT_FEATURES_INPUT,
    output: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Load features, run sensitivity checks, and write the report."""
    if not features_input.exists():
        raise FileNotFoundError(f"Feature table not found: {features_input}")
    feature_table = pd.read_parquet(features_input)
    sensitivity = run_label_sensitivity(feature_table)

    output.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output, index=False)
    return sensitivity


def _summarize_scenario(
    scenario: LabelSensitivityScenario,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    complete = labels[labels["is_label_complete"]].copy()
    label_counts = complete["label_class"].value_counts()
    complete_count = int(len(complete))
    event_count = int(len(labels))
    strong_buy_count = int(label_counts.get("Strong Buy", 0))
    good_time_count = int(label_counts.get("Good Time", 0))
    bad_time_count = int(label_counts.get("Bad Time", 0))

    return {
        "scenario_name": scenario.scenario_name,
        "horizon_days": scenario.horizon_days,
        "required_profit_bps": scenario.required_profit_bps,
        "stop_loss_bps": scenario.stop_loss_bps,
        "buy_fee_bps": scenario.fee_model.buy_fee_bps,
        "sell_fee_bps": scenario.fee_model.sell_fee_bps,
        "slippage_bps": scenario.fee_model.slippage_bps,
        "flat_round_trip_cost": scenario.fee_model.flat_round_trip_cost,
        "event_count": event_count,
        "complete_label_count": complete_count,
        "incomplete_label_count": event_count - complete_count,
        "bad_time_count": bad_time_count,
        "good_time_count": good_time_count,
        "strong_buy_count": strong_buy_count,
        "bad_time_pct": _pct(bad_time_count, complete_count),
        "good_time_pct": _pct(good_time_count, complete_count),
        "strong_buy_pct": _pct(strong_buy_count, complete_count),
        "median_net_return_at_label": _median(complete["net_return_at_label"]),
        "mean_net_return_at_label": _mean(complete["net_return_at_label"]),
        "strong_buy_median_net_return": _median(
            complete.loc[complete["label_class"] == "Strong Buy", "net_return_at_label"]
        ),
    }


def _fee_model_from_config(venue: str) -> FeeModel:
    fees = load_yaml_config("fees.yaml")
    venue_config = fees.get("venues", {}).get(venue)
    if not isinstance(venue_config, dict):
        raise ValueError(f"Missing fee configuration for venue: {venue}")
    return FeeModel(
        buy_fee_bps=float(venue_config.get("buy_fee_bps", 0.0)),
        sell_fee_bps=float(venue_config.get("sell_fee_bps", 0.0)),
        slippage_bps=float(venue_config.get("slippage_bps", 0.0)),
    )


def _pct(count: int, total: int) -> float:
    return float(count / total) if total else np.nan


def _median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def _mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 9.5 label sensitivity checks.")
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sensitivity = build_day9_5_label_sensitivity(
        features_input=args.features_input,
        output=args.output,
    )
    print(f"Scenarios: {len(sensitivity)}")
    print(f"Wrote sensitivity report: {args.output}")


if __name__ == "__main__":
    main()
