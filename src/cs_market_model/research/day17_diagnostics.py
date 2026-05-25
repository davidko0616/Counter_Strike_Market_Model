"""Day 17 diagnostics for rejection-aware normal-regime results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.backtesting.rejection import apply_rejection_policy
from cs_market_model.backtesting.rejection_policy import RejectionPolicy
from cs_market_model.config import PROJECT_ROOT, data_path, reports_path

DEFAULT_BASE_LEDGER_INPUT = reports_path("backtests", "day12_13_trade_ledger.csv")
DEFAULT_ACCEPTED_INPUT = reports_path("backtests", "day16_walk_forward_accepted_trades.csv")
DEFAULT_SELECTION_INPUT = reports_path("tables", "day16_walk_forward_threshold_selection.csv")
DEFAULT_LABELS_INPUT = data_path("labels", "labels_day7_v1.parquet")
DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")

DEFAULT_ITEM_ATTRIBUTION_OUTPUT = reports_path("tables", "day17_item_attribution.csv")
DEFAULT_PERIOD_ATTRIBUTION_OUTPUT = reports_path("tables", "day17_period_attribution.csv")
DEFAULT_ACCEPTED_REJECTED_OUTPUT = reports_path(
    "tables",
    "day17_accepted_vs_rejected_profile.csv",
)
DEFAULT_THRESHOLD_STABILITY_OUTPUT = reports_path(
    "tables",
    "day17_threshold_stability.csv",
)
DEFAULT_FEATURE_RETURN_OUTPUT = reports_path(
    "tables",
    "day17_feature_return_diagnostics.csv",
)
DEFAULT_LABEL_QUALITY_OUTPUT = reports_path("tables", "day17_label_quality_snapshot.csv")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "day17_diagnostics_report.md")
DEFAULT_NOTEBOOK_OUTPUT = PROJECT_ROOT / "notebooks" / "01_day17_rejection_diagnostics.ipynb"

DIAGNOSTIC_FEATURES = [
    "strong_buy_score",
    "liquidity_quality_score_30d",
    "cs2_index_return_7d",
    "cs2_index_return_30d",
    "cs2_index_drawdown_30d",
    "cs2_index_ma_distance_30d",
    "item_excess_log_return_7d",
    "item_excess_log_return_30d",
]


@dataclass(frozen=True)
class Day17Result:
    """Output paths and row counts for Day 17 diagnostics."""

    item_attribution_output: Path
    period_attribution_output: Path
    accepted_rejected_output: Path
    threshold_stability_output: Path
    feature_return_output: Path
    label_quality_output: Path
    report_output: Path
    notebook_output: Path
    item_rows: int
    period_rows: int
    profile_rows: int
    threshold_rows: int
    feature_rows: int
    label_rows: int


def run_day17_diagnostics(
    base_ledger_input: Path = DEFAULT_BASE_LEDGER_INPUT,
    accepted_input: Path = DEFAULT_ACCEPTED_INPUT,
    selection_input: Path = DEFAULT_SELECTION_INPUT,
    labels_input: Path = DEFAULT_LABELS_INPUT,
    features_input: Path = DEFAULT_FEATURES_INPUT,
    item_attribution_output: Path = DEFAULT_ITEM_ATTRIBUTION_OUTPUT,
    period_attribution_output: Path = DEFAULT_PERIOD_ATTRIBUTION_OUTPUT,
    accepted_rejected_output: Path = DEFAULT_ACCEPTED_REJECTED_OUTPUT,
    threshold_stability_output: Path = DEFAULT_THRESHOLD_STABILITY_OUTPUT,
    feature_return_output: Path = DEFAULT_FEATURE_RETURN_OUTPUT,
    label_quality_output: Path = DEFAULT_LABEL_QUALITY_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    notebook_output: Path = DEFAULT_NOTEBOOK_OUTPUT,
) -> Day17Result:
    """Run Day 17 diagnostics from generated Day 16 artifacts."""
    base_ledger = pd.read_csv(base_ledger_input)
    # accepted_input is not used directly — acceptance is re-derived from
    # selection thresholds applied to the base ledger in tag_walk_forward_acceptance.
    selection = pd.read_csv(selection_input)
    labels = pd.read_parquet(labels_input)
    features = pd.read_parquet(features_input)

    tagged = tag_walk_forward_acceptance(base_ledger, selection)
    item_attribution = build_item_attribution(tagged)
    period_attribution = build_period_attribution(tagged)
    accepted_rejected = build_accepted_rejected_profile(tagged)
    threshold_stability = build_threshold_stability(selection)
    feature_return = build_feature_return_diagnostics(tagged)
    label_quality = build_label_quality_snapshot(labels, features)
    report = build_markdown_report(
        item_attribution=item_attribution,
        period_attribution=period_attribution,
        accepted_rejected=accepted_rejected,
        threshold_stability=threshold_stability,
        feature_return=feature_return,
        label_quality=label_quality,
    )

    outputs = [
        item_attribution_output,
        period_attribution_output,
        accepted_rejected_output,
        threshold_stability_output,
        feature_return_output,
        label_quality_output,
        report_output,
        notebook_output,
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    item_attribution.to_csv(item_attribution_output, index=False)
    period_attribution.to_csv(period_attribution_output, index=False)
    accepted_rejected.to_csv(accepted_rejected_output, index=False)
    threshold_stability.to_csv(threshold_stability_output, index=False)
    feature_return.to_csv(feature_return_output, index=False)
    label_quality.to_csv(label_quality_output, index=False)
    report_output.write_text(report, encoding="utf-8")
    write_notebook_scaffold(notebook_output)

    return Day17Result(
        item_attribution_output=item_attribution_output,
        period_attribution_output=period_attribution_output,
        accepted_rejected_output=accepted_rejected_output,
        threshold_stability_output=threshold_stability_output,
        feature_return_output=feature_return_output,
        label_quality_output=label_quality_output,
        report_output=report_output,
        notebook_output=notebook_output,
        item_rows=len(item_attribution),
        period_rows=len(period_attribution),
        profile_rows=len(accepted_rejected),
        threshold_rows=len(threshold_stability),
        feature_rows=len(feature_return),
        label_rows=len(label_quality),
    )


def tag_walk_forward_acceptance(
    ledger: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """Tag normal-regime ledger rows using Day 16 walk-forward thresholds."""
    if ledger.empty:
        return ledger.copy()

    frame = ledger.copy()
    if "label_market_regime" in frame.columns:
        frame = frame[frame["label_market_regime"].astype(str).ne("known_event")].copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["test_period"] = (
        frame["entry_timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    frame["selected_threshold"] = 0.0
    if not selection.empty and "selected_score_threshold" in selection.columns:
        sel_lookup = selection[["model_name", "test_period", "selected_score_threshold"]].copy()
        sel_lookup["model_name"] = sel_lookup["model_name"].astype(str)
        sel_lookup["test_period"] = sel_lookup["test_period"].astype(str)
        frame["model_name"] = frame["model_name"].astype(str)
        frame = frame.merge(
            sel_lookup, on=["model_name", "test_period"], how="left", suffixes=("", "_sel"),
        )
        frame["selected_threshold"] = (
            frame["selected_score_threshold"].fillna(0.0)
        )
        frame = frame.drop(columns=["selected_score_threshold"], errors="ignore")

    tagged_frames = []
    for _, group in frame.groupby(["model_name", "test_period"], sort=False):
        threshold = float(group["selected_threshold"].iloc[0])
        policy = RejectionPolicy(min_score_threshold=threshold, exclude_event_regime=False)
        tagged_frames.append(apply_rejection_policy(group, policy))
    return pd.concat(tagged_frames, ignore_index=True, sort=False) if tagged_frames else frame


def build_item_attribution(tagged: pd.DataFrame) -> pd.DataFrame:
    """Summarize normal-regime base and accepted performance by item."""
    rows: list[dict[str, Any]] = []
    if tagged.empty:
        return pd.DataFrame()
    for (model_name, item), group in tagged.groupby(["model_name", "market_hash_name"]):
        accepted = group[group["is_accepted"]]
        rows.append({
            "model_name": model_name,
            "market_hash_name": item,
            "base_trade_count": int(len(group)),
            "accepted_trade_count": int(len(accepted)),
            "rejection_rate": float(1.0 - len(accepted) / len(group)),
            "base_pnl": float(group["pnl"].sum()),
            "accepted_pnl": float(accepted["pnl"].sum()),
            "base_avg_return": float(group["realized_net_return"].mean()),
            "accepted_avg_return": _mean_or_nan(accepted, "realized_net_return"),
            "accepted_win_rate": _win_rate_or_nan(accepted),
            "accepted_precision": _precision_or_nan(accepted),
        })
    return pd.DataFrame(rows).sort_values(["model_name", "accepted_pnl"], ascending=[True, False])


def build_period_attribution(tagged: pd.DataFrame) -> pd.DataFrame:
    """Summarize normal-regime base and accepted performance by month."""
    rows: list[dict[str, Any]] = []
    if tagged.empty:
        return pd.DataFrame()
    for (model_name, period), group in tagged.groupby(["model_name", "test_period"]):
        accepted = group[group["is_accepted"]]
        rows.append({
            "model_name": model_name,
            "test_period": period,
            "base_trade_count": int(len(group)),
            "accepted_trade_count": int(len(accepted)),
            "rejection_rate": float(1.0 - len(accepted) / len(group)),
            "selected_threshold": float(group["selected_threshold"].iloc[0]),
            "base_pnl": float(group["pnl"].sum()),
            "accepted_pnl": float(accepted["pnl"].sum()),
            "accepted_avg_return": _mean_or_nan(accepted, "realized_net_return"),
            "accepted_win_rate": _win_rate_or_nan(accepted),
            "accepted_precision": _precision_or_nan(accepted),
        })
    return pd.DataFrame(rows).sort_values(["model_name", "test_period"])


def build_accepted_rejected_profile(tagged: pd.DataFrame) -> pd.DataFrame:
    """Compare accepted and rejected trade profiles."""
    rows: list[dict[str, Any]] = []
    if tagged.empty:
        return pd.DataFrame()
    for (model_name, accepted_flag), group in tagged.groupby(["model_name", "is_accepted"]):
        row: dict[str, Any] = {
            "model_name": model_name,
            "selection_bucket": "accepted" if bool(accepted_flag) else "rejected",
            "trade_count": int(len(group)),
            "total_pnl": float(group["pnl"].sum()),
            "avg_return": float(group["realized_net_return"].mean()),
            "win_rate": float((group["realized_net_return"] > 0).mean()),
            "precision": float(group["is_actual_strong_buy"].mean())
            if "is_actual_strong_buy" in group.columns
            else np.nan,
        }
        for feature in DIAGNOSTIC_FEATURES:
            if feature in group.columns:
                row[f"{feature}_mean"] = float(pd.to_numeric(group[feature], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_name", "selection_bucket"])


def build_threshold_stability(selection: pd.DataFrame) -> pd.DataFrame:
    """Measure threshold stability across walk-forward periods."""
    if selection.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_name, group in selection.groupby("model_name"):
        thresholds = pd.to_numeric(group["selected_score_threshold"], errors="coerce")
        rows.append({
            "model_name": model_name,
            "period_count": int(len(group)),
            "threshold_mean": float(thresholds.mean()),
            "threshold_median": float(thresholds.median()),
            "threshold_std": float(thresholds.std(ddof=0)),
            "threshold_min": float(thresholds.min()),
            "threshold_max": float(thresholds.max()),
            "threshold_changes": int(thresholds.ne(thresholds.shift()).sum() - 1),
            "positive_test_period_share": float(
                (pd.to_numeric(group["test_accepted_total_pnl"], errors="coerce") > 0).mean()
            ),
            "total_test_pnl": float(
                pd.to_numeric(group["test_accepted_total_pnl"], errors="coerce").sum()
            ),
        })
    return pd.DataFrame(rows).sort_values("total_test_pnl", ascending=False)


def build_feature_return_diagnostics(tagged: pd.DataFrame) -> pd.DataFrame:
    """Bucket diagnostic features and summarize realized returns."""
    rows: list[dict[str, Any]] = []
    if tagged.empty:
        return pd.DataFrame()
    for model_name, model_rows in tagged.groupby("model_name"):
        for feature in DIAGNOSTIC_FEATURES:
            if feature not in model_rows.columns:
                continue
            values = pd.to_numeric(model_rows[feature], errors="coerce")
            if values.nunique(dropna=True) < 3:
                continue
            try:
                bins = pd.qcut(values, q=4, duplicates="drop")
            except ValueError:
                continue
            temp = model_rows.assign(_feature_bin=bins)
            for feature_bin, group in temp.groupby("_feature_bin", observed=True):
                accepted = group[group["is_accepted"]]
                rows.append({
                    "model_name": model_name,
                    "feature": feature,
                    "feature_bin": str(feature_bin),
                    "trade_count": int(len(group)),
                    "accepted_count": int(len(accepted)),
                    "base_avg_return": float(group["realized_net_return"].mean()),
                    "accepted_avg_return": _mean_or_nan(accepted, "realized_net_return"),
                    "base_pnl": float(group["pnl"].sum()),
                    "accepted_pnl": float(accepted["pnl"].sum()),
                })
    return pd.DataFrame(rows)


def build_label_quality_snapshot(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Summarize label and data-quality context."""
    rows: list[dict[str, Any]] = []
    if not labels.empty:
        group_cols = ["label_market_regime", "label_class"]
        for keys, group in labels.groupby(group_cols, dropna=False):
            rows.append({
                "source": "labels",
                "group": "|".join(str(key) for key in keys),
                "row_count": int(len(group)),
                "mean_net_return": float(group["net_return_at_label"].mean()),
                "median_net_return": float(group["net_return_at_label"].median()),
                "mean_row_coverage_30d": float(group["row_coverage_30d"].mean())
                if "row_coverage_30d" in group.columns
                else np.nan,
                "mean_staleness_days": float(group["effective_staleness_days"].mean())
                if "effective_staleness_days" in group.columns
                else np.nan,
            })
    if not features.empty:
        rows.append({
            "source": "features",
            "group": "coverage",
            "row_count": int(len(features)),
            "mean_net_return": np.nan,
            "median_net_return": np.nan,
            "mean_row_coverage_30d": float(features["row_coverage_30d"].mean())
            if "row_coverage_30d" in features.columns
            else np.nan,
            "mean_staleness_days": float(features["effective_staleness_days"].mean())
            if "effective_staleness_days" in features.columns
            else np.nan,
        })
    return pd.DataFrame(rows)


def build_markdown_report(
    *,
    item_attribution: pd.DataFrame,
    period_attribution: pd.DataFrame,
    accepted_rejected: pd.DataFrame,
    threshold_stability: pd.DataFrame,
    feature_return: pd.DataFrame,
    label_quality: pd.DataFrame,
) -> str:
    """Create a concise Markdown report from Day 17 outputs."""
    lines = [
        "# Day 17 Rejection Diagnostics",
        "",
        "## Purpose",
        "",
        "Day 17 checks whether the Day 16 rejection-aware result is broad, stable,",
        "and explainable, or whether it is concentrated in a few items, periods,",
        "or feature artifacts.",
        "",
        "## Threshold Stability",
        "",
        _markdown_table(threshold_stability.head(10)),
        "",
        "## Accepted vs Rejected Profile",
        "",
        _markdown_table(accepted_rejected.head(12)),
        "",
        "## Top Accepted Item Attribution",
        "",
        _markdown_table(item_attribution.head(15)),
        "",
        "## Weakest Accepted Periods",
        "",
        _markdown_table(period_attribution.sort_values("accepted_pnl").head(10)),
        "",
        "## Feature Diagnostics",
        "",
        f"Feature-return rows generated: {len(feature_return)}.",
        "",
        "## Label Quality Snapshot",
        "",
        _markdown_table(label_quality.head(12)),
        "",
        "## Next Plan",
        "",
        "Day 18 should add CSFloat listing features and run a SteamDT-only versus",
        "SteamDT-plus-CSFloat ablation. The Day 17 diagnostics should be used to",
        "decide which listing/liquidity features matter most.",
        "",
    ]
    return "\n".join(lines)


def write_notebook_scaffold(path: Path) -> None:
    """Write a minimal notebook that loads Day 17 outputs."""
    notebook = {
        "cells": [
            _markdown_cell("# Day 17 Rejection Diagnostics\n\nLoad generated Day 17 tables."),
            _code_cell(
                "import pandas as pd\n"
                "from pathlib import Path\n\n"
                "root = Path.cwd()\n"
                "tables = root / 'reports' / 'tables'\n"
                "backtests = root / 'reports' / 'backtests'\n"
            ),
            _code_cell(
                "threshold = pd.read_csv(tables / 'day17_threshold_stability.csv')\n"
                "profile = pd.read_csv(tables / 'day17_accepted_vs_rejected_profile.csv')\n"
                "items = pd.read_csv(tables / 'day17_item_attribution.csv')\n"
                "periods = pd.read_csv(tables / 'day17_period_attribution.csv')\n"
                "threshold"
            ),
            _code_cell("profile"),
            _code_cell("items.head(20)"),
            _code_cell("periods.sort_values('accepted_pnl').head(20)"),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _threshold_lookup(selection: pd.DataFrame) -> dict[tuple[str, str], float]:
    if selection.empty:
        return {}
    return {
        (str(row["model_name"]), str(row["test_period"])): float(
            row["selected_score_threshold"]
        )
        for _, row in selection.iterrows()
    }


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _win_rate_or_nan(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    return float((pd.to_numeric(frame["realized_net_return"], errors="coerce") > 0).mean())


def _precision_or_nan(frame: pd.DataFrame) -> float:
    if frame.empty or "is_actual_strong_buy" not in frame.columns:
        return np.nan
    return float(frame["is_actual_strong_buy"].mean())


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [str(column) for column in display.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]) if pd.notna(row[column]) else "" for column in display.columns]
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows)


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 17 rejection diagnostics.")
    parser.add_argument("--base-ledger-input", type=Path, default=DEFAULT_BASE_LEDGER_INPUT)
    parser.add_argument("--accepted-input", type=Path, default=DEFAULT_ACCEPTED_INPUT)
    parser.add_argument("--selection-input", type=Path, default=DEFAULT_SELECTION_INPUT)
    parser.add_argument("--labels-input", type=Path, default=DEFAULT_LABELS_INPUT)
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    args = parser.parse_args()

    result = run_day17_diagnostics(
        base_ledger_input=args.base_ledger_input,
        accepted_input=args.accepted_input,
        selection_input=args.selection_input,
        labels_input=args.labels_input,
        features_input=args.features_input,
    )
    print(f"Item attribution rows: {result.item_rows}")
    print(f"Period attribution rows: {result.period_rows}")
    print(f"Accepted/rejected profile rows: {result.profile_rows}")
    print(f"Threshold stability rows: {result.threshold_rows}")
    print(f"Feature diagnostic rows: {result.feature_rows}")
    print(f"Label quality rows: {result.label_rows}")
    print(f"Wrote report: {_display_path(result.report_output)}")
    print(f"Wrote notebook: {_display_path(result.notebook_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
