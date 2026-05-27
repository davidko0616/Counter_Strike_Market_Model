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
DEFAULT_POLICY_INPUTS = {
    "raw_v2_costed": reports_path("backtests", "day22_raw_v2_costed_accepted_trades.csv"),
    "min_price_1_costed": reports_path("backtests", "day22_min_price_1_costed_accepted_trades.csv"),
    "min_price_5_costed": reports_path("backtests", "day22_min_price_5_costed_accepted_trades.csv"),
}
DEFAULT_POLICY_SELECTION_INPUTS = {
    "raw_v2_costed": reports_path("tables", "day22_raw_v2_costed_threshold_selection.csv"),
    "min_price_1_costed": reports_path("tables", "day22_min_price_1_costed_threshold_selection.csv"),
    "min_price_5_costed": reports_path("tables", "day22_min_price_5_costed_threshold_selection.csv"),
}
DEFAULT_POLICY_SUMMARY_OUTPUT = reports_path("tables", "day17_policy_variant_summary.csv")
DEFAULT_POLICY_ITEM_OUTPUT = reports_path("tables", "day17_policy_variant_item_attribution.csv")
DEFAULT_POLICY_PERIOD_OUTPUT = reports_path("tables", "day17_policy_variant_period_attribution.csv")
DEFAULT_POLICY_BUCKET_OUTPUT = reports_path("tables", "day17_policy_variant_bucket_attribution.csv")
DEFAULT_POLICY_THRESHOLD_OUTPUT = reports_path(
    "tables",
    "day17_policy_variant_threshold_stability.csv",
)
DEFAULT_POLICY_REPORT_OUTPUT = reports_path("backtests", "day17_policy_variant_diagnostics_report.md")

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


@dataclass(frozen=True)
class Day17PolicyVariantResult:
    """Output paths and row counts for corrected policy-variant diagnostics."""

    summary_output: Path
    item_output: Path
    period_output: Path
    bucket_output: Path
    threshold_output: Path
    report_output: Path
    summary_rows: int
    item_rows: int
    period_rows: int
    bucket_rows: int
    threshold_rows: int


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


def run_day17_policy_variant_diagnostics(
    policy_inputs: dict[str, Path] | None = None,
    selection_inputs: dict[str, Path] | None = None,
    summary_output: Path = DEFAULT_POLICY_SUMMARY_OUTPUT,
    item_output: Path = DEFAULT_POLICY_ITEM_OUTPUT,
    period_output: Path = DEFAULT_POLICY_PERIOD_OUTPUT,
    bucket_output: Path = DEFAULT_POLICY_BUCKET_OUTPUT,
    threshold_output: Path = DEFAULT_POLICY_THRESHOLD_OUTPUT,
    report_output: Path = DEFAULT_POLICY_REPORT_OUTPUT,
) -> Day17PolicyVariantResult:
    """Run Day 17 diagnostics directly from corrected policy-variant outputs."""
    trades = load_policy_variant_trades(policy_inputs or DEFAULT_POLICY_INPUTS)
    selections = load_policy_variant_selections(
        selection_inputs or DEFAULT_POLICY_SELECTION_INPUTS,
    )
    summary = build_policy_variant_summary(trades, selections)
    item = build_policy_variant_group_attribution(
        trades,
        ["policy_variant", "model_name", "market_hash_name"],
    )
    period = build_policy_variant_group_attribution(
        trades,
        ["policy_variant", "model_name", "test_period"],
    )
    buckets = build_policy_variant_bucket_attribution(trades)
    threshold = build_policy_variant_threshold_stability(selections)
    report = build_policy_variant_markdown_report(summary, item, period, buckets, threshold)

    outputs = [summary_output, item_output, period_output, bucket_output, threshold_output, report_output]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    item.to_csv(item_output, index=False)
    period.to_csv(period_output, index=False)
    buckets.to_csv(bucket_output, index=False)
    threshold.to_csv(threshold_output, index=False)
    report_output.write_text(report, encoding="utf-8")

    return Day17PolicyVariantResult(
        summary_output=summary_output,
        item_output=item_output,
        period_output=period_output,
        bucket_output=bucket_output,
        threshold_output=threshold_output,
        report_output=report_output,
        summary_rows=len(summary),
        item_rows=len(item),
        period_rows=len(period),
        bucket_rows=len(buckets),
        threshold_rows=len(threshold),
    )


def load_policy_variant_trades(policy_inputs: dict[str, Path]) -> pd.DataFrame:
    """Load accepted trades generated by corrected policy-variant backtests."""
    frames: list[pd.DataFrame] = []
    for policy_variant, path in policy_inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Accepted trades not found for {policy_variant}: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "policy_variant", policy_variant)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True, sort=False)
    trades["entry_timestamp"] = pd.to_datetime(trades["entry_timestamp"], utc=True)
    if "test_period" not in trades.columns:
        trades["test_period"] = (
            trades["entry_timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
        )
    else:
        trades["test_period"] = trades["test_period"].astype(str)
    for column in [
        "realized_net_return",
        "pnl",
        "notional",
        "entry_close",
        "strong_buy_score",
        "liquidity_quality_score_30d",
        "selected_threshold",
        "original_notional",
        "original_pnl",
        "capacity_notional_multiplier",
        "capacity_liquidity_notional_cap",
    ]:
        if column in trades.columns:
            trades[column] = pd.to_numeric(trades[column], errors="coerce")
    return trades


def load_policy_variant_selections(selection_inputs: dict[str, Path]) -> pd.DataFrame:
    """Load corrected walk-forward threshold selections for each policy variant."""
    frames: list[pd.DataFrame] = []
    for policy_variant, path in selection_inputs.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "policy_variant", policy_variant)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_policy_variant_summary(
    trades: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize accepted-trade performance and rejection counts by policy."""
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    selection_totals = _selection_totals(selections)
    for (policy_variant, model_name), group in trades.groupby(
        ["policy_variant", "model_name"],
        sort=False,
    ):
        selection_key = (str(policy_variant), str(model_name))
        selection = selection_totals.get(selection_key, {})
        accepted_count = int(len(group))
        rejected_count = int(_zero_if_nan(selection.get("test_rejected_count", 0)))
        total_seen = accepted_count + rejected_count
        total_pnl = float(group["pnl"].sum())
        rows.append(
            {
                "policy_variant": policy_variant,
                "model_name": model_name,
                "accepted_trade_count": accepted_count,
                "rejected_trade_count": rejected_count,
                "rejection_rate": float(rejected_count / total_seen) if total_seen else np.nan,
                "item_count": int(group["market_hash_name"].nunique()),
                "period_count": int(group["test_period"].nunique()),
                "accepted_pnl": total_pnl,
                "accepted_original_pnl": _optional_sum(group, "original_pnl"),
                "accepted_notional": float(group["notional"].sum()),
                "accepted_original_notional": _optional_sum(group, "original_notional"),
                "accepted_avg_return": float(group["realized_net_return"].mean()),
                "accepted_median_return": float(group["realized_net_return"].median()),
                "accepted_win_rate": _win_rate_or_nan(group),
                "accepted_precision": _precision_or_nan(group),
                "profit_factor": _profit_factor(group["pnl"]),
                "average_capacity_multiplier": _optional_mean(
                    group,
                    "capacity_notional_multiplier",
                ),
                "median_entry_close": _optional_median(group, "entry_close"),
                "positive_period_share": float(
                    group.groupby("test_period")["pnl"].sum().gt(0).mean()
                ),
                "selected_threshold_mean": float(
                    pd.to_numeric(group["selected_threshold"], errors="coerce").mean()
                )
                if "selected_threshold" in group.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "policy_variant"]).reset_index(drop=True)


def build_policy_variant_group_attribution(
    trades: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Build accepted-trade attribution for a policy/model/group combination."""
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    total_lookup = trades.groupby(["policy_variant", "model_name"])["pnl"].sum().to_dict()
    for keys, group in trades.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        total_pnl = float(group["pnl"].sum())
        model_total = float(total_lookup.get((row["policy_variant"], row["model_name"]), np.nan))
        row.update(
            {
                "accepted_trade_count": int(len(group)),
                "accepted_pnl": total_pnl,
                "accepted_original_pnl": _optional_sum(group, "original_pnl"),
                "pnl_share_of_model": _share(total_pnl, model_total),
                "accepted_avg_return": float(group["realized_net_return"].mean()),
                "accepted_median_return": float(group["realized_net_return"].median()),
                "accepted_win_rate": _win_rate_or_nan(group),
                "accepted_precision": _precision_or_nan(group),
                "profit_factor": _profit_factor(group["pnl"]),
                "accepted_notional": float(group["notional"].sum()),
                "average_capacity_multiplier": _optional_mean(
                    group,
                    "capacity_notional_multiplier",
                ),
                "median_entry_close": _optional_median(group, "entry_close"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["policy_variant", "model_name", "accepted_pnl"],
        ascending=[True, True, False],
    )


def build_policy_variant_bucket_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize corrected accepted trades by price, wear, rarity, and regime buckets."""
    rows: list[pd.DataFrame] = []
    for column in [
        "execution_price_bucket",
        "price_bucket",
        "wear",
        "rarity",
        "category",
        "label_market_regime",
    ]:
        if column not in trades.columns:
            continue
        bucket = build_policy_variant_group_attribution(
            trades,
            ["policy_variant", "model_name", column],
        )
        bucket = bucket.rename(columns={column: "bucket_value"})
        bucket.insert(2, "bucket_type", column)
        rows.append(bucket)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_policy_variant_threshold_stability(selections: pd.DataFrame) -> pd.DataFrame:
    """Measure threshold and rejection stability for corrected policy variants."""
    if selections.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for policy_variant, group in selections.groupby("policy_variant", sort=False):
        stability = build_threshold_stability(group)
        if stability.empty:
            continue
        stability.insert(0, "policy_variant", policy_variant)
        rows.append(stability)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_policy_variant_markdown_report(
    summary: pd.DataFrame,
    item: pd.DataFrame,
    period: pd.DataFrame,
    buckets: pd.DataFrame,
    threshold: pd.DataFrame,
) -> str:
    """Create a concise report from corrected policy-variant Day 17 diagnostics."""
    lightgbm = summary[summary["model_name"].eq("lightgbm_rank_blend")]
    lines = [
        "# Day 17 Corrected Policy-Variant Diagnostics",
        "",
        "## Purpose",
        "",
        "This rerun reads the corrected Day 22 policy-variant accepted trades directly,",
        "so diagnostics use the same gates, execution costs, and capacity sizing as test evaluation.",
        "",
        "## LightGBM Summary",
        "",
        _markdown_table(lightgbm),
        "",
        "## LightGBM Threshold Stability",
        "",
        _markdown_table(threshold[threshold["model_name"].eq("lightgbm_rank_blend")]),
        "",
        "## Top LightGBM Items",
        "",
        _markdown_table(
            item[item["model_name"].eq("lightgbm_rank_blend")]
            .groupby("policy_variant", group_keys=False)
            .head(10)
        ),
        "",
        "## Weakest LightGBM Periods",
        "",
        _markdown_table(
            period[period["model_name"].eq("lightgbm_rank_blend")]
            .sort_values(["policy_variant", "accepted_pnl"])
            .groupby("policy_variant", group_keys=False)
            .head(5)
        ),
        "",
        "## LightGBM Price And Wear Attribution",
        "",
        _markdown_table(
            buckets[
                buckets["model_name"].eq("lightgbm_rank_blend")
                & buckets["bucket_type"].isin(["execution_price_bucket", "wear", "rarity"])
            ].head(40)
        ),
        "",
        "## Interpretation",
        "",
        _policy_variant_interpretation(lightgbm),
        "",
    ]
    return "\n".join(lines)


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


def _selection_totals(selections: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    if selections.empty:
        return {}
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for (policy_variant, model_name), group in selections.groupby(
        ["policy_variant", "model_name"],
        sort=False,
    ):
        totals[(str(policy_variant), str(model_name))] = {
            "test_accepted_count": _optional_sum(group, "test_accepted_count"),
            "test_rejected_count": _optional_sum(group, "test_rejected_count"),
        }
    return totals


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


def _optional_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _optional_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _optional_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").median())


def _profit_factor(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    gross_profit = values[values > 0].sum()
    gross_loss = abs(values[values < 0].sum())
    if gross_loss > 0:
        return float(gross_profit / gross_loss)
    return float("inf") if gross_profit > 0 else 0.0


def _share(part: float, total: float) -> float:
    return float(part / total) if total != 0 else np.nan


def _zero_if_nan(value: Any) -> float:
    return 0.0 if pd.isna(value) else float(value)


def _policy_variant_interpretation(lightgbm: pd.DataFrame) -> str:
    if lightgbm.empty:
        return "No LightGBM rows were available."
    notes = []
    for _, row in lightgbm.sort_values("policy_variant").iterrows():
        notes.append(
            f"{row['policy_variant']}: accepted PnL {row['accepted_pnl']:.4f}, "
            f"{int(row['accepted_trade_count'])} trades, "
            f"profit factor {row['profit_factor']:.4f}, "
            f"rejection rate {row['rejection_rate']:.2%}, "
            f"median entry close {row['median_entry_close']:.4f}."
        )
    return "\n\n".join(notes)


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
    parser.add_argument(
        "--policy-variant-mode",
        action="store_true",
        help="Read corrected Day 22 policy-variant accepted trades directly.",
    )
    parser.add_argument("--base-ledger-input", type=Path, default=DEFAULT_BASE_LEDGER_INPUT)
    parser.add_argument("--accepted-input", type=Path, default=DEFAULT_ACCEPTED_INPUT)
    parser.add_argument("--selection-input", type=Path, default=DEFAULT_SELECTION_INPUT)
    parser.add_argument("--labels-input", type=Path, default=DEFAULT_LABELS_INPUT)
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    args = parser.parse_args()

    if args.policy_variant_mode:
        result = run_day17_policy_variant_diagnostics()
        print(f"Policy summary rows: {result.summary_rows}")
        print(f"Policy item rows: {result.item_rows}")
        print(f"Policy period rows: {result.period_rows}")
        print(f"Policy bucket rows: {result.bucket_rows}")
        print(f"Policy threshold rows: {result.threshold_rows}")
        print(f"Wrote report: {_display_path(result.report_output)}")
        return

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
