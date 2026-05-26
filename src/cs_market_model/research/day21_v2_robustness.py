"""Robustness audit for expanded-universe walk-forward rejection results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path

DEFAULT_ACCEPTED_INPUT = reports_path("backtests", "day16_walk_forward_accepted_trades.csv")
DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_SCENARIO_OUTPUT = reports_path("tables", "day21_v2_scenario_robustness.csv")
DEFAULT_TOP_TRADES_OUTPUT = reports_path("tables", "day21_v2_top_trades.csv")
DEFAULT_BUCKET_OUTPUT = reports_path("tables", "day21_v2_bucket_attribution.csv")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "day21_v2_robustness_report.md")
FEATURE_JOIN_KEYS = ["market_hash_name", "entry_timestamp"]


@dataclass(frozen=True)
class Day21Result:
    """Output metadata for the v2 robustness audit."""

    scenario_output: Path
    top_trades_output: Path
    bucket_output: Path
    report_output: Path
    scenario_rows: int
    top_trade_rows: int
    bucket_rows: int


def run_day21_v2_robustness(
    accepted_input: Path = DEFAULT_ACCEPTED_INPUT,
    features_input: Path = DEFAULT_FEATURES_INPUT,
    scenario_output: Path = DEFAULT_SCENARIO_OUTPUT,
    top_trades_output: Path = DEFAULT_TOP_TRADES_OUTPUT,
    bucket_output: Path = DEFAULT_BUCKET_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> Day21Result:
    """Build scenario, top-trade, bucket, and Markdown robustness reports."""
    accepted = pd.read_csv(accepted_input)
    features = pd.read_parquet(features_input)
    enriched = enrich_trades_with_features(accepted, features)
    scenarios = build_scenario_robustness(enriched)
    top_trades = build_top_trade_audit(enriched)
    buckets = build_bucket_attribution(enriched)
    report = build_markdown_report(scenarios, top_trades, buckets)

    for output in (scenario_output, top_trades_output, bucket_output, report_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(scenario_output, index=False)
    top_trades.to_csv(top_trades_output, index=False)
    buckets.to_csv(bucket_output, index=False)
    report_output.write_text(report, encoding="utf-8")

    return Day21Result(
        scenario_output=scenario_output,
        top_trades_output=top_trades_output,
        bucket_output=bucket_output,
        report_output=report_output,
        scenario_rows=len(scenarios),
        top_trade_rows=len(top_trades),
        bucket_rows=len(buckets),
    )


def enrich_trades_with_features(
    accepted: pd.DataFrame,
    features: pd.DataFrame,
    *,
    strict_duplicate_conflicts: bool = False,
) -> pd.DataFrame:
    """Attach entry-date price and item metadata to accepted trades."""
    trades = accepted.copy()
    trades["entry_timestamp"] = pd.to_datetime(trades["entry_timestamp"], utc=True)
    features_at_entry = features.copy()
    features_at_entry["entry_timestamp"] = pd.to_datetime(features_at_entry["timestamp"], utc=True)
    feature_columns = [
        column
        for column in [
            "market_hash_name",
            "entry_timestamp",
            "close",
            "category",
            "weapon_type",
            "rarity",
            "wear",
            "source_type",
            "row_coverage_30d",
            "effective_staleness_days",
            "abs_return_max_7d",
            "abs_return_max_30d",
            "price_jump_50pct_count_7d",
            "price_jump_100pct_count_7d",
            "price_jump_50pct_count_30d",
            "price_jump_100pct_count_30d",
        ]
        if column in features_at_entry.columns
    ]
    feature_join, duplicate_diagnostics = dedupe_features_for_trade_join(
        features_at_entry[feature_columns],
    )
    if strict_duplicate_conflicts and duplicate_diagnostics[
        "feature_join_duplicate_conflict"
    ].any():
        conflicts = duplicate_diagnostics[
            duplicate_diagnostics["feature_join_duplicate_conflict"]
        ]
        examples = conflicts.head(5)[FEATURE_JOIN_KEYS].to_dict("records")
        raise ValueError(f"Conflicting duplicate feature rows detected: {examples}")

    enriched = trades.merge(
        feature_join,
        on=["market_hash_name", "entry_timestamp"],
        how="left",
        suffixes=("", "_feature"),
        validate="many_to_one",
    )
    enriched = enriched.merge(
        duplicate_diagnostics,
        on=FEATURE_JOIN_KEYS,
        how="left",
        validate="many_to_one",
    )
    enriched["feature_join_duplicate_count"] = (
        pd.to_numeric(enriched["feature_join_duplicate_count"], errors="coerce")
        .fillna(1)
        .astype(int)
    )
    enriched["feature_join_duplicate_conflict"] = enriched[
        "feature_join_duplicate_conflict"
    ].fillna(False).astype(bool)
    enriched["feature_join_conflicting_columns"] = enriched[
        "feature_join_conflicting_columns"
    ].fillna("")
    for column in ["category", "weapon_type", "rarity", "wear", "source_type"]:
        feature_column = f"{column}_feature"
        if feature_column not in enriched.columns:
            continue
        if column not in enriched.columns:
            enriched[column] = enriched[feature_column]
        else:
            base = enriched[column].replace("", pd.NA)
            enriched[column] = base.fillna(enriched[feature_column])
        enriched = enriched.drop(columns=[feature_column])
    enriched["entry_close"] = pd.to_numeric(enriched.get("close"), errors="coerce")
    enriched = enriched.drop(columns=["close"], errors="ignore")
    enriched["realized_net_return"] = pd.to_numeric(
        enriched["realized_net_return"],
        errors="coerce",
    )
    enriched["notional"] = pd.to_numeric(enriched["notional"], errors="coerce")
    enriched["pnl"] = pd.to_numeric(enriched["pnl"], errors="coerce")
    enriched["entry_month"] = (
        enriched["entry_timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    enriched["price_bucket"] = pd.cut(
        enriched["entry_close"],
        bins=[-np.inf, 1.0, 5.0, 20.0, 100.0, np.inf],
        labels=["<=1", "1-5", "5-20", "20-100", ">100"],
    ).astype("string")
    return enriched


def dedupe_features_for_trade_join(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate feature rows and return per-key duplicate diagnostics."""
    if features.empty:
        diagnostics = pd.DataFrame(
            columns=[
                *FEATURE_JOIN_KEYS,
                "feature_join_duplicate_count",
                "feature_join_duplicate_conflict",
                "feature_join_conflicting_columns",
            ]
        )
        return features.copy(), diagnostics

    frame = features.copy()
    missing = [column for column in FEATURE_JOIN_KEYS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature join keys: {missing}")
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    value_columns = [column for column in frame.columns if column not in FEATURE_JOIN_KEYS]

    diagnostic_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(FEATURE_JOIN_KEYS, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        conflicting_columns = [
            column
            for column in value_columns
            if _normalized_nunique(group[column]) > 1
        ]
        row = dict(zip(FEATURE_JOIN_KEYS, keys, strict=True))
        row.update(
            {
                "feature_join_duplicate_count": int(len(group)),
                "feature_join_duplicate_conflict": bool(conflicting_columns),
                "feature_join_conflicting_columns": "|".join(conflicting_columns),
            }
        )
        diagnostic_rows.append(row)

    diagnostics = pd.DataFrame(diagnostic_rows)
    deduped = frame.drop_duplicates(FEATURE_JOIN_KEYS, keep="first").copy()
    return deduped, diagnostics


def _normalized_nunique(values: pd.Series) -> int:
    normalized = values.map(lambda value: "<NA>" if pd.isna(value) else repr(value))
    return int(normalized.nunique(dropna=False))


def build_scenario_robustness(enriched: pd.DataFrame) -> pd.DataFrame:
    """Summarize accepted-trade performance under robustness scenarios."""
    scenarios = [
        ("raw_all_accepted", _all_mask(enriched), None),
        ("normal_only", enriched["label_market_regime"].astype(str).eq("normal"), None),
        (
            "exclude_known_event_overlap",
            ~enriched.get("label_overlaps_known_market_event", False).fillna(False).astype(bool),
            None,
        ),
        ("cap_returns_100pct", _all_mask(enriched), 1.0),
        ("cap_returns_50pct", _all_mask(enriched), 0.5),
        ("cap_returns_25pct", _all_mask(enriched), 0.25),
        ("entry_price_ge_1", enriched["entry_close"].ge(1.0), None),
        ("entry_price_ge_5", enriched["entry_close"].ge(5.0), None),
    ]
    for top_n in (1, 5, 10):
        scenarios.append(
            (
                f"exclude_top_{top_n}_pnl_trades",
                _exclude_top_pnl_by_model(enriched, top_n),
                None,
            )
        )

    rows: list[dict[str, Any]] = []
    for scenario_name, mask, cap in scenarios:
        scenario_trades = enriched[mask].copy()
        if scenario_trades.empty:
            continue
        scenario_trades["scenario_return"] = scenario_trades["realized_net_return"]
        if cap is not None:
            scenario_trades["scenario_return"] = scenario_trades["scenario_return"].clip(
                lower=-cap,
                upper=cap,
            )
        scenario_trades["scenario_pnl"] = scenario_trades["notional"] * scenario_trades[
            "scenario_return"
        ]
        for model_name, group in scenario_trades.groupby("model_name", sort=False):
            rows.append(_summary_row(group, scenario_name, model_name))
    return pd.DataFrame(rows).sort_values(["model_name", "scenario"]).reset_index(drop=True)


def build_top_trade_audit(enriched: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Return the largest accepted winners with data-quality context."""
    columns = [
        "model_name",
        "market_hash_name",
        "entry_timestamp",
        "exit_timestamp",
        "entry_close",
        "wear",
        "rarity",
        "category",
        "label_market_regime",
        "realized_net_return",
        "notional",
        "pnl",
        "strong_buy_score",
        "liquidity_quality_score_30d",
        "row_coverage_30d",
        "effective_staleness_days",
        "abs_return_max_7d",
        "abs_return_max_30d",
        "price_jump_50pct_count_7d",
        "price_jump_100pct_count_7d",
        "price_jump_50pct_count_30d",
        "price_jump_100pct_count_30d",
        "feature_join_duplicate_count",
        "feature_join_duplicate_conflict",
        "feature_join_conflicting_columns",
    ]
    available_columns = [column for column in columns if column in enriched.columns]
    top = enriched.sort_values("pnl", ascending=False).head(top_n).copy()
    top["audit_flags"] = top.apply(_trade_audit_flags, axis=1)
    return top[[*available_columns, "audit_flags"]].reset_index(drop=True)


def build_bucket_attribution(enriched: pd.DataFrame) -> pd.DataFrame:
    """Summarize PnL by model and interpretable item buckets."""
    bucket_specs = [
        ("wear", "wear"),
        ("rarity", "rarity"),
        ("category", "category"),
        ("price_bucket", "price_bucket"),
        ("entry_month", "entry_month"),
        ("label_market_regime", "label_market_regime"),
    ]
    rows: list[dict[str, Any]] = []
    total_by_model = enriched.groupby("model_name")["pnl"].sum().to_dict()
    for bucket_name, column in bucket_specs:
        if column not in enriched.columns:
            continue
        for (model_name, bucket_value), group in enriched.groupby(
            ["model_name", column],
            dropna=False,
            sort=False,
        ):
            total_pnl = float(group["pnl"].sum())
            model_total = float(total_by_model.get(model_name, np.nan))
            rows.append(
                {
                    "model_name": model_name,
                    "bucket_type": bucket_name,
                    "bucket_value": bucket_value,
                    "trade_count": int(len(group)),
                    "total_pnl": total_pnl,
                    "pnl_share_of_model": float(total_pnl / model_total)
                    if model_total != 0
                    else np.nan,
                    "average_trade_return": float(group["realized_net_return"].mean()),
                    "win_rate": float(group["realized_net_return"].gt(0).mean()),
                    "median_entry_close": float(group["entry_close"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model_name", "bucket_type", "total_pnl"],
        ascending=[True, True, False],
    )


def build_markdown_report(
    scenarios: pd.DataFrame,
    top_trades: pd.DataFrame,
    buckets: pd.DataFrame,
) -> str:
    """Build the concise Markdown robustness report."""
    lightgbm = scenarios[scenarios["model_name"].eq("lightgbm_rank_blend")].copy()
    lines = [
        "# Day 21 V2 Robustness Audit",
        "",
        "## Scenario Summary",
        "",
        _markdown_table(lightgbm),
        "",
        "## Largest LightGBM Winners",
        "",
        _markdown_table(top_trades[top_trades["model_name"].eq("lightgbm_rank_blend")].head(15)),
        "",
        "## LightGBM Bucket Attribution",
        "",
        _markdown_table(buckets[buckets["model_name"].eq("lightgbm_rank_blend")].head(40)),
        "",
        "## Interpretation",
        "",
        _interpret(lightgbm, top_trades, buckets),
        "",
    ]
    return "\n".join(lines)


def _summary_row(group: pd.DataFrame, scenario_name: str, model_name: str) -> dict[str, Any]:
    total_pnl = float(group["scenario_pnl"].sum())
    gross_profit = float(group.loc[group["scenario_pnl"].gt(0), "scenario_pnl"].sum())
    gross_loss = float(-group.loc[group["scenario_pnl"].lt(0), "scenario_pnl"].sum())
    return {
        "scenario": scenario_name,
        "model_name": model_name,
        "trade_count": int(len(group)),
        "total_pnl": total_pnl,
        "average_trade_return": float(group["scenario_return"].mean()),
        "median_trade_return": float(group["scenario_return"].median()),
        "win_rate": float(group["scenario_return"].gt(0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "max_trade_return": float(group["scenario_return"].max()),
        "min_trade_return": float(group["scenario_return"].min()),
        "positive_item_count": int(group.loc[group["scenario_pnl"].gt(0), "market_hash_name"].nunique()),
        "item_count": int(group["market_hash_name"].nunique()),
        "period_count": int(group["entry_month"].nunique()),
    }


def _all_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=frame.index)


def _exclude_top_pnl_by_model(frame: pd.DataFrame, top_n: int) -> pd.Series:
    mask = _all_mask(frame)
    if frame.empty:
        return mask
    top_indices = (
        frame.sort_values("pnl", ascending=False)
        .groupby("model_name", sort=False)
        .head(top_n)
        .index
    )
    mask.loc[top_indices] = False
    return mask


def _trade_audit_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if pd.notna(row.get("entry_close")) and row["entry_close"] < 1:
        flags.append("entry_price_lt_1")
    if row.get("realized_net_return", 0) >= 1:
        flags.append("return_ge_100pct")
    if row.get("realized_net_return", 0) >= 5:
        flags.append("return_ge_500pct")
    if row.get("label_market_regime") != "normal":
        flags.append("non_normal_regime")
    if row.get("price_jump_100pct_count_30d", 0) and row.get("price_jump_100pct_count_30d", 0) > 0:
        flags.append("recent_100pct_jump")
    if row.get("row_coverage_30d", 1) < 0.75:
        flags.append("low_30d_coverage")
    if row.get("feature_join_duplicate_conflict", False):
        flags.append("feature_join_duplicate_conflict")
    return "|".join(flags)


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
        rows.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(rows)


def _interpret(
    lightgbm: pd.DataFrame,
    top_trades: pd.DataFrame,
    buckets: pd.DataFrame,
) -> str:
    if lightgbm.empty:
        return "No LightGBM rows were available for interpretation."
    raw = _scenario(lightgbm, "raw_all_accepted")
    cap50 = _scenario(lightgbm, "cap_returns_50pct")
    exclude_top5 = _scenario(lightgbm, "exclude_top_5_pnl_trades")
    normal = _scenario(lightgbm, "normal_only")
    notes = [
        f"Raw LightGBM accepted PnL is {raw['total_pnl']:.4f} across {int(raw['trade_count'])} trades.",
        f"Normal-only PnL is {normal['total_pnl']:.4f}.",
        f"With returns capped at 50%, PnL is {cap50['total_pnl']:.4f}.",
        f"After removing the top 5 trades, PnL is {exclude_top5['total_pnl']:.4f}.",
    ]
    flagged = top_trades[
        top_trades["model_name"].eq("lightgbm_rank_blend")
        & top_trades["audit_flags"].astype(str).ne("")
    ]
    if not flagged.empty:
        notes.append(
            f"{len(flagged.head(25))} of the top 25 LightGBM winners have at least one audit flag."
        )
    price_bucket = buckets[
        buckets["model_name"].eq("lightgbm_rank_blend")
        & buckets["bucket_type"].eq("price_bucket")
    ].sort_values("total_pnl", ascending=False)
    if not price_bucket.empty:
        best = price_bucket.iloc[0]
        notes.append(
            f"The largest price bucket contributor is {best['bucket_value']} with "
            f"{best['total_pnl']:.4f} PnL."
        )
    return "\n\n".join(notes)


def _scenario(frame: pd.DataFrame, scenario: str) -> pd.Series:
    rows = frame[frame["scenario"].eq(scenario)]
    if rows.empty:
        raise ValueError(f"Scenario not found: {scenario}")
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 21 v2 robustness audit.")
    parser.add_argument("--accepted-input", type=Path, default=DEFAULT_ACCEPTED_INPUT)
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--scenario-output", type=Path, default=DEFAULT_SCENARIO_OUTPUT)
    parser.add_argument("--top-trades-output", type=Path, default=DEFAULT_TOP_TRADES_OUTPUT)
    parser.add_argument("--bucket-output", type=Path, default=DEFAULT_BUCKET_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_day21_v2_robustness(
        accepted_input=args.accepted_input,
        features_input=args.features_input,
        scenario_output=args.scenario_output,
        top_trades_output=args.top_trades_output,
        bucket_output=args.bucket_output,
        report_output=args.report_output,
    )
    print(f"Scenario rows: {result.scenario_rows}")
    print(f"Top-trade rows: {result.top_trade_rows}")
    print(f"Bucket rows: {result.bucket_rows}")
    print(f"Wrote scenarios: {_display_path(result.scenario_output)}")
    print(f"Wrote top trades: {_display_path(result.top_trades_output)}")
    print(f"Wrote buckets: {_display_path(result.bucket_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
