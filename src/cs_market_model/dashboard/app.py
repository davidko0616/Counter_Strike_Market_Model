"""Streamlit dashboard for the CS market model research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from cs_market_model.config import PROJECT_ROOT

DEFAULT_POLICY_VARIANT = "min_price_5_costed"
DEFAULT_SCENARIO = "paper_trade_5_plus_gates"
DEFAULT_MIN_ENTRY_PRICE = 5.0
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "day10_11_lightgbm_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features_day5_v1.parquet"
ACCEPTED_TRADES_PATH = (
    PROJECT_ROOT / "reports" / "backtests" / "day25_paper_trade_5_plus_gates_accepted_trades.csv"
)
EQUITY_PATH = PROJECT_ROOT / "reports" / "backtests" / "day12_13_daily_equity.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "day25_paper_trade_rejection_summary.csv"
ITEM_ATTRIBUTION_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "day23_policy_item_attribution.csv"
)
CAPACITY_SENSITIVITY_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "day25_paper_trade_rejection_summary.csv"
)
CAPACITY_PERIOD_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "day25_paper_trade_rejection_period_attribution.csv"
)
CSFLOAT_COVERAGE_PATH = PROJECT_ROOT / "reports" / "tables" / "day19_csfloat_coverage.csv"
CSFLOAT_ABLATION_PATH = PROJECT_ROOT / "reports" / "tables" / "day18_csfloat_ablation_setup.csv"
SIGNAL_EXPLANATIONS_PATH = PROJECT_ROOT / "reports" / "tables" / "signal_explanations.csv"
PAPER_TRADE_AB_PATH = PROJECT_ROOT / "reports" / "tables" / "paper_trade_ab_test_summary.csv"
MODEL_HEALTH_MONITOR_PATH = PROJECT_ROOT / "reports" / "tables" / "model_health_monitor.csv"
MODEL_HEALTH_STATUS_PATH = PROJECT_ROOT / "reports" / "tables" / "model_health_status.json"
MAX_FRESHNESS_DAYS = 7
POLICY_ARTIFACT_PATHS = {
    "Predictions": PREDICTIONS_PATH,
    "Features": FEATURES_PATH,
    "$5+ Accepted Trades": ACCEPTED_TRADES_PATH,
    "$5+ Paper-Trade Summary": SUMMARY_PATH,
    "$5+ Item Attribution": ITEM_ATTRIBUTION_PATH,
    "$5+ Paper-Trade Gates": CAPACITY_SENSITIVITY_PATH,
    "$5+ Paper-Trade Periods": CAPACITY_PERIOD_PATH,
    "CSFloat Coverage": CSFLOAT_COVERAGE_PATH,
    "CSFloat Ablation": CSFLOAT_ABLATION_PATH,
    "Signal Explanations": SIGNAL_EXPLANATIONS_PATH,
    "Paper-Trade A/B": PAPER_TRADE_AB_PATH,
    "Model Health": MODEL_HEALTH_MONITOR_PATH,
}
COMMAND_HINTS = {
    "csfloat": ".\\.venv\\Scripts\\python.exe -m cs_market_model.research.day19_csfloat_coverage",
    "explanations": ".\\.venv\\Scripts\\python.exe -m cs_market_model.models.explain",
    "paper_trade": ".\\.venv\\Scripts\\python.exe -m cs_market_model.research.paper_trade_ab_test",
    "health": ".\\.venv\\Scripts\\python.exe -m cs_market_model.research.model_health_monitor",
}


def main() -> None:
    st.set_page_config(page_title="CS Market Model", layout="wide")
    st.title("CS Market Model")

    artifacts = load_artifacts()
    if artifacts["predictions"].empty or artifacts["features"].empty:
        st.error("Required model artifacts are missing.")
        return

    model_names = sorted(artifacts["predictions"]["model_name"].dropna().unique())
    default_model = (
        "lightgbm_rank_blend" if "lightgbm_rank_blend" in model_names else model_names[0]
    )
    default_threshold = selected_policy_threshold(artifacts["summary"], default_model)
    with st.sidebar:
        st.caption("Policy: $5+ conservative candidate")
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()
        model_name = st.selectbox("Model", model_names, index=model_names.index(default_model))
        min_score = st.slider("Minimum Score", 0.0, 1.0, default_threshold, 0.01)
        min_liquidity = st.slider("Minimum Liquidity", 0.0, 1.0, 0.0, 0.05)
        min_entry_price = st.number_input(
            "Minimum Entry Price",
            min_value=0.0,
            value=DEFAULT_MIN_ENTRY_PRICE,
            step=1.0,
        )
        category = st.multiselect(
            "Category",
            sorted(_latest_features(artifacts["features"])["category"].dropna().unique()),
        )
        wear = st.multiselect(
            "Wear",
            sorted(_latest_features(artifacts["features"])["wear"].dropna().unique()),
        )
        latest_feature_rows = _latest_features(artifacts["features"])
        price_buckets = st.multiselect(
            "Price Bucket",
            sorted(
                latest_feature_rows["close"]
                .map(price_bucket)
                .dropna()
                .unique()
                .tolist()
            )
            if "close" in latest_feature_rows.columns
            else [],
        )

    signal_table = build_signal_table(
        artifacts,
        model_name=model_name,
        min_score=min_score,
        min_liquidity=min_liquidity,
        min_entry_price=min_entry_price,
        categories=category,
        wears=wear,
        price_buckets=price_buckets,
    )
    accepted_signals = signal_table[signal_table["candidate_status"].eq("accepted")].copy()
    rejected_signals = signal_table[signal_table["candidate_status"].eq("rejected")].copy()
    policy_summary_for_model(artifacts["summary"], model_name)
    capacity_status = capacity_summary_for_model(artifacts["capacity_sensitivity"], model_name)
    default_capacity = capacity_status[capacity_status["scenario"].eq(DEFAULT_SCENARIO)]
    latest_regime = latest_market_regime(artifacts["features"])
    policy_decision = capacity_decision(capacity_status)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Policy", "$5+")
    metric_cols[1].metric("Candidate Status", policy_decision)
    metric_cols[2].metric(
        "Accepted Trades", _metric_value(default_capacity, "accepted_count", integer=True)
    )
    metric_cols[3].metric("PnL", _metric_value(default_capacity, "accepted_total_pnl"))
    metric_cols[4].metric("Profit Factor", _metric_value(default_capacity, "profit_factor"))
    metric_cols[5].metric("Regime", latest_regime)

    (
        tab_signals,
        tab_rejected,
        tab_capacity,
        tab_csfloat,
        tab_explanations,
        tab_paper_trade,
        tab_health,
        tab_freshness,
        tab_item,
        tab_regime,
    ) = st.tabs(
        [
            "Signals",
            "Rejected",
            "Capacity",
            "CSFloat",
            "Explanations",
            "Paper Trade",
            "Health",
            "Freshness",
            "Item Detail",
            "Regime",
        ]
    )

    with tab_signals:
        cols = st.columns(4)
        cols[0].metric("Current Accepted", f"{len(accepted_signals):,}")
        cols[1].metric("Current Rejected", f"{len(rejected_signals):,}")
        cols[2].metric("Min Price", f"${min_entry_price:.2f}")
        cols[3].metric("Score Gate", f"{min_score:.2f}")
        st.dataframe(
            accepted_signals.head(50),
            use_container_width=True,
            hide_index=True,
            column_config=signal_column_config(),
        )
        st.download_button(
            "Export CSV",
            accepted_signals.to_csv(index=False),
            file_name="accepted_signals.csv",
            mime="text/csv",
        )
        st.download_button(
            "Export Daily Summary",
            daily_signal_markdown(accepted_signals),
            file_name="daily_signal_summary.md",
            mime="text/markdown",
        )

    with tab_rejected:
        st.dataframe(
            rejected_signals.head(200),
            use_container_width=True,
            hide_index=True,
            column_config=signal_column_config(),
        )
        st.download_button(
            "Export CSV",
            rejected_signals.to_csv(index=False),
            file_name="rejected_signals.csv",
            mime="text/csv",
        )

    with tab_capacity:
        render_capacity_status(
            artifacts["capacity_sensitivity"], artifacts["capacity_period"], model_name
        )

    with tab_csfloat:
        render_csfloat_status(artifacts["csfloat_coverage"], artifacts["csfloat_ablation"])

    with tab_explanations:
        render_signal_explanations(artifacts["signal_explanations"], signal_table)

    with tab_paper_trade:
        render_paper_trade_status(artifacts["paper_trade_ab"])

    with tab_health:
        render_health_status(artifacts["model_health"], artifacts["model_health_status"])

    with tab_freshness:
        render_freshness(POLICY_ARTIFACT_PATHS)

    with tab_item:
        selectable_items = accepted_signals["market_hash_name"].tolist()
        if not selectable_items:
            selectable_items = sorted(artifacts["features"]["market_hash_name"].dropna().unique())
        selected_item = st.selectbox("Item", selectable_items)
        render_item_detail(artifacts, selected_item, model_name)

    with tab_regime:
        render_regime(artifacts["features"])


def load_artifacts() -> dict[str, Any]:
    """Load generated research artifacts for the dashboard."""
    return {
        "predictions": _read_table(PREDICTIONS_PATH),
        "features": _read_table(FEATURES_PATH),
        "accepted": _read_table(ACCEPTED_TRADES_PATH),
        "equity": _read_table(EQUITY_PATH),
        "summary": _read_table(SUMMARY_PATH),
        "item_attribution": _read_table(ITEM_ATTRIBUTION_PATH),
        "capacity_sensitivity": _read_table(CAPACITY_SENSITIVITY_PATH),
        "capacity_period": _read_table(CAPACITY_PERIOD_PATH),
        "csfloat_coverage": _read_table(CSFLOAT_COVERAGE_PATH),
        "csfloat_ablation": _read_table(CSFLOAT_ABLATION_PATH),
        "signal_explanations": _read_table(SIGNAL_EXPLANATIONS_PATH),
        "paper_trade_ab": _read_table(PAPER_TRADE_AB_PATH),
        "model_health": _read_table(MODEL_HEALTH_MONITOR_PATH),
        "model_health_status": _read_json(MODEL_HEALTH_STATUS_PATH),
    }


def build_signal_table(
    artifacts: dict[str, Any],
    model_name: str,
    min_score: float,
    min_liquidity: float,
    min_entry_price: float,
    categories: list[str],
    wears: list[str],
    price_buckets: list[str] | None = None,
) -> pd.DataFrame:
    """Build the latest scored signal table with historical attribution context."""
    predictions = artifacts["predictions"]
    model_predictions = predictions[predictions["model_name"].eq(model_name)].copy()
    if model_predictions.empty:
        return pd.DataFrame()
    model_predictions["timestamp"] = pd.to_datetime(model_predictions["timestamp"], utc=True)
    latest_timestamp = model_predictions["timestamp"].max()
    latest = model_predictions[model_predictions["timestamp"].eq(latest_timestamp)].copy()

    latest_features = _latest_features(artifacts["features"])
    latest = latest.merge(
        latest_features[
            [
                "market_hash_name",
                "category",
                "weapon_type",
                "wear",
                "close",
                "rarity",
                "row_coverage_30d",
                "effective_staleness_days",
            ]
            + [
                column
                for column in [
                    "cs2_index_regime_bull",
                    "cs2_index_regime_neutral",
                    "cs2_index_regime_bear",
                ]
                if column in latest_features.columns
            ]
        ],
        on="market_hash_name",
        how="left",
        suffixes=("", "_feature"),
    )
    for column in [
        "category",
        "weapon_type",
        "wear",
        "close",
        "rarity",
        "row_coverage_30d",
        "effective_staleness_days",
    ]:
        latest[column] = _coalesce_columns(latest, column, f"{column}_feature")
        latest = latest.drop(columns=[f"{column}_feature"], errors="ignore")
    attribution = artifacts["item_attribution"]
    if not attribution.empty:
        item_stats = normalized_item_attribution(attribution, model_name)
        latest = latest.merge(item_stats, on="market_hash_name", how="left")

    latest["close"] = pd.to_numeric(latest["close"], errors="coerce")
    latest["price_bucket"] = latest["close"].map(price_bucket)
    latest["strong_buy_score"] = pd.to_numeric(latest["strong_buy_score"], errors="coerce")
    latest["liquidity_quality_score_30d"] = pd.to_numeric(
        latest["liquidity_quality_score_30d"],
        errors="coerce",
    ).fillna(0)
    latest["pass_price"] = latest["close"].ge(min_entry_price)
    latest["pass_score"] = latest["strong_buy_score"].ge(min_score)
    latest["pass_liquidity"] = latest["liquidity_quality_score_30d"].ge(min_liquidity)
    latest["rejection_reason"] = latest.apply(
        lambda row: candidate_rejection_reason(row, min_entry_price, min_score, min_liquidity),
        axis=1,
    )
    latest["candidate_status"] = np.where(latest["rejection_reason"].eq(""), "accepted", "rejected")
    if categories:
        latest = latest[latest["category"].isin(categories)]
    if wears:
        latest = latest[latest["wear"].isin(wears)]
    if price_buckets:
        latest = latest[latest["price_bucket"].isin(price_buckets)]
    columns = [
        "candidate_status",
        "rejection_reason",
        "market_hash_name",
        "weapon_type",
        "category",
        "wear",
        "rarity",
        "close",
        "price_bucket",
        "strong_buy_score",
        "liquidity_quality_score_30d",
        "row_coverage_30d",
        "effective_staleness_days",
        "historical_avg_return",
        "accepted_win_rate",
        "accepted_trade_count",
        "accepted_pnl",
        "csfloat_snapshot_available",
        "csfloat_listing_count",
    ]
    for column in columns:
        if column not in latest.columns:
            latest[column] = np.nan
    return (
        latest[columns]
        .sort_values(
            ["candidate_status", "strong_buy_score"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def render_item_detail(
    artifacts: dict[str, pd.DataFrame],
    market_hash_name: str,
    model_name: str,
) -> None:
    """Render price history and accepted-trade markers for one item."""
    features = artifacts["features"].copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    item_features = features[features["market_hash_name"].eq(market_hash_name)].sort_values(
        "timestamp"
    )
    if item_features.empty:
        st.warning("No feature rows found for this item.")
        return
    st.subheader(market_hash_name)
    st.line_chart(item_features.set_index("timestamp")["close"])

    accepted = artifacts["accepted"].copy()
    if not accepted.empty:
        accepted = accepted[
            accepted["model_name"].eq(model_name)
            & accepted["market_hash_name"].eq(market_hash_name)
        ].copy()
    if not accepted.empty:
        accepted["entry_timestamp"] = pd.to_datetime(accepted["entry_timestamp"], utc=True)
        st.dataframe(
            accepted[
                [
                    "entry_timestamp",
                    "exit_timestamp",
                    "strong_buy_score",
                    "realized_net_return",
                    "pnl",
                    "label_market_regime",
                ]
            ].sort_values("entry_timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def render_capacity_status(
    capacity: pd.DataFrame,
    capacity_period: pd.DataFrame,
    model_name: str,
) -> None:
    """Render Day 24 capacity stress status for the selected model."""
    if capacity.empty:
        st.warning("No capacity sensitivity table found.")
        return
    frame = capacity[capacity["model_name"].eq(model_name)].copy()
    if frame.empty:
        st.warning("No capacity sensitivity rows found for this model.")
        return
    frame = frame.sort_values("accepted_total_pnl", ascending=True)
    survived = bool(frame["survives_capacity_stress"].fillna(False).all())
    cols = st.columns(4)
    cols[0].metric("Stress Status", "survives" if survived else "fails")
    cols[1].metric("Worst PnL", f"{frame['accepted_total_pnl'].min():.4f}")
    cols[2].metric("Worst Drawdown", f"{frame['accepted_max_drawdown'].min():.2%}")
    cols[3].metric("Max Top-2 Period Share", f"{frame['top_2_period_pnl_share'].max():.2%}")
    display_columns = [
        "scenario",
        "accepted_count",
        "capacity_rejected_count",
        "accepted_total_pnl",
        "profit_factor",
        "accepted_max_drawdown",
        "top_2_period_pnl_share",
        "positive_period_share",
        "survives_capacity_stress",
        "max_open_positions",
        "max_open_positions_per_bucket",
    ]
    st.dataframe(
        frame[[column for column in display_columns if column in frame.columns]],
        use_container_width=True,
        hide_index=True,
    )
    if not capacity_period.empty:
        periods = capacity_period[capacity_period["model_name"].eq(model_name)].copy()
        st.dataframe(
            periods.sort_values(["scenario", "accepted_total_pnl"])
            .groupby("scenario", group_keys=False)
            .head(5),
            use_container_width=True,
            hide_index=True,
        )


def render_csfloat_status(coverage: pd.DataFrame, ablation: pd.DataFrame) -> None:
    """Render CSFloat point-in-time coverage and ablation readiness."""
    if coverage.empty and ablation.empty:
        st.warning(f"No CSFloat coverage artifacts found. Run `{COMMAND_HINTS['csfloat']}`.")
        return
    if not coverage.empty:
        st.subheader("Coverage")
        st.dataframe(coverage, use_container_width=True, hide_index=True)
    if not ablation.empty:
        st.subheader("Ablation Readiness")
        st.dataframe(ablation, use_container_width=True, hide_index=True)


def render_signal_explanations(explanations: pd.DataFrame, signals: pd.DataFrame) -> None:
    """Render top feature explanations for current signals."""
    if explanations.empty:
        st.warning(f"No signal explanations found. Run `{COMMAND_HINTS['explanations']}`.")
        return
    selectable_items = signals["market_hash_name"].dropna().astype(str).tolist()
    if selectable_items:
        selected = st.selectbox("Signal", selectable_items)
        explanations = explanations[explanations["market_hash_name"].astype(str).eq(selected)]
    st.dataframe(explanations, use_container_width=True, hide_index=True)


def render_paper_trade_status(summary: pd.DataFrame) -> None:
    """Render paper-trade A/B review status."""
    if summary.empty:
        st.warning(f"No paper-trade A/B summary found. Run `{COMMAND_HINTS['paper_trade']}`.")
        return
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_health_status(monitor: pd.DataFrame, status_payload: dict[str, Any]) -> None:
    """Render model/data health badge and details."""
    status = str(status_payload.get("status", "unknown")) if status_payload else "unknown"
    st.metric("Health", status)
    if not status_payload:
        st.warning(f"No health JSON found. Run `{COMMAND_HINTS['health']}`.")
    elif status_payload.get("warnings"):
        st.write(status_payload["warnings"])
    if monitor.empty:
        st.warning(f"No health monitor table found. Run `{COMMAND_HINTS['health']}`.")
        return
    st.dataframe(monitor, use_container_width=True, hide_index=True)


def render_freshness(paths: dict[str, Path]) -> None:
    """Render local artifact freshness and row counts."""
    st.dataframe(artifact_freshness_table(paths), use_container_width=True, hide_index=True)


def render_portfolio(equity: pd.DataFrame, model_name: str) -> None:
    """Render equity curve and current drawdown for the selected model."""
    if equity.empty:
        st.warning("No equity file found.")
        return
    frame = equity[equity["model_name"].eq(model_name)].copy()
    if frame.empty:
        st.warning("No equity rows found for this model.")
        return
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date")
    running_peak = frame["equity"].cummax()
    frame["drawdown"] = frame["equity"].div(running_peak).sub(1.0)
    cols = st.columns(3)
    cols[0].metric("Final Equity", f"{frame['equity'].iloc[-1]:.2f}")
    cols[1].metric("Max Drawdown", f"{frame['drawdown'].min():.2%}")
    cols[2].metric("Open Positions", f"{int(frame['open_positions'].iloc[-1])}")
    st.line_chart(frame.set_index("date")[["equity"]])
    st.area_chart(frame.set_index("date")[["drawdown"]])


def render_regime(features: pd.DataFrame) -> None:
    """Render current market-index regime and latest index features."""
    latest = _latest_features(features)
    if latest.empty:
        st.warning("No feature rows found.")
        return
    numeric_cols = [
        "cs2_index_return_7d",
        "cs2_index_return_30d",
        "cs2_index_drawdown_30d",
        "cs2_index_ma_distance_30d",
        "row_coverage_30d",
        "liquidity_quality_score_30d",
    ]
    regime_counts = {
        "bull": int(latest.get("cs2_index_regime_bull", pd.Series(False)).fillna(False).sum()),
        "neutral": int(
            latest.get("cs2_index_regime_neutral", pd.Series(False)).fillna(False).sum()
        ),
        "bear": int(latest.get("cs2_index_regime_bear", pd.Series(False)).fillna(False).sum()),
    }
    cols = st.columns(3)
    cols[0].metric("Bull Rows", regime_counts["bull"])
    cols[1].metric("Neutral Rows", regime_counts["neutral"])
    cols[2].metric("Bear Rows", regime_counts["bear"])
    available = [column for column in numeric_cols if column in latest.columns]
    st.dataframe(
        latest[["market_hash_name", *available]], use_container_width=True, hide_index=True
    )


def latest_market_regime(features: pd.DataFrame) -> str:
    """Return the dominant latest CS2 index regime."""
    latest = _latest_features(features)
    if latest.empty:
        return "unknown"
    counts = {
        "bull": int(latest.get("cs2_index_regime_bull", pd.Series(False)).fillna(False).sum()),
        "neutral": int(
            latest.get("cs2_index_regime_neutral", pd.Series(False)).fillna(False).sum()
        ),
        "bear": int(latest.get("cs2_index_regime_bear", pd.Series(False)).fillna(False).sum()),
    }
    return max(counts, key=counts.get)


def selected_policy_threshold(summary: pd.DataFrame, model_name: str) -> float:
    """Return the corrected $5+ mean threshold for the sidebar default."""
    model_summary = policy_summary_for_model(summary, model_name)
    threshold_column = (
        "selected_threshold_mean"
        if "selected_threshold_mean" in model_summary.columns
        else "mean_selected_score_threshold"
    )
    if model_summary.empty or threshold_column not in model_summary.columns:
        return 0.85
    value = pd.to_numeric(model_summary[threshold_column], errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else 0.85


def policy_summary_for_model(summary: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return corrected $5+ policy summary rows for a model."""
    if summary.empty:
        return summary
    frame = summary[summary["model_name"].eq(model_name)].copy()
    if "policy_variant" in frame.columns:
        frame = frame[frame["policy_variant"].eq(DEFAULT_POLICY_VARIANT)].copy()
    if "scenario" in frame.columns and frame["scenario"].eq(DEFAULT_SCENARIO).any():
        frame = frame[frame["scenario"].eq(DEFAULT_SCENARIO)].copy()
    return frame


def normalized_item_attribution(attribution: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return item attribution with dashboard-standard column names."""
    frame = attribution[attribution["model_name"].eq(model_name)].copy()
    if "policy_variant" in frame.columns:
        frame = frame[frame["policy_variant"].eq(DEFAULT_POLICY_VARIANT)].copy()
    rename_map = {
        "trade_count": "accepted_trade_count",
        "average_trade_return": "historical_avg_return",
        "win_rate": "accepted_win_rate",
        "total_pnl": "accepted_pnl",
        "accepted_avg_return": "historical_avg_return",
    }
    frame = frame.rename(columns=rename_map)
    columns = [
        "market_hash_name",
        "accepted_trade_count",
        "historical_avg_return",
        "accepted_win_rate",
        "accepted_pnl",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def capacity_summary_for_model(capacity: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return Day 24 capacity rows for a model."""
    if capacity.empty:
        return capacity
    return capacity[capacity["model_name"].eq(model_name)].copy()


def capacity_decision(capacity: pd.DataFrame) -> str:
    """Return a concise policy decision based on Day 24 stress rows."""
    if capacity.empty or "survives_capacity_stress" not in capacity.columns:
        return "unknown"
    if capacity["survives_capacity_stress"].fillna(False).all():
        return "paper trade"
    return "tighten gates"


def candidate_rejection_reason(
    row: pd.Series,
    min_entry_price: float,
    min_score: float,
    min_liquidity: float,
) -> str:
    """Return rejection reasons for current dashboard candidates."""
    reasons: list[str] = []
    close = row.get("close")
    if pd.isna(close):
        reasons.append("missing_price")
    elif float(close) < min_entry_price:
        reasons.append("below_min_price")
    score = row.get("strong_buy_score")
    if pd.isna(score) or float(score) < min_score:
        reasons.append("low_score")
    liquidity = row.get("liquidity_quality_score_30d")
    if pd.isna(liquidity) or float(liquidity) < min_liquidity:
        reasons.append("low_liquidity")
    return "|".join(reasons)


def price_bucket(value: object) -> str:
    """Return dashboard price bucket for a candidate price."""
    price = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(price):
        return "unknown"
    price = float(price)
    if price < 5:
        return "<$5"
    if price < 25:
        return "$5-$25"
    if price < 100:
        return "$25-$100"
    return "$100+"


def daily_signal_markdown(signals: pd.DataFrame) -> str:
    """Return a compact Markdown export for the current signal table."""
    if signals.empty:
        return "# Daily Signal Summary\n\nNo accepted signals.\n"
    display = signals.head(25).copy()
    lines = [
        "# Daily Signal Summary",
        "",
        f"Accepted signals: {len(signals):,}",
        "",
        "| Item | Price | Score | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("market_hash_name", "")),
                    _format_number(row.get("close"), prefix="$"),
                    _format_number(row.get("strong_buy_score")),
                    str(row.get("rejection_reason", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def artifact_freshness_table(
    paths: dict[str, Path],
    *,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return local artifact freshness with a red/yellow/green status field."""
    resolved_now = now or pd.Timestamp.now(tz="UTC")
    rows = []
    for label, path in paths.items():
        exists = path.exists()
        modified = pd.NaT
        age_days = np.nan
        status = "missing"
        if exists:
            modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
            age_days = float((resolved_now - modified).total_seconds() / 86400.0)
            status = "fresh" if age_days <= MAX_FRESHNESS_DAYS else "stale"
        rows.append(
            {
                "artifact": label,
                "path": _display_path(path),
                "exists": exists,
                "status": status,
                "age_days": age_days,
                "modified_utc": modified,
                "rows": _row_count(path) if exists else 0,
            }
        )
    return pd.DataFrame(rows)


def _latest_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    if frame.empty or "timestamp" not in frame.columns:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = frame["timestamp"].max()
    return frame[frame["timestamp"].eq(latest_timestamp)].copy()


def _coalesce_columns(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    if primary not in frame.columns and fallback not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    if primary not in frame.columns:
        return frame[fallback]
    if fallback not in frame.columns:
        return frame[primary]
    return frame[primary].replace("", pd.NA).fillna(frame[fallback])


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix != ".csv":
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _row_count(path: Path) -> int:
    table = _read_table(path)
    return int(len(table)) if not table.empty else 0


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_number(value: object, *, prefix: str = "") -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{prefix}{float(numeric):.3f}"


def _metric_value(
    frame: pd.DataFrame,
    column: str,
    *,
    percent: bool = False,
    integer: bool = False,
) -> str:
    if frame.empty or column not in frame.columns:
        return "n/a"
    value = pd.to_numeric(frame[column], errors="coerce").iloc[0]
    if pd.isna(value):
        return "n/a"
    if percent:
        return f"{value:.1%}"
    if integer:
        return f"{int(value):,}"
    return f"{value:.3f}"


def signal_column_config() -> dict[str, object]:
    """Return consistent table formatting for current candidate rows."""
    return {
        "strong_buy_score": st.column_config.ProgressColumn(
            "Score",
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        ),
        "liquidity_quality_score_30d": st.column_config.ProgressColumn(
            "Liquidity",
            min_value=0.0,
            max_value=1.0,
            format="%.2f",
        ),
        "historical_avg_return": st.column_config.NumberColumn(
            "Hist. Avg Return",
            format="%.4f",
        ),
        "accepted_pnl": st.column_config.NumberColumn("Hist. PnL", format="%.4f"),
        "close": st.column_config.NumberColumn("Price", format="$%.2f"),
    }


if __name__ == "__main__":
    main()
