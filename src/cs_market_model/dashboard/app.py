"""Streamlit dashboard for the CS market model research artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cs_market_model.config import PROJECT_ROOT

DEFAULT_POLICY_VARIANT = "min_price_5_costed"
DEFAULT_SCENARIO = "current_day22"
DEFAULT_MIN_ENTRY_PRICE = 5.0
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "day10_11_lightgbm_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features_day5_v1.parquet"
ACCEPTED_TRADES_PATH = (
    PROJECT_ROOT / "reports" / "backtests" / "day24_min_price_5_current_day22_accepted_trades.csv"
)
EQUITY_PATH = PROJECT_ROOT / "reports" / "backtests" / "day12_13_daily_equity.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "day17_policy_variant_summary.csv"
ITEM_ATTRIBUTION_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "day17_policy_variant_item_attribution.csv"
)
CAPACITY_SENSITIVITY_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "day24_min_price_5_capacity_sensitivity.csv"
)
CAPACITY_PERIOD_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "day24_min_price_5_capacity_period_attribution.csv"
)
POLICY_ARTIFACT_PATHS = {
    "Predictions": PREDICTIONS_PATH,
    "Features": FEATURES_PATH,
    "$5+ Accepted Trades": ACCEPTED_TRADES_PATH,
    "$5+ Policy Summary": SUMMARY_PATH,
    "$5+ Item Attribution": ITEM_ATTRIBUTION_PATH,
    "$5+ Capacity Stress": CAPACITY_SENSITIVITY_PATH,
    "$5+ Capacity Periods": CAPACITY_PERIOD_PATH,
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

    signal_table = build_signal_table(
        artifacts,
        model_name=model_name,
        min_score=min_score,
        min_liquidity=min_liquidity,
        min_entry_price=min_entry_price,
        categories=category,
        wears=wear,
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

    tab_signals, tab_rejected, tab_capacity, tab_freshness, tab_item, tab_regime = st.tabs(
        ["Signals", "Rejected", "Capacity", "Freshness", "Item Detail", "Regime"]
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

    with tab_rejected:
        st.dataframe(
            rejected_signals.head(200),
            use_container_width=True,
            hide_index=True,
            column_config=signal_column_config(),
        )

    with tab_capacity:
        render_capacity_status(
            artifacts["capacity_sensitivity"], artifacts["capacity_period"], model_name
        )

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


def load_artifacts() -> dict[str, pd.DataFrame]:
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
    }


def build_signal_table(
    artifacts: dict[str, pd.DataFrame],
    model_name: str,
    min_score: float,
    min_liquidity: float,
    min_entry_price: float,
    categories: list[str],
    wears: list[str],
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
        item_stats = attribution[
            attribution["model_name"].eq(model_name)
            & attribution["policy_variant"].eq(DEFAULT_POLICY_VARIANT)
        ][
            [
                "market_hash_name",
                "accepted_trade_count",
                "accepted_avg_return",
                "accepted_win_rate",
                "accepted_pnl",
            ]
        ].rename(columns={"accepted_avg_return": "historical_avg_return"})
        latest = latest.merge(item_stats, on="market_hash_name", how="left")

    latest["close"] = pd.to_numeric(latest["close"], errors="coerce")
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
    columns = [
        "candidate_status",
        "rejection_reason",
        "market_hash_name",
        "weapon_type",
        "category",
        "wear",
        "rarity",
        "close",
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


def render_freshness(paths: dict[str, Path]) -> None:
    """Render local artifact freshness and row counts."""
    rows = []
    for label, path in paths.items():
        exists = path.exists()
        modified = (
            pd.Timestamp(path.stat().st_mtime, unit="s").tz_localize("UTC") if exists else pd.NaT
        )
        rows.append(
            {
                "artifact": label,
                "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
                "exists": exists,
                "modified_utc": modified,
                "rows": len(_read_table(path)) if exists else 0,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
    if model_summary.empty or "selected_threshold_mean" not in model_summary.columns:
        return 0.85
    value = pd.to_numeric(model_summary["selected_threshold_mean"], errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else 0.85


def policy_summary_for_model(summary: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return corrected $5+ policy summary rows for a model."""
    if summary.empty:
        return summary
    frame = summary[summary["model_name"].eq(model_name)].copy()
    if "policy_variant" in frame.columns:
        frame = frame[frame["policy_variant"].eq(DEFAULT_POLICY_VARIANT)].copy()
    return frame


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
    return pd.read_csv(path)


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
