"""Streamlit dashboard for the CS market model research artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cs_market_model.config import PROJECT_ROOT

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "day10_11_lightgbm_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features_day5_v1.parquet"
ACCEPTED_TRADES_PATH = PROJECT_ROOT / "reports" / "backtests" / "day16_walk_forward_accepted_trades.csv"
EQUITY_PATH = PROJECT_ROOT / "reports" / "backtests" / "day12_13_daily_equity.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "day16_walk_forward_threshold_summary.csv"
ITEM_ATTRIBUTION_PATH = PROJECT_ROOT / "reports" / "tables" / "day17_item_attribution.csv"


def main() -> None:
    st.set_page_config(page_title="CS Market Model", layout="wide")
    st.title("CS Market Model")

    artifacts = load_artifacts()
    if artifacts["predictions"].empty or artifacts["features"].empty:
        st.error("Required model artifacts are missing.")
        return

    model_names = sorted(artifacts["predictions"]["model_name"].dropna().unique())
    default_model = "lightgbm_rank_blend" if "lightgbm_rank_blend" in model_names else model_names[0]
    with st.sidebar:
        model_name = st.selectbox("Model", model_names, index=model_names.index(default_model))
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.85, 0.01)
        min_liquidity = st.slider("Minimum Liquidity", 0.0, 1.0, 0.0, 0.05)
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
        categories=category,
        wears=wear,
    )
    model_summary = artifacts["summary"][artifacts["summary"]["model_name"].eq(model_name)]
    latest_regime = latest_market_regime(artifacts["features"])

    metric_cols = st.columns(5)
    metric_cols[0].metric("Accepted Trades", _metric_value(model_summary, "accepted_count", integer=True))
    metric_cols[1].metric("PnL", _metric_value(model_summary, "accepted_total_pnl"))
    metric_cols[2].metric("Win Rate", _metric_value(model_summary, "accepted_win_rate", percent=True))
    metric_cols[3].metric("Profit Factor", _metric_value(model_summary, "profit_factor"))
    metric_cols[4].metric("Regime", latest_regime)

    tab_signals, tab_item, tab_portfolio, tab_regime = st.tabs(
        ["Signals", "Item Detail", "Portfolio", "Regime"]
    )

    with tab_signals:
        st.dataframe(
            signal_table.head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
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
                    format="%.2f",
                ),
            },
        )

    with tab_item:
        selectable_items = signal_table["market_hash_name"].tolist()
        if not selectable_items:
            selectable_items = sorted(artifacts["features"]["market_hash_name"].dropna().unique())
        selected_item = st.selectbox("Item", selectable_items)
        render_item_detail(artifacts, selected_item, model_name)

    with tab_portfolio:
        render_portfolio(artifacts["equity"], model_name)

    with tab_regime:
        render_regime(artifacts["features"])


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict[str, pd.DataFrame]:
    """Load generated research artifacts for the dashboard."""
    return {
        "predictions": _read_table(PREDICTIONS_PATH),
        "features": _read_table(FEATURES_PATH),
        "accepted": _read_table(ACCEPTED_TRADES_PATH),
        "equity": _read_table(EQUITY_PATH),
        "summary": _read_table(SUMMARY_PATH),
        "item_attribution": _read_table(ITEM_ATTRIBUTION_PATH),
    }


def build_signal_table(
    artifacts: dict[str, pd.DataFrame],
    model_name: str,
    min_score: float,
    min_liquidity: float,
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
            ]
        ],
        on="market_hash_name",
        how="left",
    )
    attribution = artifacts["item_attribution"]
    if not attribution.empty:
        item_stats = attribution[attribution["model_name"].eq(model_name)][
            ["market_hash_name", "accepted_trade_count", "accepted_avg_return", "accepted_win_rate"]
        ].rename(columns={"accepted_avg_return": "historical_avg_return"})
        latest = latest.merge(item_stats, on="market_hash_name", how="left")

    latest["pass_score"] = pd.to_numeric(latest["strong_buy_score"], errors="coerce").ge(min_score)
    latest["pass_liquidity"] = pd.to_numeric(
        latest["liquidity_quality_score_30d"],
        errors="coerce",
    ).fillna(0).ge(min_liquidity)
    latest["rejection_status"] = np.where(
        latest["pass_score"] & latest["pass_liquidity"],
        "accepted",
        "rejected",
    )
    if categories:
        latest = latest[latest["category"].isin(categories)]
    if wears:
        latest = latest[latest["wear"].isin(wears)]
    latest = latest[latest["rejection_status"].eq("accepted")]
    columns = [
        "market_hash_name",
        "weapon_type",
        "category",
        "wear",
        "close",
        "strong_buy_score",
        "liquidity_quality_score_30d",
        "historical_avg_return",
        "accepted_win_rate",
        "accepted_trade_count",
        "csfloat_snapshot_available",
        "csfloat_listing_count",
        "rejection_status",
    ]
    for column in columns:
        if column not in latest.columns:
            latest[column] = np.nan
    return latest[columns].sort_values("strong_buy_score", ascending=False).reset_index(drop=True)


def render_item_detail(
    artifacts: dict[str, pd.DataFrame],
    market_hash_name: str,
    model_name: str,
) -> None:
    """Render price history and accepted-trade markers for one item."""
    features = artifacts["features"].copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    item_features = features[features["market_hash_name"].eq(market_hash_name)].sort_values("timestamp")
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
        "neutral": int(latest.get("cs2_index_regime_neutral", pd.Series(False)).fillna(False).sum()),
        "bear": int(latest.get("cs2_index_regime_bear", pd.Series(False)).fillna(False).sum()),
    }
    cols = st.columns(3)
    cols[0].metric("Bull Rows", regime_counts["bull"])
    cols[1].metric("Neutral Rows", regime_counts["neutral"])
    cols[2].metric("Bear Rows", regime_counts["bear"])
    available = [column for column in numeric_cols if column in latest.columns]
    st.dataframe(latest[["market_hash_name", *available]], use_container_width=True, hide_index=True)


def latest_market_regime(features: pd.DataFrame) -> str:
    """Return the dominant latest CS2 index regime."""
    latest = _latest_features(features)
    if latest.empty:
        return "unknown"
    counts = {
        "bull": int(latest.get("cs2_index_regime_bull", pd.Series(False)).fillna(False).sum()),
        "neutral": int(latest.get("cs2_index_regime_neutral", pd.Series(False)).fillna(False).sum()),
        "bear": int(latest.get("cs2_index_regime_bear", pd.Series(False)).fillna(False).sum()),
    }
    return max(counts, key=counts.get)


def _latest_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    if frame.empty or "timestamp" not in frame.columns:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest_timestamp = frame["timestamp"].max()
    return frame[frame["timestamp"].eq(latest_timestamp)].copy()


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


if __name__ == "__main__":
    main()
