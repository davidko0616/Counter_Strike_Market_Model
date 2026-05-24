"""Tests for the Day 15 rejection-aware trade analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cs_market_model.backtesting.rejection import (
    RejectionPolicy,
    apply_rejection_policy,
    build_rejection_curve,
    summarize_threshold_policies,
    summarize_normal_accepted_trades,
    _accepted_trade_metrics,
)


def _make_ledger(n: int = 20) -> pd.DataFrame:
    """Create a synthetic trade ledger with all columns used by rejection."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "model_name": ["test_model"] * n,
        "event_id": [f"e{i}" for i in range(n)],
        "market_hash_name": [f"AK-47 | Skin {i % 5}" for i in range(n)],
        "category": ["rifle"] * n,
        "entry_timestamp": pd.date_range("2025-06-01", periods=n, freq="D", tz="UTC"),
        "exit_timestamp": pd.date_range("2025-06-15", periods=n, freq="D", tz="UTC"),
        "holding_period_days": [14.0] * n,
        "strong_buy_score": np.linspace(0.1, 0.9, n),
        "daily_rank": list(range(1, n + 1)),
        "label_code": rng.choice([0, 1, 2], size=n),
        "label_class": rng.choice(["Bad Time", "Good Time", "Strong Buy"], size=n),
        "is_actual_strong_buy": rng.choice([True, False], size=n),
        "realized_net_return": rng.uniform(-0.05, 0.08, size=n),
        "notional": [0.05] * n,
        "pnl": rng.uniform(-0.003, 0.004, size=n),
        "label_market_regime": ["normal"] * (n - 2) + ["known_event"] * 2,
        "liquidity_quality_score_30d": np.linspace(0.2, 0.9, n),
        "effective_staleness_days": np.linspace(0.0, 5.0, n),
        "row_coverage_30d": np.linspace(0.3, 1.0, n),
        "price_jump_50pct_share_30d": np.linspace(0.0, 0.3, n),
        "cs2_index_regime_bear": [True] * (n // 2) + [False] * (n - n // 2),
        "item_excess_log_return_7d": np.linspace(-0.05, 0.05, n),
    })


class TestApplyRejectionPolicy:
    """Tests for apply_rejection_policy."""

    def test_no_filter_accepts_all(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(exclude_event_regime=False)
        result = apply_rejection_policy(ledger, policy)
        assert result["is_accepted"].all()
        assert result["rejection_reason"].isna().all()

    def test_event_regime_exclusion(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(exclude_event_regime=True)
        result = apply_rejection_policy(ledger, policy)
        event_rows = result[result["label_market_regime"] == "known_event"]
        assert not event_rows["is_accepted"].any()
        assert (event_rows["rejection_reason"] == "event_regime").all()

    def test_score_threshold(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(
            min_score_threshold=0.5,
            exclude_event_regime=False,
        )
        result = apply_rejection_policy(ledger, policy)
        low_score = result[result["strong_buy_score"] < 0.5]
        assert not low_score.empty
        # All low-score rows should be rejected for low_score
        assert (low_score["rejection_reason"] == "low_score").all()

    def test_liquidity_threshold(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(
            min_liquidity_quality=0.8,
            exclude_event_regime=False,
        )
        result = apply_rejection_policy(ledger, policy)
        rejected = result[~result["is_accepted"]]
        assert not rejected.empty
        assert "low_liquidity" in rejected["rejection_reason"].values

    def test_staleness_threshold(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(
            max_staleness_days=2.0,
            exclude_event_regime=False,
        )
        result = apply_rejection_policy(ledger, policy)
        stale = result[result["effective_staleness_days"] > 2.0]
        # Stale rows should be rejected (unless already rejected by prior gate)
        assert not stale.empty
        assert stale["rejection_reason"].notna().all()

    def test_price_jump_threshold(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(
            max_price_jump_share=0.1,
            exclude_event_regime=False,
        )
        result = apply_rejection_policy(ledger, policy)
        jump = result[result["price_jump_50pct_share_30d"] > 0.1]
        assert not jump.empty
        assert jump["rejection_reason"].notna().all()

    def test_row_coverage_threshold(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(
            min_row_coverage=0.8,
            exclude_event_regime=False,
        )
        result = apply_rejection_policy(ledger, policy)
        low_cov = result[result["row_coverage_30d"] < 0.8]
        assert not low_cov.empty
        assert low_cov["rejection_reason"].notna().all()

    def test_priority_order_event_regime_first(self) -> None:
        """Event regime gate should take priority over score gate."""
        ledger = _make_ledger()
        policy = RejectionPolicy(
            min_score_threshold=0.5,
            exclude_event_regime=True,
        )
        result = apply_rejection_policy(ledger, policy)
        event_rows = result[result["label_market_regime"] == "known_event"]
        # Even if score is high, event regime reason takes priority
        assert (event_rows["rejection_reason"] == "event_regime").all()

    def test_empty_ledger(self) -> None:
        ledger = pd.DataFrame(columns=_make_ledger().columns)
        policy = RejectionPolicy()
        result = apply_rejection_policy(ledger, policy)
        assert result.empty
        assert "is_accepted" in result.columns
        assert "rejection_reason" in result.columns

    def test_all_rejected(self) -> None:
        ledger = _make_ledger(5)
        policy = RejectionPolicy(
            min_score_threshold=0.99,
            exclude_event_regime=False,
        )
        result = apply_rejection_policy(ledger, policy)
        assert not result["is_accepted"].any()


class TestRejectionCurve:
    """Tests for build_rejection_curve."""

    def test_curve_shape(self) -> None:
        ledger = _make_ledger()
        thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
        curve = build_rejection_curve(
            ledger,
            dimension="min_score_threshold",
            thresholds=thresholds,
        )
        assert len(curve) == len(thresholds)
        assert "threshold_value" in curve.columns
        assert "accepted_count" in curve.columns
        assert "dimension" in curve.columns
        assert (curve["dimension"] == "min_score_threshold").all()

    def test_monotonically_decreasing_accepted(self) -> None:
        ledger = _make_ledger(50)
        thresholds = [round(v, 2) for v in np.arange(0.0, 0.95, 0.1)]
        curve = build_rejection_curve(
            ledger,
            dimension="min_score_threshold",
            thresholds=thresholds,
        )
        accepted_counts = curve["accepted_count"].values
        # Accepted count should never increase as threshold rises
        for i in range(1, len(accepted_counts)):
            assert accepted_counts[i] <= accepted_counts[i - 1], (
                f"accepted_count increased from {accepted_counts[i-1]} to "
                f"{accepted_counts[i]} at threshold index {i}"
            )

    def test_staleness_curve_decreasing(self) -> None:
        ledger = _make_ledger(50)
        thresholds = [float("inf"), 5.0, 3.0, 2.0, 1.0]
        curve = build_rejection_curve(
            ledger,
            dimension="max_staleness_days",
            thresholds=thresholds,
        )
        accepted_counts = curve["accepted_count"].values
        for i in range(1, len(accepted_counts)):
            assert accepted_counts[i] <= accepted_counts[i - 1]


class TestSummarizeThresholdPolicies:
    """Tests for summarize_threshold_policies."""

    def test_basic_output(self) -> None:
        ledger = _make_ledger()
        policies = [
            ("no_filter", RejectionPolicy(exclude_event_regime=False)),
            ("strict", RejectionPolicy(min_score_threshold=0.7, exclude_event_regime=True)),
        ]
        summary = summarize_threshold_policies(ledger, policies)
        assert not summary.empty
        assert "policy_name" in summary.columns
        assert "model_name" in summary.columns
        assert set(summary["policy_name"]) == {"no_filter", "strict"}

    def test_no_filter_accepts_all(self) -> None:
        ledger = _make_ledger()
        policies = [
            ("no_filter", RejectionPolicy(exclude_event_regime=False)),
        ]
        summary = summarize_threshold_policies(ledger, policies)
        no_filter = summary[summary["policy_name"] == "no_filter"].iloc[0]
        assert no_filter["accepted_count"] == len(ledger)
        assert no_filter["rejected_count"] == 0

    def test_strict_rejects_more(self) -> None:
        ledger = _make_ledger()
        policies = [
            ("no_filter", RejectionPolicy(exclude_event_regime=False)),
            ("strict", RejectionPolicy(min_score_threshold=0.7, exclude_event_regime=True)),
        ]
        summary = summarize_threshold_policies(ledger, policies)
        no_filter = summary[summary["policy_name"] == "no_filter"].iloc[0]
        strict = summary[summary["policy_name"] == "strict"].iloc[0]
        assert strict["accepted_count"] < no_filter["accepted_count"]


class TestNormalAccepted:
    """Tests for summarize_normal_accepted_trades."""

    def test_excludes_event_regime(self) -> None:
        ledger = _make_ledger()
        event_count = (ledger["label_market_regime"] == "known_event").sum()
        policies = [
            ("no_filter", RejectionPolicy(exclude_event_regime=False)),
        ]
        summary = summarize_normal_accepted_trades(ledger, policies)
        no_filter = summary[summary["policy_name"] == "no_filter"].iloc[0]
        # Normal-only summary should never include event-regime trades
        assert no_filter["accepted_count"] == len(ledger) - event_count


class TestAcceptedTradeMetrics:
    """Tests for _accepted_trade_metrics."""

    def test_empty_accepted(self) -> None:
        empty = pd.DataFrame(columns=_make_ledger().columns)
        metrics = _accepted_trade_metrics(empty, total_count=10)
        assert metrics["accepted_count"] == 0
        assert metrics["rejected_count"] == 10
        assert metrics["rejection_rate"] == 1.0
        assert np.isnan(metrics["accepted_avg_return"])

    def test_all_winners(self) -> None:
        ledger = _make_ledger(5)
        ledger["realized_net_return"] = 0.05
        ledger["pnl"] = 0.01
        metrics = _accepted_trade_metrics(ledger, total_count=5)
        assert metrics["accepted_win_rate"] == 1.0
        assert metrics["profit_factor"] == float("inf")

    def test_all_losers(self) -> None:
        ledger = _make_ledger(5)
        ledger["realized_net_return"] = -0.05
        ledger["pnl"] = -0.01
        metrics = _accepted_trade_metrics(ledger, total_count=5)
        assert metrics["accepted_win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0


class TestRejectionPolicyValidation:
    """Tests for RejectionPolicy validation."""

    def test_negative_score_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_score_threshold"):
            RejectionPolicy(min_score_threshold=-0.1)

    def test_zero_staleness_raises(self) -> None:
        with pytest.raises(ValueError, match="max_staleness_days"):
            RejectionPolicy(max_staleness_days=0.0)

    def test_negative_liquidity_raises(self) -> None:
        with pytest.raises(ValueError, match="min_liquidity_quality"):
            RejectionPolicy(min_liquidity_quality=-0.5)


class TestBearRegimeGate:
    """Tests for the bear market regime rejection gate."""

    def test_bear_no_alpha_rejects_bear_underperformers(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(
            exclude_event_regime=False,
            reject_bear_without_alpha=True,
        )
        result = apply_rejection_policy(ledger, policy)
        # Rows that are bear AND have non-positive excess return should be rejected
        bear_mask = ledger["cs2_index_regime_bear"].astype(bool)
        excess = ledger["item_excess_log_return_7d"]
        expected_rejected = bear_mask & (excess <= 0)
        rejected_bear = result[result["rejection_reason"] == "bear_no_alpha"]
        assert len(rejected_bear) == expected_rejected.sum()

    def test_bear_with_alpha_accepted(self) -> None:
        """Items outperforming in a bear market should NOT be rejected."""
        ledger = _make_ledger(10)
        # All bear, but all have positive excess return
        ledger["cs2_index_regime_bear"] = True
        ledger["item_excess_log_return_7d"] = 0.03
        policy = RejectionPolicy(
            exclude_event_regime=False,
            reject_bear_without_alpha=True,
        )
        result = apply_rejection_policy(ledger, policy)
        assert (result["is_accepted"]).all()

    def test_bear_gate_disabled_by_default(self) -> None:
        ledger = _make_ledger()
        policy = RejectionPolicy(exclude_event_regime=False)
        result = apply_rejection_policy(ledger, policy)
        # Default policy doesn't reject bear trades
        assert "bear_no_alpha" not in result["rejection_reason"].values
