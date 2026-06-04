"""Model, data, and paper-trade health monitoring."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path

DEFAULT_FEATURES_INPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_IMPORTANCE_INPUT = reports_path("tables", "day10_11_feature_importance.csv")
DEFAULT_PAPER_TRADE_SUMMARY_INPUT = reports_path("tables", "paper_trade_ab_test_summary.csv")
DEFAULT_CSFLOAT_COVERAGE_INPUT = reports_path("tables", "day19_csfloat_coverage.csv")
DEFAULT_MONITOR_OUTPUT = reports_path("tables", "model_health_monitor.csv")
DEFAULT_STATUS_OUTPUT = reports_path("tables", "model_health_status.json")
DEFAULT_REPORT_OUTPUT = reports_path("backtests", "model_health_report.md")
MAX_FRESHNESS_DAYS = 7
MIN_FEATURE_COVERAGE = 0.80
MAX_FEATURE_PSI_WARNING = 0.25
MAX_PROFIT_FACTOR_WORSENING = 0.20
MAX_DRAWDOWN_WORSENING = 0.05


@dataclass(frozen=True)
class ModelHealthResult:
    """Output metadata for the model health monitor."""

    monitor_output: Path
    status_output: Path
    report_output: Path
    status: str
    row_count: int


def run_model_health_monitor(
    features_input: Path = DEFAULT_FEATURES_INPUT,
    importance_input: Path = DEFAULT_IMPORTANCE_INPUT,
    paper_trade_summary_input: Path = DEFAULT_PAPER_TRADE_SUMMARY_INPUT,
    csfloat_coverage_input: Path = DEFAULT_CSFLOAT_COVERAGE_INPUT,
    monitor_output: Path = DEFAULT_MONITOR_OUTPUT,
    status_output: Path = DEFAULT_STATUS_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> ModelHealthResult:
    """Write health monitor CSV, JSON status, and Markdown report."""
    rows = []
    rows.extend(
        artifact_freshness_rows(
            [
                features_input,
                importance_input,
                paper_trade_summary_input,
                csfloat_coverage_input,
            ]
        )
    )
    rows.extend(paper_trade_health_rows(_read_table(paper_trade_summary_input)))
    features = _read_table(features_input)
    importance = _read_table(importance_input)
    rows.extend(feature_coverage_rows(features, importance))
    rows.extend(feature_drift_rows(features, importance))
    rows.extend(csfloat_coverage_rows(_read_table(csfloat_coverage_input)))

    monitor = pd.DataFrame(rows)
    status_payload = summarize_health_status(monitor)
    monitor_output.parent.mkdir(parents=True, exist_ok=True)
    status_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    monitor.to_csv(monitor_output, index=False)
    status_output.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    report_output.write_text(build_model_health_report(monitor, status_payload), encoding="utf-8")
    return ModelHealthResult(
        monitor_output=monitor_output,
        status_output=status_output,
        report_output=report_output,
        status=str(status_payload["status"]),
        row_count=len(monitor),
    )


def artifact_freshness_rows(
    paths: Iterable[Path],
    *,
    now: pd.Timestamp | None = None,
) -> list[dict[str, object]]:
    """Return freshness rows for important local artifacts."""
    resolved_now = now or pd.Timestamp.now(tz="UTC")
    rows = []
    for path in paths:
        exists = path.exists()
        age_days = np.nan
        status = "warning"
        detail = "missing"
        if exists:
            modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
            age_days = float((resolved_now - modified).total_seconds() / 86400.0)
            status = "healthy" if age_days <= MAX_FRESHNESS_DAYS else "warning"
            detail = "fresh" if status == "healthy" else "stale"
        rows.append(
            {
                "check": "artifact_freshness",
                "target": _display_path(path),
                "status": status,
                "value": age_days,
                "threshold": MAX_FRESHNESS_DAYS,
                "detail": detail,
            }
        )
    return rows


def paper_trade_health_rows(summary: pd.DataFrame) -> list[dict[str, object]]:
    """Return health rows from paper-trade A/B status."""
    if summary.empty:
        return [
            {
                "check": "paper_trade_review",
                "target": "paper_trade_5_plus",
                "status": "warning",
                "value": np.nan,
                "threshold": "paper_trade_summary",
                "detail": "missing_paper_trade_summary",
            }
        ]
    rows = []
    for _, row in summary.iterrows():
        review_status = str(row.get("review_status", "unknown"))
        status = "healthy" if review_status == "passes_paper_trade_review" else "warning"
        if review_status.startswith("failed_"):
            status = "critical"
        rows.append(
            {
                "check": "paper_trade_review",
                "target": row.get("scenario", "unknown"),
                "status": status,
                "value": row.get("accepted_count", np.nan),
                "threshold": "paper_trade_only",
                "detail": review_status,
            }
        )
    return rows


def feature_coverage_rows(
    features: pd.DataFrame,
    importance: pd.DataFrame,
) -> list[dict[str, object]]:
    """Return top-important-feature non-null coverage rows."""
    if features.empty:
        return [
            {
                "check": "feature_coverage",
                "target": "features",
                "status": "critical",
                "value": 0.0,
                "threshold": MIN_FEATURE_COVERAGE,
                "detail": "missing_features",
            }
        ]
    candidates = _top_feature_names(importance)
    if not candidates:
        candidates = [
            column
            for column in features.columns
            if column not in {"timestamp", "market_hash_name", "event_id"}
        ][:10]
    rows = []
    for feature in candidates:
        if feature not in features.columns:
            coverage = 0.0
        else:
            coverage = float(features[feature].notna().mean())
        rows.append(
            {
                "check": "feature_coverage",
                "target": feature,
                "status": "healthy" if coverage >= MIN_FEATURE_COVERAGE else "critical",
                "value": coverage,
                "threshold": MIN_FEATURE_COVERAGE,
                "detail": "non_null_coverage",
            }
        )
    return rows


def feature_drift_rows(features: pd.DataFrame, importance: pd.DataFrame) -> list[dict[str, object]]:
    """Return PSI drift rows for top important features."""
    if features.empty or "timestamp" not in features.columns:
        return []
    frame = features.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(frame) < 20:
        return []
    midpoint = len(frame) // 2
    baseline = frame.iloc[:midpoint]
    current = frame.iloc[midpoint:]
    rows = []
    for feature in _top_feature_names(importance):
        if feature not in frame.columns:
            continue
        psi = population_stability_index(baseline[feature], current[feature])
        rows.append(
            {
                "check": "feature_drift_psi",
                "target": feature,
                "status": "healthy" if psi <= MAX_FEATURE_PSI_WARNING else "warning",
                "value": psi,
                "threshold": MAX_FEATURE_PSI_WARNING,
                "detail": "baseline_vs_recent",
            }
        )
    return rows


def population_stability_index(
    baseline: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Return a dependency-free PSI estimate for one numeric feature."""
    baseline_values = pd.to_numeric(baseline, errors="coerce").dropna().to_numpy(dtype=float)
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(baseline_values) == 0 or len(current_values) == 0:
        return float("inf")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline_values, quantiles))
    if len(edges) < 3:
        lower = min(float(np.min(baseline_values)), float(np.min(current_values)))
        upper = max(float(np.max(baseline_values)), float(np.max(current_values)))
        if lower == upper:
            return 0.0
        edges = np.linspace(lower, upper, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    baseline_counts, _ = np.histogram(baseline_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    baseline_share = baseline_counts / max(baseline_counts.sum(), 1)
    current_share = current_counts / max(current_counts.sum(), 1)
    epsilon = 1e-6
    baseline_share = np.clip(baseline_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)
    return float(np.sum((current_share - baseline_share) * np.log(current_share / baseline_share)))


def csfloat_coverage_rows(coverage: pd.DataFrame) -> list[dict[str, object]]:
    """Return CSFloat point-in-time coverage status rows."""
    if coverage.empty:
        return [
            {
                "check": "csfloat_coverage",
                "target": "day19",
                "status": "warning",
                "value": 0.0,
                "threshold": "diagnostic_only",
                "detail": "missing_csfloat_coverage",
            }
        ]
    post_snapshot = coverage.get("has_post_snapshot_bar", pd.Series(dtype=float))
    usable = pd.to_numeric(post_snapshot, errors="coerce").fillna(0.0).astype(float)
    rate = float(usable.mean()) if len(usable) else 0.0
    return [
        {
            "check": "csfloat_coverage",
            "target": "point_in_time_rows",
            "status": "healthy" if rate >= 0.50 else "warning",
            "value": rate,
            "threshold": 0.50,
            "detail": "diagnostic_only_when_low",
        }
    ]


def should_keep_previous_model(
    *,
    previous_profit_factor: float,
    new_profit_factor: float,
    previous_drawdown: float,
    new_drawdown: float,
) -> bool:
    """Return True when conservative rollback rules reject the new model."""
    pf_worse = new_profit_factor < previous_profit_factor - MAX_PROFIT_FACTOR_WORSENING
    drawdown_worse = new_drawdown < previous_drawdown - MAX_DRAWDOWN_WORSENING
    return bool(pf_worse or drawdown_worse)


def summarize_health_status(monitor: pd.DataFrame) -> dict[str, object]:
    """Return machine-readable health status for the dashboard."""
    if monitor.empty:
        return {"status": "warning", "warnings": ["no_health_rows"]}
    critical = monitor[monitor["status"].astype(str).eq("critical")]
    warning = monitor[monitor["status"].astype(str).eq("warning")]
    if not critical.empty:
        status = "critical"
    elif not warning.empty:
        status = "warning"
    else:
        status = "healthy"
    warnings = [
        f"{row.check}:{row.target}:{row.detail}"
        for row in pd.concat([critical, warning], ignore_index=True).itertuples(index=False)
    ]
    return {"status": status, "warnings": warnings}


def build_model_health_report(monitor: pd.DataFrame, status_payload: dict[str, object]) -> str:
    """Build a short Markdown health report."""
    return "\n".join(
        [
            "# Model Health Report",
            "",
            f"Status: {status_payload['status']}",
            "",
            "## Checks",
            "",
            _markdown_table(monitor),
            "",
            "## Conservative Rollback Rule",
            "",
            (
                "Keep the previous model active if the new out-of-sample profit "
                "factor worsens by more than 0.20 or the new drawdown worsens by "
                "more than 5 percentage points."
            ),
            "",
        ]
    )


def _top_feature_names(importance: pd.DataFrame) -> list[str]:
    if importance.empty or "feature" not in importance.columns:
        return []
    frame = importance.copy()
    if "importance_gain" not in frame.columns:
        frame["importance_gain"] = 0.0
    frame["importance_gain"] = pd.to_numeric(frame["importance_gain"], errors="coerce").fillna(0.0)
    return (
        frame.groupby("feature", as_index=False)["importance_gain"]
        .mean()
        .sort_values("importance_gain", ascending=False)
        .head(10)["feature"]
        .astype(str)
        .tolist()
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else ""
        )
    columns = [str(column) for column in display.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        rows.append(
            "| " + " | ".join(_markdown_cell(row[column]) for column in display.columns) + " |"
        )
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model and data health monitor.")
    parser.add_argument("--features-input", type=Path, default=DEFAULT_FEATURES_INPUT)
    parser.add_argument("--importance-input", type=Path, default=DEFAULT_IMPORTANCE_INPUT)
    parser.add_argument(
        "--paper-trade-summary-input",
        type=Path,
        default=DEFAULT_PAPER_TRADE_SUMMARY_INPUT,
    )
    parser.add_argument(
        "--csfloat-coverage-input",
        type=Path,
        default=DEFAULT_CSFLOAT_COVERAGE_INPUT,
    )
    parser.add_argument("--monitor-output", type=Path, default=DEFAULT_MONITOR_OUTPUT)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()

    result = run_model_health_monitor(
        features_input=args.features_input,
        importance_input=args.importance_input,
        paper_trade_summary_input=args.paper_trade_summary_input,
        csfloat_coverage_input=args.csfloat_coverage_input,
        monitor_output=args.monitor_output,
        status_output=args.status_output,
        report_output=args.report_output,
    )
    print(f"Health status: {result.status}")
    print(f"Health rows: {result.row_count}")
    print(f"Wrote monitor: {_display_path(result.monitor_output)}")
    print(f"Wrote status: {_display_path(result.status_output)}")
    print(f"Wrote report: {_display_path(result.report_output)}")


if __name__ == "__main__":
    main()
