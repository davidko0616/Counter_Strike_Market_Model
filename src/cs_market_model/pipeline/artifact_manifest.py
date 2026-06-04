"""Generate reproducibility manifests for generated artifacts."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path

DEFAULT_MANIFEST_OUTPUT = reports_path("tables", "artifact_manifest.csv")
DEFAULT_ARTIFACTS = (
    data_path("processed", "day10_11_lightgbm_predictions.parquet"),
    data_path("features", "features_day5_v1.parquet"),
    data_path("features", "csfloat_listing_features.parquet"),
    reports_path("tables", "day10_11_feature_importance.csv"),
    reports_path("tables", "day18_csfloat_ablation_setup.csv"),
    reports_path("tables", "day19_csfloat_coverage.csv"),
    reports_path("tables", "paper_trade_ab_test_summary.csv"),
    reports_path("tables", "model_health_monitor.csv"),
    reports_path("tables", "signal_explanations.csv"),
    reports_path("backtests", "paper_trade_ab_test_report.md"),
    reports_path("backtests", "model_health_report.md"),
    PROJECT_ROOT / "docs" / "final_research_report.md",
    PROJECT_ROOT / "docs" / "REPRODUCE.md",
)


@dataclass(frozen=True)
class ManifestResult:
    """Output metadata for an artifact manifest."""

    output_path: Path
    artifact_count: int
    existing_artifact_count: int


def build_artifact_manifest(paths: Iterable[Path] = DEFAULT_ARTIFACTS) -> pd.DataFrame:
    """Return SHA256, byte-size, row-count, and modified-time rows."""
    return pd.DataFrame([artifact_manifest_row(path) for path in paths])


def artifact_manifest_row(path: Path) -> dict[str, object]:
    """Return one manifest row for an artifact path."""
    exists = path.exists()
    return {
        "artifact": _display_path(path),
        "exists": exists,
        "sha256": _sha256(path) if exists and path.is_file() else "",
        "bytes": path.stat().st_size if exists and path.is_file() else 0,
        "rows": _row_count(path) if exists and path.is_file() else 0,
        "modified_utc": _modified_utc(path) if exists and path.is_file() else "",
    }


def write_artifact_manifest(
    output_path: Path = DEFAULT_MANIFEST_OUTPUT,
    paths: Iterable[Path] = DEFAULT_ARTIFACTS,
) -> ManifestResult:
    """Write the local artifact manifest as CSV."""
    manifest = build_artifact_manifest(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return ManifestResult(
        output_path=output_path,
        artifact_count=len(manifest),
        existing_artifact_count=int(manifest["exists"].sum()) if not manifest.empty else 0,
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _row_count(path: Path) -> int:
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        if path.suffix == ".csv":
            return int(len(pd.read_csv(path)))
    except Exception:
        return 0
    return 0


def _modified_utc(path: Path) -> str:
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write artifact manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    result = write_artifact_manifest(output_path=args.output)
    print(f"Artifacts listed: {result.artifact_count}")
    print(f"Artifacts present: {result.existing_artifact_count}")
    print(f"Wrote manifest: {_display_path(result.output_path)}")


if __name__ == "__main__":
    main()
