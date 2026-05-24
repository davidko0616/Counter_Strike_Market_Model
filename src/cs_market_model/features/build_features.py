"""Build the Day 5 feature table from normalized price bars."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import PROJECT_ROOT, data_path, reports_path
from cs_market_model.features.events import add_market_event_features
from cs_market_model.features.indices import add_index_features
from cs_market_model.features.liquidity import add_liquidity_features
from cs_market_model.features.market_index import add_market_index_features
from cs_market_model.features.momentum import add_momentum_features
from cs_market_model.features.ranks import add_rank_features
from cs_market_model.features.volatility import add_volatility_features
from cs_market_model.normalization.items import enrich_item_metadata

DEFAULT_PRICE_BARS_INPUT = data_path("processed", "price_bars_day3_first_pull.parquet")
DEFAULT_METADATA_INPUT = data_path("processed", "item_metadata_day3_first_pull.parquet")
DEFAULT_FEATURES_OUTPUT = data_path("features", "features_day5_v1.parquet")
DEFAULT_AUDIT_OUTPUT = reports_path("tables", "day5_feature_audit.csv")

BASE_COLUMNS = {
    "item_id",
    "market_hash_name",
    "venue",
    "timestamp",
    "timestamp_exchange",
    "open",
    "high",
    "low",
    "close",
    "median_price",
    "volume",
    "trade_count",
    "currency",
    "data_resolution",
    "is_forward_filled",
    "staleness_days",
    "source",
    "weapon_type",
    "category",
    "requested_rarity",
    "rarity",
    "rarity_substituted",
    "wear",
    "source_type",
    "collections",
    "cases",
    "min_float",
    "max_float",
    "stattrak_flag",
    "souvenir_flag",
    "tradeup_output_rarity",
}


@dataclass(frozen=True)
class FeatureBuildResult:
    """Paths and shape metadata for a completed feature build."""

    features_output: Path
    audit_output: Path
    row_count: int
    item_count: int
    feature_count: int


def load_price_bars(path: Path = DEFAULT_PRICE_BARS_INPUT) -> pd.DataFrame:
    """Load normalized price bars from parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Price bars not found: {path}")
    return pd.read_parquet(path)


def load_item_metadata(path: Path = DEFAULT_METADATA_INPUT) -> pd.DataFrame | None:
    """Load item metadata if available."""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def join_item_metadata(price_bars: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    """Join item metadata onto price bars using market_hash_name."""
    metadata = enrich_item_metadata(metadata)
    if metadata is None or metadata.empty:
        return price_bars.copy()
    duplicate_keys = metadata["market_hash_name"].duplicated()
    if duplicate_keys.any():
        duplicates = sorted(metadata.loc[duplicate_keys, "market_hash_name"].unique())
        raise ValueError(f"Metadata has duplicate market_hash_name values: {duplicates}")
    return price_bars.merge(metadata, on="market_hash_name", how="left", validate="many_to_one")


def build_feature_table(
    price_bars: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build point-in-time-safe feature rows from normalized price bars."""
    features = join_item_metadata(price_bars, metadata)
    features = add_momentum_features(features)
    features = add_volatility_features(features)
    features = add_liquidity_features(features)
    features = add_market_index_features(features)
    features = add_index_features(features)
    features = add_rank_features(features)
    features = add_market_event_features(features)
    return features.sort_values(["market_hash_name", "timestamp"]).reset_index(drop=True)


def feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding identifiers and metadata."""
    return [column for column in feature_table.columns if column not in BASE_COLUMNS]


def audit_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Create per-feature data quality statistics."""
    rows: list[dict[str, Any]] = []
    for column in feature_columns(feature_table):
        series = feature_table[column]
        numeric = pd.to_numeric(series, errors="coerce")
        finite = np.isfinite(numeric)
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "row_count": len(series),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "finite_count": int(finite.sum()),
                "min": float(numeric[finite].min()) if finite.any() else np.nan,
                "max": float(numeric[finite].max()) if finite.any() else np.nan,
                "mean": float(numeric[finite].mean()) if finite.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def build_day5_features(
    price_bars_input: Path = DEFAULT_PRICE_BARS_INPUT,
    metadata_input: Path = DEFAULT_METADATA_INPUT,
    features_output: Path = DEFAULT_FEATURES_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
) -> FeatureBuildResult:
    """Load Day 3 outputs, build Day 5 features, and save artifacts."""
    price_bars = load_price_bars(price_bars_input)
    metadata = load_item_metadata(metadata_input)
    feature_table = build_feature_table(price_bars, metadata)
    audit = audit_feature_table(feature_table)

    features_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(features_output, index=False)
    audit.to_csv(audit_output, index=False)

    return FeatureBuildResult(
        features_output=features_output,
        audit_output=audit_output,
        row_count=len(feature_table),
        item_count=feature_table["market_hash_name"].nunique(),
        feature_count=len(feature_columns(feature_table)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Day 5 Feature Factory V1 outputs.")
    parser.add_argument("--price-bars-input", type=Path, default=DEFAULT_PRICE_BARS_INPUT)
    parser.add_argument("--metadata-input", type=Path, default=DEFAULT_METADATA_INPUT)
    parser.add_argument("--features-output", type=Path, default=DEFAULT_FEATURES_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    args = parser.parse_args()

    result = build_day5_features(
        price_bars_input=args.price_bars_input,
        metadata_input=args.metadata_input,
        features_output=args.features_output,
        audit_output=args.audit_output,
    )
    print(f"Feature rows: {result.row_count}")
    print(f"Items: {result.item_count}")
    print(f"Feature columns: {result.feature_count}")
    print(f"Wrote features: {_display_path(result.features_output)}")
    print(f"Wrote audit: {_display_path(result.audit_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
