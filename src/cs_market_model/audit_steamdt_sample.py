"""Day 2 audit command for a local SteamDT K-line sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cs_market_model.config import PROJECT_ROOT
from cs_market_model.normalization.prices import parse_and_audit_steamdt_kline

DEFAULT_SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kline"
    / "day2_sample_ak47_slate_field_tested_type2.json"
)
DEFAULT_MARKET_HASH_NAME = "AK-47 | Slate (Field-Tested)"


def build_day2_outputs(
    input_path: Path,
    audit_output: Path,
    bars_output: Path,
    column_docs_output: Path,
    market_hash_name: str | None = None,
) -> None:
    result = parse_and_audit_steamdt_kline(input_path, market_hash_name=market_hash_name)

    audit_output.parent.mkdir(parents=True, exist_ok=True)
    bars_output.parent.mkdir(parents=True, exist_ok=True)
    column_docs_output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result.audit]).to_csv(audit_output, index=False)
    result.column_docs.to_csv(column_docs_output, index=False)
    result.bars.to_parquet(bars_output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Day 2 SteamDT K-line sample.")
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/tables/data_audit_mvp.csv"),
    )
    parser.add_argument(
        "--bars-output",
        type=Path,
        default=Path("data/processed/price_bars_ak47_slate_field_tested.parquet"),
    )
    parser.add_argument(
        "--column-docs-output",
        type=Path,
        default=Path("reports/tables/steamdt_kline_columns.csv"),
    )
    parser.add_argument("--market-hash-name", default=DEFAULT_MARKET_HASH_NAME)
    args = parser.parse_args()

    build_day2_outputs(
        input_path=args.input,
        audit_output=args.audit_output,
        bars_output=args.bars_output,
        column_docs_output=args.column_docs_output,
        market_hash_name=args.market_hash_name,
    )
    print(f"Wrote audit: {args.audit_output}")
    print(f"Wrote normalized bars: {args.bars_output}")
    print(f"Wrote column docs: {args.column_docs_output}")


if __name__ == "__main__":
    main()
