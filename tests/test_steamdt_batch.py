from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cs_market_model.collectors.steamdt_batch import (
    build_item_metadata,
    collect_steamdt_universe,
    load_universe_items,
)


class FakeSteamDTClient:
    def kline(
        self,
        market_hash_name: str,
        kline_type: int = 2,
        platform: str = "STEAM",
        special_style: str = "",
    ) -> dict[str, Any]:
        return {
            "success": True,
            "data": [
                ["1768233600", 10.0, 11.0, 12.0, 9.5],
                ["1768320000", 11.0, 11.5, 12.5, 10.5],
            ],
        }


def test_load_day3_universe_is_first_pull_sized_and_gun_only() -> None:
    items = load_universe_items("universe_day3_first_pull.yaml")

    assert len(items) == 48
    assert {item["category"] for item in items} <= {"rifle", "sniper", "pistol", "smg"}
    assert all("Case" not in item["market_hash_name"] for item in items)
    assert all("Sticker" not in item["market_hash_name"] for item in items)


def test_build_item_metadata_flattens_collection_and_case_lists() -> None:
    metadata = build_item_metadata(
        [
            {
                "market_hash_name": "AK-47 | Slate (Field-Tested)",
                "collections": ["The Snakebite Collection"],
                "cases": ["Snakebite Case"],
            }
        ]
    )

    assert metadata.loc[0, "collections"] == "The Snakebite Collection"
    assert metadata.loc[0, "cases"] == "Snakebite Case"
    assert bool(metadata.loc[0, "stattrak_flag"]) is False
    assert bool(metadata.loc[0, "souvenir_flag"]) is False


def test_collect_steamdt_universe_writes_bars_metadata_and_log(tmp_path: Path) -> None:
    items = [
        {
            "market_hash_name": "AK-47 | Slate (Field-Tested)",
            "weapon_type": "AK-47",
            "category": "rifle",
            "collections": ["The Snakebite Collection"],
            "cases": ["Snakebite Case"],
            "wear": "Field-Tested",
            "rarity": "Restricted",
        }
    ]

    result = collect_steamdt_universe(
        items=items,
        client=FakeSteamDTClient(),  # type: ignore[arg-type]
        raw_dir=tmp_path / "raw",
        bars_output=tmp_path / "price_bars.parquet",
        metadata_output=tmp_path / "metadata.parquet",
        log_output=tmp_path / "ingestion_log.csv",
    )

    assert result.requested_items == 1
    assert result.successful_items == 1
    assert result.failed_items == 0
    assert result.price_bar_rows == 2
    assert result.bars_output.exists()
    assert result.metadata_output.exists()
    assert result.log_output.exists()
    assert len(list((tmp_path / "raw").glob("*.json"))) == 1
    assert pd.read_parquet(result.bars_output)["market_hash_name"].tolist() == [
        "AK-47 | Slate (Field-Tested)",
        "AK-47 | Slate (Field-Tested)",
    ]
