"""Item metadata normalization and market-mechanics enrichment."""

from __future__ import annotations

import pandas as pd

RARITY_ORDER = {
    "Consumer Grade": 0,
    "Industrial Grade": 1,
    "Mil-Spec Grade": 2,
    "Restricted": 3,
    "Classified": 4,
    "Covert": 5,
}
NEXT_TRADEUP_RARITY = {
    "Consumer Grade": "Industrial Grade",
    "Industrial Grade": "Mil-Spec Grade",
    "Mil-Spec Grade": "Restricted",
    "Restricted": "Classified",
    "Classified": "Covert",
    "Covert": "Knife/Gloves",
}


def enrich_item_metadata(metadata: pd.DataFrame | None) -> pd.DataFrame | None:
    """Add deterministic Trade Up and item-mechanics metadata."""
    if metadata is None:
        return None
    if metadata.empty:
        return metadata.copy()

    frame = metadata.copy()
    frame["rarity_rank"] = frame.get("rarity", pd.Series(index=frame.index)).map(RARITY_ORDER)
    frame["tradeup_output_rarity"] = frame.get(
        "rarity",
        pd.Series(index=frame.index),
    ).map(NEXT_TRADEUP_RARITY)
    frame["stattrak_flag"] = _bool_or_name_contains(frame, "stattrak_flag", "StatTrak")
    frame["souvenir_flag"] = _bool_or_name_contains(frame, "souvenir_flag", "Souvenir")

    rarity = frame.get("rarity", pd.Series(index=frame.index)).fillna("")
    frame["tradeup_input_eligible"] = rarity.isin(RARITY_ORDER)
    frame["normal_tradeup_input_eligible"] = (
        frame["tradeup_input_eligible"] & ~frame["souvenir_flag"]
    )
    frame["covert_tradeup_input_eligible"] = rarity.eq("Covert")
    frame["knife_glove_tradeup_eligible"] = (
        frame["covert_tradeup_input_eligible"] & ~frame["souvenir_flag"]
    )
    frame["stattrak_knife_tradeup_eligible"] = (
        frame["knife_glove_tradeup_eligible"] & frame["stattrak_flag"]
    )
    frame["regular_knife_glove_tradeup_eligible"] = (
        frame["knife_glove_tradeup_eligible"] & ~frame["stattrak_flag"]
    )
    frame["souvenir_tradeup_input_eligible"] = (
        frame["tradeup_input_eligible"] & frame["souvenir_flag"]
    )
    frame["souvenir_attributes_removed_on_tradeup"] = frame["souvenir_tradeup_input_eligible"]

    return frame


def _bool_or_name_contains(frame: pd.DataFrame, column: str, token: str) -> pd.Series:
    if column in frame.columns:
        values = frame[column].fillna(False).astype(bool)
    else:
        values = pd.Series(False, index=frame.index)
    if "market_hash_name" not in frame.columns:
        return values
    name_match = frame["market_hash_name"].fillna("").str.contains(token, case=False, regex=False)
    return values | name_match
