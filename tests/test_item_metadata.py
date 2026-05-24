from __future__ import annotations

import pandas as pd

from cs_market_model.normalization.items import enrich_item_metadata


def test_enrich_item_metadata_adds_tradeup_eligibility_flags() -> None:
    metadata = pd.DataFrame(
        [
            {
                "market_hash_name": "AK-47 | Red Test (Field-Tested)",
                "rarity": "Covert",
                "stattrak_flag": False,
                "souvenir_flag": False,
            },
            {
                "market_hash_name": "Souvenir AWP | Test (Field-Tested)",
                "rarity": "Classified",
            },
        ]
    )

    enriched = enrich_item_metadata(metadata)

    covert = enriched.iloc[0]
    souvenir = enriched.iloc[1]
    assert bool(covert["knife_glove_tradeup_eligible"])
    assert bool(covert["regular_knife_glove_tradeup_eligible"])
    assert covert["tradeup_output_rarity"] == "Knife/Gloves"
    assert bool(souvenir["souvenir_tradeup_input_eligible"])
    assert bool(souvenir["souvenir_attributes_removed_on_tradeup"])
