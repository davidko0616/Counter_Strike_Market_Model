from __future__ import annotations

import pandas as pd

from cs_market_model.features.events import add_market_event_features


def test_market_event_features_only_activate_during_configured_window(tmp_path) -> None:
    events_path = tmp_path / "market_events.yaml"
    events_path.write_text(
        """
events:
  - event_id: cs2_covert_knife_glove_tradeup
    start_date: "2025-10-22"
    end_date: "2025-10-23"
    affected_item_query:
      rarities: ["Covert"]
      include_terms: []
      exclude_terms: []
""",
        encoding="utf-8",
    )
    rows = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Red"] * 3 + ["AWP | Blue"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2025-10-21",
                    "2025-10-22",
                    "2025-10-24",
                    "2025-10-21",
                    "2025-10-22",
                    "2025-10-24",
                ],
                utc=True,
            ),
            "rarity": ["Covert"] * 3 + ["Classified"] * 3,
        }
    )

    features = add_market_event_features(rows, events_path)

    covert_event = features[
        (features["market_hash_name"] == "AK-47 | Red")
        & (features["timestamp"] == pd.Timestamp("2025-10-22", tz="UTC"))
    ].iloc[0]
    covert_before = features[
        (features["market_hash_name"] == "AK-47 | Red")
        & (features["timestamp"] == pd.Timestamp("2025-10-21", tz="UTC"))
    ].iloc[0]
    classified_event = features[
        (features["market_hash_name"] == "AWP | Blue")
        & (features["timestamp"] == pd.Timestamp("2025-10-22", tz="UTC"))
    ].iloc[0]

    assert bool(covert_event["is_known_market_event_window"])
    assert bool(covert_event["is_covert_tradeup_event_window"])
    assert covert_event["days_since_known_market_event_start"] == 0
    assert not bool(covert_before["is_known_market_event_window"])
    assert pd.isna(covert_before["days_since_known_market_event_start"])
    assert not bool(classified_event["is_known_market_event_window"])
