"""Known market-event feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cs_market_model.config import CONFIG_DIR

DEFAULT_MARKET_EVENTS_INPUT = CONFIG_DIR / "market_events.yaml"


def add_market_event_features(
    feature_frame: pd.DataFrame,
    market_events_input: Path = DEFAULT_MARKET_EVENTS_INPUT,
) -> pd.DataFrame:
    """Add leakage-safe known-event regime features.

    Event windows activate only when ``timestamp`` is at or after the configured
    event start. Surprise event dates are therefore not exposed to rows before
    the event.
    """
    required = {"market_hash_name", "timestamp"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ValueError(f"Missing required event feature columns: {sorted(missing)}")

    features = feature_frame.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    features["is_known_market_event_window"] = False
    features["is_known_market_event_affected_item"] = False
    features["days_since_known_market_event_start"] = np.nan
    features["known_event_count_active"] = 0
    features["is_tradeup_event_window"] = False
    features["is_covert_tradeup_event_window"] = False
    features["is_souvenir_tradeup_event_window"] = False
    for window in (7, 14, 30):
        features[f"is_post_known_market_event_{window}d"] = False
        features[f"is_post_tradeup_event_{window}d"] = False

    events = load_market_events(market_events_input)
    if not events:
        return features

    for event in events:
        start = pd.Timestamp(event["start_date"], tz="UTC")
        end = pd.Timestamp(event["end_date"], tz="UTC") + pd.Timedelta(days=1)
        event_id = str(event.get("event_id", ""))
        active_window = features["timestamp"].ge(start) & features["timestamp"].lt(end)
        affected_item = _affected_item_mask(features, event)
        mask = active_window & affected_item
        days_since_start = (features.loc[mask, "timestamp"] - start).dt.days.astype(float)

        features.loc[mask, "is_known_market_event_window"] = True
        features.loc[mask, "is_known_market_event_affected_item"] = True
        features.loc[mask, "known_event_count_active"] += 1
        features.loc[mask, "days_since_known_market_event_start"] = days_since_start
        if "tradeup" in event_id:
            features.loc[mask, "is_tradeup_event_window"] = True
        if "covert" in event_id:
            features.loc[mask, "is_covert_tradeup_event_window"] = True
        if "souvenir" in event_id:
            features.loc[mask, "is_souvenir_tradeup_event_window"] = True

        for window in (7, 14, 30):
            post_start = end
            post_end = end + pd.Timedelta(days=window)
            post_window = features["timestamp"].ge(post_start) & features["timestamp"].lt(post_end)
            post_mask = post_window & affected_item
            features.loc[post_mask, f"is_post_known_market_event_{window}d"] = True
            if "tradeup" in event_id:
                features.loc[post_mask, f"is_post_tradeup_event_{window}d"] = True

    return features


def load_market_events(path: Path = DEFAULT_MARKET_EVENTS_INPUT) -> list[dict[str, Any]]:
    """Load configured market events."""
    if not path.exists():
        return []

    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    events = data.get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _affected_item_mask(features: pd.DataFrame, event: dict[str, Any]) -> pd.Series:
    query = event.get("affected_item_query", {})
    if not isinstance(query, dict):
        return pd.Series(True, index=features.index)

    mask = pd.Series(True, index=features.index)
    rarities = [str(rarity) for rarity in query.get("rarities", [])]
    if rarities:
        if "rarity" not in features.columns:
            return pd.Series(False, index=features.index)
        mask &= features["rarity"].isin(rarities)

    include_terms = [str(term) for term in query.get("include_terms", [])]
    if include_terms:
        names = features["market_hash_name"].fillna("")
        mask &= pd.concat(
            [names.str.contains(term, case=False, regex=False) for term in include_terms],
            axis=1,
        ).any(axis=1)

    exclude_terms = [str(term) for term in query.get("exclude_terms", [])]
    if exclude_terms:
        names = features["market_hash_name"].fillna("")
        mask &= ~pd.concat(
            [names.str.contains(term, case=False, regex=False) for term in exclude_terms],
            axis=1,
        ).any(axis=1)

    return mask
