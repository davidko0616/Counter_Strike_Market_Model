# SteamDT K-Line Sample Schema

Day 2 uses a local repo-owned sample:

`data/raw/kline/day2_sample_ak47_slate_field_tested_type2.json`

The sample item is `AK-47 | Slate (Field-Tested)`.

The sample response is a JSON object with a `data` array. Each row in `data` is
a five-element K-line record.

| Raw index | Normalized field | Meaning |
| --- | --- | --- |
| 0 | `timestamp_epoch_seconds` | Exchange timestamp in Unix epoch seconds. |
| 1 | `open` | Opening price for the K-line interval. |
| 2 | `close` | Closing price for the K-line interval. |
| 3 | `high` | Highest observed price for the K-line interval. |
| 4 | `low` | Lowest observed price for the K-line interval. |

The open/close ordering is inferred from sample continuity: each row's close
matches the following row's open.

The sample does not include volume, trade count, or explicit currency fields.
For the Day 2 normalized output, those fields are represented as unavailable
where appropriate, and currency is set to the MVP default `CNY`.
