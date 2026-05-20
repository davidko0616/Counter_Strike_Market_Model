# SteamDT K-Line Sample Schema

Day 2 uses the existing sibling sample:

`C:\Users\chung\Desktop\2026-1학기\인공지능기초수학\cs2_price_model\data\raw\kline\AK-47__Redline_Field-Tested_type2.json`

The sample response is a JSON object with a `data` array. Each row in `data` is
a five-element K-line record.

| Raw index | Normalized field | Meaning |
| --- | --- | --- |
| 0 | `timestamp_epoch_seconds` | Exchange timestamp in Unix epoch seconds. |
| 1 | `open` | Opening price for the K-line interval. |
| 2 | `close` | Closing price for the K-line interval. |
| 3 | `high` | Highest observed price for the K-line interval. |
| 4 | `low` | Lowest observed price for the K-line interval. |

The open/close ordering is inferred from the sample continuity: each row's close
matches the following row's open.

The sample does not include volume, trade count, or explicit currency fields.
For the Day 2 normalized output, those fields are represented as unavailable
where appropriate, and currency is set to the MVP default `CNY`.
