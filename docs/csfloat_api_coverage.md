# CSFloat API Coverage

Official documentation checked: `https://docs.csfloat.com/`.

The documented CSFloat API is useful for active listing snapshots and item-level
microstructure. It is not sufficient as the project's primary historical time
series source.

## MVP Coverage Summary

| Need | CSFloat support | Notes |
| --- | --- | --- |
| Active listings | Yes | `GET /api/v1/listings` returns active marketplace listings. |
| Best ask | Yes | Listing objects expose `price`; querying `sort_by=lowest_price` gives the current ask ladder. |
| Listing depth | Yes, with pagination | The endpoint supports `limit` and `cursor`; full depth requires pagination and rate-limit handling. |
| Float values | Yes | Listing item objects expose `item.float_value`. |
| Stickers | Yes | Listing item objects expose `item.stickers[]`. |
| Sticker reference values | Yes, when populated | Live listing objects can include sticker `reference.price`. Treat as sparse. |
| Item reference price | Yes, when populated | Live listing objects can include `reference.base_price`. This is a reference, not executable history. |
| Predicted price | Partial | Live listing objects can include `reference.predicted_price`; use only as a snapshot feature, not as labels. |
| Seller reliability | Yes | Seller objects may expose trade statistics. |
| Bid side / true spread | No | The public listing docs expose sell-side listings, not a complete bid-side order book. |
| Historical daily bars | No | No stable historical K-line endpoint is documented. |
| Historical volume / trade count | No | Current listing data is not a historical volume feed. |

Conclusion: use CSFloat for snapshot features such as best ask, listing depth,
float distribution, stickers, reference price, and seller reliability. Keep
SteamDT as the source for returns, volatility, labels, and walk-forward backtests.
