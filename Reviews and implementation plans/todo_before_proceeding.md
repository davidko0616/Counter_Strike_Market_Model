# TODO Before Proceeding

## Immediate Cleanup

1. Commit current code changes, excluding `.env`.
2. Decide whether to track or ignore `Reviews and implementation plans/`.
3. Confirm generated data/report artifacts remain ignored and are not committed.
4. Keep the daily CSFloat/SteamDT coverage heartbeat active until Day 18 coverage becomes nonzero.

## Validation Before More Modeling

1. Add execution realism for low-price skins.
2. Add minimum entry price gates:
   - `entry_price >= $1`
   - `entry_price >= $5`
3. Add price-bucket slippage penalties.
4. Add price-bucket spread assumptions.
5. Rerun Day 16 walk-forward with these gates.
6. Rerun Day 17 diagnostics.
7. Rerun Day 21 robustness audit.
8. Compare raw v2 vs `$1+` vs `$5+` strategies.

## Low-Price Artifact Audit

1. Inspect top sub-$1 winners manually.
2. Check whether repeated `0.20` prices are real or floor/rounding artifacts.
3. Check whether the large sub-$1 PnL depends on unrealistic fill sizes.
4. Check whether low-price items have enough listing depth.
5. Add capacity limits by price bucket.
6. Add max notional per item/day.
7. Add max position count per price bucket.

## CSFloat Work

1. Wait for SteamDT bars after CSFloat snapshot timestamps.
2. Once coverage is nonzero, rerun Day 18.
3. Convert Day 18 from coverage setup into true model ablation.
4. Compare SteamDT-only vs SteamDT+CSFloat on the same splits.
5. Add CSFloat retry/backoff for `429` and `503`.
6. Add latest-snapshot-per-item summary.
7. Add stale snapshot diagnostics.
8. Add v2-only CSFloat coverage report.

## Universe Expansion Checks

1. Verify v2 universe selection logic is acceptable.
2. Review `configs/universe_mvp_v2.yaml`.
3. Check MW/WW variants for valid float ranges.
4. Confirm the use of SteamDT `sellCount` as a liquidity proxy is acceptable.
5. Consider adding `price_single` liquidity collection as a separate reusable collector.
6. Add liquidity snapshot artifact for reproducibility.

## Dashboard Work

1. Add rejected candidates view.
2. Add rejection reason details.
3. Add price-bucket filters.
4. Add minimum price policy toggle.
5. Add CSFloat coverage panel.
6. Add low-price warning indicators.
7. Add model comparison view.
8. Add Day 21 robustness summary panel.

## Research Improvements

1. Add normal-only headline metrics everywhere.
2. Add capped-return headline metrics.
3. Add top-trade-excluded headline metrics.
4. Add per-price-bucket PnL to every backtest report.
5. Add per-wear and per-rarity attribution to Day 17 or Day 21.
6. Add event-regime reports separately from normal-regime reports.
7. Add capacity-adjusted PnL.
8. Add rejection curves by price bucket.

## Code Hygiene

1. Refactor duplicated rejection gate logic.
2. Add retry/backoff to CSFloat collector.
3. Fix notebook `Path.cwd()` hardcoding.
4. Consolidate duplicated `_display_path` helpers later.
5. Add better input validation to Day 18/19 modules.
6. Add tests for low-price gates and price-bucket execution costs.

## Recommended Order

1. Commit current work.
2. Add low-price execution realism.
3. Rerun Day 16, Day 17, and Day 21 with `$1+` and `$5+` policy variants.
4. Decide whether sub-$1 items are tradable or should be excluded.
5. Improve Day 18 real ablation once CSFloat coverage becomes nonzero.
6. Upgrade dashboard after the trading policy is more realistic.
