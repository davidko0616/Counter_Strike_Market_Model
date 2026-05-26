# TODO Before Proceeding

## Bigger Project Plan

### Project Goal

Build a research-grade CS2 skin market signal system that can answer:

> Is there a skin worth buying today, after realistic costs, liquidity, market regime, and execution constraints?

The end goal is not just a model with high backtest PnL. The end goal is a defensible decision system with:

- no forced trades
- normal/event regimes separated
- realistic execution assumptions
- clear current signals
- dashboard visibility
- reproducible reports

### Where We Are

The pipeline works end to end.

Completed:

- SteamDT collection
- feature factory
- market index/regime features
- event-regime tagging
- triple-barrier labels
- baseline + LightGBM models
- portfolio backtest
- rejection-aware walk-forward validation
- CSFloat snapshot collection
- 150-item MVP v2 universe
- dashboard MVP
- v2 robustness audit
- low-price execution-cost stress test

Main finding:

- Raw v2 looks strong.
- But much of the edge comes from low-price skins.
- After execution realism:
  - raw costed still strong
  - `$1+` is positive but weaker
  - `$5+` is smaller but cleaner

So the current bottleneck is not modeling. It is deciding what is realistically tradable.

### Phase 1: Lock The Trading Policy

Purpose: decide the actual MVP trading universe and execution rules.

Tasks:

1. Finish variant-aware attribution:
   - raw v2 costed
   - `$1+` costed
   - `$5+` costed
2. Compare each policy by:
   - PnL
   - profit factor
   - drawdown
   - trade count
   - item count
   - period concentration
   - price-bucket contribution
   - wear/rarity/category attribution
3. Audit sub-$1 winners:
   - repeated `$0.20` price behavior
   - fill realism
   - listing depth
   - CSFloat availability
   - SteamDT `sellCount`
4. Choose one default policy:
   - allow sub-$1 with harsh capacity limits
   - `$1+` minimum price
   - `$5+` minimum price

Exit condition:

We have one default policy that is defensible and documented.

Current recommendation:

Use `$5+` as the conservative MVP default for headline/paper-trading signals. Keep `$1+` as research mode. Do not headline raw sub-dollar performance yet.

### Phase 2: Execution Realism

Purpose: turn backtest PnL into something closer to tradable PnL.

Tasks:

1. Add capacity rules:
   - max notional per item/day
   - max notional by price bucket
   - max open positions by price bucket
2. Add liquidity-adjusted sizing:
   - based on SteamDT `sellCount`
   - later based on CSFloat listing depth
3. Add spread/slippage model:
   - price bucket
   - listing depth
   - CSFloat spread proxy
4. Add failed-execution probability:
   - higher for low-price items
   - higher for stale data
5. Rerun:
   - Day 16
   - Day 17
   - Day 21
   - Day 22

Exit condition:

The selected policy remains positive after capacity and execution limits.

### Phase 3: CSFloat Real Ablation

Purpose: prove whether CSFloat adds predictive value.

Current blocker:

CSFloat snapshots are newer than the latest SteamDT daily bars, so point-in-time coverage is still zero.

Once coverage is nonzero:

1. Build true SteamDT-only vs SteamDT+CSFloat comparison.
2. Use same labels and same walk-forward splits.
3. Compare:
   - precision
   - accepted PnL
   - profit factor
   - drawdown
   - rejection quality
   - feature importance
4. Decide:
   - keep CSFloat features
   - keep only as execution/liquidity features
   - drop from model training until more history exists

Exit condition:

We know whether CSFloat improves alpha, execution filtering, both, or neither.

### Phase 4: Event-Regime System

Purpose: separate patch/event trades from normal-market trades.

Tasks:

1. Keep normal model and event model separate.
2. Build event metadata:
   - `known_at_utc`
   - `tradable_from_utc`
   - affected rarity/collection/wear
   - `allowed_feature_from_utc`
3. Add event-study reports:
   - October 2025 trade-up shock
   - May 2026 souvenir update
4. Build event logic:
   - what assets are mechanically affected
   - how fast repricing happens
   - whether delayed opportunities exist
5. Never include event winners in normal headline metrics.

Exit condition:

Normal strategy and event strategy have separate reports, metrics, and decision rules.

### Phase 5: Dashboard Upgrade

Purpose: make the system usable.

Dashboard should show:

1. Current accepted signals.
2. Rejected candidates and rejection reasons.
3. Policy toggle:
   - raw
   - `$1+`
   - `$5+`
   - conservative
4. Item detail:
   - price chart
   - model score
   - liquidity
   - CSFloat snapshot
   - historical trade outcomes
5. Portfolio health:
   - equity curve
   - drawdown
   - profit factor
6. Regime view:
   - bull/neutral/bear market index
   - event-regime warnings
7. Robustness panel:
   - capped return PnL
   - top-trade-excluded PnL
   - price-bucket dependency

Exit condition:

A user can open the dashboard and understand whether anything is worth buying today and why.

### Phase 6: Final Research Report

Purpose: document the project cleanly.

Report sections:

1. Objective
2. Data sources
3. Universe construction
4. Features
5. Labels
6. Models
7. Rejection policy
8. Execution assumptions
9. Normal-regime results
10. Event-regime results
11. CSFloat ablation
12. Low-price audit
13. Dashboard usage
14. Limitations
15. Final recommendation

Exit condition:

A complete project report exists and matches the code outputs.

### How We End The Project

We should end when we have a defensible MVP, not when every possible model idea is exhausted.

Final deliverables:

1. Clean committed codebase.
2. Reproducible pipeline commands.
3. Final selected trading policy.
4. Dashboard MVP.
5. Final research report.
6. Clear `do not trade / paper trade / live small-size` recommendation.

Final decision should be one of:

#### A. Do Not Trade

If results disappear after execution realism.

#### B. Paper Trade Only

If backtest is positive but depends on uncertain liquidity/fill assumptions.

#### C. Small-Size Live Test

Only if:

- normal-regime PnL remains positive
- no single price bucket dominates too much
- CSFloat/liquidity checks support execution
- dashboard can explain each accepted trade
- risk limits are defined

Current likely endpoint:

Paper trade first.

The strategy is promising, but not ready for real capital until low-price execution and CSFloat/liquidity validation are stronger.

### Recommended Next 5 Moves

1. Build variant-aware Day 17 attribution for raw / `$1+` / `$5+`.
2. Choose the default policy candidate.
3. Add capacity limits and liquidity-adjusted sizing.
4. Rerun full validation under that policy.
5. Upgrade dashboard to show policy-specific accepted/rejected signals.

That gives us a clean path from interesting research result to defensible MVP.

## Current Status After Day 22

Completed:

1. Current work was committed in `ba91527`.
2. `Reviews and implementation plans/` is now tracked in Git.
3. Low-price execution realism was added through Day 22 policy variants.
4. Minimum entry price gates were implemented for:
   - raw v2 with price-bucket costs
   - `$1+` with price-bucket costs
   - `$5+` with price-bucket costs
5. Price-bucket execution cost assumptions were applied:
   - `<= $1`: extra `500 bps`
   - `$1-$5`: extra `250 bps`
   - `$5-$20`: extra `150 bps`
   - `> $20`: extra `100 bps`
6. Day 22 walk-forward reports were generated.
7. Day 21 robustness reports were rerun for all Day 22 policy variants.

Current LightGBM policy comparison:

| Policy | Accepted Trades | PnL | Win Rate | Profit Factor |
| --- | ---: | ---: | ---: | ---: |
| raw v2 + bucket costs | 628 | 82.5692 | 82.17% | 10.8359 |
| `$1+` + bucket costs | 364 | 15.4240 | 83.79% | 2.1718 |
| `$5+` + bucket costs | 188 | 11.5163 | 87.23% | 4.8854 |

Decision point:

The raw v2 result remains profitable but is still low-price dependent. The `$1+` variant is positive but materially weaker. The `$5+` variant is smaller, cleaner, and has a stronger profit factor. Before adding more model complexity, choose whether the MVP should allow sub-dollar skins, use a `$1+` gate, or use a `$5+` gate.

## Current Status After Day 23

Completed:

1. Variant-aware attribution was added for:
   - raw v2 + bucket costs
   - `$1+` + bucket costs
   - `$5+` + bucket costs
2. Day 23 reports were generated:
   - `reports/tables/day23_policy_attribution_summary.csv`
   - `reports/tables/day23_policy_item_attribution.csv`
   - `reports/tables/day23_policy_period_attribution.csv`
   - `reports/tables/day23_policy_bucket_attribution.csv`
   - `reports/backtests/day23_policy_attribution_report.md`

LightGBM attribution summary:

| Policy | PnL | Trades | Items | Positive Items | Top-3 Item Share | Top-2 Period Share | Median Entry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw v2 + bucket costs | 82.5692 | 628 | 142 | 120 | 19.92% | 61.37% | `$1.07` |
| `$1+` + bucket costs | 15.4240 | 364 | 89 | 69 | 48.01% | 73.29% | `$4.88` |
| `$5+` + bucket costs | 11.5163 | 188 | 54 | 49 | 32.07% | 68.35% | `$22.04` |

Policy recommendation:

Use `$5+` as the conservative MVP default for paper trading and dashboard headline signals. Keep `$1+` as a research mode. Do not headline raw sub-dollar performance until capacity and fill realism for sub-dollar skins are proven.

## Immediate Cleanup

1. [x] Commit current code changes, excluding `.env`.
2. [x] Decide whether to track or ignore `Reviews and implementation plans/`.
3. [x] Confirm generated data/report artifacts remain ignored and are not committed.
4. [x] Keep the daily CSFloat/SteamDT coverage heartbeat active until Day 18 coverage becomes nonzero.

## Validation Before More Modeling

1. [x] Add execution realism for low-price skins.
2. [x] Add minimum entry price gates:
   - `entry_price >= $1`
   - `entry_price >= $5`
3. [x] Add price-bucket slippage penalties.
4. [x] Add price-bucket spread assumptions.
5. [x] Rerun Day 16 walk-forward with these gates.
6. [x] Rerun Day 17 diagnostics with explicit variant-aware item/period attribution.
7. [x] Rerun Day 21 robustness audit.
8. [x] Compare raw v2 vs `$1+` vs `$5+` strategies.

## Low-Price Artifact Audit

1. Inspect top sub-$1 winners manually.
2. Check whether repeated `0.20` prices are real or floor/rounding artifacts.
3. Check whether the large sub-$1 PnL depends on unrealistic fill sizes.
4. Check whether low-price items have enough listing depth using CSFloat and SteamDT `sellCount`.
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

Updated recommended order:

1. Commit Day 22 and Day 23 work.
2. Treat `$5+` as the conservative default policy candidate.
3. Add capacity limits and liquidity-adjusted sizing.
4. Rerun Day 22 and Day 23 with capacity constraints.
5. Upgrade dashboard to expose policy toggles and the `$5+` default.
6. Continue waiting for CSFloat point-in-time coverage so Day 18 can become a true ablation.
