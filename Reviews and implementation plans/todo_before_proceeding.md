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

Updated review finding:

The project should not proceed to capacity sensitivity, dashboard promotion, or headline policy selection until the newly identified validation defects are fixed. The current bottleneck is now correctness of execution math and walk-forward validation, not modeling.

### Critical Blockers Before Proceeding

These must be fixed before relying on Day 22/23 results:

1. Fix price-bucket execution-cost math.
   - Current issue: extra cost is subtracted as a flat percentage from return.
   - Required correction: apply costs through entry/exit execution prices or an equivalent costed gross-return formula.
   - Reason: high-return low-price trades are materially inflated. A `$0.20` skin with a `100%` move and `500 bps` extra cost can be overstated by roughly `9 percentage points`.
   - Consequence: prior raw v2 headline PnL, including the old `82.57` raw v2 result, must be treated as invalid until rerun.
2. Make capacity sizing deterministic.
   - Current issue: equal-score trades on the same day can consume capacity in undefined order.
   - Required correction: sort with a stable tiebreaker such as `market_hash_name` after model/date/score.
3. Harden feature joins for wear variants.
   - Current issue: `enrich_trades_with_features(... validate="many_to_one")` can crash or behave incorrectly when MW/WW variants share item timestamps or duplicated feature rows exist.
   - Required correction: deduplicate feature rows by `market_hash_name, entry_timestamp` before joining, and report duplicate counts.
4. Fix walk-forward train/test mismatch.
   - Current issue: threshold selection can optimize on sub-`$1` trades even when the test policy is `$5+`.
   - Required correction: apply the same base policy gates to training candidates before threshold selection and test candidates before evaluation.
   - Consequence: Day 22/23 policy comparisons must be rerun after this fix.

### Serious Review Items

These are not optional, but can follow the critical blockers:

1. Validate wear-variant physical existence.
   - Do not generate MW/WW variants that cannot exist under the item float range.
   - Add an audit table listing excluded impossible variants.
2. Fix dashboard cache behavior.
   - Dashboard data must refresh or expose a manual refresh path.
   - Stale cached signals are unacceptable for a daily trading decision tool.
3. Change top-N trade exclusion from global to per-period.
   - Global exclusion can hide period concentration and overstate robustness.
   - Reports should support both global and per-period top-N exclusions, with per-period as the default robustness metric.
4. Rename or clarify `CapacitySizingConfig` fields.
   - Current names imply dollar values but the values are portfolio fractions.
   - Rename fields with `_fraction` suffix or add explicit dollar conversion fields.

### Planning Gaps To Add

The project plan must also include:

1. Model retraining cadence.
   - Define when models are retrained: daily, weekly, or after enough new labeled bars.
   - Define minimum new data requirements before retraining.
2. Walk-forward degradation monitoring.
   - Track rolling accepted PnL, profit factor, rejection rate, calibration, and drawdown by month.
   - Trigger investigation when live or paper-trade metrics fall below historical walk-forward expectations.
3. Paper-trade A/B test duration.
   - Run paper trading before live capital.
   - Compare `$5+` conservative policy against `$1+` research mode and no-trade baseline.
   - Require a minimum test duration and minimum accepted-trade count before changing recommendation.
4. Survivorship bias acknowledgment.
   - Document that universe construction may overrepresent items that have survived data, liquidity, and availability filters.
   - Add a report section explaining how future delisted, missing, or failed-collection items are handled.

### Phase 1: Lock The Trading Policy

Purpose: decide the actual MVP trading universe and execution rules.

Tasks:

1. Fix the critical validation blockers before selecting a policy:
   - price-bucket execution-cost math
   - deterministic capacity ordering
   - duplicate-safe feature joins
   - matching train/test policy gates
2. Rerun variant-aware attribution after the fixes:
   - raw v2 costed
   - `$1+` costed
   - `$5+` costed
3. Compare each policy by:
   - PnL
   - profit factor
   - drawdown
   - trade count
   - item count
   - period concentration
   - price-bucket contribution
   - wear/rarity/category attribution
4. Audit sub-$1 winners:
   - repeated `$0.20` price behavior
   - fill realism
   - listing depth
   - CSFloat availability
   - SteamDT `sellCount`
5. Choose one default policy:
   - allow sub-$1 with harsh capacity limits
   - `$1+` minimum price
   - `$5+` minimum price

Exit condition:

We have one default policy that is defensible and documented.

Current recommendation status:

Pause final policy selection until the critical blockers are fixed and Day 22/23 are rerun. The likely candidate remains `$5+` for conservative paper trading, but it should not be finalized from the current reports.

### Phase 2: Execution Realism

Purpose: turn backtest PnL into something closer to tradable PnL.

Tasks:

1. Correct execution-cost math:
   - model cost through entry/exit prices or equivalent multiplicative return math
   - rerun all price-bucket reports after correction
2. Add capacity rules:
   - max notional per item/day
   - max notional by price bucket
   - max open positions by price bucket
3. Add liquidity-adjusted sizing:
   - based on SteamDT `sellCount`
   - later based on CSFloat listing depth
4. Add spread/slippage model:
   - price bucket
   - listing depth
   - CSFloat spread proxy
5. Add failed-execution probability:
   - higher for low-price items
   - higher for stale data
6. Rerun:
   - Day 16
   - Day 17
   - Day 21
   - Day 22
   - Day 23

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
8. Data freshness and cache status:
   - latest report timestamp
   - latest feature timestamp
   - latest CSFloat snapshot timestamp
   - manual refresh or cache-clear control

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
15. Model retraining cadence
16. Walk-forward degradation monitoring
17. Paper-trade A/B test design
18. Survivorship bias and universe limitations
19. Final recommendation

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

1. Fix execution-cost math so price-bucket costs are applied through entry/exit execution prices.
2. Fix walk-forward policy consistency so training and test use the same minimum-price/capacity gates.
3. Make capacity sizing deterministic and rename fraction-based config fields clearly.
4. Add duplicate-safe feature joins and valid-wear universe checks.
5. Rerun Day 22/23 and only then resume capacity sensitivity or dashboard promotion.

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

Invalidated LightGBM policy comparison:

| Policy | Accepted Trades | PnL | Win Rate | Profit Factor |
| --- | ---: | ---: | ---: | ---: |
| raw v2 + bucket costs | 628 | 82.5692 | 82.17% | 10.8359 |
| `$1+` + bucket costs | 364 | 15.4240 | 83.79% | 2.1718 |
| `$5+` + bucket costs | 188 | 11.5163 | 87.23% | 4.8854 |

Status:

These Day 22 numbers are retained only as historical context. They should not be used for policy selection because the price-bucket cost math and walk-forward train/test gating need correction.

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

Invalidated LightGBM attribution summary:

| Policy | PnL | Trades | Items | Positive Items | Top-3 Item Share | Top-2 Period Share | Median Entry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw v2 + bucket costs | 82.5692 | 628 | 142 | 120 | 19.92% | 61.37% | `$1.07` |
| `$1+` + bucket costs | 15.4240 | 364 | 89 | 69 | 48.01% | 73.29% | `$4.88` |
| `$5+` + bucket costs | 11.5163 | 188 | 54 | 49 | 32.07% | 68.35% | `$22.04` |

Status:

These Day 23 numbers are retained only as historical context. The current policy recommendation is paused until corrected Day 22/23 reports exist.

## Current Status After Capacity Review

Review conclusion:

1. Do not proceed to capacity sensitivity yet.
2. Do not promote `$5+` to dashboard default yet.
3. Do not headline raw v2, `$1+`, or `$5+` PnL from the current Day 22/23 reports.
4. Treat the latest Day 22/23 implementation as a useful scaffold that needs correctness fixes before results are trusted.

Immediate fix queue:

1. [x] Correct price-bucket execution-cost math.
2. [x] Add deterministic capacity tie-breaking with `market_hash_name`.
3. [x] Add feature join deduplication and duplicate diagnostics.
4. [x] Apply base policy gates consistently in walk-forward training and test evaluation.
5. [x] Rename `CapacitySizingConfig` fields that are fractions, not dollars.
6. [x] Add tests proving all five fixes.
7. [x] Rerun Day 22 and Day 23 after fixes.
8. [x] Replace invalidated report tables with corrected results.

## Current Status After Price-Cost Math Fix

Completed:

1. Price-bucket extra execution costs now use entry/exit price multipliers instead of flat return subtraction.
2. High-return cost math has a regression test.
3. Day 22 and Day 23 were rerun once with corrected cost math.

Corrected but still provisional LightGBM results:

| Policy | Capacity PnL | Trades | Profit Factor | Win Rate |
| --- | ---: | ---: | ---: | ---: |
| raw v2 costed | 0.7314 | 642 | 2.7641 | 68.54% |
| `$1+` costed | 0.2025 | 403 | 1.8471 | 72.70% |
| `$5+` costed | 0.1513 | 196 | 3.1912 | 87.24% |

Status:

These results confirm the old headline PnL was inflated. They are still provisional because deterministic capacity ordering, duplicate-safe feature joins, and walk-forward train/test gate consistency are not fixed yet.

## Current Status After Walk-Forward Gate Fix

Completed:

1. Walk-forward threshold training now uses the same base policy candidate universe as test evaluation.
2. Training and test rows are base-gated before score-threshold optimization and evaluation.
3. Selection output now records both raw and policy-eligible train/test counts.
4. A regression test verifies that sub-`$1` rows cannot influence threshold training for a `$5+` policy.
5. Day 22 and Day 23 were rerun after the fix.

Corrected but still provisional LightGBM results:

| Policy | Capacity PnL | Trades | Profit Factor | Win Rate |
| --- | ---: | ---: | ---: | ---: |
| raw v2 costed | 0.7314 | 642 | 2.7641 | 68.54% |
| `$1+` costed | 0.2025 | 403 | 1.8471 | 72.70% |
| `$5+` costed | 0.1007 | 232 | 1.7592 | 81.03% |

Status:

The `$5+` policy weakened after fixing train/test gate consistency. These results are still provisional until deterministic capacity ordering and duplicate-safe feature joins are fixed.

## Current Status After Deterministic Capacity Ordering Fix

Completed:

1. Capacity allocation now sorts by model, entry timestamp, descending score, `market_hash_name`, and `event_id`.
2. The sort uses a stable algorithm so tied rows allocate capacity repeatably.
3. A regression test verifies tied-score trades consume bucket capacity in `market_hash_name` order.
4. Day 22 and Day 23 were rerun after the fix.

Corrected but still provisional LightGBM results:

| Policy | Capacity PnL | Trades | Profit Factor | Win Rate |
| --- | ---: | ---: | ---: | ---: |
| raw v2 costed | 0.7314 | 642 | 2.7641 | 68.54% |
| `$1+` costed | 0.2025 | 403 | 1.8471 | 72.70% |
| `$5+` costed | 0.1007 | 232 | 1.7592 | 81.03% |

Status:

The deterministic ordering fix did not change the current report numbers, but it removes hidden non-repeatability. Results remain provisional until duplicate-safe feature joins and capacity config naming are fixed.

## Current Status After Duplicate-Safe Feature Join Fix

Completed:

1. `enrich_trades_with_features` now deduplicates feature rows by `market_hash_name, entry_timestamp` before merging.
2. The join still uses `validate="many_to_one"` after deduplication.
3. Enriched trades now include:
   - `feature_join_duplicate_count`
   - `feature_join_duplicate_conflict`
   - `feature_join_conflicting_columns`
4. Strict mode can fail fast on conflicting duplicate feature rows.
5. Top-trade audit flags feature-join conflicts.
6. Day 22 and Day 23 were rerun after the fix.
7. Current Day 22 `$5+` accepted trades had no duplicate feature conflicts.

Corrected but still provisional LightGBM results:

| Policy | Capacity PnL | Trades | Profit Factor | Win Rate |
| --- | ---: | ---: | ---: | ---: |
| raw v2 costed | 0.7314 | 642 | 2.7641 | 68.54% |
| `$1+` costed | 0.2025 | 403 | 1.8471 | 72.70% |
| `$5+` costed | 0.1007 | 232 | 1.7592 | 81.03% |

Strict validation:

`pytest -W error` passes for the full test suite. Current count: `100 passed`.

Status:

Results remain provisional until wear-variant validity is audited.

## Current Status After Capacity Config Naming Fix

Completed:

1. `CapacitySizingConfig` fields that represent portfolio fractions now use `_fraction` suffixes:
   - `max_notional_per_trade_fraction`
   - `max_item_daily_notional_fraction`
   - `max_bucket_daily_notional_fraction`
   - `minimum_executable_notional_fraction`
2. Call sites and tests were updated to use the clarified names.
3. Day 22 and Day 23 were rerun after the rename.

Corrected but still provisional LightGBM results:

| Policy | Capacity PnL | Trades | Profit Factor | Win Rate |
| --- | ---: | ---: | ---: | ---: |
| raw v2 costed | 0.7314 | 642 | 2.7641 | 68.54% |
| `$1+` costed | 0.2025 | 403 | 1.8471 | 72.70% |
| `$5+` costed | 0.1007 | 232 | 1.7592 | 81.03% |

Strict validation:

`pytest -W error` passes for the full test suite. Current count: `100 passed`.

Status:

The original critical code blockers are fixed. Remaining serious review items include wear-variant validity, dashboard cache refresh, per-period top-N exclusion, and broader corrected reruns.

## Immediate Cleanup

1. [x] Commit current code changes, excluding `.env`.
2. [x] Decide whether to track or ignore `Reviews and implementation plans/`.
3. [x] Confirm generated data/report artifacts remain ignored and are not committed.
4. [x] Keep the daily CSFloat/SteamDT coverage heartbeat active until Day 18 coverage becomes nonzero.

## Validation Before More Modeling

1. [ ] Correct execution realism for low-price skins.
2. [x] Add minimum entry price gates:
   - `entry_price >= $1`
   - `entry_price >= $5`
3. [x] Fix price-bucket slippage/spread math.
4. [x] Rerun walk-forward policy comparison with matching train/test gates.
5. [ ] Rerun Day 17 diagnostics with corrected policy variants.
6. [ ] Rerun Day 21 robustness audit with corrected cost math.
7. [ ] Rerun Day 22 and Day 23 with corrected capacity sizing.
8. [ ] Compare raw v2 vs `$1+` vs `$5+` strategies using corrected outputs.

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
3. Check MW/WW variants for valid float ranges and physical existence.
4. Confirm the use of SteamDT `sellCount` as a liquidity proxy is acceptable.
5. Consider adding `price_single` liquidity collection as a separate reusable collector.
6. Add liquidity snapshot artifact for reproducibility.
7. Add an excluded-variant audit table for impossible wear variants.

## Dashboard Work

1. Add rejected candidates view.
2. Add rejection reason details.
3. Add price-bucket filters.
4. Add minimum price policy toggle.
5. Add CSFloat coverage panel.
6. Add low-price warning indicators.
7. Add model comparison view.
8. Add Day 21 robustness summary panel.
9. Fix dashboard cache refresh behavior.
10. Show data/report freshness timestamps.
11. Add manual refresh or cache-clear control.

## Research Improvements

1. Add normal-only headline metrics everywhere.
2. Add capped-return headline metrics.
3. Add per-period top-trade-excluded headline metrics.
4. Add per-price-bucket PnL to every backtest report.
5. Add per-wear and per-rarity attribution to Day 17 or Day 21.
6. Add event-regime reports separately from normal-regime reports.
7. Add capacity-adjusted PnL.
8. Add rejection curves by price bucket.
9. Add model retraining cadence.
10. Add walk-forward degradation monitoring.
11. Add paper-trade A/B test duration and success criteria.
12. Add survivorship bias checks and documentation.

## Code Hygiene

1. Refactor duplicated rejection gate logic.
2. Add retry/backoff to CSFloat collector.
3. Fix notebook `Path.cwd()` hardcoding.
4. Consolidate duplicated `_display_path` helpers later.
5. Add better input validation to Day 18/19 modules.
6. Add tests for low-price gates and corrected price-bucket execution costs.
7. Add deterministic capacity-ordering tests.
8. Add duplicate-safe feature-join tests. Completed.
9. Add train/test policy-gate consistency tests.
10. Rename capacity config fields that represent fractions.

## Recommended Order

1. Commit current work.
2. Add low-price execution realism.
3. Rerun Day 16, Day 17, and Day 21 with `$1+` and `$5+` policy variants.
4. Decide whether sub-$1 items are tradable or should be excluded.
5. Improve Day 18 real ablation once CSFloat coverage becomes nonzero.
6. Upgrade dashboard after the trading policy is more realistic.

Updated recommended order after capacity review:

1. Fix price-bucket cost math.
2. Fix walk-forward train/test policy consistency.
3. Fix deterministic capacity ordering.
4. Fix duplicate-safe feature joins.
5. Validate MW/WW variants and remove impossible variants.
6. Rerun Day 22 and Day 23.
7. Rerun Day 21 robustness with corrected cost math.
8. Only then run `$5+` capacity sensitivity.
9. Upgrade dashboard after corrected `$5+` results are available.
10. Continue waiting for CSFloat point-in-time coverage so Day 18 can become a true ablation.
