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

Wear-variant validity audit is now complete; see the later wear-variant status section.

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

The original critical code blockers are fixed. Remaining serious review items include dashboard cache refresh, per-period top-N exclusion, and broader corrected reruns.

## Current Status After Wear-Variant Validity Audit

Completed:

1. MW/WW variant generation now intersects each requested wear range with the item's actual `min_float, max_float`.
2. Impossible variants are excluded instead of having their float range reset to a generic wear range.
3. `reports/tables/day19_excluded_wear_variants.csv` is written on each Day 19 universe rebuild.
4. Strict tests cover valid intersections and impossible variant exclusions.
5. The current v2 universe rebuild produced:
   - 110 base Field-Tested items
   - 40 valid MW/WW variant items
   - 0 excluded impossible variants
   - 150 total v2 universe items
6. Day 22 and Day 23 were rerun after the audit.

Corrected LightGBM results after wear-variant audit:

| Policy | Capacity PnL | Trades | Profit Factor | Win Rate |
| --- | ---: | ---: | ---: | ---: |
| raw v2 costed | 0.7314 | 642 | 2.7641 | 68.54% |
| `$1+` costed | 0.2025 | 403 | 1.8471 | 72.70% |
| `$5+` costed | 0.1007 | 232 | 1.7592 | 81.03% |

Strict validation:

`pytest -W error` passes for the full test suite. Current count: `104 passed`.

Status:

The original critical blockers, wear-variant serious review item, corrected Day 21/17 reruns, and per-period top-N exclusion are fixed. The next serious item is dashboard cache freshness, but the immediate research move is `$5+` capacity sensitivity under harsher robustness assumptions.

## Current Status After Corrected Day 21/17 Reruns

Completed on the corrected Day 22 policy outputs:

1. Day 21 robustness now includes both global top-winner removal and per-period top-winner removal.
2. Day 21 drops stale feature-join diagnostics from accepted-trade inputs before attaching fresh diagnostics.
3. Day 17 has a corrected policy-variant mode that reads Day 22 accepted trades and threshold selections directly.
4. Strict tests cover stale feature-join diagnostic replacement and corrected policy-variant summaries.

Corrected LightGBM Day 21 stress results:

| Policy | Raw PnL | 50% Cap PnL | Remove Top 5 Global | Remove Top 5 Per Period | Profit Factor Raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw v2 costed | 0.7314 | 0.5084 | 0.4334 | 0.1864 | 2.7641 |
| `$1+` costed | 0.2025 | 0.1841 | 0.1210 | -0.0302 | 1.8471 |
| `$5+` costed | 0.1007 | 0.0918 | 0.0580 | -0.0244 | 1.7592 |

Corrected Day 17 LightGBM policy summary:

| Policy | Accepted Trades | Rejected Trades | Rejection Rate | Accepted PnL | Profit Factor | Median Entry Close |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw v2 costed | 642 | 996 | 60.81% | 0.7314 | 2.7641 | 1.14 |
| `$1+` costed | 403 | 594 | 59.58% | 0.2025 | 1.8471 | 4.82 |
| `$5+` costed | 232 | 322 | 58.12% | 0.1007 | 1.7592 | 23.04 |

Interpretation:

The broader corrected diagnostics are harsher than the headline Day 22/23 result. Global top-trade removal remains positive, but per-period top-winner removal turns `$1+` and `$5+` negative. This means the next `$5+` capacity sensitivity should test concentration and fill realism aggressively before treating `$5+` as a defensible default.

## Current Status After `$5+` Capacity Sensitivity

Completed:

1. Capacity sizing now supports explicit open-position limits:
   - max open positions
   - max open positions per execution price bucket
2. Day 24 runs `$5+` capacity sensitivity without overwriting Day 22 baseline outputs.
3. Sensitivity scenarios test stricter item/day, bucket/day, liquidity participation, and open-position caps.
4. Outputs:
   - `reports/tables/day24_min_price_5_capacity_sensitivity.csv`
   - `reports/tables/day24_min_price_5_capacity_period_attribution.csv`
   - `reports/backtests/day24_min_price_5_capacity_sensitivity_report.md`

LightGBM `$5+` capacity sensitivity:

| Scenario | Trades | Capacity Rejected | PnL | Profit Factor | Max Drawdown | Top 2 Period PnL Share | Positive Period Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_day22 | 232 | 3 | 0.1007 | 1.7592 | -0.0381 | 67.77% | 90.91% |
| strict_daily_half | 228 | 7 | 0.0499 | 1.7254 | -0.0200 | 68.43% | 90.91% |
| strict_open_10 | 175 | 60 | 0.0332 | 2.0756 | -0.0090 | 63.53% | 90.91% |
| severe_open_5 | 125 | 110 | 0.0217 | 2.5699 | -0.0034 | 58.03% | 81.82% |

Interpretation:

The `$5+` policy survives the configured capacity stress suite, but not strongly enough for a live-trading recommendation. The correct status is conservative default candidate for paper trading. Period concentration remains material, so the next work should strengthen rejection gates and dashboard visibility rather than loosen capacity assumptions.

## Current Status After Dashboard Upgrade

Completed:

1. Dashboard now uses the corrected `$5+` conservative policy artifacts by default.
2. Current signal view keeps both accepted and rejected candidates.
3. Rejected candidates show explicit rejection reasons:
   - below minimum price
   - low score
   - low liquidity
   - missing price
4. Capacity tab shows Day 24 stress scenarios, survival status, drawdown, profit factor, and period concentration.
5. Freshness tab shows artifact paths, row counts, and modification timestamps.
6. Dashboard artifact loading no longer uses stale Streamlit cache.
7. Local dashboard verified at `http://localhost:8508`.

Status:

The dashboard is now aligned with the corrected research position: `$5+` is the conservative default candidate for paper trading, while capacity concentration and rejected candidates remain visible.

## Current Status After Stricter `$5+` Paper-Trade Gates

Completed:

1. Added default-off rejection policy gates for:
   - minimum recent item excess return
   - maximum 30-day absolute return instability
2. Added `paper_trade_5_plus` to `configs/backtest.yaml`.
3. Wired the new gates through scalar rejection, vectorized rejection, YAML policy loading, rejection curves, and normal-only summaries.
4. Added Day 25 paper-trade rejection analysis:
   - `src/cs_market_model/research/day25_paper_trade_rejection.py`
   - `reports/tables/day25_paper_trade_rejection_summary.csv`
   - `reports/tables/day25_paper_trade_rejection_period_attribution.csv`
   - `reports/backtests/day25_paper_trade_rejection_report.md`
5. Added regression tests for the new gates and Day 25 rejection-source accounting.

Day 25 LightGBM comparison:

| Scenario | Accepted | Base-Gate Rejected | Score Rejected | Capacity Rejected | PnL | Profit Factor | Max Drawdown | Top-2 Period Share | Positive Period Share | Survives Stress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current Day 24 `$5+` gates | 195 | 1083 | 347 | 2 | 0.0413 | 1.3241 | -0.0379 | 60.06% | 90.91% | Yes |
| stricter `$5+` gates | 94 | 1532 | 0 | 1 | 0.1022 | 1.8276 | -0.0260 | 145.86% | 36.36% | No |
| stricter `$5+` gates + strict open 10 | 93 | 1532 | 0 | 2 | 0.0349 | 1.8473 | -0.0088 | 142.49% | 36.36% | No |

Interpretation:

The stricter gates improve isolated PnL under current capacity, but they do not clear the paper-trade survival rules because period concentration worsens and only 36.36% of periods are positive. This means the stricter `$5+` policy is implemented as a candidate gate set, not a promoted trading decision. Keep live capital rejected and treat this as research-mode until dashboard visibility, degradation monitoring, and paper-trade A/B criteria are added.

CSFloat listing-count gates are intentionally not part of this stricter policy yet because point-in-time coverage is still the Day 18 blocker.

## Current Status Before Finalization

Completed:

1. Repo-wide Ruff now passes.
2. The full test suite passes locally.
3. Day 19 CSFloat coverage no longer crashes on mixed timezone values.
4. Day 19 and Day 18 both run cleanly and produce status reports.
5. Current local CSFloat coverage is explicitly reported as unavailable:
   - 150 price-bar items
   - 0 items with local CSFloat snapshot features
   - 0 post-snapshot covered items
   - 0 CSFloat-covered feature rows
6. Final submission docs were added:
   - `docs/final_submission_report.md`
   - `SUBMISSION_CHECKLIST.md`

Interpretation:

CSFloat true model ablation cannot be produced from the current local artifacts because the project has no usable local CSFloat snapshot feature rows. This is no longer a code blocker. It is documented as an external data availability limitation, and the project is ready to finalize as a research MVP with a paper-trade-only recommendation.

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
5. [x] Rerun Day 17 diagnostics with corrected policy variants.
6. [x] Rerun Day 21 robustness audit with corrected cost math.
7. [x] Rerun Day 22 and Day 23 with corrected capacity sizing.
8. [x] Compare raw v2 vs `$1+` vs `$5+` strategies using corrected outputs.

## Low-Price Artifact Audit

1. Inspect top sub-$1 winners manually.
2. Check whether repeated `0.20` prices are real or floor/rounding artifacts.
3. Check whether the large sub-$1 PnL depends on unrealistic fill sizes.
4. Check whether low-price items have enough listing depth using CSFloat and SteamDT `sellCount`.
5. Add capacity limits by price bucket.
6. Add max notional per item/day.
7. [x] Add max position count per price bucket.

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

1. [x] Verify v2 universe selection logic is acceptable.
2. [x] Review `configs/universe_mvp_v2.yaml`.
3. [x] Check MW/WW variants for valid float ranges and physical existence.
4. Confirm the use of SteamDT `sellCount` as a liquidity proxy is acceptable.
5. Consider adding `price_single` liquidity collection as a separate reusable collector.
6. Add liquidity snapshot artifact for reproducibility.
7. [x] Add an excluded-variant audit table for impossible wear variants.

## Dashboard Work

1. [x] Add rejected candidates view.
2. [x] Add rejection reason details.
3. Add price-bucket filters.
4. [x] Add minimum price policy toggle.
5. Add CSFloat coverage panel.
6. Add low-price warning indicators.
7. Add model comparison view.
8. [x] Add Day 24 capacity stress summary panel.
9. [x] Fix dashboard cache refresh behavior.
10. [x] Show data/report freshness timestamps.
11. [x] Add manual refresh or cache-clear control.

## Research Improvements

1. Add normal-only headline metrics everywhere.
2. Add capped-return headline metrics.
3. [x] Add per-period top-trade-excluded headline metrics.
4. Add per-price-bucket PnL to every backtest report.
5. [x] Add per-wear and per-rarity attribution to Day 17 or Day 21.
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
11. Add impossible wear-variant exclusion tests.

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
5. Validate MW/WW variants and remove impossible variants. Completed.
6. Rerun Day 22 and Day 23. Completed.
7. Rerun Day 21 robustness with corrected cost math. Completed.
8. Only then run `$5+` capacity sensitivity. Completed.
9. Upgrade dashboard after corrected `$5+` results are available. Completed.
10. Continue waiting for CSFloat point-in-time coverage so Day 18 can become a true ablation.
11. Add stricter rejection gates for the `$5+` paper-trade candidate. Completed.
12. Add dashboard price-bucket filters, CSFloat coverage panel, low-price warnings, and model comparison view.
13. Add model retraining cadence and walk-forward degradation monitoring.
14. Add paper-trade A/B test duration and success criteria.
