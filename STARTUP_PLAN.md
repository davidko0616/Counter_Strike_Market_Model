# Counter-Strike Skin Market Model Startup Plan

## 1. Project Goal

Build a research-grade MVP for net-return-aware Counter-Strike skin buy-signal prediction. The system should not simply forecast future prices. It should rank items by whether they are worth buying after fees, spread, slippage, liquidity constraints, and holding-period risk.

The first version should prove the pipeline on a small, reliable universe of liquid items before expanding to rare, float-sensitive, or cross-market strategies.

Target output for each item-date or item-event:

- `P(Strong Buy)`
- `P(Good Time)`
- `P(Bad Time)`
- expected net return
- downside estimate
- liquidity score
- confidence score
- reason codes

## 2. Recommended MVP Scope

Keep the first version narrow enough to finish and evaluate honestly.

MVP universe:

- 100 to 300 liquid items.
- Start with cases, AK-47 skins, AWP skins, M4A1-S/M4A4 skins, and popular stickers.
- Exclude knives, gloves, rare patterns, and complex sticker crafts from the first modeling pass unless the data is already clean.

Current implementation note:

- The first pull has been narrowed to gun skins only after project review.
- Cases, stickers, knives, gloves, and sticker price data are excluded from the current MVP data pull.
- The Day 3 first-pull universe uses 48 Field-Tested gun skins, with 4 skins each for 12 liquid weapons across Covert, Classified, Restricted, and Mil-Spec rarity slots.
- Other wears should be added later as an expansion once the first Field-Tested pipeline is validated end to end.

MVP venues and APIs:

- Use both SteamDT and CSFloat, but assign them different responsibilities.
- SteamDT is the primary historical price source.
- CSFloat is the secondary listing, float, sticker, and microstructure source.
- Build the first working pipeline with SteamDT historical bars, then add CSFloat snapshots as additional features.
- Treat cross-market spread features as Phase 2, not a blocker for the first model.

Current execution-cost note:

- Do not use Steam marketplace transaction fees for the current SteamDT-based labels.
- SteamDT prices are treated as a proxy for Youpin/Buff-style markets.
- Current baseline uses Buff's 2.5% transaction fee, with separate placeholder slippage.
- Youpin-like 1.0%, Buff-like 2.5%, and CSFloat-like 2.0% fee scenarios are tracked in label sensitivity reports.

Initial API split:

| Source | Main Role | Data Used |
| --- | --- | --- |
| SteamDT Open API | Historical market time series | K-line price history, volume, price bars |
| CSFloat API | Listing and item-level microstructure | Active listings, float values, stickers, reference prices, listing depth |

The first ablation should compare `SteamDT only` against `SteamDT + CSFloat` so the project can measure whether CSFloat-derived features actually improve buy-signal quality.

MVP prediction task:

- Daily event-based buy-signal classification.
- Labels generated using net-return-aware Triple Barrier logic.
- Walk-forward evaluation by time.

MVP success criteria:

- End-to-end reproducible pipeline from raw data to ranked signals.
- Baselines implemented before advanced models.
- Performance reported with `precision@top_k`, after-fee net return, max drawdown, and calibration quality.
- At least one ablation table showing whether each feature group improves results.

## 3. Existing Assets To Reuse

There is a sibling folder:

`C:\Users\chung\Desktop\2026-1학기\인공지능기초수학\cs2_price_model`

Useful files found there:

- `explore_api.py`: probes the SteamDT K-line endpoint and saves raw JSON.
- `probe_csfloat.py`: probes CSFloat listings/history-style endpoints.
- `data/raw/kline/AK-47__Redline_Field-Tested_type2.json`: sample raw K-line data.
- `csfloat_probe_results.json`: sample CSFloat listings response.

Important cleanup before reuse:

- Remove hardcoded API keys from scripts.
- Rotate any keys that were committed or shared.
- Load secrets from `.env` or local environment variables.
- Add `.env` to `.gitignore`.

## 4. Proposed Repository Structure

Use a simple Python project layout:

```text
Counter Strike Market Model/
  README.md
  STARTUP_PLAN.md
  pyproject.toml
  .gitignore
  .env.example
  configs/
    universe_mvp.yaml
    universe_day3_first_pull.yaml
    universe_gun_skins_candidate.yaml
    venues.yaml
    fees.yaml
    backtest.yaml
    market_events.yaml
  data/
    raw/
    interim/
    processed/
    features/
    labels/
  notebooks/
    01_data_audit.ipynb
    02_feature_distributions.ipynb
    03_model_error_analysis.ipynb
    04_regime_visualization.ipynb
  reports/
    figures/
    tables/
    backtests/
  src/
    cs_market_model/
      __init__.py
      config.py
      utils.py
      collectors/
        steamdt.py
        steamdt_batch.py
        csfloat.py
      normalization/
        items.py
        prices.py
        order_books.py
      features/
        momentum.py
        volatility.py
        liquidity.py
        indices.py
        market_index.py
        events.py
        ranks.py
        build_features.py
      labeling/
        cusum.py
        triple_barrier.py
        sensitivity.py
        build_labels.py
      models/
        baselines.py
        train.py
        lightgbm_train.py
        calibrate.py
        explain.py
      backtesting/
        execution.py
        portfolio.py
        walk_forward.py
        metrics.py
        audit_outliers.py
      dashboard/
        app.py
      security/
        env.py
        scan_secrets.py
  tests/
    test_labeling.py
    test_features.py
    test_event_features.py
    test_market_index.py
    test_baseline_models.py
    test_lightgbm_train.py
    test_calibration.py
    test_walk_forward.py
    test_portfolio_backtest.py
    test_outlier_audit.py
    test_price_normalization.py
    test_item_metadata.py
    test_steamdt_batch.py
    test_project_skeleton.py
```

## 5. Technology Choices

Recommended stack:

- Python 3.11 or newer.
- `pandas`, `numpy`, `pyarrow` for data processing.
- `scikit-learn` for baselines, calibration, and metrics.
- `lightgbm` as the first advanced model.
- `xgboost` or `catboost` later if time allows.
- `polars` optional if data grows large.
- `duckdb` optional for local analytics over parquet.
- `pydantic` or dataclasses for config/schema validation.
- `pytest` for critical logic tests.
- `streamlit` for the first dashboard, unless a web app is specifically required.

Storage format:

- Raw API responses as JSONL or JSON.
- Clean bars/features/labels as parquet.
- Model artifacts as `joblib` or model-native files.

## 6. API and Data Plan

### 6.1 API Strategy

Use both APIs from the start of the design, but do not mix their responsibilities.

SteamDT Open API:

- Base URL: `https://open.steamdt.com`
- Primary endpoint: `/open/cs2/item/v1/kline`
- Probe endpoint: `/open/cs2/v1/price/single`
- Main use: historical price bars and volume.
- Modeling role: source for returns, volatility, CUSUM events, Triple Barrier labels, and first baseline models.

CSFloat API:

- Base URL: `https://csfloat.com`
- Primary endpoint: `/api/v1/listings`
- Experimental history-style endpoints may be probed, but should not be relied on until verified.
- Main use: listing snapshots, float values, sticker data, reference prices, and liquidity proxies.
- Modeling role: source for listing depth, scarcity, float/sticker, and execution-realism features.

Join key:

- Use `market_hash_name` as the first join key.
- Normalize names into stable internal `item_id` values.
- Preserve source-specific timestamps so features remain point-in-time safe.

Implementation order:

1. Build SteamDT collector and normalized `price_bars`.
2. Build features and labels from SteamDT only.
3. Train baselines and LightGBM using SteamDT-only features.
4. Build CSFloat listing snapshot collector.
5. Add CSFloat-derived features with timestamp-safe joins.
6. Run ablation: `SteamDT only` vs. `SteamDT + CSFloat`.

### 6.2 Raw Data Layer

Store untouched API responses with both source timestamps and ingestion timestamps.

Minimum raw fields:

- `source`
- `venue`
- `item_hash_name`
- `timestamp_exchange`
- `timestamp_ingested`
- `raw_price`
- `raw_volume`
- `currency`
- `request_status`
- `raw_json`

Do not overwrite raw files. Raw data should be append-only so bugs in cleaning logic can be fixed later.

### 6.3 Clean Price Bars

Create normalized daily bars:

- `item_id`
- `market_hash_name`
- `venue`
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `median_price`
- `volume`
- `trade_count`
- `currency`
- `data_resolution`
- `is_forward_filled`
- `staleness_days`

The `is_forward_filled` and `staleness_days` fields are important because illiquid items can look artificially stable.

### 6.4 Item Metadata

Create an item table:

- `item_id`
- `market_hash_name`
- `weapon_type`
- `collection`
- `rarity`
- `wear`
- `stattrak_flag`
- `souvenir_flag`
- `case_source`
- `drop_status`
- `release_date`
- `min_float`
- `max_float`

For the MVP, some fields may be manually curated in `configs/universe_mvp.yaml`.

### 6.5 CSFloat Listings and Microstructure Data

Use listings data to estimate:

- best ask
- sell listing count
- listing depth near market price
- float distribution
- sticker count
- reference price
- predicted price
- seller trade reliability if available

Only include these features in training when their timestamps are point-in-time safe.

## 7. Feature Plan

Build features in small groups so ablation testing is easy.

Feature group A: price momentum

- 1-day, 7-day, 14-day, 30-day log returns.
- Moving average distance.
- Recent high/low position.

Feature group B: volatility

- Rolling return standard deviation.
- EWMA volatility.
- Realized volatility percentile.
- Drawdown from recent high.

Feature group C: volume and liquidity

- Rolling volume.
- Relative volume.
- Volume acceleration.
- Days since last real sale.
- Forward-fill/staleness flags.

Feature group D: synthetic indices

- Internal CS2 MVP equal-weight market index.
- CS2 index returns, volatility, drawdown, trend, and regime flags.
- Item excess return versus the CS2 MVP market index.
- Item beta, correlation, and residual volatility versus the CS2 MVP market index.
- Equal-weight category index.
- Median category index.
- Liquidity-filtered category index.
- Item return minus category return.
- Item price/category index ratio.
- Market-regime breadth features, such as universe/category share above moving average and median drawdown, so broad risk-off periods can be modeled.

Feature group E: order-book/listing features

- Best ask.
- Listing count.
- Estimated spread if bid data exists.
- Sell-side depth within 1%, 3%, and 5%.
- Listing wall distance.

Feature group F: event features

- Major tournament period.
- New case release.
- Operation launch.
- Sticker capsule release.
- Patch/update day.
- Steam sale/wallet liquidity event.

Feature group G: scarcity and item attributes

- Drop status.
- Case/collection age.
- Float percentile if asset-level data exists.
- StatTrak/souvenir flags.
- Sticker overpay proxy.

Rule for every feature:

Every feature must be computable using only data available before the prediction timestamp.

## 8. Labeling Plan

Use a two-step event-labeling approach.

### 8.1 CUSUM Event Sampling

Instead of labeling every row, trigger events when price movement exceeds a volatility-adjusted threshold.

Initial practical settings:

- Daily bars.
- Volatility lookback: 30 days.
- CUSUM threshold: 0.5 to 1.0 times rolling volatility.
- Minimum gap between events: 1 day.

Tune later during walk-forward validation.

### 8.2 Net-Return Triple Barrier Labels

For every event, simulate an entry and future exit using after-cost prices.

Entry estimate:

```text
entry_cost = executable_ask + buy_fee + expected_slippage
```

Exit estimate:

```text
exit_value = executable_bid - sell_fee - expected_slippage
```

Net return:

```text
net_return = (exit_value - entry_cost) / entry_cost
```

Initial barriers:

- Upper barrier: expected round-trip friction + required profit margin.
- Lower barrier: volatility-adjusted stop loss.
- Vertical barrier: 7 to 30 days depending on item liquidity.

Initial class logic:

- `Strong Buy`: upper net-return barrier reached first with acceptable liquidity.
- `Good Time`: vertical barrier reached with positive net return and controlled downside.
- `Bad Time`: stop-loss reached first, negative vertical return, or liquidity too poor.

## 9. Modeling Plan

Train models in this order:

1. Naive momentum rule.
2. Mean-reversion rule.
3. Moving-average crossover rule.
4. Logistic regression.
5. Random forest.
6. LightGBM multiclass classifier.

Do not start with the most complex model. The baselines define whether the ML system is actually adding value.

Probability calibration:

- Use validation folds only.
- Test isotonic calibration and Platt scaling.
- Report calibration error and reliability curves.

Optional later architecture:

- Stage 1 opportunity model predicts direction or positive gross return.
- Stage 2 trade filter predicts whether the opportunity survives costs, liquidity, and downside risk.

## 10. Backtesting Plan

The backtest should rank buy candidates, select top candidates, simulate entries/exits, and apply realistic costs.

Include:

- venue-specific fees
- bid-ask spread
- slippage
- maximum position size
- minimum liquidity threshold
- maximum portfolio concentration
- sale delay
- failed execution probability if data allows

Core outputs:

- total after-fee return
- average trade return
- win rate
- maximum drawdown
- Sharpe ratio
- Sortino ratio
- average holding period
- turnover
- precision@top_k
- return by category
- return by venue

Primary decision metrics:

- `precision@top_k`
- after-fee net return
- maximum drawdown
- calibration quality

Accuracy is secondary because the project only needs to rank a small set of attractive opportunities reliably.

## 11. Validation Plan

Use walk-forward validation only.

Example:

- Train months 1 to 6, test month 7.
- Train months 1 to 7, test month 8.
- Train months 1 to 8, test month 9.

Add purging and embargo when using overlapping Triple Barrier labels:

- Purge training rows whose label windows overlap the test window.
- Add an embargo period after each test window.

Leakage checks:

- No random train/test split.
- No future category index constituents.
- No full-sample scaler/PCA/feature selection.
- No labels based on prices unavailable at prediction time.
- No same-day close feature if the prediction is meant to happen before close.

## 12. Ablation Plan

Run experiments in this sequence:

| Experiment | Feature Set |
| --- | --- |
| A | Price momentum only |
| B | Price + volume |
| C | Price + volume + volatility |
| D | Add category-relative indices |
| E | Add listing/order-book features |
| F | Add cross-market spread features |
| G | Add scarcity and float features |
| H | Add event-calendar features |
| I | Full model |

For each experiment, report:

- `precision@top_k`
- after-fee net return
- max drawdown
- average holding period
- number of trades
- calibration error

## 13. Dashboard Plan

Build the dashboard after the backtest is reliable.

First dashboard views:

- Top buy candidates.
- Signal probabilities.
- Expected net return.
- Liquidity score.
- Confidence score.
- Reason codes.
- Feature contribution summary.
- Historical performance.
- Filters by weapon, rarity, collection, venue, and liquidity.

Reason code examples:

- Price below collection index by large z-score.
- Relative volume increased sharply.
- Listing depth improved while spread narrowed.
- Category index rising faster than item price.
- Signal rejected because liquidity is insufficient.

## 14. First Two-Week Execution Plan

### Day 1: Project Setup

- Create repository structure.
- Add `pyproject.toml`, `.gitignore`, `.env.example`, and README.
- Move API keys to environment variables.
- Copy useful probe scripts from the sibling project into `src/cs_market_model/collectors/` after removing secrets.
- Add minimal config files for venues, fees, and MVP universe.

Deliverable:

- Clean runnable project skeleton.

### Day 2: Data Audit

- Parse the existing sample K-line JSON.
- Document column meanings.
- Convert one item into normalized daily bars.
- Check date range, missing values, duplicate timestamps, volume fields, and currency.

Deliverable:

- `reports/tables/data_audit_mvp.csv`
- One clean parquet file for the sample item.

### Days 3-4: Data Collection MVP

- Build a collector for the primary price history source.
- Collect 20 to 50 liquid items first.
- Save raw responses exactly as returned.
- Normalize into daily bars.
- Add ingestion logs and request failure tracking.

Deliverable:

- Raw and clean data for the first item universe.

### Days 5-6: Feature Factory V1

- Implement momentum features.
- Implement volatility features.
- Implement volume/liquidity features.
- Add staleness indicators.
- Add feature tests to verify no future rows are used.

Deliverable:

- Feature table for all collected items.

### Day 7: Labeling V1

- Implement CUSUM event sampling.
- Implement simplified net-return Triple Barrier labels.
- Use conservative fixed fee/slippage assumptions from config.
- Manually inspect labeled examples.

Deliverable:

- Label table with `Strong Buy`, `Good Time`, and `Bad Time`.

Current implementation note:

- Day 7 writes `data/labels/labels_day7_v1.parquet`.
- Complete labels are separated from recent incomplete horizons with `is_label_complete`.
- Manual audit outputs are written to `reports/tables/day7_label_audit.csv` and `reports/tables/day7_label_examples.csv`.

### Days 8-9: Baseline Models

- Implement momentum and mean-reversion rules.
- Implement logistic regression.
- Implement random forest.
- Use walk-forward split.

Deliverable:

- Baseline metrics report.

Current implementation note:

- Days 8-9 write split-level metrics to `reports/tables/day8_9_baseline_metrics.csv`.
- Summary metrics are written to `reports/tables/day8_9_baseline_summary.csv`.
- Out-of-sample baseline predictions are written to `data/processed/day8_9_baseline_predictions.parquet`.
- Current baselines use complete Day 7 labels only and monthly purged walk-forward splits with embargo.

### Day 9.5: Baseline Improvement Gate

This step was added after the first Days 8-9 results showed weak top-k net return.
The purpose is to adjust the plan based on observed model behavior before moving to
LightGBM.

- Add cross-sectional rank features for momentum, volatility, drawdown, and relative-value signals.
- Train binary `Strong Buy` vs. `Not Strong Buy` baselines in addition to multiclass baselines.
- Add a deterministic rank-blend benchmark using the binary and multiclass random forest scores.
- Run fee/horizon label sensitivity checks before trusting one target definition.
- Report daily top-k and same-item cooldown-adjusted metrics to check for inflated month-level top-k scores.

Deliverable:

- Improved baseline metrics and label sensitivity report.

Current implementation note:

- Feature Factory V1 now includes 85 feature columns after adding rank and market-regime features.
- Baseline training now evaluates 7 models, including `binary_random_forest` and `forest_rank_blend`.
- `forest_rank_blend` is the current corrected-cost ranking benchmark to beat, with the best mean `precision@top_k` and positive mean top-k net return.
- The benchmark has been checked against stricter daily top-k and 14-day same-item cooldown metrics; both remain materially above the Strong Buy base rate.
- Label sensitivity is written to `reports/tables/day9_5_label_sensitivity.csv`.

### Days 10-11: LightGBM and Calibration

- Train LightGBM multiclass model.
- Calibrate probabilities.
- Compare against baselines.
- Generate feature importance.

Deliverable:

- Model comparison table.

Current implementation note:

- Days 10-11 write LightGBM predictions to `data/processed/day10_11_lightgbm_predictions.parquet`.
- Split-level LightGBM metrics are written to `reports/tables/day10_11_lightgbm_metrics.csv`.
- Baseline-vs-LightGBM comparison is written to `reports/tables/day10_11_model_comparison.csv`.
- Feature importance is written to `reports/tables/day10_11_feature_importance.csv`.
- Calibration diagnostics are written to `reports/tables/day10_11_calibration.csv`.
- LightGBM is competitive with the forest benchmark; `lightgbm_rank_blend` ties top-k precision and improves mean top-k net return, but Platt calibration does not improve test calibration enough to become the lead model.

### Days 12-13: Backtest V1

- Convert model probabilities into ranked signals.
- Simulate top-k trades with fees, slippage, and holding period.
- Report after-fee returns and drawdowns.

Deliverable:

- First execution-realistic backtest report.

Current implementation note:

- Backtest V1 reads Day 8-9 and Day 10-11 prediction parquet artifacts.
- It compares `forest_rank_blend`, `lightgbm_rank_blend`, and `momentum_rule_14d` by default when those models are available.
- Trade simulation applies daily top-k selection, same-item cooldown, cash-constrained max position sizing, optional category exposure limits, and expected failed-execution dilution.
- Outputs are written to `reports/backtests/day12_13_trade_ledger.csv`, `reports/backtests/day12_13_daily_equity.csv`, and `reports/tables/day12_13_backtest_summary.csv`.
- The real local report requires regenerating the ignored prediction parquet files after clone or pull.

### Day 14: Review and Next Scope Decision

- Review data quality problems.
- Review false Strong Buy predictions.
- Decide whether to expand data, features, or labeling.
- Write a concise project progress report.

Deliverable:

- MVP review memo with next steps.

Current implementation note:

- Day 14 outlier audit writes `reports/tables/day14_outlier_audit.csv` and `reports/tables/day14_outlier_model_impact.csv`.
- The first audit found 8 selected events with net returns above 100%.
- The positive `lightgbm_rank_blend` and `forest_rank_blend` Backtest V1 totals are more than fully explained by these outlier trades; excluding audited outlier PnL makes both negative.
- After adding local price bars, price-bar context confirmed that the extreme returns match normalized entry/exit closes, including a 2025-10-22 multi-item spike cluster.
- The 2025-10-22 spike cluster is a known CS2 economy event from the Covert-to-knife/glove Trade Up Contract update, not a generic bad-tick artifact.
- The 2026-05-21 souvenir Trade Up Contract update is recorded for future souvenir-universe work, but the current MVP universe excludes souvenir items.
- Robustness sensitivity is written to `reports/tables/day14_backtest_robustness.csv`; normal-regime metrics excluding known event windows make all compared models negative.
- Regime-level selected-trade metrics are written to `reports/tables/day14_backtest_regime_summary.csv`. In the current rebuild, `lightgbm_rank_blend` has +1.0350 PnL on 10 known-event selected trades and -0.7951 PnL on 889 normal-regime selected trades.
- A normal-alpha feature rebuild added relative-value z-scores, price-jump/liquidity-quality features, and post-event aftershock flags, increasing Day 5 from 101 to 145 feature columns.
- After this rebuild, normal-regime PnL improved modestly but remains negative: `lightgbm_rank_blend` is -0.7266 on 894 normal trades and `forest_rank_blend` is -0.8979 on 893 normal trades.
- A CS2 MVP equal-weight market index was then added to test whether normal-regime losses are mainly caused by broad market drawdowns. This increased Day 5 to 177 feature columns.
- The index feature set includes broad market returns, volatility, drawdown, moving-average distance, trend slope, bull/neutral/bear regime flags, item excess returns, beta, correlation, and residual volatility.
- The rebuilt LightGBM model uses the new index features, especially short-term item excess return and CS2 index volatility/return features.
- Index-regime reporting is written to `reports/tables/day14_backtest_index_regime_summary.csv`.
- The latest index-regime split shows normal-regime PnL is still negative in bear, bull, and neutral index regimes. That means broad market weakness is useful context but does not fully explain the current loss; forced trading and label/execution quality remain the core bottlenecks.
- Day 14 review memo is written to `reports/backtests/day14_mvp_review.md`.
- Next decision should be to stop forced daily top-k trading by adding a normal-regime no-trade/rejection layer and threshold diagnostics.

### Day 15: Rejection-Aware Normal Model

This step was added after the event and CS2 market-index reviews showed that
normal-regime trades remain negative even after controlling for known events and
broad index regimes.

- Keep normal-regime model selection separate from known-event results.
- Add an abstention policy so the model can select no trades on weak days.
- Replace daily forced top-k as the main metric with thresholded accepted-trade
  performance.
- Add rejection curves by score, calibrated probability, liquidity quality,
  label quality, and CS2 index regime.
- Report accepted-trade count, rejection rate, average net return, total PnL,
  max drawdown, profit factor, and liquidity-adjusted capacity.
- Add label-quality gates for stale prices, low volume, isolated one-day jumps,
  event overlap, and non-executable returns.

Deliverable:

- `reports/tables/day15_rejection_curve.csv`
- `reports/tables/day15_threshold_policy_summary.csv`
- normal-only accepted-trade report with event-regime results excluded from
  threshold tuning.

### Day 16: Code Hygiene and Utility Extraction

This step was added after a project-wide code review identified duplicated
functions and missing infrastructure patterns across the codebase.

- Create `src/cs_market_model/utils.py` to hold shared utility functions.
- Extract `safe_item_name()` from `collectors/csfloat.py` and
  `collectors/steamdt.py` into `utils.py`.
- Extract `_display_path()` from the 6 files where it is duplicated
  (`build_features.py`, `build_labels.py`, `train.py`, `lightgbm_train.py`,
  `portfolio.py`, `audit_outliers.py`) into `config.py` or `utils.py`.
- Extract `_rank_blend_for_split()` from `models/train.py` and
  `models/lightgbm_train.py` into a shared model utilities module.
- Extract `_fee_model_from_config()` from `labeling/build_labels.py` and
  `labeling/sensitivity.py` into a shared labeling utility.
- Fix cross-module private imports: refactor `_attach_split_metadata` and
  `_base_prediction_frame` in `train.py` to public functions since they are
  imported by `lightgbm_train.py`.
- Replace `print()` output with Python `logging` module across all pipeline
  scripts. Use `logging.getLogger(__name__)` in each module.
- Add `__all__` exports to all subpackage `__init__.py` files.
- Update `hashlib.sha1` in `normalization/prices.py` to `hashlib.sha256`.

Deliverable:

- `src/cs_market_model/utils.py` with shared utilities.
- Zero duplicated utility functions across the codebase.
- All pipeline scripts use `logging` instead of `print()`.
- All existing tests pass after refactor.

### Day 17: EDA Notebooks and Research Presentation

This step was added because the `notebooks/` directory is empty despite being
part of the original plan. For a research project, exploratory analysis
notebooks are essential for interpretability, presentation, and debugging.

- Create `notebooks/01_data_audit.ipynb`: visualize price bar coverage, missing
  data heatmap, volume distributions, staleness patterns, per-item time series.
- Create `notebooks/02_feature_distributions.ipynb`: feature histograms,
  correlation matrix, feature importance comparison across models, rank feature
  distributions, cross-sectional feature spreads by category.
- Create `notebooks/03_model_error_analysis.ipynb`: false Strong Buy
  investigation, confusion matrices per walk-forward split, precision@top-k
  over time, regime-conditional error rates, per-item model bias analysis.
- Create `notebooks/04_regime_visualization.ipynb`: CS2 market index time
  series, bull/bear/neutral regime overlay, event-regime trade highlighting,
  equity curve with regime shading, normal vs event PnL waterfall chart.
- Use generated report CSVs and parquet artifacts as data sources.

Deliverable:

- 4 runnable Jupyter notebooks with visualizations.
- Key charts saved to `reports/figures/` for use in presentations or papers.

### Day 18: CSFloat Feature Integration and Ablation

This step implements the Phase 2 feature expansion planned in Section 6.1.
The CSFloat collector already works; this step connects listing data into
the feature pipeline.

- Implement `normalization/order_books.py`: parse CSFloat listing snapshots
  into normalized listing depth, best ask, float distribution, and sticker
  features.
- Build a CSFloat snapshot collector schedule for the 48 MVP items.
- Add CSFloat-derived features to `build_features.py` with timestamp-safe
  joins (only use listing snapshots available before prediction time).
- New features: best ask distance from close, listing count, sell-side depth
  within 1%/3%/5%, float percentile, sticker count proxy, reference price
  gap.
- Run ablation: compare `SteamDT only` (current 177 features) against
  `SteamDT + CSFloat` to measure whether CSFloat features improve top-k
  precision and net return.
- Add tests for CSFloat normalization and point-in-time safety.

Deliverable:

- `normalization/order_books.py` fully implemented.
- Ablation table in `reports/tables/day18_csfloat_ablation.csv`.
- Updated feature audit with CSFloat feature coverage.

### Day 19: Universe Expansion

This step expands the tradable universe beyond the initial 48 Field-Tested
gun skins after the pipeline has been validated end to end.

- Select 100 to 150 items from the 408-item candidate universe
  (`universe_gun_skins_candidate.yaml`) based on liquidity, data coverage,
  and trading volume thresholds.
- Add other wear conditions beyond Field-Tested (Minimal Wear, Well-Worn)
  for high-liquidity items only.
- Re-run the full pipeline (collection, features, labels, models, backtest)
  on the expanded universe.
- Compare cross-sectional signal quality: does a larger universe improve
  category-relative and rank feature informativeness?
- Check whether rejection thresholds from Day 15 transfer to the expanded
  universe or need re-tuning.
- Add per-item and per-collection attribution for normal-regime losses.

Deliverable:

- Updated `configs/universe_mvp_v2.yaml` with expanded item list.
- Comparison report: 48-item vs expanded-universe backtest metrics.
- Per-item and per-collection loss attribution table.

### Day 20: Dashboard MVP

This step builds the first interactive dashboard after the backtest and
rejection system are validated.

- Implement `dashboard/app.py` using Streamlit.
- Views: top buy candidates, signal probabilities, expected net return,
  liquidity score, confidence score, reason codes, feature contribution
  summary.
- Add filters by weapon, rarity, collection, venue, and liquidity.
- Add historical performance charts: equity curve with regime overlay,
  drawdown, rolling precision.
- Add model comparison view: forest vs LightGBM vs baselines.
- Add rejection policy visualization: show which trades were accepted/
  rejected and why.
- Load data from local parquet artifacts and report CSVs.

Deliverable:

- Runnable Streamlit dashboard (`streamlit run src/cs_market_model/dashboard/app.py`).
- Screenshots saved to `reports/figures/`.

## 15. Key Risks

Data quality:

- APIs may be incomplete, delayed, rate-limited, or inconsistent.
- Some venues may not provide true historical executable prices.

Leakage:

- The biggest technical risk is accidentally using future information.
- Walk-forward validation and point-in-time feature generation are mandatory.

Execution realism:

- A profitable chart signal can disappear after spread, fees, slippage, and sale delays.
- Backtests must reject signals that cannot realistically be executed.

Item heterogeneity:

- Rare float, rare pattern, and sticker-heavy items may not follow median market prices.
- Start with liquid commodity-like items, then expand.

Overfitting:

- The feature space can become large quickly.
- Keep ablation testing and simple baselines in the core workflow.

Normal-regime alpha:

- Current models are unprofitable in normal market conditions after fees.
- All observed profitability comes from ~10 trades during the October 2025 CS2 Trade Up economy event.
- The rejection/abstention layer (Day 15) and CSFloat features (Day 18) are the primary mitigations.
- Universe expansion (Day 19) may improve cross-sectional signal quality.

Code maintenance:

- Several utility functions are duplicated across 2-6 files.
- All output uses `print()` instead of the `logging` module.
- Day 16 code hygiene pass addresses both issues.

## 16. Immediate Next Steps

Priority 1 — Research critical (Days 15-16):

1. Build the Day 15 normal-regime rejection curve and no-trade thresholds.
2. Add label-quality gates and hard reject flags to the label table.
3. Report normal-only threshold performance separately from known-event
   performance.
4. Run Day 16 code hygiene: extract duplicated utilities, replace `print()`
   with `logging`, fix cross-module private imports.

Priority 2 — Research expansion (Days 17-18):

5. Create EDA notebooks for data audit, feature distributions, model error
   analysis, and regime visualization.
6. Implement CSFloat feature integration and run the `SteamDT only` vs
   `SteamDT + CSFloat` ablation.
7. Add stronger liquidity/spread proxies from CSFloat listing data before
   expanding the tradable universe.

Priority 3 — Scale and present (Days 19-20):

8. Expand the tradable universe from 48 to 100-150 items with per-item and
   per-collection loss attribution.
9. Build the Streamlit dashboard MVP with signal viewer, regime overlay,
   and rejection policy visualization.
10. Add trade-up EV features as explanatory features, while keeping the
    current tradable MVP narrow.

Standing rules:

- Treat the October 2025 and May 2026 Trade Up changes as event-engine inputs,
  not normal-alpha training wins.
- Do not expand the universe until the rejection layer is validated on the
  current 48-item set.
- Every new feature group must include an ablation row showing marginal impact.

## 17. Definition of Done for the First MVP

The first MVP is done when the project can run one reproducible command that:

1. Loads raw data.
2. Builds clean daily bars.
3. Builds point-in-time-safe features.
4. Generates net-return-aware labels.
5. Trains baseline and LightGBM models.
6. Runs walk-forward validation.
7. Runs an execution-realistic top-k backtest.
8. Applies rejection thresholds to filter low-conviction trades.
9. Saves a report with metrics, charts, and top signal examples.

The output does not need to be profitable in the first version. It needs to be honest, reproducible, leakage-safe, and detailed enough to show what must improve next.
