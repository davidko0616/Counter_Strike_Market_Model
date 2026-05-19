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

MVP venues and APIs:

- Use both SteamDT and CSFloat, but assign them different responsibilities.
- SteamDT is the primary historical price source.
- CSFloat is the secondary listing, float, sticker, and microstructure source.
- Build the first working pipeline with SteamDT historical bars, then add CSFloat snapshots as additional features.
- Treat cross-market spread features as Phase 2, not a blocker for the first model.

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
    venues.yaml
    fees.yaml
    backtest.yaml
  data/
    raw/
    interim/
    processed/
    features/
    labels/
  notebooks/
    01_data_audit.ipynb
    02_feature_sanity_checks.ipynb
    03_model_error_analysis.ipynb
  reports/
    figures/
    tables/
    backtests/
  src/
    cs_market_model/
      __init__.py
      config.py
      collectors/
        steamdt.py
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
        events.py
        build_features.py
      labeling/
        cusum.py
        triple_barrier.py
        build_labels.py
      models/
        baselines.py
        train.py
        calibrate.py
        explain.py
      backtesting/
        execution.py
        portfolio.py
        walk_forward.py
        metrics.py
      dashboard/
        app.py
  tests/
    test_labeling.py
    test_features_no_leakage.py
    test_backtest_costs.py
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

- Equal-weight category index.
- Median category index.
- Liquidity-filtered category index.
- Item return minus category return.
- Item price/category index ratio.

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

### Days 8-9: Baseline Models

- Implement momentum and mean-reversion rules.
- Implement logistic regression.
- Implement random forest.
- Use walk-forward split.

Deliverable:

- Baseline metrics report.

### Days 10-11: LightGBM and Calibration

- Train LightGBM multiclass model.
- Calibrate probabilities.
- Compare against baselines.
- Generate feature importance.

Deliverable:

- Model comparison table.

### Days 12-13: Backtest V1

- Convert model probabilities into ranked signals.
- Simulate top-k trades with fees, slippage, and holding period.
- Report after-fee returns and drawdowns.

Deliverable:

- First execution-realistic backtest report.

### Day 14: Review and Next Scope Decision

- Review data quality problems.
- Review false Strong Buy predictions.
- Decide whether to expand data, features, or labeling.
- Write a concise project progress report.

Deliverable:

- MVP review memo with next steps.

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

## 16. Immediate Next Steps

1. Initialize the repository skeleton.
2. Create `.env.example` and remove hardcoded secrets from copied scripts.
3. Build the raw-to-clean parser for the existing K-line sample.
4. Define the first 20 to 50 MVP items.
5. Generate the first normalized `price_bars` table.
6. Implement momentum, volatility, and liquidity features.
7. Implement simplified CUSUM and Triple Barrier labels.
8. Train baselines before LightGBM.
9. Run walk-forward evaluation.
10. Produce the first after-fee backtest report.

## 17. Definition of Done for the First MVP

The first MVP is done when the project can run one reproducible command that:

1. Loads raw data.
2. Builds clean daily bars.
3. Builds point-in-time-safe features.
4. Generates net-return-aware labels.
5. Trains baseline and LightGBM models.
6. Runs walk-forward validation.
7. Runs an execution-realistic top-k backtest.
8. Saves a report with metrics, charts, and top signal examples.

The output does not need to be profitable in the first version. It needs to be honest, reproducible, leakage-safe, and detailed enough to show what must improve next.
