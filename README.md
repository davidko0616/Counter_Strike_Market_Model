# Counter-Strike Market Model

Research-grade MVP for net-return-aware Counter-Strike skin buy-signal prediction.

The project ranks item-date opportunities by expected after-cost value rather than raw price direction. The initial scope is a small liquid universe, SteamDT historical bars, and CSFloat listing snapshots for point-in-time microstructure features.

## Current Status

Completed through Day 14 implementation:

- Project environment, secret handling, and repository skeleton are in place.
- The MVP universe is gun skins only: no knives, gloves, cases, stickers, or sticker price data.
- The first-pull universe has 48 Field-Tested skins across rifles, snipers, pistols, and SMGs.
- SteamDT Day 3 collection produced normalized daily bars and item metadata for all 48 items.
- Feature Factory V1 builds 177 feature columns: momentum, volatility, liquidity/staleness, category/weapon-relative, cross-sectional ranks, market-regime, and known-event features.
- Labeling V1 builds CUSUM-sampled events and net-return Triple Barrier labels.
- Baseline modeling evaluates rule, logistic regression, and random forest models with purged walk-forward splits.
- Day 9.5 adds cross-sectional rank features, binary Strong Buy models, a forest rank-blend, and label sensitivity checks.
- Execution assumptions now use a SteamDT proxy baseline: 2.5% transaction fee, reflecting Buff/Youpin-style markets rather than Steam marketplace fees.
- LightGBM binary, multiclass, calibrated, and rank-blend models are compared against the Day 9.5 forest benchmark.
- Backtest V1 converts ranked prediction outputs into cash-constrained top-k trade simulations with same-item cooldown, position sizing, trade ledger, equity curve, drawdown, turnover, and selected-trade precision.
- Day 14 outlier audit separates event-regime trades (Oct 2025 Trade Up update) from normal-regime performance. Normal-regime trading is negative across all models and all index regimes.
- CS2 equal-weight market index added with item excess returns, beta, correlation, and regime classification.

Current local generated artifacts:

- `data/processed/price_bars_day3_first_pull.parquet`
- `data/processed/item_metadata_day3_first_pull.parquet`
- `data/features/features_day5_v1.parquet`
- `data/labels/labels_day7_v1.parquet`
- `reports/tables/day3_steamdt_ingestion_log.csv`
- `reports/tables/day5_feature_audit.csv`
- `reports/tables/day7_label_audit.csv`
- `reports/tables/day7_label_examples.csv`
- `reports/tables/day8_9_baseline_metrics.csv`
- `reports/tables/day8_9_baseline_summary.csv`
- `reports/tables/day9_5_label_sensitivity.csv`
- `reports/tables/day10_11_lightgbm_metrics.csv`
- `reports/tables/day10_11_model_comparison.csv`
- `reports/tables/day10_11_feature_importance.csv`
- `reports/tables/day10_11_calibration.csv`
- `reports/backtests/day12_13_trade_ledger.csv`
- `reports/backtests/day12_13_daily_equity.csv`
- `reports/tables/day12_13_backtest_summary.csv`
- `reports/tables/day14_outlier_audit.csv`
- `reports/tables/day14_outlier_model_impact.csv`
- `reports/tables/day14_backtest_robustness.csv`
- `reports/tables/day14_backtest_regime_summary.csv`
- `reports/backtests/day14_mvp_review.md`

These data and report outputs are intentionally ignored by git.

## Day 1 Skeleton

This repository now includes:

- A Python package under `src/cs_market_model`.
- Sanitized SteamDT and CSFloat collector probes that read API keys from environment variables.
- Minimal config files for venues, fees, backtesting assumptions, and the MVP universe.
- Data, report, notebook, and test directories matching the startup plan.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill in `.env` locally:

```text
STEAMDT_API_KEY=...
CSFLOAT_API_KEY=...
```

Run the secret scanner before committing:

```powershell
python -m cs_market_model.security.scan_secrets
```

## Collector Probes

SteamDT K-line probe:

```powershell
python -m cs_market_model.collectors.steamdt "AK-47 | Redline (Field-Tested)"
```

SteamDT Day 3 batch collection:

```powershell
python -m cs_market_model.collectors.steamdt_batch --config universe_day3_first_pull.yaml
```

Day 5 feature build:

```powershell
python -m cs_market_model.features.build_features
```

Day 7 label build:

```powershell
python -m cs_market_model.labeling.build_labels
```

Days 8-9 baseline training:

```powershell
python -m cs_market_model.models.train
```

Day 9.5 label sensitivity:

```powershell
python -m cs_market_model.labeling.sensitivity
```

Days 10-11 LightGBM comparison:

```powershell
python -m cs_market_model.models.lightgbm_train
```

Days 12-13 Backtest V1:

```powershell
python -m cs_market_model.backtesting.portfolio
```

Day 14 outlier audit:

```powershell
python -m cs_market_model.backtesting.audit_outliers
```

CSFloat listings probe:

```powershell
python -m cs_market_model.collectors.csfloat "AK-47 | Redline (Field-Tested)"
```

Collector commands save append-only raw JSON under `data/raw/` and do not contain hardcoded secrets.

## Next Steps

The project roadmap extends from Day 15 through Day 20. See `STARTUP_PLAN.md` for full details.

**Priority 1 — Research critical (Days 15-16):**

- Day 15: Build a rejection-aware normal-regime model with no-trade thresholds and label-quality gates. Stop forcing daily top-k trades; let the model abstain on weak days.
- Day 16: Code hygiene pass — extract duplicated utility functions, replace `print()` with `logging`, fix cross-module private imports.

**Priority 2 — Research expansion (Days 17-18):**

- Day 17: Create EDA notebooks for data audit, feature distributions, model error analysis, and regime visualization.
- Day 18: Integrate CSFloat listing features into the pipeline and run `SteamDT only` vs `SteamDT + CSFloat` ablation.

**Priority 3 — Scale and present (Days 19-20):**

- Day 19: Expand the tradable universe from 48 to 100-150 items with per-item loss attribution.
- Day 20: Build a Streamlit dashboard MVP with signal viewer, regime overlay, and rejection policy visualization.
