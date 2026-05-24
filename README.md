# Counter-Strike Market Model

Research-grade MVP for net-return-aware Counter-Strike skin buy-signal prediction.

The project ranks item-date opportunities by expected after-cost value rather than raw price direction. The initial scope is a small liquid universe, SteamDT historical bars, and CSFloat listing snapshots for point-in-time microstructure features.

## Current Status

Completed through Days 10-11:

- Project environment, secret handling, and repository skeleton are in place.
- The MVP universe is gun skins only: no knives, gloves, cases, stickers, or sticker price data.
- The first-pull universe has 48 Field-Tested skins across rifles, snipers, pistols, and SMGs.
- SteamDT Day 3 collection produced normalized daily bars and item metadata for all 48 items.
- Feature Factory V1 builds momentum, volatility, liquidity/staleness, and category/weapon-relative features.
- Labeling V1 builds CUSUM-sampled events and net-return Triple Barrier labels.
- Baseline modeling evaluates rule, logistic regression, and random forest models with purged walk-forward splits.
- Day 9.5 adds cross-sectional rank features, binary Strong Buy models, a forest rank-blend, and label sensitivity checks.
- Execution assumptions now use a SteamDT proxy baseline: 2.5% transaction fee, reflecting Buff/Youpin-style markets rather than Steam marketplace fees.
- Feature Factory V1 includes market-regime features so broad market drawdowns and bloated/risk-off periods can be modeled.
- Baseline reports include stricter daily top-k and same-item cooldown-adjusted metrics to reduce over-optimistic ranking claims.
- LightGBM binary, multiclass, calibrated, and rank-blend models are compared against the Day 9.5 forest benchmark.

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

CSFloat listings probe:

```powershell
python -m cs_market_model.collectors.csfloat "AK-47 | Redline (Field-Tested)"
```

Collector commands save append-only raw JSON under `data/raw/` and do not contain hardcoded secrets.

## Next Step

Days 12-13 should run Backtest V1:

- Convert ranked probabilities into executable top-k trade simulations.
- Apply fees, slippage, holding period, and same-item cooldown constraints.
- Compare `forest_rank_blend`, `lightgbm_rank_blend`, and the best simple baseline.
- Report after-fee returns, drawdown, turnover, and trade examples.
