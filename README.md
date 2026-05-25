# Counter-Strike Market Model

Research-grade MVP for net-return-aware Counter-Strike skin buy-signal
prediction.

The project ranks item-date opportunities by expected after-cost value rather
than raw price direction. The initial scope is a small liquid universe, SteamDT
historical bars, and later CSFloat listing snapshots for point-in-time
microstructure features.

## Current Status

Completed through Day 16 implementation:

- The MVP universe is gun skins only: no knives, gloves, cases, stickers, or
  sticker price data in the tradable set.
- The first-pull universe has 48 Field-Tested skins across rifles, snipers,
  pistols, and SMGs.
- Feature Factory V1 builds 177 feature columns: momentum, volatility,
  liquidity/staleness, relative index features, cross-sectional ranks, CS2
  market-index features, and known-event features.
- Labeling V1 builds CUSUM-sampled events and net-return Triple Barrier labels.
- Baseline and LightGBM models use purged walk-forward splits.
- Backtest V1 converts ranked prediction outputs into cash-constrained top-k
  trade simulations with cooldowns, position sizing, trade ledgers, equity
  curves, drawdown, turnover, and selected-trade precision.
- Day 14 separates event-regime trades from normal-regime performance. Normal
  trading is negative across all models and all CS2 index regimes.
- Day 15 adds post-hoc rejection curves and named no-trade policies.
- Day 16 integrates rejection into the portfolio simulator before entries and
  adds walk-forward score-threshold validation.

Generated data and report outputs are intentionally ignored by git.

## Key Local Artifacts

- `data/processed/price_bars_day3_first_pull.parquet`
- `data/processed/item_metadata_day3_first_pull.parquet`
- `data/features/features_day5_v1.parquet`
- `data/labels/labels_day7_v1.parquet`
- `reports/tables/day12_13_backtest_summary.csv`
- `reports/backtests/day12_13_trade_ledger.csv`
- `reports/tables/day14_backtest_regime_summary.csv`
- `reports/tables/day15_rejection_curve.csv`
- `reports/tables/day15_threshold_policy_summary.csv`
- `reports/tables/day15_normal_accepted_trade_summary.csv`
- `reports/backtests/day16_conservative_trade_ledger.csv`
- `reports/tables/day16_conservative_backtest_summary.csv`
- `reports/tables/day16_walk_forward_threshold_selection.csv`
- `reports/tables/day16_walk_forward_threshold_summary.csv`
- `reports/backtests/day16_walk_forward_accepted_trades.csv`

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

## Main Commands

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

Day 15 rejection analysis:

```powershell
python -m cs_market_model.backtesting.rejection
```

Day 16 integrated conservative-policy backtest:

```powershell
python -m cs_market_model.backtesting.portfolio --rejection-policy conservative
```

Day 16 walk-forward rejection validation:

```powershell
python -m cs_market_model.backtesting.walk_forward_rejection
```

CSFloat listings probe:

```powershell
python -m cs_market_model.collectors.csfloat "AK-47 | Redline (Field-Tested)"
```

## Next Steps

Priority 1: Finish validating the Day 16 rejection system on regenerated
prediction artifacts and make threshold selection stricter if needed.

Priority 2: Add CSFloat listing features and run a `SteamDT only` versus
`SteamDT + CSFloat` ablation.

Priority 3: Expand the tradable universe only after the rejection-aware normal
model remains positive under walk-forward validation.
