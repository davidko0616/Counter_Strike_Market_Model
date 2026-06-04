# Final Submission Report

## Project Summary

This project builds a research-grade Counter-Strike skin market signal system. It collects and normalizes SteamDT price data, builds point-in-time features, creates net-return-aware labels, trains baseline and LightGBM models, and evaluates trade candidates with walk-forward validation, rejection gates, transaction costs, and capacity limits.

The current MVP is intentionally conservative: gun skins only, no knives, gloves, cases, stickers, or sticker price data as tradable assets.

## Final Recommendation

Do not use live capital yet.

The strongest defensible status is paper-trade / research mode. The corrected `$5+` candidate remains positive under several tests, but the stricter paper-trade gates still fail survival criteria because gains are concentrated in too few periods. This is a good research result, not a live-trading signal.

## What Works

- SteamDT batch collection and normalized daily price bars.
- Gun-skin-only universe configuration.
- Feature Factory V1 with momentum, volatility, liquidity, staleness, market-index, and event-regime features.
- Net-return-aware triple-barrier labels.
- Baseline models and LightGBM training.
- Walk-forward validation with purging/embargo.
- Execution-realistic backtests with fees, price-bucket costs, capacity sizing, and rejection gates.
- Wear-variant validity checks.
- Streamlit dashboard MVP.
- Secret scanning and `.env` exclusion.

## Corrected LightGBM Policy Results

These are the current corrected Day 22 policy comparison results:

| Policy | Accepted Trades | PnL | Profit Factor | Win Rate | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw v2 costed | 704 | 0.6950 | 2.4118 | 66.76% | -0.0417 |
| `$1+` costed | 416 | 0.1789 | 1.6882 | 71.15% | -0.0275 |
| `$5+` costed | 195 | 0.0413 | 1.3241 | 77.44% | -0.0392 |

The raw and `$1+` policies are useful for research, but the safer candidate is `$5+` because it avoids the weakest low-price execution assumptions.

## Attribution Summary

Current Day 23 attribution shows meaningful concentration risk:

| Policy | Item Count | Positive Items | Top-3 Item PnL Share | Top-2 Period PnL Share | Median Entry Close |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw v2 costed | 143 | 96 | 35.53% | 50.85% | 1.18 |
| `$1+` costed | 88 | 53 | 37.70% | 56.18% | 4.81 |
| `$5+` costed | 52 | 30 | 65.21% | 60.06% | 21.77 |

This is why the project should not claim a live-trading recommendation.

## Stricter Paper-Trade Gate Result

Day 25 added stricter `$5+` rejection gates:

- minimum entry price: `$5`
- liquidity quality at least `0.7`
- staleness no more than `2` days
- price jump share no more than `0.05`
- row coverage at least `0.8`
- reject bear-market rows without item alpha
- require non-negative recent item excess return
- reject unstable 30-day absolute return paths over `0.35`

| Scenario | Accepted | PnL | Profit Factor | Max Drawdown | Positive Period Share | Survives Stress |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current Day 24 `$5+` gates | 195 | 0.0413 | 1.3241 | -0.0379 | 90.91% | Yes |
| stricter `$5+` gates | 94 | 0.1022 | 1.8276 | -0.0260 | 36.36% | No |
| stricter `$5+` gates + strict open 10 | 93 | 0.0349 | 1.8473 | -0.0088 | 36.36% | No |

The stricter gates improved isolated PnL but failed period-stability rules. Final status remains paper-trade only.

## Known Limitations

- CSFloat point-in-time coverage is unavailable in the current local artifact set.
  Day 18 and Day 19 now run successfully and write coverage reports, but they
  correctly report `0` CSFloat-covered feature rows because no local CSFloat
  snapshot feature rows are available for the 150-item universe. This is an
  external data availability limitation, not a code failure.
- Generated data and report artifacts are intentionally ignored by Git, so a clean clone must rerun the pipeline or receive the local artifacts separately.
- The dashboard is an MVP focused on visibility, not a finished production trading app.
- Repo-wide Ruff passes.

## Validation

Local validation completed:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The project `.venv` now includes `streamlit`, so dashboard tests can import the dashboard app.

Targeted validation completed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rejection.py tests/test_day25_paper_trade_rejection.py
.\.venv\Scripts\python.exe -m ruff check src/cs_market_model/backtesting/rejection_policy.py src/cs_market_model/backtesting/rejection.py src/cs_market_model/backtesting/portfolio.py src/cs_market_model/research/day25_paper_trade_rejection.py tests/test_rejection.py tests/test_day25_paper_trade_rejection.py
```

Repo-wide lint validation completed:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

## Submission Position

This project is submit-ready as an honest research MVP. The conclusion is not "we found a guaranteed trading bot"; the conclusion is:

> The pipeline works, the backtest is corrected for major execution and validation issues, the conservative `$5+` policy is suitable for paper trading, and live trading should wait for better CSFloat point-in-time coverage and live/paper monitoring.
