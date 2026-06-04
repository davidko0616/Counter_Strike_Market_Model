# Counter-Strike Market Model

Research MVP for Counter-Strike skin buy-signal modeling after fees, liquidity, execution costs, market regime, and rejection gates.

## Current Status

The project is complete enough for submission as an honest research MVP.

Final recommendation:

> Paper trade only. Do not use live capital yet.

The pipeline works end to end, and the corrected `$5+` gun-skin policy remains positive in several tests. However, stricter paper-trade gates still fail period-stability criteria because gains are too concentrated. The correct conclusion is that the system is useful for research and paper trading, not live trading.

See the final report:

- `docs/final_submission_report.md`

## Scope

Current tradable universe:

- gun skins only
- no knives
- no gloves
- no cases
- no sticker price data

Data sources:

- SteamDT: historical price bars and liquidity proxies
- CSFloat: listing snapshots and future microstructure features

CSFloat point-in-time coverage is unavailable in the current local artifact set.
Day 18 and Day 19 run successfully and report this explicitly, so CSFloat is
not used as a hard trading gate in the final recommendation.

## What Is Implemented

- SteamDT collectors and batch ingestion
- normalized price bars
- item metadata and gun-skin universe configs
- point-in-time feature factory
- net-return-aware triple-barrier labels
- baseline models
- LightGBM model
- walk-forward validation
- execution-realistic backtests
- rejection policies
- low-price execution-cost stress tests
- capacity sizing and open-position limits
- wear-variant validity audit
- Day 25 stricter `$5+` paper-trade gate analysis
- Streamlit dashboard MVP
- secret scanner

Generated data/report artifacts are intentionally ignored by Git.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill `.env` locally:

```text
STEAMDT_API_KEY=...
CSFLOAT_API_KEY=...
```

Never commit `.env`.

Run the secret scanner before pushing:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.security.scan_secrets
```

## Validation

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run repo-wide lint:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Repo-wide Ruff passes in the final local validation.

## Final Research Commands

Day 22 low-price policy comparison:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.research.day22_low_price_policy
```

Day 23 policy attribution:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.research.day23_policy_attribution
```

Day 24 `$5+` capacity sensitivity:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.research.day24_capacity_sensitivity
```

Day 25 stricter paper-trade rejection gates:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.research.day25_paper_trade_rejection
```

Day 18/19 CSFloat coverage status:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.research.day19_csfloat_coverage
.\.venv\Scripts\python.exe -m cs_market_model.research.day18_csfloat_ablation
```

Dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run src/cs_market_model/dashboard/app.py
```

## Final Headline Results

Corrected LightGBM Day 22 results:

| Policy | Accepted Trades | PnL | Profit Factor | Win Rate |
| --- | ---: | ---: | ---: | ---: |
| raw v2 costed | 704 | 0.6950 | 2.4118 | 66.76% |
| `$1+` costed | 416 | 0.1789 | 1.6882 | 71.15% |
| `$5+` costed | 195 | 0.0413 | 1.3241 | 77.44% |

Day 25 stricter `$5+` result:

| Scenario | Accepted | PnL | Profit Factor | Positive Period Share | Stress Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| current Day 24 `$5+` gates | 195 | 0.0413 | 1.3241 | 90.91% | Yes |
| stricter `$5+` gates | 94 | 0.1022 | 1.8276 | 36.36% | No |
| stricter `$5+` gates + strict open 10 | 93 | 0.0349 | 1.8473 | 36.36% | No |

## Submission Files

- `docs/final_submission_report.md`
- `SUBMISSION_CHECKLIST.md`
- `STARTUP_PLAN.md`
- `Reviews and implementation plans/todo_before_proceeding.md`
- `src/cs_market_model/dashboard/app.py`
- `src/cs_market_model/research/day25_paper_trade_rejection.py`
- `configs/backtest.yaml`

## Bottom Line

The project demonstrates a working, validated research pipeline and reaches a defensible recommendation: use the conservative `$5+` policy only for paper trading until CSFloat point-in-time coverage and live degradation monitoring are available.
