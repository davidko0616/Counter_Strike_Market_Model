# Reproduce The Paper-Trade Research System

Runtime: Python 3.12.13 on Windows PowerShell.

This project is paper-trade-only. Do not put API keys, `.env`, or generated private data into Git.

## 1. Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e .
```

If the lockfile install is too strict for a new platform, install the project dependencies from
`pyproject.toml`, then rerun tests before trusting results.

## 2. Data Collection

SteamDT and CSFloat collection require local credentials/configuration. Keep secrets in `.env` only.

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.collectors.csfloat_batch --config universe_mvp_v2.yaml
.\.venv\Scripts\python.exe -m cs_market_model.features.csfloat_listings
```

If CSFloat coverage is missing or too small, the system stays SteamDT-only for model validation and
uses CSFloat only as a liquidity diagnostic.

## 3. One-Command Pipeline

Dry run:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.pipeline.run_all --dry-run
```

Local non-collector run:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.pipeline.run_all
```

Collector-inclusive run:

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.pipeline.run_all --include-collectors
```

Expected runtime depends on data volume and API limits. The non-collector run should be minutes once
feature and backtest inputs already exist. Collector-inclusive runs can take longer due to API pacing.

## 4. Key Commands

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.research.day19_csfloat_coverage
.\.venv\Scripts\python.exe -m cs_market_model.research.day18_csfloat_ablation
.\.venv\Scripts\python.exe -m cs_market_model.models.explain
.\.venv\Scripts\python.exe -m cs_market_model.research.paper_trade_ab_test
.\.venv\Scripts\python.exe -m cs_market_model.research.model_health_monitor
.\.venv\Scripts\python.exe -m cs_market_model.pipeline.artifact_manifest
.\.venv\Scripts\streamlit.exe run src/cs_market_model/dashboard/app.py
```

## 5. Expected Outputs

- `reports/tables/day18_csfloat_ablation_setup.csv`
- `reports/tables/day19_csfloat_coverage.csv`
- `reports/tables/signal_explanations.csv`
- `reports/tables/paper_trade_ab_test_summary.csv`
- `reports/tables/model_health_monitor.csv`
- `reports/tables/model_health_status.json`
- `reports/tables/artifact_manifest.csv`
- `reports/backtests/paper_trade_ab_test_report.md`
- `reports/backtests/model_health_report.md`

Most generated `data/` and `reports/` files are intentionally gitignored. Use the artifact manifest
to record SHA256 hashes and row counts for local verification.

## 6. Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m cs_market_model.security.scan_secrets
```

The final recommendation is paper trade only unless future paper-trade evidence is reviewed outside
this codebase. This repository does not include live execution or live-capital approval logic.
