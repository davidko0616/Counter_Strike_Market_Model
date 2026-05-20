# Counter-Strike Market Model

Research-grade MVP for net-return-aware Counter-Strike skin buy-signal prediction.

The project ranks item-date opportunities by expected after-cost value rather than raw price direction. The initial scope is a small liquid universe, SteamDT historical bars, and CSFloat listing snapshots for point-in-time microstructure features.

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

CSFloat listings probe:

```powershell
python -m cs_market_model.collectors.csfloat "AK-47 | Redline (Field-Tested)"
```

Both commands save append-only raw JSON under `data/raw/` and do not contain hardcoded secrets.

## Next Step

Day 2 should parse the existing sample K-line JSON, document the raw columns, and write one normalized daily bars parquet file plus `reports/tables/data_audit_mvp.csv`.
