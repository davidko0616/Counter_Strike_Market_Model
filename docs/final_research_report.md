# Counter-Strike Market Model Final Research Report

Date: 2026-06-04

Runtime: Python 3.12.13

Git commit at implementation start: `b61e06d`

Recommendation: paper trade only. Do not use live capital.

## Objective

Build a reproducible research system for Counter-Strike gun-skin buy-signal analysis. The system
prioritizes conservative validation, explainability, operational monitoring, and dashboard review.
It does not include live trading, live execution, or any live-capital eligibility path.

## Data Sources

- SteamDT/YouPin/Buff-derived price history for daily price bars.
- CSFloat listing snapshots for liquidity and listing-depth diagnostics.
- Local item universe configs restricted to gun skins.
- Local generated research artifacts under `data/` and `reports/`.

CSFloat features are point-in-time safe only when listing snapshots can be joined to later price
bars. Until coverage reaches the configured threshold, CSFloat remains diagnostic-only.

## Universe

The production research universe is gun-skins-only. Knives, gloves, cases, stickers, and other
non-gun market items are excluded from the final research universe. Wear variants are expected to be
physically valid against each skin's float range.

## Features

The feature factory includes price momentum, volatility, liquidity, market-index regime, event
features, rank features, and CSFloat listing diagnostics. CSFloat listing-depth normalization now
captures listing count, ask price summaries, depth within 1%, 3%, and 5%, float summaries, and
sticker-count summaries.

## Labels

Labels are built with the existing triple-barrier and walk-forward research workflow. Validation is
time-aware and avoids using future rows for model selection or feature joins.

## Models

The model stack includes baselines and LightGBM variants. LightGBM training now writes model
artifacts with feature metadata under `models/artifacts/`, which remains gitignored. The dashboard
and report use model outputs and feature-importance artifacts for signal explanations.

## Execution Assumptions

The baseline transaction fee assumption is 2.5%, matching the conservative marketplace-fee baseline
chosen for CSFloat/Buff-like execution. The system is designed for paper-trade review only.

## Rejection Policy

The current strategy remains conservative:

- Minimum entry price gates are used to avoid fragile low-price signals.
- Liquidity, staleness, coverage, market-regime, and unstable-price gates reject weak candidates.
- Capacity and concentration checks prevent overstating small, concentrated PnL pockets.
- Paper-trade A/B review includes top-item and top-period concentration checks.

## CSFloat Ablation

True CSFloat model ablation requires at least 500 point-in-time covered rows. If coverage is below
that threshold, the report marks `csfloat_coverage_insufficient` and keeps the production research
model SteamDT-only. CSFloat can still appear in the dashboard for liquidity and execution
diagnostics.

## Explainability

`cs_market_model.models.explain` produces `reports/tables/signal_explanations.csv`. The default path
uses global LightGBM gain/split importance for current signals, and the module also includes a
native LightGBM `pred_contrib=True` helper for saved model payloads.

## Paper-Trade A/B

The headline A/B compares:

- No-trade baseline.
- Current Day 24 `$5+` gates.
- Stricter `paper_trade_5_plus` gates.

Review criteria require at least 30 calendar days, 50 accepted trades, profit factor >= 1.2,
bootstrap 95% profit-factor lower bound >= 1.0, max drawdown better than -10%, top-1 item PnL share
<= 25%, top-3 item PnL share <= 50%, top-2 period PnL share <= 80%, and positive-period share >=
50%.

If fewer than 50 trades exist after 30 days, paper trading extends to 60 days. If fewer than 50
trades exist after 60 days, the system reports `signal_too_rare_for_review`.

## Monitoring

`cs_market_model.research.model_health_monitor` writes:

- `reports/tables/model_health_monitor.csv`
- `reports/tables/model_health_status.json`
- `reports/backtests/model_health_report.md`

Monitoring checks artifact freshness, paper-trade status, top-feature coverage, PSI-style feature
drift, CSFloat coverage, and conservative rollback rules. The dashboard reads the JSON status and
shows a health badge.

## Dashboard

The Streamlit dashboard supports:

- Current signal review.
- Rejected-candidate review.
- Capacity status.
- Price-bucket filters.
- CSFloat coverage and ablation status.
- Signal explanations.
- Paper-trade A/B status.
- Model/data health.
- Artifact freshness.
- CSV and Markdown exports.

Missing artifacts degrade gracefully with command hints rather than crashing individual panels.

## Reproducibility

Use `docs/REPRODUCE.md` for setup, lockfile install, pipeline commands, expected outputs, artifact
hash generation, and validation commands. `requirements.lock` pins the validated environment. The
artifact manifest records hashes and row counts for generated files.

## Limitations

- Historical CSFloat ablation remains invalid until sufficient point-in-time coverage exists.
- Backtests can overstate robustness when PnL is concentrated by item or period.
- The current market may be bloated and dropping, which can make many negative model calls correct
  but also increases regime-dependence.
- Paper-trade evidence must be collected over time before any economic claims are trusted.
- This project intentionally excludes live-trading automation.

## Final Recommendation

Continue with paper trading only. Use the dashboard and health monitor to track whether the stricter
paper-trade policy is stable, diversified, and reproducible. Do not use live capital from this
system.
