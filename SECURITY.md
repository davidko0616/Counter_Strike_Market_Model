# API Key Handling

Use local environment variables for API keys. Do not put real keys in Python files,
YAML configs, notebooks, command examples, screenshots, or issue comments.

## Local Development

1. Copy `.env.example` to `.env`.
2. Add real values only in `.env`.
3. Keep `.env` local. It is ignored by Git.

Required variables:

```text
STEAMDT_API_KEY=
CSFLOAT_API_KEY=
```

The collectors load `.env` automatically for local runs, then read the values from
the process environment.

## Shared Environments

Use the host secret manager:

- GitHub Actions: repository or environment secrets.
- Streamlit Cloud: app secrets.
- Local machine: `.env` or OS-level environment variables.

Prefer keys with the smallest available scope. Rotate keys immediately if a key is
ever pasted into source, logs, chat, a notebook, or a committed file.

## Before Committing

Run:

```powershell
python -m cs_market_model.security.scan_secrets
```

The scanner checks repository files for likely committed credentials and intentionally
skips local-only paths such as `.env`, `.git`, virtual environments, caches, raw data,
and reports.

