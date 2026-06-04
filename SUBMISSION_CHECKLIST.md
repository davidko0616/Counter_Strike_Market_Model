# Submission Checklist

Use this file for the final handoff.

## Required Before Submitting

1. Confirm `.env` is not staged:

   ```powershell
   git status --short
   git check-ignore -v .env
   ```

2. Run tests:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest
   ```

3. Run the final Day 25 report if needed:

   ```powershell
   .\.venv\Scripts\python.exe -m cs_market_model.research.day25_paper_trade_rejection
   ```

4. Confirm CSFloat coverage status if needed:

   ```powershell
   .\.venv\Scripts\python.exe -m cs_market_model.research.day19_csfloat_coverage
   .\.venv\Scripts\python.exe -m cs_market_model.research.day18_csfloat_ablation
   ```

5. Run repo-wide Ruff:

   ```powershell
   .\.venv\Scripts\python.exe -m ruff check .
   ```

6. Run the dashboard:

   ```powershell
   .\.venv\Scripts\streamlit.exe run src/cs_market_model/dashboard/app.py
   ```

7. Read the final report:

   - `docs/final_submission_report.md`

## Files To Highlight

- `README.md`
- `STARTUP_PLAN.md`
- `docs/final_submission_report.md`
- `Reviews and implementation plans/todo_before_proceeding.md`
- `src/cs_market_model/dashboard/app.py`
- `src/cs_market_model/research/day25_paper_trade_rejection.py`
- `configs/backtest.yaml`

## Final Recommendation

Submit as a research MVP with a paper-trade-only recommendation.

Do not claim the model is ready for live capital.

## Do Not Commit

- `.env`
- API keys
- local virtual environment files
- generated data/report artifacts unless the submission specifically asks for a local artifact bundle
