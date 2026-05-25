# @analyst — Focused Review of Commit `7712321`

**Scope:** `Add rejection validation and CSFloat snapshot workflow`
**Delta:** +2,469 lines / -191 lines across 28 files

---

## 🔴 CRITICAL

### 1. Walk-Forward Threshold Selection Optimizes on PnL — Overfitting Risk

[_select_threshold_from_training](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/backtesting/walk_forward_rejection.py#L196-L232) ranks candidate thresholds by `accepted_total_pnl`, then `profit_factor`, then `accepted_count`:

```python
ranked = pd.DataFrame(candidates).sort_values(
    ["accepted_total_pnl", "profit_factor", "accepted_count"],
    ascending=[False, False, False],
)
```

**The problem:** PnL is the noisiest possible selection criterion. With 19 threshold candidates × ~100-200 training trades, the "best PnL" threshold is heavily influenced by a few lucky trades. This is in-sample optimization wearing an out-of-sample hat.

**Better approach:** Select by **profit_factor** or **win_rate** first (more robust to outliers), with a minimum accepted count constraint. Or use Sharpe ratio which penalizes variance. Even better: use the threshold with the best **worst-case monthly PnL** across training months (minimax).

### 2. `accepted_input` is Read Then Discarded

In [day17_diagnostics.py line 91](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/research/day17_diagnostics.py#L91):

```python
pd.read_csv(accepted_input)  # ← result is never assigned
```

This reads the entire Day 16 accepted trades CSV into memory and throws it away. It's dead code that wastes I/O and silently passes if the file doesn't exist (it would crash, but the user would think the *accepted_input* is being used). Should either be removed or assigned to a variable and used.

### 3. CSFloat Price Unit Ambiguity Will Corrupt Features

[parse_csfloat_envelope](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/features/csfloat_listings.py#L101) extracts prices using keys `["price", "price_cents", "lowest_price"]`:

```python
prices = [_extract_numeric(listing, ["price", "price_cents", "lowest_price"]) ...]
```

**The problem:** `price` and `price_cents` have different units (dollars vs cents — a 100x difference). The first key found wins. If one listing has `price=10.50` (dollars) and another has `price_cents=1050` (cents), they'll both be treated as the same unit. This will produce garbage price statistics.

**Fix:** Normalize units. If key is `price_cents`, divide by 100 before returning.

---

## 🟠 SERIOUS

### 4. `tag_walk_forward_acceptance` Uses `iterrows` — O(n²) Performance

[day17_diagnostics.py lines 169-172](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/research/day17_diagnostics.py#L169-L172):

```python
for idx, row in frame.iterrows():
    key = (str(row["model_name"]), str(row["test_period"]))
    threshold = threshold_lookup.get(key, 0.0)
    frame.at[idx, "selected_threshold"] = threshold
```

This iterates row-by-row over potentially thousands of trades. Should be a vectorized merge:

```python
sel = selection[["model_name", "test_period", "selected_score_threshold"]]
frame = frame.merge(sel, on=["model_name", "test_period"], how="left")
frame["selected_threshold"] = frame["selected_score_threshold"].fillna(0.0)
```

### 5. CSFloat Backward As-Of Join Has No Staleness Limit

[add_csfloat_listing_features](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/features/csfloat_listings.py#L149-L155) uses `pd.merge_asof(direction="backward")` with no `tolerance` parameter. This means a CSFloat snapshot from **6 months ago** will be joined to today's feature row as if it's current data.

The `csfloat_snapshot_age_days` column is computed (good), but nothing prevents features with age > 30 days from being used. A snapshot of CSFloat listings from months ago is stale enough to be misleading.

**Fix:** Either add `tolerance=pd.Timedelta(days=30)` to the merge, or add a downstream filter/NaN-out for `csfloat_snapshot_age_days > 30`.

### 6. No Constraint That Selected Threshold Should Be Monotonically Non-Decreasing

The walk-forward selection picks a new threshold each month independently. But there's no regularization or constraint — the threshold could swing wildly from 0.0 to 0.85 to 0.15 between adjacent months based on noise in the training data.

The [threshold_stability](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/research/day17_diagnostics.py#L252-L275) diagnostic measures this, which is great. But the actual selection algorithm has no stability penalty. Consider adding exponential smoothing or constraining threshold changes to ±0.1 per period.

### 7. Duplicate Rejection Gate Logic Across Two Modules

[rejection_policy.py](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/backtesting/rejection_policy.py#L40-L82) has `rejection_reason_for_row()` (per-row, used by `portfolio.py`), and [rejection.py](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/backtesting/rejection.py#L57-L124) has vectorized gate logic. These implement the **same 7 gates** independently. If someone adds Gate 8 to one and forgets the other, they'll silently diverge.

**Fix:** Make the vectorized version the canonical implementation and have `rejection_reason_for_row` call it (apply to a single-row DataFrame), or generate both from a shared gate definition.

---

## 🟡 MODERATE

### 8. Walk-Forward Only Sweeps `min_score_threshold`

[walk_forward_rejection.py](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/backtesting/walk_forward_rejection.py#L122-L125) constructs the policy with only `min_score_threshold` set:

```python
policy = RejectionPolicy(
    min_score_threshold=selected_threshold,
    exclude_event_regime=False,
)
```

The other 5 gates (liquidity, staleness, coverage, price jumps, bear regime) are all at their default (no-filter) values. So the walk-forward validation only validates the score threshold — it doesn't validate the full conservative policy that includes all gates.

This means the "out-of-sample" result only covers one dimension of the multi-dimensional policy.

### 9. `_numeric_value` in rejection_policy.py Creates a pd.Series per Call

[rejection_policy.py line 88](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/backtesting/rejection_policy.py#L88):

```python
value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
```

Creating a `pd.Series` for a single scalar value is ~100x slower than `float()` with a try/except. In `portfolio.py`'s simulation loop, this is called per-row per-gate — potentially 7 × 2,706 = 18,942 unnecessary Series allocations.

### 10. CSFloat Batch Collector Has No Rate-Limit or Retry Logic

[csfloat_batch.py](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/collectors/csfloat_batch.py#L67-L81) catches all exceptions and logs them as "failure", but has no retry mechanism for transient HTTP errors (429 rate limits, 503 service unavailable). With 48 items at 0.25s sleep, one transient failure loses that item's data entirely.

### 11. Day 18/19 Research Modules Reference Day 17 Outputs But Have No Dependency Check

If Day 17 diagnostics haven't been run, Day 18/19 will fail with unclear errors when trying to load files. Should validate inputs exist at the start and provide actionable error messages.

---

## 🔵 MINOR

### 12. `_display_path` Is Now in 8 Files

Still duplicated. This commit added it to 3 more modules (csfloat_batch.py, csfloat_listings.py, day17_diagnostics.py).

### 13. Notebook Scaffold Hardcodes `Path.cwd()`

[day17_diagnostics.py line 408](file:///c:/Users/chung/Desktop/2026-1학기/인공지능기초수학/Counter_Strike_Market_Model/src/cs_market_model/research/day17_diagnostics.py#L408): `"root = Path.cwd()\\n"` — if the notebook is opened from a different working directory, all paths break. Should use `__file__` resolution.

---

## Summary Table

| # | Severity | Issue | File | Effort |
|---|----------|-------|------|--------|
| 1 | 🔴 | Threshold selection by PnL overfits to noise | walk_forward_rejection.py | Medium |
| 2 | 🔴 | accepted_input read but discarded | day17_diagnostics.py | Trivial |
| 3 | 🔴 | price vs price_cents unit mismatch | csfloat_listings.py | Low |
| 4 | 🟠 | iterrows in tag_walk_forward_acceptance | day17_diagnostics.py | Low |
| 5 | 🟠 | No staleness limit on CSFloat as-of join | csfloat_listings.py | Low |
| 6 | 🟠 | No threshold stability constraint | walk_forward_rejection.py | Medium |
| 7 | 🟠 | Duplicate gate logic across 2 modules | rejection.py + rejection_policy.py | Medium |
| 8 | 🟡 | WF only validates score threshold, not full policy | walk_forward_rejection.py | Medium |
| 9 | 🟡 | pd.Series created per scalar in _numeric_value | rejection_policy.py | Low |
| 10 | 🟡 | No retry logic for CSFloat HTTP errors | csfloat_batch.py | Low |
| 11 | 🟡 | No input validation in Day 18/19 research | day18/19 research | Low |
| 12 | 🔵 | _display_path now in 8 files | Multiple | Low |
| 13 | 🔵 | Notebook hardcodes Path.cwd() | day17_diagnostics.py | Trivial |

---

## What's Good About This Commit

To be fair — this isn't all criticism. The positive aspects:

- **Walk-forward rejection is the right idea.** Addressing my previous @analyst finding about in-sample threshold selection. The architecture is correct even if the selection criterion needs work.
- **CSFloat feature pipeline is well-structured.** The backward as-of join is the correct approach for point-in-time features. Just needs a staleness guard.
- **Day 17 diagnostics are exactly what's needed.** Item attribution, period attribution, threshold stability — these are the right questions to ask.
- **rejection_policy.py extraction** is good refactoring. Just needs the duplication resolved.
- **Test coverage is reasonable.** New test files cover the happy paths. Edge cases could be stronger but the foundation is there.
