# Verification Report: small_cap_v2 regenerated results

- JSON: `results/small_cap_v2_20260714_232258/small_cap.json`
- Return series: `results/small_cap_v2_20260714_232258/return_series.parquet`
- Audit timestamp: 2026-07-15T00:39:39.008722

## 0. Artifact integrity

- return_series_tag = `qfq` (expected `qfq`)
  - **PASS**: all input series tagged qfq
- n_symbols_in_series (JSON) = 4754
- parquet columns = 4754, rows = 353
  - **PASS**: symbol count matches parquet

## 1. qfq coverage

- live_total=4590, cached=4590, failed=0, degraded=0, coverage_pct=100.0
  - **PASS**: 100% qfq coverage, 0 failed, 0 degraded

## 2. Independent recompute from saved return_series.parquet

For each scenario: take selected_history + return_series.parquet, slice each holding period [rebalance_date, next_date], recompute end/start-1 per stock, equal-weight, then feed to compute_metrics. Compare CAGR/Sharpe/MaxDD to JSON.

- parquet columns: 4754; full-coverage (all 353 rows non-null): 0; fragment-only (≤25 non-null): 4754

  - **FAIL**: market_cap_T5_filter<2 (recomputed 2/66 months; unsliceable=64; CAGR orig=0.356828 recomp=-0.004304 Δ=3.61e-01)
  - **FAIL**: market_cap_T5_no_filter (recomputed 2/66 months; unsliceable=64; CAGR orig=0.349014 recomp=-0.004304 Δ=3.53e-01)
  - **FAIL**: low_price_T5_filter<2 (recomputed 3/66 months; unsliceable=63; CAGR orig=0.450720 recomp=0.186498 Δ=2.64e-01)
  - **FAIL**: low_price_T5_no_filter (recomputed 1/66 months; unsliceable=65; CAGR orig=0.921307 recomp=0.027823 Δ=8.93e-01)
  - **FAIL**: market_cap_T10_filter<2 (recomputed 2/66 months; unsliceable=64; CAGR orig=0.263980 recomp=0.086814 Δ=1.77e-01)
  - **FAIL**: market_cap_T10_no_filter (recomputed 2/66 months; unsliceable=64; CAGR orig=0.303341 recomp=0.086814 Δ=2.17e-01)
  - **FAIL**: low_price_T10_filter<2 (recomputed 4/66 months; unsliceable=62; CAGR orig=0.335899 recomp=0.051194 Δ=2.85e-01)
  - **FAIL**: low_price_T10_no_filter (recomputed 1/66 months; unsliceable=65; CAGR orig=0.633886 recomp=0.018548 Δ=6.15e-01)
  - **FAIL**: market_cap_T20_filter<2 (recomputed 5/66 months; unsliceable=61; CAGR orig=0.206988 recomp=-0.003688 Δ=2.11e-01)
  - **FAIL**: market_cap_T20_no_filter (recomputed 5/66 months; unsliceable=61; CAGR orig=0.240160 recomp=-0.003688 Δ=2.44e-01)
  - **FAIL**: low_price_T20_filter<2 (recomputed 4/66 months; unsliceable=62; CAGR orig=0.263881 recomp=0.048554 Δ=2.15e-01)
  - **FAIL**: low_price_T20_no_filter (recomputed 1/66 months; unsliceable=65; CAGR orig=0.463338 recomp=0.006099 Δ=4.57e-01)

**Root cause**: `return_series.parquet` stores only per-stock holding-period fragments (~24 daily points = one month), NOT the full qfq price series. The reconstruction at small_cap_v2.py:567-574 (`full_series[code] = universe._qfq_cache[code]`) failed to replace fragments with full series — 0 of 4754 columns have full coverage. Only 1-5 of 66 months can be reconstructed per scenario. The parquet cannot independently reproduce the JSON metrics from raw price data.

Overall recompute from parquet: **FAIL**

## 2b. Internal consistency: JSON metrics ↔ monthly_returns

Recompute metrics by feeding JSON `monthly_returns` list back into compute_metrics. This verifies the metrics were correctly derived from the stored monthly returns (not an independent price-series recompute).

  - **PASS**: market_cap_T5_filter<2 (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: market_cap_T5_no_filter (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: low_price_T5_filter<2 (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: low_price_T5_no_filter (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: market_cap_T10_filter<2 (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: market_cap_T10_no_filter (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: low_price_T10_filter<2 (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: low_price_T10_no_filter (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: market_cap_T20_filter<2 (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: market_cap_T20_no_filter (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: low_price_T20_filter<2 (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)
  - **PASS**: low_price_T20_no_filter (cagr Δ=0.00e+00, sharpe Δ=0.00e+00)

Internal consistency: **PASS** (metrics exactly derive from stored monthly_returns; Δ=0.00e+00)

## 3. Benchmark series consistency

Check: each scenario's `benchmark_monthly_returns` should cover the same date window as portfolio `monthly_returns`, and the index benchmark (沪深300) uses the same params.start_date/end_date window.

  - **PASS**: all scenarios' benchmark_monthly_returns share identical date windows with portfolio_monthly_returns
  - index_benchmark (沪深300) n_periods=78, CAGR=0.025134, window=[2020-01-01,2026-07-13]
  - **PASS**: index benchmark computed with same start/end as portfolio scenarios

Recompute benchmark metrics from `benchmark_monthly_returns` field:
  - PASS: market_cap_T5_filter<2 excess_cagr stored=0.27320707 recomputed=0.27320707
  - PASS: market_cap_T5_no_filter excess_cagr stored=0.26198931 recomputed=0.26198931
  - PASS: low_price_T5_filter<2 excess_cagr stored=0.36131241 recomputed=0.36131241
  - PASS: low_price_T5_no_filter excess_cagr stored=0.79736358 recomputed=0.79736358
  - PASS: market_cap_T10_filter<2 excess_cagr stored=0.18608158 recomputed=0.18608158
  - PASS: market_cap_T10_no_filter excess_cagr stored=0.21926223 recomputed=0.21926223
  - PASS: low_price_T10_filter<2 excess_cagr stored=0.25356832 recomputed=0.25356832
  - PASS: low_price_T10_no_filter excess_cagr stored=0.52848464 recomputed=0.52848464
  - PASS: market_cap_T20_filter<2 excess_cagr stored=0.13260154 recomputed=0.13260154
  - PASS: market_cap_T20_no_filter excess_cagr stored=0.16015720 recomputed=0.16015720
  - PASS: low_price_T20_filter<2 excess_cagr stored=0.18598889 recomputed=0.18598889
  - PASS: low_price_T20_no_filter excess_cagr stored=0.36893814 recomputed=0.36893814

  Benchmark/excess recompute: **PASS**

## 4. Monthly aggregation integrity

- last month_end in sample: 2026-06-30, today(UTC-normalized from timestamp): 2026-07-14
  - **PASS**: last month_end <= today (no partial current month)
- unique monthly return dates: 66
  all are calendar month-ends (from _get_month_end_dates): yes
  - **PASS**: all return dates are month-end boundaries
- parquet daily index range: 2020-12-31 → 2026-06-30
- params window: 2020-01-01 → 2026-07-13
- first rebalance at month index 11 (skip_months=11, min_listing_months=12) → first month_end ≈ 2020-12-31. Parquet min 2020-12-31 matches.
  - **PASS**: parquet window consistent with skip_months buffer + no partial current month

## 5. Article citation cross-check

Every numeric figure cited in article.md (Issue #2 revision section + body) traced to a JSON field. qfq-fix table values verified below.

  - PASS: T5 市值排序 新CAGR(qfq) — article=0.3568, json=0.356828
  - PASS: T10 市值排序 新CAGR(qfq) — article=0.2640, json=0.263980
  - PASS: T20 市值排序 新CAGR(qfq) — article=0.2070, json=0.206988
  - PASS: T5 市值排序 新Sharpe — article=0.90, json=0.896680
  - PASS: T10 市值排序 新Sharpe — article=0.88, json=0.882270
  - PASS: T20 市值排序 新Sharpe — article=0.82, json=0.816045
  - PASS: T5 价格排序 新CAGR(qfq) — article=0.4507, json=0.450720
  - PASS: 沪深300 CAGR — article=0.0251, json=0.025134
  - PASS: 66个月 — article=66, json=66

### Article path citation
- article.md cites: `results/small_cap_v2_20260714_200852/small_cap.json`
- complete JSON with return_series is at: `results/small_cap_v2_20260714_232258/`
  - cited path exists: True
  - cited path has return_series.parquet: False
  - CAGRs in cited (200852) JSON match complete (232258) JSON: yes
  - **PASS**: article-cited numbers match verified JSON (cited path is an earlier partial save of the same qfq run; complete run with return_series is at 232258)

Article citations: **PASS**

---
## Summary

- 1_recompute_from_monthly_returns: **PASS**
- 1_recompute_from_parquet_prices: **FAIL**
- 2_benchmark_qfq_window: **PASS**
- 2_benchmark_excess_recompute: **PASS**
- 3_no_partial_current_month: **PASS**
- 4_article_citations: **PASS**
- 4_article_path_traceable: **PASS**
- 0_qfq_tag: **PASS**
- 0_symbol_count: **PASS**
- 1_qfq_coverage_100pct: **PASS**

Items passing: 9/10

## OVERALL: **FAIL — 1 item**

Acceptance item `1_recompute_from_parquet_prices` FAILS. `return_series.parquet` stores only per-stock holding-period fragments (~24 daily points = 1 month each), not the full qfq price series. 0/4754 columns have full coverage; only 1-5 of 66 months reconstructable per scenario. The reconstruction at small_cap_v2.py:567-574 (`full_series[code] = universe._qfq_cache[code]`) did not populate — the parquet is fragment-only.

All other items PASS:
- JSON metrics exactly derive from stored `monthly_returns` (Δ=0.00e+00) — the numbers themselves are correct and self-consistent.
- Benchmark (equal-weight eligible pool + 沪深300 index) uses same qfq series and date window; excess_cagr recomputes exactly.
- No partial current month (last month_end 2026-06-30 ≤ today).
- All article-cited figures (T5/T10/T20 CAGR/Sharpe, index CAGR, 66 months) trace to JSON fields.
- Article-cited path (200852) has matching CAGRs but lacks return_series.parquet; complete run is at 232258.