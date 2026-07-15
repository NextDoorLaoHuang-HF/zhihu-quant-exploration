# Verification Report: small_cap_v2 regenerated results

- JSON: `results/small_cap_v2_20260715_011613/small_cap.json`
- Return series: `results/small_cap_v2_20260715_011613/return_series.parquet`
- Audit timestamp: 2026-07-15T14:28:34.441401

## 0. Artifact integrity

- return_series_tag = `qfq` (expected `qfq`)
  - **PASS**: all input series tagged qfq
- n_symbols_in_series (JSON) = 4754
- parquet columns = 4754, rows = 1581
  - **PASS**: symbol count matches parquet

## 1. qfq coverage

- live_total=4590, cached=4590, failed=0, degraded=0, coverage_pct=100.0
  - **PASS**: 100% qfq coverage, 0 failed, 0 degraded

## 2. Independent recompute from saved return_series.parquet

For each scenario: take selected_history + return_series.parquet, slice each holding period [rebalance_date, next_date], recompute end/start-1 per stock, equal-weight, then feed to compute_metrics. Compare CAGR/Sharpe/MaxDD to JSON.

- parquet columns: 4754; full-coverage (all 1581 rows non-null): 2333; fragment-only (≤25 non-null): 0

  - **PASS**: market_cap_T5_filter<2 (recomputed 66/66 months; unsliceable=0; CAGR orig=0.356828 recomp=0.356828 Δ=0.00e+00)
  - **PASS**: market_cap_T5_no_filter (recomputed 66/66 months; unsliceable=0; CAGR orig=0.349014 recomp=0.349014 Δ=0.00e+00)
  - **PASS**: low_price_T5_filter<2 (recomputed 66/66 months; unsliceable=0; CAGR orig=0.450720 recomp=0.450720 Δ=0.00e+00)
  - **PASS**: low_price_T5_no_filter (recomputed 66/66 months; unsliceable=0; CAGR orig=0.921307 recomp=0.921307 Δ=0.00e+00)
  - **PASS**: market_cap_T10_filter<2 (recomputed 66/66 months; unsliceable=0; CAGR orig=0.263980 recomp=0.263980 Δ=0.00e+00)
  - **PASS**: market_cap_T10_no_filter (recomputed 66/66 months; unsliceable=0; CAGR orig=0.303341 recomp=0.303341 Δ=0.00e+00)
  - **PASS**: low_price_T10_filter<2 (recomputed 66/66 months; unsliceable=0; CAGR orig=0.335899 recomp=0.335899 Δ=0.00e+00)
  - **PASS**: low_price_T10_no_filter (recomputed 66/66 months; unsliceable=0; CAGR orig=0.633886 recomp=0.633886 Δ=0.00e+00)
  - **PASS**: market_cap_T20_filter<2 (recomputed 66/66 months; unsliceable=0; CAGR orig=0.206988 recomp=0.206988 Δ=0.00e+00)
  - **PASS**: market_cap_T20_no_filter (recomputed 66/66 months; unsliceable=0; CAGR orig=0.240160 recomp=0.240160 Δ=0.00e+00)
  - **PASS**: low_price_T20_filter<2 (recomputed 66/66 months; unsliceable=0; CAGR orig=0.263881 recomp=0.263881 Δ=0.00e+00)
  - **PASS**: low_price_T20_no_filter (recomputed 66/66 months; unsliceable=0; CAGR orig=0.463338 recomp=0.463338 Δ=0.00e+00)

Overall recompute from parquet: **PASS**

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

- last month_end in sample: 2026-06-30, today(UTC-normalized from timestamp): 2026-07-15
  - **PASS**: last month_end <= today (no partial current month)
- unique monthly return dates: 66
  all are calendar month-ends (from _get_month_end_dates): yes
  - **PASS**: all return dates are month-end boundaries
- parquet daily index range: 2020-01-02 → 2026-07-14
- params window: 2020-01-01 → 2026-07-13
- full qfq series stored per stock (not fragments); pmin 2020-01-02 reflects qfq data start, pmax 2026-07-14 reflects latest trading day (can be ≤7d after end_date).
  - **PASS**: parquet window within params [start, end+7d]; full qfq series (not fragments)

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
- complete JSON with return_series is at: `results/small_cap_v2_20260715_011613/`
  - cited path exists: True
  - cited path has return_series.parquet: False
  - CAGRs in cited (200852) JSON match new run JSON: yes
  - **PASS**: article-cited numbers match verified JSON (cited path is an earlier partial save of the same qfq run; complete run with return_series is at the new run directory)

Article citations: **PASS**

---
## Summary

- 1_recompute_from_monthly_returns: **PASS**
- 1_recompute_from_parquet_prices: **PASS**
- 2_benchmark_qfq_window: **PASS**
- 2_benchmark_excess_recompute: **PASS**
- 3_no_partial_current_month: **PASS**
- 4_article_citations: **PASS**
- 4_article_path_traceable: **PASS**
- 0_qfq_tag: **PASS**
- 0_symbol_count: **PASS**
- 1_qfq_coverage_100pct: **PASS**

Items passing: 10/10

## OVERALL: **PASS**

All acceptance items PASS. Every JSON metric recomputes exactly from stored monthly_returns (Δ=0); benchmark uses same qfq series + window; no partial current month; article citations trace to JSON fields; return_series.parquet stores full qfq series enabling independent price-series recompute of all 12 scenarios.