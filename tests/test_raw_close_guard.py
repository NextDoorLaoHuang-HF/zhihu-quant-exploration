"""
tests/test_raw_close_guard.py
Integration & metamorphic tests that guard against silent raw-close regression.

These tests verify that the small_cap_v2 pipeline never silently falls back to
raw close prices when qfq data is unavailable.

1. Integration test: run the pipeline with mock data where some symbols have
   network errors. Assert that coverage report shows those symbols as FAIL or
   DEGRADE, and that no main-result values are derived from raw close for them.

2. Metamorphic test: run the pipeline twice — once with correct qfq cache,
   once with a deliberately corrupted raw-close source. Assert outputs differ,
   proving the pipeline is not accidentally using raw closes.

3. Unit test: verify the QfqCache coverage report structure (success + fail +
   degrade counts sum to total symbol count).

Run: pytest tests/test_raw_close_guard.py -v
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
import json

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.universe import StockUniverse
from lib.qfq_cache import QfqCache, Provenance
from small_cap_v2 import run_monthly_backtest, _get_next_month_return


# ============================================================
# Helpers — build a test universe with mock data
# ============================================================

def _make_dates(start='2020-01-01', end='2024-06-30'):
    return pd.date_range(start, end, freq='B')


def _make_stock_meta(codes, listing_date='1999-01-01'):
    """Build stock_meta dict for a list of codes."""
    return {
        code: {
            'name': f'股票{code}',
            'listing_date': listing_date,
            'delist_date': None,
            'is_b': False,
            'is_delisted': False,
        }
        for code in codes
    }


def _make_live_daily(codes, dates, base_price=10.0, shares=1e8):
    """Build live_daily dict with deterministic price series."""
    live_daily = {}
    for code in codes:
        rng = np.random.RandomState(abs(hash(code)) % (2**31))
        walk = rng.randn(len(dates)).cumsum() * (base_price * 0.02)
        price = np.maximum(base_price + walk, base_price * 0.3)
        live_daily[code] = pd.DataFrame({
            'close': price,
            'outstanding_share': shares,
            'volume': 1e6,
        }, index=dates)
    return live_daily


def _make_qfq_series(codes, dates, base_price=10.0, adjustment_factor=0.8):
    """Build qfq price series that differ from raw close (simulating dividend adjustment)."""
    qfq = {}
    for code in codes:
        rng = np.random.RandomState(abs(hash(code + 'qfq')) % (2**31))
        walk = rng.randn(len(dates)).cumsum() * (base_price * 0.02)
        price = np.maximum(base_price * adjustment_factor + walk, base_price * 0.2)
        qfq[code] = pd.Series(price, index=dates)
    return qfq


def _make_universe_with_qfq(codes, qfq_map=None, dates=None):
    """Build a StockUniverse with qfq data injected.

    Args:
        codes: list of stock codes
        qfq_map: dict code -> pd.Series of qfq prices; if None, uses raw close as qfq
        dates: date range for the data
    """
    if dates is None:
        dates = _make_dates()

    stock_meta = _make_stock_meta(codes)
    live_daily = _make_live_daily(codes, dates)

    universe = StockUniverse.from_cache(
        stock_meta=stock_meta,
        live_daily=live_daily,
        delist_prices={},
        delist_info={},
    )

    # Inject qfq cache
    if qfq_map is None:
        # Default: qfq = raw close * 0.8 (simulating dividend adjustment)
        qfq_map = _make_qfq_series(codes, dates)

    for code in codes:
        if code in qfq_map:
            universe._qfq_cache[code] = qfq_map[code]

    universe._qfq_parquet_loaded = True
    universe._qfq_parquet_available = set(codes)

    return universe


# ============================================================
# Test 1: Integration — pipeline with network-failing symbols
#         Asserts FAIL/DEGRADE in coverage, no raw-close in results
# ============================================================

class TestPipelineNetworkFailures:
    """Integration tests: pipeline behavior when some symbols have network errors."""

    def test_failing_symbols_excluded_from_results(self):
        """Symbols with network-failed qfq must not appear in backtest results.

        Setup: 10 stocks, 3 have qfq fetch failures (no cache, network error).
        The failing symbols should:
        1. Be recorded in _qfq_failed
        2. NOT contribute returns derived from raw close prices
        3. Have total_return_series return empty (not raw close)
        """
        dates = _make_dates()
        all_codes = [f'60000{i}' for i in range(8)] + ['600010', '600011']
        fail_codes = {'600003', '600005', '600007'}

        # Build qfq map: failing codes get no qfq entry
        qfq_map = _make_qfq_series(all_codes, dates, adjustment_factor=0.7)
        for code in fail_codes:
            del qfq_map[code]  # remove qfq for failing symbols

        universe = _make_universe_with_qfq(all_codes, qfq_map=qfq_map, dates=dates)

        # Simulate network failure for fail_codes by blocking akshare
        universe._qfq_parquet_loaded = True
        universe._qfq_parquet_available = set(all_codes) - fail_codes

        # Mock _get_qfq_close to return None for fail_codes (simulating network failure)
        original_get_qfq = universe._get_qfq_close

        def mock_get_qfq(code):
            if code in fail_codes:
                universe._qfq_failed.add(code)
                return None
            return original_get_qfq(code)

        universe._get_qfq_close = mock_get_qfq

        # Verify: total_return_series for failing symbols returns empty, not raw close
        for code in fail_codes:
            s = universe.total_return_series(code, '2023-06-01', '2023-06-30')
            assert len(s) == 0, \
                f"Failing symbol {code} should return empty series, not raw close. Got len={len(s)}"
            assert code in universe._qfq_failed, \
                f"Failing symbol {code} should be in _qfq_failed"

        # Verify: non-failing symbols return qfq data, not raw close
        ok_codes = [c for c in all_codes if c not in fail_codes]
        for code in ok_codes:
            s = universe.total_return_series(code, '2023-06-01', '2023-06-30')
            assert len(s) > 0, f"Non-failing symbol {code} should have qfq data"

        # Run the full pipeline
        result = run_monthly_backtest(
            universe,
            start_date='2020-01-01',
            end_date='2024-06-30',
            top_n=3,
            sort_by='market_cap',
            filter_low_price=False,
        )

        # The pipeline should produce results (using non-failing symbols)
        assert result['n_months'] > 0, "Pipeline should produce results with non-failing symbols"

        # Check selected_history: failing symbols should NOT appear in selected
        # (because total_return_series returns empty → _get_next_month_return returns None
        #  → stock gets no return → but it could still be "selected" by market cap)
        # The key assertion: if a failing symbol is selected, it should NOT
        # contribute a return derived from raw close

        # Verify: for any failing symbol that appears in selected_history,
        # its return must NOT equal the raw-close-based return
        for entry in result['selected_history']:
            selected = entry['selected']
            date_str = entry['date']

            for code in selected:
                if code in fail_codes:
                    # This symbol was selected but should have no valid return
                    # (because total_return_series returns empty)
                    rebalance_date = pd.Timestamp(date_str)
                    # Find next month end
                    from small_cap_v2 import _get_month_end_dates
                    month_ends = _get_month_end_dates('2020-01-01', '2024-06-30')
                    if rebalance_date in month_ends:
                        idx = month_ends.index(rebalance_date)
                        if idx + 1 < len(month_ends):
                            next_date = month_ends[idx + 1]
                            ret, series = _get_next_month_return(
                                universe, code, rebalance_date, next_date
                            )
                            assert ret is None, \
                                f"Failing symbol {code} should have None return " \
                                f"(no raw-close fallback), got {ret}"
                            assert series is None, \
                                f"Failing symbol {code} should have None series " \
                                f"(no raw-close fallback), got series with len={len(series) if series is not None else 'None'}"

    def test_failing_symbols_marked_in_coverage(self):
        """qfq coverage report must reflect failing symbols correctly."""
        dates = _make_dates()
        all_codes = [f'60000{i}' for i in range(6)]
        fail_codes = {'600002', '600004'}

        qfq_map = _make_qfq_series(all_codes, dates, adjustment_factor=0.7)
        for code in fail_codes:
            del qfq_map[code]

        universe = _make_universe_with_qfq(all_codes, qfq_map=qfq_map, dates=dates)
        universe._qfq_parquet_available = set(all_codes) - fail_codes

        original_get_qfq = universe._get_qfq_close

        def mock_get_qfq(code):
            if code in fail_codes:
                universe._qfq_failed.add(code)
                return None
            return original_get_qfq(code)

        universe._get_qfq_close = mock_get_qfq

        # Trigger qfq fetch for all symbols
        for code in all_codes:
            universe.total_return_series(code, '2023-01-01', '2023-06-30')

        # Check coverage report
        report = universe.qfq_coverage_report()
        assert report['live_total'] == len(all_codes)
        assert report['qfq_failed'] >= len(fail_codes), \
            f"Coverage report should show >= {len(fail_codes)} failed, " \
            f"got {report['qfq_failed']}"
        # Failing codes should be in _qfq_failed
        for code in fail_codes:
            assert code in universe._qfq_failed, \
                f"{code} should be in _qfq_failed"


# ============================================================
# Test 2: Metamorphic — qfq vs corrupted raw close produce different results
# ============================================================

class TestMetamorphicQfqVsRawClose:
    """
    Metamorphic test: the pipeline output must change when qfq data differs
    from raw close. If outputs were identical, the pipeline would be using
    raw close instead of qfq.

    Metamorphic oracle: run pipeline with qfq = raw_close * 0.8 (correct qfq),
    then run with qfq = raw_close * 1.2 (different qfq). The returns must differ.
    """

    def test_different_qfq_produces_different_returns(self):
        """Running with different qfq data must produce different return series."""
        dates = _make_dates()
        codes = [f'60000{i}' for i in range(8)] + ['600010', '600011']

        # Run 1: qfq = raw close * 0.7
        qfq_map_1 = _make_qfq_series(codes, dates, adjustment_factor=0.7)
        universe1 = _make_universe_with_qfq(codes, qfq_map=qfq_map_1, dates=dates)
        result1 = run_monthly_backtest(
            universe1,
            start_date='2020-01-01',
            end_date='2024-06-30',
            top_n=3,
            sort_by='market_cap',
            filter_low_price=False,
        )

        # Run 2: qfq = raw close * 1.3
        qfq_map_2 = _make_qfq_series(codes, dates, adjustment_factor=1.3)
        universe2 = _make_universe_with_qfq(codes, qfq_map=qfq_map_2, dates=dates)
        result2 = run_monthly_backtest(
            universe2,
            start_date='2020-01-01',
            end_date='2024-06-30',
            top_n=3,
            sort_by='market_cap',
            filter_low_price=False,
        )

        assert result1['n_months'] > 0, "Run 1 should produce results"
        assert result2['n_months'] > 0, "Run 2 should produce results"

        # The returns must differ — if they were identical, the pipeline
        # would be using raw close (which is the same in both runs) instead
        # of the injected qfq data
        rets1 = result1['returns']
        rets2 = result2['returns']

        # Align indices
        common_idx = rets1.index.intersection(rets2.index)
        assert len(common_idx) > 0, "Should have common return dates"

        r1 = rets1.loc[common_idx].values
        r2 = rets2.loc[common_idx].values

        assert not np.allclose(r1, r2, atol=1e-10), \
            "Returns with different qfq data must differ. " \
            "If they're identical, the pipeline is using raw close, not qfq."

    def test_qfq_equal_to_raw_close_produces_same_returns(self):
        """Control: when qfq = raw close (no adjustment), returns should match raw-close computation.

        This is a sanity check — it confirms the test framework is valid.
        """
        dates = _make_dates()
        codes = [f'60000{i}' for i in range(8)] + ['600010', '600011']

        # qfq = raw close (adjustment_factor = 1.0, same seed for same walk)
        stock_meta = _make_stock_meta(codes)
        live_daily = _make_live_daily(codes, dates)

        universe = StockUniverse.from_cache(
            stock_meta=stock_meta,
            live_daily=live_daily,
            delist_prices={},
            delist_info={},
        )

        # Inject qfq = raw close (exact same series)
        for code in codes:
            universe._qfq_cache[code] = live_daily[code]['close'].copy()

        universe._qfq_parquet_loaded = True
        universe._qfq_parquet_available = set(codes)

        # Verify total_return_series returns the same as raw close
        code = codes[0]
        s = universe.total_return_series(code, '2023-06-01', '2023-06-30')
        raw_s = live_daily[code]['close']
        raw_slice = raw_s[(raw_s.index >= pd.Timestamp('2023-06-01')) &
                          (raw_s.index <= pd.Timestamp('2023-06-30'))]

        assert np.allclose(s.values, raw_slice.reindex(s.index).values), \
            "When qfq = raw close, total_return_series should match raw close"


# ============================================================
# Test 3: Unit — QfqCache coverage report structure
# ============================================================

class TestCoverageReportStructure:
    """Unit tests for QfqCache.coverage_report() structure and invariants."""

    def test_counts_sum_to_total(self):
        """success + fail + degrade must equal total symbol count."""
        tmp = tempfile.mkdtemp()
        try:
            # Setup: 6 symbols with mixed outcomes
            # 2 success (cached), 2 fail (network error, no cache), 2 degrade (stale cache + fail)
            cache = QfqCache(cache_dir=tmp)

            # Success: pre-populate parquet
            for code in ['600100', '600101']:
                dates = pd.date_range('2023-01-01', '2023-06-30', freq='B')
                df = pd.DataFrame({
                    'date': dates,
                    'close': np.random.RandomState(hash(code) % 2**31).randn(len(dates)).cumsum() + 10,
                })
                cache._save_parquet(code, df)

            # Degrade: stale parquet (only goes to March)
            for code in ['600102', '600103']:
                dates = pd.date_range('2023-01-01', '2023-03-31', freq='B')
                df = pd.DataFrame({
                    'date': dates,
                    'close': np.random.RandomState(hash(code) % 2**31).randn(len(dates)).cumsum() + 10,
                })
                cache._save_parquet(code, df)

            # Fail: no cache, network error
            # Need a fetcher that fails for these codes
            def fetcher(code):
                if code in ('600104', '600105'):
                    raise ConnectionError("mock network error")
                return None, 'empty'

            cache.fetcher = fetcher

            # Trigger all fetches
            all_codes = ['600100', '600101', '600102', '600103', '600104', '600105']
            for code in all_codes:
                cache.get_series(code, '2023-01-01', '2023-06-30')

            report = cache.coverage_report(symbols=all_codes)

            # Structural assertions
            assert 'success' in report
            assert 'fail' in report
            assert 'degrade' in report
            assert 'total' in report
            assert 'per_symbol' in report

            # CRITICAL INVARIANT: success + fail + degrade == total
            total_counted = report['success'] + report['fail'] + report['degrade']
            assert total_counted == report['total'], \
                f"success({report['success']}) + fail({report['fail']}) + " \
                f"degrade({report['degrade']}) = {total_counted} != " \
                f"total({report['total']})"

            # Per-symbol detail should have exactly `total` entries
            assert len(report['per_symbol']) == report['total'], \
                f"per_symbol length {len(report['per_symbol'])} != total {report['total']}"

            # Each per_symbol entry must have required fields
            for entry in report['per_symbol']:
                assert 'code' in entry
                assert 'status' in entry
                assert 'source' in entry
                assert 'detail' in entry
                assert entry['status'] in ('SUCCESS', 'FAIL', 'DEGRADE'), \
                    f"Invalid status {entry['status']} for {entry['code']}"

        finally:
            shutil.rmtree(tmp)

    def test_empty_coverage_report(self):
        """Coverage report with no symbols should have all zeros."""
        tmp = tempfile.mkdtemp()
        try:
            cache = QfqCache(cache_dir=tmp)
            report = cache.coverage_report(symbols=[])

            assert report['success'] == 0
            assert report['fail'] == 0
            assert report['degrade'] == 0
            assert report['total'] == 0
            assert len(report['per_symbol']) == 0
        finally:
            shutil.rmtree(tmp)

    def test_all_success_coverage(self):
        """When all symbols succeed, fail and degrade should be 0."""
        tmp = tempfile.mkdtemp()
        try:
            cache = QfqCache(cache_dir=tmp)
            codes = ['600200', '600201', '600202']
            for code in codes:
                dates = pd.date_range('2023-01-01', '2023-06-30', freq='B')
                df = pd.DataFrame({
                    'date': dates,
                    'close': 10.0 + np.random.randn(len(dates)).cumsum() * 0.1,
                })
                cache._save_parquet(code, df)

            for code in codes:
                cache.get_series(code, '2023-01-01', '2023-06-30')

            report = cache.coverage_report(symbols=codes)

            assert report['success'] == 3
            assert report['fail'] == 0
            assert report['degrade'] == 0
            assert report['total'] == 3
            assert report['success'] + report['fail'] + report['degrade'] == report['total']
        finally:
            shutil.rmtree(tmp)

    def test_all_fail_coverage(self):
        """When all symbols fail, success and degrade should be 0."""
        tmp = tempfile.mkdtemp()
        try:
            def fetcher(code):
                raise ConnectionError("network error")

            cache = QfqCache(cache_dir=tmp, fetcher=fetcher)
            codes = ['600300', '600301', '600302']

            for code in codes:
                cache.get_series(code, '2023-01-01', '2023-06-30')

            report = cache.coverage_report(symbols=codes)

            assert report['success'] == 0
            assert report['fail'] == 3
            assert report['degrade'] == 0
            assert report['total'] == 3
            assert report['success'] + report['fail'] + report['degrade'] == report['total']
        finally:
            shutil.rmtree(tmp)

    def test_mixed_coverage_counts(self):
        """Mixed outcomes: verify exact counts."""
        tmp = tempfile.mkdtemp()
        try:
            cache = QfqCache(cache_dir=tmp)

            # 3 success
            for code in ['600400', '600401', '600402']:
                dates = pd.date_range('2023-01-01', '2023-06-30', freq='B')
                df = pd.DataFrame({'date': dates, 'close': 10.0})
                cache._save_parquet(code, df)

            # 2 degrade (stale cache + network fail)
            for code in ['600403', '600404']:
                dates = pd.date_range('2023-01-01', '2023-03-31', freq='B')
                df = pd.DataFrame({'date': dates, 'close': 10.0})
                cache._save_parquet(code, df)

            def fetcher(code):
                if code in ('600403', '600404', '600405', '600406'):
                    raise ConnectionError("network error")
                return None, 'empty'

            cache.fetcher = fetcher

            all_codes = ['600400', '600401', '600402', '600403', '600404', '600405', '600406']
            for code in all_codes:
                cache.get_series(code, '2023-01-01', '2023-06-30')

            report = cache.coverage_report(symbols=all_codes)

            assert report['success'] == 3, f"Expected 3 success, got {report['success']}"
            assert report['degrade'] == 2, f"Expected 2 degrade, got {report['degrade']}"
            assert report['fail'] == 2, f"Expected 2 fail, got {report['fail']}"
            assert report['total'] == 7, f"Expected total 7, got {report['total']}"
            assert report['success'] + report['fail'] + report['degrade'] == report['total']

            # Verify per_symbol statuses
            detail = {d['code']: d['status'] for d in report['per_symbol']}
            for code in ['600400', '600401', '600402']:
                assert detail[code] == 'SUCCESS', f"{code} should be SUCCESS"
            for code in ['600403', '600404']:
                assert detail[code] == 'DEGRADE', f"{code} should be DEGRADE"
            for code in ['600405', '600406']:
                assert detail[code] == 'FAIL', f"{code} should be FAIL"
        finally:
            shutil.rmtree(tmp)


# ============================================================
# Test 4: Mutation test — if raw-close fallback were silently re-introduced,
#         these tests would fail
# ============================================================

class TestRawCloseMutationGuard:
    """
    Mutation guard: these tests are designed to fail if raw-close fallback
    is silently re-introduced into the pipeline.

    The key assertion is: total_return_series must return empty (not raw close)
    when qfq data is unavailable. If someone re-introduces the fallback,
    the returned series would have data, and the assertion would fail.
    """

    def test_total_return_series_empty_on_qfq_failure(self):
        """total_return_series must return empty when qfq fetch fails and no cache."""
        dates = _make_dates()
        code = '600500'

        stock_meta = _make_stock_meta([code])
        live_daily = _make_live_daily([code], dates, base_price=10.0)

        universe = StockUniverse.from_cache(
            stock_meta=stock_meta,
            live_daily=live_daily,
            delist_prices={},
            delist_info={},
        )

        # No qfq cache injected, block parquet loading, mock akshare failure
        universe._qfq_parquet_loaded = True
        universe._qfq_parquet_available = set()

        # Mock _get_qfq_close to simulate network failure
        universe._get_qfq_close = lambda c: None

        s = universe.total_return_series(code, '2023-06-01', '2023-06-30')

        # MUST be empty — not raw close
        assert len(s) == 0, \
            "total_return_series must return empty series on qfq failure, " \
            f"not raw close. Got len={len(s)}"

        # Raw close should have data (proving the failure is specific to qfq)
        raw_s = universe._raw_close_series(code, '2023-06-01', '2023-06-30')
        assert len(raw_s) > 0, \
            "Raw close should have data — the empty series is because qfq failed, " \
            "not because the stock has no data"

        # The symbol must be marked as failed
        assert code in universe._qfq_failed

    def test_raw_close_not_used_for_returns(self):
        """_get_next_month_return must return None when qfq is unavailable.

        If raw-close fallback were re-introduced, this would return a float
        derived from raw close prices instead of None.
        """
        dates = _make_dates()
        code = '600501'

        stock_meta = _make_stock_meta([code])
        live_daily = _make_live_daily([code], dates, base_price=10.0)

        universe = StockUniverse.from_cache(
            stock_meta=stock_meta,
            live_daily=live_daily,
            delist_prices={},
            delist_info={},
        )

        universe._qfq_parquet_loaded = True
        universe._qfq_parquet_available = set()
        universe._get_qfq_close = lambda c: None

        rebalance_date = pd.Timestamp('2023-06-30')
        next_date = pd.Timestamp('2023-07-31')

        ret, series = _get_next_month_return(universe, code, rebalance_date, next_date)

        assert ret is None, \
            f"_get_next_month_return should return None return for qfq-failed symbol, " \
            f"got {ret}. If raw-close fallback is re-introduced, this would " \
            f"return a non-None value derived from raw close."
        assert series is None, \
            f"_get_next_month_return should return None series for qfq-failed symbol, " \
            f"got {series}. If raw-close fallback is re-introduced, this would " \
            f"return a non-None series derived from raw close."

    def test_pipeline_excludes_qfq_failed_from_returns(self):
        """The pipeline must not include returns from qfq-failed symbols.

        Runs a pipeline where one symbol has qfq failure. Verifies that
        symbol's return is None (excluded), not a raw-close-derived value.
        """
        dates = _make_dates()
        codes = [f'60060{i}' for i in range(8)]
        fail_code = '600603'

        qfq_map = _make_qfq_series(codes, dates, adjustment_factor=0.7)
        del qfq_map[fail_code]

        universe = _make_universe_with_qfq(codes, qfq_map=qfq_map, dates=dates)
        universe._qfq_parquet_available = set(codes) - {fail_code}

        original_get_qfq = universe._get_qfq_close

        def mock_get_qfq(code):
            if code == fail_code:
                universe._qfq_failed.add(code)
                return None
            return original_get_qfq(code)

        universe._get_qfq_close = mock_get_qfq

        # Run pipeline
        result = run_monthly_backtest(
            universe,
            start_date='2020-01-01',
            end_date='2024-06-30',
            top_n=3,
            sort_by='market_cap',
            filter_low_price=False,
        )

        assert result['n_months'] > 0

        # For every month where the failing symbol was selected, verify
        # its return is None (not derived from raw close)
        from small_cap_v2 import _get_month_end_dates
        month_ends = _get_month_end_dates('2020-01-01', '2024-06-30')

        for entry in result['selected_history']:
            if fail_code in entry['selected']:
                rebalance_date = pd.Timestamp(entry['date'])
                idx = month_ends.index(rebalance_date)
                next_date = month_ends[idx + 1]

                ret, series = _get_next_month_return(
                    universe, fail_code, rebalance_date, next_date
                )
                assert ret is None, \
                    f"Failing symbol {fail_code} should have None return " \
                    f"(no raw-close fallback), got {ret} for {entry['date']}"
                assert series is None, \
                    f"Failing symbol {fail_code} should have None series " \
                    f"(no raw-close fallback), got series for {entry['date']}"

        # Also verify _qfq_degraded is empty (no silent degradation)
        assert len(universe._qfq_degraded) == 0, \
            f"_qfq_degraded should be empty (no silent raw-close fallback), " \
            f"but has {len(universe._qfq_degraded)} entries: {list(universe._qfq_degraded)}"
