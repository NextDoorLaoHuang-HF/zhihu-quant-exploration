"""
tests/test_qfq_cache.py
TDD tests for the persistent incremental qfq/total-return cache with coverage tracking.

Tests the new QfqCache layer in scripts/lib/qfq_cache.py:
- SUCCESS / FAIL / DEGRADE provenance tags
- get_qfq_series(symbol, start, end) -> (series, provenance)
- build_qfq_cache(start_date, end_date) incremental
- structured coverage report with counts + per-symbol detail
- CRITICAL: network fail + no cache => no silent raw-close fallback

Run: pytest tests/test_qfq_cache.py -v
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.qfq_cache import QfqCache, Provenance, build_qfq_cache, get_qfq_series


# ============================================================
# helpers
# ============================================================

def _make_fake_fetcher(data_map: dict, fail_codes: set | None = None):
    """Return a fake fetcher callable: (code) -> (df, status_str).

    data_map: code -> pd.DataFrame (with 'date' and 'close' columns)
    fail_codes: codes that should raise a network error.
    """
    fail_codes = fail_codes or set()

    def fetcher(code):
        if code in fail_codes:
            raise ConnectionError(f"mock network error for {code}")
        if code not in data_map:
            return None, 'empty'
        return data_map[code].copy(), 'success'

    return fetcher


def _make_dates(start='2023-01-01', end='2023-06-30'):
    return pd.date_range(start, end, freq='B')


def _make_df(code, base_price=10.0, start='2023-01-01', end='2023-06-30'):
    dates = _make_dates(start, end)
    n = len(dates)
    rng = np.random.RandomState(abs(hash(code)) % (2**31))
    walk = rng.randn(n).cumsum() * 0.1
    close = np.maximum(base_price + walk, base_price * 0.3)
    df = pd.DataFrame({'date': dates, 'close': close})
    return df


# ============================================================
# Test 1: cache hit returns SUCCESS provenance
# ============================================================

def test_cache_hit_success():
    """When qfq data is in the persistent cache, get_qfq_series returns SUCCESS."""
    tmp = tempfile.mkdtemp()
    try:
        cache = QfqCache(cache_dir=tmp)
        df = _make_df('600000')
        cache._save_parquet('600000', df)

        series, prov = get_qfq_series('600000', '2023-01-01', '2023-06-30',
                                      cache_dir=tmp)
        assert prov.status == 'SUCCESS'
        assert prov.source in ('cache', 'parquet')
        assert len(series) > 0
        # values should match what we stored
        assert series.iloc[0] == pytest.approx(df['close'].iloc[0], rel=1e-6)
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 2: fresh network fetch returns SUCCESS with source=network
# ============================================================

def test_fresh_network_fetch_success():
    """When not cached, a fresh network fetch returns SUCCESS with source=network."""
    tmp = tempfile.mkdtemp()
    try:
        df = _make_df('600001')
        fetcher = _make_fake_fetcher({'600001': df})

        cache = QfqCache(cache_dir=tmp, fetcher=fetcher)
        series, prov = cache.get_series('600001', '2023-01-01', '2023-06-30')

        assert prov.status == 'SUCCESS'
        assert prov.source == 'network'
        assert len(series) > 0
        # should now be persisted to parquet
        assert os.path.exists(os.path.join(tmp, '600001.parquet'))
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 3: network fail + no cache => FAIL, no series returned
# ============================================================

def test_network_fail_no_cache_is_fail():
    """Network error with no cached data must return FAIL, not raw close."""
    tmp = tempfile.mkdtemp()
    try:
        fetcher = _make_fake_fetcher({}, fail_codes={'600002'})
        cache = QfqCache(cache_dir=tmp, fetcher=fetcher)

        series, prov = cache.get_series('600002', '2023-01-01', '2023-06-30')

        assert prov.status == 'FAIL'
        assert prov.source == 'none'
        assert series is None, "FAIL must not return a usable series"
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 4: network fail + stale cache => DEGRADE, stale series returned
# ============================================================

def test_network_fail_with_stale_cache_is_degrade():
    """Network error but stale cache exists => DEGRADE with stale data returned."""
    tmp = tempfile.mkdtemp()
    try:
        # pre-populate cache with data up to 2023-03-31 (stale)
        stale_df = _make_df('600003', start='2023-01-01', end='2023-03-31')
        cache = QfqCache(cache_dir=tmp)
        cache._save_parquet('600003', stale_df)

        # now request a range that extends beyond cache, with network failing
        fetcher = _make_fake_fetcher({}, fail_codes={'600003'})
        cache.fetcher = fetcher

        series, prov = cache.get_series('600003', '2023-01-01', '2023-06-30')

        assert prov.status == 'DEGRADE'
        assert prov.source == 'stale_cache'
        assert series is not None, "DEGRADE should return stale cached data"
        assert len(series) > 0
        # stale data only goes to 2023-03-31
        assert series.index.max() <= pd.Timestamp('2023-04-01')
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 5: incremental update — second build skips already cached
# ============================================================

def test_incremental_update_skips_cached():
    """build_qfq_cache should skip symbols already cached (incremental)."""
    tmp = tempfile.mkdtemp()
    try:
        df1 = _make_df('600004')
        df2 = _make_df('600005')

        call_count = {'n': 0}

        def counting_fetcher(code):
            call_count['n'] += 1
            data = {'600004': df1, '600005': df2}
            if code in data:
                return data[code].copy(), 'success'
            return None, 'empty'

        # First build
        cache = QfqCache(cache_dir=tmp, fetcher=counting_fetcher)
        report1 = cache.build(['600004', '600005'], '2023-01-01', '2023-06-30')
        assert report1['success'] == 2
        assert call_count['n'] == 2

        # Second build — should skip both
        call_count['n'] = 0
        report2 = cache.build(['600004', '600005'], '2023-01-01', '2023-06-30')
        assert report2['success'] == 2
        assert call_count['n'] == 0, "incremental build should not re-fetch cached symbols"
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 6: coverage report has success/fail/degrade counts + detail
# ============================================================

def test_coverage_report_structure():
    """Coverage report must have SUCCESS/FAIL/DEGRADE counts and per-symbol detail."""
    tmp = tempfile.mkdtemp()
    try:
        # 600006: success (cached)
        cache = QfqCache(cache_dir=tmp)
        cache._save_parquet('600006', _make_df('600006'))

        # 600007: fail (network error, no cache)
        # 600008: degrade (stale cache + network error)
        stale_df = _make_df('600008', start='2023-01-01', end='2023-03-31')
        cache._save_parquet('600008', stale_df)

        fetcher = _make_fake_fetcher({}, fail_codes={'600007', '600008'})
        cache.fetcher = fetcher

        # trigger fetches
        cache.get_series('600006', '2023-01-01', '2023-06-30')
        cache.get_series('600007', '2023-01-01', '2023-06-30')
        cache.get_series('600008', '2023-01-01', '2023-06-30')

        report = cache.coverage_report()

        assert 'success' in report
        assert 'fail' in report
        assert 'degrade' in report
        assert 'total' in report
        assert 'per_symbol' in report

        assert report['success'] == 1
        assert report['fail'] == 1
        assert report['degrade'] == 1
        assert report['total'] == 3

        # per_symbol detail
        detail = {d['code']: d for d in report['per_symbol']}
        assert detail['600006']['status'] == 'SUCCESS'
        assert detail['600007']['status'] == 'FAIL'
        assert detail['600008']['status'] == 'DEGRADE'
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 7: get_qfq_series module-level function works
# ============================================================

def test_get_qfq_series_module_api():
    """The module-level get_qfq_series(symbol, start, end) returns (series, provenance)."""
    tmp = tempfile.mkdtemp()
    try:
        cache = QfqCache(cache_dir=tmp)
        cache._save_parquet('600009', _make_df('600009'))

        series, prov = get_qfq_series('600009', '2023-01-01', '2023-06-30',
                                     cache_dir=tmp)
        assert isinstance(series, pd.Series)
        assert isinstance(prov, Provenance)
        assert prov.status == 'SUCCESS'
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 8: build_qfq_cache(start_date, end_date) module-level API
# ============================================================

def test_build_qfq_cache_with_dates():
    """build_qfq_cache(start_date, end_date) builds cache for the given range."""
    tmp = tempfile.mkdtemp()
    try:
        df = _make_df('600010')
        fetcher = _make_fake_fetcher({'600010': df})

        # monkeypatch QfqCache default fetcher
        import lib.qfq_cache as qc_mod
        original = qc_mod._default_fetcher

        def fake_default(code):
            return fetcher(code)

        qc_mod._default_fetcher = fake_default
        try:
            report = build_qfq_cache(
                start_date='2023-01-01', end_date='2023-06-30',
                symbols=['600010'], cache_dir=tmp,
            )
        finally:
            qc_mod._default_fetcher = original

        assert report['success'] == 1
        assert os.path.exists(os.path.join(tmp, '600010.parquet'))
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 9: date range filtering — get_qfq_series respects start/end
# ============================================================

def test_get_qfq_series_date_filtering():
    """get_qfq_series should return only the requested date range."""
    tmp = tempfile.mkdtemp()
    try:
        cache = QfqCache(cache_dir=tmp)
        df = _make_df('600011', start='2023-01-01', end='2023-12-31')
        cache._save_parquet('600011', df)

        series, prov = cache.get_series('600011', '2023-03-01', '2023-05-31')

        assert prov.status == 'SUCCESS'
        assert series.index.min() >= pd.Timestamp('2023-03-01')
        assert series.index.max() <= pd.Timestamp('2023-05-31')
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 10: FAIL does not pollute main-result path with raw close
# ============================================================

def test_fail_provenance_is_not_raw_close():
    """A FAIL must not silently return raw close — series must be None."""
    tmp = tempfile.mkdtemp()
    try:
        fetcher = _make_fake_fetcher({}, fail_codes={'600012'})
        cache = QfqCache(cache_dir=tmp, fetcher=fetcher)

        series, prov = cache.get_series('600012', '2023-01-01', '2023-06-30')

        assert prov.status == 'FAIL'
        assert series is None, \
            "FAIL path must NOT return any series (no raw-close fallback)"
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 11: _meta.json is persisted with coverage data
# ============================================================

def test_meta_json_persisted():
    """After build, _meta.json should contain success/fail/degrade lists."""
    tmp = tempfile.mkdtemp()
    try:
        df = _make_df('600013')
        fetcher = _make_fake_fetcher({'600013': df, '600014': None},
                                    fail_codes=set())
        # 600014 returns empty
        cache = QfqCache(cache_dir=tmp, fetcher=fetcher)
        cache.build(['600013', '600014'], '2023-01-01', '2023-06-30')

        meta_path = os.path.join(tmp, '_meta.json')
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)

        assert 'success' in meta
        assert 'fail' in meta
        assert 'degrade' in meta
        assert '600013' in meta['success']
        assert '600014' in meta['fail']  # empty -> fail (no data)
    finally:
        shutil.rmtree(tmp)


# ============================================================
# Test 12: provenance carries timestamp + detail
# ============================================================

def test_provenance_fields():
    """Provenance should carry status, source, and optional detail."""
    tmp = tempfile.mkdtemp()
    try:
        cache = QfqCache(cache_dir=tmp)
        cache._save_parquet('600015', _make_df('600015'))

        _, prov = cache.get_series('600015', '2023-01-01', '2023-06-30')

        assert hasattr(prov, 'status')
        assert hasattr(prov, 'source')
        assert hasattr(prov, 'detail')
        assert hasattr(prov, 'fetched_at')
    finally:
        shutil.rmtree(tmp)
