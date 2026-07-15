"""
scripts/lib/qfq_cache.py
Persistent incremental qfq (前复权) / total-return cache with coverage tracking.

Design:
- File-based cache: one parquet file per symbol in cache_dir/{code}.parquet
- _meta.json tracks success / fail / degrade lists + build metadata
- Each parquet file stores the full available qfq history for that symbol
- Incremental: build() skips symbols already in the success list
- Provenance: every get_qfq_series call returns (series, Provenance)
    SUCCESS  — cache hit or fresh network fetch
    FAIL     — network error, no cache available (series is None)
    DEGRADE  — network failed but stale cache used

CRITICAL (requirement #3):
    When network fails and no cache exists, the pipeline MUST NOT silently
    fall back to raw close prices. get_qfq_series returns (None, FAIL) in
    that case — the caller must decide to halt or exclude the symbol.

Public API:
    build_qfq_cache(start_date, end_date, symbols=None, cache_dir=None, ...)
    get_qfq_series(symbol, start_date, end_date, cache_dir=None) -> (Series, Provenance)

    QfqCache class for direct use with injectable fetcher (testable).
"""
from __future__ import annotations

import os
import json
import time
import warnings
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_QFQ_CACHE_DIR = os.path.join(_DATA_DIR, 'qfq_cache')


# ============================================================
# Provenance
# ============================================================

@dataclass
class Provenance:
    """Records where a qfq series came from."""
    status: str   # 'SUCCESS' | 'FAIL' | 'DEGRADE'
    source: str   # 'cache' | 'network' | 'stale_cache' | 'none'
    detail: str = ''
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self):
        return f"Provenance(status={self.status}, source={self.source}, detail={self.detail!r})"


# ============================================================
# Default fetcher — uses akshare stock_zh_a_daily(adjust='qfq')
# ============================================================

def _default_fetcher(code: str) -> tuple[pd.DataFrame | None, str]:
    """
    Fetch qfq daily data for a live A-share via akshare.

    Returns (df, status):
        df: DataFrame with 'date' and 'close' columns, or None
        status: 'success' | 'empty' | 'fail:<ErrorType>'
    """
    try:
        import akshare as ak
        sym = f'sh{code}' if code.startswith('6') else f'sz{code}'
        df = ak.stock_zh_a_daily(symbol=sym, adjust='qfq')
        if df is None or len(df) == 0:
            return None, 'empty'
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df, 'success'
    except Exception as e:
        return None, f'fail:{type(e).__name__}'


# ============================================================
# QfqCache
# ============================================================

class QfqCache:
    """
    Persistent incremental qfq cache backed by parquet files.

    Args:
        cache_dir: directory for parquet files + _meta.json
        fetcher: callable(code) -> (df, status_str); defaults to akshare
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        fetcher: Callable[[str], tuple[pd.DataFrame | None, str]] | None = None,
    ):
        self.cache_dir = cache_dir or _QFQ_CACHE_DIR
        self.fetcher = fetcher or _default_fetcher
        os.makedirs(self.cache_dir, exist_ok=True)

        # in-memory caches
        self._series_cache: dict[str, pd.Series] = {}
        self._provenance: dict[str, Provenance] = {}
        self._meta: dict = self._load_meta()

    # --------------------------------------------------------
    # Meta persistence
    # --------------------------------------------------------
    @property
    def _meta_path(self) -> str:
        return os.path.join(self.cache_dir, '_meta.json')

    def _load_meta(self) -> dict:
        """Load _meta.json, returning a fresh structure if absent."""
        meta = {
            'success': [],
            'fail': [],
            'degrade': [],
            'empty': [],
            'build_time': None,
            'date_range': {},
        }
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                for key in ('success', 'fail', 'degrade', 'empty'):
                    if key in loaded:
                        meta[key] = loaded[key]
                if 'build_time' in loaded:
                    meta['build_time'] = loaded['build_time']
                if 'date_range' in loaded:
                    meta['date_range'] = loaded['date_range']
            except Exception:
                pass
        return meta

    def _save_meta(self):
        """Persist _meta.json."""
        self._meta['build_time'] = datetime.now(timezone.utc).isoformat()
        with open(self._meta_path, 'w', encoding='utf-8') as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # Parquet I/O
    # --------------------------------------------------------
    def _save_parquet(self, code: str, df: pd.DataFrame):
        """Save a symbol's qfq data to parquet."""
        path = os.path.join(self.cache_dir, f'{code}.parquet')
        df.to_parquet(path, index=False)

    def _load_parquet(self, code: str) -> pd.Series | None:
        """Load a symbol's qfq close series from parquet. Returns None if missing."""
        path = os.path.join(self.cache_dir, f'{code}.parquet')
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            if 'date' not in df.columns or 'close' not in df.columns:
                return None
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df['close']
        except Exception:
            return None

    def _has_parquet(self, code: str) -> bool:
        return os.path.exists(os.path.join(self.cache_dir, f'{code}.parquet'))

    # --------------------------------------------------------
    # Core: get_series
    # --------------------------------------------------------
    def get_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.Series | None, Provenance]:
        """
        Get the qfq close series for a symbol, sliced to [start_date, end_date].

        Returns (series, provenance):
            SUCCESS  — series from cache or fresh network fetch
            FAIL     — network failed, no cache; series is None
            DEGRADE  — network failed, stale cache used; series may be shorter than requested
        """
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # 1. Check in-memory cache
        if symbol in self._series_cache:
            s = self._series_cache[symbol]
            sliced = s[(s.index >= start_dt) & (s.index <= end_dt)]
            prov = Provenance(status='SUCCESS', source='cache', detail='in-memory')
            self._provenance[symbol] = prov
            return sliced, prov

        # 2. Check parquet cache
        cached_s = self._load_parquet(symbol)
        if cached_s is not None and len(cached_s) > 0:
            # Check if cache covers the requested range.
            # Use a tolerance of 10 calendar days to account for non-trading
            # days (weekends, holidays, CNY) — the cache can't have data on
            # a non-trading day, so a strict <= / >= would always fail when
            # start_dt or end_dt falls on a holiday.
            cache_start = cached_s.index.min()
            cache_end = cached_s.index.max()
            _tol = pd.Timedelta(days=10)
            covers_range = (
                cache_start <= start_dt + _tol
                and cache_end >= end_dt - _tol
                and cached_s.index.min() <= end_dt
                and cached_s.index.max() >= start_dt
            )

            if covers_range:
                # Full cache hit
                sliced = cached_s[(cached_s.index >= start_dt) & (cached_s.index <= end_dt)]
                self._series_cache[symbol] = cached_s
                prov = Provenance(status='SUCCESS', source='cache', detail='parquet full')
                self._provenance[symbol] = prov
                self._mark_success(symbol)
                return sliced, prov

            # Cache is stale (doesn't cover the full range) — try to refresh
            try:
                df, status = self.fetcher(symbol)
                if status == 'success' and df is not None and len(df) > 0:
                    s = self._series_from_df(df)
                    self._save_parquet(symbol, df)
                    self._series_cache[symbol] = s
                    sliced = s[(s.index >= start_dt) & (s.index <= end_dt)]
                    prov = Provenance(status='SUCCESS', source='network', detail='refreshed stale')
                    self._provenance[symbol] = prov
                    self._mark_success(symbol)
                    return sliced, prov
            except Exception:
                pass

            # Network failed — use stale cache (DEGRADE)
            sliced = cached_s[(cached_s.index >= start_dt) & (cached_s.index <= end_dt)]
            prov = Provenance(
                status='DEGRADE',
                source='stale_cache',
                detail=f'cache ends {cache_end.date()}, requested {end_dt.date()}',
            )
            self._provenance[symbol] = prov
            self._mark_degrade(symbol)
            return sliced, prov

        # 3. No cache at all — try network fetch
        try:
            df, status = self.fetcher(symbol)
        except Exception as e:
            df, status = None, f'fail:{type(e).__name__}'

        if status == 'success' and df is not None and len(df) > 0:
            s = self._series_from_df(df)
            self._save_parquet(symbol, df)
            self._series_cache[symbol] = s
            sliced = s[(s.index >= start_dt) & (s.index <= end_dt)]
            prov = Provenance(status='SUCCESS', source='network', detail='fresh fetch')
            self._provenance[symbol] = prov
            self._mark_success(symbol)
            return sliced, prov

        # 4. Network fail + no cache => FAIL (no raw-close fallback!)
        prov = Provenance(
            status='FAIL',
            source='none',
            detail=f'fetch status: {status}',
        )
        self._provenance[symbol] = prov
        self._mark_fail(symbol)
        return None, prov

    @staticmethod
    def _series_from_df(df: pd.DataFrame) -> pd.Series:
        """Convert a fetched DataFrame to a date-indexed close Series."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        return df['close']

    # --------------------------------------------------------
    # Meta marking helpers
    # --------------------------------------------------------
    def _mark_success(self, code: str):
        if code not in self._meta['success']:
            self._meta['success'].append(code)
        # remove from fail/degrade if previously there
        self._meta['fail'] = [c for c in self._meta['fail'] if c != code]
        self._meta['degrade'] = [c for c in self._meta['degrade'] if c != code]
        self._save_meta()

    def _mark_fail(self, code: str):
        if code not in self._meta['success'] and code not in self._meta['fail']:
            self._meta['fail'].append(code)
        self._save_meta()

    def _mark_degrade(self, code: str):
        if code not in self._meta['degrade']:
            self._meta['degrade'].append(code)
        self._save_meta()

    # --------------------------------------------------------
    # Build (batch)
    # --------------------------------------------------------
    def build(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        max_workers: int = 10,
        skip_existing: bool = True,
        verbose: bool = False,
    ) -> dict:
        """
        Batch-build the cache for a list of symbols.

        Incremental: symbols already in _meta['success'] are skipped
        (unless skip_existing=False).

        Returns a coverage report dict.
        """
        existing_success = set(self._meta['success'])

        to_fetch = symbols
        if skip_existing:
            to_fetch = [s for s in symbols if s not in existing_success]

        if verbose:
            print(f"  qfq cache build: {len(symbols)} symbols, "
                  f"{len(to_fetch)} to fetch, {len(existing_success)} cached")

        if len(to_fetch) == 0:
            if verbose:
                print("  all symbols cached, nothing to fetch")
        else:
            batch_size = max(max_workers * 5, 50)
            for batch_start in range(0, len(to_fetch), batch_size):
                batch = to_fetch[batch_start:batch_start + batch_size]

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(self._fetch_and_store, code, start_date, end_date): code
                               for code in batch}
                    for future in concurrent.futures.as_completed(futures):
                        code = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            prov = Provenance(
                                status='FAIL', source='none',
                                detail=f'exception: {type(e).__name__}',
                            )
                            self._provenance[code] = prov
                            self._mark_fail(code)

                if verbose:
                    done = batch_start + len(batch)
                    print(f"    {done}/{len(to_fetch)} "
                          f"(success={len(self._meta['success'])}, "
                          f"fail={len(self._meta['fail'])}, "
                          f"degrade={len(self._meta['degrade'])})")

        self._meta['date_range'] = {'start': start_date, 'end': end_date}
        self._save_meta()

        return self.coverage_report(symbols=symbols)

    def _fetch_and_store(self, code: str, start_date: str, end_date: str):
        """Fetch a single symbol and store to parquet. Used by build()."""
        # Check parquet first (might have been cached by a prior run)
        cached_s = self._load_parquet(code)
        if cached_s is not None and len(cached_s) > 0:
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            cache_start = cached_s.index.min()
            cache_end = cached_s.index.max()
            _tol = pd.Timedelta(days=10)
            if (cache_start <= start_dt + _tol
                    and cache_end >= end_dt - _tol
                    and cached_s.index.min() <= end_dt
                    and cached_s.index.max() >= start_dt):
                self._series_cache[code] = cached_s
                prov = Provenance(status='SUCCESS', source='cache', detail='parquet full')
                self._provenance[code] = prov
                self._mark_success(code)
                return

        # Try network fetch
        try:
            df, status = self.fetcher(code)
        except Exception as e:
            df, status = None, f'fail:{type(e).__name__}'

        if status == 'success' and df is not None and len(df) > 0:
            s = self._series_from_df(df)
            self._save_parquet(code, df)
            self._series_cache[code] = s
            prov = Provenance(status='SUCCESS', source='network', detail='fresh fetch')
            self._provenance[code] = prov
            self._mark_success(code)
            return

        # Network failed
        if cached_s is not None and len(cached_s) > 0:
            # Stale cache available
            prov = Provenance(
                status='DEGRADE', source='stale_cache',
                detail=f'fetch={status}',
            )
            self._provenance[code] = prov
            self._mark_degrade(code)
        else:
            # No cache at all
            prov = Provenance(
                status='FAIL', source='none',
                detail=f'fetch={status}',
            )
            self._provenance[code] = prov
            self._mark_fail(code)

    # --------------------------------------------------------
    # Coverage report
    # --------------------------------------------------------
    def coverage_report(self, symbols: list[str] | None = None) -> dict:
        """
        Emit a structured coverage report.

        Args:
            symbols: if provided, report only for these symbols;
                     otherwise use all symbols in _meta.

        Returns dict with:
            success, fail, degrade, total: counts
            per_symbol: list of {code, status, source, detail}
        """
        if symbols is None:
            symbols = sorted(set(
                self._meta.get('success', [])
                + self._meta.get('fail', [])
                + self._meta.get('degrade', [])
                + list(self._provenance.keys())
            ))

        per_symbol = []
        success_count = 0
        fail_count = 0
        degrade_count = 0

        for code in symbols:
            prov = self._provenance.get(code)
            if prov is None:
                # Infer from meta
                if code in self._meta.get('success', []):
                    prov = Provenance(status='SUCCESS', source='cache', detail='from meta')
                elif code in self._meta.get('degrade', []):
                    prov = Provenance(status='DEGRADE', source='stale_cache', detail='from meta')
                elif code in self._meta.get('fail', []):
                    prov = Provenance(status='FAIL', source='none', detail='from meta')
                else:
                    prov = Provenance(status='FAIL', source='none', detail='not fetched')

            per_symbol.append({
                'code': code,
                'status': prov.status,
                'source': prov.source,
                'detail': prov.detail,
            })

            if prov.status == 'SUCCESS':
                success_count += 1
            elif prov.status == 'DEGRADE':
                degrade_count += 1
            else:
                fail_count += 1

        return {
            'success': success_count,
            'fail': fail_count,
            'degrade': degrade_count,
            'total': len(symbols),
            'per_symbol': per_symbol,
            'build_time': self._meta.get('build_time'),
            'date_range': self._meta.get('date_range', {}),
        }


# ============================================================
# Module-level convenience API
# ============================================================

def build_qfq_cache(
    start_date: str,
    end_date: str,
    symbols: list[str] | None = None,
    cache_dir: str | None = None,
    fetcher: Callable[[str], tuple[pd.DataFrame | None, str]] | None = None,
    max_workers: int = 10,
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Build the persistent qfq cache for the given date range.

    Args:
        start_date, end_date: date range (YYYY-MM-DD)
        symbols: list of A-share codes; if None, fetch the full live list via akshare
        cache_dir: override cache directory
        fetcher: custom fetcher (for testing); defaults to akshare
        max_workers: concurrent fetch threads
        skip_existing: skip symbols already cached as success
        verbose: print progress

    Returns: coverage report dict (see QfqCache.coverage_report)
    """
    if symbols is None:
        symbols = _get_full_a_share_list()

    cache = QfqCache(cache_dir=cache_dir, fetcher=fetcher)
    return cache.build(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
        skip_existing=skip_existing,
        verbose=verbose,
    )


def get_qfq_series(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: str | None = None,
    fetcher: Callable[[str], tuple[pd.DataFrame | None, str]] | None = None,
) -> tuple[pd.Series | None, Provenance]:
    """
    Get the qfq close series for a symbol, sliced to [start_date, end_date].

    Returns (series, provenance):
        series: pd.Series of qfq close prices, or None if FAIL
        provenance: Provenance with status SUCCESS / FAIL / DEGRADE

    CRITICAL: when network fails and no cache exists, returns (None, FAIL).
    Does NOT silently fall back to raw close prices.
    """
    cache = QfqCache(cache_dir=cache_dir, fetcher=fetcher)
    return cache.get_series(symbol, start_date, end_date)


def _get_full_a_share_list() -> list[str]:
    """Get the full list of live A-share codes via akshare."""
    import akshare as ak
    live_list = ak.stock_info_a_code_name()
    live_list['code'] = live_list['code'].apply(lambda x: str(x).zfill(6))
    return [c for c in live_list['code'].tolist() if _is_a_share(c)]


def _is_a_share(code: str) -> bool:
    """Check if a code is an A-share (not B-share)."""
    if code.startswith('900') or code.startswith('200'):
        return False
    return code.startswith(('600', '601', '603', '605', '000', '001', '002', '003', '300', '301'))


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build persistent qfq cache')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD, default=today)')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated symbol codes (default=full market)')
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--no-skip-existing', action='store_true')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='Limit total stocks (testing)')
    args = parser.parse_args()

    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    elif args.max_stocks:
        symbols = _get_full_a_share_list()[:args.max_stocks]

    report = build_qfq_cache(
        start_date=args.start_date,
        end_date=end_date,
        symbols=symbols,
        cache_dir=args.cache_dir,
        max_workers=args.workers,
        skip_existing=not args.no_skip_existing,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"Coverage Report")
    print(f"{'='*60}")
    print(f"  Total:    {report['total']}")
    print(f"  Success:  {report['success']}")
    print(f"  Fail:     {report['fail']}")
    print(f"  Degrade:  {report['degrade']}")
    if report['total'] > 0:
        print(f"  Coverage: {report['success']/report['total']*100:.1f}%")
