"""
TDD tests for scripts/dividend_v2.py
Each test targets a specific bug in the original 04_div_fill_rights.py (Issue #1, Problem 5).

Run: pytest tests/test_dividend_event.py -v
"""
import sys
import os
import json
import pytest
import pandas as pd
import numpy as np

# Make scripts importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from dividend_v2 import (
    compute_holder_return,
    compute_entrant_return,
    compute_fill_flag,
    compute_index_return,
    select_stock_pool,
)


# ============================================================
# Helpers
# ============================================================

def make_price_series(prices, dates=None, freq='B'):
    """Build a close-price series with DatetimeIndex."""
    if dates is None:
        dates = pd.date_range('2020-06-01', periods=len(prices), freq=freq)
    return pd.Series(prices, index=dates, dtype=float)


def make_index_series(prices, dates=None, freq='B'):
    """Build an index close-price series."""
    return make_price_series(prices, dates, freq)


# ============================================================
# Test 1: 9667c — entrant return excludes the dividend
# Original bug: total_ret = (post_price + dividend - pre_price) / pre_price
#   This adds dividend into the entrant's return, but an entrant who buys
#   on the ex-date never receives that dividend.
# ============================================================
def test_entrant_excludes_dividend():
    """
    Entrant buys at ex-date close (post-deduction price).
    Their return should be exit_price / entry_price - 1,
    NOT (exit_price + dividend - entry_price) / entry_price.

    Setup: pre_ex_close = 10.00, dividend = 0.50 (per share)
    ex_date close should be ~ 9.50 (price drops by dividend)
    holding end close = 10.00 (recovers to pre-ex level)

    Entrant return = 10.00 / 9.50 - 1 ≈ 5.26%
    NOT (10.00 + 0.50 - 9.50) / 9.50 = 10.53% (wrong, includes dividend)
    """
    pre_ex_close = 10.0
    dividend = 0.50
    ex_date_close = 9.50  # price dropped by dividend amount
    exit_close = 10.00

    entrant_ret = compute_entrant_return(
        ex_date_entry_price=ex_date_close,
        exit_price=exit_close,
    )

    # Entrant return should be ~5.26%, NOT ~10.53%
    assert abs(entrant_ret - (exit_close / ex_date_close - 1)) < 1e-10
    # Make sure it's NOT the dividend-inclusive number
    wrong = (exit_close + dividend - ex_date_close) / ex_date_close
    assert abs(entrant_ret - wrong) > 0.01


# ============================================================
# Test 2: 9667c — holder return includes the dividend
# Original bug: the script does include dividend for total_ret, but
#   when splitting into entrant/holder, the holder MUST still include it.
# ============================================================
def test_holder_includes_dividend():
    """
    Holder (who owned before ex-date) receives the dividend.
    holder_return = (exit_price + received_dividend - pre_ex_close) / pre_ex_close

    Setup: pre_ex_close = 10.0, dividend = 0.50, exit_close = 9.50
    Holder return = (9.50 + 0.50 - 10.0) / 10.0 = 0.0% (flat, dividend offsets price drop)
    NOT just 9.50 / 10.0 - 1 = -5% (wrong, excludes dividend)
    """
    pre_ex_close = 10.0
    dividend = 0.50
    exit_close = 9.50

    holder_ret = compute_holder_return(
        pre_ex_close=pre_ex_close,
        exit_price=exit_close,
        received_dividend=dividend,
    )

    expected = (exit_close + dividend - pre_ex_close) / pre_ex_close
    assert abs(holder_ret - expected) < 1e-10
    # Holder return includes dividend, so it should be higher than price-only
    price_only = exit_close / pre_ex_close - 1
    assert holder_ret > price_only


# ============================================================
# Test 3: 9667c — fill flag uses raw (unadjusted) price, not total_ret > 0
# Original bug: filled = total_ret > 0
#   This is wrong because total_ret includes the dividend, so any stock
#   that drops less than the dividend amount shows total_ret > 0 → "filled".
#   Fill should be: raw post-ex price reaches pre-ex close within the window.
# ============================================================
def test_fill_uses_raw_price():
    """
    Fill = post-ex RAW (unadjusted) price window max >= pre_ex_close.

    Case A: pre_ex=10, ex_date_open=9.50, window max raw price = 10.10 → filled=True
    Case B: pre_ex=10, ex_date_open=9.50, window max raw price = 9.80 → filled=False
    (Even though Case B total_ret might be > 0 due to dividend)
    """
    # Case A: raw price recovers to 10.10 ≥ 10.0 → filled
    window_a = make_price_series([9.50, 9.60, 9.80, 10.00, 10.10])
    pre_ex_close_a = 10.0
    assert compute_fill_flag(window_a, pre_ex_close_a) == True

    # Case B: raw price only reaches 9.80 < 10.0 → NOT filled
    window_b = make_price_series([9.50, 9.60, 9.70, 9.75, 9.80])
    pre_ex_close_b = 10.0
    assert compute_fill_flag(window_b, pre_ex_close_b) == False

    # Case B with dividend: total_ret = (9.80 + 0.50 - 10.0)/10 = 0.03 > 0
    # but fill should still be False because raw price didn't reach pre_ex_close
    dividend = 0.50
    total_ret_b = (9.80 + dividend - pre_ex_close_b) / pre_ex_close_b
    assert total_ret_b > 0  # old logic would say "filled" — WRONG
    assert compute_fill_flag(window_b, pre_ex_close_b) == False  # correct: not filled


# ============================================================
# Test 4: 9667c — market benchmark is a real index, not the same stock
# Original bug: mkt_ret = price_df.iloc[end_idx]['close'] / price_df.iloc[idx]['close'] - 1
#   This uses the same stock's price as the "market", so excess_ret = total_ret - same_stock_ret
#   is meaningless.
# ============================================================
def test_market_benchmark_is_index():
    """
    Market return must come from an index (e.g., CSI 300), not the same stock.

    Verify: compute_index_return takes an index series, not a stock series.
    The function should return index_total_return over the same date window.
    """
    idx_prices = make_index_series([3800, 3850, 3900, 3880, 3950])
    stock_prices = make_price_series([10.0, 10.5, 10.2, 10.8, 11.0])

    # Compute index return for the full window
    mkt_ret = compute_index_return(
        index_prices=idx_prices,
        start_pos=0,
        end_pos=4,
    )
    expected = 3950 / 3800 - 1
    assert abs(mkt_ret - expected) < 1e-10

    # Verify it's NOT the stock's own return
    stock_ret = 11.0 / 10.0 - 1
    assert abs(mkt_ret - stock_ret) > 0.001, "Benchmark should be index, not same stock"


# ============================================================
# Test 5: 9667c — no lookback stock selection
# Original bug: top15 = div_rank.nlargest(15, '年均股息')
#   This selects stocks by full-history average dividend — a lookback bias.
#   The fix: use all stocks that have a cash dividend event in the period,
#   don't pre-rank by historical average.
# ============================================================
def test_no_lookback_stock_selection():
    """
    select_stock_pool should NOT rank by full-history average dividend.
    Instead, it should include all stocks with a cash dividend event
    in the specified date range, grouped by pre-ex yield.

    We test with a mock dividend_detail DataFrame that has events
    from 3 stocks. None should be excluded by historical-average ranking.
    """
    # Simulate dividend detail data from 3 stocks
    div_detail = pd.DataFrame({
        'code': ['000001', '000002', '600000'],
        '名称': ['StockA', 'StockB', 'StockC'],
        '除权除息日': pd.to_datetime(['2021-06-15', '2022-06-15', '2023-06-15']),
        '派息': [3.0, 1.0, 8.0],  # per 10 shares
        '进度': ['实施', '实施', '实施'],
    })

    # The stock pool should include ALL 3 stocks, not a Top-N by historical average
    pool = select_stock_pool(div_detail, start_date='2020-01-01', end_date='2026-12-31')

    # All 3 stocks present — no lookback filtering
    assert len(pool) == 3
    assert set(pool['code']) == {'000001', '000002', '600000'}

    # Stock with small dividend (1.0/10 = 0.10 per share) is NOT excluded
    # by a "high dividend" ranking — all events are included
    assert '000002' in set(pool['code'])

    # Stocks outside date range should be excluded
    out_of_range = pd.DataFrame({
        'code': ['000099'],
        '名称': ['StockOld'],
        '除权除息日': pd.to_datetime(['2019-06-15']),
        '派息': [5.0],
        '进度': ['实施'],
    })
    pool2 = select_stock_pool(out_of_range, start_date='2020-01-01', end_date='2026-12-31')
    assert len(pool2) == 0
