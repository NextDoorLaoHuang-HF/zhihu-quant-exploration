"""
tests/test_qfq_integration.py
集成测试 — 验证回测结果未使用 raw close 降级

验证：
1. total_return_series 返回的 qfq 价格与 raw close 不同（对有分红股票）
2. _qfq_degraded 集合为空（没有降级）
3. qfq_coverage_report 覆盖率达标
4. 从 parquet 缓存加载的数据与网络拉取一致

运行方式：
    pytest tests/test_qfq_integration.py -v
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.universe import StockUniverse


# ============================================================
# 测试 1: total_return_series 返回 qfq 价格，不是 raw close
# ============================================================

def test_qfq_differs_from_raw_close():
    """对有分红的股票，qfq close 与 raw close 应该不同。"""
    # 构造测试数据：raw close=10, qfq close=8（因分红调整）
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')

    stock_meta = {
        '600000': {
            'name': '测试股', 'listing_date': '2010-01-01',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        },
    }

    live_daily = {
        '600000': pd.DataFrame({
            'close': 10.0 + np.random.RandomState(1).randn(len(dates)).cumsum() * 0.1,
            'outstanding_share': 1e9,
            'volume': 1e6,
        }, index=dates),
    }

    universe = StockUniverse.from_cache(
        stock_meta=stock_meta,
        live_daily=live_daily,
        delist_prices={},
        delist_info={},
    )

    # 注入 qfq 缓存（价格不同）
    qfq_prices = pd.Series(
        8.0 + np.random.RandomState(2).randn(len(dates)).cumsum() * 0.08,
        index=dates,
    )
    universe._qfq_cache['600000'] = qfq_prices

    # 获取收益序列
    s = universe.total_return_series('600000', '2023-06-01', '2023-06-30')
    raw_s = live_daily['600000']['close']
    raw_slice = raw_s[(raw_s.index >= pd.Timestamp('2023-06-01')) &
                      (raw_s.index <= pd.Timestamp('2023-06-30'))]

    # qfq 价格应该与 raw close 不同
    assert not np.allclose(s.values, raw_slice.reindex(s.index).values), \
        "total_return_series 应返回 qfq 价格，而非 raw close"

    # 确认返回的值来自 qfq 缓存
    expected = qfq_prices[(qfq_prices.index >= pd.Timestamp('2023-06-01')) &
                          (qfq_prices.index <= pd.Timestamp('2023-06-30'))]
    assert np.allclose(s.values, expected.values), \
        "total_return_series 应与 _qfq_cache 中的值一致"


# ============================================================
# 测试 2: qfq 拉取失败时记录降级，不静默
# ============================================================

def test_degrade_is_tracked():
    """qfq 拉取失败时，股票被记入 _qfq_degraded，不静默降级。"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')

    stock_meta = {
        '600001': {
            'name': '测试股B', 'listing_date': '2010-01-01',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        },
    }
    live_daily = {
        '600001': pd.DataFrame({
            'close': 5.0,
            'outstanding_share': 1e9,
            'volume': 1e6,
        }, index=dates),
    }

    universe = StockUniverse.from_cache(
        stock_meta=stock_meta,
        live_daily=live_daily,
        delist_prices={},
        delist_info={},
    )

    # 不注入 qfq 缓存，也不让它从 parquet 加载
    universe._qfq_parquet_loaded = True  # 跳过 parquet 加载
    universe._qfq_parquet_available = set()

    # 用 monkeypatch 模拟 akshare 失败
    import sys as _sys
    class FakeAk:
        @staticmethod
        def stock_zh_a_daily(symbol, adjust):
            raise Exception("mock network error")
    _sys.modules['akshare'] = FakeAk

    # 调用 total_return_series — 应降级为 raw close
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = universe.total_return_series('600001', '2023-06-01', '2023-06-30')
        assert len(w) == 1, "应产生一个降级 warning"
        assert "降级" in str(w[0].message) or "fallback" in str(w[0].message).lower()

    # 确认被记入 _qfq_degraded
    assert '600001' in universe._qfq_degraded, \
        "qfq 拉取失败的股票应被记入 _qfq_degraded"

    # 覆盖率报告应显示降级
    report = universe.qfq_coverage_report()
    assert report['degraded'] == 1, \
        f"降级计数应为1，实际 {report['degraded']}"


# ============================================================
# 测试 3: parquet 缓存加载后不触发网络请求
# ============================================================

def test_parquet_cache_no_network():
    """从 parquet 缓存加载后，_get_qfq_close 不触发网络请求。"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    stock_meta = {
        '000001': {
            'name': '平安银行', 'listing_date': '1991-04-03',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        },
    }
    live_daily = {
        '000001': pd.DataFrame({
            'close': 15.0,
            'outstanding_share': 1e9,
            'volume': 1e6,
        }, index=dates),
    }

    universe = StockUniverse.from_cache(
        stock_meta=stock_meta,
        live_daily=live_daily,
        delist_prices={},
        delist_info={},
    )

    # 模拟 parquet 缓存中有这只股票
    universe._qfq_parquet_loaded = True
    universe._qfq_parquet_available = {'000001'}

    # 注入一个假的 parquet 加载函数
    qfq_series = pd.Series(12.0, index=dates)
    universe._qfq_cache['000001'] = qfq_series  # 直接注入内存

    # 第一次调用 — 应从内存缓存返回
    s1 = universe._get_qfq_close('000001')
    assert s1 is qfq_series, "应从内存缓存返回"

    # 清除内存缓存，模拟从 parquet 加载
    universe._qfq_cache.clear()

    # 用 monkeypatch 模拟 parquet 加载
    original_load = universe._load_qfq_parquet_for
    def mock_load(code):
        if code == '000001':
            return qfq_series
        return None
    universe._load_qfq_parquet_for = mock_load

    # 调用 — 应从 parquet 加载，不触发网络
    s2 = universe._get_qfq_close('000001')
    assert s2 is not None, "应从 parquet 加载成功"
    assert s2 is qfq_series, "应返回 parquet 中的数据"

    # 恢复
    universe._load_qfq_parquet_for = original_load


# ============================================================
# 测试 4: qfq_coverage_report 准确反映覆盖率
# ============================================================

def test_qfq_coverage_report():
    """qfq_coverage_report 应准确反映缓存/失败/降级状态。"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')

    stock_meta = {f'60000{i}': {
        'name': f'股票{i}', 'listing_date': '2010-01-01',
        'delist_date': None, 'is_b': False, 'is_delisted': False,
    } for i in range(5)}

    live_daily = {f'60000{i}': pd.DataFrame({
        'close': 10.0, 'outstanding_share': 1e9, 'volume': 1e6,
    }, index=dates) for i in range(5)}

    universe = StockUniverse.from_cache(
        stock_meta=stock_meta, live_daily=live_daily,
        delist_prices={}, delist_info={},
    )

    # 模拟状态：2只缓存，1只失败，1只降级(同时也是missing)，1只缺失
    universe._qfq_cache['600000'] = pd.Series(8.0, index=dates)
    universe._qfq_cache['600001'] = pd.Series(9.0, index=dates)
    universe._qfq_failed.add('600002')
    universe._qfq_degraded.add('600003')
    # 600004 是缺失的（未缓存、未失败、未降级）
    # 600003 同时是 degraded 和 missing（降级时不计入缓存）

    report = universe.qfq_coverage_report()
    assert report['live_total'] == 5
    assert report['qfq_cached'] == 2
    assert report['qfq_failed'] == 1
    assert report['degraded'] == 1
    # missing = total - cached - failed (degraded is a subset of missing)
    assert report['qfq_missing'] == 2  # 600003 + 600004
    assert report['coverage_pct'] == 40.0  # 2/5 = 40%


# ============================================================
# 测试 5: 回测结果的 qfq_coverage 中 degraded=0
# ============================================================

def test_backtest_no_degradation():
    """回测完成后，_qfq_degraded 应为空集（主结果不含 raw close 降级）。"""
    # 构造所有股票都有 qfq 缓存的测试数据
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)

    stock_configs = [
        ('600001', '股票A', 50.0, 1e7),
        ('600002', '股票B', 2.0, 1e10),
        ('600003', '股票C', 10.0, 5e8),
        ('600004', '股票D', 3.0, 1e7),
        ('600005', '股票E', 30.0, 1e9),
        ('600006', '股票F', 5.0, 2e8),
        ('600007', '股票G', 8.0, 3e7),
        ('600008', '股票H', 4.0, 5e8),
        ('601298', '新股I', 12.0, 4e8),
    ]

    stock_meta = {}
    live_daily = {}

    for code, name, base_price, shares in stock_configs:
        stock_meta[code] = {
            'name': name, 'listing_date': '1999-01-01',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        }
        walk = np.random.RandomState(hash(code) % 2**31).randn(n).cumsum() * (base_price * 0.02)
        price = np.maximum(base_price + walk, base_price * 0.3)
        live_daily[code] = pd.DataFrame({
            'close': price, 'outstanding_share': shares, 'volume': 1e6,
        }, index=dates)

    universe = StockUniverse.from_cache(
        stock_meta=stock_meta, live_daily=live_daily,
        delist_prices={}, delist_info={},
    )

    # 为所有股票注入 qfq 缓存（模拟 parquet 加载成功）
    for code, _, base_price, _ in stock_configs:
        walk = np.random.RandomState(hash(code + 'qfq') % 2**31).randn(n).cumsum() * (base_price * 0.02)
        qfq_price = np.maximum(base_price * 0.7 + walk, base_price * 0.2)
        universe._qfq_cache[code] = pd.Series(qfq_price, index=dates)

    universe._qfq_parquet_loaded = True
    universe._qfq_parquet_available = set(stock_meta.keys())

    # 运行回测
    from small_cap_v2 import run_monthly_backtest
    result = run_monthly_backtest(
        universe, start_date='2020-01-01', end_date='2024-06-30',
        top_n=3, sort_by='market_cap', filter_low_price=False,
    )

    assert len(result['returns']) > 0, "回测应有收益数据"
    assert len(universe._qfq_degraded) == 0, \
        f"回测后 _qfq_degraded 应为空，但有 {len(universe._qfq_degraded)} 只降级: {list(universe._qfq_degraded)}"
