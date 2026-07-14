"""
tests/test_universe.py
TDD 测试用例 — 对应 scripts/lib/universe.py

运行方式（从项目根目录）：
    pytest tests/test_universe.py -v

测试不依赖网络请求，使用 from_cache() 注入模拟数据。
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# 让 tests/ 能找到 scripts/lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.universe import StockUniverse, _is_b_share, _is_a_share


# ============================================================
# 辅助：构建测试用 StockUniverse
# ============================================================

def _make_test_universe() -> StockUniverse:
    """构建包含边界case的测试数据，不触发网络请求。"""

    # 股票元信息
    stock_meta = {
        # 普通存活A股
        '600000': {
            'name': '浦发银行', 'listing_date': '1999-11-10',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        },
        # 2023年上市的新股
        '601298': {
            'name': '青岛港', 'listing_date': '2023-01-15',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        },
        # 已退市A股
        '600432': {
            'name': '退市吉恩', 'listing_date': '2003-09-18',
            'delist_date': '2024-06-05', 'is_b': False, 'is_delisted': True,
        },
        # B股 — 应被排除
        '900956': {
            'name': '东贝Ｂ股', 'listing_date': '1999-10-08',
            'delist_date': '2020-11-23', 'is_b': True, 'is_delisted': True,
        },
        # 另一只普通存活A股
        '000001': {
            'name': '平安银行', 'listing_date': '1991-04-03',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        },
        # ST存活股（当前名称含ST）
        '000005': {
            'name': '*ST世纪星源', 'listing_date': '1990-12-10',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        },
    }

    # 存活股日线数据
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')

    live_daily = {}
    for code in ['600000', '000001', '000005', '601298']:
        price = 10.0 + np.random.RandomState(hash(code) % 2**31).randn(len(dates)).cumsum() * 0.5
        price = np.maximum(price, 1.0)
        df = pd.DataFrame({
            'close': price,
            'outstanding_share': 1e9,  # 10亿股
            'volume': 1e6,
        }, index=dates)
        # 601298 在上市日前不应该有数据
        if code == '601298':
            df = df[df.index >= '2023-01-15']
        live_daily[code] = df

    # 退市股价格（前复权）
    delist_dates = pd.date_range('2020-01-01', '2024-05-31', freq='B')
    delist_prices = {
        '600432': pd.Series(
            5.0 + np.random.RandomState(42).randn(len(delist_dates)).cumsum() * 0.3,
            index=delist_dates
        ),
    }

    delist_info = {
        '600432': {
            'name': '退市吉恩',
            'delist_date': '2024-06-05',
            'data_start': '2020-01-02',
            'data_end': '2024-05-31',
            'n_days': 1000,
            'last_price': 3.50,
        },
        '900956': {
            'name': '东贝Ｂ股',
            'delist_date': '2020-11-23',
            'data_start': '2020-01-02',
            'data_end': '2020-11-04',
            'n_days': 196,
            'last_price': 3.07,
        },
    }

    name_history = {
        '600000': ['浦发银行'],
        '000001': ['S深发展A', '深发展A', '平安银行'],
        '600432': ['G吉恩', '吉恩镍业', '*ST吉恩', '退市吉恩', '吉恩5'],
    }

    return StockUniverse.from_cache(
        stock_meta=stock_meta,
        live_daily=live_daily,
        delist_prices=delist_prices,
        delist_info=delist_info,
        name_history=name_history,
        st_precise=False,
    )


# ============================================================
# 测试 1: B股被排除
# ============================================================
def test_excludes_b_shares():
    """900956（东贝B股）不在 eligible_at 结果中。"""
    universe = _make_test_universe()
    eligible = universe.eligible_at('2023-06-30')
    assert '900956' not in eligible, \
        f"B股 900956 不应出现在 eligible 列表中，但出现了"

    # 也验证 B股检测函数
    assert _is_b_share('900956') is True
    assert _is_b_share('200018') is True
    assert _is_b_share('600000') is False
    assert _is_b_share('000001') is False


# ============================================================
# 测试 2: 上市日前不在候选池
# ============================================================
def test_excludes_before_listing():
    """新股在上市日前不在候选池。"""
    universe = _make_test_universe()

    # 601298 在 2023-01-15 上市
    # 2022-12-01 不应该在 eligible 中
    eligible_before = universe.eligible_at('2022-12-01')
    assert '601298' not in eligible_before, \
        f"601298 在上市日（2023-01-15）前不应出现在 eligible 列表中"

    # 2023-06-30 应该在（但需要 ≥12 个月上市时长，所以实际上不应该在）
    # 默认 min_listing_months=12，2023-01-15 上市 → 2024-01-15 后才满12个月
    eligible_6mo = universe.eligible_at('2023-06-30')
    assert '601298' not in eligible_6mo, \
        f"601298 上市仅5个月，未满足 ≥12个月 上市时长要求"

    # 2024-06-01 应该在（上市满12个月）
    eligible_after = universe.eligible_at('2024-06-01')
    assert '601298' in eligible_after, \
        f"601298 上市已满12个月，应在 eligible 列表中"


# ============================================================
# 测试 3: 退市日后不在候选池
# ============================================================
def test_excludes_after_delist():
    """退市股在退市日后不在候选池。"""
    universe = _make_test_universe()

    # 600432 在 2024-06-05 退市
    # 2024-01-01 应该在（退市前）
    eligible_before = universe.eligible_at('2024-01-01')
    assert '600432' in eligible_before, \
        f"600432 在退市日前应在 eligible 列表中"

    # 2024-07-01 不应该在（退市后）
    eligible_after = universe.eligible_at('2024-07-01')
    assert '600432' not in eligible_after, \
        f"600432 在退市日后不应出现在 eligible 列表中"


# ============================================================
# 测试 4: 市值 = 未复权价 × 股本，不是复权价
# ============================================================
def test_market_cap_uses_raw_price():
    """市值 = 未复权收盘价 × 流通股本。"""
    universe = _make_test_universe()

    # 获取 600000 在某日的市值
    test_date = '2023-06-30'
    if test_date not in universe.live_daily['600000'].index:
        # 找最近的前一个交易日
        df = universe.live_daily['600000']
        before = df[df.index <= pd.Timestamp(test_date)]
        test_date = before.index[-1]

    raw_close = universe._get_raw_close('600000', pd.Timestamp(test_date))
    shares = universe._get_outstanding_share('600000', pd.Timestamp(test_date))
    expected_cap = raw_close * shares

    actual_cap = universe.market_cap_at('600000', test_date)

    assert actual_cap is not None, "市值不应为 None（有 outstanding_share 数据）"
    assert abs(actual_cap - expected_cap) < 1e-6, \
        f"市值={actual_cap}, 期望={expected_cap} (raw_close={raw_close} * shares={shares})"

    # 退市股没有 outstanding_share，应返回 None
    cap_delist = universe.market_cap_at('600432', '2023-06-30')
    assert cap_delist is None, \
        f"退市股 600432 无 outstanding_share，市值应返回 None"


# ============================================================
# 测试 5: ST状态拿不到时返回 None
# ============================================================
def test_st_state_returns_none_when_unknown():
    """拿不到历史ST状态时返回 None 而非猜测。"""
    universe = _make_test_universe()

    # 600000 从未叫过 ST → False
    st_600000 = universe.is_st_at('600000', '2023-06-30')
    assert st_600000 is False, \
        f"600000 从未含ST名称，is_st_at 应返回 False，但返回了 {st_600000}"

    # 600432 曾用名有 *ST吉恩，但无日期 → None（不精确）
    # 当前名称"退市吉恩"不含 ST → 返回 None
    st_600432 = universe.is_st_at('600432', '2023-06-30')
    assert st_600432 is None, \
        f"600432 曾用名有ST但无日期，is_st_at 应返回 None（不精确），但返回了 {st_600432}"

    # 000005 当前名称含 *ST → True（保守判断）
    st_000005 = universe.is_st_at('000005', '2023-06-30')
    assert st_000005 is True, \
        f"000005 当前名称含 *ST，is_st_at 应返回 True，但返回了 {st_000005}"


# ============================================================
# 测试 6: 覆盖率报告
# ============================================================
def test_coverage_report():
    """报告有/无市值数据的股票数量。"""
    universe = _make_test_universe()
    report = universe.coverage_report()

    # 基本字段存在
    assert 'total_stocks' in report
    assert 'a_shares' in report
    assert 'b_shares_excluded' in report
    assert 'has_market_cap_data' in report
    assert 'market_cap_coverage_pct' in report

    # 数值合理
    assert report['total_stocks'] == 6, f"总股票数应为6，实际 {report['total_stocks']}"
    assert report['b_shares_excluded'] == 1, f"B股数应为1，实际 {report['b_shares_excluded']}"
    assert report['a_shares'] == 5, f"A股数应为5，实际 {report['a_shares']}"
    assert report['has_market_cap_data'] == 4, \
        f"有市值数据的股票应为4（4只存活A股），实际 {report['has_market_cap_data']}"

    # 可读的 JSON（能 json.dumps）
    import json
    json_str = json.dumps(report, ensure_ascii=False, indent=2)
    assert len(json_str) > 0


# ============================================================
# 测试 7: total_return_series 对存活股使用前复权价格
# ============================================================
def test_total_return_uses_qfq():
    """存活股 total_return_series 应返回前复权价格，不是未复权 close。"""
    universe = _make_test_universe()

    # 注入模拟的前复权数据到 _qfq_cache
    # 用与 live_daily 不同的价格，验证 total_return_series 确实取的是 qfq
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    qfq_prices = pd.Series(
        20.0 + np.random.RandomState(99).randn(len(dates)).cumsum() * 0.5,
        index=dates,
    )
    universe._qfq_cache['600000'] = qfq_prices

    s = universe.total_return_series('600000', '2023-06-01', '2023-06-30')
    assert len(s) > 0, "total_return_series 应返回非空序列"

    # 验证返回的是 qfq 价格，不是 live_daily 的未复权 close
    raw_close = universe.live_daily['600000']['close']
    raw_slice = raw_close[(raw_close.index >= pd.Timestamp('2023-06-01')) &
                         (raw_close.index <= pd.Timestamp('2023-06-30'))]

    # qfq 价格应与 raw close 不同（至少有一处不同）
    assert not np.allclose(s.values, raw_slice.reindex(s.index).values), \
        "total_return_series 返回的应与前复权价格一致，而非未复权 close"

    # 确认返回的值确实来自 _qfq_cache
    expected_slice = qfq_prices[(qfq_prices.index >= pd.Timestamp('2023-06-01')) &
                                (qfq_prices.index <= pd.Timestamp('2023-06-30'))]
    assert np.allclose(s.values, expected_slice.values), \
        "total_return_series 返回值应与 _qfq_cache 中的前复权价格一致"


# ============================================================
# 测试 8: _get_qfq_close 缓存命中不触发网络请求
# ============================================================
def test_qfq_cache_hit():
    """第二次调用 _get_qfq_close 不触发网络请求（从缓存取）。"""
    universe = _make_test_universe()

    # 预填充缓存
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    cached_series = pd.Series(
        15.0 + np.random.RandomState(77).randn(len(dates)).cumsum() * 0.3,
        index=dates,
    )
    universe._qfq_cache['000001'] = cached_series

    # 第一次调用 — 应从缓存返回，不触发网络
    s1 = universe._get_qfq_close('000001')
    assert s1 is not None, "_get_qfq_close 对已缓存的股票应返回数据"
    assert s1 is cached_series, "第一次调用应返回缓存中的同一对象"

    # 验证缓存未被修改（没有新增 key）
    assert '000001' in universe._qfq_cache
    assert len(universe._qfq_cache) == 1, \
        f"缓存应只有1个条目，实际 {len(universe._qfq_cache)}"

    # 第二次调用 — 仍从缓存返回
    s2 = universe._get_qfq_close('000001')
    assert s2 is cached_series, "第二次调用应返回同一缓存对象"


# ============================================================
# 测试 9: total_return_series 对退市股仍用 delist_prices
# ============================================================
def test_total_return_delist_uses_cached():
    """退市股 total_return_series 用 delist_prices（已前复权）。"""
    universe = _make_test_universe()

    s = universe.total_return_series('600432', '2023-06-01', '2023-06-30')
    assert len(s) > 0, "退市股应返回 delist_prices 中的数据"

    # 验证来自 delist_prices
    delist_s = universe.delist_prices['600432']
    expected = delist_s[(delist_s.index >= pd.Timestamp('2023-06-01')) &
                        (delist_s.index <= pd.Timestamp('2023-06-30'))]
    assert np.allclose(s.values, expected.values), \
        "退市股 total_return_series 应直接返回 delist_prices 切片"


# ============================================================
# 测试 10: 拉取失败的股票不重复网络请求
# ============================================================
def test_qfq_failed_cached(monkeypatch):
    """_get_qfq_close 拉取失败后，第二次调用不触发网络请求。"""
    universe = _make_test_universe()

    # 用 monkeypatch 模拟 akshare，统计调用次数
    call_count = {'n': 0}

    class FakeAk:
        @staticmethod
        def stock_zh_a_daily(symbol, adjust):
            call_count['n'] += 1
            raise Exception("mock network error")

    # 注入 fake akshare 模块
    import sys
    monkeypatch.setitem(sys.modules, 'akshare', FakeAk)

    # 第一次调用 — 触发网络请求（失败）
    s1 = universe._get_qfq_close('600000')
    assert s1 is None, "网络失败时应返回 None"
    assert call_count['n'] == 1, f"第一次应触发1次网络请求，实际 {call_count['n']}"

    # 第二次调用 — 不应触发网络请求（已记入 _qfq_failed）
    s2 = universe._get_qfq_close('600000')
    assert s2 is None, "已缓存的失败结果应返回 None"
    assert call_count['n'] == 1, \
        f"第二次不应触发网络请求，但触发了 {call_count['n']} 次"

    # 确认记入了 _qfq_failed
    assert '600000' in universe._qfq_failed
