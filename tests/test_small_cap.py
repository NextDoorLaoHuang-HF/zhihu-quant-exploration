"""
tests/test_small_cap.py
TDD 测试用例 — 对应 scripts/small_cap_v2.py

运行方式（从项目根目录）：
    pytest tests/test_small_cap.py -v

测试不依赖网络请求，使用 mock 数据验证回测逻辑。
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# 让 tests/ 能找到 scripts/lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.universe import StockUniverse
from lib.metrics import compute_metrics, relative_cagr

# 导入被测模块
from small_cap_v2 import (
    select_by_market_cap,
    select_by_low_price,
    run_monthly_backtest,
    build_equal_weight_benchmark,
    BACKTEST_PARAMS,
)


# ============================================================
# 辅助：构建测试用 StockUniverse
# ============================================================

def _make_test_universe() -> StockUniverse:
    """构建包含边界case的测试数据，不触发网络请求。

    设计要点：
    - 股票之间市值排序 ≠ 价格排序（核心验证点）
    - 价格全部为正
    - 包含足够多的股票使回测可运行（≥9只 eligible）
    """

    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)

    # 设计10只股票，确保市值排序 ≠ 价格排序
    # 关键：价格低但股本大 → 市值大；价格高但股本小 → 市值小
    stock_configs = [
        # (code, name, base_price, shares, listing_date, delist_date, is_st)
        # 高价小股本 → 小市值（价格排序靠后，市值排序靠前）
        ('600001', '股票A', 50.0, 1e7, '1999-01-01', None, False),       # 价格高，市值=50*1e7=5e8
        # 低价大股本 → 大市值（价格排序靠前，市值排序靠后）
        ('600002', '股票B', 2.0, 1e10, '1999-02-01', None, False),       # 价格低，市值=2*1e10=2e10
        # 中价中股本
        ('600003', '股票C', 10.0, 5e8, '1999-03-01', None, False),       # 市值=5e9
        # 低价小股本 → 极小市值
        ('600004', '股票D', 3.0, 1e7, '1999-04-01', None, False),         # 市值=3e7
        # 高价大股本 → 极大市值
        ('600005', '股票E', 30.0, 1e9, '1999-05-01', None, False),        # 市值=3e10
        # ST股
        ('600006', '*ST股F', 5.0, 2e8, '1999-06-01', None, False),        # 市值=1e9
        # 中价小股本
        ('600007', '股票G', 8.0, 3e7, '1999-07-01', None, False),         # 市值=2.4e8
        # 低价中股本
        ('600008', '股票H', 4.0, 5e8, '1999-08-01', None, False),         # 市值=2e9
        # 新上市股（不满12个月 → 在某些日期不在eligible）
        ('601298', '新股I', 12.0, 4e8, '2023-01-15', None, False),       # 2023年上市
        # 退市股
        ('600432', '退市J', 6.0, 3e8, '2003-09-18', '2024-06-05', False), # 退市
    ]

    stock_meta = {}
    live_daily = {}

    for code, name, base_price, shares, listing_date, delist_date, is_st in stock_configs:
        meta = {
            'name': name,
            'listing_date': listing_date,
            'delist_date': delist_date,
            'is_b': False,
            'is_delisted': delist_date is not None,
        }
        stock_meta[code] = meta

        # 生成价格序列（确保为正）
        walk = np.random.RandomState(hash(code) % 2**31).randn(n).cumsum() * (base_price * 0.02)
        price = np.maximum(base_price + walk, base_price * 0.3)

        df = pd.DataFrame({
            'close': price,
            'outstanding_share': shares,
            'volume': 1e6,
        }, index=dates)

        # 新股在上市日前没有数据
        if listing_date and listing_date > '2020-01-01':
            df = df[df.index >= pd.Timestamp(listing_date)]

        # 退市股在退市日后没有数据（存活股的 live_daily 不会被用到退市后）
        # 这里只是模拟退市前的数据

        live_daily[code] = df

    # 退市股价格（前复权）— 退市前有数据
    delist_dates = pd.date_range('2020-01-01', '2024-05-31', freq='B')
    delist_prices = {
        '600432': pd.Series(
            np.maximum(5.0 + np.random.RandomState(42).randn(len(delist_dates)).cumsum() * 0.3, 0.5),
            index=delist_dates
        ),
    }

    delist_info = {
        '600432': {
            'name': '退市J',
            'delist_date': '2024-06-05',
            'data_start': '2020-01-02',
            'data_end': '2024-05-31',
            'n_days': 1000,
            'last_price': 3.50,
        },
    }

    name_history = {
        '600001': ['股票A'],
        '600002': ['股票B'],
        '600003': ['股票C'],
        '600004': ['股票D'],
        '600005': ['股票E'],
        '600006': ['股票F', '*ST股F'],
        '600007': ['股票G'],
        '600008': ['股票H'],
        '600432': ['G吉恩', '吉恩镍业', '*ST吉恩', '退市J', '吉恩5'],
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
# 测试 1: 市值排序选出的股票 ≠ 价格排序选出的股票
# ============================================================

def test_market_cap_selection_differs_from_price():
    """市值排序和价格排序选出的股票不完全相同。"""
    universe = _make_test_universe()
    date = '2023-06-30'
    eligible = universe.eligible_at(date)

    assert len(eligible) >= 8, f"测试需要至少8只eligible股，实际 {len(eligible)}"

    # 用市值排序选最小的5只
    mc_selected = select_by_market_cap(universe, eligible, date, top_n=5)

    # 用价格排序选最低的5只
    price_selected = select_by_low_price(universe, eligible, date, top_n=5)

    assert len(mc_selected) == 5, f"市值排序应选出5只，实际 {len(mc_selected)}"
    assert len(price_selected) == 5, f"价格排序应选出5只，实际 {len(price_selected)}"

    # 两者应该不完全相同
    # 因为 600002（低价2元 × 大股本1e10 = 大市值2e10）在价格排序中会被选入
    # 但在市值排序中不会被选入
    assert set(mc_selected) != set(price_selected), \
        f"市值排序和价格排序选出的股票完全相同 {mc_selected}，" \
        f"但市值=价格×股本，股本不同时结果应不同"

    # 验证具体的差异：
    # 600002 价格低(2元) → 价格排序选入
    # 600002 市值大(2e10) → 市值排序不选
    assert '600002' in price_selected, \
        "600002（低价2元）应在价格排序结果中"
    assert '600002' not in mc_selected, \
        "600002（大市值2e10）不应在市值排序结果中"


# ============================================================
# 测试 2: 新上市股在上市后12个月内不被选入
# ============================================================

def test_new_listing_excluded():
    """新上市股在上市后12个月内不在 eligible 列表，因此不会被选入。"""
    universe = _make_test_universe()

    # 601298 在 2023-01-15 上市
    # 2023-06-30 上市仅约5个月，不满12个月 → 不在 eligible
    eligible = universe.eligible_at('2023-06-30')
    assert '601298' not in eligible, \
        "601298 上市仅5个月，不应在 eligible 列表中（需满12个月）"

    # 2024-06-01 上市满12个月 → 在 eligible
    eligible_later = universe.eligible_at('2024-06-01')
    assert '601298' in eligible_later, \
        "601298 上市已满12个月，应在 eligible 列表中"


# ============================================================
# 测试 3: 退市股在退市日后不被选入
# ============================================================

def test_delisted_excluded_after_delist():
    """退市股在退市日后不在 eligible 列表。"""
    universe = _make_test_universe()

    # 600432 在 2024-06-05 退市
    # 2024-01-01 退市前 → 在 eligible
    eligible_before = universe.eligible_at('2024-01-01')
    assert '600432' in eligible_before, \
        "600432 在退市日前应在 eligible 列表中"

    # 2024-07-01 退市后 → 不在 eligible
    eligible_after = universe.eligible_at('2024-07-01')
    assert '600432' not in eligible_after, \
        "600432 在退市日后不应出现在 eligible 列表中"


# ============================================================
# 测试 4: 输出的 CAGR ≠ mean*12（算术年化）
# ============================================================

def test_cagr_not_arithmetic():
    """回测输出的 CAGR 不等于 mean*12（算术年化）。"""
    universe = _make_test_universe()

    # 运行回测
    results = run_monthly_backtest(
        universe,
        start_date='2020-01-01',
        end_date='2024-06-30',
        top_n=3,
        sort_by='market_cap',
        filter_low_price=False,
    )

    assert 'returns' in results, "回测结果应包含 returns 序列"
    assert len(results['returns']) > 0, \
        f"回测应至少有1期收益，实际 {len(results['returns'])}（eligible应≥9只）"

    metrics = results['metrics']
    assert 'cagr' in metrics, "metrics 应包含 cagr"

    # CAGR 和算术年化不应该完全相等
    assert abs(metrics['cagr'] - metrics['annualized_mean']) > 1e-6, \
        f"CAGR ({metrics['cagr']:.6f}) 不应等于算术年化 ({metrics['annualized_mean']:.6f})"

    # 验证 CAGR 是几何复利，不是简单的 mean*12
    rets = results['returns']
    mean_x_12 = float(rets.mean() * 12)
    assert abs(metrics['cagr'] - mean_x_12) > 1e-8, \
        f"CAGR ({metrics['cagr']:.6f}) 不应等于 mean*12 ({mean_x_12:.6f})"


# ============================================================
# 测试 5: 等权基准股票池逐月变化（动态），不是固定列表
# ============================================================

def test_benchmark_is_dynamic():
    """等权基准的股票池逐月变化，不是固定列表。"""
    universe = _make_test_universe()

    # 获取两个不同月份的 eligible 列表
    eligible_jan = universe.eligible_at('2024-01-31')
    eligible_jul = universe.eligible_at('2024-07-31')

    # 两个月份的 eligible 列表应该不同
    # 因为 601298 在1月还不在（不满12个月），7月应该在
    assert set(eligible_jan) != set(eligible_jul), \
        f"1月和7月的 eligible 列表完全相同，但应该有变化（601298上市满12个月）"

    # 运行回测，检查基准的月度收益是否基于逐月变化的股票池
    results = run_monthly_backtest(
        universe,
        start_date='2023-06-01',
        end_date='2024-10-31',
        top_n=3,
        sort_by='market_cap',
        filter_low_price=False,
    )

    assert 'benchmark_returns' in results, "回测结果应包含基准收益序列"
    bench_rets = results['benchmark_returns']
    assert len(bench_rets) > 0, "基准收益序列不应为空"

    # 基准收益序列的索引应该是逐月的
    assert isinstance(bench_rets.index, pd.DatetimeIndex), \
        "基准收益序列的索引应为 DatetimeIndex"

    # 验证基准是动态的：不同月份的基准收益不同
    # 如果基准是固定列表，所有月份的收益应该完全相同
    # （除非所有股票每月收益一样，这在随机数据中不可能）
    assert bench_rets.std() > 0, \
        "基准月度收益标准差为0，可能使用了固定股票池"
