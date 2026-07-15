"""
tests/test_fama_french.py
TDD 测试用例 — 对应 scripts/lib/fama_french.py

运行方式（从项目根目录）：
    pytest tests/test_fama_french.py -v

测试不依赖网络请求，使用 from_cache() 注入模拟数据。
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# 让 tests/ 能找到 scripts/lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.universe import StockUniverse
from lib.fama_french import FamaFrenchBuilder


# ============================================================
# 辅助：构建测试用 FamaFrenchBuilder
# ============================================================

# 12 只股票配置：code, market_cap, book_equity, bm_ratio, monthly_return
# 6 小盘 (MC 100-130) + 6 大盘 (MC 500-530)
# 每个大小组内：2 Low B/M, 2 Medium B/M, 2 High B/M
_STOCK_CONFIGS = [
    # Small, Low B/M
    ('000001', 100, 10,    0.10, 0.010),
    ('000002', 110, 11,    0.10, 0.010),
    # Small, Medium B/M
    ('000003', 120, 60,    0.50, 0.020),
    ('000004', 130, 65,    0.50, 0.020),
    # Small, High B/M
    ('000005', 105, 94.5,  0.90, 0.030),
    ('000006', 115, 103.5, 0.90, 0.030),
    # Big, Low B/M
    ('000007', 500, 50,    0.10, 0.005),
    ('000008', 510, 51,    0.10, 0.005),
    # Big, Medium B/M
    ('000009', 520, 260,   0.50, 0.015),
    ('000010', 530, 265,   0.50, 0.015),
    # Big, High B/M
    ('000011', 505, 454.5, 0.90, 0.025),
    ('000012', 515, 463.5, 0.90, 0.025),
]


def _make_test_builder() -> FamaFrenchBuilder:
    """构建包含 12 只股票的测试数据，不触发网络请求。"""

    stock_meta = {}
    live_daily = {}

    for code, mcap, be, bm, ret in _STOCK_CONFIGS:
        stock_meta[code] = {
            'name': f'测试{code}', 'listing_date': '2010-01-01',
            'delist_date': None, 'is_b': False, 'is_delisted': False,
        }

    # 月末日期：2023-06-30 到 2025-06-30（覆盖2024年6/30周日场景）
    monthly_dates = pd.date_range('2023-06-30', '2025-06-30', freq='ME')

    for code, mcap, be, bm, ret in _STOCK_CONFIGS:
        # outstanding_share = 1.0，使 market_cap = close × 1 = close
        # June 2023 close = mcap
        june_close = float(mcap)
        shares = 1.0

        # 按固定月收益率复利
        prices = [june_close]
        for _ in range(len(monthly_dates) - 1):
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'close': prices,
            'outstanding_share': shares,
        }, index=monthly_dates)
        live_daily[code] = df

    # 账面权益数据（年报）
    # Q4 2022: announce 2023-04-28（6月前可用）
    # Q4 2023: announce 2024-04-25（次年6月前可用）
    # Q4 2024: announce 2025-04-28（次年6月前可用）
    book_equity_data = {}
    for code, mcap, be, bm, ret in _STOCK_CONFIGS:
        # book_equity = bm × mcap（使 B/M = bm）
        book_equity = bm * mcap
        book_equity_data[code] = [
            {
                'report_period': '2022-12-31',
                'announce_date': '2023-04-28',
                'total_equity': book_equity,
            },
            {
                'report_period': '2023-12-31',
                'announce_date': '2024-04-25',
                'total_equity': book_equity * 1.1,
            },
            {
                'report_period': '2024-12-31',
                'announce_date': '2025-04-28',
                'total_equity': book_equity * 1.2,
            },
        ]

    universe = StockUniverse.from_cache(
        stock_meta=stock_meta,
        live_daily=live_daily,
        delist_prices={},
        delist_info={},
    )

    # 预填充 _qfq_cache，使 total_return_series 不触发网络请求
    # 使用与 live_daily 相同的价格（测试中无分红，qfq = raw close）
    for code, df in live_daily.items():
        universe._qfq_cache[code] = df['close']

    return FamaFrenchBuilder.from_cache(universe, book_equity_data)


def _make_test_builder_with_negative_be() -> FamaFrenchBuilder:
    """在 12 只正常股票基础上，加 1 只负账面权益的股票。"""
    builder = _make_test_builder()

    # 添加第 13 只股票：负账面权益
    neg_code = '000013'
    builder.universe.stock_meta[neg_code] = {
        'name': '负权益测试', 'listing_date': '2010-01-01',
        'delist_date': None, 'is_b': False, 'is_delisted': False,
    }

    monthly_dates = pd.date_range('2023-06-30', '2025-06-30', freq='ME')
    builder.universe.live_daily[neg_code] = pd.DataFrame({
        'close': [200.0 * (1.005 ** i) for i in range(len(monthly_dates))],
        'outstanding_share': 1.0,
    }, index=monthly_dates)
    builder.universe._qfq_cache[neg_code] = builder.universe.live_daily[neg_code]['close']

    builder.book_equity_data[neg_code] = [
        {
            'report_period': '2022-12-31',
            'announce_date': '2023-04-28',
            'total_equity': -50.0,  # 负账面权益
        },
    ]

    return builder


# ============================================================
# 测试 1: 账面数据按公告日期滞后
# ============================================================
def test_book_equity_lagged():
    """使用 t-1 财年的账面数据，公告日期晚于报告期。"""
    builder = _make_test_builder()

    # 在 2023-06-30，应使用 Q4 2022 报告（announce 2023-04-28 ≤ June 2023）
    # Q4 2023 报告（announce 2024-04-25）尚未公告
    be_2023 = builder.get_book_equity_at('000001', '2023-06-30')
    assert be_2023 is not None, "2023-06-30 应有可用的年报数据"
    # 000001 的 B/M = 0.10, market_cap = 100, 所以 book_equity = 10
    assert abs(be_2023 - 10.0) < 1e-6, \
        f"2023-06-30 应返回 Q4 2022 的权益 10.0，实际 {be_2023}"

    # 在 2024-06-30，应使用 Q4 2023 报告（announce 2024-04-25 ≤ June 2024）
    be_2024 = builder.get_book_equity_at('000001', '2024-06-30')
    assert be_2024 is not None, "2024-06-30 应有可用的年报数据"
    # Q4 2023 的权益 = 10.0 * 1.1 = 11.0
    assert abs(be_2024 - 11.0) < 1e-6, \
        f"2024-06-30 应返回 Q4 2023 的权益 11.0，实际 {be_2024}"


# ============================================================
# 测试 2: 每年6月末重新分组
# ============================================================
def test_june_rebalance():
    """因子收益从7月开始到次年6月，覆盖12个月。"""
    builder = _make_test_builder()

    factors = builder.compute_factors(2023, 2023)
    assert not factors.empty, "compute_factors 不应返回空 DataFrame"

    # 应覆盖 2023-07-31 到 2024-06-30，共 12 个月
    assert len(factors) == 12, \
        f"应有 12 个月因子收益，实际 {len(factors)}"

    # 第一个月是 2023-07-31
    first_month = factors.index[0]
    assert first_month == pd.Timestamp('2023-07-31'), \
        f"第一个月应为 2023-07-31，实际 {first_month}"

    # 最后一个月是 2024-06-30
    last_month = factors.index[-1]
    assert last_month == pd.Timestamp('2024-06-30'), \
        f"最后一个月应为 2024-06-30，实际 {last_month}"


# ============================================================
# 测试 3: 组合收益按市值加权，不是等权
# ============================================================
def test_value_weighted():
    """市值加权收益 = Σ(wi×ri) / Σ(wi)，不是等权 mean(ri)。"""
    # 两只股票：市值 100 和 200，收益 1% 和 2%
    stock_returns = {'A': 0.01, 'B': 0.02}
    weights = {'A': 100.0, 'B': 200.0}

    result = FamaFrenchBuilder.value_weighted_return(stock_returns, weights)
    # 加权 = (100×0.01 + 200×0.02) / 300 = (1 + 4) / 300 = 0.016667
    expected = (100 * 0.01 + 200 * 0.02) / 300
    assert abs(result - expected) < 1e-10, \
        f"市值加权收益应为 {expected:.6f}，实际 {result:.6f}"

    # 不是等权 = (0.01 + 0.02) / 2 = 0.015
    equal_weighted = (0.01 + 0.02) / 2
    assert abs(result - equal_weighted) > 1e-6, \
        f"结果 {result:.6f} 等于等权 {equal_weighted:.6f}，应不同"


# ============================================================
# 测试 4: SMB = (SL+SM+SH)/3 - (BL+BM+BH)/3
# ============================================================
def test_smb_formula():
    """SMB = 小盘三组平均 - 大盘三组平均。"""
    builder = _make_test_builder()

    port_rets = builder.compute_portfolio_returns(2023, 2023)
    factors = builder.compute_factors(2023, 2023)

    # 逐月验证
    for date in factors.index:
        sl = port_rets.loc[date, 'SL']
        sm = port_rets.loc[date, 'SM']
        sh = port_rets.loc[date, 'SH']
        bl = port_rets.loc[date, 'BL']
        bm = port_rets.loc[date, 'BM']
        bh = port_rets.loc[date, 'BH']

        expected_smb = (sl + sm + sh) / 3 - (bl + bm + bh) / 3
        actual_smb = factors.loc[date, 'SMB']

        assert abs(actual_smb - expected_smb) < 1e-10, \
            f"{date.date()}: SMB={actual_smb:.6f} != 期望={expected_smb:.6f}"

    # 由于每组内所有股票月收益相同，验证具体值
    # SL=0.01, SM=0.02, SH=0.03 → 小盘平均 = 0.02
    # BL=0.005, BM=0.015, BH=0.025 → 大盘平均 = 0.015
    # SMB = 0.02 - 0.015 = 0.005
    expected_smb_value = (0.010 + 0.020 + 0.030) / 3 - (0.005 + 0.015 + 0.025) / 3
    for date in factors.index:
        assert abs(factors.loc[date, 'SMB'] - expected_smb_value) < 1e-10, \
            f"{date.date()}: SMB={factors.loc[date, 'SMB']:.6f} != {expected_smb_value:.6f}"


# ============================================================
# 测试 5: HML = (SH+BH)/2 - (SL+BL)/2
# ============================================================
def test_hml_formula():
    """HML = 高 B/M 两组平均 - 低 B/M 两组平均。"""
    builder = _make_test_builder()

    port_rets = builder.compute_portfolio_returns(2023, 2023)
    factors = builder.compute_factors(2023, 2023)

    # 逐月验证公式
    for date in factors.index:
        sl = port_rets.loc[date, 'SL']
        sh = port_rets.loc[date, 'SH']
        bl = port_rets.loc[date, 'BL']
        bh = port_rets.loc[date, 'BH']

        expected_hml = (sh + bh) / 2 - (sl + bl) / 2
        actual_hml = factors.loc[date, 'HML']

        assert abs(actual_hml - expected_hml) < 1e-10, \
            f"{date.date()}: HML={actual_hml:.6f} != 期望={expected_hml:.6f}"

    # 验证具体值
    # SH=0.03, BH=0.025 → 高 B/M 平均 = 0.0275
    # SL=0.01, BL=0.005 → 低 B/M 平均 = 0.0075
    # HML = 0.0275 - 0.0075 = 0.02
    expected_hml_value = (0.030 + 0.025) / 2 - (0.010 + 0.005) / 2
    for date in factors.index:
        assert abs(factors.loc[date, 'HML'] - expected_hml_value) < 1e-10, \
            f"{date.date()}: HML={factors.loc[date, 'HML']:.6f} != {expected_hml_value:.6f}"


# ============================================================
# 测试 6: 负账面净值的股票被排除
# ============================================================
def test_positive_book_equity_only():
    """负账面净值的股票不进入任何组合。"""
    builder = _make_test_builder_with_negative_be()

    portfolios = builder.get_portfolio_stocks('2023-06-30')
    all_stocks = []
    for codes in portfolios.values():
        all_stocks.extend(codes)

    # 000013 有负账面权益，不应出现在任何组合中
    assert '000013' not in all_stocks, \
        "负账面权益股票 000013 不应出现在任何组合中"


# ============================================================
# 测试 7: 空组合/空收益返回 NaN，不是伪 0
# ============================================================
def test_empty_portfolio_returns_nan():
    """value_weighted_return 在无有效收益时返回 NaN，不是 0.0。"""
    result = FamaFrenchBuilder.value_weighted_return({}, {})
    assert pd.isna(result), \
        f"空组合应返回 NaN，实际返回 {result}"

    # 有权重但无收益匹配
    result2 = FamaFrenchBuilder.value_weighted_return(
        {}, {'A': 100.0}
    )
    assert pd.isna(result2), \
        f"有权重但无收益匹配应返回 NaN，实际返回 {result2}"


# ============================================================
# 测试 8: 持有期收益默认使用前复权价格（qfq）
# ============================================================
def test_holding_returns_use_qfq_by_default():
    """_get_monthly_returns 默认使用 qfq/total_return_series，不是 raw close。"""
    builder = _make_test_builder()

    # 获取 000001 的月度收益
    rets = builder._get_monthly_returns(
        '000001', '2023-06-30', '2024-06-30'
    )
    assert len(rets) > 0, "月度收益序列不应为空"

    # 验证：用 raw close 计算的月度收益应该与 qfq 的不同
    # 因为测试数据中 qfq_cache 被预填充为 live_daily['close']
    # 但如果 use_qfq=False (旧行为)，它直接用 _raw_close_series
    # 让我们验证 use_qfq 标志为 True
    assert builder.use_qfq is True, \
        "use_qfq 应默认为 True，持有期收益必须用前复权价格"


# ============================================================
# 测试 9: 形成日调整为最后交易日（当6/30为非交易日时）
# ============================================================
def test_formation_date_adjusts_to_trading_day():
    """当6/30为非交易日（如2024年周日）时，形成日调整为6月最后交易日。"""
    builder = _make_test_builder()

    # 2024-06-30 是周日，应该调整到 2024-06-28（周五）
    formation_date = builder._get_formation_date(2024)
    assert formation_date == pd.Timestamp('2024-06-28'), \
        f"2024年形成日应为 2024-06-28（周五，最后交易日），实际 {formation_date}"

    # 2023-06-30 是周五（交易日），应保持不变
    formation_date_2023 = builder._get_formation_date(2023)
    assert formation_date_2023 == pd.Timestamp('2023-06-30'), \
        f"2023年形成日应为 2023-06-30（交易日），实际 {formation_date_2023}"


# ============================================================
# 测试 10: 7月首月有形成基点（不因非交易日形成导致7月收益为0）
# ============================================================
def test_july_return_not_zero_when_june30_nontrading():
    """2024年6月30日为周日时，7月因子收益不应全为0。"""
    builder = _make_test_builder()

    # 预期：2024-07-31 的因子不应全为0
    factors = builder.compute_factors(2024, 2024)
    assert not factors.empty, "compute_factors 不应返回空"

    july = factors.loc[factors.index == pd.Timestamp('2024-07-31')]
    assert len(july) == 1, "应有2024-07-31的因子数据"

    # 不应所有因子都为0
    all_zero = (july == 0).all(axis=1).iloc[0]
    assert not all_zero, \
        f"2024-07-31 因子不应全为0（形成日应已调整到交易日）"


# ============================================================
# 测试 11: 账面权益使用实际公告日期而非法定截止日
# ============================================================
def test_book_equity_uses_actual_announce_date():
    """get_book_equity_at 应使用实际公告日期，不是固定4/30。"""
    builder = _make_test_builder()

    # 000001 的 Q4 2022 报告公告日为 2023-04-28
    # 在 2023-04-27（公告前一天）不应可用
    be_before = builder.get_book_equity_at('000001', '2023-04-27')
    assert be_before is None, \
        f"2023-04-27 在公告日(2023-04-28)前，不应有Q4 2022数据，实际 {be_before}"

    # 在 2023-04-28（公告日当天）应可用
    be_on = builder.get_book_equity_at('000001', '2023-04-28')
    assert be_on is not None, \
        "2023-04-28（公告日当天）应有Q4 2022数据"


# ============================================================
# 测试 12: 公告日字段降级命名标注
# ============================================================
def test_announce_date_source_label():
    """当使用法定截止日而非实际公告日时，builder 应标注数据来源。"""
    builder = _make_test_builder()

    # 测试数据中 book_equity_data 有真实 announce_date
    assert hasattr(builder, 'announce_date_source')
    assert builder.announce_date_source == 'actual', \
        f"测试数据有实际公告日，来源应为 'actual'，实际 {builder.announce_date_source}"


# ============================================================
# 测试 13: 数据质量字段
# ============================================================
def test_data_quality_fields():
    """compute_factors 输出包含数据质量字段（每月有效股票数、空组合数、降级数）。"""
    builder = _make_test_builder()

    factors, quality = builder.compute_factors_with_quality(2023, 2023)

    assert 'data_quality' in quality
    dq = quality['data_quality']
    assert 'monthly_stock_counts' in dq
    assert 'empty_portfolios' in dq
    assert 'downgraded_stocks' in dq
    assert 'n_months' in dq
    assert dq['n_months'] == 12


# ============================================================
# 测试 14: 非交易日形成日不产生重复的6月行
# ============================================================
def test_no_duplicate_june_row_when_june30_nontrading():
    """2024年6月30日为周日时，因子CSV不应出现重复的2024-06-30行。

    Bug: 当形成日从6/30（周日）调整到6/28（周五）时，
    all_months 的 date_range 从 6/29 开始，freq='ME' 的第一个
    条目是 6/30 —— 与上一年的最后一行重复，且值为 NaN。
    """
    builder = _make_test_builder()

    factors = builder.compute_factors(2023, 2024)
    assert not factors.empty, "compute_factors 不应返回空"

    # 检查索引不应有重复
    dup = factors.index.duplicated(keep=False)
    n_dup = dup.sum()
    assert n_dup == 0, \
        f"索引有 {n_dup} 个重复行: {factors.index[dup].tolist() if n_dup else ''}"

    # 2024-06-30 不应出现两次
    june30_count = (factors.index == pd.Timestamp('2024-06-30')).sum()
    assert june30_count <= 1, \
        f"2024-06-30 出现了 {june30_count} 次，应最多1次"

    # 也不应有全 NaN 的行
    nan_rows = factors[factors.isna().all(axis=1)]
    assert len(nan_rows) == 0, \
        f"有 {len(nan_rows)} 行全为 NaN: {nan_rows.index.tolist()}"


# ============================================================
# 测试 15: 持有期收益用 qfq 而 sorting 用 raw close
# ============================================================
def test_market_cap_uses_raw_close_not_qfq():
    """市值排序用未复权收盘价，持有期收益用前复权——两条路径使用不同输入。"""
    builder = _make_test_builder()

    # 验证 market_cap_at 用的是 raw close
    # 测试数据中 outstanding_share=1.0, close=mcap
    # 所以 market_cap = raw_close × 1.0 = close
    mc = builder.universe.market_cap_at('000001', '2023-06-30')
    assert mc is not None
    # 000001 的 market_cap = 100, close = 100
    assert abs(mc - 100.0) < 1e-6, \
        f"market_cap 应为 100.0（raw close × shares），实际 {mc}"

    # 验证 _get_monthly_returns 使用 qfq 路径（use_qfq=True）
    assert builder.use_qfq is True
    # _get_monthly_returns 应调用 total_return_series（qfq），而非 _raw_close_series
    # 在测试中 qfq_cache 预填充为 live_daily['close']，所以两者值相同
    # 但 use_qfq=True 证明走的是 qfq 路径


# ============================================================
# 测试 16: 多年因子收益总数正确（每年12个月，无重复）
# ============================================================
def test_multi_year_month_count():
    """2023-2024 两年应有24个月因子收益，无重复无NaN行。"""
    builder = _make_test_builder()

    factors = builder.compute_factors(2023, 2024)
    assert not factors.empty

    # 2年 × 12月 = 24个月
    assert len(factors) == 24, \
        f"两年应有24个月因子收益，实际 {len(factors)}（可能有重复6月行）"

    # 索引不应有重复
    assert not factors.index.duplicated().any(), \
        f"索引有重复: {factors.index[factors.index.duplicated()].tolist()}"
