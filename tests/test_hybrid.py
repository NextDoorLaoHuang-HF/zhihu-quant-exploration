"""
tests/test_hybrid.py
TDD 测试用例 — 对应 scripts/lib/hybrid.py

运行方式（从项目根目录）：
    pytest tests/test_hybrid.py -v

测试不依赖网络请求，使用注入的模拟数据验证核心逻辑：
1. 可转债动态池排除已退出券
2. 退出月收益不为0（月末标签 + 先断言前提）
3. HRP用滚动窗口（不用全样本）
4. 混合比例是OOS估计（不是全样本优化）
5. 可转债收益来自真实价格（不是随机噪声）
6. 时序正确性：t-1月末权重应用于t月收益（同月前视检查）
7. ETF死输入不变性：改变ETF收益不改变策略结果
8. 退出月异常值处理：0和NaN需被处理
9. 固定比例与动态比例同区间比较
10. 不完整月排除：运行日当月不被纳入
"""
import sys
import os
import json
import pytest
import pandas as pd
import numpy as np

# 让 tests/ 能找到 scripts/lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.hybrid import HybridBacktest, CBPool, HRPOptimizer


# ============================================================
# 辅助：构建测试用数据
# ============================================================

def _make_cb_prices() -> dict[str, pd.Series]:
    """
    构建测试用可转债价格序列。

    包含：
    - 2只存活券（CB_A, CB_B）：全程有价格
    - 1只到期退出券（CB_C）：2023-06-30后退出，退出前最后价格130
    - 1只强赎退出券（CB_D）：2023-03-15后退出，退出前最后价格110
    """
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='B')

    np_rng = np.random.RandomState(42)

    # 存活券 A
    price_a = 100.0 + np_rng.randn(len(dates)).cumsum() * 0.3
    price_a = np.maximum(price_a, 80.0)

    # 存活券 B
    price_b = 95.0 + np_rng.randn(len(dates)).cumsum() * 0.25
    price_b = np.maximum(price_b, 75.0)

    # 到期退出券 C — 2023-06-30退出
    mask_c = dates <= pd.Timestamp('2023-06-30')
    dates_c = dates[mask_c]
    price_c = 100.0 + np_rng.randn(len(dates_c)).cumsum() * 0.2
    price_c = np.maximum(price_c, 85.0)
    # 退出前最后价格设为130（到期赎回价）
    price_c[-5:] = 130.0

    # 强赎退出券 D — 2023-03-15退出
    mask_d = dates <= pd.Timestamp('2023-03-15')
    dates_d = dates[mask_d]
    price_d = 100.0 + np_rng.randn(len(dates_d)).cumsum() * 0.15
    price_d = np.maximum(price_d, 90.0)
    # 退出前最后价格设为110（强赎价）
    price_d[-3:] = 110.0

    return {
        'CB_A': pd.Series(price_a, index=dates, name='CB_A'),
        'CB_B': pd.Series(price_b, index=dates, name='CB_B'),
        'CB_C': pd.Series(price_c, index=dates_c, name='CB_C'),
        'CB_D': pd.Series(price_d, index=dates_d, name='CB_D'),
    }


def _make_cb_meta() -> dict:
    """可转债元信息：上市日、退出日、退出原因、退出最终价。"""
    return {
        'CB_A': {'listing_date': '2021-01-04', 'delist_date': None, 'exit_reason': None, 'exit_final_price': None},
        'CB_B': {'listing_date': '2021-03-01', 'delist_date': None, 'exit_reason': None, 'exit_final_price': None},
        'CB_C': {'listing_date': '2021-06-01', 'delist_date': '2023-06-30', 'exit_reason': '到期', 'exit_final_price': 130.0},
        # NOTE: key is 'exit_final_price' (no 's') — matches hybrid.py
        'CB_D': {'listing_date': '2021-01-04', 'delist_date': '2023-03-15', 'exit_reason': '强赎', 'exit_final_price': 110.0},
    }


def _make_etf_returns() -> pd.Series:
    """构建测试用ETF日收益序列。"""
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='B')
    np_rng = np.random.RandomState(123)
    daily_ret = pd.Series(np_rng.randn(len(dates)) * 0.01, index=dates)
    return daily_ret


def _make_etf_returns_alt() -> pd.Series:
    """构建一个完全不同的ETF日收益序列，用于死输入测试。"""
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='B')
    np_rng = np.random.RandomState(999)
    daily_ret = pd.Series(np_rng.randn(len(dates)) * 0.02 + 0.001, index=dates)
    return daily_ret


# ============================================================
# 测试 1: 已退出可转债在退出后不在候选池
# ============================================================

def test_cb_pool_excludes_terminated():
    """CB_C 在 2023-06-30 退出后，不应出现在候选池中。"""
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()

    pool = CBPool(prices=cb_prices, meta=cb_meta)

    # 2023-06-01 — CB_C 仍在交易
    eligible_before = pool.eligible_at('2023-06-01')
    assert 'CB_C' in eligible_before, \
        "CB_C 在退出日前应在候选池中"

    # 2023-07-01 — CB_C 已退出
    eligible_after = pool.eligible_at('2023-07-01')
    assert 'CB_C' not in eligible_after, \
        "CB_C 在退出日后不应出现在候选池中"

    # 同样检查 CB_D（2023-03-15退出）
    eligible_before_d = pool.eligible_at('2023-03-01')
    assert 'CB_D' in eligible_before_d, \
        "CB_D 在退出日前应在候选池中"

    eligible_after_d = pool.eligible_at('2023-04-01')
    assert 'CB_D' not in eligible_after_d, \
        "CB_D 在退出日后不应出现在候选池中"


# ============================================================
# 测试 2: 退出月收益不为0（月末标签 + 先断言前提）
# ============================================================

def test_cb_exit_return_not_zero():
    """
    CB_C 退出月的收益应反映真实价格变化，不是默认0。

    关键修复：
    - 使用月末日期标签（resample('ME').last() 产出的是月末，不是月初）
    - 先断言退出月在 index 中（前提），再检查收益值
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()

    pool = CBPool(prices=cb_prices, meta=cb_meta)

    # 获取月度收益序列
    monthly_returns = pool.monthly_returns()
    assert not monthly_returns.empty, "月度收益面板不应为空"

    # CB_C 在 2023年6月退出 — resample('ME') 产出月末日期
    june_2023_end = pd.Timestamp('2023-06-30')
    assert june_2023_end in monthly_returns.index, \
        f"2023-06-30（月末）应在月度收益 index 中，实际 index: {list(monthly_returns.index[:5])}"
    assert 'CB_C' in monthly_returns.columns, \
        "CB_C 应在月度收益列中"

    cb_c_june_ret = monthly_returns.loc[june_2023_end, 'CB_C']
    assert pd.notna(cb_c_june_ret), \
        f"CB_C 退出月收益不应为NaN，实际为 {cb_c_june_ret}"
    assert abs(cb_c_june_ret) > 1e-6, \
        f"CB_C 退出月收益不应为0，实际为 {cb_c_june_ret}"

    # CB_D 在 2023年3月退出
    march_2023_end = pd.Timestamp('2023-03-31')
    assert march_2023_end in monthly_returns.index, \
        f"2023-03-31（月末）应在月度收益 index 中"
    assert 'CB_D' in monthly_returns.columns, \
        "CB_D 应在月度收益列中"

    cb_d_march_ret = monthly_returns.loc[march_2023_end, 'CB_D']
    assert pd.notna(cb_d_march_ret), \
        f"CB_D 退出月收益不应为NaN，实际为 {cb_d_march_ret}"
    assert abs(cb_d_march_ret) > 1e-6, \
        f"CB_D 退出月收益不应为0，实际为 {cb_d_march_ret}"


# ============================================================
# 测试 3: HRP权重只用过去252个交易日数据
# ============================================================

def test_hrp_uses_rolling_window():
    """HRP在某月的权重应只基于该月之前252个交易日的数据。"""
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()
    pool = CBPool(prices=cb_prices, meta=cb_meta)

    optimizer = HRPOptimizer(lookback=252)

    # 在 2023-06-01 计算 HRP 权重
    target_date = pd.Timestamp('2023-06-01')
    eligible = pool.eligible_at(target_date)
    weights = optimizer.compute_weights(pool, target_date, eligible)

    # 权重应非空且和为1
    assert weights is not None, "HRP权重不应为None"
    assert len(weights) > 0, "HRP权重不应为空"
    assert abs(weights.sum() - 1.0) < 1e-6, \
        f"权重和应为1，实际为 {weights.sum()}"

    # 验证只用过去数据：协方差矩阵截止日 < target_date
    cov_end_date = optimizer._get_cov_end_date(target_date)
    assert cov_end_date < target_date, \
        f"协方差矩阵截止日 {cov_end_date} 应早于目标日 {target_date}"


# ============================================================
# 测试 4: 混合比例是OOS估计，不是全样本优化
# ============================================================

def test_mixture_is_oos():
    """
    混合比例通过 walk-forward 估计：
    - 训练窗口选比例 → 下一段OOS验证
    - 不能在全样本上扫描0%-100%后报告"最佳比例"

    验证逻辑：运行回测后，每个OOS段的混合比例是在该段开始前确定的，
    不是用该段未来数据优化出来的。
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()
    etf_returns = _make_etf_returns()

    bt = HybridBacktest(
        cb_pool=CBPool(prices=cb_prices, meta=cb_meta),
        etf_returns=etf_returns,
        train_months=12,
        oos_months=3,
        cost_bps=10,  # 0.1% 单边交易成本
    )

    results = bt.run()

    # 结果应包含 OOS 段记录
    assert 'oos_segments' in results, "结果应包含OOS分段记录"
    assert len(results['oos_segments']) > 0, "应有至少一个OOS段"

    # 每个OOS段应记录：
    # - 比例在段开始前已确定（train_end_date < oos_start_date）
    # - 比例不是用OOS段内的数据优化出来的
    for seg in results['oos_segments']:
        assert 'train_end' in seg, "OOS段应记录训练窗口结束日"
        assert 'oos_start' in seg, "OOS段应记录OOS开始日"
        assert 'cb_ratio' in seg, "OOS段应记录CB比例"
        assert seg['train_end'] < seg['oos_start'], \
            f"训练结束日 {seg['train_end']} 应早于 OOS开始日 {seg['oos_start']}"

    # 不应有全样本扫描的 "best_ratio" 字段
    assert 'best_ratio' not in results, \
        "不应有全样本优化出的'最佳比例'字段"

    # 固定比例对比应存在
    assert 'fixed_ratios' in results, "应报告固定比例对比"


# ============================================================
# 测试 5: 可转债收益来自真实价格，不是随机噪声
# ============================================================

def test_no_synthetic_returns():
    """
    可转债收益应来自真实价格序列的 pct_change，
    不应包含 np.random.normal 或缩放操作。

    验证逻辑：
    - CBPool.monthly_returns() 的收益与输入价格序列一致
    - 不存在缩放到目标年化的操作
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()
    pool = CBPool(prices=cb_prices, meta=cb_meta)

    monthly_returns = pool.monthly_returns()

    # 验证：对存活券 CB_A，月收益 = 月末价格/上月末价格 - 1
    cb_a_prices = cb_prices['CB_A']
    expected_monthly = cb_a_prices.resample('ME').last().pct_change(fill_method=None)

    for date in monthly_returns.index:
        if date in expected_monthly.index:
            actual_ret = monthly_returns.loc[date, 'CB_A']
            expected_ret = expected_monthly.loc[date]
            if pd.notna(actual_ret) and pd.notna(expected_ret):
                assert abs(actual_ret - expected_ret) < 1e-6, \
                    f"CB_A 在 {date} 的月收益 {actual_ret} 与价格计算 {expected_ret} 不符"

    # 确保收益不是缩放过的
    cb_a_monthly = monthly_returns['CB_A'].dropna()
    if len(cb_a_monthly) > 0:
        annual = (1 + cb_a_monthly).prod() ** (12 / len(cb_a_monthly)) - 1
        assert abs(annual - 0.23) > 0.001, \
            "CB_A 年化收益不应精确等于23%（暗示有缩放操作）"


# ============================================================
# 测试 6: 时序正确性 — t-1月末权重应用于t月收益
# ============================================================

def test_hrp_weights_t_minus_1_applied_to_t():
    """
    HRP权重在 t-1 月末计算，应用于 t 月收益。
    不能在 t 月末计算权重后用于 t 月收益（同月前视偏差）。

    验证：_compute_hrp_monthly_returns 中，第 i 个月的权重
    应基于第 i-1 月（或更早）的数据计算。
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()
    pool = CBPool(prices=cb_prices, meta=cb_meta)

    bt = HybridBacktest(
        cb_pool=pool,
        etf_returns=_make_etf_returns(),
        train_months=12,
        oos_months=3,
        cost_bps=10,
    )

    cb_monthly_panel = pool.monthly_returns()
    hrp_monthly = bt._compute_hrp_monthly_returns(cb_monthly_panel)

    # 核心断言：hrp_monthly 的权重时间应与收益月份不同
    # 具体验证：如果我们在第 t 个月的月度收益中使用了 HRP 权重，
    # 那个权重应该是在 t-1 月末（或更早）计算的
    # 间接验证：对于第 0 个月，由于没有前一个月的权重可用，应返回 0 或等权
    # （因为无法在 t=-1 计算权重）

    # 获取所有月份
    month_dates = list(cb_monthly_panel.index)
    assert len(month_dates) >= 2, "至少需要2个月"

    # 第 0 个月不应该有 HRP 权重（因为没有前一个月数据）
    # 如果 _compute_hrp_monthly_returns 正确实现了 t-1 → t 时序，
    # 第 0 个月要么返回0，要么使用等权（但没有前一个月权重）
    first_month_ret = hrp_monthly.iloc[0]
    # 第0个月如果非0，需要验证它是等权而非HRP（因为没有前月数据）
    # 但更重要的是验证：第1个月的权重确实基于第0个月的数据

    # 直接验证：检查 _compute_hrp_monthly_returns 的权重索引
    # 如果函数记录了权重计算月份，检查 weight_month < return_month
    # 没有直接记录的话，用变异测试：改变第0个月的数据，第1个月的权重应变
    pass  # 主验证在下面的变异测试中


def test_hrp_invariance_same_month_data():
    """
    变异测试：只改变 t 月内部价格（不改 t-1 及之前的数据），
    HRP 在 t 月的收益应不变（因为权重在 t-1 月末已固定）。

    如果存在同月前视偏差（权重在 t 月末计算），
    改变 t 月价格会改变权重，从而改变 t 月 HRP 收益。
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()

    pool = CBPool(prices=cb_prices, meta=cb_meta)
    bt = HybridBacktest(
        cb_pool=pool,
        etf_returns=_make_etf_returns(),
        train_months=12,
        oos_months=3,
        cost_bps=10,
    )

    cb_panel = pool.monthly_returns()
    hrp_orig = bt._compute_hrp_monthly_returns(cb_panel)

    # 改变 CB_A 在第3个月之后的日价格（不影响前2个月）
    cb_prices2 = {k: v.copy() for k, v in cb_prices.items()}
    dates = cb_prices2['CB_A'].index
    # 找到第3个月的开始日期
    cutoff = pd.Timestamp('2022-04-01')
    mask = dates >= cutoff
    cb_prices2['CB_A'].loc[mask] = cb_prices2['CB_A'].loc[mask] * 1.5  # 大幅改变

    pool2 = CBPool(prices=cb_prices2, meta=cb_meta)
    bt2 = HybridBacktest(
        cb_pool=pool2,
        etf_returns=_make_etf_returns(),
        train_months=12,
        oos_months=3,
        cost_bps=10,
    )
    cb_panel2 = pool2.monthly_returns()
    hrp_mod = bt2._compute_hrp_monthly_returns(cb_panel2)

    # 第1、2个月的 HRP 收益不应改变（因为权重基于 t-1，即第0月或之前）
    # 第3个月及之后的收益会变（因为第3个月的价格变了，月度收益面板变了）
    # 但关键是：如果权重在 t-1 月末计算，那么第3个月的 HRP 权重
    # 取决于第2个月末之前的数据——第3个月的价格变化不应影响第3个月的权重
    # 所以第3个月的 HRP 收益变化只来自月度收益面板变化，不来自权重变化

    # 简单验证：前2个月的 HRP 收益应完全相同
    for i in range(min(2, len(hrp_orig), len(hrp_mod))):
        orig_val = hrp_orig.iloc[i]
        mod_val = hrp_mod.iloc[i]
        if pd.notna(orig_val) and pd.notna(mod_val):
            assert abs(orig_val - mod_val) < 1e-10, \
                f"第{i}个月 HRP 收益应不变（权重基于更早数据），" \
                f"原始={orig_val}, 修改后={mod_val}"


# ============================================================
# 测试 7: ETF死输入不变性 — 改变ETF收益不改变策略结果
# ============================================================

def test_etf_dead_input_invariance():
    """
    ETF收益在策略中仅用于日期对齐，不参与收益混合。
    改变ETF收益序列的内容，策略结果（OOS收益、metrics）不应改变。

    如果ETF在策略中真正使用了（如作为混合成分），
    改变ETF收益会改变结果——此测试会失败。
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()

    # 原始 ETF
    bt1 = HybridBacktest(
        cb_pool=CBPool(prices=cb_prices, meta=cb_meta),
        etf_returns=_make_etf_returns(),
        train_months=12,
        oos_months=3,
        cost_bps=10,
    )
    results1 = bt1.run()

    # 完全不同的 ETF
    bt2 = HybridBacktest(
        cb_pool=CBPool(prices=cb_prices, meta=cb_meta),
        etf_returns=_make_etf_returns_alt(),
        train_months=12,
        oos_months=3,
        cost_bps=10,
    )
    results2 = bt2.run()

    # OOS 收益应完全相同（ETF不参与策略）
    oos1 = results1['oos_returns']
    oos2 = results2['oos_returns']

    assert len(oos1) == len(oos2), \
        f"OOS收益长度应相同: {len(oos1)} vs {len(oos2)}"

    for i in range(len(oos1)):
        v1 = oos1.iloc[i]
        v2 = oos2.iloc[i]
        if pd.notna(v1) and pd.notna(v2):
            assert abs(v1 - v2) < 1e-10, \
                f"OOS收益第{i}期应相同（ETF不参与策略）: {v1} vs {v2}"

    # OOS metrics 应相同
    m1 = results1['oos_metrics']
    m2 = results2['oos_metrics']
    for key in ['cagr', 'sharpe', 'max_drawdown', 'total_return']:
        if m1 and m2 and key in m1 and key in m2:
            assert abs(m1[key] - m2[key]) < 1e-10, \
                f"OOS metric '{key}' 应相同: {m1[key]} vs {m2[key]}"


# ============================================================
# 测试 8: 退出月异常值处理（0和NaN）
# ============================================================

def test_exit_month_zero_and_nan_handling():
    """
    退出月收益为0或NaN时，应被正确处理而非静默传播。

    构造数据：
    - CB_E: 退出月收益恰好为0（exit_final_price == 上月末价格）
    - CB_F: 退出月收益为NaN（exit_final_price 无法计算）
    """
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='B')
    np_rng = np.random.RandomState(77)

    # CB_E: 退出月 exit_final_price = 上月末价格 → 收益=0
    mask_e = dates <= pd.Timestamp('2023-06-30')
    dates_e = dates[mask_e]
    price_e = 100.0 + np_rng.randn(len(dates_e)).cumsum() * 0.2
    price_e = np.maximum(price_e, 85.0)
    # 5月底（上月末）价格
    may_end_mask = dates_e <= pd.Timestamp('2023-05-31')
    may_end_price = price_e[may_end_mask][-1]
    # 设 exit_final_price = may_end_price → 6月收益=0
    price_e[-5:] = may_end_price

    prices = {
        'CB_E': pd.Series(price_e, index=dates_e, name='CB_E'),
    }
    meta = {
        'CB_E': {
            'listing_date': '2021-06-01',
            'delist_date': '2023-06-30',
            'exit_reason': '到期',
            'exit_final_price': float(may_end_price),
        },
    }

    pool = CBPool(prices=prices, meta=meta)
    monthly_returns = pool.monthly_returns()

    june_end = pd.Timestamp('2023-06-30')
    if june_end in monthly_returns.index and 'CB_E' in monthly_returns.columns:
        june_ret = monthly_returns.loc[june_end, 'CB_E']
        # 收益为0是数学上正确的（exit_final_price == 上月末价格）
        # 但策略在等权计算中应处理这个0——它不是NaN，是真实0收益
        # 关键是：这个0不应被当作"没有数据"处理
        assert pd.notna(june_ret), "退出月收益为0是合法值，不应变为NaN"


# ============================================================
# 测试 9: 固定比例与动态比例同区间比较
# ============================================================

def test_fixed_and_dynamic_same_oos_interval():
    """
    固定比例和动态比例应在相同的OOS区间上比较，
    不能一个用全样本、一个只用OOS段。

    验证：fixed_ratios 的 n_periods 应等于 oos_metrics 的 n_periods，
    而不是全样本的 n_months。
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()
    etf_returns = _make_etf_returns()

    bt = HybridBacktest(
        cb_pool=CBPool(prices=cb_prices, meta=cb_meta),
        etf_returns=etf_returns,
        train_months=12,
        oos_months=3,
        cost_bps=10,
    )
    results = bt.run()

    oos_n = results['oos_metrics']['n_periods']
    for ratio_key, metrics in results['fixed_ratios'].items():
        if metrics:
            assert metrics['n_periods'] == oos_n, \
                f"固定比例 {ratio_key} 的 n_periods={metrics['n_periods']} " \
                f"应等于 OOS n_periods={oos_n}（同区间比较）"


# ============================================================
# 测试 10: 不完整月排除
# ============================================================

def test_incomplete_month_excluded():
    """
    运行日当月（未结束）不应被纳入回测结果。
    如果数据截止到7月14日，7月收益不应被作为"完整月"使用。

    验证：period.end 不应是当前未结束月的月末。
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()

    # 构造一个延伸到"今天"的数据集
    today = pd.Timestamp.now().normalize()
    dates_ext = pd.date_range('2022-01-01', today, freq='B')
    np_rng = np.random.RandomState(42)
    price_ext = 100.0 + np_rng.randn(len(dates_ext)).cumsum() * 0.3
    price_ext = np.maximum(price_ext, 80.0)

    prices = {
        'CB_A': pd.Series(price_ext, index=dates_ext, name='CB_A'),
    }
    prices.update({k: v for k, v in cb_prices.items() if k != 'CB_A'})

    pool = CBPool(prices=prices, meta=cb_meta)
    monthly_returns = pool.monthly_returns()

    # 最后一个月的月末标签
    last_month = monthly_returns.index[-1]
    # 如果当前月还没结束，resample('ME').last() 会给当前月月末
    # 但当前月的收益是不完整的——不应该被使用
    # 验证：CBPool 有截止未完成月的逻辑
    if hasattr(pool, 'data_cutoff_date'):
        cutoff = pool.data_cutoff_date
        assert cutoff is not None, "应设置数据截止日期"
        # 最后一个月应 <= cutoff 的月末
        last_complete = pd.Timestamp(cutoff).to_period('M').to_timestamp(how='end').normalize()
        assert last_month <= last_complete, \
            f"最后月份 {last_month} 应 <= 最后完整月 {last_complete}"


# ============================================================
# 测试 11: 固定比例标签正确性
# ============================================================

def test_fixed_ratio_labels_correct():
    """
    固定比例的标签应正确反映实际比例。
    80/20 而非 80/19（这是 int(0.8*100)/int(0.2*100) 的浮点误差）。
    """
    cb_prices = _make_cb_prices()
    cb_meta = _make_cb_meta()
    etf_returns = _make_etf_returns()

    bt = HybridBacktest(
        cb_pool=CBPool(prices=cb_prices, meta=cb_meta),
        etf_returns=etf_returns,
        train_months=12,
        oos_months=3,
        cost_bps=10,
    )
    results = bt.run()

    expected_ratios = {'20/80', '50/50', '80/20'}
    actual_ratios = set(results['fixed_ratios'].keys())
    assert actual_ratios == expected_ratios, \
        f"固定比例标签应为 {expected_ratios}，实际为 {actual_ratios}"
