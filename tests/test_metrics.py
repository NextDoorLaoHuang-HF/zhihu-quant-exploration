"""
tests/test_metrics.py
TDD 测试用例 — 对应 scripts/lib/metrics.py

运行方式（从项目根目录）：
    pytest tests/test_metrics.py -v

或从 scripts/ 目录：
    cd scripts && python -m pytest ../tests/test_metrics.py -v
"""

import sys
import os
import math
import pytest
import pandas as pd
import numpy as np

# 让 tests/ 能找到 scripts/lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lib.metrics import compute_metrics, relative_cagr


# ── 辅助 ──────────────────────────────────────────────

def make_series(returns, start='2020-01-01', freq='MS'):
    """把 list 变成带 DatetimeIndex 的 pd.Series"""
    idx = pd.date_range(start=start, periods=len(returns), freq=freq)
    return pd.Series(returns, index=idx)


# ── 测试 1: CAGR 匹配复利 ────────────────────────────

def test_cagr_matches_compound():
    """月收益 [0.01]*12 的 CAGR 应为 (1.01^12 - 1) ≈ 12.682%，不是 12%"""
    rets = make_series([0.01] * 12)
    m = compute_metrics(rets)
    # CAGR = nav_final^(1/years) - 1，years 按日历日计算
    nav_final = 1.01 ** 12
    years = (rets.index[-1] - rets.index[0]).days / 365.2425
    expected_cagr = nav_final ** (1.0 / years) - 1
    assert abs(m['cagr'] - expected_cagr) < 1e-10, \
        f"CAGR {m['cagr']:.6%} != 复利 {expected_cagr:.6%}"
    # 算术年化 = 12% （对比用）
    assert abs(m['annualized_mean'] - 0.12) < 1e-10
    # CAGR != 算术年化
    assert abs(m['cagr'] - m['annualized_mean']) > 1e-6


# ── 测试 2: 大回撤序列两种年化差异显著 ───────────────

def test_arithmetic_mean_differs_from_cagr():
    """[0.1, -0.5, 0.1, -0.5] 算术年化 vs CAGR 差异巨大"""
    rets = make_series([0.1, -0.5, 0.1, -0.5])
    m = compute_metrics(rets)
    # 算术年化 = mean * 12 = (-0.2) * 12 = -2.4 = -240%
    assert abs(m['annualized_mean'] - (-2.4)) < 1e-10
    # CAGR: nav = 1.1 * 0.5 * 1.1 * 0.5 = 0.3025
    # years 按日历日计算（4个月 ≈ 0.334年）
    nav_final = 1.1 * 0.5 * 1.1 * 0.5
    years = (rets.index[-1] - rets.index[0]).days / 365.2425
    expected_cagr = nav_final ** (1.0 / years) - 1
    assert abs(m['cagr'] - expected_cagr) < 1e-6
    # 差异显著
    assert abs(m['cagr'] - m['annualized_mean']) > 1.0


# ── 测试 3: 最大回撤正确 ─────────────────────────────

def test_max_drawdown_correct():
    """
    [0.1, -0.2, 0.05] 的净值：
      t0: 1.0  → 1.1
      t1: 1.1  → 0.88  (回撤 0.88/1.1 - 1 = -20%)
      t2: 0.88 → 0.924
    最大回撤 = -20%，不是 -25%
    """
    rets = make_series([0.1, -0.2, 0.05])
    m = compute_metrics(rets)
    assert abs(m['max_drawdown'] - (-0.20)) < 1e-6, \
        f"max_drawdown {m['max_drawdown']:.4f} != -0.20"
    # -0.25 是错误算法（0.8/1.0 - 1）的结果
    assert abs(m['max_drawdown'] - (-0.25)) > 1e-6


# ── 测试 4: 单期收益 < -100% 触发 ValueError ──────────

def test_no_return_below_neg100():
    """任何单期收益 < -1 应触发 ValueError"""
    rets = make_series([0.05, -1.5, 0.02])
    with pytest.raises(ValueError, match="<-100%|< -1"):
        compute_metrics(rets)


# ── 测试 5: relative_cagr ─────────────────────────────

def test_relative_cagr():
    """策略翻倍、基准不动时，相对 CAGR = 100%"""
    idx = pd.date_range('2020-01-01', periods=25, freq='MS')  # 2年
    # 基准不动：净值恒为 1.0
    benchmark = pd.Series([1.0] * 25, index=idx)
    # 策略翻倍：从 1.0 线性涨到 2.0
    strategy = pd.Series(np.linspace(1.0, 2.0, 25), index=idx)
    rel = relative_cagr(strategy, benchmark)
    # 策略从1.0涨到2.0，基准不动 → 相对净值也从1.0到2.0
    # CAGR = 2^(1/years) - 1，years 按日历日计算
    years = (strategy.index[-1] - strategy.index[0]).days / 365.2425
    expected = 2.0 ** (1.0 / years) - 1
    assert abs(rel - expected) < 1e-6, f"relative_cagr {rel:.6f} != {expected:.6f}"


# ── 测试 6: 单期不崩溃 ───────────────────────────────

def test_single_period():
    """只有一个数据点不崩溃"""
    rets = make_series([0.05])
    m = compute_metrics(rets)
    assert m['n_periods'] == 1
    assert m['cagr'] == pytest.approx(0.05)  # 单期：cagr = total_return
    assert m['max_drawdown'] == 0.0  # 单期无回撤


# ── 测试 7: 空序列 ────────────────────────────────────

def test_empty_series():
    """空序列应抛清晰异常"""
    rets = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="不能为空"):
        compute_metrics(rets)
