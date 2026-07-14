"""
统一绩效指标计算模块。

所有回测脚本应使用此模块代替各自 mean()*12 的算术年化，
以消除算术平均与复合年化（CAGR）的混淆。

依赖：numpy, pandas（无第三方引入）
"""

import numpy as np
import pandas as pd


def compute_metrics(returns: pd.Series, periods_per_year: int = 12, rf: float = 0.0) -> dict:
    """
    统一绩效指标。returns 为周期收益序列（如月收益）。

    返回 dict 包含：
      cagr            — 复合年化收益率（真实几何年化）
      annualized_mean — 算术年化（均值 * periods_per_year），保留供对比
      total_return    — 累计总收益率
      annualized_vol  — 年化波动率
      sharpe          — 夏普比率（基于算术年化）
      max_drawdown    — 最大回撤（基于复合净值）
      calmar          — 卡玛比率（CAGR / |最大回撤|）
      n_periods       — 收益期数
      years           — 覆盖年数（按日历日 / 365.2425）
    """
    if returns is None or len(returns) == 0:
        raise ValueError("returns 序列不能为空")

    if (returns < -1.0).any():
        raise ValueError("存在单期收益 < -100%（即本金亏损超过全部），数据有误")

    returns = returns.astype(float)

    # 复合净值曲线
    nav = (1 + returns).cumprod()

    # 日历年数
    if isinstance(returns.index, pd.DatetimeIndex):
        if len(returns) >= 2:
            years = (returns.index[-1] - returns.index[0]).days / 365.2425
        else:
            years = 0.0
    else:
        # 非 DatetimeIndex 时按期数推算
        years = len(returns) / periods_per_year if periods_per_year > 0 else 0.0

    total_return = float(nav.iloc[-1] - 1)

    if years > 0:
        cagr = float(nav.iloc[-1] ** (1.0 / years) - 1)
    else:
        cagr = total_return  # 单期：直接用总收益

    annualized_mean = float(returns.mean() * periods_per_year)
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float((annualized_mean - rf) / annualized_vol) if annualized_vol > 0 else 0.0

    max_drawdown = float((nav / nav.cummax() - 1).min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    return {
        'cagr': cagr,
        'annualized_mean': annualized_mean,
        'total_return': total_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'n_periods': int(len(returns)),
        'years': float(years),
    }


def relative_cagr(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> float:
    """
    相对基准的复合年化超额。

    strategy_nav / benchmark_nav 形成相对净值曲线，
    取其复合年化收益率。正值表示策略跑赢基准。
    """
    if strategy_nav is None or benchmark_nav is None:
        raise ValueError("strategy_nav 和 benchmark_nav 不能为 None")
    if len(strategy_nav) == 0 or len(benchmark_nav) == 0:
        raise ValueError("净值序列不能为空")

    # 对齐索引
    common = strategy_nav.index.intersection(benchmark_nav.index)
    if len(common) == 0:
        raise ValueError("策略与基准无重叠时间区间")

    s = strategy_nav.loc[common]
    b = benchmark_nav.loc[common]

    relative = s / b

    if isinstance(relative.index, pd.DatetimeIndex):
        if len(relative) >= 2:
            years = (relative.index[-1] - relative.index[0]).days / 365.2425
        else:
            years = 0.0
    else:
        years = len(relative) / 12.0

    if years > 0:
        return float(relative.iloc[-1] ** (1.0 / years) - 1)
    else:
        return float(relative.iloc[-1] - 1)
