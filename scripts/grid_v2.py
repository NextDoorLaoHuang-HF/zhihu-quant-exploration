"""
统一网格交易回测 — 替换 6 个受污染脚本的数字

调用 lib/grid_engine.py 的 GridEngine，不再自己写交易循环。
输出 results/grid_v2/grid.json，包含四部分：
  1. scan_results    — 6品种 × 4网格 × 3底仓 = 72组参数扫描
  2. cost_sensitivity — 4档佣金下胜率变化
  3. walkforward     — 滚动窗口优化（继承资金和持仓）
  4. overfit_check   — 两段分割检验

数据源：
  - 5个品种：akshare fund_etf_hist_sina（新浪，未复权但有>40%除权检测）
  - 半导体 sh512480：腾讯前复权接口（修复>40%误判分拆/合并的bug）

所有年化数字使用 CAGR（几何年化），不用算术平均。
"""
import os
import sys
import json
import math
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# 让 import 能找到 lib 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np
import pandas as pd
import requests
import akshare as ak
from grid_engine import GridConfig, GridEngine
from metrics import compute_metrics, relative_cagr

# ============================================================
# 配置
# ============================================================
INITIAL_CAPITAL = 100_000
GRID_CAPITAL = INITIAL_CAPITAL * 0.05  # 每格5000元
GRID_PARAMS = [0.03, 0.05, 0.08, 0.10]
BASE_POSITIONS = [0.5, 0.6, 0.7]
START_DATE = '2019-01-01'

SYMBOLS = {
    '创业板': 'sz159915',
    '半导体': 'sh512480',
    '证券': 'sh512880',
    '中证1000': 'sh512100',
    '科创50': 'sh588000',
    '纳指': 'sh513100',
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'grid_v2')

# 成本敏感性：万2.5 → 万2.0 单边
COST_SCENARIOS = {
    '万2.5佣金': 0.00025,
    '万2.0佣金': 0.00020,
    '万3.0佣金': 0.00030,
    '千1.0全成本': 0.001,
}

# walk-forward 交易年份
WF_TRADE_YEARS = [2021, 2022, 2023, 2024, 2025]

# 过拟合检验区间
OVERFIT_PERIODS = {
    '2019-2022': ('2019-01-01', '2022-06-30'),
    '2022-2026': ('2022-07-01', '2026-12-31'),
}


# ============================================================
# 数据获取
# ============================================================
def get_etf_sina(code: str, start: str = START_DATE) -> pd.Series:
    """新浪源 ETF 日线（带 >40% 除权检测，非半导体品种用）。"""
    df = ak.fund_etf_hist_sina(symbol=code)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start].sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    # 复权检测：单日涨跌>40%视为除权，向前调整
    df['ret'] = df['close'].pct_change(fill_method=None)
    split_mask = df['ret'].abs() > 0.40
    if split_mask.any():
        for sd in df[split_mask].index:
            idx = df.index.get_loc(sd)
            if idx > 0:
                ratio = df.iloc[idx]['close'] / df.iloc[idx - 1]['close']
                df.iloc[:idx, df.columns.get_loc('close')] *= ratio
    return df['close']


def get_etf_tencent_qfq(code: str, start: str = START_DATE) -> pd.Series:
    """腾讯前复权 ETF 日线（半导体 sh512480 专用，修复 >40% 误判）。

    腾讯接口每次最多返回 640 条，需要按时间段分页拉取。
    """
    url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    all_klines = []
    seen_dates = set()

    # 分段拉取：每段约1.5年
    segments = [
        ('2019-01-01', '2021-06-30'),
        ('2021-06-01', '2023-06-30'),
        ('2023-06-01', '2025-06-30'),
        ('2025-06-01', '2026-12-31'),
    ]

    for seg_start, seg_end in segments:
        params = {'param': f'{code},day,{seg_start},{seg_end},640,qfq'}
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            klines = data.get('data', {}).get(code, {}).get('qfqday', [])
            if klines:
                for k in klines:
                    d = k[0]
                    if d not in seen_dates:
                        all_klines.append(k)
                        seen_dates.add(d)
        except Exception as e:
            print(f'  ⚠️ 腾讯接口 {code} {seg_start}~{seg_end} 失败: {e}')
        time.sleep(0.3)  # 礼貌限速

    if not all_klines:
        raise RuntimeError(f'腾讯前复权接口无数据: {code}')

    df = pd.DataFrame(all_klines, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].astype(float)
    df = df[df['date'] >= pd.Timestamp(start)]
    df = df.sort_values('date').drop_duplicates(subset='date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    return df['close']


def get_all_etf_data() -> dict:
    """获取全部6个品种的收盘价序列。"""
    prices = {}
    for name, code in SYMBOLS.items():
        print(f'  获取 {name} ({code})...', end=' ')
        try:
            if code == 'sh512480':
                p = get_etf_tencent_qfq(code)
                # 验证：不应有>40%的单日波动
                rets = p.pct_change(fill_method=None)
                big = rets[rets.abs() > 0.40]
                print(f'腾讯qfq {len(p)}天, >40%波动{len(big)}天')
            else:
                p = get_etf_sina(code)
                rets = p.pct_change(fill_method=None)
                big = rets[rets.abs() > 0.40]
                print(f'新浪 {len(p)}天, >40%波动{len(big)}天(已复权)')
            prices[name] = p
        except Exception as e:
            print(f'失败: {e}')
    return prices


# ============================================================
# 1. 参数扫描
# ============================================================
def run_param_scan(prices_dict: dict) -> list:
    """72组参数扫描：6品种 × 4网格 × 3底仓。"""
    print('\n' + '=' * 70)
    print('1. 参数扫描：6品种 × 4网格 × 3底仓 = 72组')
    print('=' * 70)

    results = []
    for name, p in prices_dict.items():
        for gp in GRID_PARAMS:
            for bp in BASE_POSITIONS:
                config = GridConfig(
                    grid_pct=gp,
                    base_position=bp,
                    grid_capital=GRID_CAPITAL,
                )
                engine = GridEngine(config, initial_capital=INITIAL_CAPITAL)
                result = engine.run(p)

                results.append({
                    'symbol': name,
                    'code': SYMBOLS[name],
                    'grid_pct': gp,
                    'base_position': bp,
                    'grid_cagr': round(result.grid_annual_return, 6),
                    'bh_cagr': round(result.bh_annual_return, 6),
                    'base_bm_cagr': round(result.base_benchmark_annual_return, 6),
                    'excess_vs_bh': round(result.excess_return, 6),
                    'excess_vs_base': round(result.grid_excess_vs_base, 6),
                    'trades': result.trades,
                    'final_value': round(result.final_value, 2),
                    'n_days': len(p),
                    'start_date': str(p.index[0].date()),
                    'end_date': str(p.index[-1].date()),
                })

    # 汇总
    win_count = sum(1 for r in results if r['excess_vs_bh'] > 0)
    total = len(results)
    print(f'\n  胜率: {win_count}/{total} = {win_count/total:.0%}')

    print(f"\n  {'品种':8s} {'网格':>4s} {'底仓':>4s} | {'网格CAGR':>8s} {'买持CAGR':>8s} {'底仓基准':>8s} | {'超额vs买持':>10s} {'超额vs底仓':>10s} {'交易':>4s}")
    print(f"  {'-'*85}")
    for r in results:
        if r['excess_vs_bh'] > 0:
            print(f"  {r['symbol']:8s} {r['grid_pct']:>4.0%} {r['base_position']:>4.0%} | "
                  f"{r['grid_cagr']:>8.1%} {r['bh_cagr']:>8.1%} {r['base_bm_cagr']:>8.1%} | "
                  f"{r['excess_vs_bh']:>+10.1%} {r['excess_vs_base']:>+10.1%} {r['trades']:>4d}")

    return results


# ============================================================
# 2. 成本敏感性
# ============================================================
def run_cost_sensitivity(prices_dict: dict) -> list:
    """4档佣金下的胜率变化。"""
    print('\n' + '=' * 70)
    print('2. 成本敏感性：4档佣金')
    print('=' * 70)

    all_results = []
    for cost_label, cost_rate in COST_SCENARIOS.items():
        win_count = 0
        total = 0
        detail = []
        for name, p in prices_dict.items():
            for gp in GRID_PARAMS:
                for bp in BASE_POSITIONS:
                    config = GridConfig(
                        grid_pct=gp,
                        base_position=bp,
                        grid_capital=GRID_CAPITAL,
                        commission_rate=cost_rate,
                    )
                    engine = GridEngine(config, initial_capital=INITIAL_CAPITAL)
                    result = engine.run(p)
                    excess = result.excess_return
                    total += 1
                    if excess > 0:
                        win_count += 1
                    detail.append({
                        'symbol': name,
                        'grid_pct': gp,
                        'base_position': bp,
                        'excess_vs_bh': round(excess, 6),
                        'trades': result.trades,
                    })
        win_rate = win_count / total if total > 0 else 0
        print(f'  {cost_label:12s} (单边{cost_rate:.4%}): 胜率 {win_count}/{total} = {win_rate:.0%}')
        all_results.append({
            'cost_label': cost_label,
            'cost_rate': cost_rate,
            'win_count': win_count,
            'total': total,
            'win_rate': round(win_rate, 4),
            'detail': detail,
        })
    return all_results


# ============================================================
# 3. Walk-Forward（滚动窗口优化，继承资金和持仓）
# ============================================================
def run_walkforward(prices_dict: dict) -> list:
    """滚动窗口优化：每年用历史数据选最优参数，当年执行。

    关键：交易年间继承资金和持仓（不重置）。
    """
    print('\n' + '=' * 70)
    print('3. Walk-Forward：滚动窗口优化（继承资金和持仓）')
    print('=' * 70)

    wf_results = []

    for name, p in prices_dict.items():
        print(f'\n  --- {name} ---')
        year_results = []
        cumulative_grid = 1.0  # 累计净值
        cumulative_bh = 1.0

        for year in WF_TRADE_YEARS:
            opt_end = f'{year - 1}-12-31'
            opt_start = START_DATE
            trade_start = f'{year}-01-01'
            trade_end = f'{year}-12-31'

            p_opt = p.loc[opt_start:opt_end]
            p_trade = p.loc[trade_start:trade_end]

            if len(p_opt) < 200 or len(p_trade) < 100:
                continue

            # 优化期：找最优参数
            best_excess = -999
            best_params = None
            for gp in GRID_PARAMS:
                for bp in BASE_POSITIONS:
                    config = GridConfig(
                        grid_pct=gp,
                        base_position=bp,
                        grid_capital=GRID_CAPITAL,
                    )
                    engine = GridEngine(config, initial_capital=INITIAL_CAPITAL)
                    result = engine.run(p_opt)
                    if result.excess_return > best_excess:
                        best_excess = result.excess_return
                        best_params = (gp, bp)

            # 交易期：用最优参数交易，继承上一年的资金和持仓
            config = GridConfig(
                grid_pct=best_params[0],
                base_position=best_params[1],
                grid_capital=GRID_CAPITAL,
            )

            if not year_results:
                # 第一年：从头开始
                engine = GridEngine(config, initial_capital=INITIAL_CAPITAL)
            else:
                # 后续年：继承上一年的状态
                prev = year_results[-1]
                engine = GridEngine(config, initial_capital=prev['final_value'])
                engine.cash = prev['end_cash']
                engine.shares = prev['end_shares']
                engine.current_grid = prev['end_grid']
                engine.grid_base = prev['end_grid_base']
                engine._initialized = True

            result = engine.run(p_trade)

            # 买持基准（当年独立计算，不继承 — 买持没有状态）
            bh_ann = (p_trade.iloc[-1] / p_trade.iloc[0]) ** (
                1 / ((p_trade.index[-1] - p_trade.index[0]).days / 365.25)
            ) - 1

            # 网格当年收益率
            grid_year_return = result.final_value / engine.initial_capital - 1
            # 注意：继承年份的 initial_capital 是上一年的 final_value

            cumulative_grid *= (1 + grid_year_return)
            cumulative_bh *= (1 + bh_ann)

            yr = {
                'year': year,
                'symbol': name,
                'best_grid_pct': best_params[0],
                'best_base_position': best_params[1],
                'opt_excess': round(best_excess, 6),
                'trade_grid_return': round(grid_year_return, 6),
                'trade_bh_return': round(bh_ann, 6),
                'trade_excess': round(grid_year_return - bh_ann, 6),
                'final_value': round(result.final_value, 2),
                'trades': result.trades,
                'end_cash': round(engine.cash, 2),
                'end_shares': round(engine.shares, 4),
                'end_grid': engine.current_grid,
                'end_grid_base': round(engine.grid_base, 6),
                'inherited': len(year_results) > 0,
            }
            year_results.append(yr)
            print(f'    {year}: 优化期最优={best_params[0]:.0%}/{best_params[1]:.0%} → '
                  f'当年网格{grid_year_return:+.1%} vs 买持{bh_ann:+.1%} '
                  f'(超额{grid_year_return-bh_ann:+.1%}, {"继承" if yr["inherited"] else "新启"})')

        if year_results:
            years_count = len(year_results)
            wf_grid_ann = cumulative_grid ** (1 / years_count) - 1
            wf_bh_ann = cumulative_bh ** (1 / years_count) - 1
            print(f'    → 滚动优化年化: 网格{wf_grid_ann:.1%} vs 买持{wf_bh_ann:.1%} '
                  f'(超额{wf_grid_ann - wf_bh_ann:+.1%})')
            wf_results.append({
                'symbol': name,
                'years': year_results,
                'cumulative_grid_cagr': round(wf_grid_ann, 6),
                'cumulative_bh_cagr': round(wf_bh_ann, 6),
                'cumulative_excess': round(wf_grid_ann - wf_bh_ann, 6),
            })

    return wf_results


# ============================================================
# 4. 过拟合检验
# ============================================================
def run_overfit_check(prices_dict: dict) -> dict:
    """两段分割检验：2019-2022 vs 2022-2026。"""
    print('\n' + '=' * 70)
    print('4. 过拟合检验：两段分割')
    print('=' * 70)

    all_segment_results = []

    for name, p in prices_dict.items():
        for period_name, (start, end) in OVERFIT_PERIODS.items():
            p_seg = p.loc[start:end]
            if len(p_seg) < 100:
                continue
            for gp in GRID_PARAMS:
                for bp in BASE_POSITIONS:
                    config = GridConfig(
                        grid_pct=gp,
                        base_position=bp,
                        grid_capital=GRID_CAPITAL,
                    )
                    engine = GridEngine(config, initial_capital=INITIAL_CAPITAL)
                    result = engine.run(p_seg)
                    all_segment_results.append({
                        'symbol': name,
                        'period': period_name,
                        'grid_pct': gp,
                        'base_position': bp,
                        'excess_vs_bh': round(result.excess_return, 6),
                        'grid_cagr': round(result.grid_annual_return, 6),
                        'bh_cagr': round(result.bh_annual_return, 6),
                    })

    df = pd.DataFrame(all_segment_results)

    # 1. 各品种两段最优参数对比
    print('\n  === 各品种两段最优参数对比 ===')
    best_params_comparison = []
    for name in SYMBOLS:
        if name not in df['symbol'].values:
            continue
        sub1 = df[(df['symbol'] == name) & (df['period'] == '2019-2022')]
        sub2 = df[(df['symbol'] == name) & (df['period'] == '2022-2026')]
        if len(sub1) == 0 or len(sub2) == 0:
            continue
        best1 = sub1.loc[sub1['excess_vs_bh'].idxmax()]
        best2 = sub2.loc[sub2['excess_vs_bh'].idxmax()]
        same = (best1['grid_pct'] == best2['grid_pct'] and best1['base_position'] == best2['base_position'])
        marker = '✅一致' if same else '❌不同'
        print(f"  {name:8s} | 前半段: 网格{best1['grid_pct']:.0%}/{best1['base_position']:.0%} "
              f"超额{best1['excess_vs_bh']:+.1%} | 后半段: 网格{best2['grid_pct']:.0%}/{best2['base_position']:.0%} "
              f"超额{best2['excess_vs_bh']:+.1%} | {marker}")
        best_params_comparison.append({
            'symbol': name,
            'p1_grid_pct': best1['grid_pct'],
            'p1_base_position': best1['base_position'],
            'p1_excess': float(best1['excess_vs_bh']),
            'p2_grid_pct': best2['grid_pct'],
            'p2_base_position': best2['base_position'],
            'p2_excess': float(best2['excess_vs_bh']),
            'same_params': bool(same),
        })

    # 2. 全样本最优参数的分段稳定性
    print('\n  === 全样本最优参数的分段稳定性 ===')
    stability = []
    full_best = df.groupby(['symbol', 'grid_pct', 'base_position'])['excess_vs_bh'].mean().reset_index()
    for name in SYMBOLS:
        if name not in full_best['symbol'].values:
            continue
        sub = full_best[full_best['symbol'] == name]
        best = sub.loc[sub['excess_vs_bh'].idxmax()]
        r1 = df[(df['symbol'] == name) & (df['grid_pct'] == best['grid_pct']) &
                (df['base_position'] == best['base_position']) & (df['period'] == '2019-2022')]
        r2 = df[(df['symbol'] == name) & (df['grid_pct'] == best['grid_pct']) &
                (df['base_position'] == best['base_position']) & (df['period'] == '2022-2026')]
        e1 = float(r1['excess_vs_bh'].values[0]) if len(r1) > 0 else None
        e2 = float(r2['excess_vs_bh'].values[0]) if len(r2) > 0 else None
        if e1 is not None and e2 is not None:
            stable = (e1 > 0 and e2 > 0)
            marker = '✅稳定' if stable else '⚠️不稳定'
            print(f"  {name:8s} 全样本最优: 网格{best['grid_pct']:.0%}/{best['base_position']:.0%} "
                  f"| 前半段{e1:+.1%} 后半段{e2:+.1%} → {marker}")
            stability.append({
                'symbol': name,
                'best_grid_pct': best['grid_pct'],
                'best_base_position': best['base_position'],
                'p1_excess': e1,
                'p2_excess': e2,
                'stable': stable,
            })

    # 3. 胜率统计
    p1_wins = sum(1 for r in all_segment_results if r['period'] == '2019-2022' and r['excess_vs_bh'] > 0)
    p1_total = sum(1 for r in all_segment_results if r['period'] == '2019-2022')
    p2_wins = sum(1 for r in all_segment_results if r['period'] == '2022-2026' and r['excess_vs_bh'] > 0)
    p2_total = sum(1 for r in all_segment_results if r['period'] == '2022-2026')

    return {
        'segment_results': all_segment_results,
        'best_params_comparison': best_params_comparison,
        'stability_check': stability,
        'p1_win_rate': round(p1_wins / p1_total, 4) if p1_total > 0 else 0,
        'p2_win_rate': round(p2_wins / p2_total, 4) if p2_total > 0 else 0,
    }


# ============================================================
# 资金守恒验证
# ============================================================
def verify_capital_conservation(prices_dict: dict) -> dict:
    """验证网格引擎资金守恒：每笔交易 cash 变化 + shares*price 变化 = 0（扣除手续费）。"""
    print('\n' + '=' * 70)
    print('资金守恒验证')
    print('=' * 70)

    # 用创业板数据做验证
    p = list(prices_dict.values())[0]
    config = GridConfig(grid_pct=0.05, base_position=0.6, grid_capital=GRID_CAPITAL)
    engine = GridEngine(config, initial_capital=INITIAL_CAPITAL)
    result = engine.run(p)

    # 检查首尾净值
    initial_pv = result.grid_pv.iloc[0]
    final_pv = result.grid_pv.iloc[-1]
    print(f'  初始净值: {initial_pv:.2f}')
    print(f'  期末净值: {final_pv:.2f}')
    print(f'  现金+持仓: {engine.cash:.2f} + {engine.shares * p.iloc[-1]:.2f} = '
          f'{engine.cash + engine.shares * p.iloc[-1]:.2f}')
    print(f'  final_value: {result.final_value:.2f}')

    # 买持基准验证
    bh_shares = INITIAL_CAPITAL / p.iloc[0]
    expected_bh_final = bh_shares * p.iloc[-1]
    print(f'  买持基准: {result.bh_pv.iloc[-1]:.2f} (expected {expected_bh_final:.2f})')

    # 底仓基准验证
    base_shares = engine.config.base_position * INITIAL_CAPITAL / p.iloc[0]
    # 取整（contract_size=100）
    base_shares_int = int(base_shares // 100) * 100
    base_cash = INITIAL_CAPITAL - base_shares_int * p.iloc[0]
    expected_base_bm_final = base_cash + base_shares_int * p.iloc[-1]
    print(f'  底仓基准: {result.base_benchmark_pv.iloc[-1]:.2f} (expected {expected_base_bm_final:.2f})')

    ok = abs(final_pv - (engine.cash + engine.shares * p.iloc[-1])) < 1.0
    print(f'  资金守恒: {"✅通过" if ok else "❌失败"}')

    return {
        'initial_pv': round(float(initial_pv), 2),
        'final_pv': round(float(final_pv), 2),
        'cash_plus_position': round(float(engine.cash + engine.shares * p.iloc[-1]), 2),
        'bh_final': round(float(result.bh_pv.iloc[-1]), 2),
        'base_bm_final': round(float(result.base_benchmark_pv.iloc[-1]), 2),
        'passed': ok,
    }


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('=' * 70)
    print('网格交易统一回测 (grid_v2.py)')
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'初始资金: {INITIAL_CAPITAL:,}  每格资金: {GRID_CAPITAL:,}')
    print('=' * 70)

    # 获取数据
    print('\n获取ETF数据...')
    prices_dict = get_all_etf_data()

    if len(prices_dict) < 6:
        print(f'⚠️ 只获取到 {len(prices_dict)} 个品种数据，期望6个')

    # 资金守恒验证
    conservation = verify_capital_conservation(prices_dict)

    # 1. 参数扫描
    scan_results = run_param_scan(prices_dict)

    # 2. 成本敏感性
    cost_sensitivity = run_cost_sensitivity(prices_dict)

    # 3. Walk-Forward
    walkforward = run_walkforward(prices_dict)

    # 4. 过拟合检验
    overfit_check = run_overfit_check(prices_dict)

    # 汇总输出
    output = {
        'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'initial_capital': INITIAL_CAPITAL,
            'grid_capital': GRID_CAPITAL,
            'grid_params': GRID_PARAMS,
            'base_positions': BASE_POSITIONS,
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'cost_scenarios': COST_SCENARIOS,
        },
        'capital_conservation': conservation,
        'scan_results': scan_results,
        'cost_sensitivity': cost_sensitivity,
        'walkforward': walkforward,
        'overfit_check': overfit_check,
    }

    output_path = os.path.join(RESULTS_DIR, 'grid.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f'\n{"=" * 70}')
    print(f'结果已保存: {output_path}')
    print(f'{"=" * 70}')

    # 汇总
    scan_wins = sum(1 for r in scan_results if r['excess_vs_bh'] > 0)
    print(f'\n=== 汇总 ===')
    print(f'参数扫描胜率: {scan_wins}/{len(scan_results)} = {scan_wins/len(scan_results):.0%}')
    for cs in cost_sensitivity:
        print(f'成本{cs["cost_label"]}: 胜率{cs["win_count"]}/{cs["total"]} = {cs["win_rate"]:.0%}')
    print(f'Walk-Forward: {len(walkforward)}个品种')
    for wf in walkforward:
        print(f'  {wf["symbol"]:8s}: 网格CAGR{wf["cumulative_grid_cagr"]:.1%} vs 买持CAGR{wf["cumulative_bh_cagr"]:.1%}'
              f' (超额{wf["cumulative_excess"]:+.1%})')
    print(f'过拟合: {sum(1 for b in overfit_check["best_params_comparison"] if b["same_params"])}/'
          f'{len(overfit_check["best_params_comparison"])} 品种两段最优一致')


if __name__ == '__main__':
    main()
