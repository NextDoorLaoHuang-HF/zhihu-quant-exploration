"""
scripts/run_hybrid_v2.py — 用真实可转债数据跑 CB等权/CB-HRP 混合策略回测

策略说明：
  本策略混合两个可转债组合的收益：
  - CB等权：动态池中所有在交易可转债的等权月收益
  - CB-HRP：用层次风险平价(HRP)分配权重的可转债组合月收益
  混合比例通过 walk-forward OOS 估计。

  ETF（511380 可转债ETF）仅用于日期对齐和基准对比，不参与策略收益计算。

数据来源：
  - 04.a-share/cb_cache/cb_daily_all.pkl — 可转债日线（含已退出券）
  - 04.a-share/cb_cache/cb_meta_all.pkl  — 可转债元信息（上市日、退出日、退出最终价）
  - ak.fund_etf_hist_sina(sh511380)     — 可转债ETF（仅用于日期对齐）

注意：
  exit_final_price 是最后交易日收盘价（代理值），不是实际到期赎回/强赎结算价。
  数据源 ak.bond_zh_hs_cov_daily 只返回OHLCV，无法获取退出原因和结算价。

输出：
  results/hybrid_v2/hybrid.json         — OOS回测结果
  results/hybrid_v2/oos_returns.csv    — OOS月收益序列
"""
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import akshare as ak
import pandas as pd
import numpy as np

# 让 scripts/ 能找到 lib
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from lib.hybrid import CBPool, HybridBacktest, save_results

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CB_CACHE_DIR = os.path.join(PROJECT_ROOT, '04.a-share', 'cb_cache')


def load_cb_data() -> tuple[dict[str, pd.Series], dict]:
    """
    从缓存加载可转债价格数据，构建 CBPool 所需的格式。

    返回：
        prices: {code: pd.Series(index=Date, values=收盘价)}
        meta:   {code: {listing_date, delist_date, exit_reason, exit_final_price}}
    """
    daily_path = os.path.join(CB_CACHE_DIR, 'cb_daily_all.pkl')
    meta_path = os.path.join(CB_CACHE_DIR, 'cb_meta_all.pkl')

    print('加载可转债缓存数据...')
    with open(daily_path, 'rb') as f:
        daily_data = pickle.load(f)
    with open(meta_path, 'rb') as f:
        meta_raw = pickle.load(f)

    print(f'  缓存: {len(daily_data)} 只可转债')

    # 转换为 CBPool 所需格式
    prices = {}
    meta = {}

    for code, df in daily_data.items():
        if len(df) == 0:
            continue

        # 构建价格序列
        s = pd.Series(
            df['close'].values,
            index=pd.DatetimeIndex(df['date']),
            name=code,
            dtype=float,
        )
        # 去重索引
        s = s[~s.index.duplicated(keep='last')]
        prices[code] = s

        # 构建元信息
        m = meta_raw.get(code, {})
        meta[code] = {
            'listing_date': m.get('listing_date'),
            'delist_date': m.get('delist_date'),
            'exit_reason': m.get('exit_reason'),
            'exit_final_price': m.get('exit_final_price'),
        }

    print(f'  有效: {len(prices)} 只')

    # 统计
    active = sum(1 for v in meta.values() if v.get('delist_date') is None)
    exited = sum(1 for v in meta.values() if v.get('delist_date') is not None)
    print(f'  存续: {active} 只, 已退出: {exited} 只')

    return prices, meta


def fetch_etf_returns() -> pd.Series:
    """获取可转债ETF日收益序列。"""
    print('获取可转债ETF(511380)日收益...')
    etf = ak.fund_etf_hist_sina(symbol='sh511380')
    etf['date'] = pd.to_datetime(etf['date'])
    etf = etf.sort_values('date').reset_index(drop=True)
    etf.set_index('date', inplace=True)

    etf_ret = etf['close'].pct_change(fill_method=None).dropna()
    print(f'  ETF: {etf_ret.index[0].date()} ~ {etf_ret.index[-1].date()}, {len(etf_ret)} 天')
    return etf_ret


def main():
    print('=' * 60)
    print('CB等权/CB-HRP 混合策略回测（扩展可转债池版）')
    print('ETF仅用于日期对齐，不参与策略收益')
    print('=' * 60)

    # 1. 加载数据
    cb_prices, cb_meta = load_cb_data()

    # 2. 获取ETF收益
    etf_returns = fetch_etf_returns()

    # 3. 构建CBPool
    print('\n构建可转债动态池...')
    pool = CBPool(prices=cb_prices, meta=cb_meta)

    # 验证动态池大小
    test_dates = ['2020-01-06', '2021-06-01', '2023-01-03', '2025-01-06', '2026-06-01']
    for d in test_dates:
        eligible = pool.eligible_at(d)
        print(f'  {d}: {len(eligible)} 只可转债可交易')

    # 4. 运行回测
    print('\n运行 walk-forward 回测...')
    bt = HybridBacktest(
        cb_pool=pool,
        etf_returns=etf_returns,
        train_months=36,   # 3年训练窗口
        oos_months=12,     # 1年OOS
        cost_bps=10,        # 单边0.1%交易成本
        hrp_lookback=252,   # 252个交易日滚动窗口
    )

    results = bt.run()

    # 5. 输出结果
    print('\n保存结果...')
    path = save_results(results, run_id='hybrid_v2')
    print(f'  保存到: {path}')

    # 6. 打印摘要
    print('\n' + '=' * 60)
    print('回测结果摘要')
    print('=' * 60)
    print(f'可转债池总量: {results["n_bonds_total"]} 只')
    print(f'回测期间: {results["period"]["start"]} ~ {results["period"]["end"]} ({results["period"]["n_months"]} 月)')
    print(f'配置: train={results["config"]["train_months"]}月, oos={results["config"]["oos_months"]}月, cost={results["config"]["cost_bps"]}bps')

    print(f'\nOOS 段数: {len(results["oos_segments"])}')
    for seg in results['oos_segments']:
        m = seg.get('metrics', {}) or {}
        print(f'  {seg["oos_start"]} ~ {seg["oos_end"]}: CB={seg["cb_ratio"]:.0%}, '
              f'CAGR={m.get("cagr", "N/A")}, Sharpe={m.get("sharpe", "N/A")}' if m else
              f'  {seg["oos_start"]} ~ {seg["oos_end"]}: CB={seg["cb_ratio"]:.0%}')

    oos_metrics = results.get('oos_metrics', {}) or {}
    if oos_metrics:
        print(f'\nOOS 整体:')
        print(f'  CAGR:          {oos_metrics.get("cagr", "N/A")}')
        print(f'  年化波动:       {oos_metrics.get("annualized_vol", "N/A")}')
        print(f'  Sharpe:        {oos_metrics.get("sharpe", "N/A")}')
        print(f'  最大回撤:       {oos_metrics.get("max_drawdown", "N/A")}')
        print(f'  Calmar:        {oos_metrics.get("calmar", "N/A")}')
        print(f'  总收益:         {oos_metrics.get("total_return", "N/A")}')

    print(f'\n固定比例对比（同OOS区间，同成本路径）:')
    for ratio, metrics in results.get('fixed_ratios', {}).items():
        if metrics:
            print(f'  CB_eq/CB_HRP={ratio}: CAGR={metrics.get("cagr", "N/A")}, Sharpe={metrics.get("sharpe", "N/A")}, n={metrics.get("n_periods", "N/A")}')


if __name__ == '__main__':
    main()
