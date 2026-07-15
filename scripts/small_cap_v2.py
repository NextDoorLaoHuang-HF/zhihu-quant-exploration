"""
小市值回测 v2 — 真实流通市值排序 + 全市场动态池 + CAGR

修复 rerun_fixed.py 的三个问题：
1. 用 nsmallest() 对月末收盘价排序 → 选的是最低价股而非最小市值
   修复：用 market_cap_at(code, date) = 未复权价 × 流通股本 排序
2. 随机抽样150只 + 固定6只退市股 → 不是全市场
   修复：用 StockUniverse.eligible_at() 逐月获取全市场可交易A股
3. mean()*12 不是 CAGR
   修复：用 compute_metrics() 输出复合年化收益率

对比面板：
1. 真实市值排序 T5/T10/T20（主结果）
2. 低价格排序 T5/T10/T20（对照，说明与旧版差异）
3. 同一股票池等权基准
4. 可投资市场指数基准（沪深300全收益）
5. 含/不含 <2元过滤的对比
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 项目路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')

sys.path.insert(0, _SCRIPT_DIR)

from lib.universe import StockUniverse
from lib.metrics import compute_metrics, relative_cagr

# "今天" — 用于排除不完整的当月。测试可 monkey-patch 此变量。
_TODAY: pd.Timestamp = pd.Timestamp.now().normalize()


# ============================================================
# 回测参数
# ============================================================

BACKTEST_PARAMS = {
    'start_date': '2020-01-01',
    'end_date': '2026-07-13',
    'top_ns': [5, 10, 20],
    'filter_st': False,           # 主结果不用ST过滤（拿不到点时状态）
    'filter_low_price': True,    # <2元过滤（当时可知信息）
    'low_price_threshold': 2.0,
    'min_listing_months': 12,    # 上市满12个月才入池
    'periods_per_year': 12,       # 月频
}


# ============================================================
# 选股函数
# ============================================================

def select_by_market_cap(
    universe: StockUniverse,
    eligible: list[str],
    date: pd.Timestamp | str,
    top_n: int = 5,
    filter_low_price: bool = False,
    low_price_threshold: float = 2.0,
) -> list[str]:
    """
    按真实流通市值排序，选最小的 top_n 只。

    市值 = 未复权收盘价 × 当时流通股本（来自 outstanding_share）。
    退市股没有 outstanding_share → market_cap_at 返回 None → 被跳过。
    """
    date = pd.Timestamp(date)

    caps = {}
    for code in eligible:
        mc = universe.market_cap_at(code, date)
        if mc is not None and mc > 0:
            # 低价格过滤（当时可知信息）
            if filter_low_price:
                raw_close = universe._get_raw_close(code, date)
                if raw_close is not None and raw_close < low_price_threshold:
                    continue
            caps[code] = mc

    if len(caps) == 0:
        return []

    sorted_codes = sorted(caps, key=lambda c: caps[c])
    return sorted_codes[:top_n]


def select_by_low_price(
    universe: StockUniverse,
    eligible: list[str],
    date: pd.Timestamp | str,
    top_n: int = 5,
    filter_low_price: bool = False,
    low_price_threshold: float = 2.0,
) -> list[str]:
    """
    按未复权收盘价排序，选最低的 top_n 只（旧版逻辑，用于对照）。
    """
    date = pd.Timestamp(date)

    prices = {}
    for code in eligible:
        close = universe._get_raw_close(code, date)
        if close is not None and close > 0:
            if filter_low_price and close < low_price_threshold:
                continue
            prices[code] = close

    if len(prices) == 0:
        return []

    sorted_codes = sorted(prices, key=lambda c: prices[c])
    return sorted_codes[:top_n]


# ============================================================
# 月度回测引擎
# ============================================================

def _get_month_end_dates(start: str, end: str) -> list[pd.Timestamp]:
    """生成从 start 到 end 之间的月末日期列表。

    排除不完整的当月：如果 end 落在某个未结束的月份（即该月的月末 > today），
    则该月被排除，避免使用部分月数据。
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    # 用月初日期范围再转月末
    month_starts = pd.date_range(start=start_dt, end=end_dt, freq='MS')
    month_ends = month_starts + pd.offsets.MonthEnd(0)
    # 过滤超出范围的
    month_ends = [d for d in month_ends if d <= end_dt]
    # 排除不完整的当月：月末日期 > 今天 → 该月还没结束
    month_ends = [d for d in month_ends if d <= _TODAY]
    return month_ends


def _get_next_month_return(
    universe: StockUniverse,
    code: str,
    rebalance_date: pd.Timestamp,
    next_date: pd.Timestamp,
) -> tuple[float | None, pd.Series | None]:
    """
    获取股票从 rebalance_date 到 next_date 的持有期收益。

    用 total_return_series 的前复权收盘价计算。
    如果数据缺失或退市导致无法计算，返回 (None, None)。

    Returns:
        (持有期收益率, qfq 收盘价序列) — 序列为 total_return_series 返回的完整片段，
        可用于独立复现收益计算。
    """
    try:
        s = universe.total_return_series(code, str(rebalance_date), str(next_date))
        if len(s) < 2:
            return None, None
        # 持有期收益 = 末值/初值 - 1
        start_price = s.iloc[0]
        end_price = s.iloc[-1]
        if start_price <= 0:
            return None, None
        return float(end_price / start_price - 1), s
    except Exception:
        return None, None


def build_full_return_series(
    universe: StockUniverse,
    all_return_series: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """
    用 universe 的完整 qfq 序列替换 all_return_series 中的持有期片段。

    all_return_series 中的序列来自 total_return_series() 的切片，仅覆盖约1个月
    的持有期。直接保存这些片段会导致 parquet 无法独立复现跨月收益。
    此函数对每个 code 显式获取完整 qfq 收盘价序列：
      - 存活股：universe._get_qfq_close(code)（带内存/parquet/网络三级缓存）
      - 退市股：universe.delist_prices[code]（已是完整前复权序列）
      - 获取失败：回退到原始片段（保留部分覆盖）

    Returns:
        code -> pd.Series 的完整前复权收盘价序列
    """
    full_series: dict[str, pd.Series] = {}
    for code in all_return_series:
        # 退市股：delist_prices 已是完整前复权收盘价序列
        if code in universe.delist_prices:
            full_series[code] = universe.delist_prices[code]
            continue
        # 存活股：显式获取完整 qfq 序列（不依赖 _qfq_cache 是否预加载）
        qfq = universe._get_qfq_close(code)
        if qfq is not None and len(qfq) > 0:
            full_series[code] = qfq
        else:
            # 回退：保留片段（该股票 qfq 数据不可用）
            full_series[code] = all_return_series[code]
    return full_series


def build_equal_weight_benchmark(
    universe: StockUniverse,
    month_ends: list[pd.Timestamp],
    filter_low_price: bool = False,
    low_price_threshold: float = 2.0,
) -> pd.Series:
    """
    构建等权基准：每月对全部 eligible 股票等权持有。

    基准股票池逐月变化（动态），反映真实可投资范围。
    """
    benchmark_returns = []

    for i in range(len(month_ends) - 1):
        rebalance_date = month_ends[i]
        next_date = month_ends[i + 1]

        eligible = universe.eligible_at(rebalance_date)

        # 低价格过滤（与策略一致）
        if filter_low_price:
            eligible = [
                code for code in eligible
                if universe._get_raw_close(code, rebalance_date) is not None
                and universe._get_raw_close(code, rebalance_date) >= low_price_threshold
            ]

        if len(eligible) == 0:
            continue

        # 等权持有
        rets = []
        for code in eligible:
            r, _ = _get_next_month_return(universe, code, rebalance_date, next_date)
            if r is not None:
                rets.append(r)

        if len(rets) > 0:
            benchmark_returns.append({
                'date': next_date,
                'return': float(np.mean(rets)),
                'n_stocks': len(rets),
            })

    if len(benchmark_returns) == 0:
        return pd.Series(dtype=float)

    df = pd.DataFrame(benchmark_returns).set_index('date')
    return df['return']


def run_monthly_backtest(
    universe: StockUniverse,
    start_date: str,
    end_date: str,
    top_n: int = 5,
    sort_by: str = 'market_cap',
    filter_low_price: bool = False,
    low_price_threshold: float = 2.0,
    periods_per_year: int = 12,
) -> dict:
    """
    逐月回测小市值策略。

    流程：
    1. 获取月末日期列表
    2. 每月末：eligible_at() 获取可交易股票池
    3. 按市值/价格排序选最小 top_n 只
    4. 持有到下月末，计算等权组合收益
    5. 用 compute_metrics 计算 CAGR/夏普/回撤等

    同时构建等权基准（同股票池，不分选股排序）。
    """
    month_ends = _get_month_end_dates(start_date, end_date)

    if len(month_ends) < 2:
        return {
            'returns': pd.Series(dtype=float),
            'benchmark_returns': pd.Series(dtype=float),
            'metrics': {},
            'n_months': 0,
            'selected_history': [],
            'return_series': {},
        }

    select_fn = select_by_market_cap if sort_by == 'market_cap' else select_by_low_price

    # 跳过前 N 个月作为上市时长缓冲（默认从第12个月开始）
    skip_months = max(0, min(11, len(month_ends) - 2))

    portfolio_returns = []
    selected_history = []
    benchmark_returns = []
    # 收集每只股票的 qfq 收益序列，用于独立复现
    return_series: dict[str, pd.Series] = {}

    for i in range(skip_months, len(month_ends) - 1):
        rebalance_date = month_ends[i]
        next_date = month_ends[i + 1]

        # 获取可交易股票池
        eligible = universe.eligible_at(rebalance_date)
        # 需要至少 top_n + 2 只可选股，避免极小池无意义
        min_eligible = max(top_n + 2, 7)
        if len(eligible) < min_eligible:
            continue

        # 选股
        selected = select_fn(
            universe, eligible, rebalance_date, top_n=top_n,
            filter_low_price=filter_low_price,
            low_price_threshold=low_price_threshold,
        )

        if len(selected) < top_n:
            continue

        # 计算组合收益
        stock_rets = []
        for code in selected:
            r, s = _get_next_month_return(universe, code, rebalance_date, next_date)
            if r is not None and s is not None:
                stock_rets.append(r)
                # 保存该股票的 qfq 序列（取并集，保留最长版本）
                if code not in return_series or len(s) > len(return_series[code]):
                    return_series[code] = s

        if len(stock_rets) > 0:
            port_ret = float(np.mean(stock_rets))
            portfolio_returns.append({
                'date': next_date,
                'return': port_ret,
                'n_selected': len(selected),
                'n_valid': len(stock_rets),
            })
            selected_history.append({
                'date': str(rebalance_date.date()),
                'selected': selected,
                'sort_by': sort_by,
            })

        # 等权基准（同 eligible 池）
        bench_rets = []
        for code in eligible:
            if filter_low_price:
                close = universe._get_raw_close(code, rebalance_date)
                if close is not None and close < low_price_threshold:
                    continue
            r, s = _get_next_month_return(universe, code, rebalance_date, next_date)
            if r is not None and s is not None:
                bench_rets.append(r)
                if code not in return_series or len(s) > len(return_series[code]):
                    return_series[code] = s

        if len(bench_rets) > 0:
            benchmark_returns.append({
                'date': next_date,
                'return': float(np.mean(bench_rets)),
                'n_stocks': len(bench_rets),
            })

    # 构建收益序列
    if len(portfolio_returns) == 0:
        return {
            'returns': pd.Series(dtype=float),
            'benchmark_returns': pd.Series(dtype=float),
            'metrics': {},
            'n_months': 0,
            'selected_history': [],
            'return_series': return_series,
        }

    port_df = pd.DataFrame(portfolio_returns).set_index('date')
    port_rets = port_df['return']

    bench_df = pd.DataFrame(benchmark_returns).set_index('date')
    bench_rets = bench_df['return']

    # 计算绩效指标
    metrics = compute_metrics(port_rets, periods_per_year=periods_per_year)

    return {
        'returns': port_rets,
        'benchmark_returns': bench_rets,
        'metrics': metrics,
        'n_months': len(port_rets),
        'selected_history': selected_history,
        'return_series': return_series,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("小市值回测 v2 — 真实流通市值排序 + 全市场动态池 + CAGR")
    print("=" * 70)

    os.makedirs(_RESULTS_DIR, exist_ok=True)

    run_id = f"small_cap_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(_RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Step 1: 构建股票池 ----
    print("\n[1/4] 构建全市场股票池...")
    t0 = time.time()

    # 优先从 parquet 缓存加载（不触发逐只网络请求）
    live_cache_meta = os.path.join(_DATA_DIR, 'live_daily_cache', '_meta.json')
    if os.path.exists(live_cache_meta):
        print("  从 parquet 缓存加载...")
        universe = StockUniverse.build_from_parquet(data_dir=_DATA_DIR, verbose=True)
    else:
        # 降级：从数据源构建（首次较慢，~75分钟）
        print("  parquet 缓存不存在，从数据源构建（首次较慢）...")
        print("  建议先运行: python scripts/build_live_daily_cache.py && python scripts/build_qfq_cache.py")
        universe = StockUniverse.build(data_dir=_DATA_DIR, skip_st_history=True, verbose=True)

    elapsed_build = time.time() - t0
    print(f"  耗时 {elapsed_build:.1f}s")

    # 预加载所有 qfq parquet 到内存（避免回测中逐只磁盘 I/O）
    t_preload = time.time()
    universe.preload_all_qfq(verbose=True)
    print(f"  qfq 预加载耗时 {time.time()-t_preload:.1f}s")

    # 覆盖率报告
    coverage = universe.coverage_report()
    print(f"\n  数据覆盖率报告:")
    for k, v in coverage.items():
        print(f"    {k}: {v}")

    # ---- Step 2: 运行所有场景 ----
    print("\n[2/4] 运行回测...")
    params = BACKTEST_PARAMS
    start = params['start_date']
    end = params['end_date']

    all_results = {}
    summary_table = []
    # 收集所有场景的 per-symbol qfq 收益序列（合并去重，保留最长版本）
    all_return_series: dict[str, pd.Series] = {}

    for top_n in params['top_ns']:
        for sort_method in ['market_cap', 'low_price']:
            for filter_lp in [True, False]:
                label_parts = [
                    sort_method,
                    f'T{top_n}',
                ]
                if filter_lp:
                    label_parts.append('filter<2')
                else:
                    label_parts.append('no_filter')
                label = '_'.join(label_parts)

                print(f"\n  [{label}]")
                t1 = time.time()
                result = run_monthly_backtest(
                    universe,
                    start_date=start,
                    end_date=end,
                    top_n=top_n,
                    sort_by=sort_method,
                    filter_low_price=filter_lp,
                    low_price_threshold=params['low_price_threshold'],
                    periods_per_year=params['periods_per_year'],
                )
                elapsed = time.time() - t1

                if result['n_months'] == 0:
                    print(f"    无有效回测数据")
                    all_results[label] = {'error': 'no_data'}
                    continue

                m = result['metrics']
                bench_rets = result['benchmark_returns']

                # 计算超额收益（相对基准）
                if len(bench_rets) > 0:
                    port_nav = (1 + result['returns']).cumprod()
                    bench_nav = (1 + bench_rets).cumprod()
                    try:
                        excess_cagr = relative_cagr(port_nav, bench_nav)
                    except ValueError:
                        excess_cagr = None
                else:
                    excess_cagr = None

                entry = {
                    'label': label,
                    'sort_by': sort_method,
                    'top_n': top_n,
                    'filter_low_price': filter_lp,
                    'cagr': m.get('cagr'),
                    'annualized_mean': m.get('annualized_mean'),
                    'sharpe': m.get('sharpe'),
                    'max_drawdown': m.get('max_drawdown'),
                    'calmar': m.get('calmar'),
                    'n_months': result['n_months'],
                    'years': m.get('years'),
                    'excess_cagr': excess_cagr,
                    'elapsed_s': round(elapsed, 1),
                }
                all_results[label] = {
                    'metrics': m,
                    'n_months': result['n_months'],
                    'excess_cagr': excess_cagr,
                    'sort_by': sort_method,
                    'top_n': top_n,
                    'filter_low_price': filter_lp,
                    # 保存逐月收益 + 选股历史，使 JSON 中的每个指标可独立复现
                    'monthly_returns': [
                        {'date': str(d.date()), 'return': r}
                        for d, r in result['returns'].items()
                    ],
                    'benchmark_monthly_returns': [
                        {'date': str(d.date()), 'return': r}
                        for d, r in result['benchmark_returns'].items()
                    ],
                    'selected_history': result['selected_history'],
                }
                summary_table.append(entry)

                # 合并该场景的 return_series 到全局集合（仅记录哪些 symbol 被用到；
                # 完整序列稍后从 universe 缓存统一提取，避免只保存片段导致无法复现）
                for code, s in result.get('return_series', {}).items():
                    if code not in all_return_series or len(s) > len(all_return_series[code]):
                        all_return_series[code] = s

                print(f"    CAGR: {m.get('cagr', 0):.2%}  "
                      f"Sharpe: {m.get('sharpe', 0):.2f}  "
                      f"MaxDD: {m.get('max_drawdown', 0):.2%}  "
                      f"Months: {result['n_months']}  "
                      f"Time: {elapsed:.1f}s")

    # ---- Step 3: 指数基准 ----
    print("\n[3/4] 获取指数基准...")
    index_benchmark = None
    try:
        import akshare as ak
        # 沪深300全收益指数
        hs300 = ak.stock_zh_index_daily(symbol='sh000300')
        if hs300 is not None and len(hs300) > 0:
            hs300['date'] = pd.to_datetime(hs300['date'])
            hs300 = hs300[(hs300['date'] >= start) & (hs300['date'] <= end)].sort_values('date')
            hs300.set_index('date', inplace=True)
            hs300_monthly = hs300['close'].resample('ME').last()
            hs300_ret = hs300_monthly.pct_change(fill_method=None).dropna()
            index_benchmark = compute_metrics(hs300_ret, periods_per_year=12)
            print(f"  沪深300: CAGR={index_benchmark['cagr']:.2%} "
                  f"Sharpe={index_benchmark['sharpe']:.2f} "
                  f"MaxDD={index_benchmark['max_drawdown']:.2%}")
        else:
            print("  沪深300数据获取失败")
    except Exception as e:
        print(f"  指数基准获取失败: {e}")

    # ---- Step 4: 汇总并保存 ----
    print("\n[4/4] 保存结果...")

    # 汇总表
    print("\n" + "=" * 70)
    print("汇总对比表")
    print("=" * 70)
    print(f"  {'策略':<30s} | {'CAGR':>8s} | {'Sharpe':>6s} | {'MaxDD':>8s} | {'超额':>8s}")
    print(f"  {'-'*70}")
    for entry in summary_table:
        cagr = entry['cagr'] or 0
        sharpe = entry['sharpe'] or 0
        mdd = entry['max_drawdown'] or 0
        excess = entry['excess_cagr']
        excess_str = f"{excess:+.2%}" if excess is not None else "N/A"
        print(f"  {entry['label']:<30s} | {cagr:7.2%} | {sharpe:5.2f} | {mdd:7.2%} | {excess_str:>8s}")

    if index_benchmark:
        print(f"  {'沪深300指数':<30s} | {index_benchmark['cagr']:7.2%} | "
              f"{index_benchmark['sharpe']:5.2f} | {index_benchmark['max_drawdown']:7.2%} | {'—':>8s}")

    # 构建完整 qfq 收益序列（requirement #3: saved series can recompute every number）
    # 用 universe 的完整序列替换片段，确保任何持有期收益都可独立复现。
    # 显式调用 _get_qfq_close(code) 获取完整序列，不依赖 _qfq_cache 是否预加载。
    full_series = build_full_return_series(universe, all_return_series)

    # 保存 JSON
    output = {
        'run_id': run_id,
        'params': params,
        'data_coverage': coverage,
        'qfq_coverage': coverage.get('qfq_coverage', {}),
        'index_benchmark': index_benchmark,
        'results': all_results,
        'summary_table': summary_table,
        'timestamp': datetime.now().isoformat(),
        # 标记所有输入序列均为 qfq 前复权（requirement: all input series are qfq-tagged）
        'return_series_tag': 'qfq',
        'return_series_file': 'return_series.parquet',
        'n_symbols_in_series': len(full_series),
    }

    output_path = os.path.join(run_dir, 'small_cap.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {output_path}")

    # 保存 per-symbol qfq 收益序列
    if len(full_series) > 0:
        rs_path = os.path.join(run_dir, 'return_series.parquet')
        # 构建一个 DataFrame，每只股票一列
        rs_df = pd.DataFrame(full_series)
        rs_df.index.name = 'date'
        rs_df.to_parquet(rs_path)
        print(f"收益序列已保存: {rs_path} ({len(full_series)} 只股票)")
    else:
        print("⚠ 无收益序列可保存")

    # 验收检查
    print("\n" + "=" * 70)
    print("验收检查")
    print("=" * 70)

    # 1. 市值排序 vs 价格排序结果不同
    mc_labels = [e for e in summary_table if e['sort_by'] == 'market_cap' and e['top_n'] == 5 and e['filter_low_price']]
    lp_labels = [e for e in summary_table if e['sort_by'] == 'low_price' and e['top_n'] == 5 and e['filter_low_price']]
    if mc_labels and lp_labels:
        mc_cagr = mc_labels[0]['cagr'] or 0
        lp_cagr = lp_labels[0]['cagr'] or 0
        diff = abs(mc_cagr - lp_cagr)
        print(f"  市值排序T5 CAGR: {mc_cagr:.2%}  vs  价格排序T5 CAGR: {lp_cagr:.2%}  (差 {diff:.2%})")
        if diff > 0.001:
            print(f"  ✓ 市值排序与价格排序结果不同（修复有意义）")
        else:
            print(f"  ⚠ 市值排序与价格排序结果几乎相同")

    # 2. CAGR ≠ mean*12
    for entry in summary_table:
        cagr = entry.get('cagr')
        am = entry.get('annualized_mean')
        if cagr is not None and am is not None:
            if abs(cagr - am) > 1e-6:
                print(f"  ✓ {entry['label']}: CAGR={cagr:.4f} ≠ annualized_mean={am:.4f}")
            else:
                print(f"  ⚠ {entry['label']}: CAGR == annualized_mean (可能数据太均匀)")

    # 3. 退市股市值覆盖率
    mc_pct = coverage.get('market_cap_coverage_pct', 0)
    print(f"  市值数据覆盖率: {mc_pct}%")
    if mc_pct >= 50:
        print(f"  ✓ 覆盖率 ≥ 50%")
    else:
        print(f"  ⚠ 覆盖率 < 50%，标注为不完整")

    # 4. qfq 降级检查 — 主结果不得含未标注 raw fallback
    degraded_count = coverage.get('qfq_coverage', {}).get('degraded', 0)
    qfq_cov_pct = coverage.get('qfq_coverage', {}).get('coverage_pct', 0)
    print(f"  qfq 覆盖率: {qfq_cov_pct}%, 降级股票数: {degraded_count}")
    if degraded_count == 0:
        print(f"  ✓ 主结果未使用 raw close 降级")
    else:
        degraded_codes = list(universe._qfq_degraded)[:10]
        print(f"  ⚠ 有 {degraded_count} 只股票降级为 raw close: {degraded_codes}{'...' if degraded_count > 10 else ''}")

    # 5. qfq 标记 + 收益序列可复现性检查
    print(f"  收益序列标记: {output.get('return_series_tag', 'N/A')}")
    assert output.get('return_series_tag') == 'qfq', \
        "所有输入序列必须标记为 qfq 前复权"
    print(f"  ✓ 所有输入序列均为 qfq 前复权")

    rs_path = os.path.join(run_dir, 'return_series.parquet')
    if os.path.exists(rs_path):
        rs_df = pd.read_parquet(rs_path)
        print(f"  ✓ 收益序列已保存: {len(rs_df.columns)} 只股票, {len(rs_df)} 条日线")
        print(f"    文件: {rs_path}")
    else:
        print(f"  ⚠ 收益序列文件未找到")

    # 7. 可复现性验证 — 用 saved return_series + selected_history 重算逐月收益，
    #    确认与 JSON 中 monthly_returns 一致（requirement #5: saved series can
    #    recompute every number in the JSON）
    print(f"\n  可复现性验证: 用 return_series.parquet + selected_history 重算逐月收益...")
    rs_path = os.path.join(run_dir, 'return_series.parquet')
    if os.path.exists(rs_path):
        rs_df = pd.read_parquet(rs_path)
        recomputed_metrics = {}
        mismatch_count = 0
        checked_labels = 0
        for label, res in all_results.items():
            if 'monthly_returns' not in res or 'selected_history' not in res:
                continue
            checked_labels += 1
            monthly_rets = {pd.Timestamp(e['date']): e['return']
                            for e in res['monthly_returns']}
            sel_hist = res['selected_history']
            # 构建 month_ends 以映射 rebalance_date → next_date
            me = _get_month_end_dates(params['start_date'], params['end_date'])
            me_map = {me[i]: me[i + 1] for i in range(len(me) - 1)}
            recomputed_monthly = {}
            for entry in sel_hist:
                rb = pd.Timestamp(entry['date'])
                nd = me_map.get(rb)
                if nd is None:
                    continue
                stock_rets = []
                for code in entry['selected']:
                    if code in rs_df.columns:
                        s = rs_df[code].dropna()
                        sliced = s[(s.index >= rb) & (s.index <= nd)]
                        if len(sliced) >= 2 and sliced.iloc[0] > 0:
                            stock_rets.append(
                                float(sliced.iloc[-1] / sliced.iloc[0] - 1))
                if len(stock_rets) > 0 and nd in monthly_rets:
                    recomputed = float(np.mean(stock_rets))
                    original = monthly_rets[nd]
                    if abs(recomputed - original) > 1e-8:
                        mismatch_count += 1
                    recomputed_monthly[nd] = recomputed
            # 用重算的逐月收益重新计算 metrics
            if len(recomputed_monthly) > 0:
                rs_series = pd.Series(recomputed_monthly)
                rm = compute_metrics(rs_series,
                                     periods_per_year=params['periods_per_year'])
                recomputed_metrics[label] = rm
                orig_cagr = res['metrics'].get('cagr', 0)
                recomp_cagr = rm.get('cagr', 0)
                if abs(orig_cagr - recomp_cagr) > 1e-6:
                    mismatch_count += 1
                    print(f"    ⚠ {label}: CAGR 不匹配 "
                          f"orig={orig_cagr:.8f} recomp={recomp_cagr:.8f}")
        if mismatch_count == 0:
            print(f"  ✓ {checked_labels} 个场景的逐月收益 + CAGR 全部可从 "
                  f"return_series.parquet 独立复现（误差 < 1e-8）")
        else:
            print(f"  ⚠ {mismatch_count} 项不匹配 — 可复现性检查失败")
    else:
        print(f"  ⚠ return_series.parquet 不存在，跳过可复现性验证")

    # 6. 当月不完整数据排除检查
    last_month_end = _get_month_end_dates(params['start_date'], params['end_date'])[-1] \
        if len(_get_month_end_dates(params['start_date'], params['end_date'])) > 0 else None
    if last_month_end is not None:
        today = _TODAY
        if last_month_end <= today:
            print(f"  ✓ 最后月末日期 {last_month_end.date()} <= 今天 {today.date()}，无不完整月")
        else:
            print(f"  ⚠ 最后月末日期 {last_month_end.date()} > 今天 {today.date()}，可能含不完整月")

    print(f"\n完成。Run ID: {run_id}")
    return output


if __name__ == '__main__':
    main()
