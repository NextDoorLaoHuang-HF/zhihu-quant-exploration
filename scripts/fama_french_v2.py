"""
Fama-French 三因子模型 V2 — 正式版（含实际公告日修正）
按 Kenneth French 官方规则构建 SMB/HML 因子

用法：
    python3 scripts/fama_french_v2.py [--start 2020] [--end 2024]
    python3 scripts/fama_french_v2.py --rebuild  # 删除旧缓存，重新拉取

输出：
    results/<run_id>/fama_french.json
    results/<run_id>/fama_french.csv

注意：这是 A 股样本上的 FF 风格因子构造，
不与 Kenneth French 官方因子直接比较。

与旧版的区别（02_conventional_dualma_ff.py / gen_ff3_chart.py）：
1. 市值 = 未复权收盘价 × 流通股本（旧版用收盘价×固定股本或1/股价）
2. B/M = 上一财年正账面净资产 / 当年6月末流通市值（旧版用12月涨跌幅）
3. 账面数据按实际公告日期滞后（stock_report_disclosure 实际披露日，旧版用固定4/30）
4. 持有期收益用前复权价格（含分红再投资，旧版用未复权 close）
5. 形成日调整为6月最后交易日（旧版固定6/30，非交易日时收益为0）
6. 2×3 分组：市值中位数分大小，B/M 30%/70% 分位数分低中高
7. 组合内市值加权（旧版等权）
8. 空组合返回 NaN 而非伪 0（旧版返回 0.0 导致因子精确为 0）
"""
import sys
import os
import json
import argparse
import pickle
import pandas as pd
from datetime import datetime
from scipy import stats

# 确保能找到 scripts/lib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.universe import StockUniverse
from lib.fama_french import FamaFrenchBuilder
from lib.pickle_compat import load_pickle_compat


def main():
    parser = argparse.ArgumentParser(description='Fama-French 三因子构建 V2')
    parser.add_argument('--start', type=int, default=2020, help='起始年份')
    parser.add_argument('--end', type=int, default=2024, help='结束年份')
    parser.add_argument('--max-live', type=int, default=None,
                        help='最多拉取多少只存活股（测试用）')
    parser.add_argument('--rebuild', action='store_true',
                        help='删除旧缓存，重新拉取账面数据（含实际公告日）')
    args = parser.parse_args()

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', run_id
    )
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 70)
    print('Fama-French 三因子构建 V2（正式版，含实际公告日修正）')
    print(f'  年份范围: {args.start} - {args.end}')
    print(f'  输出目录: {results_dir}')
    print('=' * 70)

    # Step 1: 构建股票池（优先从 parquet 缓存加载，回退到 pickle）
    print('\n[1/3] 构建股票池...')
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'universe_cache.pkl'
    )
    live_cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'live_daily_cache'
    )
    live_meta_path = os.path.join(live_cache_dir, '_meta.json')

    universe = None
    # 优先使用 parquet 缓存（避免 pickle 版本兼容问题）
    if os.path.exists(live_meta_path):
        print('  从 parquet 缓存加载...')
        try:
            universe = StockUniverse.build_from_parquet(verbose=True)
        except Exception as e:
            print(f'  parquet 加载失败: {e}')
            universe = None

    if universe is None and os.path.exists(cache_path):
        print('  从 pickle 缓存加载...')
        try:
            universe = load_pickle_compat(cache_path)
        except Exception as e:
            print(f'  pickle 加载失败: {e}')

    if universe is None:
        print('  从数据源构建（首次较慢）...')
        universe = StockUniverse.build(max_live=args.max_live)

    # 兼容旧缓存：确保新字段存在
    if not hasattr(universe, '_qfq_failed'):
        universe._qfq_failed = set()
    if not hasattr(universe, '_qfq_degraded'):
        universe._qfq_degraded = set()

    report = universe.coverage_report()
    print(f'  A 股: {report["a_shares"]} 只')
    print(f'  有市值数据: {report["has_market_cap_data"]} 只')
    print(f'  市值覆盖率: {report["market_cap_coverage_pct"]}%')

    # Step 2: 拉取账面数据（含实际公告日）
    print('\n[2/3] 拉取账面权益数据（含实际公告日）...')
    be_cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'book_equity_all.pkl'
    )
    if args.rebuild and os.path.exists(be_cache_path):
        print('  --rebuild: 删除旧缓存，重新拉取...')
        os.remove(be_cache_path)

    builder = FamaFrenchBuilder.build(universe)
    print(f'  有账面数据: {len(builder.book_equity_data)} 只')
    print(f'  公告日来源: {builder.announce_date_source}')

    # Step 3: 计算因子（含数据质量报告）
    print(f'\n[3/3] 计算 SMB/HML 因子 ({args.start}-{args.end})...')
    factors, quality = builder.compute_factors_with_quality(args.start, args.end)

    if factors.empty:
        print('  ⚠️ 因子构建失败（数据不足）')
        return

    # 统计
    print(f'\n  因子时间序列: {factors.index[0].date()} ~ {factors.index[-1].date()}, '
          f'{len(factors)} 月')
    print(f'\n  因子统计:')
    factor_stats = {}
    for col in ['MKT', 'SMB', 'HML']:
        s = factors[col].dropna()
        if len(s) == 0:
            print(f'    {col}: 全部为 NaN')
            factor_stats[col] = {
                'annualized_mean': None,
                'annualized_std': None,
                'sharpe': None,
                't_stat': None,
                'p_value': None,
                'n_months': 0,
                'n_nan': len(factors[col]) - len(s),
            }
            continue
        ann_mean = s.mean() * 12
        ann_std = s.std() * (12 ** 0.5)
        sharpe = ann_mean / ann_std if ann_std > 0 else 0
        # t 统计量 = mean / (std / sqrt(n))
        t_stat = s.mean() / (s.std() / (len(s) ** 0.5)) if s.std() > 0 else 0
        # 也用 scipy 验证
        t_scipy, p_scipy = stats.ttest_1samp(s, 0)
        t_scipy = float(t_scipy)
        p_scipy = float(p_scipy)
        print(f'    {col}: 年化{ann_mean:.1%}  夏普{sharpe:.2f}  t={t_stat:.2f}  '
              f'(scipy t={t_scipy:.2f}, p={p_scipy:.4f})  NaN月={len(factors[col]) - len(s)}')
        factor_stats[col] = {
            'annualized_mean': round(float(ann_mean), 6),
            'annualized_std': round(float(ann_std), 6),
            'sharpe': round(float(sharpe), 4),
            't_stat': round(float(t_stat), 4),
            'p_value': round(p_scipy, 6),
            'n_months': len(s),
            'n_nan_months': len(factors[col]) - len(s),
        }

    # 相关性
    print(f'\n  因子相关性:')
    print(f'    MKT-SMB: {factors["MKT"].corr(factors["SMB"]):.3f}')
    print(f'    MKT-HML: {factors["MKT"].corr(factors["HML"]):.3f}')
    print(f'    SMB-HML: {factors["SMB"].corr(factors["HML"]):.3f}')

    # 数据覆盖率
    total_stocks = len(universe.stock_meta)
    with_be = len(builder.book_equity_data)
    with_mc = report['has_market_cap_data']

    # 数据质量摘要
    dq = quality['data_quality']
    empty_months = sum(1 for v in dq['empty_portfolios'].values() if v > 0)
    avg_stocks = sum(dq['monthly_stock_counts'].values()) / max(len(dq['monthly_stock_counts']), 1)

    print(f'\n  数据质量:')
    print(f'    每月平均有效组合: {avg_stocks:.1f}/6')
    print(f'    有空组合的月数: {empty_months}/{dq["n_months"]}')
    print(f'    qfq降级股票数: {dq["downgraded_stocks"]}')

    # 保存结果
    output_path = os.path.join(results_dir, 'fama_french.json')
    output = {
        'description': 'A 股样本上的 Fama-French 风格因子构造',
        'note': '不与 Kenneth French 官方因子直接比较',
        'method': {
            'market_cap': '未复权收盘价 × 流通股本',
            'bm_ratio': '上一财年正账面净资产 / 当年6月末流通市值',
            'book_equity_lag': f'按公告日期滞后（{builder.announce_date_source}）',
            'holding_returns': '前复权价格（含分红再投资）',
            'formation_date': '6月最后交易日（非交易日回退）',
            'size_split': '市值中位数分 Small/Big',
            'bm_split': 'B/M 30%/70% 分位数分 Low/Medium/High',
            'weighting': '组合内市值加权',
            'rebalance': '每年6月末重新分组，持有到次年6月',
            'data_source': 'akshare stock_zcfz_em（权益）+ stock_report_disclosure（公告日）',
            'empty_handling': '空组合返回 NaN，不返回伪 0',
        },
        'start_year': args.start,
        'end_year': args.end,
        'n_months': len(factors),
        'factors': {
            col: factor_stats[col] for col in ['MKT', 'SMB', 'HML']
        },
        'data_coverage': {
            'total_stocks': total_stocks,
            'with_book_equity': with_be,
            'with_market_cap': with_mc,
            'announce_date_source': builder.announce_date_source,
            'note': '退市股无 outstanding_share，不参与市值排序和分组',
        },
        'data_quality': {
            'monthly_stock_counts': dq['monthly_stock_counts'],
            'empty_portfolios': dq['empty_portfolios'],
            'downgraded_stocks': dq['downgraded_stocks'],
            'n_months': dq['n_months'],
            'months_with_empty_portfolios': empty_months,
            'avg_portfolios_per_month': round(avg_stocks, 2),
        },
        'factor_correlations': {
            'MKT_SMB': round(float(factors['MKT'].corr(factors['SMB'])), 4),
            'MKT_HML': round(float(factors['MKT'].corr(factors['HML'])), 4),
            'SMB_HML': round(float(factors['SMB'].corr(factors['HML'])), 4),
        },
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 结果已保存: {output_path}')

    # 也保存 CSV 方便查看
    csv_path = os.path.join(results_dir, 'fama_french.csv')
    factors.to_csv(csv_path)
    print(f'✅ CSV 已保存: {csv_path}')


if __name__ == '__main__':
    main()
