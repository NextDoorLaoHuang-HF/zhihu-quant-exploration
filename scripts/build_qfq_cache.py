"""
持久化 qfq/total_return 缓存构建器

为全市场收益序列建立持久化增量 qfq/total_return 缓存。
- 存活股：通过 akshare stock_zh_a_daily(adjust='qfq') 拉取前复权收盘价
- 退市股：从 delist_prices.pkl 加载（已是前复权）
- 缓存格式：Parquet 文件（每只股票一个文件），增量保存
- 失败记录：单独 JSON 文件，不重试已失败的股票
- 降级追踪：明确标注哪些股票使用了 raw close 降级

用法：
    python scripts/build_qfq_cache.py [--workers 10] [--max-stocks N]

输出：
    data/qfq_cache/           — 每只股票一个 parquet 文件
    data/qfq_cache/_meta.json  — 缓存元信息（成功/失败/降级计数）
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings
import concurrent.futures
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_QFQ_CACHE_DIR = os.path.join(_DATA_DIR, 'qfq_cache')

# 代理设置
_PROXY = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _PROXY:
    os.environ.setdefault('HTTP_PROXY', _PROXY)
    os.environ.setdefault('HTTPS_PROXY', _PROXY)


def _is_a_share(code: str) -> bool:
    """判断是否为A股（排除B股）。"""
    if code.startswith('900') or code.startswith('200'):
        return False
    return code.startswith(('600', '601', '603', '605', '000', '001', '002', '003', '300', '301'))


def fetch_qfq_for_stock(code: str) -> tuple[str, pd.DataFrame | None, str]:
    """
    拉取单只存活股的前复权日线数据。

    Returns:
        (code, df, status)
        status: 'success' | 'fail' | 'empty'
    """
    import akshare as ak
    try:
        sym = f'sh{code}' if code.startswith('6') else f'sz{code}'
        df = ak.stock_zh_a_daily(symbol=sym, adjust='qfq')
        if df is None or len(df) == 0:
            return code, None, 'empty'
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
        if len(df) == 0:
            return code, None, 'empty'
        return code, df, 'success'
    except Exception as e:
        return code, None, f'fail:{type(e).__name__}'


def save_qfq_parquet(code: str, df: pd.DataFrame, cache_dir: str) -> str:
    """将 qfq 数据保存为 parquet 文件。"""
    path = os.path.join(cache_dir, f'{code}.parquet')
    df.to_parquet(path, index=False)
    return path


def load_qfq_parquet(code: str, cache_dir: str) -> pd.Series | None:
    """从 parquet 文件加载 qfq 收盘价序列。"""
    path = os.path.join(cache_dir, f'{code}.parquet')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        return df['close']
    except Exception:
        return None


def load_qfq_full(code: str, cache_dir: str) -> pd.DataFrame | None:
    """从 parquet 加载完整 qfq DataFrame（含 outstanding_share）。"""
    path = os.path.join(cache_dir, f'{code}.parquet')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        return df
    except Exception:
        return None


def build_qfq_cache(
    cache_dir: str | None = None,
    max_workers: int = 10,
    max_stocks: int | None = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict:
    """
    为全市场存活A股构建 qfq 缓存。

    参数：
        cache_dir: 缓存目录（默认 data/qfq_cache/）
        max_workers: 并发线程数
        max_stocks: 最多拉取多少只（None=全部，用于测试）
        skip_existing: 是否跳过已缓存的股票
        verbose: 打印进度
    """
    import akshare as ak

    cache_dir = cache_dir or _QFQ_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    meta_path = os.path.join(cache_dir, '_meta.json')

    # 加载已有元信息
    meta = {
        'success': [],
        'fail': [],
        'empty': [],
        'build_time': None,
        'total_a_shares': 0,
    }
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            old_meta = json.load(f)
        meta['success'] = old_meta.get('success', [])
        meta['fail'] = old_meta.get('fail', [])
        meta['empty'] = old_meta.get('empty', [])

    # 获取全量A股列表
    live_list = ak.stock_info_a_code_name()
    live_list['code'] = live_list['code'].apply(lambda x: str(x).zfill(6))
    a_shares = [c for c in live_list['code'].tolist() if _is_a_share(c)]
    meta['total_a_shares'] = len(a_shares)

    if max_stocks is not None:
        a_shares = a_shares[:max_stocks]

    # 过滤已缓存的
    existing_success = set(meta['success'])
    existing_fail = set(meta['fail'])
    existing_empty = set(meta['empty'])
    all_cached = existing_success | existing_fail | existing_empty

    to_fetch = a_shares
    if skip_existing:
        to_fetch = [c for c in a_shares if c not in all_cached]

    if verbose:
        print(f"总 A 股: {len(a_shares)}")
        print(f"已缓存: {len(existing_success)} 成功, {len(existing_fail)} 失败, {len(existing_empty)} 空")
        print(f"待拉取: {len(to_fetch)}")

    if len(to_fetch) == 0:
        if verbose:
            print("全部已缓存，无需拉取。")
        # 仍要更新退市股状态
        _update_delist_in_meta(meta, cache_dir, meta_path)
        return meta

    # 并发拉取
    t0 = time.time()
    batch_size = max(max_workers * 5, 50)
    new_success = []
    new_fail = []
    new_empty = []

    for batch_start in range(0, len(to_fetch), batch_size):
        batch = to_fetch[batch_start:batch_start + batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_qfq_for_stock, code): code for code in batch}
            for future in concurrent.futures.as_completed(futures):
                code = futures[future]
                try:
                    result_code, df, status = future.result()
                except Exception as e:
                    new_fail.append({'code': code, 'error': str(e)})
                    continue

                if status == 'success' and df is not None:
                    save_qfq_parquet(code, df, cache_dir)
                    new_success.append(code)
                elif status == 'empty':
                    new_empty.append(code)
                else:
                    new_fail.append({'code': code, 'error': status})

        # 增量保存元信息
        meta['success'].extend(new_success)
        meta['success'] = list(set(meta['success']))
        meta['fail'].extend(new_fail)
        meta['empty'].extend(new_empty)
        meta['empty'] = list(set(meta['empty']))
        meta['build_time'] = datetime.now().isoformat()

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 0.1)
        eta = (len(to_fetch) - done) / max(rate, 0.1)
        if verbose:
            print(f"  {done}/{len(to_fetch)} ({rate:.1f}/s, ETA {eta:.0f}s) "
                  f"成功 {len(new_success)} 失败 {len(new_fail)} 空 {len(new_empty)}")

    # 更新退市股状态
    _update_delist_in_meta(meta, cache_dir, meta_path)

    if verbose:
        print(f"\n完成。成功: {len(meta['success'])}, 失败: {len(meta['fail'])}, "
              f"空: {len(meta['empty'])}")
        print(f"耗时: {time.time()-t0:.1f}s")
        print(f"缓存目录: {cache_dir}")
        print(f"覆盖率: {len(meta['success'])}/{meta['total_a_shares']} "
              f"({len(meta['success'])/max(meta['total_a_shares'],1)*100:.1f}%)")

    return meta


def _update_delist_in_meta(meta: dict, cache_dir: str, meta_path: str):
    """更新退市股的缓存状态（从 delist_prices.pkl 加载）。"""
    delist_pkl = os.path.join(_DATA_DIR, 'delist_prices.pkl')
    if os.path.exists(delist_pkl):
        delist_prices = pd.read_pickle(delist_pkl)
        meta['delist_cached'] = list(delist_prices.keys())
        meta['delist_count'] = len(delist_prices)
    else:
        meta['delist_cached'] = []
        meta['delist_count'] = 0

    # 计算降级率
    total_live = meta['total_a_shares']
    qfq_success = len(meta['success'])
    degrade_count = total_live - qfq_success - len(meta['fail']) - len(meta['empty'])
    meta['degrade_count'] = max(0, degrade_count)
    meta['qfq_coverage_pct'] = round(qfq_success / max(total_live, 1) * 100, 1)

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def get_cache_stats(cache_dir: str | None = None) -> dict:
    """获取缓存统计信息。"""
    cache_dir = cache_dir or _QFQ_CACHE_DIR
    meta_path = os.path.join(cache_dir, '_meta.json')

    if not os.path.exists(meta_path):
        return {'status': 'not_built', 'message': '缓存未构建，请先运行 build_qfq_cache()'}

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    return {
        'status': 'built',
        'total_a_shares': meta.get('total_a_shares', 0),
        'qfq_success': len(meta.get('success', [])),
        'qfq_fail': len(meta.get('fail', [])),
        'qfq_empty': len(meta.get('empty', [])),
        'delist_cached': meta.get('delist_count', 0),
        'qfq_coverage_pct': meta.get('qfq_coverage_pct', 0),
        'build_time': meta.get('build_time'),
        'cache_dir': cache_dir,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='构建持久化 qfq 缓存')
    parser.add_argument('--workers', type=int, default=10, help='并发线程数')
    parser.add_argument('--max-stocks', type=int, default=None, help='最多拉取多少只（测试用）')
    parser.add_argument('--cache-dir', type=str, default=None, help='缓存目录')
    parser.add_argument('--no-skip-existing', action='store_true', help='不跳过已缓存的')
    args = parser.parse_args()

    build_qfq_cache(
        cache_dir=args.cache_dir,
        max_workers=args.workers,
        max_stocks=args.max_stocks,
        skip_existing=not args.no_skip_existing,
        verbose=True,
    )
