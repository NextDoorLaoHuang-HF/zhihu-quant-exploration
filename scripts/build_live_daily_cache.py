"""
并行构建全市场存活股日线缓存（raw close + outstanding_share）

用多线程并发拉取全市场存活A股的日线数据（未复权），
保存为 parquet 格式（避免 pickle 的 pandas 版本兼容问题）。

用法：
    python scripts/build_live_daily_cache.py [--workers 10] [--max-stocks N]

输出：
    data/live_daily_cache/        — 每只股票一个 parquet 文件
    data/live_daily_cache/_meta.json  — 缓存元信息
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
_LIVE_CACHE_DIR = os.path.join(_DATA_DIR, 'live_daily_cache')

_PROXY = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _PROXY:
    os.environ.setdefault('HTTP_PROXY', _PROXY)
    os.environ.setdefault('HTTPS_PROXY', _PROXY)


def _is_a_share(code: str) -> bool:
    if code.startswith('900') or code.startswith('200'):
        return False
    return code.startswith(('600', '601', '603', '605', '000', '001', '002', '003', '300', '301'))


def fetch_raw_for_stock(code: str) -> tuple[str, pd.DataFrame | None, str]:
    """
    拉取单只存活股的未复权日线数据。

    Returns:
        (code, df, status)
        status: 'success' | 'fail' | 'empty'
    """
    import akshare as ak
    try:
        sym = f'sh{code}' if code.startswith('6') else f'sz{code}'
        df = ak.stock_zh_a_daily(symbol=sym)
        if df is None or len(df) == 0:
            return code, None, 'empty'
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
        if len(df) == 0:
            return code, None, 'empty'
        return code, df, 'success'
    except Exception as e:
        return code, None, f'fail:{type(e).__name__}'


def build_live_daily_cache(
    cache_dir: str | None = None,
    max_workers: int = 10,
    max_stocks: int | None = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict:
    """
    为全市场存活A股构建 raw daily 缓存（parquet 格式）。
    """
    import akshare as ak

    cache_dir = cache_dir or _LIVE_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    meta_path = os.path.join(cache_dir, '_meta.json')

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

    existing_success = set(meta['success'])
    existing_fail = set(meta.get('fail', []))
    existing_empty = set(meta.get('empty', []))
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
        meta['build_time'] = datetime.now().isoformat()
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    t0 = time.time()
    batch_size = max(max_workers * 5, 50)
    new_success = []
    new_fail = []
    new_empty = []

    for batch_start in range(0, len(to_fetch), batch_size):
        batch = to_fetch[batch_start:batch_start + batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_raw_for_stock, code): code for code in batch}
            for future in concurrent.futures.as_completed(futures):
                code = futures[future]
                try:
                    result_code, df, status = future.result()
                except Exception as e:
                    new_fail.append({'code': code, 'error': str(e)})
                    continue

                if status == 'success' and df is not None:
                    path = os.path.join(cache_dir, f'{code}.parquet')
                    df.to_parquet(path, index=False)
                    new_success.append(code)
                elif status == 'empty':
                    new_empty.append(code)
                else:
                    new_fail.append({'code': code, 'error': status})

        # 增量保存元信息
        meta['success'].extend(new_success)
        meta['success'] = list(set(meta['success']))
        meta['fail'] = new_fail[:]
        meta['empty'] = list(set(meta.get('empty', []) + new_empty))
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

    if verbose:
        print(f"\n完成。成功: {len(meta['success'])}, 失败: {len(meta['fail'])}, "
              f"空: {len(meta.get('empty', []))}")
        print(f"耗时: {time.time()-t0:.1f}s")
        print(f"缓存目录: {cache_dir}")
        print(f"覆盖率: {len(meta['success'])}/{meta['total_a_shares']} "
              f"({len(meta['success'])/max(meta['total_a_shares'],1)*100:.1f}%)")

    return meta


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='构建存活股日线缓存（raw close）')
    parser.add_argument('--workers', type=int, default=10, help='并发线程数')
    parser.add_argument('--max-stocks', type=int, default=None, help='最多拉取多少只（测试用）')
    parser.add_argument('--cache-dir', type=str, default=None, help='缓存目录')
    parser.add_argument('--no-skip-existing', action='store_true', help='不跳过已缓存的')
    args = parser.parse_args()

    build_live_daily_cache(
        cache_dir=args.cache_dir,
        max_workers=args.workers,
        max_stocks=args.max_stocks,
        skip_existing=not args.no_skip_existing,
        verbose=True,
    )
