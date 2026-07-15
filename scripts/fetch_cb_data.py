"""
scripts/fetch_cb_data.py — 拉取全量可转债日线数据（含已退出券）

输出：
  04.a-share/cb_cache/cb_daily_all.pkl     — dict{code: DataFrame(date, open, high, low, close, volume)}
  04.a-share/cb_cache/cb_meta_all.pkl      — dict{code: {listing_date, delist_date, exit_reason, exit_final_price}}
  04.a-share/cb_cache/cb_list_all.pkl     — 全量可转债列表 DataFrame

数据源：
  - ak.bond_zh_cov() 获取全量可转债列表（含正股代码、上市时间）
  - ak.bond_zh_hs_cov_daily(symbol=sh/sz+code) 获取单只可转债日K线
  - 已退出券的最后交易日即为退出日，最后收盘价即为退出最终价
"""
import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

import akshare as ak
import pandas as pd
import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CB_CACHE_DIR = os.path.join(PROJECT_ROOT, '04.a-share', 'cb_cache')

os.makedirs(CB_CACHE_DIR, exist_ok=True)


def fetch_all_cb_data():
    """拉取全量可转债日线数据，含已退出券。"""

    # 1. 获取全量可转债列表
    print('1. 获取全量可转债列表...')
    cb_list = ak.bond_zh_cov()
    print(f'   总数: {len(cb_list)}')

    # 保存列表
    list_path = os.path.join(CB_CACHE_DIR, 'cb_list_all.pkl')
    with open(list_path, 'wb') as f:
        pickle.dump(cb_list, f)
    print(f'   保存到 {list_path}')

    # 2. 筛选目标债券：不按上市时间截断，纳入2024+新上市券
    # 旧版硬截断 listing_dates < 2024-01-01 会漏掉2024年后上市的新券
    codes = cb_list['债券代码'].astype(str).tolist()
    print(f'   全量可转债: {len(codes)} 只')

    # 3. 检查已有缓存，跳过已拉取的
    daily_path = os.path.join(CB_CACHE_DIR, 'cb_daily_all.pkl')
    meta_path = os.path.join(CB_CACHE_DIR, 'cb_meta_all.pkl')

    existing_daily = {}
    existing_meta = {}
    if os.path.exists(daily_path):
        with open(daily_path, 'rb') as f:
            existing_daily = pickle.load(f)
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            existing_meta = pickle.load(f)

    to_fetch = [c for c in codes if c not in existing_daily]
    print(f'   已缓存: {len(existing_daily)} 只, 待拉取: {len(to_fetch)} 只')

    # 4. 拉取日线
    daily_data = existing_daily
    meta_data = existing_meta

    # 从cb_list构建快速查找
    cb_dict = {}
    for _, row in cb_list.iterrows():
        code = str(row['债券代码'])
        cb_dict[code] = {
            'bond_name': row.get('债券简称', ''),
            'stock_code': str(row.get('正股代码', '')),
            'listing_date': str(row.get('上市时间', '')) if pd.notna(row.get('上市时间')) else None,
        }

    fetched = 0
    failed = 0
    t0 = time.time()

    for i, code in enumerate(to_fetch):
        prefix = 'sh' if code.startswith('11') else 'sz'
        full = f'{prefix}{code}'

        try:
            df = ak.bond_zh_hs_cov_daily(symbol=full)
            if df is None or len(df) == 0:
                failed += 1
                continue

            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            daily_data[code] = df

            # 构建元信息
            first_date = df.iloc[0]['date']
            last_date = df.iloc[-1]['date']
            last_close = float(df.iloc[-1]['close'])

            cb_info = cb_dict.get(code, {})
            listing_date = cb_info.get('listing_date')

            # 判断是否退出：如果最后交易日早于今天（且超过30天），则已退出
            today = pd.Timestamp.now().normalize()
            if (today - last_date).days > 30:
                delist_date = str(last_date.date())
                # 退出原因未知，但最后收盘价作为退出最终价
                exit_final_price = last_close
                # 推断退出原因
                exit_reason = '退出（具体原因未知）'
            else:
                delist_date = None
                exit_final_price = None
                exit_reason = None

            meta_data[code] = {
                'listing_date': str(listing_date) if listing_date else str(first_date.date()),
                'delist_date': delist_date,
                'exit_reason': exit_reason,
                'exit_final_price': exit_final_price,
                'bond_name': cb_info.get('bond_name', ''),
                'stock_code': cb_info.get('stock_code', ''),
            }

            fetched += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f'   {full} 失败: {e}')

        # 进度报告
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(to_fetch) - i - 1) / rate
            print(f'   进度: {i+1}/{len(to_fetch)} ({fetched}成功, {failed}失败), '
                  f'{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining')

            # 增量保存
            with open(daily_path, 'wb') as f:
                pickle.dump(daily_data, f)
            with open(meta_path, 'wb') as f:
                pickle.dump(meta_data, f)

    # 最终保存
    with open(daily_path, 'wb') as f:
        pickle.dump(daily_data, f)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)

    elapsed = time.time() - t0
    print(f'\n2. 拉取完成: {fetched} 成功, {failed} 失败, 耗时 {elapsed:.0f}s')
    print(f'   日线数据: {daily_path} ({len(daily_data)} 只)')
    print(f'   元信息:   {meta_path} ({len(meta_data)} 只)')

    # 3. 统计
    print('\n3. 统计:')
    active = sum(1 for v in meta_data.values() if v.get('delist_date') is None)
    exited = sum(1 for v in meta_data.values() if v.get('delist_date') is not None)
    print(f'   存续: {active} 只')
    print(f'   已退出: {exited} 只')
    print(f'   总计: {len(meta_data)} 只')

    # 时间范围
    all_dates = []
    for df in daily_data.values():
        if len(df) > 0:
            all_dates.append(df.iloc[0]['date'])
            all_dates.append(df.iloc[-1]['date'])
    if all_dates:
        print(f'   数据时间范围: {min(all_dates).date()} ~ {max(all_dates).date()}')

    return daily_data, meta_data


if __name__ == '__main__':
    fetch_all_cb_data()
