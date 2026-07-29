"""
Step 2b: 用腾讯API批量拉取深交所退市股日线数据
补充深交所退市股（0/3开头），与上交所退市股合并
"""
import requests
import pandas as pd
import numpy as np
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

# 代理设置：从环境变量读取，不硬编码
_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _proxy:
    os.environ.setdefault('HTTP_PROXY', _proxy)
    os.environ.setdefault('HTTPS_PROXY', _proxy)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

import akshare as ak

# 获取深交所退市股列表
sz_delist = ak.stock_info_sz_delist(symbol='终止上市公司')
sz_delist['证券代码'] = sz_delist['证券代码'].apply(lambda x: str(x).zfill(6))
sz_delist['终止上市日期'] = pd.to_datetime(sz_delist['终止上市日期'], errors='coerce')

# 只取2020年后退市的（与回测区间重叠）
sz_recent = sz_delist[sz_delist['终止上市日期'] >= '2020-01-01'].copy()
print(f"深交所2020年后退市: {len(sz_recent)}只")

sz_codes = sz_recent['证券代码'].tolist()
print(f"代码示例: {sz_codes[:10]}")

def get_tx_kline_fast(code):
    """腾讯前复权日线API"""
    sym = f'sz{code}' if code.startswith(('0', '3')) else f'sh{code}'
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    params = {'param': f'{sym},day,2020-01-01,2026-07-13,640,qfq'}
    
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    
    klines = data.get('data', {}).get(sym, {}).get('qfqday', [])
    if not klines:
        klines = data.get('data', {}).get(sym, {}).get('day', [])
    
    if not klines:
        return None
    
    df = pd.DataFrame(klines, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].astype(float)
    df = df.sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    return df['close']

# 批量拉取
sz_delist_prices = {}
sz_delist_info = {}
success = 0
fail = 0

for i, (_, row) in enumerate(sz_recent.iterrows()):
    code = row['证券代码']
    name = row['证券简称']
    ddate = row['终止上市日期']
    
    if i % 20 == 0:
        print(f"  进度: {i}/{len(sz_recent)}, 成功{success}, 失败{fail}")
    
    try:
        s = get_tx_kline_fast(code)
        if s is not None and len(s) > 0:
            s = s[s.index >= '2020-01-01']
            if len(s) == 0:
                fail += 1
                continue
            
            sz_delist_prices[code] = s
            sz_delist_info[code] = {
                'name': str(name),
                'delist_date': str(ddate.date()) if pd.notna(ddate) else None,
                'data_start': str(s.index[0].date()),
                'data_end': str(s.index[-1].date()),
                'n_days': len(s),
                'last_price': float(s.iloc[-1]),
                'exchange': 'SZ',
            }
            success += 1
        else:
            fail += 1
    except:
        fail += 1
    
    time.sleep(0.15)

print(f"\n深交所退市股获取完成: 成功{success}, 失败{fail}")

# 合并到已有的退市股数据中
# 加载上交所退市股
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    old_info = json.load(f)
old_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')

print(f"\n原有上交所退市股: {len(old_info)}只")

# 合并
all_info = dict(old_info)  # 上交所
all_info.update(sz_delist_info)  # 加深交所

all_prices = dict(old_prices)
all_prices.update(sz_delist_prices)

print(f"合并后退市股总数: {len(all_info)}只")
print(f"  上交所: {sum(1 for v in all_info.values() if v.get('exchange','SH') == 'SH')}")
print(f"  深交所: {sum(1 for v in all_info.values() if v.get('exchange','SZ') == 'SZ')}")

# 给上交所退市股加exchange标记
for code, info in old_info.items():
    if 'exchange' not in info:
        info['exchange'] = 'SH'

# 保存合并后的数据
pd.to_pickle(all_prices, f'{DATA_DIR}/delist_prices_all.pkl')
with open(f'{DATA_DIR}/delist_info_all.json', 'w', encoding='utf-8') as f:
    json.dump(all_info, f, ensure_ascii=False, indent=2)

# 同时更新原文件（覆盖）
pd.to_pickle(all_prices, f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'w', encoding='utf-8') as f:
    json.dump(all_info, f, ensure_ascii=False, indent=2)

print(f"\n已保存合并后的数据:")
print(f"  {DATA_DIR}/delist_prices.pkl ({len(all_prices)}只)")
print(f"  {DATA_DIR}/delist_info.json ({len(all_info)}只)")
print(f"  {DATA_DIR}/delist_prices_all.pkl (备份)")
print(f"  {DATA_DIR}/delist_info_all.json (备份)")

print(f"\n=== 深交所退市股明细 ===")
for code, info in sorted(sz_delist_info.items(), key=lambda x: x[1].get('delist_date','')):
    print(f"  {code} {info['name']:10s} 退市:{info['delist_date']} 末价:{info['last_price']:.2f}")
