"""
Step 2 (修正): 用腾讯API批量拉取退市股日线数据
直接调用 web.ifzq.gtimg.cn 前复权接口，速度快
"""
import requests
import pandas as pd
import numpy as np
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'

DATA_DIR = 'PROJECT_ROOT/data'

# 读取退市股列表
with open(f'{DATA_DIR}/delist_codes.json') as f:
    delist_codes = json.load(f)

# 获取上交所退市日期信息
import akshare as ak
sh_delist = ak.stock_info_sh_delist()
sh_delist['公司代码'] = sh_delist['公司代码'].apply(lambda x: str(x).zfill(6))
sh_delist['暂停上市日期'] = pd.to_datetime(sh_delist['暂停上市日期'], errors='coerce')

def get_tx_kline_fast(code):
    """腾讯前复权日线API，一次请求返回全部"""
    sym = f'sh{code}' if code.startswith('6') else f'sz{code}'
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
delist_prices = {}
delist_info = {}
success = 0
fail = 0

for i, code in enumerate(delist_codes):
    if i % 20 == 0:
        print(f"  进度: {i}/{len(delist_codes)}, 成功{success}, 失败{fail}")
    
    try:
        s = get_tx_kline_fast(code)
        if s is not None and len(s) > 0:
            # 过滤2020年后的数据
            s = s[s.index >= '2020-01-01']
            
            if len(s) == 0:
                fail += 1
                continue
            
            row = sh_delist[sh_delist['公司代码'] == code]
            if len(row) > 0:
                name = row.iloc[0]['公司简称']
                ddate = row.iloc[0]['暂停上市日期']
            else:
                name = code
                ddate = s.index[-1]
            
            delist_prices[code] = s
            delist_info[code] = {
                'name': str(name),
                'delist_date': str(ddate.date()) if pd.notna(ddate) else None,
                'data_start': str(s.index[0].date()),
                'data_end': str(s.index[-1].date()),
                'n_days': len(s),
                'last_price': float(s.iloc[-1]),
            }
            success += 1
        else:
            fail += 1
    except Exception as e:
        fail += 1
    
    time.sleep(0.15)  # 避免请求过快

print(f"\n获取完成: 成功{success}, 失败{fail}")
print(f"成功率: {success}/{len(delist_codes)} = {success/len(delist_codes):.1%}")

# 保存
pd.to_pickle(delist_prices, f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'w', encoding='utf-8') as f:
    json.dump(delist_info, f, ensure_ascii=False, indent=2)

print(f"\n价格数据: {DATA_DIR}/delist_prices.pkl")
print(f"元数据: {DATA_DIR}/delist_info.json")

# 打印成功获取的
print(f"\n=== 成功获取的退市股 ({success}只) ===")
for code, info in sorted(delist_info.items(), key=lambda x: x[1]['data_end']):
    print(f"  {code} {info['name']:8s} 数据:{info['data_start']}~{info['data_end']} ({info['n_days']}天) 退市:{info['delist_date']} 末价:{info['last_price']:.2f}")
