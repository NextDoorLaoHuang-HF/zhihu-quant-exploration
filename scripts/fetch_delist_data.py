"""
Step 2: 批量拉取退市股日线数据（东财前复权）
对每只退市股尝试获取2020-01-01至今的日线数据
退市后数据终止，策略中退市后收益记为-100%（价格归零）
"""
import akshare as ak
import pandas as pd
import numpy as np
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# 代理设置：从环境变量读取，不硬编码
_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _proxy:
    os.environ.setdefault('HTTP_PROXY', _proxy)
    os.environ.setdefault('HTTPS_PROXY', _proxy)

# 读取退市股列表
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'delist_codes.json')) as f:
    delist_codes = json.load(f)

print(f"退市股票数: {len(delist_codes)}")

# 获取上交所退市日期信息
sh_delist = ak.stock_info_sh_delist()
sh_delist['公司代码'] = sh_delist['公司代码'].apply(lambda x: str(x).zfill(6))
sh_delist['暂停上市日期'] = pd.to_datetime(sh_delist['暂停上市日期'], errors='coerce')

# 批量拉取日线数据
delist_prices = {}  # code -> pd.Series of close prices
delist_info = {}    # code -> {name, delist_date, data_start, data_end, n_days}

success = 0
fail = 0
for i, code in enumerate(delist_codes):
    if i % 20 == 0:
        print(f"  进度: {i}/{len(delist_codes)}, 成功{success}, 失败{fail}")
    
    try:
        # 东财前复权日线
        df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                 start_date='20200101', end_date='20260713',
                                 adjust='qfq')
        if df is None or len(df) == 0:
            # 尝试不复权
            df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                     start_date='20200101', end_date='20260713')
        
        if df is None or len(df) == 0:
            fail += 1
            continue
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        df.set_index('日期', inplace=True)
        
        close = df['收盘'].astype(float)
        
        # 获取退市日期
        row = sh_delist[sh_delist['公司代码'] == code]
        if len(row) > 0:
            name = row.iloc[0]['公司简称']
            ddate = row.iloc[0]['暂停上市日期']
        else:
            name = code
            ddate = close.index[-1]
        
        delist_prices[code] = close
        delist_info[code] = {
            'name': name,
            'delist_date': str(ddate.date()) if pd.notna(ddate) else None,
            'data_start': str(close.index[0].date()),
            'data_end': str(close.index[-1].date()),
            'n_days': len(close),
        }
        success += 1
        
        # 避免请求过快
        time.sleep(0.3)
        
    except Exception as e:
        fail += 1
        continue

print(f"\n获取完成: 成功{success}, 失败{fail}")
print(f"成功率: {success}/{len(delist_codes)} = {success/len(delist_codes):.1%}")

# 保存价格数据
prices_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'delist_prices.pkl')
pd.to_pickle(delist_prices, prices_path)
print(f"价格数据已保存: {prices_path}")

# 保存元数据
info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'delist_info.json')
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(delist_info, f, ensure_ascii=False, indent=2)
print(f"元数据已保存: {info_path}")

# 打印成功获取的退市股
print(f"\n=== 成功获取的退市股 ({success}只) ===")
for code, info in sorted(delist_info.items(), key=lambda x: x[1]['data_end']):
    print(f"  {code} {info['name']:8s} 数据:{info['data_start']}~{info['data_end']} ({info['n_days']}天) 退市:{info['delist_date']}")
