"""
Step 1: 获取上交所+深交所完整退市股票列表
"""
import akshare as ak
import pandas as pd
import os
import json
import warnings
import time
warnings.filterwarnings('ignore')

# 代理设置：从环境变量读取，不硬编码
_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _proxy:
    os.environ.setdefault('HTTP_PROXY', _proxy)
    os.environ.setdefault('HTTPS_PROXY', _proxy)

print("=== 获取退市股票完整列表 ===")

# 1. 上交所退市
sh_delist = ak.stock_info_sh_delist()
print(f"上交所退市: {len(sh_delist)}只")
sh_codes = sh_delist['公司代码'].apply(lambda x: str(x).zfill(6)).tolist()

# 2. 深交所退市 — 尝试不同参数
sz_delist = None
for param in ['终止上市', '暂停上市', '退市']:
    try:
        df = ak.stock_info_sz_delist(symbol=param)
        print(f"深交所({param}): {len(df)}只, 列名={list(df.columns)}")
        if len(df) > 0:
            if sz_delist is None:
                sz_delist = df
            else:
                sz_delist = pd.concat([sz_delist, df]).drop_duplicates()
    except Exception as e:
        print(f"深交所({param}): {e}")

if sz_delist is not None:
    sz_codes = []
    for col in sz_delist.columns:
        if '代码' in col:
            sz_codes = sz_delist[col].apply(lambda x: str(x).zfill(6)).tolist()
            print(f"深交所退市代码({col}): {len(sz_codes)}只")
            print(f"  示例: {sz_codes[:5]}")
            break
else:
    print("深交所退市获取失败，仅用上交所")
    sz_codes = []

# 合并
all_delist_codes = list(set(sh_codes + sz_codes))
print(f"\n退市股票总数: {len(all_delist_codes)}")
print(f"  上交所: {len(sh_codes)}")
print(f"  深交所: {len(sz_codes)}")

# 保存
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'delist_codes.json'), 'w') as f:
    json.dump(all_delist_codes, f)
print(f"已保存到 data/delist_codes.json")

# 筛选2020年后可能还在交易的（退市日期在2020后的更可能影响回测）
# 上交所退市有"暂停上市日期"
if '暂停上市日期' in sh_delist.columns:
    sh_delist['暂停上市日期'] = pd.to_datetime(sh_delist['暂停上市日期'], errors='coerce')
    recent_sh = sh_delist[sh_delist['暂停上市日期'] >= '2020-01-01']
    print(f"\n上交所2020后退市: {len(recent_sh)}只")
    print(recent_sh[['公司代码', '公司简称', '暂停上市日期']].head(10).to_string())
