"""
T5小市值策略改进测试：
1. 排除退市股（baseline，已完成）
2. +排除ST/*ST
3. +排除股价<1元
4. +排除亏损股（负净利润，需东财接口）
5. +换市值排序替代价格排序（需流通市值数据）
"""
import akshare as ak
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'

DATA_DIR = 'PROJECT_ROOT/data'

# 加载退市股
delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)

# 存活股
stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

survive_prices = {}
survive_names = {}
for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0: print(f"  拉取进度: {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            survive_prices[row['code']] = df['close']
            survive_names[row['code']] = row['name']
    except: pass

# 加入6只退市股（真实比例）
np.random.seed(42)
delist_sampled_codes = np.random.choice(list(delist_prices.keys()), 6, replace=False)
all_dates = pd.date_range('2020-01-01', '2026-07-13', freq='D')

delist_panel = {}
for code in delist_sampled_codes:
    s = delist_prices[code].reindex(all_dates)
    info = delist_info.get(code, {})
    dd = info.get('delist_date')
    if dd: s[s.index > pd.to_datetime(dd)] = np.nan
    if s.notna().any(): delist_panel[code] = s

prices = pd.DataFrame({**survive_prices, **delist_panel}).dropna(how='all')
print(f"\n候选池: {prices.shape[1]}只 (存活{len(survive_prices)} + 退市6)")

# ============================================================
# 简化回测：用月频跑T5，对比不同过滤条件
# ============================================================

def run_t5_filtered(prices_df, filter_price_min=None, n_stocks=5, 
                     exclude_st_name=None):
    """T5策略，可选过滤"""
    mp = prices_df.resample('M').last()
    mr = mp.pct_change()
    
    rets, dates = [], []
    for i in range(12, len(mp) - 1):
        row = mp.iloc[i].dropna()
        
        # 价格过滤
        if filter_price_min:
            row = row[row >= filter_price_min]
        
        # ST过滤（名称中含ST/*ST）
        if exclude_st_name:
            valid = []
            for code in row.index:
                name = survive_names.get(code, delist_info.get(code, {}).get('name', ''))
                if 'ST' not in str(name).upper():
                    valid.append(code)
            row = row[valid]
        
        if len(row) < n_stocks + 5: continue
        
        cheapest = row.nsmallest(n_stocks).index
        r = mr.iloc[i+1].reindex(cheapest).mean()
        rets.append(r)
        dates.append(mr.index[i+1])
    
    if len(rets) < 12: return None
    s = pd.Series(rets, index=dates)
    return {
        'annual': s.mean() * 12,
        'sharpe': s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0,
        'dd': ((1+s).cumprod() / (1+s).cumprod().cummax() - 1).min(),
        'n': len(s),
    }

# 测试不同过滤条件
print("\n" + "=" * 70)
print("T5改进方案对比（含6只退市股）")
print("=" * 70)

# 对T5、T10、T20各跑一遍
scenarios = [
    ('T5_无过滤', 5, {}),
    ('T5_排除<1元', 5, {'filter_price_min': 1.0}),
    ('T5_排除<2元', 5, {'filter_price_min': 2.0}),
    ('T5_排除ST', 5, {'exclude_st_name': True}),
    ('T5_排除ST+<1元', 5, {'filter_price_min': 1.0, 'exclude_st_name': True}),
    ('T5_排除ST+<2元', 5, {'filter_price_min': 2.0, 'exclude_st_name': True}),
    # T10对比
    ('T10_无过滤', 10, {}),
    ('T10_排除ST+<2元', 10, {'filter_price_min': 2.0, 'exclude_st_name': True}),
]

print(f"  {'方案':20s} | {'年化':>7s} | {'夏普':>6s} | {'回撤':>7s} | {'月数'}")
print(f"  {'-'*55}")
best = {}
for label, n, kwargs in scenarios:
    r = run_t5_filtered(prices, n_stocks=n, **kwargs)
    if r:
        print(f"  {label:20s} | {r['annual']:6.1%} | {r['sharpe']:5.2f} | {r['dd']:6.1%} | {r['n']:3d}")
        best[label] = r

print(f"\n=== 改进效果总结 ===")
print("  1. 排除ST/*ST：T5从10.7%→?，排除退市前ST股")
print("  2. 排除<1元：T5从10.7%→?，排除面值退市风险")
print("  3. 双重过滤：理论上应拿到'健康微盘股'的alpha")
print("  4. T10天然抗退市影响（选股范围宽），加上过滤可能更稳")
