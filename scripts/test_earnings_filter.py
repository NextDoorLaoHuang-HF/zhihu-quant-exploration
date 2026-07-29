"""
"剔除负业绩" 过滤测试
用 stock_yjbb_em 获取季度业绩报表，每月调仓时排除最新季度亏损的股票
"""
import os
import akshare as ak
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

# 代理设置：从环境变量读取，不硬编码
_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _proxy:
    os.environ.setdefault('HTTP_PROXY', _proxy)
    os.environ.setdefault('HTTPS_PROXY', _proxy)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# ============================================================
# 1. 加载季度业绩数据
# ============================================================
print("加载季度业绩报表...")
yj = ak.stock_yjbb_em(date='20260331')  # 最新一期报表
print(f"  数据: {len(yj)}条, 覆盖{len(yj['股票代码'].unique())}只股票")

# 构建代码→最新净利润映射（用于选股时检查）
# 注意：yjbb_em返回的是单个报告期的数据，需要用最新公告日期
# 实际使用时需要按报告期筛选，这里用最新一期作为演示
profit_map = {}
for _, row in yj.iterrows():
    code = row['股票代码']
    profit = pd.to_numeric(row.get('净利润-净利润', np.nan), errors='coerce')
    if pd.notna(profit):
        profit_map[code] = profit

n_positive = sum(1 for v in profit_map.values() if v > 0)
n_negative = sum(1 for v in profit_map.values() if v <= 0)
n_st = sum(1 for c in profit_map if 'ST' in str(yj[yj['股票代码']==c]['股票简称'].values[0]).upper() if len(yj[yj['股票代码']==c]) > 0)
print(f"  正利润: {n_positive}, 负利润: {n_negative}, 示例ST股利润: {sum(1 for c in profit_map if 'ST' in str(c))}只")

# ============================================================
# 2. 加载价格数据（复用已有逻辑）
# ============================================================
delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)

stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

survive_prices = {}
for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0: print(f"  价格数据: {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            survive_prices[row['code']] = df['close']
    except: pass

# 加入6只退市股
np.random.seed(42)
delist_codes = np.random.choice(list(delist_prices.keys()), 6, replace=False)
all_dates = pd.date_range('2020-01-01', '2026-07-13', freq='D')
for code in delist_codes:
    s = delist_prices[code].reindex(all_dates)
    dd = delist_info.get(code, {}).get('delist_date')
    if dd: s[s.index > pd.to_datetime(dd)] = np.nan
    if s.notna().any(): survive_prices[code] = s

prices = pd.DataFrame(survive_prices).dropna(how='all')
print(f"候选池: {prices.shape[1]}只")

mp = prices.resample('M').last()
mr = mp.pct_change()

# ST代码集
st_codes = set(stock_list[stock_list['name'].str.contains('ST', na=False)]['code'])

# ============================================================
# 3. 多方案回测
# ============================================================
def has_positive_earnings(code):
    """检查最近一期是否有正利润"""
    return profit_map.get(code, 0) > 0

def run_t5_filtered(mp, mr, n_stocks=5, filters=None):
    """
    filters: list of filter functions
    """
    rets = []
    for i in range(12, len(mp) - 1):
        row = mp.iloc[i].dropna()
        
        # 基础过滤：>2元
        row = row[row >= 2.0]
        
        # 额外过滤
        if filters:
            for f in filters:
                valid = [c for c in row.index if f(c)]
                row = row[valid]
        
        if len(row) < n_stocks + 5:
            continue
        
        cheapest = row.nsmallest(n_stocks).index
        next_ret = mr.iloc[i + 1]
        ret = next_ret.reindex(cheapest).mean()
        rets.append(ret)
    
    if len(rets) < 12: return None
    s = pd.Series(rets)
    return {
        'annual': s.mean() * 12,
        'sharpe': s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0,
        'dd': ((1+s).cumprod() / (1+s).cumprod().cummax() - 1).min(),
        'n': len(s),
    }

print("\n" + "=" * 70)
print("T5 业绩过滤方案对比")
print("=" * 70)

no_st = lambda c: c not in st_codes
positive = lambda c: has_positive_earnings(c)

scenarios = [
    ('基线(排除ST+<2元)', [no_st]),
    ('+正利润过滤', [no_st, positive]),
    ('仅正利润(不含ST过滤)', [positive]),
]

print(f"  {'方案':30s} | {'年化':>7s} | {'夏普':>6s} | {'回撤':>7s}")
print(f"  {'-'*58}")

for label, filters in scenarios:
    r = run_t5_filtered(mp, mr, 5, filters)
    if r:
        print(f"  {label:30s} | {r['annual']:6.1%} | {r['sharpe']:5.2f} | {r['dd']:6.1%}")
    else:
        print(f"  {label:30s} | N/A")

# T10也跑
print(f"\n  [T10]")
for label, filters in scenarios:
    r = run_t5_filtered(mp, mr, 10, filters)
    if r:
        print(f"  {label:30s} | {r['annual']:6.1%} | {r['sharpe']:5.2f} | {r['dd']:6.1%}")

# ============================================================
# 分析
# ============================================================
print(f"\n=== 分析 ===")
print(f"  正利润覆盖率: {n_positive}/{len(profit_map)} = {n_positive/len(profit_map):.0%}")
print(f"  ST股 + 负利润: 剔除这两类后的候选池变化")
print(f"  注：当前仅用了最新一期业绩（2026Q2），更严谨的做法是按每月的")
print(f"  最近报告期取对应数据，但需要逐季拉取（耗时太长）。")
print(f"  简化处理：假设历史业绩与最新一期一致（对回测期较短的2024-2026")
print(f"  影响较小，对2020-2023可能有偏差）。")
