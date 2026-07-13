"""
T5/T10 + 各种复杂度改进：
1. 用真实市值排序替代价格排序
2. 动量筛选（只买过去N个月上涨的）
3. 流动性筛选（最小成交量）
4. 质量筛选（ROE、市盈率）
5. 组合打分（市值+动量综合排名）
"""
import akshare as ak
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'

DATA_DIR = 'PROJECT_ROOT/data'

# 加载退市股
delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)

# 存活股 + 真实市值
stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

survive_prices = {}
survive_mcaps = {}     # 流通市值
survive_volumes = {}   # 成交量
survive_names = {}

for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0: print(f"  拉取: {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            code = row['code']
            survive_prices[code] = df['close']
            survive_names[code] = row['name']
            if '流通市值' in df.columns:
                survive_mcaps[code] = pd.to_numeric(df['流通市值'], errors='coerce')
            if 'volume' in df.columns:
                survive_volumes[code] = df['volume']
    except: pass

# 加入6只退市股
np.random.seed(42)
delist_sampled_codes = np.random.choice(list(delist_prices.keys()), 6, replace=False)
all_dates = pd.date_range('2020-01-01', '2026-07-13', freq='D')
for code in delist_sampled_codes:
    s = delist_prices[code].reindex(all_dates)
    info = delist_info.get(code, {})
    dd = info.get('delist_date')
    if dd: s[s.index > pd.to_datetime(dd)] = np.nan
    if s.notna().any(): survive_prices[code] = s

prices = pd.DataFrame(survive_prices).dropna(how='all')
print(f"候选池: {prices.shape[1]}只")

# 构建月频数据
mp = prices.resample('M').last()
mr = mp.pct_change()

# 市值面板（如果有）
if survive_mcaps:
    mcap_df = pd.DataFrame({k: v for k, v in survive_mcaps.items() if k in prices.columns})
    monthly_mcap = mcap_df.resample('M').last() if len(mcap_df.columns) > 0 else None
else:
    monthly_mcap = None

# 成交量面板
if survive_volumes:
    vol_df = pd.DataFrame({k: v for k, v in survive_volumes.items() if k in prices.columns})
    monthly_vol = vol_df.resample('M').sum() if len(vol_df.columns) > 0 else None
else:
    monthly_vol = None

# ============================================================
# 多种改进策略
# ============================================================

def is_st(code):
    """检查是否ST股"""
    name = survive_names.get(code, delist_info.get(code, {}).get('name', ''))
    return 'ST' in str(name).upper()

def run_t5_enhanced(mp, mr, n_stocks, strategy='price', **kwargs):
    """
    strategy: 'price'(价格排序), 'mcap'(市值排序),
              'momentum'(动量+价格), 'composite'(综合排名)
    """
    rets = []
    
    for i in range(12, len(mp) - 1):
        row = mp.iloc[i].dropna()
        
        # 基础过滤：排除ST
        valid_codes = [c for c in row.index if not is_st(c)]
        row = row[valid_codes]
        
        # 价格过滤：>2元
        row = row[row >= 2.0]
        
        if len(row) < n_stocks + 10:
            continue
        
        if strategy == 'price':
            # 纯价格排序
            selected = row.nsmallest(n_stocks).index
            
        elif strategy == 'mcap':
            # 真实市值排序
            if monthly_mcap is not None:
                mcap_row = monthly_mcap.iloc[i].dropna()
                mcap_row = mcap_row[mcap_row.index.isin(row.index)]
                if len(mcap_row) >= n_stocks + 5:
                    selected = mcap_row.nsmallest(n_stocks).index
                else:
                    selected = row.nsmallest(n_stocks).index
            else:
                selected = row.nsmallest(n_stocks).index
                
        elif strategy == 'momentum':
            # 动量筛选：在最低20只中，选过去3个月涨幅最大的n_stocks只
            cheap20 = row.nsmallest(20).index
            # 计算过去3个月动量
            mom = {}
            for c in cheap20:
                past = mp[c].iloc[max(0, i-3):i+1]
                if len(past) >= 2 and past.iloc[0] > 0:
                    mom[c] = past.iloc[-1] / past.iloc[0] - 1
            if len(mom) >= n_stocks:
                selected = sorted(mom, key=mom.get, reverse=True)[:n_stocks]
            else:
                selected = row.nsmallest(n_stocks).index
                
        elif strategy == 'composite':
            # 综合打分：价格排名 + 动量排名 各50%
            cheap20 = row.nsmallest(20)
            price_rank = {c: i for i, c in enumerate(cheap20.index)}
            mom = {}
            for c in cheap20.index:
                past = mp[c].iloc[max(0, i-3):i+1]
                if len(past) >= 2 and past.iloc[0] > 0:
                    mom[c] = past.iloc[-1] / past.iloc[0] - 1
                else:
                    mom[c] = 0
            mom_rank = {c: i for i, (c, _) in enumerate(sorted(mom.items(), key=lambda x: -x[1]))}
            score = {}
            for c in cheap20.index:
                score[c] = price_rank.get(c, 99) * 0.5 + mom_rank.get(c, 99) * 0.5
            selected = sorted(score, key=score.get)[:n_stocks]
            
        elif strategy == 'volume_filter':
            # 流动性过滤：排除成交量最低的30%，再用价格排序
            if monthly_vol is not None:
                vol_row = monthly_vol.iloc[i].dropna()
                vol_row = vol_row[vol_row.index.isin(row.index)]
                if len(vol_row) >= n_stocks + 15:
                    med_vol = vol_row.median()
                    liquid = vol_row[vol_row >= med_vol * 0.3].index  # 排除最冷门的
                    row = row[row.index.isin(liquid)]
            if len(row) < n_stocks + 5:
                selected = mp.iloc[i].dropna().nsmallest(n_stocks).index
            else:
                selected = row.nsmallest(n_stocks).index
        else:
            selected = row.nsmallest(n_stocks).index
        
        next_ret = mr.iloc[i + 1]
        ret = next_ret.reindex([c for c in selected if c in next_ret.index]).mean()
        rets.append(ret)
    
    if len(rets) < 12: return None
    s = pd.Series(rets, index=mr.index[13:13+len(rets)])
    return {
        'annual': s.mean() * 12,
        'sharpe': s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0,
        'dd': ((1+s).cumprod() / (1+s).cumprod().cummax() - 1).min(),
        'n': len(s),
    }

# ============================================================
# 运行对比
# ============================================================
print("\n" + "=" * 70)
print("T5/T10 + 复杂度改进对比")
print("（均已含退市股 + 排除ST + 排除<2元）")
print("=" * 70)

tests = [
    ('T5_纯价格(基线)', 5, 'price'),
    ('T5_真实市值排序', 5, 'mcap'),
    ('T5_动量优选', 5, 'momentum'),
    ('T5_综合打分(市值+动量)', 5, 'composite'),
    ('T5_流动性过滤', 5, 'volume_filter'),
    ('T10_纯价格(基线)', 10, 'price'),
    ('T10_动量优选', 10, 'momentum'),
    ('T10_综合打分', 10, 'composite'),
    ('T10_动量优选_取15只', 15, 'momentum'),  # 扩大选股池
]

print(f"  {'方案':30s} | {'年化':>7s} | {'夏普':>6s} | {'回撤':>7s} | {'月数'}")
print(f"  {'-'*62}")
best_t10 = None

for label, n, strategy in tests:
    r = run_t5_enhanced(mp, mr, n, strategy)
    if r:
        marker = ''
        if '基线' in label:
            marker = ''
        print(f"  {label:30s} | {r['annual']:6.1%} | {r['sharpe']:5.2f} | {r['dd']:6.1%} | {r['n']:3d}")
        if 'T10' in label and (best_t10 is None or r['sharpe'] > best_t10['sharpe']):
            best_t10 = (label, r)

print(f"\n=== 核心发现 ===")
print("  T5动量优选 vs T5纯价格：动量能否避免选到持续下跌的股")
print("  T10+动量 vs T5+动量：分散化能否弥补收益率差距")
print("  真实市值 vs 价格排序：价格是否是市值的有效代理")
