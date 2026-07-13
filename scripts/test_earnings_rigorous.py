"""
严谨版"剔除负业绩"回测：
逐月使用当时最新可用的财报数据，消除前视偏差。
财报公布规则：年报/一季报4月30日前，半年报8月31日前，三季报10月31日前。
简化：每年5月起用上年度Q4数据（年报），1-4月用再上年度Q4（年报未出）。
"""
import akshare as ak
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'

DATA_DIR = 'PROJECT_ROOT/data'

# ============================================================
# 1. 逐季拉取财报数据
# ============================================================
print("拉取各季度财报数据...")
quarters_needed = ['20191231', '20201231', '20211231', '20221231', '20231231', '20241231', '20251231']
quarterly_profits = {}  # quarter_date -> {code: profit}

for q in quarters_needed:
    print(f"  {q}...", end=' ')
    try:
        yj = ak.stock_yjbb_em(date=q)
        profit_dict = {}
        for _, row in yj.iterrows():
            code = row['股票代码']
            profit = pd.to_numeric(row.get('净利润-净利润', np.nan), errors='coerce')
            if pd.notna(profit):
                profit_dict[code] = profit
        quarterly_profits[q] = profit_dict
        print(f"{len(profit_dict)}只有数据, +{sum(1 for v in profit_dict.values() if v>0)}正/{sum(1 for v in profit_dict.values() if v<=0)}负")
    except Exception as e:
        print(f"失败: {str(e)[:60]}")

# ============================================================
# 2. 确定每月可用财报（财报公布时间表）
# ============================================================
def get_available_quarter(month_date):
    """根据月份返回当时最新可用的年报数据"""
    year = month_date.year
    month = month_date.month
    # 年报截止日4月30日，所以5月起可用上年度Q4
    if month >= 5:
        available_year = year - 1
    else:
        available_year = year - 2
    return f'{available_year}1231'

# 测试
for test_date in [pd.Timestamp('2021-03-31'), pd.Timestamp('2021-05-31'), 
                  pd.Timestamp('2023-01-31'), pd.Timestamp('2023-06-30')]:
    q = get_available_quarter(test_date)
    print(f"  {test_date.date()} → 可用财报: {q} (在数据中={q in quarterly_profits})")

# ============================================================
# 3. 加载价格数据
# ============================================================
print("\n加载价格数据...")
delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)

stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

survive_prices = {}
for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0: print(f"  {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            survive_prices[row['code']] = df['close']
    except: pass

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

# ST检测（修正：包含退市股名称）
def is_st(code):
    """检查是否ST股，含退市股"""
    name = stock_list[stock_list['code']==code]['name'].values
    if len(name) > 0:
        return 'ST' in str(name[0]).upper()
    # 退市股
    info = delist_info.get(code, {})
    return 'ST' in str(info.get('name', '')).upper()

# 构建ST代码集（用于快速过滤）
st_codes = set()
for code in prices.columns:
    if is_st(code):
        st_codes.add(code)
print(f"ST股: {len(st_codes)}只（含退市ST）")

# ============================================================
# 4. 三种策略回测
# ============================================================
def is_profitable(code, month_date):
    """根据调仓月份查询当时可用的最新财报利润"""
    q = get_available_quarter(month_date)
    profits = quarterly_profits.get(q, {})
    return profits.get(code, 0) > 0

def run_backtest(mp, mr, n_stocks=5, use_earnings=False):
    rets = []
    for i in range(12, len(mp) - 1):
        month_date = mp.index[i]
        row = mp.iloc[i].dropna()
        
        # 基础过滤
        row = row[row >= 2.0]
        row = row[[c for c in row.index if c not in st_codes]]
        
        # 正利润过滤
        if use_earnings:
            row = row[[c for c in row.index if is_profitable(c, month_date)]]
        
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
print("严谨版回测（逐月取财报，无前视偏差）")
print("=" * 70)

print(f"\n  {'方案':35s} | {'年化':>7s} | {'夏普':>6s} | {'回撤':>7s}")
print(f"  {'-'*62}")

for n, label in [(5, 'T5'), (10, 'T10')]:
    r_base = run_backtest(mp, mr, n, use_earnings=False)
    r_earn = run_backtest(mp, mr, n, use_earnings=True)
    
    if r_base:
        print(f"  {label} 基线(排除ST+<2元):{'':16s} | {r_base['annual']:6.1%} | {r_base['sharpe']:5.2f} | {r_base['dd']:6.1%}")
    if r_earn:
        print(f"  {label} +正利润(逐月财报):{'':13s} | {r_earn['annual']:6.1%} | {r_earn['sharpe']:5.2f} | {r_earn['dd']:6.1%}")
        if r_base:
            diff = r_earn['annual'] - r_base['annual']
            dd_diff = r_earn['dd'] - r_base['dd']
            print(f"  {'':35s} | {diff:+6.1%} | {'':>6s} | {dd_diff:+6.1%}")

print(f"\n=== 与简化版对比 ===")
print(f"  简化版（用2026Q1覆盖全程）: T5 15.4% 夏普0.63 回撤-28.0%  ← 前视偏差")
print(f"  严谨版（逐月取当时财报）: 见上                             ← 无前视偏差")
