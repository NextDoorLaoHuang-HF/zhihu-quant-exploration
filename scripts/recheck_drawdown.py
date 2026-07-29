"""
验证正利润过滤的回撤变化是否正确
逐月计算净值，画回撤曲线，找出最大回撤发生的月份
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

# 加载财报（复用已有逻辑）
print("加载数据...")
quarters_needed = ['20191231', '20201231', '20211231', '20221231', '20231231', '20241231', '20251231']
quarterly_profits = {}
for q in quarters_needed:
    yj = ak.stock_yjbb_em(date=q)
    profit_dict = {}
    for _, row in yj.iterrows():
        code = row['股票代码']
        profit = pd.to_numeric(row.get('净利润-净利润', np.nan), errors='coerce')
        if pd.notna(profit):
            profit_dict[code] = profit
    quarterly_profits[q] = profit_dict

def get_available_quarter(month_date):
    year = month_date.year
    return f'{year-1}1231' if month_date.month >= 5 else f'{year-2}1231'

# 退市股+存活股
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

# ST过滤（含退市ST）
st_codes = set()
for code in prices.columns:
    name = stock_list[stock_list['code']==code]['name'].values
    if len(name) > 0:
        if 'ST' in str(name[0]).upper(): st_codes.add(code)
    else:
        info = delist_info.get(code, {})
        if 'ST' in str(info.get('name', '')).upper(): st_codes.add(code)

mp = prices.resample('M').last()
mr = mp.pct_change()

def run_with_ret_series(use_earnings=False):
    """返回月度收益序列而非汇总统计"""
    rets = []
    for i in range(12, len(mp) - 1):
        month_date = mp.index[i]
        row = mp.iloc[i].dropna()
        row = row[row >= 2.0]
        row = row[[c for c in row.index if c not in st_codes]]
        
        if use_earnings:
            q = get_available_quarter(month_date)
            profits = quarterly_profits.get(q, {})
            row = row[[c for c in row.index if profits.get(c, 0) > 0]]
        
        if len(row) < 10: continue
        cheapest = row.nsmallest(5).index
        ret = mr.iloc[i+1].reindex(cheapest).mean()
        rets.append(ret)
    return pd.Series(rets, index=mr.index[13:13+len(rets)])

print("\n计算回撤路径...")
rets_base = run_with_ret_series(use_earnings=False)
rets_earn = run_with_ret_series(use_earnings=True)

# 对齐日期
common = rets_base.index.intersection(rets_earn.index)
rets_base = rets_base[common]
rets_earn = rets_earn[common]

# 净值
pv_base = (1 + rets_base).cumprod()
pv_earn = (1 + rets_earn).cumprod()

# 回撤序列
dd_base = pv_base / pv_base.cummax() - 1
dd_earn = pv_earn / pv_earn.cummax() - 1

# 找出最大回撤发生的时间
max_dd_idx_base = dd_base.idxmin()
max_dd_idx_earn = dd_earn.idxmin()

print(f"\n基线:")
print(f"  年化: {rets_base.mean()*12:.1%}")
print(f"  最大回撤: {dd_base.min():.1%} 发生在 {max_dd_idx_base.date()}")
print(f"  当时净值: {pv_base[max_dd_idx_base]:.3f}, 峰值: {pv_base.cummax()[max_dd_idx_base]:.3f}")

print(f"\n正利润过滤:")
print(f"  年化: {rets_earn.mean()*12:.1%}")
print(f"  最大回撤: {dd_earn.min():.1%} 发生在 {max_dd_idx_earn.date()}")
print(f"  当时净值: {pv_earn[max_dd_idx_earn]:.3f}, 峰值: {pv_earn.cummax()[max_dd_idx_earn]:.3f}")

# 比较：正利润过滤在哪些月份比基线差，哪些月份好
monthly_diff = rets_earn - rets_base
win_months = (monthly_diff > 0).sum()
print(f"\n正利润过滤月度胜率: {win_months}/{len(monthly_diff)} = {win_months/len(monthly_diff):.0%}")
print(f"  平均月度差异: {monthly_diff.mean():+.2%}")
print(f"  赢的月份平均多赚: {monthly_diff[monthly_diff>0].mean():+.2%}")
print(f"  输的月份平均少赚: {monthly_diff[monthly_diff<0].mean():+.2%}")

# 检查：是否正利润过滤让候选池太小，导致某些月份选不到股
print(f"\n候选池检查:")
for i in range(12, len(mp) - 1):
    month_date = mp.index[i]
    row = mp.iloc[i].dropna()
    row2 = row[row >= 2.0]
    row3 = row2[[c for c in row2.index if c not in st_codes]]
    q = get_available_quarter(month_date)
    profits = quarterly_profits.get(q, {})
    row4 = row3[[c for c in row3.index if profits.get(c, 0) > 0]]
    # 只打印候选池明显缩水的月份
    if len(row4) < len(row3) * 0.3 and len(row3) > 20:
        print(f"  {month_date.date()}: 基线候选{len(row3):3d} → 正利润后{len(row4):3d} (-{(1-len(row4)/max(1,len(row3)))*100:.0f}%)")
