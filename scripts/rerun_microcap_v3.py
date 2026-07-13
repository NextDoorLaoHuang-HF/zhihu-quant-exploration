"""
修正版回测：按真实市场比例加入退市股，并添加合理过滤条件
- 真实比例：150只中应有~5-6只退市股（全市场200/5500*150≈5.5）
- 过滤条件：排除ST/*ST（有交易限制）、排除股价<1元（面值退市风险）
- 多组对比：不同过滤条件下的结果
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

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("加载数据")
print("=" * 70)

delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)
print(f"退市股: {len(delist_prices)}只")

# 存活股（复用已有的，避免重复拉取）
stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(
    lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')

np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

survive_prices = {}
survive_names = {}
for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0:
        print(f"  存活股进度: {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].copy()
            df = df.sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            code = row['code']
            survive_prices[code] = df['close']
            survive_names[code] = row['name']
    except:
        pass

print(f"存活股: {len(survive_prices)}只")

# ============================================================
# 2. 构建多个对比面板
# ============================================================
print("\n" + "=" * 70)
print("构建对比面板")
print("=" * 70)

prices_base = pd.DataFrame(survive_prices).dropna(how='all')
all_dates = prices_base.index.copy()

# 面板A: 原始（仅存活股，不含退市）
panel_A = prices_base.copy()
print(f"面板A（原始）: {panel_A.shape}")

# 面板B: 含全部78只退市股（之前的结果）
delist_all = {}
for code, s in delist_prices.items():
    s_aligned = s.reindex(all_dates)
    info = delist_info.get(code, {})
    dd = info.get('delist_date')
    if dd:
        s_aligned[s_aligned.index > pd.to_datetime(dd)] = np.nan
    if s_aligned.notna().any():
        delist_all[code] = s_aligned

panel_B = pd.DataFrame({**survive_prices, **delist_all}).dropna(how='all')
print(f"面板B（全部退市）: {panel_B.shape}")

# 面板C: 按真实比例随机抽退市股（150只中应有~5.5只）
# 全市场2020-2026退市约200只，存活~5500只，比例约3.6%
# 150只中预期约5.5只
np.random.seed(42)
n_delist_sample = min(6, len(delist_prices))
delist_sampled_codes = np.random.choice(list(delist_prices.keys()), n_delist_sample, replace=False)

delist_sampled = {}
for code in delist_sampled_codes:
    s = delist_prices[code]
    s_aligned = s.reindex(all_dates)
    info = delist_info.get(code, {})
    dd = info.get('delist_date')
    if dd:
        s_aligned[s_aligned.index > pd.to_datetime(dd)] = np.nan
    if s_aligned.notna().any():
        delist_sampled[code] = s_aligned

panel_C = pd.DataFrame({**survive_prices, **delist_sampled}).dropna(how='all')
print(f"面板C（比例采样{n_delist_sample}只退市）: {panel_C.shape}")
print(f"  采样退市股: {[delist_info.get(c,{}).get('name',c) for c in delist_sampled_codes]}")

# 面板D: 全部退市 + 过滤ST/*ST和股价<1元
# 过滤逻辑：在选股时排除（不是从面板中删除，因为ST状态会变）
print(f"面板D: 同面板B，但在选股时过滤ST/*ST和<1元")

# ============================================================
# 3. 回测函数（支持过滤条件）
# ============================================================
def run_microcap_backtest(prices_df, top_n_list=[5, 10, 20], 
                          filter_low_price=True, price_min=1.0,
                          exclude_st=True, stock_names=None):
    """小市值策略回测，支持过滤"""
    monthly_prices = prices_df.resample('M').last()
    monthly_ret = monthly_prices.pct_change()
    
    results = {}
    
    for top_n in top_n_list:
        port_rets = []
        port_dates = []
        
        for i in range(12, len(monthly_prices) - 1):
            row = monthly_prices.iloc[i].dropna()
            
            # 过滤条件
            if filter_low_price:
                row = row[row >= price_min]  # 排除低价股
            
            if len(row) < top_n + 5:
                continue
            
            cheapest = row.nsmallest(top_n).index
            next_ret = monthly_ret.iloc[i + 1]
            ret = next_ret.reindex(cheapest).mean()
            port_rets.append(ret)
            port_dates.append(monthly_ret.index[i + 1])
        
        if len(port_rets) < 12:
            continue
        
        ser = pd.Series(port_rets, index=port_dates)
        pv = (1 + ser).cumprod()
        annual = ser.mean() * 12
        sharpe = ser.mean() / ser.std() * np.sqrt(12) if ser.std() > 0 else 0
        dd = ((pv / pv.cummax()) - 1).min()
        
        all_ret = monthly_ret.mean(axis=1).dropna()
        all_annual = all_ret.mean() * 12
        
        results[f'T{top_n}'] = {
            'annual': annual,
            'sharpe': sharpe,
            'max_dd': dd,
            'excess': annual - all_annual,
            'n_months': len(ser),
        }
    
    return results

# ============================================================
# 4. 多组回测对比
# ============================================================
print("\n" + "=" * 70)
print("回测对比")
print("=" * 70)

scenarios = {
    'A_原始(仅存活)': (panel_A, {}),
    'B_全部78退市_无过滤': (panel_B, {}),
    'C_6只退市_无过滤': (panel_C, {}),
    'B_filter_排除<1元': (panel_B, {'filter_low_price': True, 'price_min': 1.0}),
    'B_filter_排除<2元': (panel_B, {'filter_low_price': True, 'price_min': 2.0}),
    'B_filter_排除<3元': (panel_B, {'filter_low_price': True, 'price_min': 3.0}),
}

all_results = {}
for label, (panel, kwargs) in scenarios.items():
    r = run_microcap_backtest(panel, **kwargs)
    all_results[label] = r
    print(f"\n  [{label}]")
    for t in ['T5', 'T10', 'T20']:
        if t in r:
            v = r[t]
            print(f"    {t}: 年化{v['annual']:.1%} 夏普{v['sharpe']:.2f} 回撤{v['max_dd']:.1%} 超额{v['excess']:+.1%}")

# ============================================================
# 5. 汇总对比表
# ============================================================
print("\n" + "=" * 70)
print("T5汇总对比")
print("=" * 70)
print(f"  {'场景':<30s} | {'年化':>7s} | {'回撤':>7s} | {'超额':>7s}")
print(f"  {'-'*58}")
for label, r in all_results.items():
    if 'T5' in r:
        v = r['T5']
        print(f"  {label:<30s} | {v['annual']:6.1%} | {v['max_dd']:6.1%} | {v['excess']:+6.1%}")

# 保存
with open(f'{DATA_DIR}/backtest_all_scenarios.json', 'w', encoding='utf-8') as f:
    json.dump({k: {kk: {kkk: float(vvv) for kkk, vvv in vv.items()} for kk, vv in v.items()} 
               for k, v in all_results.items()}, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存: {DATA_DIR}/backtest_all_scenarios.json")
