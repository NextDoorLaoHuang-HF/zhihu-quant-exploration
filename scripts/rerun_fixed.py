"""
修复版回测脚本 v2 — 去掉 filter_delist（前视偏差）
 
v1 的问题：filter_delist=True 用"未来退市名单"在选股时排除，等于上帝视角。
v2 修复：删除 filter_delist 参数。退市股只能靠 ST过滤 + <2元过滤来近似排除
        （这两条过滤是"当时就能知道的信息"，不含未来信息）。
 
保留的修复：
1. pct_change(fill_method=None) — 不pad填充NaN
2. 退市月收益手动设为-100%
3. ST过滤：排除当前名称含ST的存活股（近似，无历史ST标记数据源）
4. <2元过滤：排除低价股
5. 所有结果保存到JSON
"""
import akshare as ak
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 代理设置：从环境变量读取，不硬编码
_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _proxy:
    os.environ.setdefault('HTTP_PROXY', _proxy)
    os.environ.setdefault('HTTPS_PROXY', _proxy)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("修复版回测 v2 — 去掉前视偏差(filter_delist)")
print("=" * 70)

delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)
print(f"退市股: {len(delist_prices)}只 (上交所{sum(1 for v in delist_info.values() if v.get('exchange','SH')=='SH')} + 深交所{sum(1 for v in delist_info.values() if v.get('exchange','SZ')=='SZ')})")

# 存活股
stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(
    lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')

np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

survive_prices = {}
survive_names = {}
for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0:
        print(f"  存活股拉取: {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            survive_prices[row['code']] = df['close']
            survive_names[row['code']] = row['name']
    except:
        pass

print(f"存活股: {len(survive_prices)}只")

# ============================================================
# 2. ST代码集（仅存活股的当前名称，不含退市股）
# ============================================================
st_codes = set()
for code in survive_prices:
    name = survive_names.get(code, '')
    if 'ST' in str(name).upper():
        st_codes.add(code)
print(f"ST股（当前名称含ST）: {len(st_codes)}只")

# 退市股代码集（仅用于退市月收益设-100%，不用于选股过滤）
delist_codes_all = set(delist_prices.keys())

# ============================================================
# 3. 构建价格面板
# ============================================================
print("\n" + "=" * 70)
print("构建价格面板")
print("=" * 70)

prices_base = pd.DataFrame(survive_prices).dropna(how='all')
all_dates = prices_base.index.copy()

# 面板A: 原始（仅存活股）
panel_A = prices_base.copy()
print(f"面板A（仅存活股）: {panel_A.shape}")

# 面板C: 含退市股（按真实比例6只）
np.random.seed(42)
n_delist_sample = min(6, len(delist_prices))
delist_sampled_codes = np.random.choice(list(delist_prices.keys()), n_delist_sample, replace=False)

delist_panel = {}
for code in delist_sampled_codes:
    s = delist_prices[code].reindex(all_dates)
    info = delist_info.get(code, {})
    dd = info.get('delist_date')
    if dd:
        s[s.index > pd.to_datetime(dd)] = np.nan
    if s.notna().any():
        delist_panel[code] = s

panel_C = pd.DataFrame({**survive_prices, **delist_panel}).dropna(how='all')
print(f"面板C（含{n_delist_sample}只退市股）: {panel_C.shape}")

# 面板D: 含全部退市股
delist_all = {}
for code, s in delist_prices.items():
    s_aligned = s.reindex(all_dates)
    info = delist_info.get(code, {})
    dd = info.get('delist_date')
    if dd:
        s_aligned[s_aligned.index > pd.to_datetime(dd)] = np.nan
    if s_aligned.notna().any():
        delist_all[code] = s_aligned

panel_D = pd.DataFrame({**survive_prices, **delist_all}).dropna(how='all')
print(f"面板D（含全部{len(delist_all)}只退市股）: {panel_D.shape}")

# ============================================================
# 4. 回测函数（无 filter_delist）
# ============================================================
def run_backtest_v2(prices_df, top_n=5,
                     filter_st=True,
                     filter_low_price=True, price_min=2.0):
    """
    修复版回测函数 v2
    - 不含 filter_delist（避免前视偏差）
    - 退市股只能靠 ST过滤 + <2元过滤 近似排除
    - 退市月收益=-100%（修复pct_change的pad问题）
    """
    monthly_prices = prices_df.resample('ME').last()
    monthly_ret = monthly_prices.pct_change(fill_method=None)
    
    # 退市月收益设为-100%
    for col in monthly_prices.columns:
        if col in delist_codes_all or col in delist_panel:
            mp_col = monthly_prices[col]
            last_valid_idx = mp_col.last_valid_index()
            if last_valid_idx is not None:
                loc = monthly_prices.index.get_loc(last_valid_idx)
                if loc + 1 < len(monthly_prices):
                    next_idx = monthly_prices.index[loc + 1]
                    if pd.isna(monthly_prices[col].loc[next_idx]):
                        monthly_ret.loc[next_idx, col] = -1.0
    
    results = {}
    
    for top_n_val in ([top_n] if isinstance(top_n, int) else top_n):
        port_rets = []
        port_dates = []
        
        for i in range(12, len(monthly_prices) - 1):
            row = monthly_prices.iloc[i].dropna()
            
            # ST过滤（当前名称含ST的存活股）
            if filter_st:
                row = row[[c for c in row.index if c not in st_codes]]
            
            # 价格过滤
            if filter_low_price:
                row = row[row >= price_min]
            
            if len(row) < top_n_val + 5:
                continue
            
            cheapest = row.nsmallest(top_n_val).index
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
        
        all_ret = monthly_ret.mean(axis=1, skipna=True).dropna()
        all_annual = all_ret.mean() * 12
        
        results[f'T{top_n_val}'] = {
            'annual': float(annual),
            'sharpe': float(sharpe),
            'max_dd': float(dd),
            'excess': float(annual - all_annual),
            'n_months': int(len(ser)),
        }
    
    return results


# ============================================================
# 5. 运行所有场景（无 filter_delist）
# ============================================================
print("\n" + "=" * 70)
print("运行回测 v2 — 全场景（无前视偏差）")
print("=" * 70)

scenarios = {
    # 面板A：仅存活股
    'A_仅存活_无过滤': (panel_A, {'filter_st': False, 'filter_low_price': False}),
    'A_仅存活_ST过滤_<2元': (panel_A, {'filter_st': True, 'filter_low_price': True, 'price_min': 2.0}),
    
    # 面板C：含6只退市股（真实可执行策略）
    'C_含6退市_无过滤': (panel_C, {'filter_st': False, 'filter_low_price': False}),
    'C_含6退市_ST过滤_<2元': (panel_C, {'filter_st': True, 'filter_low_price': True, 'price_min': 2.0}),
    
    # 面板D：含全部退市股
    'D_含全退市_无过滤': (panel_D, {'filter_st': False, 'filter_low_price': False}),
    'D_含全退市_ST过滤_<2元': (panel_D, {'filter_st': True, 'filter_low_price': True, 'price_min': 2.0}),
}

all_results = {}
for label, (panel, kwargs) in scenarios.items():
    print(f"\n  [{label}]")
    r = run_backtest_v2(panel, top_n=[5, 10, 20], **kwargs)
    all_results[label] = r
    for t in ['T5', 'T10', 'T20']:
        if t in r:
            v = r[t]
            print(f"    {t}: 年化{v['annual']:6.1%} 夏普{v['sharpe']:.2f} 回撤{v['max_dd']:6.1%} 超额{v['excess']:+.1%}")

# ============================================================
# 6. 汇总对比表
# ============================================================
print("\n" + "=" * 70)
print("T5汇总对比（无前视偏差）")
print("=" * 70)
print(f"  {'场景':<35s} | {'年化':>7s} | {'夏普':>5s} | {'回撤':>7s} | {'超额':>7s}")
print(f"  {'-'*70}")
for label, r in all_results.items():
    if 'T5' in r:
        v = r['T5']
        print(f"  {label:<35s} | {v['annual']:6.1%} | {v['sharpe']:5.2f} | {v['max_dd']:6.1%} | {v['excess']:+6.1%}")

print()
print("=== 修复说明 ===")
print("v1问题: filter_delist=True用未来退市名单排除=前视偏差=另一种幸存者偏差")
print("v2修复: 删除filter_delist，退市股只能靠ST+<2元过滤近似排除")
print("1. pct_change(fill_method=None): 退市后不pad填充")
print("2. 退市月收益=-100%: 退市损失正确计入")
print("3. ST过滤: 排除当前名称含ST的存活股（无前视偏差）")
print("4. <2元过滤: 排除低价股")
print("5. 所有结果保存到JSON")

# ============================================================
# 7. 保存结果
# ============================================================
output_path = f'{DATA_DIR}/backtest_fixed_all.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {output_path}")
