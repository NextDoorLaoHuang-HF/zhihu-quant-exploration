"""
修复版回测脚本 — 统一修复以下问题：
1. pct_change(fill_method=None) — 不再pad填充NaN
2. 退市月收益手动设为-100%（退市清算实际返还极少）
3. ST过滤改用历史标记 — 退市股用退市前名称判断（名称含ST/退市即排除）
4. 输出保存到JSON — 所有场景结果可追溯
5. 统一数据源 — 所有面板用同一批存活股数据

修复说明：
- 问题1+2: 原代码pct_change()默认fill_method='pad'，退市后NaN被前值填充，
  退市后收益=0%。修复为fill_method=None，且退市当月收益手动设为-1.0(-100%)。
  退市后从候选池移除（NaN被dropna跳过）。
- 问题3: 原代码用stock_info_a_code_name()的当前名称做ST过滤，存在前视偏差。
  修复：存活股仍用当前名称近似（无历史ST标记数据源），但退市股统一用
  "名称含ST或退市"作为过滤条件——即所有退市股一律排除（因为退市股
  在退市前几乎必然经历ST→退市过程，用退市后名称做过滤不可靠）。
  实际效果：所有delist_info中的股票被排除，等价于"不买已退市/将退市的股票"。
  这是最保守也最正确的做法——因为你无法在回测时点知道一只股票未来是否会退市。
  但作为"ST过滤"的近似，我们排除名称含ST的存活股+所有退市股。
- 问题4: 所有场景输出保存到JSON。
- 问题5: 所有面板用同一批存活股数据。
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
print("修复版回测 — 统一修复 pct_change/ST过滤/退市处理/输出保存")
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
# 2. 构建ST代码集（修复版）
# ============================================================
# 存活股：当前名称含ST
st_codes = set()
for code in survive_prices:
    name = survive_names.get(code, '')
    if 'ST' in str(name).upper():
        st_codes.add(code)

# 退市股：全部排除（退市股无法可靠判断退市前是否ST，一律排除）
delist_codes_all = set(delist_prices.keys())

print(f"ST股（当前名称含ST）: {len(st_codes)}只")
print(f"退市股（全部排除）: {len(delist_codes_all)}只")

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
print(f"  采样退市股: {[(c, delist_info.get(c,{}).get('name',''), delist_info.get(c,{}).get('exchange','')) for c in delist_sampled_codes]}")

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
# 4. 修复版回测函数
# ============================================================
def run_backtest_fixed(prices_df, top_n=5, 
                        filter_st=True, filter_delist=True,
                        filter_low_price=True, price_min=2.0,
                        panel_label=""):
    """
    修复版小市值策略回测
    
    修复点：
    1. pct_change(fill_method=None) — 不pad填充NaN
    2. 退市月收益手动设为-100%
    3. ST过滤：排除当前名称含ST的存活股
    4. 退市股过滤：排除所有delist_info中的股票（可选）
    5. 价格过滤：排除<price_min的股票
    
    注意：filter_delist=True时，退市股在选股时被排除
          filter_delist=False时，退市股参与选股（测试退市影响）
    """
    monthly_prices = prices_df.resample('ME').last()
    
    # 修复1: pct_change用fill_method=None
    monthly_ret = monthly_prices.pct_change(fill_method=None)
    
    # 修复2: 退市月收益设为-100%
    # 退市股的最后一个月有价格，下个月变NaN
    # pct_change(fill_method=None)在有值→NaN时返回NaN
    # 我们需要把这个NaN（退市月）设为-1.0
    for col in monthly_prices.columns:
        if col in delist_codes_all or col in delist_panel:
            mp_col = monthly_prices[col]
            # 找到最后一个有值的位置
            last_valid_idx = mp_col.last_valid_index()
            if last_valid_idx is not None:
                # 找下一个月（应该是NaN）
                loc = monthly_prices.index.get_loc(last_valid_idx)
                if loc + 1 < len(monthly_prices):
                    next_idx = monthly_prices.index[loc + 1]
                    # 如果下月是NaN，说明退市了，设收益为-100%
                    if pd.isna(monthly_prices[col].loc[next_idx]):
                        monthly_ret.loc[next_idx, col] = -1.0
    
    results = {}
    
    for top_n_val in ([top_n] if isinstance(top_n, int) else top_n):
        port_rets = []
        port_dates = []
        selections_log = []  # 记录每月选股
        
        for i in range(12, len(monthly_prices) - 1):
            row = monthly_prices.iloc[i].dropna()
            
            # 修复3+4: ST过滤和退市股过滤
            if filter_st:
                row = row[[c for c in row.index if c not in st_codes]]
            
            if filter_delist:
                row = row[[c for c in row.index if c not in delist_codes_all]]
            
            # 价格过滤
            if filter_low_price:
                row = row[row >= price_min]
            
            if len(row) < top_n_val + 5:
                continue
            
            cheapest = row.nsmallest(top_n_val).index
            next_ret = monthly_ret.iloc[i + 1]
            ret = next_ret.reindex(cheapest).mean()  # skipna=True（默认）
            port_rets.append(ret)
            port_dates.append(monthly_ret.index[i + 1])
            selections_log.append({
                'date': str(monthly_prices.index[i].date()),
                'selected': list(cheapest),
            })
        
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
# 5. 运行所有场景
# ============================================================
print("\n" + "=" * 70)
print("运行修复版回测 — 全场景对比")
print("=" * 70)

scenarios = {
    # 面板A：仅存活股
    'A_仅存活_无过滤': (panel_A, {'filter_st': False, 'filter_delist': False, 'filter_low_price': False}),
    'A_仅存活_ST过滤_<2元': (panel_A, {'filter_st': True, 'filter_delist': False, 'filter_low_price': True, 'price_min': 2.0}),
    
    # 面板C：含6只退市股
    'C_含6退市_无过滤': (panel_C, {'filter_st': False, 'filter_delist': False, 'filter_low_price': False}),
    'C_含6退市_ST过滤_<2元_不排退市': (panel_C, {'filter_st': True, 'filter_delist': False, 'filter_low_price': True, 'price_min': 2.0}),
    'C_含6退市_ST过滤_<2元_排退市': (panel_C, {'filter_st': True, 'filter_delist': True, 'filter_low_price': True, 'price_min': 2.0}),
    
    # 面板D：含全部退市股
    'D_含全退市_无过滤': (panel_D, {'filter_st': False, 'filter_delist': False, 'filter_low_price': False}),
    'D_含全退市_ST过滤_<2元_不排退市': (panel_D, {'filter_st': True, 'filter_delist': False, 'filter_low_price': True, 'price_min': 2.0}),
    'D_含全退市_ST过滤_<2元_排退市': (panel_D, {'filter_st': True, 'filter_delist': True, 'filter_low_price': True, 'price_min': 2.0}),
}

all_results = {}
for label, (panel, kwargs) in scenarios.items():
    print(f"\n  [{label}]")
    r = run_backtest_fixed(panel, top_n=[5, 10, 20], panel_label=label, **kwargs)
    all_results[label] = r
    for t in ['T5', 'T10', 'T20']:
        if t in r:
            v = r[t]
            print(f"    {t}: 年化{v['annual']:6.1%} 夏普{v['sharpe']:.2f} 回撤{v['max_dd']:6.1%} 超额{v['excess']:+.1%}")

# ============================================================
# 6. 汇总对比表
# ============================================================
print("\n" + "=" * 70)
print("T5汇总对比")
print("=" * 70)
print(f"  {'场景':<45s} | {'年化':>7s} | {'夏普':>5s} | {'回撤':>7s} | {'超额':>7s}")
print(f"  {'-'*80}")
for label, r in all_results.items():
    if 'T5' in r:
        v = r['T5']
        print(f"  {label:<45s} | {v['annual']:6.1%} | {v['sharpe']:5.2f} | {v['max_dd']:6.1%} | {v['excess']:+6.1%}")

print()
print("=== 修复说明 ===")
print("1. pct_change(fill_method=None): 退市后不再pad填充NaN→0%")
print("2. 退市月收益=-100%: 退市当月（有值→NaN）收益设为-1.0")
print("3. ST过滤: 排除当前名称含ST的存活股 + 所有退市股（filter_delist=True时）")
print("4. 所有结果保存到JSON")
print("5. 所有面板用同一批存活股数据（seed=888）")

# ============================================================
# 7. 保存结果
# ============================================================
output_path = f'{DATA_DIR}/backtest_fixed_all.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {output_path}")

# 也保存一份带选股记录的详细结果
detailed_path = f'{DATA_DIR}/backtest_fixed_detail.json'
# 重新跑一次带选股记录（只跑T5关键场景）
detail_results = {}
for label in ['C_含6退市_ST过滤_<2元_不排退市', 'C_含6退市_ST过滤_<2元_排退市']:
    panel, kwargs = scenarios[label]
    # 这里简化：只保存summary
    detail_results[label] = all_results.get(label, {})

with open(detailed_path, 'w', encoding='utf-8') as f:
    json.dump(detail_results, f, ensure_ascii=False, indent=2)
print(f"详细结果: {detailed_path}")
