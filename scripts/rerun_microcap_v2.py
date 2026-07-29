"""
Step 3+4 (修正版): 合并退市股数据，正确处理退市后从候选池移除
- 退市前：正常参与选股
- 退市当月：如果被选中，收益体现退市损失
- 退市后：价格设为NaN，不再参与选股
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
# 1. 加载退市股数据
# ============================================================
print("=" * 70)
print("Step 1: 加载退市股数据")
print("=" * 70)

delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)

print(f"退市股数据: {len(delist_prices)}只")

# ============================================================
# 2. 获取存活股数据（与原回测相同的随机种子和抽样）
# ============================================================
print("\n" + "=" * 70)
print("Step 2: 获取存活股数据")
print("=" * 70)

stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(
    lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')

np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

survive_prices = {}
for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0:
        print(f"  进度: {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].copy()
            df = df.sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            survive_prices[row['code']] = df['close']
    except:
        pass

print(f"存活股: {len(survive_prices)}只")

# ============================================================
# 3. 构建价格面板
# ============================================================
print("\n" + "=" * 70)
print("Step 3: 构建价格面板")
print("=" * 70)

# 面板A：不含退市股
prices_no_delist = pd.DataFrame(survive_prices).dropna(how='all')
print(f"面板A（不含退市）: {prices_no_delist.shape}")

# 面板B：含退市股
# 关键修正：退市后将价格设为NaN，使该股票不再被选中
# 退市前的最后一个月保留最后交易价格
all_dates = prices_no_delist.index.copy()

delist_panel = {}
for code, s in delist_prices.items():
    # 对齐到全市场日期
    s_aligned = s.reindex(all_dates)
    
    # 获取退市日期
    info = delist_info.get(code, {})
    delist_date_str = info.get('delist_date')
    
    if delist_date_str:
        delist_date = pd.to_datetime(delist_date_str)
        # 退市后将价格设为NaN（不再参与选股）
        s_aligned[s_aligned.index > delist_date] = np.nan
    
    # 只保留有数据的部分
    if s_aligned.notna().any():
        delist_panel[code] = s_aligned

prices_with_delist = pd.DataFrame({**survive_prices, **delist_panel})
prices_with_delist = prices_with_delist.dropna(how='all')
print(f"面板B（含退市）: {prices_with_delist.shape}")
print(f"  其中退市股: {len(delist_panel)}只")

# ============================================================
# 4. 回测函数
# ============================================================
def run_microcap_backtest(prices_df, top_n_list=[5, 10, 20, 50]):
    """小市值策略回测"""
    monthly_prices = prices_df.resample('M').last()
    monthly_ret = monthly_prices.pct_change(fill_method=None)
    
    results = {}
    
    for top_n in top_n_list:
        port_rets = []
        port_dates = []
        selected_delist_months = {}  # 记录退市股被选中情况
        
        for i in range(12, len(monthly_prices) - 1):
            row = monthly_prices.iloc[i].dropna()
            if len(row) < top_n + 10:
                continue
            
            cheapest = row.nsmallest(top_n).index
            next_ret = monthly_ret.iloc[i + 1]
            
            # 记录退市股被选中
            for code in cheapest:
                if code in delist_panel:
                    selected_delist_months[code] = selected_delist_months.get(code, 0) + 1
            
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
            'pv': pv,
            'returns': ser,
            'selected_delist': selected_delist_months,
        }
    
    return results

# ============================================================
# 5. 运行回测对比
# ============================================================
print("\n" + "=" * 70)
print("Step 4: 回测对比 — 含退市 vs 不含退市")
print("=" * 70)

print("\n--- 面板A：不含退市股（原始回测）---")
results_no_delist = run_microcap_backtest(prices_no_delist)
for name, r in results_no_delist.items():
    print(f"  {name:6s} | 年化{r['annual']:5.1%} 夏普{r['sharpe']:.2f} 回撤{r['max_dd']:5.1%} | 超额{r['excess']:+.1%} | {r['n_months']}月")

print("\n--- 面板B：含退市股（退市后移除）---")
results_with_delist = run_microcap_backtest(prices_with_delist)
for name, r in results_with_delist.items():
    print(f"  {name:6s} | 年化{r['annual']:5.1%} 夏普{r['sharpe']:.2f} 回撤{r['max_dd']:5.1%} | 超额{r['excess']:+.1%} | {r['n_months']}月")

print("\n--- 差异对比 ---")
print(f"  {'策略':6s} | {'不含退市':>8s} | {'含退市':>8s} | {'差异':>8s} | {'回撤变化':>8s}")
print(f"  {'-'*55}")
for name in results_no_delist:
    if name in results_with_delist:
        r_a = results_no_delist[name]
        r_b = results_with_delist[name]
        diff = r_b['annual'] - r_a['annual']
        dd_diff = r_b['max_dd'] - r_a['max_dd']
        print(f"  {name:6s} | {r_a['annual']:7.1%} | {r_b['annual']:7.1%} | {diff:+7.1%} | {dd_diff:+7.1%}")

# 检查退市股在T5组合中被选中的频率
print("\n--- T5组合中退市股被选中情况 ---")
t5_delist = results_with_delist.get('T5', {}).get('selected_delist', {})
if t5_delist:
    print(f"  被选中的退市股: {len(t5_delist)}只")
    for code, cnt in sorted(t5_delist.items(), key=lambda x: -x[1]):
        info = delist_info.get(code, {})
        print(f"    {code} {info.get('name',''):8s} 被选{cnt}个月 退市:{info.get('delist_date','?')} 末价:{info.get('last_price',0):.2f}")
else:
    print("  T5组合中未选中退市股")

# 保存结果
results_summary = {
    'no_delist': {k: {kk: vv for kk, vv in v.items() if kk not in ['pv', 'returns', 'selected_delist']} 
                  for k, v in results_no_delist.items()},
    'with_delist': {k: {kk: vv for kk, vv in v.items() if kk not in ['pv', 'returns', 'selected_delist']} 
                    for k, v in results_with_delist.items()},
    't5_selected_delist': t5_delist,
}
with open(f'{DATA_DIR}/backtest_comparison_v2.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已保存: {DATA_DIR}/backtest_comparison_v2.json")
