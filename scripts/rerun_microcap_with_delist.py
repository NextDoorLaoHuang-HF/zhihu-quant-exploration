"""
Step 3+4: 合并退市股数据到价格面板，重跑T5/T10/T20小市值策略
对比：含退市股 vs 不含退市股
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
for code, s in list(delist_prices.items())[:5]:
    info = delist_info.get(code, {})
    print(f"  {code} {info.get('name',''):8s} {s.index[0].date()}~{s.index[-1].date()} ({len(s)}天)")

# ============================================================
# 2. 获取存活股数据（与原回测相同的随机种子和抽样）
# ============================================================
print("\n" + "=" * 70)
print("Step 2: 获取存活股数据（与原回测相同）")
print("=" * 70)

stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(
    lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')

np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

print(f"抽样: {len(sample)}只")

survive_prices = {}
survive_names = {}
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
            code = row['code']
            survive_prices[code] = df['close']
            survive_names[code] = row['name']
    except:
        pass

print(f"存活股获取成功: {len(survive_prices)}只")

# ============================================================
# 3. 构建两个价格面板
# ============================================================
print("\n" + "=" * 70)
print("Step 3: 构建价格面板")
print("=" * 70)

# 面板A：不含退市股（原始回测）
prices_no_delist = pd.DataFrame(survive_prices)
prices_no_delist = prices_no_delist.dropna(how='all')
print(f"面板A（不含退市）: {prices_no_delist.shape[0]}天 × {prices_no_delist.shape[1]}只")

# 面板B：含退市股
# 退市股数据需要对齐到统一日期索引
# 退市后价格归零（模拟退市亏损-100%）
all_dates = prices_no_delist.index.copy()

delist_panel = {}
for code, s in delist_prices.items():
    # 对齐到全市场日期
    s_aligned = s.reindex(all_dates)
    # 退市后的价格设为0（退市清算通常返还很少或零）
    last_valid = s_aligned.last_valid_index()
    if last_valid is not None:
        # 退市后的日期价格设为极小值（模拟退市损失）
        # 实际上退市股最后一天可能还有残余价值，但通常很低
        # 这里用退市前最后价格 * 0.1 作为退市后价格（保守估计）
        last_price = s_aligned[last_valid]
        s_aligned[s_aligned.index > last_valid] = last_price * 0.1
    delist_panel[code] = s_aligned

prices_with_delist = pd.DataFrame({**survive_prices, **delist_panel})
prices_with_delist = prices_with_delist.dropna(how='all')
print(f"面板B（含退市）: {prices_with_delist.shape[0]}天 × {prices_with_delist.shape[1]}只")
print(f"  其中退市股: {len(delist_panel)}只")

# ============================================================
# 4. 回测函数
# ============================================================
def run_microcap_backtest(prices_df, top_n_list=[5, 10, 20, 50]):
    """小市值策略回测"""
    monthly_prices = prices_df.resample('M').last()
    monthly_ret = monthly_prices.pct_change()
    
    results = {}
    
    for top_n in top_n_list:
        port_rets = []
        port_dates = []
        
        for i in range(12, len(monthly_prices) - 1):
            row = monthly_prices.iloc[i].dropna()
            if len(row) < top_n + 10:
                continue
            
            # 按价格排序（价格最低=市值最小的粗略代理）
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
        
        # 全样本基准
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

print("\n--- 面板B：含退市股 ---")
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
print("\n--- 退市股在组合中被选中的情况 ---")
monthly_prices_b = prices_with_delist.resample('M').last()
monthly_ret_b = monthly_prices_b.pct_change()

delist_selected = {code: 0 for code in delist_prices}
for i in range(12, len(monthly_prices_b) - 1):
    row = monthly_prices_b.iloc[i].dropna()
    if len(row) < 15:
        continue
    cheapest = row.nsmallest(5).index
    for code in cheapest:
        if code in delist_selected:
            delist_selected[code] += 1

selected_delist = {k: v for k, v in delist_selected.items() if v > 0}
if selected_delist:
    print(f"  T5组合中被选中的退市股: {len(selected_delist)}只")
    for code, cnt in sorted(selected_delist.items(), key=lambda x: -x[1]):
        info = delist_info.get(code, {})
        print(f"    {code} {info.get('name',''):8s} 被选{cnt}个月 退市日期:{info.get('delist_date','?')}")
else:
    print("  T5组合中未被选中退市股")

# 保存结果
results_summary = {
    'no_delist': {k: {kk: vv for kk, vv in v.items() if kk not in ['pv', 'returns']} 
                  for k, v in results_no_delist.items()},
    'with_delist': {k: {kk: vv for kk, vv in v.items() if kk not in ['pv', 'returns']} 
                    for k, v in results_with_delist.items()},
    'delist_selected': selected_delist,
}
with open(f'{DATA_DIR}/backtest_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已保存: {DATA_DIR}/backtest_comparison.json")
