"""
检验网格交易参数是否过拟合
方法：将回测区间拆成两段（2019-2022 vs 2022-2026），
看两段的"最优参数"是否一致。如果不一致 → 过拟合。
同时检查参数稳定性：同一组参数在两段的表现差异。
"""
import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

etf_cache = {}

def get_etf_data(code):
    if code in etf_cache: return etf_cache[code]
    df = ak.fund_etf_hist_sina(symbol=code)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2019-01-01'].sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    df['ret'] = df['close'].pct_change()
    for sd in df[df['ret'].abs() > 0.40].index:
        idx = df.index.get_loc(sd)
        if idx > 0:
            ratio = df.iloc[idx]['close'] / df.iloc[idx-1]['close']
            df.iloc[:idx, df.columns.get_loc('close')] *= ratio
    etf_cache[code] = df
    return df

def grid_backtest_on_period(price, start_date, end_date, grid_pct=0.05, base_pos=0.6):
    """在指定区间跑网格回测，返回超额收益"""
    p = price.loc[start_date:end_date].copy()
    if len(p) < 100: return None
    
    cash = 100000 * (1 - base_pos)
    shares = 100000 * base_pos / p.iloc[0]
    grid_base, current_grid = p.iloc[0], 0
    grid_cap = 100000 * 0.05
    gv = [100000]
    
    for i, (dt, pr) in enumerate(p.items()):
        if i == 0: continue
        gp = int(np.log(pr / grid_base) / np.log(1 + grid_pct))
        if abs(gp) > 10: gp = 10 if gp > 0 else -10
        ch = gp - current_grid
        if ch < 0 and cash >= grid_cap * abs(ch):
            shares += grid_cap * abs(ch) / pr
            cash -= grid_cap * abs(ch) * 1.00025
        elif ch > 0 and shares >= (grid_cap * ch) / pr:
            shares -= grid_cap * ch / pr
            cash += grid_cap * ch * pr * 0.99975
        current_grid = gp
        grid_base = pr * (1 + grid_pct) ** (-gp)
        gv.append(cash + shares * pr)
    
    gpv = pd.Series(gv, index=p.index)
    years = (p.index[-1] - p.index[0]).days / 365.25
    g_ann = (gpv.iloc[-1]/100000)**(1/years) - 1
    bh_ann = (p.iloc[-1]/p.iloc[0])**(1/years) - 1
    return g_ann - bh_ann

# ============================================================
print("=" * 70)
print("网格交易参数过拟合检验")
print("=" * 70)

symbols = {
    '创业板': 'sz159915',
    '中证1000': 'sh512100',
    '科创50': 'sh588000',
    '证券': 'sh512880',
    '纳指': 'sh513100',
}

# 两段时间窗口
periods = {
    '2019-2022': ('2019-01-01', '2022-06-30'),   # 前半段：震荡+疫情
    '2022-2026': ('2022-07-01', '2026-07-13'),   # 后半段：反弹+震荡
}

grid_params = [0.03, 0.05, 0.08, 0.10]
base_positions = [0.5, 0.6, 0.7]

all_results = []

for name, code in symbols.items():
    try:
        p = get_etf_data(code)['close']
    except:
        continue
    
    for period_name, (start, end) in periods.items():
        for gp in grid_params:
            for bp in base_positions:
                excess = grid_backtest_on_period(p, start, end, gp, bp)
                if excess is not None:
                    all_results.append({
                        'symbol': name, 'period': period_name,
                        'grid_pct': gp, 'base_pos': bp, 'excess': excess
                    })

df = pd.DataFrame(all_results)

# ============================================================
# 1. 每个品种在两段中的最优参数是否一致
# ============================================================
print("\n=== 1. 各品种两段最优参数对比 ===")
print(f"  {'品种':8s} | {'前半段最优':>20s} | {'后半段最优':>20s} | {'一致?'}")
print(f"  {'-'*65}")

for name in symbols:
    sub1 = df[(df['symbol']==name) & (df['period']=='2019-2022')]
    sub2 = df[(df['symbol']==name) & (df['period']=='2022-2026')]
    if len(sub1) == 0 or len(sub2) == 0: continue
    best1 = sub1.loc[sub1['excess'].idxmax()]
    best2 = sub2.loc[sub2['excess'].idxmax()]
    same = (best1['grid_pct'] == best2['grid_pct'] and best1['base_pos'] == best2['base_pos'])
    marker = '✅一致' if same else '❌不同'
    print(f"  {name:8s} | 网格{best1['grid_pct']:.0%}/{best1['base_pos']:.0%}底仓 {best1['excess']:+.1%} | 网格{best2['grid_pct']:.0%}/{best2['base_pos']:.0%}底仓 {best2['excess']:+.1%} | {marker}")

# ============================================================
# 2. 全样本最优参数在两段中的表现（最关键的检验）
# ============================================================
print("\n=== 2. 全样本最优参数的分段稳定性 ===")

# 先找全样本最优（合并两段平均超额）
full_best = df.groupby(['symbol', 'grid_pct', 'base_pos'])['excess'].mean().reset_index()

for name in symbols:
    sub = full_best[full_best['symbol']==name]
    if len(sub) == 0: continue
    best = sub.loc[sub['excess'].idxmax()]
    
    # 这组参数在两段的表现
    r1 = df[(df['symbol']==name) & (df['grid_pct']==best['grid_pct']) & 
            (df['base_pos']==best['base_pos']) & (df['period']=='2019-2022')]
    r2 = df[(df['symbol']==name) & (df['grid_pct']==best['grid_pct']) & 
            (df['base_pos']==best['base_pos']) & (df['period']=='2022-2026')]
    
    e1 = r1['excess'].values[0] if len(r1) > 0 else None
    e2 = r2['excess'].values[0] if len(r2) > 0 else None
    
    if e1 is not None and e2 is not None:
        stable = (e1 > 0 and e2 > 0)  # 两段都跑赢才算稳定
        marker = '✅稳定' if stable else '⚠️不稳定'
        print(f"  {name:8s} 全样本最优: 网格{best['grid_pct']:.0%}/{best['base_pos']:.0%}底仓")
        print(f"    前半段超额{e1:+.1%} | 后半段超额{e2:+.1%} → {marker}")

# ============================================================
# 3. 参数敏感性：同一品种不同参数的收益分布
# ============================================================
print("\n=== 3. 参数敏感性分析（创业板） ===")
cyb = df[df['symbol']=='创业板']
pivot = cyb.pivot_table(values='excess', index='grid_pct', columns='base_pos', aggfunc='mean')
print(f"  创业板各参数组合平均超额（两段均值）:")
print(f"  {'网格':>8s} | {'底仓50%':>8s} | {'底仓60%':>8s} | {'底仓70%':>8s}")
for gp in grid_params:
    row_str = f"  {gp:7.0%}"
    for bp in base_positions:
        val = pivot.loc[gp, bp] if (gp in pivot.index and bp in pivot.columns) else np.nan
        row_str += f" | {val:7.1%}" if not np.isnan(val) else " |     N/A"
    print(row_str)

# ============================================================
# 4. 超额收益的标准差（越大越不稳定）
# ============================================================
print("\n=== 4. 超额收益的跨段波动 ===")
for name in symbols:
    sub = df[df['symbol']==name]
    if len(sub) == 0: continue
    # 同一参数在两段的标准差
    var = sub.groupby(['grid_pct', 'base_pos'])['excess'].std().mean()
    mean_excess = sub['excess'].mean()
    print(f"  {name:8s}: 平均超额{mean_excess:+.1%}, 跨段标准差均值{var:.1%}, 变异系数{var/abs(mean_excess):.1f}" if mean_excess != 0 else f"  {name:8s}: N/A")

# ============================================================
# 5. 结论
# ============================================================
print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("""
判断标准：
  1. 两段最优参数一致 → 参数稳健，非过拟合
  2. 全样本最优参数在两段都跑赢 → 参数可靠
  3. 跨段标准差/变异系数小 → 参数稳定

如果多数品种满足以上条件 → 网格交易有稳健的alpha
如果两段最优参数完全不同 → 过拟合，最优参数是运气
""")
