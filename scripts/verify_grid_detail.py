"""深入检查半导体ETF的复权问题，以及01脚本的扫描结果"""
import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== 检查半导体ETF (sh512480) 原始数据 ===")
df = ak.fund_etf_hist_sina(symbol='sh512480')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2019-01-01'].sort_values('date').reset_index(drop=True)
df.set_index('date', inplace=True)

print(f"数据范围: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")
print(f"列名: {list(df.columns)}")

# 检查单日大幅波动
df['ret'] = df['close'].pct_change()
big_moves = df[df['ret'].abs() > 0.40]
print(f"\n单日涨跌>40%的天数: {len(big_moves)}")
if len(big_moves) > 0:
    print(big_moves[['close', 'ret']].to_string())

# 检查更合理的除权阈值
big_moves_10 = df[df['ret'].abs() > 0.10]
print(f"\n单日涨跌>10%的天数: {len(big_moves_10)}")
if len(big_moves_10) > 0:
    print(big_moves_10[['close', 'ret']].head(10).to_string())

# 前复权处理
df_adj = df.copy()
for sd in df_adj[df_adj['ret'].abs() > 0.40].index:
    idx = df_adj.index.get_loc(sd)
    if idx > 0:
        ratio = df_adj.iloc[idx]['close'] / df_adj.iloc[idx-1]['close']
        df_adj.iloc[:idx, df_adj.columns.get_loc('close')] *= ratio
        print(f"  除权日 {sd.date()}: ratio={ratio:.4f}")

p_raw = df['close']
p_adj = df_adj['close']

print(f"\n原始close: 起{p_raw.iloc[0]:.3f} 终{p_raw.iloc[-1]:.3f}")
print(f"复权close: 起{p_adj.iloc[0]:.3f} 终{p_adj.iloc[-1]:.3f}")
print(f"原始总收益: {p_raw.iloc[-1]/p_raw.iloc[0]-1:.1%}")
print(f"复权总收益: {p_adj.iloc[-1]/p_adj.iloc[0]-1:.1%}")

# 网格回测用原始vs复权数据
def grid_test(p, label=""):
    grid_pct, base_pos, max_grids = 0.05, 0.6, 10
    cash, shares = 100000 * (1-base_pos), 100000 * base_pos / p.iloc[0]
    grid_base, current_grid = p.iloc[0], 0
    grid_vals, grid_cap = [100000], 100000 * 0.05
    for i, (dt, pr) in enumerate(p.items()):
        if i == 0: continue
        gp = int(np.log(pr / grid_base) / np.log(1 + grid_pct))
        if abs(gp) > max_grids: gp = max_grids if gp > 0 else -max_grids
        ch = gp - current_grid
        if ch < 0 and cash >= grid_cap * abs(ch):
            shares += grid_cap * abs(ch) / pr
            cash -= grid_cap * abs(ch) * 1.0005
        elif ch > 0 and shares >= (grid_cap * ch) / pr:
            shares -= grid_cap * ch / pr
            cash += grid_cap * ch * pr * 0.9995
        current_grid = gp
        grid_base = pr * (1 + grid_pct) ** (-gp)
        grid_vals.append(cash + shares * pr)
    gpv = pd.Series(grid_vals, index=p.index)
    bh = 100000 / p.iloc[0] * p
    years = (p.index[-1] - p.index[0]).days / 365.25
    g_ann = (gpv.iloc[-1]/100000)**(1/years)-1
    b_ann = (bh.iloc[-1]/100000)**(1/years)-1
    print(f"  {label}: 网格{g_ann:.1%} vs 买持{b_ann:.1%} 超额{g_ann-b_ann:+.1%}")
    return g_ann, b_ann

print("\n--- 半导体 原始 vs 复权 ---")
grid_test(p_raw, "原始数据")
grid_test(p_adj, "复权数据")

# 检查所有品种的除权情况
print("\n=== 所有品种除权检查 ===")
symbols = {
    '创业板': 'sz159915', '半导体': 'sh512480', '证券': 'sh512880',
    '中证1000': 'sh512100', '科创50': 'sh588000', '纳指': 'sh513100',
}
for name, code in symbols.items():
    try:
        d = ak.fund_etf_hist_sina(symbol=code)
        d['date'] = pd.to_datetime(d['date'])
        d = d[d['date'] >= '2019-01-01'].sort_values('date').reset_index(drop=True)
        d.set_index('date', inplace=True)
        d['ret'] = d['close'].pct_change()
        big = d[d['ret'].abs() > 0.40]
        big10 = d[d['ret'].abs() > 0.10]
        print(f"  {name:6s}: >40%={len(big)}, >10%={len(big10)}, 起{d['close'].iloc[0]:.3f}终{d['close'].iloc[-1]:.3f}")
    except Exception as e:
        print(f"  {name:6s}: {e}")

# 关键：01脚本的 run_grid_scan 用的是 get_etf_data，复权阈值0.40
# 如果半导体没有>40%的单日波动，复权不会被触发，数据是原始的
# 那么01脚本和我的验证用的数据应该一致

# 但01脚本的 grid_backtest 和我验证用的逻辑完全一致
# 所以01脚本如果正确运行，应该得到和我一样的结果：3赢3输
# 文章说"一组都没跑赢"是错的

# 让我检查01脚本是否有其他差异
# 01脚本 grid_backtest 用的参数扫描：grid_params = [0.03, 0.05, 0.08, 0.10], base_positions = [0.5, 0.6, 0.7]
# gen_strategy_charts 只画了 5%网格 60%底仓
# 但01脚本测了4×3=12组per品种 ×6品种 = 72组
# 如果72组里有3个品种的某些参数是网格赢的，那文章说"一组都没跑赢"就是错的

print("\n=== 72组全面扫描(01脚本逻辑复现) ===")
grid_params = [0.03, 0.05, 0.08, 0.10]
base_positions = [0.5, 0.6, 0.7]
win_count = 0
total_count = 0

for name, code in symbols.items():
    try:
        d = ak.fund_etf_hist_sina(symbol=code)
        d['date'] = pd.to_datetime(d['date'])
        d = d[d['date'] >= '2019-01-01'].sort_values('date').reset_index(drop=True)
        d.set_index('date', inplace=True)
        d['ret'] = d['close'].pct_change()
        for sd in d[d['ret'].abs() > 0.40].index:
            idx = d.index.get_loc(sd)
            if idx > 0:
                ratio = d.iloc[idx]['close'] / d.iloc[idx-1]['close']
                d.iloc[:idx, d.columns.get_loc('close')] *= ratio
        p = d['close']
        years = (p.index[-1] - p.index[0]).days / 365.25
        bh_ann = (p.iloc[-1]/p.iloc[0])**(1/years)-1
        
        for gp in grid_params:
            for bp in base_positions:
                cash, shares = 100000 * (1-bp), 100000 * bp / p.iloc[0]
                gb, cg = p.iloc[0], 0
                gv = [100000]
                gc = 100000 * 0.05
                for i, (dt, pr) in enumerate(p.items()):
                    if i == 0: continue
                    gpos = int(np.log(pr / gb) / np.log(1 + gp))
                    if abs(gpos) > 10: gpos = 10 if gpos > 0 else -10
                    ch = gpos - cg
                    if ch < 0 and cash >= gc * abs(ch):
                        shares += gc * abs(ch) / pr
                        cash -= gc * abs(ch) * 1.00025
                    elif ch > 0 and shares >= (gc * ch) / pr:
                        shares -= gc * ch / pr
                        cash += gc * ch * pr * 0.99975
                    cg = gpos
                    gb = pr * (1 + gp) ** (-gpos)
                    gv.append(cash + shares * pr)
                gpv = pd.Series(gv, index=p.index)
                g_ann = (gpv.iloc[-1]/100000)**(1/years)-1
                excess = g_ann - bh_ann
                total_count += 1
                if excess > 0:
                    win_count += 1
                    print(f"  {name:6s} 网格{gp:.0%} 底仓{bp:.0%}: 网格{g_ann:.1%} vs 买持{bh_ann:.1%} 超额{excess:+.1%} → 网格赢")
    except Exception as e:
        print(f"  {name}: {e}")

print(f"\n总计: {win_count}/{total_count} 组网格跑赢买持")
