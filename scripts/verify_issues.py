"""
验证两个问题：
1. 小市值策略幸存者偏差
2. 网格交易图与文章结论矛盾
"""
import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("验证问题1：小市值策略幸存者偏差")
print("=" * 70)

# 获取当前A股列表
stock_list = ak.stock_info_a_code_name()
print(f"\n当前A股数量: {len(stock_list)}")
print(f"列名: {list(stock_list.columns)}")
print(f"前5只:\n{stock_list.head()}")

# 获取退市股票
print("\n--- 检查退市股票 ---")

# 方法1: 上交所退市
try:
    delisted_sh = ak.stock_info_sh_delist()
    print(f"上交所退市: {len(delisted_sh)}只")
    print(f"列名: {list(delisted_sh.columns)}")
    if len(delisted_sh) > 0:
        print(delisted_sh.head(3).to_string())
except Exception as e:
    print(f"上交所退市接口失败: {e}")

# 方法2: 深交所终止上市
try:
    delisted_sz = ak.stock_info_sz_delist(symbol='终止上市')
    print(f"\n深交所终止上市: {len(delisted_sz)}只")
    print(f"列名: {list(delisted_sz.columns)}")
    if len(delisted_sz) > 0:
        print(delisted_sz.head(3).to_string())
except Exception as e:
    print(f"深交所退市接口失败: {e}")

# 方法3: 尝试 stock_info_a_code_name 是否有退市标识
# 通常 akshare 的这个接口只返回当前在交易的
print("\n--- 验证: stock_info_a_code_name 是否包含退市股 ---")
# 如果一只股票在2020年存在但后来退市了，它不应该在这个列表里
# 我们用一些已知的退市股票代码来验证
known_delisted = ['600432', '600286', '000760', '002220', '002611']  # 一些已知退市的
for code in known_delisted:
    found = code in stock_list['code'].values
    print(f"  退市股 {code}: 在列表中={found}")

# 方法4: 检查 stock_zh_a_daily 对退市股票是否能获取数据
print("\n--- 验证: 退市股票是否还能获取日线数据 ---")
# 退市股票的代码前缀: 400开头(退市板块/老三板)
# 试试已知退市股票
for code in known_delisted[:2]:
    sym = f'sh{code}' if code.startswith('6') else f'sz{code}'
    try:
        df = ak.stock_zh_a_daily(symbol=sym)
        if df is not None and len(df) > 0:
            print(f"  {sym}: 获取到{len(df)}条数据, 日期范围 {df['date'].min()} ~ {df['date'].max()}")
        else:
            print(f"  {sym}: 无数据")
    except Exception as e:
        print(f"  {sym}: 获取失败 - {e}")

print("\n" + "=" * 70)
print("验证问题2：网格交易 — 策略 vs 买持 实际数值")
print("=" * 70)

# 获取创业板ETF数据
etf = ak.fund_etf_hist_sina(symbol='sz159915')
etf['date'] = pd.to_datetime(etf['date'])
etf = etf[etf['date'] >= '2019-01-01'].sort_values('date').reset_index(drop=True)
etf.set_index('date', inplace=True)

# 复权检测
etf['ret'] = etf['close'].pct_change()
for sd in etf[etf['ret'].abs() > 0.40].index:
    idx = etf.index.get_loc(sd)
    if idx > 0:
        ratio = etf.iloc[idx]['close'] / etf.iloc[idx-1]['close']
        etf.iloc[:idx, etf.columns.get_loc('close')] *= ratio

p = etf['close']
print(f"\n创业板ETF: {p.index[0].date()} ~ {p.index[-1].date()}, {len(p)}天")
print(f"  起始价: {p.iloc[0]:.3f}")
print(f"  终止价: {p.iloc[-1]:.3f}")
print(f"  买持总收益: {(p.iloc[-1]/p.iloc[0]-1):.1%}")

# 网格交易回测 (与 gen_strategy_charts.py 完全一致)
grid_pct, base_pos, max_grids = 0.05, 0.6, 10
cash, shares = 100000 * (1-base_pos), 100000 * base_pos / p.iloc[0]
grid_base, current_grid = p.iloc[0], 0
grid_vals, grid_cap = [], 100000 * 0.05

for i, (dt, pr) in enumerate(p.items()):
    if i == 0:
        grid_vals.append(100000)
        continue
    grid_pos = int(np.log(pr / grid_base) / np.log(1 + grid_pct))
    if abs(grid_pos) > max_grids:
        grid_pos = max_grids if grid_pos > 0 else -max_grids
    change = grid_pos - current_grid
    if change < 0 and cash >= grid_cap * abs(change):
        shares += grid_cap * abs(change) / pr
        cash -= grid_cap * abs(change) * 1.0005
    elif change > 0 and shares >= (grid_cap * change) / pr:
        shares -= grid_cap * change / pr
        cash += grid_cap * change * pr * 0.9995
    current_grid = grid_pos
    grid_base = pr * (1 + grid_pct) ** (-grid_pos)
    grid_vals.append(cash + shares * pr)

grid_pv = pd.Series(grid_vals, index=p.index)

# 买持基准
bh_pv = 100000 / p.iloc[0] * p

print(f"\n网格策略:")
print(f"  最终净值: {grid_pv.iloc[-1]:.0f}")
print(f"  总收益: {(grid_pv.iloc[-1]/100000-1):.1%}")
years = (p.index[-1] - p.index[0]).days / 365.25
grid_annual = (1 + grid_pv.iloc[-1]/100000-1) ** (1/years) - 1
print(f"  年化: {grid_annual:.1%}")

print(f"\n买持基准:")
print(f"  最终净值: {bh_pv.iloc[-1]:.0f}")
print(f"  总收益: {(bh_pv.iloc[-1]/100000-1):.1%}")
bh_annual = (1 + bh_pv.iloc[-1]/100000-1) ** (1/years) - 1
print(f"  年化: {bh_annual:.1%}")

print(f"\n超额收益(网格-买持): {grid_annual - bh_annual:+.1%}")
print(f"网格是否跑赢买持: {'YES - 与文章一组都没跑赢矛盾' if grid_annual > bh_annual else 'NO - 与文章一致'}")

# 画月度净值对比
grid_mth = grid_pv.resample('M').last().pct_change().dropna()
cyb_mth = p.resample('M').last().pct_change().dropna()
grid_nav = (1 + grid_mth).cumprod()
cyb_nav = (1 + cyb_mth).cumprod()

print(f"\n月度净值对比:")
print(f"  网格最终月度净值: {grid_nav.iloc[-1]:.3f}")
print(f"  买持最终月度净值: {cyb_nav.iloc[-1]:.3f}")
print(f"  网格月度年化: {grid_mth.mean()*12:.1%}")
print(f"  买持月度年化: {cyb_mth.mean()*12:.1%}")

# 检查所有6个品种的网格结果
print("\n--- 所有品种网格 vs 买持 ---")
symbols = {
    '创业板': 'sz159915',
    '半导体': 'sh512480',
    '证券': 'sh512880',
    '中证1000': 'sh512100',
    '科创50': 'sh588000',
    '纳指': 'sh513100',
}

for name, code in symbols.items():
    try:
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

        pp = df['close']
        # 网格
        c, s = 100000 * (1-base_pos), 100000 * base_pos / pp.iloc[0]
        gb, cg = pp.iloc[0], 0
        gv = [100000]
        for i, (dt, pr) in enumerate(pp.items()):
            if i == 0: continue
            gp = int(np.log(pr / gb) / np.log(1 + grid_pct))
            if abs(gp) > max_grids: gp = max_grids if gp > 0 else -max_grids
            ch = gp - cg
            if ch < 0 and c >= grid_cap * abs(ch):
                s += grid_cap * abs(ch) / pr
                c -= grid_cap * abs(ch) * 1.0005
            elif ch > 0 and s >= (grid_cap * ch) / pr:
                s -= grid_cap * ch / pr
                c += grid_cap * ch * pr * 0.9995
            cg = gp
            gb = pr * (1 + grid_pct) ** (-gp)
            gv.append(c + s * pr)
        gpv = pd.Series(gv, index=pp.index)
        
        bh = 100000 / pp.iloc[0] * pp
        g_ret = (gpv.iloc[-1]/100000 - 1)
        b_ret = (bh.iloc[-1]/100000 - 1)
        yrs = (pp.index[-1] - pp.index[0]).days / 365.25
        g_ann = (1+g_ret)**(1/yrs)-1 if yrs > 0 else 0
        b_ann = (1+b_ret)**(1/yrs)-1 if yrs > 0 else 0
        
        win = "网格赢" if g_ann > b_ann else "买持赢"
        print(f"  {name:6s}: 网格{g_ann:.1%} vs 买持{b_ann:.1%} 超额{g_ann-b_ann:+.1%} → {win}")
    except Exception as e:
        print(f"  {name:6s}: 获取失败 - {e}")
