"""第四节：ST股/低成交量冷门股/濒临退市 三个失败方向的收益曲线"""
import os, warnings
warnings.filterwarnings('ignore')
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# 复用主脚本的数据获取逻辑
stock_list = ak.stock_info_a_code_name()
stock_list['sym'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

prices = {}
volumes = {}
names = {}
for _, r in sample.iterrows():
    try:
        df = ak.stock_zh_a_daily(symbol=r['sym'])
        if df is None or len(df) < 200: continue
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
        df.set_index('date', inplace=True)
        prices[r['code']] = df['close']
        names[r['code']] = r['name']
        if 'volume' in df.columns:
            volumes[r['code']] = df['volume']
    except: pass

pdf = pd.DataFrame(prices).dropna(how='all')
mth = pdf.resample('M').last()
mth_ret = mth.pct_change()
all_mkt = mth_ret.mean(axis=1).dropna()

def nav(r): return (1 + r.fillna(0)).cumprod()

def get_strat(selector):
    rets = []
    for i in range(12, len(mth)-1):
        row = mth.iloc[i].dropna()
        if len(row) < 20: continue
        selected = selector(row, i)
        if len(selected) < 2: continue
        rets.append(mth_ret.iloc[i+1].reindex(selected).mean())
    return pd.Series(rets, index=mth_ret.index[13:13+len(rets)])

# 1. ST股
st_codes = [c for c in prices if '*ST' in names.get(c,'') or names.get(c,'').startswith('ST')]
st_ret = get_strat(lambda row, i: [c for c in st_codes if c in row.index and pd.notna(row[c])][:10])

# 2. 低成交量冷门股
vol_df = pd.DataFrame(volumes).dropna(how='all')
mth_vol = vol_df.resample('M').mean()
lowvol_ret = get_strat(lambda row, i: mth_vol.iloc[i].dropna().nsmallest(10).index.tolist() if i < len(mth_vol) else [])

# 3. T5微盘（作为成功对照）
t5_ret = get_strat(lambda row, i: row.nsmallest(5).index.tolist())

# 画图：三合一
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

plots = [
    (axes[0], st_ret, 'ST股', '#dc2626', 'ST股（机构合规禁区）'),
    (axes[1], lowvol_ret, '低成交量冷门股', '#ea580c', '低成交量冷门股（机构不关注）'),
    (axes[2], t5_ret, 'T5极端微盘', '#2563eb', 'T5极端微盘（对照·有效）'),
]

for ax, strat, label, color, title in plots:
    # 基准
    common = strat.index.intersection(all_mkt.index)
    b_pv = nav(all_mkt[common])
    ax.plot(b_pv.index, b_pv.values, color='#9ca3af', linewidth=1.2, linestyle='--', label='全市场平均')
    
    # 策略
    s_pv = nav(strat)
    ax.plot(s_pv.index, s_pv.values, color=color, linewidth=2, label=label)
    
    ann = strat.mean() * 12
    b_ann = all_mkt[common].mean() * 12
    ax.set_title(f'{title}\n{label}: {ann:.1%} vs 基准: {b_ann:.1%}', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylabel('净值', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('"机构不买的股票"三种方向对比：只有极端微盘有效', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('../charts/strategy_failed_directions.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('✅ strategy_failed_directions.png')
