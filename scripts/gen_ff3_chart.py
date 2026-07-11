"""Fama-French 三因子模型 — 修正版：使用市值+历史涨跌幅"""
import warnings; warnings.filterwarnings('ignore')
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# ── 数据准备 ──────────────────────────────────────────
stock_list = ak.stock_info_a_code_name()
stock_list['sym'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

# 拉价格+股本
prices = {}
shares = {}
for _, r in sample.iterrows():
    try:
        df = ak.stock_zh_a_daily(symbol=r['sym'])
        if df is None or len(df) < 250: continue
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2019-01-01'].sort_values('date').reset_index(drop=True)
        df.set_index('date', inplace=True)
        prices[r['code']] = df['close']
        if 'outstanding_share' in df.columns:
            shares[r['code']] = df['outstanding_share']
    except: pass

pdf = pd.DataFrame(prices).dropna(how='all')
sdf = pd.DataFrame(shares).dropna(how='all')
# 只保留两者都有的
common_codes = pdf.columns.intersection(sdf.columns)
pdf = pdf[common_codes]
sdf = sdf[common_codes]
print(f'有效股票: {len(common_codes)}只')

# 市值 = 流通股本 × 收盘价（流通市值近似）
mcap = pdf * sdf

mth_price = pdf.resample('M').last()
mth_mcap = mcap.resample('M').last()
mth_ret = mth_price.pct_change()
all_mkt = mth_ret.mean(axis=1).dropna()

# ── FF3 因子构造 ─────────────────────────────────────
# Size = 流通市值 (mth_mcap)
# Value = 过去12个月涨跌幅 (涨幅大=成长, 涨跌幅小/跌幅大=价值)

smb_rets, hml_rets, dates = [], [], []

for i in range(24, len(mth_price)-1):  # 需要24个月历史
    row_mcap = mth_mcap.iloc[i].dropna()
    row_price = mth_price.iloc[i].dropna()
    common = row_mcap.index.intersection(row_price.index)
    if len(common) < 30: continue
    
    # 过去12个月涨跌幅
    if i >= 12:
        past_ret = (mth_price.iloc[i] / mth_price.iloc[i-12] - 1).dropna()
    else:
        past_ret = pd.Series(0, index=common)
    common = common.intersection(past_ret.index)
    if len(common) < 30: continue
    
    # 按市值分3组 (qcut从小到大 → 小/中/大)
    mc_rank = pd.qcut(row_mcap[common], 3, labels=['小市值','中市值','大市值'])
    # 按12月涨跌幅分3组 (qcut从小到大 → 价值/中性/成长)
    val_rank = pd.qcut(past_ret[common], 3, labels=['价值','中性','成长'])
    
    # 9组下月收益
    ret_next = mth_ret.iloc[i+1].dropna()
    
    sz_ret = {}
    for sz in ['小市值','中市值','大市值']:
        codes = common[(mc_rank==sz)]
        valid = codes.intersection(ret_next.index)
        if len(valid) >= 2:
            sz_ret[sz] = ret_next[valid].mean()
    
    vl_ret = {}
    for vl in ['价值','中性','成长']:
        codes = common[(val_rank==vl)]
        valid = codes.intersection(ret_next.index)
        if len(valid) >= 2:
            vl_ret[vl] = ret_next[valid].mean()
    
    if len(sz_ret) >= 2 and len(vl_ret) >= 2:
        smb_rets.append(sz_ret.get('小市值', 0) - sz_ret.get('大市值', 0))
        hml_rets.append(vl_ret.get('价值', 0) - vl_ret.get('成长', 0))
        dates.append(mth_ret.index[i+1])

smb = pd.Series(smb_rets, index=dates)
hml = pd.Series(hml_rets, index=dates)

smb_ann = smb.mean() * 12
smb_t = smb.mean() / smb.std() * np.sqrt(len(smb)) if smb.std() > 0 else 0
hml_ann = hml.mean() * 12
hml_t = hml.mean() / hml.std() * np.sqrt(len(hml)) if hml.std() > 0 else 0
corr = smb.corr(hml)

print(f'SMB(小-大): 年化{smb_ann:.1%}  t值{smb_t:.2f}')
print(f'HML(价值-成长): 年化{hml_ann:.1%}  t值{hml_t:.2f}')
print(f'SMB-HML 相关性: {corr:.3f}')

def nav(r): return (1 + r.fillna(0)).cumprod()

# ── 画图 ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))

# 左图：因子净值
common_s = smb.index.intersection(all_mkt.index)
ax1.plot(nav(all_mkt[common_s]).index, nav(all_mkt[common_s]).values, 
         color='#9ca3af', linewidth=1.2, linestyle='--', label='全市场平均')
ax1.plot(nav(smb).index, nav(smb).values, 
         color='#2563eb', linewidth=2, label=f'SMB 规模因子 (年化{smb_ann:.1%}, t={smb_t:.2f})')
ax1.plot(nav(hml).index, nav(hml).values, 
         color='#dc2626', linewidth=2, label=f'HML 价值因子 (年化{hml_ann:.1%}, t={hml_t:.2f})')
ax1.set_title('Fama-French 三因子 · 因子净值走势', fontsize=13, fontweight='bold')
ax1.legend(frameon=False, fontsize=10)
ax1.set_ylabel('净值 (初始=1)', fontsize=11)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# 右图：月度散点 + 相关性
ax2.scatter(smb * 100, hml * 100, alpha=0.5, color='#2563eb', s=30)
slope, intercept, r, _, _ = stats.linregress(smb.dropna() * 100, hml.dropna() * 100)
x_range = np.linspace(smb.min()*100, smb.max()*100, 100)
ax2.plot(x_range, slope * x_range + intercept, color='#dc2626', linewidth=1.5, 
         label=f'相关度 r={corr:.3f}')
ax2.axhline(y=0, color='#9ca3af', linewidth=0.5)
ax2.axvline(x=0, color='#9ca3af', linewidth=0.5)
ax2.set_xlabel('SMB 当月收益 (%)', fontsize=11)
ax2.set_ylabel('HML 当月收益 (%)', fontsize=11)
ax2.set_title(f'SMB vs HML 月度散点图 (相关度 {corr:.3f})', fontsize=12, fontweight='bold')
ax2.legend(frameon=False, fontsize=10)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.suptitle('Fama-French三因子在A股（简化版）：规模因子 vs 价值因子', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('../charts/strategy_ff3factor.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('\n✅ strategy_ff3factor.png')
