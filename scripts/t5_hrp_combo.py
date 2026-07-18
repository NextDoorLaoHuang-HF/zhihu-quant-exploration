#!/usr/bin/env python3
"""
T5 + HRP 组合回测
=================
加载 T5 月度收益（small_cap_v2 JSON），重新跑 HRP 80/20 回测，
扫描不同 T5 权重（0%→30%），对齐时间轴后计算组合指标。
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import akshare as ak
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ── Config ───────────────────────────────────────────────
T5_JSON = os.path.expanduser(
    '~/workspace/zhihu-quant-exploration/results/'
    'small_cap_v2_20260715_011613/small_cap.json'
)
T5_KEY = 'market_cap_T5_filter<2'  # 流通市值最小5只，排除<2元
HRP_PCT = 0.80  # HRP 80/20 中的 HRP 占比
OUT_DIR = os.path.expanduser(
    '~/workspace/zhihu-quant-exploration/results/t5_hrp_combo'
)
os.makedirs(OUT_DIR, exist_ok=True)

ETF_POOL = {
    '创业板': 'sz159915', '深红利': 'sz159905', '上证50': 'sh510050',
    '沪深300': 'sh510300', '中证500': 'sh510500', '中证1000': 'sh512100',
    '国债': 'sh511010', '黄金': 'sh518880',
}
CROSS_BORDER = {
    '纳指ETF': 'sh513100', '日经ETF': 'sh513520', '德国ETF': 'sh513030',
}
POOL = {**ETF_POOL, **CROSS_BORDER}


# ── 1. Load T5 monthly returns ──────────────────────────
print("📡 加载 T5 月度收益...")
with open(T5_JSON) as f:
    t5_data = json.load(f)

t5_result = t5_data['results'][T5_KEY]
t5_metrics = t5_result['metrics']
t5_monthly = pd.DataFrame(t5_result['monthly_returns'])
t5_monthly['date'] = pd.to_datetime(t5_monthly['date'])
t5_monthly = t5_monthly.set_index('date')['return']
t5_monthly.name = 'T5'
print(f"  T5: {len(t5_monthly)} 月, CAGR={t5_metrics['cagr']*100:.2f}%, "
      f"Sharpe={t5_metrics['sharpe']:.2f}")


# ── 2. Fetch ETF data and run HRP backtest ───────────────
print("\n📡 获取 ETF 数据...")

def fetch_etf(code):
    """Fetch ETF daily close with split adjustment"""
    df = ak.fund_etf_hist_sina(symbol=code)
    df['date'] = pd.to_datetime(df['date'])
    s = df.set_index('date').sort_index()['close'].astype(float)
    ret = s.pct_change()
    for d in ret[abs(ret) > 0.4].index:
        idx_pos = s.index.get_loc(d)
        prev = s.index[idx_pos - 1]
        s.loc[:prev] *= s[d] / s[prev]
    return s

prices = {}
for name, code in POOL.items():
    try:
        prices[name] = fetch_etf(code)
        print(f"  ✓ {name}: {len(prices[name])} 天, {prices[name].index[0].date()} → {prices[name].index[-1].date()}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

prices_df = pd.DataFrame(prices).dropna()
print(f"  对齐后: {len(prices_df)} 天, {prices_df.index[0].date()} → {prices_df.index[-1].date()}")


def rp_weights(prices_df, etf_list, le_date, vl=1):
    """Risk Parity weights (inverse vol)"""
    ls = le_date - pd.DateOffset(months=vl)
    inv_vols = {}
    for sec in etf_list:
        hist = prices_df[sec][(prices_df[sec].index >= ls) & (prices_df[sec].index <= le_date)]
        if len(hist) < 10: continue
        vol = hist.pct_change().dropna().std()
        if vol > 0: inv_vols[sec] = 1.0 / vol
    if not inv_vols: return {}
    total = sum(inv_vols.values())
    return {s: inv_vols.get(s, 0.0) / total for s in etf_list}


def hrp_weights(prices_df, etf_list, le_date, vl=1):
    """HRP weights via hierarchical clustering"""
    lsc = le_date - pd.DateOffset(months=24)
    returns = {}
    for sec in etf_list:
        hist = prices_df[sec][(prices_df[sec].index >= lsc) & (prices_df[sec].index <= le_date)]
        if len(hist) > 60: returns[sec] = hist.pct_change().dropna()
    if len(returns) < 3:
        return rp_weights(prices_df, etf_list, le_date, vl)
    ret_df = pd.DataFrame(returns).dropna()
    if len(ret_df) < 60:
        return rp_weights(prices_df, etf_list, le_date, vl)
    corr = ret_df.corr()
    dist = ((1 - corr) / 2).fillna(0.5)
    try:
        dm = dist.values[np.triu_indices_from(dist, k=1)]
        Z = linkage(dm, method='ward')
    except:
        return rp_weights(prices_df, etf_list, le_date, vl)
    nc = max(2, int(np.sqrt(len(etf_list))))
    clusters = fcluster(Z, nc, criterion='maxclust')
    cmap = {sec: clusters[i] for i, sec in enumerate(ret_df.columns)}
    intra = {}
    for cid in set(clusters):
        members = [s for s in ret_df.columns if cmap[s] == cid]
        intra.update(rp_weights(prices_df, members, le_date, vl))
    crets = {}
    for cid in set(clusters):
        members = [s for s in ret_df.columns if cmap[s] == cid]
        if len(members) == 1:
            crets[cid] = ret_df[members[0]]
        else:
            sw = {s: intra.get(s, 0) for s in members}
            tot = sum(sw.values()) or 1
            crets[cid] = sum(ret_df[s] * (sw[s]/tot) for s in members)
    cvols = {cid: 1.0/crt.std() for cid, crt in crets.items() if crt.std() > 0}
    if cvols:
        tcv = sum(cvols.values())
        return {sec: (cvols[cmap.get(sec, 0)]/tcv) * intra.get(sec, 0) for sec in etf_list}
    return rp_weights(prices_df, etf_list, le_date, vl)


def mom_weights(prices_df, etf_list, le_date, lb=12, tk=1):
    """Momentum weights (top K by lookback return)"""
    ls = le_date - pd.DateOffset(months=lb)
    scores = {}
    for sec in etf_list:
        hist = prices_df[sec][prices_df[sec].index <= le_date]
        if len(hist) < 2: continue
        early = hist[hist.index <= ls]
        if len(early) == 0 or early.iloc[-1] <= 0: continue
        scores[sec] = (hist.iloc[-1] / early.iloc[-1]) - 1
    if not scores: return {}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    sel = [n for n, _ in ranked[:tk]]
    if not sel: return {}
    w = 1.0 / len(sel)
    return {s: w for s in sel}


# ── 3. Monthly backtest ──────────────────────────────────
print("\n📊 运行 HRP 80/20 历史回测...")
etf_list = list(POOL.keys())

# Get aligned monthly dates from daily data
monthly_dates = prices_df.resample('ME').last().index

# Ensure we only use dates where all ETFs have data
hrp_returns = []
hrp_dates = []

# We need at least 12 months of warmup for HRP
min_idx = 12
for i in range(min_idx, len(monthly_dates) - 1):
    le_date = monthly_dates[i]  # last day of month i (weights computed EOM)
    next_date = monthly_dates[i + 1]  # next month end
    
    # Compute weights
    w_hrp = hrp_weights(prices_df, etf_list, le_date, 1)
    w_mom = mom_weights(prices_df, etf_list, le_date, 12, 1)
    
    # Blended 80/20
    blended = {}
    for sec in etf_list:
        blended[sec] = HRP_PCT * w_hrp.get(sec, 0) + (1 - HRP_PCT) * w_mom.get(sec, 0)
    
    if sum(blended.values()) == 0:
        continue
    
    # Next month return for each ETF
    month_ret = {}
    for sec in etf_list:
        prices_sec = prices_df[sec]
        start_idx = prices_sec.index.asof(le_date) if le_date in prices_sec.index else prices_sec.index[prices_sec.index.get_indexer([le_date], method='ffill')[0]]
        end_idx = prices_sec.index.asof(next_date) if next_date in prices_sec.index else prices_sec.index[prices_sec.index.get_indexer([next_date], method='ffill')[0]]
        if start_idx and end_idx and start_idx < end_idx:
            start_price = prices_sec[start_idx]
            end_price = prices_sec[end_idx]
            if start_price > 0:
                month_ret[sec] = (end_price / start_price) - 1
    
    # Portfolio return
    port_ret = sum(blended.get(sec, 0) * month_ret.get(sec, 0) for sec in etf_list)
    hrp_returns.append(port_ret)
    hrp_dates.append(next_date)

hrp_monthly = pd.Series(hrp_returns, index=hrp_dates)
hrp_monthly.name = 'HRP'
hrp_cagr = (1 + hrp_monthly).prod() ** (12 / len(hrp_monthly)) - 1
hrp_sharpe = hrp_monthly.mean() / hrp_monthly.std() * np.sqrt(12) if hrp_monthly.std() > 0 else 0
print(f"  HRP 80/20: {len(hrp_monthly)} 月, CAGR={hrp_cagr*100:.2f}%, Sharpe={hrp_sharpe:.2f}")


# ── 4. Align and combine ─────────────────────────────────
print("\n🔗 对齐时间轴 & 组合扫描...")

# Align T5 and HRP on common months
common_dates = t5_monthly.index.intersection(hrp_monthly.index)
t5_aligned = t5_monthly[common_dates]
hrp_aligned = hrp_monthly[common_dates]
print(f"  共同月份: {len(common_dates)}, {common_dates[0].date()} → {common_dates[-1].date()}")

# Correlation
corr = t5_aligned.corr(hrp_aligned)
print(f"  Pearson 相关系数: {corr:.4f}")


def compute_metrics(returns, periods_per_year=12):
    """Compute standard backtest metrics"""
    if len(returns) == 0:
        return {'cagr': np.nan, 'vol': np.nan, 'sharpe': np.nan, 'max_dd': np.nan}
    cagr = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = (nav / nav.cummax() - 1).min()
    return {'cagr': cagr, 'vol': annual_vol, 'sharpe': sharpe, 'max_dd': max_dd}


# Scan T5 weights from 0% to 35% (more granular at low end)
t5_weights = [0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
results = []

for w_t5 in t5_weights:
    w_hrp = 1.0 - w_t5
    combo = t5_aligned * w_t5 + hrp_aligned * w_hrp
    m = compute_metrics(combo)
    m['t5_weight'] = w_t5
    m['t5_cagr'] = t5_metrics['cagr']
    m['hrp_cagr'] = hrp_cagr
    results.append(m)
    print(f"  T5={w_t5*100:3.0f}% HRP={w_hrp*100:3.0f}% → "
          f"CAGR={m['cagr']*100:5.1f}%  "
          f"Sharpe={m['sharpe']:.2f}  "
          f"MaxDD={m['max_dd']*100:5.1f}%")

results_df = pd.DataFrame(results)


# ── 5. Write summary ─────────────────────────────────────
summary_path = os.path.join(OUT_DIR, 'combo_results.csv')
results_df.to_csv(summary_path, index=False, float_format='%.6f')

# Also save aligned returns for reproducibility
aligned_df = pd.DataFrame({'T5': t5_aligned, 'HRP': hrp_aligned})
aligned_path = os.path.join(OUT_DIR, 'aligned_returns.csv')
aligned_df.to_csv(aligned_path, float_format='%.8f')

print(f"\n💾 已保存: {summary_path}")
print(f"💾 已保存: {aligned_path}")


# ── 6. Charts ────────────────────────────────────────────
print("\n📈 生成图表...")
plt.rcParams['font.family'] = 'PingFang HK'
plt.rcParams['axes.unicode_minus'] = False

# Chart 1: NAV comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
colors = ['#2563eb', '#dc2626', '#7c3aed', '#059669', '#d97706', '#8b5cf6']
for i, w_t5 in enumerate([0, 0.05, 0.10, 0.20, 0.30, 1.0]):
    if w_t5 == 1.0:
        combo = t5_aligned
        lbl = '纯T5'
        color = '#dc2626'
        lw = 1.5
        ls = '--'
    else:
        combo = t5_aligned * w_t5 + hrp_aligned * (1 - w_t5)
        lbl = f'T5 {w_t5*100:.0f}%'
        color = colors[i % len(colors)]
        lw = 1.2
        ls = '-'
    nav = (1 + combo).cumprod()
    ax.plot(nav.index, nav.values, color=color, linewidth=lw, linestyle=ls, label=lbl)

# Add pure HRP
nav_hrp = (1 + hrp_aligned).cumprod()
ax.plot(nav_hrp.index, nav_hrp.values, color='#059669', linewidth=2, label='纯HRP')

ax.set_title('T5 + HRP 80/20 组合净值', fontsize=14, fontweight='bold')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_ylabel('净值 (初始=1)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Chart 2: Efficient frontier (Sharpe vs CAGR)
ax = axes[1]
sharpe_vals = results_df['sharpe'].values
cagr_vals = results_df['cagr'].values * 100
maxdd_vals = results_df['max_dd'].values * 100
t5_w = results_df['t5_weight'].values * 100

sc = ax.scatter(sharpe_vals, cagr_vals, c=t5_w, cmap='RdYlGn', s=80, edgecolors='white', linewidth=0.5, zorder=3)
for i, (s, c, m) in enumerate(zip(sharpe_vals, cagr_vals, maxdd_vals)):
    if t5_w[i] in [0, 5, 10, 15, 20, 30]:
        offset = 0.12 if i % 2 == 0 else -0.25
        ax.annotate(f'T5 {t5_w[i]:.0f}%\n(S={s:.2f}, DD={m:.0f}%)',
                    (s, c), textcoords="offset points", xytext=(8, offset*20),
                    fontsize=8, alpha=0.8)

ax.set_xlabel('夏普比率', fontsize=11)
ax.set_ylabel('年化收益 (CAGR %)', fontsize=11)
ax.set_title('T5+HRP 组合前沿', fontsize=14, fontweight='bold')
cbar = plt.colorbar(sc, ax=ax, label='T5 权重 (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
chart_path = os.path.join(OUT_DIR, 't5_hrp_combo.png')
fig.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  图表: {chart_path}")


# ── 7. Summary table ─────────────────────────────────────
print("\n" + "=" * 70)
print("T5 + HRP 80/20 组合回测结果")
print(f"回测区间: {common_dates[0].date()} → {common_dates[-1].date()} ({len(common_dates)} 月)")
print(f"T5 vs HRP 相关性: {corr:.4f}")
print("=" * 70)
print(f"{'T5权重':>8} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'波动率':>7}")
print("-" * 45)
for _, row in results_df.iterrows():
    print(f"{row['t5_weight']*100:7.0f}% {row['cagr']*100:7.2f}% {row['sharpe']:6.2f} {row['max_dd']*100:7.1f}% {row['vol']*100:7.1f}%")
print("-" * 45)
print(f"{'纯T5':>8} {t5_metrics['cagr']*100:7.2f}% {t5_metrics['sharpe']:6.2f}")
print(f"{'纯HRP':>8} {hrp_cagr*100:7.2f}% {hrp_sharpe:6.2f}")

# Best Sharpe combo
best_idx = results_df['sharpe'].idxmax()
best_row = results_df.iloc[best_idx]
print(f"\n🏆 最优夏普: T5 {best_row['t5_weight']*100:.0f}% + HRP {(1-best_row['t5_weight'])*100:.0f}%")
print(f"   CAGR={best_row['cagr']*100:.2f}%, Sharpe={best_row['sharpe']:.2f}, MaxDD={best_row['max_dd']*100:.1f}%")

print("\n✅ 完成!")
