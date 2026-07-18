"""
定投 vs 一次性投入 — 全指数多时段回测 + 图表生成
用于知乎回答「为什么说定投是最愚蠢的？」
"""
import os
os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'

import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import warnings
warnings.filterwarnings('ignore')

# 中文字体
plt.rcParams['font.family'] = ['PingFang HK', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ========== 数据获取 ==========
def get_yf(ticker):
    df = yf.download(ticker, start='2005-01-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df[('Close', ticker)]
    else:
        close_series = df['Close']
    out = pd.DataFrame({'date': pd.to_datetime(close_series.index), 'close': close_series.values})
    return out.dropna()

indices = {}
for code, name in [('sh000300','沪深300'), ('sh000905','中证500'), ('sz399006','创业板指')]:
    df = ak.stock_zh_index_daily(symbol=code)
    df['date'] = pd.to_datetime(df['date'])
    indices[name] = df.sort_values('date')[['date','close']].reset_index(drop=True)

indices['标普500'] = get_yf('^GSPC')
indices['纳斯达克'] = get_yf('^IXIC')

# ========== 回测函数 ==========
def run_dca(df, monthly=1000, start=None, end=None):
    data = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))].copy()
    if len(data) < 24: return None
    data = data.sort_values('date').reset_index(drop=True)
    data['ym'] = data['date'].dt.to_period('M')
    monthly_data = data.groupby('ym').first().reset_index()
    
    units, invested = 0, 0
    rows = []
    for _, r in monthly_data.iterrows():
        units += monthly / r['close']
        invested += monthly
        rows.append({'date': r['date'], 'nav': units * r['close'], 'invested': invested})
    df_out = pd.DataFrame(rows)
    return df_out

def run_lump(df, total, start=None, end=None):
    data = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))].copy()
    if len(data) < 2: return None
    data = data.sort_values('date').reset_index(drop=True)
    units = total / data['close'].iloc[0]
    data['nav'] = units * data['close']
    return data[['date','nav']].copy()

def metrics(nav_series, invested, dates=None):
    final = nav_series.iloc[-1]
    if dates is not None:
        years = (dates.max() - dates.min()).days / 365.25
    else:
        years = len(nav_series) / 12
    cagr = (final / invested) ** (1 / max(years, 0.5)) - 1
    peak = nav_series.expanding().max()
    max_dd = ((nav_series - peak) / peak).min()
    mr = nav_series.pct_change().dropna()
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if mr.std() > 0 else 0
    return {'cagr': cagr*100, 'max_dd': max_dd*100, 'sharpe': sharpe, 'final_value': final, 'years': years}

# ========== 生成图表 ==========

# 图1：全周期定投vs一次性净值对比（5指数子图）
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.flatten()
PERIOD_START = '2005-01-01'
PERIOD_END = '2026-07-17'

for i, (name, df) in enumerate(indices.items()):
    if i >= 5: break
    ax = axes[i]
    
    dca = run_dca(df, start=PERIOD_START, end=PERIOD_END)
    if dca is None: continue
    lump = run_lump(df, total=dca['invested'].iloc[-1], start=PERIOD_START, end=PERIOD_END)
    if lump is None: continue
    
    dca_m = metrics(dca['nav'], dca['invested'].iloc[-1])
    lump_m = metrics(lump['nav'], dca['invested'].iloc[-1], dates=lump['date'])
    
    ax.plot(dca['date'], dca['nav']/10000, color='#C11A21', linewidth=1.5, label=f"每月定投1000元 (年化{dca_m['cagr']:.1f}%)")
    ax.plot(lump['date'], lump['nav']/10000, color='#2E75B6', linewidth=1.5, linestyle='--', label=f"期初一次性投入 (年化{lump_m['cagr']:.1f}%)")
    ax.plot(dca['date'], dca['invested']/10000, color='gray', linewidth=0.8, alpha=0.5, label='累计投入')
    
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylabel('净值 (万元)', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.grid(True, alpha=0.3)

axes[5].set_visible(False)
fig.suptitle('定投 vs 一次性投入 — 全周期净值对比（2005-2026）', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/dca_vs_lump_all.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ 图1: dca_vs_lump_all.png")

# 图2：关键数据对比柱状图（定投vs一次性年化 + delta）
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 左：定投vs一次性年化
periods = [
    ('沪深300', '10-20'), ('沪深300', '15-25'), ('沪深300', '05-26'),
    ('中证500', '10-20'), ('中证500', '15-25'), ('中证500', '05-26'),
    ('标普500', '10-20'), ('标普500', '15-25'), ('标普500', '05-26'),
    ('纳斯达克', '10-20'), ('纳斯达克', '15-25'), ('纳斯达克', '05-26'),
]
p_map = {'10-20': ('2010-01-01','2019-12-31'), '15-25': ('2015-01-01','2024-12-31'), '05-26': ('2005-01-01','2026-07-17')}

x_labels = []
dca_vals, lump_vals, deltas = [], [], []
for name, p_key in periods:
    ps, pe = p_map[p_key]
    df = indices[name]
    dca = run_dca(df, start=ps, end=pe)
    if dca is None: continue
    lump = run_lump(df, total=dca['invested'].iloc[-1], start=ps, end=pe)
    if lump is None: continue
    dm = metrics(dca['nav'], dca['invested'].iloc[-1])
    lm = metrics(lump['nav'], dca['invested'].iloc[-1], dates=lump['date'])
    x_labels.append(f'{name} {p_key}')
    dca_vals.append(dm['cagr'])
    lump_vals.append(lm['cagr'])
    deltas.append(dm['cagr'] - lm['cagr'])

x = np.arange(len(x_labels))
w = 0.35
ax1 = axes[0]
bars1 = ax1.bar(x - w/2, dca_vals, w, label='定投年化', color='#C11A21', alpha=0.85)
bars2 = ax1.bar(x + w/2, lump_vals, w, label='一次性年化', color='#2E75B6', alpha=0.85)
for bar, val in zip(bars1, dca_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontsize=7)
for bar, val in zip(bars2, lump_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontsize=7)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('年化收益率 (%)', fontsize=10)
ax1.set_title('定投 vs 一次性 年化收益对比', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2, axis='y')
ax1.axhline(y=0, color='black', linewidth=0.5)

# 右：差值图
ax2 = axes[1]
colors = ['#C11A21' if d < 0 else '#2E75B6' for d in deltas]
bars = ax2.bar(x, deltas, color=colors, alpha=0.85)
for bar, val in zip(bars, deltas):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if val >= 0 else -0.8), f'{val:+.1f}%', ha='center', fontsize=8, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('定投 - 一次性 (百分点)', fontsize=10)
ax2.set_title('定投相对一次性投入的收益差（负值=定投跑输）', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.2, axis='y')
ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/dca_comparison_bars.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ 图2: dca_comparison_bars.png")

# 图3：不同时段定投胜率
fig, ax = plt.subplots(figsize=(12, 6))
summary_data = []
for name in ['沪深300', '中证500', '创业板指', '标普500', '纳斯达克']:
    df = indices[name]
    wins = 0
    total = 0
    avg_diff = 0
    all_diffs = []
    for p_name, ps, pe in [('05-15','2005-01-01','2014-12-31'),('10-20','2010-01-01','2019-12-31'),
                               ('15-25','2015-01-01','2024-12-31'),('05-26','2005-01-01','2026-07-17'),
                               ('10-26','2010-01-01','2026-07-17'),('20-26','2020-01-01','2026-07-17')]:
        dca = run_dca(df, start=ps, end=pe)
        if dca is None: continue
        lump = run_lump(df, total=dca['invested'].iloc[-1], start=ps, end=pe)
        if lump is None: continue
        dm = metrics(dca['nav'], dca['invested'].iloc[-1])
        lm = metrics(lump['nav'], dca['invested'].iloc[-1], dates=lump['date'])
        diff = dm['cagr'] - lm['cagr']
        all_diffs.append(diff)
        total += 1
        if diff > 0: wins += 1
    
    avg_dca = np.mean([d['cagr'] for d in [metrics(run_dca(df, start=s, end=e)['nav'], run_dca(df, start=s, end=e)['invested'].iloc[-1]) 
                        for s,e in [('2005-01-01','2014-12-31'),('2010-01-01','2019-12-31'),('2015-01-01','2024-12-31')] if run_dca(df, start=s, end=e) is not None]])
    summary_data.append({'指数': name, '胜率': wins/max(total,1)*100, '平均落后': np.mean(all_diffs)})

names_s = [d['指数'] for d in summary_data]
win_rates = [d['胜率'] for d in summary_data]
avg_diffs = [d['平均落后'] for d in summary_data]

bars = ax.bar(names_s, win_rates, color=['#C11A21' if w < 50 else '#2E75B6' for w in win_rates], alpha=0.85)
for bar, wr, ad in zip(bars, win_rates, avg_diffs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{wr:.0f}%\n(平均{ad:+.1f}pp)', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('定投胜出比例 (%)', fontsize=11)
ax.set_title('定投跑赢一次性的概率（6个时段中胜出的比例）', fontsize=13, fontweight='bold')
ax.axhline(y=50, color='gray', linewidth=0.8, linestyle='--', label='50%基准线')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, 100)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/dca_win_rate.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ 图3: dca_win_rate.png")

# ========== 保存完整回测数据 ==========
all_results = []
periods_full = [
    ('2005-2015', '2005-01-01', '2014-12-31'),
    ('2010-2020', '2010-01-01', '2019-12-31'),
    ('2015-2025', '2015-01-01', '2024-12-31'),
    ('2005-全期', '2005-01-01', '2026-07-17'),
    ('2010-全期', '2010-01-01', '2026-07-17'),
    ('2020-至今', '2020-01-01', '2026-07-17'),
]
for name, df in indices.items():
    for pn, ps, pe in periods_full:
        dca = run_dca(df, start=ps, end=pe)
        if dca is None: continue
        lump = run_lump(df, total=dca['invested'].iloc[-1], start=ps, end=pe)
        if lump is None: continue
        dm = metrics(dca['nav'], dca['invested'].iloc[-1])
        lm = metrics(lump['nav'], dca['invested'].iloc[-1], dates=lump['date'])
        all_results.append({
            '指数': name, '时段': pn,
            '定投年化': round(float(dm['cagr']),2), '一次性年化': round(float(lm['cagr']),2),
            '差值': round(float(dm['cagr']-lm['cagr']),2),
            '定投回撤': round(float(dm['max_dd']),2), '一次性回撤': round(float(lm['max_dd']),2),
            '定投夏普': round(float(dm['sharpe']),2),
            '定投终值': int(dm['final_value']), '投入总额': int(dca['invested'].iloc[-1]),
            '年数': round(float(dm['years']),1)
        })

with open(f'{DATA_DIR}/dca_backtest_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# 输出汇总统计
print(f"\n=== 汇总 ===")
for name in ['沪深300','中证500','创业板指','标普500','纳斯达克']:
    sub = [r for r in all_results if r['指数']==name]
    wins = sum(1 for r in sub if r['差值']>0)
    avg_cagr = np.mean([r['定投年化'] for r in sub])
    avg_lump = np.mean([r['一次性年化'] for r in sub])
    print(f"{name}: 定投{avg_cagr:.1f}% vs 一次性{avg_lump:.1f}% | 胜率{wins}/{len(sub)} | 平均落后{np.mean([r['差值'] for r in sub]):.1f}pp")

print(f"\n✅ 3张图表已保存到 {OUT_DIR}/")
print(f"✅ 回测数据已保存到 {DATA_DIR}/dca_backtest_results.json")
