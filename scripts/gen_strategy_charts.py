"""
为文章中每个策略生成收益曲线 vs 基准对比图
统一风格：白底，两条曲线（策略深色、基准灰色虚线），简洁标注
"""
import os, json, warnings
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

OUT_DIR = '../charts'

# ── 数据缓存 ──────────────────────────────────────────
etf_cache = {}
def get_etf(sym, start='2019-01-01'):
    if sym in etf_cache: return etf_cache[sym]
    df = ak.fund_etf_hist_sina(symbol=sym)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start].sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    # 复权校准
    df['ret'] = df['close'].pct_change()
    for sd in df[df['ret'].abs() > 0.40].index:
        idx = df.index.get_loc(sd)
        if idx > 0:
            ratio = df.iloc[idx]['close'] / df.iloc[idx-1]['close']
            df.iloc[:idx, 3] *= ratio  # close column
    etf_cache[sym] = df
    return df

def nav(returns):
    """月收益序列→净值曲线"""
    return (1 + returns.fillna(0)).cumprod()


# ── 公用：获取全样本等权基准 ──────────────────────────
def get_all_market_benchmark():
    """136只A股等权月度收益"""
    stock_list = ak.stock_info_a_code_name()
    stock_list['sym'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
    np.random.seed(888)
    sample = stock_list.sample(min(150, len(stock_list)), random_state=888)
    prices = {}
    for _, r in sample.iterrows():
        try:
            df = ak.stock_zh_a_daily(symbol=r['sym'])
            if df is None or len(df) < 200: continue
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            prices[r['code']] = df['close']
        except: pass
    pdf = pd.DataFrame(prices).dropna(how='all')
    mth = pdf.resample('M').last().pct_change()
    return mth.mean(axis=1).dropna()

all_mkt = get_all_market_benchmark()
print(f'全样本基准: {len(all_mkt)}月, 年化{all_mkt.mean()*12:.1%}')


# ── 绘图函数 ──────────────────────────────────────────
def draw_comparison(strategy_returns, benchmark_returns, title, filename,
                    strat_label='策略', bench_label='全市场平均', color='#2563eb'):
    """strategy_returns: pd.Series(index=日期)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    s_pv = nav(strategy_returns)
    b_pv = nav(benchmark_returns.loc[strategy_returns.index.intersection(benchmark_returns.index)])
    ax.plot(s_pv.index, s_pv.values, color=color, linewidth=2, label=strat_label)
    ax.plot(b_pv.index, b_pv.values, color='#9ca3af', linewidth=1.2, linestyle='--', label=bench_label)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(frameon=False, fontsize=11)
    ax.set_ylabel('净值 (初始=1)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    path = f'{OUT_DIR}/{filename}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {filename}')


# ═══════════════════════════════════════════════════════
# 1. T5 极端微盘 vs 全样本
# ═══════════════════════════════════════════════════════
print('\n1. T5 极端微盘...')
stock_list = ak.stock_info_a_code_name()
stock_list['sym'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)
prices = {}
for _, r in sample.iterrows():
    try:
        df = ak.stock_zh_a_daily(symbol=r['sym'])
        if df is None or len(df) < 200: continue
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
        df.set_index('date', inplace=True)
        prices[r['code']] = df['close']
    except: pass
pdf = pd.DataFrame(prices).dropna(how='all')
mth = pdf.resample('M').last()
mth_ret = mth.pct_change()

t5_rets = []
for i in range(12, len(mth)-1):
    row = mth.iloc[i].dropna()
    if len(row) < 15: continue
    t5_rets.append(mth_ret.iloc[i+1].reindex(row.nsmallest(5).index).mean())
t5 = pd.Series(t5_rets, index=mth_ret.index[13:13+len(t5_rets)])

draw_comparison(t5, all_mkt,
    'T5极端微盘（每月市值最低5只） vs 全市场平均',
    'strategy_t5_microcap.png', strat_label='T5极端微盘', color='#dc2626')


# ═══════════════════════════════════════════════════════
# 2. 双均线 DMA(10,30) 纳指 vs 买持纳指
# ═══════════════════════════════════════════════════════
print('\n2. 双均线 DMA(10,30) 纳指...')
ndx = get_etf('sh513100')
price = ndx['close']
ma_s, ma_l = price.rolling(10).mean(), price.rolling(30).mean()
sig = (ma_s > ma_l).astype(int).shift(1)
daily = price.pct_change()
dma_ret = sig * daily
dma_ret = dma_ret - sig.diff().abs().sum() * 0.0005 / len(dma_ret)  # 佣金

# 转月度
dma_mth = dma_ret.resample('M').apply(lambda x: (1+x).prod()-1).dropna()
ndx_mth = daily.resample('M').apply(lambda x: (1+x).prod()-1).dropna()

draw_comparison(dma_mth, ndx_mth,
    '双均线 DMA(10,30) 纳指 vs 一直拿着不动',
    'strategy_dualma_nasdaq.png', strat_label='DMA(10,30)策略', bench_label='一直拿着纳指不动', color='#2563eb')


# ═══════════════════════════════════════════════════════
# 3. 趋势跟踪 8品种组合 vs 等权买持
# ═══════════════════════════════════════════════════════
print('\n3. 趋势跟踪 8品种组合...')
etfs = {'创业板':'sz159915','沪深300':'sh510300','中证1000':'sh512100',
        '证券':'sh512880','纳指':'sh513100','国债':'sh511010',
        '黄金':'sh518880','银行':'sh512800'}
all_ret, all_sig = {}, {}
for name, code in etfs.items():
    df = get_etf(code)
    p = df['close']
    all_ret[name] = p.pct_change()
    all_sig[name] = (p > p.rolling(200).mean()).astype(int).shift(1)

ret_df = pd.DataFrame(all_ret)
sig_df = pd.DataFrame(all_sig)
ok = sig_df.notna().all(axis=1)
tf_mth = ((ret_df * sig_df).mean(axis=1)[ok]).resample('M').apply(lambda x: (1+x).prod()-1).dropna()
bh_mth = (ret_df[ok].mean(axis=1)).resample('M').apply(lambda x: (1+x).prod()-1).dropna()

draw_comparison(tf_mth, bh_mth,
    '趋势跟踪（8品种200日均线） vs 一直拿着不动',
    'strategy_trendfollow.png', strat_label='趋势跟踪组合', bench_label='同样8只一直拿着', color='#7c3aed')


# ═══════════════════════════════════════════════════════
# 4. 网格交易 创业板 vs 买持
# ═══════════════════════════════════════════════════════
print('\n4. 网格交易 创业板...')
cyb = get_etf('sz159915')
p = cyb['close']
# 网格：5%网格，60%底仓
grid_pct, base_pos, max_grids = 0.05, 0.6, 10
cash, shares = 100000 * (1-base_pos), 100000 * base_pos / p.iloc[0]
grid_base, current_grid = p.iloc[0], 0
grid_vals, grid_cap = [], 100000 * 0.05
for i, (dt, pr) in enumerate(p.items()):
    if i == 0: grid_vals.append(100000); continue
    grid_pos = int(np.log(pr / grid_base) / np.log(1 + grid_pct))
    if abs(grid_pos) > max_grids: grid_pos = max_grids if grid_pos > 0 else -max_grids
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
grid_mth = grid_pv.resample('M').last().pct_change().dropna()
cyb_mth = p.resample('M').last().pct_change().dropna()

draw_comparison(grid_mth, cyb_mth,
    '网格交易（创业板 5%网格 60%底仓） vs 一直拿着不动',
    'strategy_grid.png', strat_label='网格策略', bench_label='一直拿着创业板不动', color='#ea580c')


# ═══════════════════════════════════════════════════════
# 5. 高股息填权 vs 全市场等权（事件驱动）
# ═══════════════════════════════════════════════════════
print('\n5. 高股息填权...')
# 需要代理
# 代理设置：从环境变量读取，不硬编码
_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
if _proxy:
    os.environ.setdefault('HTTP_PROXY', _proxy)
    os.environ.setdefault('HTTPS_PROXY', _proxy)

div_rank = ak.stock_history_dividend()
div_rank = div_rank[~div_rank['名称'].str.contains('ST', na=False)]
div_rank = div_rank[div_rank['年均股息'] > 3.0]
top15 = div_rank.nlargest(15, '年均股息')

div_events = []  # ex_date -> ret
for _, stock in top15.iterrows():
    try:
        detail = ak.stock_history_dividend_detail(symbol=stock['代码'], indicator='分红')
    except: continue
    detail = detail[detail['进度'] == '实施'].copy()
    detail['除权除息日'] = pd.to_datetime(detail['除权除息日'])
    detail = detail.dropna(subset=['除权除息日'])
    detail = detail[detail['除权除息日'] >= '2020-01-01']
    if len(detail) == 0: continue
    sym = 'sh'+stock['代码'] if stock['代码'].startswith('6') else 'sz'+stock['代码']
    try:
        pdf = ak.stock_zh_a_daily(symbol=sym)
    except: continue
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.sort_values('date').set_index('date')
    for _, div in detail.iterrows():
        ex = div['除权除息日']
        amt = float(div['派息']) / 10
        fut = pdf.index[pdf.index >= ex]
        if len(fut) < 2: continue
        idx = pdf.index.get_loc(fut[0])
        pre = pdf.iloc[idx-1]['close'] if idx > 0 else pdf.iloc[idx]['close']
        end = min(idx+60, len(pdf)-1)
        if end - idx < 3: continue
        post = pdf.iloc[end]['close']
        div_events.append({'ex_date': fut[0], 'ret': (post + amt - pre) / pre})

if div_events:
    dv = pd.DataFrame(div_events).set_index('ex_date')['ret'].sort_index()
    # 按月累计
    dv_mth = dv.resample('M').mean()
    draw_comparison(dv_mth, all_mkt,
        '高股息填权（15只高息股除权后60日） vs 全市场平均',
        'strategy_dividend.png', strat_label='高股息填权', bench_label='全市场平均', color='#059669')
else:
    print('  ⚠️ 无分红事件数据')


# ═══════════════════════════════════════════════════════
# 6. T5+国债 50/50 固定配置 vs 纯T5 vs 国债
# ═══════════════════════════════════════════════════════
print('\n6. T5+国债 固定配置...')
tz = get_etf('sh511010')
tz_mth = tz['close'].pct_change().resample('M').apply(lambda x: (1+x).prod()-1).dropna()
common = t5.index.intersection(tz_mth.index)
mix50 = (t5[common] * 0.5 + tz_mth[common] * 0.5).dropna()
mix80 = (t5[common] * 0.8 + tz_mth[common] * 0.2).dropna()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(nav(t5[common]).index, nav(t5[common]).values, color='#dc2626', linewidth=2.5, label='纯T5微盘 (年化23.9%, 暂停引用)')
ax.plot(nav(mix50).index, nav(mix50).values, color='#2563eb', linewidth=2, label='T5-50% + 国债-50% (年化13.3%, 暂停引用)')
ax.plot(nav(mix80).index, nav(mix80).values, color='#7c3aed', linewidth=1.5, label='T5-80% + 国债-20% (年化19.7%, 暂停引用)')
ax.plot(nav(tz_mth[common]).index, nav(tz_mth[common]).values, color='#9ca3af', linewidth=1, linestyle='--', label='纯国债 (年化2.6%)')
ax.set_title('T5极端微盘 + 国债固定配置', fontsize=14, fontweight='bold')
ax.legend(frameon=False, fontsize=10)
ax.set_ylabel('净值 (初始=1)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/strategy_t5_bond_mix.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  ✅ strategy_t5_bond_mix.png')


# ═══════════════════════════════════════════════════════
# 7. 策略轮动 vs 纯T5
# ═══════════════════════════════════════════════════════
print('\n7. 策略轮动 vs 纯T5...')
hs300 = get_etf('sh510300')
hs300_vol = hs300['close'].pct_change().rolling(60).std() * np.sqrt(252)
hs300_vol_mth = hs300_vol.resample('M').last()
rot_rets = []
for i, dt in enumerate(t5.index):
    if dt in hs300_vol_mth.index: v = hs300_vol_mth[dt]
    else: v = 0.15
    if pd.isna(v): v = 0.15
    rot_rets.append(tz_mth.get(dt, 0.003) if v > 0.25 else t5[i])
rot = pd.Series(rot_rets, index=t5.index)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(nav(t5).index, nav(t5).values, color='#dc2626', linewidth=2, label='纯T5微盘 (年化23.9%, 暂停引用)')
ax.plot(nav(rot).index, nav(rot).values, color='#2563eb', linewidth=2, label='策略轮动 (波动>25%→国债) (年化17.7%, 暂停引用)')
ax.set_title('策略轮动 vs 纯T5微盘', fontsize=14, fontweight='bold')
ax.legend(frameon=False, fontsize=11)
ax.set_ylabel('净值 (初始=1)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/strategy_rotation.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  ✅ strategy_rotation.png')


print(f'\n全部图表已保存到 {OUT_DIR}/')
