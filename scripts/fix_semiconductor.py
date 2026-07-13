"""
修复半导体ETF复权bug：对比三种方案
1. 原始错误复权（>40%阈值误判除权）
2. 不处理（原样数据）
3. 正确前复权（用东财接口重拉）
然后给出双均线回测的修正数字
"""
import akshare as ak
import pandas as pd
import numpy as np
import os, warnings, json
warnings.filterwarnings('ignore')

os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'

# ============================================================
# 方案1: 原始错误复权（当前代码做的事）
# ============================================================
df_raw = ak.fund_etf_hist_sina(symbol='sh512480')
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw[df_raw['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
df_raw.set_index('date', inplace=True)

# 当前错误复权逻辑
df_wrong = df_raw.copy()
df_wrong['ret'] = df_wrong['close'].pct_change()
for sd in df_wrong[df_wrong['ret'].abs() > 0.40].index:
    idx = df_wrong.index.get_loc(sd)
    if idx > 0:
        ratio = df_wrong.iloc[idx]['close'] / df_wrong.iloc[idx-1]['close']
        df_wrong.iloc[:idx, df_wrong.columns.get_loc('close')] *= ratio

print("=== 半导体ETF 三组数据对比 ===")
print(f"原始数据: 起价{df_raw['close'].iloc[0]:.3f} 终价{df_raw['close'].iloc[-1]:.3f}")
print(f"错误复权: 起价{df_wrong['close'].iloc[0]:.3f} 终价{df_wrong['close'].iloc[-1]:.3f}")

# ============================================================
# 方案2: 东财前复权接口（正确数据）
# ============================================================
try:
    df_correct = ak.stock_zh_a_hist(symbol='512480', period='daily',
                                      start_date='20200101', end_date='20260713',
                                      adjust='qfq')
    if df_correct is not None and len(df_correct) > 0:
        df_correct['date'] = pd.to_datetime(df_correct['日期'])
        df_correct = df_correct.sort_values('date').reset_index(drop=True)
        df_correct.set_index('date', inplace=True)
        print(f"东财前复权: 起价{df_correct['收盘'].iloc[0]:.3f} 终价{df_correct['收盘'].iloc[-1]:.3f}")
    else:
        df_correct = None
        print("东财前复权: 无数据(接口不可用)")
except Exception as e:
    df_correct = None
    print(f"东财前复权: {str(e)[:60]}")

# ============================================================
# 方案3: 腾讯前复权（备选）
# ============================================================
try:
    import requests
    url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    r = requests.get(url, params={'param': 'sh512480,day,2020-01-01,2026-07-13,640,qfq'}, timeout=10)
    data = r.json()
    klines = data.get('data', {}).get('sh512480', {}).get('qfqday', [])
    if klines:
        df_tx = pd.DataFrame(klines, columns=['date','open','close','high','low','volume'])
        df_tx['date'] = pd.to_datetime(df_tx['date'])
        df_tx['close'] = df_tx['close'].astype(float)
        df_tx = df_tx.sort_values('date').reset_index(drop=True)
        df_tx.set_index('date', inplace=True)
        print(f"腾讯前复权: 起价{df_tx['close'].iloc[0]:.3f} 终价{df_tx['close'].iloc[-1]:.3f}")
    else:
        df_tx = None
except:
    df_tx = None

# ============================================================
# 跑双均线回测对比（5日/20日均线，原文用的参数）
# ============================================================
def dual_ma_test(prices, label):
    """双均线DMA(5,20)回测，返回年化和超额"""
    p = prices.copy()
    daily_ret = p.pct_change()
    ma_s, ma_l = p.rolling(5).mean(), p.rolling(20).mean()
    sig = (ma_s > ma_l).astype(int).shift(1)
    strategy_ret = sig * daily_ret
    # 佣金
    strategy_ret = strategy_ret - sig.diff().abs() * 0.00025
    
    years = (p.index[-1] - p.index[0]).days / 365.25
    dma_cum = (1 + strategy_ret).prod()
    bh_cum = (1 + daily_ret).prod()
    dma_ann = dma_cum ** (1/years) - 1
    bh_ann = bh_cum ** (1/years) - 1
    
    # 夏普
    dma_sharpe = strategy_ret.mean() / strategy_ret.std() * np.sqrt(252) if strategy_ret.std() > 0 else 0
    bh_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    
    # 胜率（月度）
    dma_mth = strategy_ret.resample('M').apply(lambda x: (1+x).prod()-1)
    bh_mth = daily_ret.resample('M').apply(lambda x: (1+x).prod()-1)
    win_rate = (dma_mth > bh_mth).mean()
    
    print(f"\n  [{label}]")
    print(f"    策略年化{dma_ann:.1%} 夏普{dma_sharpe:.2f} | 买持年化{bh_ann:.1%} 夏普{bh_sharpe:.2f}")
    print(f"    超额{dma_ann-bh_ann:+.1%} | 月度胜率{win_rate:.0%}")
    return dma_ann, bh_ann, dma_ann - bh_ann, win_rate

print("\n=== 双均线 DMA(5,20) 回测对比 ===")

# 用原始数据（不复权）
r1 = dual_ma_test(df_raw['close'], "原始数据(不复权)——注意有分拆缺口")

# 用错误复权数据
r2 = dual_ma_test(df_wrong['close'], "错误复权(>40%阈值)——当前脚本结果")

# 用东财前复权
if df_correct is not None:
    r3 = dual_ma_test(df_correct['收盘'], "东财前复权(正确)")

# 用腾讯前复权
if df_tx is not None:
    r4 = dual_ma_test(df_tx['close'], "腾讯前复权(正确)")

# ============================================================
# 结论
# ============================================================
print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("""
受影响的内容：
  1. chart2_heatmap.png 半导体一行 → 数字基于错误复权
  2. 文章正文\"在半导体上勉强追平买持\" → 基于错误数据
  3. 01/02脚本中所有涉及 sh512480 的回测结果

不受影响的内容：
  1. strategy_grid.png（画的创业板，非半导体）
  2. 其余5个品种的网格/双均线结果
  3. T5微盘、FF三因子、趋势跟踪等其他章节
""")
