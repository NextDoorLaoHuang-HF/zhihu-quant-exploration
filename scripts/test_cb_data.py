"""
散户优势策略：可转债低价+动量
来源: 土木老哥(3赞)可转债量化, 之前skill中CB动量策略年化23%

逻辑: 
1. 低价策略 — 买入价格最低的可转债（债底保护，机构合规限制买不了低评级）
2. 双低策略 — 价格×溢价率最低（经典可转债策略）
"""
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

OUT = '../charts'

# === 基准: 可转债ETF ===
print('获取可转债ETF基准...')
etf = ak.fund_etf_hist_sina(symbol='sh511380')
etf['date'] = pd.to_datetime(etf['date'])
etf = etf[etf['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
etf.set_index('date', inplace=True)
etf['ret'] = etf['close'].pct_change()
etf_mth = etf['ret'].resample('M').apply(lambda x: (1+x).prod()-1).dropna()
print(f'可转债ETF: {etf.index[0].date()} ~ {etf.index[-1].date()}, {len(etf_mth)}月, 年化{etf_mth.mean()*12:.1%}')

# === 获取全市场可转债列表 ===
print('\n获取可转债列表...')
cb_list = ak.bond_zh_cov()
# 只要有正股代码的
cb_list = cb_list[cb_list['正股代码'].notna()].copy()
print(f'可转债总数: {len(cb_list)}')

# 采样可转债日线 — 用bond_zh_cov_value_analysis
# 但这个API只返回单只的历史
# 试 bond_zh_hs_cov_spot 获取实时行情
print('\n获取可转债实时行情...')
try:
    spot = ak.bond_zh_hs_cov_spot()
    print(f'字段: {list(spot.columns)}')
    print(f'可转债行情: {len(spot)}只')
    print(spot.head(3).to_string())
except Exception as e:
    print(f'bond_zh_hs_cov_spot失败: {e}')

# 试 bond_zh_hs_cov_daily 
print('\n获取可转债日线(单券测试)...')
try:
    df = ak.bond_zh_hs_cov_daily(symbol='113001')
    print(f'113001日线: {len(df)}天')
    print(f'字段: {list(df.columns)}')
    print(df.tail(3))
except Exception as e:
    print(f'bond_zh_hs_cov_daily失败: {e}')
    # 试替代方案: 从value_analysis拿
    try:
        df = ak.bond_zh_cov_value_analysis(symbol='113001')
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        print(f'bond_zh_cov_value_analysis: {len(df)}天, 字段={list(df.columns)}')
        print(df.tail(3))
    except Exception as e2:
        print(f'bond_zh_cov_value_analysis也失败: {e2}')
