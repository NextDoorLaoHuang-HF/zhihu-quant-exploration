#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉取银行股分红回测所需的全部数据 → ../data/bank_dividend/

数据源（全部经 akshare，免代理直连）：
- 不复权真实价格：新浪财经 stock_zh_a_daily(adjust='')
- 后复权价格（仅用于"跌了加仓"回撤判定）：新浪 stock_zh_a_daily(adjust='hfq')
- 分红除权明细：东方财富 stock_history_dividend_detail(indicator='分红')
- 沪深300指数：新浪 stock_zh_index_daily

⚠️ 不要使用腾讯 fqkline 的 hfq 数据——其累积复权因子严重偏小
（工行2007→2026总回报1.47x vs 真实2.31x），本项目早期踩过此坑。

用法: python3 fetch_bank_data.py
"""
import os, time
import akshare as ak
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'bank_dividend')
os.makedirs(DATA, exist_ok=True)

STOCKS = {'601398': 'sh601398', '601288': 'sh601288', '601988': 'sh601988',
          '601939': 'sh601939', '600016': 'sh600016', '600036': 'sh600036'}

for code, scode in STOCKS.items():
    for adjust, suffix in [('', 'raw'), ('hfq', 'hfq_sina')]:
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_daily(symbol=scode, adjust=adjust)
                df.to_csv(f'{DATA}/{code}_{suffix}.csv', index=False)
                print(f'{code}_{suffix}: {len(df)} rows, {df["date"].iloc[0]} ~ {df["date"].iloc[-1]}')
                break
            except Exception as e:
                print(f'{code}_{suffix} attempt {attempt+1}: {type(e).__name__} {str(e)[:100]}')
                time.sleep(6)
        time.sleep(2)
    for attempt in range(3):
        try:
            div = ak.stock_history_dividend_detail(symbol=code, indicator='分红')
            div.to_csv(f'{DATA}/{code}_dividend.csv', index=False)
            print(f'{code}_dividend: {len(div)} rows')
            break
        except Exception as e:
            print(f'{code}_dividend attempt {attempt+1}: {type(e).__name__} {str(e)[:100]}')
            time.sleep(6)
    time.sleep(2)

idx = ak.stock_zh_index_daily(symbol='sh000300')
idx.to_csv(f'{DATA}/sh000300.csv', index=False)
print(f'sh000300: {len(idx)} rows')
print('done →', DATA)
