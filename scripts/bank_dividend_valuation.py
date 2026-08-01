#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读者评论：各历史大顶时点的市盈率/股息率参考数据。

背景（2026-08-01 知乎读者评论）：
"话说有没有各个历史大顶的市盈率股息率这些数据可以参考一下？"

对应正文"情景一"表格的四个买入时点，给出工行当日估值：
- PE_TTM：2018-01-02 起用东财估值分析接口（RPT_VALUEANALYSIS_DET）权威口径；
  2007/2015 两个大顶东财接口无数据，用 F10 财务指标（EPSJB 基本每股收益）
  自算：静态PE=当日收盘/最近已披露年报EPS，PE-TTM=当日收盘/TTM EPS
  （TTM=最近报告期累计EPS - 上年同期累计EPS + 上年全年EPS）。
- 股息率：与正文变体口径一致——过去365个自然日已除权派息合计 ÷ 当日收盘价
  （分红明细来自 data/bank_dividend/601398_dividend.csv，可完全复现）。
- PB_MRQ：2018+ 用东财；2007/2015 自算 = 当日收盘 / 最近已披露报告期每股净资产。

数据文件：data/bank_dividend/valuation_icbc.csv（本次抓取快照）。
"""
import os
import time
import pandas as pd
import requests

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'bank_dividend')
PEAKS = [  # (日期, 备注) 与正文"情景一"表格一致
    ('2007-11-01', '07年大顶（工行历史最高8.84）'),
    ('2015-06-08', '杠杆牛顶'),
    ('2018-01-24', '蓝筹顶'),
    ('2021-02-18', '核心资产顶'),
]

HDR = {'Referer': 'https://data.eastmoney.com/'}


def fetch_valuation():
    """东财估值分析（2018-01-02 起）：PE_TTM / PB_MRQ / 收盘价"""
    rows = []
    page = 1
    while True:
        url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
               "?reportName=RPT_VALUEANALYSIS_DET&columns=ALL"
               "&filter=(SECURITY_CODE%3D%22601398%22)"
               f"&pageNumber={page}&pageSize=500"
               "&sortColumns=TRADE_DATE&sortTypes=-1")
        d = requests.get(url, headers=HDR, timeout=30).json()
        batch = (d.get('result') or {}).get('data') or []
        if not batch:
            break
        rows.extend(batch)
        if len(rows) >= (d['result'].get('count') or 0):
            break
        page += 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['TRADE_DATE'])
    df = df.set_index('date').sort_index()
    return df[['CLOSE_PRICE', 'PE_TTM', 'PB_MRQ']]


def fetch_f10():
    """东财F10主要财务指标：各报告期基本每股收益EPSJB、每股净资产BPS"""
    url = ("https://datacenter.eastmoney.com/securities/api/data/v1/get"
           "?reportName=RPT_F10_FINANCE_MAINFINADATA"
           "&columns=SECUCODE,REPORT_DATE,EPSJB,BPS"
           "&filter=(SECUCODE%3D%22601398.SH%22)"
           "&pageNumber=1&pageSize=100&sortColumns=REPORT_DATE&sortTypes=-1")
    d = requests.get(url, headers={'Referer': 'https://emweb.securities.eastmoney.com/'}, timeout=30).json()
    df = pd.DataFrame((d.get('result') or {}).get('data') or [])
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
    df = df.sort_values('REPORT_DATE').set_index('REPORT_DATE')
    return df[['EPSJB', 'BPS']].dropna(subset=['EPSJB'])


def ttm_eps(f10, asof):
    """asof 时点可得的 TTM EPS：最近已披露报告期累计EPS - 上年同期累计 + 上年全年"""
    avail = f10[f10.index <= asof]
    if len(avail) < 2:
        return None
    latest = avail.iloc[-1]
    m, y = latest.name.month, latest.name.year
    if m == 12:  # 年报已出：全年EPS即TTM
        return latest['EPSJB']
    ly = y - 1
    prev_same = avail[(avail.index.year == ly) & (avail.index.month == m)]
    prev_full = avail[(avail.index.year == ly) & (avail.index.month == 12)]
    same_period = prev_same.iloc[-1]['EPSJB'] if len(prev_same) else 0.0
    full = prev_full.iloc[-1]['EPSJB'] if len(prev_full) else same_period
    return latest['EPSJB'] - same_period + full


def ttm_dividend_yield(raw_close, div, asof):
    """过去365自然日已除权派息合计 ÷ 当日收盘价（与正文变体口径一致）"""
    cutoff = asof - pd.Timedelta(days=365)
    recent = div[(div['除权除息日'] > cutoff) & (div['除权除息日'] <= asof)]
    dps = (recent['派息'] / 10.0).sum()
    idx = raw_close.index.searchsorted(asof)
    if idx >= len(raw_close):
        return None
    td = raw_close.index[idx]
    return dps / raw_close.iloc[idx], td


def main():
    raw = pd.read_csv(f'{DATA}/601398_raw.csv')
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.set_index('date').sort_index()
    div = pd.read_csv(f'{DATA}/601398_dividend.csv')
    div['除权除息日'] = pd.to_datetime(div['除权除息日'])

    val = fetch_valuation()
    f10 = fetch_f10()

    print(f"{'大顶日':<12} {'收盘价':>6} {'PE-TTM':>8} {'PB':>6} {'股息率TTM':>8}  说明")
    print('-' * 68)
    rows = []
    for d, note in PEAKS:
        ts = pd.Timestamp(d)
        close = raw['close'].iloc[raw['close'].index.searchsorted(ts)]
        # PE-TTM：2018+ 用东财口径，更早自算
        if ts >= val.index.min():
            pe = val['PE_TTM'].iloc[val.index.searchsorted(ts)]
            pb = val['PB_MRQ'].iloc[val.index.searchsorted(ts)]
            pe_src = '东财'
        else:
            eps_ttm = ttm_eps(f10, ts)
            pe = close / eps_ttm if eps_ttm else None
            bps = f10[f10.index <= ts].iloc[-1]['BPS']
            pb = close / bps if bps and bps > 0 else None
            pe_src = f'自算(EPS-TTM {eps_ttm:.2f})' if eps_ttm else 'N/A'
        dy, dy_td = ttm_dividend_yield(raw['close'], div, ts)
        print(f"{d:<12} {close:>6.2f} {pe:>7.2f}× {pb:>5.2f}× {dy*100:>7.2f}%  {note} [{pe_src}]")
        rows.append({'日期': d, '收盘价': close, 'PE_TTM': pe, 'PB': pb,
                     '股息率TTM': round(dy * 100, 2), 'PE口径': pe_src, '说明': note})
    out = pd.DataFrame(rows)
    out.to_csv(f'{DATA}/valuation_icbc.csv', index=False, encoding='utf-8-sig')
    print(f'\n已保存: {DATA}/valuation_icbc.csv')


if __name__ == '__main__':
    main()
