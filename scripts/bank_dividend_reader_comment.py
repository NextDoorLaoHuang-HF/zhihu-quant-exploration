#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读者评论验证：定投工行的收益对"起点" vs "终点"的敏感性。

背景（2026-08-01 知乎读者评论）：
"从最顶端买工行，到现在工行又差不多回到顶端，这个定投上是一个非常好的结果，
叫做微笑曲线。你再换个情况，从06年的最底端3.4开始买工行，然后一直定投到
2022年4.6，再看这个收益率，会惨不忍睹。"

本脚本用与 bank_dividend_backtest.py 完全相同的规则（月投1000、hfq回撤10%加倍、
分红再投、红利税分档、佣金万2.5最低5元），只改变起点/终点，验证：
- 读者场景（06年低点起投 → 2022年）的 XIRR 到底是多少；
- 同样终点下换起点、同样起点下换终点，哪个影响更大。

结论（2026-08-01 实证）：定投 XIRR 对终点高度敏感、对起点几乎不敏感。
同样终点2022年末：06低点起投4.42% vs 07大顶起投4.55%（几乎一样）；
同样起点2006年11月：终点2022年末4.42% vs 终点2026-07 9.73%（差一倍多）。
文章的10.1%本质是"终点选在2026年高位"的终点效应，与起点无关。
"""
import pandas as pd
from bank_dividend_backtest import load_stock, Portfolio, xirr

DATA = load_stock.__globals__['DATA']


def dca_to(start_date, end_date, monthly=1000, dip_multiplier=2.0,
           dip_threshold=0.10, dip_lookback=250):
    """月定投到指定终点（与 bank_dividend_backtest.dca() 同规则，仅终点可变）"""
    raw, hfq, div = load_stock('601398')
    pf = Portfolio(raw, div)
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    months = pd.date_range(start, end, freq='MS')
    invest_days = []
    for m in months:
        idx = raw.index.searchsorted(m)
        if idx < len(raw.index) and raw.index[idx] <= end:
            invest_days.append(raw.index[idx])
    invest_days = sorted(set(invest_days))
    div_events = [(row['除权除息日'], row['派息'] / 10.0,
                   row.get('送股', 0) or 0, row.get('转增', 0) or 0)
                  for _, row in div.iterrows() if start < row['除权除息日'] <= end]
    hfq_close = hfq['close']
    n_dip = 0
    invest_set = set(invest_days)
    div_map = {}
    for exd, ps, sg, zh in div_events:
        td, _ = pf.price_on(exd)
        if td is not None:
            div_map[td] = (exd, ps, sg, zh)
    for d in raw.index[raw.index >= invest_days[0]]:
        if d > end:
            break
        if d in invest_set:
            loc = hfq_close.index.searchsorted(d)
            if loc > 0:
                win = hfq_close.iloc[max(0, loc - dip_lookback):loc + 1]
                peak = win.max()
                cur = hfq_close.iloc[min(loc, len(hfq_close) - 1)]
                amt = monthly * dip_multiplier if cur < peak * (1 - dip_threshold) else monthly
                if amt > monthly:
                    n_dip += 1
            else:
                amt = monthly
            pf.buy(d, amt)
        if d in div_map:
            exd, ps, sg, zh = div_map[d]
            pf.apply_dividend(exd, ps, sg, zh)
    final = pf.market_value(end)
    r = xirr(pf.cashflows, end, final)
    return dict(start=str(start.date()), end=str(end.date()),
                invested=round(pf.total_invested, 0), final=round(final, 0),
                xirr=round(r * 100, 2) if r else None, n_dip=n_dip,
                price=round(raw['close'].loc[end], 2))


if __name__ == '__main__':
    scenarios = [
        ('2006-11-01', '2022-01-17', '读者场景: 06年低点起投 -> 2022年价格4.65'),
        ('2006-11-01', '2022-12-30', '读者场景: 06年低点起投 -> 2022年末(4.34)'),
        ('2006-11-01', '2026-07-28', '对照: 06年低点起投 -> 2026-07(文章终点)'),
        ('2007-10-08', '2022-12-30', '对照: 07年大顶起投 -> 2022年末'),
        ('2007-10-08', '2026-07-28', '对照: 07年大顶起投 -> 2026-07(文章场景)'),
    ]
    print(f"{'起点':<12} {'终点':<12} {'投入':>8} {'终值':>10} {'XIRR':>7} {'终点价':>6} 说明")
    for s, e, note in scenarios:
        r = dca_to(s, e)
        print(f"{r['start']:<12} {r['end']:<12} {r['invested']:>8,} {r['final']:>10,} "
              f"{r['xirr']:>6}% {r['price']:>6}  {note}")
