#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读者提问变体：只在股息率>=5%时定投，收益率会更高吗？（知乎 Q443919043 评论）

策略规则化：
- 每月首个交易日，TTM股息率 = 过去365天已除权的每股派息之和 / 当日真实收盘价
- 股息率 >= 5%：当月买入（当月预算1000 + 场外攒的全部现金）
- 股息率 < 5%：当月1000攒成场外现金（不消失，等触发时一并投入；闲置期0收益，XIRR自然体现拖累）
- 分红再投、红利税、佣金与正文回测完全一致（复用 bank_dividend_backtest.Portfolio）
- 对照组：普通月定投（每月1000无条件）
- 公平口径：XIRR（本金投入时点不同，终值不可直接比）；期末资产=市值+场外现金

结论（2007-11大顶起点，工行）：择时11.13% vs 无脑定投10.16%，+0.97pct——
有效（避开2007-2010高价期，起点股息率仅0.18%），但幅度有限，且是后视镜规则。
"""
import pandas as pd
from bank_dividend_backtest import load_stock, Portfolio, xirr, END_DATE, STOCKS

YIELD_THRESHOLD = 0.05
TTM_DAYS = 365


def div_events_all(div):
    """全部派息事件（TTM判定需要start之前的分红历史）"""
    return [(row['除权除息日'], row['派息'] / 10.0, row.get('送股', 0) or 0, row.get('转增', 0) or 0)
            for _, row in div.iterrows() if row['除权除息日'] <= END_DATE]


def ttm_yield(events, d, price):
    """过去365天已除权派息之和 / 当日价。修正接缝伪影：跨年除权日错位导致
    窗口短暂无派息时（如2021-07-01），沿用最近一次除权派息（市场通用口径）。"""
    ttm = sum(ps for exd, ps, sg, zh in events
              if d - pd.Timedelta(days=TTM_DAYS) < exd <= d)
    if ttm == 0:
        past = [ps for exd, ps, sg, zh in events if exd <= d]
        ttm = past[-1] if past else 0.0
    return ttm / price if price > 0 else 0.0


def dca_yield_timing(code, start_date, monthly=1000):
    raw, hfq, div = load_stock(code)
    pf = Portfolio(raw, div)
    start = pd.Timestamp(start_date)
    events = div_events_all(div)
    days = raw.index[raw.index >= start]
    months = pd.Series(days, index=days).groupby([days.year, days.month]).min().tolist()
    month_set = set(months)
    div_map = {}
    for exd, ps, sg, zh in events:
        if start < exd <= END_DATE:
            td, _ = pf.price_on(exd)
            if td is not None:
                div_map[td] = (exd, ps, sg, zh)
    pool = 0.0          # 场外攒的本金（0收益）
    n_buy = n_skip = 0
    wealth = []
    for d in days:
        if d in month_set:
            pool += monthly
            y = ttm_yield(events, d, raw['close'].loc[d])
            if y >= YIELD_THRESHOLD:
                pf.buy(d, pool)
                pool = 0.0
                n_buy += 1
            else:
                n_skip += 1
        if d in div_map:
            exd, ps, sg, zh = div_map[d]
            pf.apply_dividend(exd, ps, sg, zh)
        wealth.append((d, pf.current_value(raw['close'].loc[d]) + pool))
    final = pf.market_value(END_DATE) + pool
    vs = pd.Series(dict(wealth)).sort_index()
    r = xirr(pf.cashflows, END_DATE, final) if final else None
    first_buy = pf.lots[0][0] if pf.lots else None
    return {
        'stock': STOCKS[code], 'start': str(start.date()),
        'total_invested': round(pf.total_invested, 0),
        'final_value': round(final, 0),
        'xirr': round(r * 100, 2) if r is not None else None,
        'n_buy': n_buy, 'n_skip': n_skip,
        'first_buy': str(first_buy.date()) if first_buy is not None else None,
        'pool_left': round(pool, 0),
        'max_dd': round((vs / vs.cummax() - 1).min() * 100, 1),
    }


def dca_plain(code, start_date, monthly=1000):
    """对照组：普通月定投（无条件）"""
    raw, hfq, div = load_stock(code)
    pf = Portfolio(raw, div)
    start = pd.Timestamp(start_date)
    events = div_events_all(div)
    days = raw.index[raw.index >= start]
    months = pd.Series(days, index=days).groupby([days.year, days.month]).min().tolist()
    month_set = set(months)
    div_map = {}
    for exd, ps, sg, zh in events:
        if start < exd <= END_DATE:
            td, _ = pf.price_on(exd)
            if td is not None:
                div_map[td] = (exd, ps, sg, zh)
    wealth = []
    for d in days:
        if d in month_set:
            pf.buy(d, monthly)
        if d in div_map:
            exd, ps, sg, zh = div_map[d]
            pf.apply_dividend(exd, ps, sg, zh)
        wealth.append((d, pf.current_value(raw['close'].loc[d])))
    final = pf.market_value(END_DATE)
    vs = pd.Series(dict(wealth)).sort_index()
    r = xirr(pf.cashflows, END_DATE, final) if final else None
    return {
        'stock': STOCKS[code], 'start': str(start.date()),
        'total_invested': round(pf.total_invested, 0),
        'final_value': round(final, 0),
        'xirr': round(r * 100, 2) if r is not None else None,
        'n_buy': len(months), 'n_skip': 0,
        'max_dd': round((vs / vs.cummax() - 1).min() * 100, 1),
    }


if __name__ == '__main__':
    print('=== 工商银行（601398），2007-11-01 大顶起点，每月1000 ===')
    t = dca_yield_timing('601398', '2007-11-01')
    p = dca_plain('601398', '2007-11-01')
    print(f"择时版：投入{t['total_invested']:,.0f}，终值{t['final_value']:,.0f}，XIRR {t['xirr']}%")
    print(f"  买入{t['n_buy']}个月 / 跳过{t['n_skip']}个月，首买{t['first_buy']}，期末场外现金{t['pool_left']:,.0f}，最大回撤{t['max_dd']}%")
    print(f"对照组：投入{p['total_invested']:,.0f}，终值{p['final_value']:,.0f}，XIRR {p['xirr']}%，最大回撤{p['max_dd']}%")
    print()
    print('=== 四行组合（2010-08 起，每行250/月，各自判定各自攒）===')
    ti_t = fi_t = ti_p = fi_p = 0
    for code in ['601398', '601288', '601988', '601939']:
        t = dca_yield_timing(code, '2010-08-02', monthly=250)
        p = dca_plain(code, '2010-08-02', monthly=250)
        ti_t += t['total_invested']; fi_t += t['final_value']
        ti_p += p['total_invested']; fi_p += p['final_value']
        print(f"  {t['stock']}：择时XIRR {t['xirr']}%（买{t['n_buy']}/跳{t['n_skip']}，首买{t['first_buy']}） vs 对照 {p['xirr']}%")
    print(f"  组合合计：择时投入{ti_t:,.0f}→{fi_t:,.0f}；对照投入{ti_p:,.0f}→{fi_p:,.0f}")
    print()
    print('=== 敏感性：阈值 4.5% / 5% / 5.5% / 6%（工行 2007-11 起）===')
    for thr in [0.045, 0.05, 0.055, 0.06]:
        globals()['YIELD_THRESHOLD'] = thr
        t = dca_yield_timing('601398', '2007-11-01')
        print(f"  阈值{thr*100:.1f}%：XIRR {t['xirr']}%，买{t['n_buy']}/跳{t['n_skip']}，首买{t['first_buy']}，期末现金{t['pool_left']:,.0f}")
