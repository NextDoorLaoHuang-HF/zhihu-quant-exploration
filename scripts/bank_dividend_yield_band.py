#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读者提问变体2：股息率仓位管理——高于5%逐步买入(越高越买)，低于4%逐步卖出(越低越卖)

规则化：
- 目标仓位 = clamp((股息率-4%)/(6%-4%), 0, 1)：≤4%清仓，5%半仓，≥6%满仓，中间线性
- 缓冲带版：4%~5%之间维持当前仓位不动（忠实于读者"高于5%买、低于4%卖"的表述）
- 每月首个交易日调仓一次；调仓用内部现金，不记本金投入
- 卖出成本：佣金万2.5(最低5元) + 印花税0.1%（历史大部分区间口径，偏保守）；A股个人免资本利得税
- 存量版：初始10万，无新投入——干净检验择时信号本身的价值
- 对照组：同起点买入持有（分红再投）
"""
import pandas as pd
from bank_dividend_backtest import load_stock, Portfolio, xirr, commission, END_DATE, STOCKS

STAMP_TAX = 0.001  # 卖出印花税0.1%
Y_LOW, Y_HIGH = 0.04, 0.06


def div_events_all(div):
    return [(row['除权除息日'], row['派息'] / 10.0, row.get('送股', 0) or 0, row.get('转增', 0) or 0)
            for _, row in div.iterrows() if row['除权除息日'] <= END_DATE]


def ttm_yield(events, d, price):
    """TTM股息率。修正接缝伪影：跨年除权日错位导致365天窗口短暂无派息时
    （如2021-07-01：2020年度除权07-13未至、2019年度除权06-19已滚出），
    沿用最近一次除权派息——市场通用口径，股息率不会某天突然归零。"""
    ttm = sum(ps for exd, ps, sg, zh in events
              if d - pd.Timedelta(days=365) < exd <= d)
    if ttm == 0:
        past = [ps for exd, ps, sg, zh in events if exd <= d]
        ttm = past[-1] if past else 0.0
    return ttm / price if price > 0 else 0.0


def target_pct_linear(y):
    if y >= Y_HIGH:
        return 1.0
    if y <= Y_LOW:
        return 0.0
    return (y - Y_LOW) / (Y_HIGH - Y_LOW)


def sell_to(pf, d, target_value, price):
    """FIFO卖出，使持仓市值降至target_value。卖出净额进现金账户。"""
    cur_val = sum(s for _, s in pf.lots) * price
    excess = cur_val - target_value
    if excess <= 1:
        return 0.0
    shares_to_sell = excess / price
    gross = shares_to_sell * price
    fee = commission(gross) + gross * STAMP_TAX
    pf.cash += gross - fee
    pf.total_commission += fee
    remaining = shares_to_sell
    new_lots = []
    for bd, s in pf.lots:
        if remaining <= 0:
            new_lots.append((bd, s))
        elif s <= remaining:
            remaining -= s
        else:
            new_lots.append((bd, s - remaining))
            remaining = 0
    pf.lots = new_lots
    return gross


def buy_with_cash(pf, d, cash_amount, price):
    """用内部现金买入指定金额（调仓不记本金）"""
    td, _ = pf.price_on(d)
    use = min(cash_amount, pf.cash)
    if use <= commission(use) + 1:
        return 0.0
    fee = commission(use)
    pf.lots.append((td, (use - fee) / price))
    pf.cash -= use
    pf.total_commission += fee
    return use


def band_timing(code, start_date, initial=100000, buffer_zone=False):
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
    # 初始：按首日股息率决定初始仓位（读者规则的一致应用）
    d0 = days[0]
    p0 = raw['close'].loc[d0]
    y0 = ttm_yield(events, d0, p0)
    tp = target_pct_linear(y0)
    init_tp = tp
    pf.cash = float(initial)
    pf.total_invested = float(initial)
    pf.cashflows = [(d0, -initial)]
    buy_with_cash(pf, d0, initial * tp, p0)
    n_sell = n_buy_adj = 0
    total_sold = 0.0
    trades = []
    wealth = []
    for d in days:
        p = raw['close'].loc[d]
        if d in month_set and d > d0:
            y = ttm_yield(events, d, p)
            cur_val = sum(s for _, s in pf.lots) * p
            total = cur_val + pf.cash
            cur_pct = cur_val / total if total > 0 else 0.0
            tp = target_pct_linear(y)
            if buffer_zone and Y_LOW < y < Y_HIGH:
                tp = cur_pct  # 缓冲带：维持现状
            target_val = tp * total
            if target_val < cur_val - 1:
                amt = sell_to(pf, d, target_val, p)
                total_sold += amt
                n_sell += 1
                trades.append((str(d.date()), f'{y*100:.2f}%', f'卖{amt:,.0f}→仓位{tp*100:.0f}%'))
            elif target_val > cur_val + 1:
                amt = buy_with_cash(pf, d, target_val - cur_val, p)
                if amt > 0:
                    n_buy_adj += 1
                    trades.append((str(d.date()), f'{y*100:.2f}%', f'买{amt:,.0f}→仓位{tp*100:.0f}%'))
        if d in div_map:
            exd, ps, sg, zh = div_map[d]
            pf.apply_dividend(exd, ps, sg, zh)
        wealth.append((d, pf.current_value(p)))
    final = pf.market_value(END_DATE)
    vs = pd.Series(dict(wealth)).sort_index()
    years = (END_DATE - start).days / 365.0
    cagr = (final / initial) ** (1 / years) - 1
    return {
        'stock': STOCKS[code], 'start': str(start.date()),
        'initial': initial, 'final_value': round(final, 0),
        'cagr': round(cagr * 100, 2),
        'n_sell': n_sell, 'n_buy_adj': n_buy_adj,
        'total_sold': round(total_sold, 0),
        'init_pct': round(init_tp * 100, 0),
        'init_yield': round(y0 * 100, 2),
        'max_dd': round((vs / vs.cummax() - 1).min() * 100, 1),
        'min_value': round(vs.min(), 0),
        'trades': trades,
    }


def buy_and_hold(code, start_date, initial=100000):
    raw, hfq, div = load_stock(code)
    pf = Portfolio(raw, div)
    start = pd.Timestamp(start_date)
    events = div_events_all(div)
    days = raw.index[raw.index >= start]
    div_map = {}
    for exd, ps, sg, zh in events:
        if start < exd <= END_DATE:
            td, _ = pf.price_on(exd)
            if td is not None:
                div_map[td] = (exd, ps, sg, zh)
    pf.buy(days[0], initial, use_cash=False)
    wealth = []
    for d in days:
        if d in div_map:
            exd, ps, sg, zh = div_map[d]
            pf.apply_dividend(exd, ps, sg, zh)
        # 分红现金攒够1000次日买（与正文一致：滚入下次买入）
        if pf.cash >= 1000:
            buy_with_cash(pf, d, pf.cash, raw['close'].loc[d])
        wealth.append((d, pf.current_value(raw['close'].loc[d])))
    final = pf.market_value(END_DATE)
    vs = pd.Series(dict(wealth)).sort_index()
    years = (END_DATE - start).days / 365.0
    cagr = (final / initial) ** (1 / years) - 1
    return {
        'stock': STOCKS[code], 'start': str(start.date()),
        'initial': initial, 'final_value': round(final, 0),
        'cagr': round(cagr * 100, 2),
        'max_dd': round((vs / vs.cummax() - 1).min() * 100, 1),
        'min_value': round(vs.min(), 0),
    }


if __name__ == '__main__':
    for code, starts in [('601398', ['2007-11-01', '2010-08-02', '2015-06-08'])]:
        for st in starts:
            print(f"=== {STOCKS[code]} {st} 起，10万存量 ===")
            h = buy_and_hold(code, st)
            l = band_timing(code, st, buffer_zone=False)
            b = band_timing(code, st, buffer_zone=True)
            print(f"  买入持有：终值{h['final_value']:,.0f}，年化{h['cagr']}%，最大回撤{h['max_dd']}%（最惨{h['min_value']:,.0f}）")
            print(f"  线性调仓：终值{l['final_value']:,.0f}，年化{l['cagr']}%，回撤{l['max_dd']}% | 初始{l['init_pct']:.0f}%(股息率{l['init_yield']}%)，卖{l['n_sell']}次(共{l['total_sold']:,.0f}元)/补{l['n_buy_adj']}次")
            print(f"  缓冲带版：终值{b['final_value']:,.0f}，年化{b['cagr']}%，回撤{b['max_dd']}% | 初始{b['init_pct']:.0f}%(股息率{b['init_yield']}%)，卖{b['n_sell']}次/补{b['n_buy_adj']}次")
            print(f"  缓冲带调仓明细：")
            for dt, yv, act in b['trades']:
                print(f"    {dt} 股息率{yv} {act}")
            print()
    print('=== 四行组合（2010-08 起，各2.5万）===')
    th = tl = tb = 0
    for code in ['601398', '601288', '601988', '601939']:
        h = buy_and_hold(code, '2010-08-02', 25000)
        l = band_timing(code, '2010-08-02', 25000, buffer_zone=False)
        b = band_timing(code, '2010-08-02', 25000, buffer_zone=True)
        th += h['final_value']; tl += l['final_value']; tb += b['final_value']
        print(f"  {h['stock']}：持有{h['cagr']}% vs 线性{l['cagr']}% vs 缓冲带{b['cagr']}%")
    years = (END_DATE - pd.Timestamp('2010-08-02')).days / 365.0
    print(f"  组合终值：持有{th:,.0f}（年化{(th/100000)**(1/years)-1:.2%}） vs 线性{tl:,.0f}（{(tl/100000)**(1/years)-1:.2%}） vs 缓冲带{tb:,.0f}（{(tb/100000)**(1/years)-1:.2%}）")
    print()
    print('=== 敏感性：缓冲带阈值 (3.5%,6.5%) / (4%,6%) / (4.5%,5.5%)（工行 2010-08 起）===')
    for lo, hi in [(0.035, 0.065), (0.04, 0.06), (0.045, 0.055)]:
        globals()['Y_LOW'], globals()['Y_HIGH'] = lo, hi
        b = band_timing('601398', '2010-08-02', buffer_zone=True)
        print(f"  ({lo*100:.1f}%,{hi*100:.1f}%)：年化{b['cagr']}%，卖{b['n_sell']}次/补{b['n_buy_adj']}次，回撤{b['max_dd']}%")
    globals()['Y_LOW'], globals()['Y_HIGH'] = 0.04, 0.06
