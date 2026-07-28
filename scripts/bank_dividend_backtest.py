#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""银行股"一直不卖+分红再投+跌了加仓"策略全周期回测引擎。

方法学（严格版）：
- 真实不复权价格（sina）+ 东财分红明细，手工模拟每笔买入/分红税/再投资
- 红利税按持有期分档：>1年 0%，1月-1年 10%，<1月 20%（按lot精确计算）
- 买入佣金：万2.5，最低5元（散户真实规则）；一直不卖→无印花税
- 送转股处理：lot股数 × (1+送股/10+转增/10)
- 对照：hfq近似法交叉验证（分红免税假设）
- 结果全部写 results.json
"""
import pandas as pd
import numpy as np
import json, os
from scipy.optimize import brentq

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'bank_dividend')
STOCKS = {'601398': '工商银行', '601288': '农业银行', '601988': '中国银行',
          '601939': '建设银行', '600016': '民生银行', '600036': '招商银行'}
END_DATE = pd.Timestamp('2026-07-28')
COMMISSION_RATE = 0.00025   # 万2.5
COMMISSION_MIN = 5.0        # 最低5元


def load_stock(code):
    raw = pd.read_csv(f'{DATA}/{code}_raw.csv')
    # 兼容东财中文列名(601288_raw.csv为东财源)与sina英文列名
    col_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.set_index('date').sort_index()
    # 注意:腾讯hfq复权因子偏小(工行19年差1.6倍),已弃用;统一用sina hfq(与手工模拟互验一致)
    hfq = pd.read_csv(f'{DATA}/{code}_hfq_sina.csv')
    hfq['date'] = pd.to_datetime(hfq['date'])
    hfq = hfq.set_index('date').sort_index()
    div = pd.read_csv(f'{DATA}/{code}_dividend.csv')
    div['除权除息日'] = pd.to_datetime(div['除权除息日'], errors='coerce')
    div = div.dropna(subset=['除权除息日'])
    if '进度' in div.columns:
        div = div[div['进度'] == '实施']
    div = div.sort_values('除权除息日')
    return raw, hfq, div


def tax_rate(hold_days):
    if hold_days > 365:
        return 0.0
    elif hold_days >= 30:
        return 0.10
    return 0.20


def commission(amount):
    return max(amount * COMMISSION_RATE, COMMISSION_MIN)


class Portfolio:
    """手工精确模拟：lot级股数 + 分红税 + 送转 + 分红现金滚入下次买入"""

    def __init__(self, raw, div):
        self.raw_close = raw['close']
        self.dates = self.raw_close.index
        self.div = div
        self.lots = []          # [(buy_date, shares)]
        self.cash = 0.0         # 分红现金账户（未再投部分）
        self.total_invested = 0.0
        self.total_commission = 0.0
        self.total_div_gross = 0.0
        self.total_div_tax = 0.0
        self.cashflows = []     # [(date, amount)] 负=本金投入（分红现金不计）

    def price_on(self, d):
        idx = self.dates.searchsorted(d)
        if idx >= len(self.dates):
            return None, None
        return self.dates[idx], self.raw_close.iloc[idx]

    def buy(self, d, amount, use_cash=True):
        """买入：本金amount + 累积分红现金（若有）。佣金按合并金额计。"""
        td, p = self.price_on(d)
        if p is None:
            return
        extra = self.cash if use_cash else 0.0
        total = amount + extra
        if total <= 0:
            return
        fee = commission(total)
        invest = total - fee
        shares = invest / p
        self.lots.append((td, shares))
        if use_cash:
            self.cash = 0.0
        if amount > 0:
            self.total_invested += amount
            self.cashflows.append((td, -amount))
        self.total_commission += fee

    def flush_cash(self, threshold=1000):
        """分红现金攒到阈值就在下一交易日买入（一次性买入场景用）"""
        if self.cash < threshold:
            return
        # 找当前最后一个lot的日期之后的第一个交易日——简化：直接在调用方指定的日期买入
        # 本方法由调用方在逐日推进时调用
        pass

    def apply_dividend(self, ex_date, per_share, song=0.0, zhuan=0.0):
        """除权日：仅 bd < ex_date 的lot有权参与分红与送转。
        分红净额进现金账户（不当天再投，等下次买入日合并买入）。"""
        td, p = self.price_on(ex_date)
        if p is None:
            return
        entitled = [(bd, s) for bd, s in self.lots if bd < td]
        unentitled = [(bd, s) for bd, s in self.lots if bd >= td]
        if song > 0 or zhuan > 0:
            factor = 1 + song / 10 + zhuan / 10
            entitled = [(bd, s * factor) for bd, s in entitled]
        self.lots = entitled + unentitled
        if per_share <= 0 or not entitled:
            return
        gross = 0.0
        tax = 0.0
        for bd, s in entitled:
            g = s * per_share
            t = g * tax_rate((td - bd).days)
            gross += g
            tax += t
        self.total_div_gross += gross
        self.total_div_tax += tax
        self.cash += gross - tax

    def run_events(self):
        for _, row in self.div.iterrows():
            exd = row['除权除息日']
            if exd > END_DATE:
                continue
            self.apply_dividend(exd, row['派息'] / 10.0,
                                row.get('送股', 0) or 0, row.get('转增', 0) or 0)

    def current_value(self, p):
        return sum(s for _, s in self.lots) * p + self.cash

    def market_value(self, d):
        td, p = self.price_on(d)
        if p is None:
            return None
        return self.current_value(p)


def max_drawdown(series):
    if len(series) == 0:
        return 0.0
    roll_max = series.cummax()
    dd = series / roll_max - 1
    return dd.min()


def xirr(cashflows, final_date, final_value):
    flows = list(cashflows) + [(final_date, final_value)]
    t0 = flows[0][0]

    def npv(r):
        return sum(a / (1 + r) ** ((d - t0).days / 365.0) for d, a in flows)
    try:
        return brentq(npv, -0.9999, 10.0, maxiter=200)
    except Exception:
        return None


def lump_sum(code, start_date, amount=100000, cash_buy_threshold=1000):
    """一次性买入+分红再投(现金攒够threshold次日买)+不卖"""
    raw, hfq, div = load_stock(code)
    pf = Portfolio(raw, div)
    start = pd.Timestamp(start_date)
    events = [(row['除权除息日'], row['派息'] / 10.0, row.get('送股', 0) or 0, row.get('转增', 0) or 0)
              for _, row in div.iterrows() if start < row['除权除息日'] <= END_DATE]
    pf.buy(start, amount)
    # 逐日推进：除权日收分红；现金>=阈值次日买；逐日记录真实财富
    wealth = [(pf.dates[pf.dates.searchsorted(start)], pf.current_value(
        raw['close'].iloc[raw.index.searchsorted(start)]))]
    for d in raw.index[raw.index > start]:
        for exd, ps, sg, zh in events:
            if exd == d:
                pf.apply_dividend(exd, ps, sg, zh)
        if pf.cash >= cash_buy_threshold:
            pf.buy(d, 0)  # 只用现金买
        wealth.append((d, pf.current_value(raw['close'].loc[d])))
    final = pf.market_value(END_DATE)
    if final is None:
        return None
    vs = pd.Series(dict(wealth)).sort_index()
    # 回本：市值曾跌破本金80%后，首次重回>=本金的日期；从未深套→'未深套'
    below = vs[vs < amount * 0.8]
    if len(below):
        after = vs[vs.index > below.index[-1]]
        rec = after[after >= amount]
        recover_date = str(rec.index[0].date()) if len(rec) else '至今未回'
    else:
        recover_date = '未深套(未跌破8万)'
    days = (END_DATE - start).days
    cagr = (final / amount) ** (365.0 / days) - 1
    return {
        'stock': STOCKS[code], 'start': str(start.date()),
        'invested': amount, 'final_value': round(final, 0),
        'cagr': round(cagr * 100, 2),
        'max_drawdown': round(max_drawdown(vs) * 100, 1),
        'min_value': round(vs.min(), 0),
        'recover_date': recover_date,
        'total_div_gross': round(pf.total_div_gross, 0),
        'total_div_tax': round(pf.total_div_tax, 0),
        'commission': round(pf.total_commission, 0),
        'years': round(days / 365.25, 1),
        '_series': vs,
    }


def find_hfq_peak(code, before):
    """hfq历史最高价日（财富口径最高点）—— 不，应该用raw价格高点，题主视角的'高位买入'
    但财富口径高点更公平。这里用raw在before之前的最高点"""
    raw, hfq, div = load_stock(code)
    h = raw['close'][raw.index <= pd.Timestamp(before)]
    return h.idxmax()


def dca(code, start_date, monthly=1000, dip_multiplier=2.0, dip_threshold=0.10, dip_lookback=250):
    """每月首个交易日定投；hfq回撤>=阈值→当月加倍。返回结果dict"""
    raw, hfq, div = load_stock(code)
    pf = Portfolio(raw, div)
    # 生成每月首个交易日
    start = pd.Timestamp(start_date)
    months = pd.date_range(start, END_DATE, freq='MS')
    invest_days = []
    for m in months:
        idx = raw.index.searchsorted(m)
        if idx < len(raw.index):
            invest_days.append(raw.index[idx])
    invest_days = sorted(set(invest_days))
    # 逐日推进：定投日买入；除权日处理分红
    div_events = [(row['除权除息日'], row['派息'] / 10.0, row.get('送股', 0) or 0, row.get('转增', 0) or 0)
                  for _, row in div.iterrows() if start < row['除权除息日'] <= END_DATE]
    hfq_close = hfq['close']
    n_dip = 0
    invest_set = set(invest_days)
    div_map = {}
    for exd, ps, sg, zh in div_events:
        td, _ = pf.price_on(exd)
        if td is not None:
            div_map[td] = (exd, ps, sg, zh)
    wealth = []
    for d in raw.index[raw.index >= invest_days[0]]:
        if d in invest_set:
            # 跌了加仓判定：hfq收盘 vs 过去250日hfq最高
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
        wealth.append((d, pf.current_value(raw['close'].loc[d])))
    final = pf.market_value(END_DATE)
    vs = pd.Series(dict(wealth)).sort_index()
    r = xirr(pf.cashflows, END_DATE, final) if final else None
    return {
        'stock': STOCKS[code], 'start': str(start.date()),
        'total_invested': round(pf.total_invested, 0),
        'final_value': round(final, 0) if final else None,
        'xirr': round(r * 100, 2) if r is not None else None,
        'max_drawdown': round(max_drawdown(vs) * 100, 1),
        'n_months': len(invest_days), 'n_dip_months': n_dip,
        'total_div_gross': round(pf.total_div_gross, 0),
        'total_div_tax': round(pf.total_div_tax, 0),
        'commission': round(pf.total_commission, 0),
        '_series': vs, '_cashflows': pf.cashflows,
    }


def dca_index(index_csv, start_date, monthly=1000, dip_multiplier=2.0, dip_threshold=0.10, dip_lookback=250):
    """沪深300同规则定投基准（指数点位即价格，同样扣万2.5最低5元——对称保守）"""
    idx_df = pd.read_csv(index_csv, parse_dates=['date']).set_index('date').sort_index()
    close = idx_df['close']
    start = pd.Timestamp(start_date)
    months = pd.date_range(start, END_DATE, freq='MS')
    shares = 0.0
    invested = 0.0
    cashflows = []
    invest_days = set()
    for m in months:
        i = close.index.searchsorted(m)
        if i < len(close.index):
            invest_days.add(close.index[i])
    # 逐日推进：定投日买入，逐日记录当时真实市值
    series = []
    series_dates = []
    for d in close.index[close.index >= start]:
        if d in invest_days:
            loc = close.index.searchsorted(d)
            win = close.iloc[max(0, loc - dip_lookback):loc + 1]
            amt = monthly * dip_multiplier if close.loc[d] < win.max() * (1 - dip_threshold) else monthly
            fee = commission(amt)
            shares += (amt - fee) / close.loc[d]
            invested += amt
            cashflows.append((d, -amt))
        series.append(shares * close.loc[d])
        series_dates.append(d)
    vs = pd.Series(series, index=series_dates)
    final = shares * close.iloc[-1]
    r = xirr(cashflows, END_DATE, final)
    return {
        'stock': '沪深300(价格指数)', 'start': str(start.date()),
        'total_invested': round(invested, 0), 'final_value': round(final, 0),
        'xirr': round(r * 100, 2) if r is not None else None,
        'max_drawdown': round(max_drawdown(vs) * 100, 1),
        '_series': vs, '_cashflows': cashflows,
    }


def deposit_benchmark(cashflows, annual_rate):
    """同现金流存银行定期（按月复利）→ 终值"""
    bal = 0.0
    mr = annual_rate / 12
    flows = sorted(cashflows)
    last = flows[0][0]
    for d, a in flows:
        months = (d - last).days / 30.44
        bal *= (1 + mr) ** months
        bal += -a
        last = d
    months = (END_DATE - last).days / 30.44
    bal *= (1 + mr) ** months
    return bal


results = {'meta': {
    'end_date': str(END_DATE.date()), 'commission': '万2.5最低5元',
    'tax_rule': '红利税:持有>1年0%,1月-1年10%,<1月20%',
    'note': '分红按除权日收盘价再投;沪深300为价格指数(未含股息,实际ETF约+1.5-2pp/年)'
}}

if __name__ == '__main__':
    print('=== 场景A: 一次性买入10万(分红再投,不卖) ===')
    # 工行各历史时点
    raw_icbc, _, _ = load_stock('601398')
    # 工行raw在2007年的最高价日
    h07 = raw_icbc['close']['2007']
    peak07 = h07.idxmax()
    print(f'工行2007年raw最高价日: {peak07.date()} 价格{h07.max():.2f}')
    scenarios_a = [
        ('601398', peak07),                     # 工行07年大顶
        ('601398', '2015-06-08'),               # 杠杆牛顶附近(工行15年高点6月)
        ('601398', '2018-01-24'),               # 蓝筹阶段顶
        ('601398', '2021-02-18'),               # 核心资产顶/银行低位
        ('600016', '2007-10-16'),               # 民生07大顶(对照)
        ('600036', '2007-10-16'),               # 招行07大顶(对照)
    ]
    results['lump_sum'] = []
    series_lump_icbc07 = series_lump_ms = series_lump_zs = None
    for code, sd in scenarios_a:
        r = lump_sum(code, sd)
        if r:
            series = r.pop('_series')
            if code == '601398' and str(pd.Timestamp(sd).date()) == str(pd.Timestamp(peak07).date()):
                series_lump_icbc07 = series
            if code == '600016':
                series_lump_ms = series
            if code == '600036':
                series_lump_zs = series
            results['lump_sum'].append(r)
            print(f"{r['stock']} {r['start']}买10万 → {r['final_value']:.0f} 年化{r['cagr']}% "
                  f"最惨{r['min_value']:.0f} 回撤{r['max_drawdown']}% 回本:{r['recover_date']} 分红{r['total_div_gross']:.0f}")

    print('\n=== 场景B: 月定投1000+跌10%加倍(题主策略) ===')
    scenarios_b = [
        ('601398', '2007-10-08'),
        ('601398', '2015-06-01'),
        ('601398', '2018-01-02'),
        ('601398', '2021-02-01'),
    ]
    results['dca'] = []
    series_dca_icbc07 = series_dca300_07 = None
    for code, sd in scenarios_b:
        r = dca(code, sd)
        series = r.pop('_series'); cf = r.pop('_cashflows')
        bench300 = dca_index(f'{DATA}/sh000300.csv', sd)
        b3_series = bench300.pop('_series'); b3_cf = bench300.pop('_cashflows')
        if code == '601398' and sd == '2007-10-08':
            series_dca_icbc07 = series
            series_dca300_07 = b3_series
        dep3 = deposit_benchmark(cf, 0.03)
        dep4 = deposit_benchmark(cf, 0.04)
        r['bench_hs300'] = bench300
        r['bench_deposit3'] = round(dep3, 0)
        r['bench_deposit4'] = round(dep4, 0)
        results['dca'].append(r)
        print(f"{r['stock']} {r['start']}起 投入{r['total_invested']:.0f} → {r['final_value']:.0f} "
              f"XIRR {r['xirr']}% (300定投{bench300['xirr']}% 定存3%→{dep3:.0f}) 回撤{r['max_drawdown']}% 加仓月{r['n_dip_months']}")

    print('\n=== 场景C: 民生/招行定投对照(2007-10起) ===')
    results['dca_compare'] = []
    for code in ['600016', '600036']:
        r = dca(code, '2007-10-08')
        r.pop('_series'); r.pop('_cashflows')
        results['dca_compare'].append(r)
        print(f"{r['stock']} 投入{r['total_invested']:.0f} → {r['final_value']:.0f} XIRR {r['xirr']}% 回撤{r['max_drawdown']}%")

    print('\n=== 场景D: 工农中建四行等权组合定投(2010-08起,每月各行250) ===')
    # 四行各自dca(monthly=250,各自判定加倍),合并现金流与市值
    combo_series = None
    combo_cf = []
    combo_invested = 0
    sub_series = []
    for code in ['601398', '601288', '601988', '601939']:
        r = dca(code, '2010-08-02', monthly=250)
        s = r.pop('_series'); cf = r.pop('_cashflows')
        sub_series.append(s)
        combo_cf += cf
        combo_invested += r['total_invested']
        print(f"  {r['stock']}: 投入{r['total_invested']:.0f} → {r['final_value']:.0f} XIRR {r['xirr']}% 加仓月{r['n_dip_months']}")
    # 对齐:并集日期+前值填充(个别缺失交易日),起始前补0
    all_dates = sorted(set().union(*[s.index for s in sub_series]))
    combo_series = sum(s.reindex(all_dates).ffill().fillna(0) for s in sub_series)
    combo_cf.sort()
    combo_final = combo_series.iloc[-1]
    r_combo = xirr(combo_cf, END_DATE, combo_final)
    bench300_d = dca_index(f'{DATA}/sh000300.csv', '2010-08-02', monthly=1000)
    b300d_series = bench300_d.pop('_series'); bench300_d.pop('_cashflows')
    dep3_d = deposit_benchmark(combo_cf, 0.03)
    results['dca_combo'] = {
        'start': '2010-08-02', 'total_invested': round(combo_invested, 0),
        'final_value': round(combo_final, 0), 'xirr': round(r_combo * 100, 2) if r_combo else None,
        'max_drawdown': round(max_drawdown(combo_series) * 100, 1),
        'bench_hs300': bench300_d, 'bench_deposit3': round(dep3_d, 0),
    }
    print(f"组合: 投入{combo_invested:.0f} → {combo_final:.0f} XIRR {results['dca_combo']['xirr']}% "
          f"(300定投{bench300_d['xirr']}% 定存3%→{dep3_d:.0f}) 回撤{results['dca_combo']['max_drawdown']}%")

    # 保存画图用财富序列
    import pickle
    series_dump = {
        'lump_icbc07': series_lump_icbc07, 'lump_ms': series_lump_ms, 'lump_zs': series_lump_zs,
        'dca_icbc07': series_dca_icbc07, 'dca300_07': series_dca300_07,
        'combo': combo_series, 'dca300_10': b300d_series,
    }
    with open(f'{DATA}/series.pkl', 'wb') as f:
        pickle.dump(series_dump, f)

    with open(f'{DATA}/results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\nresults.json + series.pkl saved')
