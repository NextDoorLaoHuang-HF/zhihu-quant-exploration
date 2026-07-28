#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成知乎回答配图：大白话标签，关键点位注释。
图2/图6采用"市值/累计投入"归一化口径——因为跌了加仓触发次数不同，
工行与沪深300的实际投入不同(31.9万 vs 36.5万)，直接比终值有误导。"""
import pickle, json, sys, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_dividend_backtest import dca, dca_index, DATA, END_DATE

for f in ['PingFang HK', 'PingFang SC', 'Hiragino Sans GB']:
    if any(f in x.name for x in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
with open(f'{DATA}/series.pkl', 'rb') as fp:
    S = pickle.load(fp)
R = json.load(open(f'{DATA}/results.json'))

C_BANK = '#c0392b'
C_300 = '#7f8c8d'
C_DEP = '#2980b9'
C_ZS = '#e67e22'
C_MS = '#8e44ad'


def wan(x, pos=None):
    return f'{x/10000:.0f}万'


def invested_curve(cashflows, index):
    """由现金流构建累计投入曲线（对齐到index交易日）。同日期多笔须先聚合（四行组合同日4笔）"""
    df = pd.DataFrame(cashflows, columns=['d', 'a'])
    cf = (-df.groupby('d')['a'].sum()).sort_index().cumsum()
    cf = cf.reindex(cf.index.union(index)).ffill().reindex(index).ffill().fillna(0)
    return cf


# ---------- 图1: 一把梭在2007年大顶 ----------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
S['lump_icbc07'].plot(ax=ax, color=C_BANK, lw=1.8, label='工商银行（2007-11-01历史大顶买入）')
S['lump_zs'].plot(ax=ax, color=C_ZS, lw=1.4, label='招商银行（2007-10-16买入）')
S['lump_ms'].plot(ax=ax, color=C_MS, lw=1.4, label='民生银行（2007-10-16买入）')
ax.axhline(100000, color='black', ls='--', lw=1, alpha=0.6)
ax.text(pd.Timestamp('2008-01-01'), 104000, '本金10万', fontsize=10)
ax.annotate('最惨剩3.96万\n（2008-10）', xy=(pd.Timestamp('2008-10-28'), 39550), xytext=(pd.Timestamp('2009-06-01'), 18000),
            fontsize=9, arrowprops=dict(arrowstyle='->', color=C_BANK))
ax.annotate('9.7年后才回本\n（2017-07）', xy=(pd.Timestamp('2017-07-26'), 100000), xytext=(pd.Timestamp('2018-06-01'), 50000),
            fontsize=9, arrowprops=dict(arrowstyle='->', color=C_BANK))
ax.annotate('民生：持有17年\n才勉强回本又跌回\n终值9.5万，还亏着', xy=(pd.Timestamp('2024-10-08'), 100000), xytext=(pd.Timestamp('2019-06-01'), 155000),
            fontsize=9, arrowprops=dict(arrowstyle='->', color=C_MS))
ax.set_title('10万元在历史大顶一次性买入银行股，一直不卖+分红再投，后来怎么样了', fontsize=13)
ax.set_ylabel('账户市值（元）')
ax.yaxis.set_major_formatter(plt.FuncFormatter(wan))
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/bank_lump_sum.png')
plt.close(fig)

# ---------- 图2: 归一化定投对比（2007-10起） ----------
r_icbc = dca('601398', '2007-10-08')
s_icbc = r_icbc['_series']; cf_icbc = r_icbc['_cashflows']
r_300 = dca_index(f'{DATA}/sh000300.csv', '2007-10-08')
s_300 = r_300['_series']; cf_300 = r_300['_cashflows']
inv_icbc = invested_curve(cf_icbc, s_icbc.index)
inv_300 = invested_curve(cf_300, s_300.index)
ratio_icbc = s_icbc / inv_icbc
ratio_300 = s_300 / inv_300
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
ratio_icbc.plot(ax=ax, color=C_BANK, lw=1.8,
                label=f"定投工行+跌了加倍（投入{r_icbc['total_invested']/10000:.1f}万，年化{r_icbc['xirr']}%）")
ratio_300.plot(ax=ax, color=C_300, lw=1.4,
               label=f"同规则定投沪深300（投入{r_300['total_invested']/10000:.1f}万，年化{r_300['xirr']}%）")
ax.axhline(435599/319000, color=C_DEP, ls='--', lw=1.2)
ax.text(pd.Timestamp('2008-03-01'), 435599/319000 + 0.02, '同样节奏存3%定期：每1元变成1.37元', fontsize=10, color=C_DEP)
ax.axhline(1.0, color='black', ls=':', lw=1, alpha=0.5)
ax.text(pd.Timestamp('2008-03-01'), 1.012, '回本线', fontsize=9, alpha=0.7)
ax.annotate('每1元变成2.97元', xy=(ratio_icbc.index[-1], 2.97), xytext=(pd.Timestamp('2018-01-01'), 2.75),
            fontsize=11, color=C_BANK, fontweight='bold', arrowprops=dict(arrowstyle='->', color=C_BANK))
ax.annotate('每1元变成1.41元', xy=(ratio_300.index[-1], 1.41), xytext=(pd.Timestamp('2021-06-01'), 1.9),
            fontsize=11, color=C_300, fontweight='bold', arrowprops=dict(arrowstyle='->', color=C_300))
ax.set_title('每月1000元定投、跌超10%当月加倍，从2007年10月大顶开始：每投入1元变成多少', fontsize=12.5)
ax.set_ylabel('账户市值 ÷ 累计投入（倍）')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/bank_dca_vs_300.png')
plt.close(fig)

# ---------- 图3: 四个起点XIRR对比 ----------
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
starts = ['2007-10\n（大顶）', '2015-06\n（杠杆牛顶）', '2018-01\n（蓝筹顶）', '2021-02\n（核心资产顶）']
icbc_x = [10.10, 12.38, 13.76, 19.88]
hs_x = [3.42, 3.01, 3.16, 3.59]
x = range(len(starts))
w = 0.35
b1 = ax.bar([i - w/2 for i in x], icbc_x, w, color=C_BANK, label='定投工行+跌了加倍')
b2 = ax.bar([i + w/2 for i in x], hs_x, w, color=C_300, label='同规则定投沪深300')
ax.axhline(3.0, color=C_DEP, ls='--', lw=1)
ax.text(3.35, 3.3, '3年定存≈3%', fontsize=9, color=C_DEP)
for b, v in zip(b1, icbc_x):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f'{v}%', ha='center', fontsize=10, color=C_BANK, fontweight='bold')
for b, v in zip(b2, hs_x):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f'{v}%', ha='center', fontsize=9, color=C_300)
ax.set_xticks(list(x))
ax.set_xticklabels(starts, fontsize=10)
ax.set_ylabel('年化收益率 XIRR（%）')
ax.set_title('从四个不同的"高点"开始定投工行，年化收益全部跑赢指数和定存', fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/bank_xirr_by_start.png')
plt.close(fig)

# ---------- 图4: 股价19年没涨 vs 含分红总回报 ----------
hfq = pd.read_csv(f'{DATA}/601398_hfq_sina.csv', parse_dates=['date']).set_index('date')['close']
raw = pd.read_csv(f'{DATA}/601398_raw.csv', parse_dates=['date']).set_index('date')['close']
start = pd.Timestamp('2007-11-01')
hfq_n = hfq[hfq.index >= start] / hfq[hfq.index >= start].iloc[0]
raw_n = raw[raw.index >= start] / raw[raw.index >= start].iloc[0]
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
(hfq_n * 100).plot(ax=ax, color=C_BANK, lw=1.8, label='含分红再投的总回报（100 → 231）')
(raw_n * 100).plot(ax=ax, color=C_300, lw=1.4, label='只看股价（100 → 90）')
ax.axhline(100, color='black', ls='--', lw=1, alpha=0.5)
ax.annotate('2007年大顶8.84元\n2026年7月7.97元\n股价19年还亏10%', xy=(raw_n.index[-1], 90), xytext=(pd.Timestamp('2017-01-01'), 60),
            fontsize=10, arrowprops=dict(arrowstyle='->', color=C_300))
ax.annotate('但年年分红，分红再买\n19年累计分红4.62元/股\n总回报2.31倍', xy=(hfq_n.index[-1], 231), xytext=(pd.Timestamp('2014-06-01'), 215),
            fontsize=10, color=C_BANK, arrowprops=dict(arrowstyle='->', color=C_BANK))
ax.set_title('工商银行2007年大顶至今：股价没涨，但"吃分红"赚了1.3倍', fontsize=13)
ax.set_ylabel('2007-11-01 = 100')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/bank_price_vs_total_return.png')
plt.close(fig)

# ---------- 图5: 同一策略三家银行三种命 ----------
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
banks = ['工商银行', '招商银行', '民生银行']
xirr_dca = [10.10, 11.48, 1.88]
lump = [4.42, 4.82, -0.25]
x = range(3)
w = 0.35
b1 = ax.bar([i - w/2 for i in x], xirr_dca, w, color=C_BANK, label='月定投+跌了加倍（2007-10起）年化')
b2 = ax.bar([i + w/2 for i in x], lump, w, color='#95a5a6', label='07年大顶一次性买入年化')
ax.axhline(3.0, color=C_DEP, ls='--', lw=1)
ax.text(-0.45, 3.3, '定存≈3%', fontsize=9, color=C_DEP)
for b, v in zip(b1, xirr_dca):
    ax.text(b.get_x() + b.get_width()/2, v + 0.25, f'{v}%', ha='center', fontsize=11, fontweight='bold', color=C_BANK)
for b, v in zip(b2, lump):
    ax.text(b.get_x() + b.get_width()/2, v + 0.25 if v > 0 else v - 0.9, f'{v}%', ha='center', fontsize=10, color='#555')
ax.set_xticks(list(x))
ax.set_xticklabels(banks, fontsize=12)
ax.set_ylabel('年化收益（%）')
ax.set_title('同一个策略用在三家银行：方法不保命，选谁才是命', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/bank_three_banks.png')
plt.close(fig)

# ---------- 图6: 四行组合定投（归一化） ----------
combo = S['combo']
d300 = S['dca300_10']
r_c = R['dca_combo']
inv_combo = invested_curve(combo_cf_rebuild(), combo.index) if False else None
# 组合现金流重建：四行各自dca(monthly=250)的cf合并
combo_cf = []
for code in ['601398', '601288', '601988', '601939']:
    rr = dca(code, '2010-08-02', monthly=250)
    combo_cf += rr['_cashflows']
combo_cf.sort()
inv_combo = invested_curve(combo_cf, combo.index)
r_300d = dca_index(f'{DATA}/sh000300.csv', '2010-08-02', monthly=1000)
inv_300d = invested_curve(r_300d['_cashflows'], d300.index)
ratio_combo = combo / inv_combo
ratio_300d = d300 / inv_300d
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
ratio_combo.plot(ax=ax, color=C_BANK, lw=1.8,
                 label=f"工农中建组合（投入{r_c['total_invested']/10000:.1f}万，年化{r_c['xirr']}%）")
ratio_300d.plot(ax=ax, color=C_300, lw=1.4,
                label=f"同规则定投沪深300（投入{r_300d['total_invested']/10000:.1f}万，年化{r_300d['xirr']}%）")
ax.axhline(341301/263250, color=C_DEP, ls='--', lw=1.2)
ax.text(pd.Timestamp('2011-01-01'), 341301/263250 + 0.02, '同样节奏存3%定期：每1元变成1.30元', fontsize=10, color=C_DEP)
ax.axhline(1.0, color='black', ls=':', lw=1, alpha=0.5)
ax.annotate('每1元变成3.03元', xy=(ratio_combo.index[-1], 3.03), xytext=(pd.Timestamp('2018-06-01'), 2.8),
            fontsize=11, color=C_BANK, fontweight='bold', arrowprops=dict(arrowstyle='->', color=C_BANK))
ax.annotate('每1元变成1.32元', xy=(ratio_300d.index[-1], ratio_300d.iloc[-1]), xytext=(pd.Timestamp('2020-06-01'), 2.0),
            fontsize=11, color=C_300, fontweight='bold', arrowprops=dict(arrowstyle='->', color=C_300))
ax.set_title('每月1000元等权买入工农中建、跌超10%加倍，2010年8月至今：每投入1元变成多少', fontsize=12.5)
ax.set_ylabel('账户市值 ÷ 累计投入（倍）')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/bank_combo.png')
plt.close(fig)

print('6张图已生成（图2/图6为归一化口径）')
print(f"图2: 工行 ratio={ratio_icbc.iloc[-1]:.2f} 300 ratio={ratio_300.iloc[-1]:.2f}")
print(f"图6: 组合 ratio={ratio_combo.iloc[-1]:.2f} 300 ratio={ratio_300d.iloc[-1]:.2f} 300投入={r_300d['total_invested']:.0f} 300终值={r_300d['final_value']:.0f}")
