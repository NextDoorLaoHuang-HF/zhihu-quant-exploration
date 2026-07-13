"""
检查含退市回测是否存在逻辑错误
重点：
1. 退市股占候选池比例是否合理
2. 退市后价格处理是否正确
3. 月度收益计算是否有bug（如NaN被跳过）
4. 前复权价格对退市股的影响
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'PROJECT_ROOT/data'

# 加载数据
delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)

# ============================================================
# 检查1: 退市股的价格分布 vs 存活股的价格分布
# 如果退市股价格显著低于存活股，那T5几乎必然选退市股
# ============================================================
print("=" * 70)
print("检查1: 价格分布对比")
print("=" * 70)

# 获取2020年1月的价格
for code, s in delist_prices.items():
    s_first = s[s.index >= '2020-01-01']
    if len(s_first) > 0:
        info = delist_info.get(code, {})
        name = info.get('name', '')
        first_price = s_first.iloc[0]
        last_price = info.get('last_price', 0)
        delist_date = info.get('delist_date', '')
        print(f"  {code} {name:8s} 首价:{first_price:7.2f} 末价:{last_price:6.2f} 退市:{delist_date}")

# ============================================================
# 检查2: 退市后月度收益计算 — NaN是否被skip_na跳过
# ============================================================
print("\n" + "=" * 70)
print("检查2: 月度收益计算中NaN的行为")
print("=" * 70)

# 构造一个例子：退市股在某月退市，当月收益是-50%
# 模拟价格面板
dates = pd.date_range('2020-01-01', '2020-06-30', freq='D')
test_prices = pd.DataFrame({
    'healthy1': [10 + np.sin(i/10) for i in range(len(dates))],
    'healthy2': [8 + np.cos(i/5) for i in range(len(dates))],
    'healthy3': [12 + np.sin(i/7) for i in range(len(dates))],
    'healthy4': [6 + np.cos(i/6) for i in range(len(dates))],
    'delisted': np.linspace(2.0, 0.3, min(120, len(dates))).tolist() + [np.nan] * max(0, len(dates)-120),
}, index=dates)

# 月频
mp = test_prices.resample('M').last()
mr = mp.pct_change()
print(f"月度价格:\n{mp}")
print(f"\n月度收益:\n{mr}")

# 假设在某月选中了delisted + 4个healthy
for i in range(1, len(mp)):
    row = mp.iloc[i-1].dropna()
    if len(row) < 5: continue
    cheapest = row.nsmallest(5).index
    next_ret = mr.iloc[i]
    ret = next_ret.reindex(cheapest).mean()
    print(f"\n  月份{i}: 选中={list(cheapest)}, 下月收益中delisted={next_ret['delisted']:.2%}" if not pd.isna(next_ret['delisted']) else f"\n  月份{i}: 选中={list(cheapest)}, 下月收益中delisted=NaN")
    print(f"    mean()结果={ret:.4%}")

# ============================================================
# 检查3: 真实退市股的月度收益表现
# ============================================================
print("\n" + "=" * 70)
print("检查3: 真实退市股 — 退市前几个月的月度收益")
print("=" * 70)

# 挑几个被选中最多的退市股，看退市前6个月的月度收益
import akshare as ak
all_dates = pd.date_range('2020-01-01', '2026-07-13', freq='D')

for code in ['900951', '600086', '600074', '600175']:
    if code not in delist_prices: continue
    s = delist_prices[code]
    if len(s) < 10: continue
    
    s_aligned = s.reindex(all_dates)
    info = delist_info.get(code, {})
    delist_date = pd.to_datetime(info.get('delist_date'))
    
    # 退市前价格设为NaN
    s_aligned[s_aligned.index > delist_date] = np.nan
    
    # 月频
    mp_s = s_aligned.resample('M').last()
    mr_s = mp_s.pct_change()
    
    # 找最后6个有数据的月份
    valid = mr_s.dropna()
    last_6 = valid.tail(6)
    
    if len(last_6) > 0:
        name = info.get('name', code)
        print(f"\n  {code} {name} (退市日:{delist_date.date()})")
        for dt, r in last_6.items():
            price = mp_s[dt]
            print(f"    {dt.date()}: 价格{price:.2f} 月收益{r:+.2%}")
        if len(last_6) > 1:
            total = (1 + last_6).prod() - 1
            print(f"    最后{len(last_6)}月累计: {total:+.2%}")

# ============================================================
# 检查4: 候选池中退市股比例是否合理
# ============================================================
print("\n" + "=" * 70)
print("检查4: 候选池比例分析")
print("=" * 70)

n_healthy = 136
n_delist = 78
n_total = n_healthy + n_delist
print(f"当前候选池: {n_total}只 (存活{n_healthy} + 退市{n_delist}), 退市占比{n_delist/n_total:.1%}")

# 真实市场: ~5500只A股, 2020-2026退市约200只
# 如果从全市场随机抽150只，预期退市股 = 200/5500 * 150 ≈ 5.5只
expected_delist = 200 / 5500 * 150
print(f"全市场: ~5500只, 退市~200只, 占比{200/5500:.1%}")
print(f"150只随机抽样中预期退市股: {expected_delist:.1f}只")
print(f"当前退市股: {n_delist}只, 是预期的{n_delist/expected_delist:.0f}倍")

# 但是！退市股集中在低价股中
# 我们关心的是在"价格最低5只"这个子集中，退市股的比例
# 如果全市场按价格排，最低的1%（5500*1%=55只）中退市股占比可能很高
print(f"\n关键问题：T5选的是价格最低5只")
print(f"在5500只全市场中，最低5只的价格区间大约是多少？")
print(f"退市股价格(0.1-1.5元)基本都在这个区间")
print(f"所以T5几乎必然会选中退市股，无论候选池多大")

# ============================================================
# 检查5: 如果不把退市股看作"可选的"，而是单独统计它们被选中的概率
# ============================================================
print("\n" + "=" * 70)
print("检查5: 退市股对T5组合收益的边际贡献")
print("=" * 70)

# 策略核心问题：T5用"价格"作为"市值"的代理
# 对正常股票：低价 ≈ 小市值 ✓
# 对退市股：低价 ≈ 濒临退市 ✗（不是小市值，是价值毁灭）
# 
# 修复思路：
# 1. 排除ST/*ST股票（有交易限制，公募也不能买）
# 2. 排除股价<1元的股票（面值退市风险）
# 3. 用真正的市值（而不是价格）来排序
print()
print("策略设计缺陷:")
print("1. 用'价格最低'代理'市值最小'——对正常股票有效，对退市股无效")
print("2. 退市股价格极低(0.1-1元)但这不是'小市值'，是'价值归零的过程'")
print("3. 没有排除ST/*ST/面值退市风险的股票")
print("4. 退市股在候选池中占比36%，远高于真实市场的3.6%")
print()
print("合理的修正:")
print("A. 排除ST/*ST和股价<1元的股票（符合实际投资约束）")
print("B. 只按真实比例加入退市股（150只中加5-6只，而不是78只）")
print("C. 用市值排序代替价格排序（但akshare free数据可能没有流通市值）")
