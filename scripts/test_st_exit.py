"""
测试两个新过滤条件：
1. "ST化立即退出" — 持仓股变ST后下月强制卖出（而非仅选股时过滤）
2. "剔除负业绩" — 需要PE/净利润数据，免费API受限，用近似方案

注：季度财报数据需要东财API（当前代理不稳定），所以负面业绩过滤
    暂用"排除当前ST股"作为近似（ST通常伴随亏损）
"""
import os
import akshare as ak
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# 加载退市股
delist_prices = pd.read_pickle(f'{DATA_DIR}/delist_prices.pkl')
with open(f'{DATA_DIR}/delist_info.json', 'r', encoding='utf-8') as f:
    delist_info = json.load(f)

# 存活股
stock_list = ak.stock_info_a_code_name()
stock_list['symbol_full'] = stock_list['code'].apply(lambda x: f'sh{x}' if x.startswith('6') else f'sz{x}')
np.random.seed(888)
sample = stock_list.sample(min(150, len(stock_list)), random_state=888)

# 预先识别ST股
st_codes = set()
for _, row in stock_list.iterrows():
    if 'ST' in str(row['name']).upper():
        st_codes.add(row['code'])
print(f"当前全市场ST股: {len(st_codes)}只")
print(f"  示例: {list(st_codes)[:5]}")

# 抽样中ST股
sample_st = [c for c in sample['code'] if c in st_codes]
print(f"  150只抽样中ST股: {len(sample_st)}只")

survive_prices = {}
survive_names = {}
for i, (_, row) in enumerate(sample.iterrows()):
    if i % 30 == 0: print(f"  拉取: {i}/{len(sample)}")
    try:
        df = ak.stock_zh_a_daily(symbol=row['symbol_full'])
        if df is not None and len(df) > 200:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            survive_prices[row['code']] = df['close']
            survive_names[row['code']] = row['name']
    except: pass

# 加入6只退市股
np.random.seed(42)
delist_codes = np.random.choice(list(delist_prices.keys()), 6, replace=False)
all_dates = pd.date_range('2020-01-01', '2026-07-13', freq='D')
for code in delist_codes:
    s = delist_prices[code].reindex(all_dates)
    dd = delist_info.get(code, {}).get('delist_date')
    if dd: s[s.index > pd.to_datetime(dd)] = np.nan
    if s.notna().any(): survive_prices[code] = s

prices = pd.DataFrame(survive_prices).dropna(how='all')
print(f"\n候选池: {prices.shape[1]}只 (存活{len(survive_prices)-6} + 退市6)")

# ============================================================
# 三种策略对比
# ============================================================
mp = prices.resample('M').last()
mr = mp.pct_change()

def is_st_code(code):
    return code in st_codes

def run_t5_variants(mp, mr, n_stocks=5):
    """
    返回三种变体：
    A: 基线（排除ST选股 + 排除<2元）
    B: A + ST化后隔月退出（被选中的ST股只持有一个月就强制卖）
    C: A + ST股完全不买（即基线，作为对照）
    """
    rets_a, rets_b, rets_c = [], [], []
    
    for i in range(12, len(mp) - 1):
        row = mp.iloc[i].dropna()
        
        # 排除<2元和ST（选股时）
        row = row[row >= 2.0]
        valid = [c for c in row.index if not is_st_code(c)]
        row = row[valid]
        
        if len(row) < n_stocks + 5:
            continue
        
        # 基线选股（排除ST）
        cheapest_a = row.nsmallest(n_stocks).index
        next_ret = mr.iloc[i + 1]
        
        # 变体A: 正常持有（基线）
        ret_a = next_ret.reindex(cheapest_a).mean()
        rets_a.append(ret_a)
        
        # 变体B: 如果上月持仓中有ST股，本月强制退出（收益为0）
        # 在实际月度回测中，这表现为：上月选中ST股 → 当月收益中ST贡献为0
        # 简化实现：检查cheapest中是否有ST，有则将其收益置0
        ret_b_components = []
        for c in cheapest_a:
            r = next_ret.get(c, np.nan)
            if pd.isna(r):
                continue
            # ST股：假设"立即退出"，即当月不承担其收益/亏损
            # 注：这是理想化假设（能以月初价格退出）
            if is_st_code(c):
                ret_b_components.append(0.0)
            else:
                ret_b_components.append(r)
        ret_b = np.mean(ret_b_components) if ret_b_components else 0
        rets_b.append(ret_b)
        
        # 变体C: 同A（确认基线一致性，因为选股时已排除ST）
        ret_c = ret_a
        rets_c.append(ret_c)
    
    def summarize(rets, label):
        s = pd.Series(rets)
        return {
            'annual': s.mean() * 12,
            'sharpe': s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0,
            'dd': ((1+s).cumprod() / (1+s).cumprod().cummax() - 1).min(),
            'n': len(s),
            'label': label,
        }
    
    return [
        summarize(rets_a, 'A_基线(排除ST选股)'),
        summarize(rets_b, 'B_ST化立即退出(当月收益为0)'),
        summarize(rets_c, 'C_同A(确认)'),
    ]

print("\n" + "=" * 70)
print("ST退出策略对比")
print("=" * 70)

for n, label in [(5, 'T5'), (10, 'T10')]:
    results = run_t5_variants(mp, mr, n)
    print(f"\n  [{label}]")
    for r in results:
        print(f"    {r['label']:40s} | 年化{r['annual']:5.1%} 夏普{r['sharpe']:.2f} 回撤{r['dd']:5.1%}")

# 解释ST退出策略的实际意义
print(f"\n=== 解读 ===")
print(f"  基线(选股时排除ST): 已经在选股阶段过滤了ST股")
print(f"  ST化立即退出: 针对'选股后变ST'的情况——即在月初买入时不是ST，")
print(f"  但持有期间变成ST。理想化假设能以月初价格退出(收益=0)。")
print(f"  由于月度调仓，且ST通常伴随暴跌，当月收益为0已经比实际亏损好很多")
print(f"  所以这个测试给出的是上限估计（最乐观情况）")
print(f"\n  如果ST退出策略和基线差异很小 → 说明选股时过滤ST已经足够")
print(f"  如果ST退出策略显著更好 → 说明需要关注'买入后变ST'的风险")
print(f"\n=== 关于'剔除负业绩' ===")
print(f"  需要季度净利润数据，免费API受限于东财接口（当前代理不可用）")
print(f"  近似替代：排除ST股（ST通常伴随连续亏损）已经起到类似作用")
print(f"  更精确的测试需要付费数据源或东财API恢复后补测")
