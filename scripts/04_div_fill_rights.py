import os
# 如果设置了代理环境变量则使用，否则直连（东财源在中国电信网络下可能被屏蔽）
if 'HTTP_PROXY' not in os.environ and 'HTTPS_PROXY' not in os.environ:
    # 如需代理，取消下面两行的注释并修改地址
    # os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
    # os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'
    pass
import akshare as ak
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

print('高股息填权策略 - 回测')
print('='*60)

# TOP15高息股
div_rank = ak.stock_history_dividend()
div_rank = div_rank[~div_rank['名称'].str.contains('ST',na=False)]
div_rank = div_rank[div_rank['年均股息']>3.0]
top15 = div_rank.nlargest(15, '年均股息')

fill_rights_events = []

for i,(_, stock) in enumerate(top15.iterrows()):
    code = stock['代码']
    name = stock['名称']
    print(f'  [{i+1}/15] {code} {name}...')
    
    try:
        detail = ak.stock_history_dividend_detail(symbol=code, indicator='分红')
    except:
        continue
    
    detail = detail[detail['进度']=='实施'].copy()
    detail['除权除息日'] = pd.to_datetime(detail['除权除息日'])
    detail = detail.dropna(subset=['除权除息日'])
    detail = detail[detail['除权除息日'] >= '2020-01-01']
    
    if len(detail) == 0:
        continue
    
    symbol_full = 'sh'+code if code.startswith('6') else 'sz'+code
    try:
        price_df = ak.stock_zh_a_daily(symbol=symbol_full)
    except:
        continue
    
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date').set_index('date')
    
    for _, div in detail.iterrows():
        ex_date = div['除权除息日']
        dividend = float(div['派息']) / 10
        
        future = price_df.index[price_df.index >= ex_date]
        if len(future) == 0:
            continue
        ex_date_actual = future[0]
        idx = price_df.index.get_loc(ex_date_actual)
        
        pre_price = price_df.iloc[idx-1]['close'] if idx>0 else price_df.iloc[idx]['close']
        div_yield = dividend / pre_price
        
        for days in [5, 20, 60, 120]:
            end_idx = min(idx + days, len(price_df) - 1)
            actual_days = end_idx - idx
            if actual_days < 3:  # 至少需要3个交易日
                continue
            post_price = price_df.iloc[end_idx]['close']
            total_ret = (post_price + dividend - pre_price) / pre_price
            mkt_ret = (price_df.iloc[end_idx]['close'] / price_df.iloc[idx]['close']) - 1
            
            fill_rights_events.append({
                'code': code, 'name': name, 'ex_date': ex_date_actual,
                'div_yield': div_yield, 'days': days, 'actual_days': actual_days,
                'total_ret': total_ret, 'excess_ret': total_ret - mkt_ret,
                'filled': total_ret > 0, 'partial': actual_days < days,
            })

if len(fill_rights_events) == 0:
    print('无分红事件数据')
    exit()

events_df = pd.DataFrame(fill_rights_events)
print(f'\n总分红事件: {len(events_df)}条 (来自{events_df["code"].nunique()}只股票)')

header = f'  {"持有":>6s} | {"事件数":>5s} | {"总收益":>7s} | {"超额":>7s} | {"填权率":>6s} | {"中位超额":>7s}'
print(header)
print('-' * len(header))

for days in [5, 20, 60, 120]:
    sub = events_df[events_df['days']==days]
    if len(sub)==0: continue
    print(f'  {days:3d}天 | {len(sub):5d} | {sub["total_ret"].mean():6.1%} | {sub["excess_ret"].mean():+6.1%} | {sub["filled"].mean():5.1%} | {sub["excess_ret"].median():+6.1%}')

# 股息率分组
print('\n按股息率分组(持有60天):')
sub60 = events_df[events_df['days']==60].copy()
sub60['yield_bucket'] = pd.cut(sub60['div_yield'], bins=[0,0.03,0.05,0.08,1.0], labels=['<3%','3-5%','5-8%','>8%'])
for bucket in ['<3%','3-5%','5-8%','>8%']:
    b = sub60[sub60['yield_bucket']==bucket]
    if len(b)==0: continue
    print(f'  股息率{bucket}: {len(b):3d}次 总收益{b["total_ret"].mean():.1%} 超额{b["excess_ret"].mean():+.1%} 填权率{b["filled"].mean():.0%}')

# 年度
print('\n按年度(持有60天):')
sub60['year'] = sub60['ex_date'].dt.year
for y in sorted(sub60['year'].unique()):
    b = sub60[sub60['year']==y]
    print(f'  {y}年: {len(b):3d}次 总收益{b["total_ret"].mean():.1%} 超额{b["excess_ret"].mean():+.1%} 填权率{b["filled"].mean():.0%}')

# 按股票
print('\n按股票(持有60天):')
for code in sorted(sub60['code'].unique()):
    b = sub60[sub60['code']==code]
    name = b['name'].iloc[0]
    print(f'  {code} {name:10s}: {len(b):2d}次 总收益{b["total_ret"].mean():.1%} 超额{b["excess_ret"].mean():+.1%} 填权率{b["filled"].mean():.0%}')
