"""验证东财接口能否获取退市股票数据"""
import akshare as ak
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# 设置代理
os.environ['HTTP_PROXY'] = 'PROXY_PLACEHOLDER'
os.environ['HTTPS_PROXY'] = 'PROXY_PLACEHOLDER'

print("=== 验证东财接口能否获取退市股票 ===")

# 已知退市股票
test_codes = ['600432', '600286', '000760', '002220']

for code in test_codes:
    print(f"\n--- {code} ---")
    # 方法1: stock_zh_a_hist (东财)
    try:
        df = ak.stock_zh_a_hist(symbol=code, period='daily', 
                                 start_date='20200101', end_date='20260701',
                                 adjust='qfq')
        if df is not None and len(df) > 0:
            print(f"  东财前复权: {len(df)}条, {df['日期'].min()} ~ {df['日期'].max()}")
            print(f"  起价: {df['收盘'].iloc[0]:.2f}, 终价: {df['收盘'].iloc[-1]:.2f}")
            # 检查是否在某个时间点终止（退市）
            last_date = pd.to_datetime(df['日期'].max())
            if last_date < pd.Timestamp('2026-07-01'):
                print(f"  *** 最后数据日期 {last_date.date()} → 可能退市 ***")
        else:
            print(f"  东财前复权: 无数据")
    except Exception as e:
        print(f"  东财前复权: {e}")

# 方法2: 获取退市股票列表
print("\n=== 获取退市股票完整列表 ===")

# 上交所退市
try:
    sh_delist = ak.stock_info_sh_delist()
    print(f"上交所退市: {len(sh_delist)}只")
    print(f"列名: {list(sh_delist.columns)}")
    if len(sh_delist) > 0:
        print(sh_delist.head(5).to_string())
except Exception as e:
    print(f"上交所退市: {e}")

# 深交所退市
for param in ['终止上市', '暂停上市', '退市']:
    try:
        sz_delist = ak.stock_info_sz_delist(symbol=param)
        print(f"\n深交所{param}: {len(sz_delist)}只")
        print(f"列名: {list(sz_delist.columns)}")
        if len(sz_delist) > 0:
            print(sz_delist.head(3).to_string())
        break
    except Exception as e:
        print(f"深交所{param}: {e}")

# 统计：如果用东财数据，能获取多少退市股的数据
print("\n=== 测试退市股数据可获取性 ===")
try:
    sh_delist = ak.stock_info_sh_delist()
    success = 0
    total = min(20, len(sh_delist))
    for _, row in sh_delist.head(total).iterrows():
        code = str(row['公司代码']).zfill(6)
        try:
            df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                     start_date='20200101', end_date='20260701',
                                     adjust='qfq')
            if df is not None and len(df) > 0:
                success += 1
                last = pd.to_datetime(df['日期'].max())
                if last < pd.Timestamp('2024-01-01'):
                    print(f"  {code} {row['公司简称']:8s}: {len(df)}条, 最后日期{last.date()} (已退市)")
        except:
            pass
    print(f"\n东财可获取退市股数据: {success}/{total}")
except Exception as e:
    print(f"退市股测试失败: {e}")
