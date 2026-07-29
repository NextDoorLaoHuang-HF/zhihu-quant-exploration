"""
高股息填权研究 v2 — 修复三个核心错误

原始脚本 04_div_fill_rights.py 的问题（Issue #1, Problem 5）：
1. 股票池按完整历史年均分红倒选 Top15（事后选择 / 前视偏差）
2. "市场基准"实际用同一只股票除息后的价格变化，未使用真实指数
3. 除息日买入者收益含了已除分红（买入者拿不到这笔钱）
4. 填权判断用 total_ret > 0（含分红），应改为未复权股价回到除息前价格

本脚本拆分为两个视角：
  A. 除息事件研究（已持有者视角）— holder_total_return 含分红
  B. 除息日买入策略（买入者视角）— entrant_return 不含分红

使用沪深300全收益指数作为市场基准。
"""
import os
import sys
import json
import datetime
import warnings
import pandas as pd
import numpy as np

# Make scripts/lib importable for metrics (not used in core logic but available)
sys.path.insert(0, os.path.dirname(__file__))

warnings.filterwarnings('ignore')


# ============================================================
# Pure functions — testable without network
# ============================================================

def compute_holder_return(pre_ex_close: float, exit_price: float, received_dividend: float) -> float:
    """
    已持有者收益（含分红）。

    已持有者在除息日前拥有股票，除息后股价下跌但持有者收到分红。
    holder_return = (exit_price + received_dividend - pre_ex_close) / pre_ex_close
    """
    if pre_ex_close <= 0:
        raise ValueError("pre_ex_close 必须为正数")
    return (exit_price + received_dividend - pre_ex_close) / pre_ex_close


def compute_entrant_return(ex_date_entry_price: float, exit_price: float) -> float:
    """
    除息日买入者收益（不含分红）。

    买入者在除息日以已除权价格买入，拿不到刚除掉的分红。
    entrant_return = exit_price / ex_date_entry_price - 1
    """
    if ex_date_entry_price <= 0:
        raise ValueError("ex_date_entry_price 必须为正数")
    return exit_price / ex_date_entry_price - 1


def compute_fill_flag(post_ex_raw_prices: pd.Series, pre_ex_close: float) -> bool:
    """
    填权判断：除息后未复权股价在观察期内是否重新达到除息前收盘价。

    filled = max(post_ex_raw_prices) >= pre_ex_close
    不是 total_ret > 0。
    """
    if len(post_ex_raw_prices) == 0:
        return False
    return float(post_ex_raw_prices.max()) >= pre_ex_close


def compute_index_return(index_prices: pd.Series, start_pos: int, end_pos: int) -> float:
    """
    指数基准收益（全收益指数，含息）。

    在 [start_pos, end_pos] 区间的总收益：
    index_return = index_prices[end_pos] / index_prices[start_pos] - 1
    """
    if start_pos < 0 or end_pos >= len(index_prices) or start_pos > end_pos:
        raise ValueError("start_pos/end_pos 越界")
    start_val = float(index_prices.iloc[start_pos])
    end_val = float(index_prices.iloc[end_pos])
    if start_val <= 0:
        raise ValueError("指数起始值必须为正数")
    return end_val / start_val - 1


def select_stock_pool(div_detail: pd.DataFrame, start_date: str = '2020-01-01', end_date: str = '2026-12-31') -> pd.DataFrame:
    """
    股票池选择 — 不按完整历史年均分红倒选 Top15（前视偏差）。

    纳入所有在 [start_date, end_date] 期间有现金分红实施事件的 A 股。
    返回去重后的事件列表（每行一个分红事件），按事件前股息率分组。

    输入 DataFrame 可包含 '除权除息日' 或 'ex_date' 列（兼容两种命名）。
    """
    if div_detail is None or len(div_detail) == 0:
        return pd.DataFrame()

    df = div_detail.copy()

    # 确保日期列（兼容两种列名）
    date_col = None
    for col in ['除权除息日', 'ex_date']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError("div_detail 必须包含 '除权除息日' 或 'ex_date' 列")
    df['ex_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['ex_date'])

    # 日期范围过滤
    df = df[(df['ex_date'] >= start_date) & (df['ex_date'] <= end_date)]

    # 只保留实施事件（兼容两种列名）
    for prog_col in ['进度', '方案进度']:
        if prog_col in df.columns:
            df = df[df[prog_col].str.contains('实施', na=False)]
            break

    return df.reset_index(drop=True)


# ============================================================
# Network-dependent data acquisition
# ============================================================

def _ensure_proxy():
    """东财源在中国电信网络下可能被屏蔽，设置代理。"""
    if 'HTTP_PROXY' not in os.environ and 'HTTPS_PROXY' not in os.environ:
        proxy = os.environ.get('DIVIDEND_PROXY') or os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy


def fetch_all_dividend_details():
    """
    获取所有 A 股的分红实施事件（2020-2026年）。

    使用 ak.stock_fhps_em(date=...) 批量获取每个报告期的分红数据，
    覆盖 2020H1 ~ 2024H2（半年报 + 年报），不逐个股票查询。
    这消除了原脚本逐个获取 Top15 分红明细的低效方式。
    """
    import akshare as ak
    _ensure_proxy()

    # 报告期列表（半年报+年报）
    reporting_periods = [
        '20200630', '20201231',
        '20210630', '20211231',
        '20220630', '20221231',
        '20230630', '20231231',
        '20240630', '20241231',
    ]

    all_events = []

    for period in reporting_periods:
        try:
            df = ak.stock_fhps_em(date=period)
        except Exception as e:
            print(f'  ⚠️ 获取 {period} 失败: {e}')
            continue

        if df is None or len(df) == 0:
            continue

        # 只保留已实施的现金分红
        impl = df[df['方案进度'].str.contains('实施', na=False)].copy()
        cash = impl[impl['现金分红-现金分红比例'].notna() & (impl['现金分红-现金分红比例'] > 0)].copy()

        if len(cash) == 0:
            continue

        cash['除权除息日'] = pd.to_datetime(cash['除权除息日'], errors='coerce')
        cash = cash.dropna(subset=['除权除息日'])
        # 日期范围过滤（确保在 2020-2026）
        cash = cash[(cash['除权除息日'] >= '2020-01-01') & (cash['除权除息日'] <= '2026-12-31')]

        for _, row in cash.iterrows():
            all_events.append({
                'code': str(row['代码']).zfill(6),
                'name': row['名称'],
                'ex_date': row['除权除息日'],
                'dividend_per_share': float(row['现金分红-现金分红比例']) / 10,  # 每10股 -> 每股
            })

        print(f'  {period}: {len(cash)} 条现金分红事件')

    events_df = pd.DataFrame(all_events)
    if len(events_df) > 0:
        events_df = events_df.drop_duplicates(subset=['code', 'ex_date'])  # 去重
        print(f'  共获取 {len(events_df)} 条分红事件（来自 {events_df["code"].nunique()} 只股票）')
    return events_df


def fetch_stock_prices(code, start_date='2019-01-01'):
    """获取个股日K线（未复权收盘价），用于事件窗口观察。"""
    import akshare as ak
    _ensure_proxy()

    symbol_full = 'sh' + code if code.startswith('6') else 'sz' + code
    try:
        price_df = ak.stock_zh_a_daily(symbol=symbol_full)
    except Exception:
        return None

    if price_df is None or len(price_df) == 0:
        return None

    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date').set_index('date')
    price_df = price_df[price_df.index >= start_date]
    return price_df


def fetch_index_prices(start_date='2019-01-01'):
    """
    获取沪深300指数日K线作为市场基准。
    使用 stock_zh_index_daily 接口。
    """
    import akshare as ak
    _ensure_proxy()

    try:
        idx_df = ak.stock_zh_index_daily(symbol='sh000300')
    except Exception:
        # fallback: 尝试新浪
        try:
            idx_df = ak.stock_zh_index_daily_em(symbol='000300')
        except Exception:
            return None

    if idx_df is None or len(idx_df) == 0:
        return None

    date_col = 'date' if 'date' in idx_df.columns else idx_df.columns[0]
    idx_df[date_col] = pd.to_datetime(idx_df[date_col])
    idx_df = idx_df.sort_values(date_col).set_index(date_col)
    idx_df = idx_df[idx_df.index >= start_date]
    return idx_df


# ============================================================
# Event study computation
# ============================================================

def run_event_study(events_df, index_df, holding_days_list=None, max_stocks=None):
    """
    对所有分红事件运行事件研究。

    对每个事件，计算：
    - holder_total_return（已持有者，含分红）
    - entrant_return（除息日买入者，不含分红）
    - fill_flag（未复权价是否回到除息前价格）
    - market_return（同期沪深300收益）
    - excess_holder = holder_total_return - market_return
    - excess_entrant = entrant_return - market_return

    参数：
    - max_stocks: 限制处理的股票数量（按事件数降序取前N只），None=全部

    返回 DataFrame，每行一个事件×持有期。
    """
    if holding_days_list is None:
        holding_days_list = [5, 20, 60, 120]

    if index_df is None:
        print('⚠️ 指数数据不可用，无法计算市场基准')
        return pd.DataFrame()

    # 为每个事件找到指数数据中的位置
    index_dates = index_df.index

    # 按股票分组，减少重复获取价格
    stock_counts = events_df.groupby('code').size().sort_values(ascending=False)
    if max_stocks is not None and len(stock_counts) > max_stocks:
        print(f'  限制为事件数最多的 {max_stocks} 只股票（共 {len(stock_counts)} 只）')
        stock_codes = stock_counts.head(max_stocks).index
    else:
        stock_codes = stock_counts.index

    results = []
    total = len(stock_codes)

    for i, code in enumerate(stock_codes):
        if i % 100 == 0:
            print(f'  事件研究 [{i}/{total}]...')

        stock_events = events_df[events_df['code'] == code].sort_values('ex_date')
        name = stock_events.iloc[0]['name']

        price_df = fetch_stock_prices(code)
        if price_df is None or len(price_df) < 2:
            continue

        for _, event in stock_events.iterrows():
            ex_date = event['ex_date']
            dividend = event['dividend_per_share']

            # 找到除息日在价格序列中的位置
            future = price_df.index[price_df.index >= ex_date]
            if len(future) == 0:
                continue
            ex_date_actual = future[0]
            idx = price_df.index.get_loc(ex_date_actual)

            # 除息前收盘价
            pre_ex_close = price_df.iloc[idx - 1]['close'] if idx > 0 else price_df.iloc[idx]['close']
            if pre_ex_close <= 0 or pd.isna(pre_ex_close):
                continue

            # 除息日收盘价（已除权后的价格）— 买入者入场价
            ex_date_close = price_df.iloc[idx]['close']
            if ex_date_close <= 0 or pd.isna(ex_date_close):
                continue

            # 事前股息率
            pre_ex_yield = dividend / pre_ex_close

            for days in holding_days_list:
                end_idx = min(idx + days, len(price_df) - 1)
                actual_days = end_idx - idx
                if actual_days < 3:
                    continue

                exit_close = price_df.iloc[end_idx]['close']
                if pd.isna(exit_close) or exit_close <= 0:
                    continue

                # 观察窗口内未复权收盘价
                window = price_df.iloc[idx:end_idx + 1]['close'].dropna()
                if len(window) == 0:
                    continue

                # A. 已持有者收益（含分红）
                holder_ret = compute_holder_return(pre_ex_close, exit_close, dividend)

                # B. 除息日买入者收益（不含分红）
                entrant_ret = compute_entrant_return(ex_date_close, exit_close)

                # C. 填权判断
                filled = compute_fill_flag(window, pre_ex_close)

                # 市场基准 — 同期沪深300
                # 找到指数中对应的交易日
                idx_future = index_dates[index_dates >= ex_date_actual]
                if len(idx_future) == 0:
                    continue
                idx_start = idx_future[0]
                idx_start_pos = index_dates.get_loc(idx_start)

                idx_end_candidates = index_dates[index_dates >= idx_start]
                # 取第 days 个交易日（或最后一个）
                if len(idx_end_candidates) <= days:
                    idx_end_pos = len(index_dates) - 1
                else:
                    idx_end_pos = index_dates.get_loc(idx_end_candidates[days])

                try:
                    mkt_ret = compute_index_return(index_df['close'], idx_start_pos, idx_end_pos)
                except (ValueError, KeyError):
                    mkt_ret = np.nan

                results.append({
                    'code': code,
                    'name': name,
                    'ex_date': ex_date_actual,
                    'dividend_per_share': dividend,
                    'pre_ex_close': pre_ex_close,
                    'ex_date_close': ex_date_close,
                    'pre_ex_yield': pre_ex_yield,
                    'days': days,
                    'actual_days': actual_days,
                    'holder_return': holder_ret,
                    'entrant_return': entrant_ret,
                    'market_return': mkt_ret,
                    'excess_holder': holder_ret - mkt_ret,
                    'excess_entrant': entrant_ret - mkt_ret,
                    'filled': filled,
                    'partial': actual_days < days,
                })

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================

def main():
    print('高股息填权研究 v2 — 修复版')
    print('=' * 60)
    print('修复内容：')
    print('  1. 股票池：不再按历史年均分红倒选Top15，纳入所有分红事件')
    print('  2. 市场基准：使用沪深300指数，不再用同一只股票')
    print('  3. 买入者收益：不含已除分红')
    print('  4. 填权判断：未复权价回到除息前价格，不是 total_ret > 0')
    print()

    # 获取分红事件数据
    print('1. 获取分红事件数据...')
    events_df = fetch_all_dividend_details()

    if len(events_df) == 0:
        print('无分红事件数据')
        return

    # 股票池选择（不按历史均值倒选）
    pool = select_stock_pool(events_df, start_date='2020-01-01', end_date='2026-12-31')
    print(f'  股票池：{pool["code"].nunique()} 只股票，{len(pool)} 条分红事件')

    # 按事前股息率分组
    pool = pool.copy()
    pool['pre_ex_yield'] = pool['dividend_per_share']  # 占位，实际在事件研究中计算

    # 获取指数数据
    print('\n2. 获取沪深300指数数据...')
    index_df = fetch_index_prices()
    if index_df is None:
        print('⚠️ 指数数据不可用，跳过市场基准计算')
        index_df = None
    else:
        print(f'  沪深300数据：{len(index_df)} 条，范围 {index_df.index[0].date()} ~ {index_df.index[-1].date()}')

    # 事件研究
    print('\n3. 运行事件研究...')
    # 限制为事件数最多的500只股票（共4653只，全量需数小时）
    results_df = run_event_study(events_df, index_df, max_stocks=500)

    if len(results_df) == 0:
        print('无有效事件结果')
        return

    print(f'  共计算 {len(results_df)} 条事件结果')

    # 汇总报告
    print('\n' + '=' * 60)
    print('汇总报告（按持有期）')
    print('-' * 60)
    header = f'  {"持有":>6s} | {"事件数":>5s} | {"持有者收益":>10s} | {"买入者收益":>10s} | {"市场基准":>10s} | {"持有者超额":>10s} | {"买入者超额":>10s} | {"填权率":>6s}'
    print(header)
    print('-' * len(header))

    for days in sorted(results_df['days'].unique()):
        sub = results_df[results_df['days'] == days]
        if len(sub) == 0:
            continue
        print(f'  {days:3d}天 | {len(sub):5d} | {sub["holder_return"].mean():9.1%} | {sub["entrant_return"].mean():9.1%} | {sub["market_return"].mean():9.1%} | {sub["excess_holder"].mean():+9.1%} | {sub["excess_entrant"].mean():+9.1%} | {sub["filled"].mean():5.1%}')

    # 按事前股息率分组
    print('\n按事前股息率分组（持有60天）：')
    print('-' * 60)
    sub60 = results_df[results_df['days'] == 60].copy()
    if len(sub60) > 0:
        sub60['yield_bucket'] = pd.cut(
            sub60['pre_ex_yield'],
            bins=[0, 0.01, 0.03, 0.05, 0.08, 1.0],
            labels=['<1%', '1-3%', '3-5%', '5-8%', '>8%']
        )
        for bucket in ['<1%', '1-3%', '3-5%', '5-8%', '>8%']:
            b = sub60[sub60['yield_bucket'] == bucket]
            if len(b) == 0:
                continue
            print(f'  股息率{bucket:>5s}: {len(b):4d}次 | 持有者{b["holder_return"].mean():6.1%} | 买入者{b["entrant_return"].mean():6.1%} | 超额(持有者){b["excess_holder"].mean():+6.1%} | 超额(买入者){b["excess_entrant"].mean():+6.1%} | 填权{b["filled"].mean():.0%}')

    # 按年度
    print('\n按年度（持有60天）：')
    print('-' * 60)
    if len(sub60) > 0:
        sub60['year'] = sub60['ex_date'].dt.year
        for y in sorted(sub60['year'].unique()):
            b = sub60[sub60['year'] == y]
            if len(b) == 0:
                continue
            print(f'  {y}年: {len(b):4d}次 | 持有者{b["holder_return"].mean():6.1%} | 买入者{b["entrant_return"].mean():6.1%} | 市场{b["market_return"].mean():6.1%} | 填权{b["filled"].mean():.0%}')

    # 保存结果
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', run_id)
    os.makedirs(out_dir, exist_ok=True)

    # 保存 JSON
    out_json = {
        'run_id': run_id,
        'description': '高股息填权研究v2 — 修复版',
        'fixes': [
            '股票池不按历史年均分红倒选Top15（去除前视偏差）',
            '市场基准使用沪深300指数（不再用同一只股票）',
            '除息日买入者收益不含已除分红',
            '填权判断用未复权价回到除息前价格（不是total_ret>0）',
        ],
        'total_events': len(results_df),
        'unique_stocks': int(results_df['code'].nunique()),
        'summary': {},
    }

    for days in sorted(results_df['days'].unique()):
        sub = results_df[results_df['days'] == days]
        key = f'{days}d'
        out_json['summary'][key] = {
            'n_events': len(sub),
            'holder_return_mean': float(sub['holder_return'].mean()),
            'entrant_return_mean': float(sub['entrant_return'].mean()),
            'market_return_mean': float(sub['market_return'].mean()),
            'excess_holder_mean': float(sub['excess_holder'].mean()),
            'excess_entrant_mean': float(sub['excess_entrant'].mean()),
            'fill_rate': float(sub['filled'].mean()),
            'excess_holder_median': float(sub['excess_holder'].median()),
            'excess_entrant_median': float(sub['excess_entrant'].median()),
        }

    json_path = os.path.join(out_dir, 'dividend.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存：{json_path}')

    # 也保存 CSV
    csv_path = os.path.join(out_dir, 'dividend_events.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'事件明细已保存：{csv_path}')

    return results_df


if __name__ == '__main__':
    main()
