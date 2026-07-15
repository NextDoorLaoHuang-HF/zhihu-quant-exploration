"""
按正式 Fama-French 规则构建 SMB/HML 因子 — scripts/lib/fama_french.py

与旧版 02_conventional_dualma_ff.py / gen_ff3_chart.py 的区别：
- 市值：未复权收盘价 × 流通股本（不是收盘价 × 固定股本，也不是 1/股价）
- B/M：上一财年正账面净资产 / 当年6月末流通市值（不是过去12月涨跌幅）
- 账面数据按公告日期滞后（年报4月30日前披露，5月起用上年Q4）
- 2×3 分组：市值中位数分大小，B/M 30%/70% 分位数分低中高
- 组合内市值加权（不是等权）
- 每年6月末重新分组，持有到次年6月

注意：这是 A 股样本上的 FF 风格因子构造，不与 Kenneth French 官方因子直接比较。

依赖：numpy, pandas（akshare 仅在 build() 中需要）
"""
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.universe import StockUniverse

warnings.filterwarnings('ignore')

_PROXY = 'PROXY_PLACEHOLDER'
os.environ.setdefault('HTTP_PROXY', _PROXY)
os.environ.setdefault('HTTPS_PROXY', _PROXY)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')


@dataclass
class FamaFrenchBuilder:
    """
    按正式 Fama-French 规则构建 SMB/HML 因子。

    使用方式：
        universe = StockUniverse.build()
        builder = FamaFrenchBuilder.build(universe)
        factors = builder.compute_factors(2020, 2024)

    测试：
        builder = FamaFrenchBuilder.from_cache(universe, book_equity_data)
    """

    universe: "StockUniverse"
    book_equity_data: dict = field(default_factory=dict)
    use_qfq: bool = True  # 持有期收益必须用前复权价格（含分红再投资）
    announce_date_source: str = 'actual'  # 'actual'=实际公告日, 'statutory'=法定截止日近似

    @classmethod
    def from_cache(cls, universe, book_equity_data):
        """用预加载数据构建（用于测试，不触发网络请求）。"""
        # 检测数据中是否有实际公告日期
        # 如果所有 announce_date 都是 X-04-30，说明用的是法定截止日近似
        all_dates = []
        for records in book_equity_data.values():
            for r in records:
                all_dates.append(r.get('announce_date', ''))
        statutory = all(d.endswith('-04-30') for d in all_dates if d)
        source = 'statutory' if statutory and len(all_dates) > 0 else 'actual'
        return cls(universe=universe, book_equity_data=book_equity_data,
                   announce_date_source=source)

    @classmethod
    def build(cls, universe):
        """
        从 akshare 批量拉取账面权益数据构建。

        数据源：
        1. ak.stock_zcfz_em(date='YYYY1231') — 资产负债表，提取股东权益合计
        2. ak.stock_report_disclosure(period='YYYY年报') — 实际公告日期

        两个数据源按股票代码+报告期合并：
        - 股东权益合计 → total_equity
        - 实际披露日期 → announce_date（避免前视偏差）

        若 stock_report_disclosure 缺失某只股票，降级使用法定截止日（次年4月30日），
        并在 announce_date_source 字段中标注。

        缓存到 data/book_equity_all.pkl 避免重复拉取。
        """
        import akshare as ak

        cache_path = os.path.join(_PROJECT_ROOT, 'data', 'book_equity_all.pkl')
        if os.path.exists(cache_path):
            import pickle
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            # 检查缓存是否含 _meta 字段
            if isinstance(cached, dict) and '_meta' in cached:
                book_equity_data = cached['data']
                source = cached['_meta'].get('announce_date_source', 'unknown')
            else:
                # 旧格式缓存，只有 data
                book_equity_data = cached
                source = 'unknown'
            return cls(universe=universe, book_equity_data=book_equity_data,
                       announce_date_source=source)

        book_equity_data = {}
        actual_dates = {}  # code -> {report_period: announce_date}
        downgraded_count = 0

        # Step 1: 拉取实际公告日期
        print('  拉取实际公告日期 (stock_report_disclosure)...')
        for year in range(2019, 2026):
            period = f'{year}年报'
            try:
                df_disc = ak.stock_report_disclosure(market='沪深京', period=period)
                if df_disc is None or len(df_disc) == 0:
                    print(f'    {period}: 无数据')
                    continue
                # 实际披露日期
                for _, row in df_disc.iterrows():
                    code = str(row.get('股票代码', '')).zfill(6)
                    disc_date = row.get('实际披露')
                    if pd.notna(disc_date) and disc_date is not None:
                        report_period = f'{year}-12-31'
                        if code not in actual_dates:
                            actual_dates[code] = {}
                        try:
                            actual_dates[code][report_period] = str(pd.Timestamp(disc_date).date())
                        except Exception:
                            pass
                print(f'    {period}: {len(df_disc)} 只股票')
            except Exception as e:
                print(f'    {period}: 拉取失败 {e}')

        # Step 2: 拉取账面权益数据并合并公告日期
        print('  拉取账面权益数据 (stock_zcfz_em)...')
        for year in range(2019, 2026):
            date_str = f'{year}1231'
            try:
                df = ak.stock_zcfz_em(date=date_str)
                if df is None or len(df) == 0:
                    continue
                report_period = f'{year}-12-31'
                statutory_date = f'{year + 1}-04-30'

                for _, row in df.iterrows():
                    code = str(row.get('股票代码', '')).zfill(6)
                    equity = row.get('股东权益合计')

                    if equity is not None and pd.notna(equity) and equity > 0:
                        # 优先使用实际公告日，降级到法定截止日
                        actual_ad = actual_dates.get(code, {}).get(report_period)
                        if actual_ad:
                            announce_date = actual_ad
                        else:
                            announce_date = statutory_date
                            downgraded_count += 1

                        if code not in book_equity_data:
                            book_equity_data[code] = []
                        book_equity_data[code].append({
                            'report_period': report_period,
                            'announce_date': announce_date,
                            'total_equity': float(equity),
                        })
                print(f'    {date_str}: {len(df)} 只股票')
            except Exception as e:
                print(f'    {date_str}: 拉取失败 {e}')

        # 判断来源
        total_records = sum(len(v) for v in book_equity_data.values())
        all_dates_flat = []
        for records in book_equity_data.values():
            for r in records:
                all_dates_flat.append(r.get('announce_date', ''))
        statutory_count = sum(1 for d in all_dates_flat if d.endswith('-04-30'))
        source = 'actual' if statutory_count < total_records * 0.5 else 'statutory'
        if downgraded_count > 0:
            source = f'actual ({downgraded_count} stocks downgraded to statutory)'

        # 缓存（含元数据）
        import pickle
        cache_data = {
            'data': book_equity_data,
            '_meta': {
                'announce_date_source': source,
                'total_stocks': len(book_equity_data),
                'downgraded_count': downgraded_count,
            },
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f'  公告日来源: {source}')
        print(f'  降级数: {downgraded_count}')

        return cls(universe=universe, book_equity_data=book_equity_data,
                   announce_date_source=source)

    # --------------------------------------------------------
    # 核心方法
    # --------------------------------------------------------

    def get_book_equity_at(self, code: str, date) -> float | None:
        """
        获取在 date 时可用的最近年报的归属母公司股东权益。

        Kenneth French 规则：使用 t-1 财年的账面权益。
        在 A 股，年报须在次年 4 月 30 日前披露。

        筛选条件：
        - report_period 以 12-31 结尾（年报）
        - announce_date ≤ date（公告日期不晚于指定日期，避免前视偏差）
        - 取 report_period 最晚的一期
        """
        date = pd.Timestamp(date)
        records = self.book_equity_data.get(code, [])

        # 筛选年报（report_period 以 12-31 结尾）
        annual_reports = [r for r in records if r['report_period'].endswith('12-31')]

        # 筛选公告日期 ≤ date 的报告（避免使用尚未公告的数据）
        available = []
        for r in annual_reports:
            ad = r.get('announce_date', '')
            if not ad or ad in ('None', 'nan', 'NaT'):
                continue
            try:
                ad_ts = pd.Timestamp(ad)
                if ad_ts <= date:
                    available.append(r)
            except Exception:
                continue

        if not available:
            return None

        # 取 report_period 最晚的一期
        available.sort(key=lambda r: r['report_period'], reverse=True)
        equity = available[0]['total_equity']

        if equity is None or (isinstance(equity, float) and np.isnan(equity)):
            return None
        return float(equity)

    @staticmethod
    def form_portfolios(codes, market_caps, bm_ratios,
                        size_median, bm_low, bm_high):
        """
        按 2×3 规则分组。

        - Size: 市值 ≤ 中位数 = Small (S), > 中位数 = Big (B)
        - B/M: ≤ 30 分位 = Low (L), ≥ 70 分位 = High (H), 中间 = Medium (M)
        - 6 个组合: SL, SM, SH, BL, BM, BH

        返回: dict, 键为组合名，值为股票代码列表
        """
        portfolios = {'SL': [], 'SM': [], 'SH': [],
                      'BL': [], 'BM': [], 'BH': []}

        for code in codes:
            mc = market_caps.get(code)
            bm = bm_ratios.get(code)
            if mc is None or bm is None:
                continue

            if mc <= size_median:
                size = 'S'
            else:
                size = 'B'

            if bm <= bm_low:
                bm_group = 'L'
            elif bm >= bm_high:
                bm_group = 'H'
            else:
                bm_group = 'M'

            port_name = f'{size}{bm_group}'
            portfolios[port_name].append(code)

        return portfolios

    @staticmethod
    def value_weighted_return(stock_returns, weights):
        """
        市值加权收益 = Σ(wi × ri) / Σ(wi)

        stock_returns: dict, code -> 月收益
        weights: dict, code -> 市值（形成日）

        返回: float 或 NaN（当无有效收益时返回 NaN，不返回伪 0）
        """
        total_weight = sum(weights[c] for c in stock_returns if c in weights)
        if total_weight == 0:
            return float('nan')
        return sum(weights[c] * r for c, r in stock_returns.items()
                    if c in weights) / total_weight

    def _get_monthly_returns(self, code, start, end):
        """获取月度收益序列。"""
        if self.use_qfq:
            s = self.universe.total_return_series(code, start, end)
        else:
            s = self.universe._raw_close_series(code, start, end)
        if len(s) == 0:
            return pd.Series(dtype=float)
        monthly_close = s.resample('ME').last()
        return monthly_close.pct_change(fill_method=None)

    def _get_formation_date(self, year: int) -> pd.Timestamp:
        """
        返回该年6月最后一个交易日作为组合形成日。

        当6月30日为非交易日（周末）时，回退到最后一个交易日。
        例如 2024-06-30 是周日 → 返回 2024-06-28（周五）。
        """
        june_30 = pd.Timestamp(year=year, month=6, day=30)
        # 如果是交易日（周一~周五），直接用
        if june_30.weekday() < 5:
            return june_30
        # 否则回退到最近的交易日
        offset = june_30.weekday() - 4  # 回退到周五
        return june_30 - pd.Timedelta(days=offset)

    def _form_at_date(self, formation_date):
        """
        在给定日期形成组合。

        返回: (portfolios, stock_data) 或 (None, None)
        - portfolios: dict, 组合名 -> 股票代码列表
        - stock_data: dict, 股票代码 -> {market_cap, book_equity, bm_ratio}
        """
        formation_date = pd.Timestamp(formation_date)

        eligible = self.universe.eligible_at(formation_date)

        stock_data = {}
        for code in eligible:
            mc = self.universe.market_cap_at(code, formation_date)
            be = self.get_book_equity_at(code, formation_date)
            if mc is None or mc <= 0:
                continue
            if be is None or be <= 0:
                continue  # 排除负/零账面权益
            bm = be / mc
            stock_data[code] = {
                'market_cap': mc,
                'book_equity': be,
                'bm_ratio': bm,
            }

        if len(stock_data) < 6:
            return None, None

        mcs = pd.Series({c: d['market_cap'] for c, d in stock_data.items()})
        bms = pd.Series({c: d['bm_ratio'] for c, d in stock_data.items()})

        portfolios = self.form_portfolios(
            list(stock_data.keys()),
            {c: d['market_cap'] for c, d in stock_data.items()},
            {c: d['bm_ratio'] for c, d in stock_data.items()},
            mcs.median(),
            bms.quantile(0.30),
            bms.quantile(0.70),
        )
        return portfolios, stock_data

    def get_portfolio_stocks(self, formation_date) -> dict:
        """
        返回给定日期的 6 个组合的股票列表。

        用于调试和测试。
        """
        portfolios, _ = self._form_at_date(formation_date)
        if portfolios is None:
            return {}
        return portfolios

    def compute_portfolio_returns(self, start_year, end_year):
        """
        计算 6 个组合 + 全市场的月度收益。

        返回: pd.DataFrame
        列: SL, SM, SH, BL, BM, BH, MKT
        索引: 月末日期
        """
        all_rows = []

        for year in range(start_year, end_year + 1):
            formation_date = self._get_formation_date(year)
            hold_start = formation_date
            # hold_end 用日历6/30而非交易日，确保6月末价格被包含
            hold_end = pd.Timestamp(year=year + 1, month=6, day=30)

            portfolios, stock_data = self._form_at_date(formation_date)
            if portfolios is None:
                continue

            # 获取所有股票的月度收益
            monthly_returns = {}
            for code in stock_data:
                monthly_returns[code] = self._get_monthly_returns(
                    code, hold_start, hold_end
                )

            # 逐月计算组合收益
            # 使用日历6/30作为月末上界（确保6月被包含），而非交易日
            all_months = pd.date_range(
                formation_date + pd.Timedelta(days=1),
                pd.Timestamp(year=year + 1, month=6, day=30),
                freq='ME',
            )

            for month_end in all_months:
                row = {}

                # 6 个组合
                for port_name, codes in portfolios.items():
                    stock_rets = {}
                    wts = {}
                    for code in codes:
                        if (code in monthly_returns and
                                month_end in monthly_returns[code].index):
                            r = monthly_returns[code].loc[month_end]
                            if pd.notna(r):
                                stock_rets[code] = r
                                wts[code] = stock_data[code]['market_cap']
                    row[port_name] = self.value_weighted_return(
                        stock_rets, wts
                    )

                # 全市场
                all_rets = {}
                all_wts = {}
                for code, d in stock_data.items():
                    if (code in monthly_returns and
                            month_end in monthly_returns[code].index):
                        r = monthly_returns[code].loc[month_end]
                        if pd.notna(r):
                            all_rets[code] = r
                            all_wts[code] = d['market_cap']
                row['MKT'] = self.value_weighted_return(all_rets, all_wts)

                row['date'] = month_end
                all_rows.append(row)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).set_index('date')
        return df

    def compute_factors(self, start_year, end_year):
        """
        计算 MKT, SMB, HML 月度因子收益。

        返回: pd.DataFrame, 列为 MKT, SMB, HML, 索引为月末日期

        注意：这是 A 股样本上的 FF 风格因子构造，
        不与 Kenneth French 官方因子直接比较。
        """
        port_rets = self.compute_portfolio_returns(start_year, end_year)

        if port_rets.empty:
            return pd.DataFrame()

        factors = pd.DataFrame(index=port_rets.index)
        factors['MKT'] = port_rets['MKT']

        # SMB = (SL + SM + SH)/3 - (BL + BM + BH)/3
        factors['SMB'] = (
            (port_rets['SL'] + port_rets['SM'] + port_rets['SH']) / 3
            - (port_rets['BL'] + port_rets['BM'] + port_rets['BH']) / 3
        )

        # HML = (SH + BH)/2 - (SL + BL)/2
        factors['HML'] = (
            (port_rets['SH'] + port_rets['BH']) / 2
            - (port_rets['SL'] + port_rets['BL']) / 2
        )

        return factors[['MKT', 'SMB', 'HML']]

    def compute_factors_with_quality(self, start_year, end_year):
        """
        计算因子收益 + 数据质量报告。

        返回: (factors_df, quality_dict)
        - factors_df: pd.DataFrame, 列 MKT/SMB/HML
        - quality_dict: 包含 data_quality 子字典
          - monthly_stock_counts: 每月有效股票数
          - empty_portfolios: 每月空组合数
          - downgraded_stocks: qfq 拉取失败降级数
          - n_months: 月数
        """
        port_rets = self.compute_portfolio_returns(start_year, end_year)

        if port_rets.empty:
            return pd.DataFrame(), {'data_quality': {
                'monthly_stock_counts': {},
                'empty_portfolios': {},
                'downgraded_stocks': 0,
                'n_months': 0,
            }}

        factors = pd.DataFrame(index=port_rets.index)
        factors['MKT'] = port_rets['MKT']

        factors['SMB'] = (
            (port_rets['SL'] + port_rets['SM'] + port_rets['SH']) / 3
            - (port_rets['BL'] + port_rets['BM'] + port_rets['BH']) / 3
        )

        factors['HML'] = (
            (port_rets['SH'] + port_rets['BH']) / 2
            - (port_rets['SL'] + port_rets['BL']) / 2
        )

        # 数据质量统计
        monthly_counts = {}
        empty_ports = {}
        port_cols = [c for c in ['SL', 'SM', 'SH', 'BL', 'BM', 'BH']
                    if c in port_rets.columns]
        for date in port_rets.index:
            row = port_rets.loc[date]
            # 处理可能的重复索引（返回DataFrame的情况）
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            valid_count = int(row[port_cols].notna().sum())
            monthly_counts[str(date.date())] = valid_count
            empty_ports[str(date.date())] = 6 - valid_count

        # 降级股票数
        downgraded = len(getattr(self.universe, '_qfq_failed', set()))

        quality = {
            'data_quality': {
                'monthly_stock_counts': monthly_counts,
                'empty_portfolios': empty_ports,
                'downgraded_stocks': downgraded,
                'n_months': len(factors),
            }
        }

        return factors[['MKT', 'SMB', 'HML']], quality
