"""
点时（point-in-time）全市场A股股票池构建器 — scripts/lib/universe.py

解决三个偏差：
1. 幸存者偏差 — 只用当前存活股列表会漏掉已退市的股票
2. ST 前视偏差 — 用当前名称做历史ST过滤等于用了未来信息
3. 退市股数据缺失 — akshare 新浪接口对退市股返回 JSONDecodeError

数据源策略：
- 存活股：ak.stock_info_a_code_name() + ak.stock_zh_a_daily() (含 outstanding_share)
- 退市股：data/delist_prices.pkl (腾讯qfq接口已预取) + data/delist_info.json
- 上市/退市日期：ak.stock_info_sh_delist() 上交所退市列表含上市日期+暂停上市日期
- ST状态：ak.stock_info_change_name() 曾用名列表（无日期，标记为不精确）

qfq 缓存策略：
- 持久化 parquet 文件存于 data/qfq_cache/，每只股票一个文件
- 失败记录存于 data/qfq_cache/_meta.json
- 不使用 pickle（避免 pandas 版本升级导致反序列化失败）
- 网络失败时标记为 degraded，不静默降级

依赖：akshare, pandas, numpy, pyarrow（已在 requirements.txt 中）
"""
from __future__ import annotations

import json
import os
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 代理设置（akshare 需要访问新浪/东财）
_PROXY = 'PROXY_PLACEHOLDER'
os.environ.setdefault('HTTP_PROXY', _PROXY)
os.environ.setdefault('HTTPS_PROXY', _PROXY)

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_QFQ_CACHE_DIR = os.path.join(_DATA_DIR, 'qfq_cache')


def _is_b_share(code: str) -> bool:
    """判断是否为 B 股：沪市 900xxx，深市 200xxx。"""
    return code.startswith('900') or code.startswith('200')


def _is_a_share(code: str) -> bool:
    """判断是否为 A 股主板/中小板/创业板：沪市 60xxxx，深市 000xxx/001xxx/002xxx/300xxx。"""
    if _is_b_share(code):
        return False
    return code.startswith(('600', '601', '603', '605', '000', '001', '002', '003', '300', '301'))


@dataclass
class StockUniverse:
    """
    点时股票池。

    使用方式：
        universe = StockUniverse.build()   # 从数据源构建（首次较慢，会缓存）
        eligible = universe.eligible_at('2023-06-30')

    也可用预加载数据构建：
        universe = StockUniverse.from_cache(
            live_prices=...,
            delist_prices=...,
            ...
        )
    """

    # 全量股票元信息：code -> {name, listing_date, delist_date, is_b, is_delisted}
    stock_meta: dict = field(default_factory=dict)

    # 存活股日线数据：code -> pd.DataFrame (date index, columns: close, outstanding_share, ...)
    live_daily: dict = field(default_factory=dict)

    # 退市股前复权收盘价：code -> pd.Series (date index)
    delist_prices: dict = field(default_factory=dict)

    # 退市股信息：code -> {name, delist_date, data_start, data_end, ...}
    delist_info: dict = field(default_factory=dict)

    # ST 曾用名记录：code -> list[str]（无日期，标记为不精确）
    name_history: dict = field(default_factory=dict)

    # 标记：ST状态是否精确（有日期）
    st_precise: bool = False

    # 缓存的存活股前复权价格：code -> pd.Series
    _qfq_cache: dict = field(default_factory=dict, repr=False)

    # 拉取失败的存活股code集合，避免重复网络请求
    _qfq_failed: set = field(default_factory=set, repr=False)

    # 降级到未复权close的存活股code集合（qfq拉取失败后fallback）
    _qfq_degraded: set = field(default_factory=set, repr=False)

    # 是否已从 parquet 缓存加载过 qfq 数据
    _qfq_parquet_loaded: bool = field(default=False, repr=False)

    # --------------------------------------------------------
    # 构建方法
    # --------------------------------------------------------
    @classmethod
    def build(cls, data_dir: str | None = None, max_live: int | None = None,
              skip_st_history: bool = True, verbose: bool = True) -> "StockUniverse":
        """
        从数据源构建股票池（首次较慢）。
        """
        import akshare as ak

        ddir = data_dir or _DATA_DIR

        live_list = ak.stock_info_a_code_name()
        live_codes = set(live_list['code'].apply(lambda x: str(x).zfill(6)).tolist())

        delist_prices = {}
        delist_info = {}
        pkl_path = os.path.join(ddir, 'delist_prices.pkl')
        json_path = os.path.join(ddir, 'delist_info.json')
        if os.path.exists(pkl_path) and os.path.exists(json_path):
            delist_prices = pd.read_pickle(pkl_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                delist_info = json.load(f)

        stock_meta = {}
        try:
            sh_delist = ak.stock_info_sh_delist()
            sh_delist['公司代码'] = sh_delist['公司代码'].apply(lambda x: str(x).zfill(6))
            for _, row in sh_delist.iterrows():
                code = row['公司代码']
                stock_meta[code] = {
                    'name': str(row.get('公司简称', code)),
                    'listing_date': str(row['上市日期']) if pd.notna(row.get('上市日期')) else None,
                    'delist_date': str(row['暂停上市日期']) if pd.notna(row.get('暂停上市日期')) else None,
                    'is_b': _is_b_share(code),
                    'is_delisted': True,
                }
        except Exception:
            pass

        for code, info in delist_info.items():
            if code not in stock_meta:
                stock_meta[code] = {
                    'name': info.get('name', code),
                    'listing_date': None,
                    'delist_date': info.get('delist_date'),
                    'is_b': _is_b_share(code),
                    'is_delisted': True,
                }
            else:
                if not stock_meta[code].get('delist_date'):
                    stock_meta[code]['delist_date'] = info.get('delist_date')
                stock_meta[code]['name'] = info.get('name', stock_meta[code]['name'])

        for code in live_codes:
            if code not in stock_meta:
                stock_meta[code] = {
                    'name': str(live_list[live_list['code'] == code]['name'].iloc[0])
                            if code in live_list['code'].values else code,
                    'listing_date': None,
                    'delist_date': None,
                    'is_b': _is_b_share(code),
                    'is_delisted': False,
                }

        live_daily = {}
        codes_to_fetch = list(live_codes)
        if max_live is not None:
            codes_to_fetch = codes_to_fetch[:max_live]

        for i, code in enumerate(codes_to_fetch):
            try:
                sym = f'sh{code}' if code.startswith('6') else f'sz{code}'
                df = ak.stock_zh_a_daily(symbol=sym)
                if df is not None and len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[df['date'] >= '2020-01-01'].sort_values('date').reset_index(drop=True)
                    df.set_index('date', inplace=True)
                    live_daily[code] = df
            except Exception:
                pass
            if verbose and (i + 1) % 200 == 0:
                print(f"    日线拉取: {i+1}/{len(codes_to_fetch)}")

        name_history = {}
        st_precise = False
        if not skip_st_history:
            for i, code in enumerate(list(stock_meta.keys())):
                try:
                    df = ak.stock_info_change_name(symbol=code)
                    if df is not None and len(df) > 0:
                        names = df['name'].tolist()
                        name_history[code] = names
                except Exception:
                    pass
                if verbose and (i + 1) % 200 == 0:
                    print(f"    ST曾用名拉取: {i+1}/{len(stock_meta)}")

        universe = cls(
            stock_meta=stock_meta,
            live_daily=live_daily,
            delist_prices=delist_prices,
            delist_info=delist_info,
            name_history=name_history,
            st_precise=st_precise,
        )
        universe._load_qfq_from_parquet(verbose=verbose)
        return universe

    @classmethod
    def from_cache(
        cls,
        stock_meta: dict,
        live_daily: dict,
        delist_prices: dict,
        delist_info: dict,
        name_history: dict | None = None,
        st_precise: bool = False,
    ) -> "StockUniverse":
        """用预加载数据构建（用于测试，不触发网络请求）。"""
        return cls(
            stock_meta=stock_meta,
            live_daily=live_daily,
            delist_prices=delist_prices,
            delist_info=delist_info,
            name_history=name_history or {},
            st_precise=st_precise,
        )

    @classmethod
    def build_from_parquet(cls, data_dir: str | None = None,
                           verbose: bool = True) -> "StockUniverse":
        """
        从 parquet 缓存构建股票池（不触发逐只网络请求）。

        需要 data/live_daily_cache/ 和 data/qfq_cache/ 已构建。
        用 build_live_daily_cache.py 和 build_qfq_cache.py 预先构建。
        """
        import akshare as ak

        ddir = data_dir or _DATA_DIR
        live_cache_dir = os.path.join(ddir, 'live_daily_cache')

        live_meta_path = os.path.join(live_cache_dir, '_meta.json')
        if not os.path.exists(live_meta_path):
            raise FileNotFoundError(
                f"live_daily 缓存不存在: {live_meta_path}\n"
                f"请先运行: python scripts/build_live_daily_cache.py"
            )

        with open(live_meta_path, 'r', encoding='utf-8') as f:
            live_meta = json.load(f)

        if verbose:
            print(f"  从 parquet 加载 live_daily: {len(live_meta.get('success', []))} 只")

        live_daily = {}
        for code in live_meta.get('success', []):
            path = os.path.join(live_cache_dir, f'{code}.parquet')
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                live_daily[code] = df
            except Exception:
                pass

        if verbose:
            print(f"  live_daily 加载完成: {len(live_daily)} 只")

        live_list = ak.stock_info_a_code_name()
        live_list['code'] = live_list['code'].apply(lambda x: str(x).zfill(6))
        live_codes = set(live_list['code'].tolist())

        delist_prices = {}
        delist_info = {}
        pkl_path = os.path.join(ddir, 'delist_prices.pkl')
        json_path = os.path.join(ddir, 'delist_info.json')
        if os.path.exists(pkl_path) and os.path.exists(json_path):
            delist_prices = pd.read_pickle(pkl_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                delist_info = json.load(f)

        stock_meta = {}
        try:
            sh_delist = ak.stock_info_sh_delist()
            sh_delist['公司代码'] = sh_delist['公司代码'].apply(lambda x: str(x).zfill(6))
            for _, row in sh_delist.iterrows():
                code = row['公司代码']
                stock_meta[code] = {
                    'name': str(row.get('公司简称', code)),
                    'listing_date': str(row['上市日期']) if pd.notna(row.get('上市日期')) else None,
                    'delist_date': str(row['暂停上市日期']) if pd.notna(row.get('暂停上市日期')) else None,
                    'is_b': _is_b_share(code),
                    'is_delisted': True,
                }
        except Exception:
            pass

        for code, info in delist_info.items():
            if code not in stock_meta:
                stock_meta[code] = {
                    'name': info.get('name', code),
                    'listing_date': None,
                    'delist_date': info.get('delist_date'),
                    'is_b': _is_b_share(code),
                    'is_delisted': True,
                }
            else:
                if not stock_meta[code].get('delist_date'):
                    stock_meta[code]['delist_date'] = info.get('delist_date')
                stock_meta[code]['name'] = info.get('name', stock_meta[code]['name'])

        for code in live_codes:
            if code not in stock_meta:
                stock_meta[code] = {
                    'name': str(live_list[live_list['code'] == code]['name'].iloc[0])
                            if code in live_list['code'].values else code,
                    'listing_date': None,
                    'delist_date': None,
                    'is_b': _is_b_share(code),
                    'is_delisted': False,
                }

        universe = cls(
            stock_meta=stock_meta,
            live_daily=live_daily,
            delist_prices=delist_prices,
            delist_info=delist_info,
            name_history={},
            st_precise=False,
        )
        universe._load_qfq_from_parquet(verbose=verbose)
        return universe

    # --------------------------------------------------------
    # 核心API
    # --------------------------------------------------------
    def eligible_at(self, date: pd.Timestamp | str,
                    min_listing_months: int = 12) -> list[str]:
        """返回该日期可交易的A股代码列表。"""
        date = pd.Timestamp(date)

        eligible = []
        for code, meta in self.stock_meta.items():
            if meta.get('is_b', False):
                continue

            delist_date = meta.get('delist_date')
            if delist_date:
                try:
                    dd = pd.Timestamp(delist_date)
                    if dd <= date:
                        continue
                except Exception:
                    pass

            listing_date = meta.get('listing_date')
            if listing_date:
                try:
                    ld = pd.Timestamp(listing_date)
                    if ld > date:
                        continue
                    if (date - ld).days < min_listing_months * 30:
                        continue
                except Exception:
                    pass

            if not self._has_price_in_month(code, date):
                continue

            eligible.append(code)

        return sorted(eligible)

    def _has_price_in_month(self, code: str, date: pd.Timestamp) -> bool:
        """检查该股票在给定日期所在月份是否有可交易价格。"""
        month_start = date.replace(day=1)
        month_end = (month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1))

        if code in self.live_daily:
            df = self.live_daily[code]
            mask = (df.index >= month_start) & (df.index <= month_end)
            return mask.any()

        if code in self.delist_prices:
            s = self.delist_prices[code]
            mask = (s.index >= month_start) & (s.index <= month_end)
            return mask.any()

        return False

    def _get_raw_close(self, code: str, date: pd.Timestamp) -> float | None:
        """获取未复权收盘价（用于市值计算）。"""
        if code in self.live_daily:
            df = self.live_daily[code]
            if date in df.index:
                close = df.loc[date, 'close']
                if pd.notna(close):
                    return float(close)
            before = df[df.index <= date]
            if len(before) > 0:
                close = before.iloc[-1]['close']
                if pd.notna(close):
                    return float(close)
        return None

    def _get_outstanding_share(self, code: str, date: pd.Timestamp) -> float | None:
        """获取流通股本（用于市值计算）。"""
        if code in self.live_daily:
            df = self.live_daily[code]
            if date in df.index:
                share = df.loc[date, 'outstanding_share']
                if pd.notna(share) and share > 0:
                    return float(share)
            before = df[df.index <= date]
            if len(before) > 0:
                share = before.iloc[-1]['outstanding_share']
                if pd.notna(share) and share > 0:
                    return float(share)
        return None

    def market_cap_at(self, code: str, date: pd.Timestamp | str) -> float | None:
        """返回该股票在该日期的流通市值 = 未复权收盘价 × 当时流通股本。"""
        date = pd.Timestamp(date)

        if _is_b_share(code):
            return None

        raw_close = self._get_raw_close(code, date)
        if raw_close is None:
            return None

        shares = self._get_outstanding_share(code, date)
        if shares is None:
            return None

        return raw_close * shares

    def is_st_at(self, code: str, date: pd.Timestamp | str) -> bool | None:
        """返回该股票在该日期是否为ST。"""
        date = pd.Timestamp(date)

        if code not in self.stock_meta:
            return None

        names = self.name_history.get(code, [])
        current_name = self.stock_meta[code].get('name', '')

        if not names:
            if current_name and ('ST' in str(current_name).upper()):
                return True
            return None

        has_st_name = any('ST' in str(n).upper() for n in names)

        if not has_st_name:
            return False

        if not self.st_precise:
            if current_name and 'ST' in str(current_name).upper():
                return True
            return None

        return None

    # --------------------------------------------------------
    # 持久化 qfq 缓存（parquet 格式）
    # --------------------------------------------------------
    def _load_qfq_from_parquet(self, verbose: bool = False) -> tuple[int, int]:
        """
        从 data/qfq_cache/ parquet 文件加载前复权收盘价元信息。

        惰性加载：只记录哪些 code 有 parquet 文件可用，不一次性全部加载。
        后续 _get_qfq_close 按需从 parquet 读取。
        """
        if self._qfq_parquet_loaded:
            return len(self._qfq_cache), len(self._qfq_failed)

        self._qfq_parquet_loaded = True

        meta_path = os.path.join(_QFQ_CACHE_DIR, '_meta.json')
        if not os.path.exists(meta_path):
            if verbose:
                print(f"  qfq parquet 缓存不存在于 {_QFQ_CACHE_DIR}，将在运行时逐只拉取")
            return 0, 0

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            return 0, 0

        for item in meta.get('fail', []):
            if isinstance(item, dict):
                self._qfq_failed.add(item.get('code', ''))
            elif isinstance(item, str):
                self._qfq_failed.add(item)

        for code in meta.get('empty', []):
            self._qfq_failed.add(code)

        self._qfq_parquet_available: set = set(meta.get('success', []))

        n_failed = len(self._qfq_failed)
        n_available = len(self._qfq_parquet_available)

        if verbose:
            print(f"  qfq parquet 缓存: {n_available} 只可用, {n_failed} 只失败/空")

        return n_available, n_failed

    def _load_qfq_parquet_for(self, code: str) -> pd.Series | None:
        """从 parquet 文件加载单只股票的 qfq 收盘价。"""
        path = os.path.join(_QFQ_CACHE_DIR, f'{code}.parquet')
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df['close']
        except Exception:
            return None

    def preload_all_qfq(self, verbose: bool = False) -> int:
        """
        将所有 parquet 可用的 qfq 数据预加载到内存。

        回测前调用可避免逐只磁盘 I/O，加速回测。
        Returns: 加载的股票数
        """
        if not hasattr(self, '_qfq_parquet_available'):
            return 0

        count = 0
        codes = list(self._qfq_parquet_available)
        for i, code in enumerate(codes):
            if code not in self._qfq_cache:
                s = self._load_qfq_parquet_for(code)
                if s is not None and len(s) > 0:
                    self._qfq_cache[code] = s
                    count += 1
            if verbose and (i + 1) % 1000 == 0:
                print(f"    qfq 预加载: {i+1}/{len(codes)}")

        if verbose:
            print(f"  qfq 预加载完成: {count} 只加载到内存 (总 {len(self._qfq_cache)} 只)")
        return count

    def _get_qfq_close(self, code: str) -> pd.Series | None:
        """
        获取存活股的前复权收盘价（带缓存）。

        加载优先级：
        1. 内存缓存 _qfq_cache
        2. parquet 持久化缓存（data/qfq_cache/）
        3. akshare 网络拉取（adjust='qfq'）
        拉取失败的股票记入 _qfq_failed，避免重复网络请求。
        """
        if code in self._qfq_cache:
            return self._qfq_cache[code]

        if not hasattr(self, '_qfq_failed'):
            self._qfq_failed = set()
        if code in self._qfq_failed:
            return None

        # 尝试从 parquet 缓存加载
        if hasattr(self, '_qfq_parquet_available') and code in self._qfq_parquet_available:
            s = self._load_qfq_parquet_for(code)
            if s is not None and len(s) > 0:
                self._qfq_cache[code] = s
                return s
            self._qfq_failed.add(code)
            return None

        # 尝试网络拉取
        try:
            import akshare as ak
            sym = f'sh{code}' if code.startswith('6') else f'sz{code}'
            df = ak.stock_zh_a_daily(symbol=sym, adjust='qfq')
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] >= '2020-01-01'].sort_values('date')
                df.set_index('date', inplace=True)
                s = df['close']
                self._qfq_cache[code] = s
                return s
        except Exception:
            pass

        self._qfq_failed.add(code)
        return None

    def load_qfq_cache(self, path: str | None = None) -> tuple[int, int]:
        """
        从磁盘加载持久化的前复权缓存。
        兼容旧 pickle 格式和新的 parquet 格式。
        """
        n_cached, n_failed = self._load_qfq_from_parquet(verbose=True)
        if n_cached > 0 or n_failed > 0:
            return n_cached, n_failed

        p = path or os.path.join(_DATA_DIR, 'qfq_cache.pkl')
        if not os.path.exists(p):
            p = os.path.join(_DATA_DIR, 'qfq_prices.pkl')
        if not os.path.exists(p):
            return 0, 0

        try:
            import pickle as _pkl
            with open(p, 'rb') as f:
                disk = _pkl.load(f)
            qfq_data = disk.get('qfq', {})
            failed_set = set(disk.get('failed', []))
            for code, s in qfq_data.items():
                if code not in self._qfq_cache and code not in self._qfq_failed:
                    self._qfq_cache[code] = s
            for code in failed_set:
                self._qfq_failed.add(code)
            return len(qfq_data), len(failed_set)
        except Exception:
            pass
        return 0, 0

    def save_qfq_cache(self, path: str | None = None) -> str:
        """保存内存中的 qfq 缓存到 parquet 文件。"""
        os.makedirs(_QFQ_CACHE_DIR, exist_ok=True)

        meta = {
            'success': list(self._qfq_cache.keys()),
            'fail': list(self._qfq_failed),
            'empty': [],
            'build_time': pd.Timestamp.now().isoformat(),
        }

        for code, s in self._qfq_cache.items():
            if isinstance(s, pd.Series) and len(s) > 0:
                df = s.reset_index()
                df.columns = ['date', 'close']
                df.to_parquet(os.path.join(_QFQ_CACHE_DIR, f'{code}.parquet'), index=False)

        meta_path = os.path.join(_QFQ_CACHE_DIR, '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return _QFQ_CACHE_DIR

    def prefetch_qfq(self, codes: list[str] | None = None, verbose: bool = True) -> dict:
        """批量预拉取存活股的前复权数据并持久化缓存。"""
        if codes is None:
            codes = list(self.live_daily.keys())

        self._load_qfq_from_parquet(verbose=verbose)

        success = 0
        failed = 0
        already = 0
        total = len(codes)

        for i, code in enumerate(codes):
            if code in self._qfq_cache or code in self._qfq_failed:
                already += 1
                continue

            s = self._get_qfq_close(code)
            if s is not None:
                success += 1
            else:
                failed += 1

            if verbose and (i + 1) % 200 == 0:
                print(f"    qfq 预拉取: {i+1}/{total} (新增成功 {success}, 新增失败 {failed})")

        cache_path = self.save_qfq_cache()
        degraded = len(self._qfq_degraded)

        stats = {
            'total': total,
            'already_cached': already,
            'success_new': success,
            'failed_new': failed,
            'total_cached': len(self._qfq_cache),
            'total_failed': len(self._qfq_failed),
            'degraded': degraded,
            'cache_path': cache_path,
        }

        if verbose:
            print(f"  qfq 预拉取完成: "
                  f"成功 {len(self._qfq_cache)}, 失败 {len(self._qfq_failed)}, "
                  f"降级 {degraded}")
            print(f"  缓存已保存: {cache_path}")

        return stats

    def qfq_coverage_report(self) -> dict:
        """返回 qfq 数据覆盖率统计（含内存缓存 + parquet 可用缓存）。"""
        live_codes = set(self.live_daily.keys())
        all_cached = set(self._qfq_cache.keys())
        if hasattr(self, '_qfq_parquet_available'):
            all_cached = all_cached | self._qfq_parquet_available
        cached = len(live_codes & all_cached)
        failed = len(live_codes & self._qfq_failed)
        degraded = len(live_codes & self._qfq_degraded)
        missing = len(live_codes - all_cached - self._qfq_failed)
        total = len(live_codes)

        return {
            'live_total': total,
            'qfq_cached': cached,
            'qfq_failed': failed,
            'qfq_missing': missing,
            'degraded': degraded,
            'coverage_pct': round(cached / max(total, 1) * 100, 1),
        }

    def _raw_close_series(self, code: str, start: str, end: str) -> pd.Series:
        """返回未复权收盘价序列（不触发网络请求）。"""
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        if code in self.live_daily:
            s = self.live_daily[code]['close']
            return s[(s.index >= start_dt) & (s.index <= end_dt)]

        if code in self.delist_prices:
            s = self.delist_prices[code]
            return s[(s.index >= start_dt) & (s.index <= end_dt)]

        return pd.Series(dtype=float)

    def total_return_series(self, code: str, start: str, end: str) -> pd.Series:
        """
        返回前复权收盘价序列（含分红再投资收益）。

        存活股：优先用 parquet 缓存或 akshare adjust='qfq' 的前复权收盘价，
               拉取失败时降级为 live_daily 的未复权 close（记入 _qfq_degraded）。
        退市股：delist_prices 已经是前复权收盘价。
        """
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        if code in self.delist_prices:
            s = self.delist_prices[code]
            return s[(s.index >= start_dt) & (s.index <= end_dt)]

        if code in self.live_daily:
            qfq = self._get_qfq_close(code)
            if qfq is not None:
                return qfq[(qfq.index >= start_dt) & (qfq.index <= end_dt)]

            self._qfq_degraded.add(code)
            import warnings as _w
            _w.warn(
                f"存活股 {code} 前复权数据拉取失败，降级使用未复权 close。"
                f"分红除权日的收益计算可能不准确。",
                stacklevel=2,
            )
            df = self.live_daily[code]
            s = df['close']
            return s[(s.index >= start_dt) & (s.index <= end_dt)]

        return pd.Series(dtype=float)

    def coverage_report(self) -> dict:
        """返回数据覆盖率报告。"""
        total_stocks = len(self.stock_meta)
        a_share_count = sum(1 for m in self.stock_meta.values() if not m.get('is_b', False))
        b_share_count = sum(1 for m in self.stock_meta.values() if m.get('is_b', False))
        live_count = sum(1 for m in self.stock_meta.values() if not m.get('is_delisted', False) and not m.get('is_b', False))
        delist_count = sum(1 for m in self.stock_meta.values() if m.get('is_delisted', False) and not m.get('is_b', False))

        has_live_daily = len(self.live_daily)
        has_delist_prices = len(self.delist_prices)

        has_market_cap = 0
        for code, df in self.live_daily.items():
            if 'outstanding_share' in df.columns and (df['outstanding_share'] > 0).any():
                has_market_cap += 1

        has_name_history = len(self.name_history)
        has_listing_date = sum(1 for m in self.stock_meta.values() if m.get('listing_date'))
        has_delist_date = sum(1 for m in self.stock_meta.values() if m.get('delist_date'))

        return {
            'total_stocks': total_stocks,
            'a_shares': a_share_count,
            'b_shares_excluded': b_share_count,
            'live_a_shares': live_count,
            'delisted_a_shares': delist_count,
            'has_daily_prices': has_live_daily,
            'has_delist_prices': has_delist_prices,
            'has_market_cap_data': has_market_cap,
            'market_cap_coverage_pct': round(has_market_cap / max(a_share_count, 1) * 100, 1),
            'has_name_history': has_name_history,
            'has_listing_date': has_listing_date,
            'has_delist_date': has_delist_date,
            'st_state_precise': self.st_precise,
            'qfq_coverage': self.qfq_coverage_report(),
            'note': '退市股市值数据缺失（无outstanding_share），不可用于小市值排序',
        }
