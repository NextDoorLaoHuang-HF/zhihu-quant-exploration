"""
scripts/lib/hybrid.py — 可转债等权/CB-HRP混合策略回测模块

策略：混合两个可转债组合的收益：
1. CB等权：动态池中所有在交易可转债的等权月收益
2. CB-HRP：用层次风险平价(HRP)分配权重的可转债组合月收益
混合比例通过 walk-forward OOS 估计。

ETF（511380 可转债ETF）仅用于日期对齐和基准对比，不参与策略收益计算。

替代 01_grid_etf_premium.py 中 run_hybrid_strategy() 的三个问题：
1. 可转债收益是随机噪声缩放到23%目标年化 → 改用真实可转债价格
2. "HRP"实际是全样本1/波动率固定权重 → 改用真正滚动252日协方差+层次聚类+递归二分
3. 混合比例在全样本扫描0%-100%后报"最佳" → 改用 walk-forward OOS 估计

数据源策略：
- ak.bond_zh_cov() 获取可转债列表（含上市日）
- ak.bond_zh_hs_cov_daily() 获取每只可转债历史日K线
- 已退出券的最后收盘价作为退出最终价（代理值，非实际赎回/强赎结算价）
- ETF日收益仅用于日期对齐（调用方负责拉取）

依赖：numpy, pandas, scipy（已在 requirements.txt 中）
"""
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform, pdist

warnings.filterwarnings('ignore')

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')


# ============================================================
# A. 可转债动态池
# ============================================================

class CBPool:
    """
    可转债动态候选池。

    核心功能：
    - eligible_at(date) — 返回该日期仍在交易的可转债列表
    - monthly_returns() — 返回月度收益面板（退出月收益不为0）
    - 退出券的最后价格用于计算退出月真实收益

    使用方式：
        pool = CBPool(prices=..., meta=...)
        eligible = pool.eligible_at('2023-06-01')
        monthly_ret = pool.monthly_returns()

    注意：
    - exit_final_price 是最后交易日收盘价（代理值），不是实际到期赎回/强赎结算价。
      数据源 ak.bond_zh_hs_cov_daily 只返回OHLCV，退出原因和结算价无法获取。
    - 部分退出券的 exit_final_price 可能为 0 或 NaN，这些在 monthly_returns
      中会被处理：0 → 真实0收益（合法），NaN → 该券退出月收益为NaN。
    """

    def __init__(self, prices: dict[str, pd.Series], meta: dict):
        """
        参数：
            prices: {bond_code: pd.Series(index=Date, values=收盘价)}
            meta:   {bond_code: {listing_date, delist_date, exit_reason, exit_final_price}}
        """
        self.prices = prices
        self.meta = meta

        # 构建价格面板（DataFrame, columns=bond_code）
        self._price_panel = self._build_price_panel()

        # 数据截止日期：用于排除不完整的当月
        self.data_cutoff_date = self._compute_cutoff_date()

    def _build_price_panel(self) -> pd.DataFrame:
        """将各券价格序列合并为 DataFrame，列=债券代码。"""
        if not self.prices:
            return pd.DataFrame()

        df = pd.DataFrame(self.prices)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def _compute_cutoff_date(self) -> pd.Timestamp | None:
        """
        计算数据截止日期：所有价格序列的最后日期。

        用于排除未结束的当月——如果数据只到7月14日，
        7月的月度收益不完整，不应被使用。
        """
        if not self.prices:
            return None
        max_dates = []
        for s in self.prices.values():
            if len(s) > 0:
                max_dates.append(s.index[-1])
        if not max_dates:
            return None
        return max(max_dates)

    def eligible_at(self, date: str | pd.Timestamp) -> list[str]:
        """
        返回该日期仍在交易的可转债列表。

        条件：
        - 已上市（上市日 ≤ date；无上市日则假设已上市）
        - 未退出（退出日 > date 或无退出日）
        - 当月有价格数据
        """
        date = pd.Timestamp(date)
        eligible = []

        for code, meta_info in self.meta.items():
            if code not in self.prices:
                continue

            # 上市日检查
            listing_date = meta_info.get('listing_date')
            if listing_date:
                try:
                    ld = pd.Timestamp(listing_date)
                    if ld > date:
                        continue
                except Exception:
                    pass

            # 退出日检查
            delist_date = meta_info.get('delist_date')
            if delist_date:
                try:
                    dd = pd.Timestamp(delist_date)
                    if dd <= date:
                        continue
                except Exception:
                    pass

            # 当月有价格
            s = self.prices[code]
            month_start = date.replace(day=1)
            month_end = (month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1))
            mask = (s.index >= month_start) & (s.index <= month_end)
            if not mask.any():
                continue

            eligible.append(code)

        return sorted(eligible)

    def monthly_returns(self) -> pd.DataFrame:
        """
        返回月度收益面板（DataFrame, index=月末日期, columns=债券代码）。

        收益计算：
        - 正常月：月末收盘 / 月初收盘 - 1
        - 退出月：用退出最终价（exit_final_price）作为月末价
        - 退出后月：NaN（不在池中）
        - 上市首月：NaN（不满月）

        关键：
        - 退出月收益不为0，反映真实价格变化。
        - exit_final_price 是最后收盘价（代理值），不是实际赎回结算价。
        - 排除不完整的当月：如果数据截止到7月14日，7月不作为完整月。
        """
        if self._price_panel.empty:
            return pd.DataFrame()

        panel = self._price_panel.copy()

        # 按月重采样：取每月最后一个交易日的价格
        # 用 pd.concat(axis=1) 而非逐列赋值，确保各券不同时间范围正确合并
        monthly_series_list = []

        for code in panel.columns:
            s = panel[code].dropna()
            if len(s) == 0:
                continue

            # 获取该券的元信息
            meta_info = self.meta.get(code, {})
            delist_date = meta_info.get('delist_date')
            exit_final_price = meta_info.get('exit_final_price')

            # 按月分组，取每月最后一个有效价格
            monthly = s.resample('ME').last()

            # 如果有退出最终价，在退出月用它替代
            if delist_date and exit_final_price is not None:
                delist_month = pd.Timestamp(delist_date).to_period('M')
                delist_month_end = delist_month.to_timestamp(how='end').normalize()
                # 确保退出月在 index 中
                if delist_month_end not in monthly.index:
                    monthly.loc[delist_month_end] = float(exit_final_price)
                    monthly = monthly.sort_index()
                else:
                    monthly.loc[delist_month_end] = float(exit_final_price)

            monthly.name = code
            monthly_series_list.append(monthly)

        if not monthly_series_list:
            return pd.DataFrame()

        monthly_prices = pd.concat(monthly_series_list, axis=1)

        # 计算月收益
        monthly_ret = monthly_prices.pct_change(fill_method=None)

        # 排除不完整的当月
        # 数据截止日所在月如果还没结束（即截止日不是月末），删掉该月
        if self.data_cutoff_date is not None:
            cutoff = self.data_cutoff_date
            # 月末日期 > 数据截止日 的月份都不完整
            # resample('ME').last() 给的是每月最后一天的 00:00
            # 如果 cutoff 是7月14日，7月31日的标签 > cutoff，7月不完整
            incomplete_mask = monthly_ret.index > cutoff
            if incomplete_mask.any():
                monthly_ret = monthly_ret[~incomplete_mask]

        return monthly_ret

    def daily_returns(self, code: str, start: str, end: str) -> pd.Series:
        """返回某只可转债的日收益序列。"""
        if code not in self.prices:
            return pd.Series(dtype=float)

        s = self.prices[code]
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        s = s[(s.index >= start_dt) & (s.index <= end_dt)]
        return s.pct_change(fill_method=None).dropna()


# ============================================================
# B. 真正的滚动 HRP (Hierarchical Risk Parity)
# ============================================================

class HRPOptimizer:
    """
    滚动 HRP 权重优化器。

    实现步骤（每月调仓时）：
    1. 用过去 252 个交易日的日收益估计协方差矩阵
    2. 距离矩阵 → 层次聚类（scipy.cluster.hierarchy.linkage）
    3. 递归二分配置（bisection）
    4. 只使用调仓日之前的数据（无前视偏差）
    5. 权重归一化到和为1

    参考实现：López de Prado (2016) "Building Diversified Portfolios that
    Outperform Out-of-Sample"
    """

    def __init__(self, lookback: int = 252):
        """
        参数：
            lookback: 滚动窗口长度（交易日），默认252（约1年）
        """
        self.lookback = lookback

    def _get_cov_end_date(self, target_date: pd.Timestamp) -> pd.Timestamp:
        """
        返回协方差矩阵的截止日期。

        确保截止日严格早于 target_date（无前视偏差）。
        target_date 所在月的数据不被使用。
        """
        return target_date - pd.Timedelta(days=1)  # type: ignore[return-value]

    def compute_weights(
        self,
        pool: CBPool,
        target_date: pd.Timestamp | str,
        eligible_codes: list[str] | None = None,
    ) -> pd.Series | None:
        """
        计算给定日期的 HRP 权重。

        参数：
            pool: 可转债池
            target_date: 调仓日
            eligible_codes: 可选，限定候选券列表

        返回：
            pd.Series(index=bond_code, values=权重)，和为1
            如果候选券不足2只，返回等权
        """
        target_date = pd.Timestamp(target_date)

        if eligible_codes is None:
            eligible_codes = pool.eligible_at(target_date)

        if len(eligible_codes) == 0:
            return None

        if len(eligible_codes) == 1:
            return pd.Series({eligible_codes[0]: 1.0})

        # 截取过去 lookback 个交易日的日收益
        # 确保不用 target_date 当天及之后的数据
        cov_end = self._get_cov_end_date(target_date)

        daily_returns = {}
        for code in eligible_codes:
            if code not in pool.prices:
                continue
            s = pool.prices[code]
            # 只取 cov_end 之前的数据
            s = s[s.index <= cov_end]
            if len(s) < self.lookback:
                # 数据不足，用全部可用数据
                pass
            else:
                s = s.iloc[-self.lookback:]

            ret = s.pct_change(fill_method=None).dropna()
            if len(ret) > 0:
                daily_returns[code] = ret

        if len(daily_returns) < 2:
            # 不足2只券，等权
            codes = list(daily_returns.keys()) if daily_returns else eligible_codes
            n = len(codes)
            return pd.Series({c: 1.0/n for c in codes}) if n > 0 else None

        # 构建日收益矩阵
        ret_df = pd.DataFrame(daily_returns)
        # 对齐日期，删除含NaN的行
        ret_df = ret_df.dropna(how='all')

        # 计算协方差矩阵
        cov = ret_df.cov()

        if cov.empty or cov.shape[0] < 2:
            n = len(daily_returns)
            return pd.Series({c: 1.0/n for c in daily_returns.keys()})

        # HRP 步骤 1: 相关性 → 距离矩阵
        corr = ret_df.corr()
        # 距离 d = sqrt(0.5 * (1 - corr))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0, None))

        # 确保对角线为0
        np.fill_diagonal(dist, 0.0)

        # 对称距离矩阵 → condensed distance vector → 层次聚类
        try:
            condensed_1d = squareform(dist, checks=False)
            Z = linkage(condensed_1d, method='single')
        except Exception:
            # 如果聚类失败，降级为等权
            n = len(daily_returns)
            return pd.Series({c: 1.0/n for c in daily_returns.keys()})

        # HRP 步骤 2: 递归二分配置
        weights = self._get_recursive_bisection_weights(cov, Z, list(cov.columns))

        return weights

    def _get_recursive_bisection_weights(
        self,
        cov: pd.DataFrame,
        Z: np.ndarray,
        codes: list[str],
    ) -> pd.Series:
        """
        递归二分配置（recursive bisection）。

        算法：
        1. 将所有资产归为一个簇
        2. 用聚类树二分
        3. 对每个子簇，按逆波动率（1/σ）分配权重
        4. 递归直到每个子簇只有一个资产
        """
        n = len(codes)
        weights = pd.Series(1.0, index=codes)

        # 获取聚类树中的所有合并
        # Z 的每行: [idx1, idx2, dist, count]
        # 我们需要从根节点开始递归二分

        # 构建簇的层次结构
        clusters = self._get_cluster_leaves(Z, n)

        # 递归二分
        self._recursive_bisection(cov, Z, n, clusters, weights, list(range(n)))

        # 归一化
        weights = weights / weights.sum()
        return weights

    def _get_cluster_leaves(self, Z: np.ndarray, n: int) -> dict:
        """获取每个聚类节点的叶子索引列表。"""
        clusters = {}
        # 初始叶子
        for i in range(n):
            clusters[i] = [i]

        # Z 的每行创建一个新节点
        for i, row in enumerate(Z):
            node_id = n + i
            left = int(row[0])
            right = int(row[1])
            left_members = clusters.get(left, [left] if left < n else [])
            right_members = clusters.get(right, [right] if right < n else [])
            clusters[node_id] = left_members + right_members

        return clusters

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        Z: np.ndarray,
        n: int,
        clusters: dict,
        weights: pd.Series,
        cluster_indices: list[int],
    ):
        """递归二分权重分配。"""
        # 找到当前簇对应的聚类节点
        # 从最高层节点开始
        root_id = n + len(Z) - 1

        # 递归处理
        self._bisect_cluster(cov, Z, n, clusters, weights, root_id)

    def _bisect_cluster(
        self,
        cov: pd.DataFrame,
        Z: np.ndarray,
        n: int,
        clusters: dict,
        weights: pd.Series,
        node_id: int,
    ):
        """递归二分一个簇。"""
        members = clusters.get(node_id, [])
        if len(members) <= 1:
            return

        # 找到该节点的两个子簇
        # Z 中 node_id 对应的行
        z_idx = node_id - n
        if z_idx < 0 or z_idx >= len(Z):
            return

        row = Z[z_idx]
        left_child = int(row[0])
        right_child = int(row[1])

        left_members = clusters.get(left_child, [left_child] if left_child < n else [])
        right_members = clusters.get(right_child, [right_child] if right_child < n else [])

        if len(left_members) == 0 or len(right_members) == 0:
            return

        # 计算两个子簇的逆波动率
        left_vol = self._cluster_volatility(cov, left_members)
        right_vol = self._cluster_volatility(cov, right_members)

        # 分配权重：逆波动率加权
        alpha = 1.0 / left_vol
        beta = 1.0 / right_vol
        total = alpha + beta

        # 获取当前权重
        left_weight = weights.iloc[left_members].sum()
        right_weight = weights.iloc[right_members].sum()
        total_weight = left_weight + right_weight

        if total_weight > 0:
            # 按逆波动率比例分配
            new_left = total_weight * (alpha / total)
            new_right = total_weight * (beta / total)

            # 缩放子簇内的权重
            if left_weight > 0:
                scale_left = new_left / left_weight
                for idx in left_members:
                    weights.iloc[idx] *= scale_left

            if right_weight > 0:
                scale_right = new_right / right_weight
                for idx in right_members:
                    weights.iloc[idx] *= scale_right

        # 递归
        self._bisect_cluster(cov, Z, n, clusters, weights, left_child)
        self._bisect_cluster(cov, Z, n, clusters, weights, right_child)

    def _cluster_volatility(self, cov: pd.DataFrame, members: list[int]) -> float:
        """计算簇的波动率（组合方差的开方）。"""
        if len(members) == 0:
            return 1e-6

        if len(members) == 1:
            return float(np.sqrt(cov.iloc[members[0], members[0]]))

        sub_cov = cov.iloc[members, :].iloc[:, members]
        # 等权组合方差 = w' Σ w，w = 1/n
        n = len(members)
        w = np.ones(n) / n
        portfolio_var = w @ sub_cov.values @ w
        return float(np.sqrt(max(portfolio_var, 1e-12)))


# ============================================================
# C. 混合比例 Walk-Forward
# ============================================================

class HybridBacktest:
    """
    可转债等权/CB-HRP混合策略 walk-forward 回测。

    策略：混合两个可转债组合的收益：
    - CB等权：动态池中所有在交易可转债的等权月收益
    - CB-HRP：用层次风险平价(HRP)分配权重的可转债组合月收益
    ETF仅用于日期对齐，不参与策略收益计算。

    流程：
    1. 训练窗口（如3年）→ 选择 CB等权/CB-HRP 比例（按训练窗口内夏普最大化）
    2. 下一段（如1年）作为 OOS → 用训练出的比例计算混合收益
    3. 滚动拼接所有 OOS 收益
    4. 同时报告固定比例（20/80, 50/50, 80/20）的对比

    关键：
    - 比例在OOS段开始前已确定，不是全样本优化
    - 扣除真实交易成本（双边 cost_bps）
    - 不使用 np.random.normal 或缩放操作
    - ETF仅用于日期对齐，不参与收益计算
    """

    def __init__(
        self,
        cb_pool: CBPool,
        etf_returns: pd.Series,
        train_months: int = 36,
        oos_months: int = 12,
        cost_bps: float = 10.0,
        hrp_lookback: int = 252,
    ):
        """
        参数：
            cb_pool: 可转债池
            etf_returns: ETF日收益序列（仅用于日期对齐，不参与策略收益）
            train_months: 训练窗口长度（月）
            oos_months: 每段OOS长度（月）
            cost_bps: 单边交易成本（基点），默认10bps=0.1%
            hrp_lookback: HRP滚动窗口长度（交易日）
        """
        self.cb_pool = cb_pool
        self.etf_returns = etf_returns
        self.train_months = train_months
        self.oos_months = oos_months
        self.cost_bps = cost_bps
        self.hrp_optimizer = HRPOptimizer(lookback=hrp_lookback)

    def run(self) -> dict:
        """
        运行完整 walk-forward 回测。

        返回 dict 包含：
        - oos_segments: 每个OOS段的记录（train_end, oos_start, oos_end, cb_ratio, metrics）
        - oos_returns: 拼接的OOS月收益序列
        - oos_metrics: OOS整体绩效指标
        - fixed_ratios: 固定比例对比（{ratio: metrics}）
        - cb_monthly: 可转债等权月收益
        - hrp_monthly: HRP组合月收益
        - etf_monthly: ETF月收益（仅用于基准对比）
        """
        # 获取可转债月度收益面板
        cb_monthly_panel = self.cb_pool.monthly_returns()

        if cb_monthly_panel.empty:
            raise ValueError("可转债月度收益面板为空")

        # 计算可转债等权月收益
        cb_equal_weight_monthly = cb_monthly_panel.mean(axis=1).dropna()

        # 计算 HRP 组合月收益（滚动调仓）
        hrp_monthly = self._compute_hrp_monthly_returns(cb_monthly_panel)

        # ETF月收益
        etf_monthly = self.etf_returns.resample('ME').apply(lambda x: (1+x).prod()-1).dropna()

        # 对齐时间
        common_idx = cb_equal_weight_monthly.index.intersection(hrp_monthly.index)
        common_idx = common_idx.intersection(etf_monthly.index)

        if len(common_idx) == 0:
            raise ValueError("可转债与ETF收益无重叠时间区间")

        cb_eq = cb_equal_weight_monthly.loc[common_idx]
        hrp = hrp_monthly.loc[common_idx]
        etf = etf_monthly.loc[common_idx]

        # Walk-forward OOS
        oos_segments = []
        oos_returns_list = []

        # 按月滚动
        total_months = len(common_idx)
        start_idx = self.train_months  # 第一个OOS段从 train_months 开始

        for i in range(start_idx, total_months, self.oos_months):
            train_start = i - self.train_months
            train_end_idx = i - 1
            oos_start_idx = i
            oos_end_idx = min(i + self.oos_months, total_months)

            if oos_end_idx <= oos_start_idx:
                break

            train_end_date = common_idx[train_end_idx]
            oos_start_date = common_idx[oos_start_idx]
            oos_end_date = common_idx[min(oos_end_idx - 1, total_months - 1)]

            # 训练窗口内选择最优比例（按夏普）
            train_cb = cb_eq.iloc[train_start:train_end_idx+1]
            train_hrp = hrp.iloc[train_start:train_end_idx+1]

            best_ratio = self._optimize_ratio(train_cb, train_hrp)

            # OOS段收益
            oos_cb = cb_eq.iloc[oos_start_idx:oos_end_idx]
            oos_hrp = hrp.iloc[oos_start_idx:oos_end_idx]

            # 混合OOS收益（扣交易成本）
            mix_oos = oos_cb * best_ratio + oos_hrp * (1 - best_ratio)

            # 交易成本：每次调仓的成本
            # 简化：假设每次调仓换手率50%，双边成本
            turnover = 0.5
            cost = turnover * self.cost_bps / 10000
            mix_oos = mix_oos - cost

            oos_returns_list.append(mix_oos)

            # OOS段指标
            from lib.metrics import compute_metrics
            seg_metrics = compute_metrics(mix_oos) if len(mix_oos) > 0 else None

            oos_segments.append({
                'train_end': str(train_end_date.date()) if hasattr(train_end_date, 'date') else str(train_end_date),
                'oos_start': str(oos_start_date.date()) if hasattr(oos_start_date, 'date') else str(oos_start_date),
                'oos_end': str(oos_end_date.date()) if hasattr(oos_end_date, 'date') else str(oos_end_date),
                'cb_ratio': float(best_ratio),
                'n_oos_months': len(mix_oos),
                'metrics': seg_metrics,
            })

        # 拼接OOS收益
        oos_returns = pd.concat(oos_returns_list) if oos_returns_list else pd.Series(dtype=float)

        # OOS整体指标
        from lib.metrics import compute_metrics
        oos_metrics = compute_metrics(oos_returns) if len(oos_returns) > 0 else None

        # 固定比例对比（与动态比例在相同OOS区间上比较）
        fixed_ratios = {}
        for ratio in [0.2, 0.5, 0.8]:
            mix_fixed = cb_eq * ratio + hrp * (1 - ratio)

            # 只在OOS区间内计算（与动态比例同区间）
            oos_mask = pd.Series(False, index=mix_fixed.index)
            for seg in oos_segments:
                start = seg['oos_start']
                end = seg['oos_end']
                mask = (mix_fixed.index >= start) & (mix_fixed.index <= end)
                oos_mask |= mask
                # 同成本路径：与动态比例一样扣交易成本
                turnover = 0.5
                cost = turnover * self.cost_bps / 10000
                mix_fixed.loc[mask] -= cost

            mix_fixed_oos = mix_fixed[oos_mask]
            fixed_metrics = compute_metrics(mix_fixed_oos) if len(mix_fixed_oos) > 0 else None
            # 标签用 round 避免浮点误差（0.8*100=80, 0.2*100=20，但 int 可能出80/19）
            cb_pct = round(ratio * 100)
            hrp_pct = round((1 - ratio) * 100)
            fixed_ratios[f'{cb_pct}/{hrp_pct}'] = fixed_metrics

        # 统计每月候选券数
        n_candidates_per_month = {}
        for date in common_idx:
            n = len(self.cb_pool.eligible_at(date))
            n_candidates_per_month[str(date.date())] = n

        # 数据截止日期
        data_cutoff = self.cb_pool.data_cutoff_date

        # 退出券异常值统计
        exited_bonds = [c for c, m in self.cb_pool.meta.items() if m.get('delist_date')]
        exit_anomalies = {
            'zero_return': 0,
            'nan_return': 0,
            'total_exited': len(exited_bonds),
        }
        # 统计 exit_final_price 为 0 或 NaN 的退出券
        for code in exited_bonds:
            efp = self.cb_pool.meta[code].get('exit_final_price')
            if efp is None or (isinstance(efp, float) and np.isnan(efp)):
                exit_anomalies['nan_return'] += 1
            elif efp == 0:
                exit_anomalies['zero_return'] += 1

        return {
            'oos_segments': oos_segments,
            'oos_returns': oos_returns,
            'oos_metrics': oos_metrics,
            'fixed_ratios': fixed_ratios,
            'cb_monthly': cb_eq,
            'hrp_monthly': hrp,
            'etf_monthly': etf,
            'n_bonds_total': len(self.cb_pool.meta),
            'n_candidates_per_month': n_candidates_per_month,
            'data_cutoff_date': str(data_cutoff.date()) if data_cutoff else None,
            'exit_anomalies': exit_anomalies,
            'exit_final_price_note': (
                'exit_final_price is last daily close (proxy), not actual redemption/settlement price. '
                'Data source ak.bond_zh_hs_cov_daily only provides OHLCV; '
                'exit reason and settlement value are not available from this source. '
                'Actual settlement may differ (e.g. redemption at par ~100, forced redemption at ~100-110).'
            ),
            'period': {
                'start': str(common_idx[0].date()) if len(common_idx) > 0 else None,
                'end': str(common_idx[-1].date()) if len(common_idx) > 0 else None,
                'n_months': len(common_idx),
            },
            'config': {
                'train_months': self.train_months,
                'oos_months': self.oos_months,
                'cost_bps': self.cost_bps,
                'hrp_lookback': self.hrp_optimizer.lookback,
            },
        }

    def _compute_hrp_monthly_returns(self, cb_monthly_panel: pd.DataFrame) -> pd.Series:
        """
        计算 HRP 组合的月收益序列。

        时序：t-1 月末权重应用于 t 月收益。
        - 第 i 个月的权重基于第 i-1 月末（或更早）的数据计算。
        - 第 0 个月无前月权重，使用等权（或0）。
        - 这消除了同月前视偏差（旧版在 t 月末算权重用于 t 月收益）。
        """
        # 获取所有月末日期
        month_dates = cb_monthly_panel.index

        hrp_returns = []

        for i, date in enumerate(month_dates):
            if i == 0:
                # 第0个月无前月权重，用等权
                eligible = self.cb_pool.eligible_at(date)
                if len(eligible) == 0:
                    hrp_returns.append(0.0)
                else:
                    # 等权
                    n = len(eligible)
                    portfolio_ret = 0.0
                    for code in eligible:
                        if code in cb_monthly_panel.columns:
                            r = cb_monthly_panel.loc[date, code]
                            if pd.notna(r):
                                portfolio_ret += r / n
                    hrp_returns.append(portfolio_ret)
                continue

            # 用上月末作为权重计算日
            prev_date = month_dates[i - 1]
            eligible = self.cb_pool.eligible_at(prev_date)

            if len(eligible) < 2:
                # 不足2只券，用等权
                if len(eligible) == 1:
                    code = eligible[0]
                    if code in cb_monthly_panel.columns:
                        ret = cb_monthly_panel.loc[date, code]
                        if pd.notna(ret):
                            hrp_returns.append(ret)
                        else:
                            hrp_returns.append(0.0)
                    else:
                        hrp_returns.append(0.0)
                else:
                    hrp_returns.append(0.0)
                continue

            # 在上月末计算 HRP 权重（无前视）
            weights = self.hrp_optimizer.compute_weights(
                self.cb_pool, prev_date, eligible
            )

            if weights is None or len(weights) == 0:
                hrp_returns.append(0.0)
                continue

            # 用上月末权重加权当月收益
            portfolio_ret = 0.0
            for code, w in weights.items():
                if code in cb_monthly_panel.columns:
                    r = cb_monthly_panel.loc[date, code]
                    if pd.notna(r):
                        portfolio_ret += r * w

            hrp_returns.append(portfolio_ret)

        return pd.Series(hrp_returns, index=month_dates, name='hrp_monthly')

    def _optimize_ratio(self, cb_returns: pd.Series, hrp_returns: pd.Series) -> float:
        """
        在训练窗口内选择最优 CB等权/CB-HRP 比例。

        优化目标：最大化夏普比率。
        扫描范围：0% 到 100%，步长5%。

        返回最优 CB等权 比例。
        """
        best_sharpe = -np.inf
        best_ratio = 0.5  # 默认50/50

        for cb_pct in np.arange(0, 1.01, 0.05):
            mix = cb_returns * cb_pct + hrp_returns * (1 - cb_pct)
            if mix.std() > 0:
                sharpe = mix.mean() / mix.std() * np.sqrt(12)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_ratio = cb_pct

        return best_ratio


# ============================================================
# 结果输出
# ============================================================

def save_results(results: dict, run_id: str | None = None) -> str:
    """
    保存回测结果到 results/<run_id>/hybrid.json。

    返回保存路径。
    """
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    result_dir = os.path.join(_RESULTS_DIR, run_id)
    os.makedirs(result_dir, exist_ok=True)

    # 序列化（pd.Series → list）
    def serialize(obj):
        if isinstance(obj, pd.Series):
            return {
                'index': [str(x.date()) if hasattr(x, 'date') else str(x) for x in obj.index],
                'values': obj.tolist(),
            }
        if isinstance(obj, pd.DataFrame):
            return {
                'index': [str(x.date()) if hasattr(x, 'date') else str(x) for x in obj.index],
                'columns': list(obj.columns),
                'values': obj.values.tolist(),
            }
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    output = {
        'oos_segments': results['oos_segments'],
        'oos_metrics': results['oos_metrics'],
        'fixed_ratios': results['fixed_ratios'],
        'n_bonds_total': results['n_bonds_total'],
        'n_candidates_per_month': results.get('n_candidates_per_month', {}),
        'data_cutoff_date': results.get('data_cutoff_date'),
        'exit_anomalies': results.get('exit_anomalies', {}),
        'exit_final_price_note': results.get('exit_final_price_note', ''),
        'period': results['period'],
        'config': results['config'],
    }

    output = serialize(output)

    path = os.path.join(result_dir, 'hybrid.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 同时保存 OOS 收益序列为 CSV
    if 'oos_returns' in results and len(results['oos_returns']) > 0:
        csv_path = os.path.join(result_dir, 'oos_returns.csv')
        results['oos_returns'].to_csv(csv_path, header=['oos_return'])

    return path
