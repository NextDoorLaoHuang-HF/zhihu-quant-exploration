"""
统一网格交易回测引擎 — scripts/lib/grid_engine.py

修复 GitHub Issue #1 中的 6+1 个问题：
1. 卖出记账错误（现金多乘一次价格）
2. 资金/持仓不足时仍推进 current_grid
3. 用当日收盘价信号同日成交（执行时点问题）
4. 一天跨多格全部按最终收盘价成交（应按各格挂单价）
5. 忽略最低5元佣金
6. walk-forward 每年重置不继承资金/持仓
7. 60%底仓网格只对比100%买持，无法隔离网格操作贡献

所有网格脚本（扫描/成本/WF/图表）应调用此引擎。
依赖：pandas, numpy（已在 requirements.txt 中）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GridConfig:
    """网格策略参数配置。"""
    grid_pct: float        # 每格幅度（如 0.05 = 5%）
    base_position: float   # 底仓比例（如 0.6 = 60%）
    grid_capital: float    # 每格资金（元）
    max_grids: int = 10    # 最大网格层数
    commission_rate: float = 0.00025   # 佣金费率（万2.5）
    min_commission: float = 5.0       # 最低佣金（5元）
    contract_size: int = 100           # ETF整手（100股/手，设1则不取整）


@dataclass
class GridResult:
    """回测结果，包含三条净值曲线和统计指标。"""
    grid_pv: pd.Series            # 网格策略净值
    bh_pv: pd.Series              # 100%买入持有净值
    base_benchmark_pv: pd.Series  # 底仓比例买持+剩余现金 净值（隔离网格贡献）
    final_value: float            # 期末总资产
    trades: int                   # 总成交笔数
    grid_annual_return: float     # 网格年化收益
    bh_annual_return: float       # 买持年化收益
    base_benchmark_annual_return: float  # 底仓基准年化收益
    excess_return: float         # 网格超额 = grid_annual - bh_annual
    grid_excess_vs_base: float   # 网格相对底仓基准超额 = grid_annual - base_benchmark_annual


class GridEngine:
    """
    网格交易回测引擎。

    核心逻辑：
    - 以底仓比例建仓，剩余现金做网格
    - 价格每下跌 grid_pct 买入一格，每上涨 grid_pct 卖出一格
    - 信号在 T 日收盘产生，T+1 日执行（避免同日信号+成交）
    - 多格日内变动：逐格在各自触发价成交
    - 资金/持仓不足时不成交，不推进 current_grid
    - walk-forward 期间继承资金和持仓

    用法：
        config = GridConfig(grid_pct=0.05, base_position=0.6, grid_capital=5000)
        engine = GridEngine(config, initial_capital=100000)
        result = engine.run(price_series)

        # walk-forward 第二段（继承状态）：
        engine2 = GridEngine(config, initial_capital=result.final_value)
        engine2.cash = engine.cash
        engine2.shares = engine.shares
        engine2.current_grid = engine.current_grid
        engine2.grid_base = engine.grid_base
        engine2._initialized = True
        result2 = engine2.run(price_series_2)
    """

    def __init__(self, config: GridConfig, initial_capital: float = 100000):
        self.config = config
        self.initial_capital = initial_capital

        # 初始状态
        self.cash: float = initial_capital * (1 - config.base_position)
        self.shares: float = 0.0
        self.current_grid: int = 0
        self.grid_base: float = 0.0
        self._initialized: bool = False

        # 交易记录
        self.trades: int = 0

    def _init_base(self, first_price: float):
        """用首日价格建底仓。"""
        base_cash = self.initial_capital * self.config.base_position
        raw_shares = base_cash / first_price
        cs = self.config.contract_size
        if cs > 1:
            raw_shares = int(raw_shares // cs) * cs
        self.shares = raw_shares
        self.cash = self.initial_capital - self.shares * first_price
        self.grid_base = first_price
        self.current_grid = 0
        self._initialized = True

    # --------------------------------------------------------
    # 手续费计算
    # --------------------------------------------------------
    def _compute_fee(self, traded_value: float) -> float:
        """佣金 = max(最低佣金, 成交额 × 费率)。"""
        return max(self.config.min_commission, traded_value * self.config.commission_rate)

    # --------------------------------------------------------
    # 网格位置计算
    # --------------------------------------------------------
    def _grid_position(self, price: float) -> int:
        """根据当前价格计算网格层数（相对 grid_base）。
        用 round() 而非 int() 避免浮点精度导致的截断（如 2.999→2）。
        """
        if self.grid_base <= 0 or price <= 0:
            return 0
        log_ratio = math.log(price / self.grid_base)
        log_grid = math.log(1 + self.config.grid_pct)
        if log_grid == 0:
            return 0
        gp = round(log_ratio / log_grid)
        if abs(gp) > self.config.max_grids:
            gp = self.config.max_grids if gp > 0 else -self.config.max_grids
        return gp

    def _grid_trigger_price(self, grid_level: int) -> float:
        """计算某网格层的触发价格。"""
        return self.grid_base * (1 + self.config.grid_pct) ** grid_level

    # --------------------------------------------------------
    # 单笔交易执行
    # --------------------------------------------------------
    def _execute_buy(self, grid_capital: float, price: float) -> bool:
        """
        买入一格。
        shares_delta = +grid_capital / price（按整手取整）
        cash_delta = -(shares_bought * price + fee)
        资金不足时返回 False，不成交。
        """
        cs = self.config.contract_size
        raw_shares = grid_capital / price
        if cs > 1:
            raw_shares = int(raw_shares // cs) * cs
        if raw_shares <= 0:
            return False

        cost = raw_shares * price
        fee = self._compute_fee(cost)
        if self.cash < cost + fee:
            return False

        self.shares += raw_shares
        self.cash -= cost + fee
        self.trades += 1
        return True

    def _execute_sell(self, grid_capital: float, price: float) -> bool:
        """
        卖出一格。
        shares_delta = -grid_capital / price（按整手取整）
        cash_delta = +(shares_sold * price - fee)  ← 修复：不额外乘价格
        持仓不足时返回 False，不成交。
        """
        cs = self.config.contract_size
        raw_shares = grid_capital / price
        if cs > 1:
            raw_shares = int(raw_shares // cs) * cs
        if raw_shares <= 0:
            return False
        if self.shares < raw_shares:
            return False

        proceeds = raw_shares * price
        fee = self._compute_fee(proceeds)
        self.shares -= raw_shares
        self.cash += proceeds - fee  # 修复：现金增加 = 卖出所得 - 佣金
        self.trades += 1
        return True

    # --------------------------------------------------------
    # 完整回测（run 方法）
    # --------------------------------------------------------
    def run(self, prices: pd.Series) -> GridResult:
        """
        运行完整网格回测。

        prices: 收盘价序列（pd.Series，带日期索引）
        返回 GridResult，包含三条净值曲线和统计指标。

        信号/执行分离：T 日收盘产生信号，T+1 日执行。
        多格变动：逐格在各自触发价成交（非全部用最终收盘价）。
        """
        if not self._initialized and len(prices) > 0:
            self._init_base(prices.iloc[0])

        # 净值序列
        grid_values = []
        bh_values = []
        base_bm_values = []

        # 买入持有基准（100% 买入持有）
        bh_shares = self.initial_capital / prices.iloc[0]

        # 底仓基准：底仓比例买持 + 剩余现金（隔离网格贡献）
        base_bm_shares = self.shares  # 与网格策略相同的底仓
        base_bm_cash = self.initial_capital - base_bm_shares * prices.iloc[0]

        # 延迟信号：记录 T 日的网格位置，T+1 日执行
        pending_signal: Optional[int] = None

        for i, (date, price) in enumerate(prices.items()):
            if i == 0:
                # 首日：建仓，记录净值
                grid_values.append(self.cash + self.shares * price)
                bh_values.append(bh_shares * price)
                base_bm_values.append(base_bm_cash + base_bm_shares * price)
                # 首日信号
                pending_signal = self._grid_position(price)
                continue

            # 执行昨日信号（用今日价格环境判断触发）
            if pending_signal is not None and pending_signal != self.current_grid:
                old_grid = self.current_grid
                new_grid = pending_signal

                if new_grid < old_grid:
                    # 买入：逐格往下买，每格在各自触发价成交
                    # 从 level L 降到 L-1 时，买入触发价 = _grid_trigger_price(L-1)
                    gc = self.config.grid_capital
                    executed_to = old_grid
                    for level in range(old_grid, new_grid, -1):
                        trigger_price = self._grid_trigger_price(level - 1)
                        if trigger_price <= 0 or not math.isfinite(trigger_price):
                            trigger_price = price
                        if self._execute_buy(gc, trigger_price):
                            executed_to = level - 1
                        else:
                            break  # 资金不足，停止
                    self.current_grid = executed_to

                elif new_grid > old_grid:
                    # 卖出：逐格往上卖，每格在各自触发价成交
                    gc = self.config.grid_capital
                    executed_to = old_grid
                    for level in range(old_grid + 1, new_grid + 1):
                        trigger_price = self._grid_trigger_price(level)
                        if trigger_price <= 0 or not math.isfinite(trigger_price):
                            trigger_price = price
                        if self._execute_sell(gc, trigger_price):
                            executed_to = level
                        else:
                            break  # 持仓不足，停止
                    self.current_grid = executed_to

            # 记录今日净值
            grid_values.append(self.cash + self.shares * price)
            bh_values.append(bh_shares * price)
            base_bm_values.append(base_bm_cash + base_bm_shares * price)

            # 生成今日信号（明日执行）
            pending_signal = self._grid_position(price)

        # 构建净值序列
        grid_pv = pd.Series(grid_values, index=prices.index, name='grid')
        bh_pv = pd.Series(bh_values, index=prices.index, name='buy_hold')
        base_bm_pv = pd.Series(base_bm_values, index=prices.index, name='base_benchmark')

        # 年化收益
        if len(prices) > 1:
            years = (prices.index[-1] - prices.index[0]).days / 365.25
        else:
            years = 1
        if years <= 0:
            years = 1

        final_value = grid_pv.iloc[-1]
        grid_ann = (grid_pv.iloc[-1] / grid_pv.iloc[0]) ** (1 / years) - 1
        bh_ann = (bh_pv.iloc[-1] / bh_pv.iloc[0]) ** (1 / years) - 1
        base_bm_ann = (base_bm_pv.iloc[-1] / base_bm_pv.iloc[0]) ** (1 / years) - 1

        return GridResult(
            grid_pv=grid_pv,
            bh_pv=bh_pv,
            base_benchmark_pv=base_bm_pv,
            final_value=final_value,
            trades=self.trades,
            grid_annual_return=grid_ann,
            bh_annual_return=bh_ann,
            base_benchmark_annual_return=base_bm_ann,
            excess_return=grid_ann - bh_ann,
            grid_excess_vs_base=grid_ann - base_bm_ann,
        )
