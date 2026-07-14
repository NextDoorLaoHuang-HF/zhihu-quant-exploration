"""
TDD tests for scripts/lib/grid_engine.py
Each test targets a specific bug documented in GitHub Issue #1.
Run: pytest tests/test_grid_engine.py -v
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Make scripts/lib importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'lib'))
from grid_engine import GridConfig, GridEngine, GridResult


# ============================================================
# Helper: build a simple close-price series
# ============================================================
def make_prices(prices, dates=None):
    """Build a pd.Series of close prices with a DatetimeIndex."""
    if dates is None:
        dates = pd.date_range('2020-01-01', periods=len(prices), freq='B')
    return pd.Series(prices, index=dates, dtype=float)


# ============================================================
# Test 1: Selling should not multiply cash by price
# Issue 6: verify_grid_walkforward.py:27-28
#   shares -= gc * ch / pr        → shares reduce by gc*ch/pr
#   cash += gc * ch * pr * (1-cost) → cash increase = gc*ch*pr (WRONG!)
# Correct: cash should increase by grid_capital minus fee,
#          NOT grid_capital * price.
# ============================================================
def test_sell_cash_not_multiplied_by_price():
    """When selling one grid, cash increases by ~grid_capital, not grid_capital*price."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.5,
        grid_capital=5000.0,
        max_grids=10,
        commission_rate=0.0,   # zero commission for clarity
        min_commission=0.0,
        contract_size=1,       # no lot rounding — test pure math
    )
    engine = GridEngine(config, initial_capital=100000)

    # Day 0: close=10, signal=grid_position(10)=0
    # Day 1: close=10.5, execute yesterday's signal (0→0, no trade), new signal=grid_position(10.5)=1
    # Day 2: close=10.5, execute yesterday's signal (0→1, SELL 1 grid at trigger price 10.5)
    #   shares_sold = 5000 / 10.5 ≈ 476.19
    #   cash_increase = 476.19 * 10.5 - 0 = 5000 (NOT 5000*10.5=52500)
    prices = make_prices([10.0, 10.5, 10.5])
    result = engine.run(prices)

    cash_after = engine.cash
    # Initial cash = 100000 * (1-0.5) = 50000
    # After selling 1 grid: cash = 50000 + 5000 = 55000
    # If bug: cash = 50000 + 5000*10.5 = 102500
    assert 54000 <= cash_after <= 56000, \
        f"Sell cash bug: cash={cash_after}, expected ~55000 (not ~102500)"


# ============================================================
# Test 2: Zero-cost roundtrip — buy then sell at same price
# ============================================================
def test_zero_cost_roundtrip():
    """Buy one grid then sell one grid at same price, with zero commission → total assets unchanged."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.5,
        grid_capital=5000.0,
        max_grids=10,
        commission_rate=0.0,
        min_commission=0.0,
        contract_size=1,
    )
    engine = GridEngine(config, initial_capital=100000)

    # Day 0: close=10, signal=0
    # Day 1: close=9.524 (≈10/1.05), execute sig 0→0 (no trade), signal=-1 (BUY)
    # Day 2: close=10.0, execute sig 0→-1 (BUY 1 grid at trigger 9.524), signal=0
    # Day 3: close=10.0, execute sig -1→0 (SELL 1 grid at trigger 10.0)
    prices = make_prices([10.0, 10.0 / 1.05, 10.0, 10.0])
    result = engine.run(prices)

    # After roundtrip with zero cost, total assets should reflect grid profit.
    # Buy at trigger 9.524 (=10/1.05), sell at trigger 10.0:
    #   buy 5000/9.524 = 525 shares, sell 525*10.0 = 5250 → profit = 250
    # Grid profit = grid_capital * grid_pct = 5000 * 0.05 = 250
    expected_profit = config.grid_capital * config.grid_pct
    assert abs(result.final_value - (100000 + expected_profit)) < 10, \
        f"Zero-cost roundtrip: final_value={result.final_value}, expected ~{100000 + expected_profit}"


# ============================================================
# Test 3: Insufficient cash → no trade, current_grid unchanged
# ============================================================
def test_insufficient_cash_no_trade():
    """If cash < grid_capital, buy should not execute and current_grid should not advance."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.99,   # nearly all-in: cash = 1000 < 5000
        grid_capital=5000.0,
        max_grids=10,
        commission_rate=0.0,
        min_commission=0.0,
        contract_size=1,
    )
    engine = GridEngine(config, initial_capital=100000)

    # Day 0: close=10, signal=0
    # Day 1: close=9.524, execute 0→0 (no trade), signal=-1 (BUY)
    # Day 2: close=9.524, execute 0→-1 (BUY but cash=1000 < 5000 → FAIL)
    #        current_grid stays 0
    prices = make_prices([10.0, 10.0 / 1.05, 10.0 / 1.05])
    result = engine.run(prices)

    assert engine.current_grid == 0, \
        f"Insufficient cash: current_grid={engine.current_grid}, expected 0 (no advancement)"


# ============================================================
# Test 4: Insufficient shares → no trade
# ============================================================
def test_insufficient_shares_no_trade():
    """If shares < grid_capital/price, sell should not execute and current_grid should not advance."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.01,   # tiny base: shares = 100000*0.01/10 = 100
        grid_capital=5000.0,  # need 5000/10 = 500 shares to sell → not enough
        max_grids=10,
        commission_rate=0.0,
        min_commission=0.0,
        contract_size=1,
    )
    engine = GridEngine(config, initial_capital=100000)

    # Day 0: close=10, signal=0
    # Day 1: close=10.5, execute 0→0, signal=1 (SELL)
    # Day 2: close=10.5, execute 0→1 (SELL but shares=100 < 500 → FAIL)
    prices = make_prices([10.0, 10.5, 10.5])
    result = engine.run(prices)

    assert engine.current_grid == 0, \
        f"Insufficient shares: current_grid={engine.current_grid}, expected 0 (no advancement)"


# ============================================================
# Test 5: Minimum commission — small trades still cost >= 5 yuan
# ============================================================
def test_min_commission():
    """Trades smaller than min_commission threshold should still incur at least min_commission fee."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.5,
        grid_capital=100.0,   # small grid → commission would be 100*0.00025 = 0.025 < 5
        max_grids=10,
        commission_rate=0.00025,
        min_commission=5.0,
        contract_size=1,
    )
    engine = GridEngine(config, initial_capital=100000)

    # Day 0: close=10, signal=0
    # Day 1: close=10.5, execute 0→0, signal=1 (SELL)
    # Day 2: close=10.5, execute 0→1 (SELL 1 grid at 10.5)
    #   proceeds = 100/10.5 * 10.5 = 100
    #   fee = max(5, 100*0.00025) = max(5, 0.025) = 5.0
    #   cash += 100 - 5 = 95
    prices = make_prices([10.0, 10.5, 10.5])
    result = engine.run(prices)

    cash_after = engine.cash
    # Initial cash = 100000 * (1-0.5) = 50000
    # After sell: cash = 50000 + 100 - 5 = 50095
    expected_with_min = 50000 + 100 - 5.0     # 50095
    expected_without_min = 50000 + 100 - 0.025  # 50099.975

    assert abs(cash_after - expected_with_min) < 1, \
        f"Min commission: cash={cash_after}, expected ~{expected_with_min} (not ~{expected_without_min:.3f})"


# ============================================================
# Test 6: Multi-grid intraday — 3 grids triggered, each at own price
# ============================================================
def test_multi_grid_intraday():
    """If price moves through 3 grid levels, each grid trades at its own trigger price."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.5,
        grid_capital=5000.0,
        max_grids=10,
        commission_rate=0.0,
        min_commission=0.0,
        contract_size=1,
    )
    engine = GridEngine(config, initial_capital=100000)

    # Day 0: close=10, signal=0
    # Day 1: close=10*1.05^3=11.576, execute 0→0 (no trade), signal=3
    # Day 2: close=11.576, execute 0→3 (SELL 3 grids)
    #   Each grid sells at its own trigger: 10.5, 11.025, 11.576
    #   NOT all at 11.576
    p0 = 10.0
    p3 = p0 * (1.05 ** 3)  # ≈ 11.576
    prices = make_prices([p0, p3, p3])
    result = engine.run(prices)

    # current_grid should be 3 (all sells executed)
    assert engine.current_grid == 3, \
        f"Multi-grid intraday: current_grid={engine.current_grid}, expected 3"

    # Cash: 3 sells, each worth grid_capital=5000 → cash += 15000
    # (each sell: shares_sold = 5000/trigger_price, cash += shares_sold * trigger_price = 5000)
    cash_after = engine.cash
    expected_cash = 50000 + 15000  # initial + 3 * grid_capital
    assert abs(cash_after - expected_cash) < 50, \
        f"Multi-grid cash: cash={cash_after}, expected ~{expected_cash}"


# ============================================================
# Test 7: Next-day execution — signal on day T, execution on T+1
# ============================================================
def test_next_day_execution():
    """Signal generated on day T (close), execution happens on T+1."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.5,
        grid_capital=5000.0,
        max_grids=10,
        commission_rate=0.0,
        min_commission=0.0,
        contract_size=1,
    )

    # Case A: signal on day 1, execution on day 2 at day 2's price
    # Day 0: close=10, signal=0
    # Day 1: close=10.5, execute 0→0 (no trade), signal=1 (SELL)
    # Day 2: close=11.0, execute 0→1 (SELL at trigger 10.5)
    engine = GridEngine(config, initial_capital=100000)
    prices = make_prices([10.0, 10.5, 11.0])
    result = engine.run(prices)

    assert engine.current_grid == 1, \
        f"Next-day execution: current_grid={engine.current_grid}, expected 1"

    # The sell executes at the trigger price 10.5 (= grid_base * 1.05^1)
    # not at day 1's close (10.5) — wait, the trigger IS 10.5 here.
    # To distinguish same-day vs next-day, we check whether the sell
    # happened at all (it should only happen on day 2, not day 1).
    # With only 2 days (same-day bug), the sell would execute on day 1.
    # With 3 days (next-day), it executes on day 2.
    # We verify by checking shares after just 2 days (no execution yet):
    engine_check = GridEngine(config, initial_capital=100000)
    prices_check = make_prices([10.0, 10.5])
    result_check = engine_check.run(prices_check)
    # With next-day: no trade executed yet (signal pending)
    assert engine_check.current_grid == 0, \
        f"Next-day: after 2 days, current_grid={engine_check.current_grid}, expected 0 (signal pending)"
    assert engine_check.trades == 0, \
        f"Next-day: after 2 days, trades={engine_check.trades}, expected 0 (no execution yet)"

    # Case B: verify execution uses trigger price, not day 2's close
    # Day 0: close=10, signal=0
    # Day 1: close=10.5, signal=1
    # Day 2: close=12.0, execute 0→1 (SELL at trigger 10.5)
    #   shares_sold = 5000/10.5 ≈ 476.19
    #   If bug (sell at close 12.0): shares_sold = 5000/12.0 ≈ 416.67
    engine2 = GridEngine(config, initial_capital=100000)
    prices2 = make_prices([10.0, 10.5, 12.0])
    result2 = engine2.run(prices2)

    initial_shares = 100000 * 0.5 / 10.0  # 5000
    expected_shares_trigger = initial_shares - 5000 / 10.5  # ≈ 4523.81
    expected_shares_close = initial_shares - 5000 / 12.0   # ≈ 4583.33

    assert abs(engine2.shares - expected_shares_trigger) < 10, \
        f"Next-day execution: shares={engine2.shares}, expected ~{expected_shares_trigger:.2f} "
    f"(trigger price, not ~{expected_shares_close:.2f} at close)"


# ============================================================
# Test 8: Walk-forward continuity — carry over capital and shares
# ============================================================
def test_walkforward_continuity():
    """When running walk-forward, capital and positions should carry over between periods."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.5,
        grid_capital=5000.0,
        max_grids=10,
        commission_rate=0.0,
        min_commission=0.0,
        contract_size=1,
    )

    # Period 1: prices go up (trigger sell)
    prices1 = make_prices([10.0, 10.5, 10.5],
                          dates=pd.date_range('2020-01-01', periods=3, freq='B'))
    engine = GridEngine(config, initial_capital=100000)
    result1 = engine.run(prices1)

    # After period 1: should have sold 1 grid
    # cash > 50000, shares < 5000 — state has changed
    assert engine.cash > 50000, "After sell, cash should be > initial"
    assert engine.shares < 5000, "After sell, shares should be < initial"
    assert engine.current_grid == 1, f"After sell, current_grid={engine.current_grid}, expected 1"

    # Period 2: create new engine, carry over state
    engine2 = GridEngine(config, initial_capital=result1.final_value)
    engine2.cash = engine.cash
    engine2.shares = engine.shares
    engine2.current_grid = engine.current_grid
    engine2.grid_base = engine.grid_base
    engine2._initialized = True

    prices2 = make_prices([10.5, 10.0, 10.0],
                          dates=pd.date_range('2020-01-06', periods=3, freq='B'))
    result2 = engine2.run(prices2)

    # Key: engine2 should NOT reset to 100k initial state
    # It should start from engine's end state
    # If it reset (bug), it would have 50000 cash and 5000 shares
    assert engine2.cash != 50000 or engine2.shares != 5000, \
        "Walk-forward: engine should carry over state, not reset to initial 100k"

    # The grid_base should also carry over (not reset to period 2's first price)
    assert engine2.grid_base == 10.0, \
        f"Walk-forward: grid_base={engine2.grid_base}, expected 10.0 (carried over)"


# ============================================================
# Test 9: Base-position benchmark — engine outputs 3 benchmarks
# ============================================================
def test_base_position_benchmark():
    """Engine result should include base_position + cash benchmark to isolate grid contribution."""
    config = GridConfig(
        grid_pct=0.05,
        base_position=0.6,
        grid_capital=5000.0,
        max_grids=10,
        commission_rate=0.0,
        min_commission=0.0,
        contract_size=1,
    )
    engine = GridEngine(config, initial_capital=100000)

    prices = make_prices([10.0, 9.5, 10.5, 10.0])
    result = engine.run(prices)

    # Result must contain all 3 benchmarks
    assert hasattr(result, 'grid_pv'), "Result must have grid_pv"
    assert hasattr(result, 'bh_pv'), "Result must have bh_pv (100% buy & hold)"
    assert hasattr(result, 'base_benchmark_pv'), \
        "Result must have base_benchmark_pv (base position + cash, isolating grid contribution)"

    # Base benchmark: 60% buy & hold + 40% cash
    # Initial: 60000 in stock at 10 → 6000 shares, 40000 cash
    # Final price = 10.0 → base_benchmark = 60000 + 40000 = 100000
    expected_base_bm = 100000 * config.base_position * (prices.iloc[-1] / prices.iloc[0]) \
        + 100000 * (1 - config.base_position)
    assert abs(result.base_benchmark_pv.iloc[-1] - expected_base_bm) < 100, \
        f"Base benchmark: {result.base_benchmark_pv.iloc[-1]}, expected ~{expected_base_bm}"

    # All PV series should have same length as price input
    assert len(result.grid_pv) == len(prices), \
        f"Grid PV length {len(result.grid_pv)} != {len(prices)}"
    assert len(result.bh_pv) == len(prices), \
        f"BH PV length {len(result.bh_pv)} != {len(prices)}"
    assert len(result.base_benchmark_pv) == len(prices), \
        f"Base benchmark PV length {len(result.base_benchmark_pv)} != {len(prices)}"
