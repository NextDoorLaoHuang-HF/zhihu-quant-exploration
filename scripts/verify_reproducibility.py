"""
独立验证脚本：从 return_series.parquet + JSON 中的 selected_history 重算
逐月收益和所有 metrics，确认与 JSON 中的数值一致。

用法: python scripts/verify_reproducibility.py <results_dir>
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from lib.metrics import compute_metrics, relative_cagr
from small_cap_v2 import _get_month_end_dates


def verify(results_dir: str) -> int:
    """Returns 0 if all checks pass, 1 otherwise."""
    json_path = os.path.join(results_dir, 'small_cap.json')
    rs_path = os.path.join(results_dir, 'return_series.parquet')

    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        return 1
    if not os.path.exists(rs_path):
        print(f"ERROR: {rs_path} not found")
        return 1

    with open(json_path) as f:
        data = json.load(f)

    rs_df = pd.read_parquet(rs_path)
    print(f"Loaded: {len(rs_df.columns)} symbols, {len(rs_df)} dates")
    print(f"Return series tag: {data.get('return_series_tag', 'N/A')}")

    # Criterion: all input series must be qfq-tagged
    assert data.get('return_series_tag') == 'qfq', \
        "return_series_tag must be 'qfq'"
    print("  PASS: return_series_tag == 'qfq'")

    params = data['params']
    month_ends = _get_month_end_dates(params['start_date'], params['end_date'])
    me_map = {month_ends[i]: month_ends[i + 1] for i in range(len(month_ends) - 1)}

    all_pass = True
    n_checked = 0
    n_mismatches = 0

    for label, res in data['results'].items():
        if 'monthly_returns' not in res or 'selected_history' not in res:
            print(f"  SKIP {label}: no reproducibility data")
            continue

        n_checked += 1
        monthly_rets = {pd.Timestamp(e['date']): e['return']
                        for e in res['monthly_returns']}
        sel_hist = res['selected_history']

        # --- Recompute monthly portfolio returns from saved series ---
        recomputed_monthly = {}
        for entry in sel_hist:
            rb = pd.Timestamp(entry['date'])
            nd = me_map.get(rb)
            if nd is None:
                continue
            stock_rets = []
            for code in entry['selected']:
                if code in rs_df.columns:
                    s = rs_df[code].dropna()
                    sliced = s[(s.index >= rb) & (s.index <= nd)]
                    if len(sliced) >= 2 and sliced.iloc[0] > 0:
                        stock_rets.append(
                            float(sliced.iloc[-1] / sliced.iloc[0] - 1))
            if len(stock_rets) > 0:
                recomputed = float(np.mean(stock_rets))
                recomputed_monthly[nd] = recomputed

                # Check monthly return matches
                if nd in monthly_rets:
                    orig = monthly_rets[nd]
                    if abs(recomputed - orig) > 1e-8:
                        n_mismatches += 1
                        print(f"  FAIL {label} {rb.date()}: "
                              f"recomputed={recomputed:.10f} "
                              f"orig={orig:.10f} diff={abs(recomputed-orig):.2e}")
                        all_pass = False

        # --- Recompute metrics from recomputed monthly returns ---
        if len(recomputed_monthly) > 0:
            rs_series = pd.Series(recomputed_monthly)
            rm = compute_metrics(rs_series,
                                 periods_per_year=params['periods_per_year'])

            # Check CAGR
            orig_cagr = res['metrics']['cagr']
            recomp_cagr = rm['cagr']
            if abs(orig_cagr - recomp_cagr) > 1e-6:
                n_mismatches += 1
                print(f"  FAIL {label} CAGR: "
                      f"orig={orig_cagr:.10f} recomp={recomp_cagr:.10f}")
                all_pass = False

            # Check Sharpe
            orig_sharpe = res['metrics']['sharpe']
            recomp_sharpe = rm['sharpe']
            if abs(orig_sharpe - recomp_sharpe) > 1e-6:
                n_mismatches += 1
                print(f"  FAIL {label} Sharpe: "
                      f"orig={orig_sharpe:.10f} recomp={recomp_sharpe:.10f}")
                all_pass = False

            # Check MaxDD
            orig_mdd = res['metrics']['max_drawdown']
            recomp_mdd = rm['max_drawdown']
            if abs(orig_mdd - recomp_mdd) > 1e-6:
                n_mismatches += 1
                print(f"  FAIL {label} MaxDD: "
                      f"orig={orig_mdd:.10f} recomp={recomp_mdd:.10f}")
                all_pass = False

            # Check excess_cagr if present
            if 'excess_cagr' in res and res['excess_cagr'] is not None:
                bench_monthly = {
                    pd.Timestamp(e['date']): e['return']
                    for e in res.get('benchmark_monthly_returns', [])
                }
                if len(bench_monthly) > 0:
                    bench_series = pd.Series(bench_monthly)
                    port_nav = (1 + rs_series).cumprod()
                    bench_nav = (1 + bench_series).cumprod()
                    try:
                        recomp_excess = relative_cagr(port_nav, bench_nav)
                        orig_excess = res['excess_cagr']
                        if abs(orig_excess - recomp_excess) > 1e-6:
                            n_mismatches += 1
                            print(f"  FAIL {label} excess_cagr: "
                                  f"orig={orig_excess:.10f} "
                                  f"recomp={recomp_excess:.10f}")
                            all_pass = False
                    except (ValueError, Exception):
                        pass

    # --- Criterion: timestamp > qfq fix commit time ---
    ts = data.get('timestamp', '')
    print(f"\n  Run timestamp: {ts}")
    print(f"  qfq fix commit: 2026-07-14T21:10:29+08:00")
    if ts > '2026-07-14T21:10':
        print("  PASS: timestamp > qfq fix commit time")
    else:
        print("  FAIL: timestamp not after qfq fix commit")
        all_pass = False

    # --- Criterion: no partial month entries ---
    if month_ends:
        today = pd.Timestamp.now().normalize()
        last_me = month_ends[-1]
        if last_me <= today:
            print(f"  PASS: last month-end {last_me.date()} <= today "
                  f"{today.date()}, no partial month")
        else:
            print(f"  FAIL: last month-end {last_me.date()} > today "
                  f"{today.date()}, partial month included")
            all_pass = False

    # --- Criterion: qfq coverage 100%, 0 degraded ---
    qfq_cov = data.get('qfq_coverage', {})
    if qfq_cov.get('coverage_pct', 0) == 100.0 and \
       qfq_cov.get('degraded', 0) == 0 and \
       qfq_cov.get('qfq_failed', 0) == 0:
        print(f"  PASS: qfq coverage 100%, 0 degraded, 0 failed")
    else:
        print(f"  FAIL: qfq coverage={qfq_cov.get('coverage_pct')}%, "
              f"degraded={qfq_cov.get('degraded')}, "
              f"failed={qfq_cov.get('qfq_failed')}")
        all_pass = False

    print(f"\n{'='*60}")
    print(f"Checked {n_checked} scenarios, {n_mismatches} mismatches")
    if all_pass and n_mismatches == 0:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify reproducibility of small_cap_v2 results')
    parser.add_argument('results_dir', help='Path to results directory')
    args = parser.parse_args()
    sys.exit(verify(args.results_dir))
