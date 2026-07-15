# 数据缓存清单

本文件说明项目中所有大型数据缓存的生成方式和校验方法。
这些缓存文件不入 Git（见 .gitignore），但可通过以下命令完全复现。

## 1. 存活股日线缓存（raw close + outstanding_share）

- 目录: `data/live_daily_cache/`
- 格式: Parquet（每只股票一个文件 + _meta.json）
- 股票数: 4590 只 A 股
- 覆盖率: 100%（0 失败、0 空）
- 大小: ~328MB
- 生成命令:
  ```bash
  python scripts/build_live_daily_cache.py --workers 10
  ```
- 校验:
  ```bash
  python -c "
  import json
  with open('data/live_daily_cache/_meta.json') as f:
      m = json.load(f)
  assert len(m['success']) == 4590
  assert len(m['fail']) == 0
  print('live_daily_cache OK:', len(m['success']), 'stocks')
  "
  ```

## 2. 前复权收盘价缓存（qfq close）

- 目录: `data/qfq_cache/`
- 格式: Parquet（每只股票一个文件 + _meta.json）
- 股票数: 4590 只 A 股 + 199 只退市股（退市股在 delist_prices.pkl 中）
- 覆盖率: 100%（0 失败、0 降级）
- 大小: ~325MB
- 生成命令:
  ```bash
  python scripts/build_qfq_cache.py --workers 10
  ```
- 校验:
  ```bash
  python -c "
  import json
  with open('data/qfq_cache/_meta.json') as f:
      m = json.load(f)
  assert len(m['success']) == 4590
  assert len(m['fail']) == 0
  assert m.get('qfq_coverage_pct') == 100.0
  print('qfq_cache OK:', len(m['success']), 'stocks,', m.get('delist_count'), 'delist')
  "
  ```

## 3. 退市股价格（delist_prices.pkl）

- 文件: `data/delist_prices.pkl` + `data/delist_info.json`
- 来源: 腾讯K线接口（前复权）
- 股票数: 199 只（上交所 78 + 深交所 121）
- 生成命令:
  ```bash
  python scripts/fetch_delist_tx.py  # 上交所退市股
  python scripts/fetch_delist_sz.py  # 深交所退市股
  ```

## 4. 回测结果

- 最新结果: `results/small_cap_v2_20260715_011613/small_cap.json`（生成时间 2026-07-15 01:27，含 `return_series.parquet`）
- 旧结果（暂停引用）: `results/small_cap_v2_20260714_232258/`、`results/small_cap_v2_20260714_200852/` 和 `results/small_cap_v2_20260713_194952/`
- 生成命令:
  ```bash
  # 先构建缓存（约15分钟）
  python scripts/build_live_daily_cache.py --workers 10
  python scripts/build_qfq_cache.py --workers 10
  # 再运行回测（约15分钟）
  python scripts/small_cap_v2.py
  ```

## 校验摘要

| 缓存 | 股票数 | 成功率 | 降级数 | 生成时间 |
|------|--------|--------|--------|---------|
| live_daily_cache | 4590 | 100% | N/A | 2026-07-14 19:13 |
| qfq_cache | 4590 | 100% | 0 | 2026-07-14 19:17 |
| delist_prices | 199 | N/A | N/A | 2026-07-13 |
