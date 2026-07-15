# 可复现性清单 (Reproducibility Manifest)

本文件记录所有不入 Git 的大型数据缓存的生成命令、预期大小、校验和与行数，
确保任何人从 fresh clone 即可完整复现回测结果。

## 1. 存活股未复权日线缓存 (live_daily_cache)

| 属性 | 值 |
|------|-----|
| 目录 | `data/live_daily_cache/` |
| 格式 | Parquet（每只股票一个文件 + `_meta.json`） |
| 股票数 | 4590 只 A 股 |
| 覆盖率 | 100%（0 失败、0 空） |
| 预期大小 | ~328 MB |
| 文件数 | 4590 个 parquet + 1 个 `_meta.json` |
| 生成命令 | `python scripts/build_live_daily_cache.py --workers 10` |
| 构建时间 | 约 10-15 分钟（取决于网络与并发数） |
| 增量构建 | 支持，已缓存的股票自动跳过（`--no-skip-existing` 可强制重建） |
| `_meta.json` SHA-256 | `43cfb9795316e9a0e9ada662a62abb64012105669193bdb6db983312aaa9822c` |

校验命令：
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

## 2. 前复权收盘价缓存 (qfq_cache)

| 属性 | 值 |
|------|-----|
| 目录 | `data/qfq_cache/` |
| 格式 | Parquet（每只股票一个文件 + `_meta.json`） |
| 股票数 | 4590 只存活 A 股 + 199 只退市股（退市股在 `delist_prices.pkl`） |
| 覆盖率 | 100%（0 失败、0 降级） |
| 预期大小 | ~325 MB |
| 文件数 | 4590 个 parquet + 1 个 `_meta.json` |
| 生成命令 | `python scripts/build_qfq_cache.py --workers 10` |
| 构建时间 | 约 10-15 分钟（取决于网络与并发数） |
| 增量构建 | 支持，已缓存的股票自动跳过 |
| `_meta.json` SHA-256 | `714a96f97a738bc444d85d85691bd5dc4c39be883d7460a31650dd813d3f65be` |

校验命令：
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

## 3. 退市股价格 (delist_prices.pkl)

| 属性 | 值 |
|------|-----|
| 文件 | `data/delist_prices.pkl` + `data/delist_info.json` |
| 来源 | 腾讯K线接口（前复权） |
| 股票数 | 199 只（上交所 78 + 深交所 121） |
| 预期大小 | ~2.5 MB |
| 生成命令 | `python scripts/fetch_delist_tx.py && python scripts/fetch_delist_sz.py` |

## 4. 其他缓存文件

| 文件 | 大小 | 说明 | 生成方式 |
|------|------|------|---------|
| `data/universe_cache.pkl` | ~512 MB | 全市场股票池缓存（含流通股本等） | 运行 `small_cap_v2.py` 时自动生成 |
| `data/book_equity_all.pkl` | ~1.2 MB | 全市场资产负债表缓存 | 运行 `fama_french_v2.py` 时自动生成 |
| `data/qfq_cache.pkl` | ~32 KB | 旧版 qfq 缓存（pickle，已弃用） | 旧版 `build_qfq_cache.py` 生成，不再使用 |

## 5. 回测结果复现

### 完整复现流程

```bash
# Step 1: 构建存活股日线缓存（约 328 MB，10-15 分钟）
python scripts/build_live_daily_cache.py --workers 10

# Step 2: 构建前复权缓存（约 325 MB，10-15 分钟）
python scripts/build_qfq_cache.py --workers 10

# Step 3: 运行小市值回测（自动读取缓存，输出到 results/small_cap_v2_<timestamp>/）
python scripts/small_cap_v2.py

# Step 4: 验证结果可复现性（从 parquet 独立重算 CAGR/Sharpe/MaxDD）
python scripts/verify_reproducibility.py results/small_cap_v2_<timestamp>/

# Step 5: 全量审计（验证 JSON 内部一致性、基准对比、文章引用数字溯源）
python scripts/verify_small_cap_v2_results.py results/small_cap_v2_<timestamp>/
```

### 最新结果校验

| 文件 | SHA-256 | 大小 |
|------|---------|:---:|
| `results/small_cap_v2_20260715_011613/small_cap.json` | `d7716c3b1e882fb47bfe5a3deadac8a29f00e0ecb632d1797132bce8cf0b975c` | 456 KB |
| `results/small_cap_v2_20260715_011613/return_series.parquet` | `b10b8d8af3fce0506d60d38de8ea7d66778acb07830af8a43cad247b40e593e3` | 24 MB |

### 验证结果摘要

- `verify_reproducibility.py`: 全部 12 个场景 PASS，独立重算 CAGR 与 JSON 精确匹配（Δ=0.00e+00）
- `verify_small_cap_v2_results.py` 审计报告: `results/small_cap_v2_20260715_011613/verification_report.md`
  - qfq 覆盖率: 100% (4590/4590)
  - parquet 独立重算: 66/66 个月全部 PASS（无 unsliceable）
  - 文章引用数字溯源: 全部 PASS
  - 基准对比: PASS（沪深300 CAGR=2.51%）

### 旧结果（已弃用）

| 目录 | 说明 |
|------|------|
| `results/small_cap_v2_20260714_232258/` | parquet 仅有片段数据（24点/股），独立重算失败，已被 011613 替代 |
| `results/small_cap_v2_20260714_200852/` | 修复前版本 |
| `results/small_cap_v2_20260713_194952/` | 最初版本 |
| `results/20260714_220329/` | FF 修复前版本（raw close 持有期、法定4/30公告日、重复6月行） |
| `results/small_cap_v2_20260714_224909/` | FF 中间版本（部分修复） |

## 6. Fama-French 三因子结果复现

### 完整复现流程

```bash
# 前置：需要已构建 live_daily_cache 和 qfq_cache（见上方 Step 1-2）

# 运行 FF 三因子构建（输出到 results/<timestamp>/）
python scripts/fama_french_v2.py --start 2020 --end 2024

# 运行测试
python -m pytest tests/test_fama_french.py -v
```

### 最新结果校验

| 文件 | SHA-256 | 大小 |
|------|---------|:---:|
| `results/20260715_173639/fama_french.json` | (见文件) | 5 KB |
| `results/20260715_173639/fama_french.csv` | (见文件) | 4.5 KB |

### 验证结果摘要

- 16/16 测试全部 PASS（含 raw-vs-qfq 路径测试、非交易日形成日测试、空组合 NaN 测试）
- CSV 60 个月因子收益，0 个 all-zero 月，0 个 NaN 月
- 公告日来源: actual（6044 records across 2677 stocks downgraded to statutory）
- 持有期收益: 前复权价格（qfq），市值排序: 未复权收盘价 × 流通股本
