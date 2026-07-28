# 知乎「个人做量化交易是否可行」— 策略回测复现

本项目对知乎问题[「个人做量化交易是否可行呢？」](https://www.zhihu.com/question/529408913)下50个高赞回答中提到的量化策略进行了逐条回测验证。

## 核心结论

**散户真正的alpha不在择时，在去机构去不了的地方。**

| 策略方向 | 年化 | 夏普 | 超额(vs全样本) | 状态 | ⚠️暂停引用 |
|---------|:---:|:---:|:---:|:---:|:---:|
| T5极端微盘 | 23.9% | 0.81 | +14.5% | ✅ 最优 | ⚠️暂停引用 |
| T10微盘 | 22.2% | 0.88 | +12.8% | ✅ | ⚠️暂停引用 |
| T20小盘 | 14.9% | 0.61 | +5.5% | ✅ | ⚠️暂停引用 |
| 高股息填权 | -2.3%超额 | — | 填权率46% | ❌ | ⚠️暂停引用 |
| 双均线(5/20) | — | 0.38 | 胜率17% | ❌ | |
| 趋势跟踪(MA200) | 7.0% | 0.73 | -5.7% | ❌ | |
| 网格交易 | — | — | 72组中21组赢 | ⚠️ | ⚠️暂停引用 |
| ST股 | 4.7% | 0.14 | -4.7% | ❌ | |
| 低成交量冷门股 | 6.9% | 0.24 | -2.5% | ❌ | |

> **⚠️ 暂停引用说明**：标记 ⚠️暂停引用 的数字存在根本性问题（见 [Issue #1](https://github.com/NextDoorLaoHuang-HF/zhihu-quant-exploration/issues/1)），系统性修复完成前请勿直接引用。受影响项目：T5/T10/T20 收益（选股用最低股价而非最小市值）、19.0% "真实可执行"结论（仍含幸存者偏差）、SMB 21.2%/HML 7.1%（非标准 Fama-French 因子）、填权 -2.3% 超额（股票池事后选择+市场基准用错）、网格胜率/参数稳定性/成本（卖出记账错误+多相邻问题）、CB/HRP 最优配比（可转债收益模拟生成，HRP 全样本逆波动率）。

详见 [article.md](article.md)。

## 目录结构

```
zhihu-quant-exploration/
├── README.md
├── article.md                         # 文章正文
├── requirements.txt
├── scripts/
│   ├── 01_grid_etf_premium.py         # 网格交易 + ETF折溢价 + CB/HRP混合
│   ├── 02_conventional_dualma_ff.py   # 双均线 + Fama-French三因子 + 趋势跟踪
│   ├── 03_retail_edge_microcap.py     # T5极端微盘 + ST/冷门/退市 + 策略轮动
│   ├── 04_div_fill_rights.py          # 高股息填权
│   ├── fetch_delist_tx.py             # 获取上交所退市股日线数据（腾讯API前复权）
│   ├── fetch_delist_sz.py             # 获取深交所退市股日线数据
│   ├── rerun_microcap_v2.py           # 含退市股重跑T5/T10/T20回测
│   ├── rerun_fixed.py                 # 修复版回测（pct_change+ST过滤+退市处理）
│   ├── small_cap_v2.py               # 真实流通市值排序+全市场动态池+CAGR
│   ├── build_live_daily_cache.py      # 并行构建存活股日线缓存（parquet格式）
│   ├── build_qfq_cache.py             # 并行构建前复权缓存（parquet格式）
│   ├── fama_french_v2.py              # 真实Fama-French三因子（批量资产负债表）
│   ├── dividend_v2.py                 # 分红事件处理v2
│   ├── fetch_cb_data.py               # 可转债数据拉取
│   ├── grid_v2.py                     # 网格交易v2（walk-forward）
│   ├── run_hybrid_v2.py               # CB/HRP混合策略v2
│   ├── verify_reproducibility.py      # 从parquet独立重算CAGR/Sharpe验证
│   ├── verify_small_cap_v2_results.py # 小市值v2全量审计脚本
│   ├── lib/universe.py               # 点时股票池构建器
│   ├── lib/metrics.py                # 绩效指标计算
│   ├── lib/qfq_cache.py              # 前复权缓存模块（覆盖率追踪+降级保护）
│   ├── lib/fama_french.py            # Fama-French因子计算
│   └── verify_issues.py               # 验证幸存者偏差+网格交易矛盾
├── data/
│   ├── delist_prices.pkl              # 退市股前复权日线（腾讯API）
│   ├── delist_info.json               # 退市股元信息
│   ├── live_daily_cache/              # 存活股日线缓存（parquet，4590只）
│   ├── qfq_cache/                     # 前复权收盘价缓存（parquet，4590只）
│   └── backtest_fixed_all.json        # 修复后回测结果
├── results/
│   └── small_cap_v2_20260715_011613/ # 最新回测结果（qfq修复后，2026-07-15 01:27 生成）
│       ├── small_cap.json
│       ├── return_series.parquet
│       └── verification_report.md
├── tests/                            # 测试套件（96 tests）
│   ├── test_small_cap.py             # 小市值回测测试
│   ├── test_raw_close_guard.py       # 原始收盘价降级守卫+变异测试
│   ├── test_qfq_cache.py             # 前复权缓存单元测试
│   ├── test_qfq_integration.py      # 前复权集成测试
│   ├── test_universe.py              # 股票池构建测试
│   ├── test_metrics.py               # 绩效指标测试
│   ├── test_fama_french.py           # Fama-French因子测试
│   ├── test_hybrid.py                # CB/HRP混合策略测试
│   ├── test_grid_engine.py          # 网格引擎测试
│   └── test_dividend_event.py       # 分红事件测试
├── charts/                            # 回测结果图表
│   ├── chart1_overview.png            # 策略总览表
│   ├── chart2_heatmap.png             # 双均线夏普热力图
│   ├── chart3_smallcap.png            # 小市值vs基准
│   ├── chart4_trend.png               # 趋势跟踪vs买持
│   ├── chart5_ff3factor.png           # FF三因子
│   ├── chart6_quadrant.png            # 四象限总结
│   └── retail_edge_strategies.png     # T5微盘策略四合图
└── html/
    └── zhihu_article_charts.html      # 图表HTML源文件
```

## 环境要求

- Python 3.9+
- macOS / Linux（Windows 未经测试）

```bash
pip install -r requirements.txt
```

## 数据源说明

本项目使用两个数据源：

| 数据源 | API | 可用性 | 说明 |
|--------|-----|--------|------|
| **新浪** | `ak.fund_etf_hist_sina()` / `ak.stock_zh_a_daily()` | ✅ 直连 | ETF和个股日线，无需代理。**ETF不复权，脚本已自动检测>40%跳跃并前复权校准。** |
| **东财** | `ak.stock_history_dividend_detail()` / `ak.stock_zh_a_hist()` | ⚠️ 需代理 | 分红数据和前复权日线。中国电信网络下被屏蔽，需设置代理。 |

### 代理设置

如果在中国电信网络下运行，需要在运行前设置环境变量：

```bash
export HTTP_PROXY=PROXY_PLACEHOLDER
export HTTPS_PROXY=PROXY_PLACEHOLDER
python scripts/04_div_fill_rights.py
```

如果你的代理地址不同，修改对应的 host:port。

前三个脚本（01-03）使用新浪源，无需代理。第四个脚本（分红填权）需要东财源。

## 运行脚本

按顺序运行（每个脚本约需 3-10 分钟，取决于网络速度）：

```bash
cd scripts

# 第一轮：网格交易 + ETF折溢价 + CB/HRP混合
python 01_grid_etf_premium.py

# 第二轮：双均线 + FF三因子 + 趋势跟踪（需个股数据，较慢）
python 02_conventional_dualma_ff.py

# 第三轮：T5极端微盘 + ST/冷门/策略轮动（需个股数据，较慢）
python 03_retail_edge_microcap.py

# 第四轮：高股息填权（需代理 + 东财源）
HTTP_PROXY=PROXY_PLACEHOLDER HTTPS_PROXY=PROXY_PLACEHOLDER python 04_div_fill_rights.py
```

## 复现注意事项

1. **随机种子固定**：所有抽样使用 `np.random.seed(42)` 或 `np.random.seed(888)`，结果可复现。
2. **数据窗口**：回测区间为 2020-01 至 2026-07，基于akshare实时拉取的最新数据。
3. **个股抽样**：受限于 API 频率，每次随机抽样 100-150 只。样本量越大，统计显著性越高。
4. **滑点**：基础回测仅扣佣金（万2.5双边）。文章中有滑点敏感性分析，T5微盘月换手率5.5%，滑点影响约0.4%年化。
5. **幸存者偏差（重要）** ⚠️暂停引用：新浪源 `stock_info_a_code_name()` 不含已退市股票。用腾讯API补充退市股数据（`scripts/fetch_delist_tx.py` 上交所 + `scripts/fetch_delist_sz.py` 深交所，合计199只），按真实比例（150只中6只退市股）重跑（`scripts/rerun_fixed.py`）：T5 年化从 23.9% 降至 -12.9%（含退市无过滤），加 ST过滤+<2元过滤后恢复到 19.0%。幸存者偏差将真实收益高估了约5个百分点。详见 `data/backtest_fixed_all.json`。**注意：19.0% 仍含幸存者偏差，且选股用最低股价而非最小市值，数字暂停引用。**
6. **回测引擎修复** ⚠️暂停引用：三个 bug 修复——①`pct_change(fill_method=None)` 修复退市后收益被pad填充为0%的问题，退市月收益手动设为-100%；②ST过滤改用退市股一律排除（旧版用当前名称做历史过滤存在前视偏差）；③补充深交所退市股（旧版仅有上交所78只）。修复后 T5 含退市+ST+<2元 = 19.0%（旧版为21.9%）。详见 `scripts/rerun_fixed.py` 和 `data/backtest_fixed_all.json`。**注意：修复后的 19.0% 仍存在选股用最低股价而非最小市值的问题，数字暂停引用。**
7. **前复权数据修复（Issue #2）** ⚠️暂停引用：存活股持有期收益从"未复权收盘价"改为"前复权收盘价"（qfq close）。旧版把分红除权日股价下调当成亏损，系统性低估小市值策略收益。为全市场4590只存活股持久化parquet格式qfq缓存（`scripts/build_qfq_cache.py` + `scripts/build_live_daily_cache.py`），qfq覆盖率 100%（4590/4590 缓存成功，0 失败，0 降级）。重跑后 T5 市值排序 CAGR 从 19.82% → 35.68%（+15.86个百分点），T10 从 17.98% → 26.40%，T20 从 11.70% → 20.70%。价格排序策略受影响极小（+0.15%~0.93%）。最新结果：`results/small_cap_v2_20260715_011613/small_cap.json`（生成时间 2026-07-15 01:27，含 `return_series.parquet` 月度收益序列，`return_series_tag: qfq`）；旧结果 `results/small_cap_v2_20260714_200852/` 和 `results/small_cap_v2_20260713_194952/` 暂停引用。**注意：新数字仍含幸存者偏差，标记 ⚠️暂停引用。**
8. **网格交易结论更正** ⚠️暂停引用：原文章称"一组都没跑赢"有误。实际 72 组中 21 组网格跑赢买持（胜率 29%），集中在震荡品种（创业板 +5.3%、中证1000 +3.1%、科创50 +4.4%）。单边上涨品种（纳指 -6.0%）网格跑输。`strategy_grid.png` 图中策略线在基准线上方，与原结论矛盾。**注意：网格回测存在卖出记账错误、参数稳定性检验和成本影响等多相邻问题，数字暂停引用。**

## 数据缓存与复现

`data/live_daily_cache/`、`data/qfq_cache/` 及若干 `.pkl` 文件体积较大（合计约 650 MB），已通过 `.gitignore` 排除，不在 Git 仓库中。首次复现需自行生成，步骤如下：

```bash
# 1. 构建存活股未复权日线缓存（raw close，parquet，4590 只，约 328 MB）
python scripts/build_live_daily_cache.py --workers 10

# 2. 构建前复权缓存（qfq close，parquet，4590 只，约 325 MB）
python scripts/build_qfq_cache.py --workers 10

# 3. 运行小市值回测（自动读取上述缓存，输出到 results/small_cap_v2_<timestamp>/）
python scripts/small_cap_v2.py
```

两个缓存脚本均为增量构建：已缓存的股票会自动跳过（除非加 `--no-skip-existing`），断点续跑友好。`--max-stocks N` 可用于小批量测试。

### 缓存与结果校验摘要

| 路径 | 文件数 | 大小 | 说明 |
|------|:---:|:---:|------|
| `data/live_daily_cache/` | 4590 | ~328 MB | 存活股 raw close 日线（parquet，git-excluded） |
| `data/qfq_cache/` | 4590 | ~325 MB | 存活股前复权 close 日线（parquet，git-excluded，覆盖率 100%） |

最新回测结果（已提交 Git）：

| 文件 | SHA-256 | 大小 |
|------|---------|:---:|
| `results/small_cap_v2_20260715_011613/small_cap.json` | `d7716c3b1e882fb47bfe5a3deadac8a29f00e0ecb632d1797132bce8cf0b975c` | 456 KB |
| `results/small_cap_v2_20260715_011613/return_series.parquet` | `b10b8d8af3fce0506d60d38de8ea7d66778acb07830af8a43cad247b40e593e3` | 24 MB |

`small_cap.json` 顶层字段 `run_id`、`timestamp`（2026-07-15T01:27:13）、`qfq_coverage`（100%）、`return_series_tag`（`qfq`）可用于校验数据来源。验证脚本 `scripts/verify_reproducibility.py <results_dir>` 可从 `return_series.parquet` 独立重算 CAGR/Sharpe/MaxDD 并与 JSON 比对。独立审计报告 `results/small_cap_v2_20260715_011613/verification_report.md` 确认全部 12 个场景的独立重算结果与 JSON 精确匹配（Δ=0.00e+00）。

## 图表生成

图表使用 HTML+CSS 编写，通过 Playwright 截图生成：

```bash
python -c "
from playwright.sync_api import sync_playwright
import os

html_path = os.path.abspath('../html/zhihu_article_charts.html')
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1000, 'height': 800})
    page.goto(f'file://{html_path}')
    page.wait_for_timeout(500)
    for cid, fn in [('chart1','chart1_overview.png'),('chart2','chart2_heatmap.png'),
                     ('chart3','chart3_smallcap.png'),('chart4','chart4_trend.png'),
                     ('chart5','chart5_ff3factor.png'),('chart6','chart6_quadrant.png')]:
        page.locator('#'+cid).screenshot(path=f'../charts/{fn}')
        print(fn)
    browser.close()
"
```

## 银行股分红回测（知乎 Q443919043，2026-07）

对应知乎问题[「如果一直只买工农中建银行股，一直不卖等着分红，跌了就加仓，是不是就不会被割韭菜了？」](https://www.zhihu.com/question/443919043)的回答复现材料。

**核心结论**：题主策略（月定投+跌10%加倍+分红再投+不卖）从四个历史大顶任一个起投工行，年化 10.1%~19.9%（同期沪深300定投仅 3.0%~3.6%）；但 2007-11-01 历史大顶一把梭工行要套 9.7 年、年化仅 4.42%；同策略用在民生银行定投 19 年年化仅 1.88%，跑输 3% 定存。

**方法学**（与主项目量化回测不同的点）：
- 真实不复权价格（新浪）+ 逐笔分红事件（东财）手工模拟，红利税按持有期分档（>1年0%/1月-1年10%/<1月20%），佣金万2.5最低5元
- 交叉验证：手工模拟 vs 新浪后复权总回报（工行 4.42% vs 4.56%，民生 -0.25% vs -0.26%）
- ⚠️ 腾讯 fqkline 的 hfq 数据累积复权因子严重偏小（工行 19 年 1.47x vs 真实 2.31x），已弃用——用后复权数据前请先用"除权日跳水≈每股分红"验证

**复现**：

```bash
python3 scripts/fetch_bank_data.py        # 拉取数据→data/bank_dividend/（仓库已含快照，可跳过）
python3 scripts/bank_dividend_backtest.py # 回测→results.json + series.pkl
python3 scripts/bank_dividend_charts.py   # 6张图→charts/bank_*.png
```

数据快照 `data/bank_dividend/`（回测截止 2026-07-28）：6 家银行不复权/后复权日线、分红明细、沪深300指数、results.json。

## 许可

MIT
