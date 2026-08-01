# 知乎量化策略回测复现

本仓库包含两个独立的知乎回答回测项目：

| 项目 | 知乎问题 | 状态 |
|------|---------|------|
| [A. 个人做量化交易是否可行](#项目a个人做量化交易是否可行) | [Q529408913](https://www.zhihu.com/question/529408913) | ✅ 7+4 项问题已修复（[Issue #1](https://github.com/NextDoorLaoHuang-HF/zhihu-quant-exploration/issues/1)/[#2](https://github.com/NextDoorLaoHuang-HF/zhihu-quant-exploration/issues/2)），104 测试通过 |
| [B. 银行股分红养老](#银行股分红回测知乎-q4439190432026-07) | [Q443919043](https://www.zhihu.com/question/443919043) | ✅ 关键数字经双源互验 |
| [C. 指数基金定投 vs 一次性](#项目c指数基金定投-vs-一次性) | [Q810847946](https://www.zhihu.com/question/810847946) | ✅ 已发布 |

---

## 项目A：个人做量化交易是否可行

对知乎问题[「个人做量化交易是否可行呢？」](https://www.zhihu.com/question/529408913)下50个高赞回答中提到的量化策略进行了逐条回测验证。

## 核心结论

**散户唯一稳定跑赢的方向，是去机构因为规模进不去的地方，而且越极端越好。**

| 策略方向 | 年化 | 夏普 | 备注 | 状态 |
|---------|:---:|:---:|:---:|:---:|
| T5极端微盘 | 35.68% | — | 回撤-33.3%，收益靠少数暴涨月 | ✅ 最优方向 |
| T10微盘 | 26.40% | — | | ✅ |
| T20小盘 | 20.70% | — | 回撤反而更大(-29.1%) | ✅ |
| T5+国债各半 | 19.1% | 0.97 | 回撤-14.7%，稳健替代 | ✅ |
| T5(8%)+ETF-HRP(92%) | 12.1% | **1.79** | 回撤-7.1%，持有体验最优（文章后新演进，未入正文） | ✅ |
| CB等权/HRP混合 | 6.62% | 0.59 | 回撤-15.7%，200+策略里最稳 | ✅ |
| 网格交易 | — | — | 胜率29%（21/72），震荡品种有效 | ⚠️ 有限 |
| 高股息填权 | 60天+2.79%超额 | — | 修复后结论反转，但幅度小 | ⚠️ |
| SMB规模因子 | 10.3% | — | t=1.38，方向正不够显著 | ⚠️ |
| HML价值因子 | 7.3% | — | t=1.31，方向正不够显著 | ⚠️ |
| 策略轮动(波动率切换) | 最优33.7% | — | 仍跑输纯T5 | ❌ |
| 双均线(5/20) | <7% | 0.38 | 胜率17% | ❌ |
| 趋势跟踪(MA200) | 7.0% | 0.73 | 买持12.7% | ❌ |
| ST股 | 4.7% | 0.14 | -4.7%超额 | ❌ |
| 低成交量冷门股 | 6.9% | 0.24 | -2.5%超额 | ❌ |

> **修复说明（2026-07-15）**：[Issue #1](https://github.com/NextDoorLaoHuang-HF/zhihu-quant-exploration/issues/1)（7项根本性问题）与 [Issue #2](https://github.com/NextDoorLaoHuang-HF/zhihu-quant-exploration/issues/2)（4项遗留问题）已通过 Kanban 工作链系统性修复并关闭：选股改为真实市值排序（未复权价×流通股本）、全市场动态股票池5802只（含258只退市股，不再随机抽样）、持有期收益计入分红（qfq，T5 从 19.82%→35.68%）、标准 Fama-French 因子、网格账本守恒引擎、填权改用指数基准（-2.3%→+2.79%，结论反转）、CB 池扩至 1006 只。文章正文已同步修订，104 项测试通过。过程详见 Issue 评论与 [article.md](article.md) 顶部修订记录。

详见 [article.md](article.md)。

## 目录结构

```
zhihu-quant-exploration/
├── README.md
├── MANIFEST.md                        # 缓存生成命令+校验摘要
├── article.md                         # 项目A文章正文
├── requirements.txt
├── scripts/
│   ├── ── 项目A：量化可行性 ──
│   ├── 01_grid_etf_premium.py         # 网格交易 + ETF折溢价 + CB/HRP混合
│   ├── 02_conventional_dualma_ff.py   # 双均线 + FF三因子 + 趋势跟踪
│   ├── 03_retail_edge_microcap.py     # T5极端微盘 + ST/冷门/退市
│   ├── 04_div_fill_rights.py          # 高股息填权
│   ├── small_cap_v2.py                # 真实流通市值排序+全市场动态池（最终版）
│   ├── fama_french_v2.py              # 标准FF三因子
│   ├── dividend_v2.py                 # 分红事件研究v2
│   ├── grid_v2.py / run_hybrid_v2.py  # 网格/CB-HRP v2
│   ├── build_live_daily_cache.py      # 存活股日线缓存构建（parquet）
│   ├── build_qfq_cache.py             # 前复权缓存构建（parquet）
│   ├── rerun_fixed.py / rerun_microcap_*.py   # 修复链（历史中间态）
│   ├── t5_hrp_combo.py                # T5+ETF-HRP组合（文章后新演进）
│   ├── fetch_*.py / verify_*.py / gen_*.py    # 数据拉取/校验/图表生成
│   ├── lib/
│   │   ├── universe.py               # 点时股票池构建器（全市场动态池）
│   │   ├── metrics.py                # 绩效指标（CAGR/夏普/回撤/Calmar）
│   │   ├── qfq_cache.py              # 前复权缓存模块
│   │   ├── fama_french.py            # FF因子计算
│   │   ├── grid_engine.py            # 网格统一引擎（资金守恒）
│   │   ├── hybrid.py                 # CB/HRP混合策略
│   │   └── pickle_compat.py          # pickle 兼容层
│   ├── ── 项目B：银行股分红 ──
│   ├── fetch_bank_data.py             # 银行股/沪深300数据拉取
│   ├── bank_dividend_backtest.py      # 主回测引擎（真实价+逐笔分红手工模拟）
│   ├── bank_dividend_charts.py        # 6张图
│   ├── bank_dividend_yield_timing.py  # 变体1：股息率≥5%才定投
│   ├── bank_dividend_yield_band.py    # 变体2：股息率仓位管理（含卖出）
│   └── ── 项目C：定投 vs 一次性 ──
│       └── dca_backtest.py            # 5指数×6时段回测+3图
├── data/
│   ├── bank_dividend/                 # 项目B数据快照（6银行+沪深300）
│   ├── live_daily_cache/              # 存活股日线缓存（parquet，4590只，git-excluded）
│   ├── qfq_cache/                     # 前复权缓存（parquet，4590只，git-excluded）
│   ├── delist_prices.pkl / delist_info.json   # 退市股数据
│   └── backtest_fixed_all.json        # 修复链结果
├── results/
│   ├── small_cap_v2_20260715_011613/  # ✅ 权威小市值结果（qfq口径，SHA-256校验见下文）
│   ├── t5_hrp_combo/                  # T5+ETF-HRP组合结果+图
│   └── （其余时间戳目录为历史迭代，已废弃）
├── tests/                             # 测试套件（104 tests）
├── charts/
│   ├── chart1-6 + strategy_* + cover.png      # 项目A图表
│   ├── bank_*.png                     # 项目B图表（6张）
│   └── dca_*.png                      # 项目C图表（3张）
└── html/
    └── zhihu_article_charts.html      # 项目A图表HTML源文件
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

如果在中国电信网络下运行，需要在运行前设置代理环境变量：

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
python scripts/04_div_fill_rights.py
```

脚本会从环境变量读取代理地址，不再硬编码。前三个脚本（01-03）使用新浪源，无需代理。第四个脚本（分红填权）需要东财源。

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
HTTP_PROXY=http://your-proxy:port HTTPS_PROXY=http://your-proxy:port python 04_div_fill_rights.py
```

## 复现注意事项

1. **随机种子固定**：所有抽样使用 `np.random.seed(42)` 或 `np.random.seed(888)`，结果可复现。
2. **数据窗口**：回测区间为 2020-01 至 2026-07，基于akshare实时拉取的最新数据。
3. **个股抽样**：受限于 API 频率，每次随机抽样 100-150 只。样本量越大，统计显著性越高。
4. **滑点**：基础回测仅扣佣金（万2.5双边）。文章中有滑点敏感性分析，T5微盘月换手率5.5%，滑点影响约0.4%年化。
5. **幸存者偏差（已根治 ✅）**：新浪源 `stock_info_a_code_name()` 不含已退市股票。修复路径：腾讯API补充退市股（`scripts/fetch_delist_tx.py` + `scripts/fetch_delist_sz.py`）→ 最终由 `scripts/lib/universe.py` 全市场动态股票池（5802只含258只退市股，逐月 `eligible_at()` 筛选，不再随机抽样）根治。历史中间态：150只抽样+退市修补时 T5 为 19.0%（选股口径还是最低股价，已废弃）。详见 `data/backtest_fixed_all.json`。
6. **回测引擎修复（已合并 ✅）**：三个 bug 修复——①`pct_change(fill_method=None)` 退市后收益不再被pad为0%，退市月手动设为-100%；②ST过滤改用退市股一律排除（旧版用当前名称做历史过滤存在前视偏差）；③补充深交所退市股（旧版仅上交所78只）。详见 `scripts/rerun_fixed.py`。
7. **前复权数据修复（Issue #2，当前最终口径 ✅）**：存活股持有期收益从"未复权收盘价"改为"前复权收盘价"（qfq close）。旧版把分红除权日股价下调当成亏损，系统性低估小市值策略收益。全市场4590只存活股qfq缓存（`scripts/build_qfq_cache.py` + `scripts/build_live_daily_cache.py`），覆盖率 100%。重跑后 T5 市值排序 CAGR 从 19.82% → 35.68%，T10 → 26.40%，T20 → 20.70%。当前权威结果：`results/small_cap_v2_20260715_011613/small_cap.json`（含 `return_series.parquet`，`return_series_tag: qfq`）；旧结果目录已废弃。
8. **网格交易结论更正（已修复 ✅）**：原文章称"一组都没跑赢"有误。实际 72 组中 21 组网格跑赢买持（胜率 29%），集中在震荡品种（创业板 +5.3%、中证1000 +3.1%、科创50 +4.4%）；单边上涨品种（纳指 -6.0%）网格跑输。卖出记账错误已由 `scripts/lib/grid_engine.py` 统一引擎修复（资金守恒单元测试覆盖）。

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
python3 scripts/bank_dividend_yield_timing.py  # 读者提问变体：股息率择时定投
python3 scripts/bank_dividend_yield_band.py    # 读者提问变体2：股息率仓位管理（含卖出）
python3 scripts/bank_dividend_reader_comment.py # 读者评论验证：起点 vs 终点敏感性
python3 scripts/bank_dividend_valuation.py     # 读者评论：各历史大顶 PE/PB/股息率（需联网抓东财）
```

### 变体1：只在股息率≥5%时定投（`bank_dividend_yield_timing.py`）

> 评论区提问：如果只在股息率高于5%的时候定投，收益率会高一些吗？

**规则**：每月定投日，TTM 股息率（过去365天已除权派息之和 ÷ 当日价）≥5% 才买入；不足则当月预算攒成场外现金（0收益），等触发时一并投入。

**结果**（工行，2007-11大顶起点，每月1000）：

| 策略 | 投入 | 终值 | 年化(XIRR) |
|------|-----:|-----:|:---:|
| 股息率≥5%才买 | 22.5万 | **65.7万** | **11.13%** |
| 无脑月定投 | 22.5万 | 63.9万 | 10.16% |

**解读**：
- 有效的原因：2007年大顶股息率仅0.18%，首笔买入自动推迟至2010-06（股息率7.9%），避开了最贵的一段
- 幅度有限：近15年工行股息率大半时间在5%附近以上，择时版仍买入48%的月份
- 阈值4.5%~6%结果在11.0%~11.4%之间，不挑参数；四行组合（2010-08起）择时全面小胜0.6~0.9pct
- ⚠️ 后视镜规则：股息率长期低于5%的品种（如沪深300，常年1-2%）该策略永远不出手；场外现金闲置期收益为0（XIRR已计入此拖累）

### 变体2：股息率仓位管理——>5%越高越买，<4%越低越卖（`bank_dividend_yield_band.py`）

> 评论区提问：以股息率为标杆，高于5%逐步买入、越高越买，低于4%卖出、越低越卖，收益是否会更高？

**规则**：10万存量，目标仓位=clamp((股息率-4%)/2%, 0, 1)（≤4%清仓、5%半仓、≥6%满仓），每月调仓；卖出计佣金万2.5+印花税0.1%。另设缓冲带版：4%~5%之间不动。

**结果**（年化）：

| 起点 | 买入持有 | 线性调仓 | 缓冲带版 |
|------|:---:|:---:|:---:|
| 工行 2007-11 大顶 | 4.42% | 6.75% | **8.26%** |
| 工行 2010-08 | **9.25%** | 8.22% ❌ | 9.54% ≈ |
| 四行组合 2010-08 | 10.05% | 9.35% ❌ | 10.20% ≈ |

**解读**：
- **线性调仓跑输**：19年买卖70+次、10万本金成交额180万——成本磨损+牛市半仓踏空
- **缓冲带版≈打平**（组合+0.15pct），建行单品种反而跑输1.25pct
- 唯一大幅跑赢是2007大顶起点，几乎全部来自起点空仓避开腰斩的**后视镜**
- 阈值±0.5pct结果在7.5%~11.2%乱跳——参数运气，无稳健信号
- ⚠️ **TTM接缝伪影**：初版缓冲带曾显示11.16%，追查调仓明细发现1.6pct来自数据伪影（跨年除权日错位→365天窗口无派息→股息率瞬间为0→2015/2021两次"神级逃顶"）；修正口径后优势消失。**择时策略的回测优势，常藏在数据伪影和参数运气里**

数据快照 `data/bank_dividend/`（回测截止 2026-07-28）：6 家银行不复权/后复权日线、分红明细、沪深300指数、results.json。

### 读者评论验证：定投收益对"终点"敏感、对"起点"不敏感（`bank_dividend_reader_comment.py`）

> 评论区质疑（2026-08-01）："从06年的最底端3.4开始买工行，然后一直定投到2022年4.6，再看这个收益率，会惨不忍睹。"

**验证**（同规则月定投，仅换起点/终点）：

| 起点 | 终点 | XIRR |
|------|------|:---:|
| 2006-11（股价≈3.4） | 2022-01（4.65） | 4.98% |
| 2006-11（股价≈3.4） | 2022年末（4.34） | 4.42% |
| 2007-10 大顶（8.84） | 2022年末（4.34） | 4.55% |
| 2006-11（股价≈3.4） | 2026-07（7.97） | 9.73% |
| 2007-10 大顶（8.84） | 2026-07（7.97） | 10.1%（正文场景） |

**结论**：读者场景确实难看（XIRR 4.4%~5.0%，与定存相当），但根因不是"起点在低点"——同样终点2022年末，06低点起投4.42% vs 07大顶起投4.55%，几乎无差；真正决定收益率的是**终点估值**：同样06年起投，终点2022年末4.42% vs 终点2026-07 9.73%。正文10.1%的本质是"终点选在2026年高位"的终点效应，与起点选择无关（定投摊薄成本、钝化起点；一次性买入才看起点）。

### 各历史大顶的估值参考（`bank_dividend_valuation.py`）

> 评论区提问："话说有没有各个历史大顶的市盈率股息率这些数据可以参考一下？"

工行在正文"情景一"四个大顶时点的估值（数据快照 `data/bank_dividend/valuation_icbc.csv`）：

| 大顶日 | 收盘价 | PE-TTM | PB | 股息率TTM |
|--------|:---:|:---:|:---:|:---:|
| 2007-11-01（工行历史最高8.84） | 8.84 | 40.2× | 5.67× | 0.18% |
| 2015-06-08（杠杆牛顶） | 5.63 | 7.2× | 1.25× | 4.65% |
| 2018-01-24（蓝筹顶） | 7.40 | 9.3× | 1.29× | 3.17% |
| 2021-02-18（核心资产顶） | 5.24 | 6.5× | 0.70× | 5.02% |

口径：2018 起 PE-TTM/PB 为东财估值接口值；2007/2015 东财无数据，按当日可得财报自算（PE-TTM=收盘价/最近4季EPS，PB=收盘价/最近报告期每股净资产）；股息率TTM=过去365天已除权派息÷当日收盘价（与正文变体口径一致，本地分红明细可复现）。**四个大顶只有2007年是估值泡沫（PE 40倍、股息率0.18%），后三次PE 6~9倍、股息率3%~5%——是情绪顶不是估值顶，这正是"从大顶开始定投也没被割"的估值层面的原因。**

---

## 项目C：指数基金定投 vs 一次性

对应知乎问题[「为什么说定投是最愚蠢的？」](https://www.zhihu.com/question/810847946)的回答复现材料。

**核心结论**：30 组回测（5 指数 × 6 时段）定投胜出仅 6 组（**胜率 20%**）——定投的数学劣势真实存在（沪深300 自2005：定投 2.8% vs 一次性 7.3%；纳指 9.0% vs 12.2%），但它的价值在行为学（让拿不住的人拿得住），不在收益。

**复现**：

```bash
python3 scripts/dca_backtest.py   # 5指数×6时段回测+3图→charts/dca_*.png
# 注：脚本从环境变量 HTTP_PROXY/HTTPS_PROXY 读取代理地址，未设置则直连
```

图表：`charts/dca_comparison_bars.png`、`dca_vs_lump_all.png`、`dca_win_rate.png`。

## 许可

MIT
