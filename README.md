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
│   ├── lib/universe.py               # 点时股票池构建器
│   ├── lib/metrics.py                # 绩效指标计算
│   └── verify_issues.py               # 验证幸存者偏差+网格交易矛盾
├── data/
│   ├── delist_prices.pkl              # 退市股前复权日线（腾讯API）
│   ├── delist_info.json               # 退市股元信息
│   ├── live_daily_cache/              # 存活股日线缓存（parquet，4590只）
│   ├── qfq_cache/                     # 前复权收盘价缓存（parquet，4590只）
│   └── backtest_fixed_all.json        # 修复后回测结果
├── results/
│   └── small_cap_v2_20260714_200852/ # 最新回测结果（qfq修复后）
│       └── small_cap.json
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
7. **前复权数据修复（Issue #2）** ⚠️暂停引用：存活股持有期收益从"未复权收盘价"改为"前复权收盘价"（qfq close）。旧版把分红除权日股价下调当成亏损，系统性低估小市值策略收益。为全市场4590只存活股持久化parquet格式qfq缓存（`scripts/build_qfq_cache.py` + `scripts/build_live_daily_cache.py`），覆盖率100%、0降级。重跑后 T5 市值排序 CAGR 从 19.82% → 35.68%（+15.86个百分点），T10 从 17.98% → 26.40%，T20 从 11.70% → 20.70%。价格排序策略受影响极小（+0.15%~0.93%）。详见 `results/small_cap_v2_20260714_200852/small_cap.json`，旧结果 `results/small_cap_v2_20260713_194952/` 暂停引用。**注意：新数字仍含幸存者偏差，标记 ⚠️暂停引用。**
8. **网格交易结论更正** ⚠️暂停引用：原文章称"一组都没跑赢"有误。实际 72 组中 21 组网格跑赢买持（胜率 29%），集中在震荡品种（创业板 +5.3%、中证1000 +3.1%、科创50 +4.4%）。单边上涨品种（纳指 -6.0%）网格跑输。`strategy_grid.png` 图中策略线在基准线上方，与原结论矛盾。**注意：网格回测存在卖出记账错误、参数稳定性检验和成本影响等多相邻问题，数字暂停引用。**

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

## 许可

MIT
