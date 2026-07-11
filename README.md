# 知乎「个人做量化交易是否可行」— 策略回测复现

本项目对知乎问题[「个人做量化交易是否可行呢？」](https://www.zhihu.com/question/529408913)下50个高赞回答中提到的量化策略进行了逐条回测验证。

## 核心结论

**散户真正的alpha不在择时，在去机构去不了的地方。**

| 策略方向 | 年化 | 夏普 | 超额(vs全样本) | 状态 |
|---------|:---:|:---:|:---:|:---:|
| T5极端微盘 | 23.9% | 0.81 | +14.5% | ✅ 最优 |
| T10微盘 | 22.2% | 0.88 | +12.8% | ✅ |
| T20小盘 | 14.9% | 0.61 | +5.5% | ✅ |
| 高股息填权 | -2.3%超额 | — | 填权率46% | ❌ |
| 双均线(5/20) | — | 0.38 | 胜率17% | ❌ |
| 趋势跟踪(MA200) | 7.0% | 0.73 | -5.7% | ❌ |
| 网格交易 | — | — | 72组全输 | ❌ |
| ST股 | 4.7% | 0.14 | -4.7% | ❌ |
| 低成交量冷门股 | 6.9% | 0.24 | -2.5% | ❌ |

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
│   └── 04_div_fill_rights.py          # 高股息填权
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
5. **幸存者偏差**：新浪源不含已退市股票，微盘股收益可能被系统性高估。

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
