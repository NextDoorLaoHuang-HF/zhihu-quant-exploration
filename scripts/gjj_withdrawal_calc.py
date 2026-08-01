# -*- coding: utf-8 -*-
"""
公积金「取 vs 留」算账 —— 知乎问题 564845718「公积金有必要取出来吗？」配套计算
所有输入数字均为 2026-08-01 从官方来源逐条核实的现行值，来源见 data/gjj_verified_sources.json。
输出：results/gjj_withdrawal/results.json + charts/gjj_rate_landscape.png + charts/gjj_loan_gap.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 已核实输入（2026-08-01 核实，来源逐条登记） ============
RATES = {
    "gjj_deposit":   0.015,   # 公积金存款利率：一年期定存基准利率，2016-02起（条例第21条+央行2016机制；年报反推1.52%）
    "yuebao":        0.0121,  # 余额宝7日年化（天天基金000198，2026-08-01）
    "dep_1y":        0.0095,  # 一年定存挂牌（中行2025-05-20表，至核实日最新）
    "dep_3y":        0.0125,  # 三年定存挂牌（同上）
    "gjj_loan_5y":   0.021,   # 公积金贷款利率·首套·5年以下(含)（2025-05-08起，北京中心官网历年利率表）
    "gjj_loan_5yp":  0.026,   # 公积金贷款利率·首套·5年以上（同上）
    "lpr_5y":        0.035,   # LPR 5年期以上（2026-07-20，中国货币网）
}

SOURCES = {
    "gjj_deposit_rate": {
        "value": "1.5%（一年期定期存款基准利率）",
        "basis": "《住房公积金管理条例》第21条（gov.cn 公报 content_61418）；央行等三部门2016年2月《关于完善职工住房公积金账户存款利率形成机制的通知》统一按一年期定存基准利率执行",
        "crosscheck": "《全国住房公积金2024年年度报告》（住建部/财政部/央行，2025-06-01，gov.cn/zhengce/zhengceku/202506/content_7026110.htm）：2024年支付缴存人利息1597.29亿元 ÷ 平均缴存余额约10.49万亿元 ≈ 1.52%",
    },
    "gjj_loan_rate": {
        "value": "首套 5年以下(含)2.1% / 5年以上2.6%（2025-05-08起）",
        "source": "北京住房公积金管理中心官网《公积金/商业贷款利率》历年利率表 https://gjj.beijing.gov.cn/web/zwfw5/gjjsydkll/index.html",
    },
    "lpr": {
        "value": "1年期3.0% / 5年期以上3.5%（2026-07-20）",
        "source": "全国银行间同业拆借中心（中国货币网 chinamoney.com.cn LPR接口）",
    },
    "bank_deposit": {
        "value": "活期0.05 / 3月0.65 / 6月0.85 / 1年0.95 / 2年1.05 / 3年1.25 / 5年1.30（%）",
        "source": "中国银行人民币存款利率表（2025-05-20，至2026-08-01为最新一期）https://www.boc.cn/fimarkets/lilv/fd31/",
    },
    "yuebao": {
        "value": "7日年化 1.21%（2026-08-01）",
        "source": "天天基金 天弘余额宝货币 000198 页面",
    },
    "reform_2026": {
        "value": "“深化住房公积金制度改革，扩大使用范围，着力满足缴存人不同阶段的多样化住房需求”",
        "source": "国务院《关于〈扩大消费“十五五”规划〉的批复》（2026-07-13）https://www.gov.cn/zhengce/zhengceku/202607/content_7075217.htm",
        "note": "国家层面为方向性定调；“提取限制全国统一取消”未见任何全国性文件，具体放宽由各地公积金中心陆续出台",
    },
    "withdraw_situations": {
        "value": "6种法定情形：购/建/翻建/大修自住住房；离休退休；完全丧失劳动能力并终止劳动关系；出境定居；偿还购房贷款本息；房租超出家庭工资收入规定比例",
        "source": "《住房公积金管理条例》第24条（国务院令第350号，2002年修订；2019年国务院令第710号部分修订）",
    },
    "national_stats": {
        "value": "实缴人数1.76亿；缴存余额10.93万亿元（2024年末）；2024年8127万人提取2.77万亿元、提取率76.15%；累计提取占累计缴存66.69%",
        "source": "《全国住房公积金2024年年度报告》",
    },
    "beijing_quota": {
        "value": "每缴存1年可贷15万元；上浮后最高不超过160万元；月还款额不超过月收入60%",
        "source": "北京住房公积金管理中心《贷款业务问答》https://gjj.beijing.gov.cn/web/zwfw5/1747335/1747338/index.html",
    },
    "beijing_rent": {
        "value": "无房且连续缴存3个月可租房提取：告知承诺制2000元/人/月；有发票且备案按实际月租金",
        "source": "《关于进一步优化租房提取业务的通知》京房公积金发〔2023〕6号（现行有效）",
    },
}

def compound(p, r, years):
    return p * (1 + r) ** years

def mortgage(p, annual_r, years):
    """等额本息：返回(月供, 总利息)"""
    r = annual_r / 12
    n = years * 12
    f = (1 + r) ** n
    monthly = p * r * f / (f - 1)
    return monthly, monthly * n - p

# ============ 算账1：纯利息对比（10万 × 10年） ============
P, Y = 100000, 10
interest_cmp = {
    "留公积金(1.5%)":      compound(P, RATES["gjj_deposit"], Y) - P,
    "取出来放余额宝(1.21%)": compound(P, RATES["yuebao"], Y) - P,
    "取出来存三年定存滚动(1.25%)": compound(P, RATES["dep_3y"], Y) - P,
    "取出来存一年定存滚动(0.95%)": compound(P, RATES["dep_1y"], Y) - P,
}

# ============ 算账2：贷款资格价值（100万 × 30年 等额本息） ============
LP, LY = 1000000, 30
m_gjj,  i_gjj  = mortgage(LP, RATES["gjj_loan_5yp"], LY)
m_lpr,  i_lpr  = mortgage(LP, RATES["lpr_5y"], LY)
m_low,  i_low  = mortgage(LP, 0.0305, LY)   # 情景：商贷按 LPR-45BP（部分城市首套最低执行口径）
m_high, i_high = mortgage(LP, 0.041, LY)    # 情景：商贷按 LPR+60BP（加点较高城市/二套）

# ============ 算账3：有商贷的人——提取提前还贷的无风险利差 ============
# 每提取1万元提前还商贷 ≈ 买入收益率=商贷利率的无风险资产
prepay = {}
for label, r in [("商贷按LPR 3.5%", 0.035), ("商贷 4.1%", 0.041)]:
    _, saved = mortgage(10000, r, 20)   # 1万元20年等额本息的利息
    keep = compound(10000, RATES["gjj_deposit"], 20) - 10000
    prepay[label] = {"提前还贷省利息": round(saved), "留公积金得利息": round(keep), "净赚": round(saved - keep)}

# ============ 年报反推存款利率 ============
interest_paid = 1597.29e8
bal_2024 = 109252.79e8
bal_2023 = bal_2024 / 1.0861
implied = interest_paid / ((bal_2023 + bal_2024) / 2)

results = {
    "verified_at": "2026-08-01",
    "rates": RATES,
    "sources": SOURCES,
    "calc1_interest_10w_10y": {k: round(v) for k, v in interest_cmp.items()},
    "calc2_loan_100w_30y": {
        "公积金2.6%":       {"月供": round(m_gjj),  "总利息": round(i_gjj)},
        "商贷3.05%(LPR-45BP情景)": {"月供": round(m_low),  "总利息": round(i_low)},
        "商贷3.5%(LPR)":    {"月供": round(m_lpr),  "总利息": round(i_lpr)},
        "商贷4.1%(LPR+60BP情景)": {"月供": round(m_high), "总利息": round(i_high)},
        "利息差": {
            "vs LPR口径": round(i_lpr - i_gjj),
            "vs LPR-45BP": round(i_low - i_gjj),
            "vs LPR+60BP": round(i_high - i_gjj),
        },
    },
    "calc3_prepay_arbitrage_per_1w_20y": prepay,
    "annual_report_crosscheck": {"implied_deposit_rate": f"{implied*100:.2f}%"},
}

os.makedirs(f"{ROOT}/results/gjj_withdrawal", exist_ok=True)
with open(f"{ROOT}/results/gjj_withdrawal/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
with open(f"{ROOT}/data/gjj_verified_sources.json", "w", encoding="utf-8") as f:
    json.dump({"verified_at": "2026-08-01", "rates": RATES, "sources": SOURCES}, f, ensure_ascii=False, indent=2)

# ============ 图1：利率全景（存款端 vs 贷款端，别再把两个数搞混） ============
fig, ax = plt.subplots(figsize=(9, 5.2))
items = [
    ("公积金账户存款\n（你的钱躺在这）", 1.5,  "#2e7d32"),
    ("三年定存\n（取出来存银行）",   1.25, "#888888"),
    ("余额宝\n（取出来放货基）",     1.21, "#888888"),
    ("一年定存\n（取出来存银行）",   0.95, "#888888"),
    ("公积金贷款·首套\n（你借钱的成本）", 2.6,  "#c62828"),
    ("商贷基准LPR·5年以上\n（找银行借的成本）", 3.5, "#e65100"),
]
labels = [x[0] for x in items]
vals = [x[1] for x in items]
colors = [x[2] for x in items]
bars = ax.barh(range(len(items)), vals, color=colors, height=0.62)
ax.set_yticks(range(len(items)))
ax.set_yticklabels(labels, fontsize=10.5)
ax.invert_yaxis()
for b, v in zip(bars, vals):
    ax.text(v + 0.04, b.get_y() + b.get_height()/2, f"{v}%", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 4.1)
ax.set_xlabel("年利率（%）", fontsize=11)
ax.set_title("2026年8月现行利率：存钱端和借钱端是两组数，别搞混\n（绿色=你拿钱，红色/橙色=你付钱）", fontsize=13)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{ROOT}/charts/gjj_rate_landscape.png", dpi=150)
plt.close()

# ============ 图2：贷100万30年，公积金 vs 商贷总利息 ============
fig, ax = plt.subplots(figsize=(8.5, 4.8))
names = ["公积金贷款\n2.6%（首套）", "商贷 3.05%\n（LPR-45BP情景）", "商贷 3.5%\n（按LPR）", "商贷 4.1%\n（LPR+60BP情景）"]
interests = [i_gjj/1e4, i_low/1e4, i_lpr/1e4, i_high/1e4]
cols = ["#2e7d32", "#9e9e9e", "#e65100", "#c62828"]
bars = ax.bar(names, interests, color=cols, width=0.55)
for b, v, ref in zip(bars, interests, interests):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}万", ha="center", fontsize=12, fontweight="bold")
for i, g in enumerate([i_low-i_gjj, i_lpr-i_gjj, i_high-i_gjj]):
    ax.text(i+1, interests[i+1] + 6.5, f"比公积金多还{g/1e4:.1f}万", ha="center", fontsize=10.5, color="#c62828", fontweight="bold")
ax.set_ylabel("30年总利息（万元）", fontsize=11)
ax.set_title("贷100万、30年、等额本息：用不用公积金贷款，利息最多差近30万", fontsize=13)
ax.set_ylim(0, max(interests)*1.22)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{ROOT}/charts/gjj_loan_gap.png", dpi=150)
plt.close()

print(json.dumps(results["calc1_interest_10w_10y"], ensure_ascii=False, indent=1))
print(json.dumps(results["calc2_loan_100w_30y"], ensure_ascii=False, indent=1))
print(json.dumps(results["calc3_prepay_arbitrage_per_1w_20y"], ensure_ascii=False, indent=1))
print("implied deposit rate:", results["annual_report_crosscheck"])
