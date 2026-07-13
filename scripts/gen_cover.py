"""公众号封面图 — 高级感版本"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(11.75, 5))
ax.set_xlim(0, 11.75)
ax.set_ylim(0, 5)
ax.axis('off')

# ── 渐变背景 ──
gradient = np.linspace(0, 1, 256).reshape(1, -1)
bg_top = np.array([0.02, 0.06, 0.18])  # 深海军蓝
bg_bot = np.array([0.06, 0.02, 0.14])  # 深紫
for i in range(256):
    t = i / 255
    color = bg_top * (1-t) + bg_bot * t
    ax.axhspan(i*5/256, (i+1)*5/256, facecolor=color, alpha=1, zorder=0)

# ── 装饰：涨势曲线（微光） ──
np.random.seed(42)
t = np.linspace(0.5, 6.5, 80)
trend = 1.2 + t * 0.35
noise = np.cumsum(np.random.randn(80) * 0.06)
y = trend + noise
y = (y - y.min()) / (y.max() - y.min()) * 1.5 + 3.1
for alpha_val in np.linspace(0.08, 0.25, 6):
    ax.plot(t, y + alpha_val*0.03, color='#f87171', linewidth=1.5, alpha=alpha_val)
ax.plot(t, y, color='#f87171', linewidth=2.0, alpha=0.35)

# ── 装饰：散落的数据点 ──
np.random.seed(99)
dots_x = np.random.uniform(7, 11.3, 12)
dots_y = np.random.uniform(0.3, 4.5, 12)
ax.scatter(dots_x, dots_y, s=15, c='#475569', alpha=0.5)

# ── 装饰：细横线 ──
for y_pos, alpha_val in [(0.15, 0.15), (4.85, 0.15), (2.5, 0.06)]:
    ax.axhline(y=y_pos, color='#334155', linewidth=0.4, alpha=alpha_val, 
               xmin=0.04, xmax=0.96)

# ── 主角数字 ──
ax.text(5.875, 2.6, '23.9%', fontsize=150, fontweight='bold', color='#fca5a5',
        ha='center', va='center')

# ── 副标题 ──
ax.text(5.875, 0.85, 'AI 回测发现 · 极端微盘策略', fontsize=24, color='#94a3b8',
        ha='center', va='center', fontweight='bold')

# ── 顶部小字 ──
ax.text(0.5, 4.65, '量化回测笔记', fontsize=10, color='#475569',
        fontweight='bold')
ax.text(11.25, 4.65, '2020–2026', fontsize=10, color='#475569',
        ha='right', fontweight='bold')

fig.tight_layout(pad=0)
fig.savefig('../charts/cover.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print('✅ cover.png')
