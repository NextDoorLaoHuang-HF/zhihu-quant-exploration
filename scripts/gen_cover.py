"""公众号封面图 — 大字版"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(11.75, 5), facecolor='#0f172a')
ax.set_facecolor('#0f172a')
ax.set_xlim(0, 11.75)
ax.set_ylim(0, 5)
ax.axis('off')

# 超大主数字
ax.text(5.875, 2.7, '23.9%', fontsize=160, fontweight='bold', color='#f87171',
        ha='center', va='center')

# 大字副标题
ax.text(5.875, 0.7, 'AI回测发现·极端微盘策略', fontsize=28, color='#cbd5e1',
        ha='center', va='center', fontweight='bold')

# 极小标签
ax.text(0.4, 4.7, '量化回测笔记', fontsize=12, color='#475569',
        fontweight='bold', va='center')

fig.tight_layout(pad=0)
fig.savefig('../charts/cover.png', dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close(fig)
print('✅ cover.png')
