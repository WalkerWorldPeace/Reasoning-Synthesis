import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
import seaborn as sns

# =========================
# 全局字体缩放参数
# 改这个数字即可调整所有字体大小
# =========================
FONT_SCALE = 1.5  # 1.0 为原大小，论文建议 1.5~2.0

parser = argparse.ArgumentParser(
    description="Figure 10: contour density plot of solver Consistency vs Accuracy."
)
parser.add_argument(
    "--input_file",
    default="math12k_score_accuracy_analysis.json",
    help="JSON file produced by analysis/evaluate_consistency.py (list of "
    "{score, accuracy} dicts).",
)
parser.add_argument(
    "--output_file",
    default="math12k_score_accuracy_contour_density.pdf",
    help="Output figure path.",
)
args = parser.parse_args()

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("Reading data files...")

# 1. 读取数据
with open(args.input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取 score 和 accuracy
scores = np.array([item['score'] for item in data])
accuracies = np.array([item['accuracy'] for item in data])

print(f"✅ Loaded {len(scores)} questions")
print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"Accuracy range: [{accuracies.min():.3f}, {accuracies.max():.3f}]")

# 计算相关系数
correlation, p_value = pearsonr(scores, accuracies)
print(f"Pearson correlation: {correlation:.4f} (p-value: {p_value:.2e})")

# 计算趋势线
z = np.polyfit(scores, accuracies, 1)
p = np.poly1d(z)
x_line = np.linspace(0, 1, 100)

# 2. 创建等高线密度图
print("\nComputing KDE for contour plot...")

fig, ax = plt.subplots(figsize=(16, 12))  # 放大画布

# 创建密集网格用于KDE
x_grid = np.linspace(scores.min() - 0.05, scores.max() + 0.05, 150)
y_grid = np.linspace(accuracies.min() - 0.05, accuracies.max() + 0.05, 150)
X, Y = np.meshgrid(x_grid, y_grid)
positions = np.vstack([X.ravel(), Y.ravel()])

# 计算核密度估计
print("Calculating kernel density estimation...")
kernel = gaussian_kde(np.vstack([scores, accuracies]), bw_method='scott')
Z = np.reshape(kernel(positions).T, X.shape)

# 绘制填充等高线
contourf = ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.85)

# 绘制等高线边界
contour_lines = ax.contour(X, Y, Z, levels=10, colors='white', 
                           linewidths=0.5, alpha=0.4, linestyles='solid')

# 添加等高线标签
ax.clabel(contour_lines, inline=True, fontsize=int(8 * FONT_SCALE), fmt='%.1e')

# 叠加散点图
ax.scatter(scores, accuracies, alpha=0.08, s=8, c='white', 
           edgecolors='none', rasterized=True)

# 添加趋势线
ax.plot(x_line, p(x_line), color='red', linestyle='--', 
        alpha=0.9, linewidth=3, 
        label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')

# 添加对角线参考
ax.plot([0, 1], [0, 1], 'yellow', linestyle=':', 
        alpha=0.6, linewidth=2.5, 
        label='Perfect correlation (y=x)')

# 颜色条
cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
cbar.set_label('Probability Density', fontsize=int(15 * FONT_SCALE), fontweight='bold')
cbar.ax.tick_params(labelsize=int(11 * FONT_SCALE))

# 坐标轴设置
ax.set_xlabel('Consistency', fontsize=int(16 * FONT_SCALE), fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=int(16 * FONT_SCALE), fontweight='bold')

# 设置刻度
ax.tick_params(axis='both', which='major', labelsize=int(11 * FONT_SCALE))
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# 网格
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray')

# 图例
ax.legend(loc='lower right', fontsize=int(12 * FONT_SCALE), framealpha=0.95, 
          edgecolor='black', fancybox=True, shadow=True)

# 添加简化的统计信息框
stats_text = (
    f'📊 Statistical Summary\n'
    f'{"="*30}\n'
    f'Data Points: {len(scores):,}\n'
    f'\n'
    f'Consistency:\n'
    f'  Mean (μ) = {np.mean(scores):.4f}\n'
    f'  Std (σ) = {np.std(scores):.4f}\n'
    f'\n'
    f'Accuracy:\n'
    f'  Mean (μ) = {np.mean(accuracies):.4f}\n'
    f'  Std (σ) = {np.std(accuracies):.4f}\n'
    f'\n'
    f'Correlation Analysis:\n'
    f'  Pearson r = {correlation:.4f}\n'
    f'  p-value = {p_value:.2e}\n'
    f'  R² = {correlation**2:.4f}\n'
    f'  Interpretation: {"Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.5 else "Weak"}'
)

# 文本框样式
props = dict(boxstyle='round,pad=1.0', facecolor='white', 
             alpha=0.92, edgecolor='black', linewidth=2)
ax.text(0.015, 0.985, stats_text, transform=ax.transAxes, 
        fontsize=int(10 * FONT_SCALE), verticalalignment='top', bbox=props, 
        family='monospace', linespacing=1.4)

plt.tight_layout()

# 保存图片
plt.savefig(args.output_file, dpi=400, bbox_inches='tight', facecolor='white')
print(f"\n✅ Contour density plot saved as {args.output_file}")

plt.close()