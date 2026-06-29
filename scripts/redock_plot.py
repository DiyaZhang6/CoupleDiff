import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 数据准备 ---
# 创建一个模拟的DataFrame，请将其替换为您自己的真实数据。
# 这里的分布是随机生成的，仅用于演示。
np.random.seed(42)
methods = [
    'CarsıDock', 'Glide SP', 'Glide XP', 'AutoDock4',
    'AutoDock-GPU', 'Vinardo', 'Vina-GPU', 'Gnina',
    'DeepDock', 'TankBind', 'EDM-Dock', 'Uni-Mol'
]
data = []
for method in methods:
    # 模拟不同方法的性能差异
    if 'Dock' in method or 'Vina' in method:
        mean_rmsd = np.random.uniform(3, 7)
        std_dev = 2.5
    else:
        mean_rmsd = np.random.uniform(1.5, 4)
        std_dev = 1.8
    num_samples = np.random.randint(250, 300)
    rmsd_values = np.random.lognormal(mean=np.log(mean_rmsd), sigma=0.6, size=num_samples)
    rmsd_values = np.clip(rmsd_values, 0, 20)  # 限制最大值
    for rmsd in rmsd_values:
        data.append({'Method': method, 'RMSD': rmsd})

df = pd.DataFrame(data)

# --- 2. 绘图设置 ---
# 设置整体风格和颜色
sns.set_theme(style="whitegrid")
# 使用一个好看的调色板
palette = sns.color_palette("muted", len(methods))

# 创建一个包含两个子图的画布 (1行2列)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.tight_layout(pad=5.0)  # 增加子图间距

# --- 3. 绘制累积分布函数图 (CDF Plot) ---
ax1 = axes[0]
for i, method in enumerate(methods):
    # 筛选出当前方法的数据
    method_rmsd = df[df['Method'] == method]['RMSD'].sort_values()
    # 计算累积频率
    cumulative_freq = np.arange(1, len(method_rmsd) + 1) / len(method_rmsd)
    # 绘制曲线
    ax1.plot(method_rmsd, cumulative_freq, label=method, color=palette[i])

# 添加 RMSD=2Å 的成功率阈值线
ax1.axvline(x=2.0, color='black', linestyle='--', linewidth=1.5)

# 设置图表细节
ax1.set_xlabel('RMSD (Å)', fontsize=14)
ax1.set_ylabel('Cumulative Frequency', fontsize=14)
ax1.set_title('A', loc='left', fontsize=16, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1.0)
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- 4. 绘制箱形图 (Box Plot) ---
ax2 = axes[1]

# 使用Seaborn绘制箱形图
sns.boxplot(
    x='Method',
    y='RMSD',
    data=df,
    order=methods,  # 保证和CDF图例顺序一致
    palette=palette,
    ax=ax2,
    showmeans=True,  # 显示均值点
    meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"}
)

# 计算并添加每个方法的样本数量 (N=...)
sample_counts = df.groupby('Method').size().loc[methods]
for i, method in enumerate(methods):
    ax2.text(i, -2.5, f"N={sample_counts[method]}", ha='center', size=9)

# 设置图表细节
ax2.set_xlabel('', fontsize=14)  # X轴标签通常在箱形图中省略
ax2.set_ylabel('RMSD (Å)', fontsize=14)
ax2.set_title('B', loc='left', fontsize=16, fontweight='bold')
ax2.set_ylim(-0.5, 20)
ax2.tick_params(axis='x', rotation=45)  # 旋转X轴标签以防重叠
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- 5. 保存和显示图像 ---
plt.savefig('rmsd_performance_plot.png', dpi=300, bbox_inches='tight')
plt.show()