import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
csv_path = 'data\\read_data\\DE_12k_1797.csv'
df = pd.read_csv(csv_path)

# 只取前10列（如果有表头，自动用表头，否则用列号）
columns = df.columns[:10]

# 每列取前1000个数据点
data = df[columns].iloc[:1000]

# 创建3列4行的子图布局
fig, axes = plt.subplots(4, 3, figsize=(14, 10))
axes = axes.flatten()

# 只在第一个子图绘制第一列数据
axes[0].plot(data[columns[0]])
axes[0].set_title(columns[0])
axes[0].set_xticks([0, 200, 400, 600, 800, 1000])
axes[0].set_xlabel('Index')
axes[0].set_yticks([])

# 隐藏第一行的其余两个子图
axes[1].axis('off')
axes[2].axis('off')

# 从第二行开始，依次绘制剩下的9个数据
for i, col in enumerate(columns[1:]):
    ax_idx = i + 3  # 跳过前3个子图
    axes[ax_idx].plot(data[col])
    axes[ax_idx].set_title(col)
    axes[ax_idx].set_xticks([0, 200, 400, 600, 800, 1000])
    axes[ax_idx].set_xlabel('Index')
    axes[ax_idx].set_yticks([])

# 隐藏多余的子图（如果有）
for j in range(len(columns) + 2, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('data\\read_data\\origin_data_plot.png', dpi=300)  # 保存图片到当前目录，分辨率300dpi
plt.show()
