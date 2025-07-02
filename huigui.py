import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 数据
# x = np.array([0.021, 0.031, 0.042, 0.05, 0.06, 0.066, 0.076, 0.086, 0.097, 0.1]).reshape(-1, 1)
# y = np.array([6.3, 16.5, 26.7, 36.4, 46.5, 54.5, 64.3, 68.2, 77.2, 94.5])

x = np.array([0.021, 0.031, 0.042, 0.05, 0.06]).reshape(-1, 1)
y = np.array([6.3, 16.5, 26.7, 36.4, 46.5])  # 不用reshape

# 线性回归
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# 打印回归直线方程
print(f"回归直线方程: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")

# 绘图
plt.figure(figsize=(8, 6), dpi=120)
plt.scatter(x, y, color='dodgerblue', label='原始数据', s=80, edgecolor='k')
plt.plot(x, y_pred, color='crimson', linewidth=2, label='回归直线')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('线性回归拟合', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
