import matplotlib.pyplot as plt
import numpy as np

# --- 绘图风格设置 ---
plt.style.use('seaborn-v0_8') # 使用带白色背景和网格的风格，更清晰

# --- 预蒸发实验数据 ---
# 从您提供的LaTeX表格中提取
time_s = np.array([
    0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 
    51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87
])
# 压强数据，单位已从 10^-3 Pa 转换为 Pa
pressure_pa = np.array([
    5.7, 5.8, 5.9, 6.0, 6.1, 6.3, 6.5, 6.7, 6.8, 6.8, 6.8, 6.7, 6.6, 6.6, 
    6.6, 6.6, 6.6, 6.1, 5.7, 5.7, 5.6, 5.6, 5.5, 5.5, 5.5, 5.5, 5.5, 5.4, 
    5.4, 5.4
]) * 1e-3

# --- 核心：估算测量误差 ---
# 您的数据记录到小数点后一位 (如5.7 x 10^-3 Pa)。
# 一个合理的误差估算是最小刻度的一半，即 0.05 x 10^-3 Pa。
# 这代表了仪器读数的不确定性。
pressure_error = 0.05 * 1e-3

# --- 绘图 ---
fig, ax = plt.subplots(figsize=(12, 7))

# 1. 绘制带误差棒的散点图
ax.errorbar(time_s, pressure_pa, yerr=pressure_error,
            fmt='o',             # 'o' 表示数据点是圆形
            color='#D32F2F',      # 数据点颜色
            ecolor='#D32F2F',     # 误差棒颜色
            elinewidth=1.5,      # 误差棒线宽
            capsize=4,           # 误差棒两端短横线的长度
            markersize=6,        # 数据点大小
            markeredgecolor='black',
            markerfacecolor='#EF5350',
            linewidth=0.5,
            label='Experimental Data with Uncertainty', 
            zorder=3)

# 2. (可选) 添加连接线以显示时间顺序
# 这条线不代表“拟合”，仅用于引导视


# --- 图表美化 ---
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Pressure (Pa)', fontsize=12)

# 为了更清晰地展示微小变化，Y轴不再使用对数刻度，而是自动缩放到数据范围
# ax.set_yscale('log') # 在这个数据量级下，线性刻度更佳

# 调整Y轴的格式，以科学记数法显示
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax.legend()
ax.grid(True, which="major", linestyle=':', linewidth=0.7)

# 自动调整坐标轴范围
ax.autoscale(enable=True, axis='both', tight=True)
ax.margins(0.05)

plt.tight_layout()
plt.show()
