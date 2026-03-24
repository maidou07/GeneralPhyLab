import matplotlib.pyplot as plt
import numpy as np
# 导入 B-spline 相关的函数
from scipy.interpolate import splrep, splev

plt.style.use('seaborn-v0_8')
# --- 实验数据 ---
# 时间 t (单位: 秒)
time = np.array([30, 35, 40, 45, 50, 55, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300])

# 压强 P (单位: 帕斯卡)
pressure = np.array([
     1.0, 1.1e-1, 7.4e-2, 5.5e-2, 4.3e-2, 3.6e-2, 3.2e-2,
    2.4e-2, 1.9e-2, 1.5e-2, 1.2e-2, 1.1e-2, 1.0e-2, 9.2e-3, 8.5e-3, 8.0e-3
])
# 同样在对数空间进行拟合，效果更好
log_pressure = np.log10(pressure)

# --- B-Spline 平滑拟合 ---
# splrep 函数找到曲线的 B-spline 表示
# s 是一个正的平滑因子(smoothing factor)。s值越大，曲线越平滑，但与数据点的偏差可能越大。
# s=0 会尝试通过所有点，可能导致震荡。从一个较小的值开始尝试是个好方法。
tck = splrep(time, log_pressure, s=0.02)

# 创建一个更密集的横坐标点用于绘制平滑曲线（增加点数以确保视觉平滑）
time_smooth = np.linspace(time.min(), time.max(), 10000)
# splev 函数在新的点上评估样条曲线
log_pressure_smooth = splev(time_smooth, tck)
# 将对数压强转换回原始压强值
pressure_smooth = 10**log_pressure_smooth

# --- 绘图设置 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制原始数据点 (散点图)
ax.scatter(time, pressure, color='#D32F2F', s=20, alpha=0.9, marker='o', edgecolor='black', linewidth=0.02, label='Experimental Data', zorder=3)

# 绘制 B-spline 平滑拟合曲线
ax.plot(time_smooth, pressure_smooth, linestyle='-', color='#D32F2F', label='B-Spline Smooth Fit')

# 设置Y轴为对数刻度
ax.set_yscale('log')

# --- 图表美化 ---
ax.set_title('Pressure vs. Time with B-Spline Smoothing', fontsize=16)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Pressure (Pa) - Log Scale', fontsize=12)
ax.legend()
ax.grid(True, which="both", linestyle='--', linewidth=0.5)

plt.show()
