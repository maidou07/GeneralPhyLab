import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev

# --- 绘图风格设置 ---
plt.style.use('seaborn-v0_8')

# --- 实验数据直接嵌入 ---
# 决策：t=0时刻的数据点未被选用。
time_min = np.array([
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 
    8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 
    14.5, 15.0
])

# 压强数据：根据用户指示进行单位换算
# t < 6min 的原始记录值为 Pa
# t >= 6min 的原始记录值为 10^-3 Pa，需乘以 1e-3 转换为 Pa
pressure_pa = np.array([
    # t < 6 min (单位: Pa)
    1500, 100, 42, 23, 15, 11, 8.8, 2.7, 0.03, 15e-3, 11e-3,
    # t >= 6 min (单位: 10^-3 Pa, 转换为 Pa)
    9.4e-3, 8.0e-3, 7.0e-3, 6.3e-3, 5.8e-3, 5.3e-3, 4.9e-3, 4.6e-3, 4.3e-3,
    4.1e-3, 3.9e-3, 3.7e-3, 3.6e-3, 3.4e-3, 3.3e-3, 3.1e-3, 3.0e-3, 2.9e-3, 2.8e-3
])

# 将压强数据转换为对数尺度，以便进行更稳定的拟合
log_pressure = np.log10(pressure_pa)


# --- B-Spline 平滑拟合 ---
# s 是平滑因子，可以微调。
tck = splrep(time_min, log_pressure, s=0.3)

# 创建一个更密集的时间点数组以绘制平滑曲线
time_smooth = np.linspace(time_min.min(), time_min.max(), 1000)
log_pressure_smooth = splev(time_smooth, tck)
pressure_smooth = 10**log_pressure_smooth


# --- 绘图 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制原始数据点 (散点图)，应用您指定的样式
ax.scatter(time_min, pressure_pa, color="#523535", s=25, alpha=0.9, marker='o', 
           edgecolor='black', linewidth=0.2, label='Experimental Data', zorder=3)

# 绘制 B-spline 平滑拟合曲线
ax.plot(time_smooth, pressure_smooth, linestyle='-', color='#1976D2', 
        linewidth=2, label='B-Spline Smooth Fit')

# 设置Y轴为对数刻度
ax.set_yscale('log')


# --- 图表美化 ---

ax.set_xlabel('Time (min)', fontsize=12)
ax.set_ylabel('Pressure (Pa) - Log Scale', fontsize=12)
ax.legend()
ax.grid(True, which="both", linestyle='--', linewidth=0.5)

# 自动调整坐标轴范围并增加一些边距
ax.autoscale(enable=True, axis='both', tight=False)
ax.margins(0.05)


plt.tight_layout()
plt.show()

