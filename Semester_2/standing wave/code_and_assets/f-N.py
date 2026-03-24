import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

plt.style.use('seaborn-v0_8')
# --- 数据输入 ---
# 使用第二部分最终确认的数据
N = np.array([1, 2, 3, 4])
fc = np.array([60.5, 121.0, 181.5, 242.0]) # 理论频率
fe = np.array([61.9, 126.3, 192, 258])      # 实验频率

# --- 颜色配置 ---
# #AC2B40 (深红), #515CC0 (蓝), #E6A58C (浅橙), #DEDEDE (灰)
color_exp_data = '#AC2B40'  # 实验数据点和拟合线 (深红)
color_th_data = '#515CC0'   # 理论数据点和拟合线 (蓝)

# --- 最小二乘法拟合 ---
# (This part is unchanged)
slope_c, intercept_c, r_value_c, _, _ = linregress(N, fc)
slope_e, intercept_e, r_value_e, _, _ = linregress(N, fe)

# --- 打印拟合结果 ---
# (This part is unchanged)
print("--- 理论数据拟合结果 ---")
print(f"斜率 (k_c): {slope_c:.1f} Hz")
print(f"截距 (b_c): {intercept_c:.1f} Hz")
print(f"相关系数 (R^2): {r_value_c**2:.6f}\n")

print("--- 实验数据拟合结果 ---")
print(f"斜率 (k_e): {slope_e:.2f} Hz")
print(f"截距 (b_e): {intercept_e:.2f} Hz")
print(f"相关系数 (R^2): {r_value_e**2:.6f}\n")


# --- 绘图 ---
plt.figure(figsize=(10, 7))

# 绘制数据点
# CHANGED: Reduced size, added specific markers, and added a crisp edge
plt.scatter(N, fc, label='Theoratical Data ($f_c$)', color=color_th_data, marker='o', s=60, zorder=5, edgecolors='black', linewidths=0.5)
plt.scatter(N, fe, label='Experimental Data ($f_e$)', color=color_exp_data, marker='^', s=60, zorder=5, edgecolors='black', linewidths=0.5)

# 绘制拟合直线
# 创建一个更平滑的x轴用于绘制直线
N_fit = np.linspace(0, 4.5, 100)
fc_fit = slope_c * N_fit + intercept_c
fe_fit = slope_e * N_fit + intercept_e

# CHANGED: Adjusted line widths for better visual distinction
plt.plot(N_fit, fc_fit, label=f'Theoratical fit: $f_c = ({slope_c:.1f}N {intercept_c:+.1f})  Hz$', color=color_th_data, linestyle='--', linewidth=2)
plt.plot(N_fit, fe_fit, label=f'Experimental fit: $f_e = ({slope_e:.1f}N {intercept_e:+.1f})  Hz$', color=color_exp_data, linestyle='-', linewidth=2.5)

# --- 图像美化 ---
# (This part is unchanged)
plt.xlabel('$N$', fontsize=12, family='SimHei')
plt.ylabel('$f$ (Hz)', fontsize=12, family='SimHei')
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, 4.5)
plt.ylim(0, max(fe) * 1.1)
plt.xticks(np.arange(0, 5, 1))
plt.minorticks_on()
# 显示图像
plt.savefig('f-N.pdf', dpi=300, bbox_inches='tight')
plt.show()