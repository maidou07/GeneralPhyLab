# sound/multi_reflection_scan.py

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# Create image directory if it doesn't exist
img_dir = 'sound/img'
os.makedirs(img_dir, exist_ok=True)

# === 计算函数 (从 field.py 复制) ===
def calculate_pressure_at_points(field_points_xyz, source_positions, source_amplitudes, k_wavenumber):
    """Calculates complex pressure at specific field points from sources with individual amplitudes (with 1/r decay)."""
    num_field_points = field_points_xyz.shape[0]
    if num_field_points == 0:
        return np.array([], dtype=np.complex128)
        
    num_source_points = source_positions.shape[0]
    # Ensure source_amplitudes is a numpy array of the correct shape and complex type
    if not isinstance(source_amplitudes, np.ndarray) or source_amplitudes.shape[0] != num_source_points:
        if np.isscalar(source_amplitudes):
             source_amplitudes = np.full(num_source_points, source_amplitudes, dtype=np.complex128)
        else:
            raise ValueError("source_amplitudes must be a scalar or a numpy array matching source_positions")
            
    P_total = np.zeros(num_field_points, dtype=np.complex128)
    sx, sy, sz = source_positions[:, 0], source_positions[:, 1], source_positions[:, 2]

    for i in range(num_field_points):
        px, py, pz = field_points_xyz[i, 0], field_points_xyz[i, 1], field_points_xyz[i, 2]
        r = np.sqrt((px - sx)**2 + (py - sy)**2 + (pz - sz)**2)
        r = np.maximum(r, 1e-10)
        
        # Sum contributions (P = A_s * exp(ikr) / r) where A_s varies per source
        P_contributions = source_amplitudes * np.exp(1j * k_wavenumber * r) / r
        P_total[i] = np.sum(P_contributions)
        
    return P_total

# --- 声学和换能器参数 (空气) ---
c = 346.0       # 声速 (m/s) - 空气
f = 36981.0     # 频率 (Hz) - 36.981 kHz
a = 0.0191       # 换能器半径 (m) - 19.1mm
lambda_ = c / f
k = 2 * np.pi / lambda_

print(f"介质: 空气, c = {c} m/s")
print(f"频率 f = {f/1000:.3f} kHz")
print(f"换能器半径 a = {a*1000:.1f} mm")
print(f"波长 λ = {lambda_*1000:.4f} mm")
nf_dist = a**2 / lambda_
print(f"近场距离 (菲涅尔区) 大约延伸至 a^2/λ ≈ {nf_dist*1000:.1f} mm")

# --- 换能器表面离散化 ---
N_points_per_radius = 50 # 沿半径方向的离散点数
source_points_orig = []
xs_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1)
ys_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1)

for xs in xs_grid:
    for ys in ys_grid:
        if xs**2 + ys**2 <= a**2:
            source_points_orig.append(np.array([xs, ys, 0])) # 原始换能器位于 z=0 平面

source_points_orig = np.array(source_points_orig)
num_source_points = len(source_points_orig)
source_strength_per_point = 1.0
A_point = source_strength_per_point
print(f"将换能器表面离散化为 {num_source_points} 个点源。")

# --- 多重反射参数 ---
N_pairs = 100
print(f"镜像源对数 N_pairs = {N_pairs}")

# === 定义要测试的反射系数值 ===
R_values = [1.0]
print(f"将测试以下反射系数值 R: {R_values}")

# === 仿真参数 ===
l_min_m = 0.001
l_max_m = 0.150 # 保持 150mm 或按需调整
num_l_points = 2000
l_values = np.linspace(l_min_m, l_max_m, num_l_points)
l_values_mm = l_values * 1000
print(f"扫描距离 l 从 {l_min_m*1000:.1f} mm 到 {l_max_m*1000:.1f} mm ({num_l_points} 点)")

# 预计算常量
lambda_half_mm = (lambda_ / 2) * 1000
db_min_global = -40.0
epsilon_global = 1e-10

# === 准备组合图 ===
fig_amp, ax_amp = plt.subplots(figsize=(15, 6))
fig_spl, ax_spl = plt.subplots(figsize=(15, 6))

# === 循环计算不同 R 值 ===
print("\n=== 开始仿真不同反射系数 R ===")
for R_val in R_values:
    R_0 = R_val # Assume R0 = Rl = R_val
    R_l = R_val
    print(f"\n--- 计算 R = {R_val:.2f} (R0=Rl={R_val:.2f}) ---")

    P_total_multi_at_l = np.zeros(num_l_points, dtype=np.complex128)

    print(f"  循环计算 l 处声压 (N={N_pairs}, R={R_val:.2f})...")
    start_scan_time = time.time()

    for i, l_val in enumerate(l_values):
        if l_val < 1e-9: continue
        target_point_l = np.array([[0, 0, l_val]])
        P_total_at_this_l = 0 + 0j
        P_0 = calculate_pressure_at_points(target_point_l, source_points_orig, A_point, k)[0]
        P_total_at_this_l += P_0
        current_amplitude_factor_pos = R_l
        current_amplitude_factor_neg = R_l * R_0
        for n in range(1, N_pairs + 1):
            src_pos_plus = source_points_orig + np.array([0, 0, 2 * n * l_val])
            amp_plus = A_point * current_amplitude_factor_pos
            P_plus = calculate_pressure_at_points(target_point_l, src_pos_plus, amp_plus, k)[0]
            P_total_at_this_l += P_plus
            current_amplitude_factor_pos *= (R_0 * R_l)
            src_pos_neg = source_points_orig + np.array([0, 0, -2 * n * l_val])
            amp_neg = A_point * current_amplitude_factor_neg
            P_neg = calculate_pressure_at_points(target_point_l, src_pos_neg, amp_neg, k)[0]
            P_total_at_this_l += P_neg
            current_amplitude_factor_neg *= (R_l * R_0)
        P_total_multi_at_l[i] = P_total_at_this_l
        if (i + 1) % max(1, num_l_points // 10) == 0:
             elapsed = time.time() - start_scan_time
             percent_done = ((i + 1) / num_l_points) * 100
             eta = (elapsed / percent_done) * (100 - percent_done) if percent_done > 0 else 0
             print(f"    l-scan (N={N_pairs}, R={R_val:.2f}) 进度: {percent_done:.1f}% (预计剩余时间: {eta:.1f}s)")
    print(f"  l-scan (R={R_val:.2f}) 计算完成，耗时 {time.time() - start_scan_time:.2f} 秒。")

    # --- 分析当前 R 值的振幅和 SPL --- 
    P_amplitude_multi_at_l = np.abs(P_total_multi_at_l)

    p_max_current_R = np.max(P_amplitude_multi_at_l[P_amplitude_multi_at_l > 0])
    if p_max_current_R == 0: p_max_current_R = 1.0
    if not np.isfinite(p_max_current_R) or p_max_current_R <= 0:
        print(f"警告: R={R_val:.2f} 时计算得到无效的最大声压 {p_max_current_R}，跳过绘图。")
        continue

    SPL_dB_at_l = 20 * np.log10(P_amplitude_multi_at_l / p_max_current_R + epsilon_global)
    SPL_dB_at_l[SPL_dB_at_l < db_min_global] = db_min_global

    # --- 添加到组合图 --- 
    ax_amp.plot(l_values_mm, P_amplitude_multi_at_l, linestyle='-', label=f'R={R_val:.2f}')
    ax_spl.plot(l_values_mm, SPL_dB_at_l, linestyle='-', label=f'R={R_val:.2f}')

# === 完成组合图绘制 ===
print("\n绘制组合图...")

# --- 振幅组合图 --- 
ax_amp.set_xlabel('反射面距离 l (mm)')
ax_amp.set_ylabel('相对声压振幅')
ax_amp.set_title(f'反射面中心点总振幅 vs l (不同反射系数 R, N={N_pairs})')
ax_amp.grid(True, linestyle=':')
ax_amp.legend()
ax_amp.set_xlim(0, l_max_m * 1000)
ax_amp.set_ylim(bottom=0)
# 添加理论λ/2标记
y_max_amp = ax_amp.get_ylim()[1]
for i in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
    ax_amp.axvline(i * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
    if i > 0:
        ax_amp.text(i * lambda_half_mm, y_max_amp * 0.95, f'{i}λ/2',
                    fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)
# 保存振幅组合图
save_path_amp = os.path.join(img_dir, f'multi_reflection_amp_vs_l_comparison_N{N_pairs}.pdf')
fig_amp.savefig(save_path_amp, format='pdf', bbox_inches='tight')
print(f"振幅对比图已保存至: {save_path_amp}")

# --- SPL 组合图 --- 
ax_spl.set_xlabel('反射面距离 l (mm)')
ax_spl.set_ylabel('相对声压级 (dB, 各自归一化)')
ax_spl.set_title(f'反射面中心点声压级 vs l (不同反射系数 R, N={N_pairs})')
ax_spl.grid(True, linestyle=':')
ax_spl.legend()
ax_spl.set_xlim(0, l_max_m * 1000)
ax_spl.set_ylim(db_min_global - 5, 5)
# 添加理论λ/2标记
for i in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
    ax_spl.axvline(i * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
    if i > 0:
        ax_spl.text(i * lambda_half_mm, db_min_global, f'{i}λ/2',
                    fontsize=8, ha='center', va='bottom', rotation=90, alpha=0.7)
# 保存SPL组合图
save_path_spl = os.path.join(img_dir, f'multi_reflection_spl_vs_l_comparison_N{N_pairs}.pdf')
fig_spl.savefig(save_path_spl, format='pdf', bbox_inches='tight')
print(f"声压级对比图已保存至: {save_path_spl}")

plt.show() # 显示两张组合图

print("\n多重反射仿真结束。") 