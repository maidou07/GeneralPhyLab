import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from scipy.signal import find_peaks
import os # Import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# Create image directory if it doesn't exist
img_dir = 'sound/img'
os.makedirs(img_dir, exist_ok=True)

# --- 声学和换能器参数 (空气) ---
c = 346.0       # 声速 (m/s) - 空气
f = 36981.0     # 频率 (Hz) - 36.981 kHz
a = 0.0191        # 换能器半径 (m) - 30mm
lambda_ = c / f
k = 2 * np.pi / lambda_

print(f"介质: 空气, c = {c} m/s")
print(f"频率 f = {f/1000:.3f} kHz")
print(f"换能器半径 a = {a*1000:.1f} mm")
print(f"波长 λ = {lambda_*1000:.4f} mm")
nf_dist = a**2 / lambda_
print(f"近场距离 (菲涅尔区) 大约延伸至 a^2/λ ≈ {nf_dist*1000:.1f} mm")

# === 计算函数 (加入 1/r 衰减) ===
def calculate_pressure_field_piston(field_points_X, field_points_Z, source_positions, k_wavenumber, amplitude_per_source):
    """
    计算由活塞表面点源在指定场点产生的总复声压 (用于2D场)。
    加入了 1/r 衰减。
    """
    num_field_points = len(field_points_X)
    num_source_points = len(source_positions)
    P_total_flat = np.zeros(num_field_points, dtype=np.complex128)

    sx, sy, sz = source_positions[:, 0], source_positions[:, 1], source_positions[:, 2]

    print("开始计算声场 (含 1/r 衰减)...")
    start_time = time.time()
    for i in range(num_field_points):
        px, pz = field_points_X[i], field_points_Z[i]
        py = 0 # 场点在 XZ 平面

        # 计算到所有源点的距离
        r = np.sqrt((px - sx)**2 + (py - sy)**2 + (pz - sz)**2)
        r = np.maximum(r, 1e-10) # 防止除零

        # 叠加来自所有源点的贡献 (P = A * exp(ikr) / r)
        P_contributions = amplitude_per_source * np.exp(1j * k_wavenumber * r) / r
        P_total_flat[i] = np.sum(P_contributions)

        # 打印进度
        if (i + 1) % max(1, num_field_points // 20) == 0:
            elapsed = time.time() - start_time
            percent_done = ((i + 1) / num_field_points) * 100
            eta = (elapsed / percent_done) * (100 - percent_done) if percent_done > 0 else 0
            print(f"  计算进度: {percent_done:.1f}% (预计剩余时间: {eta:.1f}s)")

    end_time = time.time()
    print(f"声场计算完成，耗时 {end_time - start_time:.2f} 秒。")
    return P_total_flat

# === 定义要测试的 N 值 ===
N_values = [50, 100, 200,500]

# === 全局 dB 范围和 epsilon ===
db_min_global = -40.0
epsilon_global = 1e-10

# === 准备组合轴向 SPL 图 ===
fig_on_axis, ax_on_axis = plt.subplots(figsize=(12, 5))

# === 循环计算不同 N 值 ===
print("\n=== 开始仿真不同 N_points_per_radius ===")
for N_points_per_radius in N_values:
    print(f"\n--- 计算 N_points_per_radius = {N_points_per_radius} ---")

    # --- 换能器表面离散化 ---
    source_points = []
    xs_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1)
    ys_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1)
    source_strength_per_point = 1.0 # 保持每个点强度一致

    print(f"使用 N_points_per_radius = {N_points_per_radius} 进行离散化...")
    for xs in xs_grid:
        for ys in ys_grid:
            if xs**2 + ys**2 <= a**2:
                source_points.append(np.array([xs, ys, 0]))

    source_points = np.array(source_points)
    num_source_points = len(source_points)
    A_point = source_strength_per_point
    print(f"将换能器表面离散化为 {num_source_points} 个点源。")

    # --- 仿真网格 (保持不变) ---
    z_min_single, z_max_single, nz_single = 0.0001, 0.2, 300
    x_min_single, x_max_single, nx_single = -a*1.5, a*1.5, 150
    z_grid_single = np.linspace(z_min_single, z_max_single, nz_single)
    x_grid_single = np.linspace(x_min_single, x_max_single, nx_single)
    X_single, Z_single = np.meshgrid(x_grid_single, z_grid_single)
    X_flat_single = X_single.flatten()
    Z_flat_single = Z_single.flatten()
    num_field_points_single = len(X_flat_single)
    print(f"  仿真网格: X=[{x_min_single*1000:.0f},{x_max_single*1000:.0f}]mm ({nx_single} pts), Z=[{z_min_single*1000:.1f},{z_max_single*1000:.0f}]mm ({nz_single} pts)")

    # --- 计算声场 ---
    P_total_flat_single = calculate_pressure_field_piston(X_flat_single, Z_flat_single, source_points, k, A_point)
    P_amplitude_single_raw = np.abs(P_total_flat_single)

    # --- 计算声压级 (dB) ---
    print("  计算声压级 (dB)...")
    p_max_single = np.max(P_amplitude_single_raw[P_amplitude_single_raw > 0])
    if p_max_single == 0: p_max_single = 1.0
    P_amplitude_normalized_single = P_amplitude_single_raw / p_max_single
    SPL_dB_flat_single = 20 * np.log10(P_amplitude_normalized_single + epsilon_global)
    SPL_dB_flat_single[SPL_dB_flat_single < db_min_global] = db_min_global
    SPL_dB_single = SPL_dB_flat_single.reshape((nz_single, nx_single))

    # --- 可视化单个换能器声场 (每个 N 单独保存) ---
    print(f"  开始绘制 N={N_points_per_radius} 的二维声场图...")
    plt.figure(figsize=(12, 6))
    X_mm_single = X_single * 1000
    Z_mm_single = Z_single * 1000
    nf_dist_mm = nf_dist * 1000
    a_mm = a * 1000
    im = plt.pcolormesh(Z_mm_single.T, X_mm_single.T, SPL_dB_single.T, cmap='jet', vmin=db_min_global, vmax=0, shading='auto')
    cbar = plt.colorbar(im, label='相对声压级 (dB)')
    cbar.set_ticks(np.linspace(db_min_global, 0, 5))
    plt.ylabel('x / mm')
    plt.xlabel('z / mm')
    plt.title(f'单换能器声场 (N_radius={N_points_per_radius})')
    plt.plot([0, 0], [-a_mm, a_mm], 'r-', lw=2, label='换能器')
    plt.axhline(0, color='white', linestyle=':', linewidth=0.8)
    plt.axvline(nf_dist_mm, color='lime', linestyle=':', linewidth=1, label=f'近场极限 z≈{nf_dist_mm:.1f}mm')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylim(x_min_single*1000, x_max_single*1000)
    plt.xlim(0, z_max_single * 1000)
    plt.gca().set_facecolor('#EEEEEE')
    plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    # Save before showing
    # --- 恢复所有 N 值都保存为 PDF ---
    save_format = 'pdf'
    save_path_field = os.path.join(img_dir, f'single_transducer_field_N{N_points_per_radius}.pdf')
    plt.savefig(save_path_field, format=save_format, bbox_inches='tight') # 移除条件 dpi
    print(f"  二维声场图已保存至: {save_path_field} (格式: {save_format})")
    plt.close() # 关闭当前图形，避免过多窗口

    # --- 提取轴线上声压 (dB) 并添加到组合图中 ---
    x_center_index_single = np.argmin(np.abs(x_grid_single))
    on_axis_SPL_dB_single = SPL_dB_single[:, x_center_index_single]
    z_grid_mm_single = z_grid_single * 1000

    ax_on_axis.plot(z_grid_mm_single, on_axis_SPL_dB_single, linestyle='-', label=f'N_radius={N_points_per_radius}')

# === 完成组合轴向 SPL 图的绘制 ===
print("\n绘制组合轴向声压级图...")
ax_on_axis.set_xlabel('轴向距离 z (mm)')
ax_on_axis.set_ylabel('相对轴向声压级 (dB)')
ax_on_axis.set_title(f'单换能器轴向声压级比较 (空气, {f/1000:.1f}kHz, a={a*1000:.0f}mm)')
ax_on_axis.grid(True, linestyle=':')
ax_on_axis.axvline(nf_dist * 1000, color='r', linestyle='--', label=f'近场极限 ≈ {nf_dist*1000:.1f} mm')
ax_on_axis.legend()
ax_on_axis.set_xlim(0, z_max_single * 1000)
ax_on_axis.set_ylim(db_min_global - 5, 5)
# 保存组合图
save_path_on_axis = os.path.join(img_dir, 'single_transducer_on_axis_spl_comparison.pdf')
fig_on_axis.savefig(save_path_on_axis, format='pdf', bbox_inches='tight') # 轴向图保持PDF
print(f"组合轴向声压级图已保存至: {save_path_on_axis}")
plt.show() # 显示组合图

print("\n仿真结束。") 