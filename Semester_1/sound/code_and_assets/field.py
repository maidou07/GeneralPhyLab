import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from scipy.signal import find_peaks
import os # Import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 指定默认字体
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

# --- 换能器表面离散化 ---
N_points_per_radius = 50 # 沿半径方向的离散点数
source_points = []
# 在圆内均匀（或近似均匀）生成点 - 这里用网格法筛选
xs_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1)
ys_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1)
dA_approx = (xs_grid[1]-xs_grid[0]) * (ys_grid[1]-ys_grid[0]) # 每个格点代表的面积
source_strength_per_point = 1.0 # 假设每个点源强度相同 (可调整)

for xs in xs_grid:
    for ys in ys_grid:
        if xs**2 + ys**2 <= a**2:
            source_points.append(np.array([xs, ys, 0])) # 换能器位于 z=0 平面

source_points = np.array(source_points)
num_source_points = len(source_points)
# 每个点源的复振幅，归一化或保持固定取决于应用
A_point = source_strength_per_point # / num_source_points # 暂时不除以点数，观察效果

print(f"将换能器表面离散化为 {num_source_points} 个点源。")

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

# === 修改 calculate_pressure_at_points ===
def calculate_pressure_at_points(field_points_xyz, source_positions, source_amplitudes, k_wavenumber):
    """Calculates complex pressure at specific field points from sources with individual amplitudes (with 1/r decay)."""
    num_field_points = field_points_xyz.shape[0]
    if num_field_points == 0:
        return np.array([], dtype=np.complex128)
        
    num_source_points = source_positions.shape[0]
    # Ensure source_amplitudes is a numpy array of the correct shape and complex type
    if not isinstance(source_amplitudes, np.ndarray) or source_amplitudes.shape[0] != num_source_points:
        # If a single amplitude is given, broadcast it
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

# === 控制运行哪些仿真 ===
run_single_transducer_sim = False
run_dual_active_sim = False
run_reflector_scan_sim = False
run_fixed_l_field_sim = True

# 全局 dB 范围和 epsilon
db_min_global = -40.0
epsilon_global = 1e-10

# === 1. 仿真单个换能器 ===
if run_single_transducer_sim:
    print("\n=== 1. 仿真单个换能器 ===")
    # 仿真网格 (XZ 平面, Y=0) 
    z_min_single, z_max_single, nz_single = 0.0001, 0.2, 300   # 轴向距离 (0 到 200 mm), 减少点数加速
    x_min_single, x_max_single, nx_single = -a*1.5, a*1.5, 150 # 横向距离 (-45 到 45 mm)
    z_grid_single = np.linspace(z_min_single, z_max_single, nz_single)
    x_grid_single = np.linspace(x_min_single, x_max_single, nx_single)
    X_single, Z_single = np.meshgrid(x_grid_single, z_grid_single) 

    X_flat_single = X_single.flatten()
    Z_flat_single = Z_single.flatten()
    num_field_points_single = len(X_flat_single)
    print(f"  仿真网格: X=[{x_min_single*1000:.0f},{x_max_single*1000:.0f}]mm ({nx_single} pts), Z=[{z_min_single*1000:.1f},{z_max_single*1000:.0f}]mm ({nz_single} pts)")

    P_total_flat_single = calculate_pressure_field_piston(X_flat_single, Z_flat_single, source_points, k, A_point)
    P_amplitude_single_raw = np.abs(P_total_flat_single)

    # --- 计算声压级 (dB) ---
    print("  计算声压级 (dB)...")
    p_max_single = np.max(P_amplitude_single_raw[P_amplitude_single_raw > 0]) # Find max ignoring zeros
    if p_max_single == 0: p_max_single = 1.0 
    P_amplitude_normalized_single = P_amplitude_single_raw / p_max_single
    SPL_dB_flat_single = 20 * np.log10(P_amplitude_normalized_single + epsilon_global)
    SPL_dB_flat_single[SPL_dB_flat_single < db_min_global] = db_min_global
    SPL_dB_single = SPL_dB_flat_single.reshape((nz_single, nx_single))

    # --- 可视化单个换能器声场 ---
    print("  开始绘图...")
    plt.figure(figsize=(12, 6))
    X_mm_single = X_single * 1000
    Z_mm_single = Z_single * 1000
    nf_dist_mm = nf_dist * 1000
    a_mm = a * 1000
    # 使用 pcolormesh 可能更清晰显示网格点
    im = plt.pcolormesh(Z_mm_single.T, X_mm_single.T, SPL_dB_single.T, cmap='jet', vmin=db_min_global, vmax=0, shading='gouraud')
    # im = plt.contourf(Z_mm_single.T, X_mm_single.T, SPL_dB_single.T, levels=np.linspace(db_min_global, 0, 31), cmap='jet', extend='min') 
    cbar = plt.colorbar(im, label='Relative SPL (dB)')
    cbar.set_ticks(np.linspace(db_min_global, 0, 5)) 
    plt.ylabel('x / mm') 
    plt.xlabel('z / mm') 
    plt.title(f'Single Transducer (Air, {f/1000:.1f}kHz, a={a_mm:.0f}mm)')
    plt.plot([0, 0], [-a_mm, a_mm], 'r-', lw=2, label=f'Transducer') 
    plt.axhline(0, color='white', linestyle=':', linewidth=0.8) 
    plt.axvline(nf_dist_mm, color='lime', linestyle=':', linewidth=1, label=f'Near Field Limit z≈{nf_dist_mm:.1f}mm') 
    plt.legend(loc='upper right', fontsize='small')
    plt.ylim(x_min_single*1000, x_max_single*1000)  
    plt.xlim(0, z_max_single * 1000)  
    plt.gca().set_facecolor('#EEEEEE') 
    plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    # Save before showing
    plt.savefig(os.path.join(img_dir, f'single_transducer_field_a{a_mm:.0f}_f{f/1000:.1f}.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

    # --- 绘制单换能器轴线上声压 (dB) ---
    x_center_index_single = np.argmin(np.abs(x_grid_single))
    on_axis_SPL_dB_single = SPL_dB_single[:, x_center_index_single]
    z_grid_mm_single = z_grid_single * 1000
    peaks_single_indices, _ = find_peaks(on_axis_SPL_dB_single)
    peak_z_single_mm = z_grid_mm_single[peaks_single_indices]
    peak_db_single = on_axis_SPL_dB_single[peaks_single_indices]

    plt.figure(figsize=(12, 5))
    plt.plot(z_grid_mm_single, on_axis_SPL_dB_single, linestyle='-', label='On-axis SPL')
    plt.plot(peak_z_single_mm, peak_db_single, "rx", markersize=8, mew=1.5, label=f'Maxima ({len(peaks_single_indices)} found)')
    plt.xlabel('Axial Distance z (mm)')
    plt.ylabel('Relative On-axis SPL (dB)')
    plt.title(f'Single Transducer On-axis SPL (Air, {f/1000:.1f}kHz, a={a_mm:.0f}mm)')
    plt.grid(True, linestyle=':')
    plt.axvline(nf_dist_mm, color='r', linestyle='--', label=f'Near Field Limit ≈ {nf_dist_mm:.1f} mm')
    plt.legend()
    plt.xlim(0, z_max_single * 1000)
    plt.ylim(db_min_global - 5, 5) 
    # Save before showing
    plt.savefig(os.path.join(img_dir, f'single_transducer_on_axis_spl_a{a_mm:.0f}_f{f/1000:.1f}.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

# === 2. 仿真两个主动面对面换能器 ===
if run_dual_active_sim:
    print("\n=== 2. 仿真两个主动面对面换能器 ===")
    separation_dual = nf_dist * 1.5 # 选择一个间距，例如 1.5 倍近场距离
    separation_dual_mm = separation_dual * 1000
    print(f"  设置间距 D = {separation_dual_mm:.1f} mm")

    # --- 定义仿真网格 (覆盖两个换能器之间及稍外区域) ---
    z_min_dual, z_max_dual, nz_dual = 0.0001, separation_dual * 1.1, 300 # 稍超出间距
    x_min_dual, x_max_dual, nx_dual = x_min_single, x_max_single, nx_single # 复用单换能器的X范围
    z_grid_dual = np.linspace(z_min_dual, z_max_dual, nz_dual)
    x_grid_dual = np.linspace(x_min_dual, x_max_dual, nx_dual)
    X_dual, Z_dual = np.meshgrid(x_grid_dual, z_grid_dual) 
    X_flat_dual = X_dual.flatten()
    Z_flat_dual = Z_dual.flatten()
    num_field_points_dual = len(X_flat_dual)
    print(f"  仿真网格: X=[{x_min_dual*1000:.0f},{x_max_dual*1000:.0f}]mm ({nx_dual} pts), Z=[{z_min_dual*1000:.1f},{z_max_dual*1000:.1f}]mm ({nz_dual} pts)")

    # --- 计算 P1 (来自 z=0 的 T1) ---
    print("  计算来自 T1 (z=0) 的声场 P1...")
    P_total_flat_t1_dual = calculate_pressure_field_piston(X_flat_dual, Z_flat_dual, source_points, k, A_point)

    # --- 计算 P2 (来自 z=separation 的 T2) ---
    print(f"  计算来自 T2 (z={separation_dual_mm:.1f}mm) 的声场 P2...")
    source_points_t2_dual = source_points + np.array([0, 0, separation_dual])
    P_total_flat_t2_dual = calculate_pressure_field_piston(X_flat_dual, Z_flat_dual, source_points_t2_dual, k, A_point)

    # --- 计算总声场和 SPL --- 
    print("  计算组合声场的声压级 (dB)...")
    P_total_flat_two_active = P_total_flat_t1_dual + P_total_flat_t2_dual
    P_amplitude_two_active_raw = np.abs(P_total_flat_two_active)
    
    p_max_two_active = np.max(P_amplitude_two_active_raw[P_amplitude_two_active_raw > 0])
    if p_max_two_active == 0: p_max_two_active = 1.0
    P_amplitude_normalized_two_active = P_amplitude_two_active_raw / p_max_two_active
    SPL_dB_flat_two_active = 20 * np.log10(P_amplitude_normalized_two_active + epsilon_global)
    SPL_dB_flat_two_active[SPL_dB_flat_two_active < db_min_global] = db_min_global
    SPL_dB_two_active = SPL_dB_flat_two_active.reshape((nz_dual, nx_dual))

    # --- 可视化两个面对面换能器声场 ---
    print("  开始绘图...")
    plt.figure(figsize=(12, 6))
    X_mm_dual = X_dual * 1000
    Z_mm_dual = Z_dual * 1000
    a_mm = a * 1000

    im = plt.pcolormesh(Z_mm_dual.T, X_mm_dual.T, SPL_dB_two_active.T, cmap='jet', vmin=db_min_global, vmax=0, shading='gouraud')
    cbar = plt.colorbar(im, label='Relative SPL (dB)')
    cbar.set_ticks(np.linspace(db_min_global, 0, 5))
    plt.ylabel('x / mm')
    plt.xlabel('z / mm')
    plt.title(f'Two Active Transducers (Air, D={separation_dual_mm:.1f}mm)')
    plt.plot([0, 0], [-a_mm, a_mm], 'r-', lw=2, label='Transducer 1 (z=0)')
    plt.plot([separation_dual_mm, separation_dual_mm], [-a_mm, a_mm], 'b-', lw=2, label=f'Transducer 2 (z={separation_dual_mm:.1f}mm)')
    plt.axhline(0, color='white', linestyle=':', linewidth=0.8)
    plt.legend(loc='upper right', fontsize='small')
    plt.ylim(x_min_dual*1000, x_max_dual*1000)
    plt.xlim(0, z_max_dual * 1000)
    plt.gca().set_facecolor('#EEEEEE')
    plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    # Save before showing
    plt.savefig(os.path.join(img_dir, f'two_active_transducers_field_D{separation_dual_mm:.1f}.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

    # --- 绘制轴线上声压 --- 
    x_center_index_dual = np.argmin(np.abs(x_grid_dual))
    on_axis_SPL_dB_two_active = SPL_dB_two_active[:, x_center_index_dual]
    z_grid_mm_dual = z_grid_dual * 1000
    peaks_dual_indices, _ = find_peaks(on_axis_SPL_dB_two_active)
    peak_z_dual_mm = z_grid_mm_dual[peaks_dual_indices]
    peak_db_dual = on_axis_SPL_dB_two_active[peaks_dual_indices]

    plt.figure(figsize=(12, 5))
    plt.plot(z_grid_mm_dual, on_axis_SPL_dB_two_active, linestyle='-', label='On-axis SPL')
    plt.plot(peak_z_dual_mm, peak_db_dual, "rx", markersize=8, mew=1.5, label=f'Maxima ({len(peaks_dual_indices)} found)')
    plt.xlabel('Axial Distance z (mm)')
    plt.ylabel('Relative On-axis SPL (dB)')
    plt.title(f'Two Active Transducers On-axis SPL (Air, D={separation_dual_mm:.1f}mm)')
    plt.grid(True, linestyle=':')
    plt.axvline(0, color='r', linestyle='--', label='T1')
    plt.axvline(separation_dual_mm, color='b', linestyle='--', label='T2')
    plt.legend()
    plt.xlim(0, z_max_dual * 1000)
    plt.ylim(db_min_global - 5, 5)
    # Save before showing
    plt.savefig(os.path.join(img_dir, f'two_active_transducers_on_axis_spl_D{separation_dual_mm:.1f}.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

# === 3. 仿真反射面 (z=l) 处声压随距离 l 的变化 (镜像源 方法) ===
if run_reflector_scan_sim:
    print("\n=== 3. 仿真反射面 (z=l) 处声压随距离 l 的变化 (镜像源 方法) ===")

    # --- 定义 l 的范围 ---
    l_min_m = 0.0001 # 0.1 mm
    l_max_m = 0.150  # 150 mm
    num_l_points = 500 # 增加点数以获得更精细的曲线
    l_values = np.linspace(l_min_m, l_max_m, num_l_points)
    l_values_mm = l_values * 1000
    print(f"扫描距离 l 从 {l_min_m*1000:.1f} mm 到 {l_max_m*1000:.1f} mm ({num_l_points} 点)")

    # --- 准备 ---
    # 源 T1 的参数 (位于 z=0)
    source_points_t1 = source_points
    source_amplitudes_t1 = np.full(num_source_points, A_point, dtype=np.complex128)

    # 存储结果的数组
    P1_direct_at_target_l = np.zeros(num_l_points, dtype=np.complex128)
    P2_image_at_target_l = np.zeros(num_l_points, dtype=np.complex128)
    
    # --- 预计算源点的角频率和波长 ---
    omega = 2 * np.pi * f
    lambda_half_mm = (lambda_ / 2) * 1000
    
    # --- 循环计算每个距离 l 处的声场 (使用镜像源) ---
    print("循环计算 P1_direct 和 P2_image 在所有 z=l 处的值 (镜像源方法)...")
    start_scan_time = time.time()

    num_reflections_pairs = 10 

    for i, l_val in enumerate(l_values):
        # 当前的目标点 (0,0,l) - 反射面上的中心点
        target_point_l = np.array([[0, 0, l_val]]) 

        # 1. 计算直达波 P1_direct 在目标点的值 (源在 z=0)
        P1_direct_at_target_l[i] = calculate_pressure_at_points(
            target_point_l, source_points_t1, source_amplitudes_t1, k
        )[0]

        # 2. 定义镜像源的位置 (位于 z=2l)
        source_points_t2_image = source_points_t1 + np.array([0, 0, 2 * l_val])
        # 镜像源振幅 (假设刚性反射，相位反转，即乘以 -1)
        # 注意：这里的振幅 A_point 是针对原始源的，镜像源也用它，但总效果带负号
        # 我们将 P2_image 计算出来，然后在总和时体现负号（或者将振幅设为 -A_point）
        # 为保持与 Section 4 一致，直接使用 P1 - P2' (假设 P2' 是镜像源贡献)
        # 但更清晰的是 P = P1 + (-P2_image_calc) 
        source_amplitudes_t2_image = np.full(num_source_points, A_point, dtype=np.complex128)

        # 3. 计算镜像波 P2_image 在目标点的值
        P2_image_at_target_l[i] = calculate_pressure_at_points(
            target_point_l, source_points_t2_image, source_amplitudes_t2_image, k
        )[0]
        
        # 打印进度
        if (i + 1) % max(1, num_l_points // 10) == 0: 
             elapsed = time.time() - start_scan_time
             percent_done = ((i + 1) / num_l_points) * 100
             eta = (elapsed / percent_done) * (100 - percent_done) if percent_done > 0 else 0
             print(f"  l-scan (镜像源)进度: {percent_done:.1f}% (预计剩余时间: {eta:.1f}s)")

    # 计算各波振幅
    P1_amplitude_at_l = np.abs(P1_direct_at_target_l) # 直接波振幅
    P2_image_amplitude_at_l = np.abs(P2_image_at_target_l) # 镜像波贡献振幅 (未乘-1)
    print(f"  l-scan (镜像源) 计算完成，耗时 {time.time() - start_scan_time:.2f} 秒。")

    # --- 计算总声压 P_total = P1_direct + P2_image (刚性反射) ---
    print("  计算复振幅总和 P_total = P1_direct + P2_image (考虑刚性反射)..")
    P_total_at_l = P1_direct_at_target_l + P2_image_at_target_l  # P1 + (+1)*P2_image
    P_amplitude_at_l = np.abs(P_total_at_l)  # 取模得到总振幅
    
    # --- 计算相位 (可选) ---
    # P1_phase_at_l = np.angle(P1_direct_at_target_l, deg=True)
    # P2_image_phase_at_l = np.angle(P2_image_at_target_l, deg=True)
    # P_total_phase_at_l = np.angle(P_total_at_l, deg=True)
    
    # --- 绘图 0: 检查 |P1_direct| 和 |P2_image| ---
    print("绘制 |P1_direct| 和 |P2_image| 随 l 的变化...")
    plt.figure(figsize=(15, 6))
    plt.plot(l_values_mm, P1_amplitude_at_l, linestyle='-', label='|P1(l)| (直达波振幅)')
    plt.plot(l_values_mm, P2_image_amplitude_at_l, linestyle='--', label="|P2_image(l)| (镜像波贡献振幅)")
    plt.plot(l_values_mm, P_amplitude_at_l, linestyle='-.', color='green', 
             label="|P1+P2_image| (总振幅 - 镜像源法)")
    
    plt.xlabel('反射面距离 l (mm)')
    plt.ylabel('相对线性振幅')
    plt.title('直达波、镜像波及总波振幅随距离的变化 (镜像源法)')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.xlim(0, l_max_m * 1000)
    plt.ylim(bottom=0) 
    
    # 添加理论λ/2标记
    y_max = plt.ylim()[1]
    for i in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
        plt.axvline(i * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
        if i > 0:
            plt.text(i * lambda_half_mm, y_max * 0.95, f'{i}λ/2', 
                    fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)
    
    # Save before showing
    plt.savefig(os.path.join(img_dir, 'reflector_scan_p1_p2_image_vs_l.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

    # --- 在线性总振幅上寻找极大值 ---
    print("在线性总振幅曲线上查找极大值...")
    min_peak_distance = int(num_l_points * lambda_half_mm / (l_max_m * 1000) * 0.8)
    peaks_l_indices, properties = find_peaks(P_amplitude_at_l, height=None, distance=min_peak_distance) 
    peak_l_mm = l_values_mm[peaks_l_indices]
    peak_l_amp = P_amplitude_at_l[peaks_l_indices] 
    print(f"  找到 {len(peaks_l_indices)} 个极大值 (在线性振幅上, distance={min_peak_distance})。")

    # 估算平均峰间距
    if len(peak_l_mm) > 1:
        avg_peak_spacing = np.mean(np.diff(peak_l_mm))
        print(f"  估算平均峰间距: {avg_peak_spacing:.2f} mm (理论 λ/2 ≈ {lambda_half_mm:.2f} mm)")
        print(f"  峰间距与理论值的比率: {avg_peak_spacing/lambda_half_mm:.2f}")
    elif len(peak_l_mm) == 1:
         print(f"  只找到一个极大值，无法计算间距。")
    else:
         print(f"  未找到极大值。")

    # --- 绘图 1: Linear Amplitude vs l (Total) --- (绘图内容已调整)
    print("绘制 Linear Amplitude (Total) vs l 曲线图 (镜像源法)...")
    plt.figure(figsize=(15, 6))
    plt.plot(l_values_mm, P_amplitude_at_l, linestyle='-', label='总振幅 |P1+P2_image| (镜像源法)')
    plt.plot(peak_l_mm, peak_l_amp, "rx", markersize=8, mew=1.5, label=f'极大值 ({len(peaks_l_indices)} 个)') 
    plt.xlabel('反射面距离 l (mm)')
    plt.ylabel('相对线性振幅')
    plt.title(f'反射面处的总振幅随距离变化 (镜像源法, 空气, {f/1000:.1f}kHz, a={a*1000:.0f}mm)')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.xlim(0, l_max_m * 1000)
    plt.ylim(bottom=0) 
    
    # 添加理论λ/2标记
    y_max = plt.ylim()[1]
    for i in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
        plt.axvline(i * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
        if i > 0:
            plt.text(i * lambda_half_mm, y_max * 0.95, f'{i}λ/2', 
                    fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)
    
    # Save before showing
    plt.savefig(os.path.join(img_dir, 'reflector_scan_amp_vs_l.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

    # --- 计算并绘图 2: SPL (dB) vs l (Total) --- (逻辑不变)
    print("计算并绘制 SPL (dB) vs l 曲线图 (镜像源法)...")
    p_max_overall = np.max(P_amplitude_at_l[P_amplitude_at_l > 0])
    if p_max_overall == 0: p_max_overall = 1.0
    SPL_dB_at_l = 20 * np.log10(P_amplitude_at_l / p_max_overall + epsilon_global)
    SPL_dB_at_l[SPL_dB_at_l < db_min_global] = db_min_global
    peak_l_db = SPL_dB_at_l[peaks_l_indices] 

    plt.figure(figsize=(15, 6))
    plt.plot(l_values_mm, SPL_dB_at_l, linestyle='-', label='反射面处声压级')
    plt.plot(peak_l_mm, peak_l_db, "rx", markersize=8, mew=1.5, label=f'极大值 ({len(peaks_l_indices)} 个)')
    plt.xlabel('反射面距离 l (mm)')
    plt.ylabel('相对声压级 (dB)')
    plt.title(f'反射面处声压级随距离变化 (镜像源法, 空气, {f/1000:.1f}kHz, a={a*1000:.0f}mm)')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.xlim(0, l_max_m * 1000)
    plt.ylim(db_min_global - 5, 5) 
    
    # 添加理论λ/2标记
    for i in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
        plt.axvline(i * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
        if i > 0:
            plt.text(i * lambda_half_mm, db_min_global, f'{i}λ/2', 
                    fontsize=8, ha='center', va='bottom', rotation=90, alpha=0.7)
    
    # Save before showing
    plt.savefig(os.path.join(img_dir, 'reflector_scan_spl_vs_l.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

# === 4. 仿真固定距离 l 下，主动换能器与反射面之间的二维声场 ===
if run_fixed_l_field_sim:
    print("\n=== 4. 仿真固定距离 l 下，主动换能器与反射面之间的二维声场 ===")
    
    # --- 定义要绘制的多个固定间距 l ---
    # 使用计算出的近场距离 nf_dist 来定义 l
    l_fixed_values = [nf_dist / 4, nf_dist / 2, nf_dist, nf_dist * 1.5, nf_dist * 2, nf_dist * 3] # 米单位, 增加了范围
    print(f"  将绘制以下间距的声场图: {[f'{l*1000:.1f}' for l in l_fixed_values]} mm")

    for l_fixed in l_fixed_values:
        l_fixed_mm = l_fixed * 1000
        print(f"\n--- 计算固定间距 l = {l_fixed_mm:.1f} mm ---")

        # --- 定义仿真网格 (0 到 l_fixed) ---
        z_min_fixed, z_max_fixed, nz_fixed = 0.0001, l_fixed, 200 # 轴向网格点数
        x_min_fixed, x_max_fixed, nx_fixed = -a*1.5, a*1.5, 150 # 横向范围和点数
        z_grid_fixed = np.linspace(z_min_fixed, z_max_fixed, nz_fixed)
        x_grid_fixed = np.linspace(x_min_fixed, x_max_fixed, nx_fixed)
        X_fixed, Z_fixed = np.meshgrid(x_grid_fixed, z_grid_fixed) 
        X_flat_fixed = X_fixed.flatten()
        Z_flat_fixed = Z_fixed.flatten()
        num_field_points_fixed = len(X_flat_fixed)
        print(f"  仿真网格: X=[{x_min_fixed*1000:.0f},{x_max_fixed*1000:.0f}]mm ({nx_fixed} pts), Z=[{z_min_fixed*1000:.1f},{z_max_fixed*1000:.1f}]mm ({nz_fixed} pts)")

        # --- 计算声场 P1 (来自 z=0 的 T1) ---
        print(f"  计算来自 T1 (z=0) 的声场 P1...")
        P_total_flat_t1_fixed = calculate_pressure_field_piston(
            X_flat_fixed, Z_flat_fixed, source_points, k, A_point
        )

        # --- 计算声场 P2' (来自 z=2l_fixed 的镜像源 T2') ---
        print(f"  计算来自镜像源 T2' (z={2*l_fixed*1000:.1f} mm) 的声场 P2'...")
        source_points_t2_image_fixed = source_points + np.array([0, 0, 2 * l_fixed])
        P_total_flat_t2_image_fixed = calculate_pressure_field_piston(
            X_flat_fixed, Z_flat_fixed, source_points_t2_image_fixed, k, A_point
        )

        # --- 计算总声场 P_total = P1 + P2' ---
        P_total_flat_combined_fixed = P_total_flat_t1_fixed + P_total_flat_t2_image_fixed
        P_amplitude_combined_fixed_raw = np.abs(P_total_flat_combined_fixed)

        # --- 计算声压级 (dB), 相对于当前场最大值 ---
        print(f"  计算组合声场的声压级 (dB)...")
        # 查找当前场的最大值（忽略0）用于归一化
        p_max_current_field = np.max(P_amplitude_combined_fixed_raw[P_amplitude_combined_fixed_raw > 0])
        if p_max_current_field == 0: p_max_current_field = 1.0
        P_amplitude_normalized_combined_fixed = P_amplitude_combined_fixed_raw / p_max_current_field
        SPL_dB_flat_combined_fixed = 20 * np.log10(P_amplitude_normalized_combined_fixed + epsilon_global)
        SPL_dB_flat_combined_fixed[SPL_dB_flat_combined_fixed < db_min_global] = db_min_global
        SPL_dB_combined_fixed = SPL_dB_flat_combined_fixed.reshape((nz_fixed, nx_fixed))

        # --- 可视化组合声场 ---
        print(f"  绘制 l = {l_fixed_mm:.1f} mm 的二维声场图...")
        plt.figure(figsize=(max(8, 10 * (l_fixed / 0.1)), 6)) # 根据 l 调整宽度

        X_mm_fixed = X_fixed * 1000
        Z_mm_fixed = Z_fixed * 1000
        a_mm = a * 1000

        im = plt.pcolormesh(Z_mm_fixed.T, X_mm_fixed.T, SPL_dB_combined_fixed.T, 
                          cmap='jet', vmin=db_min_global, vmax=0, shading='gouraud')
        cbar = plt.colorbar(im, label='Relative SPL (dB)')
        cbar.set_ticks(np.linspace(db_min_global, 0, 5))

        plt.ylabel('x / mm')
        plt.xlabel('z / mm')
        plt.title(f'声场分布：换能器 (z=0) 与 反射面 (z={l_fixed_mm:.1f} mm)')

        # 标记换能器和反射面
        plt.plot([0, 0], [-a_mm, a_mm], 'r-', lw=2, label=f'Transducer')
        plt.axvline(l_fixed_mm, color='black', linestyle='-', lw=2, label=f'Reflector')
        plt.axhline(0, color='white', linestyle=':', linewidth=0.8)

        plt.legend(loc='upper right', fontsize='small')
        plt.ylim(x_min_fixed*1000, x_max_fixed*1000)
        plt.xlim(0, l_fixed_mm)
        plt.gca().set_facecolor('#EEEEEE')
        plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        # Save before showing
        plt.savefig(os.path.join(img_dir, f'fixed_l_field_l{l_fixed_mm:.1f}.pdf'), format='pdf', bbox_inches='tight', dpi=150)
        plt.show()

        # 可选的轴线图代码仍然注释掉
        # ... （省略） ...

print("\n仿真结束。")
