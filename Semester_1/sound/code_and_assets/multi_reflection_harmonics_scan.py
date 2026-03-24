# sound/py/multi_reflection_harmonics_scan.py

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import os
# import torch # No longer needed for core calculation

# --- Set PyTorch Device (commented out) ---
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("PyTorch using MPS device (GPU)")
# elif torch.cuda.is_available():
#      device = torch.device("cuda") # Should not happen on Mac usually
#      print("PyTorch using CUDA device")
# else:
#     device = torch.device("cpu")
#     print("PyTorch using CPU device")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# Create image directory if it doesn't exist
img_dir = 'sound/img'
os.makedirs(img_dir, exist_ok=True)

# === Calculation Function (Copied from field.py) ===
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
c = 346.0
f1 = 36981.0 # Fundamental frequency
a = 0.0191 # Radius (1.91cm, Diameter 3.82cm)

f2 = 2 * f1 # 2nd Harmonic
f3 = 3 * f1 # 3rd Harmonic

lambda1 = c / f1
lambda2 = c / f2
lambda3 = c / f3

k1_np = 2 * np.pi / lambda1
k2_np = 2 * np.pi / lambda2
k3_np = 2 * np.pi / lambda3

# k1_pt = torch.tensor(k1_np, dtype=torch.complex64, device=device) # No longer needed
# k2_pt = torch.tensor(k2_np, dtype=torch.complex64, device=device) # No longer needed
# k3_pt = torch.tensor(k3_np, dtype=torch.complex64, device=device) # No longer needed

print(f"介质: 空气, c = {c} m/s")
print(f"换能器半径 a = {a*1000:.1f} mm")
print(f"基频 f1 = {f1/1000:.3f} kHz (λ1 = {lambda1*1000:.4f} mm)")
print(f"二次谐波 f2 = {f2/1000:.3f} kHz (λ2 = {lambda2*1000:.4f} mm)")
print(f"三次谐波 f3 = {f3/1000:.3f} kHz (λ3 = {lambda3*1000:.4f} mm)")

nf_dist1 = a**2 / lambda1
print(f"基频近场距离 大约延伸至 a^2/λ1 ≈ {nf_dist1*1000:.1f} mm")

# --- 换能器表面离散化 (Same for all frequencies) ---
N_points_per_radius = 50
source_points_list = []
xs_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1, dtype=np.float32)
ys_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1, dtype=np.float32)
for xs in xs_grid:
    for ys in ys_grid:
        if xs**2 + ys**2 <= a**2:
            source_points_list.append(np.array([xs, ys, 0], dtype=np.float32))
source_points_orig_np = np.array(source_points_list, dtype=np.float32)
num_source_points = len(source_points_orig_np)
# source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device) # No longer needed
print(f"将换能器表面离散化为 {num_source_points} 个点源 (Using NumPy)")

# --- 多重反射参数 ---
N_pairs = 50
print(f"镜像源对数 N_pairs = {N_pairs}")

# === 固定反射系数 和 移除基频衰减 ===
R_val = 0.95
phi_R_val = 0.0
# gamma1_base = 0.2 # Base attenuation for fundamental frequency (Np/m) - REMOVED

print(f"固定反射系数 R = {R_val}")
print(f"固定反射相位 phi_R = {phi_R_val} rad")
print("衰减系数 gamma 不再使用") # Updated print

# --- 计算固定的复数反射系数 --- (NumPy)
R_complex_np = (R_val * np.exp(1j * phi_R_val)).astype(np.complex128)
R_0_np = R_complex_np
R_l_np = R_complex_np
# R_0_pt = torch.tensor(R_complex_np, dtype=torch.complex64, device=device) # No longer needed
# R_l_pt = torch.tensor(R_complex_np, dtype=torch.complex64, device=device) # No longer needed
print(f"固定复数反射系数 Complex R = {R_complex_np:.2f}")

# === 计算频率相关衰减系数 (gamma ~ f^2) - REMOVED ===
# gamma1_pt = torch.tensor(gamma1_base, dtype=torch.float32, device=device) # No longer needed
# gamma2_pt = gamma1_pt * (f2 / f1)**2 # No longer needed
# gamma3_pt = gamma1_pt * (f3 / f1)**2 # No longer needed
# print(f"二次谐波衰减 gamma2 = {gamma2_pt.item():.2f} Np/m") # No longer needed
# print(f"三次谐波衰减 gamma3 = {gamma3_pt.item():.2f} Np/m") # No longer needed

# === 定义谐波场景 ===
# (relative_amplitude_f1, relative_amplitude_f2, relative_amplitude_f3, scenario_label)
harmonic_scenarios = [
    #(1.0, 0.0,  0.15,  "f1_plus_0p15f3"),
    #(1.0, 0.2, 0.0,  "f1_plus_0p2f2"),
    (1.0, 0.5, 0.00, "f1_plus_0p5f2"),
    (1.0, 0.4, 0.00, "f1_plus_0p4f2"),
]

# === 仿真参数 ===
l_min_m = 0.0001
l_max_m = 0.050
num_l_points = 5000
l_values_np = np.linspace(l_min_m, l_max_m, num_l_points, dtype=np.float32)
l_values_mm = l_values_np * 1000
print(f"扫描距离 l 从 {l_min_m*1000:.1f} mm 到 {l_max_m*1000:.1f} mm ({num_l_points} 点)")

# 预计算常量 和 绘图设置
lambda1_half_mm = (lambda1 / 2) * 1000 # Use fundamental wavelength for reference lines
db_min_global = -40.0
epsilon_global = np.float32(1e-10)
near_field_limit_mm = 3.0
print(f"将忽略 l < {near_field_limit_mm} mm 范围内的极大值")

# === 循环计算不同谐波场景 ===
print("\n=== 开始仿真不同谐波场景 ===")
for amp1_rel, amp2_rel, amp3_rel, scenario_label in harmonic_scenarios:
    print(f"\n===== 计算场景: {scenario_label} (A1:{amp1_rel:.2f}, A2:{amp2_rel:.2f}, A3:{amp3_rel:.2f}) =====")

    # Initialize total pressure array for this scenario
    P_total_scenario_at_l_np = np.zeros(num_l_points, dtype=np.complex128)

    # Base source strengths per point for each harmonic component (NumPy)
    A1_point_np = np.complex128(1.0 * amp1_rel)
    A2_point_np = np.complex128(1.0 * amp2_rel)
    A3_point_np = np.complex128(1.0 * amp3_rel)
    # A1_point_pt = torch.tensor(1.0 * amp1_rel, dtype=torch.complex64, device=device) # No longer needed
    # A2_point_pt = torch.tensor(1.0 * amp2_rel, dtype=torch.complex64, device=device) # No longer needed
    # A3_point_pt = torch.tensor(1.0 * amp3_rel, dtype=torch.complex64, device=device) # No longer needed

    # Pre-calculate base amplitude arrays for all source points for each harmonic (NumPy)
    base_amplitude1_per_point_np = np.full(num_source_points, A1_point_np, dtype=np.complex128)
    base_amplitude2_per_point_np = np.full(num_source_points, A2_point_np, dtype=np.complex128)
    base_amplitude3_per_point_np = np.full(num_source_points, A3_point_np, dtype=np.complex128)
    # base_amplitude1_per_point_pt = torch.full((num_source_points,), A1_point_pt.item(), dtype=torch.complex64, device=device) # No longer needed
    # base_amplitude2_per_point_pt = torch.full((num_source_points,), A2_point_pt.item(), dtype=torch.complex64, device=device) # No longer needed
    # Base source strengths per point for each harmonic component
    A1_point_pt = torch.tensor(1.0 * amp1_rel, dtype=torch.complex64, device=device)
    A2_point_pt = torch.tensor(1.0 * amp2_rel, dtype=torch.complex64, device=device)
    A3_point_pt = torch.tensor(1.0 * amp3_rel, dtype=torch.complex64, device=device)

    # Pre-calculate base amplitude tensors for all source points for each harmonic
    base_amplitude1_per_point_pt = torch.full((num_source_points,), A1_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude2_per_point_pt = torch.full((num_source_points,), A2_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude3_per_point_pt = torch.full((num_source_points,), A3_point_pt.item(), dtype=torch.complex64, device=device)


    print(f"  循环计算 l 处声压 (N={N_pairs}, R={R_val:.2f}, phi={phi_R_val:.2f}, gamma1={gamma1_base:.2f}) 使用 PyTorch...")
    start_scan_time = time.time()

    for i, l_val_np in enumerate(l_values_np):
        if l_val_np < 1e-9: continue
        target_point_l_pt = torch.tensor([[0, 0, l_val_np]], dtype=torch.float32, device=device) # Shape (1, 3)

        # --- Optimized Calculation for single l (Applied to each harmonic component) ---
        # We need to build the source list for *all* harmonics together if we want one calculation call?
        # No, the k and gamma are different. We must calculate each harmonic separately and sum.

        P_total_at_this_l_pt = torch.tensor(0.0, dtype=torch.complex64, device=device)

        # --- Calculate Contribution from Harmonic 1 (if amplitude > 0) ---
        if amp1_rel > 1e-9:
            all_source_positions_list = [source_points_orig_pt]
            all_source_amplitudes_list = [base_amplitude1_per_point_pt]
            current_amplitude_factor_pos_pt = R_l_pt.clone()
            current_amplitude_factor_neg_pt = (R_l_pt * R_0_pt).clone()
            for n in range(1, N_pairs + 1):
                z_offset_pos = 2.0 * n * l_val_np
                amp_factor_pos = current_amplitude_factor_pos_pt
                src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_plus_pt)
                all_source_amplitudes_list.append(base_amplitude1_per_point_pt * amp_factor_pos)
                current_amplitude_factor_pos_pt *= (R_0_pt * R_l_pt)

                z_offset_neg = -2.0 * n * l_val_np
                amp_factor_neg = current_amplitude_factor_neg_pt
                src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_neg_pt)
                all_source_amplitudes_list.append(base_amplitude1_per_point_pt * amp_factor_neg)
                current_amplitude_factor_neg_pt *= (R_l_pt * R_0_pt)

            all_src_pt = torch.cat(all_source_positions_list, dim=0)
            all_amp_pt = torch.cat(all_source_amplitudes_list, dim=0)
            P_total_at_this_l_pt += calculate_pressure_at_points_pytorch(target_point_l_pt, all_src_pt, all_amp_pt, k1_pt, gamma1_pt)

        # --- Calculate Contribution from Harmonic 2 (if amplitude > 0) ---
        if amp2_rel > 1e-9:
            # Rebuild source/amplitude lists for f2 (amplitude factor is different)
            all_source_positions_list = [source_points_orig_pt]
            all_source_amplitudes_list = [base_amplitude2_per_point_pt]
            current_amplitude_factor_pos_pt = R_l_pt.clone()
            current_amplitude_factor_neg_pt = (R_l_pt * R_0_pt).clone()
            for n in range(1, N_pairs + 1):
                z_offset_pos = 2.0 * n * l_val_np
                amp_factor_pos = current_amplitude_factor_pos_pt
                src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_plus_pt)
                all_source_amplitudes_list.append(base_amplitude2_per_point_pt * amp_factor_pos)
                current_amplitude_factor_pos_pt *= (R_0_pt * R_l_pt)

                z_offset_neg = -2.0 * n * l_val_np
                amp_factor_neg = current_amplitude_factor_neg_pt
                src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_neg_pt)
                all_source_amplitudes_list.append(base_amplitude2_per_point_pt * amp_factor_neg)
                current_amplitude_factor_neg_pt *= (R_l_pt * R_0_pt)

            all_src_pt = torch.cat(all_source_positions_list, dim=0)
            all_amp_pt = torch.cat(all_source_amplitudes_list, dim=0)
            P_total_at_this_l_pt += calculate_pressure_at_points_pytorch(target_point_l_pt, all_src_pt, all_amp_pt, k2_pt, gamma2_pt)

        # --- Calculate Contribution from Harmonic 3 (if amplitude > 0) ---
        if amp3_rel > 1e-9:
            # Rebuild source/amplitude lists for f3
            all_source_positions_list = [source_points_orig_pt]
            all_source_amplitudes_list = [base_amplitude3_per_point_pt]
            current_amplitude_factor_pos_pt = R_l_pt.clone()
            current_amplitude_factor_neg_pt = (R_l_pt * R_0_pt).clone()
            for n in range(1, N_pairs + 1):
                z_offset_pos = 2.0 * n * l_val_np
                amp_factor_pos = current_amplitude_factor_pos_pt
                src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_plus_pt)
                all_source_amplitudes_list.append(base_amplitude3_per_point_pt * amp_factor_pos)
                current_amplitude_factor_pos_pt *= (R_0_pt * R_l_pt)

                z_offset_neg = -2.0 * n * l_val_np
                amp_factor_neg = current_amplitude_factor_neg_pt
                src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_neg_pt)
                all_source_amplitudes_list.append(base_amplitude3_per_point_pt * amp_factor_neg)
                current_amplitude_factor_neg_pt *= (R_l_pt * R_0_pt)

            all_src_pt = torch.cat(all_source_positions_list, dim=0)
            all_amp_pt = torch.cat(all_source_amplitudes_list, dim=0)
            P_total_at_this_l_pt += calculate_pressure_at_points_pytorch(target_point_l_pt, all_src_pt, all_amp_pt, k3_pt, gamma3_pt)

        # Store total result for this l
        P_total_scenario_at_l_np[i] = P_total_at_this_l_pt.cpu().numpy()

        # Progress update
        if (i + 1) % max(1, num_l_points // 10) == 0:
             elapsed = time.time() - start_scan_time
             percent_done = ((i + 1) / num_l_points) * 100
             eta = (elapsed / percent_done) * (100 - percent_done) if percent_done > 0 else 0
             processed_points = i + 1
             points_per_sec = processed_points / elapsed if elapsed > 0 else 0
             print(f"    l-scan ({scenario_label}) 进度: {percent_done:.1f}% ({processed_points}/{num_l_points}), Speed: {points_per_sec:.1f} l/s, ETA: {eta:.1f}s")

    print(f"  l-scan ({scenario_label}) 计算完成，耗时 {time.time() - start_scan_time:.2f} 秒。")

    # --- 分析当前场景的振幅和 SPL ---
    P_amplitude = np.abs(P_total_scenario_at_l_np)
    # Use a fixed reference pressure (e.g., max amplitude from f1_only case?)
    # For simplicity, let's use max of current curve for relative SPL plot
    p_ref = np.max(P_amplitude)
    if p_ref <= 0 or not np.isfinite(p_ref):
         print(f"警告: 场景 {scenario_label} 计算得到无效的最大声压 {p_ref}，跳过 SPL 计算。")
         SPL_dB = np.full_like(P_amplitude, db_min_global)
    else:
         SPL_dB = 20 * np.log10(P_amplitude / p_ref + epsilon_global)
         SPL_dB[SPL_dB < db_min_global] = db_min_global

    # --- 查找极大值 --- (Find all peaks)
    peaks_indices, _ = find_peaks(P_amplitude, height=0, distance=1)
    all_valid_peaks_indices = []
    if len(peaks_indices) > 0:
        peak_l_values_mm = l_values_mm[peaks_indices]
        for idx, peak_l in zip(peaks_indices, peak_l_values_mm):
            if peak_l < near_field_limit_mm:
                continue
            all_valid_peaks_indices.append(idx)
        print(f"  找到 {len(peaks_indices)} 个原始峰值. 近场(l<{near_field_limit_mm}mm)排除后找到 {len(all_valid_peaks_indices)} 个极大值.")
        if len(all_valid_peaks_indices) > 0:
             valid_peak_l_values = l_values_mm[all_valid_peaks_indices]
             print(f"    极大值位置 (l in mm): {np.round(valid_peak_l_values, 2)}")
    else:
        print("  未找到明显峰值。")
    all_valid_peaks_indices = np.array(all_valid_peaks_indices, dtype=int)

    # --- 绘制当前场景的独立图 ---
    print(f"  绘制场景 {scenario_label} 的独立图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Update title to include scenario
    fig.suptitle(f'中心点声压 vs l (场景: {scenario_label}, R={R_val:.2f}, $\gamma_1$={gamma1_base:.1f}, N={N_pairs})', fontsize=14)

    # 子图1: 线性振幅与极大值标记
    ax1.plot(l_values_mm, P_amplitude, linestyle='-', label='总振幅', color='blue')
    if len(all_valid_peaks_indices) > 0:
        ax1.plot(l_values_mm[all_valid_peaks_indices], P_amplitude[all_valid_peaks_indices],
                 'kx', markersize=6, mew=1.5, linestyle='None', label='极大值')
    ax1.set_ylabel('相对声压振幅')
    ax1.grid(True, linestyle=':')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    y_max_amp = ax1.get_ylim()[1]
    # Reference lines based on fundamental frequency
    for k in range(int(l_max_m * 1000 / lambda1_half_mm) + 1):
        ax1.axvline(k * lambda1_half_mm, color='gray', linestyle=':', alpha=0.5)
        if k > 0:
            ax1.text(k * lambda1_half_mm, y_max_amp * 0.95, f'{k}λ$_1$/2', fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)

    # 子图2: 相对声压级
    ax2.plot(l_values_mm, SPL_dB, linestyle='-', label='相对声压级 (dB)', color='green')
    if len(all_valid_peaks_indices) > 0:
         ax2.plot(l_values_mm[all_valid_peaks_indices], SPL_dB[all_valid_peaks_indices],
                  'kx', markersize=6, mew=1.5, linestyle='None')
    ax2.set_xlabel('反射面距离 l (mm)')
    ax2.set_ylabel('相对声压级 (dB, 参考最大值)')
    ax2.grid(True, linestyle=':')
    ax2.set_ylim(db_min_global - 5, 5)
    # Reference lines based on fundamental frequency
    for k in range(int(l_max_m * 1000 / lambda1_half_mm) + 1):
        ax2.axvline(k * lambda1_half_mm, color='gray', linestyle=':', alpha=0.5)
        if k > 0:
             ax2.text(k * lambda1_half_mm, db_min_global, f'{k}λ$_1$/2', fontsize=8, ha='center', va='bottom', rotation=90, alpha=0.7)

    # Finalize and save plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Update filename to include scenario
    save_path = os.path.join(img_dir, f'multi_reflection_harmonics_scan_{scenario_label}_R{R_val:.2f}_g1{gamma1_base:.1f}_N{N_pairs}.pdf')
    try:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"  图表已保存至: {save_path}")
    except Exception as e:
        print(f"  保存图表失败: {e}")
    plt.close(fig)

print("\n多重反射与谐波仿真结束。") 