# sound/py/multi_reflection_phase_scan.py

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import os
import torch # Import PyTorch

# --- Set PyTorch Device ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("PyTorch using MPS device (GPU)")
elif torch.cuda.is_available():
     device = torch.device("cuda") # Should not happen on Mac usually
     print("PyTorch using CUDA device")
else:
    device = torch.device("cpu")
    print("PyTorch using CPU device")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# Create image directory if it doesn't exist
img_dir = 'sound/img'
os.makedirs(img_dir, exist_ok=True)

# === PyTorch Calculation Function ===
def calculate_pressure_at_points_pytorch(field_points_pt, source_positions_pt, source_amplitudes_pt, k_pt):
    """Calculates complex pressure at specific field points using PyTorch (vectorized)."""
    # field_points_pt: (Nf, 3)
    # source_positions_pt: (Ns, 3)
    # source_amplitudes_pt: (Ns,) or scalar

    field_points_expanded = field_points_pt.unsqueeze(1) # (Nf, 1, 3)
    source_positions_expanded = source_positions_pt.unsqueeze(0) # (1, Ns, 3)

    diff = field_points_expanded - source_positions_expanded # (Nf, Ns, 3)
    # Ensure dim=-1 is used for sum, regardless of specific dimension index
    r_sq = torch.sum(torch.square(diff), dim=-1) # (Nf, Ns)
    r = torch.sqrt(r_sq)
    r = torch.clamp(r, min=1e-9)
    r_complex = r.to(torch.complex64) # (Nf, Ns)

    j_pt = torch.tensor(1j, dtype=torch.complex64, device=k_pt.device)
    exp_term = torch.exp(j_pt * k_pt * r_complex) # (Nf, Ns)

    if source_amplitudes_pt.ndim == 0:
         source_amplitudes_expanded = source_amplitudes_pt # Scalar broadcasts
    elif source_amplitudes_pt.ndim == 1:
         source_amplitudes_expanded = source_amplitudes_pt.unsqueeze(0) # (1, Ns)
    else:
         source_amplitudes_expanded = source_amplitudes_pt

    # Broadcasting should work: (1, Ns)*(Nf, Ns)/(Nf, Ns) -> (Nf, Ns)
    P_contributions = source_amplitudes_expanded * exp_term / r_complex

    # Sum over the source dimension (Ns), which is dim 1
    P_total = torch.sum(P_contributions, dim=1) # (Nf,)

    # Squeeze to remove dimensions of size 1, resulting in scalar if Nf=1
    return P_total.squeeze()

# --- 声学和换能器参数 (空气) ---
c = 346.0
f = 36981.0
# a = 0.02 # Old radius (2cm, Diameter 4cm)
a = 0.0191 # New radius (1.91cm, Diameter 3.82cm)
lambda_ = c / f
k_np = 2 * np.pi / lambda_
k_pt = torch.tensor(k_np, dtype=torch.complex64, device=device)
print(f"介质: 空气, c = {c} m/s")
print(f"频率 f = {f/1000:.3f} kHz")
print(f"换能器半径 a = {a*1000:.1f} mm") # Updated print
print(f"波长 λ = {lambda_*1000:.4f} mm")
nf_dist = a**2 / lambda_
print(f"近场距离 (菲涅尔区) 大约延伸至 a^2/λ ≈ {nf_dist*1000:.1f} mm") # Updated print

# --- 换能器表面离散化 ---
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
source_strength_per_point = 1.0
source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device)
A_point_pt = torch.tensor(source_strength_per_point, dtype=torch.complex64, device=device)
print(f"将换能器表面离散化为 {num_source_points} 个点源 (PyTorch Tensor created on {device})")

# --- 多重反射参数 ---
N_pairs = 100
print(f"镜像源对数 N_pairs = {N_pairs}")

# === 定义要测试的反射系数值和相位 ===
# R_values = np.array([0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float32)
R_values = np.array([0.95], dtype=np.float32) # Only simulate R = 0.95
phi_R_values = np.array([-0.2*np.pi, -0.1*np.pi, 0, 0.1*np.pi, 0.2*np.pi], dtype=np.float32)
print(f"将测试反射系数 R: {R_values}")
print(f"将测试反射相位 phi_R (rad): {phi_R_values}")

# === 仿真参数 ===
l_min_m = 0.0001
l_max_m = 0.050
num_l_points = 5000
l_values_np = np.linspace(l_min_m, l_max_m, num_l_points, dtype=np.float32)
l_values_mm = l_values_np * 1000
print(f"扫描距离 l 从 {l_min_m*1000:.1f} mm 到 {l_max_m*1000:.1f} mm ({num_l_points} 点)")

# 预计算常量 和 绘图设置
lambda_half_mm = (lambda_ / 2) * 1000
db_min_global = -40.0
epsilon_global = np.float32(1e-10)
near_field_limit_mm = 3.0
print(f"将忽略 l < {near_field_limit_mm} mm 范围内的极大值进行分类")

# === 循环计算不同反射相位 ===
print("\n=== 开始仿真不同反射相位 phi_R ===")
for phi_R_val in phi_R_values:
    print(f"\n===== 计算相位 phi_R = {phi_R_val:.3f} rad =====")

    # === 循环计算不同 R 值 ===
    for R_val in R_values:
        # Calculate complex reflection coefficient in NumPy and Torch
        R_complex_np = (R_val * np.exp(1j * phi_R_val)).astype(np.complex64)
        R_0_pt = torch.tensor(R_complex_np, dtype=torch.complex64, device=device)
        R_l_pt = torch.tensor(R_complex_np, dtype=torch.complex64, device=device)
        print(f"\n--- 计算 R = {R_val:.2f} (phi_R={phi_R_val:.3f}) --- Complex R = {R_complex_np:.2f} ---")

        P_total_multi_at_l_np = np.zeros(num_l_points, dtype=np.complex64)

        print(f"  循环计算 l 处声压 (N={N_pairs}, R={R_val:.2f}, phi_R={phi_R_val:.3f}) 使用 PyTorch (优化版)..." )
        start_scan_time = time.time()

        # Pre-calculate base amplitude tensor for all source points
        base_amplitude_per_point_pt = torch.full((num_source_points,), A_point_pt.item(), dtype=torch.complex64, device=device)

        for i, l_val_np in enumerate(l_values_np):
            if l_val_np < 1e-9: continue
            target_point_l_pt = torch.tensor([[0, 0, l_val_np]], dtype=torch.float32, device=device) # Shape (1, 3)

            # --- Optimized Calculation for single l --- START
            all_source_positions_list = [source_points_orig_pt]
            all_source_amplitudes_list = [base_amplitude_per_point_pt]

            current_amplitude_factor_pos_pt = R_l_pt.clone()
            current_amplitude_factor_neg_pt = (R_l_pt * R_0_pt).clone()

            for n in range(1, N_pairs + 1):
                # Positive side reflections
                z_offset_pos = 2.0 * n * l_val_np
                amp_factor_pos = current_amplitude_factor_pos_pt
                src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_plus_pt)
                all_source_amplitudes_list.append(base_amplitude_per_point_pt * amp_factor_pos)
                current_amplitude_factor_pos_pt *= (R_0_pt * R_l_pt)

                # Negative side reflections
                z_offset_neg = -2.0 * n * l_val_np
                amp_factor_neg = current_amplitude_factor_neg_pt
                src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_neg_pt)
                all_source_amplitudes_list.append(base_amplitude_per_point_pt * amp_factor_neg)
                current_amplitude_factor_neg_pt *= (R_l_pt * R_0_pt)

            all_src_pt = torch.cat(all_source_positions_list, dim=0)
            all_amp_pt = torch.cat(all_source_amplitudes_list, dim=0)

            # Corrected call: Pass target_point_l_pt directly (shape (1, 3))
            P_total_at_this_l_pt = calculate_pressure_at_points_pytorch(target_point_l_pt, # Shape (1, 3)
                                                                        all_src_pt,        # Shape (N_total_points, 3)
                                                                        all_amp_pt,        # Shape (N_total_points,)
                                                                        k_pt)
            # --- Optimized Calculation for single l --- END

            # Store result (output of function is now scalar due to squeeze)
            P_total_multi_at_l_np[i] = P_total_at_this_l_pt.cpu().numpy()

            # Progress update
            if (i + 1) % max(1, num_l_points // 10) == 0: # Update less frequently
                 elapsed = time.time() - start_scan_time
                 percent_done = ((i + 1) / num_l_points) * 100
                 eta = (elapsed / percent_done) * (100 - percent_done) if percent_done > 0 else 0
                 processed_points = i + 1
                 points_per_sec = processed_points / elapsed if elapsed > 0 else 0
                 print(f"    l-scan (N={N_pairs}, R={R_val:.2f}, phi={phi_R_val:.2f}) 进度: {percent_done:.1f}% ({processed_points}/{num_l_points}), Speed: {points_per_sec:.1f} l/s, ETA: {eta:.1f}s")

        print(f"  l-scan (R={R_val:.2f}, phi={phi_R_val:.2f}) 计算完成，耗时 {time.time() - start_scan_time:.2f} 秒。")

        # --- 分析当前 R 和 phi_R 值的振幅和 SPL ---
        P_amplitude = np.abs(P_total_multi_at_l_np)
        p_ref = np.max(P_amplitude)
        if p_ref <= 0 or not np.isfinite(p_ref):
             print(f"警告: R={R_val:.2f}, phi={phi_R_val:.2f} 时计算得到无效的最大声压 {p_ref}，跳过 SPL 计算。")
             SPL_dB = np.full_like(P_amplitude, db_min_global)
        else:
             SPL_dB = 20 * np.log10(P_amplitude / p_ref + epsilon_global)
             SPL_dB[SPL_dB < db_min_global] = db_min_global

        # --- 查找极大值 (尝试找到所有) ---
        # min_dist_points = max(1, int(num_l_points * (lambda_half_mm / 2) / (l_max_m * 1000) * 0.8))
        # Set distance=1 to find all local maxima
        peaks_indices, _ = find_peaks(P_amplitude, height=0, distance=1)

        # main_maxima_indices = []
        # secondary_maxima_indices = []
        all_valid_peaks_indices = [] # Store all peaks outside near field

        if len(peaks_indices) > 0:
            # l_theo_main_max_mm = np.arange(1, int(l_max_m * 1000 / lambda_half_mm) + 1) * lambda_half_mm
            # tolerance_mm = lambda_half_mm / 4.0
            peak_l_values_mm = l_values_mm[peaks_indices]

            for idx, peak_l in zip(peaks_indices, peak_l_values_mm):
                if peak_l < near_field_limit_mm:
                    continue
                # --- Classification removed --- START
                # if len(l_theo_main_max_mm) > 0:
                #      min_dist_to_theo = np.min(np.abs(peak_l - l_theo_main_max_mm))
                #      if min_dist_to_theo < tolerance_mm:
                #          main_maxima_indices.append(idx)
                #      else:
                #          secondary_maxima_indices.append(idx)
                # else:
                #      secondary_maxima_indices.append(idx)
                all_valid_peaks_indices.append(idx) # Add all valid peaks
                # --- Classification removed --- END

            # print(f"  找到 {len(peaks_indices)} 个峰值. 近场排除后: 主极大: {len(main_maxima_indices)}, 次极大: {len(secondary_maxima_indices)}")
            print(f"  找到 {len(peaks_indices)} 个原始峰值. 近场(l<{near_field_limit_mm}mm)排除后找到 {len(all_valid_peaks_indices)} 个极大值.")
            # --- 输出所有极大值位置 ---
            if len(all_valid_peaks_indices) > 0:
                 valid_peak_l_values = l_values_mm[all_valid_peaks_indices]
                 print(f"    极大值位置 (l in mm): {np.round(valid_peak_l_values, 2)}")
            # ---
        else:
            print("  未找到明显峰值。")

        # main_maxima_indices = np.array(main_maxima_indices, dtype=int)
        # secondary_maxima_indices = np.array(secondary_maxima_indices, dtype=int)
        all_valid_peaks_indices = np.array(all_valid_peaks_indices, dtype=int)

        # --- 绘制当前 R 和 phi_R 的独立图 ---
        print(f"  绘制 R={R_val:.2f}, phi_R={phi_R_val:.3f} 的独立图表...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f'中心点声压 vs l (R={R_val:.2f}, $\phi_R$={phi_R_val/np.pi:.2f}$\pi$, N={N_pairs})', fontsize=14)

        # 子图1: 线性振幅与极大值标记
        ax1.plot(l_values_mm, P_amplitude, linestyle='-', label='总振幅', color='blue')
        # Mark all valid peaks
        if len(all_valid_peaks_indices) > 0:
            ax1.plot(l_values_mm[all_valid_peaks_indices], P_amplitude[all_valid_peaks_indices],
                     'kx', markersize=6, mew=1.5, linestyle='None', label='极大值') # Black 'x'
        # --- Old peak plotting removed --- START
        # if len(main_maxima_indices) > 0:
        #     ax1.plot(l_values_mm[main_maxima_indices], P_amplitude[main_maxima_indices],
        #              'ro', markersize=7, label='主极大 (近 nλ/2)')
        # if len(secondary_maxima_indices) > 0:
        #     ax1.plot(l_values_mm[secondary_maxima_indices], P_amplitude[secondary_maxima_indices],
        #              'gx', markersize=6, mew=1.5, linestyle='None', label='次极大')
        # --- Old peak plotting removed --- END
        ax1.set_ylabel('相对声压振幅')
        ax1.grid(True, linestyle=':')
        ax1.legend()
        ax1.set_ylim(bottom=0)
        y_max_amp = ax1.get_ylim()[1]
        for k in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
            ax1.axvline(k * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
            if k > 0:
                ax1.text(k * lambda_half_mm, y_max_amp * 0.95, f'{k}λ/2', fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)

        # 子图2: 相对声压级
        ax2.plot(l_values_mm, SPL_dB, linestyle='-', label='相对声压级 (dB)', color='green')
        # Mark all valid peaks
        if len(all_valid_peaks_indices) > 0:
             ax2.plot(l_values_mm[all_valid_peaks_indices], SPL_dB[all_valid_peaks_indices],
                      'kx', markersize=6, mew=1.5, linestyle='None') # Black 'x'
        # --- Old peak plotting removed --- START
        # if len(main_maxima_indices) > 0:
        #      ax2.plot(l_values_mm[main_maxima_indices], SPL_dB[main_maxima_indices], 'ro', markersize=7, linestyle='None')
        # if len(secondary_maxima_indices) > 0:
        #      ax2.plot(l_values_mm[secondary_maxima_indices], SPL_dB[secondary_maxima_indices], 'gx', markersize=6, mew=1.5, linestyle='None')
        # --- Old peak plotting removed --- END
        ax2.set_xlabel('反射面距离 l (mm)')
        ax2.set_ylabel('相对声压级 (dB, 参考最大值)')
        ax2.grid(True, linestyle=':')
        ax2.set_ylim(db_min_global - 5, 5)
        for k in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
            ax2.axvline(k * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
            if k > 0:
                 ax2.text(k * lambda_half_mm, db_min_global, f'{k}λ/2', fontsize=8, ha='center', va='bottom', rotation=90, alpha=0.7)

        # Finalize and save plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        phi_suffix = f"{phi_R_val:.2f}".replace('.', 'p')
        save_path = os.path.join(img_dir, f'multi_reflection_scan_R{R_val:.2f}_phi{phi_suffix}_N{N_pairs}.pdf')
        try:
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"  图表已保存至: {save_path}")
        except Exception as e:
            print(f"  保存图表失败: {e}")
        plt.close(fig)

print("\n多重反射与相位仿真结束。") 