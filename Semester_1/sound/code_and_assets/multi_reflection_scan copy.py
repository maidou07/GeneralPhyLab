# sound/multi_reflection_scan.py

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
# No @torch.jit.script for now, let's ensure eager mode works first
def calculate_pressure_at_points_pytorch(field_points_pt, source_positions_pt, source_amplitudes_pt, k_pt):
    """Calculates complex pressure at specific field points using PyTorch (vectorized)."""
    # k_pt should already be complex
    # source_amplitudes_pt should already be complex

    # Expand dimensions for broadcasting: field_points (Nf, 1, 3), source_positions (1, Ns, 3)
    # Ensure tensors are on the same device
    field_points_expanded = field_points_pt.unsqueeze(1).to(k_pt.device) # Shape (Nf, 1, 3)
    source_positions_expanded = source_positions_pt.unsqueeze(0).to(k_pt.device) # Shape (1, Ns, 3)

    # Calculate distances: sqrt(sum((px-sx)^2 + (py-sy)^2 + (pz-sz)^2)) for all pairs
    diff = field_points_expanded - source_positions_expanded # Shape (Nf, Ns, 3)
    r_sq = torch.sum(torch.square(diff), dim=2) # Shape (Nf, Ns)
    r = torch.sqrt(r_sq)
    r = torch.clamp(r, min=1e-9) # Avoid division by zero
    r_complex = r.to(torch.complex64)

    # Calculate complex pressure contribution: A * exp(j*k*r) / r
    j_pt = torch.tensor(1j, dtype=torch.complex64, device=k_pt.device)

    exp_term = torch.exp(j_pt * k_pt * r_complex)
    # Ensure amplitude is broadcast correctly (Ns,) -> (1, Ns)
    # If source_amplitudes_pt is scalar, it broadcasts automatically
    if source_amplitudes_pt.ndim == 0:
         source_amplitudes_expanded = source_amplitudes_pt # Scalar broadcasts
    elif source_amplitudes_pt.ndim == 1:
         source_amplitudes_expanded = source_amplitudes_pt.unsqueeze(0) # Shape (1, Ns)
    else:
         source_amplitudes_expanded = source_amplitudes_pt # Assume already correct shape

    P_contributions = source_amplitudes_expanded * exp_term / r_complex # Shape (Nf, Ns)

    # Sum contributions from all source points for each field point
    P_total = torch.sum(P_contributions, dim=1) # Shape (Nf,)

    return P_total

# --- 声学和换能器参数 (空气) ---
c = 346.0       # 声速 (m/s) - 空气
f = 36981.0     # 频率 (Hz) - 36.981 kHz
a = 0.0191        # 换能器半径 (m) - 30mm
lambda_ = c / f
k_np = 2 * np.pi / lambda_
# PyTorch constant for k (complex) on the chosen device
k_pt = torch.tensor(k_np, dtype=torch.complex64, device=device)

print(f"介质: 空气, c = {c} m/s")
print(f"频率 f = {f/1000:.3f} kHz")
print(f"换能器半径 a = {a*1000:.1f} mm")
print(f"波长 λ = {lambda_*1000:.4f} mm")
nf_dist = a**2 / lambda_
print(f"近场距离 (菲涅尔区) 大约延伸至 a^2/λ ≈ {nf_dist*1000:.1f} mm")

# --- 换能器表面离散化 (使用 NumPy first) ---
N_points_per_radius = 50 # 沿半径方向的离散点数
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
# Convert source points to PyTorch Tensor on the chosen device
source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device)
# Amplitude per point as complex tensor
A_point_pt = torch.tensor(source_strength_per_point, dtype=torch.complex64, device=device)
print(f"将换能器表面离散化为 {num_source_points} 个点源 (PyTorch Tensor created on {device})")

# --- 多重反射参数 ---
N_pairs = 100
print(f"镜像源对数 N_pairs = {N_pairs}")

# === 定义要测试的反射系数值 (增加取值) ===
R_values = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]
print(f"将测试以下反射系数值 R: {R_values}")

# === 仿真参数 (修改 l 范围) ===
l_min_m = 0.0001 # 0.1 mm
l_max_m = 0.050  # 50 mm (修改)
num_l_points = 1000 # 点数 (保持或按需调整)
l_values_np = np.linspace(l_min_m, l_max_m, num_l_points, dtype=np.float32)
l_values_mm = l_values_np * 1000
print(f"扫描距离 l 从 {l_min_m*1000:.1f} mm 到 {l_max_m*1000:.1f} mm ({num_l_points} 点)")

# 预计算常量 和 绘图设置
lambda_half_mm = (lambda_ / 2) * 1000
db_min_global = -40.0
epsilon_global = np.float32(1e-10)

# === 循环计算不同 R 值 ===
print("\n=== 开始仿真不同反射系数 R ===")
for R_val in R_values:
    R_0_np = np.float32(R_val)
    R_l_np = np.float32(R_val)
    print(f"\n--- 计算 R = {R_val:.2f} (R0=Rl={R_val:.2f}) ---")

    P_total_multi_at_l_np = np.zeros(num_l_points, dtype=np.complex64)

    print(f"  循环计算 l 处声压 (N={N_pairs}, R={R_val:.2f}) 使用 PyTorch...")
    start_scan_time = time.time()
    for i, l_val_np in enumerate(l_values_np):
        if l_val_np < 1e-9: continue
        target_point_l_pt = torch.tensor([[0, 0, l_val_np]], dtype=torch.float32, device=device)
        P_total_at_this_l_pt = torch.zeros((1,), dtype=torch.complex64, device=device)
        P_0_pt = calculate_pressure_at_points_pytorch(target_point_l_pt, source_points_orig_pt, A_point_pt, k_pt)
        P_total_at_this_l_pt += P_0_pt
        current_amplitude_factor_pos = R_l_np
        current_amplitude_factor_neg = R_l_np * R_0_np
        for n in range(1, N_pairs + 1):
            z_offset_pos = 2.0 * n * l_val_np
            amp_factor_pos = current_amplitude_factor_pos
            src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
            amp_plus_pt = torch.tensor(amp_factor_pos, dtype=torch.complex64, device=device) * A_point_pt
            P_plus_pt = calculate_pressure_at_points_pytorch(target_point_l_pt, src_pos_plus_pt, amp_plus_pt, k_pt)
            P_total_at_this_l_pt += P_plus_pt
            current_amplitude_factor_pos *= (R_0_np * R_l_np)
            z_offset_neg = -2.0 * n * l_val_np
            amp_factor_neg = current_amplitude_factor_neg
            src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
            amp_neg_pt = torch.tensor(amp_factor_neg, dtype=torch.complex64, device=device) * A_point_pt
            P_neg_pt = calculate_pressure_at_points_pytorch(target_point_l_pt, src_pos_neg_pt, amp_neg_pt, k_pt)
            P_total_at_this_l_pt += P_neg_pt
            current_amplitude_factor_neg *= (R_l_np * R_0_np)
        P_total_multi_at_l_np[i] = P_total_at_this_l_pt.cpu().numpy()[0]
        if (i + 1) % max(1, num_l_points // 5) == 0:
             elapsed = time.time() - start_scan_time
             percent_done = ((i + 1) / num_l_points) * 100
             eta = (elapsed / percent_done) * (100 - percent_done) if percent_done > 0 else 0
             print(f"    l-scan (N={N_pairs}, R={R_val:.2f}) 进度: {percent_done:.1f}% (预计剩余时间: {eta:.1f}s)")
    print(f"  l-scan (R={R_val:.2f}) 计算完成，耗时 {time.time() - start_scan_time:.2f} 秒。")

    # --- 分析当前 R 值的振幅和 SPL --- 
    P_amplitude_multi_at_l = np.abs(P_total_multi_at_l_np)

    p_max_current_R = np.max(P_amplitude_multi_at_l[P_amplitude_multi_at_l > 0])
    if p_max_current_R == 0: p_max_current_R = 1.0
    if not np.isfinite(p_max_current_R) or p_max_current_R <= 0:
        print(f"警告: R={R_val:.2f} 时计算得到无效的最大声压 {p_max_current_R}，跳过绘图。")
        continue

    SPL_dB_at_l = 20 * np.log10(P_amplitude_multi_at_l / p_max_current_R + epsilon_global)
    SPL_dB_at_l[SPL_dB_at_l < db_min_global] = db_min_global

    # --- 查找并区分主/次极大值 --- 
    min_dist_points = max(1, int(num_l_points * (lambda_half_mm / 2) / (l_max_m * 1000) * 0.8))
    peaks_indices, properties = find_peaks(P_amplitude_multi_at_l, height=0, distance=min_dist_points)
    
    main_maxima_indices = []
    secondary_maxima_indices = []
    
    if len(peaks_indices) > 0:
        # 计算理论主极大位置 (mm)
        n_max_theo = int(l_max_m * 1000 / lambda_half_mm) + 1
        l_theo_main_max_mm = np.arange(1, n_max_theo) * lambda_half_mm
        
        # 定义判断接近的阈值 (例如 lambda/4)
        tolerance_mm = lambda_half_mm / 4.0 
        
        peak_l_values_mm = l_values_mm[peaks_indices]
        
        for idx, peak_l in zip(peaks_indices, peak_l_values_mm):
            # 找到距离当前峰值最近的理论主极大位置
            if len(l_theo_main_max_mm) > 0:
                 min_dist_to_theo = np.min(np.abs(peak_l - l_theo_main_max_mm))
                 # 如果足够接近，则认为是主极大
                 if min_dist_to_theo < tolerance_mm:
                     main_maxima_indices.append(idx)
                 else:
                     secondary_maxima_indices.append(idx)
            else: # 如果没有理论主极大（范围太小），都算次极大
                 secondary_maxima_indices.append(idx)
                     
        print(f"  找到 {len(peaks_indices)} 个峰值 (主极大: {len(main_maxima_indices)}, 次极大: {len(secondary_maxima_indices)}) (dist={min_dist_points}, tol={tolerance_mm:.2f}mm)")
    else:
        print("  未找到明显峰值。")
        
    main_maxima_indices = np.array(main_maxima_indices, dtype=int)
    secondary_maxima_indices = np.array(secondary_maxima_indices, dtype=int)

    # --- 绘制当前 R 值的独立图 --- 
    print(f"  绘制 R={R_val:.2f} 的独立图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'反射面中心点声压 vs l (R={R_val:.2f}, N={N_pairs})', fontsize=14)

    # 子图1: 线性振幅与极大值标记
    ax1.plot(l_values_mm, P_amplitude_multi_at_l, linestyle='-', label='总振幅', color='blue')
    # 标记主极大值
    if len(main_maxima_indices) > 0:
        ax1.plot(l_values_mm[main_maxima_indices], P_amplitude_multi_at_l[main_maxima_indices], 
                 'ro', markersize=7, label='主极大 (近 nλ/2)')
    # 标记次极大值
    if len(secondary_maxima_indices) > 0:
        ax1.plot(l_values_mm[secondary_maxima_indices], P_amplitude_multi_at_l[secondary_maxima_indices], 
                 'gx', markersize=6, mew=1.5, linestyle='None', label='次极大')
    
    ax1.set_ylabel('相对声压振幅')
    ax1.grid(True, linestyle=':')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    # 添加理论半波长标记
    y_max_amp = ax1.get_ylim()[1]
    for k in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
        ax1.axvline(k * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
        if k > 0:
            ax1.text(k * lambda_half_mm, y_max_amp * 0.95, f'{k}λ/2', fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)

    # 子图2: 相对声压级
    ax2.plot(l_values_mm, SPL_dB_at_l, linestyle='-', label='相对声压级 (dB)', color='green')
    # 标记极大值对应的SPL (可选，避免图形混乱)
    if len(main_maxima_indices) > 0:
         ax2.plot(l_values_mm[main_maxima_indices], SPL_dB_at_l[main_maxima_indices], 'ro', markersize=7, linestyle='None')
    if len(secondary_maxima_indices) > 0:
         ax2.plot(l_values_mm[secondary_maxima_indices], SPL_dB_at_l[secondary_maxima_indices], 'gx', markersize=6, mew=1.5, linestyle='None')
         
    ax2.set_xlabel('反射面距离 l (mm)')
    ax2.set_ylabel('相对声压级 (dB)')
    ax2.grid(True, linestyle=':')
    # ax2.legend() # Legend might be redundant
    ax2.set_ylim(db_min_global - 5, 5)
    # 添加理论半波长标记
    for k in range(int(l_max_m * 1000 / lambda_half_mm) + 1):
        ax2.axvline(k * lambda_half_mm, color='gray', linestyle=':', alpha=0.5)
        if k > 0:
             ax2.text(k * lambda_half_mm, db_min_global, f'{k}λ/2', fontsize=8, ha='center', va='bottom', rotation=90, alpha=0.7)

    # 保存当前 R 值的图
    save_path = os.path.join(img_dir, f'multi_reflection_copy_R{R_val:.2f}_N{N_pairs}_scan.pdf')
    try:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"  图表已保存至: {save_path}")
    except Exception as e:
        print(f"  保存图表失败: {e}")
    plt.close(fig) # 关闭图形，防止最后显示出来

print("\n多重反射仿真结束。") 