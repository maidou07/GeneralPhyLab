# sound/py/plot_simulation_with_params.py

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import os
import torch

# --- Set PyTorch Device ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("PyTorch using MPS device (GPU)")
elif torch.cuda.is_available():
     device = torch.device("cuda")
     print("PyTorch using CUDA device")
else:
    device = torch.device("cpu")
    print("PyTorch using CPU device")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Create directories if they don't exist
img_dir = 'sound/img'
fit_dir = 'sound/fit_results'
os.makedirs(img_dir, exist_ok=True)
os.makedirs(fit_dir, exist_ok=True)

# === PyTorch Calculation Function (with Attenuation) ===
def calculate_pressure_at_points_pytorch(field_points_pt, source_positions_pt, source_amplitudes_pt, k_pt, gamma_pt):
    """Calculates complex pressure at specific field points including propagation attenuation."""
    field_points_expanded = field_points_pt.unsqueeze(1)
    source_positions_expanded = source_positions_pt.unsqueeze(0)
    diff = field_points_expanded - source_positions_expanded
    r_sq = torch.sum(torch.square(diff), dim=-1)
    r = torch.sqrt(r_sq)
    r = torch.clamp(r, min=1e-9)
    r_complex = r.to(torch.complex64)
    j_pt = torch.tensor(1j, dtype=torch.complex64, device=k_pt.device)
    exp_term = torch.exp(j_pt * k_pt * r_complex)
    attenuation_factor = torch.exp(-gamma_pt * r)
    if source_amplitudes_pt.ndim == 0:
        source_amplitudes_expanded = source_amplitudes_pt
    elif source_amplitudes_pt.ndim == 1:
        source_amplitudes_expanded = source_amplitudes_pt.unsqueeze(0)
    else:
        source_amplitudes_expanded = source_amplitudes_pt
    P_contributions = source_amplitudes_expanded * exp_term * attenuation_factor / r_complex
    P_total = torch.sum(P_contributions, dim=1)
    return P_total.squeeze()

# --- Get Parameters from User Input ---
def get_float_input(prompt, default_value):
    """Helper function to get float input with error handling and default value."""
    while True:
        try:
            user_input = input(f"{prompt} [默认: {default_value}]: ")
            if not user_input: # User pressed Enter
                return default_value
            return float(user_input)
        except ValueError:
            print("  错误：请输入一个有效的数值。")

print("请输入仿真参数:")
print("\n基频参数:")
R1_val     = get_float_input("  R1 (基频反射系数, 0 到 1)", 0.95)
phi1_rad   = get_float_input("  phi1 (基频相位, 弧度)", 0.0)
gamma1     = get_float_input("  gamma1 (基频衰减系数, Np/m)", 1.0)

print("\n二次谐波参数:")
R2_val     = get_float_input("  R2 (二次谐波反射系数, 0 到 1)", 0.95)
phi2_rad   = get_float_input("  phi2 (二次谐波相位, 弧度)", 0.0)
gamma2     = get_float_input("  gamma2 (二次谐波衰减系数, Np/m)", 4.0)
A2_rel     = get_float_input("  A2_rel (二次谐波相对幅值, >=0)", 0.1)

print("\n三次谐波参数:")
R3_val     = get_float_input("  R3 (三次谐波反射系数, 0 到 1)", 0.95)
phi3_rad   = get_float_input("  phi3 (三次谐波相位, 弧度)", 0.0)
gamma3     = get_float_input("  gamma3 (三次谐波衰减系数, Np/m)", 9.0)
A3_rel     = get_float_input("  A3_rel (三次谐波相对幅值, >=0)", 0.05)

# Assign to params list
params = [R1_val, phi1_rad, gamma1, R2_val, phi2_rad, gamma2, R3_val, phi3_rad, gamma3, A2_rel, A3_rel]

print("\n--- 使用参数 ---")
print("基频参数:")
print(f"  R1 = {params[0]:.4f}")
print(f"  phi1 (rad) = {params[1]:.4f}")
print(f"  gamma1 = {params[2]:.4f}")
print("二次谐波参数:")
print(f"  R2 = {params[3]:.4f}")
print(f"  phi2 (rad) = {params[4]:.4f}")
print(f"  gamma2 = {params[5]:.4f}")
print(f"  A2_rel = {params[9]:.4f}")
print("三次谐波参数:")
print(f"  R3 = {params[6]:.4f}")
print(f"  phi3 (rad) = {params[7]:.4f}")
print(f"  gamma3 = {params[8]:.4f}")
print(f"  A3_rel = {params[10]:.4f}")

# --- Load Experimental Data (Optional, for comparison) ---
exp_data_path = 'sound/csv/submax.csv'
l_peaks_exp_mm = None
amp_peaks_exp = None
try:
    exp_data = np.loadtxt(exp_data_path, delimiter=',')
    if exp_data.shape[1] >= 2:  # 确认有至少两列数据
        l_peaks_exp_mm = exp_data[:, 0]  # 第一列是位置
        amp_peaks_exp = exp_data[:, 1]   # 第二列是实测幅度
        # 可选：按位置排序
        sort_indices = np.argsort(l_peaks_exp_mm)
        l_peaks_exp_mm = l_peaks_exp_mm[sort_indices]
        amp_peaks_exp = amp_peaks_exp[sort_indices]
        print(f"成功加载实验极大值数据 ({len(l_peaks_exp_mm)} 点) 用于对比。")
    else:
        l_peaks_exp_mm = exp_data.flatten()
        print(f"实验数据文件中仅包含位置信息，加载了 {len(l_peaks_exp_mm)} 个点。")
except Exception as e:
    print(f"提示: 未能加载实验数据文件 '{exp_data_path}' ({e})，仅绘制仿真结果。")

# --- Fixed Simulation Parameters (Same as fitting script) ---
c = 346.0
f1 = 36981.0
a = 0.0191
N_points_per_radius = 50
N_pairs = 50

# --- Derived Acoustic Parameters ---
f2 = 2 * f1
f3 = 3 * f1
lambda1 = c / f1
lambda2 = c / f2
lambda3 = c / f3
k1_np = 2 * np.pi / lambda1
k2_np = 2 * np.pi / lambda2
k3_np = 2 * np.pi / lambda3
k1_pt = torch.tensor(k1_np, dtype=torch.complex64, device=device)
k2_pt = torch.tensor(k2_np, dtype=torch.complex64, device=device)
k3_pt = torch.tensor(k3_np, dtype=torch.complex64, device=device)

# --- Transducer Discretization ---
source_points_list = []
xs_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1, dtype=np.float32)
ys_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1, dtype=np.float32)
for xs in xs_grid:
    for ys in ys_grid:
        if xs**2 + ys**2 <= a**2:
            source_points_list.append(np.array([xs, ys, 0], dtype=np.float32))
source_points_orig_np = np.array(source_points_list, dtype=np.float32)
num_source_points = len(source_points_orig_np)
source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device)
print(f"换能器离散化: {num_source_points} 点")

# --- Simulation l-range for Plotting ---
# 修改绘图范围为0-20mm，分辨率为1000个点
l_min_plot_m = 0.0001  # 起始点稍微偏移0以避免奇异点
l_max_plot_m = 0.020   # 20mm 上限
num_l_points_plot = 1000  # 设置为1000个点
l_values_plot_np = np.linspace(l_min_plot_m, l_max_plot_m, num_l_points_plot, dtype=np.float32)
l_values_plot_mm = l_values_plot_np * 1000
print(f"绘图 L 范围: {l_min_plot_m*1000:.2f} mm to {l_max_plot_m*1000:.2f} mm ({num_l_points_plot} 点)")

# --- Simulation Function (identical to the one in fit_experimental_peaks.py) ---
def simulate_pressure_curve(l_values_np, params):
    # Expanded parameters for each harmonic
    # [R1, phi1, gamma1, R2, phi2, gamma2, R3, phi3, gamma3, A2_rel, A3_rel]
    R1_val, phi1_rad, gamma1, R2_val, phi2_rad, gamma2, R3_val, phi3_rad, gamma3, A2_rel, A3_rel = params
    
    # Complex reflection coefficients for each harmonic
    R1_complex_np = (R1_val * np.exp(1j * phi1_rad)).astype(np.complex64)
    R2_complex_np = (R2_val * np.exp(1j * phi2_rad)).astype(np.complex64)
    R3_complex_np = (R3_val * np.exp(1j * phi3_rad)).astype(np.complex64)
    
    # Transfer to PyTorch tensors
    R1_0_pt = torch.tensor(R1_complex_np, dtype=torch.complex64, device=device)
    R1_l_pt = torch.tensor(R1_complex_np, dtype=torch.complex64, device=device)
    
    R2_0_pt = torch.tensor(R2_complex_np, dtype=torch.complex64, device=device)
    R2_l_pt = torch.tensor(R2_complex_np, dtype=torch.complex64, device=device)
    
    R3_0_pt = torch.tensor(R3_complex_np, dtype=torch.complex64, device=device)
    R3_l_pt = torch.tensor(R3_complex_np, dtype=torch.complex64, device=device)
    
    # Attenuation coefficients for each harmonic
    gamma1_pt = torch.tensor(gamma1, dtype=torch.float32, device=device)
    gamma2_pt = torch.tensor(gamma2, dtype=torch.float32, device=device) 
    gamma3_pt = torch.tensor(gamma3, dtype=torch.float32, device=device)
    
    # Amplitude coefficients
    A1_point_pt = torch.tensor(1.0, dtype=torch.complex64, device=device)
    A2_point_pt = torch.tensor(A2_rel, dtype=torch.complex64, device=device)
    A3_point_pt = torch.tensor(A3_rel, dtype=torch.complex64, device=device)
    
    base_amplitude1_per_point_pt = torch.full((num_source_points,), A1_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude2_per_point_pt = torch.full((num_source_points,), A2_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude3_per_point_pt = torch.full((num_source_points,), A3_point_pt.item(), dtype=torch.complex64, device=device)
    
    P_total_scenario_at_l_np = np.zeros(len(l_values_np), dtype=np.complex64)
    print("  开始计算仿真曲线...")
    sim_start_time = time.time()
    for i, l_val_np in enumerate(l_values_np):
        if l_val_np < 1e-9: continue
        target_point_l_pt = torch.tensor([[0, 0, l_val_np]], dtype=torch.float32, device=device)
        P_total_at_this_l_pt = torch.tensor(0.0, dtype=torch.complex64, device=device)
        
        # Process each harmonic with its own parameters
        # [frequency, wavenumber, attenuation, amplitude, R coefficients]
        for h_idx, (k_pt_h, gamma_pt_h, base_amp_pt_h, amp_rel_h, R0_pt_h, Rl_pt_h) in enumerate([
            (k1_pt, gamma1_pt, base_amplitude1_per_point_pt, 1.0, R1_0_pt, R1_l_pt),
            (k2_pt, gamma2_pt, base_amplitude2_per_point_pt, A2_rel, R2_0_pt, R2_l_pt),
            (k3_pt, gamma3_pt, base_amplitude3_per_point_pt, A3_rel, R3_0_pt, R3_l_pt)
        ]):
            if amp_rel_h < 1e-6: continue
            all_source_positions_list = [source_points_orig_pt]
            all_source_amplitudes_list = [base_amp_pt_h]
            current_amplitude_factor_pos_pt = Rl_pt_h.clone()
            current_amplitude_factor_neg_pt = (Rl_pt_h * R0_pt_h).clone()
            for n in range(1, N_pairs + 1):
                z_offset_pos = 2.0 * n * l_val_np
                amp_factor_pos = current_amplitude_factor_pos_pt
                src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_plus_pt)
                all_source_amplitudes_list.append(base_amp_pt_h * amp_factor_pos)
                current_amplitude_factor_pos_pt *= (R0_pt_h * Rl_pt_h)
                
                z_offset_neg = -2.0 * n * l_val_np
                amp_factor_neg = current_amplitude_factor_neg_pt
                src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_neg_pt)
                all_source_amplitudes_list.append(base_amp_pt_h * amp_factor_neg)
                current_amplitude_factor_neg_pt *= (Rl_pt_h * R0_pt_h)
            
            all_src_pt = torch.cat(all_source_positions_list, dim=0)
            all_amp_pt = torch.cat(all_source_amplitudes_list, dim=0)
            P_total_at_this_l_pt += calculate_pressure_at_points_pytorch(
                target_point_l_pt, all_src_pt, all_amp_pt, k_pt_h, gamma_pt_h
            )
        P_total_scenario_at_l_np[i] = P_total_at_this_l_pt.cpu().numpy()
        # Optional: Add progress indicator for plotting simulation
        if (i + 1) % max(1, len(l_values_np) // 10) == 0:
             print(f"    仿真进度: {(i+1)/len(l_values_np)*100:.1f}%", end='\r')
    sim_end_time = time.time()
    print(f"\n  仿真计算完成，耗时: {sim_end_time - sim_start_time:.2f} 秒。")
    return np.abs(P_total_scenario_at_l_np)

# --- Run Simulation with Loaded/Default Parameters ---
P_amplitude_sim = simulate_pressure_curve(l_values_plot_np, params)

# --- Find Peaks in Simulation ---
peak_finding_distance_points = 5 # Adjust if needed for plotting resolution
sim_peaks_indices, _ = find_peaks(P_amplitude_sim,
                                   distance=peak_finding_distance_points,
                                   height=np.max(P_amplitude_sim)*0.05) # Optional height threshold
l_peaks_sim_mm = l_values_plot_mm[sim_peaks_indices]
print(f"在仿真曲线中找到 {len(l_peaks_sim_mm)} 个极大值。")

# --- Plotting ---
print("开始绘图...")
plt.figure(figsize=(14, 7))
plt.plot(l_values_plot_mm, P_amplitude_sim, label='仿真结果', color='dodgerblue', linewidth=1.5)
plt.plot(l_peaks_sim_mm, P_amplitude_sim[sim_peaks_indices], '^', markersize=8, label='仿真极大值', color='blue', alpha=0.9)

# Overlay experimental data if loaded
if l_peaks_exp_mm is not None:
    if amp_peaks_exp is not None:
        # 归一化实验数据幅度，使其与仿真幅度在同一尺度
        max_sim_amp = np.max(P_amplitude_sim)
        max_exp_amp = np.max(amp_peaks_exp)
        amp_peaks_exp_norm = amp_peaks_exp * (max_sim_amp / max_exp_amp)
        
        # 使用实际测量的幅度值（归一化后）
        # 只显示20mm范围内的实验数据
        valid_exp_indices = l_peaks_exp_mm <= 20.0
        plt.plot(l_peaks_exp_mm[valid_exp_indices], amp_peaks_exp_norm[valid_exp_indices], 'o', markersize=8, 
                 label='实验极大值', color='red', alpha=0.7, markerfacecolor='none', mew=1.5)
    else:
        # 如果没有幅度数据，则使用插值（保持原代码行为）
        valid_exp_indices = l_peaks_exp_mm <= 20.0
        exp_peak_amps_interp = np.interp(l_peaks_exp_mm[valid_exp_indices], l_values_plot_mm, P_amplitude_sim)
        plt.plot(l_peaks_exp_mm[valid_exp_indices], exp_peak_amps_interp, 'o', markersize=8, 
                 label='实验极大值位置', color='red', alpha=0.7, markerfacecolor='none', mew=1.5)

plt.xlabel('反射面距离 l (mm)')
plt.ylabel('相对声压振幅')
param_str = rf'R1={params[0]:.3f}, R2={params[3]:.3f}, R3={params[6]:.3f}, A2={params[9]:.2f}, A3={params[10]:.2f}'
plt.title(f'仿真声压 vs 距离 (0-20mm, {param_str})')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(l_values_plot_mm[0], l_values_plot_mm[-1])
plt.ylim(bottom=0)

# Add theoretical markers (optional)
lambda1_half_mm = (lambda1 / 2) * 1000
y_max_plot = plt.ylim()[1]
for k in range(int(l_values_plot_mm[-1] / lambda1_half_mm) + 1):
    l_mark = k * lambda1_half_mm
    if l_mark > l_values_plot_mm[0]:
        plt.axvline(l_mark, color='grey', linestyle=':', alpha=0.5, linewidth=1)
        # Optional: Add text label for markers
        # if k > 0:
        #     plt.text(l_mark, y_max_plot * 0.95, f'{k}λ$_1$/2', fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)

# Save the plot
plot_filename = f'simulation_plot_R1{params[0]:.2f}_R2{params[3]:.2f}_R3{params[6]:.2f}_A2{params[9]:.2f}_A3{params[10]:.2f}.pdf'
final_plot_path = os.path.join(img_dir, plot_filename)
try:
    plt.savefig(final_plot_path, format='pdf', bbox_inches='tight')
    print(f"绘图已保存至: {final_plot_path}")
except Exception as e:
    print(f"保存绘图失败: {e}")

plt.show()

print("绘图程序结束。") 