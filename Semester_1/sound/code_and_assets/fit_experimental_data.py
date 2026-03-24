# sound/py/fit_experimental_data.py

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
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
    field_points_expanded = field_points_pt.unsqueeze(1) # (Nf, 1, 3)
    source_positions_expanded = source_positions_pt.unsqueeze(0) # (1, Ns, 3)
    diff = field_points_expanded - source_positions_expanded # (Nf, Ns, 3)
    r_sq = torch.sum(torch.square(diff), dim=-1) # (Nf, Ns)
    r = torch.sqrt(r_sq)
    r = torch.clamp(r, min=1e-9)
    r_complex = r.to(torch.complex64) # (Nf, Ns)
    j_pt = torch.tensor(1j, dtype=torch.complex64, device=k_pt.device)
    exp_term = torch.exp(j_pt * k_pt * r_complex) # Phase term (Nf, Ns)
    attenuation_factor = torch.exp(-gamma_pt * r) # (Nf, Ns)
    if source_amplitudes_pt.ndim == 0:
         source_amplitudes_expanded = source_amplitudes_pt
    elif source_amplitudes_pt.ndim == 1:
         source_amplitudes_expanded = source_amplitudes_pt.unsqueeze(0) # (1, Ns)
    else:
         source_amplitudes_expanded = source_amplitudes_pt
    P_contributions = source_amplitudes_expanded * exp_term * attenuation_factor / r_complex
    P_total = torch.sum(P_contributions, dim=1) # (Nf,)
    return P_total.squeeze()

# --- Load Experimental Data ---
exp_data_path = 'sound/csv/submax.csv'
try:
    exp_data = np.loadtxt(exp_data_path, delimiter=',', usecols=(0,)) # Load only first column (peak locations in mm)
    l_peaks_exp_mm = np.sort(exp_data)
    l_peaks_exp = l_peaks_exp_mm / 1000.0 # Convert to meters
    l_exp_min, l_exp_max = l_peaks_exp[0], l_peaks_exp[-1]
    print(f"成功加载实验数据: {len(l_peaks_exp)} 个极大值点 (从 {l_exp_min*1000:.2f} mm 到 {l_exp_max*1000:.2f} mm)")
    print(f"  实验极大值位置 (mm): {np.round(l_peaks_exp_mm, 2)}")
except Exception as e:
    print(f"错误：无法加载或处理实验数据文件 '{exp_data_path}': {e}")
    exit()

# --- Fixed Simulation Parameters ---
c = 346.0
f1 = 36981.0
a = 0.0191 # Use consistent radius
N_points_per_radius = 50 # Density of transducer points
N_pairs = 50 # Number of image source pairs for multi-reflection

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

# --- Simulation l-range ---
# Ensure simulation covers the experimental range + some margin
l_min_m = max(0.0001, l_exp_min - 0.005)
l_max_m = l_exp_max + 0.005
num_l_points_sim = 2000 # Resolution for simulation during fitting
l_values_sim_np = np.linspace(l_min_m, l_max_m, num_l_points_sim, dtype=np.float32)
l_values_sim_mm = l_values_sim_np * 1000
print(f"仿真 L 范围: {l_min_m*1000:.2f} mm to {l_max_m*1000:.2f} mm ({num_l_points_sim} 点)")

# --- Simulation Function (incorporating all parameters) ---
def simulate_pressure_curve(l_values_np, params):
    """Simulates pressure amplitude curve vs l for given parameters."""
    R_val, phi_R_rad, gamma_base, A2_rel, A3_rel = params

    # Calculate complex R and gamma for each harmonic
    R_complex_np = (R_val * np.exp(1j * phi_R_rad)).astype(np.complex64)
    R_0_pt = torch.tensor(R_complex_np, dtype=torch.complex64, device=device)
    R_l_pt = torch.tensor(R_complex_np, dtype=torch.complex64, device=device)

    gamma1_pt = torch.tensor(gamma_base, dtype=torch.float32, device=device)
    gamma2_pt = gamma1_pt * (f2 / f1)**2
    gamma3_pt = gamma1_pt * (f3 / f1)**2

    # Base amplitudes for each harmonic
    A1_point_pt = torch.tensor(1.0, dtype=torch.complex64, device=device) # Fundamental is reference
    A2_point_pt = torch.tensor(A2_rel, dtype=torch.complex64, device=device)
    A3_point_pt = torch.tensor(A3_rel, dtype=torch.complex64, device=device)
    base_amplitude1_per_point_pt = torch.full((num_source_points,), A1_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude2_per_point_pt = torch.full((num_source_points,), A2_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude3_per_point_pt = torch.full((num_source_points,), A3_point_pt.item(), dtype=torch.complex64, device=device)

    P_total_scenario_at_l_np = np.zeros(len(l_values_np), dtype=np.complex64)

    for i, l_val_np in enumerate(l_values_np):
        if l_val_np < 1e-9: continue
        target_point_l_pt = torch.tensor([[0, 0, l_val_np]], dtype=torch.float32, device=device)
        P_total_at_this_l_pt = torch.tensor(0.0, dtype=torch.complex64, device=device)

        # Calculate contributions for each harmonic
        for k_pt_h, gamma_pt_h, base_amp_pt_h, amp_rel_h in [
            (k1_pt, gamma1_pt, base_amplitude1_per_point_pt, 1.0),
            (k2_pt, gamma2_pt, base_amplitude2_per_point_pt, A2_rel),
            (k3_pt, gamma3_pt, base_amplitude3_per_point_pt, A3_rel)
        ]:
            if amp_rel_h < 1e-6: continue # Skip if amplitude is negligible

            all_source_positions_list = [source_points_orig_pt]
            all_source_amplitudes_list = [base_amp_pt_h]
            current_amplitude_factor_pos_pt = R_l_pt.clone()
            current_amplitude_factor_neg_pt = (R_l_pt * R_0_pt).clone()

            for n in range(1, N_pairs + 1):
                z_offset_pos = 2.0 * n * l_val_np
                amp_factor_pos = current_amplitude_factor_pos_pt
                src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_plus_pt)
                all_source_amplitudes_list.append(base_amp_pt_h * amp_factor_pos)
                current_amplitude_factor_pos_pt *= (R_0_pt * R_l_pt)

                z_offset_neg = -2.0 * n * l_val_np
                amp_factor_neg = current_amplitude_factor_neg_pt
                src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
                all_source_positions_list.append(src_pos_neg_pt)
                all_source_amplitudes_list.append(base_amp_pt_h * amp_factor_neg)
                current_amplitude_factor_neg_pt *= (R_l_pt * R_0_pt)

            all_src_pt = torch.cat(all_source_positions_list, dim=0)
            all_amp_pt = torch.cat(all_source_amplitudes_list, dim=0)

            # Add contribution from this harmonic (using its k and gamma)
            P_total_at_this_l_pt += calculate_pressure_at_points_pytorch(
                target_point_l_pt, all_src_pt, all_amp_pt, k_pt_h, gamma_pt_h
            )

        P_total_scenario_at_l_np[i] = P_total_at_this_l_pt.cpu().numpy()

    return np.abs(P_total_scenario_at_l_np)

# --- Objective Function ---
peak_finding_distance_points = 5 # Min distance between peaks in simulation points
peak_count_mismatch_penalty_weight = 5.0 # Weight for penalty

def objective(params, l_values_sim_np, l_peaks_exp):
    """Calculates the loss between simulated and experimental peak locations."""
    l_values_sim_mm = l_values_sim_np * 1000
    l_peaks_exp_mm = l_peaks_exp * 1000
    l_exp_min_mm, l_exp_max_mm = l_peaks_exp_mm[0], l_peaks_exp_mm[-1]

    # Simulate the curve
    P_amplitude_sim = simulate_pressure_curve(l_values_sim_np, params)

    # Find peaks in simulation
    sim_peaks_indices, _ = find_peaks(P_amplitude_sim,
                                       distance=peak_finding_distance_points,
                                       height=np.max(P_amplitude_sim)*0.05) # Require some height
    l_peaks_sim_mm = l_values_sim_mm[sim_peaks_indices]

    # Filter simulated peaks to be within the experimental range
    l_peaks_sim_filtered = [p for p in l_peaks_sim_mm if l_exp_min_mm <= p <= l_exp_max_mm]

    if not l_peaks_sim_filtered:
        # Heavy penalty if no peaks are found in the range
        return 1e6 + (len(l_peaks_exp_mm))**2 * peak_count_mismatch_penalty_weight

    # Calculate nearest-neighbor squared distance sum
    loss = 0
    for l_exp in l_peaks_exp_mm:
        min_dist_sq = min((l_exp - l_sim)**2 for l_sim in l_peaks_sim_filtered)
        loss += min_dist_sq

    # Add penalty for mismatch in peak count within the range
    loss += peak_count_mismatch_penalty_weight * (len(l_peaks_sim_filtered) - len(l_peaks_exp_mm))**2

    # Normalize loss by number of exp peaks? Maybe not needed if penalty works.
    # loss = loss / len(l_peaks_exp_mm) if len(l_peaks_exp_mm) > 0 else loss

    print(f"  Params: R={params[0]:.3f}, phi={params[1]:.3f}, gam={params[2]:.3f}, A2={params[3]:.3f}, A3={params[4]:.3f} -> Loss: {loss:.4f} (Sim Peaks: {len(l_peaks_sim_filtered)})")
    return loss

# --- Optimization ---
# Parameter order: [R_val, phi_R_rad, gamma_base, A2_rel, A3_rel]
bounds = [
    (0.5, 1.0),       # R (reasonably high reflection)
    (-np.pi/4, np.pi/4), # phi_R (rad, limit range initially)
    (0.0, 10.0),      # gamma_base (Np/m at f1, allow some attenuation)
    (0.0, 0.6),       # A2_rel (relative amplitude of f2)
    (0.0, 0.3)        # A3_rel (relative amplitude of f3, usually smaller)
]

print("开始差分进化优化...")
start_opt_time = time.time()
result = differential_evolution(
    objective,
    bounds,
    args=(l_values_sim_np, l_peaks_exp),
    strategy='best1bin',
    maxiter=100, # Increase iterations for potentially better result
    popsize=20, # Increase population size
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    disp=True, # Show progress
    polish=True # Try to improve result with local search at the end
)
end_opt_time = time.time()
print(f"优化完成，耗时: {end_opt_time - start_opt_time:.2f} 秒")

# --- Results ---
if result.success:
    best_params = result.x
    print("找到最优参数:")
    print(f"  R (反射系数幅值) = {best_params[0]:.4f}")
    print(f"  phi_R (反射相位, rad) = {best_params[1]:.4f}")
    print(f"  phi_R (反射相位, deg) = {np.degrees(best_params[1]):.2f}")
    print(f"  gamma_base (基频衰减系数, Np/m) = {best_params[2]:.4f}")
    print(f"  A2_rel (二次谐波相对幅值) = {best_params[3]:.4f}")
    print(f"  A3_rel (三次谐波相对幅值) = {best_params[4]:.4f}")
    print(f"  对应最小损失值: {result.fun:.4f}")

    # Save best parameters
    fit_params_path = os.path.join(fit_dir, 'best_fit_params.txt')
    with open(fit_params_path, 'w') as f:
        f.write(f"R={best_params[0]:.6f}\n")
        f.write(f"phi_R_rad={best_params[1]:.6f}\n")
        f.write(f"phi_R_deg={np.degrees(best_params[1]):.4f}\n")
        f.write(f"gamma_base={best_params[2]:.6f}\n")
        f.write(f"A2_rel={best_params[3]:.6f}\n")
        f.write(f"A3_rel={best_params[4]:.6f}\n")
        f.write(f"min_loss={result.fun:.6f}\n")
    print(f"最优参数已保存至: {fit_params_path}")

    # --- Plot final comparison ---
    print("绘制最优参数下的仿真结果与实验数据对比图...")
    P_amplitude_best_fit = simulate_pressure_curve(l_values_sim_np, best_params)
    best_fit_peaks_indices, _ = find_peaks(P_amplitude_best_fit, distance=peak_finding_distance_points, height=np.max(P_amplitude_best_fit)*0.05)
    l_peaks_best_fit_mm = l_values_sim_mm[best_fit_peaks_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(l_values_sim_mm, P_amplitude_best_fit, label='仿真 (最优参数)', color='blue')
    plt.plot(l_peaks_best_fit_mm, P_amplitude_best_fit[best_fit_peaks_indices], 'bx', markersize=8, label='仿真极大值', mew=1.5)
    plt.plot(l_peaks_exp_mm, np.interp(l_peaks_exp_mm, l_values_sim_mm, P_amplitude_best_fit), 'ro', markersize=8, label='实验极大值位置', alpha=0.8) # Plot experimental points on the curve
    plt.xlabel('反射面距离 l (mm)')
    plt.ylabel('相对声压振幅')
    plt.title('实验数据与最优仿真结果对比')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.xlim(l_values_sim_mm[0], l_values_sim_mm[-1])
    plt.ylim(bottom=0)

    # Add theoretical markers
    lambda1_half_mm = (lambda1 / 2) * 1000
    y_max_plot = plt.ylim()[1]
    for k in range(int(l_values_sim_mm[-1] / lambda1_half_mm) + 1):
        l_mark = k * lambda1_half_mm
        if l_mark > l_values_sim_mm[0]:
            plt.axvline(l_mark, color='gray', linestyle=':', alpha=0.5)
            if k > 0:
                plt.text(l_mark, y_max_plot * 0.95, f'{k}λ$_1$/2', fontsize=8, ha='center', va='top', rotation=90, alpha=0.7)

    final_plot_path = os.path.join(img_dir, 'fit_result_comparison.pdf')
    try:
        plt.savefig(final_plot_path, format='pdf', bbox_inches='tight')
        print(f"对比图已保存至: {final_plot_path}")
    except Exception as e:
        print(f"保存对比图失败: {e}")
    plt.show()

else:
    print("优化未成功。")
    print(f"  状态: {result.message}")

print("拟合程序结束。") 