# sound/py/fit_experimental_peaks_cuda.py

import numpy as np
import matplotlib
# 设置Matplotlib使用Agg后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
import logging
import os
import sys
import json
import argparse
from scipy.signal import find_peaks
# from scipy.optimize import differential_evolution # We use custom one
from scipy.spatial.distance import cdist
import torch
import scipy.optimize
import multiprocessing as mp
from functools import partial
import tempfile

# --- CUDA Utilities Import ---
sys.path.append('sound/py') # Assuming cuda_utils is in sound/py
try:
    from cuda_utils import setup_cuda_for_multiprocessing, print_cuda_info, get_optimal_device
except ImportError:
    print("ERROR: cuda_utils.py not found or cannot be imported from sound/py/")
    # Define dummy functions if import fails, to avoid crashing later
    def setup_cuda_for_multiprocessing(): print("Warning: cuda_utils not found, CUDA multiprocessing might not work.")
    def print_cuda_info(): print("Warning: cuda_utils not found.")
    def get_optimal_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Falling back to default device selection.")

# --- Global Constants (Safe outside main guard) ---
C_SOUND = 346.0
BASE_FREQ = 36981.0
TRANSDUCER_RADIUS = 0.0191
N_POINTS_PER_RADIUS = 50
N_REFLECTION_PAIRS = 20 # Reduced from 50 for potentially faster simulation
PEAK_FINDING_DISTANCE_POINTS = 5
LOSS_WEIGHT_LOCATION = 500.0
LOSS_WEIGHT_AMPLITUDE = 25.0
LOSS_WEIGHT_COUNT = 5.0

# Derived constants
LAMBDA1 = C_SOUND / BASE_FREQ
LAMBDA1_HALF_MM = (LAMBDA1 / 2.0) * 1000.0
K1_NP = 2 * np.pi / LAMBDA1
K2_NP = 2 * np.pi / (C_SOUND / (2 * BASE_FREQ))
K3_NP = 2 * np.pi / (C_SOUND / (3 * BASE_FREQ))

# Parameter names (consistent with bounds)
PARAM_NAMES = [
    "R1", "phi1_rad", "gamma1",
    "R2", "phi2_rad", "gamma2",
    "R3", "phi3_rad", "gamma3",
    "A2_rel", "A3_rel"
]

# Optimization Bounds
BOUNDS = [
    (0.8, 1.0),        # R1
    (-np.pi/3, np.pi/3), # phi1
    (0.0, 10.0),       # gamma1
    (0.3, 0.8),        # R2
    (-np.pi/3, np.pi/3), # phi2
    (0, 20),       # gamma2
    (0.3,0.8),        # R3
    (-np.pi/6, np.pi/6), # phi3
    (0, 40),       # gamma3
    (0.0, 0.5),        # A2_rel
    (0.0, 0.5)         # A3_rel
]

# --- Logger Setup Function (can be defined globally) ---
def setup_logger(log_dir, run_timestamp):
    logger = logging.getLogger('plasma_fit')
    # Prevent adding handlers multiple times if re-imported
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s][%(process)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_file = os.path.join(log_dir, f'plasma_fit_{run_timestamp}.log')
    try:
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")
    except Exception as e:
        logger.error(f"Failed to create file handler for {log_file}: {e}")

    return logger

# --- Core Physics/Simulation Functions (can be defined globally) ---

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

def simulate_pressure_curve(l_values_np, params, device, num_source_points, source_points_orig_pt, k_vals_pt, logger):
    """
    模拟声压曲线，计算给定距离范围内的声压幅度
    接收 device, num_source_points, source_points_orig_pt, k_vals_pt, logger 作为参数
    """
    pid = os.getpid()
    logger.debug(f"[PID:{pid}] simulate_pressure_curve started")
    R1_val, phi1_rad, gamma1, R2_val, phi2_rad, gamma2, R3_val, phi3_rad, gamma3, A2_rel, A3_rel = params
    k1_pt, k2_pt, k3_pt = k_vals_pt

    active_harmonics = []
    if R1_val > 1e-6: active_harmonics.append(1)
    if R2_val > 1e-6 and A2_rel > 1e-6: active_harmonics.append(2)
    if R3_val > 1e-6 and A3_rel > 1e-6: active_harmonics.append(3)
    logger.debug(f"[PID:{pid}] Active harmonics: {active_harmonics}")

    l_values_pt = torch.tensor(l_values_np, dtype=torch.float32, device=device)

    R1_complex_pt = torch.tensor((R1_val * np.exp(1j * phi1_rad)).astype(np.complex64), device=device)
    R2_complex_pt = torch.tensor((R2_val * np.exp(1j * phi2_rad)).astype(np.complex64), device=device)
    R3_complex_pt = torch.tensor((R3_val * np.exp(1j * phi3_rad)).astype(np.complex64), device=device)

    gamma1_pt = torch.tensor(gamma1, dtype=torch.float32, device=device)
    gamma2_pt = torch.tensor(gamma2, dtype=torch.float32, device=device)
    gamma3_pt = torch.tensor(gamma3, dtype=torch.float32, device=device)

    A1_point_pt = torch.tensor(1.0, dtype=torch.complex64, device=device)
    A2_point_pt = torch.tensor(A2_rel, dtype=torch.complex64, device=device)
    A3_point_pt = torch.tensor(A3_rel, dtype=torch.complex64, device=device)

    base_amplitude1_pt = torch.full((num_source_points,), A1_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude2_pt = torch.full((num_source_points,), A2_point_pt.item(), dtype=torch.complex64, device=device)
    base_amplitude3_pt = torch.full((num_source_points,), A3_point_pt.item(), dtype=torch.complex64, device=device)

    P_total_at_l_pt = torch.zeros(len(l_values_np), dtype=torch.complex64, device=device)
    batch_size = min(100, len(l_values_np))

    harmonic_configs = [
        (1, k1_pt, gamma1_pt, base_amplitude1_pt, 1.0, R1_complex_pt, R1_complex_pt), # R0 = Rl for simplicity assumed
        (2, k2_pt, gamma2_pt, base_amplitude2_pt, A2_rel, R2_complex_pt, R2_complex_pt),
        (3, k3_pt, gamma3_pt, base_amplitude3_pt, A3_rel, R3_complex_pt, R3_complex_pt)
    ]

    zero_offset_pt = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)

    for h_num, k_pt_h, gamma_pt_h, base_amp_pt_h, amp_rel_h, R0_pt_h, Rl_pt_h in harmonic_configs:
        if h_num not in active_harmonics: continue
        logger.debug(f"[PID:{pid}] Calculating harmonic {h_num}...")

        for batch_start in range(0, len(l_values_np), batch_size):
            batch_end = min(batch_start + batch_size, len(l_values_np))
            l_batch_pt = l_values_pt[batch_start:batch_end]
            batch_len = len(l_batch_pt)
            P_batch = torch.zeros(batch_len, dtype=torch.complex64, device=device)

            for i, l_val_pt in enumerate(l_batch_pt):
                if l_val_pt < 1e-9: continue

                l_val_float = l_val_pt.item() # Use float for multiplication
                all_src_pos_list = [source_points_orig_pt]
                all_src_amp_list = [base_amp_pt_h]

                current_amp_factor_pos = Rl_pt_h.clone()
                current_amp_factor_neg = (Rl_pt_h * R0_pt_h).clone()
                two_l_val_pt = torch.tensor([[0, 0, 2.0 * l_val_float]], dtype=torch.float32, device=device)
                minus_two_l_val_pt = torch.tensor([[0, 0, -2.0 * l_val_float]], dtype=torch.float32, device=device)
                R0_Rl_prod = R0_pt_h * Rl_pt_h

                for n in range(1, N_REFLECTION_PAIRS + 1):
                    amp_pos = base_amp_pt_h * current_amp_factor_pos
                    pos_offset = two_l_val_pt * n
                    all_src_pos_list.append(source_points_orig_pt + pos_offset)
                    all_src_amp_list.append(amp_pos)
                    current_amp_factor_pos *= R0_Rl_prod

                    amp_neg = base_amp_pt_h * current_amp_factor_neg
                    neg_offset = minus_two_l_val_pt * n
                    all_src_pos_list.append(source_points_orig_pt + neg_offset)
                    all_src_amp_list.append(amp_neg)
                    current_amp_factor_neg *= R0_Rl_prod

                all_src_pt = torch.cat(all_src_pos_list, dim=0)
                all_amp_pt = torch.cat(all_src_amp_list, dim=0)

                target_point_l_pt = torch.tensor([[0, 0, l_val_float]], dtype=torch.float32, device=device)
                P_batch[i] += calculate_pressure_at_points_pytorch(
                    target_point_l_pt, all_src_pt, all_amp_pt, k_pt_h, gamma_pt_h
                )

            P_total_at_l_pt[batch_start:batch_end] += P_batch
        logger.debug(f"[PID:{pid}] Harmonic {h_num} calculation finished.")

    P_amplitude = torch.abs(P_total_at_l_pt).cpu().numpy()
    logger.debug(f"[PID:{pid}] simulate_pressure_curve finished")
    return P_amplitude

def objective(params, l_values_sim_np, l_peaks_exp_mm, amp_peaks_exp_norm, 
              device, num_source_points, source_points_orig_pt, k_vals_pt, 
              weight_loc, weight_amp, weight_count, logger): # Added weights to signature
    """Calculates loss. Needs simulation params passed explicitly."""
    pid = os.getpid()
    logger.debug(f"[PID:{pid}] objective: Started") # Simplified start log
    l_values_sim_mm = l_values_sim_np * 1000.0 # Use 1000.0 for float division
    l_exp_min_mm, l_exp_max_mm = l_peaks_exp_mm[0], l_peaks_exp_mm[-1]

    # 1. Simulate
    P_amplitude_sim = simulate_pressure_curve(l_values_sim_np, params, device, num_source_points, source_points_orig_pt, k_vals_pt, logger)
    max_amp_sim = np.max(P_amplitude_sim)
    if max_amp_sim <= 1e-9:
        logger.warning(f"[PID:{pid}] objective: Simulation amplitude too small.")
        return 1e7
    P_amplitude_sim_norm = P_amplitude_sim / max_amp_sim

    # 2. Find sim peaks
    sim_peaks_indices, _ = find_peaks(P_amplitude_sim, distance=PEAK_FINDING_DISTANCE_POINTS, height=0.05) # Height relative to normalized
    l_peaks_sim_mm = l_values_sim_mm[sim_peaks_indices]
    amp_peaks_sim_norm = P_amplitude_sim_norm[sim_peaks_indices]

    # 3. Filter sim peaks
    valid_indices = (l_peaks_sim_mm >= l_exp_min_mm) & (l_peaks_sim_mm <= l_exp_max_mm)
    l_peaks_sim_filtered = l_peaks_sim_mm[valid_indices]
    amp_peaks_sim_norm_filtered = amp_peaks_sim_norm[valid_indices]
    num_exp_peaks = len(l_peaks_exp_mm)
    num_sim_peaks_filt = len(l_peaks_sim_filtered)

    # 4. Handle no sim peaks
    if num_sim_peaks_filt == 0:
        logger.warning(f"[PID:{pid}] objective: No sim peaks in range.")
        # Penalize based on the number of exp peaks missed using passed weights
        return 1e6 + weight_count * num_exp_peaks**2 + num_exp_peaks * ( (weight_loc * (LAMBDA1_HALF_MM/2)**2) + (weight_amp * 0.5**2) )

    # 5. Match peaks (ONE-TO-ONE) based on LOCATION ONLY
    matched_exp_to_sim = {} # dict: {exp_idx: sim_filt_idx}
    used_sim_indices = set() # Track sim indices already matched
    if num_sim_peaks_filt > 0:
        # Iterate through experimental peaks
        for i, l_exp in enumerate(l_peaks_exp_mm):
            distances = np.abs(l_peaks_sim_filtered - l_exp)
            
            # Find the closest *available* sim peak
            best_sim_idx = -1
            min_dist = float('inf')
            
            # Check distances in order, considering only available sim peaks
            for sim_idx, dist in enumerate(distances):
                if sim_idx not in used_sim_indices and dist < min_dist:
                    min_dist = dist
                    best_sim_idx = sim_idx
            
            # If a suitable *and available* sim peak is found within threshold
            if best_sim_idx != -1 and min_dist < LAMBDA1_HALF_MM / 8.0:
                matched_exp_to_sim[i] = best_sim_idx
                used_sim_indices.add(best_sim_idx)
                
    logger.debug(f"[PID:{pid}] objective: Matched {len(matched_exp_to_sim)} peaks (1-to-1) based on L distance < {LAMBDA1_HALF_MM / 8.0:.2f}mm.")
    logger.debug(f"[PID:{pid}]   Used sim indices: {used_sim_indices}")

    # 6. Calculate optimal amplitude scale using ONLY the one-to-one matches
    optimal_scale = 1.0
    # Require a reasonable number of matches to calculate scale reliably
    if len(matched_exp_to_sim) >= max(3, 0.3 * min(num_exp_peaks, num_sim_peaks_filt)): # Require at least 3 matches or 30% of the smaller peak count
        matched_pairs = list(matched_exp_to_sim.items()) # Use only these pairs for scaling
        exp_amps = np.array([amp_peaks_exp_norm[i] for i, _ in matched_pairs])
        sim_amps = np.array([amp_peaks_sim_norm_filtered[j] for _, j in matched_pairs])
        sum_exp_sq = np.sum(exp_amps ** 2)
        if sum_exp_sq > 1e-9:
            optimal_scale = np.sum(exp_amps * sim_amps) / sum_exp_sq
            logger.debug(f"[PID:{pid}] objective: Calculated optimal_scale = {optimal_scale:.4f} from {len(matched_pairs)} pairs.")
        else:
            logger.debug(f"[PID:{pid}] objective: Denominator zero for optimal_scale.")
            optimal_scale = 1.0 # Fallback
    else:
        logger.debug(f"[PID:{pid}] objective: Too few matched peaks ({len(matched_exp_to_sim)}), using scale=1.0")

    # Apply scaling to experimental amplitudes
    exp_amplitude_scaled = amp_peaks_exp_norm * optimal_scale

    # 7. Calculate Loss based on the LOCATION matching (from matched_exp_to_sim)
    loss_loc_sq_sum = 0
    loss_amp_sq_sum = 0
    matched_sim_indices_used_in_loss = set() # Keep track to potentially penalize unused sim peaks later if needed

    for exp_idx, sim_filt_idx in matched_exp_to_sim.items():
        # Ensure sim_filt_idx is valid (should be, as it came from filtered indices)
        if sim_filt_idx < len(l_peaks_sim_filtered):
             # Direct calculation using the matched pair indices
            loss_loc_sq_sum += (l_peaks_exp_mm[exp_idx] - l_peaks_sim_filtered[sim_filt_idx])**2
            loss_amp_sq_sum += (exp_amplitude_scaled[exp_idx] - amp_peaks_sim_norm_filtered[sim_filt_idx])**2
            matched_sim_indices_used_in_loss.add(sim_filt_idx)
        else:
            # This case should ideally not happen if logic is correct
             logger.error(f"[PID:{pid}] objective: Invalid sim_filt_idx {sim_filt_idx} encountered during loss calc! Skipping this pair.")

    # Add penalty for experimental peaks that didn't find a close simulated peak
    unmatched_exp_count = num_exp_peaks - len(matched_exp_to_sim)
    # Simple penalty: add a large fixed loss for each unmatched experimental peak
    # This penalizes simulations that miss expected peaks. Use weighted max possible error.
    # Use passed weights
    unmatched_loc_penalty_per_peak = weight_loc * (LAMBDA1_HALF_MM / 2.0)**2 
    unmatched_amp_penalty_per_peak = weight_amp * (1.0)**2 # Max possible amplitude error is ~1.0 (normalized)
    unmatched_penalty = unmatched_exp_count * (unmatched_loc_penalty_per_peak + unmatched_amp_penalty_per_peak)
    if unmatched_exp_count > 0:
         logger.debug(f"[PID:{pid}] objective: Adding penalty for {unmatched_exp_count} unmatched experimental peaks.")

    # 8. Count Penalty for overall difference in peak numbers using passed weight
    count_mismatch_penalty = weight_count * (num_sim_peaks_filt - num_exp_peaks)**2

    # 9. Combine losses using passed weights
    # Normalize matched loss terms by number of exp peaks to make it somewhat independent
    loss_loc_term = weight_loc * loss_loc_sq_sum / num_exp_peaks if num_exp_peaks > 0 else 0
    loss_amp_term = weight_amp * loss_amp_sq_sum / num_exp_peaks if num_exp_peaks > 0 else 0
    # Add the penalties
    total_loss = loss_loc_term + loss_amp_term + count_mismatch_penalty + unmatched_penalty

    # Logging
    param_loss_str = (f"R1={params[0]:.3f},φ1={params[1]:.3f},γ1={params[2]:.3f}, "
                    f"R2={params[3]:.3f},φ2={params[4]:.3f},γ2={params[5]:.3f}, "
                    f"R3={params[6]:.3f},φ3={params[7]:.3f},γ3={params[8]:.3f}, "
                    f"A2={params[9]:.3f},A3={params[10]:.3f} -> Loss: {total_loss:.3f}")
    logger.info(f"[PID:{pid}] 评估: {param_loss_str}")
    logger.debug(f"[PID:{pid}]   Loss Breakdown: Loc={loss_loc_term:.3f}, Amp={loss_amp_term:.3f}, CountPen={count_mismatch_penalty:.3f}, UnmatchedPen={unmatched_penalty:.3f}")
    logger.debug(f"[PID:{pid}]   Peaks: Exp={num_exp_peaks}, SimFilt={num_sim_peaks_filt}, Matched={len(matched_exp_to_sim)}, ScaleFactor={optimal_scale:.4f}")

    return total_loss

# --- Optimization Callback Class (can be defined globally) ---
class OptimizationCallback:
    # ... (class implementation - needs self.logger)
    def __init__(self, logger, patience=10, min_improvement=0.01, max_time_seconds=1800,
                 min_loss_threshold=1.0, window_size=5, stability_threshold=0.01,
                 divergence_threshold=2.0, plots_dir=None):
        self.logger = logger
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_time_seconds = max_time_seconds
        self.min_loss_threshold = min_loss_threshold
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.divergence_threshold = divergence_threshold
        self.plots_dir = plots_dir

        self.iteration = 0
        self.best_loss = float('inf')
        self.best_params = None
        self.best_iteration = -1
        self.start_time = time.time()
        self.wait = 0
        self.loss_history = []
        self.param_history = [] # Consider removing if memory becomes an issue
        self.convergence_status = "未完成"
        self.should_stop = False
        self.objective_func = None # Will be set later
        self.args = None # Will be set later
        self.last_plotted_iteration = -1 # Track last best plot

        self.logger.info(f"优化回调初始化，Patience={patience}, MinImprove={min_improvement:.3f}, MaxTime={max_time_seconds}s")
        if plots_dir:
            self.logger.info(f"最佳结果图表将保存至: {plots_dir}")

    def __call__(self, xk, convergence=None): # Accept convergence for compatibility if needed
        current_time = time.time()
        self.iteration += 1

        # Note: DE provides the *best* solution so far (xk) to the callback.
        # We need the actual loss for this xk to decide if it's an improvement.
        # Re-evaluating xk here might be slow. Can we get the current best fitness?
        # Scipy's DE callback doesn't directly provide the current best fitness easily.
        # We rely on the fact that `evaluate_single_param` logs fitness, and we track `best_loss` internally.
        current_loss = self.objective_func(xk, *self.args)
        self.loss_history.append(current_loss)

        improvement = float('inf')
        is_new_best = False
        if current_loss < self.best_loss:
            if self.best_loss != float('inf'):
                improvement = (self.best_loss - current_loss) / abs(self.best_loss) if abs(self.best_loss) > 1e-9 else float('inf')
            
            if improvement >= self.min_improvement:
                self.logger.info(f"Iter {self.iteration}: 新最佳解! Loss={current_loss:.4f} (改进: {improvement:.4f} >= {self.min_improvement:.4f}), Patience重置.")
                self.best_loss = current_loss
                self.best_params = np.copy(xk)
                self.best_iteration = self.iteration
                self.wait = 0
                is_new_best = True
                 # Save best parameters to a file immediately
                if self.plots_dir:
                    best_params_file = os.path.join(self.plots_dir, f"best_params_iter_{self.iteration}.json")
                    try:
                        params_dict = {name: val for name, val in zip(PARAM_NAMES, self.best_params)}
                        params_dict['iteration'] = self.iteration
                        params_dict['loss'] = self.best_loss
                        with open(best_params_file, 'w') as f:
                            json.dump(params_dict, f, indent=2)
                        self.logger.info(f"最佳参数已保存: {best_params_file}")
                    except Exception as e:
                        self.logger.error(f"保存最佳参数失败: {e}")
            else:
                self.wait += 1
                self.logger.info(f"Iter {self.iteration}: Loss={current_loss:.4f} (未达最小改进 {improvement:.4f} < {self.min_improvement:.4f}), Patience={self.wait}/{self.patience}")
        else:
            self.wait += 1
            self.logger.info(f"Iter {self.iteration}: Loss={current_loss:.4f} (无改进), Patience={self.wait}/{self.patience}")

        # Generate plot only for the best parameters found so far
        if is_new_best and self.plots_dir and self.iteration != self.last_plotted_iteration:
             best_plot_file = os.path.join(self.plots_dir, f"best_iter_{self.iteration:04d}.png")
             self.logger.info(f"尝试生成最佳参数图像: {best_plot_file}")
             try:
                 # Use param_id=0 for best plots, iteration is the main identifier
                 plot_success = generate_iteration_plot(self.best_params, self.best_loss, self.iteration, 0, best_plot_file, is_best=True)
                 if plot_success:
                     self.logger.info(f"最佳参数图像已保存: {best_plot_file}")
                     self.last_plotted_iteration = self.iteration
                 else:
                     self.logger.error(f"保存最佳参数图像失败: {best_plot_file}")
             except Exception as e:
                 self.logger.error(f"生成最佳参数图像时出错: {e}")
                 self.logger.exception("详细错误堆栈:")

        # Check stopping conditions
        stop_reason = None
        current_time = time.time()

        # 1. Check for success condition FIRST (loss threshold)
        if self.best_loss < self.min_loss_threshold:
            stop_reason = f"达到目标损失阈值 {self.min_loss_threshold}"
            self.should_stop = True
            self.convergence_status = stop_reason # Mark as successful stop
            self.logger.info(f"触发停止条件: {stop_reason}")

        # 2. Check fail-safe conditions ONLY if not already stopped by target loss
        if not self.should_stop:
            if self.wait >= self.patience:
                stop_reason = f"达到耐心值 {self.patience} (未达到目标损失)"
                self.should_stop = True
                self.convergence_status = stop_reason # Mark as stopped due to patience
                self.logger.info(f"触发停止条件: {stop_reason}")
            elif current_time - self.start_time > self.max_time_seconds:
                stop_reason = f"达到最大时间 {self.max_time_seconds}s (未达到目标损失)"
                self.should_stop = True
                self.convergence_status = stop_reason # Mark as stopped due to time
                self.logger.info(f"触发停止条件: {stop_reason}")

        # Log iteration summary
        time_elapsed = current_time - self.start_time
        self.logger.info(f"--- Iter {self.iteration} | Best Loss: {self.best_loss:.4f} (Iter {self.best_iteration}) | Time: {time_elapsed:.1f}s ---")
        return self.should_stop

# --- Plotting Function (can be defined globally) ---
def generate_iteration_plot(params, loss_value, iteration, param_id, output_file, is_best=False, logger=None, l_values_sim_mm=None, l_peaks_exp_mm=None, amp_peaks_exp_norm=None, simulate_func=None, device=None, num_source_points=None, source_points_orig_pt=None, k_vals_pt=None):
    """生成当前参数下的模拟与实验对比图 (简化版)
       Requires logger and necessary simulation data passed as arguments.
    """
    # --- Argument Validation --- 
    if logger is None:
        print("FATAL ERROR in generate_iteration_plot: Logger not provided.")
        # Cannot log the error if logger is None, print instead.
        return False
        
    pid = os.getpid()
    logger.info(f"[PID:{pid}] generate_iteration_plot: 开始 (Iter:{iteration}, ParamID:{param_id}, Best:{is_best}) -> {output_file}")

    # Check required arguments individually for clarity and avoiding the ambiguous error
    required_args_check = {
        # Skip checking logger as it's already verified
        "l_values_sim_mm": l_values_sim_mm,
        # Allow l_peaks_exp_mm and amp_peaks_exp_norm to be None if experimental data wasn't loaded
        # "l_peaks_exp_mm": l_peaks_exp_mm, 
        # "amp_peaks_exp_norm": amp_peaks_exp_norm,
        "simulate_func": simulate_func,
        "device": device,
        "num_source_points": num_source_points,
        "source_points_orig_pt": source_points_orig_pt,
        "k_vals_pt": k_vals_pt
    }
    missing_args = [name for name, arg in required_args_check.items() if arg is None]
    if missing_args:
         logger.error(f"[PID:{pid}] 错误: generate_iteration_plot缺少必要的参数: {', '.join(missing_args)}")
         return False
    # Additionally check if experimental data arrays exist if they are expected
    # This depends on whether the main process guarantees they are non-None if loaded.
    # Assuming for now that if they are passed, they should be valid.
    if l_peaks_exp_mm is None or amp_peaks_exp_norm is None:
        logger.warning(f"[PID:{pid}] 缺少实验数据 (l_peaks_exp_mm or amp_peaks_exp_norm is None)，绘图将不包含实验点.")

    # --- Plotting Logic --- 
    fig = None
    try:
        # ... (rest of the function: directory check, simulation, plotting, saving) ...
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.warning(f"[PID:{pid}] 图像输出目录不存在，已创建: {output_dir}")
            except Exception as mkdir_err:
                logger.error(f"[PID:{pid}] 无法创建图像目录 {output_dir}: {mkdir_err}")
                return False

        dpi = 150 if is_best else 75
        fig_size = (10, 6) if is_best else (7, 5)

        logger.debug(f"[PID:{pid}] 模拟压力曲线 for plot...")
        # Call the passed simulate_func
        P_amplitude_sim = simulate_func(l_values_sim_mm / 1000.0, params, device, num_source_points, source_points_orig_pt, k_vals_pt, logger)
        max_amp_sim = np.max(P_amplitude_sim)
        if max_amp_sim <= 1e-9:
            P_amplitude_sim_norm = np.zeros_like(P_amplitude_sim)
        else:
            P_amplitude_sim_norm = P_amplitude_sim / max_amp_sim

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.plot(l_values_sim_mm, P_amplitude_sim_norm, 'b-', linewidth=1.0, label='模拟(归一化)')
        # Only plot experimental data if it's available and not None
        if l_peaks_exp_mm is not None and amp_peaks_exp_norm is not None:
             ax.plot(l_peaks_exp_mm, amp_peaks_exp_norm, 'rx', markersize=6, label='实验(归一化)')

        title = f'Iter {iteration} ID {param_id} Loss {loss_value:.3f}' if not is_best else f'Best Iter {iteration} Loss {loss_value:.3f}'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Distance (mm)', fontsize=9)
        ax.set_ylabel('Normalized Amp', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set xlim based on available data
        if l_peaks_exp_mm is not None and len(l_peaks_exp_mm) > 0:
            ax.set_xlim(l_peaks_exp_mm[0] * 0.95, l_peaks_exp_mm[-1] * 1.05)
        elif l_values_sim_mm is not None and len(l_values_sim_mm) > 0:
            ax.set_xlim(l_values_sim_mm[0], l_values_sim_mm[-1])
            
        ax.set_ylim(bottom=-0.05) # Ensure baseline is visible

        output_file_abs = os.path.abspath(output_file)
        logger.info(f"[PID:{pid}] 准备保存图像到: {output_file_abs}")
        try:
            plt.tight_layout(pad=0.5)
            plt.savefig(output_file_abs, dpi=dpi, format='png', bbox_inches='tight')
            logger.info(f"[PID:{pid}] savefig 完成: {output_file_abs}")
            if os.path.exists(output_file_abs) and os.path.getsize(output_file_abs) > 0:
                logger.info(f"[PID:{pid}] 图像验证成功: {output_file_abs}")
                return True
            else:
                logger.error(f"[PID:{pid}] 图像验证失败: {output_file_abs}")
                return False
        except Exception as save_err:
            logger.error(f"[PID:{pid}] savefig 失败: {save_err}")
            logger.exception("详细错误堆栈:")
            return False
    except Exception as plot_err:
        logger.error(f"[PID:{pid}] generate_iteration_plot 意外错误: {plot_err}")
        logger.exception("详细错误堆栈:")
        return False
    finally:
        if fig is not None:
            plt.close(fig)
            logger.debug(f"[PID:{pid}] 图形已关闭")

# --- Worker Function (replaces evaluate_single_param for clarity) ---
def worker_evaluate_and_plot(params, core_args, iter_plots_dir):
    """子进程执行的函数：评估参数并生成图像"""
    # Unpack core arguments needed by objective and plotting
    # Correctly unpack including the new weights
    l_values_sim_np, l_peaks_exp_mm, amp_peaks_exp_norm, device_name, \
    num_source_points, source_points_orig_np, k_vals_np, \
    weight_loc, weight_amp, weight_count, \
    run_dir, timestamp = core_args

    pid = os.getpid()
    # Setup logger for this specific worker process
    # Log to the *same* file as the main process (using append mode)
    logger = setup_logger(run_dir, timestamp)
    logger.info(f"[PID:{pid}] Worker started.")

    # Re-initialize device and tensors for this process
    device = torch.device(device_name)
    source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device)
    k1_pt = torch.tensor(k_vals_np[0], dtype=torch.complex64, device=device)
    k2_pt = torch.tensor(k_vals_np[1], dtype=torch.complex64, device=device)
    k3_pt = torch.tensor(k_vals_np[2], dtype=torch.complex64, device=device)
    k_vals_pt = (k1_pt, k2_pt, k3_pt)

    # --- 1. Execute objective function --- 
    loss = -1.0 # Default loss if objective fails
    try:
        # Pass weights to objective function
        loss = objective(params, l_values_sim_np, l_peaks_exp_mm, amp_peaks_exp_norm,
                         device, num_source_points, source_points_orig_pt, k_vals_pt,
                         weight_loc, weight_amp, weight_count, logger) # Pass unpacked weights
        logger.info(f"[PID:{pid}] Objective function completed, Loss: {loss:.4f}")
    except Exception as e:
        logger.error(f"[PID:{pid}] Objective function failed: {e}")
        logger.exception("Detailed stack trace:")
        loss = 1e8 # Return high penalty

    # --- 2. Generate plot --- 
    if iter_plots_dir is not None and loss >= 0: # Only plot if objective ran and loss is valid
        param_id = np.random.randint(10000)
        # Iteration number isn't easily available here, use 0
        iteration_for_filename = 0
        # Include loss in the filename
        plot_filename = os.path.join(iter_plots_dir, f"param_{iteration_for_filename}_{param_id:04d}_loss{loss:.3f}.png")
        logger.info(f"[PID:{pid}] Worker attempting to generate plot: {plot_filename}")
        try:
            # Prepare args for generate_iteration_plot
            l_values_sim_mm = l_values_sim_np * 1000 # Recalculate mm scale
            plot_args = {
                'params': params,
                'loss_value': loss, # Pass the calculated loss
                'iteration': iteration_for_filename,
                'param_id': param_id,
                'output_file': plot_filename,
                'is_best': False,
                'logger': logger,
                'l_values_sim_mm': l_values_sim_mm,
                'l_peaks_exp_mm': l_peaks_exp_mm,
                'amp_peaks_exp_norm': amp_peaks_exp_norm,
                'simulate_func': simulate_pressure_curve, # Pass function itself
                'device': device,
                'num_source_points': num_source_points,
                'source_points_orig_pt': source_points_orig_pt,
                'k_vals_pt': k_vals_pt
            }
            plot_success = generate_iteration_plot(**plot_args)
            if plot_success:
                logger.info(f"[PID:{pid}] Worker successfully generated plot: {plot_filename}")
            else:
                logger.error(f"[PID:{pid}] Worker failed to generate plot (returned False): {plot_filename}")
        except Exception as e:
            logger.error(f"[PID:{pid}] Worker failed to generate plot (exception): {e}")
            logger.exception("Detailed stack trace:")
    elif loss < 0:
         logger.warning(f"[PID:{pid}] Worker: Objective failed (loss={loss}), skipping plot generation.")
    else: # iter_plots_dir is None
         logger.warning(f"[PID:{pid}] Worker: iter_plots_dir is None, skipping plot generation.")

    logger.info(f"[PID:{pid}] Worker finished.")
    return loss

# --- Custom Differential Evolution (handles multiprocessing) ---
def custom_parallel_differential_evolution(objective_func, bounds,
                                          core_args, # Replaces generic args
                                          max_workers=4, maxiter=100,
                                          popsize=15, tol=0.01,
                                          mutation=(0.5, 1.0),
                                          recombination=0.7, seed=None,
                                          callback=None, disp=False, logger=None):
    # ... (implementation - needs logger, uses worker_evaluate_and_plot) ...
    if logger is None:
        print("ERROR: Logger not provided to custom_parallel_differential_evolution")
        # Setup a dummy logger maybe?
        logger = logging.getLogger('de_dummy')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.WARNING)
        
    logger.info(f"使用自定义并行微分进化，进程数: {max_workers}")
    rng = np.random.RandomState(seed)
    n_params = len(bounds)
    
    # Unpack run_dir and timestamp from core_args for worker logger setup
    run_dir = core_args[-2] 
    timestamp = core_args[-1]
    iter_plots_dir = os.path.join(run_dir, 'iter_plots') # Define plot dir based on run_dir

    # Create initial population
    population = np.zeros((popsize, n_params))
    for i in range(n_params):
        population[:, i] = rng.uniform(bounds[i][0], bounds[i][1], popsize)

    # Initial evaluation (can be done in parallel too, but maybe simpler sequentially first)
    logger.info("评估初始种群...")
    fitness = np.zeros(popsize)
    initial_eval_tasks = [(pop, core_args, iter_plots_dir) for pop in population]
    
    # Use spawn context for pool
    mp_context = mp.get_context('spawn')
    try:
        with mp_context.Pool(processes=max_workers) as pool:
            fitness = pool.starmap(worker_evaluate_and_plot, initial_eval_tasks)
        fitness = np.array(fitness)
        logger.info("初始种群评估完成.")
    except Exception as e:
         logger.error(f"初始种群评估失败: {e}")
         logger.exception("详细错误堆栈:")
         return None # Indicate failure

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    logger.info(f"初始最佳适应度: {best_fitness:.4f}")

    # Setup callback
    if callback:
        callback.logger = logger # Ensure callback uses the same logger
        callback.start_time = time.time()
        # Pass objective function and args needed by callback's internal evaluation
        # Note: The callback might need access to the simulation parameters in core_args
        callback.objective_func = objective_func 
        callback.args = core_args[:-2] # Pass only the args relevant to objective, not run_dir/timestamp
    else:
         # Create a default callback if none provided
         callback = OptimizationCallback(logger=logger, plots_dir=iter_plots_dir) 
         callback.start_time = time.time()
         callback.objective_func = objective_func
         callback.args = core_args[:-2]

    # Main DE loop
    for iteration in range(maxiter):
        logger.info(f"--- 开始迭代 {iteration + 1}/{maxiter} ---")
        trial_population = np.zeros_like(population)

        # Create trial vectors (mutation and crossover)
        for i in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != i]
            a, b, c = population[rng.choice(idxs, 3, replace=False)]
            mutant = np.clip(best_solution + rng.uniform(mutation[0], mutation[1]) * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = rng.rand(n_params) < recombination
            if not np.any(cross_points):
                cross_points[rng.randint(n_params)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_population[i] = trial

        # Parallel evaluation of trial population
        logger.info(f"分发 {popsize} 个评估任务...")
        eval_tasks = [(trial, core_args, iter_plots_dir) for trial in trial_population]
        try:
            with mp_context.Pool(processes=max_workers) as pool:
                trial_fitness = pool.starmap(worker_evaluate_and_plot, eval_tasks)
            trial_fitness = np.array(trial_fitness)
            logger.info("评估任务完成.")
        except Exception as e:
            logger.error(f"迭代 {iteration+1} 评估失败: {e}")
            logger.exception("详细错误堆栈:")
            # Maybe break or try to continue?
            break # Let's break if evaluation fails

        # Selection
        improved_mask = trial_fitness < fitness
        population[improved_mask] = trial_population[improved_mask]
        fitness[improved_mask] = trial_fitness[improved_mask]

        # Update best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = fitness[current_best_idx]
            # Log improvement within callback now

        # Callback execution (provides best solution found *in this iteration*)
        if callback:
            stop = callback(best_solution) # Callback now handles logging improvement
            if stop:
                logger.info(f"回调函数请求停止在迭代 {iteration + 1}")
                break
        else: # Minimal logging if no callback
             logger.info(f"Iter {iteration+1}: Current Best Loss = {best_fitness:.4f}")

        # Check convergence (optional, callback might handle this)
        if best_fitness < tol:
            logger.info(f"达到收敛阈值 {tol} 在迭代 {iteration + 1}")
            break

    # Final result construction
    result = scipy.optimize.OptimizeResult()
    result.x = callback.best_params if callback else best_solution # Get best from callback if possible
    result.fun = callback.best_loss if callback else best_fitness
    result.nit = iteration + 1
    result.message = getattr(callback, 'convergence_status', 'Optimization loop finished.')
    result.success = True # Assume success if loop finishes

    logger.info("自定义并行微分进化完成.")
    return result


# ================================================
#               MAIN EXECUTION BLOCK
# ================================================
if __name__ == "__main__":

    # --- 1. Initial Setup --- 
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create base logs directory relative to script
    logs_dir = os.path.join(script_dir, '..', 'logs') # Place logs outside py folder
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception as e:
        print(f"FATAL: Cannot create base log directory {logs_dir}: {e}")
        sys.exit(1)
        
    # Create directory for this specific run
    run_dir = os.path.join(logs_dir, f'run_{run_timestamp}')
    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as e:
         print(f"FATAL: Cannot create run directory {run_dir}: {e}")
         sys.exit(1)
         
    # Setup root logger for the main process
    logger = setup_logger(run_dir, run_timestamp)
    logger.info(f"Run directory: {run_dir}")

    # Setup multiprocessing context (important for CUDA)
    setup_cuda_for_multiprocessing()
    mp.set_start_method('spawn', force=True)
    logger.info(f"Multiprocessing start method set to 'spawn'")

    # --- 2. Device and Environment Info --- 
    logger.info("--- System Information ---")
    logger.info(f"Python: {sys.version}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Matplotlib: {matplotlib.__version__}")
    logger.info(f"SciPy: {scipy.__version__}")
    print_cuda_info() # Log CUDA details if available
    device = get_optimal_device()
    logger.info(f"Using device: {device}")
    
    # --- 3. Setup Directories (Output related) ---
    # Base img and fit_results dir relative to script parent
    base_output_dir = os.path.join(script_dir, '..') 
    img_dir = os.path.join(base_output_dir, 'img')
    fit_dir = os.path.join(base_output_dir, 'fit_results')
    try:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(fit_dir, exist_ok=True)
        logger.info(f"Output directories ensured: {img_dir}, {fit_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directories: {e}")
        # Decide if this is fatal

    # --- 4. Load Experimental Data --- 
    l_peaks_exp_mm = None
    amp_peaks_exp_norm = None
    data_loaded = False
    final_exp_data_path = None

    # Helper function for loading
    def load_exp_data(path_to_load):
        exp_data = np.loadtxt(path_to_load, delimiter=',')
        if exp_data.ndim == 1: exp_data = exp_data.reshape(1, -1)
        exp_data = exp_data[exp_data[:, 0].argsort()] # Sort by location
        l_peaks = exp_data[:, 0]
        amp_peaks = exp_data[:, 1]
        max_amp = np.max(amp_peaks)
        if max_amp <= 1e-9:
            raise ValueError("Max experimental amplitude is too small or zero.")
        amp_peaks_norm = amp_peaks / max_amp
        return l_peaks, amp_peaks_norm

    # Attempt 1: Path relative to the script file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exp_data_path_script_rel = os.path.normpath(os.path.join(script_dir, '..', 'csv', 'submax.csv'))
        logger.info(f"Attempting to load data relative to script: {exp_data_path_script_rel}")
        l_peaks_exp_mm, amp_peaks_exp_norm = load_exp_data(exp_data_path_script_rel)
        logger.info(f"Successfully loaded data using script-relative path: {exp_data_path_script_rel}")
        final_exp_data_path = exp_data_path_script_rel
        data_loaded = True
    except FileNotFoundError:
        logger.warning(f"Script-relative path not found: {exp_data_path_script_rel}")
    except Exception as e:
        logger.error(f"Failed loading from script-relative path '{exp_data_path_script_rel}': {e}")

    # Attempt 2: Path relative to Current Working Directory (if first attempt failed)
    if not data_loaded:
        try:
            cwd = os.getcwd()
            exp_data_path_cwd_rel = os.path.join(cwd, "sound", "csv", "submax.csv")
            # Normalize path just in case CWD has unusual components
            exp_data_path_cwd_rel = os.path.normpath(exp_data_path_cwd_rel)
            logger.info(f"Attempting to load data relative to CWD ({cwd}): {exp_data_path_cwd_rel}")
            l_peaks_exp_mm, amp_peaks_exp_norm = load_exp_data(exp_data_path_cwd_rel)
            logger.info(f"Successfully loaded data using CWD-relative path: {exp_data_path_cwd_rel}")
            final_exp_data_path = exp_data_path_cwd_rel
            data_loaded = True
        except FileNotFoundError:
            logger.warning(f"CWD-relative path not found: {exp_data_path_cwd_rel}")
        except Exception as e:
            logger.error(f"Failed loading from CWD-relative path '{exp_data_path_cwd_rel}': {e}")

    # Final check and exit if data loading failed
    if not data_loaded:
        logger.critical("FATAL: Could not load experimental data from any attempted path. Exiting.")
        logger.critical(f"  Attempted script-relative: {exp_data_path_script_rel}")
        logger.critical(f"  Attempted CWD-relative: {exp_data_path_cwd_rel if 'exp_data_path_cwd_rel' in locals() else '(Not Attempted)'}")
        sys.exit(1)

    logger.info(f"Using experimental data from: {final_exp_data_path}")
    logger.info(f"  {len(l_peaks_exp_mm)} peaks loaded.")
    logger.info(f"  L Range (mm): {l_peaks_exp_mm[0]:.2f} to {l_peaks_exp_mm[-1]:.2f}")

    # --- 5. Prepare Simulation Setup Data (Constants, Tensors) --- 
    logger.info("--- Preparing Simulation Setup ---")
    # Transducer Discretization
    source_points_list = []
    xs_grid = np.linspace(-TRANSDUCER_RADIUS, TRANSDUCER_RADIUS, 2 * N_POINTS_PER_RADIUS + 1, dtype=np.float32)
    ys_grid = np.linspace(-TRANSDUCER_RADIUS, TRANSDUCER_RADIUS, 2 * N_POINTS_PER_RADIUS + 1, dtype=np.float32)
    radius_sq = TRANSDUCER_RADIUS**2
    for xs in xs_grid:
        for ys in ys_grid:
            if xs**2 + ys**2 <= radius_sq:
                source_points_list.append([xs, ys, 0.0])
    source_points_orig_np = np.array(source_points_list, dtype=np.float32)
    num_source_points = len(source_points_orig_np)
    logger.info(f"Transducer discretized into {num_source_points} points.")
    # Note: source_points_orig_pt is created within the worker process now

    # Simulation L-range (based on experimental data)
    l_exp_min_m = l_peaks_exp_mm[0] / 1000.0
    l_exp_max_m = l_peaks_exp_mm[-1] / 1000.0
    l_min_m = max(0.0001, l_exp_min_m - 0.005)
    l_max_m = l_exp_max_m + 0.005
    num_l_points_sim = 1000
    l_values_sim_np = np.linspace(l_min_m, l_max_m, num_l_points_sim, dtype=np.float32)
    l_values_sim_mm = l_values_sim_np * 1000 # For plotting
    logger.info(f"Simulation L Range (m): {l_min_m:.4f} to {l_max_m:.4f} ({num_l_points_sim} points)")

    # Wave numbers (k values)
    k_vals_np = (K1_NP, K2_NP, K3_NP) # Pass numpy values to workers
    # k_vals_pt created within worker
    
    # --- 6. Prepare Arguments for Workers/Objective ---
    # Bundle core arguments that are constant across evaluations
    # These need to be pickleable
    core_args_for_worker = (
        l_values_sim_np,
        l_peaks_exp_mm,
        amp_peaks_exp_norm,
        str(device), # Pass device name as string
        num_source_points,
        source_points_orig_np, # Pass numpy array
        k_vals_np,             # Pass numpy k-values
        LOSS_WEIGHT_LOCATION,  # Pass location weight
        LOSS_WEIGHT_AMPLITUDE, # Pass amplitude weight
        LOSS_WEIGHT_COUNT,     # Pass count weight
        run_dir,               # Pass run directory for worker logger
        run_timestamp          # Pass timestamp for worker logger
    )

    # --- 7. Setup Iteration Plot Directory ---
    iteration_plots_dir = os.path.join(run_dir, 'iter_plots')
    try:
        os.makedirs(iteration_plots_dir, exist_ok=True)
        # Test write permission
        test_file = os.path.join(iteration_plots_dir, f'write_test_{os.getpid()}.txt')
        with open(test_file, 'w') as f: f.write('test')
        os.remove(test_file)
        logger.info(f"Iteration plot directory ready: {iteration_plots_dir}")
    except Exception as e:
        logger.error(f"Failed to create or test iteration plot directory {iteration_plots_dir}: {e}")
        iteration_plots_dir = None # Disable iteration plotting if dir fails
        logger.warning("Iteration plotting disabled due to directory error.")
        
    # --- 8. Parse Arguments (Example: --workers) ---
    parser = argparse.ArgumentParser(description='Acoustic peak fitting using Differential Evolution.')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel worker processes.')
    parser.add_argument('--maxiter', type=int, default=100, help='Maximum number of iterations for DE.')
    parser.add_argument('--popsize', type=int, default=15, help='Population size multiplier for DE (actual size = popsize * num_params).')
    args = parser.parse_args()
    actual_popsize = args.popsize * len(BOUNDS)
    logger.info(f"Workers={args.workers}, MaxIter={args.maxiter}, PopSizeMult={args.popsize} (Actual={actual_popsize})")
    logger.info(f"Loss Weights: Location={LOSS_WEIGHT_LOCATION}, Amplitude={LOSS_WEIGHT_AMPLITUDE}, Count={LOSS_WEIGHT_COUNT}")

    # --- 9. Setup Optimization Callback ---
    opt_callback = OptimizationCallback(
        logger=logger,
        patience=15,
        min_improvement=0.005,
        max_time_seconds=3600, # Increased max time to 1 hour
        min_loss_threshold=0.1,
        window_size=8,
        stability_threshold=0.001,
        divergence_threshold=1.5,
        plots_dir=iteration_plots_dir # Pass the verified/created directory
    )

    # --- 10. Run Optimization --- 
    logger.info("--- Starting Optimization ---")
    opt_start_time = time.time()
    try:
        result = custom_parallel_differential_evolution(
            objective_func=objective, # Pass the objective function itself
            bounds=np.array(BOUNDS), # Pass bounds as numpy array
            core_args=core_args_for_worker, # Pass the bundled constant args
            max_workers=args.workers,
            maxiter=args.maxiter,
            popsize=actual_popsize,
            tol=0.001, # Stricter tolerance
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=int(run_timestamp[-6:]), # Use timestamp for seed
            callback=opt_callback,
            disp=True, # Display progress (handled by callback logging now)
            logger=logger
        )
        opt_end_time = time.time()
        logger.info(f"Optimization finished in {opt_end_time - opt_start_time:.2f} seconds.")

        # --- 11. Process Results --- 
        if result and result.success:
            final_params = result.x
            final_loss = result.fun
            logger.info(f"Optimization successful! Final Loss: {final_loss:.6f}")
            logger.info("Best Parameters Found:")
            for name, value in zip(PARAM_NAMES, final_params):
                logger.info(f"  {name}: {value:.6g}")

            # Save final parameters
            best_params_path = os.path.join(run_dir, f'best_params_final.json')
            try:
                params_dict = {name: val for name, val in zip(PARAM_NAMES, final_params)}
                params_dict['final_loss'] = final_loss
                params_dict['iterations'] = result.nit
                params_dict['message'] = result.message
                with open(best_params_path, 'w') as f:
                    json.dump(params_dict, f, indent=2)
                logger.info(f"Final best parameters saved to: {best_params_path}")
            except Exception as e:
                logger.error(f"Failed to save final parameters: {e}")

            # Generate final comparison plot (using best params from callback)
            final_plot_path = os.path.join(run_dir, 'final_result_comparison.png')
            logger.info(f"Generating final comparison plot: {final_plot_path}")
            try:
                # Prepare args for generate_iteration_plot
                plot_args = {
                    'params': final_params,
                    'loss_value': final_loss,
                    'iteration': result.nit, # Use final iteration count
                    'param_id': 0, # Indicates final plot
                    'output_file': final_plot_path,
                    'is_best': True, # High quality plot
                    'logger': logger,
                    'l_values_sim_mm': l_values_sim_mm,
                    'l_peaks_exp_mm': l_peaks_exp_mm,
                    'amp_peaks_exp_norm': amp_peaks_exp_norm,
                    'simulate_func': simulate_pressure_curve,
                    'device': device,
                    'num_source_points': num_source_points,
                    # Need to re-create tensors for main process plot if needed
                    'source_points_orig_pt': torch.tensor(source_points_orig_np, dtype=torch.float32, device=device),
                    'k_vals_pt': (
                        torch.tensor(k_vals_np[0], dtype=torch.complex64, device=device),
                        torch.tensor(k_vals_np[1], dtype=torch.complex64, device=device),
                        torch.tensor(k_vals_np[2], dtype=torch.complex64, device=device)
                    )
                }
                final_plot_success = generate_iteration_plot(**plot_args)
                if final_plot_success:
                     logger.info(f"Final comparison plot saved successfully.")
                else:
                     logger.error(f"Failed to save final comparison plot.")
            except Exception as e:
                logger.error(f"Error generating final plot: {e}")
                logger.exception("Plotting error details:")

        elif result:
             logger.warning(f"Optimization did not report success. Message: {result.message}")
        else:
             logger.error("Optimization function returned None, indicating a failure.")
             
    except Exception as e:
        logger.error(f"An unexpected error occurred during optimization: {e}")
        logger.exception("Optimization Process Error:")

    finally:
        logger.info("--- Optimization Run Finished --- ")
        # Close logger handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
