import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import sys
import logging
import argparse
from scipy.signal import find_peaks

# --- CUDA Utilities Import ---
# Assuming cuda_utils.py is in the same directory or accessible via sys.path
try:
    from cuda_utils import setup_cuda_for_multiprocessing, print_cuda_info, get_optimal_device
except ImportError:
    print("ERROR: cuda_utils.py not found or cannot be imported.")
    def setup_cuda_for_multiprocessing(): print("Warning: cuda_utils not found.")
    def print_cuda_info(): print("Warning: cuda_utils not found.")
    def get_optimal_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Falling back to default device selection.")

# --- Global Constants ---
C_SOUND = 346.0
BASE_FREQ = 36981.0
TRANSDUCER_RADIUS = 0.0191
N_POINTS_PER_RADIUS = 50
N_REFLECTION_PAIRS = 100 # Use the increased value

# Derived constants
LAMBDA1 = C_SOUND / BASE_FREQ
K1_NP = 2 * np.pi / LAMBDA1
K2_NP = 2 * np.pi / (C_SOUND / (2 * BASE_FREQ))
K3_NP = 2 * np.pi / (C_SOUND / (3 * BASE_FREQ))

# --- Logger Setup Function ---
def setup_logger(log_filename='single_R_plot.log'):
    logger = logging.getLogger('single_R_plot')
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File Handler
    try:
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_filename}")
    except Exception as e:
        logger.error(f"Failed to create file handler for {log_filename}: {e}")
    return logger

# --- Simulation Functions (Copied/adapted) ---
def calculate_pressure_at_points_pytorch(field_points_pt, source_positions_pt, source_amplitudes_pt, k_pt, gamma_pt):
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
    pid = os.getpid()
    logger.debug(f"[PID:{pid}] simulate_pressure_curve started")
    num_harmonics = len(k_vals_pt)
    try:
        if num_harmonics == 3:
            expected_params = 11
            if len(params) != expected_params: raise ValueError(f"Expected {expected_params} params for 3 harmonics, got {len(params)}")
            R1_val, phi1_rad, gamma1, R2_val, phi2_rad, gamma2, R3_val, phi3_rad, gamma3, A2_rel, A3_rel = params
            k1_pt, k2_pt, k3_pt = k_vals_pt
        elif num_harmonics == 2:
            expected_params = 7
            if len(params) != expected_params: raise ValueError(f"Expected {expected_params} params for 2 harmonics, got {len(params)}")
            R1_val, phi1_rad, gamma1, R2_val, phi2_rad, gamma2, A2_rel = params
            k1_pt, k2_pt = k_vals_pt
            R3_val, phi3_rad, gamma3, A3_rel = 0, 0, 0, 0
            k3_pt = torch.tensor(0+0j, device=device)
        elif num_harmonics == 1:
            expected_params = 3
            if len(params) != expected_params: raise ValueError(f"Expected {expected_params} params for 1 harmonic, got {len(params)}")
            R1_val, phi1_rad, gamma1 = params
            k1_pt, = k_vals_pt
            R2_val, phi2_rad, gamma2, A2_rel = 0, 0, 0, 0
            R3_val, phi3_rad, gamma3, A3_rel = 0, 0, 0, 0
            k2_pt = torch.tensor(0+0j, device=device)
            k3_pt = torch.tensor(0+0j, device=device)
        else:
            raise ValueError(f"Unsupported number of harmonics: {num_harmonics}")
    except ValueError as e:
        logger.error(f"[PID:{pid}] Parameter or k_vals mismatch: {e}")
        return np.zeros_like(l_values_np)

    active_harmonics = []
    if R1_val > 1e-6: active_harmonics.append(1)
    if num_harmonics >= 2 and R2_val > 1e-6 and A2_rel > 1e-6: active_harmonics.append(2)
    if num_harmonics >= 3 and R3_val > 1e-6 and A3_rel > 1e-6: active_harmonics.append(3)
    logger.debug(f"[PID:{pid}] Active harmonics: {active_harmonics}")
    if not active_harmonics: return np.zeros_like(l_values_np)

    l_values_pt = torch.tensor(l_values_np, dtype=torch.float32, device=device)
    R1_complex_pt = torch.tensor((R1_val * np.exp(1j * phi1_rad)).astype(np.complex64), device=device)
    gamma1_pt = torch.tensor(gamma1, dtype=torch.float32, device=device)
    base_amplitude1_pt = torch.full((num_source_points,), 1.0, dtype=torch.complex64, device=device)
    if 2 in active_harmonics:
        R2_complex_pt = torch.tensor((R2_val * np.exp(1j * phi2_rad)).astype(np.complex64), device=device)
        gamma2_pt = torch.tensor(gamma2, dtype=torch.float32, device=device)
        base_amplitude2_pt = torch.full((num_source_points,), A2_rel, dtype=torch.complex64, device=device)
    else: R2_complex_pt, gamma2_pt, base_amplitude2_pt = None, None, None
    if 3 in active_harmonics:
        R3_complex_pt = torch.tensor((R3_val * np.exp(1j * phi3_rad)).astype(np.complex64), device=device)
        gamma3_pt = torch.tensor(gamma3, dtype=torch.float32, device=device)
        base_amplitude3_pt = torch.full((num_source_points,), A3_rel, dtype=torch.complex64, device=device)
    else: R3_complex_pt, gamma3_pt, base_amplitude3_pt = None, None, None

    harmonic_configs = []
    if 1 in active_harmonics: harmonic_configs.append({"h_num": 1, "k": k1_pt, "gamma": gamma1_pt, "base_amp": base_amplitude1_pt, "R0": R1_complex_pt, "Rl": R1_complex_pt})
    if 2 in active_harmonics: harmonic_configs.append({"h_num": 2, "k": k2_pt, "gamma": gamma2_pt, "base_amp": base_amplitude2_pt, "R0": R2_complex_pt, "Rl": R2_complex_pt})
    if 3 in active_harmonics: harmonic_configs.append({"h_num": 3, "k": k3_pt, "gamma": gamma3_pt, "base_amp": base_amplitude3_pt, "R0": R3_complex_pt, "Rl": R3_complex_pt})

    P_total_at_l_pt = torch.zeros(len(l_values_np), dtype=torch.complex64, device=device)
    batch_size = min(100, len(l_values_np))
    zero_offset_pt = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)

    for config in harmonic_configs:
        h_num, k_pt_h, gamma_pt_h = config["h_num"], config["k"], config["gamma"]
        base_amp_pt_h, R0_pt_h, Rl_pt_h = config["base_amp"], config["R0"], config["Rl"]
        logger.debug(f"[PID:{pid}] Calculating harmonic {h_num}...")
        P_harmonic_total = torch.zeros_like(P_total_at_l_pt)
        for batch_start in range(0, len(l_values_np), batch_size):
            batch_end = min(batch_start + batch_size, len(l_values_np))
            l_batch_pt = l_values_pt[batch_start:batch_end]
            P_batch = torch.zeros(len(l_batch_pt), dtype=torch.complex64, device=device)
            for i, l_val_pt in enumerate(l_batch_pt):
                if l_val_pt < 1e-9: continue
                l_val_float = l_val_pt.item()
                all_src_pos_list = [source_points_orig_pt]
                all_src_amp_list = [base_amp_pt_h]
                current_amp_factor_pos = Rl_pt_h.clone()
                current_amp_factor_neg = (Rl_pt_h * R0_pt_h).clone()
                two_l_val_tensor = torch.tensor([0, 0, 2.0 * l_val_float], dtype=torch.float32, device=device)
                R0_Rl_prod = R0_pt_h * Rl_pt_h
                for n in range(1, N_REFLECTION_PAIRS + 1):
                    amp_pos = base_amp_pt_h * current_amp_factor_pos
                    pos_offset = two_l_val_tensor * n
                    all_src_pos_list.append(source_points_orig_pt + pos_offset)
                    all_src_amp_list.append(amp_pos)
                    current_amp_factor_pos *= R0_Rl_prod
                    amp_neg = base_amp_pt_h * current_amp_factor_neg
                    neg_offset = -two_l_val_tensor * n
                    all_src_pos_list.append(source_points_orig_pt + neg_offset)
                    all_src_amp_list.append(amp_neg)
                    current_amp_factor_neg *= R0_Rl_prod
                all_src_pt = torch.cat(all_src_pos_list, dim=0)
                all_amp_pt = torch.cat(all_src_amp_list, dim=0)
                target_point_l_pt = torch.tensor([[0, 0, l_val_float]], dtype=torch.float32, device=device)
                P_batch[i] = calculate_pressure_at_points_pytorch(
                    target_point_l_pt, all_src_pt, all_amp_pt, k_pt_h, gamma_pt_h
                ).squeeze()
            P_harmonic_total[batch_start:batch_end] = P_batch
        P_total_at_l_pt += P_harmonic_total
        logger.debug(f"[PID:{pid}] Harmonic {h_num} calculation finished.")
    P_amplitude = torch.abs(P_total_at_l_pt).cpu().numpy()
    logger.debug(f"[PID:{pid}] simulate_pressure_curve finished")
    return P_amplitude

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot pressure amplitude vs distance for R=0.95 with peak identification.')
    parser.add_argument('--disable_third_harmonic', action='store_true', help='Disable simulation of the third harmonic (keeps 1st and 2nd).')
    parser.add_argument('--disable_higher_harmonics', action='store_true', help='Disable simulation of ALL higher harmonics (keeps only 1st).')
    parser.add_argument('--output', type=str, default='sound/img/R_0.95_plot', help='Base output plot file path (without extension).')
    parser.add_argument('--log_file', type=str, default='sound/logs/R_0.95_plot.log', help='Log file path.')
    parser.add_argument('--L_min', type=float, default=0.001, help='Minimum distance (m) for simulation (suggested: 1mm).')
    parser.add_argument('--L_max', type=float, default=0.080, help='Maximum distance (m) for simulation (suggested: 80mm).')
    parser.add_argument('--num_L_points', type=int, default=2000, help='Number of distance points for simulation (increase for smoother curve/peaks).')
    parser.add_argument('--peak_height', type=float, default=0.05, help='Minimum normalized peak height for detection.')
    parser.add_argument('--peak_distance_mm', type=float, default=LAMBDA1*1000/4, help='Minimum distance between peaks in mm.')

    args = parser.parse_args()

    # Handle harmonic disabling logic
    if args.disable_higher_harmonics: args.disable_third_harmonic = True

    # Ensure output/log directories exist
    output_dir = os.path.dirname(args.output)
    log_dir = os.path.dirname(args.log_file)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.log_file)
    logger.info("--- Starting Single R Plot Generation (R=0.95) ---")

    # Setup CUDA
    setup_cuda_for_multiprocessing()
    device = get_optimal_device()
    print_cuda_info()
    logger.info(f"Using device: {device}")

    # --- Define Fixed Parameters for R=0.95 ---
    R_fixed = 0.95
    if args.disable_higher_harmonics:
        fixed_params_tuple = (0.0, 1.0) # phi1, gam1
        k_vals_np = (K1_NP,)
        param_indices = {"phi1": 1, "gam1": 2}
        num_sim_params = 3
        title_suffix = "(1st Harmonic Only)"
        logger.info("Simulating only 1st harmonic for R=0.95.")
    elif args.disable_third_harmonic:
        fixed_params_tuple = (0.0, 1.0, 0.0, 5.0, 0.3) # phi1, gam1, phi2, gam2, A2
        k_vals_np = (K1_NP, K2_NP)
        param_indices = {"phi1": 1, "gam1": 2, "phi2": 4, "gam2": 5, "A2": 6}
        num_sim_params = 7
        title_suffix = "(1st & 2nd Harmonics)"
        logger.info("Simulating 1st and 2nd harmonics for R=0.95.")
    else:
        fixed_params_tuple = (0.0, 1.0, 0.0, 5.0, 0.3, 0.0, 10.0, 0.1) # Full set
        k_vals_np = (K1_NP, K2_NP, K3_NP)
        param_indices = {"phi1": 1, "gam1": 2, "phi2": 4, "gam2": 5, "A2": 9, "phi3": 7, "gam3": 8, "A3": 10}
        num_sim_params = 11
        title_suffix = "(1st, 2nd & 3rd Harmonics)"
        logger.info("Simulating 1st, 2nd, and 3rd harmonics for R=0.95.")

    k_vals_pt = tuple(torch.tensor(k, dtype=torch.complex64, device=device) for k in k_vals_np)
    logger.info(f"Fixed parameters (excluding R): {fixed_params_tuple}")

    # --- Prepare Simulation Setup ---
    logger.info("Preparing transducer geometry...")
    source_points_list = []
    xs_grid = np.linspace(-TRANSDUCER_RADIUS, TRANSDUCER_RADIUS, 2 * N_POINTS_PER_RADIUS + 1, dtype=np.float32)
    ys_grid = np.linspace(-TRANSDUCER_RADIUS, TRANSDUCER_RADIUS, 2 * N_POINTS_PER_RADIUS + 1, dtype=np.float32)
    radius_sq = TRANSDUCER_RADIUS**2
    for xs in xs_grid:
        for ys in ys_grid:
            if xs**2 + ys**2 <= radius_sq: source_points_list.append([xs, ys, 0.0])
    source_points_orig_np = np.array(source_points_list, dtype=np.float32)
    num_source_points = len(source_points_orig_np)
    source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device)
    logger.info(f"Transducer discretized into {num_source_points} points.")

    # Simulation L-range
    l_values_sim_np = np.linspace(args.L_min, args.L_max, args.num_L_points, dtype=np.float32)
    l_values_sim_mm = l_values_sim_np * 1000
    logger.info(f"Simulation L Range: {args.L_min*1000:.1f} mm to {args.L_max*1000:.1f} mm ({args.num_L_points} points)")

    # --- Simulation for R=0.95 ---
    logger.info(f"Starting simulation for R = {R_fixed:.2f}...")
    sim_start_time = time.time()

    current_params = [0.0] * num_sim_params
    current_params[0] = R_fixed # R1
    current_params[param_indices["phi1"]] = fixed_params_tuple[0]
    current_params[param_indices["gam1"]] = fixed_params_tuple[1]
    if num_sim_params >= 7:
        current_params[3] = R_fixed # R2
        current_params[param_indices["phi2"]] = fixed_params_tuple[2]
        current_params[param_indices["gam2"]] = fixed_params_tuple[3]
        current_params[param_indices["A2"]] = fixed_params_tuple[4]
    if num_sim_params >= 11:
        current_params[6] = R_fixed # R3
        current_params[param_indices["phi3"]] = fixed_params_tuple[5]
        current_params[param_indices["gam3"]] = fixed_params_tuple[6]
        current_params[param_indices["A3"]] = fixed_params_tuple[7]

    P_amplitude = simulate_pressure_curve(
         l_values_sim_np, current_params, device,
         num_source_points, source_points_orig_pt, k_vals_pt, logger
     )
    sim_end_time = time.time()
    logger.info(f"Simulation completed in {sim_end_time - sim_start_time:.2f} seconds.")

    # --- Peak Finding & Classification ---
    logger.info("Finding and classifying peaks...")
    max_amp = np.max(P_amplitude)
    P_amp_norm = P_amplitude / max_amp if max_amp > 1e-9 else P_amplitude

    l_step_mm = (l_values_sim_mm[1] - l_values_sim_mm[0]) if len(l_values_sim_mm) > 1 else 1
    peak_distance_points = int(np.ceil(args.peak_distance_mm / l_step_mm))
    logger.info(f"Peak finding: Min height={args.peak_height}, Min distance={args.peak_distance_mm}mm (~{peak_distance_points} points)")

    # 1. Find all local maxima
    peaks_indices, properties = find_peaks(P_amp_norm, height=args.peak_height, distance=peak_distance_points)

    principal_max_indices = []
    secondary_max_indices = []

    if len(peaks_indices) > 0:
        l_peaks_found_mm = l_values_sim_mm[peaks_indices]
        lambda1_half_mm = (LAMBDA1 / 2.0) * 1000.0
        
        # 2. Calculate expected principal maxima locations (n * lambda1 / 2)
        min_l_mm = args.L_min * 1000
        max_l_mm = args.L_max * 1000
        n_max = int(np.floor(max_l_mm / lambda1_half_mm))
        expected_principal_locs_mm = np.arange(1, n_max + 1) * lambda1_half_mm
        # Filter locations within the actual L range being plotted
        expected_principal_locs_mm = expected_principal_locs_mm[(expected_principal_locs_mm >= min_l_mm) & (expected_principal_locs_mm <= max_l_mm)]
        logger.debug(f"Expected principal maxima locations (mm): {expected_principal_locs_mm}")

        # 3. Classify peaks based on proximity to expected locations
        # Define tolerance for matching (e.g., lambda1 / 8)
        tolerance_mm = lambda1_half_mm / 4.0 
        logger.info(f"Classifying peaks using tolerance: +/- {tolerance_mm:.3f} mm around expected locations.")

        expected_locs_reshaped = expected_principal_locs_mm.reshape(-1, 1)
        l_peaks_reshaped = l_peaks_found_mm.reshape(1, -1)
        
        # Calculate distance from each found peak to each expected location
        dist_matrix = np.abs(expected_locs_reshaped - l_peaks_reshaped)
        
        # Find the minimum distance for each found peak to any expected location
        min_dist_to_expected = np.min(dist_matrix, axis=0)
        
        # Classify based on the minimum distance
        is_principal = min_dist_to_expected < tolerance_mm
        
        principal_max_indices = peaks_indices[is_principal]
        secondary_max_indices = peaks_indices[~is_principal]

        logger.info(f"Found {len(peaks_indices)} peaks. Classified {len(principal_max_indices)} as Principal, {len(secondary_max_indices)} as Secondary.")
        if len(principal_max_indices) > 0:
             logger.debug(f"Principal maxima L (mm): {l_values_sim_mm[principal_max_indices]}")
        if len(secondary_max_indices) > 0:
             logger.debug(f"Secondary maxima L (mm): {l_values_sim_mm[secondary_max_indices]}")

    else:
        logger.warning("No peaks found meeting the criteria.")

    # --- Plotting ---
    logger.info("Generating plot...")
    plot_start_time = time.time()
    try:
        plt.style.use('seaborn-v0_8-talk')
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot the main curve
        ax.plot(l_values_sim_mm, P_amplitude, label=f'Simulated Amplitude (R={R_fixed:.2f})', color='dodgerblue', linewidth=2, zorder=5)

        # Plot Principal Maxima (if any)
        if len(principal_max_indices) > 0:
            ax.plot(l_values_sim_mm[principal_max_indices], P_amplitude[principal_max_indices],
                    "x", markersize=10, markerfacecolor='black', markeredgecolor='black', label='Principal Maxima', zorder=10, linestyle='None')

        # Plot expected principal locations for reference (optional)
        # if len(expected_principal_locs_mm) > 0:
        #     ax.vlines(expected_principal_locs_mm, 0, ax.get_ylim()[1], colors='grey', linestyles='dotted', alpha=0.6, label='n * λ/2')

        # Enhance plot readability
        ax.set_xlabel('Distance from Source (mm)', fontsize=16)
        ax.set_ylabel('Simulated Pressure Amplitude (Arb. Units)', fontsize=16)
        ax.set_title(f'Simulated Pressure Amplitude for R={R_fixed:.2f} {title_suffix}', fontsize=18, pad=20)
        # Ensure legend handles potentially missing peak types
        handles, labels = ax.get_legend_handles_labels()
        if handles: # Only show legend if there is something to show
             ax.legend(handles=handles, labels=labels, fontsize=14, loc='upper right')
             
        ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.minorticks_on()

        ax.set_xlim(args.L_min * 1000, args.L_max * 1000)
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        # --- Save Plot ---
        base_output_path = args.output
        png_path = base_output_path + ".png"
        pdf_path = base_output_path + ".pdf"

        try:
            plt.savefig(png_path, dpi=150)
            logger.info(f"Plot saved successfully to: {png_path}")
        except Exception as e:
            logger.error(f"Failed to save PNG plot to {png_path}: {e}")
        try:
            plt.savefig(pdf_path, format='pdf')
            logger.info(f"Plot saved successfully to: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to save PDF plot to {pdf_path}: {e}")

    except Exception as plot_err:
        logger.error(f"An error occurred during plotting: {plot_err}")
        logger.exception("Plotting Error Details:")
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

    plot_end_time = time.time()
    logger.info(f"Plotting finished in {plot_end_time - plot_start_time:.2f} seconds.")
    logger.info("--- Script Finished ---") 