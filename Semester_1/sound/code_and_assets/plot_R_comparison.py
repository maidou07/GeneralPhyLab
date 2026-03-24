import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import sys
import logging
import argparse

# --- CUDA Utilities Import ---
# Assuming cuda_utils.py is in the same directory or accessible via sys.path
try:
    from cuda_utils import setup_cuda_for_multiprocessing, print_cuda_info, get_optimal_device
except ImportError:
    print("ERROR: cuda_utils.py not found or cannot be imported.")
    # Define dummy functions if import fails
    def setup_cuda_for_multiprocessing(): print("Warning: cuda_utils not found.")
    def print_cuda_info(): print("Warning: cuda_utils not found.")
    def get_optimal_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Falling back to default device selection.")

# --- Global Constants ---
C_SOUND = 346.0
BASE_FREQ = 36981.0
TRANSDUCER_RADIUS = 0.0191
N_POINTS_PER_RADIUS = 50 # Discretization level for transducer simulation
N_REFLECTION_PAIRS = 100 # Number of reflections to simulate

# Derived constants
LAMBDA1 = C_SOUND / BASE_FREQ
K1_NP = 2 * np.pi / LAMBDA1
K2_NP = 2 * np.pi / (C_SOUND / (2 * BASE_FREQ))
K3_NP = 2 * np.pi / (C_SOUND / (3 * BASE_FREQ))

# --- Logger Setup Function ---
def setup_logger(log_filename='R_comparison.log'):
    logger = logging.getLogger('R_comparison')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional, good practice)
    try:
        file_handler = logging.FileHandler(log_filename, mode='w') # Overwrite log each run
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_filename}")
    except Exception as e:
        logger.error(f"Failed to create file handler for {log_filename}: {e}")

    return logger

# --- Simulation Functions (Copied/adapted from fit_experimental_peaks_cuda.py) ---

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
    Simulates pressure curve, adapted to handle variable number of harmonics based on k_vals_pt length.
    """
    pid = os.getpid() # Using PID though not multiprocessing, for consistency if logs merged
    logger.debug(f"[PID:{pid}] simulate_pressure_curve started")
    num_harmonics = len(k_vals_pt)

    # Unpack parameters based on the expected number of harmonics
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
            # Set defaults for unused 3rd harmonic variables
            R3_val, phi3_rad, gamma3, A3_rel = 0, 0, 0, 0
            k3_pt = torch.tensor(0+0j, device=device) # Placeholder
        elif num_harmonics == 1:
            expected_params = 3 # R1, phi1, gamma1
            if len(params) != expected_params: raise ValueError(f"Expected {expected_params} params for 1 harmonic, got {len(params)}")
            R1_val, phi1_rad, gamma1 = params
            k1_pt, = k_vals_pt # Unpack single k-value
            # Set defaults for unused higher harmonic variables
            R2_val, phi2_rad, gamma2, A2_rel = 0, 0, 0, 0
            R3_val, phi3_rad, gamma3, A3_rel = 0, 0, 0, 0
            k2_pt = torch.tensor(0+0j, device=device) # Placeholder
            k3_pt = torch.tensor(0+0j, device=device) # Placeholder
        else:
            raise ValueError(f"Unsupported number of harmonics: {num_harmonics}")
    except ValueError as e:
        logger.error(f"[PID:{pid}] Parameter or k_vals mismatch: {e}")
        return np.zeros_like(l_values_np)

    # Determine active harmonics based on parameters
    active_harmonics = []
    if R1_val > 1e-6: active_harmonics.append(1)
    if num_harmonics >= 2 and R2_val > 1e-6 and A2_rel > 1e-6: active_harmonics.append(2)
    if num_harmonics >= 3 and R3_val > 1e-6 and A3_rel > 1e-6: active_harmonics.append(3)
    logger.debug(f"[PID:{pid}] Active harmonics: {active_harmonics}")

    if not active_harmonics:
        logger.warning(f"[PID:{pid}] No active harmonics based on parameters. Returning zeros.")
        return np.zeros_like(l_values_np)

    l_values_pt = torch.tensor(l_values_np, dtype=torch.float32, device=device)

    # --- Create Tensors for Active Harmonics ---
    R1_complex_pt = torch.tensor((R1_val * np.exp(1j * phi1_rad)).astype(np.complex64), device=device)
    gamma1_pt = torch.tensor(gamma1, dtype=torch.float32, device=device)
    A1_point_pt = torch.tensor(1.0, dtype=torch.complex64, device=device) # Base amplitude for harmonic 1
    base_amplitude1_pt = torch.full((num_source_points,), A1_point_pt.item(), dtype=torch.complex64, device=device)

    # Conditional creation for 2nd harmonic
    if 2 in active_harmonics:
        R2_complex_pt = torch.tensor((R2_val * np.exp(1j * phi2_rad)).astype(np.complex64), device=device)
        gamma2_pt = torch.tensor(gamma2, dtype=torch.float32, device=device)
        A2_point_pt = torch.tensor(A2_rel, dtype=torch.complex64, device=device)
        base_amplitude2_pt = torch.full((num_source_points,), A2_point_pt.item(), dtype=torch.complex64, device=device)
    else: # Provide placeholders if not active, though harmonic_configs prevents use
        R2_complex_pt, gamma2_pt, base_amplitude2_pt = None, None, None

    # Conditional creation for 3rd harmonic
    if 3 in active_harmonics:
        R3_complex_pt = torch.tensor((R3_val * np.exp(1j * phi3_rad)).astype(np.complex64), device=device)
        gamma3_pt = torch.tensor(gamma3, dtype=torch.float32, device=device)
        A3_point_pt = torch.tensor(A3_rel, dtype=torch.complex64, device=device)
        base_amplitude3_pt = torch.full((num_source_points,), A3_point_pt.item(), dtype=torch.complex64, device=device)
    else: # Placeholders
        R3_complex_pt, gamma3_pt, base_amplitude3_pt = None, None, None

    # --- Build Harmonic Configurations ---
    harmonic_configs = []
    if 1 in active_harmonics:
        harmonic_configs.append({"h_num": 1, "k": k1_pt, "gamma": gamma1_pt, "base_amp": base_amplitude1_pt, "R0": R1_complex_pt, "Rl": R1_complex_pt})
    if 2 in active_harmonics:
        harmonic_configs.append({"h_num": 2, "k": k2_pt, "gamma": gamma2_pt, "base_amp": base_amplitude2_pt, "R0": R2_complex_pt, "Rl": R2_complex_pt})
    if 3 in active_harmonics:
        harmonic_configs.append({"h_num": 3, "k": k3_pt, "gamma": gamma3_pt, "base_amp": base_amplitude3_pt, "R0": R3_complex_pt, "Rl": R3_complex_pt})

    # --- Simulation Loop ---
    P_total_at_l_pt = torch.zeros(len(l_values_np), dtype=torch.complex64, device=device)
    batch_size = min(100, len(l_values_np)) # Process L-values in batches
    zero_offset_pt = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)

    for config in harmonic_configs:
        h_num = config["h_num"]
        k_pt_h = config["k"]
        gamma_pt_h = config["gamma"]
        base_amp_pt_h = config["base_amp"]
        R0_pt_h = config["R0"]
        Rl_pt_h = config["Rl"]
        logger.debug(f"[PID:{pid}] Calculating harmonic {h_num}...")

        P_harmonic_total = torch.zeros_like(P_total_at_l_pt) # Accumulate for this harmonic

        # Batch processing for L values to manage memory
        for batch_start in range(0, len(l_values_np), batch_size):
            batch_end = min(batch_start + batch_size, len(l_values_np))
            l_batch_pt = l_values_pt[batch_start:batch_end]
            P_batch = torch.zeros(len(l_batch_pt), dtype=torch.complex64, device=device)

            for i, l_val_pt in enumerate(l_batch_pt):
                if l_val_pt < 1e-9: continue # Avoid division by zero at source

                l_val_float = l_val_pt.item()
                all_src_pos_list = [source_points_orig_pt]
                all_src_amp_list = [base_amp_pt_h]

                # Reflection calculation using PyTorch operations for efficiency
                current_amp_factor_pos = Rl_pt_h.clone()
                current_amp_factor_neg = (Rl_pt_h * R0_pt_h).clone() # First reflection from source end
                two_l_val_tensor = torch.tensor([0, 0, 2.0 * l_val_float], dtype=torch.float32, device=device)
                R0_Rl_prod = R0_pt_h * Rl_pt_h

                for n in range(1, N_REFLECTION_PAIRS + 1):
                    # Positive direction reflections (image sources at 2nl, 4nl, ...)
                    amp_pos = base_amp_pt_h * current_amp_factor_pos
                    pos_offset = two_l_val_tensor * n
                    all_src_pos_list.append(source_points_orig_pt + pos_offset)
                    all_src_amp_list.append(amp_pos)
                    current_amp_factor_pos = current_amp_factor_pos * R0_Rl_prod # Apply both R for next pair

                    # Negative direction reflections (image sources at -2nl, -4nl, ...)
                    amp_neg = base_amp_pt_h * current_amp_factor_neg
                    neg_offset = -two_l_val_tensor * n # Negate the offset
                    all_src_pos_list.append(source_points_orig_pt + neg_offset)
                    all_src_amp_list.append(amp_neg)
                    current_amp_factor_neg = current_amp_factor_neg * R0_Rl_prod # Apply both R

                # Concatenate all sources and amplitudes for this L point
                all_src_pt = torch.cat(all_src_pos_list, dim=0)
                all_amp_pt = torch.cat(all_src_amp_list, dim=0)

                # Calculate pressure at the target point (L) from all image sources
                target_point_l_pt = torch.tensor([[0, 0, l_val_float]], dtype=torch.float32, device=device)
                P_batch[i] = calculate_pressure_at_points_pytorch(
                    target_point_l_pt, all_src_pt, all_amp_pt, k_pt_h, gamma_pt_h
                ).squeeze() # Squeeze potential extra dim

            P_harmonic_total[batch_start:batch_end] = P_batch

        P_total_at_l_pt += P_harmonic_total # Add contribution of this harmonic
        logger.debug(f"[PID:{pid}] Harmonic {h_num} calculation finished.")

    # Final amplitude is the absolute value of the total complex pressure
    P_amplitude = torch.abs(P_total_at_l_pt).cpu().numpy()
    logger.debug(f"[PID:{pid}] simulate_pressure_curve finished")
    return P_amplitude


# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot pressure amplitude vs distance for different R values.')
    parser.add_argument('--disable_third_harmonic', action='store_true', help='Disable simulation of the third harmonic (keeps 1st and 2nd).')
    parser.add_argument('--disable_higher_harmonics', action='store_true', help='Disable simulation of ALL higher harmonics (keeps only 1st).')
    parser.add_argument('--output', type=str, default='sound/img/R_comparison_plot', help='Base output plot file path (without extension).')
    parser.add_argument('--log_file', type=str, default='sound/logs/R_comparison.log', help='Log file path.')
    parser.add_argument('--L_min', type=float, default=0.001, help='Minimum distance (m) for simulation (1mm default).')
    parser.add_argument('--L_max', type=float, default=0.080, help='Maximum distance (m) for simulation (80mm default).')
    parser.add_argument('--num_L_points', type=int, default=2000, help='Number of distance points for simulation.')
    parser.add_argument('--inset_near_L_max_mm', type=float, default=20.0, help='Maximum distance (mm) for the near-field inset plot (1-20mm).')
    parser.add_argument('--far_field_center_mm', type=float, default=60.0, help='Center distance (mm) for the far-field inset plot.')
    parser.add_argument('--far_field_width_mm', type=float, default=LAMBDA1*1000/2, help='Width (mm) for the far-field inset plot (default lambda1/2).')
    args = parser.parse_args()

    # Handle harmonic disabling logic
    if args.disable_higher_harmonics:
        args.disable_third_harmonic = True # Disabling higher implies disabling third

    # Ensure output/log directories exist
    output_dir = os.path.dirname(args.output)
    log_dir = os.path.dirname(args.log_file)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.log_file)
    logger.info("--- Starting R Comparison Plot Generation ---")

    # Setup CUDA
    setup_cuda_for_multiprocessing() # For consistency, though not strictly needed here
    device = get_optimal_device()
    print_cuda_info() # Log CUDA details
    logger.info(f"Using device: {device}")

    # --- Define Fixed Parameters based on harmonic flags ---
    if args.disable_higher_harmonics:
        # Keep only 1st harmonic
        # Parameters: phi1, gam1
        fixed_params_tuple = (0.0, 1.0)
        k_vals_np = (K1_NP,)
        param_indices = {"phi1": 1, "gam1": 2} # Indices in the 3-param list [R1, phi1, gam1]
        num_sim_params = 3
        title_suffix = "(1st Harmonic Only)"
        logger.info("Simulating only 1st harmonic.")
    elif args.disable_third_harmonic:
        # Keep 1st and 2nd harmonics
        # Parameters: phi1, gam1, phi2, gam2, A2
        fixed_params_tuple = (0.0, 1.0, 0.0, 5.0, 0.3)
        k_vals_np = (K1_NP, K2_NP)
        param_indices = {"phi1": 1, "gam1": 2, "phi2": 4, "gam2": 5, "A2": 6} # Indices in the 7-param list
        num_sim_params = 7
        title_suffix = "(1st & 2nd Harmonics)"
        logger.info("Simulating 1st and 2nd harmonics.")
    else:
        # Keep all three harmonics (default)
        # Parameters: phi1, gam1, phi2, gam2, A2, phi3, gam3, A3
        fixed_params_tuple = (0.0, 1.0, 0.0, 5.0, 0.3, 0.0, 10.0, 0.1)
        k_vals_np = (K1_NP, K2_NP, K3_NP)
        param_indices = {"phi1": 1, "gam1": 2, "phi2": 4, "gam2": 5, "A2": 9, "phi3": 7, "gam3": 8, "A3": 10} # Indices in the 11-param list
        num_sim_params = 11
        title_suffix = "(1st, 2nd & 3rd Harmonics)"
        logger.info("Simulating 1st, 2nd, and 3rd harmonics.")

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
            if xs**2 + ys**2 <= radius_sq:
                source_points_list.append([xs, ys, 0.0])
    source_points_orig_np = np.array(source_points_list, dtype=np.float32)
    num_source_points = len(source_points_orig_np)
    source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device)
    logger.info(f"Transducer discretized into {num_source_points} points.")

    # Simulation L-range
    l_values_sim_np = np.linspace(args.L_min, args.L_max, args.num_L_points, dtype=np.float32)
    l_values_sim_mm = l_values_sim_np * 1000 # For plotting
    logger.info(f"Simulation L Range: {args.L_min*1000:.1f} mm to {args.L_max*1000:.1f} mm ({args.num_L_points} points)")

    # --- Simulation Loop ---
    R_values_to_plot = [0.7, 0.8, 0.9, 0.95, 1.0]
    results = {} # Store P_amplitude for each R

    logger.info(f"Starting simulations for R = {R_values_to_plot}...")
    sim_start_time = time.time()
    for R in R_values_to_plot:
        iter_start_time = time.time()
        logger.info(f"  Simulating for R = {R:.2f}...")

        # Construct the parameter list dynamically based on num_sim_params
        current_params = [0.0] * num_sim_params
        current_params[0] = R # R1
        current_params[param_indices["phi1"]] = fixed_params_tuple[0]
        current_params[param_indices["gam1"]] = fixed_params_tuple[1]
        
        if num_sim_params >= 7: # If 2nd harmonic is included
            current_params[3] = R # R2
            current_params[param_indices["phi2"]] = fixed_params_tuple[2]
            current_params[param_indices["gam2"]] = fixed_params_tuple[3]
            current_params[param_indices["A2"]] = fixed_params_tuple[4]
            
        if num_sim_params >= 11: # If 3rd harmonic is included
            current_params[6] = R # R3
            current_params[param_indices["phi3"]] = fixed_params_tuple[5]
            current_params[param_indices["gam3"]] = fixed_params_tuple[6]
            current_params[param_indices["A3"]] = fixed_params_tuple[7]

        # Call simulation function (ensure it handles variable param length correctly)
        P_amp = simulate_pressure_curve(
             l_values_sim_np,
             current_params,
             device,
             num_source_points,
             source_points_orig_pt,
             k_vals_pt,
             logger
         )
        results[R] = P_amp
        iter_end_time = time.time()
        logger.info(f"    -> R = {R:.2f} simulation finished in {iter_end_time - iter_start_time:.2f} s")

    sim_end_time = time.time()
    logger.info(f"All simulations completed in {sim_end_time - sim_start_time:.2f} seconds.")

    # --- Plotting ---
    logger.info("Generating plot...")
    plot_start_time = time.time()
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8)) # Slightly wider figure

        colors = plt.cm.viridis(np.linspace(0, 0.9, len(R_values_to_plot)))
        max_overall_amp = 0

        # --- Main Plot --- 
        for i, R in enumerate(R_values_to_plot):
            P_amp = results.get(R)
            if P_amp is not None and len(P_amp) > 0:
                current_max = np.max(P_amp)
                if current_max > max_overall_amp:
                    max_overall_amp = current_max
                ax.plot(l_values_sim_mm, P_amp, label=f'R = {R:.2f}', color=colors[i], linewidth=1.5, alpha=0.9)
            else:
                 logger.warning(f"No simulation data found for R = {R:.2f}, skipping main plot.")

        # Main plot enhancements
        ax.set_xlabel('Distance from Source (mm)', fontsize=13)
        ax.set_ylabel('Simulated Pressure Amplitude (Arb. Units)', fontsize=13)
        ax.set_title(f'Effect of Reflection Coefficient (R) on Pressure Amplitude {title_suffix}', fontsize=15, pad=15)
        ax.legend(title="Reflection Coeff (R)", fontsize=11, title_fontsize=12, loc='upper right', frameon=True, shadow=True)
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=11, direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.set_xlim(1.0, args.L_max * 1000)
        if max_overall_amp > 0:
            ax.set_ylim(bottom=0, top=max_overall_amp * 1.1)

        # --- Inset Plot 1: Near Field (dB) ---
        axins1 = ax.inset_axes([0.05, 0.55, 0.35, 0.35]) # Position near-field inset top-left

        # Set the limits for the near-field inset plot (1-20 mm)
        x1_near, x2_near = 1.0, args.inset_near_L_max_mm
        axins1.set_xlim(x1_near, x2_near)

        # Calculate max amplitude for dB reference ONLY within near-field inset range
        max_near_amp = 0
        for R in R_values_to_plot:
            P_amp = results.get(R)
            if P_amp is not None and len(P_amp) > 0:
                mask = (l_values_sim_mm >= x1_near) & (l_values_sim_mm <= x2_near)
                if np.any(mask):
                    current_near_max = np.max(P_amp[mask])
                    if current_near_max > max_near_amp:
                         max_near_amp = current_near_max
        
        near_epsilon = max_near_amp * 1e-7 if max_near_amp > 1e-9 else 1e-9
        logger.info(f"Near Field Inset max amplitude for dB ref: {max_near_amp:.4g}, Epsilon: {near_epsilon:.4g}")
        min_db_near = 0 

        # Plot dB values on the near-field inset axes
        for i, R in enumerate(R_values_to_plot):
            P_amp = results.get(R)
            if P_amp is not None and len(P_amp) > 0 and max_near_amp > 0:
                P_amp_safe = np.maximum(P_amp, near_epsilon)
                P_db_near = 20 * np.log10(P_amp_safe / max_near_amp)
                axins1.plot(l_values_sim_mm, P_db_near, color=colors[i], linewidth=1.0, alpha=0.8)
                mask = (l_values_sim_mm >= x1_near) & (l_values_sim_mm <= x2_near)
                if np.any(mask):
                     min_db_in_range = np.min(P_db_near[mask])
                     if min_db_in_range < min_db_near:
                         min_db_near = min_db_in_range
            else: pass

        lower_db_limit_near = max(min_db_near - 5, -60)
        upper_db_limit_near = 3
        axins1.set_ylim(lower_db_limit_near, upper_db_limit_near)
        axins1.set_title(f'Near Field ({x1_near:.0f}-{x2_near:.0f} mm)', fontsize=10)
        axins1.set_ylabel("Amplitude (dB re near max)", fontsize=10)
        axins1.tick_params(axis='both', which='major', labelsize=9)
        axins1.grid(True, alpha=0.5, linestyle=':')
        ax.indicate_inset_zoom(axins1, edgecolor="black", alpha=0.8, linewidth=0.8)

        # --- Inset Plot 2: Far Field (Linear Amplitude) ---
        axins2 = ax.inset_axes([0.55, 0.55, 0.4, 0.35]) # Position far-field inset middle/top-right

        # Calculate limits for the far-field inset plot
        half_width = args.far_field_width_mm / 2.0
        x1_far = max(args.L_min * 1000, args.far_field_center_mm - half_width)
        x2_far = min(args.L_max * 1000, args.far_field_center_mm + half_width)
        axins2.set_xlim(x1_far, x2_far)
        logger.info(f"Far Field Inset range: {x1_far:.2f} mm to {x2_far:.2f} mm")

        # Plot linear amplitude and find max within the far-field range
        max_far_amp = 0
        for i, R in enumerate(R_values_to_plot):
            P_amp = results.get(R)
            if P_amp is not None and len(P_amp) > 0:
                axins2.plot(l_values_sim_mm, P_amp, color=colors[i], linewidth=1.0, alpha=0.8)
                mask = (l_values_sim_mm >= x1_far) & (l_values_sim_mm <= x2_far)
                if np.any(mask):
                    current_far_max = np.max(P_amp[mask])
                    if current_far_max > max_far_amp:
                         max_far_amp = current_far_max
            else: pass

        if max_far_amp > 0:
            axins2.set_ylim(0, max_far_amp * 1.1)
        else:
            axins2.set_ylim(0, 1)

        # Style the far-field inset plot
        axins2.set_title(f'Far Field Detail ({x1_far:.1f}-{x2_far:.1f} mm)', fontsize=10)
        axins2.set_ylabel("Amplitude (Arb. Units)", fontsize=10) # Linear scale
        axins2.tick_params(axis='both', which='major', labelsize=9)
        axins2.grid(True, alpha=0.5, linestyle=':')
        ax.indicate_inset_zoom(axins2, edgecolor="black", alpha=0.8, linewidth=0.8)

        plt.tight_layout()

        # --- Save Main Plot ---
        base_output_path = args.output
        main_png_path = base_output_path + ".png"
        main_pdf_path = base_output_path + ".pdf"
        try:
            plt.savefig(main_png_path, dpi=200)
            logger.info(f"Main plot saved successfully to: {main_png_path}")
        except Exception as e:
            logger.error(f"Failed to save main PNG plot to {main_png_path}: {e}")
        try:
            plt.savefig(main_pdf_path, format='pdf')
            logger.info(f"Main plot saved successfully to: {main_pdf_path}")
        except Exception as e:
            logger.error(f"Failed to save main PDF plot to {main_pdf_path}: {e}")

        # --- Create and Save Separate Far-Field Plot ---
        logger.info("Generating separate far-field plot...")
        try:
            fig_far, ax_far = plt.subplots(figsize=(8, 5))
            max_far_amp_standalone = 0
            for i, R in enumerate(R_values_to_plot):
                 P_amp = results.get(R)
                 if P_amp is not None and len(P_amp) > 0:
                     ax_far.plot(l_values_sim_mm, P_amp, color=colors[i], linewidth=1.5, alpha=0.9, label=f'R = {R:.2f}')
                     mask = (l_values_sim_mm >= x1_far) & (l_values_sim_mm <= x2_far)
                     if np.any(mask):
                         current_far_max = np.max(P_amp[mask])
                         if current_far_max > max_far_amp_standalone:
                              max_far_amp_standalone = current_far_max
                 else: pass
            
            ax_far.set_xlim(x1_far, x2_far)
            if max_far_amp_standalone > 0:
                ax_far.set_ylim(0, max_far_amp_standalone * 1.1)
            else:
                ax_far.set_ylim(0, 1)
                
            ax_far.set_xlabel('Distance from Source (mm)', fontsize=12)
            ax_far.set_ylabel('Simulated Pressure Amplitude (Arb. Units)', fontsize=12)
            ax_far.set_title(f'Far Field Zoom ({x1_far:.1f}-{x2_far:.1f} mm) {title_suffix}', fontsize=14)
            ax_far.legend(title="Reflection Coeff (R)", fontsize=10, title_fontsize=11)
            ax_far.grid(True, which='major', linestyle='--', alpha=0.7)
            ax_far.grid(True, which='minor', linestyle=':', alpha=0.5)
            ax_far.tick_params(axis='both', which='major', labelsize=10)
            ax_far.minorticks_on()
            plt.tight_layout()
            
            far_png_path = base_output_path + "_far_field_zoom.png"
            far_pdf_path = base_output_path + "_far_field_zoom.pdf"
            
            try:
                plt.savefig(far_png_path, dpi=150)
                logger.info(f"Separate far-field plot saved to: {far_png_path}")
            except Exception as e:
                logger.error(f"Failed to save separate PNG plot to {far_png_path}: {e}")
            try:
                plt.savefig(far_pdf_path, format='pdf')
                logger.info(f"Separate far-field plot saved to: {far_pdf_path}")
            except Exception as e:
                logger.error(f"Failed to save separate PDF plot to {far_pdf_path}: {e}")
                
        except Exception as far_plot_err:
            logger.error(f"An error occurred during separate far-field plotting: {far_plot_err}")
        finally:
             if 'fig_far' in locals() and plt.fignum_exists(fig_far.number):
                  plt.close(fig_far)

    except Exception as plot_err:
        logger.error(f"An error occurred during plotting: {plot_err}")
        logger.exception("Plotting Error Details:")
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)

    plot_end_time = time.time()
    logger.info(f"Plotting finished in {plot_end_time - plot_start_time:.2f} seconds.")
    logger.info("--- Script Finished ---") 