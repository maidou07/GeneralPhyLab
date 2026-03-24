import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, UnivariateSpline
from scipy.optimize import curve_fit
import os
import glob
import re

# --- Configuration ---
DATA_DIR = "/Users/matt/Documents/Physics/课程/Fall,25 普通物理实验/General PhyExp 2/virtual instrument/data/week 02"
OUTPUT_DIR = "/Users/matt/Documents/Physics/课程/Fall,25 普通物理实验/General PhyExp 2/virtual instrument/report_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# --- Helper Functions ---

def fano_func(x, A, q, x0, gamma, C):
    """
    Fano resonance formula:
    I(x) = A * ((q + epsilon)^2) / (1 + epsilon^2) + C
    where epsilon = 2 * (x - x0) / gamma
    """
    epsilon = 2 * (x - x0) / gamma
    return A * ((q + epsilon)**2) / (1 + epsilon**2) + C

def read_data(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None, None
        
    try:
        files = os.listdir(folder_path)
    except Exception as e:
        print(f"Error listing {folder_path}: {e}")
        return None, None

    target_file = None
    # Priority 1: "extra precision"
    for f in files:
        if "extra precision" in f and f.endswith(".txt"):
            target_file = f
            break
    
    # Priority 2: "detailed"
    if not target_file:
        for f in files:
            if "detailed" in f and f.endswith(".txt"):
                target_file = f
                break
    
    # Priority 3: Any txt starting with digit
    if not target_file:
        for f in files:
            if f.endswith(".txt") and f[0].isdigit():
                target_file = f
                break
                
    if not target_file:
        print(f"No data file found in {folder_path}")
        return None, None

    file_path = os.path.join(folder_path, target_file)
    try:
        data = np.loadtxt(file_path)
        # Assuming Col 0 is Freq, Col 1 is Amplitude
        freq = data[:, 0]
        amp = data[:, 1]
        return freq, amp
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def plot_group(folders, param_values, param_name, title, filename_suffix):
    plt.figure(figsize=(10, 6))
    
    # Sort folders/values based on param_values
    sorted_pairs = sorted(zip(param_values, folders), key=lambda x: x[0])
    sorted_values, sorted_folders = zip(*sorted_pairs)
    
    # Color scheme from reference
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_folders)))
    
    for i, (val, folder_name) in enumerate(zip(sorted_values, sorted_folders)):
        full_path = os.path.join(DATA_DIR, "Fano共振", folder_name)
        freq, amp = read_data(full_path)
        
        if freq is None:
            continue
            
        # Sorting by frequency just in case
        sort_idx = np.argsort(freq)
        freq = freq[sort_idx]
        amp = amp[sort_idx]
        
        # Plotting
        # Plotting
        # Scatter is done inside the loop now to match color

        
        # Fano fitting
        try:
            # Initial guesses
            # x0: peak position
            max_idx = np.argmax(amp)
            x0_guess = freq[max_idx]
            # C: baseline (min value)
            C_guess = np.min(amp)
            # A: amplitude scale
            A_guess = (np.max(amp) - np.min(amp)) / (1 + 1) # rough estimate
            # gamma: width (take a fraction of range)
            gamma_guess = (freq.max() - freq.min()) / 10.0
            # q: asymmetry (start with 1)
            q_guess = 1.0
            
            p0 = [A_guess, q_guess, x0_guess, gamma_guess, C_guess]
            
            # Fit
            popt, pcov = curve_fit(fano_func, freq, amp, p0=p0, maxfev=10000)
            
            x_smooth = np.linspace(freq.min(), freq.max(), 1000)
            y_smooth = fano_func(x_smooth, *popt)
            
            # Plot smooth line with DARK color (high alpha)
            plt.plot(x_smooth, y_smooth, color=colors[i], linewidth=2, alpha=1.0, label=f"{param_name} = {val}")
            
            # Plot scatter points with LIGHT color (low alpha)
            plt.scatter(freq, amp, color=colors[i], s=15, alpha=0.3)
            
        except Exception as e:
            print(f"Fano fit failed for {folder_name}: {e}, plotting raw line")
            # Fallback to simple line
            plt.plot(freq, amp, color=colors[i], linewidth=2, alpha=1.0, label=f"{param_name} = {val}")
            plt.scatter(freq, amp, color=colors[i], s=15, alpha=0.3)

    plt.title(title, fontsize=16)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Amplitude (V)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fano_comparison_{filename_suffix}.png"), dpi=300)
    plt.close()

def plot_component(folder_subpath, filename_pattern, title, output_name):
    folder_path = os.path.join(DATA_DIR, "表征值测量", folder_subpath)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    try:
        files = os.listdir(folder_path)
    except Exception as e:
        print(f"Error listing {folder_path}: {e}")
        return

    # Simple pattern matching (glob-like but manual)
    # filename_pattern e.g. "L=16mH*.txt" or "C=0.047muF*"
    prefix = filename_pattern.replace("*", "").replace(".txt", "")
    
    matched_files = []
    for f in files:
        if f.startswith(prefix):
            matched_files.append(os.path.join(folder_path, f))
    
    if not matched_files:
        print(f"No component files found for {filename_pattern} in {folder_path}")
        return

    plt.figure(figsize=(8, 5))
    
    # Sort by size (larger file usually means more points)
    matched_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    file_path = matched_files[0]
    
    try:
        data = np.loadtxt(file_path)
        # Col 0: Freq, Col 1: Value (L or C)
        freq = data[:, 0]
        val = data[:, 1]
        
        # Determine unit based on magnitude
        mean_val = np.mean(val)
        if mean_val < 1e-4: # Likely Farads or Henrys
             # If it's C (around 1e-7), plot in uF
             # If it's L (around 1e-2), plot in mH
             if "C=" in title or "0.047muF" in title:
                 val = val * 1e6
                 unit = r"$\mu$F"
             else:
                 val = val * 1e3
                 unit = "mH"
        else:
            unit = "Unknown"

        # Spline fitting for component
        # Sort by freq first
        sort_idx = np.argsort(freq)
        freq = freq[sort_idx]
        val = val[sort_idx]
        
        try:
            # Use B-spline as requested
            # Filter duplicates just in case
            unique_freq, unique_indices = np.unique(freq, return_index=True)
            unique_val = val[unique_indices]
            
            # k=3 for cubic B-spline
            spl = make_interp_spline(unique_freq, unique_val, k=3)
            x_smooth = np.linspace(unique_freq.min(), unique_freq.max(), 1000)
            y_smooth = spl(x_smooth)
            
            plt.plot(x_smooth, y_smooth, '-', linewidth=2, color='teal', alpha=1.0)
        except Exception as e:
            print(f"B-spline failed: {e}, using raw")
            plt.plot(freq, val, '-', linewidth=2, color='teal', alpha=1.0)
            
        plt.scatter(freq, val, color='teal', s=20, alpha=0.3)
        plt.title(f"Measured {title} vs Frequency", fontsize=14)
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel(f"Value ({unit})", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting component {file_path}: {e}")

# --- Main Execution ---

# 1. Varying C2
folders_c2 = [
    "[standard] C = 0.5muF, C2 = 0.2muF, R = 500Ohm",
    "[alter C2] C = 0.5muF, C2 = 0.03muF, R = 500Ohm",
    "[alter C2] C = 0.5muF, C2 = 0.04muF, R = 500Ohm",
    "[alter C2] C = 0.5muF, C2 = 0.05muF, R = 500Ohm",
    "[alter C2] C = 0.5muF, C2 = 0.1muF, R = 500Ohm",
    "[alter C2] C = 0.5muF, C2 = 0.3muF, R = 500Ohm"
]
vals_c2 = [0.2, 0.03, 0.04, 0.05, 0.1, 0.3] # uF
plot_group(folders_c2, vals_c2, r"C2 ($\mu$F)", r"Fano Resonance: Varying C2 (C=0.5$\mu$F, R=500$\Omega$)", "vary_c2")

# 2. Varying C
folders_c = [
    "[standard] C = 0.5muF, C2 = 0.2muF, R = 500Ohm",
    "[alter C] C = 0.02muF, C2 = 0.2muF, R = 500Ohm",
    "[alter C] C = 0.05muF, C2 = 0.2muF, R = 500Ohm",
    "[alter C] C = 0.1muF, C2 = 0.2muF, R = 500Ohm",
    "[alter C] C = 0.3muF, C2 = 0.2muF, R = 500Ohm",
    "[alter C] C = 1.0muF, C2 = 0.2muF, R = 500Ohm"
]
vals_c = [0.5, 0.02, 0.05, 0.1, 0.3, 1.0] # uF
plot_group(folders_c, vals_c, r"C ($\mu$F)", r"Fano Resonance: Varying C (C2=0.2$\mu$F, R=500$\Omega$)", "vary_c")

# 3. Varying R
folders_r = [
    "[standard] C = 0.5muF, C2 = 0.2muF, R = 500Ohm",
    "[alter R] C = 0.5muF, C2 = 0.2muF, R = 10000Ohm",
    "[alter R] C = 0.5muF, C2 = 0.2muF, R = 1000Ohm",
    "[alter R] C = 0.5muF, C2 = 0.2muF, R = 50Ohm",
    "[alter R] C = 0.5muF, C2 = 0.2muF, R = 5Ohm"
]
vals_r = [500, 10000, 1000, 50, 5] # Ohm
plot_group(folders_r, vals_r, r"R ($\Omega$)", r"Fano Resonance: Varying R (C=0.5$\mu$F, C2=0.2$\mu$F)", "vary_r")

# 4. Component Characterization
plot_component("L", "L=16mH (0.5k,10k,0.5k).txt", "L=16mH", "component_16mH.png")
plot_component("L", "L=18mH (0.5k,10k,0.5k).txt", "L=18mH", "component_18mH.png") # Assuming 18mH is the one
plot_component("C", "C=0.047muF*", r"C=0.047$\mu$F", "component_047uF.png")

print("All plots generated.")
