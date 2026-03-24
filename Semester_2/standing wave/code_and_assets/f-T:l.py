import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress # ADDED: Import linregress

# --- Style and Color Configuration ---
plt.style.use('seaborn-v0_8')
color_data = '#AC2B40'  # Data points (deep red)
color_fit = '#E6A58C'   # Fit line (light orange)

# --- Define the Linear Model for Fitting ---
def linear_model(x, m, b):
    """A simple linear model: y = mx + b"""
    return m * x + b

# ==============================================================================
# PART 3: Analyzing ln(fe) vs ln(T)
# ==============================================================================
print("--- Part 3: Analysis of ln(fe) vs ln(T) ---")

# --- Data Input (Final Corrected Values) ---
ln_T = np.array([2.2825, 2.97561, 3.38108, 3.66878, 3.89191])
ln_fe_T = np.array([3.575, 3.942, 4.148, 4.290, 4.391])

# --- Least-Squares Fitting with Uncertainty ---
popt_T, pcov_T = curve_fit(linear_model, ln_T, ln_fe_T)
slope_T, intercept_T = popt_T
uncertainty_slope_T, uncertainty_intercept_T = np.sqrt(np.diag(pcov_T))

# --- Calculate R^2 for goodness of fit ---
# ADDED: Calculate and print the R-squared value
_, _, r_value_T, _, _ = linregress(ln_T, ln_fe_T)
print(f"Theoretical Slope: 0.5")
print(f"Fitted Slope (m_T): {slope_T:.4f} ± {uncertainty_slope_T:.4f}")
print(f"Fitted Intercept (b_T): {intercept_T:.4f} ± {uncertainty_intercept_T:.4f}")
print(f"Coefficient of Determination (R^2): {r_value_T**2:.6f}\n") # ADDED

# --- Plotting ---
plt.figure(figsize=(10, 7))
plt.scatter(ln_T, ln_fe_T, label='data', color=color_data, marker='^', s=60, zorder=5, edgecolors='black', linewidths=0.5)
x_fit = np.linspace(ln_T.min(), ln_T.max(), 100)
y_fit = linear_model(x_fit, slope_T, intercept_T)
plt.plot(x_fit, y_fit, label=f'fit: $y = ({slope_T:.3f} \pm {uncertainty_slope_T:.3f})x + {intercept_T:.3f}$', color=color_fit, linestyle='-', linewidth=2.5)
plt.xlabel('ln($T$)', fontsize=12)
plt.ylabel('ln($f_e$)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.minorticks_on()
plt.savefig('lnfe_vs_lnT_fit.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# PART 4: Analyzing ln(fe) vs ln(L)
# ==============================================================================
print("\n--- Part 4: Analysis of ln(fe) vs ln(L) ---")

# --- Data Input (Final Corrected Values) ---
ln_L = np.array([3.689, 3.800, 3.912, 4.025, 4.137, 4.248])
ln_fe_L = np.array([4.564, 4.449, 4.332, 4.212, 4.103, 4.005])

# --- Least-Squares Fitting with Uncertainty ---
popt_L, pcov_L = curve_fit(linear_model, ln_L, ln_fe_L)
slope_L, intercept_L = popt_L
uncertainty_slope_L, uncertainty_intercept_L = np.sqrt(np.diag(pcov_L))

# --- Calculate R^2 for goodness of fit ---
# ADDED: Calculate and print the R-squared value
_, _, r_value_L, _, _ = linregress(ln_L, ln_fe_L)
print(f"Theoretical Slope: -1.0")
print(f"Fitted Slope (m_L): {slope_L:.4f} ± {uncertainty_slope_L:.4f}")
print(f"Fitted Intercept (b_L): {intercept_L:.4f} ± {uncertainty_intercept_L:.4f}")
print(f"Coefficient of Determination (R^2): {r_value_L**2:.6f}\n") # ADDED

# --- Plotting ---
plt.figure(figsize=(10, 7))
plt.scatter(ln_L, ln_fe_L, label='data', color=color_data, marker='^', s=60, zorder=5, edgecolors='black', linewidths=0.5)
x_fit = np.linspace(ln_L.min(), ln_L.max(), 100)
y_fit = linear_model(x_fit, slope_L, intercept_L)
plt.plot(x_fit, y_fit, label=f'fit: $y = ({slope_L:.3f} \pm {uncertainty_slope_L:.3f})x + {intercept_L:.3f}$', color=color_fit, linestyle='-', linewidth=2.5)
plt.xlabel('ln($L$)', fontsize=12)
plt.ylabel('ln($f_e$)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.minorticks_on()
plt.savefig('lnfe_vs_lnL_fit.pdf', dpi=300, bbox_inches='tight')
plt.show()