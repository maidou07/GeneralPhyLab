import os
import matplotlib.pyplot as plt
import numpy as np
# from scipy.interpolate import CubicSpline # No longer needed for scatter only
import csv
# import math # No longer needed

# --- Font Setup (Includes attempt for CJK characters, kept for robustness) ---
# Attempt to use a broader list of fonts for potential CJK display needs elsewhere.
# Note: If none of these fonts are available, CJK characters might still render as boxes.
# You might need to install a CJK-supporting font (e.g., SimHei, Noto Sans CJK SC)
# and potentially clear the matplotlib font cache.
try:
    # Prioritize common English/Latin fonts, then CJK fonts
    plt.rcParams['font.sans-serif'] = [
        'DejaVu Sans',       # Common on Linux/macOS
        'Arial',             # Common on Windows/macOS
        'Helvetica',         # Common on macOS
        'Arial Unicode MS',  # macOS/Windows (if installed)
        'PingFang SC',       # macOS (newer versions)
        'SimHei',            # Windows/Linux (if installed)
        'Microsoft YaHei',   # Windows
        'WenQuanYi Micro Hei', # Linux
        'Noto Sans CJK SC',  # Cross-platform (if installed)
        'sans-serif'         # Default fallback
    ]
    plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs render correctly
except Exception as e:
    print(f"Warning: Error setting fonts: {e}")
    print("Ensure suitable fonts are installed and recognized by matplotlib.")

# --- Parameters and Data Loading ---
filename = 'sound/csv/submax2.csv'
x_data = []
y_data = []

try:
    with open(filename, 'r', encoding='utf-8') as csvfile: # Specify utf-8 just in case
        reader = csv.reader(csvfile)
        # Skip header row if necessary
        # next(reader, None) # Skip the header row - Commented out as the file likely has no header
        for i, row in enumerate(reader):
            if len(row) == 2:
                try:
                    x_data.append(float(row[0]))
                    y_data.append(float(row[1]))
                except ValueError:
                    # Use i+2 because we skipped the header (row 1) and enumerate starts at 0
                    print(f"Warning: Skipping invalid data row {i+2}: {row}")
            else:
                 # Use i+2 because we skipped the header (row 1) and enumerate starts at 0
                print(f"Warning: Skipping row {i+2} with incorrect column count: {row}")

    if not x_data or not y_data:
        print(f"Error: No valid numerical data found in '{filename}' after skipping header.")
        exit()

    # Convert to numpy arrays for processing
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Sort by x-value (ensures points are plotted in order if connecting lines were used)
    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]

    print(f"First data point (after sorting): x={x_data[0]}, y={y_data[0]}")

except FileNotFoundError:
    print(f"Error: File not found '{filename}'")
    exit()
except Exception as e:
    print(f"Error reading or processing file '{filename}': {e}")
    exit()

# --- Plotting Setup ---
plt.style.use('seaborn-v0_8-whitegrid')  # Use a clean grid style
plt.figure(figsize=(12, 7), dpi=150)  # Adjusted figure size and increased DPI for better quality

# Create a visually appealing scatter plot
plt.scatter(x_data, y_data,
            label='Measured Maxima',    # English legend label
            color='#D32F2F',            # Slightly different shade of red
            s=60,                       # Reduced point size
            alpha=0.9,                  # Slightly increased opacity
            marker='o',                 # Circle marker
            edgecolor='black',          # Black edge for contrast
            linewidth=1.0,              # Adjusted edge width
            zorder=3)                   # Ensure points are above the grid

# --- Trendline Removed ---
# Trendline code was previously here and commented out.

# --- Plot Beautification ---
#plt.title('Distribution of Measured Sound Pressure Maxima', fontsize=20, fontweight='bold', pad=20) # English title
plt.xlabel('$x$ (mm)', fontsize=16, fontweight='bold', labelpad=12) # English label (kept math notation)
plt.ylabel('$U$ (mV)', fontsize=16, fontweight='bold', labelpad=12) # English label (kept math notation)

# Configure legend
legend = plt.legend(fontsize=14, loc='best', frameon=True, framealpha=0.95,
                    edgecolor='darkgray', fancybox=True, shadow=False)
legend.get_frame().set_linewidth(1.0)
# legend.get_title().set_fontweight('bold') # Optional: Bold legend title if needed

# Configure grid
plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6, color='gray') # Slightly more visible major grid
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, color='lightgray') # Slightly more visible minor grid
plt.minorticks_on()  # Enable minor ticks

# Set axis limits with padding
x_margin = (x_data.max() - x_data.min()) * 0.05
y_max_val = np.max(y_data)
y_margin = y_max_val * 0.05 # Keep 5% margin below 0
plt.xlim(x_data.min() - x_margin, x_data.max() + x_margin)
plt.ylim(0 - y_margin, y_max_val * 1.1) # Start slightly below 0, leave 10% space at top

# Customize tick parameters
plt.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=7, direction='in', top=True, right=True) # Ticks on all sides
plt.tick_params(axis='both', which='minor', width=1, length=4, direction='in', top=True, right=True) # Minor ticks on all sides

# Enhance axis spines
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

# Add data source annotation
plt.figtext(0.99, 0.01, 'Data Source: submax2.csv', fontsize=10, alpha=0.7, ha='right', va='bottom') # English text

plt.tight_layout(pad=1.5) # Adjust layout padding

# --- Save High-Quality Image ---
output_dir = 'sound/img'
if not os.path.exists(output_dir):
    os.makedirs(output_dir) # Ensure directory exists

# Define output filenames (using a more descriptive name)
base_filename = 'measured_sound_pressure_maxima_distribution'
pdf_path = os.path.join(output_dir, f'{base_filename}.pdf')
png_path = os.path.join(output_dir, f'{base_filename}.png')

try:
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {pdf_path} and {png_path}")
except Exception as e:
    print(f"Error saving image: {e}")

# --- Display Plot ---
plt.show()
