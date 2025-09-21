import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# --- 1. Helper Function to Calculate FLOPs ---
def flops(tokens):
    """
    Calculates the flops based on the number of tokens.
    The result is returned as a power of 2 (log2).
    """
    # Estimate FLOPs based on a simplified model
    # 2.62B is an example parameter, representing FLOPs per token
    flops_val = 2_620_000_000 * tokens
    # Convert to log base 2 for the plot's x-axis
    return round(np.log2(flops_val), 2)

def is_pareto_dominant(data):
    """
    Identifies Pareto-dominant points based on accuracy (higher is better) 
    and computational cost (lower is better).
    Returns a list of indices of Pareto-dominant points.
    """
    pareto_indices = []
    
    for i, point_i in enumerate(data):
        flops_i = flops(point_i['tokens generated'])
        accuracy_i = point_i['accuracy']
        
        is_dominated = False
        for j, point_j in enumerate(data):
            if i == j:
                continue
                
            flops_j = flops(point_j['tokens generated'])
            accuracy_j = point_j['accuracy']
            
            # Point i is dominated if point j has better or equal accuracy AND lower or equal FLOPs
            # and at least one of these is strictly better
            if (accuracy_j >= accuracy_i and flops_j <= flops_i and 
                (accuracy_j > accuracy_i or flops_j < flops_i)):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    return pareto_indices

# --- 2. Data Input ---
data = [
    {'B1': 2, 'B3': 4, 'tokens generated': 3015, 'accuracy': .5740},
    {'B1': 2, 'B3': 8, 'tokens generated': 6088, 'accuracy': .6500},
    {'B1': 2, 'B3': 16, 'tokens generated': 12965, 'accuracy': .6720},
    {'B1': 2, 'B3': 20, 'tokens generated': 17127, 'accuracy': .6844},
    {'B1': 2, 'B3': 32, 'tokens generated': 28390, 'accuracy': .6837},
    {'B1': 4, 'B3': 8, 'tokens generated': 6673, 'accuracy': .6500},
    {'B1': 4, 'B3': 16, 'tokens generated': 13773, 'accuracy': .6800},
    {'B1': 4, 'B3': 20, 'tokens generated': 17537, 'accuracy': .6700},
    {'B1': 4, 'B3': 32, 'tokens generated': 28277, 'accuracy': .6992},
    {'B1': 5, 'B3': 16, 'tokens generated': 14256, 'accuracy': .6874},
    {'B1': 5, 'B3': 20, 'tokens generated': 17555, 'accuracy': .6755},
    {'B1': 5, 'B3': 32, 'tokens generated': 26931, 'accuracy': .7042},
]


# Extract unique B1 values for color normalization and colorbar ticks
b1_values = sorted(set(d['B1'] for d in data)) if data else [0]

# --- 3. Plotting Configuration ---
# Define mappings for shapes and colors
marker_map = {
    4: 'o',   # Circle
    8: 's',   # Square
    16: '^',  # Triangle Up
    20: 'D',  # Diamond
    32: 'P'   # Plus (filled)
}

# Create a colormap and a normalizer for B1 values
cmap = plt.get_cmap('plasma')
norm = mcolors.Normalize(vmin=min(b1_values), vmax=max(b1_values))

# --- 4. Create the Plot ---
fig, ax = plt.subplots(figsize=(10, 7))

# Plot data points, iterating through each unique shape (B3)
for b3_val, marker_style in marker_map.items():
    # Filter data for the current B3 value
    subset = [d for d in data if d['B3'] == b3_val]
    if not subset:
        continue

    # Extract values for plotting
    x_flops = [flops(d['tokens generated']) for d in subset]
    y_accuracy = [d['accuracy'] for d in subset]
    colors = [cmap(norm(d['B1'])) for d in subset]

    ax.scatter(x_flops, y_accuracy, marker=marker_style, c=colors, s=100,
               edgecolors='black', alpha=0.8, zorder=3)

# --- 5. Customize the Plot ---
ax.set_title('Model Performance vs. Computational Cost', fontsize=16, pad=20)
ax.set_xlabel('Inference FLOPs ($log_2$)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

# --- 6. Create Legends ---
# Colorbar for B1 (beam_width)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('B1 (Beam Width)', fontsize=12, rotation=270, labelpad=15)
cbar.set_ticks(b1_values)

# Custom legend for B3 (branching_factor) markers
legend_elements = [
    Line2D([0], [0], marker=m, color='w', label=f'{k}',
           markerfacecolor='gray', markersize=10) for k, m in marker_map.items()
]
ax.legend(handles=legend_elements, title='B3 (Total Branching Factor)',
          bbox_to_anchor=(1.20, 0.6), loc='upper left', fontsize=10)


# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust rect to make space for legends
plt.savefig("sbs-uncertain-scaling.png", dpi=300)
plt.show()

# --- 7. Create Pareto-Dominant Plot ---
# Identify Pareto-dominant points
pareto_indices = is_pareto_dominant(data)
pareto_data = [data[i] for i in pareto_indices]

# Create second plot with only Pareto-dominant points
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Plot Pareto-dominant data points
for b3_val, marker_style in marker_map.items():
    # Filter Pareto data for the current B3 value
    subset = [d for d in pareto_data if d['B3'] == b3_val]
    if not subset:
        continue

    # Extract values for plotting
    x_flops = [flops(d['tokens generated']) for d in subset]
    y_accuracy = [d['accuracy'] for d in subset]
    colors = [cmap(norm(d['B1'])) for d in subset]

    ax2.scatter(x_flops, y_accuracy, marker=marker_style, c=colors, s=100,
                edgecolors='black', alpha=0.8, zorder=3)

# Customize the Pareto plot
ax2.set_title('Model Performance vs. Computational Cost (Pareto-Dominant Points)', fontsize=16, pad=20)
ax2.set_xlabel('Inference FLOPs ($log_2$)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

# Create legends for Pareto plot
sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=ax2, pad=0.02)
cbar2.set_label('B1 (Beam Width)', fontsize=12, rotation=270, labelpad=15)
cbar2.set_ticks(b1_values)

# Custom legend for B3 (branching_factor) markers in Pareto plot
legend_elements2 = [
    Line2D([0], [0], marker=m, color='w', label=f'{k}',
           markerfacecolor='gray', markersize=10) for k, m in marker_map.items()
]
ax2.legend(handles=legend_elements2, title='B3 (Total Branching Factor)',
           bbox_to_anchor=(1.20, 0.6), loc='upper left', fontsize=10)

# Adjust layout and save Pareto plot
plt.tight_layout(rect=(0, 0, 0.85, 1))
plt.savefig("sbs-scaling-pareto.png", dpi=300)
plt.show()