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
sbs_data = [
    {'B1': 2, 'B3': 4, 'tokens generated': 2687, 'accuracy': .5300},
    {'B1': 2, 'B3': 8, 'tokens generated': 5855, 'accuracy': .5760},
    {'B1': 2, 'B3': 16, 'tokens generated': 12954, 'accuracy': .5760},
    {'B1': 2, 'B3': 32, 'tokens generated': 28849, 'accuracy': .5640},
    {'B1': 4, 'B3': 8, 'tokens generated': 6403, 'accuracy': .5440},
    {'B1': 4, 'B3': 16, 'tokens generated': 13902, 'accuracy': .5780},
    {'B1': 4, 'B3': 20, 'tokens generated': 17436, 'accuracy': .5592},
    {'B1': 4, 'B3': 32, 'tokens generated': 25418, 'accuracy': .4920},
    {'B1': 5, 'B3': 16, 'tokens generated': 14372, 'accuracy': .5935},
    {'B1': 5, 'B3': 20, 'tokens generated': 27770, 'accuracy': .5940},
    {'B1': 5, 'B3': 32, 'tokens generated': 27778, 'accuracy': .5947},
]

sbs_uncertain_data = [
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

# --- 3. Plotting Configuration ---
# Colors and markers for the two datasets
sbs_color = 'blue'
sbs_uncertain_color = 'red'
sbs_marker = 'o'
sbs_uncertain_marker = 's'

# --- 4. Create the Plot ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plot SBS data
sbs_x = [flops(d['tokens generated']) for d in sbs_data]
sbs_y = [d['accuracy'] for d in sbs_data]
ax.scatter(sbs_x, sbs_y, color=sbs_color, marker=sbs_marker, s=100,
           edgecolors='black', alpha=0.8, label='SBS', zorder=3)

# Plot SBS Uncertain data
sbs_uncertain_x = [flops(d['tokens generated']) for d in sbs_uncertain_data]
sbs_uncertain_y = [d['accuracy'] for d in sbs_uncertain_data]
ax.scatter(sbs_uncertain_x, sbs_uncertain_y, color=sbs_uncertain_color, marker=sbs_uncertain_marker, s=100,
           edgecolors='black', alpha=0.8, label='SBS-Uncertain', zorder=3)

# --- 5. Add Pareto Frontiers ---
# Calculate Pareto frontiers for both datasets
sbs_pareto_indices = is_pareto_dominant(sbs_data)
sbs_uncertain_pareto_indices = is_pareto_dominant(sbs_uncertain_data)

# Sort Pareto points by FLOPs for line plotting
sbs_pareto_points = [(flops(sbs_data[i]['tokens generated']), sbs_data[i]['accuracy']) for i in sbs_pareto_indices]
sbs_uncertain_pareto_points = [(flops(sbs_uncertain_data[i]['tokens generated']), sbs_uncertain_data[i]['accuracy']) for i in sbs_uncertain_pareto_indices]

sbs_pareto_points.sort()
sbs_uncertain_pareto_points.sort()

if sbs_pareto_points:
    pareto_x, pareto_y = zip(*sbs_pareto_points)
    ax.plot(pareto_x, pareto_y, color=sbs_color, linestyle='--', linewidth=2, alpha=0.7, label='SBS Pareto Frontier')

if sbs_uncertain_pareto_points:
    pareto_x, pareto_y = zip(*sbs_uncertain_pareto_points)
    ax.plot(pareto_x, pareto_y, color=sbs_uncertain_color, linestyle='--', linewidth=2, alpha=0.7, label='SBS-U Pareto Frontier')

# --- 6. Customize the Plot ---
ax.set_title('SBS vs SBS-Uncertain: Performance vs Computational Cost', fontsize=16, pad=20)
ax.set_xlabel('Inference FLOPs ($log_2$)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
ax.legend(fontsize=11, loc='upper left')

# Adjust layout and save
plt.tight_layout()
plt.savefig("sbs-vs-sbs-uncertain-comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 7. Dominance Analysis ---
print("=== Dominance Analysis ===")
print(f"SBS Pareto-dominant points: {len(sbs_pareto_indices)} out of {len(sbs_data)}")
print(f"SBS-Uncertain Pareto-dominant points: {len(sbs_uncertain_pareto_indices)} out of {len(sbs_uncertain_data)}")

# Compare individual points to see which dataset dominates
combined_data = []
for i, point in enumerate(sbs_data):
    combined_data.append({**point, 'dataset': 'SBS', 'original_index': i})
for i, point in enumerate(sbs_uncertain_data):
    combined_data.append({**point, 'dataset': 'SBS-Uncertain', 'original_index': i})

# Find overall Pareto frontier
overall_pareto_indices = is_pareto_dominant(combined_data)
overall_pareto_points = [combined_data[i] for i in overall_pareto_indices]

sbs_pareto_count = sum(1 for p in overall_pareto_points if p['dataset'] == 'SBS')
sbs_uncertain_pareto_count = sum(1 for p in overall_pareto_points if p['dataset'] == 'SBS-Uncertain')

print(f"\nOverall Pareto frontier:")
print(f"SBS points on frontier: {sbs_pareto_count}")
print(f"SBS-Uncertain points on frontier: {sbs_uncertain_pareto_count}")

print(f"\nPareto-dominant points by dataset:")
for point in sorted(overall_pareto_points, key=lambda x: flops(x['tokens generated'])):
    print(f"{point['dataset']}: B1={point['B1']}, B3={point['B3']}, "
          f"FLOPs(log2)={flops(point['tokens generated']):.1f}, "
          f"Accuracy={point['accuracy']:.4f}")

# --- 8. Create Detailed Comparison Plot ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: All points with overall Pareto frontier
ax1.scatter(sbs_x, sbs_y, color=sbs_color, marker=sbs_marker, s=100,
           edgecolors='black', alpha=0.6, label='SBS')
ax1.scatter(sbs_uncertain_x, sbs_uncertain_y, color=sbs_uncertain_color, marker=sbs_uncertain_marker, s=100,
           edgecolors='black', alpha=0.6, label='SBS-Uncertain')

# Highlight overall Pareto points
for point in overall_pareto_points:
    x_val = flops(point['tokens generated'])
    y_val = point['accuracy']
    color = sbs_color if point['dataset'] == 'SBS' else sbs_uncertain_color
    marker = sbs_marker if point['dataset'] == 'SBS' else sbs_uncertain_marker
    ax1.scatter(x_val, y_val, color=color, marker=marker, s=150,
               edgecolors='gold', linewidth=3, alpha=1.0, zorder=5)

ax1.set_title('All Points with Overall Pareto Frontier\n(Gold outline = Pareto dominant)', fontsize=12)
ax1.set_xlabel('Inference FLOPs ($log_2$)')
ax1.set_ylabel('Accuracy')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Right plot: Only Pareto points comparison
pareto_sbs_x = [flops(sbs_data[i]['tokens generated']) for i in sbs_pareto_indices]
pareto_sbs_y = [sbs_data[i]['accuracy'] for i in sbs_pareto_indices]
pareto_uncertain_x = [flops(sbs_uncertain_data[i]['tokens generated']) for i in sbs_uncertain_pareto_indices]
pareto_uncertain_y = [sbs_uncertain_data[i]['accuracy'] for i in sbs_uncertain_pareto_indices]

ax2.scatter(pareto_sbs_x, pareto_sbs_y, color=sbs_color, marker=sbs_marker, s=120,
           edgecolors='black', alpha=0.8, label='SBS Pareto')
ax2.scatter(pareto_uncertain_x, pareto_uncertain_y, color=sbs_uncertain_color, marker=sbs_uncertain_marker, s=120,
           edgecolors='black', alpha=0.8, label='SBS-U Pareto')

if sbs_pareto_points:
    pareto_x, pareto_y = zip(*sbs_pareto_points)
    ax2.plot(pareto_x, pareto_y, color=sbs_color, linestyle='--', linewidth=2, alpha=0.7)

if sbs_uncertain_pareto_points:
    pareto_x, pareto_y = zip(*sbs_uncertain_pareto_points)
    ax2.plot(pareto_x, pareto_y, color=sbs_uncertain_color, linestyle='--', linewidth=2, alpha=0.7)

ax2.set_title('Pareto Frontiers Comparison', fontsize=12)
ax2.set_xlabel('Inference FLOPs ($log_2$)')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("sbs-vs-sbs-uncertain-detailed.png", dpi=300, bbox_inches='tight')
plt.show()