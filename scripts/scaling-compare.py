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
    {'B1': 2, 'B3': 4, 'tokens generated': 2749, 'accuracy': 0.5772},
    {'B1': 2, 'B3': 8, 'tokens generated': 6065, 'accuracy': 0.6250},
    {'B1': 2, 'B3': 16, 'tokens generated': 13191, 'accuracy': 0.6700},
    {'B1': 2, 'B3': 20, 'tokens generated': 16771, 'accuracy': 0.6687},
    {'B1': 4, 'B3': 8, 'tokens generated': 5969, 'accuracy': 0.6520},
    {'B1': 4, 'B3': 16, 'tokens generated': 14721, 'accuracy': 0.7028},
    {'B1': 4, 'B3': 32, 'tokens generated': 29802, 'accuracy': 0.7059},
    {'B1': 5, 'B3': 20, 'tokens generated': 17420, 'accuracy': 0.7051},
    {'B1': 5, 'B3': 32, 'tokens generated': 27236, 'accuracy': 0.6991}
]

# SBS-PUMxValue
sbs_uncertain_data = [
    {'B1': 2, 'B3': 16, 'tokens generated': 13516, 'accuracy': 0.6828},
    {'B1': 2, 'B3': 32, 'tokens generated': 26956, 'accuracy': 0.7115},
    {'B1': 2, 'B3': 4,  'tokens generated': 2759,  'accuracy': 0.6052},
    {'B1': 2, 'B3': 8,  'tokens generated': 6028,  'accuracy': 0.6620},
    {'B1': 4, 'B3': 16, 'tokens generated': 13920, 'accuracy': 0.6949},
    {'B1': 4, 'B3': 20, 'tokens generated': 17973, 'accuracy': 0.6915},
    {'B1': 4, 'B3': 32, 'tokens generated': 26680, 'accuracy': 0.7075},
    {'B1': 4, 'B3': 8,  'tokens generated': 6420,  'accuracy': 0.6774},
    {'B1': 5, 'B3': 20,  'tokens generated': 19583,  'accuracy': .7143},
    {'B1': 5, 'B3': 32, 'tokens generated': 29741, 'accuracy': 0.7246},
]

# SBS-Variance
# sbs_uncertain_data = [
#     {'B1': 2, 'B3': 4,  'tokens generated': 2855,  'accuracy': 0.6020},
#     {'B1': 2, 'B3': 8,  'tokens generated': 6002,  'accuracy': 0.6560},
#     {'B1': 2, 'B3': 20, 'tokens generated': 15279, 'accuracy': 0.7030},
#     {'B1': 4, 'B3': 8,  'tokens generated': 5912,  'accuracy': 0.6653},
#     {'B1': 4, 'B3': 16, 'tokens generated': 13159, 'accuracy': 0.6938},
#     {'B1': 5, 'B3': 20, 'tokens generated': 16917, 'accuracy': 0.6959},
#     {'B1': 5, 'B3': 32, 'tokens generated': 27898, 'accuracy': 0.7256},
# ]

# SBS-CWE
# sbs_uncertain_data = [
#     {'B1': 2, 'B3': 4, 'tokens generated': 3130, 'accuracy': 0.5940},
#     {'B1': 2, 'B3': 8, 'tokens generated': 6437, 'accuracy': 0.6513},
#     {'B1': 2, 'B3': 16, 'tokens generated': 13523, 'accuracy': 0.6599},
#     {'B1': 2, 'B3': 20, 'tokens generated': 16961, 'accuracy': 0.6694},
#     {'B1': 4, 'B3': 8, 'tokens generated': 7152, 'accuracy': 0.6500},
#     {'B1': 4, 'B3': 16, 'tokens generated': 14548, 'accuracy': 0.6942},
#     {'B1': 4, 'B3': 32, 'tokens generated': 30926, 'accuracy': 0.7069},
#     {'B1': 5, 'B3': 20, 'tokens generated': 19738, 'accuracy': 0.6874},
#     {'B1': 5, 'B3': 32, 'tokens generated': 30851, 'accuracy': 0.7218}
# ]



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