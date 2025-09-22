import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


train_data = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "regression", split="train[:30%]")
test_data = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "regression", split="test")
scores = [d["labels"] for d in test_data]

# --- Data Processing ---

# We'll look at all positions using actual indices.
# Find the maximum length to determine how many positions we need to check
max_length = max(len(score_list) for score_list in scores)

# Create a dictionary to store the scores for each position index.
# The structure will be: {index: [list of scores]}
# e.g., {0: [0.2, 0.8, 0.1, ...], 1: [0.5, 0.9, ...]}
position_scores = defaultdict(list)

# Iterate over each list of scores
for score_list in scores:
    # Iterate through each position in the list
    for i in range(len(score_list)):
        score_value = score_list[i]
        position_scores[i].append(score_value)

# --- Plotting ---

# Get the positions we have data for (from 0 to max_length-1)
positions = sorted(position_scores.keys())

# Calculate statistics for each position
mean_scores = [np.mean(position_scores[pos]) for pos in positions]
std_scores = [np.std(position_scores[pos]) for pos in positions]
median_scores = [np.median(position_scores[pos]) for pos in positions]

# Create a single figure with prettier styling
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Use a prettier color scheme
main_color = '#2E86AB'  # Beautiful blue
fill_color = '#A23B72'  # Complementary pink/purple for the gradient

# Plot the mean line
line = ax.plot(positions, mean_scores, marker='o', linestyle='-', 
               color=main_color, markersize=8, linewidth=2.5, 
               markerfacecolor='white', markeredgecolor=main_color, 
               markeredgewidth=2, label='Mean Score')

# Add the standard deviation as a gradient fill
ax.fill_between(positions, 
                [mean - std for mean, std in zip(mean_scores, std_scores)],
                [mean + std for mean, std in zip(mean_scores, std_scores)],
                alpha=0.3, color=fill_color, label='± 1 Standard Deviation')

ax.set_xlabel("Position Index", fontsize=14, fontweight='bold')
ax.set_ylabel("Mean Score", fontsize=14, fontweight='bold')
ax.set_title("Score Distribution by Position Index", fontsize=16, fontweight='bold', pad=20)

# Prettier grid
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_facecolor('#fafafa')  # Light background

# Set limits with some padding
ax.set_ylim(0, 1)
ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)

# Add legend
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

# Style the spines
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#666666')

# Set the x-axis ticks
ax.set_xticks(positions[::max(1, len(positions)//20)])  # Show every nth tick to avoid crowding
ax.tick_params(axis='both', which='major', labelsize=12)

# Make the layout tight to prevent labels from overlapping
plt.tight_layout()

# Save the plot to a file
plt.savefig("/scratch/jacopo04/regression_scores_dist.png", dpi=300, bbox_inches='tight')

print("Plot saved as regression_scores_dist.png")

# Also, let's print the statistics for the first few positions
print("\nStatistics for the first 5 positions:")
for i in range(5):
    if i in position_scores:
        scores_at_pos = position_scores[i]
        print(f"Position {i}:")
        print(f"  Count: {len(scores_at_pos)}")
        print(f"  Mean:  {np.mean(scores_at_pos):.4f}")
        print(f"  Std:   {np.std(scores_at_pos):.4f}")
        print(f"  Min:   {np.min(scores_at_pos):.4f}")
        print(f"  Max:   {np.max(scores_at_pos):.4f}")
        print(f"  Median: {np.median(scores_at_pos):.4f}")
        print(f"  Q1:    {np.percentile(scores_at_pos, 25):.4f}")
        print(f"  Q3:    {np.percentile(scores_at_pos, 75):.4f}")
        print("-" * 30)

# Additional analysis: Show how many scores are above/below certain thresholds
print("\nThreshold analysis for the first position (0):")
if 0 in position_scores:
    first_pos_scores = position_scores[0]
    total_count = len(first_pos_scores)
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        above_threshold = sum(1 for score in first_pos_scores if score > threshold)
        percentage = (above_threshold / total_count) * 100
        print(f"  Scores > {threshold}: {above_threshold}/{total_count} ({percentage:.1f}%)")

# --- Linear Regression Analysis ---

# Prepare training data from train_data
train_scores = [d["labels"] for d in train_data]

# Process training data similarly to test data
train_position_scores = defaultdict(list)
for score_list in train_scores:
    for i in range(len(score_list)):
        score_value = score_list[i]
        train_position_scores[i].append(score_value)

# Get training positions and mean scores
train_positions = sorted(train_position_scores.keys())
train_mean_scores = [np.mean(train_position_scores[pos]) for pos in train_positions]

# Prepare data for linear regression
# X_train will be the position indices from training data
X_train = np.array(train_positions).reshape(-1, 1)
y_train = np.array(train_mean_scores)

# X_test will be the position indices from test data (for prediction/evaluation)
X_test = np.array(positions).reshape(-1, 1)
y_test = np.array(mean_scores)

# Create and fit the model on training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)

# Create regression plot
plt.figure(figsize=(12, 8))
plt.scatter(positions, mean_scores, color=main_color, label='Mean Uncertainty Scores', s=60, alpha=0.7)
plt.plot(positions, y_pred, color='red', linewidth=3, label=f'Linear Regression (R² = {r2:.4f})')

plt.xlabel("Step Position Index", fontsize=14, fontweight='bold')
plt.ylabel("Mean Uncertainty Score", fontsize=14, fontweight='bold')
plt.title("Linear Regression: Position vs Uncertainty Score", fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)

plt.grid(True, linestyle='--', alpha=0.3, color='gray')
plt.gca().set_facecolor('#fafafa')
plt.tight_layout()

# Save the regression plot
plt.savefig("/scratch/jacopo04/uncertainty_regression_analysis.png", dpi=300, bbox_inches='tight')

print(f"\nLinear Regression Analysis:")
print(f"  R² Score: {r2:.4f}")
print(f"  Coefficient (slope): {model.coef_[0]:.6f}")
print(f"  Intercept: {model.intercept_:.4f}")
print(f"  Position explains {r2*100:.2f}% of variance in uncertainty scores")

if r2 > 0.5:
    print("  → Strong relationship: Position strongly predicts uncertainty")
elif r2 > 0.3:
    print("  → Moderate relationship: Position moderately predicts uncertainty")
elif r2 > 0.1:
    print("  → Weak relationship: Position weakly predicts uncertainty")
else:
    print("  → Very weak relationship: Position barely predicts uncertainty")

plt.show()
