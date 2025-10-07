import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Find the most recent results directory
results_base = 'probing/scripts/results'
run_dirs = [d for d in glob.glob(os.path.join(results_base, '*_*')) if os.path.isdir(d)]
if not run_dirs:
    raise ValueError(f"No timestamped results directories found in {results_base}")
latest_run = max(run_dirs, key=os.path.getmtime)
print(f"Using results from: {latest_run}")

# Load the regression predictions
pred_df = pd.read_csv(os.path.join(latest_run, 'data', 'linear_probe_transplant_regression_predictions.csv'))

# Filter to test set only
test_df = pred_df[pred_df['split'] == 'test'].copy()

# Get unique problems in test set
test_problems = sorted(test_df['pn'].unique())

print(f"Found {len(test_problems)} problems in test set")
print(f"Problems: {test_problems}")

# For each problem, compute average cue_p and predicted cue_p across all layers/sentences
problem_stats = []
for pn in test_problems:
    pn_data = test_df[test_df['pn'] == pn]
    avg_true = pn_data['true_cue_p'].mean()
    avg_pred = pn_data['predicted_cue_p'].mean()
    problem_stats.append({
        'pn': pn,
        'avg_true_cue_p': avg_true,
        'avg_predicted_cue_p': avg_pred,
        'diff': avg_true - avg_pred
    })

stats_df = pd.DataFrame(problem_stats)

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: True vs Predicted cue_p for each problem
x = np.arange(len(test_problems))
width = 0.35

bars1 = ax1.bar(x - width/2, stats_df['avg_true_cue_p'], width, label='True cue_p', alpha=0.8)
bars2 = ax1.bar(x + width/2, stats_df['avg_predicted_cue_p'], width, label='Predicted cue_p', alpha=0.8)

ax1.set_xlabel('Problem ID (pn)')
ax1.set_ylabel('Average cue_p (across all layers/sentences)')
ax1.set_title('True vs Predicted cue_p per Problem (Test Set)')
ax1.set_xticks(x)
ax1.set_xticklabels(test_problems, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Residuals (difference) for each problem
bars3 = ax2.bar(x, stats_df['diff'], alpha=0.8, color='orange')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect prediction')
ax2.set_xlabel('Problem ID (pn)')
ax2.set_ylabel('Residual (True - Predicted)')
ax2.set_title('Prediction Residuals per Problem (Test Set)')
ax2.set_xticks(x)
ax2.set_xticklabels(test_problems, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save the figure
output_path = os.path.join(latest_run, 'figures', 'per_problem_cue_p_comparison_test_set.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved figure to {output_path}")

# Print summary statistics
print("\nPer-problem statistics (test set):")
print(stats_df.to_string(index=False))
print(f"\nMean absolute error: {stats_df['diff'].abs().mean():.4f}")
print(f"RMSE: {np.sqrt((stats_df['diff']**2).mean()):.4f}")
