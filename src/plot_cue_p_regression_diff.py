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

# Calculate the difference (residual): actual - predicted
# Note: 'error' column already contains predicted - true, so we need to negate it
pred_df['diff'] = pred_df['true_cue_p'] - pred_df['predicted_cue_p']

# Group by sentence position and calculate statistics
sentence_stats = pred_df.groupby('sentence')['diff'].agg(['mean', 'std', 'count']).reset_index()

# Calculate standard error
sentence_stats['se'] = sentence_stats['std'] / np.sqrt(sentence_stats['count'])

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(sentence_stats['sentence'], sentence_stats['mean'], marker='o', linewidth=2)
plt.fill_between(sentence_stats['sentence'],
                  sentence_stats['mean'] - sentence_stats['se'],
                  sentence_stats['mean'] + sentence_stats['se'],
                  alpha=0.3)

plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect prediction')
plt.xlabel('Sentence Position in Transplanted CoT')
plt.ylabel('Residual (Actual cue_p - Predicted cue_p)')
plt.title('Regression Probe Residuals by Sentence Position')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure to the same run directory
output_path = os.path.join(latest_run, 'figures', 'cue_p_regression_residuals_by_sentence.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_path}")

# Print summary statistics
print("\nSummary statistics by sentence:")
print(sentence_stats.to_string(index=False))
