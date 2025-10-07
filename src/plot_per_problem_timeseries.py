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

# Load the faithfulness CSV to get sentence contents
faith_df = pd.read_csv('CoT_Faithfulness_demo/faith_counterfactual_qwen-14b_demo.csv')

# Filter to test set only
test_df = pred_df[pred_df['split'] == 'test'].copy()

# Get unique problems in test set
test_problems = sorted(test_df['pn'].unique())

print(f"Found {len(test_problems)} problems in test set")

# Create output directory for per-problem plots
plots_dir = os.path.join(latest_run, 'figures', 'per_problem_timeseries')
os.makedirs(plots_dir, exist_ok=True)

# Create a master dataframe with sentence contents
all_data = []

for pn in test_problems:
    # Get data for this problem
    pn_data = test_df[test_df['pn'] == pn].copy()

    # Average predictions across all layers for each sentence
    sentence_avg = pn_data.groupby('sentence').agg({
        'true_cue_p': 'mean',
        'predicted_cue_p': 'mean'
    }).reset_index()

    # Sort by sentence number
    sentence_avg = sentence_avg.sort_values('sentence')

    # Create plot for this problem
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sentence_avg['sentence'], sentence_avg['true_cue_p'],
            marker='o', label='True cue_p', linewidth=2, markersize=6)
    ax.plot(sentence_avg['sentence'], sentence_avg['predicted_cue_p'],
            marker='s', label='Predicted cue_p', linewidth=2, markersize=6, alpha=0.7)

    ax.set_xlabel('Sentence Number in CoT')
    ax.set_ylabel('cue_p (probability of cued answer)')
    ax.set_title(f'Problem {pn}: True vs Predicted cue_p over CoT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'problem_{pn}_timeseries.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot for problem {pn}")

    # Get sentence contents from faithfulness CSV
    pn_faith = faith_df[faith_df['pn'] == pn].copy()

    # Merge with predictions
    for _, row in sentence_avg.iterrows():
        sent_num = row['sentence']

        # Get sentence content
        faith_row = pn_faith[pn_faith['sentence_num'] == sent_num]
        sentence_text = faith_row['sentence'].values[0] if len(faith_row) > 0 else "N/A"

        all_data.append({
            'pn': pn,
            'sentence_num': sent_num,
            'sentence': sentence_text,
            'true_cue_p': row['true_cue_p'],
            'predicted_cue_p': row['predicted_cue_p'],
            'error': row['true_cue_p'] - row['predicted_cue_p']
        })

# Create master dataframe
master_df = pd.DataFrame(all_data)

# Save to CSV
csv_path = os.path.join(latest_run, 'data', 'per_problem_timeseries_with_sentences.csv')
master_df.to_csv(csv_path, index=False)
print(f"\nSaved master dataframe to {csv_path}")

# Print summary for first few problems
print("\nSample data (first problem):")
print(master_df[master_df['pn'] == test_problems[0]].to_string(index=False))

print(f"\nCreated {len(test_problems)} timeseries plots in {plots_dir}")
print(f"Master CSV has {len(master_df)} rows covering {len(test_problems)} problems")
