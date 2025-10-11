"""
Create scatter plot of cue_p diff vs isolated effect, colored by primary mechanism.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load classification results
with open("CoT_Faithfulness_demo/high_diff_sentence_classifications.json", "r") as f:
    classifications = json.load(f)

# Load isolated sentence effect results
with open("CoT_Faithfulness_demo/isolated_sentence_effect_results.json", "r") as f:
    isolated_results = json.load(f)

# Create lookup dicts
mechanism_lookup = {}
for result in classifications:
    problem_id = result['problem_id']
    if result.get('classification') and result['classification'].get('classifications'):
        primary_mechanism = result['classification']['classifications'][0]['mechanism']
        mechanism_lookup[problem_id] = primary_mechanism
    else:
        mechanism_lookup[problem_id] = "Unknown"

# Extract data for plotting
data = []
for result in isolated_results:
    problem_id = result['problem_id']
    diff = result['diff']

    # Use sentence_cue_pct as the isolated effect (from previous run)
    # This is the percentage that answered with cue when shown the sentence
    isolated_effect = result.get('sentence_cue_pct')

    mechanism = mechanism_lookup.get(problem_id, "Unknown")

    if isolated_effect is not None:
        data.append({
            'problem_id': problem_id,
            'diff': diff,
            'isolated_effect': isolated_effect,
            'mechanism': mechanism
        })

df = pd.DataFrame(data)

# Create color mapping for mechanisms
mechanisms = df['mechanism'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(mechanisms)))
color_map = dict(zip(mechanisms, colors))

# Create the scatter plot
fig, ax = plt.subplots(figsize=(12, 8))

for mechanism in mechanisms:
    mask = df['mechanism'] == mechanism
    ax.scatter(
        df[mask]['diff'],
        df[mask]['isolated_effect'],
        label=mechanism,
        color=color_map[mechanism],
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

ax.set_xlabel('Cue_p Diff (in-context effect)', fontsize=12)
ax.set_ylabel('Isolated Effect (% cue answer)', fontsize=12)
ax.set_title('Sentence Isolated Effect vs In-Context Diff\nColored by Primary Bias Mechanism', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('CoT_Faithfulness_demo/mechanism_vs_isolated_effect.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to CoT_Faithfulness_demo/mechanism_vs_isolated_effect.png")

# Print summary statistics by mechanism
print("\n=== Summary by Mechanism ===")
summary = df.groupby('mechanism').agg({
    'diff': ['mean', 'count'],
    'isolated_effect': 'mean'
}).round(3)
print(summary)
