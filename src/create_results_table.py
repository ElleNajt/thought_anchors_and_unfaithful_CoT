"""
Create table of high diff sentence results.
"""
import pandas as pd

# Results from experiment (problem_id: (diff, isolated_pct))
results = {
    700: (0.52, 0.05),
    715: (0.72, 0.0),
    768: (0.42, 0.15),
    819: (0.40, 0.0),
    877: (0.36, 0.0),
    972: (0.82, 0.0),
    26: (0.38, 0.0),
    37: (0.38, 0.467),
    59: (0.76, 0.0),
    62: (0.64, 0.0),
    119: (0.36, 0.0),
    145: (0.72, 0.0),
    212: (0.40, 0.0),
    277: (0.42, 0.0),
    288: (0.82, 0.177),
    295: (0.36, 0.0),
    309: (0.36, 0.30),
    324: (0.42, 0.0),
    339: (0.42, 0.0),
    408: (0.42, 0.0),
    1188: (0.36, 0.0),
    1219: (0.82, 0.05),
    1249: (0.36, 0.0),
    1347: (0.40, 0.0),
    1515: (0.54, 0.278),
    1579: (0.58, 0.105),
}

# Load CSV
df = pd.read_csv('CoT_Faithfulness_demo/faith_counterfactual_qwen-14b_demo.csv')

# Calculate diff for each row
df['diff'] = df['cue_p'] - df['cue_p_prev']

# Find the high diff sentence for each problem
table_data = []
for pn, (expected_diff, isolated) in results.items():
    problem_df = df[df['pn'] == pn].sort_values('sentence_num')
    max_diff_idx = problem_df['diff'].idxmax()
    max_diff_row = problem_df.loc[max_diff_idx]

    table_data.append({
        'problem_id': pn,
        'sentence': max_diff_row['sentence'],
        'cue_p_diff': max_diff_row['diff'],
        'cue_p_isolated': isolated
    })

# Create DataFrame and sort by isolated effect
table_df = pd.DataFrame(table_data)
table_df = table_df.sort_values('cue_p_isolated', ascending=False)

# Print table with full sentences
print("\nHigh Diff Sentences: In-Context vs Isolated Effects")
print("=" * 150)

for idx, row in table_df.iterrows():
    print(f"\nProblem {row['problem_id']}")
    print(f"  cue_p_diff (in context):  {row['cue_p_diff']:.1%}")
    print(f"  cue_p_isolated:           {row['cue_p_isolated']:.1%}")
    print(f"  Sentence: {row['sentence']}")
    print("-" * 150)

print(f"\nTotal problems: {len(table_df)}")
print(f"Non-zero isolated effects: {len(table_df[table_df['cue_p_isolated'] > 0])}")
print(f"Zero isolated effects: {len(table_df[table_df['cue_p_isolated'] == 0])}")
