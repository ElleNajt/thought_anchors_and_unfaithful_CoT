#!/usr/bin/env python3
"""
Analyze delta cue_p across all problems.

This script calculates the increase in cue_p from the previous transplanted sentence
for each sentence, sorted by highest delta at the top.
"""

import pandas as pd
import sys
from pathlib import Path


def calculate_delta_cue_p(input_csv_path, output_csv_path=None):
    """
    Calculate delta cue_p for each sentence across all problems.
    
    Args:
        input_csv_path: Path to the input CSV file with per_problem_timeseries_with_sentences
        output_csv_path: Optional path to save the output CSV. If None, saves to same directory
                        with '_delta_analysis.csv' suffix
    
    Returns:
        DataFrame with delta_cue_p analysis
    """
    # Read the input CSV
    df = pd.read_csv(input_csv_path)
    
    # Sort by problem number and sentence number to ensure correct order
    df = df.sort_values(['pn', 'sentence_num']).reset_index(drop=True)
    
    # Calculate delta_cue_p (increase from previous sentence)
    delta_data = []
    
    for problem_num in df['pn'].unique():
        problem_df = df[df['pn'] == problem_num].sort_values('sentence_num')
        
        for idx, row in problem_df.iterrows():
            sentence_num = row['sentence_num']
            current_cue_p = row['true_cue_p']
            
            # For the first sentence (sentence_num 0), delta is relative to 0
            if sentence_num == 0:
                delta_cue_p = current_cue_p - 0.0
            else:
                # Get the previous sentence's cue_p
                prev_row = problem_df[problem_df['sentence_num'] == sentence_num - 1]
                if not prev_row.empty:
                    prev_cue_p = prev_row.iloc[0]['true_cue_p']
                    delta_cue_p = current_cue_p - prev_cue_p
                else:
                    # If previous sentence not found, skip
                    continue
            
            delta_data.append({
                'problem_number': int(problem_num),
                'sentence_number': int(sentence_num),
                'sentence': row['sentence'],
                'true_cue_p': current_cue_p,
                'delta_cue_p': delta_cue_p
            })
    
    # Create DataFrame and sort by highest delta_cue_p
    result_df = pd.DataFrame(delta_data)
    result_df = result_df.sort_values('delta_cue_p', ascending=False).reset_index(drop=True)
    
    # Determine output path
    if output_csv_path is None:
        input_path = Path(input_csv_path)
        output_csv_path = input_path.parent / f"{input_path.stem}_delta_analysis.csv"
    
    # Save to CSV
    result_df.to_csv(output_csv_path, index=False)
    
    print(f"Analysis complete!")
    print(f"Total sentences analyzed: {len(result_df)}")
    print(f"Results saved to: {output_csv_path}")
    print("\n" + "="*80)
    print("Top 10 sentences with highest delta_cue_p:")
    print("="*80)
    
    # Display top 10
    for idx, row in result_df.head(10).iterrows():
        print(f"\nRank {idx + 1}:")
        print(f"  Problem: {row['problem_number']}, Sentence: {row['sentence_number']}")
        print(f"  Delta cue_p: {row['delta_cue_p']:.4f}")
        print(f"  True cue_p: {row['true_cue_p']:.4f}")
        print(f"  Sentence: {row['sentence'][:100]}{'...' if len(row['sentence']) > 100 else ''}")
    
    print("\n" + "="*80)
    print("\nTop 10 sentences with lowest delta_cue_p (largest decreases):")
    print("="*80)
    
    # Display bottom 10 (largest decreases)
    for idx, row in result_df.tail(10).iloc[::-1].iterrows():
        print(f"\n  Problem: {row['problem_number']}, Sentence: {row['sentence_number']}")
        print(f"  Delta cue_p: {row['delta_cue_p']:.4f}")
        print(f"  True cue_p: {row['true_cue_p']:.4f}")
        print(f"  Sentence: {row['sentence'][:100]}{'...' if len(row['sentence']) > 100 else ''}")
    
    return result_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_delta_cue_p.py <input_csv_path> [output_csv_path]")
        print("\nExample:")
        print("  python analyze_delta_cue_p.py probing/scripts/results/7048225_20251009_010420/data_n/per_problem_timeseries_with_sentences.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_csv).exists():
        print(f"Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    calculate_delta_cue_p(input_csv, output_csv)


if __name__ == "__main__":
    main()

